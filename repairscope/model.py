"""Action-conditioned B-lite dynamics with typed soft and hard state patches."""
from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .tensorizer import KINDS, Tensorizer, canonical, to_device


class RelationBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, learned_edges: bool):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.action_query = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        self.edge_query = nn.Linear(width, 32, bias=False) if learned_edges else None
        self.edge_key = nn.Linear(width, 32, bias=False) if learned_edges else None
        self.static_strength = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))

    def forward(self, x, action, static_edges, scope, field_mask):
        b, n, d = x.shape
        normed = self.norm(x)
        q, k, v = self.qkv(normed).chunk(3, dim=-1)
        q = q + self.action_query(action).unsqueeze(1)
        def split(z):
            return z.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        bias = self.static_strength * static_edges
        if self.edge_query is not None:
            bias = bias + torch.tanh(torch.matmul(self.edge_query(normed), self.edge_key(normed).transpose(-1, -2)) / math.sqrt(32))
        # Hard selection is an explicit gradient boundary.  Static edges are a
        # finite bias, so selected fields can interact across a missing edge.
        allowed = scope & field_mask
        safe = allowed.clone()
        safe[:, 0] |= ~safe.any(dim=-1)
        attention_bias = bias.unsqueeze(1).expand(-1, self.heads, -1, -1).masked_fill(~safe[:, None, None, :], float("-inf"))
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias, dropout_p=self.dropout.p if self.training else 0.0)
        h = h.transpose(1, 2).reshape(b, n, d)
        x = x + self.dropout(self.output(h))
        return (x + self.dropout(self.ff(x))) * field_mask.unsqueeze(-1)


class Predictor(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = copy.deepcopy(config)
        cfg = config.get("model", config)
        self.width = int(cfg.get("d_model", 384))
        self.heads = int(cfg.get("n_heads", 6))
        if self.width % self.heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.history_length = int(cfg.get("history", 8))
        self.scope_budget = int(cfg.get("scope_budget", 128))
        if self.scope_budget < 1:
            raise ValueError("scope_budget must be positive")
        self.backbone = cfg.get("backbone", "blite")
        if self.backbone not in ("blite", "unified"):
            raise ValueError("backbone must be blite or unified")
        self.use_temporal = bool(cfg.get("temporal", True))
        self.static_scope = bool(cfg.get("static_scope", False))
        self.learned_edges = bool(cfg.get("learned_edges", True))
        self.use_escape = float(config.get("loss", {}).get("escape", 1)) > 0
        feature_dim = int(cfg.get("feature_dim", 64))
        dropout = float(cfg.get("dropout", 0.1))
        self.feature_encoder = nn.Linear(feature_dim, self.width)
        self.action_encoder = nn.Linear(feature_dim, self.width)
        self.value_encoder = nn.Linear(feature_dim, self.width)
        self.state_encoder = nn.Linear(4, self.width)
        self.kind_embedding = nn.Embedding(5, self.width)
        self.input_norm = nn.LayerNorm(self.width)
        self.scope_head = nn.Sequential(nn.Linear(2 * self.width, self.width), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.width, 1))
        self.expansion_head = nn.Sequential(nn.Linear(2 * self.width, self.width // 2), nn.GELU(), nn.Linear(self.width // 2, 1))
        self.relations = nn.ModuleList(RelationBlock(self.width, self.heads, dropout, self.learned_edges) for _ in range(int(cfg.get("relation_layers", 4))))
        self.time_embedding = nn.Embedding(self.history_length + 1, self.width)
        self.temporal = nn.ModuleList(nn.TransformerEncoderLayer(self.width, self.heads, 4 * self.width, dropout, batch_first=True, norm_first=True, activation="gelu") for _ in range(int(cfg.get("temporal_layers", 2))))
        if self.backbone == "unified":
            # Matched width and depth; a single joint field/time Transformer.
            layers = int(cfg.get("relation_layers", 4)) + (int(cfg.get("temporal_layers", 2)) if self.use_temporal else 0)
            self.relations = nn.ModuleList()
            self.temporal = nn.ModuleList()
            self.unified = nn.ModuleList(nn.TransformerEncoderLayer(self.width, self.heads, 4 * self.width, dropout, batch_first=True, norm_first=True, activation="gelu") for _ in range(layers))
        if not self.use_temporal:
            self.temporal = nn.ModuleList()
        self.output_norm = nn.LayerNorm(self.width)
        self.op_head = nn.Linear(self.width, 4)
        self.number_head = nn.Linear(self.width, 1)
        self.null_head = nn.Linear(self.width, 1)
        self.copy_query = nn.Linear(self.width, self.width)
        self.copy_key = nn.Linear(self.width, self.width)
        self.escape_head = nn.Sequential(nn.Linear(4 * self.width + 2, self.width), nn.GELU(), nn.Linear(self.width, 1))
        self.goal_encoder = nn.Linear(feature_dim, self.width)
        self.goal_head = nn.Sequential(nn.Linear(3 * self.width, self.width), nn.GELU(), nn.Linear(self.width, 4))

    def _state_embedding(self, batch, state, slots):
        return self.input_norm(self.feature_encoder(batch["features"]) + self.kind_embedding(batch["kind"]) + torch.einsum("bnv,bnvd->bnd", state["values"], slots) + self.state_encoder(torch.stack((state["numeric"], state["known"], state["exists"], state["null"]), dim=-1)))

    def _initial_history(self, batch, slots):
        # [batch,field,time,width], preserving separate histories per field.
        values = torch.einsum("bntv,bnvd->bntd", batch["history_values"], slots)
        state = torch.stack((batch["history_numeric"], batch["history_known"], batch["history_exists"], batch["history_null"]), dim=-1)
        h = self.input_norm(self.feature_encoder(batch["features"]).unsqueeze(2) + self.kind_embedding(batch["kind"]).unsqueeze(2) + values + self.state_encoder(state))
        h = h + self.action_encoder(batch["history_action"]).unsqueeze(1)
        return h, batch["history_mask"]

    def _scope(self, x, action, batch, state):
        b, n, _ = x.shape
        action_fields = action.unsqueeze(1).expand(-1, n, -1)
        logits = self.scope_head(torch.cat((x, action_fields), dim=-1)).squeeze(-1)
        pooled = (x * batch["mask"].unsqueeze(-1)).sum(1) / batch["mask"].sum(1, keepdim=True).clamp_min(1)
        expansion = self.expansion_head(torch.cat((pooled, action), dim=-1)).sigmoid()
        probability = logits.sigmoid()
        uncertainty = 4 * probability * (1 - probability)
        # Goal priority changes scheduling only, never the dynamics features.
        if self.static_scope:
            score = batch["static_prior"]
        else:
            learned = probability if self.learned_edges else torch.zeros_like(probability)
            risk = expansion if self.use_escape else torch.zeros_like(expansion)
            score = batch["static_prior"] + learned + 0.2 * uncertainty + 0.2 * (1 - state["known"]) + 0.25 * risk * (1 - batch["static_prior"]) + 0.2 * batch["goal_priority"].clamp(max=5)
        score = score.masked_fill(~batch["mask"], float("-inf"))
        selected = torch.zeros_like(batch["mask"])
        for i in range(b):
            available = int(batch["mask"][i].sum())
            count = min(self.scope_budget, available)
            if not count:
                continue
            # A small deterministic coverage reservation is independent of labels.
            reserve = min(max(1, count // 20), count - 1) if not self.static_scope else 0
            top = torch.topk(score[i], count - reserve).indices
            selected[i, top] = True
            if reserve:
                remainder = torch.where(batch["mask"][i] & ~selected[i])[0]
                # Stable pseudo-random field permutation; no RNG/target dependence.
                hashes = ((remainder.to(torch.long) * 1103515245 + 12345) % 2147483647)
                chosen = remainder[torch.argsort(hashes)[:reserve]]
                selected[i, chosen] = True
        return logits, selected, expansion.squeeze(-1)

    def reliability(self, hidden, action, scope, mask):
        def pool(selection):
            return (hidden * selection.unsqueeze(-1)).sum(1) / selection.sum(1, keepdim=True).clamp_min(1)
        inside, outside = scope & mask, ~scope & mask
        size = scope.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        uncovered = outside.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.escape_head(torch.cat((pool(mask), pool(inside), pool(outside), action, size, uncovered), -1)).squeeze(-1)

    def forward(self, batch: dict, soft_state: dict | None = None, history: tuple | None = None) -> dict:
        state = soft_state if soft_state is not None else {key: batch[key] for key in ("values", "numeric", "known", "exists", "null")}
        slots = self.value_encoder(batch["slot_features"])
        raw = self._state_embedding(batch, state, slots)
        action = self.action_encoder(batch["action"])
        x = raw + action.unsqueeze(1)
        scope_logits, scope, expansion = self._scope(x, action, batch, state)
        if history is None:
            history = self._initial_history(batch, slots)
        old_h, old_mask = history
        if self.backbone == "unified":
            b, n, d = x.shape
            tokens = torch.cat((old_h, x.unsqueeze(2)), dim=2) if self.use_temporal else x.unsqueeze(2)
            valid = torch.cat((old_mask, batch["mask"].unsqueeze(-1)), dim=-1) if self.use_temporal else batch["mask"].unsqueeze(-1)
            length = tokens.shape[2]
            pos = self.time_embedding(torch.arange(length, device=x.device)).view(1, 1, length, d)
            tokens = (tokens + pos).reshape(b, n * length, d)
            valid_flat = (valid & scope.unsqueeze(-1)).reshape(b, n * length).clone()
            valid_flat[:, -1] |= ~valid_flat.any(-1)
            time = torch.arange(length, device=x.device).repeat(n)
            causal = time.unsqueeze(0) > time.unsqueeze(1)
            for layer in self.unified:
                tokens = layer(tokens, src_mask=causal, src_key_padding_mask=~valid_flat)
            x = tokens.reshape(b, n, length, d)[:, :, -1]
        else:
            for layer in self.relations:
                x = layer(x, action, batch["static_edges"], scope, batch["mask"])
            if self.use_temporal:
                b, n, d = x.shape
                sequence = torch.cat((old_h, x.unsqueeze(2)), dim=2)
                length = sequence.shape[2]
                position = self.time_embedding(torch.arange(length, device=x.device)).view(1, 1, length, d)
                sequence = (sequence + position).reshape(b * n, length, d)
                padding = ~torch.cat((old_mask, torch.ones(b, n, 1, dtype=torch.bool, device=x.device)), dim=-1).reshape(b * n, length)
                causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
                for layer in self.temporal:
                    sequence = layer(sequence, src_mask=causal, src_key_padding_mask=padding)
                x = sequence[:, -1].reshape(b, n, d)
        x = self.output_norm(x) * batch["mask"].unsqueeze(-1)
        op_logits = self.op_head(x)
        raw_ops = op_logits.softmax(-1)
        existence = state["exists"]
        op_prob = raw_ops * torch.stack((torch.ones_like(existence), existence, 1-existence, existence), -1)
        op_prob = op_prob / op_prob.sum(-1, keepdim=True).clamp_min(1e-8)
        gate = scope.unsqueeze(-1).to(op_prob.dtype)
        keep = torch.zeros_like(op_prob)
        keep[..., 0] = 1
        applied_ops = gate * op_prob + (1 - gate) * keep
        value_logits = (self.copy_query(x).unsqueeze(2) * self.copy_key(slots)).sum(-1) / math.sqrt(self.width)
        safe_slots = batch["slot_mask"].clone()
        safe_slots[..., 0] = True
        value_logits = value_logits.masked_fill(~safe_slots, -1e4)
        value_prob = value_logits.softmax(-1)
        number = state["numeric"] + self.number_head(x).squeeze(-1)
        null_logits = self.null_head(x).squeeze(-1)
        null_probability = null_logits.sigmoid() * batch["nullable"]
        retain = applied_ops[..., 0] + applied_ops[..., 3]
        write = applied_ops[..., 1] + applied_ops[..., 2]
        next_state = {
            "values": retain.unsqueeze(-1) * state["values"] + write.unsqueeze(-1) * value_prob,
            "numeric": retain * state["numeric"] + write * number,
            "exists": (applied_ops[..., 0] + applied_ops[..., 1]) * state["exists"] + applied_ops[..., 2],
            "known": state["known"] + (1 - state["known"]) * (write + applied_ops[..., 3]),
            "null": retain * state["null"] + write * null_probability,
        }
        goal_fields = batch["goal_field"].unsqueeze(-1).expand(-1, -1, self.width)
        field_features = x.gather(1, goal_fields)
        next_embedding = self._state_embedding(batch, next_state, slots).gather(1, goal_fields)
        goal_logits = self.goal_head(torch.cat((field_features, next_embedding, self.goal_encoder(batch["goal_features"])), dim=-1))
        return {"op_logits": op_logits, "op_probabilities": applied_ops, "value_logits": value_logits,
                "value_probabilities": value_prob, "numeric": number, "null_logits": null_logits, "null_probabilities": null_probability, "soft_state": next_state,
                "scope_logits": scope_logits, "scope": scope,
                "escape_logits": self.reliability(x, action, scope, batch["mask"]),
                "recorded_escape_logits": self.reliability(x, action, batch["recorded_scope"], batch["mask"]),
                "expansion": expansion, "goal_logits": goal_logits, "hidden": x,
                "next_history": (torch.cat((old_h, (raw + action.unsqueeze(1)).unsqueeze(2)), dim=2)[:, :, -self.history_length:],
                                 torch.cat((old_mask, batch["mask"].unsqueeze(-1)), dim=-1)[:, :, -self.history_length:])}


def decode_prediction(output: dict, batch: dict, index: int = 0) -> dict:
    record, meta = batch["records"][index], batch["metadata"][index]
    next_state = copy.deepcopy(record.get("state", {"fields": []}))
    fields = {f["key"]: f for f in next_state.get("fields", [])}
    patches, selected, probabilities, changes, operations = [], [], {}, {}, {}
    for j, key in enumerate(meta["keys"]):
        probabilities[key] = float(output["scope_logits"][index, j].sigmoid().detach().cpu())
        operations[key] = output["op_probabilities"][index, j].detach().cpu().tolist()
        changes[key] = float(1 - output["op_probabilities"][index, j, 0].detach().cpu())
        if not bool(output["scope"][index, j]):
            continue
        selected.append(key)
        op = int(output["op_probabilities"][index, j].argmax())
        if op == 0:
            continue
        source = fields.get(key, copy.deepcopy(meta["fields"][j]))
        field = copy.deepcopy(source)
        field["known"] = True
        field["exists"] = op != 3
        if op == 3:
            field["value"] = None
        elif field.get("nullable", False) and float(output["null_probabilities"][index, j].detach().cpu()) >= 0.5:
            field["value"] = None
        elif meta["fields"][j].get("kind") == "number":
            scale = float(meta["fields"][j].get("scale", 1.) or 1.)
            field["value"] = float(output["numeric"][index, j].detach().cpu()) * scale
        else:
            slot = int(output["value_probabilities"][index, j].argmax())
            field["value"] = copy.deepcopy(meta["tables"][j][slot])
            # An unbound copy is explicitly unknown, never an invented ID/text.
            if field["value"] is None:
                field["known"] = False
        fields[key] = field
        patches.append({"key": key, "op": ("keep", "modify", "create", "delete")[op], "kind": field.get("kind", "text"), "value": field.get("value"), "known": field["known"], "exists": field["exists"], "probability": float(output["op_probabilities"][index, j, op].detach().cpu())})
    next_state["fields"] = list(fields.values())
    goals = output["goal_logits"][index].softmax(-1).detach().cpu().tolist()[:len(record.get("goals", []))]
    return {"next_state": next_state, "patch": patches, "scope": selected,
            "escape": float(output["escape_logits"][index].sigmoid().detach().cpu()),
            "goal_probabilities": goals, "scope_probabilities": probabilities, "change_probabilities": changes, "operation_probabilities": operations, "uncertainty": 0.0}


def predict_record(model: Predictor, tensorizer: Tensorizer, record: dict, device: str = "cpu", mc_samples: int = 0) -> dict:
    original_flags = {module: module.training for module in model.modules()}
    batch = to_device(tensorizer.batch([record]), device)
    try:
        model.eval()
        with torch.inference_mode():
            output = model(batch)
            result = decode_prediction(output, batch)
            use_goal_head = float(model.config.get("loss", {}).get("goal", 1)) > 0
            if not use_goal_head:
                result["goal_probabilities"] = []
            if mc_samples > 1:
                # MC dropout changes only the explicitly stochastic layers;
                # restore every module's original flag even for mixed modes.
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                samples, escapes, hypotheses = [], [], []
                for _ in range(mc_samples):
                    draw = model(batch)
                    samples.append(draw["scope_logits"].sigmoid())
                    escapes.append(draw["escape_logits"].sigmoid())
                    hypothesis = decode_prediction(draw, batch)
                    if not use_goal_head:
                        hypothesis["goal_probabilities"] = []
                    hypotheses.append(hypothesis)
                scope_var = torch.stack(samples).var(0, unbiased=False)[batch["mask"]].mean() if batch["mask"].any() else torch.tensor(0., device=batch["mask"].device)
                result["uncertainty"] = float((scope_var + torch.stack(escapes).var(0, unbiased=False).mean()).cpu())
                result["mc_scope_probabilities"] = [item["scope_probabilities"] for item in hypotheses]
                result["mc_predictions"] = hypotheses
        return result
    finally:
        for module, flag in original_flags.items():
            module.training = flag
