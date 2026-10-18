"""Leakage-safe conversion of public transition observations into fixed field slots.

Targets are built separately after candidate and copy tables have been frozen.  In
particular, ``next_state`` never contributes a key, a copy value, or an input.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

import torch

from .schema import effective_protected, evaluate_goals, goal_state_keys, semantic_diff

KINDS = {"number": 0, "category": 1, "bool": 2, "text": 3, "reference": 4}
OPS = {"keep": 0, "modify": 1, "create": 2, "delete": 3}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any, width: int = 64) -> list[float]:
    """A stable, stateless signed hash encoding; no fitted vocabulary is required."""
    text = canonical(value)
    words = re.findall(r"[\w]+|[^\s\w]", text.lower())
    out = [0.0] * width
    for word in words or [text]:
        digest = hashlib.blake2b(word.encode(), digest_size=16).digest()
        idx = int.from_bytes(digest[:8], "little") % width
        out[idx] += 1.0 if digest[8] % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


def public_fields(record: dict) -> list[dict]:
    fields = copy.deepcopy(record.get("state", {}).get("fields", []))
    seen = {f["key"] for f in fields}
    schema = record.get("public_schema", {})
    slots = schema.get("fields", []) if isinstance(schema, dict) else schema
    # Only explicitly public slots may introduce an unseen key.
    for slot in slots or []:
        if isinstance(slot, dict) and slot.get("key") not in seen:
            field = copy.deepcopy(slot)
            field.update(value=None, known=False, exists=False)
            fields.append(field)
            seen.add(field["key"])
    return fields


def action_tokens(action: dict) -> set[str]:
    public = {k: action[k] for k in ("name", "arguments", "code") if k in action}
    return set(re.findall(r"[^\W_]+", canonical(public).lower()))


def _scalars(value: Any):
    if isinstance(value, dict):
        for v in value.values():
            yield from _scalars(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _scalars(v)
    elif value is not None:
        yield value


def known(field: dict | None) -> bool:
    return field is not None and bool(field.get("known", True))


class Tensorizer:
    """The returned dictionaries contain tensors and auditable public metadata."""
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        model = config.get("model", config)
        self.feature_dim = int(model.get("feature_dim", 64))
        self.value_slots = int(model.get("value_slots", 64))
        self.history = int(model.get("history", 8))
        self.candidate_limit = int(model.get("candidate_limit", 256))
        if min(self.feature_dim, self.value_slots, self.history, self.candidate_limit) < 1:
            raise ValueError("Tensorizer dimensions must be positive")

    def _metadata(self, record: dict) -> dict:
        fields = public_fields(record)
        keys = [f["key"] for f in fields]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate public state field key")
        # Public action/graph retrieval precedes candidate truncation. A small
        # label-independent coverage reservation keeps cross-graph fields eligible.
        terms = action_tokens(record.get("action", {}))
        priorities = {field["key"]: len(set(re.findall(r"[^\W_]+", field["key"].lower())) & terms) for field in fields}
        for left, right in record.get("state", {}).get("relations", []):
            if left in priorities and right in priorities:
                old_left, old_right = priorities[left], priorities[right]
                priorities[left] = max(old_left, .5*old_right)
                priorities[right] = max(old_right, .5*old_left)
        if len(fields) > self.candidate_limit:
            reserve = min(max(1, self.candidate_limit//20), self.candidate_limit-1)
            ordered = sorted(fields, key=lambda field: (-priorities[field["key"]], field["key"]))
            selected = ordered[:self.candidate_limit-reserve]
            remaining = ordered[self.candidate_limit-reserve:]
            remaining.sort(key=lambda field: hashlib.blake2b(field["key"].encode(), digest_size=8).digest())
            fields = selected + remaining[:reserve]
        fields = sorted(fields, key=lambda field: field["key"])
        tables = []
        action_values = list(_scalars(record.get("action", {}).get("arguments", {})))
        historical = {}
        for item in record.get("history", [])[-self.history:]:
            for f in item.get("state", {}).get("fields", []):
                if known(f):
                    historical.setdefault(f["key"], []).append(f.get("value"))
        for field in fields:
            kind = field.get("kind", "text")
            if kind not in KINDS:
                raise ValueError(f"Unknown field kind: {kind}")
            # All copy alternatives come from visible state/history/action/schema.
            candidates = ([False, True] if kind == "bool" else [])
            candidates += list(field.get("choices", []))
            if known(field):
                candidates += [field.get("value")]
            candidates += historical.get(field["key"], []) + action_values
            candidates += [item.get("value") for item in record.get("state", {}).get("fields", []) if known(item)]
            if kind in ("category", "text"):
                candidates = [value for value in candidates if isinstance(value, str) and (kind != "category" or not field.get("choices") or value in field["choices"])]
            elif kind == "bool":
                candidates = [False, True]
            elif kind == "reference":
                candidates = [value for value in candidates if type(value) in (int, str)]
            unique, seen = [], set()
            for value in candidates:
                token = canonical(value)
                if token not in seen:
                    seen.add(token)
                    unique.append(value)
            tables.append(unique[:self.value_slots] or [None])
        return {"fields": fields, "keys": [f["key"] for f in fields], "tables": tables}

    def batch(self, records: list[dict], template: list[dict] | None = None) -> dict:
        """Batch observations; ``template`` freezes initial rollout field/copy slots."""
        if not records:
            raise ValueError("Cannot tensorize an empty batch")
        metas = copy.deepcopy(template) if template is not None else [self._metadata(r) for r in records]
        if len(metas) != len(records):
            raise ValueError("Template batch size differs")
        b, n = len(records), max(1, max(len(m["keys"]) for m in metas))
        f, v, t = self.feature_dim, self.value_slots, self.history
        tensors = {
            "features": torch.zeros(b, n, f), "action": torch.zeros(b, f),
            "values": torch.zeros(b, n, v), "numeric": torch.zeros(b, n),
            "known": torch.zeros(b, n), "exists": torch.zeros(b, n), "null": torch.zeros(b, n),
            "nullable": torch.zeros(b, n, dtype=torch.bool),
            "kind": torch.zeros(b, n, dtype=torch.long),
            "mask": torch.zeros(b, n, dtype=torch.bool),
            "slot_mask": torch.zeros(b, n, v, dtype=torch.bool),
            "slot_features": torch.zeros(b, n, v, f),
            "static_edges": torch.zeros(b, n, n), "static_prior": torch.zeros(b, n),
            "goal_priority": torch.zeros(b, n),
            "recorded_scope": torch.zeros(b, n, dtype=torch.bool),
            "history_values": torch.zeros(b, n, t, v),
            "history_numeric": torch.zeros(b, n, t),
            "history_known": torch.zeros(b, n, t), "history_exists": torch.zeros(b, n, t), "history_null": torch.zeros(b, n, t),
            "history_action": torch.zeros(b, t, f),
            "history_mask": torch.zeros(b, n, t, dtype=torch.bool),
            "target_op": torch.zeros(b, n, dtype=torch.long),
            "target_numeric": torch.zeros(b, n),
            "target_value": torch.full((b, n), -100, dtype=torch.long),
            "target_known": torch.zeros(b, n, dtype=torch.bool),
            "target_exists": torch.zeros(b, n), "target_null": torch.zeros(b, n),
            "target_changed": torch.zeros(b, n),
            "audit_mask": torch.zeros(b, n, dtype=torch.bool),
            "target_escape": torch.zeros(b), "escape_mask": torch.zeros(b, dtype=torch.bool),
        }
        max_goals = max(1, max(len(r.get("goals", [])) for r in records))
        tensors.update(goal_field=torch.zeros(b, max_goals, dtype=torch.long),
                       goal_features=torch.zeros(b, max_goals, f),
                       goal_mask=torch.zeros(b, max_goals, dtype=torch.bool),
                       target_goal=torch.full((b, max_goals), -100, dtype=torch.long))
        for i, (record, meta) in enumerate(zip(records, metas)):
            keys, fields, tables = meta["keys"], meta["fields"], meta["tables"]
            index = {key: j for j, key in enumerate(keys)}
            current = {x["key"]: x for x in record.get("state", {}).get("fields", [])}
            target_state = record.get("audit_next_state") or record.get("next_state", {})
            target_before = record.get("audit_state") or record.get("state", {})
            target_before_fields = {x["key"]: x for x in target_before.get("fields", [])}
            after = {x["key"]: x for x in target_state.get("fields", [])}
            action = record.get("action", {})
            tensors["action"][i] = torch.tensor(fingerprint(action, f))
            action_words = action_tokens(action)
            audit = record.get("audit", {})
            complete = bool(audit.get("complete", False))
            domain = set(audit.get("field_domain", []))
            audited_changes = set(audit.get("changes", []))
            recorded_scope = set(record.get("scope", []))
            for j, field in enumerate(fields):
                key, table = keys[j], tables[j]
                observed = current.get(key, {**field, "value": None, "known": False, "exists": False})
                scale = max(float(field.get("scale", 1.0) or 1.0), 1e-8)
                tensors["mask"][i, j] = True
                tensors["kind"][i, j] = KINDS[field.get("kind", "text")]
                tensors["features"][i, j] = torch.tensor(fingerprint({"key": key, "entity": field.get("entity"), "kind": field.get("kind"), "nullable": field.get("nullable", False)}, f))
                tensors["known"][i, j] = float(known(observed))
                tensors["exists"][i, j] = float(observed.get("exists", True))
                tensors["nullable"][i, j] = bool(field.get("nullable", False))
                tensors["null"][i, j] = float(known(observed) and observed.get("exists", True) and observed.get("value") is None)
                tensors["slot_mask"][i, j, :len(table)] = True
                for si, value in enumerate(table):
                    tensors["slot_features"][i, j, si] = torch.tensor(fingerprint(value, f))
                if known(observed):
                    if field.get("kind") == "number" and isinstance(observed.get("value"), (int, float)):
                        tensors["numeric"][i, j] = float(observed["value"]) / scale
                    token = canonical(observed.get("value"))
                    for si, value in enumerate(table):
                        if canonical(value) == token:
                            tensors["values"][i, j, si] = 1.0
                            break
                key_words = set(re.findall(r"[^\W_]+", key.lower()))
                tensors["static_prior"][i, j] = min(1.0, len(key_words & action_words) / max(1, min(len(key_words), 3)))
                tensors["recorded_scope"][i, j] = key in recorded_scope
                tensors["static_edges"][i, j, j] = 1.0
                for jj, other in enumerate(fields):
                    if field.get("entity") is not None and field.get("entity") == other.get("entity"):
                        tensors["static_edges"][i, j, jj] = 1.0
                target = after.get(key)
                # Partial audits contribute positives and explicitly observed known fields;
                # an absent or unknown target never becomes an unchanged negative.
                is_audited = key in domain or key in audited_changes
                if complete and not domain:
                    is_audited = key in after
                target_valid = target is not None and known(target) and is_audited
                tensors["audit_mask"][i, j] = target_valid
                tensors["target_known"][i, j] = target_valid
                if target_valid:
                    label_before = target_before_fields.get(key, observed)
                    old_exists, new_exists = bool(label_before.get("exists", True)), bool(target.get("exists", True))
                    changed = key in audited_changes or (known(label_before) and bool(semantic_diff({"fields": [label_before]}, {"fields": [target]})))
                    op = "create" if not old_exists and new_exists else "delete" if old_exists and not new_exists else "modify" if changed else "keep"
                    tensors["target_op"][i, j] = OPS[op]
                    tensors["target_exists"][i, j] = float(new_exists)
                    tensors["target_null"][i, j] = float(new_exists and target.get("value") is None)
                    tensors["target_changed"][i, j] = float(changed)
                    if field.get("kind") == "number" and isinstance(target.get("value"), (int, float)):
                        tensors["target_numeric"][i, j] = float(target["value"]) / scale
                    for si, value in enumerate(table):
                        if canonical(value) == canonical(target.get("value")):
                            tensors["target_value"][i, j] = si
                            break
            for edge in record.get("state", {}).get("relations", []):
                if len(edge) == 2 and edge[0] in index and edge[1] in index:
                    u, w = index[edge[0]], index[edge[1]]
                    tensors["static_edges"][i, u, w] = tensors["static_edges"][i, w, u] = 1.0
            # Expand the static prior one hop; static graph remains a soft prior.
            if keys:
                tensors["static_prior"][i] = torch.maximum(tensors["static_prior"][i], (tensors["static_edges"][i] @ tensors["static_prior"][i]).clamp(max=1.0) * 0.5)
            hist = record.get("history", [])[-t:]
            for ti, entry in enumerate(hist, start=t-len(hist)):
                old = {x["key"]: x for x in entry.get("state", {}).get("fields", [])}
                tensors["history_action"][i, ti] = torch.tensor(fingerprint(entry.get("action", {}), f))
                for j, field in enumerate(fields):
                    item = old.get(keys[j])
                    if item is None:
                        continue
                    tensors["history_mask"][i, j, ti] = True
                    tensors["history_known"][i, j, ti] = float(known(item))
                    tensors["history_exists"][i, j, ti] = float(item.get("exists", True))
                    tensors["history_null"][i, j, ti] = float(known(item) and item.get("exists", True) and item.get("value") is None)
                    if known(item):
                        if field.get("kind") == "number" and isinstance(item.get("value"), (int, float)):
                            tensors["history_numeric"][i, j, ti] = float(item["value"]) / max(float(field.get("scale", 1.) or 1.), 1e-8)
                        for si, value in enumerate(tables[j]):
                            if canonical(value) == canonical(item.get("value")):
                                tensors["history_values"][i, j, ti, si] = 1.0
                                break
            visible_goals = evaluate_goals(record.get("state", {"fields": []}), record.get("goals", []))
            for gi, goal in enumerate(record.get("goals", [])):
                goal_keys = [key for key in goal_state_keys(goal, record.get("state", {"fields": []})) if key in index]
                if goal_keys:
                    j = index[goal_keys[0]]
                    tensors["goal_field"][i, gi] = j
                    tensors["goal_features"][i, gi] = torch.tensor(fingerprint(goal, f))
                    tensors["goal_mask"][i, gi] = True
                    priority = (float(goal.get("weight", 1)) *
                                (2.0 if effective_protected(goal, int(record.get("step", 0)), visible_goals[gi]) else 1.0))
                    for key in goal_keys:
                        tensors["goal_priority"][i, index[key]] += priority / len(goal_keys)
                before = record.get("goal_before", [])
                after_goals = record.get("goal_after", [])
                if gi < len(before) and gi < len(after_goals) and before[gi] is not None and after_goals[gi] is not None:
                    tensors["target_goal"][i, gi] = 2 * int(before[gi]) + int(after_goals[gi])
            escaped = bool(audited_changes - recorded_scope)
            tensors["target_escape"][i] = float(escaped)
            tensors["escape_mask"][i] = escaped or complete
        tensors["metadata"] = metas
        tensors["records"] = records
        return tensors


def to_device(batch: dict, device: str | torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
