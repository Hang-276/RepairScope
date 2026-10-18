"""CPU/CUDA training, reproducible checkpoints and bounded online updates.

The training path never initializes AppWorld, an Actor, or a pretrained encoder.
A checkpoint describes its evidence origin; synthetic smoke runs remain synthetic.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .losses import counterfactual_losses, rollout_state_loss, transition_losses
from .decision import scope_supervision_audit, goal_sufficiency_loss
from .model import Predictor
from .tensorizer import Tensorizer, canonical, to_device

DEFAULT_CONFIG = {
    "model": {"d_model": 384, "n_heads": 6, "relation_layers": 4, "temporal_layers": 2, "history": 8, "scope_budget": 128, "candidate_limit": 256, "feature_dim": 64, "value_slots": 64, "dropout": 0.1, "backbone": "blite", "learned_edges": True, "temporal": True, "static_scope": False},
    "loss": {"rollout": 1., "boundary": 1., "escape": 1., "paired": 0., "scope": 1., "patch": 1., "goal": 1.},
    "train": {"epochs": 30, "micro_batch": 4, "grad_accum": 4, "learning_rate": 3e-4, "weight_decay": .01, "clip": 1., "precision": "bf16", "warmup_fraction": .1, "seed": 100, "device": "auto", "horizon": 4},
}


def resolve_config(config: dict) -> dict:
    result = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(copy.deepcopy(value))
        else:
            result[key] = copy.deepcopy(value)
    return result


def _device(config):
    requested = config["train"].get("device", "auto")
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; run GPU experiments on the server")
    if str(device).startswith("mps"):
        raise ValueError("Local verification is CPU-only; configure device=cpu or use CUDA on the server")
    return torch.device(device)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _windows(records: list[dict], horizon: int) -> list[list[dict]]:
    trajectories = defaultdict(dict)
    for i, record in enumerate(records):
        identity = tuple(record.get(key) for key in ("task_id", "group_id", "split", "trajectory_id", "cf_pair_id", "cf_condition", "cf_role"))
        step = int(record.get("step", i))
        if step in trajectories[identity]:
            raise ValueError("Duplicate trajectory steps; supply distinct trajectory_id values")
        trajectories[identity][step] = i
    windows = [[record] for record in records]
    for timeline in trajectories.values():
        for step, idx in timeline.items():
            window = [records[idx]]
            for next_step in range(step+1, step+horizon):
                if next_step not in timeline:
                    break
                following = records[timeline[next_step]]
                # A reset/intervention cannot be silently joined into a free
                # rollout, even when its step numbers happen to be contiguous.
                if window[-1].get("next_state") != following.get("state"):
                    break
                window.append(following)
            windows[idx] = window
    return windows


def _effective_precision(config: dict, device: torch.device) -> str:
    requested = config["train"].get("precision", "bf16")
    if requested not in ("fp32", "bf16"):
        raise ValueError("Supported training precision is fp32 or bf16; fp16 is not implemented")
    effective = "bf16" if device.type == "cuda" and requested == "bf16" else "fp32"
    if effective == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 is unsupported on this CUDA device")
    return effective


def _bundles(records, windows):
    paired = defaultdict(list)
    for i, record in enumerate(records):
        if record.get("cf_pair_id"):
            paired[record["cf_pair_id"]].append(i)
    grouped, used = [], set()
    for indices in paired.values():
        # Keep valid pairs/quartets together so their joint objective is evaluated
        # every epoch, with no extra data or additional optimizer updates.
        signatures = [(records[i].get("cf_condition"), records[i].get("cf_role")) for i in indices]
        if len(indices) in (2, 4) and len(set(signatures)) == len(indices):
            grouped.append(indices)
            used.update(indices)
    grouped.extend([i] for i in range(len(windows)) if i not in used)
    return grouped


def _minibatches(order, bundles, micro_batch):
    batches, current = [], []
    for index in order:
        unit = bundles[index]
        if current and len(current) + len(unit) > micro_batch:
            batches.append(current)
            current = []
        current.extend(unit)
        if len(current) >= micro_batch:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def _loss_for_windows(model, tensorizer, windows, config, device, auxiliary):
    records = [window[0] for window in windows]
    initial = to_device(tensorizer.batch(records), device)
    output = model(initial)
    components = transition_losses(output, initial, config)
    counter = counterfactual_losses(output, initial, config)
    decision = goal_sufficiency_loss(output, initial, config)
    weights = config["loss"]
    total = components["step"]
    roll_losses = [rollout_state_loss(output, initial)]
    escapes = [components["escape"]]
    goal_escapes = [components["goal_relevant_escape"]]
    boundaries, paired = [counter["boundary"]], [counter["paired"]]
    horizon = max(len(window) for window in windows) if float(weights.get("rollout", 1)) else 1
    state, history = output["soft_state"], output["next_history"]
    for step in range(1, horizon):
        active = torch.tensor([step < len(window) for window in windows], dtype=torch.bool, device=device)
        # Only action/targets of later transitions are read.  Candidate/copy slots,
        # states, relations, and histories stay rooted at the initial observation.
        later_records = [window[step] if step < len(window) else window[-1] for window in windows]
        safe_later_records = [{**later, "state": records[i].get("state", {}), "history": records[i].get("history", []), "goals": records[i].get("goals", [])} for i, later in enumerate(later_records)]
        target_batch = to_device(tensorizer.batch(safe_later_records, template=initial["metadata"]), device)
        for key in ("features", "kind", "mask", "nullable", "slot_mask", "slot_features", "static_edges", "goal_priority", "goal_field", "goal_mask", "history_values", "history_numeric", "history_known", "history_exists", "history_null", "history_action", "history_mask"):
            target_batch[key] = initial[key]
        # Dynamic static-prior scoring for the new action uses the initial public
        # schema, not a future true state. No future field keys can be introduced.
        for key in ("audit_mask", "target_known"):
            target_batch[key] = target_batch[key] & active.unsqueeze(-1)
        target_batch["target_goal"] = target_batch["target_goal"].masked_fill(~active.unsqueeze(-1), -100)
        target_batch["escape_mask"] = target_batch["escape_mask"] & active
        target_batch["records"] = [r if bool(active[i]) else {**r, "audit": {}, "cf_pair_id": None} for i, r in enumerate(later_records)]
        output = model(target_batch, soft_state=state, history=history)
        state, history = output["soft_state"], output["next_history"]
        roll_losses.append(rollout_state_loss(output, target_batch))
        step_losses = transition_losses(output, target_batch, config)
        escapes.append(step_losses["escape"])
        goal_escapes.append(step_losses["goal_relevant_escape"])
        counter = counterfactual_losses(output, target_batch, config)
        boundaries.append(counter["boundary"])
        paired.append(counter["paired"])
    roll = torch.stack(roll_losses).mean()
    escape, boundary, pair = (torch.stack(items).mean() for items in (escapes, boundaries, paired))
    goal_escape = torch.stack(goal_escapes).mean()
    total = total + float(weights.get("rollout", 1)) * roll
    if auxiliary:
        total = total + float(weights.get("escape", 1)) * escape + float(weights.get("boundary", 1)) * boundary + float(weights.get("paired", 0)) * pair
        total = total + float(weights.get("goal_sufficiency", 0)) * decision["goal_sufficiency"] + float(weights.get("goal_relevant_escape", 0)) * goal_escape
    metrics = {key: float(value.detach().cpu()) for key, value in components.items() if isinstance(value, torch.Tensor)}
    metrics.update(loss=float(total.detach().cpu()), rollout=float(roll.detach().cpu()), escape=float(escape.detach().cpu()), boundary=float(boundary.detach().cpu()), paired=float(pair.detach().cpu()), auxiliary_active=bool(auxiliary))
    metrics.update(goal_sufficiency=float(decision["goal_sufficiency"].detach().cpu()), goal_relevant_escape=float(goal_escape.detach().cpu()),
                   decision_groups=decision["decision_groups"], decision_eligible_groups=decision["decision_eligible_groups"],
                   decision_informative_groups=decision["decision_informative_groups"])
    return total, metrics


def _manifest(records, config):
    origins = sorted(set(r.get("evidence_origin", "unspecified") for r in records))
    groups = sorted(set(str(r.get("group_id", r.get("task_id", ""))) for r in records))
    schema = {}
    for record in records:
        for field in record.get("state", {}).get("fields", []):
            schema[field["key"]] = {k: field[k] for k in ("key", "kind", "entity", "choices", "scale") if k in field}
    digest = hashlib.sha256(canonical(records).encode()).hexdigest()
    return {"format_version": 1, "records": len(records), "groups": groups, "evidence_origin": origins, "dataset_sha256": digest, "backend": "synthetic" if any(origin in ("synthetic", "fixture") for origin in origins) else "recorded", "schema": list(schema.values()), "gradient_boundaries": ["hard candidate retrieval", "v2 scope: exact hard forward / fixed-budget implicit sigmoid STE backward" if config.get("model", {}).get("goal_sufficient", False) else "hard top-k scope", "JSON hard decode", "unbound entity/free-text creation"], "soft_rollout_horizon": int(config["train"].get("horizon", 4))}


def train_model(records: list[dict], config: dict, output_dir: str | Path, resume: str | Path | None = None) -> dict:
    if not records:
        raise ValueError("Training requires at least one transition")
    if any(r.get("split") in ("test", "test_normal", "test_challenge", "dev", "validation", "val") for r in records):
        raise ValueError("Held-out dev/test transitions must not be used for training")
    config = resolve_config(config)
    decision_audit = scope_supervision_audit(records, Tensorizer(config), config)
    if float(config["loss"].get("goal_sufficiency", 0)) > 0 and not decision_audit["decision_trainable_groups"]:
        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {key: value for key, value in decision_audit.items() if key != "groups"}
        (report_dir / "goal_supervision_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        raise ValueError("Goal-sufficiency training has no informative trainable legal counterfactual decisions; inspect goal_supervision_audit.json. Unknown or unretrievable labels cannot train v2.")
    train = config["train"]
    if int(train["epochs"]) < 1 or int(train["micro_batch"]) < 1 or int(train["grad_accum"]) < 1:
        raise ValueError("epochs, micro_batch and grad_accum must be positive")
    device = _device(config)
    _seed(int(train.get("seed", 100)))
    model, tensorizer = Predictor(config).to(device), Tensorizer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]), weight_decay=float(train["weight_decay"]))
    windows = _windows(records, max(1, int(train.get("horizon", 4))))
    bundles = _bundles(records, windows)
    base_batches = _minibatches(list(range(len(bundles))), bundles, int(train["micro_batch"]))
    estimated_steps = math.ceil(len(base_batches) / int(train["grad_accum"])) * int(train["epochs"])
    max_steps = int(train.get("max_steps") or estimated_steps)
    warmup = max(0, math.ceil(estimated_steps * float(train.get("warmup_fraction", .1))))
    def rate(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1., max(0., (step - warmup) / max(1, estimated_steps-warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    effective_precision = _effective_precision(config, device)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(records, config)
    (outdir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    (outdir / "manifest.json").write_text(json.dumps({**manifest, "effective_precision": effective_precision, "device": str(device)}, ensure_ascii=False, indent=2) + "\n")
    (outdir / "schema.json").write_text(json.dumps(manifest["schema"], ensure_ascii=False, indent=2) + "\n")
    (outdir / "goal_supervision_audit.json").write_text(json.dumps({key: value for key, value in decision_audit.items() if key != "groups"}, indent=2) + "\n")
    global_step, epoch, batch_cursor, order = 0, 0, 0, None
    history, latest = [], {}
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("manifest", {}).get("dataset_sha256") != manifest["dataset_sha256"]:
            raise ValueError("Resume dataset differs from checkpoint")
        if checkpoint["config"]["model"] != config["model"] or checkpoint["config"]["loss"] != config["loss"]:
            raise ValueError("Resume architecture/loss configuration differs")
        if checkpoint["config"].get("goal_decision", {}) != config.get("goal_decision", {}):
            raise ValueError("Resume goal-decision utility configuration differs")
        for key in ("micro_batch", "grad_accum", "learning_rate", "weight_decay", "seed", "warmup_fraction", "epochs", "horizon"):
            if checkpoint["config"]["train"].get(key) != train.get(key):
                raise ValueError(f"Resume train.{key} differs; exact resume requires the same schedule")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step, epoch, batch_cursor = checkpoint["global_step"], checkpoint["epoch"], checkpoint["batch_cursor"]
        order, history = checkpoint.get("order"), checkpoint.get("history", [])
        _restore_rng(checkpoint["rng"])
    checkpoint_path = outdir / "checkpoint.pt"
    def save():
        payload = {"format_version": 1, "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": _rng_state(), "global_step": global_step, "epoch": epoch, "batch_cursor": batch_cursor, "order": order, "history": history, "schema": manifest["schema"], "manifest": manifest}
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while epoch < int(train["epochs"]) and global_step < max_steps:
        if order is None:
            order = list(range(len(bundles)))
            random.shuffle(order)
        minibatches = _minibatches(order, bundles, int(train["micro_batch"]))
        cursor = batch_cursor
        while cursor < len(minibatches) and global_step < max_steps:
            accumulation = min(int(train["grad_accum"]), len(minibatches)-cursor)
            accumulated_metrics = defaultdict(float)
            for offset in range(accumulation):
                indices = minibatches[cursor+offset]
                auxiliary = global_step >= warmup
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=effective_precision == "bf16"):
                    loss, metrics = _loss_for_windows(model, tensorizer, [windows[i] for i in indices], config, device, auxiliary)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at update {global_step}")
                (loss / accumulation).backward()
                for key, value in metrics.items():
                    accumulated_metrics[key] += float(value) / accumulation
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train.get("clip", 1.)), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            cursor += accumulation
            batch_cursor = cursor
            latest = {**dict(accumulated_metrics), "step": global_step, "epoch": epoch, "gradient_norm": float(norm.detach().cpu()), "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(latest)
            with (outdir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(latest) + "\n")
            if global_step % max(1, int(train.get("checkpoint_every", 500))) == 0:
                save()
        if cursor >= len(minibatches):
            epoch += 1
            batch_cursor, order = 0, None
    save()
    result = {"checkpoint": str(checkpoint_path.resolve()), "config": str((outdir / "config.json").resolve()), "manifest": str((outdir / "manifest.json").resolve()), "updates": global_step, "epochs_completed": epoch, "metrics": latest or (history[-1] if history else {}), "device": str(device), "effective_precision": effective_precision, "evidence_origin": manifest["evidence_origin"]}
    (outdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_predictor(checkpoint: str | Path, device: str = "cpu") -> tuple[Predictor, Tensorizer, dict]:
    # Local project checkpoints contain optimizer/RNG metadata and are trusted
    # artifacts, rather than arbitrary downloaded pickle files.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = resolve_config(payload["config"])
    config["checkpoint_manifest"] = copy.deepcopy(payload.get("manifest", {}))
    model = Predictor(config).to(device)
    weights = dict(payload["model"])
    # Legacy inference migration only: new dormant heads copy their old state
    # counterparts. Exact optimizer resume intentionally remains strict.
    if not config.get("model", {}).get("goal_sufficient", False):
        for new, old in (("goal_escape_head.", "escape_head."), ("goal_expansion_head.", "expansion_head.")):
            for key in model.state_dict():
                if key.startswith(new) and key not in weights:
                    weights[key] = weights[old + key[len(new):]]
    model.load_state_dict(weights)
    model.eval()
    return model, Tensorizer(config), config


def update_predictor(model: Predictor, tensorizer: Tensorizer, records: list[dict], config: dict, optimizer=None) -> dict:
    """Update the existing Predictor only; Actor and environment are untouched."""
    config = resolve_config(config)
    if any(r.get("split") in ("test", "test_normal", "test_challenge", "dev", "validation", "val") for r in records):
        raise ValueError("Query/held-out results cannot enter online training")
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    if not records:
        return {"optimizer": optimizer, "updates": 0, "records": 0, "metrics": {}}
    device = next(model.parameters()).device
    effective_precision = _effective_precision(config, device)
    windows = _windows(records, max(1, int(config["train"].get("horizon", 4))))
    bundles = _bundles(records, windows)
    updates, totals = 0, defaultdict(float)
    was_training = model.training
    model.train()
    for _ in range(max(1, int(config["train"].get("online_epochs", 1)))):
        order = list(range(len(bundles)))
        random.shuffle(order)
        batches = _minibatches(order, bundles, int(config["train"]["micro_batch"]))
        for start in range(0, len(batches), int(config["train"]["grad_accum"])):
            group = batches[start:start+int(config["train"]["grad_accum"])]
            optimizer.zero_grad(set_to_none=True)
            for indices in group:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=effective_precision == "bf16"):
                    loss, metrics = _loss_for_windows(model, tensorizer, [windows[i] for i in indices], config, device, True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite online update loss")
                (loss/len(group)).backward()
                for key, value in metrics.items():
                    totals[key] += float(value) / len(group)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"].get("clip", 1.)), error_if_nonfinite=True)
            optimizer.step()
            updates += 1
    model.train(was_training)
    return {"optimizer": optimizer, "updates": updates, "records": len(records), "metrics": {key: value/max(updates, 1) for key, value in totals.items()}}


def save_predictor(model: Predictor, tensorizer: Tensorizer, config: dict, path: str | Path, records: list[dict] | None = None, optimizer=None) -> str:
    """Save an inference/online snapshot. Offline exact resume uses train_model.

    When updated records are supplied their origin is merged with the parent
    manifest: a model touched by synthetic data cannot become AppWorld evidence.
    """
    config = resolve_config(config)
    parent = copy.deepcopy(config.get("checkpoint_manifest", {}))
    if records:
        current = _manifest(records, config)
        current["evidence_origin"] = sorted(set(parent.get("evidence_origin", []) + current["evidence_origin"]))
        current["parent_dataset_sha256"] = parent.get("dataset_sha256")
        current["backend"] = "synthetic" if "synthetic" in current["evidence_origin"] else current["backend"]
        manifest = current
    else:
        manifest = parent or {"evidence_origin": ["unspecified"], "backend": "unspecified"}
    target = Path(path)
    if target.suffix not in (".pt", ".pth"):
        target = target / "checkpoint.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "config": config, "model": model.state_dict(), "manifest": manifest, "schema": manifest.get("schema", []), "rng": _rng_state(), "optimizer": optimizer.state_dict() if optimizer is not None else None, "checkpoint_kind": "online_inference"}
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return str(target.resolve())
