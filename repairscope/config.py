"""Explicit, reproducible configuration and actual ablation switches."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "experiment": "repairscope",
    "variant": "full",
    "model": {
        "d_model": 384, "n_heads": 6, "relation_layers": 4,
        "temporal_layers": 2, "history": 8, "scope_budget": 128,
        "candidate_limit": 256, "feature_dim": 64, "value_slots": 64,
        "dropout": 0.1, "backbone": "blite", "learned_edges": True,
        "temporal": True, "static_scope": False, "goal_sufficient": False,
    },
    "loss": {"scope": 1.0, "patch": 1.0, "goal": 1.0, "rollout": 1.0,
             "boundary": 1.0, "escape": 1.0, "paired": 0.0,
             "goal_sufficiency": 0.0, "goal_relevant_escape": 0.0},
    "goal_decision": {"temperature": 1.0, "progress_weight": 1.0,
                      "damage_weight": 5.0, "cost_weight": 0.01,
                      "protected_rule": "satisfied_and_effective_protected"},
    "train": {
        "epochs": 30, "micro_batch": 4, "grad_accum": 4,
        "learning_rate": 0.0003, "weight_decay": 0.01, "clip": 1.0,
        "precision": "bf16", "warmup_fraction": 0.1, "seed": 100,
        "device": "auto", "horizon": 4, "max_steps": None,
    },
    "data": {"train": "data/repairscope/train.jsonl",
             "dev": "data/repairscope/dev.jsonl", "test": None,
             "allow_synthetic": False, "counterfactual_plans": None,
             "goal_map": None},
    "environment": {"backend": "appworld", "root": None,
                    "appworld_revision": "42b5bcf3cd334fee33f0c37c02070a9f5807add5",
                    "observation_map": None, "offline_audit": False,
                    "max_steps": 50},
    "collection": {"split": "train", "task_ids": [], "max_steps": 50,
                   "seed": 100, "action_plans": None},
    "actor": {"kind": "openai", "model": None,
              "base_url_env": "OPENAI_BASE_URL", "api_key_env": "OPENAI_API_KEY",
              "temperature": 0.7, "max_tokens": 2048, "timeout": 120,
              "cache_path": "data/repairscope/actor_candidates.json"},
    "controller": {"candidate_count": 4, "max_calls": 100, "max_steps": 50,
                   "risk_threshold": 0.5, "progress_weight": 1.0, "damage_weight": 5.0,
                   "cost_weight": 0.01, "escape_weight": 1.0,
                   "repair": True, "max_repairs": 3, "mc_samples": 4},
    "evaluation": {"task_ids": [], "split": "test_normal",
                   "experiment_name": "repairscope", "bootstrap_samples": 1000,
                   "seed": 100},
    "probing": {"strategy": "risk", "budget_calls": 2000, "epsilon": 0.1,
                "mc_samples": 4, "plans": None, "update_every": 16,
                "replay_capacity": 10000, "query_task_ids": [],
                "checkpoints": [0.25, 0.5, 0.75, 1.0]},
}

VARIANTS: dict[str, dict[str, Any]] = {
    "actor_only": {"loss": {"boundary": 0, "escape": 0},
                   "controller": {"actor_only": True, "escape_weight": 0}},
    "dynamics": {"loss": {"boundary": 0, "escape": 0},
                 "controller": {"escape_weight": 0}},
    "paired_effect": {"loss": {"boundary": 0, "escape": 0, "paired": 1},
                      "controller": {"escape_weight": 0}},
    "full": {},
    "no_boundary": {"loss": {"boundary": 0}},
    "no_escape": {"loss": {"escape": 0}, "controller": {"escape_weight": 0}},
    "static_scope": {"model": {"static_scope": True}},
    "no_learned_edges": {"model": {"learned_edges": False}},
    "no_temporal": {"model": {"temporal": False}},
    "unified": {"model": {"backbone": "unified"}},
    "no_rollout": {"loss": {"rollout": 0}},
    "no_goal": {"loss": {"goal": 0}},
    "no_probing": {"probing": {"budget_calls": 0}},
}


def merge(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


# Legacy variants keep their original objective even when selected from a v2 YAML.
VARIANTS = {name: merge({"model": {"goal_sufficient": False},
                         "loss": {"goal_sufficiency": 0.0,
                                  "goal_relevant_escape": 0.0}}, patch)
            for name, patch in VARIANTS.items()}
_V2 = {"model": {"goal_sufficient": True},
       "loss": {"goal_sufficiency": 1.0, "goal_relevant_escape": 1.0}}
VARIANTS.update({
    "goal_sufficient": deepcopy(_V2),
    "no_goal_sufficiency": merge(_V2, {"loss": {"goal_sufficiency": 0.0}}),
    "v2_no_boundary": merge(_V2, {"loss": {"boundary": 0.0}}),
    "no_goal_relevant_escape": merge(_V2, {"loss": {"goal_relevant_escape": 0.0}}),
    "v2_static_scope": merge(_V2, {"model": {"static_scope": True},
                                   "loss": {"goal_sufficiency": 0.0}}),
})

_UTILITY_KEYS = ("progress_weight", "damage_weight", "cost_weight")


def _resolve_goal_decision(config: dict, explicit: dict) -> None:
    """One utility for training references and execution; support legacy configs."""
    for key in _UTILITY_KEYS:
        decision = explicit.get("goal_decision", {})
        controller = explicit.get("controller", {})
        if key in decision and key in controller and decision[key] != controller[key]:
            raise ValueError(f"Conflicting goal_decision.{key} and controller.{key}")
        value = decision.get(key, controller.get(key, config["goal_decision"][key]))
        config["goal_decision"][key] = config["controller"][key] = value


def _read(path: Path, visited: set[Path]) -> dict:
    path = path.expanduser().resolve()
    if path in visited:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    visited = visited | {path}
    with path.open(encoding="utf-8") as fh:
        value = yaml.safe_load(fh)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    parent = value.pop("base", None)
    if parent:
        value = merge(_read(path.parent / parent, visited), value)
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                variant: str | None = None) -> dict:
    explicit = _read(Path(path), set()) if path else {}
    config = merge(DEFAULTS, explicit)
    selected = variant or config.get("variant", "full")
    if selected not in VARIANTS:
        raise ValueError(f"Unknown variant {selected!r}; choose {sorted(VARIANTS)}")
    config = merge(config, VARIANTS[selected])
    config["variant"] = selected
    _resolve_goal_decision(config, explicit)
    for expression in overrides or []:
        key, separator, raw = expression.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"Expected key.path=value, got {expression!r}")
        current = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                raise ValueError(f"Unknown configuration section {key}")
            current = current[part]
        current[parts[-1]] = yaml.safe_load(raw)
        if len(parts) == 2 and parts[0] in {"controller", "goal_decision"} and parts[1] in _UTILITY_KEYS:
            other = "controller" if parts[0] == "goal_decision" else "goal_decision"
            config[other][parts[1]] = current[parts[-1]]
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    m, t = config["model"], config["train"]
    for key in ("d_model", "n_heads", "relation_layers", "history", "scope_budget",
                "candidate_limit", "feature_dim", "value_slots"):
        if not isinstance(m[key], int) or isinstance(m[key], bool) or m[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if m["d_model"] % m["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    if m["scope_budget"] > m["candidate_limit"]:
        raise ValueError("scope_budget cannot exceed candidate_limit")
    if not 0 <= m["dropout"] < 1:
        raise ValueError("dropout must be in [0,1)")
    for key in ("epochs", "micro_batch", "grad_accum", "horizon"):
        if not isinstance(t[key], int) or isinstance(t[key], bool) or t[key] < 1:
            raise ValueError(f"train.{key} must be positive")
    if not 0 <= t["warmup_fraction"] < 1:
        raise ValueError("warmup_fraction must be in [0,1)")
    if t["precision"] not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16; fp16 is not supported")
    if config["environment"]["backend"] not in {"fixture", "appworld"}:
        raise ValueError("environment.backend must be fixture or appworld")
    if not 0 <= config["probing"]["epsilon"] <= 1:
        raise ValueError("probing.epsilon must be in [0,1]")
    if config["probing"]["budget_calls"] < 0:
        raise ValueError("probing budget must be nonnegative")
    if not isinstance(m["goal_sufficient"], bool):
        raise ValueError("model.goal_sufficient must be bool")
    decision = config["goal_decision"]
    for key in (*_UTILITY_KEYS, "temperature"):
        value = decision[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"goal_decision.{key} must be finite and nonnegative")
    if decision["temperature"] == 0:
        raise ValueError("goal_decision.temperature must be positive")
    if decision["protected_rule"] != "satisfied_and_effective_protected":
        raise ValueError("Unsupported goal_decision.protected_rule")
    for key in ("goal_sufficiency", "goal_relevant_escape"):
        value = config["loss"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{key} must be finite and nonnegative")


def dump_config(config: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                          encoding="utf-8")
