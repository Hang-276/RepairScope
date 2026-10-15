"""Small strict experiment config; paths resolve from the repository cwd."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tomllib

DEFAULT = {
    "root": "vendor/appworld", "output_dir": "runs", "seed": 100,
    "actor": {"base_url": "http://127.0.0.1:8000/v1", "model": "Qwen/Qwen3-8B-AWQ",
              "tokenizer": "Qwen/Qwen3-8B-AWQ", "tokenizer_revision": None, "api_key_env": "B2_ACTOR_API_KEY",
              "max_tokens": 3000, "context_length": 8192, "temperature": 0.0},
    "policy": {"checkpoint_interval": 5, "goal_tokens": 512,
               "candidate_tokens": 2400, "candidate_code_tokens": 600},
    "budget": {"max_steps": 50, "max_output_tokens": 100000},
}


def load_config(path: str | Path | None = None) -> dict:
    result = copy.deepcopy(DEFAULT)
    if path:
        path = Path(path)
        values = tomllib.loads(path.read_text()) if path.suffix == ".toml" else json.loads(path.read_text())
        for key, value in values.items():
            if key not in result:
                raise ValueError(f"Unknown config field: {key}")
            if isinstance(result[key], dict):
                optional = {"max_total_tokens", "max_calls", "max_seconds", "max_classifier_tokens"} if key == "budget" else set()
                if not isinstance(value, dict) or set(value) - set(result[key]) - optional:
                    raise ValueError(f"Unknown fields in {key}")
                result[key].update(value)
            else:
                result[key] = value
    for name in ("root", "output_dir"):
        result[name] = str(Path(result[name]).expanduser().resolve())
    if not isinstance(result["seed"], int) or isinstance(result["seed"], bool):
        raise ValueError("seed must be an integer")
    for key in ("max_tokens", "context_length"):
        if isinstance(result["actor"][key], bool) or not isinstance(result["actor"][key], int) or result["actor"][key] <= 0:
            raise ValueError(f"actor.{key} must be positive")
    if result["actor"]["context_length"] <= result["actor"]["max_tokens"]:
        raise ValueError("Actor context must leave room for the prompt")
    for section, keys in (("budget", ("max_steps",)),
                          ("policy", ("checkpoint_interval", "goal_tokens", "candidate_tokens", "candidate_code_tokens"))):
        if any(isinstance(result[section][key], bool) or not isinstance(result[section][key], int)
               or result[section][key] < 1 for key in keys):
            raise ValueError("Step, policy, and generation limits must be positive integers")
    from .budget import Budget
    Budget(**{key: value for key, value in result["budget"].items() if key != "max_steps"})
    return result
