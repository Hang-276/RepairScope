"""Frozen candidate policies. Only public observations enter policy prompts.

The API backend is OpenAI compatible and deliberately has no training interface.
Scripted/cached candidates are explicit engineering fixtures, never benchmark labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agent.wma_core import aggregate_candidates


def public_state(state: dict) -> dict:
    """Use an allowlist rather than stripping a few known hidden-state names."""
    fields = []
    for item in state.get("fields", []):
        field = {k: copy.deepcopy(item[k]) for k in
                 ("key", "kind", "value", "known", "exists", "choices", "entity", "entity_type", "scale", "tolerance", "nullable", "updated_at")
                 if k in item}
        if not field.get("known", True):
            field["value"] = None
        fields.append(field)
    return {"fields": fields, "relations": copy.deepcopy(state.get("relations", []))}


def public_goals(goals: list[dict]) -> list[dict]:
    return [{k: copy.deepcopy(g[k]) for k in
             ("id", "field", "op", "value", "protected", "weight", "protect_from_step", "protect_until_step",
              "active_from", "active_until", "protect_after_satisfied", "description")
             if k in g} for g in goals]


def public_history(history: list[dict], limit: int = 8) -> list[dict]:
    return [{"state": public_state(h.get("state", {})),
             "action": normalize_action(h.get("action", {"name": "noop"}))}
            for h in history[-limit:]]


def normalize_action(action: Any) -> dict:
    if isinstance(action, str):
        action = {"name": "code", "arguments": {}, "code": action}
    if not isinstance(action, dict) or not isinstance(action.get("name"), str):
        raise ValueError("Action must have a string name")
    arguments = action.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("Action arguments must be an object")
    cost = float(action.get("cost", 1.0))
    if not 0 < cost < float("inf"):
        raise ValueError("Action cost must be finite and positive")
    result = {"name": action["name"], "arguments": copy.deepcopy(arguments), "cost": cost}
    if action.get("code") is not None:
        result["code"] = str(action["code"])
    for key in ("read_only", "wma_action", "prob", "description"):
        if key in action:
            result[key] = copy.deepcopy(action[key])
    return result


def noop_action() -> dict:
    return {"name": "noop", "arguments": {}, "cost": 1.0}


class FrozenActor:
    """Fixed API policy or deterministic scripted policy with optional disk cache."""

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.backend = self.config.get("backend", self.config.get("kind", "openai"))
        if self.backend == "fixture":
            self.backend = "scripted"
            self.config.setdefault("actions", {"execute": [
                {"name": "shop.set_shipping", "arguments": {"shipping": "express"}},
                {"name": "shop.set_shipping_preserve_payment", "arguments": {"shipping": "express"}}],
                "repair": [{"name": "shop.pay_order", "arguments": {}}]})
        self.calls = 0
        self.last_cache_hit = False
        self.cache: dict = {}
        self.cache_path = self.config.get("cache_path")
        if self.cache_path and Path(self.cache_path).exists():
            self.cache = json.loads(Path(self.cache_path).read_text())
        if self.backend not in {"openai", "scripted", "cached"}:
            raise ValueError(f"Unknown frozen Actor backend: {self.backend}")

    def _messages(self, state, goals, history, count, mode, instruction) -> list[dict]:
        return [{"role": "system", "content": self.config.get("system_prompt",
                 "Generate legal candidate AppWorld actions as JSON. Return an object with an actions array; "
                 "each action has name, arguments, optional executable Python code, and cost. "
                 "Use only the supplied public API and observations. Protect satisfied requirements. "
                 "Do not claim to have executed an action.")},
                {"role": "user", "content": json.dumps({"instruction": instruction,
                 "state": public_state(state), "goals": public_goals(goals),
                 "history": public_history(history), "candidate_count": count, "mode": mode},
                 ensure_ascii=False, sort_keys=True)}]

    def _request(self, messages: list[dict], count: int) -> list[dict]:
        key_name = self.config.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(key_name)
        if not key:
            raise RuntimeError(f"Frozen Actor requires environment variable {key_name}")
        base = self.config.get("base_url") or os.environ.get(self.config.get("base_url_env", "OPENAI_BASE_URL"), "https://api.openai.com/v1")
        url = base.rstrip("/") + "/chat/completions"
        body = {"model": self.config.get("model") or os.environ.get("ACTOR_MODEL", "gpt-4o-mini"),
                "messages": messages, "temperature": self.config.get("temperature", 0.7),
                "top_p": self.config.get("top_p", 1.0), "n": 1,
                "max_tokens": int(self.config.get("max_tokens", 2048))}
        request = urllib.request.Request(url, json.dumps(body).encode(),
                  {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        # No implicit retry: failed requests consume the same externally managed budget.
        with urllib.request.urlopen(request, timeout=float(self.config.get("timeout", 120))) as response:
            payload = json.load(response)
        content = payload["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        parsed = json.loads(content)
        return parsed.get("actions", []) if isinstance(parsed, dict) else parsed

    def generate(self, state: dict, goals: list[dict], history: list[dict], count: int = 4,
                 mode: str = "execute", instruction: str = "") -> list[dict]:
        self.calls += 1
        messages = self._messages(state, goals, history, count, mode, instruction)
        policy = {"model": self.config.get("model") or os.environ.get("ACTOR_MODEL", "gpt-4o-mini"),
                  "base_url": self.config.get("base_url") or os.environ.get(self.config.get("base_url_env", "OPENAI_BASE_URL")),
                  "temperature": self.config.get("temperature", 0.7), "top_p": self.config.get("top_p", 1.0),
                  "max_tokens": self.config.get("max_tokens", 2048),
                  "policy_revision": self.config.get("policy_revision")}
        key = hashlib.sha256(json.dumps({"messages": messages, "policy": policy}, sort_keys=True).encode()).hexdigest()
        if self.cache_path and Path(self.cache_path).exists():
            self.cache.update(json.loads(Path(self.cache_path).read_text()))
        self.last_cache_hit = key in self.cache
        if self.last_cache_hit:
            raw = self.cache[key]
        elif self.backend == "cached":
            raise KeyError(f"Missing frozen Actor cache entry {key}; API fallback is disabled")
        elif self.backend == "scripted":
            scripted = self.config.get("actions", [noop_action()])
            if isinstance(scripted, dict):
                scripted = scripted.get(mode, scripted.get("execute", [noop_action()]))
            raw = copy.deepcopy(scripted)
        else:
            raw = self._request(messages, count)
            self.cache[key] = raw
            if self.cache_path:
                path = Path(self.cache_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                # A shared policy cache freezes candidates for identical public
                # inputs across methods. Preserve the first completed response
                # even when independent ablation jobs finish concurrently.
                import fcntl
                with path.with_name(path.name + '.lock').open('a') as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    merged = json.loads(path.read_text()) if path.exists() else {}
                    merged.setdefault(key, raw)
                    raw = merged[key]
                    self.cache = merged
                    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
                    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
                    temporary.replace(path)
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        if not isinstance(raw, list):
            raise ValueError("Frozen Actor actions must be an array")
        selected = aggregate_candidates(raw, parse=normalize_action, count=count,
            key=lambda action: json.dumps({k: v for k, v in action.items() if k not in {"prob", "cost"}}, sort_keys=True))
        result = [{**item["action"], "prob": item["prob"]} for item in selected]
        return result or [noop_action()]


def create_actor(config: dict) -> FrozenActor:
    return FrozenActor(config.get("actor", config))
