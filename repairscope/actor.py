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
    from .schema import GOAL_KEYS
    return [{k: copy.deepcopy(g[k]) for k in sorted(GOAL_KEYS) if k in g} for g in goals]


def public_history(history: list[dict], limit: int = 8) -> list[dict]:
    return [{"state": public_state(h.get("state", {})),
             "action": normalize_action(h.get("action", {"name": "noop"}))}
            for h in history[-limit:]]


def normalize_action(action: Any) -> dict:
    if isinstance(action, str):
        action = {"name": "code", "arguments": {}, "code": action}
    if isinstance(action, dict):
        action = copy.deepcopy(action)
        # Served models sometimes use a descriptive key for the executable block.
        if action.get("code") is None:
            for alias in ("executable_code", "executable", "program"):
                if isinstance(action.get(alias), str) and action[alias].strip():
                    action["code"] = action[alias]
                    break
        if not isinstance(action.get("name"), str) or not action.get("name"):
            if action.get("code"):
                action["name"] = "code"
            else:
                raise ValueError("Action must have a string name")
        # A qualitative cost is a model formatting slip, not an environment cost.
        # Actual execution cost is settled from the environment response.
        if isinstance(action.get("cost"), str):
            try:
                action["cost"] = float(action["cost"])
            except ValueError:
                action["cost"] = 1.0
    if not isinstance(action, dict) or not isinstance(action.get("name"), str):
        raise ValueError("Action must have a string name")
    arguments = action.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("Action arguments must be an object")
    cost = float(action.get("cost", 1.0))
    if not 0 < cost < float("inf"):
        raise ValueError("Action cost must be finite and positive")
    result = {"name": action["name"], "arguments": copy.deepcopy(arguments), "cost": cost}
    if action.get("code") is not None and str(action["code"]).strip():
        result["code"] = str(action["code"])
    for key in ("read_only", "wma_action", "prob", "description"):
        if key in action:
            result[key] = copy.deepcopy(action[key])
    return result


def noop_action() -> dict:
    return {"name": "noop", "arguments": {}, "cost": 1.0}


ACTION_FORMAT_GUIDE = (
    'ACTION FORMAT (mandatory): return {"actions": [...]} with each item exactly '
    '{"name": "<app>.<api_name>", "arguments": {...}, "code": "<complete executable Python block or omitted>", "cost": 1}. '
    'Use the exact endpoint names listed in public_apis. cost must be a number. '
    'LEGAL CODE: code may call only apis.<app>.<api_name>(...). Never import or use requests, urllib, http, socket, '
    'os, subprocess or any network/file access; never invent URLs or placeholder values. '
    'IMPORTANT: inspect the state and history. If the last action failed, do not repeat it; choose a different legal action. '
    'If a previous action already succeeded and its output is in the state, do not repeat that action; use the output to advance. '
    'When the task asks a question and the answer is present in the state, call apis.supervisor.complete_task(answer=<answer>). '
    'When the task asks for state changes, apply them and then call apis.supervisor.complete_task(). '
    'Never call an app detail/write API before that app login succeeds in the same code block. '
    'If an API needs an entity id, first call the app search/list API and read the real id from the output; never pass a placeholder id. '
    'If a required parameter is unclear, call apis.api_docs.show_api_doc(app_name=..., api_name=...) and use its exact name and type. '
    'Every failed call is recorded and charged; read its error, do not repeat it, and choose a different legal action. '
    'Unless the state already contains a valid access_token for the app, a code block must obtain it first: '
    'call apis.supervisor.show_profile() and apis.supervisor.show_account_passwords(), find the row whose account_name '
    'matches the app, call that app login API with the supervisor email and the returned password, and pass the returned '
    'access_token to the task API in the SAME code block. The state contains the exact values returned by earlier calls. '
    'Do not guess an API response shape in advance: execute one API, observe its returned value in the next state, then continue. '
    'Example code block (replace the final spotify API with the needed app/API): '
    '"code":"import json\\nprofile = apis.supervisor.show_profile()\\npasswords = apis.supervisor.show_account_passwords()\\n'
    'password = [row[\\"password\\"] for row in passwords if row[\\"account_name\\"] == \\"spotify\\"][0]\\n'
    'login = apis.spotify.login(username=profile[\\"email\\"], password=password)\\n'
    'print(json.dumps(apis.spotify.show_liked_songs(access_token=login[\\"access_token\\"])))"'
)


def public_api_catalog(metadata: dict) -> list[str]:
    """Compact public API catalog; response payloads remain runtime observations."""
    rows = []
    for app, apis in sorted((metadata.get("api_docs") or {}).items()):
        if not isinstance(apis, dict):
            continue
        for name, doc in sorted(apis.items()):
            if not isinstance(doc, dict):
                continue
            params = ", ".join(
                f"{item.get('name')}:{item.get('type')}" + ("" if item.get("required") else "?")
                for item in doc.get("parameters", []) if isinstance(item, dict))
            rows.append(f"{app}.{name}({params}) - {doc.get('description', '')}")
    return rows


_PLACEHOLDER_MARKERS = ('obtained', 'your_token', 'placeholder', 'dummy_token', 'token_here',
                        '<access_token>', 'access_token_here')


def candidate_is_legal(action: dict) -> bool:
    """Static legality gate for the public code block.

    This is not a planner: it only rejects a candidate whose executable block
    uses a placeholder token or whose named app API is absent from the code.
    """
    code = action.get('code')
    if not code:
        return True
    calls = re.findall(r'apis\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(', code)
    if not calls:
        return False
    name = action.get('name', '')
    if name != 'code' and '.' in name:
        if tuple(name.split('.', 1)) not in calls:
            return False
    lowered = code.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return False
    if re.search(r"access_token\s*=\s*['\"]access_token['\"]", code):
        return False
    return True


def actor_instruction(metadata: dict) -> str:
    """Render public task context within the served model's context budget.

    AppWorld's full api_docs include every response schema (hundreds of KB) and
    cannot fit any candidate context. The catalog keeps the same public endpoint
    names, descriptions, and parameters; actual responses arrive through legal
    execution, so no information boundary is changed.
    """
    payload = {
        "instruction": metadata.get("instruction", ""),
        "supervisor": metadata.get("supervisor", {}),
        "allowed_apps": metadata.get("allowed_apps", []),
        "public_apis": public_api_catalog(metadata),
        "action_format": ACTION_FORMAT_GUIDE,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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
        self.parse_repairs = 0
        self.rejected_candidates = 0
        self.rejected_illegal = 0
        self.retries = 0
        self.http_calls = 0
        self.cache_hits = 0
        self.last_http_calls = 0
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
                "max_tokens": int(self.config.get("max_tokens", 4096))}
        # A truncated completion is a formatting failure, not a candidate. Retry
        # once or twice with a larger output cap; both arms share this policy.
        last_error = None
        for attempt in range(3):
            self.http_calls += 1
            self.last_http_calls += 1
            request = urllib.request.Request(url, json.dumps(body).encode(),
                      {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(request, timeout=float(self.config.get("timeout", 120))) as response:
                payload = json.load(response)
            try:
                content = payload["choices"][0]["message"]["content"]
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
                parsed = json.loads(content)
                return parsed.get("actions", []) if isinstance(parsed, dict) else parsed
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                last_error = error
                if attempt == 2:
                    break
                self.retries += 1
                body["max_tokens"] = min(8192, int(body["max_tokens"]) * 2)
        raise ValueError(f"Frozen Actor returned no parseable JSON actions: {last_error}")

    def stats(self) -> dict:
        return {"calls": self.calls, "http_calls": self.http_calls,
                "retries": self.retries, "cache_hits": self.cache_hits,
                "rejected_candidates": self.rejected_candidates,
                "rejected_illegal": self.rejected_illegal}

    def generate(self, state: dict, goals: list[dict], history: list[dict], count: int = 4,
                 mode: str = "execute", instruction: str = "") -> list[dict]:
        self.calls += 1
        self.last_http_calls = 0
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
            self.cache_hits += 1
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
        parsed = []
        for item in raw:
            try:
                action = normalize_action(item)
            except (ValueError, TypeError):
                # One malformed candidate must not discard the other legal
                # proposals; the rejection is counted for the run receipt.
                self.rejected_candidates += 1
                continue
            if not candidate_is_legal(action):
                self.rejected_candidates += 1
                self.rejected_illegal += 1
                continue
            parsed.append(action)
        selected = aggregate_candidates(parsed, parse=lambda action: action, count=count,
            key=lambda action: json.dumps({k: v for k, v in action.items() if k not in {"prob", "cost"}}, sort_keys=True))
        result = [{**item["action"], "prob": item["prob"]} for item in selected]
        return result or [noop_action()]


def create_actor(config: dict) -> FrozenActor:
    return FrozenActor(config.get("actor", config))
