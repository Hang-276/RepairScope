"""Small, versioned repair knowledge; deployment uses observable conditions only.

The store is deliberately a serial JSON archive. Receipts remain local audit
material; ``snapshot`` and ``rule_prompt`` never expose those receipts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


RULE_FIELDS = {
    "rule_id", "version", "intent", "applicability", "repair", "goal_effects",
    "evidence_ids", "counterexample_ids", "status",
}
STATUSES = {"provisional", "validated", "restricted", "deprecated"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _copy(value: Any) -> Any:
    return json.loads(_json(value))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")


def _strings(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{label} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate values")


def validate_rule(rule: dict, forbidden_literals: Iterable[str] = ()) -> None:
    """Validate the deployable schema, rejecting supplied task-specific literals.

    Evidence identifiers are audit references and are excluded from the literal
    scan. Callers supply the actual task IDs, answers, and entity values to block;
    this function cannot infer which arbitrary strings are private task answers.
    """
    _json(rule)
    _keys(rule, RULE_FIELDS, "rule")
    for key in ("rule_id", "intent"):
        if not isinstance(rule[key], str) or not rule[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if type(rule["version"]) is not int or rule["version"] < 1:
        raise ValueError("rule version must be a positive integer")
    if rule["status"] not in STATUSES:
        raise ValueError("invalid rule status")
    app = rule["applicability"]
    _keys(app, {"observable_conditions", "required_queries", "unknown_policy"}, "applicability")
    if app["unknown_policy"] != "do_not_treat_as_verified":
        raise ValueError("unknown conditions must not be treated as verified")
    if not isinstance(app["observable_conditions"], list):
        raise ValueError("observable_conditions must be a list")
    condition_keys: set[str] = set()
    for condition in app["observable_conditions"]:
        _keys(condition, {"key", "equals"}, "observable condition")
        key = condition["key"]
        if not isinstance(key, str) or not key.strip() or key in condition_keys:
            raise ValueError("observable condition keys must be nonempty and unique")
        condition_keys.add(key)
        if condition["equals"] is None:
            raise ValueError("null is an unknown observation, not a verified condition")
    _strings(app["required_queries"], "required_queries")
    repair = rule["repair"]
    _keys(repair, {"base_template", "conditional_compensation", "read_scope", "write_scope"}, "repair")
    if not isinstance(repair["base_template"], str) or not repair["base_template"].strip():
        raise ValueError("base_template must be nonempty")
    if not isinstance(repair["conditional_compensation"], list):
        raise ValueError("conditional_compensation must be a JSON list")
    for key in ("read_scope", "write_scope"):
        _strings(repair[key], key)
    effects = rule["goal_effects"]
    _keys(effects, {"recovery_claims", "preservation_claims"}, "goal_effects")
    for key in effects:
        _strings(effects[key], key)
    for key in ("evidence_ids", "counterexample_ids"):
        _strings(rule[key], key)
    deployable = {k: v for k, v in rule.items() if k not in {"evidence_ids", "counterexample_ids"}}
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.casefold()]
        if isinstance(value, dict):
            return [text for key, nested in value.items() for text in strings(key) + strings(nested)]
        if isinstance(value, list):
            return [text for nested in value for text in strings(nested)]
        return []
    deployed_strings = strings(deployable)
    for literal in forbidden_literals:
        if not isinstance(literal, str) or not literal.strip():
            raise ValueError("forbidden_literals must contain nonempty strings")
        if any(literal.casefold() in text for text in deployed_strings):
            raise ValueError("task-specific literal found in deployable rule")


def rule_prompt(rules: list[dict]) -> str:
    """Serialize verified mechanism knowledge, omitting audit evidence identifiers."""
    public = []
    for rule in rules:
        validate_rule(rule)
        if rule["status"] not in {"validated", "restricted"}:
            raise ValueError("only validated or restricted rules may be deployed")
        public.append({k: v for k, v in rule.items() if k not in {"evidence_ids", "counterexample_ids"}})
    return "Conditional repair knowledge (use only after checking every observable condition):\n" + _json(public)


class RuleStore:
    """Append immutable knowledge snapshots; a path is one JSON archive file.

    ``commit`` accepts a receipt returned by ``validate_update`` and binds that
    acceptance to the exact rule and parent. Rollback returns a historical copy;
    it never discards versions or changes the active head. Concurrent writers are
    rejected when a loaded archive is stale; callers must serialize experiments.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._archive = {"format_version": 1, "versions": [{"version": 0, "rules": [], "receipt": None}]}
        self._disk_text: str | None = None
        if self.path is not None and self.path.exists():
            self._disk_text = self.path.read_text(encoding="utf-8")
            self._archive = json.loads(self._disk_text)
            self._validate_archive()

    def _validate_archive(self) -> None:
        _keys(self._archive, {"format_version", "versions"}, "knowledge archive")
        versions = self._archive["versions"]
        if self._archive["format_version"] != 1 or not isinstance(versions, list) or not versions:
            raise ValueError("invalid knowledge archive")
        prior: dict[str, dict] = {}
        for index, snapshot in enumerate(versions):
            _keys(snapshot, {"version", "rules", "receipt"}, "stored snapshot")
            if type(snapshot["version"]) is not int or snapshot["version"] != index:
                raise ValueError("knowledge snapshot sequence is invalid")
            if not isinstance(snapshot["rules"], list):
                raise ValueError("stored rules must be a list")
            current: dict[str, dict] = {}
            for rule in snapshot["rules"]:
                validate_rule(rule)
                if rule["rule_id"] in current:
                    raise ValueError("duplicate stored rule identifier")
                current[rule["rule_id"]] = rule
            if index == 0:
                if current or snapshot["receipt"] is not None:
                    raise ValueError("initial knowledge snapshot must be empty")
            else:
                if not set(prior) <= set(current):
                    raise ValueError("stored snapshot deleted a rule")
                changed = [r for key, r in current.items() if key not in prior or r != prior[key]]
                if len(changed) != 1:
                    raise ValueError("each knowledge snapshot must change exactly one rule")
                self._check_commit(changed[0], snapshot["receipt"], prior)
            prior = current

    def snapshot(self) -> dict:
        head = self._archive["versions"][-1]
        return _copy({"version": head["version"], "rules": head["rules"]})

    def retrieve(self, observations: dict, max_rules: int = 3) -> list[dict]:
        if not isinstance(observations, dict):
            raise ValueError("observations must be an explicit public dictionary")
        if type(max_rules) is not int or not 0 <= max_rules <= 3:
            raise ValueError("retrieval limit must be between 0 and 3")
        _json(observations)
        eligible = []
        for rule in self.snapshot()["rules"]:
            if rule["status"] not in {"validated", "restricted"}:
                continue
            conditions = rule["applicability"]["observable_conditions"]
            if all(c["key"] in observations and observations[c["key"]] is not None
                   and _json(observations[c["key"]]) == _json(c["equals"]) for c in conditions):
                eligible.append(rule)
        # Stable ordering is frozen and CPU-only, with no hidden-state ranking.
        eligible.sort(key=lambda r: r["rule_id"])
        return _copy(eligible[:max_rules])

    @staticmethod
    def _check_commit(rule: dict, receipt: dict, existing: dict[str, dict]) -> None:
        validate_rule(rule)
        if not isinstance(receipt, dict) or receipt.get("accepted") is not True:
            raise ValueError("knowledge commit requires an accepted validation receipt")
        old = existing.get(rule["rule_id"])
        if receipt.get("checked_rule_hash") != _digest(rule) or receipt.get("checked_parent_hash") != _digest(old):
            raise ValueError("validation receipt is not bound to this rule and parent")
        if receipt.get("automatic") is not True or receipt.get("task_scored") is not True:
            raise ValueError("commit must be automatic and follow original task scoring")
        if not receipt.get("frozen_before") or receipt.get("frozen_before") != receipt.get("frozen_after"):
            raise ValueError("frozen protocol changed")
        expected = 1 if old is None else old["version"] + 1
        if rule["version"] != expected:
            raise ValueError("rule version must increment by exactly one")
        if old is not None:
            for key in ("evidence_ids", "counterexample_ids"):
                if not set(old[key]) <= set(rule[key]):
                    raise ValueError(f"knowledge commit cannot delete {key}")
        candidate = dict(existing)
        candidate[rule["rule_id"]] = rule
        if sum(r["status"] != "deprecated" for r in candidate.values()) > 32:
            raise ValueError("active rule limit of 32 exceeded")

    def commit(self, rule: dict, receipt: dict) -> dict:
        rule, receipt = _copy(rule), _copy(receipt)
        snapshot = self.snapshot()
        existing = {r["rule_id"]: r for r in snapshot["rules"]}
        self._check_commit(rule, receipt, existing)
        existing[rule["rule_id"]] = rule
        stored = {"version": snapshot["version"] + 1, "rules": list(existing.values()), "receipt": receipt}
        archive = _copy(self._archive)
        archive["versions"].append(stored)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            actual = self.path.read_text(encoding="utf-8") if self.path.exists() else None
            if actual != self._disk_text:
                raise ValueError("knowledge archive changed; reload before committing")
            serialized = _json(archive) + "\n"
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
                    temporary = handle.name
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                temporary = None
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
            self._disk_text = serialized
        self._archive = archive
        return self.snapshot()

    def rollback(self, version: int) -> dict:
        if type(version) is not int or not 0 <= version < len(self._archive["versions"]):
            raise ValueError("unknown knowledge snapshot version")
        snapshot = self._archive["versions"][version]
        return _copy({"version": snapshot["version"], "rules": snapshot["rules"]})
