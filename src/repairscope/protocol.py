"""Frozen P0/P1 limits. No import opens task data or starts a model."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

APPWORLD_COMMIT = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"
SOURCES = {"natural_regression", "natural_failure", "natural_incomplete",
           "cf_intervention", "official_generator", "synthetic_fixture"}
NATURAL = {"natural_regression", "natural_failure"}
DEFAULT = {
    "schema_version": 1, "name": "RepairScope-Evo-v1", "stage": "P0/P1",
    "eligibility": "EvoEligibility-v1", "learning_split": "train",
    "appworld_commit": APPWORLD_COMMIT,
    "model": {"name": "Qwen/Qwen3-8B-AWQ", "frozen": True, "context_length": 32000,
              "max_output_tokens": 3000, "enable_thinking": False},
    "actor": {"max_blocks": 50, "max_output_tokens": 100000,
              "triggers": ["visible_error", "every_5_blocks", "before_completion"]},
    "p1": {"initial_tasks_max": 5, "initial_states_max": 15,
           "initial_branches_max": 180, "extensions_max": 1,
           "extension_states_max": 15, "extension_branches_max": 180,
           "candidates_per_state": 4, "branches_per_event": 12,
           "model_calls_per_event": 6, "output_tokens_per_event": 8000,
           "events_per_task": 2, "explanations_max": 3,
           "transfer_cases_max": 3, "transfer_repeats_max": 3,
           "transfer_evaluations_max": 18},
    "knowledge": {"active_rules_max": 32, "retrieval_max": 3, "injection_tokens_max": 2048},
    "execution": {"workers": 1, "score_before_learning": True,
                  "test_enabled": False, "p2_enabled": False,
                  "official_generator_enabled": False},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def default_protocol():
    return deepcopy(DEFAULT)


def read_json(path):
    from .llm import strict_json_loads
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def require_train(value):
    if value != "train":
        raise ValueError("Persistent learning is restricted to train; dev/test contents are sealed")


def check_human_edits(record):
    for key in ("human_task_edits", "human_rule_edits", "human_intervention_edits"):
        if type(record.get(key)) is not int or record[key] != 0:
            raise ValueError(f"{key} must be explicitly zero for main mechanism evidence")


class BranchBudget:
    def __init__(self, total=180, per_event=12):
        self.total, self.per_event = total, per_event
        self.used = 0
        self.events = {}

    def reserve(self, event_id):
        from .budget import BudgetExceeded
        if self.used >= self.total or self.events.get(event_id, 0) >= self.per_event:
            raise BudgetExceeded("P1 shadow branch budget exhausted")
        self.used += 1  # Failure and timeout consume an attempted branch too.
        self.events[event_id] = self.events.get(event_id, 0) + 1


def round_robin(tasks):
    groups = {}
    for task in sorted(tasks, key=lambda t: t["task_id"]):
        groups.setdefault(task["scenario_id"], []).append(task["task_id"])
    result = []
    for index in range(max((len(g) for g in groups.values()), default=0)):
        for scenario in sorted(groups):
            if index < len(groups[scenario]):
                result.append(groups[scenario][index])
    return result
