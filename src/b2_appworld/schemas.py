"""Strict, dependency-free records shared by collection, learning, and execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

LABELS = ("00", "01", "10", "11")
IGNORE_LABEL = -100
RESEARCH_SPLITS = frozenset(("train", "dev"))
ALL_SPLITS = frozenset(("train", "dev", "test_normal", "test_challenge"))


def ensure_research_split(split: str) -> None:
    if split not in RESEARCH_SPLITS:
        raise ValueError(f"Fine-grained collection/analysis is restricted to train/dev, got {split!r}")


@dataclass(frozen=True)
class Goal:
    id: str
    text: str
    evidence_span: str
    kind: str = "state_write"

    def __post_init__(self):
        if not self.id or not self.text or not self.evidence_span:
            raise ValueError("Goal id, text and instruction evidence must be nonempty")
        if self.kind not in {"state_write", "final_answer", "trajectory", "unknown"}:
            raise ValueError(f"Invalid goal kind: {self.kind}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Goal:
        return cls(str(value["id"]), str(value["text"]), str(value["evidence_span"]), str(value.get("kind", value.get("type", "state_write"))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    id: str
    code: str
    target_goal_ids: tuple[str, ...]
    estimated_api_calls: float = 1.0

    def __post_init__(self):
        if not self.id or not self.code.strip():
            raise ValueError("Candidate id and code must be nonempty")
        if not math.isfinite(self.estimated_api_calls) or self.estimated_api_calls < 0:
            raise ValueError("Estimated API cost must be finite and nonnegative")
        if len(self.target_goal_ids) != len(set(self.target_goal_ids)):
            raise ValueError("Duplicate candidate goal ids")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Candidate:
        return cls(str(value["id"]), str(value["code"]), tuple(map(str, value.get("target_goal_ids", ()))), float(value.get("estimated_api_calls", 1)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Example:
    task_id: str
    scenario_id: str
    state_id: str
    split: str
    instruction: str
    history: list[dict[str, str]]
    goals: tuple[Goal, ...]
    candidate: Candidate
    labels: tuple[int, ...]
    source: str = "natural"
    before_fingerprint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        ensure_research_split(self.split)
        if self.scenario_id != self.task_id.split("_")[0]:
            raise ValueError("Scenario id must match task id prefix")
        if not self.goals or len(self.labels) != len(self.goals):
            raise ValueError("One label per nonempty goal set is required")
        if any(x not in (-100, 0, 1, 2, 3) for x in self.labels):
            raise ValueError("Labels must be 00/01/10/11 indices or -100")
        if len({g.id for g in self.goals}) != len(self.goals):
            raise ValueError("Duplicate goals")
        if any(g.evidence_span not in self.instruction for g in self.goals):
            raise ValueError("Goal evidence must be an exact substring of the original instruction")
        if any(g not in {x.id for x in self.goals} for g in self.candidate.target_goal_ids):
            raise ValueError("Candidate refers to an unknown goal")
        if self.source not in {"natural", "synthetic_fixture", "synthetic_fault"}:
            raise ValueError("Unknown evidence origin")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Example:
        fields = dict(value)
        fields["goals"] = tuple(Goal.from_dict(g) for g in fields["goals"])
        fields["candidate"] = Candidate.from_dict(fields["candidate"])
        fields["labels"] = tuple(int(x) for x in fields["labels"])
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
