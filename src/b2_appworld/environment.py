"""Lazy, local-only AppWorld construction and deterministic replay receipts.

Only this module touches AppWorld's process-global state. Collection constructs a
new process for every world; online execution constructs a single task world.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

from .schemas import ALL_SPLITS, ensure_research_split


class AppWorldFactory:
    def __init__(self, root: str | Path, seed: int = 100):
        self.root = Path(root).expanduser().resolve()
        self.seed = seed

    @contextmanager
    def create(self, task_id: str, experiment_name: str, split: str,
               full_ground_truth: bool = False) -> Iterator[Any]:
        if split not in ALL_SPLITS:
            raise ValueError(f"Unknown dataset split: {split}")
        if full_ground_truth:
            ensure_research_split(split)
        if not experiment_name or any(p in experiment_name for p in ("..", "/", "\\")):
            raise ValueError("experiment_name must be a single nonempty directory name")
        # Validate membership using ID lists only, without opening test contents.
        from .datasets import load_split_ids
        if task_id not in load_split_ids(self.root, split):
            raise ValueError(f"Task {task_id} is not in split {split}")
        os.environ["APPWORLD_ROOT"] = str(self.root)
        from appworld.common.path_store import path_store
        from appworld import AppWorld
        path_store.update_root(str(self.root))
        world = AppWorld(
            task_id=task_id, experiment_name=experiment_name,
            random_seed=self.seed, load_ground_truth=True,
            ground_truth_mode="full" if full_ground_truth else "minimal",
            # Controller enforces the 50 executed-code-block budget. Offline
            # replay has its own length and may also execute one candidate.
            max_interactions=1000,
        )
        try:
            yield world
        finally:
            world.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    raise TypeError(f"Cannot fingerprint value of type {type(value).__name__}")


def stable_digest(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_changes(changes: dict[str, Any]) -> dict[str, Any]:
    """Keep ordered SQL + parameters, discard instrumentation envelope only.

    Task timestamps, IDs, row order and partial writes are NEVER scrubbed. In
    particular, stripping arbitrary timestamp fields would hide state mismatch.
    AppWorld supplies (statement, parameters, executemany) triples. An envelope
    accepted for adapters may add process/experiment context, which is ignored.
    """
    result = {}
    for app, entries in sorted(changes.items()):
        normalized = []
        for entry in entries:
            if isinstance(entry, dict):
                if not {"statement", "parameters"}.issubset(entry):
                    raise ValueError("Unknown DB change record; refusing lossy fingerprint")
                entry = [entry["statement"], entry["parameters"], entry.get("executemany", False)]
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise ValueError("DB change must be SQL, parameters, executemany")
            normalized.append([str(entry[0]).strip(), _jsonable(entry[1]), bool(entry[2])])
        result[str(app)] = normalized
    return result


def fingerprint_world(world: Any) -> str:
    """Hash actual tracked DB writes, independent of experiment paths/logs.

    A replay digest is meaningful only with identical task input and code prefix;
    collection separately checks both, along with every replay observation.
    """
    changes = world.models.get_tracked_db_changes()
    return stable_digest({"task_id": world.task_id, "changes": semantic_changes(changes)})


def tracker_status(tracker: Any) -> dict[str, bool]:
    """Extract requirement truth values; never expose failure traces to models."""
    data = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
    if not isinstance(data, dict) or not {"passes", "failures"}.issubset(data):
        raise ValueError("Evaluator did not return full requirement statuses")
    status: dict[str, bool] = {}
    for key, passed in (("passes", True), ("failures", False)):
        for entry in data[key]:
            requirement = str(entry["requirement"])
            if requirement in status:
                raise ValueError("Duplicate evaluator requirement")
            status[requirement] = passed
    return status
