"""Dataset receipts and split guards; test task contents are never inspected."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any

from .environment import stable_digest
from .schemas import ALL_SPLITS

DATA_VERSION = "0.2.0"
EXPECTED_COUNTS = {"train": 90, "dev": 57, "test_normal": 168, "test_challenge": 417}
UPSTREAM_COMMIT = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"


def load_split_ids(root: str | Path, split: str) -> list[str]:
    if split not in ALL_SPLITS:
        raise ValueError(f"Unknown dataset split: {split}")
    path = Path(root) / "data" / "datasets" / f"{split}.txt"
    ids = [line.split(":", 1)[0].strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task IDs in {split}")
    if any(not re.fullmatch(r"[A-Za-z0-9]+_[1-3]", x) for x in ids):
        raise ValueError(f"Malformed task ID in {split}")
    return ids


def dataset_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    by_split = {split: load_split_ids(root, split) for split in EXPECTED_COUNTS}
    all_ids: set[str] = set()
    all_scenarios: set[str] = set()
    split_records = {}
    for split, ids in by_split.items():
        if len(ids) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]} IDs, found {len(ids)}")
        scenarios = {x.split("_")[0] for x in ids}
        if all_ids.intersection(ids) or all_scenarios.intersection(scenarios):
            raise ValueError(f"Task/scenario leakage into {split}")
        if any({f"{scenario}_{n}" for n in (1, 2, 3)} - set(ids) for scenario in scenarios):
            raise ValueError(f"{split}: scenarios must contain all three task instances")
        all_ids.update(ids)
        all_scenarios.update(scenarios)
        id_file = root / "data" / "datasets" / f"{split}.txt"
        split_records[split] = {"count": len(ids), "scenario_count": len(scenarios),
                                "ids_sha256": stable_digest(sorted(ids)),
                                "scenarios_sha256": stable_digest(sorted(scenarios)),
                                "list_file_sha256": hashlib.sha256(id_file.read_bytes()).hexdigest()}
    # Read only train/dev task specifications. No test instructions, evaluator
    # requirements, private data, solutions or per-test labels enter the receipt.
    versions: set[str] = set()
    specs_hashes = []
    for split in ("train", "dev"):
        for task_id in by_split[split]:
            path = root / "data" / "tasks" / task_id / "specs.json"
            contents = path.read_bytes()
            versions.add(str(json.loads(contents)["db_version"]))
            specs_hashes.append([task_id, hashlib.sha256(contents).hexdigest()])
    if versions != {DATA_VERSION}:
        raise ValueError(f"Expected AppWorld data {DATA_VERSION}, found train/dev versions {sorted(versions)}")
    return {"schema_version": 1, "expected_data_version": DATA_VERSION,
            "observed_train_dev_versions": sorted(versions),
            "upstream_commit": UPSTREAM_COMMIT, "splits": split_records,
            "split_root_sha256": stable_digest(split_records),
            "train_dev_specs_root_sha256": stable_digest(sorted(specs_hashes)),
            "test_contents_inspected": False,
            "test_version_verification": "deferred to AppWorld task loader at native aggregate evaluation"}


def scenario_holdout(examples: list[Any], heldout_scenarios: set[str]) -> tuple[list[Any], list[Any]]:
    """Split every state/candidate of one scenario together; never split rows."""
    train, validation = [], []
    for example in examples:
        if example.split != "train":
            raise ValueError("Training-side scenario holdout requires train examples only")
        (validation if example.scenario_id in heldout_scenarios else train).append(example)
    if not train or not validation:
        raise ValueError("Scenario holdout must produce nonempty train and validation sets")
    return train, validation


def collection_quality_report(paths: str | Path | list[str | Path], output: str | Path | None = None,
                              expected_candidate_ids: tuple[str, ...] =
                              ("original", "local", "composite", "broad")) -> dict[str, Any]:
    """Summarize coverage without reading any evaluator or test task contents.

    Structural validity and class presence are diagnostics, not a certificate of
    correct goal/checker semantics. That needs the mapping audit described in
    docs/LABEL_AUDIT.md before GPU training or reporting paper results.
    """
    from .io import atomic_json, read_jsonl
    from .schemas import Example, IGNORE_LABEL, LABELS
    files = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    pools: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    split_scenarios: dict[str, set[str]] = defaultdict(set)
    split_tasks: dict[str, set[str]] = defaultdict(set)
    class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    goal_kind_counts: Counter[str] = Counter()
    per_goal: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    candidate_counts: Counter[str] = Counter()
    provenance = []
    for path in files:
        receipt_path = Path(str(path) + ".receipt.json")
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        provenance.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "receipt_present": receipt is not None,
                           "receipt_evidence_origin": receipt.get("evidence_origin") if receipt else None})
        for row in read_jsonl(path):
            example = Example.from_dict(row)
            key = example.split, example.task_id, example.state_id, example.candidate.id
            if key in seen:
                raise ValueError(f"Duplicate state/candidate in quality data: {key}")
            seen.add(key)
            pools[key[:3]].append(example)
            split_scenarios[example.split].add(example.scenario_id)
            split_tasks[example.split].add(example.task_id)
            source_counts[example.source] += 1
            candidate_counts[example.candidate.id] += 1
            for goal, label in zip(example.goals, example.labels):
                label_name = "unknown" if label == IGNORE_LABEL else LABELS[label]
                class_counts[label_name] += 1
                per_goal[(example.task_id, goal.id)][label_name] += 1
                goal_kind_counts[goal.kind] += 1
    if split_scenarios["train"] & split_scenarios["dev"]:
        raise ValueError("Scenario leakage between train and dev collection files")
    complete = recovery_states = damage_states = joint_states = choice_states = 0
    fully_labeled_complete = 0
    recovery_complete = damage_complete = 0
    for key, rows in pools.items():
        anchor = rows[0]
        for row in rows[1:]:
            if (row.instruction != anchor.instruction or row.goals != anchor.goals or
                row.history != anchor.history or row.before_fingerprint != anchor.before_fingerprint):
                raise ValueError(f"Candidate pool does not share identical model input/state: {key}")
        # Shared known goals must agree on pre-candidate satisfaction.
        for index in range(len(anchor.goals)):
            before_bits = {row.labels[index] // 2 for row in rows if row.labels[index] != IGNORE_LABEL}
            if len(before_bits) > 1:
                raise ValueError(f"Candidate pool has inconsistent before-goal labels: {key}")
        is_complete = {row.candidate.id for row in rows} == set(expected_candidate_ids)
        has_recovery = any(1 in row.labels for row in rows)
        has_damage = any(2 in row.labels for row in rows)
        complete += is_complete
        fully_labeled_complete += is_complete and all(IGNORE_LABEL not in row.labels for row in rows)
        recovery_states += has_recovery
        damage_states += has_damage
        joint_states += has_recovery and has_damage
        recovery_complete += is_complete and has_recovery
        damage_complete += is_complete and has_damage
        outcomes = {(row.labels.count(1), row.labels.count(2)) for row in rows}
        choice_states += has_recovery and len(outcomes) > 1
    total = sum(class_counts.values())
    known = total - class_counts["unknown"]
    missing = [label for label in LABELS if not class_counts[label]]
    warnings = []
    if not seen:
        warnings.append("No accepted examples")
    if source_counts["synthetic_fixture"]:
        warnings.append("Contains synthetic fixtures; exclude from all AppWorld scientific results")
    if any(not item["receipt_present"] or item["receipt_evidence_origin"] != "AppWorld" for item in provenance):
        warnings.append("At least one source lacks an AppWorld collection receipt")
    if missing:
        warnings.append("Missing known transition classes: " + ", ".join(missing))
    if not complete:
        warnings.append("No complete four-candidate pools")
    if not joint_states:
        warnings.append("No state with both recovery and damage opportunities across candidates")
    report = {"schema_version": 1, "num_examples": len(seen), "num_states": len(pools),
              "num_task_goals": len(per_goal), "known_label_count": known,
              "unknown_label_fraction": class_counts["unknown"] / total if total else None,
              "class_counts": {label: class_counts[label] for label in (*LABELS, "unknown")},
              "source_example_counts": dict(source_counts), "goal_kind_label_counts": dict(goal_kind_counts),
              "candidate_example_counts": dict(candidate_counts),
              "expected_candidate_ids": list(expected_candidate_ids),
              "complete_candidate_pools": complete, "incomplete_candidate_pools": len(pools) - complete,
              "fully_labeled_complete_candidate_pools": fully_labeled_complete,
              "states_with_01": recovery_states, "states_with_10": damage_states,
              "states_with_both_01_and_10": joint_states,
              "complete_pools_with_01": recovery_complete, "complete_pools_with_10": damage_complete,
              "states_with_differential_recovery_damage_counts": choice_states,
              "splits": {split: {"tasks": len(split_tasks[split]),
                                  "scenarios": len(split_scenarios[split]),
                                  "scenario_ids": sorted(split_scenarios[split])}
                         for split in ("train", "dev")},
              "per_task_goal": [{"task_id": task, "goal_id": goal,
                                 "class_counts": {label: counts[label] for label in (*LABELS, "unknown")}}
                                for (task, goal), counts in sorted(per_goal.items())],
              "source_files": provenance, "warnings": warnings,
              "semantic_mapping_audit_required": True,
              "interpretation": "Coverage diagnostic only; class presence and valid replay do not certify goal-mapping truth."}
    if output is not None:
        atomic_json(output, report)
    return report
