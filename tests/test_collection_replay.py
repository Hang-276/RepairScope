"""Synthetic fake-world checks; these do not qualify a live AppWorld run."""
from contextlib import contextmanager
import json
import os

import pytest

from b2_appworld.collection import (build_mappings_from_trajectories, collect_from_trajectories,
    create_mapping, mapped_goal_status, transition_labels, _trajectory_paths)
from b2_appworld.environment import semantic_changes, stable_digest
from b2_appworld.io import read_jsonl
from b2_appworld.schemas import Goal

_CREATED = 0


class FakeWorld:
    def __init__(self, root, experiment):
        self.task_id = "fake_1"
        self.root, self.experiment = root, experiment
        self.x, self.y = 0, 1
        self.changes = []
        self.models = self
        self.has_python_local = False

    def get_tracked_db_changes(self):
        # Only SQL and values are semantic; process/path context must not cause
        # the same source prefix to get distinct fingerprints.
        return {"app": [{"statement": sql, "parameters": params,
                         "executemany": False, "context": {"pid": os.getpid(), "experiment": self.experiment}}
                        for sql, params in self.changes]}

    def execute(self, code):
        if code == "partial_fail":
            self.x = 1
            self.has_python_local = True
            self.changes.append(("UPDATE goals SET x=?", [1]))
            if str(self.root).endswith("mismatch") and self.experiment.endswith("_c1"):
                self.x = 9
                self.changes.append(("UPDATE goals SET x=?", [9]))
            return "Execution failed after writing x=1"
        if code == "repair":
            if not self.has_python_local or self.x != 1:
                return "Execution failed: lost partial write or Python local"
            self.x = 2
            self.changes.append(("UPDATE goals SET x=?", [2]))
            return "repaired"
        if code == "break_other":
            self.y = 0
            self.changes.append(("UPDATE goals SET y=?", [0]))
            return "broke other goal"
        if code == "nondeterministic":
            return str(os.getpid())
        raise ValueError(f"Unsupported fake code: {code}")

    def evaluate(self):
        truth = {"x is two": self.x == 2, "y is one": self.y == 1}
        return {"passes": [{"requirement": r} for r, status in truth.items() if status],
                "failures": [{"requirement": r} for r, status in truth.items() if not status]}

    def save(self):
        pass


class FakeFactory:
    def __init__(self, root, seed):
        self.root = root

    @contextmanager
    def create(self, task_id, experiment_name, split, full_ground_truth=False):
        global _CREATED
        _CREATED += 1
        assert _CREATED == 1, "Reused a process containing AppWorld global state"
        assert full_ground_truth and split in {"train", "dev"}
        yield FakeWorld(self.root, experiment_name)


def make_trajectory(tmp_path):
    instruction = "Set x to two and preserve y at one."
    goals = [Goal("x", "Set x to two", "Set x to two"),
             Goal("y", "preserve y at one", "preserve y at one")]
    value = {"task_id": "fake_1", "split": "train", "instruction": instruction,
             "source": "synthetic_fixture", "goals": [g.to_dict() for g in goals],
             "steps": [{"code": "partial_fail", "output": "Execution failed after writing x=1"}],
             "checkpoints": [{"state_id": "s1", "prefix_length": 1,
                 "history": [{"role": "user", "content": instruction},
                             {"role": "assistant", "content": "partial_fail"},
                             {"role": "tool", "content": "Execution failed after writing x=1"}],
                 "candidates": [{"id": "repair", "code": "repair", "target_goal_ids": ["x"]},
                                {"id": "break", "code": "break_other", "target_goal_ids": ["x", "y"]}]}]}
    directory = tmp_path / "trajectories"
    directory.mkdir()
    (directory / "fake_1.json").write_text(json.dumps(value))
    return directory, value, goals


def mapper(**kwargs):
    assert kwargs["requirements"] == ["x is two", "y is one"]
    assert set(kwargs) == {"instruction", "goals", "requirements"}
    return {"goal_requirements": {"x": {"requirements": ["x is two"], "complete": True},
                                   "y": {"requirements": ["y is one"], "complete": True}}}


def test_replays_failed_partial_writes_and_python_state_in_isolated_processes(tmp_path):
    directory, trajectory, goals = make_trajectory(tmp_path)
    # An injected fake factory must not inherit a caller's optimistic label.
    trajectory["source"] = "natural"
    (directory / "fake_1.json").write_text(json.dumps(trajectory))
    mapping_file, output = tmp_path / "private_mapping.json", tmp_path / "examples.jsonl"
    factory = f"{__name__}:FakeFactory"
    built = build_mappings_from_trajectories(directory, mapping_file, tmp_path, mapper,
                                            factory_spec=factory, timeout=15)
    assert built["tasks"] == 1
    receipt = collect_from_trajectories(directory, mapping_file, output, tmp_path,
                                         factory_spec=factory, timeout=15)
    examples = read_jsonl(output)
    assert receipt["evidence_origin"] == "synthetic_fixture"
    assert receipt["num_examples"] == 2
    assert examples[0]["labels"] == [1, 3]  # repaired x, retained y
    assert examples[1]["labels"] == [0, 2]  # did not inherit x=2 from sibling
    assert all(row["source"] == "synthetic_fixture" for row in examples)
    assert examples[0]["before_fingerprint"] == examples[1]["before_fingerprint"]
    assert "x is two" not in output.read_text()  # checker stays labeler-only
    states = read_jsonl(str(output) + ".states.jsonl")
    assert states[0]["accepted_count"] == 2


def test_observation_mismatch_rejects_entire_state(tmp_path):
    directory, trajectory, goals = make_trajectory(tmp_path)
    trajectory["steps"][0]["output"] = "wrong observation"
    (directory / "fake_1.json").write_text(json.dumps(trajectory))
    mapping = create_mapping("fake_1", "train", trajectory["instruction"], goals,
                             ["x is two", "y is one"], mapper)
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"schema_version": 1, "tasks": {"fake_1": mapping}}))
    output = tmp_path / "examples.jsonl"
    receipt = collect_from_trajectories(directory, mapping_file, output, tmp_path,
                                        factory_spec=f"{__name__}:FakeFactory", timeout=15)
    assert receipt["num_examples"] == 0
    assert receipt["num_rejected_candidates"] == 2
    assert "mismatch" in read_jsonl(str(output) + ".states.jsonl")[0]["error"]


def test_db_fork_mismatch_rejected_even_when_observations_match(tmp_path):
    directory, trajectory, goals = make_trajectory(tmp_path)
    trajectory["checkpoints"][0]["candidates"][1]["estimated_api_calls"] = 9
    (directory / "fake_1.json").write_text(json.dumps(trajectory))
    mapping = create_mapping("fake_1", "train", trajectory["instruction"], goals,
                             ["x is two", "y is one"], mapper)
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"schema_version": 1, "tasks": {"fake_1": mapping}}))
    output = tmp_path / "examples.jsonl"
    receipt = collect_from_trajectories(directory, mapping_file, output, tmp_path / "mismatch",
                                        factory_spec=f"{__name__}:FakeFactory", timeout=15)
    assert receipt["num_examples"] == 1
    assert receipt["num_rejected_candidates"] == 1
    accepted = read_jsonl(output)[0]
    assert accepted["metadata"]["candidate_group_max_cost"] == 9
    assert accepted["metadata"]["candidate_pool_ids"] == ["repair", "break"]
    assert accepted["metadata"]["candidate_pool_size"] == 2
    candidate = read_jsonl(str(output) + ".states.jsonl")[0]["candidates"][1]
    assert candidate["accepted"] is False
    assert "Fork before-state" in candidate["error"]


def test_unknown_checker_and_non_state_goals_are_masked():
    goals = (Goal("a", "save", "save"), Goal("b", "answer", "answer", "final_answer"))
    groups = {"a": {"requirements": ["known", "missing"], "complete": True},
              "b": {"requirements": ["known"], "complete": True}}
    assert mapped_goal_status(goals, groups, {"known": True}) == [None, None]
    assert transition_labels([None, False, True], [True, True, False]) == (-100, 1, 2)


def test_mapping_rejects_test_and_non_instruction_evidence():
    with pytest.raises(ValueError, match="train/dev"):
        create_mapping("fake_1", "test_normal", "save", [Goal("a", "save", "save")], ["r"], mapper)
    with pytest.raises(ValueError, match="evidence"):
        create_mapping("fake_1", "train", "save", [Goal("a", "other", "other")], ["r"], mapper)


def test_joint_requirement_cannot_supply_independent_goal_labels():
    goals = [Goal("a", "save x", "save x"), Goal("b", "save y", "save y")]
    def joint_mapper(**kwargs):
        return {g: {"requirements": ["both x and y saved"], "complete": True} for g in ("a", "b")}
    mapping = create_mapping("fake_1", "train", "save x and save y", goals,
                             ["both x and y saved"], joint_mapper)
    assert not mapping["goal_requirements"]["a"]["complete"]
    assert not mapping["goal_requirements"]["b"]["complete"]


def test_semantic_fingerprint_drops_context_but_keeps_task_values():
    a = {"app": [{"statement": " UPDATE x SET timestamp=? ", "parameters": ["2023-01-01"],
                   "executemany": False, "context": {"pid": 1}}]}
    b = {"app": [["UPDATE x SET timestamp=?", ["2023-01-01"], False]]}
    assert stable_digest(semantic_changes(a)) == stable_digest(semantic_changes(b))
    b["app"][0][1] = ["2023-01-02"]
    assert stable_digest(semantic_changes(a)) != stable_digest(semantic_changes(b))
    with pytest.raises(ValueError):
        semantic_changes({"app": [{"surprise": "data"}]})


def test_trajectory_discovery_supports_runner_layout_and_ignores_summaries(tmp_path):
    task = tmp_path / "run" / "tasks" / "fake_1"
    task.mkdir(parents=True)
    trajectory = task / "trajectory.json"
    trajectory.write_text("{}")
    (tmp_path / "run" / "run.json").write_text("{}")
    assert _trajectory_paths(tmp_path / "run") == [trajectory]
    assert _trajectory_paths(tmp_path / "run" / "tasks") == [trajectory]
    other = tmp_path / "other"
    other.mkdir()
    flat, _, _ = make_trajectory(other)
    (flat / "run.json").write_text("{}")
    (flat / "examples.receipt.json").write_text("{}")
    assert _trajectory_paths(flat) == [flat / "fake_1.json"]


def test_ineligible_tasks_are_skipped_before_goal_validation(tmp_path):
    directory, trajectory, _ = make_trajectory(tmp_path)
    trajectory.update({"goals": [], "checkpoints": []})
    (directory / "fake_1.json").write_text(json.dumps(trajectory))
    mapping_file = tmp_path / "mapping.json"
    result = build_mappings_from_trajectories(directory, mapping_file, tmp_path, mapper,
                                            factory_spec=f"{__name__}:FakeFactory", timeout=15)
    assert result["tasks"] == 0
    assert result["skipped_no_checkpoints"] == 1
    receipt = collect_from_trajectories(directory, mapping_file, tmp_path / "examples.jsonl", tmp_path,
                                        factory_spec=f"{__name__}:FakeFactory", timeout=15)
    assert receipt["status"] == "no_accepted_examples"
    assert receipt["skipped_no_checkpoints"] == 1
