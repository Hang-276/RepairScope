"""Spawned synthetic-world tests; these cannot qualify AppWorld feasibility."""
from contextlib import contextmanager, redirect_stdout
import io
import os
from types import SimpleNamespace

import pytest

from repairscope.probes import run_joint_probe, mechanism_signature
from repairscope.replay import run_branch, receipt_success, stable_digest


_CREATED = 0


class FakeWorld:
    def __init__(self, task_id, experiment_name):
        self.task_id, self.experiment_name = task_id, experiment_name
        self.x, self.y, self.condition = 0, 1, 1
        self.models, self.changes = self, []
        self.namespace = {"apis": SimpleNamespace(app=SimpleNamespace(
            set_x=self.set_x, restore_y=self.restore_y,
            set_condition=self.set_condition, get_condition=lambda: self.condition))}

    def write(self, name, value):
        setattr(self, name, value)
        self.changes.append([f"UPDATE state SET {name}=?", [value], False])

    def set_x(self, value):
        self.write("x", value)
        if value == 2 and self.condition:
            self.write("y", 0)

    def restore_y(self):
        self.write("y", 1)

    def set_condition(self, value):
        self.write("condition", value)

    def get_tracked_db_changes(self):
        return {"app": [{"statement": row[0], "parameters": row[1],
                         "executemany": row[2], "context": {"pid": os.getpid(),
                         "experiment_name": self.experiment_name}} for row in self.changes]}

    def execute(self, code):
        output = io.StringIO()
        with redirect_stdout(output):
            try:
                exec(code, self.namespace)
            except Exception as error:
                print(f"Execution failed: {type(error).__name__}: {error}")
        return output.getvalue()

    def evaluate(self):
        values = {"x is two": self.x == 2, "y is one": self.y == 1}
        return {"passes": [{"requirement": key} for key, value in values.items() if value],
                "failures": [{"requirement": key} for key, value in values.items() if not value]}

    def save(self):
        pass


class FakeFactory:
    def __init__(self, root, seed):
        pass

    @contextmanager
    def create(self, task_id, experiment_name, split, full_ground_truth=False):
        global _CREATED
        _CREATED += 1
        assert _CREATED == 1, "Process-global world leaked into sibling"
        assert split == "train" and full_ground_truth
        yield FakeWorld(task_id, experiment_name)


def checker(world, mapping):
    return {"x": world.x == 2, "y": world.y == 1}


def invalid_checker(world, mapping):
    return {"x": "true", "y": True}


def mutating_checker(world, mapping):
    world.set_x(2)
    return {"x": True, "y": True}


def request(tmp_path):
    listing = tmp_path / "data" / "datasets"
    listing.mkdir(parents=True, exist_ok=True)
    (listing / "train.txt").write_text("fake_1: task\n")
    mapping = {"x": {"requirements": ["x is two"], "complete": True},
               "y": {"requirements": ["y is one"], "complete": True}}
    identity = {name: stable_digest(name) for name in (
        "environment_sha256", "task_sha256", "goals_sha256", "protocol_sha256")}
    identity["mapping_sha256"] = stable_digest(mapping)
    return {"task_id": "fake_1", "scenario_id": "fake", "split": "train",
            "source": "natural_failure", "root": str(tmp_path), "seed": 100,
            "identity": identity, "steps": [], "program": "apis.app.set_x(2)\napis.app.restore_y()",
            "intervention": None, "goal_mapping": mapping, "g_fix": ["x"], "g_keep": ["y"],
            "factory_spec": f"{__name__}:FakeFactory"}


def condition():
    return {"code": "apis.app.set_condition(0)",
            "query": "print(apis.app.get_condition())", "expected_output": "0\n",
            "public_api_only": True, "semantics_preserved": True, "checker_invariant": True}


def test_failed_prefix_replays_partial_writes_and_python_locals(tmp_path):
    value = request(tmp_path)
    value["steps"] = [{"code": "token = 7\napis.app.set_x(1)\nraise RuntimeError('planned')",
                       "output": "Execution failed: RuntimeError: planned\n"}]
    value["program"] = "assert token == 7\napis.app.set_x(2)\napis.app.restore_y()"
    good = run_branch(value, timeout=15)
    assert receipt_success(good), good
    assert good["evidence_origin"] == "synthetic_fixture"
    assert good["source"] == "natural_failure" and good["prefix_verified"]
    assert good["identity"] == value["identity"]
    assert len(good["prefix_fingerprints"]) == 1
    assert good["changes_initial"]["app"][0][1] == [1]
    value["program"] = "assert token == 7\nprint('unchanged')"
    sibling = run_branch(value, timeout=15)
    assert sibling["goals_after"] == {"x": False, "y": True}
    assert not receipt_success(sibling)
    assert sibling["prefix_fingerprint"] == good["prefix_fingerprint"]
    assert sibling["experiment_name"] != good["experiment_name"]
    assert sibling["process_id"] != os.getpid()


def test_mismatched_observation_rejected_before_repair(tmp_path):
    value = request(tmp_path)
    value["steps"] = [{"code": "print('actual')", "output": "other\n"}]
    result = run_branch(value, timeout=15)
    assert not result["ok"]
    assert "observation mismatch" in " ".join(result["errors"])
    assert "program_output" not in result


def test_train_membership_and_test_guards_precede_factory(tmp_path):
    value = request(tmp_path)
    value.update(split="test_normal", factory_spec="does_not_exist:Factory")
    assert "train-only" in str(run_branch(value)["errors"])
    value.update(split="train", task_id="foreign_1")
    assert "official train list" in str(run_branch(value)["errors"])


def test_intervention_needs_audited_invariant_checker(tmp_path):
    value = request(tmp_path)
    value["intervention"] = condition()
    result = run_branch(value, timeout=15)
    assert result["ok"] and result["intervention_verified"]
    assert result["goals_after"] == {"x": None, "y": None}
    assert not receipt_success(result)
    value.update(checker_spec=f"{__name__}:checker", checker_audit_sha256=stable_digest("audit"))
    result = run_branch(value, timeout=15)
    assert receipt_success(result), result
    value["intervention"]["checker_invariant"] = False
    assert not receipt_success(run_branch(value, timeout=15))


def test_invalid_checker_and_error_program_cannot_establish_success(tmp_path):
    value = request(tmp_path)
    value.update(checker_spec=f"{__name__}:invalid_checker", checker_audit_sha256=stable_digest("audit"))
    result = run_branch(value, timeout=15)
    assert not result["ok"] and "bool/null" in str(result["errors"])
    value.pop("checker_spec")
    value["program"] += "\nmissing_name()"
    result = run_branch(value, timeout=15)
    assert result["goals_after"] == {"x": True, "y": True}
    assert not result["ok"] and not receipt_success(result)


def test_audit_checker_cannot_modify_the_world(tmp_path):
    value = request(tmp_path)
    value.update(checker_spec=f"{__name__}:mutating_checker", checker_audit_sha256=stable_digest("audit"))
    result = run_branch(value, timeout=15)
    assert not result["ok"] and "checker changed" in str(result["errors"])


@pytest.mark.parametrize("code", ["import os", "open('x', 'w')", "eval('1')",
    "apis.app.__dict__", "apis.app.set_condition.__call__(0)", "print = apis.app.set_condition\nprint(0)"])
def test_unsafe_interventions_rejected_without_world(tmp_path, code):
    value = request(tmp_path)
    value["intervention"] = dict(condition(), code=code)
    result = run_branch(value)
    assert not result["ok"] and "process_id" not in result


def test_joint_probe_observable_interaction_and_reservation(tmp_path):
    value = request(tmp_path)
    value.update(checker_spec=f"{__name__}:checker", checker_audit_sha256=stable_digest("audit"))
    plan = {"mechanism_id": "conditional_compensation", "r0": "apis.app.set_x(2)",
            "r1": "apis.app.set_x(2)\napis.app.restore_y()", "conditions": [None, condition()],
            "explanations": ["compensation always needed", "condition-dependent"]}
    reservations = []
    result = run_joint_probe(value, plan, reserve=lambda: reservations.append(1))
    assert result["valid"] and result["conditional_effect"], result
    assert len(reservations) == 4
    assert result["goal_interaction"] == {"x": 0, "y": -1}
    assert result["evidence_origin"] == "synthetic_fixture"
    assert {c["query_output"] for c in result["cells"].values()} == {"0\n", "1\n"}
    plan["r0"] = "missing_variable()"
    result = run_joint_probe(value, plan)
    assert not result["valid"] and not result["conditional_effect"]
    assert "cannot establish necessity" in str(result["reasons"])


def test_joint_probe_syntax_deletion_and_budget_rejected(tmp_path):
    value = request(tmp_path)
    plan = {"mechanism_id": "m", "r0": "if:", "r1": "print(1)",
            "conditions": [None, condition()]}
    calls = []
    assert not run_joint_probe(value, plan, run=lambda r: calls.append(r))["valid"]
    assert calls == []
    plan["r0"] = "print(0)"
    result = run_joint_probe(value, plan, run=lambda r: calls.append(r), reserve=lambda: False)
    assert calls == [] and "budget exhausted" in str(result["reasons"])


def test_mechanism_signature_deduplicates_entity_and_variable_renaming():
    first = {"mechanism_id": "first-label", "r0": "file_id = 'file-a'\napis.files.move(file_id, destination='folder-a')",
             "r1": "file_id = 'file-a'\napis.files.move(file_id, destination='folder-a')\napis.files.share(file_id, user='alice')",
             "conditions": [None, dict(condition(), query="print(apis.files.get_access('file-a'))")]}
    renamed = {"mechanism_id": "different-label", "r0": "document = 'file-b'\napis.files.move(document, destination='folder-b')",
               "r1": "document = 'file-b'\napis.files.move(document, destination='folder-b')\napis.files.share(document, user='bob')",
               "conditions": [None, dict(condition(), query="print(apis.files.get_access('file-b'))")]}
    assert mechanism_signature(first) == mechanism_signature(renamed)
    changed = dict(renamed, r1=renamed["r1"].replace(".share(", ".delete("))
    assert mechanism_signature(first) != mechanism_signature(changed)
    query_changed = dict(renamed, conditions=[None, dict(condition(), query="print(apis.files.get_owner('file-b'))")])
    assert mechanism_signature(first) != mechanism_signature(query_changed)
