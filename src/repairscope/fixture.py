"""Clearly artificial mechanics smoke; never qualifies real P0 or P1 evidence."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace

from .io import atomic_json
from .knowledge import RuleStore
from .p1 import summarize
from .probes import run_joint_probe
from .protocol import digest
from .replay import run_branch, receipt_success
from .validate import validate_update


class FixtureWorld:
    def __init__(self, task_id):
        self.task_id, self.x, self.y, self.condition = task_id, 0, 1, True
        self.models, self.changes = self, []
        self.namespace = {"apis": SimpleNamespace(toy=SimpleNamespace(
            set_target=self.set_target, restore=self.restore, set_condition=self.set_condition,
            get_condition=lambda: self.condition))}

    def write(self, key, value):
        setattr(self, key, value)
        self.changes.append([f"UPDATE toy SET {key}=?", [value], False])

    def set_target(self, value):
        self.write("x", value)
        if self.condition:
            self.write("y", 0)

    def restore(self):
        self.write("y", 1)

    def set_condition(self, value):
        self.write("condition", value)

    def get_tracked_db_changes(self):
        return {"toy": self.changes}

    def save(self):
        pass

    def execute(self, code):
        stream = io.StringIO()
        with redirect_stdout(stream):
            try:
                exec(code, self.namespace)
            except Exception as error:
                print(f"Execution failed: {type(error).__name__}: {error}")
        return stream.getvalue()


class FixtureFactory:
    def __init__(self, root, seed):
        pass

    @contextmanager
    def create(self, task_id, experiment_name, split, full_ground_truth=False):
        assert split == "train"
        yield FixtureWorld(task_id)


def fixture_checker(world, mapping):
    return {"target": world.x == 2, "preserve": world.y == 1}


def smoke(output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Fixture output exists; choose a new directory")
    listing = output / "world/data/datasets"
    listing.mkdir(parents=True)
    (listing / "train.txt").write_text("fixture_1\n", encoding="utf-8")
    mapping = {"target": {"requirements": ["toy target"], "complete": True},
               "preserve": {"requirements": ["toy preserve"], "complete": True}}
    request = {"task_id": "fixture_1", "scenario_id": "fixture", "split": "train", "source": "synthetic_fixture",
               "root": str(output / "world"), "seed": 100,
               "identity": {key: digest(key) for key in ("environment_sha256", "task_sha256", "goals_sha256", "protocol_sha256")},
               "steps": [{"code": "local_value = 2\napis.toy.set_target(1)\napis.toy.restore()\nraise RuntimeError('fixture')",
                          "output": "Execution failed: RuntimeError: fixture\n"}],
               "goal_mapping": mapping, "g_fix": ["target"], "g_keep": ["preserve"],
               "factory_spec": "repairscope.fixture:FixtureFactory", "checker_spec": "repairscope.fixture:fixture_checker",
               "checker_audit_sha256": digest("synthetic_fixture_only")}
    request["identity"]["mapping_sha256"] = digest(mapping)
    witness = run_branch({**request, "program": "apis.toy.set_target(local_value)\napis.toy.restore()", "intervention": None})
    witness.update(event_id="fixture_event", recovery_success=receipt_success(witness))
    joint = run_joint_probe(request, {
        "mechanism_id": "synthetic_toy_dependency", "r0": "apis.toy.set_target(local_value)",
        "r1": "apis.toy.set_target(local_value)\napis.toy.restore()", "explanations": ["toy conditional dependency"],
        "conditions": [None, {"code": "apis.toy.set_condition(False)",
                              "query": "print(apis.toy.get_condition())", "expected_output": "False\n",
                              "public_api_only": True, "semantics_preserved": True, "checker_invariant": True}]})
    rule = {"rule_id": "synthetic_toy", "version": 1, "intent": "Toy mechanism for software checking only",
            "applicability": {"observable_conditions": [{"key": "toy_dependency", "equals": True}],
                              "required_queries": ["observe toy dependency"], "unknown_policy": "do_not_treat_as_verified"},
            "repair": {"base_template": "parameterized toy target update", "conditional_compensation": ["restore protected toy state"],
                       "read_scope": ["toy dependency"], "write_scope": ["toy target", "toy protected state"]},
            "goal_effects": {"recovery_claims": ["toy target"], "preservation_claims": ["toy protected state"]},
            "evidence_ids": [joint["evidence_id"]], "counterexample_ids": [], "status": "validated"}
    decision = validate_update(None, rule, [{"evidence_id": joint["evidence_id"], "executed": True,
        "valid": joint["valid"], "recovery_success": receipt_success(witness), "known_harm": False,
        "source": "synthetic_fixture", "split": "train"}],
        {"task_scored": True, "automatic": True, "frozen_before": digest("fixture"), "frozen_after": digest("fixture"),
         "regression_checks": [], "human_task_edits": 0, "human_rule_edits": 0, "human_intervention_edits": 0})
    store = RuleStore(output / "knowledge.json")
    if decision["accepted"]:
        store.commit(rule, decision)
    report = summarize([witness], [joint], [{"validation": decision}], [], [])
    passed = receipt_success(witness) and joint["valid"] and joint["conditional_effect"] and decision["accepted"]
    result = {"status": "passed" if passed else "failed", "evidence_origin": "synthetic_fixture",
              "software_checks": {"partial_prefix_and_repair": receipt_success(witness), "joint_probe": joint["valid"],
                                  "knowledge_roundtrip": RuleStore(output / "knowledge.json").snapshot() == store.snapshot() if passed else False},
              "P1_decision": report, "scientific_validation": False}
    atomic_json(output / "witness.json", witness)
    atomic_json(output / "joint.json", joint)
    atomic_json(output / "update.json", decision)
    atomic_json(output / "receipt.json", result)
    return result
