"""P1 contract checks use synthetic metadata and never instantiate AppWorld."""

from copy import deepcopy

import pytest

from repairscope.p1 import _repair_with_knowledge, _validate_inputs, public_view, summarize
from repairscope.protocol import default_protocol, digest


def frozen_inputs():
    tasks, goal_tasks, events = [], {}, []
    protocol = default_protocol()
    protocol["task_order"] = [f"fixture_task_{index}" for index in range(4)]
    assets = {"environment_sha256": digest("synthetic environment identity")}
    for index, task_id in enumerate(protocol["task_order"]):
        instruction = f"Update alpha and preserve beta ({task_id})."
        task = {"task_id": task_id, "scenario_id": f"fixture_scenario_{index}", "split": "train",
                "source": "natural_failure", "task_sha256": digest(task_id),
                "trajectory_sha256": digest(f"prefix-{index}")}
        tasks.append(task)
        frozen = {"split": "train", "instruction": instruction,
                  "goals": [{"id": "fix", "evidence_span": "Update alpha"},
                            {"id": "keep", "evidence_span": "preserve beta"}]}
        goal_tasks[task_id] = frozen
        mapping = {"fix": {"predicate": "synthetic fix predicate"},
                   "keep": {"predicate": "synthetic preservation predicate"}}
        request = {"task_id": task_id, "scenario_id": task["scenario_id"], "split": "train",
                   "source": "natural_failure", "steps": [{"code": "pass"}], "goal_mapping": mapping,
                   "g_fix": ["fix"], "g_keep": ["keep"], "identity": {
                       "protocol_sha256": digest(protocol), "goals_sha256": digest(frozen),
                       "environment_sha256": assets["environment_sha256"], "task_sha256": task["task_sha256"],
                       "mapping_sha256": digest(mapping)}}
        result = {"original_success": False, "synthetic_fixture": True}
        events.append({"event_id": f"fixture_event_{index}", "request": request,
                       "candidates": [{"id": "existing", "code": "pass"}],
                       "public_context": {"instruction": instruction, "observations": {}},
                       "forbidden_literals": [task_id],
                       "original_score": {"scored": True, "task_id": task_id, "split": "train",
                                          "result": result, "result_sha256": digest(result)},
                       "human_task_edits": 0, "human_rule_edits": 0, "human_intervention_edits": 0})
    transfer = {key: deepcopy(events[3][key]) for key in
                ("request", "public_context", "human_task_edits", "human_rule_edits", "human_intervention_edits")}
    transfer.update(case_id="fixture_holdout", repeats=1)
    document = {"schema_version": 1, "events": events[:3], "transfer_cases": [transfer], "allow_extension": False}
    return document, assets, protocol, {"tasks": tasks}, {"tasks": goal_tasks}


def test_valid_metadata_can_be_checked_without_executing_a_task():
    inputs = frozen_inputs()
    events, transfers = _validate_inputs(*inputs)
    assert len(events) == 3 and len(transfers) == 1


@pytest.mark.parametrize("split", ["dev", "test_normal", "test_challenge"])
def test_learning_event_cannot_cross_data_boundary(split):
    inputs = frozen_inputs()
    inputs[0]["events"][0]["request"]["split"] = split
    with pytest.raises(ValueError, match="train"):
        _validate_inputs(*inputs)


def test_event_hash_and_original_instruction_cannot_drift():
    inputs = frozen_inputs()
    inputs[0]["events"][0]["request"]["identity"]["environment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        _validate_inputs(*inputs)
    inputs = frozen_inputs()
    inputs[0]["events"][0]["public_context"]["instruction"] = "Drop the preservation requirement"
    with pytest.raises(ValueError, match="instruction"):
        _validate_inputs(*inputs)


def test_transfer_cannot_reuse_an_induction_task_or_eval_partition():
    inputs = frozen_inputs()
    inputs[0]["transfer_cases"][0]["request"]["task_id"] = "fixture_task_0"
    with pytest.raises(ValueError, match="independent"):
        _validate_inputs(*inputs)
    inputs = frozen_inputs()
    inputs[0]["transfer_cases"][0]["request"]["split"] = "dev"
    with pytest.raises(ValueError, match="train"):
        _validate_inputs(*inputs)


@pytest.mark.parametrize("mutation", ["identity", "instruction", "scenario", "goal"])
def test_transfer_uses_the_same_frozen_identity_checks_as_learning(mutation):
    inputs = frozen_inputs()
    case = inputs[0]["transfer_cases"][0]
    if mutation == "identity":
        case["request"]["identity"]["task_sha256"] = "0" * 64
    elif mutation == "instruction":
        case["public_context"]["instruction"] = "Different task instruction"
    elif mutation == "scenario":
        case["request"]["scenario_id"] = "invented scenario coverage"
    else:
        case["request"]["g_fix"] = ["invented goal"]
    with pytest.raises(ValueError):
        _validate_inputs(*inputs)


def test_learning_cannot_invent_extra_scenario_coverage():
    inputs = frozen_inputs()
    inputs[0]["events"][0]["request"]["scenario_id"] = "invented extra scenario"
    with pytest.raises(ValueError, match="Scenario identity"):
        _validate_inputs(*inputs)


def test_duplicate_states_do_not_increase_support():
    inputs = frozen_inputs()
    duplicate = deepcopy(inputs[0]["events"][0])
    duplicate["event_id"] = "different event identifier, same state"
    inputs[0]["events"].insert(1, duplicate)
    with pytest.raises(ValueError, match="Duplicate"):
        _validate_inputs(*inputs)


def test_original_score_cannot_be_overwritten_with_a_new_result():
    inputs = frozen_inputs()
    inputs[0]["events"][0]["original_score"]["result"] = {"original_success": True}
    with pytest.raises(ValueError, match="already be scored"):
        _validate_inputs(*inputs)


@pytest.mark.parametrize("field", ["hidden_checker", "goal_mapping", "reference_solution", "budget"])
def test_public_context_rejects_private_or_protected_top_level_fields(field):
    with pytest.raises(ValueError, match="only"):
        public_view({"instruction": "original instruction", field: {"hidden": True}})


def test_public_context_returns_an_independent_copy():
    original = {"instruction": "original instruction", "observations": {"condition": True}}
    copied = public_view(original)
    copied["observations"]["condition"] = False
    assert original["observations"]["condition"] is True


def test_synthetic_evidence_can_never_satisfy_p1_gates():
    witnesses = [{"task_id": f"t{i}", "scenario_id": f"s{i}", "event_id": f"e{i}",
                  "source": "natural_failure", "evidence_origin": "synthetic_fixture", "recovery_success": True}
                 for i in range(3)]
    probes = [{"mechanism_id": f"m{i}", "valid": True, "conditional_effect": True,
               "evidence_origin": "synthetic_fixture"} for i in range(2)]
    updates = [{"validation": {"accepted": True, "evidence_origin": "synthetic_fixture"}} for _ in range(2)]
    transfers = [{"independent": True, "source": "synthetic_fixture", "improved": True, "behavior_changed": True}]
    report = summarize(witnesses, probes, updates, transfers, [{"status": "unresolved"}])
    assert not report["A"] and not report["B"] and not report["C"]
    assert report["candidate_attempts"] == 3 and report["unresolved_events"] == 1
    assert report["decision"] == "stop_or_analyze_existing_evidence"
    assert report["p2_started"] is False and report["native_test_run"] is False


def test_candidate_denominator_keeps_failures_unknowns_and_duplicate_successes():
    base = {"task_id": "task", "scenario_id": "scenario", "event_id": "state",
            "source": "natural_failure", "evidence_origin": "AppWorld"}
    witnesses = [{**base, "recovery_success": value} for value in (True, True, False, None)]
    report = summarize(witnesses, [], [], [], [{"status": "unresolved"}])
    assert report["candidate_attempts"] == 4 and report["witness_states"] == 1
    assert not report["A"]


def test_online_condition_matching_distinguishes_true_from_integer_one():
    rule = {
        "rule_id": "conditional_mechanism_marker", "version": 1, "intent": "use a conditional compensation",
        "applicability": {"observable_conditions": [{"key": "condition", "equals": True}],
                          "required_queries": [], "unknown_policy": "do_not_treat_as_verified"},
        "repair": {"base_template": "parameterized repair", "conditional_compensation": [],
                   "read_scope": [], "write_scope": []},
        "goal_effects": {"recovery_claims": [], "preservation_claims": []},
        "evidence_ids": [], "counterexample_ids": [], "status": "validated",
    }
    calls = []
    def model(messages):
        calls.append(messages)
        return {"program": "pass"}
    snapshot = {"version": 1, "rules": [rule]}
    _repair_with_knowledge(model, {"instruction": "complete all original requirements", "observations": {"condition": 1}}, snapshot)
    assert "conditional_mechanism_marker" not in calls[-1][0]["content"]
    _repair_with_knowledge(model, {"instruction": "complete all original requirements", "observations": {"condition": True}}, snapshot)
    assert "conditional_mechanism_marker" in calls[-1][0]["content"]
