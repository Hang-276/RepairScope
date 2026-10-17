"""Knowledge guarantees use explicit synthetic fixtures, never research evidence."""

from copy import deepcopy
import json

import pytest

from repairscope.knowledge import RuleStore, rule_prompt, validate_rule
from repairscope.revise import propose_revision
from repairscope.validate import validate_update


def rule(rule_id="preserve_progress", version=1):
    return {
        "rule_id": rule_id, "version": version, "intent": "restore a failed write while preserving progress",
        "applicability": {"observable_conditions": [{"key": "needs_compensation", "equals": True}],
                          "required_queries": ["query the observable dependency"],
                          "unknown_policy": "do_not_treat_as_verified"},
        "repair": {"base_template": "apply the parameterized repair", "conditional_compensation": [],
                   "read_scope": ["public dependency status"], "write_scope": ["requested object"]},
        "goal_effects": {"recovery_claims": ["requested write completes"],
                         "preservation_claims": ["existing requested progress remains satisfied"]},
        "evidence_ids": ["fixture-support"], "counterexample_ids": [], "status": "validated",
    }


def evidence(evidence_id="fixture-support", **updates):
    item = {"evidence_id": evidence_id, "source": "synthetic_fixture", "executed": True,
            "valid": True, "known_harm": False, "recovery_success": True}
    item.update(updates)
    return item


def context(**updates):
    item = {"task_scored": True, "frozen_before": "fixed-protocol-hash", "frozen_after": "fixed-protocol-hash",
            "automatic": True, "regression_checks": [], "human_task_edits": 0,
            "human_rule_edits": 0, "human_intervention_edits": 0}
    item.update(updates)
    return item


def accepted(candidate, old=None, items=None):
    checks = [] if old is None else [{"valid": True, "passed": True, "known_harm": False}]
    receipt = validate_update(old, candidate, items or [evidence()], context(regression_checks=checks))
    assert receipt["accepted"], receipt["reasons"]
    return receipt


def test_immutable_history_persistence_and_nondestructive_rollback(tmp_path):
    path = tmp_path / "knowledge.json"
    store = RuleStore(path)
    original = rule()
    store.commit(original, accepted(original))
    revised = rule(version=2)
    revised["applicability"]["observable_conditions"].append({"key": "dependency_is_observable", "equals": True})
    revised["counterexample_ids"] = ["fixture-counterexample"]
    items = [evidence(), evidence("fixture-counterexample", known_harm=True, recovery_success=False)]
    store.commit(revised, accepted(revised, original, items))
    old = store.rollback(1)
    old["rules"][0]["intent"] = "outside mutation"
    assert store.snapshot()["version"] == 2
    assert store.rollback(1)["rules"][0]["version"] == 1
    assert RuleStore(path).snapshot() == store.snapshot()
    archive = json.loads(path.read_text())
    assert archive["versions"][2]["receipt"]["evidence_origin"] == "synthetic_fixture"
    assert archive["versions"][2]["receipt"]["research_evidence"] is False


def test_unknown_and_wrong_typed_conditions_do_not_match():
    store = RuleStore()
    candidate = rule()
    store.commit(candidate, accepted(candidate))
    for observations in ({}, {"needs_compensation": None}, {"needs_compensation": False}, {"needs_compensation": 1}):
        assert store.retrieve(observations) == []
    retrieved = store.retrieve({"needs_compensation": True})
    retrieved[0]["intent"] = "mutated"
    assert store.retrieve({"needs_compensation": True})[0]["intent"] == candidate["intent"]
    with pytest.raises(ValueError):
        store.retrieve({}, max_rules=4)


@pytest.mark.parametrize("status", ["provisional", "deprecated"])
def test_unverified_and_deprecated_rules_are_not_deployed(status):
    candidate = rule()
    candidate["status"] = status
    store = RuleStore()
    store.commit(candidate, accepted(candidate))
    assert store.retrieve({"needs_compensation": True}) == []
    with pytest.raises(ValueError):
        rule_prompt([candidate])


@pytest.mark.parametrize("field", ["evidence_ids", "counterexample_ids"])
def test_update_cannot_delete_prior_evidence_or_counterexamples(field):
    old = rule()
    old["counterexample_ids"] = ["fixture-counterexample"]
    revised = deepcopy(old)
    revised["version"] = 2
    revised["intent"] = "narrowed intent after counterexample"
    revised[field] = []
    receipt = validate_update(old, revised, [evidence()], context(regression_checks=[
        {"valid": True, "passed": True, "known_harm": False}]))
    assert not receipt["accepted"]
    assert f"deleted_{field}" in receipt["reasons"]


@pytest.mark.parametrize("overrides", [
    {"task_scored": False}, {"automatic": False}, {"frozen_after": "tampered"},
    {"human_rule_edits": 1}, {"human_intervention_edits": None},
    {"regression_checks": [{"valid": True, "passed": None, "known_harm": False}]},
])
def test_score_protocol_and_regression_validation_are_fail_closed(overrides):
    receipt = validate_update(None, rule(), [evidence()], context(**overrides))
    assert not receipt["accepted"]


@pytest.mark.parametrize("overrides", [
    {"executed": False}, {"valid": None}, {"recovery_success": False},
    {"known_harm": True}, {"source": "unverified_server_report"},
])
def test_support_must_be_executed_valid_and_nonharmful(overrides):
    receipt = validate_update(None, rule(), [evidence(**overrides)], context())
    assert not receipt["accepted"]


def test_receipt_cannot_be_reused_for_another_rule():
    store = RuleStore()
    candidate = rule()
    receipt = accepted(candidate)
    candidate["intent"] = "different unsupported mechanism"
    with pytest.raises(ValueError, match="bound"):
        store.commit(candidate, receipt)
    assert store.snapshot() == {"version": 0, "rules": []}


def test_stale_store_cannot_replace_another_writer(tmp_path):
    path = tmp_path / "knowledge.json"
    first, stale = RuleStore(path), RuleStore(path)
    candidate = rule()
    first.commit(candidate, accepted(candidate))
    with pytest.raises(ValueError, match="changed"):
        stale.commit(candidate, accepted(candidate))
    assert RuleStore(path).snapshot()["version"] == 1


def test_rule_limit_retains_evidence_instead_of_evicting_silently():
    store = RuleStore()
    for index in range(32):
        candidate = rule(f"rule_{index:02d}")
        store.commit(candidate, accepted(candidate))
    extra = rule("extra")
    with pytest.raises(ValueError, match="32"):
        store.commit(extra, accepted(extra))
    assert len(store.snapshot()["rules"]) == 32
    assert len(store.retrieve({"needs_compensation": True})) == 3


def test_task_literals_are_rejected_but_audit_references_not_in_prompt():
    candidate = rule()
    candidate["intent"] = "repair task_1234"
    with pytest.raises(ValueError, match="task-specific"):
        validate_rule(candidate, forbidden_literals=["task_1234"])
    candidate = rule()
    candidate["evidence_ids"] = ["task_1234"]
    validate_rule(candidate, forbidden_literals=["task_1234"])
    assert "task_1234" not in rule_prompt([candidate])
    candidate["intent"] = 'repair entity "private"'
    with pytest.raises(ValueError, match="task-specific"):
        validate_rule(candidate, forbidden_literals=['entity "private"'])


@pytest.mark.parametrize("split", [None, "dev", "test", "test_normal", "test_challenge"])
def test_appworld_learning_requires_train_partition(split):
    item = evidence(source="AppWorld", split=split)
    assert not validate_update(None, rule(), [item], context())["accepted"]
    with pytest.raises(ValueError):
        propose_revision(lambda messages, max_tokens: rule(), {"version": 0, "rules": []}, {}, [item])


def test_real_environment_provenance_requires_train_and_preserves_receipt():
    receipt = validate_update(None, rule(), [evidence(source="AppWorld", split="train")], context())
    assert receipt["accepted"]
    assert receipt["research_evidence"] is True
    assert receipt["evidence_origin"] == "AppWorld"
    assert receipt["diff"]["intent"]["after"] == rule()["intent"]
    assert receipt["updated_at"]


def test_revision_calls_model_once_and_separates_offline_labels():
    calls = []
    def model(messages, max_tokens):
        calls.append((messages, max_tokens))
        return json.dumps(rule())
    candidate = propose_revision(model, {"version": 0, "rules": []}, {"instruction": "visible task"},
                                 [evidence(checker_labels={"fix": True})])
    assert candidate == rule()
    assert len(calls) == 1 and calls[0][1] == 2048
    payload = json.loads(calls[0][0][1]["content"])
    assert "checker_labels" not in payload["public_context"]
    assert payload["offline_execution_evidence"][0]["checker_labels"] == {"fix": True}
    with pytest.raises(ValueError, match="not public"):
        propose_revision(model, {"version": 0, "rules": []}, {"nested": {"hidden_checker": True}}, [evidence()])
    assert len(calls) == 1


@pytest.mark.parametrize("response", [
    '```json\n{}\n```', '{"rule_id":"first","rule_id":"second"}',
    '{"value": NaN}', {"goals": "changed"},
])
def test_revision_rejects_malformed_or_protected_model_outputs(response):
    with pytest.raises((ValueError, TypeError)):
        propose_revision(lambda messages, max_tokens: response,
                         {"version": 0, "rules": []}, {"instruction": "visible"}, [evidence()])
