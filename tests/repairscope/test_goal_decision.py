from copy import deepcopy

import pytest
import torch

from repairscope.decision import (decision_groups, goal_relevant_evidence,
                                 goal_sufficiency_loss, restricted_utility,
                                 scope_supervision_audit)
from repairscope.fixtures import fixture_records
from repairscope.tensorizer import Tensorizer


CONFIG = {"goal_decision": {"progress_weight": 1., "damage_weight": 5.,
                             "cost_weight": .01, "temperature": 1.},
          "model": {"feature_dim": 8, "value_slots": 8, "history": 2, "candidate_limit": 16}}
PAID = "shop.Order[1].paid"
SHIPPING = "shop.Order[1].shipping"


def pair():
    return deepcopy(fixture_records(1)[7:9])  # Fragile condition, action/reference.


def test_full_and_restricted_utility_keep_cost_and_weighted_protection():
    records = pair()
    built = decision_groups(records, CONFIG)
    assert built["decision_groups"] == built["decision_eligible_groups"] == 1
    assert built["decision_candidates"] == 2
    action, reference = built["groups"][0]["candidates"]
    assert action["contributions"] == {SHIPPING: 1., PAID: -12.}
    assert action["full_utility"] == pytest.approx(-11.01)
    assert reference["full_utility"] == pytest.approx(.99)
    assert restricted_utility(action, [SHIPPING]) == pytest.approx(.99)
    assert restricted_utility(action, [SHIPPING, PAID]) == pytest.approx(-11.01)


@pytest.mark.parametrize("failure", ["action", "missing_action_valid", "label", "snapshot", "domain", "diff", "audit",
                                      "prefix", "goals", "duplicate", "unpaired"])
def test_unreliable_or_noncomparable_groups_are_masked(failure):
    records = pair()
    if failure == "action":
        records[0]["action_valid"] = False
    elif failure == "missing_action_valid":
        records[0].pop("action_valid")
    elif failure == "label":
        records[0]["goal_after"][1] = None
    elif failure == "snapshot":
        records[0]["audit_next_state"]["fields"][1].update(known=False, value=None)
    elif failure == "domain":
        records[0]["audit"]["field_domain"].remove(PAID)
        records[0]["audit"]["changes"].remove(PAID)
    elif failure == "diff":
        records[0]["audit"]["changes"].remove(PAID)
    elif failure == "audit":
        records[0].pop("audit")
    elif failure == "prefix":
        records[1]["prefix_id"] = "different"
    elif failure == "goals":
        records[1]["goals"] = deepcopy(records[1]["goals"])
        records[1]["goals"][0]["weight"] = 2.
    elif failure == "duplicate":
        records.append(deepcopy(records[0]))
    elif failure == "unpaired":
        records.pop()
    built = decision_groups(records, CONFIG)
    assert built["decision_eligible_groups"] == built["decision_candidates"] == 0
    assert built["decision_masked_groups"] == 1
    assert built["groups"][0]["reason"]


def test_label_snapshot_disagreement_is_not_repaired_to_a_negative():
    records = pair()
    records[0]["goal_after"][1] = True
    assert decision_groups(records, CONFIG)["decision_eligible_groups"] == 0
    evidence = goal_relevant_evidence(records[0], CONFIG)
    assert not evidence["valid"]
    assert evidence["keys"] == []


def test_goal_sufficiency_has_direct_scope_score_gradient_and_hard_forward():
    records = pair()
    batch = Tensorizer(CONFIG).batch(records)
    scores = torch.zeros_like(batch["numeric"], requires_grad=True)
    soft = torch.sigmoid(scores)
    hard = torch.zeros_like(soft)
    for row, meta in enumerate(batch["metadata"]):
        hard[row, meta["keys"].index(SHIPPING)] = 1.
    gate = hard + soft - soft.detach()
    torch.testing.assert_close(gate.detach(), hard)
    losses = goal_sufficiency_loss({"scope_gate": gate}, batch, CONFIG)
    assert losses["goal_sufficiency"].item() > .5
    assert losses["decision_loss_groups"] == 1
    losses["goal_sufficiency"].backward()
    paid_index = batch["metadata"][0]["keys"].index(PAID)
    assert scores.grad[0, paid_index].abs().item() > .1


def test_full_utility_does_not_shrink_when_public_candidates_omit_goal_field():
    records = pair()
    built = decision_groups(records, CONFIG)
    for record in records:
        # Offline audited pre-state preserves label truth while the public
        # observation lacks the field. It must remain in the full teacher.
        record["audit_state"] = deepcopy(record["state"])
        record["state"]["fields"] = [field for field in record["state"]["fields"] if field["key"] != PAID]
        record["state"]["relations"] = []
    batch = Tensorizer(CONFIG).batch(records)
    assert PAID not in batch["metadata"][0]["keys"]
    new = decision_groups(records, CONFIG)
    assert new["groups"][0]["candidates"][0]["full_utility"] == built["groups"][0]["candidates"][0]["full_utility"]
    gate = torch.ones_like(batch["numeric"], requires_grad=True)
    assert goal_sufficiency_loss({"scope_gate": gate}, batch, CONFIG)["goal_sufficiency"].item() > .5


def test_unknown_and_unpaired_loss_is_connected_zero():
    records = pair()[:1]
    batch = Tensorizer(CONFIG).batch(records)
    gate = torch.zeros_like(batch["numeric"], requires_grad=True)
    losses = goal_sufficiency_loss({"scope_gate": gate}, batch, CONFIG)
    assert losses["goal_sufficiency"].item() == 0
    losses["goal_sufficiency"].backward()
    assert gate.grad is not None and not gate.grad.any()


def test_ties_are_eligible_for_evaluation_but_masked_from_training():
    records = deepcopy(fixture_records(1)[5:7])
    built = decision_groups(records, CONFIG)
    assert built["decision_eligible_groups"] == built["decision_tied_groups"] == 1
    batch = Tensorizer(CONFIG).batch(records)
    gate = torch.zeros_like(batch["numeric"], requires_grad=True)
    losses = goal_sufficiency_loss({"scope_gate": gate}, batch, CONFIG)
    assert losses["decision_loss_groups"] == 0


def test_goal_relevant_keys_require_changed_protection_judgment():
    action, reference = pair()
    assert goal_relevant_evidence(action, CONFIG) == {
        "valid": True, "keys": [PAID], "observed_positive_keys": [PAID], "reason": None}
    assert goal_relevant_evidence(reference, CONFIG)["keys"] == []
    # Shipping progress and a changed note do not alter achieved paid protection.
    for record in (fixture_records(1)[0], fixture_records(1)[2]):
        evidence = goal_relevant_evidence(record, CONFIG)
        assert evidence["valid"] and not evidence["keys"]


def test_unknown_protected_goal_never_becomes_complete_negative():
    record = pair()[1]
    record["goal_before"][1] = record["goal_after"][1] = None
    evidence = goal_relevant_evidence(record, CONFIG)
    assert not evidence["valid"] and not evidence["keys"]
    # The same applies when a known label is contradicted by an unknown audit.
    record = pair()[1]
    record["audit_next_state"]["fields"][1].update(known=False, value=None)
    assert not goal_relevant_evidence(record, CONFIG)["valid"]


def test_missing_action_valid_is_unknown_for_decisions_and_goal_escape():
    records = pair()
    records[0].pop("action_valid")
    built = decision_groups(records, CONFIG)
    assert built["decision_eligible_groups"] == 0
    assert built["mask_reasons"] == {"unknown_action_valid": 1}
    evidence = goal_relevant_evidence(records[0], CONFIG)
    assert not evidence["valid"] and evidence["reason"] == "unknown_action_valid"


def test_protection_intervals_and_previously_unsatisfied_goals():
    records = pair()
    for record in records:
        record["goals"] = deepcopy(record["goals"])
        record["goals"][1]["protect_from_step"] = 1
    evidence = goal_relevant_evidence(records[0], CONFIG)
    assert evidence["valid"] and evidence["keys"] == []
    candidate = decision_groups(records, CONFIG)["groups"][0]["candidates"][0]
    assert candidate["contributions"][PAID] == -2.  # Lost progress, no active protection.


def test_before_false_protected_goal_needs_no_after_label_for_damage_negative():
    record = deepcopy(fixture_records(2)[10])  # Second task's pay_order, paid=False before.
    assert record["goal_before"][1] is False
    record["goal_after"][1] = None
    record["audit_next_state"]["fields"][1].update(known=False, value=None)
    evidence = goal_relevant_evidence(record, CONFIG)
    assert evidence["valid"] and not evidence["keys"]


def test_complete_labels_in_a_partial_but_explicit_goal_domain_are_usable():
    records = pair()
    for record in records:
        record["audit"]["complete"] = False
        record["audit"]["field_domain"] = [SHIPPING, PAID]
    assert decision_groups(records, CONFIG)["decision_eligible_groups"] == 1


def test_revealing_every_goal_increment_matches_full_information_loss():
    batch = Tensorizer(CONFIG).batch(pair())
    gate = torch.ones_like(batch["numeric"], requires_grad=True)
    losses = goal_sufficiency_loss({"scope_gate": gate}, batch, CONFIG)
    assert losses["goal_sufficiency"].item() == pytest.approx(0., abs=1e-7)


def test_changed_protected_field_with_unchanged_true_predicate_is_not_relevant():
    record = deepcopy(fixture_records(1)[2])
    key = "shop.Order[1].amount"
    record["goals"] = [{"id": "amount", "field": key, "op": "ge", "value": 10., "protected": True}]
    record["goal_before"] = record["goal_after"] = [True]
    record["audit_next_state"]["fields"][2]["value"] = 21.
    record["audit"]["changes"].append(key)
    evidence = goal_relevant_evidence(record, CONFIG)
    assert evidence["valid"] and evidence["keys"] == []


def supervision_config(**model):
    config = deepcopy(CONFIG)
    config["model"].update(goal_sufficient=True, scope_budget=2, static_scope=False)
    config["model"].update(model)
    return config


def test_supervision_preflight_identifies_competitive_retrievable_groups():
    config = supervision_config()
    audit = scope_supervision_audit(pair(), Tensorizer(config), config)
    assert audit["decision_eligible_groups"] == audit["decision_informative_groups"] == 1
    assert audit["decision_trainable_groups"] == audit["decision_retrievable_groups"] == 1
    assert audit["decision_budget_competitive_groups"] == 1
    assert audit["contribution_fields_total"] == audit["contribution_fields_retrievable"] == 3


def test_supervision_preflight_detects_all_critical_fields_unretrievable():
    records = pair()
    for record in records:
        record["audit_state"] = deepcopy(record["state"])
        record["state"]["fields"] = [field for field in record["state"]["fields"]
                                       if field["key"] not in {SHIPPING, PAID}]
        record["state"]["relations"] = []
    config = supervision_config()
    audit = scope_supervision_audit(records, Tensorizer(config), config)
    assert audit["decision_eligible_groups"] == audit["decision_informative_groups"] == 1
    assert audit["decision_budget_competitive_groups"] == 1
    assert audit["decision_trainable_groups"] == audit["decision_retrievable_groups"] == 0
    assert audit["contribution_fields_total"] == 3
    assert audit["contribution_fields_retrievable"] == 0
    batch = Tensorizer(config).batch(records)
    gate = torch.ones_like(batch["numeric"], requires_grad=True)
    loss = goal_sufficiency_loss({"scope_gate": gate}, batch, config)["goal_sufficiency"]
    assert loss.item() > .5
    loss.backward()
    assert not gate.grad.any()  # Positive KL alone does not prove a learning signal.


@pytest.mark.parametrize("model", [{"scope_budget": 16}, {"static_scope": True},
                                    {"goal_sufficient": False}])
def test_supervision_preflight_rejects_noncompetitive_or_nondifferentiable_selector(model):
    config = supervision_config(**model)
    audit = scope_supervision_audit(pair(), Tensorizer(config), config)
    assert audit["decision_retrievable_groups"] == 1
    assert audit["decision_trainable_groups"] == 0
    if model.get("scope_budget") == 16:
        assert audit["decision_budget_competitive_groups"] == 0


def test_supervision_preflight_excludes_guaranteed_hash_coverage_fields():
    records = pair()
    amount = "shop.Order[1].amount"  # Sorted slot zero has the first hash.
    goals = [{"id": "amount", "field": amount, "op": "eq", "value": 20., "protected": True}]
    for record in records:
        record["goals"] = deepcopy(goals)
        record["goal_before"] = [True]
        record["goal_after"] = [record["cf_role"] == "reference"]
    records[0]["audit_next_state"]["fields"][2]["value"] = 21.
    records[0]["audit"]["changes"].append(amount)
    config = supervision_config(scope_budget=2)
    audit = scope_supervision_audit(records, Tensorizer(config), config)
    assert audit["decision_retrievable_groups"] == audit["decision_budget_competitive_groups"] == 1
    assert audit["contribution_fields_fixed_coverage"] == 1
    assert audit["decision_trainable_groups"] == 0
    # B=1 has no reservation: this field can again be selected or omitted.
    config = supervision_config(scope_budget=1)
    assert scope_supervision_audit(records, Tensorizer(config), config)["decision_trainable_groups"] == 1


def test_supervision_preflight_detects_constant_utility_across_competing_fields():
    records = pair()
    fields = sorted(records[0]["state"]["fields"], key=lambda item: item["key"])
    variable_keys = {field["key"] for field in fields[1:]}  # Slot zero is guaranteed coverage at B=2.
    goals = [{"id": key, "field": key, "op": "exists", "protected": True} for key in sorted(variable_keys)]
    # Use explicit present before snapshots for every competing field.
    for record in records:
        record["goals"] = deepcopy(goals)
        record["goal_before"] = [True] * len(goals)
        record["goal_after"] = [record["cf_role"] == "reference"] * len(goals)
        for state_name in ("state", "audit_next_state"):
            for field in record[state_name]["fields"]:
                if field["key"] in variable_keys:
                    field.update(exists=True, known=True)
                    if field["value"] is None:
                        field["value"] = "standard"
        if record["cf_role"] == "action":
            for field in record["audit_next_state"]["fields"]:
                if field["key"] in variable_keys:
                    field.update(exists=False, value=None)
            record["audit"]["changes"] = sorted(variable_keys)
        else:
            record["audit_next_state"] = deepcopy(record["state"])
            record["audit"]["changes"] = []
    config = supervision_config()
    audit = scope_supervision_audit(records, Tensorizer(config), config)
    assert audit["decision_informative_groups"] == audit["decision_retrievable_groups"] == 1
    assert audit["contribution_fields_competitive"] > 0
    assert audit["decision_trainable_groups"] == 0
