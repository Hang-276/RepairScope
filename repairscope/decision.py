"""Offline, goal-sufficient scope supervision from matched action branches.

This is an oracle-patch action-selection surrogate: a restricted scope reveals
only audited field utility increments.  It is not regret of the learned state
predictor.  Full utilities always retain all goal fields, including fields that
the public candidate retriever could not supply to the scope model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable, Mapping

import torch
from torch.nn import functional as F

from .schema import effective_protected, evaluate_goals, field_map, semantic_diff


def _weights(config: Mapping[str, Any]) -> tuple[float, float, float, float]:
    controller = config.get("controller", {})
    decision = config.get("goal_decision", {})
    values = tuple(float(decision.get(key, controller.get(key, default)))
                   for key, default in (("progress_weight", 1.), ("damage_weight", 4.),
                                        ("cost_weight", .02)))
    temperature = float(decision.get("temperature", 1.))
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Goal decision weights must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Goal decision temperature must be finite and positive")
    return (*values, temperature)


def _audit(record: Mapping[str, Any]) -> tuple[dict, dict, dict, str | None]:
    """Require explicit audit coverage; observed state alone proves no negative."""
    audit = record.get("audit")
    if not isinstance(audit, dict) or type(audit.get("complete")) is not bool:
        return {}, {}, {}, "missing_audit"
    domain, changes = audit.get("field_domain"), audit.get("changes")
    if (not isinstance(domain, list) or not isinstance(changes, list)
            or any(not isinstance(key, str) or not key for key in domain + changes)
            or len(domain) != len(set(domain)) or len(changes) != len(set(changes))
            or not set(changes) <= set(domain)):
        return {}, {}, {}, "invalid_audit"
    before = record.get("audit_state") or record.get("state")
    after = record.get("audit_next_state") or record.get("next_state")
    for state in (before, after):
        if not isinstance(state, dict) or not isinstance(state.get("fields"), list):
            return {}, {}, {}, "missing_audit_state"
        fields = state["fields"]
        if (any(not isinstance(item, dict) or not isinstance(item.get("key"), str) for item in fields)
                or len({item["key"] for item in fields}) != len(fields)):
            return {}, {}, {}, "invalid_audit_state"
    return audit, before, after, None


def _goal_evidence(record: Mapping[str, Any], goals: list[dict]) -> tuple[list[dict], str | None]:
    audit, before, after, reason = _audit(record)
    if reason:
        return [], reason
    labels_before, labels_after = record.get("goal_before"), record.get("goal_after")
    if (not isinstance(labels_before, list) or not isinstance(labels_after, list)
            or len(labels_before) != len(goals) or len(labels_after) != len(goals)
            or any(value is not None and type(value) is not bool for value in labels_before + labels_after)):
        return [], "invalid_goal_labels"
    try:
        derived_before = evaluate_goals(before, goals)
        derived_after = evaluate_goals(after, goals)
    except (KeyError, TypeError, ValueError):
        return [], "invalid_goal_predicate"
    left, right = field_map(before), field_map(after)
    domain, changes = set(audit["field_domain"]), set(audit["changes"])
    evidence = []
    for i, goal in enumerate(goals):
        key = goal.get("field")
        before_known = (key in domain and key in left and left[key].get("known") is True
                        and type(left[key].get("exists")) is bool and type(labels_before[i]) is bool
                        and derived_before[i] is labels_before[i])
        after_known = (key in domain and key in right and right[key].get("known") is True
                       and type(right[key].get("exists")) is bool and type(labels_after[i]) is bool
                       and derived_after[i] is labels_after[i])
        known = before_known and after_known
        # The audit diff and known snapshots must agree on a goal field. A
        # changed field is not, by itself, evidence that its goal was damaged.
        if known:
            try:
                changed = bool(semantic_diff({"fields": [left[key]]}, {"fields": [right[key]]}))
                known = changed == (key in changes)
            except (KeyError, TypeError, ValueError):
                known = False
        evidence.append({"known": bool(known), "before_known": bool(before_known),
                         "after_known": bool(after_known), "before": labels_before[i],
                         "after": labels_after[i], "field": key})
    return evidence, None


def _candidate(record: Mapping[str, Any], index: int, config: Mapping[str, Any]) -> tuple[dict | None, str | None]:
    if record.get("action_valid") is not True:
        return None, "invalid_action" if record.get("action_valid") is False else "unknown_action_valid"
    goals = record.get("goals")
    if not isinstance(goals, list) or any(not isinstance(goal, dict) for goal in goals):
        return None, "invalid_goals"
    evidence, reason = _goal_evidence(record, goals)
    if reason:
        return None, reason
    progress_weight, damage_weight, cost_weight, _ = _weights(config)
    contributions = defaultdict(float)
    step = record.get("step", 0)
    if type(step) is not int or step < 0:
        return None, "invalid_step"
    for goal, item in zip(goals, evidence):
        weight = goal.get("weight", 1.)
        if type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0:
            return None, "invalid_goal_weight"
        if not weight:
            continue
        if not item["known"]:
            return None, "unknown_goal_evidence"
        old, new = item["before"], item["after"]
        damage = old is True and effective_protected(goal, step, old) and new is False
        contributions[item["field"]] += float(weight) * (progress_weight * (int(new) - int(old))
                                                         - damage_weight * int(damage))
    action = record.get("action", {})
    cost = action.get("cost", 1.) if isinstance(action, dict) else None
    if type(cost) not in (float, int) or not math.isfinite(cost) or cost <= 0:
        return None, "invalid_action_cost"
    baseline = -cost_weight * float(cost)
    candidate = {"index": index, "role": record.get("cf_role"), "baseline": baseline,
                 "contributions": dict(contributions),
                 "full_utility": baseline + sum(contributions.values())}
    return candidate, None


def goal_relevant_evidence(record: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    """Return audited protection-changing fields, and whether negatives are safe.

    ``valid`` requires every positive-weight goal that could currently be
    protected to have a known before label; achieved protected goals also need
    a reliable after label and audited field. ``keys`` contains only confirmed
    True-to-False protected goals. A partial positive remains available in
    ``observed_positive_keys``, but an unknown goal never supplies a negative.
    Explicit ``action_valid=True`` is required; an omitted execution-validity
    label is unknown, including on otherwise schema-compatible legacy records.
    """
    _weights(config)
    if record.get("action_valid") is not True:
        reason = "invalid_action" if record.get("action_valid") is False else "unknown_action_valid"
        return {"valid": False, "keys": [], "observed_positive_keys": [], "reason": reason}
    goals = record.get("goals")
    if not isinstance(goals, list) or any(not isinstance(goal, dict) for goal in goals):
        return {"valid": False, "keys": [], "observed_positive_keys": [], "reason": "invalid_goals"}
    evidence, reason = _goal_evidence(record, goals)
    if reason:
        return {"valid": False, "keys": [], "observed_positive_keys": [], "reason": reason}
    step = record.get("step", 0)
    if type(step) is not int or step < 0:
        return {"valid": False, "keys": [], "observed_positive_keys": [], "reason": "invalid_step"}
    keys, valid = set(), True
    for goal, item in zip(goals, evidence):
        weight = goal.get("weight", 1.)
        if type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0:
            valid = False
            continue
        # Passing True checks whether the public interval permits protection,
        # without silently treating an unknown achieved-state label as False.
        if not weight or not effective_protected(goal, step, True):
            continue
        if not item["before_known"]:
            valid = False
        elif item["before"] is True:
            if not item["known"]:
                valid = False
            elif item["after"] is False:
                keys.add(item["field"])
    positives = sorted(keys)
    return {"valid": valid, "keys": positives, "observed_positive_keys": positives,
            "reason": None if valid else "unknown_protection_evidence"}


def decision_groups(records: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> dict:
    """Build strict two-action groups from each counterfactual pair/condition.

    The returned ``groups`` includes masked groups with explicit reasons. Each
    eligible candidate retains a full ``contributions`` map, independent of any
    tensorizer candidate limit. Duplicate cells invalidate their whole group.
    ``decision_groups`` counts observed pair/condition groups; candidate counts
    refer only to eligible groups. Ties are eligible for evaluation but masked
    from the training loss. Both branches require explicit ``action_valid=True``;
    absence of an execution-validity label is masked rather than presumed legal.
    """
    _weights(config)
    # Local import keeps losses free to call this module without an import loop.
    from .losses import _matched_prefix

    records = list(records)
    cells = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        pair, condition, role = record.get("cf_pair_id"), record.get("cf_condition"), record.get("cf_role")
        if isinstance(pair, str) and pair and condition in ("c0", "c1") and role in ("action", "reference"):
            cells[(pair, condition)][role].append(index)
    groups, reasons = [], Counter()
    eligible, informative, tied, candidates = 0, 0, 0, 0
    for (pair, condition), roles in cells.items():
        group = {"pair_id": pair, "condition": condition, "indices": [], "eligible": False,
                 "informative": False, "reason": None, "candidates": []}
        reason = None
        if any(len(indices) > 1 for indices in roles.values()):
            reason = "duplicate_cell"
        elif set(roles) != {"action", "reference"}:
            reason = "unpaired"
        else:
            indices = [roles[role][0] for role in ("action", "reference")]
            group["indices"] = indices
            left, right = (records[index] for index in indices)
            if not _matched_prefix(records, *indices) or left.get("step", 0) != right.get("step", 0):
                reason = "unmatched_prefix"
            elif left.get("goals") != right.get("goals"):
                reason = "unmatched_goals"
            else:
                for index in indices:
                    candidate, reason = _candidate(records[index], index, config)
                    if reason:
                        break
                    group["candidates"].append(candidate)
        if reason:
            group.update(reason=reason, candidates=[])
            reasons[reason] += 1
        else:
            group["eligible"] = True
            eligible += 1
            candidates += len(group["candidates"])
            utilities = [candidate["full_utility"] for candidate in group["candidates"]]
            is_tied = math.isclose(utilities[0], utilities[1], rel_tol=0., abs_tol=1e-10)
            has_signal = any(abs(value) > 1e-10 for candidate in group["candidates"]
                             for value in candidate["contributions"].values())
            group.update(tied=is_tied, informative=has_signal and not is_tied)
            tied += int(is_tied)
            informative += int(group["informative"])
        groups.append(group)
    return {"groups": groups, "decision_groups": len(groups), "decision_candidates": candidates,
            "decision_eligible_groups": eligible, "decision_masked_groups": len(groups) - eligible,
            "decision_tied_groups": tied, "decision_informative_groups": informative,
            "mask_reasons": dict(reasons)}


def restricted_utility(candidate: Mapping[str, Any], scope: Iterable[str]) -> float:
    """Evaluate an exact hard scope against the saved oracle field patch."""
    selected = set(scope)
    return float(candidate["baseline"] + sum(value for key, value in candidate["contributions"].items()
                                             if key in selected))


def scope_supervision_audit(records: Iterable[Mapping[str, Any]], tensorizer, config: Mapping[str, Any]) -> dict:
    """Preflight whether audited decisions can teach a competitive scope scorer.

    Only public retrieval metadata is built. A positive KL value alone is not
    enough: omitted audit fields may be unretrievable, or the budget may select
    every candidate. The globally first ``reserve`` fields in the selector's
    hash permutation are always selected, whether by Top-k or by reservation.
    Their contributions cannot change under any hard scope decision.

    A trainable group is eligible and informative, has a learned v2 selector,
    and at least one candidate with more fields than the hard budget whose
    non-reserved field contributions differ. This is a necessary structural
    opportunity to learn; it does not assert a nonzero gradient for every set
    of model parameters. Counts of contribution fields are candidate-field
    incidences over informative eligible groups, rather than global field IDs.
    """
    records = list(records)
    built = decision_groups(records, config)
    model = config.get("model", config)
    budget = model.get("scope_budget", 128)
    if type(budget) is not int or budget < 1:
        raise ValueError("scope_budget must be a positive integer")
    static = bool(model.get("static_scope", False))
    learned_selector = bool(model.get("goal_sufficient", False)) and not static
    metadata = {}
    totals = {"decision_trainable_groups": 0, "decision_retrievable_groups": 0,
              "decision_budget_competitive_groups": 0,
              "contribution_fields_total": 0, "contribution_fields_retrievable": 0,
              "contribution_fields_fixed_coverage": 0,
              "contribution_fields_competitive": 0}
    for group in built["groups"]:
        if not group["eligible"] or not group["informative"]:
            continue
        retrievable, competitive, trainable = False, False, False
        for candidate in group["candidates"]:
            index = candidate["index"]
            if index not in metadata:
                metadata[index] = tensorizer._metadata(records[index])
            keys = metadata[index]["keys"]
            if len(keys) != len(set(keys)):
                raise ValueError("Decision candidate field metadata is invalid")
            contributions = candidate["contributions"]
            nonzero = {key for key, value in contributions.items() if abs(value) > 1e-10}
            retrieved = nonzero.intersection(keys)
            totals["contribution_fields_total"] += len(nonzero)
            totals["contribution_fields_retrievable"] += len(retrieved)
            retrievable |= bool(retrieved)
            count = min(budget, len(keys))
            reserve = min(max(1, count // 20), count - 1) if count and not static else 0
            permutation = sorted(range(len(keys)), key=lambda slot: (slot * 1103515245 + 12345) % 2147483647)
            fixed = {keys[slot] for slot in permutation[:reserve]}
            totals["contribution_fields_fixed_coverage"] += len(retrieved & fixed)
            candidate_competes = len(keys) > budget
            competitive |= candidate_competes
            variable_keys = [key for key in keys if key not in fixed]
            selectable = retrieved - fixed
            if candidate_competes:
                totals["contribution_fields_competitive"] += len(selectable)
            # With a fixed number of selected fields, equal utility increments
            # across every variable field also yield a constant hard utility.
            variable_values = [contributions.get(key, 0.) for key in variable_keys]
            variable_utility = bool(variable_values) and max(variable_values) - min(variable_values) > 1e-10
            trainable |= learned_selector and candidate_competes and bool(selectable) and variable_utility
        totals["decision_retrievable_groups"] += int(retrievable)
        totals["decision_budget_competitive_groups"] += int(competitive)
        totals["decision_trainable_groups"] += int(trainable)
    return {**{key: value for key, value in built.items() if key != "groups"}, **totals}


def goal_sufficiency_loss(output: dict, batch: dict, config: dict) -> dict:
    """Distil full-information action preferences directly through scope gates.

    ``scope_gate`` must have exact hard Top-k values in its forward pass and a
    differentiable budgeted surrogate in its backward pass. The predictor's
    patch and auxiliary goal heads are deliberately absent from this loss.
    """
    gate = output["scope_gate"]
    if gate.ndim != 2 or gate.shape[0] != len(batch["records"]):
        raise ValueError("scope_gate must have shape [records, candidate_fields]")
    if len(batch["metadata"]) != len(batch["records"]):
        raise ValueError("Decision metadata must align with records")
    built = decision_groups(batch["records"], config)
    temperature = _weights(config)[-1]
    terms = []
    for group in built["groups"]:
        if not group["eligible"] or not group["informative"]:
            continue
        restricted = []
        for candidate in group["candidates"]:
            row = candidate["index"]
            keys = batch["metadata"][row]["keys"]
            if len(keys) > gate.shape[1] or len(keys) != len(set(keys)):
                raise ValueError("Decision candidate field metadata is invalid")
            coefficients = gate.new_tensor([candidate["contributions"].get(key, 0.) for key in keys])
            restricted.append(gate[row, :len(keys)].dot(coefficients) + candidate["baseline"])
        full = gate.new_tensor([candidate["full_utility"] for candidate in group["candidates"]])
        teacher = F.softmax(full / temperature, dim=0).detach()
        terms.append(F.kl_div(F.log_softmax(torch.stack(restricted) / temperature, dim=0),
                              teacher, reduction="sum"))
    zero = gate.sum() * 0.
    return {"goal_sufficiency": torch.stack(terms).mean() if terms else zero,
            **{key: value for key, value in built.items() if key != "groups"},
            "decision_loss_groups": len(terms)}
