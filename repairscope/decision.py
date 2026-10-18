"""Offline, goal-sufficient scope supervision from matched action branches.

This is an oracle-patch action-selection surrogate: a restricted scope reveals
only audited field utility increments.  It is not regret of the learned state
predictor.  Full utilities always retain all goal fields, including fields that
the public candidate retriever could not supply to the scope model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import ast
import math
from typing import Any, Iterable, Mapping

import torch
from torch.nn import functional as F

from .schema import effective_protected, evaluate_goals, field_map, semantic_diff
from . import schema as _schema


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


def _generic_goal(goal: Mapping[str, Any]) -> bool:
    """Submission plumbing cannot stand in for a measurable business target."""
    return (goal.get("semantic_class") == "submission" or goal.get("kind") == "complete"
            or goal.get("kind") == "answer" and "expr" not in goal and goal.get("value") in (None, "", "<<NOT_GIVEN>>"))


def _evaluate(state, goals, previous):
    if hasattr(_schema, "goal_state_keys"):
        return evaluate_goals(state, goals, previous=previous)
    return evaluate_goals(state, goals)  # Atomic compatibility before schema upgrade.


def _dependencies(goal, before, after, audit):
    """Bind every dependency; dynamic lookup requires a closed audited universe."""
    if goal.get("where"):
        # The approved schema currently validates where but does not evaluate
        # it. Silently dropping its filter would invent a different objective.
        return [], "unsupported_where", False
    patterns = list(goal.get("fields") or ([goal["field"]] if goal.get("field") else []))
    dynamic = any("*" in pattern for pattern in patterns)
    kind = goal.get("kind", "predicate")
    if kind == "answer" and not patterns:
        patterns = sorted({key for state in (before, after) for key in field_map(state)
                           if key.endswith(".answer") or key.endswith("].answer")})
        dynamic = True  # Implicit first-answer discovery needs a closed universe.
    if not patterns:
        return [], "unbound_goal_dependencies", False
    expressions = [goal[key] for key in ("expr", "filter_expr") if key in goal]
    value = goal.get("value")
    if isinstance(value, dict):
        if "expr" in value:
            expressions.append(value["expr"])
        elif "field" in value and isinstance(value["field"], str):
            patterns.append(value["field"])
        else:
            return [], "unsupported_target", False
    names = {"value", "current", "key", "previous_value", "previous_known", "previous_exists",
             "min", "max", "abs", "round", "len", "field", "aggregate", "previous_field",
             "previous_aggregate", "number", "text", "bool"}
    for expression in expressions:
        dynamic = True  # Even literal expression lookups require full certification.
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, TypeError):
            return [], "unsupported_expression", False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in names:
                return [], "unbound_expression", False
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.keywords:
                return [], "unbound_expression", False
            call = node.func.id
            if call in ("field", "previous_field"):
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    return [], "unbound_expression", False
                patterns.append(node.args[0].value)
            elif call in ("aggregate", "previous_aggregate"):
                if (len(node.args) != 2 or any(not isinstance(arg, ast.Constant) or not isinstance(arg.value, str) for arg in node.args)
                        or node.args[0].value not in ("count", "sum", "mean", "min", "max")):
                    return [], "unbound_expression", False
                patterns.append(node.args[1].value)
    dynamic |= any("*" in pattern for pattern in patterns)
    if dynamic and not audit["complete"]:
        return [], "dynamic_goal_requires_complete_audit", False
    keys = set()
    for pattern in patterns:
        if "*" in pattern:
            if not hasattr(_schema, "goal_state_keys"):
                return [], "unsupported_goal_schema", False
            matched = set(_schema.goal_state_keys({"field": pattern}, before)) | set(_schema.goal_state_keys({"field": pattern}, after))
            if not matched:
                return [], "unbound_goal_dependencies", False
            keys.update(matched)
        else:
            keys.add(pattern)
    atomic = (kind == "predicate" and not goal.get("quantifier") and not goal.get("fields")
              and not expressions and not isinstance(value, dict) and len(keys) == 1 and not dynamic)
    return sorted(keys), None, atomic


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
        derived_before = _evaluate(before, goals, before)
        derived_after = _evaluate(after, goals, before)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return [], "invalid_goal_predicate"
    left, right = field_map(before), field_map(after)
    domain, changes = set(audit["field_domain"]), set(audit["changes"])
    evidence = []
    for i, goal in enumerate(goals):
        keys, dependency_reason, atomic = _dependencies(goal, before, after, audit) if not _generic_goal(goal) else ([], None, False)
        before_known = (not dependency_reason and bool(keys) and type(labels_before[i]) is bool
                        and derived_before[i] is labels_before[i]
                        and all(key in domain and key in left and left[key].get("known") is True
                                and type(left[key].get("exists")) is bool for key in keys))
        after_known = (not dependency_reason and bool(keys) and type(labels_after[i]) is bool
                       and derived_after[i] is labels_after[i]
                       and all(key in domain and key in right and right[key].get("known") is True
                               and type(right[key].get("exists")) is bool for key in keys))
        known = before_known and after_known
        # The audit diff and known snapshots must agree on a goal field. A
        # changed field is not, by itself, evidence that its goal was damaged.
        if known:
            try:
                known = all(bool(semantic_diff({"fields": [left[key]]}, {"fields": [right[key]]})) == (key in changes)
                            for key in keys)
            except (KeyError, TypeError, ValueError):
                known = False
        evidence.append({"known": bool(known), "before_known": bool(before_known),
                         "after_known": bool(after_known), "before": labels_before[i],
                         "after": labels_after[i], "field": goal.get("field"), "fields": keys,
                         "atomic": atomic, "reason": dependency_reason, "generic": _generic_goal(goal)})
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
    business, old_values, relevant = [], [], set()
    full_delta, hybrid = 0., False
    step = record.get("step", 0)
    if type(step) is not int or step < 0:
        return None, "invalid_step"
    for goal, item in zip(goals, evidence):
        weight = goal.get("weight", 1.)
        if type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0:
            return None, "invalid_goal_weight"
        if not weight or item["generic"]:
            continue
        if not item["known"]:
            return None, item["reason"] or "unknown_goal_evidence"
        old, new = item["before"], item["after"]
        damage = old is True and effective_protected(goal, step, old) and new is False
        increment = float(weight) * (progress_weight * (int(new) - int(old)) - damage_weight * int(damage))
        full_delta += increment
        business.append(goal)
        old_values.append(old)
        if item["atomic"]:
            contributions[item["field"]] += increment
        else:
            hybrid = True
            relevant.update(set(item["fields"]) & set(record["audit"]["changes"]))
    if not business:
        return None, "no_business_goals"
    relevant.update(key for key, value in contributions.items() if abs(value) > 1e-10)
    action = record.get("action", {})
    cost = action.get("cost", 1.) if isinstance(action, dict) else None
    if type(cost) not in (float, int) or not math.isfinite(cost) or cost <= 0:
        return None, "invalid_action_cost"
    baseline = -cost_weight * float(cost)
    candidate = {"index": index, "role": record.get("cf_role"), "baseline": baseline,
                 "contributions": dict(contributions),
                 "full_utility": baseline + full_delta,
                 "utility_mode": "oracle_hybrid" if hybrid else "additive",
                 "relevant_fields": sorted(relevant)}
    if hybrid:
        _, before, after, _ = _audit(record)
        candidate["oracle"] = {"before": deepcopy(before), "after": deepcopy(after),
                               "goals": deepcopy(business), "before_values": old_values,
                               "step": step, "progress_weight": progress_weight, "damage_weight": damage_weight}
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
    keys, valid, business = set(), True, 0
    joint, joint_fields = [], set()
    failure_reason = None
    for goal, item in zip(goals, evidence):
        weight = goal.get("weight", 1.)
        if type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0:
            valid = False
            continue
        # Passing True checks whether the public interval permits protection,
        # without silently treating an unknown achieved-state label as False.
        if not weight or item["generic"]:
            continue
        business += 1
        if not effective_protected(goal, step, True):
            continue
        if not item["before_known"]:
            valid = False
            failure_reason = item["reason"] or "unknown_protection_evidence"
        elif item["before"] is True:
            if not item["known"]:
                valid = False
                failure_reason = item["reason"] or "unknown_protection_evidence"
            elif item["after"] is False:
                if item["atomic"]:
                    keys.add(item["field"])
                    continue
                audit, before, after, _ = _audit(record)
                changed = set(item["fields"]) & set(audit["changes"])
                critical = set()
                for key in changed:
                    rollback = _hybrid_state(before, after, changed - {key})
                    try:
                        restored = _evaluate(rollback, [goal], before)[0]
                    except (KeyError, TypeError, ValueError, ArithmeticError):
                        restored = None
                    if restored is True:
                        critical.add(key)
                    elif restored is None:
                        valid = False
                        failure_reason = "unknown_compound_attribution"
                keys.update(critical)
                # A changed dependency without a single-field witness may
                # participate jointly. Do not fabricate an exhaustive ranking
                # of relevant fields or a complete escape negative for it.
                if changed - critical or not changed:
                    valid = False
                    failure_reason = "joint_protection_attribution"
                    joint.append(goal.get("id"))
                    joint_fields.update(changed - critical)
    positives = sorted(keys)
    if not business:
        valid, failure_reason = False, "no_business_goals"
    result = {"valid": valid, "keys": positives, "observed_positive_keys": positives,
              "reason": None if valid else failure_reason or "unknown_protection_evidence"}
    if joint:
        result.update(joint_goals=joint, joint_fields=sorted(joint_fields))
    return result


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
            has_signal = any(candidate["relevant_fields"] for candidate in group["candidates"])
            group.update(tied=is_tied, informative=has_signal and not is_tied)
            tied += int(is_tied)
            informative += int(group["informative"])
        groups.append(group)
    return {"groups": groups, "decision_groups": len(groups), "decision_candidates": candidates,
            "decision_eligible_groups": eligible, "decision_masked_groups": len(groups) - eligible,
            "decision_tied_groups": tied, "decision_informative_groups": informative,
            "decision_compound_groups": sum(group["eligible"] and any(candidate["utility_mode"] == "oracle_hybrid" for candidate in group["candidates"]) for group in groups),
            "decision_generic_only_groups": reasons["no_business_goals"],
            "goal_generic_ignored": sum(_generic_goal(goal) for record in records for goal in record.get("goals", []) if isinstance(goal, dict)),
            "goal_business_total": sum(not _generic_goal(goal) and goal.get("weight", 1.) > 0 for record in records for goal in record.get("goals", []) if isinstance(goal, dict)),
            "decision_unsupported_groups": sum(count for reason, count in reasons.items() if reason.startswith(("unsupported", "unbound", "dynamic_goal"))),
            "mask_reasons": dict(reasons)}


def _hybrid_state(before, after, scope):
    result = deepcopy(before)
    selected, right = set(scope), field_map(after)
    result["fields"] = [deepcopy(right[field["key"]]) if field["key"] in selected and field["key"] in right else field
                        for field in result["fields"]]
    return result


def restricted_utility(candidate: Mapping[str, Any], scope: Iterable[str]) -> float | None:
    """Evaluate an exact hard scope against the saved oracle field patch."""
    selected = set(scope)
    if candidate.get("utility_mode") == "oracle_hybrid":
        oracle = candidate["oracle"]
        hybrid = _hybrid_state(oracle["before"], oracle["after"], selected)
        try:
            after = _evaluate(hybrid, oracle["goals"], oracle["before"])
        except (KeyError, TypeError, ValueError, ArithmeticError):
            return None
        utility = float(candidate["baseline"])
        for goal, old, new in zip(oracle["goals"], oracle["before_values"], after):
            if type(old) is not bool or type(new) is not bool:
                return None
            damage = old is True and effective_protected(goal, oracle["step"], old) and new is False
            utility += float(goal.get("weight", 1.)) * (oracle["progress_weight"] * (int(new) - int(old))
                                                       - oracle["damage_weight"] * int(damage))
        return utility
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
              "contribution_fields_competitive": 0,
              "decision_compound_scope_probes": 0, "decision_compound_unknown_probes": 0}
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
            nonzero = set(candidate["relevant_fields"])
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
            if candidate["utility_mode"] == "oracle_hybrid" and candidate_competes and selectable:
                # Conservative O(n) witness search over legal B-field scopes,
                # not an additive approximation of a nonlinear predicate.
                scopes = []
                slots = count - len(fixed)
                for order in (variable_keys, sorted(variable_keys, key=lambda key: (key not in nonzero, key))):
                    seed = set(order[:slots])
                    scopes.append(fixed | seed)
                    outside = [key for key in order if key not in seed]
                    inside = [key for key in order if key in seed]
                    for key in variable_keys:
                        if key in seed and outside:
                            scopes.append(fixed | (seed - {key}) | {outside[0]})
                        elif key not in seed and inside:
                            scopes.append(fixed | (seed - {inside[-1]}) | {key})
                values = []
                for scope in {frozenset(scope) for scope in scopes}:
                    value = restricted_utility(candidate, scope)
                    totals["decision_compound_scope_probes"] += 1
                    totals["decision_compound_unknown_probes"] += int(value is None)
                    if value is not None:
                        values.append(value)
                variable_utility = bool(values) and max(values) - min(values) > 1e-10
            else:
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
    terms, compound_mask_reasons = [], Counter()
    for group in built["groups"]:
        if not group["eligible"] or not group["informative"]:
            continue
        restricted = []
        valid = True
        for candidate in group["candidates"]:
            row = candidate["index"]
            keys = batch["metadata"][row]["keys"]
            if len(keys) > gate.shape[1] or len(keys) != len(set(keys)):
                raise ValueError("Decision candidate field metadata is invalid")
            if candidate["utility_mode"] == "oracle_hybrid":
                selected = {key for slot, key in enumerate(keys) if gate[row, slot].detach().item() > .5}
                exact = restricted_utility(candidate, selected)
                if exact is None:
                    valid = False
                    compound_mask_reasons["unknown_restricted_goal"] += 1
                    break
                coefficients = []
                for key in keys:
                    if key not in candidate["relevant_fields"]:
                        coefficients.append(0.)
                        continue
                    on = exact if key in selected else restricted_utility(candidate, selected | {key})
                    off = restricted_utility(candidate, selected - {key}) if key in selected else exact
                    if on is None or off is None:
                        valid = False
                        compound_mask_reasons["unknown_finite_difference"] += 1
                        break
                    coefficients.append(on - off)
                if not valid:
                    break
                coefficient_tensor = gate.new_tensor(coefficients)
                restricted.append(gate.new_tensor(exact) + (gate[row, :len(keys)] - gate[row, :len(keys)].detach()).dot(coefficient_tensor))
            else:
                coefficients = gate.new_tensor([candidate["contributions"].get(key, 0.) for key in keys])
                restricted.append(gate[row, :len(keys)].dot(coefficients) + candidate["baseline"])
        if not valid:
            continue
        full = gate.new_tensor([candidate["full_utility"] for candidate in group["candidates"]])
        teacher = F.softmax(full / temperature, dim=0).detach()
        terms.append(F.kl_div(F.log_softmax(torch.stack(restricted) / temperature, dim=0),
                              teacher, reduction="sum"))
    zero = gate.sum() * 0.
    return {"goal_sufficiency": torch.stack(terms).mean() if terms else zero,
            **{key: value for key, value in built.items() if key != "groups"},
            "decision_loss_groups": len(terms), "actual_label_loss_groups": len(terms),
            "compound_mask_reasons": dict(compound_mask_reasons)}
