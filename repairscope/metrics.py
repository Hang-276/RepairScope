"""Auditable metrics. Synthetic checkers never produce official benchmark scores."""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .control import evaluate_goals, goal_transition, protected_mask


def _mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def _changed_keys(audit):
    changes = audit.get("changes", audit.get("changed_fields", []))
    if isinstance(changes, dict):
        return set(changes)
    return {item if isinstance(item, str) else item.get("key", item.get("field")) for item in changes} - {None}


def calibration(probabilities, labels, bins=10):
    pairs = [(min(1.0, max(0.0, float(p))), int(y)) for p, y in zip(probabilities, labels)]
    if not pairs:
        return {"brier": None, "ece": None, "bins": [], "n": 0, "risk_coverage": []}
    reliability, ece = [], 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [(p, y) for p, y in pairs if low <= p and (p < high or index == bins - 1)]
        confidence, frequency = _mean([p for p, _ in bucket]), _mean([y for _, y in bucket])
        if bucket:
            ece += len(bucket) / len(pairs) * abs(confidence - frequency)
        reliability.append({"low": low, "high": high, "count": len(bucket),
                            "predicted_risk": confidence, "observed_risk": frequency})
    ordered = sorted(pairs)
    curve = [{"coverage": n / len(ordered), "escape_rate": _mean([y for _, y in ordered[:n]])}
             for n in sorted(set(max(1, round(len(ordered) * fraction / 10)) for fraction in range(1, 11)))]
    return {"brier": _mean([(p - y) ** 2 for p, y in pairs]), "ece": ece,
            "bins": reliability, "n": len(pairs), "risk_coverage": curve}


def macro_f1(truth, predicted, classes=4):
    if not truth:
        return None
    scores = []
    for label in range(classes):
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return _mean(scores)


def typed_patch_metrics(records):
    correct, total, numeric_errors, by_kind = 0, 0, [], defaultdict(lambda: [0, 0])
    operation_correct, operation_total = 0, 0
    for record in records:
        prediction = record.get("prediction", {})
        predicted_state = prediction.get("next_state", {})
        truth = {f["key"]: f for f in record.get("next_state", {}).get("fields", [])}
        predicted = {f["key"]: f for f in predicted_state.get("fields", [])}
        previous = {f["key"]: f for f in record.get("state", {}).get("fields", [])}
        predicted_ops = {patch["key"]: patch.get("op", "keep") for patch in prediction.get("patch", [])}
        audit = record.get("audit", {})
        domain = set(audit.get("field_domain", truth))
        for key in domain:
            target, guess = truth.get(key), predicted.get(key)
            if target is None or not target.get("known", True):
                continue
            source = previous.get(key)
            if source and source.get("known", True):
                from .schema import semantic_diff
                if not source.get("exists", True) and target.get("exists", True):
                    op = "create"
                elif source.get("exists", True) and not target.get("exists", True):
                    op = "delete"
                else:
                    op = "modify" if semantic_diff({"fields": [source]}, {"fields": [target]}) else "keep"
                operation_correct += predicted_ops.get(key, "keep") == op
                operation_total += 1
            total += 1
            kind = target.get("kind", "text")
            by_kind[kind][1] += 1
            same = bool(guess and guess.get("known", True) and guess.get("exists", True) == target.get("exists", True)
                        and guess.get("value") == target.get("value"))
            correct += same
            by_kind[kind][0] += same
            if kind == "number" and guess and guess.get("known", True):
                try:
                    numeric_errors.append(abs(float(target["value"]) - float(guess["value"])))
                except (TypeError, ValueError):
                    pass
    return {"field_exact_accuracy": correct / total if total else None, "n_fields": total,
            "operation_accuracy": operation_correct / operation_total if operation_total else None,
            "numeric_mae": _mean(numeric_errors), "by_kind": {k: {"accuracy": a / n, "n": n} for k, (a, n) in by_kind.items()}}


def compute_metrics(records: list[dict]) -> dict:
    covered = changed = zero_change = escaped = escape_n = 0
    scopes, probs, labels, goal_truth, goal_pred = [], [], [], [], []
    regression = protected_n = recovery = repair_damaged_n = repair_new_damage = repair_protected_n = 0
    for record in records:
        audit, prediction = record.get("audit", {}), record.get("prediction", {})
        scope = set(record.get("scope", prediction.get("scope", [])))
        scopes.append(len(scope))
        if audit:
            actual = _changed_keys(audit)
            if actual:
                covered += len(actual & scope)
                changed += len(actual)
            elif audit.get("complete", False):
                zero_change += 1
            # A positive is observed even in a partial audit; a zero requires completeness.
            miss = bool(actual - scope)
            if miss or audit.get("complete", False):
                escaped += miss
                escape_n += 1
                if "escape" in prediction:
                    probs.append(prediction["escape"])
                    labels.append(int(miss))
        goals = record.get("goals", [])
        before = record.get("goal_before", evaluate_goals(record.get("state", {}), goals))
        after = record.get("goal_after", evaluate_goals(record.get("next_state", {}), goals))
        transitions = goal_transition(before, after)
        predicted_after = evaluate_goals(prediction.get("next_state", {}), goals)
        predicted_transitions = goal_transition(before, predicted_after)
        for i, transition in enumerate(transitions):
            if transition is None:
                continue
            head = prediction.get("goal_probabilities", [])
            guess = predicted_transitions[i]
            if guess is None and i < len(head):
                guess = max(range(4), key=lambda k: head[i][k])
            if guess is not None:
                goal_truth.append(transition)
                goal_pred.append(guess)
        protection = protected_mask(goals, before, int(record.get("step", 0)))
        for i, protected in enumerate(protection):
            if protected and after[i] is not None:
                protected_n += 1
                regression += after[i] is False
                if record.get("phase") == "repair":
                    repair_protected_n += 1
                    repair_new_damage += after[i] is False
            if record.get("phase") == "repair" and before[i] is False and after[i] is not None:
                repair_damaged_n += 1
                recovery += after[i] is True
    return {"scope_recall_at_budget": covered / changed if changed else None,
            "scope_changed_fields": changed, "scope_covered_fields": covered, "zero_change_steps": zero_change,
            "escape_rate": escaped / escape_n if escape_n else None, "escape_evaluable_steps": escape_n,
            "mean_scope_size": _mean(scopes), "escape_calibration": calibration(probs, labels),
            "goal_four_state_macro_f1": macro_f1(goal_truth, goal_pred), "goal_labeled_transitions": len(goal_truth),
            "goal_regression_rate": regression / protected_n if protected_n else None,
            "protected_goal_steps": protected_n, "repair_recovery_rate": recovery / repair_damaged_n if repair_damaged_n else None,
            "goal_regressions": regression,
            "repair_new_damage_rate": repair_new_damage / repair_protected_n if repair_protected_n else None,
            "typed_patch": typed_patch_metrics(records)}


def official_metrics(results: list[dict], expected_task_ids=None):
    """Match AppWorld Metric: TGC mean success; SGC mean minimum per generator.

    Scores use the native percentage scale (0..100). Require native checkers and
    complete planned coverage; absence/fixture failure never becomes a fake zero.
    """
    by_task, scenarios = {}, defaultdict(list)
    for result in results:
        checker = result.get("official", result)
        origin = str(checker.get("evidence_origin", result.get("evidence_origin", "")))
        if origin != "appworld":
            continue
        tracker = checker.get("tracker", {})
        if "success" not in tracker:
            continue
        task_id = str(checker.get("task_id", result.get("task_id", "")))
        if not task_id:
            continue
        by_task[task_id] = bool(tracker["success"])
    expected = set(map(str, expected_task_ids or [r.get("task_id") for r in results]))
    if not expected or set(by_task) != expected:
        return {"TGC": None, "SGC": None, "native_tasks": len(by_task), "planned_tasks": len(expected),
                "complete_native_coverage": False}
    for task_id, success in by_task.items():
        # Official IDs have one '_' separating generator and task number.
        scenarios[task_id.split("_")[0]].append(float(success))
    return {"TGC": round(100 * _mean(list(by_task.values())), 1),
            "SGC": round(100 * _mean([min(values) for values in scenarios.values()]), 1),
            "native_tasks": len(by_task), "planned_tasks": len(expected), "complete_native_coverage": True}


def grouped_bootstrap(results, key, samples=1000, seed=0):
    groups = defaultdict(list)
    for result in results:
        value = result.get(key)
        if value is not None:
            groups[str(result.get("group_id", str(result.get("task_id", "")).split("_")[0]))].append(float(value))
    if not groups:
        return None
    rng, names = random.Random(seed), sorted(groups)
    estimates = []
    for _ in range(samples):
        chosen = [rng.choice(names) for _ in names]
        estimates.append(_mean([v for name in chosen for v in groups[name]]))
    estimates.sort()
    if not estimates:
        return None
    return {"low": estimates[int((len(estimates) - 1) * 0.025)],
            "high": estimates[int((len(estimates) - 1) * 0.975)], "groups": len(groups), "samples": samples}


def aggregate_runs(results, bootstrap_samples=1000, seed=0):
    records = [record for result in results for record in result.get("logs", [])]
    summary = {"tasks": len(results), **official_metrics(results), "metrics": compute_metrics(records),
               "cost": sum(float(r.get("cost", 0)) for r in results)}
    per_task = [{"task_id": r.get("task_id"), "group_id": r.get("group_id", str(r.get("task_id", "")).split("_")[0]),
                 **compute_metrics(r.get("logs", []))} for r in results]
    def ratio_interval(numerator, denominator):
        groups = defaultdict(list)
        for task in per_task:
            groups[task["group_id"]].append(task)
        if not groups or not sum(float(t.get(denominator, 0)) for t in per_task):
            return None
        rng, names, values = random.Random(seed), sorted(groups), []
        for _ in range(bootstrap_samples):
            sampled = [task for _ in names for task in groups[rng.choice(names)]]
            bottom = sum(float(t.get(denominator, 0)) for t in sampled)
            if bottom:
                top = sum(float(t.get(numerator, 0)) for t in sampled)
                values.append(top / bottom)
        values.sort()
        if not values:
            return None
        return {"low": values[int((len(values)-1) * .025)], "high": values[int((len(values)-1) * .975)],
                "groups": len(groups), "samples": bootstrap_samples}
    for task in per_task:
        task["escapes"] = (task["escape_rate"] or 0) * task["escape_evaluable_steps"]
    summary["bootstrap_95"] = {
        "scope_recall_at_budget": ratio_interval("scope_covered_fields", "scope_changed_fields"),
        "escape_rate": ratio_interval("escapes", "escape_evaluable_steps"),
        "goal_regression_rate": ratio_interval("goal_regressions", "protected_goal_steps")}
    native = []
    if summary["complete_native_coverage"]:
        for result in results:
            official = result.get("official", result)
            native.append({"task_id": result["task_id"], "group_id": result.get("group_id", str(result["task_id"]).split("_")[0]),
                           "TGC": 100 * float(official["tracker"]["success"])})
        summary["bootstrap_95"]["TGC"] = grouped_bootstrap(native, "TGC", bootstrap_samples, seed)
        by_group = defaultdict(list)
        for task in native:
            by_group[task["group_id"]].append(task["TGC"])
        summary["bootstrap_95"]["SGC"] = grouped_bootstrap(
            [{"group_id": group, "SGC": min(values)} for group, values in by_group.items()], "SGC", bootstrap_samples, seed)
    return summary
