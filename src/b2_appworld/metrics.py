"""Mechanism diagnostics on labeled train/dev records, never native AppWorld SGC.

Four-state label metrics and paired candidate regret are distinct from official
whole-task TGC/SGC; this module intentionally does not derive SGC from goals.
"""
from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Mapping, Sequence

from .schemas import Example, IGNORE_LABEL, LABELS


def _probability_row(row: Sequence[float]) -> list[float]:
    values = [float(x) for x in row]
    if len(values) != 4 or any(not math.isfinite(x) or x < 0 or x > 1 for x in values):
        raise ValueError("Expected four finite probabilities in [0,1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-4):
        raise ValueError("Four-state probabilities must sum to one")
    return values


def classification_metrics(labels: Sequence[int], probabilities: Sequence[Sequence[float]],
                           n_bins: int = 15) -> dict[str, Any]:
    if len(labels) != len(probabilities) or n_bins < 1:
        raise ValueError("Labels/probabilities must align and n_bins must be positive")
    pairs = []
    ignored = 0
    for label, row in zip(labels, probabilities):
        row = _probability_row(row)
        if label == IGNORE_LABEL:
            ignored += 1
            continue
        if label not in range(4):
            raise ValueError("Unknown four-state label")
        pairs.append((int(label), row))
    if not pairs:
        return {"n_labeled_goals": 0, "n_ignored_goals": ignored, "macro_f1": None,
                "harm_recall": None, "ece": None, "brier": None, "nll": None,
                "per_class": {label: {"support": 0} for label in LABELS}}
    confusion = [[0] * 4 for _ in range(4)]
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    harm_bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    brier = nll = 0.0
    for target, row in pairs:
        pred = max(range(4), key=row.__getitem__)
        confusion[target][pred] += 1
        confidence = row[pred]
        bins[min(n_bins - 1, int(confidence * n_bins))].append((confidence, float(pred == target)))
        harm_prob = row[2]
        harm_bins[min(n_bins - 1, int(harm_prob * n_bins))].append((harm_prob, float(target == 2)))
        brier += sum((p - float(c == target)) ** 2 for c, p in enumerate(row))
        nll -= math.log(max(1e-12, row[target]))
    per_class = {}
    for c, name in enumerate(LABELS):
        tp = confusion[c][c]
        support = sum(confusion[c])
        predicted = sum(row[c] for row in confusion)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}

    def ece(groups: Any) -> float:
        return sum(len(group) / len(pairs) * abs(sum(x for x, _ in group) / len(group)
                   - sum(y for _, y in group) / len(group)) for group in groups if group)
    present = [v for v in per_class.values() if v["support"]]
    return {"n_labeled_goals": len(pairs), "n_ignored_goals": ignored,
            "accuracy": sum(confusion[c][c] for c in range(4)) / len(pairs),
            "macro_f1": sum(x["f1"] for x in per_class.values()) / 4,
            "macro_f1_present_classes": sum(x["f1"] for x in present) / len(present),
            "balanced_accuracy": sum(x["recall"] for x in present) / len(present),
            "harm_recall": per_class["10"]["recall"] if per_class["10"]["support"] else None,
            "harm_false_negative_rate": 1 - per_class["10"]["recall"] if per_class["10"]["support"] else None,
            "ece": ece(bins), "harm_ece": ece(harm_bins),
            "brier": brier / len(pairs), "nll": nll / len(pairs),
            "confusion_true_rows_predicted_columns": confusion, "per_class": per_class}


def state_key(example: Example) -> tuple[str, str, str]:
    return example.scenario_id, example.task_id, example.state_id


def candidate_cost_denominators(examples: Sequence[Example]) -> dict[tuple[str, str, str], float]:
    denominators: dict[tuple[str, str, str], float] = {}
    seen: set[tuple[str, str, str, str]] = set()
    context: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    for example in examples:
        key = state_key(example)
        candidate_key = (*key, example.candidate.id)
        if candidate_key in seen:
            raise ValueError(f"Duplicate candidate row: {candidate_key}")
        seen.add(candidate_key)
        signature = (example.instruction, tuple((g.id, g.text, g.evidence_span, g.kind) for g in example.goals),
                     tuple((x.get("role"), x.get("content")) for x in example.history), example.before_fingerprint)
        if key in context and context[key] != signature:
            raise ValueError(f"Candidate pool has inconsistent context: {key}")
        context[key] = signature
        denominators[key] = max(denominators.get(key, 1.0), example.candidate.estimated_api_calls)
    grouped: dict[tuple[str, str, str], list[Example]] = defaultdict(list)
    for example in examples:
        grouped[state_key(example)].append(example)
    for key, pool in grouped.items():
        declared = [ex.metadata.get("candidate_group_max_cost") for ex in pool]
        if any(value is not None for value in declared):
            if any(value is None for value in declared):
                raise ValueError("Candidate pool has inconsistent original cost metadata")
            values = [float(value) for value in declared]
            if any(not math.isfinite(value) or value < 1 for value in values):
                raise ValueError("Original candidate pool cost denominator must be finite and at least one")
            if any(not math.isclose(value, values[0]) for value in values):
                raise ValueError("Candidate pool has conflicting original cost denominators")
            if values[0] < denominators[key] - 1e-8:
                raise ValueError("Original candidate cost denominator is smaller than an observed candidate")
            # Failed/rejected replay branches may be absent. Their cost still
            # belongs to the online candidate pool used to form these inputs.
            denominators[key] = values[0]
        declared_ids = [ex.metadata.get("candidate_pool_ids") for ex in pool]
        if any(value is not None for value in declared_ids):
            if any(value is None for value in declared_ids):
                raise ValueError("Candidate pool has inconsistent original ID metadata")
            if any(not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item for item in value)
                   for value in declared_ids):
                raise ValueError("Original candidate pool IDs must be nonempty strings")
            ids = [tuple(value) for value in declared_ids]
            if len(set(ids[0])) != len(ids[0]) or any(value != ids[0] for value in ids):
                raise ValueError("Candidate pool has conflicting original candidate IDs")
            if not {ex.candidate.id for ex in pool}.issubset(set(ids[0])):
                raise ValueError("Observed candidate is absent from original pool IDs")
            if any(ex.metadata.get("candidate_pool_size", len(ids[0])) != len(ids[0]) for ex in pool):
                raise ValueError("Original pool size and IDs disagree")
        declared_sizes = [ex.metadata.get("candidate_pool_size") for ex in pool]
        if any(value is not None for value in declared_sizes):
            if any(not isinstance(value, int) or isinstance(value, bool) or value < len(pool)
                   for value in declared_sizes) or len(set(declared_sizes)) != 1:
                raise ValueError("Invalid or inconsistent original candidate pool size")
        complete_original = (declared_ids[0] is not None and set(declared_ids[0]) == {ex.candidate.id for ex in pool}) or declared_sizes[0] == len(pool)
        observed_max = max(1.0, max(ex.candidate.estimated_api_calls for ex in pool))
        if complete_original and declared[0] is not None and not math.isclose(float(declared[0]), observed_max):
            raise ValueError("Complete original pool has an inconsistent cost denominator")
    return denominators


def label_utility(labels: Sequence[int], normalized_cost: float, harm_weight: float = 2,
                  cost_weight: float = 0.05) -> float:
    if not labels or any(label not in range(4) for label in labels):
        raise ValueError("Scalar utility requires a complete, known goal label vector")
    return (sum(label == 1 for label in labels) - harm_weight * sum(label == 2 for label in labels)) / len(labels) - cost_weight * normalized_cost


def expected_utility(probabilities: Sequence[Sequence[float]], normalized_cost: float,
                     harm_weight: float = 2, cost_weight: float = 0.05) -> float:
    rows = [_probability_row(row) for row in probabilities]
    if not rows:
        raise ValueError("No goal predictions")
    return sum(row[1] - harm_weight * row[2] for row in rows) / len(rows) - cost_weight * normalized_cost


def _group_decisions(examples: Sequence[Example], predictions: Sequence[Any], *,
                     kind: str, harm_weight: float, cost_weight: float) -> tuple[list[dict[str, Any]], int]:
    if len(examples) != len(predictions):
        raise ValueError("One prediction per candidate record is required")
    if kind not in {"effect", "scalar"}:
        raise ValueError("kind must be effect or scalar")
    denominators = candidate_cost_denominators(examples)
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[state_key(example)].append(index)
    decisions = []
    incomplete = 0
    for key, indices in sorted(groups.items()):
        observed_ids = {examples[i].candidate.id for i in indices}
        original_ids = examples[indices[0]].metadata.get("candidate_pool_ids")
        original_size = examples[indices[0]].metadata.get("candidate_pool_size", len(indices))
        missing_original = (original_ids is not None and set(original_ids) != observed_ids) or original_size != len(indices)
        if len(indices) < 2 or missing_original or any(IGNORE_LABEL in examples[i].labels for i in indices):
            incomplete += 1
            continue
        # The pre-repair goal state must be identical in all candidate branches.
        before = tuple(label // 2 for label in examples[indices[0]].labels)
        if any(tuple(label // 2 for label in examples[i].labels) != before for i in indices):
            raise ValueError(f"Counterfactual candidate labels disagree on the pre-state: {key}")
        actual = []
        scores = []
        for i in indices:
            ex = examples[i]
            cost = ex.candidate.estimated_api_calls / denominators[key]
            actual.append(label_utility(ex.labels, cost, harm_weight, cost_weight))
            if kind == "effect":
                if len(predictions[i]) != len(ex.goals):
                    raise ValueError("Prediction goal count does not match the candidate")
                scores.append(expected_utility(predictions[i], cost, harm_weight, cost_weight))
            else:
                score = float(predictions[i])
                if not math.isfinite(score):
                    raise ValueError("Scalar score must be finite")
                scores.append(score)
        chosen = max(range(len(indices)), key=lambda j: (scores[j],
                     -examples[indices[j]].candidate.estimated_api_calls, -j))
        oracle = max(actual)
        selected = examples[indices[chosen]]
        decisions.append({"scenario_id": key[0], "task_id": key[1], "state_id": key[2],
                          "candidate_id": selected.candidate.id, "selected_utility": actual[chosen],
                          "oracle_utility": oracle, "regret": max(0.0, oracle - actual[chosen]),
                          "oracle_tie_success": math.isclose(actual[chosen], oracle, abs_tol=1e-8),
                          "restored_fraction": sum(x == 1 for x in selected.labels) / len(selected.labels),
                          "harmed_fraction": sum(x == 2 for x in selected.labels) / len(selected.labels),
                          "normalized_cost": selected.candidate.estimated_api_calls / denominators[key]})
    return decisions, incomplete


def evaluate_predictions(examples: Sequence[Example], predictions: Sequence[Any], *,
                         kind: str = "effect", harm_weight: float = 2,
                         cost_weight: float = 0.05) -> dict[str, Any]:
    """Evaluate labeled held-out mechanism data; no task-success claim is made."""
    result: dict[str, Any] = {"evaluation_level": "heldout_candidate_mechanism",
                              "not_native_tgc_or_sgc": True, "kind": kind}
    if kind == "effect":
        labels: list[int] = []
        probs: list[Sequence[float]] = []
        if len(examples) != len(predictions):
            raise ValueError("Prediction count differs from data rows")
        for ex, rows in zip(examples, predictions):
            if len(rows) != len(ex.labels):
                raise ValueError("Each goal requires a four-state prediction")
            labels.extend(ex.labels)
            probs.extend(rows)
        result["classification"] = classification_metrics(labels, probs)
    decisions, incomplete = _group_decisions(examples, predictions, kind=kind,
                                             harm_weight=harm_weight, cost_weight=cost_weight)
    result["decision"] = {"n_complete_candidate_pools": len(decisions),
                          "n_excluded_incomplete_or_single_candidate_pools": incomplete,
                          "mean_regret": sum(x["regret"] for x in decisions) / len(decisions) if decisions else None,
                          "oracle_choice_rate": sum(x["oracle_tie_success"] for x in decisions) / len(decisions) if decisions else None,
                          "mean_selected_utility": sum(x["selected_utility"] for x in decisions) / len(decisions) if decisions else None,
                          "mean_harmed_fraction": sum(x["harmed_fraction"] for x in decisions) / len(decisions) if decisions else None,
                          "pools": decisions}
    return result


def paired_scenario_bootstrap(differences: Mapping[str, Sequence[float]], *,
                              n_resamples: int = 2000, seed: int = 100,
                              confidence: float = 0.95) -> dict[str, Any]:
    """Cluster bootstrap paired differences, giving every held-out scenario equal weight."""
    if n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap parameters")
    means = []
    for scenario, values in sorted(differences.items()):
        values = list(values)
        if not values or any(not math.isfinite(float(v)) for v in values):
            raise ValueError(f"Invalid paired scenario differences: {scenario}")
        means.append(sum(values) / len(values))
    if not means:
        raise ValueError("No paired held-out scenarios")
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    return {"mean_difference": sum(means) / len(means), "n_scenarios": len(means),
            "confidence": confidence,
            "ci_low": samples[max(0, int(alpha * n_resamples))],
            "ci_high": samples[min(n_resamples - 1, int((1 - alpha) * n_resamples))],
            "n_resamples": n_resamples, "seed": seed, "unit": "paired_scenario_mean",
            "warning": "single-scenario interval is uninformative" if len(means) < 2 else None}


def paired_policy_comparison(examples: Sequence[Example], left: Sequence[Any], right: Sequence[Any], *,
                             left_kind: str = "effect", right_kind: str = "scalar",
                             harm_weight: float = 2, cost_weight: float = 0.05,
                             n_resamples: int = 2000, seed: int = 100) -> dict[str, Any]:
    a, _ = _group_decisions(examples, left, kind=left_kind, harm_weight=harm_weight, cost_weight=cost_weight)
    b, _ = _group_decisions(examples, right, kind=right_kind, harm_weight=harm_weight, cost_weight=cost_weight)
    index = {(x["scenario_id"], x["task_id"], x["state_id"]): x for x in b}
    differences: dict[str, list[float]] = defaultdict(list)
    for row in a:
        key = row["scenario_id"], row["task_id"], row["state_id"]
        differences[key[0]].append(row["selected_utility"] - index[key]["selected_utility"])
    return paired_scenario_bootstrap(differences, n_resamples=n_resamples, seed=seed)
