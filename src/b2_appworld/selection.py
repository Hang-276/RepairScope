"""Candidate ranking with a fixed, goal-normalized, probability-checked utility."""
from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

from .schemas import Candidate


def normalized_costs(candidates: Sequence[Candidate]) -> list[float]:
    if not candidates:
        raise ValueError("Cannot score an empty candidate pool")
    denominator = max(1.0, max(candidate.estimated_api_calls for candidate in candidates))
    return [candidate.estimated_api_calls / denominator for candidate in candidates]


def validate_probabilities(probabilities: Sequence, candidate_count: int,
                           goal_count: int | None = None) -> list[list[list[float]]]:
    if len(probabilities) != candidate_count or candidate_count < 1:
        raise ValueError("Effect predictions must have one row per candidate")
    if goal_count is None:
        goal_count = len(probabilities[0])
    if goal_count < 1:
        raise ValueError("Effect predictions require at least one goal")
    result = []
    for candidate in probabilities:
        if len(candidate) != goal_count:
            raise ValueError("Effect predictions must have one distribution per goal")
        row = []
        for distribution in candidate:
            if len(distribution) != 4:
                raise ValueError("Probability label order is [00,01,10,11]")
            if any(isinstance(p, bool) or not isinstance(p, Real) for p in distribution):
                raise ValueError("Probabilities must be numeric and cannot be booleans")
            converted = [float(p) for p in distribution]
            if any(not math.isfinite(p) or p < 0 or p > 1 for p in converted):
                raise ValueError("Probabilities must be finite and within [0,1]")
            if not math.isclose(sum(converted), 1.0, rel_tol=0, abs_tol=1e-5):
                raise ValueError("Probability distribution must sum to one; no silent renormalization")
            row.append(converted)
        result.append(row)
    return result


def effect_scores(candidates: Sequence[Candidate], probabilities: Sequence,
                  goal_count: int | None = None) -> list[float]:
    probs = validate_probabilities(probabilities, len(candidates), goal_count)
    costs = normalized_costs(candidates)
    return [(sum(d[1] for d in row) - 2 * sum(d[2] for d in row)) / len(row) - 0.05 * cost
            for row, cost in zip(probs, costs)]


def select_by_scalar(candidates: Sequence[Candidate], values: Sequence[float]) -> Candidate:
    if not candidates or len(candidates) != len(values):
        raise ValueError("One scalar score is required per nonempty candidate pool")
    values = [float(v) for v in values]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Candidate scores must be finite")
    # Equal utilities choose lower estimated cost, then original pool order.
    index = max(range(len(candidates)), key=lambda i: (values[i], -candidates[i].estimated_api_calls, -i))
    return candidates[index]


def select_by_effects(candidates: Sequence[Candidate], probabilities: Sequence,
                      goal_count: int | None = None) -> Candidate:
    return select_by_scalar(candidates, effect_scores(candidates, probabilities, goal_count))
