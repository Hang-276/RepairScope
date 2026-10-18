"""Candidate aggregation and stable predicted-state ranking for RepairScope.

Extracted from WMAgent.next_action. Stable first-occurrence ordering, frequency
Top-k, and probabilities renormalized over retained actions match upstream WMA.
The original browser agent is no longer part of this repository.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable


def aggregate_candidates(responses: Iterable, parse: Callable = lambda value: value,
                         count: int = 4, key: Callable = lambda value: value):
    if count < 1:
        raise ValueError("Candidate count must be positive")
    candidates = {}
    for response in responses:
        parsed = parse(response)
        identity = key(parsed)
        if identity not in candidates:
            candidates[identity] = {"action": parsed, "raw": response, "count": 0}
        candidates[identity]["count"] += 1
    selected = sorted(candidates.values(), key=lambda item: item["count"], reverse=True)[:count]
    denominator = sum(item["count"] for item in selected)
    return [{**item, "prob": item["count"] / denominator} for item in selected]


def stable_rank(candidates, predictions, utilities):
    """Align candidates with predictions and scores, preserving tie order."""
    if not (len(candidates) == len(predictions) == len(utilities)):
        raise ValueError("Candidate/prediction/utility lengths differ")
    ranked = []
    for action, prediction, utility in zip(candidates, predictions, utilities):
        score = dict(utility) if isinstance(utility, dict) else {"score": float(utility)}
        if not math.isfinite(float(score["score"])):
            raise ValueError("Candidate utility is not finite")
        ranked.append({"action": action, "prediction": prediction, **score})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def predict_and_rank(candidates, predict: Callable, score: Callable):
    predictions = [predict(action) for action in candidates]
    utilities = [score(action, prediction) for action, prediction in zip(candidates, predictions)]
    return stable_rank(candidates, predictions, utilities)
