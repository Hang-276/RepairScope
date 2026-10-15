"""Finite synthetic metric examples; no native-task success interpretation."""
import pytest

from b2_appworld.metrics import (classification_metrics, evaluate_predictions, label_utility,
                                expected_utility, paired_policy_comparison, paired_scenario_bootstrap)
from b2_appworld.schemas import Candidate, Example, Goal


def rows():
    result = []
    for s in ("s0", "s1"):
        for candidate, labels, cost in (("restore", (1, 3), 2), ("harm", (0, 2), 1)):
            result.append(Example(s + "_1", s, "state", "dev", "Set A and keep B", [],
                                  (Goal("a", "Set A", "Set A"), Goal("b", "keep B", "keep B")),
                                  Candidate(candidate, "print('" + candidate + "')", ("a",), cost),
                                  labels, source="synthetic_fixture"))
    return result


def one_hot(labels):
    return [[float(c == label) for c in range(4)] for label in labels]


def test_imbalance_and_unknown_labels_are_reported():
    metric = classification_metrics([0, 0, 1, 2, 3, -100],
                                    one_hot([0, 0, 1, 0, 3, 2]))
    assert metric["n_labeled_goals"] == 5 and metric["n_ignored_goals"] == 1
    assert metric["harm_recall"] == 0
    assert metric["harm_false_negative_rate"] == 1
    assert metric["per_class"]["10"]["support"] == 1
    assert metric["ece"] == pytest.approx(0.2)
    assert metric["brier"] == pytest.approx(0.4)
    assert metric["macro_f1"] < metric["accuracy"]


def test_utility_matches_effect_expectation_and_normalizes_all_goals():
    labels = [1, 2, 3, 0]
    assert label_utility(labels, .25) == pytest.approx((1 - 2) / 4 - .05 * .25)
    assert expected_utility(one_hot(labels), .25) == pytest.approx(label_utility(labels, .25))
    with pytest.raises(ValueError, match="complete"):
        label_utility([1, -100], 1)


def test_regret_and_paired_bootstrap_use_same_complete_candidate_pool():
    data = rows()
    perfect = [one_hot(ex.labels) for ex in data]
    scores_always_harm = [0, 1, 0, 1]
    report = evaluate_predictions(data, perfect)
    assert report["not_native_tgc_or_sgc"] is True
    assert report["decision"]["mean_regret"] == 0
    assert report["decision"]["oracle_choice_rate"] == 1
    worse = evaluate_predictions(data, scores_always_harm, kind="scalar")
    assert worse["decision"]["mean_regret"] > 0
    paired = paired_policy_comparison(data, perfect, scores_always_harm, n_resamples=100)
    assert paired["mean_difference"] > 0 and paired["ci_low"] > 0
    assert paired["n_scenarios"] == 2


def test_inconsistent_branch_prestate_is_an_error():
    data = rows()[:2]
    changed = data[1].to_dict()
    changed["labels"] = [2, 2]
    with pytest.raises(ValueError, match="pre-state"):
        evaluate_predictions([data[0], Example.from_dict(changed)], [one_hot([1, 3]), one_hot([2, 2])])


def test_bootstrap_is_deterministic_and_scenario_balanced():
    values = {"a": [1] * 10, "b": [-1]}
    result = paired_scenario_bootstrap(values, n_resamples=200)
    assert result["mean_difference"] == 0
    assert result == paired_scenario_bootstrap(values, n_resamples=200)


def test_invalid_probability_vectors_fail_loudly():
    with pytest.raises(ValueError, match="sum to one"):
        classification_metrics([0], [[.5, .5, .5, .5]])


def test_equal_scalar_scores_match_online_cost_tiebreak():
    data = rows()[:2]  # first restore costs2, second harm costs1
    report = evaluate_predictions(data, [0, 0], kind="scalar")
    assert report["decision"]["pools"][0]["candidate_id"] == "harm"


def test_rejected_original_candidates_exclude_pool_from_regret():
    data = rows()[:2]
    fixed = []
    for ex in data:
        raw = ex.to_dict()
        raw["metadata"] = {"candidate_group_max_cost": 10,
                           "candidate_pool_ids": ["restore", "harm", "rejected"], "candidate_pool_size": 3}
        fixed.append(Example.from_dict(raw))
    report = evaluate_predictions(fixed, [one_hot(ex.labels) for ex in fixed])
    assert report["decision"]["n_complete_candidate_pools"] == 0
    assert report["decision"]["mean_regret"] is None
