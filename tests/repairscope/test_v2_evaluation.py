"""Goal labels, real candidate choices and budget accounting stay distinguishable."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from repairscope.fixtures import fixture_goals, fixture_records, fixture_state
from repairscope.schema import field_map
from repairscope.probing import candidate_statistics, selection_probabilities
import repairscope.prediction_eval as evaluation


PAID = 'shop.Order[1].paid'
SHIPPING = 'shop.Order[1].shipping'


def config(**evaluation_options):
    return {'model': {'goal_sufficient': True, 'scope_budget': 1, 'candidate_limit': 7},
            'data': {'allow_synthetic': True}, 'train': {'device': 'cpu'},
            'controller': {'mc_samples': 0}, 'evaluation': evaluation_options}


def held_out_pair():
    records = deepcopy(fixture_records(1)[7:9])
    for record in records:
        record.update(split='dev', source_split='dev', used_for_training=False)
    return records


def spy(monkeypatch, predict=None):
    model = SimpleNamespace(eval=lambda: None, scope_budget=1, goal_sufficient=True)
    tensorizer = SimpleNamespace(history=8, candidate_limit=7)
    saved = {**config(), 'checkpoint_manifest': {'backend': 'synthetic', 'evidence_origin': ['synthetic']}}
    monkeypatch.setattr(evaluation, 'load_predictor', lambda *args, **kwargs: (model, tensorizer, saved))
    calls = []
    def predictor(model, tensorizer, record, **kwargs):
        calls.append((model.scope_budget, tensorizer.candidate_limit, deepcopy(record)))
        if predict:
            return predict(model, record)
        return {'next_state': deepcopy(record['state']), 'scope': [SHIPPING, PAID][:model.scope_budget],
                'escape': .9, 'goal_escape': .1, 'patch': []}
    monkeypatch.setattr(evaluation, 'predict_record', predictor)
    return model, tensorizer, calls


def test_state_change_does_not_become_goal_relevant(monkeypatch, tmp_path):
    spy(monkeypatch)
    record = fixture_records(1, split='dev')[2]  # Only an irrelevant note changes.
    summary = evaluation.evaluate_predictions(config(), 'spy.pt', [record], tmp_path, horizons=(1,))
    result = summary['horizons']['1']['final']
    assert result['scope_changed_fields'] == 1 and result['escape_rate'] == 1
    assert result['goal_relevant_fields'] == 0
    assert result['goal_relevant_recall_at_budget'] is None
    assert result['goal_relevant_escape_rate'] == 0
    assert result['goal_relevant_escape_evaluable_steps'] == 1
    assert summary['goal_sufficient_decision']['decision_regret'] is None
    assert summary['goal_sufficient_decision']['decision_regret_coverage'] is None


def test_unknown_goal_label_never_supplies_negative(monkeypatch, tmp_path):
    spy(monkeypatch)
    record = held_out_pair()[1]
    record['goal_after'][1] = None
    summary = evaluation.evaluate_predictions(config(), 'spy.pt', [record], tmp_path, horizons=(1,))
    result = summary['horizons']['1']['final']
    assert result['goal_relevant_escape_evaluable_steps'] == 0
    assert result['goal_relevant_escape_rate'] is None
    assert result['goal_relevant_label_coverage'] == 0
    assert result['goal_relevant_escape_calibration']['n'] == 0
    event = json.loads((tmp_path / 'predictions.jsonl').read_text())
    assert event['label_receipt']['goal_escape_label'] is None
    decision = summary['goal_sufficient_decision']
    assert decision['decision_regret_denominator'] == 0
    assert decision['decision_regret_coverage_denominator'] == 1
    assert decision['decision_regret_coverage'] == 0


def test_partial_verified_positive_remains_escape_but_not_negative():
    record = held_out_pair()[0]
    extra = deepcopy(record['goals'][1])
    extra.update(id='unknown_protected', field='shop.Order[1].note', value='missing')
    record['goals'].append(extra)
    record['goal_before'].append(None)
    record['goal_after'].append(None)
    assert evaluation.goal_escape_label(record, [SHIPPING], config()) == 1
    assert evaluation.goal_escape_label(record, [SHIPPING, PAID], config()) is None


def test_prediction_regret_is_separate_from_oracle_patch_diagnostic(monkeypatch, tmp_path):
    def backwards(model, record):
        state = deepcopy(record['state'])
        # The learned dynamics reverse the true ranking, despite observing the
        # correct goal field; this must be counted as real prediction regret.
        field_map(state)[PAID]['value'] = record['action']['name'] == 'shop.set_shipping'
        field_map(state)[SHIPPING]['value'] = 'express'
        return {'next_state': state, 'scope': [PAID, SHIPPING], 'escape': 0., 'goal_escape': 0.}
    spy(monkeypatch, backwards)
    records = held_out_pair()
    result = evaluation.evaluate_predictions(config(), 'spy.pt', records, tmp_path, horizons=(1,))['goal_sufficient_decision']
    assert result['decision_regret_denominator'] == 1
    assert result['decision_regret_coverage'] == 1
    assert result['decision_regret'] > 0
    assert result['selected_goal_preservation_rate'] == 0
    diagnostic = result['scope_isolation_oracle_patch_diagnostic']
    assert diagnostic['mean_regret'] == 0 and diagnostic['is_prediction_regret'] is False


def test_incomplete_or_unmatched_candidate_groups_report_coverage(monkeypatch, tmp_path):
    spy(monkeypatch)
    records = held_out_pair()
    records[1]['goal_after'][0] = None
    result = evaluation.evaluate_predictions(config(), 'spy.pt', records, tmp_path, horizons=(1,))['goal_sufficient_decision']
    assert result['decision_regret'] is None
    assert result['decision_regret_coverage_numerator'] == 0
    assert result['decision_regret_coverage_denominator'] == 1
    assert result['decision_excluded_groups']['unknown_goal_evidence'] == 1
    records = held_out_pair()
    records[1]['history'] = [{'state': fixture_state(), 'action': {'name': 'shop.noop', 'arguments': {}, 'cost': 1}}]
    result = evaluation.evaluate_predictions(config(), 'spy.pt', records, tmp_path / 'unmatched', horizons=(1,))['goal_sufficient_decision']
    assert result['decision_regret'] is None
    assert result['decision_excluded_groups']['unmatched_prefix'] == 1


def test_explicit_budget_curve_keeps_checkpoint_candidates_and_dev_fixed(monkeypatch, tmp_path):
    model, tensorizer, calls = spy(monkeypatch)
    summary = evaluation.evaluate_predictions(config(scope_budgets=[1, 2]), 'spy.pt', held_out_pair(), tmp_path, horizons=(1,))
    curve = summary['goal_preservation_cost_curve']
    assert [point['scope_budget'] for point in curve['points']] == [1, 2]
    assert [point['metrics']['goal_relevant_recall_at_budget'] for point in curve['points']] == [0, 1]
    assert curve['candidate_limit_fixed'] and curve['candidate_limit'] == 7
    assert curve['environment_executions'] == curve['optimizer_updates'] == 0
    assert model.scope_budget == 1 and tensorizer.candidate_limit == 7
    assert {limit for _, limit, _ in calls} == {7}
    assert {tuple(sorted(record)) for _, _, record in calls} == {('action', 'goals', 'history', 'state', 'step')}
    assert curve['claim_boundary'].startswith('Recorded candidate decisions')


def test_budget_curve_is_opt_in(monkeypatch, tmp_path):
    _, _, calls = spy(monkeypatch)
    result = evaluation.evaluate_predictions(config(), 'spy.pt', held_out_pair(), tmp_path, horizons=(1,))
    assert result['goal_preservation_cost_curve']['status'] == 'disabled'
    assert len(calls) == 2
    with pytest.raises(ValueError, match='candidate_limit'):
        evaluation.evaluate_predictions(config(scope_budgets=[8]), 'spy.pt', held_out_pair(), tmp_path / 'invalid', horizons=(1,))


def test_v2_probe_uses_goal_escape_and_same_model_candidate_decisions():
    state = fixture_state()
    goals = fixture_goals()
    plan = {'state': state, 'goals': goals, 'action': {'name': 'a', 'arguments': {}},
            'reference': {'name': 'b', 'arguments': {}}}
    def predict(record):
        draws = []
        for draw in range(2):
            next_state = deepcopy(state)
            field_map(next_state)[PAID]['value'] = (record['action']['name'] == 'a') == (draw == 0)
            draws.append({'next_state': next_state, 'scope': [PAID], 'escape': 1., 'goal_escape': .4})
        return {'next_state': state, 'scope': [PAID], 'escape': 1., 'goal_escape': .4,
                'mc_predictions': draws, 'uncertainty': 0}
    result = candidate_statistics(plan, predict, config=config())
    assert result['goal_decision_risk'] == .4
    assert result['decision_disagreement'] == 1
    assert result['ranking_uncertainty'] > 0
    assert result['D'] == 1
    assert result['C'] == 2
    assert selection_probabilities([result, dict(result, D=0)], epsilon=.2)[1] == pytest.approx(.1)


def test_v1_baseline_receives_same_goal_metrics_without_goal_escape_head(monkeypatch, tmp_path):
    def legacy(model, record):
        return {'next_state': deepcopy(record['state']), 'scope': [SHIPPING], 'escape': .9}
    model, _, _ = spy(monkeypatch, legacy)
    model.goal_sufficient = False
    legacy_config = config()
    legacy_config['model']['goal_sufficient'] = False
    summary = evaluation.evaluate_predictions(legacy_config, 'spy.pt', held_out_pair(), tmp_path, horizons=(1,))
    result = summary['horizons']['1']['final']
    assert result['goal_relevant_recall_at_budget'] == 0
    assert result['goal_relevant_escape_evaluable_steps'] == 2
    assert result['goal_relevant_escape_calibration']['n'] == 0
    assert summary['goal_sufficient_decision']['decision_regret_denominator'] == 1


def test_probe_step_matches_condition_prefix_and_continuation_labels():
    from repairscope.control import BudgetLedger
    from repairscope.fixtures import FixtureEnv
    from repairscope.probing import _branch_records
    goals = fixture_goals()
    goals[1]['protect_from_step'] = 6
    noop = {'name': 'shop.noop', 'arguments': {}, 'cost': 1}
    plan = {'task_id': 'fixture', 'state': fixture_state(), 'goals': goals,
            'prefix': [{'action': deepcopy(noop)} for _ in range(2)],
            'history': [{'state': fixture_state(), 'action': deepcopy(noop)}],
            'action': {'name': 'shop.set_shipping', 'arguments': {'shipping': 'express'}, 'cost': 1},
            'reference': {'name': 'shop.set_shipping_preserve_payment', 'arguments': {'shipping': 'express'}, 'cost': 1},
            'continuation': [{'name': 'shop.pay_order', 'arguments': {}, 'cost': 1}]}
    calls = []
    def predict(record):
        calls.append(deepcopy(record))
        state = deepcopy(record['state'])
        field_map(state)[PAID]['value'] = False
        return {'next_state': state, 'scope': [PAID], 'goal_escape': 0., 'escape': 0.}
    assert candidate_statistics(plan, predict, config=config())['R'] == 0
    assert [record['step'] for record in calls] == [2, 2]  # Prefix beats truncated history.
    calls.clear()
    plan['step'] = 5
    plan['conditions'] = [{'state': fixture_state(), 'step': 6}]
    assert candidate_statistics(plan, predict, config=config())['R'] > 0
    assert [record['step'] for record in calls] == [6, 6]  # Condition overrides plan.
    calls.clear()
    env = FixtureEnv({'environment': {'backend': 'fixture', 'offline_audit': True}}, 'fixture', 'train', 'step-test')
    try:
        records, _ = _branch_records(env, plan, predict, BudgetLedger(100), 'step-test')
    finally:
        env.close()
    assert [record['step'] for record in calls] == [6, 7, 6, 7]
    assert [record['step'] for record in records] == [6, 7, 6, 7]
