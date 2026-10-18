from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from repairscope.fixtures import fixture_records
from repairscope.schema import field_map
import repairscope.prediction_eval as prediction_eval


def _setup_spy(monkeypatch, real=False):
    calls = []
    model = SimpleNamespace(eval=lambda: None)
    saved = {'model': {'history': 8}, 'checkpoint_manifest': {
        'backend': 'recorded' if real else 'synthetic', 'evidence_origin': ['appworld' if real else 'synthetic']}}
    monkeypatch.setattr(prediction_eval, 'load_predictor', lambda checkpoint, device: (model, SimpleNamespace(history=8), deepcopy(saved)))
    def predict(model, tensorizer, record, device='cpu', mc_samples=0):
        calls.append(deepcopy(record))
        state = deepcopy(record['state'])
        fields = field_map(state)
        key = 'shop.Order[1].amount'
        old = fields[key]['value']
        fields[key]['value'] += 100
        return {'next_state': state, 'scope': [key], 'escape': .25,
                'patch': [{'key': key, 'op': 'modify', 'value': old + 100}], 'goal_probabilities': []}
    monkeypatch.setattr(prediction_eval, 'predict_record', predict)
    return calls


def _config():
    return {'data': {'allow_synthetic': True}, 'train': {'device': 'cpu'}}


def test_hard_rollout_never_reads_true_intermediate_state_history_or_audit(monkeypatch, tmp_path):
    calls = _setup_spy(monkeypatch)
    records = fixture_records(1, split='dev')
    future = {'key': 'future.only_key', 'kind': 'text', 'value': 'REAL_FUTURE_SECRET',
              'known': True, 'exists': True, 'choices': []}
    history_only = {'key': 'future.history_key', 'kind': 'text', 'value': 'REAL_FUTURE_HISTORY',
                    'known': True, 'exists': True, 'choices': []}
    # The new field is actually observed only after the first executed action.
    for i, record in enumerate(records):
        if i:
            record['state']['fields'].append(deepcopy(future))
        record['next_state']['fields'].append(deepcopy(future))
        if i:
            record['history'] = [{'state': {'fields': [deepcopy(history_only)], 'relations': []},
                                  'action': {'name': 'shop.noop', 'arguments': {}, 'cost': 1}}]
        # A hidden audit target must not even contribute a candidate key.
        record['audit_next_state']['fields'].append({'key': 'audit.hidden', 'kind': 'text', 'value': 'AUDIT_SECRET',
                                                    'known': True, 'exists': True, 'choices': []})
        record['audit']['field_domain'].append('audit.hidden')
    before = deepcopy(records)
    summary = prediction_eval.evaluate_predictions(_config(), 'spy.pt', records, tmp_path, horizons=(4,))
    events = [json.loads(line) for line in (tmp_path / 'predictions.jsonl').read_text().splitlines()]
    first = [event for event in events if event['window'] == 0]
    assert len(first) == 4
    assert [field_map(event['input']['state'])['shop.Order[1].amount']['value'] for event in first] == [20., 120., 220., 320.]
    for event in first:
        encoded = json.dumps(event['input'])
        assert 'REAL_FUTURE_SECRET' not in encoded
        assert 'REAL_FUTURE_HISTORY' not in encoded
        assert 'future.only_key' not in encoded
        assert 'audit.hidden' not in encoded
        assert 'AUDIT_SECRET' not in encoded
        assert 'next_state' not in event['input'] and 'audit' not in event['input']
    assert first[1]['input']['history'][-1]['state'] == first[0]['input']['state']
    assert first[2]['input']['history'][-1]['state'] == first[1]['input']['state']
    assert first[2]['input']['state'] == first[1]['prediction']['next_state']
    assert records == before
    assert summary['horizons']['4']['windows'] == 2
    assert summary['horizons']['4']['final']['typed_patch']['numeric_mae'] > 0
    assert summary['environment_executions'] == summary['optimizer_updates'] == 0
    assert len(calls) == 8


def test_horizons_metrics_and_origin_are_explicit(monkeypatch, tmp_path):
    _setup_spy(monkeypatch)
    records = fixture_records(1, split='dev')
    summary = prediction_eval.evaluate_predictions(_config(), 'spy.pt', records, tmp_path)
    assert {h: summary['horizons'][h]['windows'] for h in ('1', '2', '4')} == {'1': 5, '2': 4, '4': 2}
    assert summary['prediction_rows'] == 21
    assert summary['held_out_evaluation']
    assert not summary['scientific_evidence']
    assert summary['evidence_origins'] == ['synthetic']
    assert summary['horizons']['4']['steps']['4']['transitions'] == 2
    metrics = summary['horizons']['1']['all_steps']
    assert metrics['typed_patch']['n_fields'] > 0
    assert metrics['scope_recall_at_budget'] == 0
    assert metrics['escape_calibration']['n'] == 5
    assert 'goal_four_state_confusion' in metrics
    assert json.loads((tmp_path / 'summary.json').read_text())['prediction_rows'] == 21


def test_synthetic_checkpoint_cannot_claim_appworld_evaluation(monkeypatch, tmp_path):
    _setup_spy(monkeypatch)
    records = fixture_records(1, split='dev')
    for record in records:
        record['evidence_origin'] = 'appworld'
    with pytest.raises(ValueError, match='synthetic/fixture checkpoint'):
        prediction_eval.evaluate_predictions(_config(), 'spy.pt', records, tmp_path)
    assert not (tmp_path / 'summary.json').exists()


def test_partial_audit_cannot_become_escape_negative(monkeypatch, tmp_path):
    _setup_spy(monkeypatch)
    records = fixture_records(1, split='dev')[:1]
    records[0]['audit'].update(complete=False, changes=[])
    summary = prediction_eval.evaluate_predictions(_config(), 'spy.pt', records, tmp_path, horizons=(1, 4))
    metrics = summary['horizons']['1']['final']
    assert metrics['escape_evaluable_steps'] == 0
    assert metrics['escape_rate'] is None
    assert metrics['escape_calibration']['n'] == 0
    assert summary['horizons']['4']['windows'] == 0
    assert summary['horizons']['4']['status'] == 'no_contiguous_full_windows'
    assert summary['horizons']['4']['final']['typed_patch']['field_exact_accuracy'] is None


def test_unknown_numeric_predictions_report_coverage(monkeypatch, tmp_path):
    _setup_spy(monkeypatch)
    def unknown(model, tensorizer, record, **kwargs):
        state = deepcopy(record['state'])
        field_map(state)['shop.Order[1].amount'].update(known=False, value=None)
        return {'next_state': state, 'scope': [], 'patch': [], 'escape': .5}
    monkeypatch.setattr(prediction_eval, 'predict_record', unknown)
    summary = prediction_eval.evaluate_predictions(_config(), 'spy.pt', fixture_records(1, split='dev')[:1], tmp_path, horizons=(1,))
    typed = summary['horizons']['1']['final']['typed_patch']
    assert typed['n_numeric_targets'] == 1
    assert typed['n_numeric_predictions'] == 0
    assert typed['numeric_mae'] is None and typed['numeric_prediction_coverage'] == 0
