import json
from pathlib import Path

import pytest

from repairscope.calibration import calibrate_threshold, apply_calibration
from repairscope.config import load_config
from repairscope.provenance import file_digest


def rows(path, checkpoint, split='dev'):
    records = []
    for score, changed, complete in [(0.1, [], True), (0.7, ['b'], True), (0.9, ['b'], True), (0.8, [], False)]:
        records.append({'horizon': 1, 'offset': 1, 'split': split, 'checkpoint_id': file_digest(checkpoint),
            'prediction': {'escape': score, 'uncertainty': score, 'scope': ['a']},
            'label_receipt': {'changes': changed, 'complete': complete}})
    path.write_text(''.join(json.dumps(row) + '\n' for row in records))


def test_dev_threshold_masks_unknowns_respects_budget_and_freezes_checkpoint(tmp_path):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint-identity')
    predictions = tmp_path / 'predictions.jsonl'
    rows(predictions, checkpoint)
    receipt = tmp_path / 'calibration.json'
    value = calibrate_threshold(predictions, checkpoint, receipt, max_check_rate=1/3)
    assert value['dev_records'] == 3
    assert value['dev_check_rate'] <= 1/3
    assert value['risk_threshold'] == 0.9
    config = load_config(variant='full', overrides=[f'controller.calibration_file={receipt}'])
    assert apply_calibration(config, checkpoint)['controller']['risk_threshold'] == 0.9
    checkpoint.write_bytes(b'different-model')
    with pytest.raises(ValueError, match='checkpoint'):
        apply_calibration(config, checkpoint)


def test_test_predictions_cannot_fit_a_threshold(tmp_path):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    predictions = tmp_path / 'predictions.jsonl'
    rows(predictions, checkpoint, split='test_normal')
    with pytest.raises(ValueError, match='dev'):
        calibrate_threshold(predictions, checkpoint, tmp_path / 'calibration.json')
