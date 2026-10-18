"""Frozen recorded-data evaluation of H=1/2/4 hard state-patch recursion.

Only the starting legal observation and history enter a rollout. Subsequent
states and history entries are model predictions; each next action is revealed
only at its own step. Audits and true future observations are metric labels.
No environment, optimizer, checkpoint branch or model update is constructed.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from .actor import public_goals, public_history, public_state
from .data import make_windows, validate_dataset
from .metrics import calibration
from .schema import effective_protected, evaluate_goals, field_map
from .model import predict_record
from .training import load_predictor


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _checkpoint_digest(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None  # A test spy may replace the loader without a disk artifact.
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _known(field: dict | None) -> bool:
    return field is not None and bool(field.get('known', False))


def _semantic_equal(a: Any, b: Any) -> bool:
    if (type(a) is bool) != (type(b) is bool):
        return False
    if type(a) in (float, int) and type(b) in (float, int):
        return a == b
    return type(a) is type(b) and a == b


def _operation(before: dict | None, after: dict | None, tolerance: float = 0.) -> str | None:
    if not _known(before) or not _known(after):
        return None
    if before['exists'] and not after['exists']:
        return 'delete'
    if not before['exists'] and after['exists']:
        return 'create'
    if not before['exists']:
        return 'keep'
    left, right = before.get('value'), after.get('value')
    if type(left) in (float, int) and type(right) in (float, int) and not isinstance(left, bool) and not isinstance(right, bool):
        return 'modify' if abs(left - right) > tolerance else 'keep'
    return 'keep' if _semantic_equal(left, right) else 'modify'


def _goal_labels(record: dict, goals: list[dict], which: str) -> list[bool | None]:
    ids = [g['id'] for g in record.get('goals', [])]
    values = record.get('goal_before' if which == 'before' else 'goal_after', [])
    supplied = dict(zip(ids, values))
    state = (record.get('audit_state') or record['state']) if which == 'before' else (record.get('audit_next_state') or record['next_state'])
    fallback = evaluate_goals(state, goals)
    return [supplied.get(goal['id'], fallback[i]) for i, goal in enumerate(goals)]


class _Accumulator:
    """Compact counts; full audited states are never retained between windows."""
    def __init__(self):
        self.rows = self.changed = self.covered = self.zero_change = 0
        self.escape_n = self.escaped = self.scope_size = 0
        self.probabilities: list[float] = []
        self.escape_targets: list[int] = []
        self.confusion = [[0] * 4 for _ in range(4)]
        self.goal_targets = self.goal_predictions = self.protected_n = self.regressions = 0
        self.field_n = self.field_correct = self.numeric_n = self.numeric_predicted = 0
        self.numeric_error = 0.
        self.operation_n = self.operation_correct = 0
        self.public_n = self.public_correct = self.known_predictions = 0
        self.kinds: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def add(self, record: dict, prediction: dict, input_state: dict, goals: list[dict], before_goals: list[bool | None], public_keys: set[str]) -> None:
        self.rows += 1
        audit = record['audit']
        selected = set(prediction.get('scope', []))
        changed = set(audit['changes'])
        self.scope_size += len(selected)
        self.changed += len(changed)
        self.covered += len(changed & selected)
        if not changed and audit['complete']:
            self.zero_change += 1
        escaped = bool(changed - selected)
        if escaped or audit['complete']:
            self.escape_n += 1
            self.escaped += int(escaped)
            probability = prediction.get('escape')
            if probability is not None and math.isfinite(float(probability)):
                self.probabilities.append(float(probability))
                self.escape_targets.append(int(escaped))
        truth_state = record.get('audit_next_state') or record['next_state']
        truth, guessed = field_map(truth_state), field_map(prediction.get('next_state', {}))
        old_truth = field_map(record.get('audit_state') or record['state'])
        patch_ops = {patch['key']: patch.get('op', 'keep') for patch in prediction.get('patch', [])}
        for key in audit['field_domain']:
            target, guess = truth.get(key), guessed.get(key)
            if not _known(target):
                continue
            self.field_n += 1
            kind = target['kind']
            self.kinds[kind][1] += 1
            comparable = _known(guess) and guess['exists'] == target['exists']
            equal = comparable and (not target['exists'] or _semantic_equal(guess.get('value'), target.get('value')))
            self.field_correct += int(equal)
            self.kinds[kind][0] += int(equal)
            self.known_predictions += int(_known(guess))
            if key in public_keys:
                self.public_n += 1
                self.public_correct += int(equal)
            if kind == 'number' and target['exists'] and type(target.get('value')) in (float, int):
                self.numeric_n += 1
                if _known(guess) and guess.get('exists') and type(guess.get('value')) in (float, int):
                    self.numeric_predicted += 1
                    self.numeric_error += abs(float(guess['value']) - float(target['value']))
            target_op = _operation(old_truth.get(key), target, float(target.get('tolerance', 0)))
            if target_op is not None:
                self.operation_n += 1
                self.operation_correct += int(patch_ops.get(key, 'keep') == target_op)
        truth_goals = _goal_labels(record, goals, 'after')
        guessed_goals = evaluate_goals(prediction.get('next_state', {}), goals)
        heads = prediction.get('goal_probabilities', [])
        for i, (before, after, guess) in enumerate(zip(before_goals, truth_goals, guessed_goals)):
            if before is not None and after is not None:
                target = 2 * int(before) + int(after)
                self.goal_targets += 1
                predicted_class = 2 * int(before) + int(guess) if guess is not None else None
                if predicted_class is None and i < len(heads) and len(heads[i]) == 4:
                    predicted_class = max(range(4), key=lambda k: heads[i][k])
                if predicted_class is not None:
                    self.confusion[target][predicted_class] += 1
                    self.goal_predictions += 1
            if before is True and effective_protected(goals[i], int(record['step']), before) and after is not None:
                self.protected_n += 1
                self.regressions += int(after is False)

    def result(self) -> dict[str, Any]:
        scores = []
        for label in range(4):
            tp = self.confusion[label][label]
            fp = sum(row[label] for i, row in enumerate(self.confusion) if i != label)
            fn = sum(self.confusion[label][j] for j in range(4) if j != label)
            scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        def ratio(a, b):
            return a / b if b else None
        return {
            'transitions': self.rows,
            'scope_recall_at_budget': ratio(self.covered, self.changed),
            'scope_changed_fields': self.changed, 'scope_covered_fields': self.covered,
            'zero_change_steps': self.zero_change, 'mean_scope_size': ratio(self.scope_size, self.rows),
            'escape_rate': ratio(self.escaped, self.escape_n), 'escape_evaluable_steps': self.escape_n,
            'escape_calibration': calibration(self.probabilities, self.escape_targets),
            'goal_four_state_macro_f1': sum(scores) / 4 if self.goal_predictions else None,
            'goal_four_state_confusion': self.confusion,
            'goal_labeled_transitions': self.goal_targets, 'goal_predicted_transitions': self.goal_predictions,
            'goal_prediction_coverage': ratio(self.goal_predictions, self.goal_targets),
            'goal_regression_rate': ratio(self.regressions, self.protected_n), 'protected_goal_steps': self.protected_n,
            'typed_patch': {
                'field_exact_accuracy': ratio(self.field_correct, self.field_n), 'n_fields': self.field_n,
                'known_prediction_coverage': ratio(self.known_predictions, self.field_n),
                'public_candidate_field_accuracy': ratio(self.public_correct, self.public_n), 'n_public_candidate_fields': self.public_n,
                'numeric_mae': ratio(self.numeric_error, self.numeric_predicted), 'n_numeric_targets': self.numeric_n,
                'n_numeric_predictions': self.numeric_predicted, 'numeric_prediction_coverage': ratio(self.numeric_predicted, self.numeric_n),
                'operation_accuracy': ratio(self.operation_correct, self.operation_n), 'n_operation_fields': self.operation_n,
                'by_kind': {kind: {'accuracy': ratio(correct, count), 'n': count} for kind, (correct, count) in sorted(self.kinds.items())},
            },
        }


def evaluate_predictions(config: dict[str, Any], checkpoint: str | Path, records: Iterable[dict[str, Any]], output_dir: str | Path, horizons=(1, 2, 4)) -> dict[str, Any]:
    records = list(records)
    if not records:
        raise ValueError('Recorded prediction evaluation requires nonempty held-out transitions')
    horizons = tuple(dict.fromkeys(horizons))
    if not horizons or any(type(h) is not int or h not in (1, 2, 4) for h in horizons):
        raise ValueError('Prediction evaluation supports H=1,2,4')
    dataset = validate_dataset(records, allow_synthetic=bool(config.get('data', {}).get('allow_synthetic', False)))
    device = config.get('train', {}).get('device', config.get('device', 'cpu'))
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tensorizer, saved_config = load_predictor(checkpoint, device=device)
    if hasattr(model, 'eval'):
        model.eval()
    manifest = saved_config.get('checkpoint_manifest', {})
    trained_groups = set(map(str, manifest.get('groups', [])))
    if any(r['split'] != 'train' and str(r['group_id']) in trained_groups for r in records):
        raise ValueError('Held-out prediction records overlap checkpoint training groups')
    origins = manifest.get('evidence_origin', [])
    origins = [origins] if isinstance(origins, str) else origins
    synthetic_checkpoint = manifest.get('backend') in ('synthetic', 'fixture') or any('synthetic' in str(origin) or 'fixture' in str(origin) for origin in origins)
    if synthetic_checkpoint and any(r['evidence_origin'] == 'appworld' for r in records):
        raise ValueError('A synthetic/fixture checkpoint cannot produce real AppWorld prediction-evaluation results')
    checkpoint_sha256 = _checkpoint_digest(checkpoint)
    checkpoint_verified = bool(checkpoint_sha256) and bool(origins) and not synthetic_checkpoint and all(origin == 'appworld' for origin in origins)
    history_limit = int(getattr(tensorizer, 'history', saved_config.get('model', {}).get('history', 8)))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_digest = _digest(records)
    checkpoint_id = checkpoint_sha256 or _digest({'checkpoint': str(checkpoint), 'manifest': manifest, 'disk_artifact_present': False})
    indexes = {_digest(record): index for index, record in enumerate(records)}
    summary: dict[str, Any] = {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha256, 'checkpoint_manifest': manifest, 'dataset': dataset, 'dataset_sha256': data_digest,
        'recorded_only': True, 'optimizer_updates': 0, 'environment_executions': 0,
        'held_out_evaluation': all(r['split'] != 'train' for r in records),
        'evidence_origins': dataset['evidence_origins'], 'checkpoint_provenance_verified': checkpoint_verified,
        'scientific_evidence': checkpoint_verified and dataset['evidence_origins'] == ['appworld'] and all(r['split'] != 'train' for r in records),
        'input_policy': 'Starting public state/schema/goals/history, current action, then only predicted states and predicted history; no true intermediate state/history/audit enters Predictor.',
        'metric_domain': 'Declared audited field domain; public-candidate accuracy and missing prediction coverage reported separately.',
        'final_scope_semantics': 'Final action scope at B versus audited changes of that final action; final goal transition uses the real window-start goal vector.',
        'horizons': {}, 'predictions_path': str((output / 'predictions.jsonl').resolve()),
    }
    row_count = 0
    partial = output / 'predictions.jsonl.partial'
    with partial.open('w', encoding='utf-8') as stream:
        for horizon in horizons:
            windows = make_windows(records, horizon)
            per_step = [_Accumulator() for _ in range(horizon)]
            all_steps, final = _Accumulator(), _Accumulator()
            for window_number, window in enumerate(windows):
                start = window[0]
                state = public_state(deepcopy(start['state']))
                history = public_history(deepcopy(start['history']), limit=history_limit)
                goals = public_goals(deepcopy(start['goals']))
                schema = deepcopy(start.get('public_schema'))
                public_keys = set(field_map(state))
                if schema is not None:
                    public_keys.update(f['key'] for f in (schema.get('fields', []) if isinstance(schema, dict) else schema))
                initial_goal_vector = _goal_labels(start, goals, 'before')
                for offset, truth_record in enumerate(window):
                    action = deepcopy(truth_record['action'])
                    # Deliberate allowlist. Not even a true future state key or
                    # metadata envelope is handed to Tensorizer or Predictor.
                    prediction_input = {'state': deepcopy(state), 'action': action, 'history': deepcopy(history),
                                        'goals': deepcopy(goals), 'step': int(start['step']) + offset}
                    if schema is not None:
                        prediction_input['public_schema'] = deepcopy(schema)
                    prediction = predict_record(model, tensorizer, prediction_input, device=device,
                                                mc_samples=int(config.get('controller', {}).get('mc_samples', 0)))
                    before_goal_vector = _goal_labels(truth_record, goals, 'before')
                    per_step[offset].add(truth_record, prediction, state, goals, before_goal_vector, public_keys)
                    all_steps.add(truth_record, prediction, state, goals, before_goal_vector, public_keys)
                    if offset == horizon - 1:
                        final.add(truth_record, prediction, state, goals, initial_goal_vector, public_keys)
                    event = {
                        'horizon': horizon, 'window': window_number, 'offset': offset + 1,
                        'source_record_index': indexes[_digest(truth_record)],
                        'task_id': start['task_id'], 'group_id': start['group_id'], 'split': start['split'],
                        'evidence_origin': start['evidence_origin'], 'checkpoint_id': checkpoint_id,
                        'window_start_step': start['step'], 'input': prediction_input, 'prediction': prediction,
                        'label_receipt': {'complete': truth_record['audit']['complete'],
                                          'field_domain_size': len(truth_record['audit']['field_domain']),
                                          'changes': truth_record['audit']['changes'],
                                          'truth_next_sha256': _digest(truth_record.get('audit_next_state') or truth_record['next_state'])},
                    }
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False, sort_keys=True) + '\n')
                    row_count += 1
                    history = (history + [{'state': deepcopy(state), 'action': action}])[-history_limit:]
                    state = public_state(deepcopy(prediction['next_state']))
            summary['horizons'][str(horizon)] = {
                'windows': len(windows), 'status': 'evaluated' if windows else 'no_contiguous_full_windows',
                'steps': {str(index + 1): result.result() for index, result in enumerate(per_step)},
                'final': final.result(), 'all_steps': all_steps.result(),
            }
    partial.replace(output / 'predictions.jsonl')
    summary['prediction_rows'] = row_count
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output / 'summary.json')
    return summary
