"""Executed natural trajectories and isolated two/four-cell counterfactuals."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable
import uuid

from .data import validate_dataset, write_jsonl
from .environment import create_environment, load_mapping, normalize_config, official_task_ids
from .schema import evaluate_goals, field_map, validate_action, validate_transition


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _task_ids(config: dict[str, Any], split: str) -> list[str]:
    settings = config.get('collection', {})
    ids = settings.get('task_ids') or config.get('data', {}).get('task_ids')
    if not ids:
        ids = ['fixture-0'] if config['environment']['backend'] == 'fixture' else official_task_ids(config['environment']['root'], split)
    if isinstance(ids, str):
        ids = [line.strip() for line in Path(ids).read_text().splitlines() if line.strip()]
    maximum = settings.get('max_tasks', config.get('data', {}).get('max_tasks'))
    return list(ids)[:maximum] if maximum is not None else list(ids)


def _goals(config: dict[str, Any], env: Any) -> list[dict[str, Any]]:
    return deepcopy(env.metadata.get('goals', []))


def _scope(state: dict[str, Any], action: dict[str, Any], config: dict[str, Any], predictor: Any = None, history: list | None = None) -> tuple[list[str], str]:
    if predictor is not None:
        result = predictor({'state': state, 'action': action, 'history': history or []}) if callable(predictor) else predictor.predict(state, action, history or [])
        return list(result['scope']), str(result.get('model_version', getattr(predictor, 'model_version', 'predictor')))
    maximum = int(config.get('model', {}).get('scope_budget', config.get('model', {}).get('max_scope', 128)))
    app = action['name'].split('.')[0]
    # This is a recorded collection prior, not a learned RepairScope result.
    keys = list(field_map(state))
    keys.sort(key=lambda key: (not key.startswith(app + '.'), key))
    return keys[:maximum], 'collection-public-prior-v1'


def _record(task_id: str, split: str, step: int, before: dict[str, Any], action: dict[str, Any], result: dict[str, Any], history: list[dict[str, Any]], goals: list[dict[str, Any]], scope: list[str], model_version: str, *, group_id: str | None = None, trajectory_id: str | None = None) -> dict[str, Any]:
    # Audited values provide labels only; state/next_state remain legal inputs.
    label_before = result.get('audit_before_state') or before
    label_after = result.get('audit_state') or result['state']
    record = {'task_id': task_id, 'source_task_id': task_id, 'source_split': split,
              'group_id': group_id or task_id, 'trajectory_id': trajectory_id or task_id + '-natural',
              'split': split, 'evidence_origin': result['evidence_origin'], 'step': step,
              'state': deepcopy(before), 'action': deepcopy(action), 'next_state': deepcopy(result['state']),
              'history': deepcopy(history[-8:]), 'goals': deepcopy(goals),
              'goal_before': (result['goal_before'] if 'goal_before' in result else
                              evaluate_goals(label_before, goals,
                                             previous=history[-1]['state'] if history else None)),
              'goal_after': (result['goal_after'] if 'goal_after' in result else
                             evaluate_goals(label_after, goals, previous=label_before)),
              'scope': list(scope), 'scope_model_version': model_version,
              'audit': {**deepcopy(result['audit']), 'model_version': model_version},
              'action_valid': bool(result['valid']), 'execution_cost': float(result['cost']),
              'execution_error': result.get('error'), 'observation': result['observation'], 'fingerprint': result.get('fingerprint'),
              'used_for_training': split == 'train'}
    if result.get('audit_state') is not None:
        record['audit_next_state'] = deepcopy(result['audit_state'])
        record['audit_state'] = deepcopy(result.get('audit_before_state'))
    validate_transition(record)
    return record


def collect(config: dict[str, Any], output_dir: str | Path, actor: Any = None, predictor: Any = None) -> dict[str, Any]:
    """Collect one frozen-policy run. No silent API retries or fake trajectories.

    `collection.action_plans` is an optional task->action-list JSON/YAML mapping;
    otherwise the same frozen actor generates one actual next action per step.
    All public API docs and supervisor identity are included in its context.
    """
    config = normalize_config(config)
    settings = config.get('collection', {})
    split = settings.get('split', 'train')
    if split == 'test':
        split = 'test_normal'
    if split not in ('train', 'dev'):
        raise ValueError('Recorded trajectory collection supports train/dev only; use frozen evaluation for official test runs')
    if config['environment']['backend'] == 'appworld':
        config['environment']['offline_audit'] = True
    if actor is None and not settings.get('action_plans'):
        from .actor import create_actor
        actor = create_actor(config)
    plans = load_mapping(settings.get('action_plans')) or {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records, task_summaries = [], []
    total_cost = 0.
    max_steps = int(settings.get('max_steps', config['environment'].get('max_steps', 50)))
    for task_id in _task_ids(config, split):
        env = create_environment(config, task_id, split, 'rs_collect_' + uuid.uuid4().hex[:12])
        history, task_cost, task_records = [], 0., []
        start = time.monotonic()
        try:
            state = env.reset()
            goals = _goals(config, env)
            from .actor import actor_instruction
            public_context = actor_instruction(env.metadata)
            actions = plans.get(task_id, plans.get('actions')) if isinstance(plans, dict) else plans
            for step in range(max_steps):
                if actions is not None:
                    if step >= len(actions):
                        break
                    action = deepcopy(actions[step])
                else:
                    candidates = actor.generate(state, goals, history, count=1, mode='execute', instruction=public_context)
                    if not candidates:
                        raise ValueError('Frozen Actor returned no candidate action')
                    action = candidates[0]
                action.setdefault('arguments', {})
                action.setdefault('cost', 1)
                validate_action(action)
                selected, version = _scope(state, action, config, predictor, history)
                result = env.execute(action)
                record = _record(task_id, split, step, state, action, result, history, goals, selected, version)
                records.append(record)
                task_records.append(record)
                history.append({'state': deepcopy(state), 'action': deepcopy(action)})
                state = result['state']
                task_cost += result['cost']
                total_cost += result['cost']
                # Preserve each completed execution before another external call.
                write_jsonl(output / 'transitions.jsonl', records)
                if result.get('done'):
                    break
            task_summaries.append({'task_id': task_id, 'steps': len(task_records), 'cost': task_cost,
                                   'duration_seconds': time.monotonic() - start, 'evidence_origin': env.evidence_origin})
        except BaseException as error:
            _json_write(output / 'collection_failure.json', {'task_id': task_id, 'error': f'{type(error).__name__}: {error}',
                        'completed_records': len(records), 'known_cost': total_cost, 'cost_unresolved': isinstance(error, TimeoutError)})
            raise
        finally:
            env.close()
    summary = {**validate_dataset(records, allow_synthetic=config['environment']['backend'] == 'fixture'),
               'total_cost': total_cost, 'tasks_completed': task_summaries, 'actor_frozen': True, 'used_for_training': split == 'train', 'collection_role': 'training' if split == 'train' else 'held_out_recorded_evaluation',
               'scientific_evidence': config['environment']['backend'] == 'appworld', 'path': str((output / 'transitions.jsonl').resolve())}
    _json_write(output / 'summary.json', summary)
    return summary


def collect_counterfactual(config: dict[str, Any], plans: Iterable[dict[str, Any]] | str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run each plan's same-prefix reference/action cells in independent worlds.

    Plans define legal conditions as executed API action lists plus a read-only
    condition query. Incomplete two-cell plans are permitted; four-cell boundary
    masks remain false until all four aligned valid outcomes exist.
    """
    config = normalize_config(config)
    config['environment']['offline_audit'] = True
    if config.get('collection', {}).get('split', 'train') != 'train':
        raise ValueError('Counterfactual acquisition is train-only')
    plans = load_mapping(plans)
    if isinstance(plans, dict):
        plans = plans.get('plans', [plans])
    plans = list(plans)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records, receipts = [], []
    total_cost = 0.
    seen_pairs = set()
    for plan in plans:
        if plan.get('split', 'train') != 'train' or plan.get('source_split', 'train') != 'train':
            raise ValueError('Official dev/test branch records cannot be used for training')
        task_id = plan['task_id']
        pair_id = str(plan.get('pair_id', plan.get('cf_pair_id', task_id + '-' + uuid.uuid4().hex[:12])))
        if pair_id in seen_pairs:
            raise ValueError('Duplicate counterfactual pair ID')
        seen_pairs.add(pair_id)
        prefix = deepcopy(plan.get('prefix', []))
        if config['environment']['backend'] == 'appworld' and any('output' not in step or not step.get('fingerprint') for step in prefix):
            raise ValueError('Real counterfactual prefix entries require exact action/output/fingerprint receipts from natural collection')
        env = create_environment(config, task_id, 'train', 'rs_cf_' + uuid.uuid4().hex[:12])
        env.reset()
        goals = deepcopy(plan.get('goals', env.metadata.get('goals', [])))
        conditions = plan.get('conditions') or {'c0': None}
        if isinstance(conditions, list):
            conditions = {f'c{i}': condition for i, condition in enumerate(conditions)}
        if set(conditions) - {'c0', 'c1'}:
            raise ValueError('Counterfactual conditions must use c0/c1 or a list of at most two legal conditions')
        prefix_identity = None
        try:
            for condition_name in ('c0', 'c1'):
                if condition_name not in conditions:
                    continue
                for role in ('reference', 'action'):
                    action = deepcopy(plan[role])
                    action.setdefault('arguments', {})
                    action.setdefault('cost', 1)
                    result = env.branch(prefix, action, conditions[condition_name])
                    total_cost += float(result['cost'])
                    identity = result.get('prefix_fingerprint') or json.dumps(result.get('prefix_receipts', result.get('prefix_state')), sort_keys=True)
                    if prefix_identity is not None and identity != prefix_identity:
                        raise ValueError('Counterfactual siblings do not have identical replayed prefixes')
                    prefix_identity = identity
                    before = result['before_state']
                    scope = list(plan.get('scope', _scope(before, action, config)[0]))
                    version = plan.get('model_version', 'collection-public-prior-v1')
                    record = _record(task_id, 'train', len(prefix), before, action, result, [], goals, scope, version,
                                     group_id=plan.get('group_id', task_id), trajectory_id=pair_id + '-' + condition_name + '-' + role)
                    record.update(cf_pair_id=pair_id, cf_condition=condition_name, cf_role=role,
                                  prefix_verified=result.get('prefix_verified', False),
                                  condition_verified=result.get('condition_verified', True),
                                  counterfactual_cost_breakdown=result.get('cost_breakdown', {}))
                    validate_transition(record)
                    records.append(record)
                    receipts.append({'pair_id': pair_id, 'condition': condition_name, 'role': role, 'cost': result['cost'],
                                     'valid': result['valid'], 'error': result.get('error'),
                                     'prefix_fingerprint': result.get('prefix_fingerprint'), 'process_isolation': result.get('process_isolation')})
                    write_jsonl(output / 'transitions.jsonl', records)
                    _json_write(output / 'receipts.json', receipts)
        except BaseException as error:
            _json_write(output / 'collection_failure.json', {'pair_id': pair_id, 'error': f'{type(error).__name__}: {error}',
                        'known_cost': total_cost, 'completed_cells': len(records), 'cost_unresolved': True,
                        'action': 'Acquisition stopped; failed/timeout branches are never silently retried or assigned zero cost.'})
            raise
        finally:
            env.close()
    _align_counterfactual_audits(records)
    write_jsonl(output / 'transitions.jsonl', records)
    summary = {**validate_dataset(records, allow_synthetic=config['environment']['backend'] == 'fixture'),
               'total_cost': total_cost, 'cells': len(records), 'receipts': str((output / 'receipts.json').resolve()),
               'scientific_evidence': config['environment']['backend'] == 'appworld', 'path': str((output / 'transitions.jsonl').resolve())}
    _json_write(output / 'summary.json', summary)
    return summary


def extract_prefix(records: Iterable[dict[str, Any]], stop_step: int) -> list[dict[str, Any]]:
    """Build verified replay entries from one natural trajectory prefix."""
    rows = sorted((r for r in records if r['step'] < stop_step), key=lambda r: r['step'])
    if [r['step'] for r in rows] != list(range(stop_step)):
        raise ValueError('Prefix must contain every step from reset up to stop_step')
    if len({(r['task_id'], r.get('trajectory_id')) for r in rows}) > 1:
        raise ValueError('Prefix records must belong to one trajectory')
    result = []
    for record in rows:
        if record['evidence_origin'] == 'appworld' and not record.get('fingerprint'):
            raise ValueError('Real prefix requires stored complete-audit fingerprint')
        result.append({'action': deepcopy(record['action']), 'output': record['observation'], 'fingerprint': record.get('fingerprint')})
    return result


def _align_counterfactual_audits(records: list[dict[str, Any]]) -> None:
    """Creation/deletion support from complete closed-domain audit snapshots only.

    This augments audit labels, never public states, and is forbidden for partial
    domains. A missing row in an explicitly complete SQLite snapshot is absent;
    an omitted field in a partial read remains unknown.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record['cf_pair_id'], []).append(record)
    for cells in groups.values():
        if not all(r['audit']['complete'] and r['audit'].get('domain_kind') in ('global_sqlite', 'fixture_closed_world') and r.get('audit_next_state') is not None for r in cells):
            continue
        templates = {f['key']: f for record in cells for f in record['audit_next_state']['fields']}
        for record in cells:
            after = field_map(record['audit_next_state'])
            for key, template in templates.items():
                if key not in after:
                    after[key] = {**deepcopy(template), 'known': True, 'exists': False, 'value': None}
            record['audit_next_state']['fields'] = list(after.values())
            record['audit']['field_domain'] = sorted(set(record['audit']['field_domain']) | set(templates))
            validate_transition(record)
