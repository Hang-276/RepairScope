"""Server-only AppWorld integration, isolated from model-facing observations.

API verified against AppWorld revision 42b5bcf3cd334fee33f0c37c02070a9f5807add5:
AppWorld(task_id, experiment_name, load_ground_truth, random_seed), execute(code),
models.get_tracked_db_changes(), models[app].SQLModel.db.connection, save(), and
appworld.evaluator.evaluate_task(). Real execution is intentionally not imported
or invoked by the Mac smoke test.
"""
from __future__ import annotations

from copy import deepcopy
from fnmatch import fnmatchcase
import hashlib
import json
import multiprocessing as mp
import math
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any

from .environment import action_code, public_state_update, validate_condition
from .schema import field_map, semantic_diff, validate_state

_ERROR = re.compile(r'Traceback \(most recent call last\)|(?:^|\n)(?:ExecutionError|SyntaxError|Exception|ValueError|TypeError|HTTPException|NameError):|Execution failed', re.I)


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {'__bytes__': value.hex()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Datetimes/UUIDs occur in public API metadata; string representation keeps
    # their public value without exposing arbitrary object attributes.
    return str(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _fingerprint(world: Any) -> str:
    return _digest({'task_id': world.task_id, 'changes': world.models.get_tracked_db_changes()})


def _set_root(root: str) -> None:
    root = str(Path(root).expanduser().resolve())
    os.environ['APPWORLD_ROOT'] = root
    from appworld.common.path_store import path_store
    path_store.update_root(root)


def snapshot_world(world: Any, ignore: list[str] | None = None) -> dict[str, Any]:
    """Full SQLite audit, never a model observation or legal retrieval.

    Keys use app.table[primary-key].column. Null remains known null. Creation and
    deletion are aligned later with explicit tombstones. Only fixed declared
    filtering is accepted; timestamps are retained unless the run manifest says
    otherwise. Internal record_hash instrumentation is excluded by default.
    """
    ignore = ['*.record_hash', *(ignore or [])]
    fields = []
    for app, models in sorted(world.models.items()):
        connection = models.SQLModel.db.connection
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        for (table,) in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = connection.execute('PRAGMA table_info(' + quoted + ')').fetchall()
            column_names = [column[1] for column in columns]
            primary = [column[1] for column in sorted(columns, key=lambda row: row[5]) if column[5]]
            cursor = connection.execute('SELECT rowid, * FROM ' + quoted) if not primary else connection.execute('SELECT * FROM ' + quoted)
            for raw_row in cursor.fetchall():
                row = dict(zip(column_names, raw_row if primary else raw_row[1:]))
                identity_values = [row[name] for name in primary] if primary else [raw_row[0]]
                identity = ','.join(str(v) for v in identity_values)
                entity = f'{app}.{table}[{identity}]'
                for column in columns:
                    name, sqltype = column[1], (column[2] or '').upper()
                    key = entity + '.' + name
                    if any(fnmatchcase(key, pattern) for pattern in ignore):
                        continue
                    value = row[name]
                    if name == 'id' or name.endswith('_id'):
                        kind = 'reference'
                        value = str(value) if value is not None else None
                    elif 'BOOL' in sqltype:
                        kind, value = 'bool', None if value is None else bool(value)
                    elif re.match(r'^(?:INT(?:EGER)?|BIGINT|SMALLINT|TINYINT|REAL|FLOAT|DOUBLE|NUMERIC|DECIMAL)\b', sqltype) or isinstance(value, (int, float)):
                        kind = 'number'
                    else:
                        kind = 'text'
                        value = value.hex() if isinstance(value, bytes) else str(value) if value is not None else None
                    field = {'key': key, 'entity': entity, 'kind': kind, 'known': True, 'exists': True, 'value': value, 'choices': []}
                    if value is None:
                        field['nullable'] = True
                    fields.append(field)
    state = {'fields': fields, 'relations': []}
    validate_state(state)
    return state


def align_audit_states(before: dict[str, Any], after: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    before, after = deepcopy(before), deepcopy(after)
    left, right = field_map(before), field_map(after)
    for key in sorted(set(left) | set(right)):
        if key not in left:
            left[key] = {**deepcopy(right[key]), 'value': None, 'exists': False, 'known': True}
        if key not in right:
            right[key] = {**deepcopy(left[key]), 'value': None, 'exists': False, 'known': True}
    before['fields'], after['fields'] = list(left.values()), list(right.values())
    return before, after


def _evaluate_entry(connection: Any, request: dict[str, Any]) -> None:
    try:
        _set_root(request['root'])
        from appworld.evaluator import evaluate_task
        tracker = evaluate_task(task_id=request['task_id'], experiment_name=request['experiment'], suppress_errors=False, save_report=True)
        data = _jsonable(tracker.to_dict())
        connection.send({'ok': True, 'result': {'task_id': request['task_id'], 'split': request['split'], 'evidence_origin': 'appworld',
                                               'tracker': data, 'TGC': float(data['success']), 'SGC': None,
                                               'requires_scenario_aggregation': True, 'evaluator_isolation': 'fresh_spawn'}})
    except BaseException as error:
        connection.send({'ok': False, 'error': f'{type(error).__name__}: {error}'})
    finally:
        connection.close()


def evaluate_fresh(world: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Disk evaluation in a fresh spawn avoids CachedDBHandler aliasing."""
    world.save()
    context = mp.get_context('spawn')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_evaluate_entry, args=(child, {'root': request['config']['environment']['root'], 'task_id': request['task_id'], 'experiment': request['experiment'], 'split': request['split']}))
    process.start()
    child.close()
    try:
        timeout = request['config']['environment'].get('worker_timeout', 300)
        if not parent.poll(timeout):
            raise TimeoutError('Fresh evaluator timed out')
        response = parent.recv()
        if not response['ok']:
            raise RuntimeError(response['error'])
        return response['result']
    finally:
        parent.close()
        process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(3)
        if process.is_alive():
            process.kill()
            process.join()


def _public_metadata(world: Any, request: dict[str, Any]) -> dict[str, Any]:
    task = world.task
    env = request['config']['environment']
    mapping = env.get('goal_map') or {}
    goals = mapping.get(request['task_id'], mapping.get('goals', [])) if isinstance(mapping, dict) else mapping
    if isinstance(goals, dict):
        goals = goals.get('goals', [])
    return {'instruction': task.instruction, 'supervisor': _jsonable(task.supervisor),
            'allowed_apps': list(task.allowed_apps), 'api_docs': _jsonable(dict(task.api_docs)),
            'goals': deepcopy(goals), 'read_only_actions': deepcopy(env.get('read_only_actions', [])),
            'evidence_origin': 'appworld', 'granularity': env.get('granularity', 'code_block'),
            'api_revision': env.get('appworld_revision'), 'audit_domain': 'SQLite fields, offline only'}


def _initial_state(metadata: dict[str, Any]) -> dict[str, Any]:
    # Only the same public instruction/supervisor available to the actor.
    from .environment import public_state_update
    state = {'fields': [{'key': 'task.instruction', 'kind': 'text', 'value': metadata['instruction'], 'known': True, 'exists': True, 'choices': []}], 'relations': []}
    return public_state_update(state, {'name': 'supervisor.public_identity', 'arguments': {}}, json.dumps(metadata['supervisor']))


def _execute(world: Any, action: dict[str, Any], state: dict[str, Any], env: dict[str, Any], offline: bool, budget_remaining: float | None = None) -> dict[str, Any]:
    if budget_remaining is not None and budget_remaining < 1:
        raise ValueError('Insufficient budget for one code block; charged_cost=0')
    tracker = world.requester.request_tracker
    previous_limit = tracker.max_num_requests
    if budget_remaining is not None:
        api_limit = max(0, math.floor(budget_remaining - 1))
        tracker.max_num_requests = min(previous_limit, api_limit) if previous_limit is not None else api_limit
    before = snapshot_world(world, env.get('audit_ignore_fields')) if offline else None
    calls_before = len(world.requester.request_tracker.requests)
    start = time.monotonic()
    output, error = '', None
    try:
        output = world.execute(action_code(action, granularity=env.get('granularity', 'code_block')))
        if not isinstance(output, str):
            raise TypeError('AppWorld.execute must return text')
        if _ERROR.search(output):
            error = 'AppWorld execution reported an error; partial writes are retained'
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
    tracker.max_num_requests = previous_limit
    calls = len(world.requester.request_tracker.requests) - calls_before
    # Code blocks and all API calls are distinct charged resource components.
    cost = 1. + max(calls, 0)
    public = public_state_update(state, action, output, env.get('observations', []))
    after = snapshot_world(world, env.get('audit_ignore_fields')) if offline else None
    if before is not None and after is not None:
        before, after = align_audit_states(before, after)
        domain = sorted(set(field_map(before)) | set(field_map(after)))
        changes = semantic_diff(before, after)
    else:
        # Retained public-cache fields can be stale after a side effect. Only
        # fields actually returned by this execution are in a partial audit;
        # cached-but-unread values must never become unchanged negatives.
        fresh = public_state_update({'fields': [], 'relations': []}, action, output, env.get('observations', []))
        domain = sorted(key for key, field in field_map(fresh).items() if field.get('known'))
        changes = sorted(set(semantic_diff(state, public)) & set(domain))
    result = {'state': public, 'observation': output, 'valid': error is None, 'error': error,
              'cost': cost, 'cost_breakdown': {'code_blocks': 1, 'api_calls': max(calls, 0)}, 'api_calls': max(calls, 0),
              'duration_seconds': time.monotonic() - start,
              'audit': {'complete': bool(offline), 'field_domain': domain, 'changes': changes, 'model_version': 'unselected', 'domain_kind': 'global_sqlite' if offline else 'public_partial'},
              'audit_before_state': before, 'audit_state': after, 'fingerprint': _fingerprint(world) if offline else None,
              'done': bool(world.task_completed()), 'evidence_origin': 'appworld'}
    return result


def worker_main(connection: Any, request: dict[str, Any]) -> None:
    world = None
    total_cost = 0.
    pending_cost = False
    try:
        env = request['config']['environment']
        if request['mode'] == 'branch' and request['split'] != 'train':
            raise ValueError('Counterfactual replay is train-only')
        if env.get('offline_audit') and request['split'] not in ('train', 'dev'):
            raise ValueError('Offline audit is train/dev only')
        _set_root(str(Path(env['root']).expanduser().resolve()))
        from appworld import AppWorld
        world = AppWorld(task_id=request['task_id'], experiment_name=request['experiment'],
                         load_ground_truth=False, random_seed=int(request['config'].get('collection', {}).get('seed', 100)),
                         max_interactions=int(env.get('max_interactions', 10000)),
                         max_api_calls_per_interaction=int(env.get('max_api_calls_per_interaction', 1000)),
                         raise_on_unsafe_syntax=True, raise_on_unsafe_execution=True)
        metadata = _public_metadata(world, request)
        state = _initial_state(metadata)
        offline = bool(env.get('offline_audit', False))
        if request['mode'] == 'branch':
            if not offline:
                raise ValueError('Training counterfactuals require offline_audit=true for verified prefix and condition labels')
            total_cost, details = 0., {'replay': 0., 'condition': 0., 'target': 0.}
            prefix_receipts = []
            condition_receipts = []
            def remaining() -> float | None:
                return None if request.get('max_cost') is None else float(request['max_cost']) - total_cost
            def execute_charged(action: dict[str, Any]) -> dict[str, Any]:
                nonlocal pending_cost
                pending_cost = True
                try:
                    result = _execute(world, action, state, env, offline, remaining())
                except ValueError as error:
                    if 'charged_cost=0' in str(error):
                        pending_cost = False  # Rejected before executing a block.
                    raise
                pending_cost = False
                return result
            for index, entry in enumerate(request.get('prefix', [])):
                action = entry.get('action', entry)
                result = execute_charged(action)
                state = result['state']
                total_cost += result['cost']
                details['replay'] += result['cost']
                prefix_receipts.append({'step': index, 'output': result['observation'], 'fingerprint': result['fingerprint'], 'valid': result['valid'], 'cost': result['cost']})
                # Failed original blocks replay unchanged; partial writes and
                # interpreter locals remain part of their exact prefix.
                if 'output' in entry and result['observation'] != entry['output']:
                    raise ValueError(f'Prefix observation mismatch at step {index}; charged_cost={total_cost}')
                if entry.get('fingerprint') and result['fingerprint'] != entry['fingerprint']:
                    raise ValueError(f'Prefix database fingerprint mismatch at step {index}; charged_cost={total_cost}')
            prefix_state = deepcopy(state)
            prefix_fingerprint = _fingerprint(world)
            condition = request.get('condition')
            if condition:
                validate_condition(condition)
                for intervention in condition.get('actions', []):
                    result = execute_charged(intervention)
                    state = result['state']
                    total_cost += result['cost']
                    details['condition'] += result['cost']
                    condition_receipts.append({'action': deepcopy(intervention), 'output': result['observation'], 'fingerprint': result['fingerprint'], 'valid': result['valid'], 'cost': result['cost']})
                    if not result['valid']:
                        raise ValueError(f'Condition action failed; charged_cost={total_cost}')
                fingerprint = _fingerprint(world)
                result = execute_charged(condition['query'])
                state = result['state']
                total_cost += result['cost']
                details['condition'] += result['cost']
                condition_receipts.append({'action': deepcopy(condition['query']), 'output': result['observation'], 'fingerprint': result['fingerprint'], 'valid': result['valid'], 'cost': result['cost']})
                if not result['valid'] or result['observation'] != condition['expected_output'] or _fingerprint(world) != fingerprint:
                    raise ValueError(f'Condition query failed to establish a read-only observable condition; charged_cost={total_cost}')
            before = deepcopy(state)
            result = execute_charged(request['action'])
            details['target'] = result['cost']
            result.update(before_state=before, prefix_state=prefix_state, prefix_verified=True,
                          prefix_receipts=prefix_receipts, prefix_fingerprint=prefix_fingerprint, condition_receipts=condition_receipts, condition_verified=True, cost=total_cost + result['cost'],
                          cost_breakdown=details, process_isolation='fresh_spawn', process_id=os.getpid())
            connection.send({'ok': True, 'result': result})
            return
        connection.send({'ok': True, 'result': {'state': state, 'metadata': metadata}})
        while True:
            message = connection.recv()
            command = message['command']
            if command == 'close':
                break
            if command == 'execute':
                result = _execute(world, message['action'], state, env, offline, message.get('budget_remaining'))
                state = result['state']
            elif command == 'snapshot':
                if not offline or request['split'] not in ('train', 'dev'):
                    raise ValueError('Global snapshots are offline train/dev only')
                result = {'state': snapshot_world(world, env.get('audit_ignore_fields'))}
            elif command == 'evaluate':
                result = evaluate_fresh(world, request)
            else:
                raise ValueError(f'Unknown worker command {command}')
            connection.send({'ok': True, 'result': result})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            connection.send({'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(), 'charged_cost': total_cost, 'cost_unresolved': pending_cost or request.get('mode') != 'branch'})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if world is not None:
            world.close()
        connection.close()
