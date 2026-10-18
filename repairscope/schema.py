"""Typed, observable transitions and conservative supervision masks.

Unknown is information, never an unchanged label.  Global audits are separate
from state: neither this module nor an adapter merges audit fields into inputs.
"""
from __future__ import annotations

from copy import deepcopy
import math
import json
import operator
import re
from typing import Any, Iterable, Mapping

KINDS = frozenset({'number', 'category', 'bool', 'text', 'reference'})
SPLITS = frozenset({'train', 'dev', 'test_normal', 'test_challenge'})
OPS = frozenset({'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'in', 'not_in', 'contains', 'exists', 'absent'})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strings(value: Any, label: str) -> list[str]:
    _require(isinstance(value, list) and all(isinstance(x, str) and x for x in value), f'{label} must be a list of nonempty strings')
    _require(len(value) == len(set(value)), f'{label} has duplicates')
    return value


def validate_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(isinstance(state, dict), 'state must be an object')
    fields = state.get('fields')
    _require(isinstance(fields, list), 'state.fields must be a list')
    keys = []
    for field in fields:
        _require(isinstance(field, dict), 'field must be an object')
        key, kind = field.get('key'), field.get('kind')
        _require(isinstance(key, str) and bool(key), 'field.key must be nonempty')
        keys.append(key)
        _require(kind in KINDS, f'{key}: invalid field kind')
        _require(type(field.get('known')) is bool and type(field.get('exists')) is bool, f'{key}: known and exists must be bool')
        _require('value' in field, f'{key}: missing value')
        value = field['value']
        if field.get('nullable', False) and field['known'] and field['exists'] and value is None:
            pass
        elif not field['known'] or not field['exists']:
            _require(value is None, f'{key}: unknown/deleted fields must have null value')
        elif kind == 'number':
            _require(type(value) in (int, float) and math.isfinite(value), f'{key}: number must be finite')
        elif kind == 'bool':
            _require(type(value) is bool, f'{key}: bool must be bool')
        elif kind in ('category', 'text'):
            _require(isinstance(value, str), f'{key}: text/category must be a string')
        elif kind == 'reference':
            _require(type(value) in (int, str), f'{key}: reference must be an ID, never a numeric regression target')
        choices = field.get('choices', [])
        _require(isinstance(choices, list), f'{key}: choices must be a list')
        _require(all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in choices), f'{key}: invalid choices')
        _require(len(choices) == len(set(choices)), f'{key}: duplicate choices')
        if kind == 'category' and field['known'] and field['exists'] and choices:
            _require(value in choices, f'{key}: category not in choices')
        if 'scale' in field:
            _require(type(field['scale']) in (float, int) and math.isfinite(field['scale']) and field['scale'] > 0, f'{key}: scale must be positive')
        if 'tolerance' in field:
            _require(type(field['tolerance']) in (float, int) and math.isfinite(field['tolerance']) and field['tolerance'] >= 0, f'{key}: tolerance must be finite and nonnegative')
        if 'entity' in field:
            _require(isinstance(field['entity'], str), f'{key}: entity must be a string')
    _require(len(keys) == len(set(keys)), 'state has duplicate field keys')
    relations = state.get('relations', [])
    _require(isinstance(relations, list), 'relations must be a list')
    for relation in relations:
        _require(isinstance(relation, (list, tuple)) and len(relation) == 2 and all(isinstance(k, str) and k in keys for k in relation), 'relations must connect present field tokens')
    return state


def field_map(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {f['key']: f for f in state.get('fields', [])}


def validate_action(action: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(isinstance(action, dict), 'action must be an object')
    _require(isinstance(action.get('name'), str) and bool(action['name']), 'action.name required')
    _require(isinstance(action.get('arguments'), dict), 'action.arguments must be an object')
    try:
        json.dumps(action['arguments'], allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('action arguments must be finite JSON values') from error
    if 'code' in action:
        _require(isinstance(action['code'], str) and bool(action['code'].strip()), 'action.code must be nonempty text')
    cost = action.get('cost', 1)
    _require(type(cost) in (int, float) and math.isfinite(cost) and cost > 0, 'action.cost must be positive')
    return action


def validate_goals(goals: Any) -> list[dict[str, Any]]:
    _require(isinstance(goals, list), 'goals must be a list')
    ids = []
    for goal in goals:
        _require(isinstance(goal, dict), 'goal must be an object')
        _require(isinstance(goal.get('id'), str) and bool(goal['id']), 'goal.id required')
        ids.append(goal['id'])
        _require(isinstance(goal.get('field'), str) and bool(goal['field']), 'goal.field required')
        _require(goal.get('op') in OPS, 'unsupported goal predicate')
        _require(type(goal.get('protected', False)) is bool, 'goal.protected must be bool')
        _require(type(goal.get('weight', 1)) in (int, float) and math.isfinite(goal.get('weight', 1)) and goal.get('weight', 1) >= 0, 'goal.weight invalid')
        for key in ('active_from', 'active_until', 'protect_from_step', 'protect_until_step'):
            if key in goal:
                _require(type(goal[key]) is int and goal[key] >= 0, f'{key} must be a nonnegative step')
        if 'protect_after_satisfied' in goal:
            _require(type(goal['protect_after_satisfied']) is bool, 'protect_after_satisfied must be bool')
        if goal['op'] not in ('exists', 'absent'):
            _require('value' in goal, 'predicate value required')
    _require(len(ids) == len(set(ids)), 'duplicate goal IDs')
    return goals


def effective_protected(goal: Mapping[str, Any], step: int = 0, before: bool | None = None) -> bool:
    """Explicit protection intervals; optionally protect only achieved subgoals."""
    if not goal.get('protected', False):
        return False
    start = goal.get('protect_from_step', goal.get('active_from', 0))
    end = goal.get('protect_until_step', goal.get('active_until'))
    if step < start or end is not None and step >= end:
        return False
    return not goal.get('protect_after_satisfied', False) or before is True


def evaluate_goals(state: Mapping[str, Any], goals: Iterable[Mapping[str, Any]]) -> list[bool | None]:
    fields = field_map(state)
    result: list[bool | None] = []
    operations = {'eq': operator.eq, 'ne': operator.ne, 'gt': operator.gt, 'ge': operator.ge, 'lt': operator.lt, 'le': operator.le}
    for goal in goals:
        field = fields.get(goal['field'])
        if field is None or not field.get('known', False):
            result.append(None)
            continue
        op, exists = goal['op'], field['exists']
        if op in ('exists', 'absent'):
            result.append(exists if op == 'exists' else not exists)
        elif not exists:
            result.append(False)
        else:
            a, b = field['value'], goal.get('value')
            try:
                if op in operations:
                    # True must not equal numeric 1 in typed goal semantics.
                    if op in ('eq', 'ne') and ((type(a) is bool) != (type(b) is bool)):
                        value = op == 'ne'
                    else:
                        value = operations[op](a, b)
                elif op == 'in':
                    value = a in b
                elif op == 'not_in':
                    value = a not in b
                elif op == 'contains':
                    value = b in a
                else:
                    raise ValueError(f'Unsupported goal operation: {op}')
                result.append(bool(value))
            except (TypeError, KeyError):
                result.append(None)
    return result


def semantic_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Semantic field differences, including explicit creation/deletion.

    Missing fields and unknown values cannot prove a change or an unchanged
    target. Deleted fields require explicit known=True, exists=False tombstones.
    Numeric thresholds are field-level fixed `tolerance` values (default zero).
    """
    left, right = field_map(before), field_map(after)
    changed = []
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if not a.get('known') or not b.get('known'):
            continue
        if a['exists'] != b['exists']:
            changed.append(key)
        elif not a['exists']:
            continue
        elif a['kind'] != b['kind']:
            changed.append(key)
        elif a['kind'] == 'number' and a['value'] is not None and b['value'] is not None:
            tolerance = max(float(a.get('tolerance', 0)), float(b.get('tolerance', 0)))
            if abs(a['value'] - b['value']) > tolerance:
                changed.append(key)
        elif type(a['value']) is not type(b['value']) or a['value'] != b['value']:
            changed.append(key)
    return changed


def escape_label(record: Mapping[str, Any], scope: Iterable[str] | None = None) -> bool | None:
    audit = record['audit']
    selected = set(record['scope'] if scope is None else scope)
    if set(audit['changes']) - selected:
        return True
    return False if audit['complete'] else None


def validate_transition(record: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(isinstance(record, dict), 'transition must be an object')
    for key in ('task_id', 'group_id'):
        _require(isinstance(record.get(key), str) and bool(record[key]), f'{key} required')
    _require(record.get('split') in SPLITS, 'invalid split; use official AppWorld split names')
    _require(record.get('evidence_origin') in ('appworld', 'synthetic'), 'explicit evidence_origin required')
    _require(type(record.get('step')) is int and record['step'] >= 0, 'step must be nonnegative integer')
    validate_state(record.get('state'))
    validate_state(record.get('next_state'))
    validate_action(record.get('action'))
    goals = validate_goals(record.get('goals'))
    for key in ('goal_before', 'goal_after'):
        values = record.get(key)
        _require(isinstance(values, list) and len(values) == len(goals) and all(v is None or type(v) is bool for v in values), f'{key} must align with goals and contain bool/null')
    history = record.get('history')
    _require(isinstance(history, list), 'history must be a list')
    for entry in history:
        _require(isinstance(entry, dict), 'history entry must be an object')
        validate_state(entry.get('state'))
        validate_action(entry.get('action'))
    _strings(record.get('scope'), 'scope')
    audit = record.get('audit')
    _require(isinstance(audit, dict) and type(audit.get('complete')) is bool, 'audit.complete must be bool')
    domain, changes = _strings(audit.get('field_domain'), 'audit.field_domain'), _strings(audit.get('changes'), 'audit.changes')
    _require(set(changes) <= set(domain), 'audit changes must lie within declared field domain')
    _require(isinstance(audit.get('model_version'), str) and bool(audit['model_version']), 'audit.model_version required')
    if 'scope_model_version' in record:
        _require(record['scope_model_version'] == audit['model_version'], 'historical scope model version cannot drift')
    if any(k in record for k in ('cf_pair_id', 'cf_condition', 'cf_role')):
        _require(isinstance(record.get('cf_pair_id'), str) and bool(record['cf_pair_id']), 'cf_pair_id required')
        _require(record.get('cf_condition') in ('c0', 'c1') and record.get('cf_role') in ('action', 'reference'), 'counterfactual condition/role invalid')
    if 'action_valid' in record:
        _require(type(record['action_valid']) is bool, 'action_valid must be bool')
    return record


def cf_four_grid_targets(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Align a four-cell pair; unavailable/partial cells never become zeros."""
    records = list(records)
    cells: dict[tuple[str, str], Mapping[str, Any]] = {}
    pair_ids = {r.get('cf_pair_id') for r in records}
    _require(len(pair_ids) == 1 and None not in pair_ids, 'one counterfactual pair required')
    groups = {(r['task_id'], r['group_id'], r['split']) for r in records}
    _require(len(groups) == 1, 'four cells must share source task/group/split')
    for record in records:
        cell = record['cf_condition'], record['cf_role']
        _require(cell not in cells, 'duplicate counterfactual cell')
        cells[cell] = record
    keys = sorted({k for r in records for k in r['audit']['field_domain']})
    target = {'keys': keys, 'y0': [], 'y1': [], 'mask': [], 'pair_id': next(iter(pair_ids))}
    for key in keys:
        ys, valid = [], True
        for condition in ('c0', 'c1'):
            a, b = cells.get((condition, 'action')), cells.get((condition, 'reference'))
            if a is None or b is None:
                ys.append(None)
                valid = False
                continue
            fa, fb = field_map(a.get('audit_next_state', a['next_state'])).get(key), field_map(b.get('audit_next_state', b['next_state'])).get(key)
            aligned = (a.get('action_valid', True) and b.get('action_valid', True)
                       and key in a['audit']['field_domain'] and key in b['audit']['field_domain']
                       and fa is not None and fb is not None and fa['known'] and fb['known'])
            if not aligned:
                ys.append(None)
                valid = False
            else:
                ys.append(float(bool(semantic_diff({'fields': [fa]}, {'fields': [fb]}))))
        target['y0'].append(ys[0])
        target['y1'].append(ys[1])
        target['mask'].append(bool(valid))
    return target


def relabel_scope(record: Mapping[str, Any], scope: Iterable[str], model_version: str) -> dict[str, Any]:
    """Explicitly derive a new record without mutating historical evidence."""
    _require(record['audit']['complete'], 'scope reselection needs complete stored audit')
    out = deepcopy(record)
    out['previous_scope'] = deepcopy(record['scope'])
    out['previous_scope_model_version'] = record['audit']['model_version']
    out['scope'] = list(scope)
    out['audit']['model_version'] = model_version
    out['scope_model_version'] = model_version
    out['escape_label'] = escape_label(out)
    validate_transition(out)
    return out
