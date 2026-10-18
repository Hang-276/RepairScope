"""Typed, observable transitions and conservative supervision masks.

Unknown is information, never an unchanged label.  Global audits are separate
from state: neither this module nor an adapter merges audit fields into inputs.
"""
from __future__ import annotations

from copy import deepcopy
import ast
import math
import json
import operator
import re
from typing import Any, Iterable, Mapping

KINDS = frozenset({'number', 'category', 'bool', 'text', 'reference'})
SPLITS = frozenset({'train', 'dev', 'test_normal', 'test_challenge'})
OPS = frozenset({'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'in', 'not_in', 'contains', 'exists', 'absent'})
GOAL_KINDS = frozenset({'predicate', 'all', 'any', 'none', 'count', 'aggregate', 'answer', 'complete'})
AGGREGATES = frozenset({'count', 'sum', 'mean', 'min', 'max'})
GOAL_KEYS = frozenset({
    'id', 'kind', 'field', 'fields', 'op', 'value', 'aggregate', 'quantifier',
    'where', 'expr', 'filter_op', 'filter_value', 'filter_expr', 'source',
    'protected', 'weight', 'protect_from_step', 'protect_until_step',
    'active_from', 'active_until', 'protect_after_satisfied', 'description', 'entity',
})


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
        unknown = set(goal) - GOAL_KEYS
        _require(not unknown, f'goal {goal["id"]} has unsupported keys: {sorted(unknown)}')
        kind = goal.get('kind', 'predicate')
        _require(kind in GOAL_KINDS, 'unsupported goal kind')
        if 'quantifier' in goal:
            _require(goal['quantifier'] in ('all', 'any', 'none'), 'unsupported goal quantifier')
            _require(kind == 'predicate', 'quantifier and kind cannot both be set')
        fields = goal.get('fields')
        if fields is not None:
            _require(isinstance(fields, list) and fields and all(isinstance(x, str) and x for x in fields),
                     'goal.fields must be a nonempty string list')
        if kind in ('predicate', 'all', 'any', 'none', 'count', 'aggregate'):
            _require(isinstance(goal.get('field'), str) and bool(goal['field']) or fields,
                     'goal.field or goal.fields required')
            _require(goal.get('op') in OPS, 'unsupported goal predicate')
            if goal['op'] not in ('exists', 'absent'):
                _require('value' in goal or 'expr' in goal, 'predicate value or expr required')
            if kind == 'aggregate':
                _require(goal.get('aggregate') in AGGREGATES, 'aggregate kind requires a supported aggregate')
                if 'source' in goal:
                    _require(goal['source'] in ('current', 'previous'), 'aggregate source must be current or previous')
            if kind == 'count' and 'filter_op' in goal:
                _require(goal['filter_op'] in OPS, 'unsupported count filter operator')
                if goal['filter_op'] not in ('exists', 'absent'):
                    _require('filter_value' in goal or 'filter_expr' in goal,
                             'count filter needs filter_value or filter_expr')
        elif kind == 'answer':
            # Answer goals may omit field and use the public/audit answer slot.
            if goal.get('op') is not None:
                _require(goal['op'] in OPS, 'unsupported answer operator')
            _require('value' in goal or 'expr' in goal or 'field' in goal,
                     'answer goal needs a value, expression, or field')
        elif kind == 'complete':
            if goal.get('op') is not None:
                _require(goal['op'] in OPS, 'unsupported complete operator')
        _require(type(goal.get('protected', False)) is bool, 'goal.protected must be bool')
        _require(type(goal.get('weight', 1)) in (int, float) and math.isfinite(goal.get('weight', 1)) and goal.get('weight', 1) >= 0, 'goal.weight invalid')
        for key in ('active_from', 'active_until', 'protect_from_step', 'protect_until_step'):
            if key in goal:
                _require(type(goal[key]) is int and goal[key] >= 0, f'{key} must be a nonnegative step')
        if 'protect_after_satisfied' in goal:
            _require(type(goal['protect_after_satisfied']) is bool, 'protect_after_satisfied must be bool')
        if 'expr' in goal:
            _require(isinstance(goal['expr'], str) and bool(goal['expr']), 'goal.expr must be nonempty text')
            _validate_expression(goal['expr'])
        if 'filter_expr' in goal:
            _require(isinstance(goal['filter_expr'], str) and bool(goal['filter_expr']),
                     'goal.filter_expr must be nonempty text')
            _validate_expression(goal['filter_expr'])
        if isinstance(goal.get('value'), dict) and 'expr' in goal['value']:
            _require(isinstance(goal['value']['expr'], str) and bool(goal['value']['expr']), 'value.expr must be nonempty text')
            _validate_expression(goal['value']['expr'])
        if goal.get('where') is not None:
            _require(isinstance(goal['where'], list), 'goal.where must be a list')
            for predicate in goal['where']:
                _require(isinstance(predicate, dict) and predicate.get('op') in OPS, 'invalid goal.where predicate')
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


def _pattern_regex(pattern: str) -> re.Pattern:
    """Translate a public field key pattern into a matcher.

    ``*`` matches within one dot-separated segment (including an entity id or
    field name), ``**`` matches across segments. Patterns are only matched
    against public/audit state keys; they never query a hidden store.
    """
    chunks = pattern.split('**')
    body = '.*'.join(re.escape(chunk).replace(r'\*', '[^.]*') for chunk in chunks)
    return re.compile('^' + body + '$')


def _patterns(goal: Mapping[str, Any]) -> list[str]:
    if goal.get('fields'):
        return list(goal['fields'])
    if goal.get('field'):
        return [goal['field']]
    return []


def _matching_fields(state: Mapping[str, Any], goal: Mapping[str, Any]) -> list[tuple[str, dict]]:
    fields = field_map(state)
    out, seen = [], set()
    for pattern in _patterns(goal):
        if '*' in pattern:
            matcher = _pattern_regex(pattern)
            keys = [key for key in fields if matcher.match(key)]
        else:
            keys = [pattern] if pattern in fields else []
        for key in keys:
            if key not in seen:
                seen.add(key)
                out.append((key, fields[key]))
    return out


def _known(field: Mapping[str, Any] | None) -> bool:
    return field is not None and bool(field.get('known', False))


def _compare(a: Any, b: Any, op: str) -> bool | None:
    if op in ('exists', 'absent'):
        return None
    try:
        if op == 'eq':
            return not ((type(a) is bool) != (type(b) is bool)) and a == b
        if op == 'ne':
            return ((type(a) is bool) != (type(b) is bool)) or a != b
        if op == 'gt':
            return a > b
        if op == 'ge':
            return a >= b
        if op == 'lt':
            return a < b
        if op == 'le':
            return a <= b
        if op == 'in':
            return a in b
        if op == 'not_in':
            return a not in b
        if op == 'contains':
            return b in a
    except (TypeError, KeyError):
        return None
    return None


def _tri_and(values: Iterable[bool | None]) -> bool | None:
    values = list(values)
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True if values else None


def _tri_or(values: Iterable[bool | None]) -> bool | None:
    values = list(values)
    if any(value is True for value in values):
        return True
    if any(value is None for value in values):
        return None
    return False if values else None


def _tri_none(values: Iterable[bool | None]) -> bool | None:
    values = list(values)
    if any(value is True for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True if values else None


def _aggregate_values(name: str, values: list[float], unknown: int) -> float | None:
    if not values:
        return None
    if name == 'count':
        return float(len(values))
    if name == 'sum':
        return float(sum(values))
    if name == 'mean':
        return float(sum(values) / len(values))
    if name == 'min':
        return float(min(values))
    if name == 'max':
        return float(max(values))
    return None


def _aggregate_compare(name: str, values: list[float], unknown: int, op: str, target: Any) -> bool | None:
    known = _aggregate_values(name, values, unknown)
    if known is None:
        return None
    if not unknown:
        return _compare(known, target, op)
    # Unknown values can only move the aggregate in a known direction for
    # sum/count (up), max (up) and min (down). mean cannot be bounded.
    if name == 'mean':
        return None
    if name in ('sum', 'count', 'max'):
        if op in ('ge', 'gt'):
            return _compare(known, target, op) if _compare(known, target, op) is True else None
        if op in ('le', 'lt'):
            result = _compare(known, target, op)
            return False if result is False else None
    if name == 'min':
        if op in ('le', 'lt'):
            return _compare(known, target, op) if _compare(known, target, op) is True else None
        if op in ('ge', 'gt'):
            result = _compare(known, target, op)
            return False if result is False else None
    return None


_ALLOWED_EXPR_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Tuple, ast.List, ast.Dict,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.arguments, ast.arg,
)


def _validate_expression(expression: str) -> None:
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as error:
        raise ValueError(f'Invalid goal expression: {error}') from error
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_EXPR_NODES):
            raise ValueError(f'Goal expression forbids {type(node).__name__}')
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in {
                    'min', 'max', 'abs', 'round', 'len', 'field', 'aggregate',
                    'previous_field', 'previous_aggregate', 'number', 'text', 'bool'}:
                raise ValueError('Goal expression forbids this call; use only min/max/abs/round/len/field/'
                                 'aggregate/previous_field/previous_aggregate/number/text/bool')
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('Goal expression may not read private names')


def _aggregate_from_state(state: Mapping[str, Any], name: str, pattern: str) -> float | None:
    fields = _matching_fields(state, {'field': pattern})
    values, unknown = [], 0
    for _, field in fields:
        if _known(field) and field.get('exists', True) and type(field.get('value')) in (int, float) and not isinstance(field.get('value'), bool):
            values.append(float(field['value']))
        else:
            unknown += 1
    return _aggregate_values(name, values, unknown)


def _field_lookup(state: Mapping[str, Any], key: str) -> Any:
    field = field_map(state).get(key)
    return field.get('value') if _known(field) and field.get('exists', True) else None


def _eval_expression(expression: str, state: Mapping[str, Any], extra: Mapping[str, Any] | None = None,
                     previous: Mapping[str, Any] | None = None) -> Any:
    _validate_expression(expression)
    tree = ast.parse(expression, mode='eval')
    env = dict(extra or {})
    env.update({
        'field': lambda key: _field_lookup(state, key),
        'aggregate': lambda name, pattern: _aggregate_from_state(state, name, pattern),
        'previous_field': lambda key: _field_lookup(previous, key) if previous is not None else None,
        'previous_aggregate': lambda name, pattern: _aggregate_from_state(previous, name, pattern) if previous is not None else None,
        'number': float,
        'text': str,
        'bool': bool,
    })

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise KeyError(node.id)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                          ast.Mod: operator.mod, ast.Pow: operator.pow}
            return operations[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.Not):
                return not value
            raise ValueError('Unsupported unary operator')
        if isinstance(node, ast.BoolOp):
            values = [visit(value) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for op_node, comparator in zip(node.ops, node.comparators):
                right = visit(comparator)
                operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                              ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
                if not operations[type(op_node)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return visit(node.body) if visit(node.test) else visit(node.orelse)
        if isinstance(node, (ast.Tuple, ast.List)):
            return [visit(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {visit(key): visit(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Call):
            name = node.func.id
            args = [visit(arg) for arg in node.args]
            if name == 'field':
                return _field_lookup(state, str(args[0]))
            if name == 'aggregate':
                return _aggregate_from_state(state, str(args[0]), str(args[1]))
            functions = {'min': min, 'max': max, 'abs': abs, 'round': round, 'len': len,
                         'number': float, 'text': str, 'bool': bool}
            return functions[name](*args)
        raise ValueError(f'Unsupported goal expression node {type(node).__name__}')

    return visit(tree)


def _target_value(goal: Mapping[str, Any], state: Mapping[str, Any], field: Mapping[str, Any],
                  previous: Mapping[str, Any] | None = None) -> Any:
    value = goal.get('value')
    previous_field = field_map(previous).get(field.get('key')) if previous is not None else None
    context = {
        'value': field.get('value'),
        'current': field.get('value'),
        'key': field.get('key'),
        'previous_value': previous_field.get('value') if _known(previous_field) else None,
        'previous_known': _known(previous_field),
        'previous_exists': bool(previous_field.get('exists', True)) if previous_field is not None else None,
    }
    if 'expr' in goal:
        return _eval_expression(goal['expr'], state, context, previous)
    if isinstance(value, dict):
        if 'expr' in value:
            return _eval_expression(value['expr'], state, context, previous)
        source = previous if value.get('source') == 'previous' else state
        if 'aggregate' in value and 'field' in value:
            return _aggregate_from_state(source, str(value['aggregate']), str(value['field']))
        if 'field' in value:
            base = _field_lookup(source, str(value['field']))
            if base is None:
                return None
            operation = value.get('op', 'add')
            amount = value.get('value', value.get('amount', 0))
            operations = {'add': operator.add, 'sub': operator.sub, 'mul': operator.mul,
                          'truediv': operator.truediv}
            if operation not in operations:
                return None
            try:
                return operations[operation](base, amount)
            except (TypeError, ZeroDivisionError):
                return None
    return value


def _evaluate_predicate(state: Mapping[str, Any], goal: Mapping[str, Any], field: Mapping[str, Any],
                        previous: Mapping[str, Any] | None = None) -> bool | None:
    op = goal.get('op', 'eq')
    if op in ('exists', 'absent'):
        exists = bool(field.get('exists', True)) if _known(field) else None
        if exists is None:
            return None
        return exists if op == 'exists' else not exists
    if not _known(field):
        return None
    if not field.get('exists', True):
        return False
    target = _target_value(goal, state, field, previous)
    if target is None and 'value' in goal and goal['value'] is not None:
        return None
    return _compare(field.get('value'), target, op)


def _answer_field(state: Mapping[str, Any], goal: Mapping[str, Any]) -> dict | None:
    fields = _matching_fields(state, goal) if _patterns(goal) else []
    if not fields:
        candidates = [(key, field) for key, field in field_map(state).items()
                      if key.endswith('.answer') or key.endswith('].answer')]
        fields = candidates
    for _, field in fields:
        if _known(field) and field.get('exists', True):
            return field
    return None


def _complete_field(state: Mapping[str, Any], goal: Mapping[str, Any]) -> dict | None:
    fields = _matching_fields(state, goal) if _patterns(goal) else []
    if not fields:
        fields = [(key, field) for key, field in field_map(state).items()
                  if (key.endswith('.status') or key.endswith('].status')) and 'supervisor' in key]
    for _, field in fields:
        if _known(field) and field.get('exists', True):
            return field
    return None


def _evaluate_goal(state: Mapping[str, Any], goal: Mapping[str, Any],
                   previous: Mapping[str, Any] | None = None) -> bool | None:
    kind = goal.get('kind', 'predicate')
    quantifier = goal.get('quantifier')
    if quantifier:
        kind = quantifier
    if kind == 'answer':
        field = _answer_field(state, goal)
        if field is None:
            return None
        if 'value' in goal or 'expr' in goal:
            target = _target_value(goal, state, field, previous)
            if target is None:
                return None
            return _compare(field.get('value'), target, goal.get('op', 'eq'))
        value = field.get('value')
        return bool(value) if isinstance(value, (str, int, float, bool)) else None
    if kind == 'complete':
        field = _complete_field(state, goal)
        if field is None:
            return None
        expected = goal.get('value', 'success')
        return _compare(field.get('value'), expected, goal.get('op', 'eq'))
    if kind in ('predicate', 'all', 'any', 'none'):
        matches = _matching_fields(state, goal)
        if not matches:
            return None
        values = [_evaluate_predicate(state, goal, field, previous) for _, field in matches]
        if kind == 'predicate':
            return values[0] if len(values) == 1 else _tri_and(values)
        if kind == 'all':
            return _tri_and(values)
        if kind == 'any':
            return _tri_or(values)
        return _tri_none(values)
    if kind == 'count':
        matches = _matching_fields(state, goal)
        if not matches:
            return None
        if 'filter_op' in goal:
            filter_goal = {'op': goal['filter_op']}
            if 'filter_value' in goal:
                filter_goal['value'] = goal['filter_value']
            if 'filter_expr' in goal:
                filter_goal['expr'] = goal['filter_expr']
            values = [_evaluate_predicate(state, filter_goal, field, previous) for _, field in matches]
        else:
            values = [True if _known(field) else None for _, field in matches]
        true_count = sum(value is True for value in values)
        unknown = sum(value is None for value in values)
        op, target = goal.get('op', 'ge'), goal.get('value')
        if op == 'eq':
            return None if unknown else _compare(true_count, target, 'eq')
        if op in ('ge', 'gt'):
            if _compare(true_count, target, op) is True:
                return True
            if _compare(true_count + unknown, target, op) is False:
                return False
            return None
        if op in ('le', 'lt'):
            if _compare(true_count, target, op) is True:
                return True if unknown == 0 else None
            return False
        return None
    if kind == 'aggregate':
        source = previous if goal.get('source') == 'previous' and previous is not None else state
        matches = _matching_fields(source, goal)
        if not matches:
            return None
        name = goal.get('aggregate')
        values, unknown = [], 0
        for _, field in matches:
            value = field.get('value') if _known(field) and field.get('exists', True) else None
            if type(value) in (int, float) and not isinstance(value, bool):
                values.append(float(value))
            else:
                unknown += 1
        target = _target_value(goal, state, matches[0][1], previous)
        return _aggregate_compare(name, values, unknown, goal.get('op', 'ge'), target)
    return None


def goal_state_keys(goal: Mapping[str, Any], state: Mapping[str, Any]) -> list[str]:
    """Keys referenced by a goal in this state; used only for slot priorities."""
    return [key for key, _ in _matching_fields(state, goal)]


def evaluate_goals(state: Mapping[str, Any], goals: Iterable[Mapping[str, Any]],
                   previous: Mapping[str, Any] | None = None) -> list[bool | None]:
    return [_evaluate_goal(state, goal, previous) for goal in goals]


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
