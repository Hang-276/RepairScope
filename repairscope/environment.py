"""Public-observation adapters and isolated AppWorld worker transport.

No AppWorld import occurs here. Real environments are constructed exclusively in
spawned workers; global SQLite snapshots are separate offline audit outputs.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Protocol
import uuid

from .schema import SPLITS, validate_action, validate_state


class EnvironmentExecutionError(RuntimeError):
    def __init__(self, message: str, charged_cost: float | None = None, cost_unresolved: bool = False):
        self.charged_cost = charged_cost
        self.cost_unresolved = cost_unresolved
        message = re.sub(r';?\s*charged_cost[=:]\s*[0-9.]+', '', message)
        suffix = f'; charged_cost={charged_cost}' if charged_cost is not None and not cost_unresolved else '; cost_unresolved=true' if cost_unresolved else ''
        super().__init__(message + suffix)


class Environment(Protocol):
    metadata: dict[str, Any]
    history: list[dict[str, Any]]
    def reset(self) -> dict[str, Any]: ...
    def observe(self) -> dict[str, Any]: ...
    def execute(self, action: dict[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


def load_mapping(value: Any) -> Any:
    if not isinstance(value, (str, Path)):
        return deepcopy(value)
    path = Path(value).expanduser()
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        return yaml.safe_load(path.read_text(encoding='utf-8'))
    return json.loads(path.read_text(encoding='utf-8'))


def official_task_ids(root: str | Path, split: str) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f'Unknown official split {split}')
    path = Path(root).expanduser().resolve() / 'data' / 'datasets' / (split + '.txt')
    ids = [line.strip().split(':', 1)[0] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate official task IDs in {path}')
    return ids


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(config))
    if 'environment' not in result:
        result = {'environment': result}
    env = result['environment']
    if env.get('observation_map'):
        value = load_mapping(env['observation_map'])
        env['observations'] = value.get('observations', value) if isinstance(value, dict) else value
    goal_map = result.get('data', {}).get('goal_map', env.get('goal_map'))
    if goal_map:
        env['goal_map'] = load_mapping(goal_map)
    env.setdefault('backend', 'appworld')
    env.setdefault('granularity', 'code_block')
    if env['granularity'] not in ('api', 'code_block'):
        raise ValueError('environment.granularity must be api or code_block')
    env.setdefault('offline_audit', False)
    return result


def action_code(action: Mapping[str, Any], *, granularity: str = 'code_block') -> str:
    validate_action(action)
    if 'code' in action:
        code = action['code']
    else:
        name = action['name'].removeprefix('apis.')
        if not re.fullmatch(r'[a-zA-Z]\w*\.[a-zA-Z]\w*', name):
            raise ValueError('AppWorld action.name must be app.public_api')
        # Use JSON through literal Python strings; no action text interpolation.
        args = repr(dict(action['arguments']))
        code = f'import json\nprint(json.dumps(apis.{name}(**{args}), default=str))'
    if granularity == 'api':
        tree = ast.parse(code)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Attribute) and isinstance(n.func.value.value, ast.Name)
                 and n.func.value.value.id == 'apis']
        if len(calls) != 1 or any(isinstance(n, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.If, ast.IfExp, ast.Try, ast.BoolOp, ast.Match, ast.Lambda)) for n in ast.walk(tree)):
            raise ValueError('api granularity requires exactly one unconditional public API call')
    return code


def validate_condition(condition: Mapping[str, Any]) -> None:
    """Conditions execute public APIs; a read query must verify actual changes.

    This syntax gate restricts branch interventions only. Actor programs retain
    AppWorld's native sandbox and normal code-block semantics.
    """
    if not isinstance(condition, dict) or not isinstance(condition.get('actions', []), list):
        raise ValueError('condition.actions must be a list')
    query = condition.get('query')
    if not query or not isinstance(condition.get('expected_output'), str):
        raise ValueError('Every condition needs a public read query and exact expected_output')
    actions = [*condition.get('actions', []), query]
    allowed_nodes = (ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant,
                     ast.Call, ast.Attribute, ast.keyword, ast.List, ast.Tuple, ast.Dict,
                     ast.Subscript, ast.UnaryOp, ast.USub, ast.UAdd, ast.Import, ast.alias)
    for action in actions:
        validate_action(action)
        code = action_code(action)
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f'Condition syntax forbidden: {type(node).__name__}')
            if isinstance(node, ast.Import) and any(alias.name != 'json' or alias.asname for alias in node.names):
                raise ValueError('Only json may be imported in conditions')
            if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
                raise ValueError('Private condition attributes forbidden')
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in ('print', 'str'):
                    continue
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'json' and func.attr == 'dumps':
                    continue
                if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute)
                        and isinstance(func.value.value, ast.Name) and func.value.value.id == 'apis'):
                    raise ValueError('Conditions may call only public apis.app.endpoint')
                if action is query and not func.attr.startswith(('get_', 'show_', 'list_', 'search_', 'find_', 'read_', 'fetch_', 'count_', 'check_')):
                    raise ValueError('Condition verification query must use a public read API')


def _json_output(output: str) -> Any:
    text = output.strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Some programs print a label followed by one JSON result.
            decoder = json.JSONDecoder()
            for index, char in enumerate(text):
                if char in '{[':
                    try:
                        value, _ = decoder.raw_decode(text[index:])
                        return value
                    except ValueError:
                        pass
    return None


def _kind(value: Any, key: str) -> str:
    if key == 'id' or key.endswith('_id'):
        return 'reference'
    if type(value) is bool:
        return 'bool'
    if type(value) in (int, float):
        return 'number'
    return 'text'


def _field(key: str, value: Any, kind: str | None = None, **kwargs: Any) -> dict[str, Any]:
    field = {'key': key, 'kind': kind or _kind(value, key.rsplit('.', 1)[-1]), 'value': value,
             'known': True, 'exists': True, 'choices': [], **kwargs}
    if value is None:
        field['nullable'] = True
    elif field['kind'] == 'reference':
        field['value'] = str(value)
    return field


def _scalar_matches(field: dict[str, Any]) -> bool:
    """A mapped value that cannot be represented stays unknown, never a guess."""
    if not field.get('known', True) or field.get('value') is None:
        return True
    value, kind = field['value'], field['kind']
    if kind == 'reference':
        return type(value) in (int, str)
    if kind == 'bool':
        return type(value) is bool
    if kind == 'number':
        return type(value) in (int, float) and not isinstance(value, bool)
    if kind in ('text', 'category'):
        return isinstance(value, str)
    return False


def public_state_update(state: Mapping[str, Any], action: Mapping[str, Any], output: str, observations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Parse only actual API-returned content, never read models or database IDs.

    Schema mappings align public API field names with audit keys. Generic output
    tokens remain useful when no map exists, but do not claim entity alignment.
    A mapping can use a JSON path (dot-separated keys, '*' for lists), entity,
    id_field, key_template and field-level {kind,key,choices,scale} settings.
    """
    from .schema import field_map
    result = deepcopy(dict(state))
    fields = field_map(result)
    payload = _json_output(output)
    api = action['name'].removeprefix('apis.')
    selected = [spec for spec in (observations or []) if spec.get('api') in (api, '*')]
    new_relations = []
    def extract(value: Any, parts: list[str]) -> list[Any]:
        if not parts:
            return [value]
        head, *tail = parts
        if head == '*':
            return [out for item in (value if isinstance(value, list) else []) for out in extract(item, tail)]
        if isinstance(value, dict) and head in value:
            return extract(value[head], tail)
        if isinstance(value, list) and head.isdigit() and int(head) < len(value):
            return extract(value[int(head)], tail)
        return []
    if selected and payload is not None:
        for spec in selected:
            rows = extract(payload, [p for p in spec.get('path', '').split('.') if p])
            for row in rows:
                if not isinstance(row, dict):
                    continue
                identity = row.get(spec.get('id_field', 'id'), action.get('arguments', {}).get(spec.get('id_argument', 'id')))
                if identity is None:
                    continue  # Unknown identity cannot be fabricated from an audit.
                entity = spec.get('entity', api.split('.')[0] + '.Object')
                for name, field_spec in spec.get('fields', {}).items():
                    field_spec = {'kind': field_spec} if isinstance(field_spec, str) else field_spec
                    source_name = field_spec.get('source', name)
                    source_known = source_name in row
                    template = field_spec.get('key', spec.get('key_template', entity + '[{id}].{field}'))
                    key = template.format(id=identity, field=name, **{k: v for k, v in action.get('arguments', {}).items() if k not in ('id', 'field')})
                    attrs = {k: deepcopy(v) for k, v in field_spec.items() if k in ('choices', 'scale', 'tolerance', 'nullable', 'references')}
                    attrs['entity'] = entity + '[' + str(identity) + ']'
                    if fields.get(key, {}).get('nullable'):
                        attrs.setdefault('nullable', True)
                    exists_source = field_spec.get('exists_source')
                    observed_exists = bool(row[exists_source]) if exists_source and exists_source in row else True
                    fields[key] = _field(key, row.get(source_name), field_spec.get('kind'), **attrs)
                    fields[key]['known'] = source_known or not observed_exists
                    fields[key]['exists'] = observed_exists
                    if not source_known or not observed_exists:
                        fields[key]['value'] = None
                    if not _scalar_matches(fields[key]):
                        fields[key].update(known=False, value=None)
                for relation in spec.get('relations', []):
                    new_relations.append([endpoint.format(id=identity, **{k: v for k, v in action.get('arguments', {}).items() if k != 'id'}) for endpoint in relation])
    elif payload is not None:
        def flatten(value: Any, prefix: str) -> None:
            if isinstance(value, dict):
                for key, item in sorted(value.items()):
                    flatten(item, prefix + '.' + str(key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    identity = item.get('id', index) if isinstance(item, dict) else index
                    flatten(item, prefix + '[' + str(identity) + ']')
            else:
                fields[prefix] = _field(prefix, value, nullable=True) if fields.get(prefix, {}).get('nullable') else _field(prefix, value)
                if not _scalar_matches(fields[prefix]):
                    fields[prefix].update(known=False, value=None)
        flatten(payload, api + '.output')
    else:
        # Public code output is legitimate information, not a hidden-state view.
        key = api + '.output_text'
        fields[key] = _field(key, output, 'text')
    result['fields'] = list(fields.values())
    relations = result.get('relations', []) + new_relations
    for key, field in fields.items():
        if field.get('kind') == 'reference' and field.get('known') and field.get('exists') and field.get('references') and field.get('value') is not None:
            entity = str(field['references']) + '[' + str(field['value']) + ']'
            relations.extend([key, other_key] for other_key, other in fields.items() if other.get('entity') == entity)
    result['relations'] = [list(edge) for edge in sorted({tuple(edge) for edge in relations if len(edge) == 2 and all(k in fields for k in edge)})]
    validate_state(result)
    return result


def _worker_entry(connection: Any, request: dict[str, Any]) -> None:
    # The parent imports no AppWorld modules or process-global database handles.
    from .appworld_worker import worker_main
    worker_main(connection, request)


class AppWorldAdapter:
    evidence_origin = 'appworld'

    def __init__(self, config: dict[str, Any], task_id: str, split: str, experiment: str):
        self.config = normalize_config(config)
        self.env_config = self.config['environment']
        self.task_id, self.split, self.experiment = task_id, split, experiment
        self.timeout = float(self.env_config.get('worker_timeout', 300))
        self.history: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self._process = self._connection = None
        self._state: dict[str, Any] = {'fields': [], 'relations': []}
        self._budget_remaining: float | None = None
        root = self.env_config.get('root')
        if not root:
            raise ValueError('environment.root must point to an installed AppWorld data root on the server')
        if task_id not in official_task_ids(root, split):
            raise ValueError(f'{task_id} is not in official split {split}')
        if self.env_config['offline_audit'] and split not in ('train', 'dev'):
            raise ValueError('Global database audits are forbidden during test interaction')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', experiment) or '..' in experiment:
            raise ValueError('experiment must be a safe directory name')

    def _start(self, *, mode: str = 'interactive', extra: dict[str, Any] | None = None) -> dict[str, Any]:
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        request = {'config': self.config, 'task_id': self.task_id, 'split': self.split, 'experiment': self.experiment,
                   'mode': mode, **(extra or {})}
        process = context.Process(target=_worker_entry, args=(child, request), daemon=False)
        process.start()
        child.close()
        self._process, self._connection = process, parent
        return self._receive()

    def _receive(self) -> dict[str, Any]:
        if not self._connection.poll(self.timeout):
            self.close(force=True)
            raise TimeoutError(f'AppWorld worker timed out after {self.timeout}s')
        try:
            response = self._connection.recv()
        except EOFError as error:
            self.close(force=True)
            raise RuntimeError('AppWorld worker exited without a result') from error
        if not response.get('ok'):
            raise EnvironmentExecutionError(response.get('error', 'AppWorld worker failed'), response.get('charged_cost'), response.get('cost_unresolved', False))
        return response['result']

    def _request(self, command: str, **kwargs: Any) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError('Call reset() before using the environment')
        self._connection.send({'command': command, **kwargs})
        return self._receive()

    def reset(self) -> dict[str, Any]:
        self.close()
        result = self._start()
        self.metadata, self._state = result['metadata'], result['state']
        self.history = []
        return self.observe()

    def observe(self) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError('Call reset() before observe()')
        return deepcopy(self._state)

    def set_budget_remaining(self, remaining: float) -> None:
        if remaining < 0:
            raise ValueError('remaining budget must be nonnegative')
        self._budget_remaining = float(remaining)

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        validate_action(action)
        result = self._request('execute', action=action, budget_remaining=self._budget_remaining)
        self._state = deepcopy(result['state'])
        self.history.append({'action': deepcopy(action), 'output': result['observation'], 'fingerprint': result.get('fingerprint')})
        return result

    def snapshot(self) -> dict[str, Any]:
        if not self.env_config['offline_audit'] or self.split not in ('train', 'dev'):
            raise ValueError('Global snapshots are offline train/dev audit only')
        return self._request('snapshot')['state']

    def evaluate(self) -> dict[str, Any]:
        # A separate Python interpreter has no inherited CachedDBHandler state.
        return self._request('evaluate')

    def branch(self, prefix: list[dict[str, Any]], action: dict[str, Any], condition: dict[str, Any] | None = None, max_cost: float | None = None) -> dict[str, Any]:
        evaluation_only = (self.split == 'dev' and self.env_config.get('offline_audit') is True
                           and self.config.get('collection', {}).get('split') == 'dev'
                           and self.config.get('collection', {}).get('counterfactual_evaluation_only') is True)
        if self.split != 'train' and not evaluation_only:
            raise ValueError('Counterfactual replay/branch collection is train-only unless explicit dev evaluation-only mode is enabled')
        if condition:
            validate_condition(condition)
        branch = AppWorldAdapter(self.config, self.task_id, self.split, self.experiment + '_cf_' + uuid.uuid4().hex[:12])
        try:
            return branch._start(mode='branch', extra={'prefix': prefix, 'action': action, 'condition': condition, 'max_cost': max_cost})
        finally:
            branch.close()

    def close(self, force: bool = False) -> None:
        process, connection = self._process, self._connection
        self._process = self._connection = None
        if connection is not None:
            try:
                if not force and process is not None and process.is_alive():
                    connection.send({'command': 'close'})
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        if process is not None:
            process.join(2 if not force else 0)
            if process.is_alive():
                process.terminate()
                process.join(3)
            if process.is_alive():
                process.kill()
                process.join()


def create_environment(config: dict[str, Any], task_id: str, split: str, experiment: str) -> Environment:
    config = normalize_config(config)
    if split == 'test':
        split = 'test_normal'
    if split not in SPLITS:
        raise ValueError(f'Unsupported split {split}')
    backend = config['environment']['backend']
    if backend == 'fixture':
        from .fixtures import FixtureEnv
        return FixtureEnv(config, task_id, split, experiment)
    if backend == 'appworld':
        return AppWorldAdapter(config, task_id, split, experiment)
    raise ValueError(f'Unknown backend {backend}')
