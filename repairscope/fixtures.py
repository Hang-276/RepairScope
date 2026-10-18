"""Small deterministic engineering fixture; never scientific benchmark evidence."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schema import field_map, semantic_diff


def fixture_state(condition: str = 'standard') -> dict[str, Any]:
    fields = [
        {'key': 'shop.Order[1].shipping', 'kind': 'category', 'value': 'standard', 'choices': ['standard', 'express'], 'entity': 'shop.Order[1]'},
        {'key': 'shop.Order[1].paid', 'kind': 'bool', 'value': True, 'entity': 'shop.Order[1]'},
        {'key': 'shop.Order[1].amount', 'kind': 'number', 'value': 20., 'scale': 100., 'entity': 'shop.Order[1]'},
        {'key': 'shop.Order[1].condition', 'kind': 'category', 'value': condition, 'choices': ['standard', 'fragile'], 'entity': 'shop.Order[1]'},
        {'key': 'shop.Order[1].customer_id', 'kind': 'reference', 'value': 'customer-1', 'entity': 'shop.Order[1]'},
        {'key': 'shop.Order[1].note', 'kind': 'text', 'value': '', 'entity': 'shop.Order[1]'},
        {'key': 'shop.Order[2].shipping', 'kind': 'category', 'value': None, 'choices': ['standard', 'express'], 'known': True, 'exists': False, 'entity': 'shop.Order[2]'},
    ]
    for field in fields:
        field.setdefault('known', True)
        field.setdefault('exists', True)
        field.setdefault('choices', [])
    return {'fields': fields, 'relations': [[fields[0]['key'], fields[3]['key']], [fields[0]['key'], fields[1]['key']]]}


def fixture_goals() -> list[dict[str, Any]]:
    return [
        {'id': 'shipping', 'field': 'shop.Order[1].shipping', 'op': 'eq', 'value': 'express', 'protected': False, 'weight': 1.},
        {'id': 'paid', 'field': 'shop.Order[1].paid', 'op': 'eq', 'value': True, 'protected': True, 'weight': 2.},
    ]


class FixtureEnv:
    evidence_origin = 'synthetic'

    def __init__(self, config: dict[str, Any], task_id: str, split: str, experiment: str):
        self.config, self.task_id, self.split, self.experiment = deepcopy(config), task_id, split, experiment
        env = config.get('environment', config)
        self.initial_condition = env.get('condition', 'standard')
        self.offline_audit = env.get('offline_audit', split == 'train')
        self.metadata = {'instruction': 'Set express shipping while preserving payment.', 'goals': fixture_goals(),
                         'read_only_actions': [{'name': 'shop.read_order', 'arguments': {}, 'cost': 1, 'read_only': True}],
                         'evidence_origin': 'synthetic', 'benchmark_evidence': False}
        self.history: list[dict[str, Any]] = []
        self.closed = False
        self._budget_remaining = None
        self.reset()

    def reset(self) -> dict[str, Any]:
        self._state = fixture_state(self.initial_condition)
        self.history = []
        self.closed = False
        return self.observe()

    def observe(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError('Environment is closed')
        return deepcopy(self._state)

    def snapshot(self) -> dict[str, Any]:
        if not self.offline_audit or self.split not in ('train', 'dev'):
            raise ValueError('Global snapshots are offline train/dev audit only')
        return self.observe()

    def set_budget_remaining(self, remaining: float) -> None:
        if remaining < 0:
            raise ValueError('remaining budget must be nonnegative')
        self._budget_remaining = float(remaining)

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._budget_remaining is not None and self._budget_remaining < float(action.get('cost', 1)):
            raise ValueError('Execution would exceed remaining budget; charged_cost=0')
        from .schema import validate_action
        validate_action(action)
        before = self.observe()
        fields = field_map(self._state)
        name, args = action['name'].split('.')[-1], action['arguments']
        error = None
        if name == 'set_shipping':
            value = args.get('shipping', 'express')
            if value not in ('standard', 'express'):
                error = 'invalid shipping'
            else:
                fields['shop.Order[1].shipping']['value'] = value
                # Conditional side effect is real execution in this fixture.
                if fields['shop.Order[1].condition']['value'] == 'fragile' and value == 'express':
                    fields['shop.Order[1].paid']['value'] = False
        elif name == 'set_shipping_preserve_payment':
            fields['shop.Order[1].shipping']['value'] = args.get('shipping', 'express')
        elif name == 'set_condition':
            value = args.get('condition')
            if value not in ('standard', 'fragile'):
                error = 'invalid condition'
            else:
                fields['shop.Order[1].condition']['value'] = value
        elif name == 'pay_order':
            fields['shop.Order[1].paid']['value'] = True
        elif name == 'set_note':
            fields['shop.Order[1].note']['value'] = str(args.get('note', ''))
        elif name == 'create_order':
            fields['shop.Order[2].shipping'].update(known=True, exists=True, value='standard')
        elif name == 'delete_order':
            fields['shop.Order[2].shipping'].update(known=True, exists=False, value=None)
        elif name not in ('read_order', 'noop'):
            error = f'unknown fixture action: {name}'
        changes = semantic_diff(before, self._state)
        output = self._state['fields'][3]['value'] if name == 'read_order' else ('error: ' + error if error else 'ok')
        self.history.append({'action': deepcopy(action), 'output': output})
        domain = list(field_map(self._state))
        return {'state': self.observe(), 'observation': output, 'valid': error is None, 'error': error, 'done': False,
                'cost': float(action.get('cost', 1)), 'api_calls': 1,
                'audit': {'complete': bool(self.offline_audit), 'field_domain': domain,
                          'changes': changes, 'model_version': 'unselected', 'domain_kind': 'fixture_closed_world'},
                'audit_state': deepcopy(self._state) if self.offline_audit else None,
                'evidence_origin': 'synthetic'}

    def branch(self, prefix: list[dict[str, Any]], action: dict[str, Any], condition: dict[str, Any] | None = None, max_cost: float | None = None) -> dict[str, Any]:
        evaluation_only = (self.split == 'dev' and self.offline_audit is True
                           and self.config.get('collection', {}).get('split') == 'dev'
                           and self.config.get('collection', {}).get('counterfactual_evaluation_only') is True)
        if self.split != 'train' and not evaluation_only:
            raise ValueError('Counterfactual branches are train-only unless explicit dev evaluation-only mode is enabled')
        other = FixtureEnv(self.config, self.task_id, self.split, self.experiment + '_branch')
        cost, verified = 0., True
        condition_receipts = []
        required = sum(float(entry.get('action', entry).get('cost', 1)) for entry in prefix) + float(action.get('cost', 1))
        if condition:
            required += sum(float(a.get('cost', 1)) for a in condition.get('actions', [])) + (float(condition['query'].get('cost', 1)) if condition.get('query') else 0)
        if max_cost is not None and required > max_cost:
            other.close()
            raise ValueError('Branch would exceed max_cost; charged_cost=0')
        try:
            for entry in prefix:
                result = other.execute(entry.get('action', entry))
                cost += result['cost']
                if 'output' in entry and result['observation'] != entry['output']:
                    raise ValueError('Prefix replay output mismatch')
            prefix_state = other.observe()
            if condition:
                for intervention in condition.get('actions', []):
                    result = other.execute(intervention)
                    cost += result['cost']
                    condition_receipts.append({'action': deepcopy(intervention), 'output': result['observation'], 'fingerprint': None, 'valid': result['valid'], 'cost': result['cost']})
                    if not result['valid']:
                        raise ValueError('Invalid condition action')
                if condition.get('query'):
                    before_query = other.observe()
                    result = other.execute(condition['query'])
                    cost += result['cost']
                    condition_receipts.append({'action': deepcopy(condition['query']), 'output': result['observation'], 'fingerprint': None, 'valid': result['valid'], 'cost': result['cost']})
                    if semantic_diff(before_query, other.observe()):
                        raise ValueError('Condition query must be read-only')
                    if result['observation'] != condition['expected_output']:
                        raise ValueError('Condition query did not prove condition')
            before = other.observe()
            result = other.execute(action)
            target_cost = result['cost']
            result.update(before_state=before, prefix_state=prefix_state, prefix_verified=verified,
                          cost=cost + target_cost, cost_breakdown={'replay_condition': cost, 'target': target_cost},
                          audit_before_state=before, condition_receipts=condition_receipts, process_isolation='independent_fixture_instance')
            return result
        finally:
            other.close()

    def evaluate(self) -> dict[str, Any]:
        from .schema import evaluate_goals
        values = evaluate_goals(self._state, self.metadata['goals'])
        return {'task_id': self.task_id, 'split': self.split, 'evidence_origin': 'synthetic', 'TGC': float(all(v is True for v in values)), 'SGC': None,
                'goal_values': values, 'benchmark_evidence': False}

    def close(self) -> None:
        self.closed = True


def fixture_records(num_tasks: int = 4, split: str = 'train') -> list[dict[str, Any]]:
    """Executed fixture trajectories plus valid conditional four-cell siblings."""
    from .schema import evaluate_goals
    records: list[dict[str, Any]] = []
    for task_index in range(num_tasks):
        task_id = f'fixture-{task_index}'
        config = {'environment': {'backend': 'fixture', 'offline_audit': True, 'condition': 'fragile' if task_index % 2 else 'standard'}}
        env = FixtureEnv(config, task_id, split, 'fixture')
        history = []
        actions = [
            {'name': 'shop.set_shipping', 'arguments': {'shipping': 'express'}, 'cost': 1},
            {'name': 'shop.pay_order', 'arguments': {}, 'cost': 1},
            {'name': 'shop.set_note', 'arguments': {'note': f'task-{task_index}'}, 'cost': 1},
            {'name': 'shop.create_order', 'arguments': {}, 'cost': 1},
            {'name': 'shop.delete_order', 'arguments': {}, 'cost': 1},
        ]
        for step, action in enumerate(actions):
            before = env.observe()
            result = env.execute(action)
            scope = [k for k in field_map(before) if k.endswith('.shipping')]
            record = {'task_id': task_id, 'source_task_id': task_id, 'group_id': task_id,
                      'split': split, 'source_split': split, 'step': step, 'trajectory_id': task_id + '-natural',
                      'evidence_origin': 'synthetic', 'state': before, 'action': action, 'next_state': result['state'],
                      'history': deepcopy(history), 'goals': env.metadata['goals'],
                      'goal_before': evaluate_goals(before, env.metadata['goals']), 'goal_after': evaluate_goals(result['state'], env.metadata['goals']),
                      'scope': scope, 'scope_model_version': 'fixture-fixed-v1', 'audit': {**result['audit'], 'model_version': 'fixture-fixed-v1'},
                      'audit_next_state': result['audit_state'], 'action_valid': result['valid'], 'used_for_training': split == 'train'}
            records.append(record)
            history.append({'state': before, 'action': deepcopy(action)})
        if split == 'train':
            for condition in ('c0', 'c1'):
                for role in ('action', 'reference'):
                    branch_env = FixtureEnv({'environment': {'backend': 'fixture', 'condition': 'standard' if condition == 'c0' else 'fragile', 'offline_audit': True}}, task_id, split, 'cf')
                    before = branch_env.observe()
                    action = {'name': 'shop.set_shipping' if role == 'action' else 'shop.set_shipping_preserve_payment', 'arguments': {'shipping': 'express'}, 'cost': 1}
                    result = branch_env.execute(action)
                    records.append({'task_id': task_id, 'source_task_id': task_id, 'group_id': task_id,
                                    'split': split, 'source_split': split, 'step': 0, 'trajectory_id': task_id + '-cf',
                                    'evidence_origin': 'synthetic', 'state': before, 'action': action, 'next_state': result['state'], 'history': [],
                                    'goals': branch_env.metadata['goals'], 'goal_before': evaluate_goals(before, branch_env.metadata['goals']),
                                    'goal_after': evaluate_goals(result['state'], branch_env.metadata['goals']), 'scope': ['shop.Order[1].shipping'],
                                    'scope_model_version': 'fixture-fixed-v1', 'audit': {**result['audit'], 'model_version': 'fixture-fixed-v1'},
                                    'audit_next_state': result['audit_state'], 'action_valid': True, 'used_for_training': True,
                                    'cf_pair_id': task_id + '-conditional-shipping', 'cf_condition': condition, 'cf_role': role})
                    branch_env.close()
        env.close()
    return records
