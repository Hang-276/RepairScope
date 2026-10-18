from copy import deepcopy
import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from repairscope.schema import (cf_four_grid_targets, effective_protected, escape_label,
                               evaluate_goals, field_map, relabel_scope, semantic_diff,
                               validate_state, validate_transition)
from repairscope.data import grouped_partition, make_windows, read_jsonl, validate_dataset, write_jsonl
from repairscope.fixtures import FixtureEnv, fixture_records, fixture_state
from repairscope.environment import action_code, create_environment, official_task_ids, public_state_update, validate_condition
from repairscope.collection import collect, collect_counterfactual, extract_prefix
from repairscope.appworld_worker import align_audit_states, snapshot_world


def test_fixture_is_explicit_synthetic_and_executed():
    records = fixture_records()
    assert validate_dataset(records, allow_synthetic=True)['records'] == 36
    with pytest.raises(ValueError, match='Synthetic'):
        validate_dataset(records)
    assert 'appworld' not in sys.modules
    fragile = records[9]
    assert fragile['audit']['changes'] == ['shop.Order[1].paid', 'shop.Order[1].shipping']


def test_known_null_unknown_tombstone_and_types():
    state = fixture_state()
    f = state['fields'][0]
    f.update(known=False, value=None)
    validate_state(state)
    f['value'] = 'express'
    with pytest.raises(ValueError, match='null'):
        validate_state(state)
    f.update(known=True, exists=False, value=None)
    validate_state(state)
    f.update(exists=True, kind='number', value=True)
    with pytest.raises(ValueError, match='finite'):
        validate_state(state)
    f.update(value=None, nullable=True)
    validate_state(state)


def test_semantic_create_delete_unknown_missing():
    before = fixture_state()
    after = deepcopy(before)
    after['fields'][-1].update(exists=True, value='standard')
    assert semantic_diff(before, after) == ['shop.Order[2].shipping']
    assert semantic_diff(after, before) == ['shop.Order[2].shipping']
    after['fields'][-1].update(known=False, value=None)
    assert semantic_diff(before, after) == []
    after['fields'] = after['fields'][:-1]
    assert semantic_diff(before, after) == []


def test_partial_audit_no_negative_and_scope_history_immutable():
    record = fixture_records()[0]
    old = deepcopy(record)
    record['audit']['complete'] = False
    record['scope'] = list(record['audit']['field_domain'])
    assert escape_label(record) is None
    record['scope'] = []
    assert escape_label(record) is True
    with pytest.raises(ValueError, match='complete'):
        relabel_scope(record, [], 'new')
    new = relabel_scope(old, [], 'new')
    assert old['audit']['model_version'] == 'fixture-fixed-v1'
    assert new['previous_scope'] == old['scope']
    assert escape_label(new) is True
    new['scope_model_version'] = 'bad'
    with pytest.raises(ValueError, match='drift'):
        validate_transition(new)


def test_goals_typed_and_protection_intervals():
    state = fixture_state()
    goals = [{'id': 'paid', 'field': 'shop.Order[1].paid', 'op': 'eq', 'value': True,
              'protected': True, 'protect_from_step': 2, 'protect_until_step': 4}]
    assert evaluate_goals(state, goals) == [True]
    assert not effective_protected(goals[0], 1)
    assert effective_protected(goals[0], 2)
    assert not effective_protected(goals[0], 4)
    goals[0]['value'] = 1
    assert evaluate_goals(state, goals) == [False]
    state['fields'][1].update(known=False, value=None)
    assert evaluate_goals(state, goals) == [None]


def test_four_grid_type_support_and_masks():
    cells = fixture_records(1)[5:9]
    targets = cf_four_grid_targets(cells)
    paid = targets['keys'].index('shop.Order[1].paid')
    assert targets['y0'][paid] == 0 and targets['y1'][paid] == 1 and targets['mask'][paid]
    assert not any(cf_four_grid_targets(cells[:-1])['mask'])
    cells[-1]['action_valid'] = False
    assert not any(cf_four_grid_targets(cells)['mask'])


def test_split_isolation_and_trajectory_windows():
    records = fixture_records(1)
    assert [len(make_windows(records, h)) for h in (1, 2, 4)] == [9, 4, 2]
    leak = deepcopy(records[0])
    leak['split'] = 'dev'
    with pytest.raises(ValueError, match='crosses splits'):
        validate_dataset([records[0], leak], allow_synthetic=True)
    cf = deepcopy(records[5])
    cf['group_id'] = 'other'
    with pytest.raises(ValueError, match='siblings'):
        validate_dataset([records[5], cf], allow_synthetic=True)
    broken = deepcopy(records[:5])
    broken[1]['state']['fields'][0]['value'] = 'standard'
    assert len(make_windows(broken, 4)) < 2
    buckets = grouped_partition(records, .5, 10)
    assert len(buckets) == 1


def test_jsonl_round_trip(tmp_path):
    records = fixture_records(1)
    path = tmp_path / 'records.jsonl'
    write_jsonl(path, records)
    assert read_jsonl(path) == records
    path.write_text('{broken}\n')
    with pytest.raises(ValueError, match=':1:'):
        read_jsonl(path)


def test_fixture_public_api_and_test_branch_block():
    env = create_environment({'environment': {'backend': 'fixture'}}, 'fixture-0', 'train', 'test')
    result = env.execute({'name': 'shop.set_shipping', 'arguments': {'shipping': 'express'}, 'cost': 1})
    assert result['valid'] and result['evidence_origin'] == 'synthetic'
    env.close()
    test = create_environment({'environment': {'backend': 'fixture'}}, 'fixture-0', 'test', 'test')
    with pytest.raises(ValueError, match='train-only'):
        test.branch([], {'name': 'shop.noop', 'arguments': {}})
    with pytest.raises(ValueError, match='offline'):
        test.snapshot()


def test_fixture_same_prefix_four_cells_and_full_cost(tmp_path):
    plans = json.loads(open('repairscope/fixtures/example_cf_plans.json').read())
    summary = collect_counterfactual({'environment': {'backend': 'fixture'}}, plans, tmp_path)
    assert summary['cells'] == 4 and summary['total_cost'] == 12
    targets = cf_four_grid_targets(read_jsonl(tmp_path / 'transitions.jsonl'))
    index = targets['keys'].index('shop.Order[1].paid')
    assert targets['y1'][index] == 1
    assert targets['y0'][index] == 0


def test_natural_collection_script_is_real_execution(tmp_path):
    config = {'environment': {'backend': 'fixture'}, 'collection': {'task_ids': ['fixture-0'], 'action_plans': {'actions': [
        {'name': 'shop.set_shipping', 'arguments': {'shipping': 'express'}}, {'name': 'shop.pay_order', 'arguments': {}}]}}}
    summary = collect(config, tmp_path)
    records = read_jsonl(tmp_path / 'transitions.jsonl')
    assert summary['records'] == 2
    assert extract_prefix(records, 2)[0]['output'] == 'ok'
    assert records[1]['state'] == records[0]['next_state']


def test_condition_gate_requires_public_read_query():
    condition = {'actions': [{'name': 'shop.set_condition', 'arguments': {'condition': 'fragile'}}],
                 'query': {'name': 'shop.read_order', 'arguments': {}}, 'expected_output': 'fragile'}
    validate_condition(condition)
    condition['query']['name'] = 'shop.delete_order'
    with pytest.raises(ValueError, match='read API'):
        validate_condition(condition)
    condition['query'] = {'name': 'evil', 'arguments': {}, 'code': 'print(open("private"))'}
    with pytest.raises(ValueError, match='only public'):
        validate_condition(condition)


def test_public_state_mapping_does_not_fill_missing_values():
    mapping = [{'api': 'shop.read_order', 'path': 'orders.*', 'entity': 'shop.orders', 'id_field': 'id',
                'fields': {'paid': {'kind': 'bool'}, 'note': {'kind': 'text'}}}]
    state = public_state_update({'fields': [], 'relations': []}, {'name': 'shop.read_order', 'arguments': {}},
                               '{"orders":[{"id":1,"paid":true}]}', mapping)
    assert list(field_map(state)) == ['shop.orders[1].paid', 'shop.orders[1].note']
    assert field_map(state)['shop.orders[1].note']['value'] is None
    assert not field_map(state)['shop.orders[1].note']['known']
    assert not any('audit' in f['key'] for f in state['fields'])


def test_code_granularity_and_official_membership(tmp_path):
    action = {'name': 'amazon.show_product', 'arguments': {'product_id': 1}}
    assert 'apis.amazon.show_product' in action_code(action, granularity='api')
    action['code'] = 'for x in range(3):\n    apis.amazon.show_product(product_id=x)'
    with pytest.raises(ValueError, match='exactly one'):
        action_code(action, granularity='api')
    (tmp_path / 'data/datasets').mkdir(parents=True)
    (tmp_path / 'data/datasets/train.txt').write_text('task-1:tag\ntask-2\n')
    assert official_task_ids(tmp_path, 'train') == ['task-1', 'task-2']


def test_global_sqlite_audit_create_delete_and_null_are_offline():
    connection = sqlite3.connect(':memory:')
    connection.execute('CREATE TABLE orders (id INTEGER PRIMARY KEY, paid BOOLEAN, note TEXT, record_hash TEXT)')
    connection.execute('INSERT INTO orders VALUES (1,1,NULL,"a")')
    world = SimpleNamespace(models={'shop': SimpleNamespace(SQLModel=SimpleNamespace(db=SimpleNamespace(connection=connection)))})
    before = snapshot_world(world)
    assert 'shop.orders[1].record_hash' not in field_map(before)
    assert field_map(before)['shop.orders[1].note']['value'] is None
    connection.execute('DELETE FROM orders')
    connection.execute('INSERT INTO orders VALUES (2,0,"new","b")')
    left, right = align_audit_states(before, snapshot_world(world))
    changed = semantic_diff(left, right)
    assert 'shop.orders[1].paid' in changed and 'shop.orders[2].paid' in changed
    assert not field_map(right)['shop.orders[1].paid']['exists']
    connection.close()


def test_atomic_environment_budget_caps_before_api_side_effects():
    from repairscope.appworld_worker import _execute
    class Tracker:
        def __init__(self):
            self.requests = []
            self.max_num_requests = 1000
    tracker = Tracker()
    class World:
        requester = SimpleNamespace(request_tracker=tracker)
        def execute(self, code):
            # Mirrors the verified upstream RequestTracker.add_request ordering:
            # call limit is enforced before appending or sending a request.
            since_reset = 0
            for _ in range(3):
                if since_reset >= tracker.max_num_requests:
                    return 'Execution failed. Traceback:\nNumRequestsLimitError'
                tracker.requests.append({'method': 'post'})
                since_reset += 1
            return '{}'
        def task_completed(self):
            return False
    result = _execute(World(), {'name': 'shop.change', 'arguments': {}}, {'fields': [], 'relations': []}, {}, False, 2)
    assert result['cost'] == 2 and result['api_calls'] == 1 and not result['valid']
    assert tracker.max_num_requests == 1000


def test_fixture_branch_budget_never_executes_an_unaffordable_cell():
    env = FixtureEnv({'environment': {'backend': 'fixture'}}, 'fixture-0', 'train', 'fixture')
    action = {'name': 'shop.set_shipping', 'arguments': {'shipping': 'express'}, 'cost': 1}
    before = env.observe()
    with pytest.raises(ValueError, match='charged_cost=0'):
        env.branch([], action, max_cost=.5)
    assert env.observe() == before
    env.set_budget_remaining(.5)
    with pytest.raises(ValueError, match='charged_cost=0'):
        env.execute(action)
    assert env.observe() == before


def test_fixture_high_risk_performs_explicitly_legal_check():
    from repairscope.actor import create_actor
    from repairscope.control import RepairController
    env = FixtureEnv({'environment': {'backend': 'fixture'}}, 'fixture-0', 'train', 'high-risk')
    assert env.metadata['read_only_actions'][0]['read_only'] is True
    actor = create_actor({'actor': {'kind': 'fixture'}})
    predictor = lambda record: {'next_state': deepcopy(record['state']), 'scope': [], 'escape': 1., 'uncertainty': 0.}
    result = RepairController(actor, predictor, {'max_steps': 1, 'max_calls': 30, 'risk_threshold': .5,
                              'repair': False, 'candidate_count': 2}).run(env)
    assert result['checks'] == 1
    assert result['logs'][0]['phase'] == 'check'
    assert result['logs'][0]['action']['name'] == 'shop.read_order'
    assert result['errors'] == []
    env.close()


def test_dev_collection_is_held_out_and_not_training(tmp_path):
    config = {'environment': {'backend': 'fixture'}, 'collection': {'split': 'dev', 'task_ids': ['fixture-dev'],
              'action_plans': {'actions': [{'name': 'shop.set_shipping', 'arguments': {'shipping': 'express'}}]}}}
    summary = collect(config, tmp_path)
    records = read_jsonl(tmp_path / 'transitions.jsonl')
    assert records[0]['split'] == records[0]['source_split'] == 'dev'
    assert records[0]['used_for_training'] is False
    assert summary['used_for_training'] is False
    assert summary['collection_role'] == 'held_out_recorded_evaluation'


def test_sqlite_nullable_numbers_keep_declared_type():
    connection = sqlite3.connect(':memory:')
    connection.execute('CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL NULL)')
    connection.execute('INSERT INTO orders VALUES (1,NULL)')
    world = SimpleNamespace(models={'shop': SimpleNamespace(SQLModel=SimpleNamespace(db=SimpleNamespace(connection=connection)))})
    before = snapshot_world(world)
    amount = field_map(before)['shop.orders[1].amount']
    assert amount['kind'] == 'number' and amount['known'] and amount['nullable'] and amount['value'] is None
    connection.execute('UPDATE orders SET amount=3')
    after = snapshot_world(world)
    assert field_map(after)['shop.orders[1].amount']['kind'] == 'number'
    assert semantic_diff(before, after) == ['shop.orders[1].amount']
    connection.close()
