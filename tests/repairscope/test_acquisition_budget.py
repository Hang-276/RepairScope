"""Hard acquisition limits survive partial failures and collection restarts.

These are fixture/transport regressions; they perform no AppWorld or HTTP calls.
"""
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import repairscope.actor as actor_module
import repairscope.collection as collection
from repairscope.acquisition_budget import AcquisitionBudgetExceeded, CollectionBudgetError
from repairscope.actor import FrozenActor
from repairscope.data import read_jsonl
from repairscope.environment import EnvironmentExecutionError
from repairscope.fixtures import FixtureEnv


def budget_config(tmp_path, *, actions=None, **limits):
    budget = {
        'ledger_path': str((tmp_path / 'acquisition_ledger.json').resolve()),
        'run_id': 'hard-budget-regression',
        'environment_limit': 10,
        'natural_step_limit': 5,
        'cf_cell_limit': 5,
        'actor_generation_limit': 10,
        'actor_http_limit': 10,
    }
    budget.update(limits)
    return {
        'environment': {'backend': 'fixture', 'offline_audit': True},
        'collection': {
            'split': 'train', 'task_ids': ['fixture-0'], 'max_steps': 2,
            'action_plans': {'actions': actions if actions is not None else [noop()]},
            'budget': budget,
        },
    }


def noop():
    return {'name': 'shop.noop', 'arguments': {}, 'cost': 1}


def two_cell_plan():
    return [{
        'task_id': 'fixture-0', 'pair_id': 'fixed-two-cell', 'split': 'train',
        'source_split': 'train', 'prefix': [], 'conditions': {'c0': None},
        'reference': noop(),
        'action': {'name': 'shop.set_shipping', 'arguments': {'shipping': 'express'}, 'cost': 1},
    }]


def read_ledger(config):
    return json.loads(Path(config['collection']['budget']['ledger_path']).read_text())


class TrackedEnv(FixtureEnv):
    def __init__(self, config, task_id, split, experiment, calls):
        self.calls_seen = calls
        super().__init__(config, task_id, split, experiment)

    def execute(self, action):
        self.calls_seen['execute'].append(self._budget_remaining)
        return super().execute(action)

    def branch(self, prefix, action, condition=None, max_cost=None):
        self.calls_seen['branch'].append(max_cost)
        return super().branch(prefix, action, condition, max_cost=max_cost)


def track_environment(monkeypatch, env_class=TrackedEnv):
    calls = {'created': [], 'execute': [], 'branch': []}

    def factory(config, task_id, split, experiment):
        calls['created'].append(task_id)
        return env_class(config, task_id, split, experiment, calls)

    monkeypatch.setattr(collection, 'create_environment', factory)
    return calls


def test_natural_and_cf_share_persistent_environment_ceiling(monkeypatch, tmp_path):
    config = budget_config(tmp_path, environment_limit=3, natural_step_limit=9, cf_cell_limit=9)
    calls = track_environment(monkeypatch)

    natural = collection.collect(config, tmp_path / 'natural')
    counterfactual = collection.collect_counterfactual(config, two_cell_plan(), tmp_path / 'cf')

    assert natural['total_cost'] == 1
    assert counterfactual['total_cost'] == 2
    assert calls['execute'] == [3]
    assert calls['branch'] == [2, 1]
    assert counterfactual['budget']['spent']['environment'] == 3
    assert read_ledger(config)['spent']['environment'] == 3

    # A new output folder is not a new allowance and cannot execute a fourth unit.
    with pytest.raises(AcquisitionBudgetExceeded):
        collection.collect(config, tmp_path / 'not_a_new_budget')
    assert calls['execute'] == [3]
    assert calls['branch'] == [2, 1]
    assert len(read_jsonl(tmp_path / 'natural' / 'transitions.jsonl')) == 1


def test_natural_step_cap_is_passed_to_environment(monkeypatch, tmp_path):
    config = budget_config(tmp_path, actions=[noop(), noop()], environment_limit=7, natural_step_limit=2)
    calls = track_environment(monkeypatch)
    summary = collection.collect(config, tmp_path / 'natural')
    assert calls['execute'] == [2, 2]
    assert summary['total_cost'] == 2
    assert summary['budget']['spent']['environment'] == 2


def test_cf_cell_cap_is_independent_of_plan_action_cost(monkeypatch, tmp_path):
    config = budget_config(tmp_path, environment_limit=10, cf_cell_limit=2)
    plans = two_cell_plan()
    plans[0]['prefix'] = [{'action': noop(), 'output': 'ok'}]
    plans[0]['conditions'] = {'c0': {'actions': [noop()]}}
    calls = track_environment(monkeypatch)
    with pytest.raises(ValueError, match='max_cost'):
        collection.collect_counterfactual(config, plans, tmp_path / 'cf')
    assert calls['branch'] == [2]
    assert read_ledger(config)['spent']['environment'] == 0
    assert not read_ledger(config)['cost_unresolved']


def test_partial_cf_failure_preserves_known_execution_cost(monkeypatch, tmp_path):
    class PartialFailure(TrackedEnv):
        def branch(self, prefix, action, condition=None, max_cost=None):
            self.calls_seen['branch'].append(max_cost)
            raise EnvironmentExecutionError('condition failed after two executed blocks', charged_cost=2)

    config = budget_config(tmp_path, environment_limit=7, cf_cell_limit=4)
    calls = track_environment(monkeypatch, PartialFailure)
    with pytest.raises(EnvironmentExecutionError, match='condition failed'):
        collection.collect_counterfactual(config, two_cell_plan(), tmp_path / 'cf')

    ledger = read_ledger(config)
    failure = json.loads((tmp_path / 'cf' / 'collection_failure.json').read_text())
    assert calls['branch'] == [4]
    assert ledger['spent']['environment'] == 2
    assert not ledger['cost_unresolved'] and not ledger['sealed']
    assert failure['known_cost'] == 2
    assert not failure['cost_unresolved']


@pytest.mark.parametrize('counterfactual', [False, True])
def test_record_conversion_failure_does_not_erase_executed_cost(monkeypatch, tmp_path, counterfactual):
    config = budget_config(tmp_path)
    calls = track_environment(monkeypatch)

    def invalid_record(*args, **kwargs):
        raise ValueError('deliberately malformed record after execution')

    monkeypatch.setattr(collection, '_record', invalid_record)
    output = tmp_path / ('cf' if counterfactual else 'natural')
    with pytest.raises(ValueError, match='malformed record'):
        if counterfactual:
            collection.collect_counterfactual(config, two_cell_plan(), output)
        else:
            collection.collect(config, output)

    ledger = read_ledger(config)
    failure = json.loads((output / 'collection_failure.json').read_text())
    assert len(calls['branch' if counterfactual else 'execute']) == 1
    assert ledger['spent']['environment'] == 1
    assert not ledger['cost_unresolved'] and not ledger['sealed']
    assert failure['known_cost'] == 1
    assert not failure['cost_unresolved']


@pytest.mark.parametrize('counterfactual', [False, True])
def test_real_collection_requires_explicit_budget_before_environment(monkeypatch, tmp_path, counterfactual):
    config = budget_config(tmp_path)
    config['environment']['backend'] = 'appworld'
    del config['collection']['budget']
    calls = track_environment(monkeypatch)
    with pytest.raises((ValueError, CollectionBudgetError), match='budget'):
        if counterfactual:
            collection.collect_counterfactual(config, two_cell_plan(), tmp_path / 'cf')
        else:
            collection.collect(config, tmp_path / 'natural')
    assert calls['created'] == []


def test_fixture_collection_without_budget_keeps_legacy_behavior(tmp_path):
    config = budget_config(tmp_path)
    del config['collection']['budget']
    summary = collection.collect(config, tmp_path / 'natural')
    assert summary['records'] == 1 and summary['total_cost'] == 1
    assert summary['scientific_evidence'] is False
    assert not (tmp_path / 'acquisition_ledger.json').exists()


def test_reset_and_observe_do_not_consume_execution_units(monkeypatch, tmp_path):
    config = budget_config(tmp_path, actions=[])
    config['collection']['max_steps'] = 0
    calls = track_environment(monkeypatch)
    summary = collection.collect(config, tmp_path / 'empty')
    assert calls['created'] == ['fixture-0'] and calls['execute'] == []
    assert summary['total_cost'] == 0
    assert summary['budget']['spent']['environment'] == 0
    assert summary['budget']['spent']['actor_generations'] == 0
    assert summary['budget']['spent']['actor_http_attempts'] == 0


def test_unknown_execution_cost_seals_ledger_across_restart(monkeypatch, tmp_path):
    class LostReceipt(TrackedEnv):
        def execute(self, action):
            self.calls_seen['execute'].append(self._budget_remaining)
            raise TimeoutError('worker receipt lost after dispatch')

    config = budget_config(tmp_path)
    calls = track_environment(monkeypatch, LostReceipt)
    with pytest.raises(TimeoutError, match='receipt lost'):
        collection.collect(config, tmp_path / 'first')
    ledger = read_ledger(config)
    assert ledger['sealed'] and ledger['cost_unresolved']
    created_before_restart = list(calls['created'])

    with pytest.raises(CollectionBudgetError):
        collection.collect(config, tmp_path / 'restart')
    assert len(calls['execute']) == 1
    assert calls['created'] == created_before_restart


def test_process_exit_pending_reservation_cannot_be_replayed(monkeypatch, tmp_path):
    config = budget_config(tmp_path)
    script = (
        'import json, os, sys\n'
        'from repairscope.acquisition_budget import CollectionBudget\n'
        'with CollectionBudget(json.loads(sys.argv[1])) as budget:\n'
        '    budget.environment_call("natural", lambda cap: os._exit(17))\n'
    )
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(Path(collection.__file__).resolve().parents[1])
    child = subprocess.run([sys.executable, '-c', script, json.dumps(config)],
                           env=environment, capture_output=True, text=True, timeout=20)
    assert child.returncode == 17, child.stderr
    assert read_ledger(config)['pending']
    calls = track_environment(monkeypatch)
    with pytest.raises(CollectionBudgetError):
        collection.collect(config, tmp_path / 'restart')
    ledger = read_ledger(config)
    assert ledger['sealed'] and ledger['cost_unresolved']
    assert calls['created'] == []


def test_actor_generation_and_http_are_separate_hard_limits(monkeypatch, tmp_path):
    config = budget_config(tmp_path, actor_generation_limit=1, actor_http_limit=0)
    config['collection'].pop('action_plans')
    calls = track_environment(monkeypatch)
    actor = FrozenActor({'backend': 'scripted', 'actions': [noop()]})
    with pytest.raises(AcquisitionBudgetExceeded):
        collection.collect(config, tmp_path / 'natural', actor=actor)
    ledger = read_ledger(config)
    assert actor.calls == 1 and actor.http_calls == 0
    assert len(calls['execute']) == 1
    assert ledger['spent']['actor_generations'] == 1
    assert ledger['spent']['actor_http_attempts'] == 0
    assert ledger['spent']['environment'] == 1


def test_actor_parse_retry_is_blocked_before_second_http_attempt(monkeypatch, tmp_path):
    config = budget_config(tmp_path, actor_generation_limit=1, actor_http_limit=1)
    config['collection'].pop('action_plans')
    config['collection']['max_steps'] = 1
    calls = track_environment(monkeypatch)
    actor = FrozenActor({'backend': 'openai', 'api_key_env': 'BUDGET_TEST_API_KEY',
                         'base_url': 'https://invalid.test/v1', 'model': 'frozen-test-policy'})
    monkeypatch.setenv('BUDGET_TEST_API_KEY', 'local-test-value')
    http = []

    def fake_urlopen(request, timeout):
        http.append(request.full_url)
        return BytesIO(json.dumps({'choices': [{'message': {'content': 'not JSON'}}]}).encode())

    monkeypatch.setattr(actor_module.urllib.request, 'urlopen', fake_urlopen)
    with pytest.raises(AcquisitionBudgetExceeded):
        collection.collect(config, tmp_path / 'natural', actor=actor)
    ledger = read_ledger(config)
    assert http == ['https://invalid.test/v1/chat/completions']
    assert calls['execute'] == []
    assert ledger['spent']['actor_generations'] == 1
    assert ledger['spent']['actor_http_attempts'] == 1
    assert ledger['spent']['environment'] == 0
    assert actor.set_http_budget_hook(None) is None


def test_actor_http_hook_tracks_requests_but_not_cache_or_script(monkeypatch):
    actor = FrozenActor({'backend': 'openai', 'api_key_env': 'BUDGET_TEST_API_KEY',
                         'base_url': 'https://invalid.test/v1', 'model': 'frozen-test-policy'})
    monkeypatch.setenv('BUDGET_TEST_API_KEY', 'local-test-value')
    hooks, http = [], []
    hook = lambda: hooks.append('http')
    assert actor.set_http_budget_hook(hook) is None

    def fake_urlopen(request, timeout):
        http.append(request.full_url)
        return BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps({'actions': [noop()]})}}]}).encode())

    monkeypatch.setattr(actor_module.urllib.request, 'urlopen', fake_urlopen)
    state = {'fields': [], 'relations': []}
    first = actor.generate(state, [], [], count=1)
    second = actor.generate(deepcopy(state), [], [], count=1)
    assert first == second
    assert hooks == ['http'] and len(http) == 1
    assert actor.calls == 2 and actor.http_calls == 1 and actor.cache_hits == 1
    assert actor.set_http_budget_hook(None) is hook

    scripted = FrozenActor({'backend': 'scripted', 'actions': [noop()]})
    assert scripted.set_http_budget_hook(hook) is None
    scripted.generate(state, [], [], count=1)
    assert hooks == ['http'] and scripted.http_calls == 0
