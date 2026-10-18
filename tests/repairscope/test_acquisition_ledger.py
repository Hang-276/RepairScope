"""No-resource regressions for persistent acquisition hard caps."""
import json
from types import SimpleNamespace

import pytest

from repairscope.acquisition_budget import (
    AcquisitionBudgetExceeded, CollectionBudget, CollectionBudgetError,
)


def _config(tmp_path, **limits):
    return {'environment': {'backend': 'fixture'}, 'collection': {'budget': {
        'ledger_path': str(tmp_path / 'acquisition.json'), 'run_id': 'bounded-run',
        'environment_limit': 12, 'natural_step_limit': 4, 'cf_cell_limit': 8,
        'actor_generation_limit': 3, 'actor_http_limit': 4, **limits,
    }}}


def _disk(config):
    return json.loads(open(config['collection']['budget']['ledger_path']).read())


def _raise(error):
    raise error


class _Actor:
    def __init__(self):
        self.hook = None
        self.sent = 0

    def set_http_budget_hook(self, hook):
        previous, self.hook = self.hook, hook
        return previous

    def generate(self, attempts=1):
        for _ in range(attempts):
            if self.hook is not None:
                self.hook()
            self.sent += 1
        return ['candidate']


def test_appworld_requires_explicit_budget_and_fixture_legacy_is_uncapped():
    with pytest.raises(CollectionBudgetError, match='requires explicit'):
        CollectionBudget({'environment': {'backend': 'appworld'}})
    with CollectionBudget({'environment': {'backend': 'fixture'}}) as budget:
        assert budget.active is False
        assert budget.environment_cap('natural') is None
        assert budget.environment_call('cf', lambda cap: {'cost': 19}) == {'cost': 19}
        assert budget.run_cost == 19
        assert budget.snapshot()['spent']['environment'] == 19
        assert budget.snapshot()['remaining']['environment'] is None


@pytest.mark.parametrize('missing', ['ledger_path', 'run_id', 'environment_limit',
                                    'natural_step_limit', 'cf_cell_limit',
                                    'actor_generation_limit', 'actor_http_limit'])
def test_every_budget_field_must_be_explicit(tmp_path, missing):
    config = _config(tmp_path)
    del config['collection']['budget'][missing]
    with pytest.raises(CollectionBudgetError, match=missing):
        CollectionBudget(config)


@pytest.mark.parametrize(('field', 'value'), [
    ('environment_limit', float('inf')), ('environment_limit', -1),
    ('natural_step_limit', True), ('cf_cell_limit', '8'),
    ('actor_generation_limit', 3.0), ('actor_http_limit', False),
    ('ledger_path', 'relative.json'), ('run_id', ''),
])
def test_invalid_explicit_budget_is_rejected(tmp_path, field, value):
    with pytest.raises(CollectionBudgetError):
        CollectionBudget(_config(tmp_path, **{field: value}))


def test_caps_and_pending_receipt_precede_side_effects(tmp_path):
    config = _config(tmp_path, environment_limit=10)
    with CollectionBudget(config) as budget:
        def execute(cap):
            receipt = _disk(config)
            assert receipt['pending'] == {'event_id': 1, 'kind': 'natural', 'cap': 4}
            assert receipt['events'][0]['status'] == 'pending'
            assert receipt['spent']['environment'] == 0
            assert cap == 4
            return {'cost': 3, 'state': 'private observation is not a ledger field'}
        budget.environment_call('natural', execute)
        assert budget.environment_cap('cf') == 7
        assert budget.run_cost == 3
    receipt = _disk(config)
    assert receipt['pending'] is None
    assert receipt['spent']['environment'] == 3
    assert 'private observation' not in json.dumps(receipt)


def test_new_phase_loads_spend_without_reset_and_run_cost_is_local(tmp_path):
    config = _config(tmp_path, environment_limit=7)
    with CollectionBudget(config) as natural:
        natural.environment_call('natural', lambda cap: {'cost': 4})
    with CollectionBudget(config) as cf:
        assert cf.run_cost == 0
        assert cf.environment_cap('cf') == 3
        cf.environment_call('cf', lambda cap: {'cost': cap})
        assert cf.run_cost == 3
        assert cf.snapshot()['spent']['environment'] == 7
        with pytest.raises(AcquisitionBudgetExceeded):
            cf.environment_call('natural', lambda cap: pytest.fail('unaffordable call executed'))
    assert len(_disk(config)['events']) == 2


def test_preflight_is_read_only_and_rejects_fractional_or_zero_caps(tmp_path):
    config = _config(tmp_path, natural_step_limit=.5)
    with CollectionBudget(config) as budget:
        before = budget.snapshot()
        with pytest.raises(AcquisitionBudgetExceeded):
            budget.environment_cap('natural')
        assert budget.snapshot() == before
        assert budget.environment_cap('cf') == 8
        assert budget.snapshot() == before
        with pytest.raises(ValueError, match='natural or cf'):
            budget.environment_cap('typo')


@pytest.mark.parametrize('changed', [{'run_id': 'new-run'}, {'environment_limit': 13},
                                    {'cf_cell_limit': 9}, {'actor_http_limit': 5}])
def test_existing_run_identity_and_limits_cannot_be_reset(tmp_path, changed):
    config = _config(tmp_path)
    with CollectionBudget(config) as budget:
        budget.environment_call('natural', lambda cap: {'cost': 2})
    before = _disk(config)
    with pytest.raises(CollectionBudgetError, match='immutable'):
        with CollectionBudget(_config(tmp_path, **changed)):
            pytest.fail('different allowance reused existing ledger')
    assert _disk(config) == before


def test_corrupt_ledger_is_not_recreated(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / 'acquisition.json'
    path.write_text('{interrupted/corrupt')
    with pytest.raises(CollectionBudgetError, match='refusing to reset'):
        with CollectionBudget(config):
            pytest.fail('corrupt ledger accepted')
    assert path.read_text() == '{interrupted/corrupt'


def test_one_command_holds_nonblocking_process_lock(tmp_path):
    config = _config(tmp_path)
    with CollectionBudget(config):
        with pytest.raises(CollectionBudgetError, match='already in use'):
            with CollectionBudget(config):
                pytest.fail('concurrent acquisition obtained lock')
    with CollectionBudget(config) as reopened:
        assert reopened.snapshot()['spent']['environment'] == 0


def test_path_aliases_use_the_same_ledger_and_lock(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    config = _config(real)
    with CollectionBudget(config) as budget:
        budget.environment_call('natural', lambda cap: {'cost': 2})
        with pytest.raises(CollectionBudgetError, match='already in use'):
            with CollectionBudget(_config(alias)):
                pytest.fail('path alias bypassed the ledger lock')
    with CollectionBudget(_config(alias)) as reopened:
        assert reopened.snapshot()['spent']['environment'] == 2
        assert reopened.snapshot()['ledger_path'] == str(real / 'acquisition.json')


def test_known_failed_cost_is_settled_once_and_can_be_read_after_failure(tmp_path):
    config = _config(tmp_path)
    error = RuntimeError('opaque failure')
    error.charged_cost = 3
    budget = CollectionBudget(config)
    with pytest.raises(RuntimeError, match='opaque failure'):
        with budget:
            budget.environment_call('cf', lambda cap: _raise(error))
    assert budget.run_cost == 3
    assert budget.cost_unresolved is False
    assert _disk(config)['spent']['environment'] == 3
    assert _disk(config)['events'][0]['status'] == 'failed'
    with CollectionBudget(config) as reopened:
        assert reopened.environment_cap('cf') == 8


def test_fixture_zero_cost_preexecution_refusal_is_recognized(tmp_path):
    config = _config(tmp_path)
    with CollectionBudget(config) as budget:
        with pytest.raises(ValueError, match='charged_cost=0'):
            budget.environment_call('cf', lambda cap: _raise(ValueError('Branch would exceed max_cost; charged_cost=0')))
        assert budget.run_cost == 0
        assert budget.cost_unresolved is False
        assert budget.snapshot()['events'][0]['charged_cost'] == 0


@pytest.mark.parametrize('error', [
    TimeoutError('no receipt'), RuntimeError('charged_cost=0junk'),
    RuntimeError('charged_cost=0; cost_unresolved=true'),
])
def test_unknown_cost_is_never_mistaken_for_zero_and_blocks_restart(tmp_path, error):
    config = _config(tmp_path)
    budget = CollectionBudget(config)
    with pytest.raises(type(error)):
        with budget:
            budget.environment_call('cf', lambda cap: _raise(error))
    assert budget.cost_unresolved is True
    event = _disk(config)['events'][0]
    assert event['status'] == 'unresolved'
    assert 'charged_cost' not in event
    assert 'known_cost_lower_bound' not in event
    with pytest.raises(CollectionBudgetError, match='sealed'):
        with CollectionBudget(config):
            pytest.fail('unknown operation was automatically resumed')


def test_unresolved_error_preserves_positive_known_lower_bound_without_resolving(tmp_path):
    config = _config(tmp_path)
    error = RuntimeError('failed after prefix')
    error.charged_cost, error.cost_unresolved = 3, True
    budget = CollectionBudget(config)
    with pytest.raises(RuntimeError):
        with budget:
            budget.environment_call('cf', lambda cap: _raise(error))
    assert budget.cost_unresolved is True
    assert budget.run_cost == 3
    event = _disk(config)['events'][0]
    assert event['known_cost_lower_bound'] == 3
    assert 'charged_cost' not in event
    assert _disk(config)['spent']['environment'] == 3
    with pytest.raises(CollectionBudgetError):
        with CollectionBudget(config):
            pass
    assert _disk(config)['spent']['environment'] == 3


@pytest.mark.parametrize('result', [{}, {'cost': float('nan')}, {'cost': True}, {'cost': -1}])
def test_missing_or_invalid_actual_cost_seals_instead_of_estimating(tmp_path, result):
    config = _config(tmp_path)
    budget = CollectionBudget(config)
    with pytest.raises(CollectionBudgetError, match='finite actual cost'):
        with budget:
            budget.environment_call('natural', lambda cap: result)
    assert budget.cost_unresolved is True
    assert 'charged_cost' not in _disk(config)['events'][0]


def test_result_with_explicit_unknown_flag_cannot_claim_zero_actual_cost(tmp_path):
    config = _config(tmp_path)
    budget = CollectionBudget(config)
    with pytest.raises(CollectionBudgetError):
        with budget:
            budget.environment_call('natural', lambda cap: {'cost': 0, 'cost_unresolved': True})
    assert budget.cost_unresolved
    assert 'charged_cost' not in _disk(config)['events'][0]


def test_backend_overrun_keeps_actual_cost_and_permanently_seals(tmp_path):
    config = _config(tmp_path, environment_limit=3, natural_step_limit=2)
    budget = CollectionBudget(config)
    with pytest.raises(AcquisitionBudgetExceeded, match='actual cost recorded'):
        with budget:
            budget.environment_call('natural', lambda cap: {'cost': 4})
    assert budget.run_cost == 4
    assert budget.cost_unresolved is False
    assert _disk(config)['spent']['environment'] == 4
    assert _disk(config)['seal_reason'] == 'environment_cost_overrun'
    with pytest.raises(CollectionBudgetError, match='sealed'):
        with CollectionBudget(config):
            pass


def test_loaded_pending_is_sealed_without_charging_reservation_as_actual(tmp_path):
    config = _config(tmp_path)
    with CollectionBudget(config):
        pass
    disk = _disk(config)
    disk['pending'] = {'event_id': 1, 'kind': 'cf', 'cap': 8}
    (tmp_path / 'acquisition.json').write_text(json.dumps(disk))
    budget = CollectionBudget(config)
    with pytest.raises(CollectionBudgetError, match='sealed'):
        with budget:
            pass
    assert budget.cost_unresolved is True
    assert budget.run_cost == 0
    assert _disk(config)['spent']['environment'] == 0
    assert _disk(config)['pending']['cap'] == 8


def test_actor_counts_generations_separately_and_restores_existing_hook(tmp_path):
    config = _config(tmp_path)
    actor = _Actor()
    old_attempts = []
    previous = actor.hook = lambda: old_attempts.append('attempt')
    with CollectionBudget(config) as budget:
        assert budget.actor_generate(actor, lambda: actor.generate(attempts=2)) == ['candidate']
        assert actor.hook is previous
        assert old_attempts == ['attempt', 'attempt']
        assert budget.snapshot()['spent'] == {
            'environment': 0, 'actor_generations': 1, 'actor_http_attempts': 2}
        assert budget.actor_generate(actor, lambda: ['cache hit']) == ['cache hit']
        assert budget.snapshot()['spent']['actor_generations'] == 2
        assert budget.snapshot()['spent']['actor_http_attempts'] == 2
    with CollectionBudget(config) as resumed:
        resumed.actor_generate(actor, lambda: ['scripted'])
        with pytest.raises(AcquisitionBudgetExceeded, match='actor_generations'):
            resumed.actor_generate(actor, lambda: pytest.fail('over-budget generation executed'))
        assert actor.hook is previous
        assert resumed.snapshot()['spent']['actor_generations'] == 3


def test_previous_actor_hook_can_still_block_http_and_is_restored(tmp_path):
    config = _config(tmp_path)
    actor = _Actor()
    previous = actor.hook = lambda: _raise(RuntimeError('previous allowance exhausted'))
    with CollectionBudget(config) as budget:
        with pytest.raises(RuntimeError, match='previous allowance'):
            budget.actor_generate(actor, actor.generate)
        assert budget.snapshot()['spent']['actor_generations'] == 1
        assert budget.snapshot()['spent']['actor_http_attempts'] == 0
        assert actor.hook is previous
        assert actor.sent == 0


def test_actor_http_retry_cannot_send_past_its_cap_and_exception_restores_hook(tmp_path):
    config = _config(tmp_path, actor_http_limit=1)
    actor = _Actor()
    budget = CollectionBudget(config)
    with pytest.raises(AcquisitionBudgetExceeded, match='actor_http_attempts'):
        with budget:
            budget.actor_generate(actor, lambda: actor.generate(attempts=3))
    assert actor.sent == 1
    assert actor.hook is None
    assert _disk(config)['spent']['actor_http_attempts'] == 1
    assert _disk(config)['spent']['actor_generations'] == 1


def test_http_failure_is_already_charged_without_recording_error_secrets(tmp_path):
    config = _config(tmp_path)
    actor = _Actor()
    def failed_request():
        actor.hook()
        assert _disk(config)['spent']['actor_http_attempts'] == 1
        raise RuntimeError('secret token and private content')
    with CollectionBudget(config) as budget:
        with pytest.raises(RuntimeError, match='secret token'):
            budget.actor_generate(actor, failed_request)
    assert actor.hook is None
    assert 'secret token' not in json.dumps(_disk(config))


def test_active_budget_rejects_uncontrolled_actor_without_charging_generation(tmp_path):
    config = _config(tmp_path)
    with CollectionBudget(config) as budget:
        with pytest.raises(CollectionBudgetError, match='set_http_budget_hook'):
            budget.actor_generate(SimpleNamespace(last_http_calls=0), lambda: pytest.fail('uncontrolled Actor called'))
        assert budget.snapshot()['spent']['actor_generations'] == 0


def test_budget_methods_require_context_and_snapshot_is_independent(tmp_path):
    budget = CollectionBudget(_config(tmp_path))
    with pytest.raises(CollectionBudgetError, match='context manager'):
        budget.environment_cap('cf')
    with budget:
        snapshot = budget.snapshot()
        snapshot['spent']['environment'] = 999
        snapshot['events'].append({'secret': 'not persistent'})
        assert budget.snapshot()['spent']['environment'] == 0
        assert budget.snapshot()['events'] == []
