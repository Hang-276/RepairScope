"""Dev CF labels require an explicit frozen evaluation route, never training."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import repairscope.collection as collection
import repairscope.environment as environment
from repairscope.appworld_worker import worker_main
from repairscope.cli import _training_input
from repairscope.data import read_jsonl
from repairscope.decision import decision_groups
from repairscope.fixtures import FixtureEnv
from repairscope.probing import ExperienceReplay


def dev_config():
    return {'environment': {'backend': 'fixture', 'offline_audit': True},
            'collection': {'split': 'dev', 'counterfactual_evaluation_only': True},
            'data': {'allow_synthetic': True, 'dev': None}}


def dev_plans():
    plans = json.loads(Path('repairscope/fixtures/example_cf_plans.json').read_text())['plans']
    for plan in plans:
        plan.update(split='dev', source_split='dev')
    return plans


def test_dev_four_cells_are_frozen_evaluation_labels_and_cannot_train(tmp_path):
    plans = dev_plans()
    before = deepcopy(plans)
    summary = collection.collect_counterfactual(dev_config(), plans, tmp_path)
    records = read_jsonl(tmp_path / 'transitions.jsonl')
    assert plans == before
    assert summary['cells'] == 4 and summary['total_cost'] == 12
    assert summary['eval_only'] and summary['frozen_query'] and summary['nonadaptive']
    assert not summary['used_for_training'] and not summary['scientific_evidence']
    assert summary['splits'] == {'dev': 4}
    assert summary['collection_role'] == 'held_out_counterfactual_evaluation'
    assert len(summary['plans_sha256']) == 64
    manifest = json.loads((tmp_path / 'evaluation_plan_manifest.json').read_text())
    assert manifest['plans_sha256'] == summary['plans_sha256']
    assert manifest['split'] == manifest['source_split'] == 'dev'
    for record in records:
        assert record['split'] == record['source_split'] == 'dev'
        assert record['frozen_query'] and record['counterfactual_evaluation_only']
        assert record['used_for_training'] is False
        assert record['prefix_verified'] and len(record['source_prefix_id']) == 64
        assert record['evaluation_plans_sha256'] == summary['plans_sha256']
    assert len({record['source_prefix_id'] for record in records}) == 1
    groups = decision_groups(records, {})
    assert groups['decision_eligible_groups'] == 2
    assert groups['decision_informative_groups'] == 1
    with pytest.raises(ValueError, match='Only official train'):
        _training_input(dev_config(), str(tmp_path / 'transitions.jsonl'))
    with pytest.raises(ValueError, match='Only training records'):
        ExperienceReplay().add(records)


@pytest.mark.parametrize('split,enabled', [('dev', False), ('test', True), ('test_normal', True),
                                          ('test_challenge', True), ('train', True)])
def test_eval_mode_cannot_open_unapproved_splits(monkeypatch, tmp_path, split, enabled):
    config = dev_config()
    config['collection'].update(split=split, counterfactual_evaluation_only=enabled)
    calls = []
    monkeypatch.setattr(collection, 'create_environment', lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        collection.collect_counterfactual(config, dev_plans(), tmp_path)
    assert calls == []


@pytest.mark.parametrize('bad_split,bad_source', [('train', 'dev'), ('dev', 'train'), ('dev', None)])
def test_entire_dev_plan_pool_is_checked_before_execution(monkeypatch, tmp_path, bad_split, bad_source):
    plans = dev_plans()
    invalid = deepcopy(plans[0])
    invalid.update(pair_id='invalid', split=bad_split, source_split=bad_source)
    calls = []
    monkeypatch.setattr(collection, 'create_environment', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='explicit split=dev and source_split=dev'):
        collection.collect_counterfactual(dev_config(), plans + [invalid], tmp_path)
    assert calls == []


def test_train_collection_keeps_original_route(tmp_path):
    plans = json.loads(Path('repairscope/fixtures/example_cf_plans.json').read_text())
    summary = collection.collect_counterfactual({'environment': {'backend': 'fixture'}}, plans, tmp_path)
    records = read_jsonl(tmp_path / 'transitions.jsonl')
    assert summary['cells'] == 4 and summary['total_cost'] == 12
    assert summary['eval_only'] is False and summary['used_for_training']
    assert all(record['split'] == 'train' and record['used_for_training'] for record in records)
    assert all(not record.get('frozen_query', False) for record in records)
    assert not (tmp_path / 'evaluation_plan_manifest.json').exists()


def test_dev_adapter_branches_preserve_official_split(monkeypatch):
    adapter_class = environment.AppWorldAdapter
    adapter = adapter_class.__new__(adapter_class)
    adapter.config = {**dev_config(), 'environment': {'backend': 'appworld', 'offline_audit': True}}
    adapter.env_config = adapter.config['environment']
    adapter.split, adapter.task_id, adapter.experiment = 'dev', 'dev-task', 'dev-evaluation'
    seen = []
    class Child:
        def __init__(self, config, task_id, split, experiment):
            seen.append(split)
        def _start(self, **kwargs):
            return {'split': seen[-1]}
        def close(self):
            pass
    monkeypatch.setattr(environment, 'AppWorldAdapter', Child)
    assert adapter_class.branch(adapter, [], {'name': 'shop.noop', 'arguments': {}}) == {'split': 'dev'}
    assert seen == ['dev']
    adapter.config['collection']['counterfactual_evaluation_only'] = False
    with pytest.raises(ValueError, match='train-only'):
        adapter_class.branch(adapter, [], {'name': 'shop.noop', 'arguments': {}})


@pytest.mark.parametrize('split,enabled,offline,passes_gate', [
    ('dev', True, True, True), ('dev', False, True, False), ('dev', True, False, False),
    ('test_normal', True, True, False), ('test_challenge', True, True, False)])
def test_worker_gate_rejects_before_appworld_import(monkeypatch, split, enabled, offline, passes_gate):
    import repairscope.appworld_worker as worker
    messages = []
    connection = SimpleNamespace(send=lambda value: messages.append(value), close=lambda: None)
    def stop_before_appworld(root):
        raise RuntimeError('gate_passed_no_appworld_execution')
    monkeypatch.setattr(worker, '_set_root', stop_before_appworld)
    config = dev_config()
    config['collection'].update(split=split, counterfactual_evaluation_only=enabled)
    config['environment'].update(root='/nonexistent-test-root', offline_audit=offline)
    worker_main(connection, {'config': config, 'mode': 'branch', 'split': split})
    assert len(messages) == 1 and messages[0]['ok'] is False
    assert ('gate_passed_no_appworld_execution' in messages[0]['error']) is passes_gate
    if not passes_gate:
        assert 'train-only' in messages[0]['error']


def test_fixture_dev_branch_requires_all_explicit_evaluation_gates():
    config = dev_config()
    config['environment']['offline_audit'] = False
    env = FixtureEnv(config, 'fixture-0', 'dev', 'dev-evaluation')
    try:
        with pytest.raises(ValueError, match='train-only'):
            env.branch([], {'name': 'shop.noop', 'arguments': {}})
    finally:
        env.close()


def test_existing_counterfactual_cli_accepts_explicit_dev_switch_without_execution():
    from repairscope.cli import parser, run
    args = parser().parse_args(['counterfactual', '--config', 'configs/repairscope/goal_sufficient.yaml',
        '--plans', '/frozen/dev_cf_plans.json', '--set', 'collection.split=dev',
        '--set', 'collection.counterfactual_evaluation_only=true', '--dry-run'])
    receipt = run(args)
    assert receipt['will_execute'] is False
    assert receipt['config']['collection']['split'] == 'dev'
    assert receipt['config']['collection']['counterfactual_evaluation_only'] is True
