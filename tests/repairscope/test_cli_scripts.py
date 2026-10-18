"""Exercise the actual CLI/shell boundaries, without environments or GPU work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from repairscope.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def json_stream(text):
    decoder = json.JSONDecoder()
    values = []
    rest = text.strip()
    while rest:
        value, end = decoder.raw_decode(rest)
        values.append(value)
        rest = rest[end:].lstrip()
    return values


@pytest.mark.parametrize('script,variant,loss_key,expected', [
    ('a1_dynamics.sh', 'dynamics', 'boundary', 0),
    ('a2_paired_effect.sh', 'paired_effect', 'paired', 1),
    ('a3_full.sh', 'full', 'boundary', 1),
])
def test_shell_runs_resolve_correct_ablations_without_training(script, variant, loss_key, expected, tmp_path):
    env = {**os.environ, 'DRY_RUN': '1', 'SEEDS': '100 101',
           'REPAIRSCOPE_PYTHON': sys.executable, 'RUN_ROOT': str(tmp_path / 'path with spaces')}
    completed = subprocess.run(['bash', str(ROOT / 'scripts/repairscope' / script)],
                               cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    plans = json_stream(completed.stdout)
    assert [p['command'] for p in plans] == ['train', 'evaluate', 'train', 'evaluate']
    assert [p['config']['train']['seed'] for p in plans] == [100, 100, 101, 101]
    assert all(p['config']['variant'] == variant and p['config']['loss'][loss_key] == expected for p in plans)
    assert all(p['will_execute'] is False for p in plans)
    assert not (tmp_path / 'path with spaces').exists()


@pytest.mark.parametrize('strategy', ['random', 'uncertainty', 'novelty_learnability', 'risk'])
def test_probe_wrappers_use_same_checkpoint_and_real_budget(strategy, tmp_path):
    env = {**os.environ, 'DRY_RUN': '1', 'SEEDS': '100',
           'REPAIRSCOPE_PYTHON': sys.executable, 'RUN_ROOT': str(tmp_path),
           'CHECKPOINT': '/shared/checkpoint.pt', 'PROBE_PLANS': '/shared/plans.json',
           'TRAIN_DATA': '/shared/train with spaces.jsonl'}
    result = subprocess.run(['bash', str(ROOT / f'scripts/repairscope/probe_{strategy}.sh')],
                            cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    plan, = json_stream(result.stdout)
    assert plan['checkpoint'] == '/shared/checkpoint.pt'
    assert plan['config']['probing']['strategy'] == strategy
    assert plan['config']['probing']['budget_calls'] == 2000
    assert plan['config']['probing']['plans'] == '/shared/plans.json'
    assert plan['config']['data']['train'] == '/shared/train with spaces.jsonl'
    assert plan['will_execute'] is False


def test_all_sh_files_have_valid_bash_syntax():
    scripts = list((ROOT / 'scripts/repairscope').glob('*.sh'))
    assert scripts
    for script in scripts:
        subprocess.run(['bash', '-n', str(script)], check=True, capture_output=True)


def test_variants_do_not_mutate_defaults_and_config_inheritance_cycle_fails(tmp_path):
    assert load_config(variant='no_escape')['loss']['escape'] == 0
    assert load_config()['loss']['escape'] == 1
    first, second = tmp_path / 'a.yaml', tmp_path / 'b.yaml'
    first.write_text('base: b.yaml\n')
    second.write_text('base: a.yaml\n')
    with pytest.raises(ValueError, match='cycle'):
        load_config(first)


def test_cli_refuses_heldout_data_before_any_optimizer(tmp_path):
    from repairscope.fixtures import fixture_records
    from repairscope.data import write_jsonl
    records = fixture_records(1, split='dev')
    dataset = tmp_path / 'heldout.jsonl'
    write_jsonl(dataset, records)
    result = subprocess.run([sys.executable, '-m', 'repairscope', 'train',
        '--data', str(dataset), '--set', 'data.allow_synthetic=true',
        '--output-dir', str(tmp_path / 'training')], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 2
    assert 'Only official train' in result.stderr
    assert not (tmp_path / 'training/checkpoint.pt').exists()
