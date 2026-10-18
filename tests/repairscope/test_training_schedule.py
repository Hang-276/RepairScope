"""Engineering regressions for shuffled atomic CF bundles, never benchmark evidence."""
from copy import deepcopy
import json
import math
import random

import pytest
import torch

from repairscope.fixtures import fixture_records
from repairscope.training import (_bundles, _minibatches, _training_schedule,
                                  _windows, load_predictor, resolve_config, train_model)


def config(epochs=4):
    return resolve_config({
        "model": {"d_model": 16, "n_heads": 2, "relation_layers": 1,
                  "temporal_layers": 1, "feature_dim": 8, "value_slots": 8,
                  "history": 2, "scope_budget": 4, "candidate_limit": 16,
                  "dropout": .1},
        "train": {"device": "cpu", "precision": "fp32", "epochs": epochs,
                  "micro_batch": 4, "grad_accum": 1, "horizon": 1,
                  "max_steps": None, "warmup_fraction": .2, "seed": 100},
    })


def test_schedule_counts_actual_cf_packing_without_changing_global_rng():
    # The proposed candidate has 120 natural starts and twelve atomic quartets.
    bundles = [list(range(4 * i, 4 * i + 4)) for i in range(12)]
    bundles.extend([[48 + i] for i in range(120)])
    train = {"epochs": 30, "micro_batch": 4, "grad_accum": 4, "seed": 100}
    before = random.getstate()
    schedule = _training_schedule(bundles, train)
    assert random.getstate() == before
    assert schedule == _training_schedule(bundles, train)
    assert schedule != _training_schedule(bundles, {**train, "seed": 101})
    actual = [math.ceil(len(_minibatches(order, bundles, 4)) / 4)
              for order in schedule["orders"]]
    assert schedule["updates_per_epoch"] == actual
    assert schedule["total_updates"] == sum(actual)
    # The former unshuffled estimate was 330, which cuts this schedule short.
    assert schedule["total_updates"] > 330
    assert len(set(schedule["minibatches_per_epoch"])) > 1
    for order in schedule["orders"]:
        assert sorted(order) == list(range(len(bundles)))


def test_null_step_cap_completes_all_30_epochs_with_variable_packing(tmp_path):
    fixture = fixture_records(1)
    records, cfg = fixture[:4] + fixture[5:], config(epochs=30)
    bundles = _bundles(records, _windows(records, 1))
    old_cap = len(_minibatches(list(range(len(bundles))), bundles, 4)) * 30
    result = train_model(records, cfg, tmp_path)
    schedule = json.loads((tmp_path / "training_schedule.json").read_text())
    checkpoint = torch.load(result["checkpoint"], weights_only=False)
    assert result["epochs_completed"] == 30
    assert result["updates"] == result["scheduled_updates"] == schedule["total_updates"]
    assert result["updates"] > old_cap
    assert len({entry["epoch"] for entry in checkpoint["history"]}) == 30
    assert checkpoint["training_schedule"] == schedule
    assert checkpoint["scheduler"]["last_epoch"] == result["updates"]


@pytest.mark.parametrize("stop_after", [1, 4])
def test_bounded_stop_and_resume_reproduce_full_schedule_and_dropout(tmp_path, stop_after):
    fixture = fixture_records(1)
    records, cfg = fixture[:4] + fixture[5:], config()
    full = train_model(records, cfg, tmp_path / "full")
    partial_cfg = deepcopy(cfg)
    partial_cfg["train"]["max_steps"] = stop_after
    partial = train_model(records, partial_cfg, tmp_path / "partial")
    assert partial["updates"] == stop_after
    resumed = train_model(records, cfg, tmp_path / "resumed", resume=partial["checkpoint"])
    expected = torch.load(full["checkpoint"], weights_only=False)
    actual = torch.load(resumed["checkpoint"], weights_only=False)
    assert resumed["epochs_completed"] == cfg["train"]["epochs"]
    assert actual["global_step"] == expected["global_step"]
    assert actual["training_schedule"] == expected["training_schedule"]
    assert actual["history"] == expected["history"]
    assert actual["scheduler"] == expected["scheduler"]
    for name, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], value, rtol=0, atol=0)
    assert torch.equal(actual["rng"]["torch"], expected["rng"]["torch"])


def test_old_checkpoint_remains_inference_loadable_but_exact_resume_is_explicitly_rejected(tmp_path):
    records, cfg = fixture_records(1), config()
    cfg["train"]["max_steps"] = 1
    partial = train_model(records, cfg, tmp_path / "partial")
    payload = torch.load(partial["checkpoint"], weights_only=False)
    payload.pop("training_schedule")
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    model, _, _ = load_predictor(legacy)
    assert not model.training
    with pytest.raises(ValueError, match="Legacy checkpoint.*exact resume"):
        train_model(records, cfg, tmp_path / "resumed", resume=legacy)


def test_resume_rejects_a_different_saved_schedule(tmp_path):
    records, cfg = fixture_records(1), config()
    cfg["train"]["max_steps"] = 1
    partial = train_model(records, cfg, tmp_path / "partial")
    payload = torch.load(partial["checkpoint"], weights_only=False)
    payload["training_schedule"]["orders"][0].reverse()
    tampered = tmp_path / "different_schedule.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="frozen training schedule differs"):
        train_model(records, cfg, tmp_path / "resumed", resume=tampered)
