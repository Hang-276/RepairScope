"""Resolved v2 comparisons share budgets/data and never reuse v1.1 output labels."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from repairscope.config import DEFAULTS, load_config

ROOT = Path(__file__).resolve().parents[2]
V2_ABLATIONS = ("no_goal_sufficiency", "v2_no_boundary",
                "no_goal_relevant_escape", "v2_static_scope")


def _stream(payload):
    decoder, values = json.JSONDecoder(), []
    rest = payload.strip()
    while rest:
        value, end = decoder.raw_decode(rest)
        values.append(value)
        rest = rest[end:].lstrip()
    return values


def _environment(tmp_path):
    env = {key: value for key, value in os.environ.items() if key not in
           {"SEEDS", "RUN_ROOT", "CONFIG", "CHECKPOINT", "CALIBRATION_FILE",
            "PROBE_PLANS", "TASK_FILE", "RS_PROBE_LABEL"}}
    env.update(DRY_RUN="1", REPAIRSCOPE_PYTHON=sys.executable,
               PYTHONPATH=str(ROOT), RUN_ROOT=str(tmp_path / "repairscope_v2_seed100"),
               TRAIN_DATA="/shared/train with spaces.jsonl",
               EVAL_DATA="/shared/heldout dev.jsonl")
    return env


def _script(name, env, *arguments):
    return subprocess.run(["bash", str(ROOT / "scripts/repairscope" / name), *arguments],
                          env=env, cwd=ROOT, text=True, capture_output=True)


def test_v2_variants_keep_model_capacity_and_fixed_training_budget():
    variants = ("actor_only", "dynamics", "paired_effect", "full",
                "goal_sufficient", *V2_ABLATIONS)
    configs = [load_config(variant=variant) for variant in variants]
    comparable = lambda model: {k: v for k, v in model.items()
                                if k not in {"goal_sufficient", "static_scope"}}
    for config in configs:
        assert comparable(config["model"]) == comparable(configs[0]["model"])
        assert config["train"] == configs[0]["train"]
        assert config["actor"] == configs[0]["actor"]
        assert config["controller"]["candidate_count"] == 4
        assert config["controller"]["max_calls"] == 100
        assert config["model"]["scope_budget"] == 128
        assert config["train"]["horizon"] == 4
        assert config["train"]["epochs"] == 30
        assert config["probing"]["budget_calls"] == 2000
    assert all(not config["model"]["goal_sufficient"] for config in configs[:4])
    assert all(config["model"]["goal_sufficient"] for config in configs[4:])
    assert configs[4]["loss"]["goal_sufficiency"] == 1
    assert configs[4]["loss"]["goal_relevant_escape"] == 1
    assert load_config(variant="no_goal_sufficiency")["loss"]["goal_sufficiency"] == 0
    assert load_config(variant="v2_no_boundary")["loss"]["boundary"] == 0
    assert load_config(variant="no_goal_relevant_escape")["loss"]["goal_relevant_escape"] == 0
    static = load_config(variant="v2_static_scope")
    assert static["model"]["static_scope"] and static["loss"]["goal_sufficiency"] == 0
    assert DEFAULTS["variant"] == "goal_sufficient"
    assert DEFAULTS["model"]["goal_sufficient"] is True


def test_v2_yaml_and_explicit_legacy_baseline_are_unambiguous():
    folder = ROOT / "configs/repairscope"
    assert load_config(folder / "goal_sufficient.yaml")["variant"] == "goal_sufficient"
    for variant in V2_ABLATIONS:
        assert load_config(folder / f"{variant}.yaml")["variant"] == variant
    baseline = load_config(folder / "goal_sufficient.yaml", variant="full")
    assert baseline["model"]["goal_sufficient"] is False
    assert baseline["loss"]["goal_sufficiency"] == 0
    assert baseline["loss"]["goal_relevant_escape"] == 0


def test_reference_and_execution_utility_weights_have_one_resolution(tmp_path):
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("controller:\n  damage_weight: 8\n", encoding="utf-8")
    assert load_config(legacy)["goal_decision"]["damage_weight"] == 8
    config = load_config(overrides=["goal_decision.damage_weight=7",
                                    "controller.damage_weight=9"])
    assert config["goal_decision"]["damage_weight"] == config["controller"]["damage_weight"] == 9
    assert config["goal_decision"]["protected_rule"] == "satisfied_and_effective_protected"
    conflicting = tmp_path / "conflicting.yaml"
    conflicting.write_text("controller:\n  damage_weight: 8\ngoal_decision:\n  damage_weight: 6\n")
    with pytest.raises(ValueError, match="Conflicting"):
        load_config(conflicting)
    with pytest.raises(ValueError, match="temperature must be positive"):
        load_config(overrides=["goal_decision.temperature=0"])


def test_initial_v2_dry_run_is_a0_through_a4_then_only_required_ablations(tmp_path):
    env = _environment(tmp_path)
    result = _script("run_goal_sufficient_main.sh", env)
    assert result.returncode == 0, result.stderr
    plans = _stream(result.stdout)
    assert len(plans) == 33  # Actor evaluation + eight train/predict/calibrate/evaluate runs.
    assert plans[0]["command"] == "evaluate"
    assert plans[0]["config"]["variant"] == "actor_only"
    training = [plan for plan in plans if plan["command"] == "train"]
    assert [plan["config"]["variant"] for plan in training] == [
        "dynamics", "paired_effect", "full", "goal_sufficient", *V2_ABLATIONS]
    for plan in plans:
        config = plan["config"]
        assert config["train"]["seed"] == config["evaluation"]["seed"] == 100
        assert config["train"]["epochs"] == 30 and config["train"]["max_steps"] is None
        assert config["train"]["horizon"] == 4 and config["model"]["scope_budget"] == 128
        assert config["data"]["train"] == env["TRAIN_DATA"]
        assert config["data"]["dev"] == env["EVAL_DATA"]
        assert plan["will_execute"] is False
        assert "a4_no_boundary" not in plan["output_dir"]
        assert "a5_no_escape" not in plan["output_dir"]
    assert not Path(env["RUN_ROOT"]).exists()


def test_ours_only_entry_runs_exactly_the_four_fixed_protocol_stages(tmp_path):
    env = _environment(tmp_path)
    result = _script("a4_goal_sufficient.sh", env)
    assert result.returncode == 0, result.stderr
    plans = _stream(result.stdout)
    assert [plan["command"] for plan in plans] == ["train", "predict-eval", "calibrate", "evaluate"]
    for plan in plans:
        config = plan["config"]
        assert config["variant"] == "goal_sufficient"
        assert config["train"]["seed"] == 100
        assert config["train"]["epochs"] == 30 and config["train"]["max_steps"] is None
        assert config["train"]["horizon"] == 4
        assert config["model"]["scope_budget"] == 128
        assert config["model"]["candidate_limit"] == 256
        assert config["data"]["train"] == env["TRAIN_DATA"]
        assert config["data"]["dev"] == env["EVAL_DATA"]
        assert config["loss"]["goal_sufficiency"] == 1
        assert plan["will_execute"] is False
    assert not Path(env["RUN_ROOT"]).exists()


@pytest.mark.parametrize("entry", ["run_goal_sufficient_main.sh", "a4_goal_sufficient.sh"])
@pytest.mark.parametrize("case", ["missing_root", "legacy_name", "existing_root",
                                  "extra_seeds", "changed_loss", "changed_epochs"])
def test_initial_v2_rejects_overwrite_and_unplanned_protocol_changes(entry, case, tmp_path):
    env, args = _environment(tmp_path), ()
    if case == "missing_root":
        env.pop("RUN_ROOT")
    elif case == "legacy_name":
        env["RUN_ROOT"] = str(tmp_path / "repairscope")
    elif case == "existing_root":
        Path(env["RUN_ROOT"]).mkdir()
    elif case == "extra_seeds":
        env["SEEDS"] = "100 101 102"
    elif case == "changed_loss":
        args = ("--set=loss.goal_sufficiency=0",)
    elif case == "changed_epochs":
        args = ("--set", "train.epochs=60")
    result = _script(entry, env, *args)
    assert result.returncode == 2
    assert not result.stdout


def test_v2_probing_is_separate_and_reuses_one_a4_checkpoint_and_budget(tmp_path):
    env = _environment(tmp_path)
    env.update(CHECKPOINT="/shared/a4_goal_sufficient/checkpoint.pt",
               PROBE_PLANS="/shared/train_only_plans.json")
    result = _script("run_goal_sufficient_b.sh", env)
    assert result.returncode == 0, result.stderr
    plans = _stream(result.stdout)
    assert len(plans) == 4
    assert [plan["config"]["probing"]["strategy"] for plan in plans] == [
        "random", "uncertainty", "novelty_learnability", "risk"]
    for plan in plans:
        assert plan["command"] == "probe" and plan["will_execute"] is False
        assert plan["checkpoint"] == env["CHECKPOINT"]
        assert plan["config"]["variant"] == "goal_sufficient"
        assert plan["config"]["probing"]["budget_calls"] == 2000
        assert plan["config"]["probing"]["plans"] == env["PROBE_PLANS"]
    assert not Path(env["RUN_ROOT"]).exists()
