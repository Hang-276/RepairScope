"""Default entrypoints select v2 without changing explicit legacy baselines."""
from pathlib import Path

import pytest

from repairscope.cli import parser, run
from repairscope.config import load_config
from repairscope.training import resolve_config


ROOT = Path(__file__).resolve().parents[2]


def assert_objective(config, variant):
    v2 = variant == "goal_sufficient"
    assert config["variant"] == variant
    assert config["model"]["goal_sufficient"] is v2
    assert config["loss"]["goal_sufficiency"] == int(v2)
    assert config["loss"]["goal_relevant_escape"] == int(v2)


@pytest.mark.parametrize("path", [None, "base.yaml", "goal_sufficient.yaml", "local_smoke.yaml"])
def test_unqualified_configuration_uses_the_v2_objective(path):
    config = load_config(ROOT / "configs/repairscope" / path if path else None)
    assert_objective(config, "goal_sufficient")


@pytest.mark.parametrize("variant", ["actor_only", "dynamics", "paired_effect", "full"])
@pytest.mark.parametrize("source", ["base.yaml", "goal_sufficient.yaml"])
def test_explicit_legacy_variant_cannot_inherit_v2_selector_or_losses(variant, source):
    config = load_config(ROOT / "configs/repairscope" / source, variant=variant)
    assert_objective(config, variant)
    # Both training/checkpoint resolution and CLI loading retain the switches.
    assert_objective(resolve_config(config), variant)
    if variant == "full":
        assert config["loss"]["boundary"] == config["loss"]["escape"] == 1
    else:
        assert config["loss"]["boundary"] == config["loss"]["escape"] == 0
    assert config["loss"]["paired"] == int(variant == "paired_effect")


def test_explicit_legacy_yaml_remains_a3_when_no_cli_variant_is_given():
    config = load_config(ROOT / "configs/repairscope/full.yaml")
    assert_objective(config, "full")


@pytest.mark.parametrize("command", ["train", "evaluate", "collect", "doctor", "smoke"])
def test_cli_default_dry_runs_select_v2_without_executing(command):
    plan = run(parser().parse_args([command, "--dry-run"]))
    assert plan["will_execute"] is False
    assert plan["output_dir"] == "runs/repairscope_v2"
    assert_objective(plan["config"], "goal_sufficient")


def test_cli_explicit_variant_takes_precedence_over_set_and_yaml():
    plan = run(parser().parse_args(["train", "--dry-run", "--variant", "full",
                                   "--set", "variant=goal_sufficient"]))
    assert_objective(plan["config"], "full")
    assert_objective(load_config(variant="goal_sufficient", overrides=["variant=full"]), "goal_sufficient")


def test_set_variant_selects_an_objective_not_just_a_run_label():
    assert_objective(load_config(overrides=["variant=full"]), "full")
    assert_objective(load_config(overrides=["variant=full", "variant=goal_sufficient"]), "goal_sufficient")
    with pytest.raises(ValueError, match="Unknown variant"):
        load_config(overrides=["variant=unknown"])


def test_low_level_legacy_checkpoint_resolution_is_not_reinterpreted_as_v2():
    # Archived checkpoint configs may predate the variant and v2 fields. Keep
    # their old architecture/loss defaults; public new runs use load_config().
    legacy = resolve_config({"model": {"d_model": 32, "n_heads": 4}})
    assert not legacy["model"].get("goal_sufficient", False)
    assert legacy["loss"].get("goal_sufficiency", 0) == 0
    assert legacy["loss"].get("goal_relevant_escape", 0) == 0
