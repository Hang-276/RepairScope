from copy import deepcopy

import pytest
import torch

from repairscope.fixtures import fixture_records
from repairscope.losses import counterfactual_losses, transition_losses
from repairscope.model import Predictor, predict_record, decode_prediction
from repairscope.tensorizer import Tensorizer
from repairscope.training import _windows, _loss_for_windows, load_predictor, resolve_config, train_model, update_predictor


def small_config(**model):
    return resolve_config({"model": {"d_model": 32, "n_heads": 4, "relation_layers": 1, "temporal_layers": 1, "feature_dim": 16, "value_slots": 16, "history": 4, "scope_budget": 4, "candidate_limit": 16, "dropout": 0., **model}, "train": {"device": "cpu", "precision": "fp32", "epochs": 1, "micro_batch": 4, "grad_accum": 1, "max_steps": 2, "warmup_fraction": 0.}})


def test_future_labels_cannot_change_inputs_or_copy_slots():
    record = fixture_records(1)[0]
    altered = deepcopy(record)
    altered["next_state"]["fields"][0]["value"] = "secret-future-value"
    altered["next_state"]["fields"].append({"key": "hidden.new", "kind": "text", "value": "secret"})
    tensorizer = Tensorizer(small_config())
    first, second = tensorizer.batch([record]), tensorizer.batch([altered])
    for key in ("features", "action", "values", "numeric", "known", "exists", "kind", "mask", "slot_features", "static_edges", "history_values"):
        torch.testing.assert_close(first[key], second[key])
    assert first["metadata"] == second["metadata"]
    assert "hidden.new" not in first["metadata"][0]["keys"]


def test_unknown_and_partial_audit_never_become_negatives():
    record = deepcopy(fixture_records(1)[0])
    record["audit"] = {"complete": False, "field_domain": [], "changes": []}
    record["next_state"]["fields"][0].update(known=False, value=None)
    batch = Tensorizer(small_config()).batch([record])
    assert not batch["audit_mask"].any()
    assert not batch["escape_mask"].any()
    model = Predictor(small_config())
    losses = transition_losses(model(batch), batch, small_config())
    assert losses["scope"].item() == 0
    assert losses["patch"].item() == 0
    assert losses["escape"].item() == 0


@pytest.mark.parametrize("backbone,temporal,learned,static", [("blite", True, True, False), ("blite", False, True, False), ("blite", True, False, False), ("blite", True, True, True), ("unified", True, True, False)])
def test_backbone_ablations_forward_backward_and_hard_budget(backbone, temporal, learned, static):
    config = small_config(backbone=backbone, temporal=temporal, learned_edges=learned, static_scope=static)
    batch = Tensorizer(config).batch(fixture_records(1)[5:])
    model = Predictor(config)
    output = model(batch)
    losses = transition_losses(output, batch, config)
    cf = counterfactual_losses(output, batch, config)
    loss = losses["step"] + losses["escape"] + cf["boundary"] + cf["paired"]
    assert torch.isfinite(loss)
    loss.backward()
    assert output["scope"].sum(1).max().item() <= 4
    assert model.op_head.weight.grad is not None
    assert cf["boundary_fields"] > 0
    assert cf["paired_fields"] > 0


def test_free_rollout_is_differentiable_and_never_uses_future_state():
    records = fixture_records(1)[:4]
    config = small_config(scope_budget=16)
    model = Predictor(config).eval()
    tensorizer = Tensorizer(config)
    loss, metrics = _loss_for_windows(model, tensorizer, [records], config, torch.device("cpu"), True)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.number_head.weight.grad is not None
    first = tensorizer.batch([records[0]])
    output = model(first)
    future = tensorizer.batch([records[1]], template=first["metadata"])
    # Corrupt all future true input states/history; recurrence must override them.
    corrupt = dict(future)
    for name in ("values", "numeric", "known", "exists", "history_values", "history_numeric"):
        corrupt[name] = torch.randn_like(future[name])
    a = model(future, soft_state=output["soft_state"], history=output["next_history"])
    b = model(corrupt, soft_state=output["soft_state"], history=output["next_history"])
    torch.testing.assert_close(a["numeric"], b["numeric"])
    assert metrics["rollout"] > 0


def test_hard_decode_and_local_training_checkpoint(tmp_path):
    records = fixture_records(1)
    config = small_config()
    result = train_model(records, config, tmp_path)
    assert result["updates"] == 2
    assert result["evidence_origin"] == ["synthetic"]
    model, tensorizer, loaded_config = load_predictor(result["checkpoint"])
    prediction = predict_record(model, tensorizer, records[0], mc_samples=3)
    assert len(prediction["scope"]) <= 4
    assert 0 <= prediction["escape"] <= 1
    assert len(prediction["goal_probabilities"][0]) == 4
    assert (tmp_path/"schema.json").exists()
    updates = update_predictor(model, tensorizer, records[5:], loaded_config)
    assert updates["updates"] == 1
    assert updates["optimizer"] is not None


def test_resume_restores_sampler_optimizer_and_rng_exactly(tmp_path):
    records = fixture_records(1)
    config = small_config(dropout=.1)
    config["train"].update(max_steps=3, checkpoint_every=1)
    uninterrupted = train_model(records, config, tmp_path/"full")
    # A bounded stop shares the same full epoch schedule. max_steps only cuts it.
    partial_config = deepcopy(config)
    partial_config["train"]["max_steps"] = 1
    partial = train_model(records, partial_config, tmp_path/"part")
    resumed = train_model(records, config, tmp_path/"resumed", resume=partial["checkpoint"])
    a = torch.load(uninterrupted["checkpoint"], weights_only=False)
    b = torch.load(resumed["checkpoint"], weights_only=False)
    assert a["global_step"] == b["global_step"] == 3
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    assert a["history"] == b["history"]


def test_heldout_training_rejected(tmp_path):
    records = fixture_records(1)
    records[0]["split"] = "dev"
    with pytest.raises(ValueError, match="Held-out"):
        train_model(records, small_config(), tmp_path)


def test_typed_decode_and_nullable_values_are_schema_valid():
    from repairscope.schema import validate_state
    records = fixture_records(1)
    config = small_config(scope_budget=16)
    record = deepcopy(records[0])
    record["state"]["fields"][2].update(value=None, nullable=True)
    record["next_state"]["fields"][2].update(value=None, nullable=True)
    record["audit_next_state"]["fields"][2].update(value=None, nullable=True)
    model, tensorizer = Predictor(config), Tensorizer(config)
    batch = tensorizer.batch([record])
    output = model(batch)
    assert batch["null"].sum() == 1
    assert batch["target_null"].sum() == 1
    assert torch.isfinite(transition_losses(output, batch, config)["patch"])
    # Force the heads through create/delete/modify and nullable decode branches.
    for op in (1, 2, 3):
        forced = dict(output)
        forced["op_probabilities"] = torch.zeros_like(output["op_probabilities"])
        forced["op_probabilities"][..., op] = 1
        forced["scope"] = batch["mask"]
        forced["null_probabilities"] = torch.ones_like(output["null_probabilities"])
        validate_state(decode_prediction(forced, batch)["next_state"])


def test_missing_or_invalid_four_grid_is_masked():
    config = small_config(scope_budget=16)
    records = fixture_records(1)[5:]
    model, tensorizer = Predictor(config), Tensorizer(config)
    batch = tensorizer.batch(records[:3])
    cf = counterfactual_losses(model(batch), batch, config)
    assert cf["boundary_fields"] == 0
    records[0]["action_valid"] = False
    batch = tensorizer.batch(records)
    cf = counterfactual_losses(model(batch), batch, config)
    assert cf["boundary_fields"] == 0


def test_global_audit_changes_labels_but_never_model_inputs():
    record = deepcopy(fixture_records(1)[0])
    changed = deepcopy(record)
    changed["audit_next_state"]["fields"][2]["value"] = 999.
    changed["audit_next_state"]["fields"].append({"key": "hidden.db", "kind": "text", "value": "unavailable", "known": True, "exists": True})
    changed["audit"]["changes"].append("hidden.db")
    tensorizer = Tensorizer(small_config())
    a, b = tensorizer.batch([record]), tensorizer.batch([changed])
    for name in ("features", "values", "numeric", "slot_features", "action", "history_values", "static_edges"):
        torch.testing.assert_close(a[name], b[name])
    assert a["metadata"] == b["metadata"]
    assert b["target_escape"].item() == 1
    assert not torch.equal(a["target_numeric"], b["target_numeric"])


def test_rollout_windows_stop_at_resets_and_reject_duplicate_steps():
    records = deepcopy(fixture_records(1)[:4])
    assert len(_windows(records, 4)[0]) == 4
    records[2]["state"]["fields"][2]["value"] = 12345.
    assert len(_windows(records, 4)[0]) == 2
    with pytest.raises(ValueError, match="Duplicate"):
        _windows([records[0], deepcopy(records[0])], 4)


def test_fp16_is_explicitly_rejected(tmp_path):
    config = small_config()
    config["train"]["precision"] = "fp16"
    with pytest.raises(ValueError, match="fp16 is not implemented"):
        train_model(fixture_records(1), config, tmp_path)


def test_cf_prefix_and_same_action_requirements_mask_invalid_boundaries():
    config = small_config(scope_budget=16)
    records = deepcopy(fixture_records(1)[5:])
    records[0]["state"]["fields"][2]["value"] = 12.
    tensorizer, model = Tensorizer(config), Predictor(config)
    batch = tensorizer.batch(records)
    cf = counterfactual_losses(model(batch), batch, config)
    assert cf["boundary_fields"] == 0
    records = deepcopy(fixture_records(1)[5:])
    records[2]["action"]["arguments"]["shipping"] = "standard"
    batch = tensorizer.batch(records)
    assert counterfactual_losses(model(batch), batch, config)["boundary_fields"] == 0


def test_mc_dropout_varies_scope_and_restores_mixed_module_flags():
    torch.manual_seed(123)
    config = small_config(dropout=.5)
    model, tensorizer = Predictor(config).eval(), Tensorizer(config)
    model.relations[0].dropout.train()
    before = [module.training for module in model.modules()]
    prediction = predict_record(model, tensorizer, fixture_records(1)[0], mc_samples=5)
    assert [module.training for module in model.modules()] == before
    scopes = prediction["mc_scope_probabilities"]
    assert any(any(scopes[0][key] != draw[key] for key in scopes[0]) for draw in scopes[1:])
    assert prediction["uncertainty"] > 0
    assert all("operation_probabilities" in draw for draw in prediction["mc_predictions"])


def test_no_goal_head_is_not_exposed_to_metrics_or_mc_hypotheses():
    config = small_config(dropout=.2)
    config["loss"]["goal"] = 0
    model, tensorizer = Predictor(config), Tensorizer(config)
    prediction = predict_record(model, tensorizer, fixture_records(1)[0], mc_samples=3)
    assert prediction["goal_probabilities"] == []
    assert all(sample["goal_probabilities"] == [] for sample in prediction["mc_predictions"])
    assert prediction["next_state"]["fields"]
