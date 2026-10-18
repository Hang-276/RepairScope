"""CPU regressions for server dtype, probability-BCE, and history-mask fixes."""
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from repairscope.config import load_config
from repairscope.fixtures import fixture_goals, fixture_state
from repairscope.losses import _bce
from repairscope.model import Predictor, causal_padding_mask
from repairscope.tensorizer import Tensorizer


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(backbone="blite"):
    config = load_config(variant="goal_sufficient")
    config["model"].update(
        d_model=32, n_heads=4, relation_layers=1, temporal_layers=2,
        feature_dim=16, value_slots=16, history=4, scope_budget=4,
        candidate_limit=16, dropout=0., backbone=backbone,
    )
    return config


def _batch(config):
    # Static fixture construction: no environment execution or optimization.
    record = {
        "task_id": "padding-regression", "split": "train", "step": 0,
        "state": fixture_state(), "goals": fixture_goals(), "history": [],
        "action": {"name": "shop.set_shipping", "arguments": {"shipping": "express"}},
    }
    shorter = deepcopy(record)
    shorter["state"]["fields"] = shorter["state"]["fields"][:-2]
    return Tensorizer(config).batch([record, shorter])


def _history(model, batch, pattern):
    with torch.no_grad():
        slots = model.value_encoder(batch["slot_features"])
        hidden, mask = model._initial_history(batch, slots)
    mask = mask.clone()
    if pattern == "full":
        mask[:] = batch["mask"].unsqueeze(-1)
    elif pattern == "partial":
        mask[:, :, -2:] = batch["mask"].unsqueeze(-1)
        mask[:, ::2, -1] = False
    return hidden.detach(), mask


def test_goal_context_casts_projection_before_bf16_scatter_and_backpropagates():
    torch.manual_seed(123)
    config = _config()
    model, batch = Predictor(config), _batch(config)
    x = torch.zeros(*batch["mask"].shape, model.width, dtype=torch.bfloat16)
    assert model.goal_encoder(batch["goal_features"]).dtype == torch.float32
    context = model._goal_context(batch, x)
    assert context.dtype == torch.bfloat16
    assert torch.isfinite(context).all()
    context.float().square().mean().backward()
    gradient = model.goal_encoder.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize("reduction", ["none", "mean"])
def test_probability_bce_is_fp32_inside_bf16_autocast(reduction):
    probability = torch.tensor(
        [0., .125, .25, .75, .875, 1.], dtype=torch.bfloat16, requires_grad=True,
    )
    target = torch.tensor([1., 0., 1., 0., 1., 0.])
    expected = F.binary_cross_entropy(
        probability.detach().float().clamp(1e-6, 1 - 1e-6), target,
        reduction=reduction,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = _bce(probability, target, reduction=reduction)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(probability.grad).all()
    assert probability.grad[1:-1].abs().sum() > 0


@pytest.mark.parametrize("pattern", ["empty", "partial", "full"])
@pytest.mark.parametrize("times", [[0, 1, 2, 3, 4], [0, 1, 2, 0, 1, 2]])
def test_causal_padding_mask_preserves_valid_keys_and_causality(pattern, times):
    length, heads = len(times), 2
    time = torch.tensor(times)
    causal = time.unsqueeze(0) > time.unsqueeze(1)
    padding = torch.ones(2, length, dtype=torch.bool)
    padding[:, -1] = False
    if pattern == "partial":
        padding[0, 2] = False
        padding[1, 1:3] = False
    elif pattern == "full":
        padding[:] = False
    blocked = causal_padding_mask(causal, padding, heads)
    assert blocked.shape == (2 * heads, length, length)
    assert blocked.dtype == torch.bool
    assert not blocked.all(-1).any()
    for batch_index in range(2):
        for query in range(length):
            expected = causal[query] | padding[batch_index]
            if bool(expected.all()):
                expected = expected.clone()
                expected[query] = False
            for head in range(heads):
                torch.testing.assert_close(blocked[batch_index * heads + head, query], expected)
    # Even the fallback cannot expose a future key to an earlier query.
    assert blocked[:, causal].all()


@pytest.mark.parametrize("backbone", ["blite", "unified"])
@pytest.mark.parametrize("pattern", ["empty", "partial", "full"])
@pytest.mark.parametrize("mode", ["train", "eval", "eval_no_grad"])
def test_history_masked_payload_does_not_change_finite_current_prediction(backbone, pattern, mode):
    torch.manual_seed(123)
    config = _config(backbone)
    model, batch = Predictor(config), _batch(config)
    model.train(mode == "train")
    hidden, mask = _history(model, batch, pattern)
    # Absent slots are not zero after encoders. Their payload must remain hidden.
    if (~mask).any():
        assert hidden[~mask].abs().sum() > 0
    changed = hidden + (~mask).unsqueeze(-1) * torch.randn_like(hidden) * 7
    context = torch.no_grad() if mode == "eval_no_grad" else nullcontext()
    with context:
        actual = model(batch, history=(hidden, mask))
        perturbed = model(batch, history=(changed, mask))
    for name in ("op_logits", "numeric", "goal_logits", "hidden", "escape_logits"):
        assert torch.isfinite(actual[name]).all(), name
        assert torch.isfinite(perturbed[name]).all(), name
        torch.testing.assert_close(actual[name], perturbed[name])
    for name, value in actual["soft_state"].items():
        assert torch.isfinite(value).all(), name
        torch.testing.assert_close(value, perturbed["soft_state"][name])
    if mode == "train" and pattern == "empty":
        actual["op_logits"].square().mean().backward()
        assert model.op_head.weight.grad is not None
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
