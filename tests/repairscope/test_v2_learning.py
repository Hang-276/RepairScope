from copy import deepcopy
import json

import pytest
import torch

from repairscope.config import load_config
from repairscope.fixtures import fixture_records
from repairscope.model import Predictor, budget_gate, predict_record
from repairscope.tensorizer import Tensorizer
from repairscope.decision import goal_sufficiency_loss
from repairscope.losses import transition_losses
from repairscope.training import train_model
from repairscope.calibration import calibrate_threshold, apply_calibration
from repairscope.provenance import file_digest


def config():
    cfg = load_config(variant="goal_sufficient")
    cfg["model"].update(d_model=32, n_heads=4, relation_layers=1, temporal_layers=1,
                        feature_dim=16, value_slots=16, history=4, scope_budget=1,
                        candidate_limit=16, dropout=0.)
    cfg["train"].update(device="cpu", precision="fp32", epochs=1, micro_batch=4,
                        grad_accum=1, max_steps=2, warmup_fraction=0.)
    return cfg


def test_fixed_mass_gate_has_exact_hard_forward_competitive_backward():
    score = torch.tensor([[1., 0., -1., 9.], [2., 3., 0., 0.]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    hard = torch.tensor([[True, False, False, False], [True, True, False, False]])
    gate = budget_gate(score, mask, hard)
    assert torch.equal(gate, hard.float())
    (gate * torch.tensor([[0., 1., 0., 100.], [1., 2., 3., 4.]])).sum().backward()
    assert score.grad[0, 1] > 0 and score.grad[0, 0] < 0
    assert abs(float(score.grad[0].sum())) < 1e-6
    assert score.grad[0, 3] == 0 and not score.grad[1].any()
    assert torch.isfinite(score.grad).all()


def test_decision_loss_updates_actual_scope_scorer_and_no_audit_leakage():
    torch.manual_seed(1)
    cfg = config()
    records = fixture_records(1)[5:]
    tensorizer, model = Tensorizer(cfg), Predictor(cfg).eval()
    # Deliberately initialize a legal omitted payment field, not a lucky
    # already-sufficient scope: then the isolated objective must reach scorer.
    with torch.no_grad():
        model.scope_head[-1].weight.zero_()
        model.scope_head[-1].bias.zero_()
    batch = tensorizer.batch(records)
    output = model(batch)
    loss = goal_sufficiency_loss(output, batch, cfg)["goal_sufficiency"]
    if float(loss.detach()) == 0:
        # A fixed public prior favoring an irrelevant field creates omission.
        batch["static_prior"].zero_()
        batch["static_prior"][:, batch["metadata"][0]["keys"].index("shop.Order[1].note")] = 10
        output = model(batch)
        loss = goal_sufficiency_loss(output, batch, cfg)["goal_sufficiency"]
    assert loss > 0
    loss.backward()
    gradient = model.scope_head[-1].weight.grad
    assert gradient is not None and gradient.norm() > 0
    before = model.scope_head[-1].weight.detach().clone()
    torch.optim.SGD(model.scope_head.parameters(), lr=.1).step()
    assert not torch.equal(before, model.scope_head[-1].weight)
    altered = deepcopy(records)
    for record in altered:
        record["audit"] = {"complete": False, "field_domain": [], "changes": []}
        record["goal_after"] = [None] * len(record["goals"])
        record["audit_next_state"] = {"fields": []}
    a, b = model(tensorizer.batch(records)), model(tensorizer.batch(altered))
    torch.testing.assert_close(a["scope_logits"], b["scope_logits"])
    assert torch.equal(a["scope"], b["scope"])
    assert int(a["scope"].sum(1).max()) == 1


def test_goal_escape_head_is_supervised_and_ablation_does_not_deploy_it():
    cfg = config()
    records = fixture_records(1)[7:]
    model, tensorizer = Predictor(cfg), Tensorizer(cfg)
    batch = tensorizer.batch(records)
    output = model(batch)
    losses = transition_losses(output, batch, cfg)
    assert losses["goal_escape_valid"] > 0
    losses["goal_relevant_escape"].backward()
    assert model.goal_escape_head[-1].weight.grad.norm() > 0
    assert "goal_escape" in predict_record(model, tensorizer, records[0])
    cfg["loss"]["goal_relevant_escape"] = 0
    assert "goal_escape" not in predict_record(Predictor(cfg), tensorizer, records[0])


def test_goal_risk_conditions_on_public_protection_at_the_current_step():
    torch.manual_seed(3)
    cfg = config()
    cfg["model"]["scope_budget"] = 16
    model, tensorizer = Predictor(cfg).eval(), Tensorizer(cfg)
    active = deepcopy(fixture_records(1)[7])
    active["goals"][1]["protect_until_step"] = 1
    expired = deepcopy(active)
    expired["step"] = 1
    a, b = model(tensorizer.batch([active])), model(tensorizer.batch([expired]))
    assert torch.equal(a["scope"], b["scope"])
    assert not torch.allclose(a["goal_escape_logits"], b["goal_escape_logits"])
    assert not torch.allclose(a["goal_expansion"], b["goal_expansion"])


def test_training_rejects_formal_objective_without_decision_labels(tmp_path):
    records = fixture_records(1)[:5]
    with pytest.raises(ValueError, match="no informative"):
        train_model(records, config(), tmp_path)
    audit = json.loads((tmp_path / "goal_supervision_audit.json").read_text())
    assert audit["decision_informative_groups"] == 0


def test_goal_risk_calibration_uses_goal_labels_and_masks_unknown(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint digest")
    source = tmp_path / "predictions.jsonl"
    rows = []
    for risk, label in ((.8, 1), (.1, 0), (.9, None)):
        rows.append({"horizon": 1, "offset": 1, "split": "dev",
                     "checkpoint_id": file_digest(checkpoint),
                     "prediction": {"goal_escape": risk, "scope": []},
                     "label_receipt": {"goal_escape_label": label, "changes": [], "complete": True}})
    source.write_text("\n".join(json.dumps(row) for row in rows))
    target = tmp_path / "calibration.json"
    result = calibrate_threshold(source, checkpoint, target, .5, "goal_escape")
    assert result["dev_records"] == 2 and result["dev_escape_positives"] == 1
    cfg = config()
    cfg["controller"]["calibration_file"] = str(target)
    assert apply_calibration(cfg, checkpoint)["controller"]["risk_threshold"] == .8
