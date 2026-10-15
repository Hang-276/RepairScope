"""Real CPU autograd/checkpoint round trips on explicitly synthetic byte-GRU data."""
import json

import pytest

torch = pytest.importorskip("torch")

from b2_appworld.io import write_jsonl
from b2_appworld.learning import create_model_bundle, load_predictor
from b2_appworld.schemas import Candidate, Example, Goal
from b2_appworld.training import evaluate_checkpoint, train_from_config


def fixture_rows(split, scenarios):
    result = []
    for scenario in scenarios:
        for candidate, labels, cost in (("restore", (1, 3), 2), ("harm", (0, 2), 1)):
            result.append(Example(scenario + "_1", scenario, "state", split, "Set A and keep B", [],
                                  (Goal("a", "Set A", "Set A"), Goal("b", "keep B", "keep B")),
                                  Candidate(candidate, "print('" + candidate + "')", ("a",), cost),
                                  labels, source="synthetic_fixture"))
    return result


@pytest.fixture(autouse=True)
def cpu_thread_limit():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("kind", ["effect", "scalar"])
def test_train_save_reload_predict_and_heldout_diagnostics(tmp_path, kind):
    train = fixture_rows("train", ["train0", "train1"])
    dev = fixture_rows("dev", ["dev0", "dev1"])
    train_path, dev_path = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    write_jsonl(train_path, [ex.to_dict() for ex in train])
    write_jsonl(dev_path, [ex.to_dict() for ex in dev])
    config = {"train_data": str(train_path), "dev_data": str(dev_path),
              "output_dir": str(tmp_path / kind), "kind": kind, "backend": "tiny",
              "device": "cpu", "quantized": False, "tiny_hidden_size": 8, "max_length": 2048,
              "epochs": 2, "batch_size": 2, "gradient_accumulation": 3, "lr": .01, "seed": 7}
    summary = train_from_config(config)
    assert summary["fixture_backend"] is True
    assert summary["optimizer_steps"] == 2  # final partial windows are stepped
    assert summary["processed_tokens_including_padding"] > 0
    checkpoint = summary["checkpoint"]
    predictor = load_predictor(checkpoint, device="cpu")
    first = dev[0]
    candidates = [ex.candidate for ex in dev[:2]]
    count = predictor.estimate_tokens(first.instruction, first.history, first.goals, candidates)
    predicted = predictor.predict(first.instruction, first.history, first.goals, candidates)
    assert predictor.last_usage["processed_tokens"] == count
    assert predictor.last_usage["forward_passes"] == 2
    assert predictor.last_usage["seconds"] > 0
    reloaded = load_predictor(checkpoint, device="cpu")
    actual = reloaded.predict(first.instruction, first.history, first.goals, candidates)
    assert actual == predicted
    if kind == "effect":
        assert len(actual) == 2 and len(actual[0]) == 2
        assert all(sum(row) == pytest.approx(1) for candidate in actual for row in candidate)
    else:
        assert len(actual) == 2 and all(isinstance(x, float) for x in actual)
    report = evaluate_checkpoint(checkpoint, dev_path, device="cpu")
    assert report["fixture_backend"] is True
    assert report["decision"]["n_complete_candidate_pools"] == 2
    with pytest.raises(ValueError, match="held-out scenarios"):
        evaluate_checkpoint(checkpoint, train_path, device="cpu")


def test_scalar_omits_unknown_rows_and_effect_masks_them(tmp_path):
    train = fixture_rows("train", ["train0"])
    dev = fixture_rows("dev", ["dev0"])
    raw = train[0].to_dict()
    raw["labels"] = [1, -100]
    train[0] = Example.from_dict(raw)
    for name, rows in (("train", train), ("dev", dev)):
        write_jsonl(tmp_path / (name + ".jsonl"), [ex.to_dict() for ex in rows])
    for kind in ("effect", "scalar"):
        result = train_from_config({"train_data": str(tmp_path / "train.jsonl"),
                                    "dev_data": str(tmp_path / "dev.jsonl"),
                                    "output_dir": str(tmp_path / kind), "kind": kind,
                                    "backend": "tiny", "device": "cpu", "quantized": False,
                                    "epochs": 1, "tiny_hidden_size": 4, "max_length": 2048,
                                    "require_complete_labels": False})
        assert result["excluded_unknown_training_rows"] == (1 if kind == "scalar" else 0)
        metadata = json.loads((tmp_path / kind / "best" / "metadata.json").read_text())
        assert metadata["unknown_label_policy"] == ("masked" if kind == "effect" else "exclude_incomplete_rows")


def test_nf4_on_cpu_fails_explicitly():
    with pytest.raises(ValueError, match="requires CUDA"):
        create_model_bundle({"backend": "hf", "device": "cpu", "quantized": True})


def test_unknown_goal_loss_has_zero_gradient_and_shared_queries_use_one_forward():
    from b2_appworld.training import _loss
    logits = torch.randn(1, 2, 4, requires_grad=True)
    loss = _loss(torch, logits, torch.tensor([[1, -100]]), "effect")
    loss.backward()
    assert logits.grad[0, 1].abs().sum().item() == 0
    assert logits.grad[0, 0].abs().sum().item() > 0
    model, _tokenizer, _metadata = create_model_bundle({"backend": "tiny", "device": "cpu",
                                                       "quantized": False, "tiny_hidden_size": 4})
    calls = []
    original_forward = model.encoder.forward

    def counted_forward(*args, **kwargs):
        calls.append(True)
        return original_forward(*args, **kwargs)

    model.encoder.forward = counted_forward
    output = model(input_ids=torch.tensor([[1, 3, 4, 5]]), attention_mask=torch.ones(1, 4),
                   goal_positions=torch.tensor([[1, 3]]))
    assert output.shape == (1, 2, 4) and len(calls) == 1
