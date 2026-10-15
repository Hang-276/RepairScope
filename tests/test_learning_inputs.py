"""Input/split guards use explicit synthetic fixtures, not AppWorld evidence."""
import copy
import sys
import subprocess

import pytest

from b2_appworld.metrics import candidate_cost_denominators
from b2_appworld.model_inputs import ByteTokenizer, InputTooLongError, build_effect_text, encode_effect_input
from b2_appworld.schemas import Candidate, Example, Goal
from b2_appworld.training import scenario_folds, split_scenario_holdout, train_from_config
from b2_appworld.io import write_jsonl


def fixture_example(scenario="s0", candidate="a", labels=(1, 3), split="train", cost=1):
    return Example(
        task_id=scenario + "_1", scenario_id=scenario, state_id="s1", split=split,
        instruction="Set A and keep B.", history=[{"role": "tool", "content": "A absent; B present."}],
        goals=(Goal("a", "Set A", "Set A"), Goal("b", "keep B", "keep B")),
        candidate=Candidate(candidate, "apis.a.set()" if candidate == "a" else "apis.b.delete()", ("a",), cost),
        labels=labels, source="synthetic_fixture")


def test_queries_follow_full_candidate_and_all_goals_without_labels():
    ex = fixture_example()
    tokenizer = ByteTokenizer()
    encoded = encode_effect_input(tokenizer, ex.instruction, ex.history, ex.goals, ex.candidate,
                                  max_length=2048)
    text = build_effect_text(ex.instruction, ex.history, ex.goals, ex.candidate)
    assert "labels" not in text and "evaluator" not in text
    assert text.index("apis.a.set()") < text.index("GOAL_QUERY")
    assert text.index('"id":"b"') < text.index("GOAL_QUERY")
    assert len(encoded.goal_positions) == 2
    assert encoded.goal_positions[0] < encoded.goal_positions[1] < len(encoded.input_ids)
    assert encoded.text == text
    # Changing outcomes cannot alter any model input.
    other = copy.deepcopy(ex.to_dict())
    other["labels"] = [0, 2]
    changed = Example.from_dict(other)
    assert build_effect_text(changed.instruction, changed.history, changed.goals, changed.candidate) == text


def test_only_old_history_can_be_dropped_and_required_context_is_never_truncated():
    ex = fixture_example()
    tok = ByteTokenizer()
    newest = [{"role": "tool", "content": "LATEST_VISIBLE_OBSERVATION"}]
    needed = len(encode_effect_input(tok, ex.instruction, newest, ex.goals, ex.candidate,
                                     max_length=10000).input_ids)
    history = [{"role": "assistant", "content": "old " * 1000}] + newest
    encoded = encode_effect_input(tok, ex.instruction, history, ex.goals, ex.candidate,
                                  max_length=needed)
    assert encoded.dropped_history_messages == 1
    assert "LATEST_VISIBLE_OBSERVATION" in encoded.text
    assert "old old" not in encoded.text
    with pytest.raises(InputTooLongError, match="Latest history"):
        encode_effect_input(tok, ex.instruction, newest, ex.goals, ex.candidate, max_length=needed - 1)
    with pytest.raises(InputTooLongError, match="Full task/goals/candidate"):
        encode_effect_input(tok, ex.instruction, [], ex.goals, ex.candidate, max_length=16)


def test_runtime_history_rejects_evaluator_records():
    ex = fixture_example()
    with pytest.raises(ValueError, match="non-runtime fields"):
        build_effect_text(ex.instruction, [{"role": "tool", "content": "ok", "ground_truth": "secret"}],
                          ex.goals, ex.candidate)


def test_scenario_folds_hold_every_task_state_candidate_together():
    rows = [fixture_example(f"s{i}", candidate=c, labels=(1, 3) if c == "a" else (0, 2))
            for i in range(10) for c in ("a", "b")]
    folds = scenario_folds(rows)
    assert sorted(x for fold in folds for x in fold) == sorted({r.scenario_id for r in rows})
    assert folds == scenario_folds(rows)
    train, heldout = split_scenario_holdout(rows, fold=2)
    assert not ({r.scenario_id for r in train} & {r.scenario_id for r in heldout})
    assert len(train) + len(heldout) == len(rows)
    assert all(sum(r.scenario_id == scenario for r in heldout) == 2 for scenario in folds[2])


def test_training_refuses_dev_optimization_before_importing_torch(tmp_path):
    path = tmp_path / "dev.jsonl"
    write_jsonl(path, [fixture_example(split="dev").to_dict()])
    with pytest.raises(ValueError, match="exclusively split=train"):
        train_from_config({"train_data": str(path), "output_dir": str(tmp_path / "out")})


def test_pool_integrity_and_cost_metadata_are_checked():
    a, b = fixture_example(cost=1), fixture_example(candidate="b", labels=(0, 2), cost=4)
    denominators = candidate_cost_denominators([a, b])
    assert list(denominators.values()) == [4]
    data = b.to_dict()
    data["history"][0]["content"] = "different state"
    with pytest.raises(ValueError, match="inconsistent context"):
        candidate_cost_denominators([a, Example.from_dict(data)])


def test_ml_import_does_not_load_torch():
    code = "import b2_appworld.learning,b2_appworld.training; import sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_main_hf_training_refuses_synthetic_fixture_before_loading_model(tmp_path):
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    write_jsonl(train, [fixture_example("train0").to_dict()])
    write_jsonl(dev, [fixture_example("dev0", split="dev").to_dict()])
    with pytest.raises(ValueError, match="source=natural"):
        train_from_config({"train_data": str(train), "dev_data": str(dev),
                           "output_dir": str(tmp_path / "model"), "backend": "hf"})


def test_training_refuses_nonempty_run_directory_before_model_loading(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "important.json").write_text('{}')
    with pytest.raises(FileExistsError, match="new run directory"):
        train_from_config({"train_data": "unused.jsonl", "output_dir": str(existing)})


def test_main_effect_and_scalar_default_to_identical_complete_label_records():
    from b2_appworld.training import _prepare
    a = fixture_example()
    b = fixture_example(candidate="b", labels=(0, -100))
    for kind in ("effect", "scalar"):
        usable, ignored = _prepare([a, b], ByteTokenizer(), {"max_length": 2048}, kind)
        assert len(usable) == 1 and ignored == 1
    diagnostic, ignored = _prepare([a, b], ByteTokenizer(),
                                   {"max_length": 2048, "require_complete_labels": False}, "effect")
    assert len(diagnostic) == 2 and ignored == 0


def test_filtering_unknown_or_rejected_candidates_preserves_original_pool_norm():
    from b2_appworld.training import _prepare
    originals = []
    for ex in (fixture_example(cost=1), fixture_example(candidate="b", labels=(0, -100), cost=4)):
        data = ex.to_dict()
        data["metadata"] = {"candidate_group_max_cost": 10,
                            "candidate_pool_ids": ["a", "b", "rejected"], "candidate_pool_size": 3}
        originals.append(Example.from_dict(data))
    assert list(candidate_cost_denominators(originals).values()) == [10]
    effect, _ = _prepare(originals, ByteTokenizer(), {"max_length": 2048}, "effect")
    scalar, _ = _prepare(originals, ByteTokenizer(), {"max_length": 2048}, "scalar")
    assert '"normalized_candidate_cost":0.1' in effect[0]["input"].text
    assert effect[0]["input"].input_ids == scalar[0]["input"].input_ids
    assert scalar[0]["target"] == pytest.approx(.5 - .05 * .1)
