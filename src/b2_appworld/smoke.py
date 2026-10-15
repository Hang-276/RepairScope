"""Deliberately synthetic local end-to-end pipeline verification."""
from __future__ import annotations

from pathlib import Path

from .io import atomic_json, write_jsonl
from .schemas import Candidate, Example, Goal


def fixture_smoke(output_dir):
    from .training import train_from_config, evaluate_checkpoint
    from .learning import load_predictor
    from .selection import select_by_effects, select_by_scalar
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError("Use a fresh fixture output directory")
    instruction = "Set A and preserve B."
    goals = (Goal("g1", "Set A", "Set A"), Goal("g2", "preserve B", "preserve B"))
    candidates = (Candidate("original", "set_a()", ("g1",), 1),
                  Candidate("local", "set_a(); clear_b()", ("g1",), 2),
                  Candidate("composite", "pass", (), 0),
                  Candidate("broad", "set_a(); keep_b()", ("g1", "g2"), 2))
    labels = ((1, 3), (1, 2), (0, 3), (1, 3))
    data = {}
    for split, count in (("train", 3), ("dev", 2)):
        rows = []
        for number in range(count):
            scenario = f"fixture{split}{number}"
            for candidate, target in zip(candidates, labels):
                ex = Example(f"{scenario}_1", scenario, "s1", split, instruction,
                             [{"role": "user", "content": "A is unset. B is present."}], goals,
                             candidate, target, source="synthetic_fixture", before_fingerprint="fixture-state")
                rows.append(ex.to_dict())
        data[split] = directory / f"{split}.jsonl"
        write_jsonl(data[split], rows)
    summaries = {}
    for kind in ("effect", "scalar"):
        config = {"train_data": str(data["train"]), "dev_data": str(data["dev"]),
                  "output_dir": str(directory / kind), "kind": kind, "backend": "tiny",
                  "device": "cpu", "quantized": False, "max_length": 4096,
                  "tiny_hidden_size": 8, "epochs": 1, "batch_size": 2,
                  "gradient_accumulation": 2, "seed": 100, "lr": 0.001}
        summary = train_from_config(config)
        predictor = load_predictor(summary["checkpoint"], device="cpu")
        args = (instruction, [{"role": "user", "content": "A is unset. B is present."}], goals, candidates)
        values = predictor.predict(*args)
        selected = (select_by_effects(candidates, values, len(goals)) if kind == "effect"
                    else select_by_scalar(candidates, values))
        report = evaluate_checkpoint(summary["checkpoint"], data["dev"],
                                     directory / f"{kind}_diagnostics.json", device="cpu")
        summaries[kind] = {"checkpoint": summary["checkpoint"], "selected_id": selected.id,
                           "fixture_backend": True, "diagnostics": report}
    result = {"evidence_origin": "synthetic_fixture", "fixture_backend": True,
              "real_appworld_tasks_run": 0, "real_models_trained": 0,
              "research_validation": False, "pipelines": summaries}
    atomic_json(directory / "receipt.json", result)
    return result
