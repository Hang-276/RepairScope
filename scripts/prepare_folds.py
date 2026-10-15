"""Write exact whole-scenario validation files for post-training diagnostics."""
from b2_appworld.io import atomic_json, read_jsonl, write_jsonl
from b2_appworld.schemas import Example
from b2_appworld.training import scenario_folds

rows = read_jsonl("runs/labels/train.jsonl")
examples = [Example.from_dict(row) for row in rows]
if any(ex.split != "train" for ex in examples):
    raise ValueError("Only official train rows may form training-side folds")
folds = scenario_folds(examples, n_folds=5, seed=100)
atomic_json("runs/folds/manifest.json", {"seed": 100, "folds": folds})
for fold, ids in enumerate(folds):
    write_jsonl(f"runs/folds/fold{fold}.validation.jsonl",
                [row for row in rows if row["scenario_id"] in set(ids)])
