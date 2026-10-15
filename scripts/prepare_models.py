"""Resolve immutable HF revisions once; optional full snapshot download."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from b2_appworld.config import load_config
from b2_appworld.io import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--download", action="store_true")
args = parser.parse_args()
target = Path("runs/models.lock.json")
if target.exists():
    raise SystemExit("Model lock already exists; reuse it instead of changing revisions mid-experiment")
api = HfApi()
records = {}
for role, repo in (("actor", "Qwen/Qwen3-8B-AWQ"), ("predictor", "Qwen/Qwen3-4B")):
    revision = api.model_info(repo).sha
    if not revision:
        raise RuntimeError("HF did not return an immutable revision")
    path = snapshot_download(repo, revision=revision) if args.download else None
    records[role] = {"repo_id": repo, "revision": revision, "local_path": path}
config = load_config("configs/appworld.toml")
config["actor"]["tokenizer"] = records["actor"]["local_path"] or records["actor"]["repo_id"]
config["actor"]["tokenizer_revision"] = records["actor"]["revision"]
atomic_json("runs/configs/appworld.json", config)
for kind in ("effect", "scalar"):
    training = json.loads(Path(f"configs/train_{kind}.json").read_text())
    training["base_model"] = records["predictor"]["local_path"] or records["predictor"]["repo_id"]
    training["revision"] = records["predictor"]["revision"]
    atomic_json(f"runs/configs/train_{kind}.json", training)
atomic_json(target, records)
print(json.dumps(records, indent=2))
