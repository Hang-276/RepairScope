"""Record/verify immutable code, configs, input labels and trained heads before test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from b2_appworld.io import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("create", "verify"))
args = parser.parse_args()
target = Path("runs/frozen_protocol.json")
files = [Path("runs/models.lock.json"), Path("runs/configs/appworld.json"),
         Path("runs/configs/train_effect.json"), Path("runs/configs/train_scalar.json"),
         Path("runs/labels/gate.json"), Path("runs/labels/mapping_audit.json"),
         Path("runs/labels/train.jsonl"), Path("runs/labels/dev.jsonl")]
for split in ("train", "dev"):
    files.extend(Path(f"runs/labels/{split}{suffix}") for suffix in
                 (".mapping.json", ".jsonl.receipt.json", ".quality.json"))
    quality = json.loads(Path(f"runs/labels/{split}.quality.json").read_text())
    current_digest = hashlib.sha256(Path(f"runs/labels/{split}.jsonl").read_bytes()).hexdigest()
    if quality.get("source_files", [{}])[0].get("sha256") != current_digest:
        raise SystemExit(f"Stale collection gate for {split}; rerun scripts/check_collection.py")
for kind in ("effect", "scalar"):
    for seed in (100, 101, 102):
        directory = Path(f"runs/checkpoints/{kind}_s{seed}/best")
        if not (directory / "metadata.json").is_file():
            raise SystemExit(f"Missing actual final checkpoint {directory}")
        files.extend(path for path in directory.rglob("*") if path.is_file())
git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
subprocess.run(["git", "diff", "--quiet", "HEAD"], check=True)
if subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True).strip():
    raise SystemExit("Commit all non-ignored source/config changes before freezing the test protocol")
hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(files))}
record = {"git_head": git_head, "files_sha256": hashes,
          "test_policy": "full native aggregate only; no task-wise test tuning"}
if not json.loads(Path("runs/labels/gate.json").read_text()).get("passed"):
    raise SystemExit("Collection gate has not passed")
if args.mode == "create":
    if target.exists():
        raise SystemExit("Protocol already frozen; preserve the original record")
    atomic_json(target, record)
elif json.loads(target.read_text()) != record:
    raise SystemExit("Frozen protocol changed; do not run/reinterpret test results")
print(f"Protocol {args.mode}: {git_head}")
