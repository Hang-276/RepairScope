"""Live entry points and aggregate-only native benchmark evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import UPSTREAM_COMMIT
from .datasets import dataset_manifest, load_split_ids
from .io import atomic_json
from .schemas import ALL_SPLITS, ensure_research_split
from .runtime import validate_live_predictor_metadata


def verify_upstream(root):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if result != UPSTREAM_COMMIT:
        raise ValueError(f"Upstream source must be pinned to {UPSTREAM_COMMIT}, got {result}")
    return result


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def verify_test_protocol(config, checkpoint=None):
    """The --final flag cannot bypass the repository's frozen protocol gate."""
    repository = Path.cwd()
    script = repository / "scripts/freeze_protocol.py"
    frozen = Path(config["output_dir"]) / "frozen_protocol.json"
    frozen_config_path = repository / "runs/configs/appworld.json"
    if not script.is_file() or not frozen.is_file() or not frozen_config_path.is_file():
        raise ValueError("Create the frozen protocol from the repository root before any test execution")
    subprocess.run([sys.executable, str(script), "verify"], cwd=repository, check=True,
                   capture_output=True, text=True)
    frozen_config = json.loads(frozen_config_path.read_text())
    if config_digest(config) != config_digest(frozen_config):
        raise ValueError("Test config or actor seed differs from the frozen protocol")
    if checkpoint:
        record = json.loads(frozen.read_text())
        checkpoint_metadata = (Path(checkpoint) / "metadata.json").resolve()
        recorded = {(repository / name).resolve() for name in record["files_sha256"]}
        if checkpoint_metadata not in recorded:
            raise ValueError("Test predictor checkpoint was not included in the frozen protocol")


def run_experiment(config, split, method, experiment_name, checkpoint=None, limit=None, final=False):
    if split not in ALL_SPLITS or method not in {"baseline", "capture", "direct", "rule", "scalar", "effect"}:
        raise ValueError("Unknown split or method")
    if not experiment_name or any(x in experiment_name for x in ("/", "\\", "..")):
        raise ValueError("experiment_name must be a single directory name")
    if split.startswith("test"):
        if not final or limit is not None or method == "capture":
            raise ValueError("Test execution requires --final, full split, and a frozen online method")
        verify_test_protocol(config, checkpoint)
    if method == "capture":
        ensure_research_split(split)
    if (method in {"effect", "scalar"}) != bool(checkpoint):
        raise ValueError("Exactly the learned methods require --checkpoint")
    verify_upstream(config["root"])
    manifest = dataset_manifest(config["root"])
    ids = load_split_ids(config["root"], split)
    if limit is not None:
        if limit < 1 or split.startswith("test"):
            raise ValueError("Positive train/dev limit required")
        ids = ids[:limit]
    run_dir = Path(config["output_dir"]) / experiment_name
    if run_dir.exists():
        raise FileExistsError("Experiment directory exists; use a new name to preserve provenance")
    native_dir = Path(config["root"]) / "experiments/outputs" / experiment_name
    if native_dir.exists():
        raise FileExistsError("Native experiment exists; refusing to overwrite prior state")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["actor"]["tokenizer"],
                    revision=config["actor"].get("tokenizer_revision"), trust_remote_code=False)
    predictor = None
    checkpoint_metadata = None
    if checkpoint:
        # Inspect metadata before importing Torch or allocating accelerator
        # memory. A tiny fixture must never masquerade as a live benchmark.
        checkpoint_metadata = json.loads((Path(checkpoint) / "metadata.json").read_text())
        validate_live_predictor_metadata(checkpoint_metadata, method)
        from .learning import load_predictor
        predictor = load_predictor(checkpoint, device="cuda")
        checkpoint_metadata = predictor.metadata
        validate_live_predictor_metadata(checkpoint_metadata, method)
    os.environ["APPWORLD_ROOT"] = config["root"]
    from appworld.common.path_store import path_store
    path_store.update_root(config["root"])
    # Upstream constructs an unused OpenAI SDK client while checking kwargs.
    # A local placeholder satisfies that constructor, and is restored immediately.
    had_key, old_key = "OPENAI_API_KEY" in os.environ, os.environ.get("OPENAI_API_KEY")
    if not had_key:
        os.environ["OPENAI_API_KEY"] = "local-no-key"
    try:
        from .agent import RepairScopeAgent
        agent = RepairScopeAgent(config, method, split, experiment_name, tokenizer, predictor)
    finally:
        if not had_key:
            os.environ.pop("OPENAI_API_KEY", None)
        elif old_key is not None:
            os.environ["OPENAI_API_KEY"] = old_key
    run_record = {"config": config, "config_sha256": config_digest(config),
                "split": split, "method": method, "task_ids": ids, "complete": False,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_metadata": checkpoint_metadata, "dataset": manifest,
                "upstream_commit": UPSTREAM_COMMIT, "final_test": final}
    atomic_json(run_dir / "run.json", run_record)
    # Use the original official task loop/logger setup and completion markers.
    agent.solve_tasks(ids, experiment_name=experiment_name)
    records = [json.loads((run_dir / "tasks" / task / "trajectory.json").read_text()) for task in ids]
    receipt = {"split": split, "method": method, "task_count": len(ids), "complete": True,
               "eligible_count": sum(r["eligible"] for r in records),
               "intervention_count": sum(len(r["checkpoints"]) for r in records),
               "executed_code_blocks": sum(len(r["steps"]) for r in records),
               "actual_api_calls": sum(r["actual_api_calls"] for r in records),
               "model_input_tokens": sum(r["budget"]["input_tokens"] for r in records),
               "model_output_tokens": sum(r["budget"]["output_tokens"] for r in records),
               "model_http_calls": sum(r["budget"]["calls"] for r in records),
               "model_http_seconds": sum(r["budget"]["seconds"] for r in records),
               "classifier_processed_tokens": sum(r["budget"]["classifier_processed_tokens"] for r in records),
               "classifier_seconds": sum(r["budget"]["classifier_seconds"] for r in records),
               "wall_seconds": sum(r["wall_seconds"] for r in records),
               "terminations": {v: sum(r["termination"] == v for r in records)
                                for v in sorted({r["termination"] for r in records})},
               "evidence_origin": "appworld_runtime"}
    atomic_json(run_dir / "receipt.json", receipt)
    atomic_json(run_dir / "run.json", {**run_record, "complete": True})
    return receipt


def evaluate_native(config, experiment_name):
    if not experiment_name or any(x in experiment_name for x in ("/", "\\", "..")):
        raise ValueError("experiment_name must be a single directory name")
    verify_upstream(config["root"])
    run_dir = Path(config["output_dir"]) / experiment_name
    run = json.loads((run_dir / "run.json").read_text())
    receipt = json.loads((run_dir / "receipt.json").read_text())
    if not receipt.get("complete") or not run.get("complete") or run["config_sha256"] != config_digest(config):
        raise ValueError("Incomplete run or mismatched config")
    if (run_dir / "native_aggregate.json").exists():
        raise FileExistsError("Native aggregate already evaluated; preserve the first result")
    if run["split"].startswith("test") and set(run["task_ids"]) != set(load_split_ids(config["root"], run["split"])):
        raise ValueError("Native test evaluation requires the full task denominator")
    os.environ["APPWORLD_ROOT"] = config["root"]
    from appworld.common.path_store import path_store
    from appworld.evaluator import evaluate_tasks
    path_store.update_root(config["root"])
    result = evaluate_tasks(run["task_ids"], experiment_name=experiment_name,
                            include_details=False, save_reports=False)
    # Never export task-wise test reports. SGC is all variants in a scenario.
    aggregate = {"experiment": experiment_name, "split": run["split"],
                 "task_count": len(run["task_ids"]), **result["aggregate"],
                 "evidence_origin": "appworld_native_evaluator"}
    atomic_json(run_dir / "native_aggregate.json", aggregate)
    return aggregate
