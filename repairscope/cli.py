"""RepairScope v2: goal-sufficient dynamic scope learning and evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from .config import load_config, merge, dump_config, VARIANTS
from .provenance import receipt, write_json


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None,
                        help="YAML configuration; defaults to RepairScope v2")
    parser.add_argument("--variant", choices=sorted(VARIANTS),
                        help="Override configuration variant (default: goal_sufficient; full is the explicit A3 baseline)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output-dir", default="runs/repairscope_v2")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and validate a plan without environment, API or training calls")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="repairscope", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "predict-eval", "calibrate", "collect", "counterfactual", "probe", "doctor", "smoke"):
        _common(sub.add_parser(name))
    by_name = sub.choices
    by_name["train"].add_argument("--data")
    by_name["train"].add_argument("--resume")
    by_name["predict-eval"].add_argument("--data")
    by_name["predict-eval"].add_argument("--checkpoint", required=True)
    by_name["calibrate"].add_argument("--predictions", required=True)
    by_name["calibrate"].add_argument("--checkpoint", required=True)
    by_name["calibrate"].add_argument("--max-check-rate", type=float, default=0.25)
    for name in ("evaluate", "probe"):
        by_name[name].add_argument("--checkpoint")
    for name in ("evaluate", "collect"):
        by_name[name].add_argument("--task-ids", nargs="+")
        by_name[name].add_argument("--task-file", help="JSON list or one task ID per line")
        by_name[name].add_argument("--split")
    for name in ("counterfactual", "probe"):
        by_name[name].add_argument("--plans")
    by_name["doctor"].add_argument("--require-live", action="store_true")
    agg = sub.add_parser("aggregate")
    agg.add_argument("--inputs", nargs="+", required=True)
    agg.add_argument("--output", required=True)
    agg.add_argument("--bootstrap-samples", type=int, default=1000)
    agg.add_argument("--seed", type=int, default=100)
    check = sub.add_parser("validate-data")
    check.add_argument("--inputs", nargs="+", required=True)
    check.add_argument("--allow-synthetic", action="store_true")
    return p


def _task_ids(args: argparse.Namespace, config: dict, section: str) -> list[str]:
    ids = args.task_ids or config[section].get("task_ids", [])
    if args.task_file:
        text = Path(args.task_file).read_text(encoding="utf-8")
        ids = json.loads(text) if text.lstrip().startswith("[") else text.splitlines()
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError("task IDs must be a list of strings")
    return list(dict.fromkeys(x.strip() for x in ids if x.strip()))


def doctor(config: dict) -> dict:
    import torch
    backend = config["environment"]["backend"]
    root = config["environment"].get("root") or os.environ.get("APPWORLD_ROOT")
    checks = {"torch_import": True, "python_supported": sys.version_info >= (3, 11),
              "linux_server": platform.system() == "Linux",
              "cuda_available": torch.cuda.is_available(),
              "appworld_installed": importlib.util.find_spec("appworld") is not None,
              "appworld_root_exists": bool(root and Path(root).is_dir()),
              "actor_model_configured": bool(config["actor"].get("model") or os.environ.get("ACTOR_MODEL")),
              "actor_key_configured": bool(os.environ.get(config["actor"].get("api_key_env", "OPENAI_API_KEY"))),
              "observation_mapping_configured": bool(config["environment"].get("observation_map") or config["environment"].get("observations")),
              "goal_mapping_configured": bool(config["data"].get("goal_map") or config.get("goals"))}
    return {"backend": backend, "checks": checks,
            "local_code_ready": checks["torch_import"] and checks["python_supported"],
            "live_ready": backend == "appworld" and all(checks.values()),
            "torch_version": torch.__version__, "evidence_origin": "engineering_check",
            "is_benchmark_result": False}


def _training_input(config: dict, path: str) -> list[dict]:
    from .data import read_jsonl, validate_dataset
    records = read_jsonl(path)
    if not records:
        raise ValueError("Training data is empty")
    if any(r["split"] != "train" for r in records):
        raise ValueError("Only official train records may update Predictor parameters")
    validate_dataset(records, allow_synthetic=config["data"]["allow_synthetic"])
    dev = config["data"].get("dev")
    if dev and Path(dev).is_file():
        # Check sibling/source/group isolation together, not one file at a time.
        validate_dataset(records + read_jsonl(dev),
                         allow_synthetic=config["data"]["allow_synthetic"])
    return records


def smoke(config: dict, output_dir: Path) -> dict:
    import math
    import torch
    from .fixtures import fixture_records
    from .data import validate_dataset, write_jsonl
    from .training import train_model, load_predictor
    from .model import predict_record
    from .evaluation import evaluate_run
    config = merge(config, {"environment": {"backend": "fixture", "offline_audit": True},
        "actor": {"kind": "fixture"}, "data": {"allow_synthetic": True, "dev": None},
        "model": {"d_model": 32, "n_heads": 4, "relation_layers": 1,
                  "temporal_layers": 1, "history": 4, "scope_budget": 4,
                  "candidate_limit": 16, "feature_dim": 16, "value_slots": 16},
        "train": {"epochs": 1, "micro_batch": 2, "grad_accum": 1,
                  "max_steps": 5, "device": "cpu", "precision": "fp32"},
        "controller": {"max_steps": 4, "max_calls": 20, "mc_samples": 2},
        "evaluation": {"split": "train", "task_ids": ["fixture-smoke-eval"]}})
    torch.set_num_threads(1)
    records = fixture_records(num_tasks=2)
    validation = validate_dataset(records, allow_synthetic=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "synthetic_transitions.jsonl", records)
    trained = train_model(records, config, output_dir / "training")
    checkpoint = _checkpoint_path(trained, output_dir / "training")
    model, tensorizer, saved_config = load_predictor(checkpoint, device="cpu")
    predictions = []
    record = deepcopy(records[0])
    for offset in range(4):
        prediction = predict_record(model, tensorizer, record, device="cpu", mc_samples=0)
        if not math.isfinite(float(prediction["escape"])):
            raise RuntimeError("Non-finite escape probability")
        predictions.append(prediction)
        record["history"] = (record.get("history", []) + [{"state": record["state"], "action": record["action"]}])[-4:]
        record["state"] = prediction["next_state"]
        record["action"] = records[min(offset + 1, 4)]["action"]
    evaluated = evaluate_run(config, str(checkpoint), ["fixture-smoke-eval"], str(output_dir / "evaluation"))
    result = {"passed": True, "scope": "bounded local engineering smoke only",
              "evidence_origin": "synthetic", "benchmark_evidence": False,
              "real_appworld_executed": False, "gpu_executed": False,
              "dataset": validation, "training": trained,
              "hard_rollout_steps": len(predictions), "controller": evaluated,
              "checkpoint": str(checkpoint)}
    write_json(output_dir / "smoke_report.json", result)
    return result


def _checkpoint_path(trained: Any, folder: Path) -> Path:
    if isinstance(trained, dict):
        for key in ("checkpoint", "checkpoint_path"):
            if trained.get(key):
                return Path(trained[key])
    for name in ("checkpoint.pt", "last.pt", "model.pt"):
        if (folder / name).is_file():
            return folder / name
    raise RuntimeError("Training did not produce a checkpoint")


def run(args: argparse.Namespace) -> Any:
    if args.command == "validate-data":
        from .data import read_jsonl, validate_dataset
        records = [r for path in args.inputs for r in read_jsonl(path)]
        return validate_dataset(records, allow_synthetic=args.allow_synthetic)
    if args.command == "aggregate":
        from .metrics import aggregate_runs
        values = [json.loads(Path(path).read_text()) for path in args.inputs]
        result = aggregate_runs(values, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        write_json(args.output, result)
        return result
    config = load_config(args.config, args.set, args.variant)
    if args.seed is not None:
        config["train"]["seed"] = config["evaluation"]["seed"] = args.seed
    if os.environ.get("APPWORLD_ROOT") and not config["environment"].get("root"):
        config["environment"]["root"] = os.environ["APPWORLD_ROOT"]
    if os.environ.get("ACTOR_MODEL") and not config["actor"].get("model"):
        config["actor"]["model"] = os.environ["ACTOR_MODEL"]
    if args.command in ("collect", "evaluate"):
        section = "collection" if args.command == "collect" else "evaluation"
        if args.split:
            config[section]["split"] = "test_normal" if args.split == "test" else args.split
        config[section]["task_ids"] = _task_ids(args, config, section)
    if args.command == "train" and args.data:
        config["data"]["train"] = args.data
    if args.command == "predict-eval" and args.data:
        config["data"]["dev"] = args.data
    if args.command in ("counterfactual", "probe") and args.plans:
        config["data"]["counterfactual_plans"] = args.plans
        config["probing"]["plans"] = args.plans
    if args.dry_run:
        return {"command": args.command, "config": config,
                "output_dir": args.output_dir, "checkpoint": getattr(args, "checkpoint", None),
                "will_execute": False}
    out = Path(args.output_dir)
    if args.command == "doctor":
        result = doctor(config)
        if args.require_live and not result["live_ready"]:
            raise RuntimeError("Live server requirements are incomplete: " + json.dumps(result["checks"]))
        return result
    if args.command == "smoke":
        return smoke(config, out)
    if (out / 'command_summary.json').exists() and not getattr(args, 'resume', None):
        raise FileExistsError("Output contains an existing run; use a new output directory or --resume for training")
    out.mkdir(parents=True, exist_ok=True)
    dump_config(config, out / "resolved_config.yaml")
    write_json(out / "run_receipt.json", receipt(config))
    if args.command == "train":
        from .training import train_model
        records = _training_input(config, config["data"]["train"])
        result = train_model(records, config, out, resume=args.resume)
        write_json(out / "run_receipt.json", receipt(config, records))
    elif args.command == "collect":
        from .collection import collect
        result = collect(config, out)
    elif args.command == "counterfactual":
        from .collection import collect_counterfactual
        plan_path = config["data"].get("counterfactual_plans")
        if not plan_path:
            raise ValueError("counterfactual requires --plans or data.counterfactual_plans")
        plans = json.loads(Path(plan_path).read_text())
        result = collect_counterfactual(config, plans, out)
    elif args.command == "evaluate":
        from .evaluation import evaluate_run
        if not args.checkpoint and not config["controller"].get("actor_only", False):
            raise ValueError("evaluate requires --checkpoint except for actor_only")
        result = evaluate_run(config, args.checkpoint, config["evaluation"]["task_ids"], out)
    elif args.command == "predict-eval":
        from .prediction_eval import evaluate_predictions
        from .data import read_jsonl, validate_dataset
        path = config["data"].get("dev")
        if not path:
            raise ValueError("predict-eval requires --data or data.dev recorded transitions")
        records = read_jsonl(path)
        validate_dataset(records, allow_synthetic=config["data"]["allow_synthetic"])
        result = evaluate_predictions(config, args.checkpoint, records, out)
    elif args.command == "calibrate":
        from .calibration import calibrate_threshold, calibration_source
        source = calibration_source(config)
        result = calibrate_threshold(args.predictions, args.checkpoint, out / "calibration.json",
                                     args.max_check_rate, source)
    elif args.command == "probe":
        from .probing import run_probing
        if not args.checkpoint:
            raise ValueError("probe requires --checkpoint")
        result = run_probing(config, args.checkpoint, out)
    else:
        raise ValueError(args.command)
    write_json(out / "command_summary.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    try:
        value = run(args)
        print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, default=str))
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        p.exit(2, f"repairscope: {type(error).__name__}: {error}\n")
