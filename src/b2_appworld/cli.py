"""Explicit stages: no benchmark/data/model download at import time."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys

from .config import load_config
from .io import atomic_json, read_jsonl


def _parser():
    parser = argparse.ArgumentParser(prog="repairscope", description="RepairScope / AppWorld research toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect dependencies, pinned source, data, GPU; no downloads")
    doctor.add_argument("--config")
    doctor.add_argument("--output")
    manifest = sub.add_parser("manifest", help="Verify exact dataset IDs/scenarios/versions")
    manifest.add_argument("--config")
    manifest.add_argument("--output", default="runs/dataset_manifest.json")
    run = sub.add_parser("run", help="Execute official ReAct plus selected online method")
    run.add_argument("--config", default="configs/appworld.toml")
    run.add_argument("--split", choices=("train", "dev", "test_normal", "test_challenge"), required=True)
    run.add_argument("--method", choices=("baseline", "capture", "direct", "rule", "scalar", "effect"), required=True)
    run.add_argument("--experiment", required=True)
    run.add_argument("--checkpoint")
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--final", action="store_true", help="Allow full frozen test execution")
    native = sub.add_parser("evaluate-native", help="Official whole-task/scenario aggregate metrics")
    native.add_argument("--config", default="configs/appworld.toml")
    native.add_argument("--experiment", required=True)
    mapping = sub.add_parser("map-goals", help="Offline train/dev-only requirement grouping")
    mapping.add_argument("--config", default="configs/appworld.toml")
    mapping.add_argument("--trajectories", required=True)
    mapping.add_argument("--output", required=True)
    collect = sub.add_parser("collect", help="Fresh-process branch replay; train/dev only")
    collect.add_argument("--config", default="configs/appworld.toml")
    collect.add_argument("--trajectories", required=True)
    collect.add_argument("--mapping", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--timeout", type=float, default=300)
    train = sub.add_parser("train", help="Train QLoRA effect/scalar head, or explicit CPU fixture")
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--kind", choices=("effect", "scalar"))
    train.add_argument("--output-dir")
    train.add_argument("--fold", type=int)
    evaluate = sub.add_parser("evaluate-predictor", help="Held-out train/dev mechanism diagnostics")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda")
    folds = sub.add_parser("folds", help="Whole-scenario train-side fold manifest")
    folds.add_argument("--data", required=True)
    folds.add_argument("--output", required=True)
    folds.add_argument("--seed", type=int, default=100)
    folds.add_argument("--n-folds", type=int, default=5)
    quality = sub.add_parser("quality", help="Label coverage/pool diagnostics; never TGC/SGC")
    quality.add_argument("--data", required=True)
    quality.add_argument("--output", required=True)
    smoke = sub.add_parser("fixture-smoke", help="Synthetic CPU integrity test; no research claims")
    smoke.add_argument("--output-dir", default="runs/fixture_smoke")
    return parser


def doctor(config):
    modules = {m: importlib.util.find_spec(m) is not None
               for m in ("appworld", "appworld_agents", "torch", "transformers", "peft", "bitsandbytes")}
    result = {"python": sys.version.split()[0], "platform": platform.platform(),
              "dependencies": modules, "root": config["root"], "synthetic_is_not_live_validation": True}
    from .runner import verify_upstream
    from .datasets import dataset_manifest
    for name, function in (("upstream", verify_upstream), ("dataset", dataset_manifest)):
        try:
            result[name] = function(config["root"])
        except (OSError, ValueError, KeyError, __import__("subprocess").SubprocessError) as exc:
            result[name] = {"ready": False, "reason": str(exc)}
    if modules["torch"]:
        try:
            import torch
            result["torch_version"] = torch.__version__
            result["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                result["gpu"] = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info(0)
                result["gpu_free_bytes"], result["gpu_total_bytes"] = free, total
        except (ImportError, OSError, RuntimeError) as exc:
            result["torch_error"] = type(exc).__name__
            result["cuda_available"] = False
    root = Path(config["root"])
    result["appworld_unpacked"] = (root / "src/appworld/apps/api_docs").is_dir() and (root / "tests/package/apps").is_dir()
    result["live_ready"] = (all(modules[m] for m in ("appworld", "appworld_agents", "torch", "transformers", "peft", "bitsandbytes"))
                            and isinstance(result.get("upstream"), str)
                            and "splits" in result.get("dataset", {}) and result.get("cuda_available", False)
                            and result["appworld_unpacked"])
    result["live_ready_scope"] = "Static prerequisites only; actor service, 4B loading, memory coexistence and real task execution remain unverified"
    return result


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        result = doctor(load_config(args.config))
        if args.output:
            atomic_json(args.output, result)
    elif args.command == "manifest":
        from .datasets import dataset_manifest
        result = dataset_manifest(load_config(args.config)["root"])
        atomic_json(args.output, result)
    elif args.command == "run":
        from .runner import run_experiment
        config = load_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        result = run_experiment(config, args.split, args.method, args.experiment,
                                args.checkpoint, args.limit, args.final)
    elif args.command == "evaluate-native":
        from .runner import evaluate_native
        result = evaluate_native(load_config(args.config), args.experiment)
    elif args.command == "map-goals":
        from .collection import build_mappings_from_trajectories
        from .llm import ChatClient, strict_json_loads
        from transformers import AutoTokenizer
        config = load_config(args.config)
        actor = config["actor"]
        tokenizer = AutoTokenizer.from_pretrained(actor["tokenizer"],
                        revision=actor.get("tokenizer_revision"), trust_remote_code=False)
        client = ChatClient(actor["base_url"], actor["model"], actor["api_key_env"], tokenizer=tokenizer,
                            chat_template_kwargs={"enable_thinking": False}, context_length=actor["context_length"])

        def mapper(**values):
            prompt = (
                "Offline TRAIN/DEV annotation only. Group the supplied exact evaluator requirement strings "
                "under the supplied instruction-grounded final state goals. Return only JSON "
                '{"goal_requirements":{"g1":{"requirements":["exact string"],"complete":true}}}. '
                "A group must be sufficient to decide that ONE goal and exclusive to it. Global/joint checks "
                "that also require another independent goal must remain unmapped. Never reuse a requirement "
                "across goals. Mark complete false when coverage/sufficiency is ambiguous; never guess. "
                "Answer/sequence goals cannot be annotated as persistent state. Do not invent checker strings."
            )
            response = client.complete([{"role": "system", "content": prompt},
                                        {"role": "user", "content": json.dumps(values)}],
                                       max_tokens=2000, purpose="offline_requirement_mapping")
            return strict_json_loads(response.content)
        result = build_mappings_from_trajectories(args.trajectories, args.output, config["root"],
                                                 mapper, seed=config["seed"])
        atomic_json(str(args.output) + ".usage.json", client.usage)
    elif args.command == "collect":
        from .collection import collect_from_trajectories
        config = load_config(args.config)
        result = collect_from_trajectories(args.trajectories, args.mapping, args.output,
                                           config["root"], seed=config["seed"], timeout=args.timeout)
    elif args.command == "train":
        from .training import train_from_config
        config = json.loads(Path(args.config).read_text())
        for key, value in (("seed", args.seed), ("kind", args.kind), ("output_dir", args.output_dir)):
            if value is not None:
                config[key] = value
        if args.fold is not None:
            config.pop("dev_data", None)
            config["validation_fold"] = args.fold
        result = train_from_config(config)
    elif args.command == "evaluate-predictor":
        from .training import evaluate_checkpoint
        result = evaluate_checkpoint(args.checkpoint, args.data, args.output, device=args.device)
    elif args.command == "folds":
        from .training import scenario_folds
        from .schemas import Example, ensure_research_split
        examples = [Example.from_dict(row) for row in read_jsonl(args.data)]
        if any(ex.split != "train" for ex in examples):
            raise ValueError("Fold diagnostics require train data only")
        result = {"seed": args.seed, "folds": scenario_folds(examples, args.n_folds, args.seed)}
        atomic_json(args.output, result)
    elif args.command == "quality":
        from .datasets import collection_quality_report
        result = collection_quality_report(args.data)
        atomic_json(args.output, result)
    elif args.command == "fixture-smoke":
        from .smoke import fixture_smoke
        result = fixture_smoke(args.output_dir)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
