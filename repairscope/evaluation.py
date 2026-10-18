"""Frozen evaluation: no optimizer, no checkpoint branches, no hidden inputs."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from pathlib import Path

from .actor import create_actor
from .control import RepairController
from .metrics import aggregate_runs, official_metrics


def _dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def create_run_context(output_dir, config, base="repairscope"):
    """Reserve evidence paths and a unique native AppWorld experiment namespace."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Existing run evidence at {output}; use a new output directory")
    run_id = uuid.uuid4().hex[:12]
    location = hashlib.sha256(str(output).encode()).hexdigest()[:10]
    safe = lambda text: re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip(".-_") or "run"
    variant = safe(config.get("variant", "full"))[:24]
    seed = config.get("train", {}).get("seed", config.get("evaluation", {}).get("seed", 0))
    experiment = f"{safe(base)[:32]}-{variant}-s{safe(seed)}-{location}-{run_id}"
    context = {"run_id": run_id, "experiment": experiment, "output_dir": str(output),
               "variant": variant, "seed": seed}
    with (output / "run_manifest.json").open("x") as stream:
        json.dump(context, stream, ensure_ascii=False, indent=2)
    return context


def evaluate_run(config, checkpoint, task_ids, output_dir):
    """Evaluate every planned task once with a fixed checkpoint and policy.

    The environment's official evaluation runs separately from online control.
    Native TGC/SGC require all planned tasks and official tracker outputs.
    """
    from .environment import create_environment, official_task_ids
    from .calibration import apply_calibration
    config = apply_calibration(config, checkpoint)
    evaluation = config.get("evaluation", {})
    split = evaluation.get("split", "test_normal")
    task_ids = list(task_ids or evaluation.get("task_ids", []))
    if not task_ids:
        root = config.get("environment", {}).get("root")
        if config.get("environment", {}).get("backend", "appworld") == "appworld" and root:
            task_ids = official_task_ids(root, split)
        else:
            raise ValueError("Evaluation needs explicit task IDs or an installed AppWorld root with the complete official split")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_context = create_run_context(output, config, evaluation.get("experiment_name", "repairscope"))
    actor = create_actor(config)
    predictor = None
    actor_only = config.get("controller", {}).get("actor_only", False) or evaluation.get("method", "") == "actor" or config.get("variant") == "actor_only"
    device = config.get("train", {}).get("device", config.get("device", "cpu"))
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint and not actor_only:
        from .training import load_predictor
        from .model import predict_record
        model, tensorizer, model_config = load_predictor(checkpoint, device)
        manifest = model_config.get("checkpoint_manifest", {})
        origins = manifest.get("evidence_origin", [])
        origins = [origins] if isinstance(origins, str) else origins
        if config.get("environment", {}).get("backend", "appworld") == "appworld" and (
                manifest.get("backend") == "synthetic" or any("synthetic" in str(o) or "fixture" in str(o) for o in origins)):
            raise ValueError("A synthetic/fixture checkpoint cannot be used for official AppWorld evaluation")
        model.eval()
        predictor = lambda record: predict_record(model, tensorizer, record,
            device=device, mc_samples=int(config.get("controller", {}).get("mc_samples", 0)))
    elif not actor_only and config.get("method", "") not in {"actor", "A0"}:
        raise ValueError("Predictor evaluation requires a checkpoint; explicitly select method=actor for A0")
    results = []
    frozen_config = copy.deepcopy(config)
    frozen_config.setdefault("environment", {})["offline_audit"] = False
    for task_id in task_ids:
        env = create_environment(frozen_config, str(task_id), split, run_context["experiment"])
        try:
            controller_options = {**config.get("controller", {}),
                                  "use_goal_head": float(config.get("loss", {}).get("goal", 1)) > 0}
            controller = RepairController(actor, predictor, controller_options)
            result = controller.run(env)
            # This receipt is never fed back to actor/predictor or a replay buffer.
            official = env.evaluate()
            result.update(task_id=str(task_id), group_id=str(task_id).split("_")[0], split=split,
                          evidence_origin=getattr(env, "evidence_origin", getattr(env, "metadata", {}).get("evidence_origin", "unknown")),
                          official=official, model_frozen=True, test_time_updates=0, test_time_branches=0)
            result["experiment"] = run_context["experiment"]
            results.append(result)
            safe_name = str(task_id).replace("/", "_").replace("..", "_")
            _dump(output / f"{safe_name}.json", result)
        finally:
            env.close()
    summary = aggregate_runs(results, int(evaluation.get("bootstrap_samples", 1000)), int(evaluation.get("seed", 0)))
    summary.update(official_metrics(results, task_ids))
    summary.update(split=split, checkpoint=str(checkpoint) if checkpoint else None, task_ids=task_ids,
                   evidence_origins=sorted({r["evidence_origin"] for r in results}),
                   test_time_updates=0, test_time_branches=0)
    summary.update(run_context)
    _dump(output / "summary.json", summary)
    return summary
