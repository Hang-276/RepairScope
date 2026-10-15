"""Scenario-isolated supervised effect/scalar training and held-out diagnostics.

Only train records are optimized. Official dev records may be used for early
stopping; alternatively all rows of held-out train scenarios form validation.
No native test labels or native task-success aggregate enter this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from .io import atomic_json, read_jsonl
from .learning import create_model_bundle, load_predictor, require_torch, save_predictor_checkpoint
from .metrics import candidate_cost_denominators, evaluate_predictions, label_utility, state_key
from .model_inputs import encode_effect_input
from .schemas import Example, IGNORE_LABEL


def scenario_folds(examples: Sequence[Example], n_folds: int = 5,
                   seed: int = 100) -> list[list[str]]:
    """Whole-scenario folds: tasks, states and candidate branches cannot straddle folds."""
    scenarios = sorted({example.scenario_id for example in examples})
    if n_folds < 2 or len(scenarios) < n_folds:
        raise ValueError(f"Need at least {n_folds} scenarios for {n_folds}-fold validation; got {len(scenarios)}")
    random.Random(seed).shuffle(scenarios)
    return [sorted(scenarios[i::n_folds]) for i in range(n_folds)]


def split_scenario_holdout(examples: Sequence[Example], fold: int = 0,
                           n_folds: int = 5, seed: int = 100,
                           holdout_scenario_ids: Sequence[str] | None = None) -> tuple[list[Example], list[Example]]:
    if any(example.split != "train" for example in examples):
        raise ValueError("Scenario holdout must be constructed exclusively from train records")
    all_scenarios = {ex.scenario_id for ex in examples}
    if holdout_scenario_ids is None:
        if not 0 <= fold < n_folds:
            raise ValueError("Validation fold is out of range")
        heldout = set(scenario_folds(examples, n_folds, seed)[fold])
    else:
        heldout = set(holdout_scenario_ids)
        if not heldout or not heldout < all_scenarios:
            raise ValueError("Holdout scenarios must be a nonempty proper subset of train scenarios")
    train = [ex for ex in examples if ex.scenario_id not in heldout]
    validation = [ex for ex in examples if ex.scenario_id in heldout]
    return train, validation


def _load_examples(path: str | Path, required_split: str | None = None) -> list[Example]:
    examples = [Example.from_dict(row) for row in read_jsonl(path)]
    if not examples:
        raise ValueError(f"No examples in {path}")
    if required_split and any(ex.split != required_split for ex in examples):
        raise ValueError(f"{path}: expected exclusively split={required_split}")
    # Check pool integrity before filtering masked labels or assigning folds.
    candidate_cost_denominators(examples)
    return examples


def _dataset_digest(examples: Sequence[Example]) -> str:
    digest = hashlib.sha256()
    for ex in examples:
        digest.update(json.dumps(ex.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _prepare(examples: Sequence[Example], tokenizer: Any, config: Mapping[str, Any],
             kind: str) -> tuple[list[dict[str, Any]], int]:
    denominators = candidate_cost_denominators(examples)
    prepared = []
    excluded = 0
    complete_only = bool(config.get("require_complete_labels", True))
    for ex in examples:
        known = [label for label in ex.labels if label != IGNORE_LABEL]
        if not known or ((kind == "scalar" or complete_only) and len(known) != len(ex.labels)):
            excluded += 1
            continue
        cost = ex.candidate.estimated_api_calls / denominators[state_key(ex)]
        encoded = encode_effect_input(
            tokenizer, ex.instruction, ex.history, ex.goals, ex.candidate,
            max_length=int(config.get("max_length", 4096)), normalized_cost=cost,
            harm_weight=float(config.get("harm_weight", 2)),
            cost_weight=float(config.get("cost_weight", 0.05)))
        target: Any = list(ex.labels) if kind == "effect" else label_utility(
            ex.labels, cost, float(config.get("harm_weight", 2)), float(config.get("cost_weight", 0.05)))
        prepared.append({"input": encoded, "target": target})
    if not prepared:
        raise ValueError(f"No usable {kind} training/validation records after masking unknown labels")
    return prepared, excluded


def _batch(torch: Any, records: Sequence[Mapping[str, Any]], pad_id: int, device: str,
           kind: str) -> tuple[dict[str, Any], Any]:
    longest = max(len(row["input"].input_ids) for row in records)
    max_goals = max(len(row["input"].goal_positions) for row in records)
    ids = []
    attention = []
    positions = []
    masks = []
    targets = []
    for row in records:
        encoded = row["input"]
        padding = longest - len(encoded.input_ids)
        goal_padding = max_goals - len(encoded.goal_positions)
        ids.append(encoded.input_ids + [pad_id] * padding)
        attention.append(encoded.attention_mask + [0] * padding)
        positions.append(encoded.goal_positions + [0] * goal_padding)
        masks.append([1] * len(encoded.goal_positions) + [0] * goal_padding)
        targets.append(row["target"] + [IGNORE_LABEL] * goal_padding if kind == "effect" else row["target"])
    inputs = {"input_ids": torch.tensor(ids, device=device),
              "attention_mask": torch.tensor(attention, device=device),
              "goal_positions": torch.tensor(positions, device=device),
              "goal_mask": torch.tensor(masks, device=device)}
    target = torch.tensor(targets, dtype=torch.long if kind == "effect" else torch.float32, device=device)
    return inputs, target


def _loss(torch: Any, output: Any, target: Any, kind: str, weights: Any = None) -> Any:
    if kind == "effect":
        return torch.nn.functional.cross_entropy(output.reshape(-1, 4), target.reshape(-1),
                                                weight=weights, ignore_index=IGNORE_LABEL)
    return torch.nn.functional.mse_loss(output, target)


def _read_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as stream:
            return dict(json.load(stream))
    if not isinstance(config, Mapping):
        raise TypeError("Training config must be a mapping or JSON path")
    return dict(config)


def train_from_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Train an effect or scalar checkpoint; stable entry point for the CLI.

    Required: train_data, output_dir. Defaults are the 4B NF4+LoRA main recipe.
    ``dev_data`` selects official dev validation. Without it, ``validation_fold``
    (default 0) holds out whole scenarios from the train file using five folds.
    """
    total_started = time.perf_counter()
    config = _read_config(config)
    for key in ("train_data", "output_dir"):
        if not config.get(key):
            raise ValueError(f"Missing training configuration: {key}")
    output = Path(config["output_dir"])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Training output already exists and is nonempty: {output}. Use a new run directory.")
    kind = str(config.get("kind", "effect"))
    if kind not in {"effect", "scalar"}:
        raise ValueError("kind must be effect or scalar")
    batch_size = int(config.get("batch_size", 1))
    accumulation = int(config.get("gradient_accumulation", 16))
    epochs = int(config.get("epochs", 3))
    patience = int(config.get("patience", 2))
    learning_rate = float(config.get("lr", 2e-4))
    head_learning_rate = float(config.get("head_lr", learning_rate))
    scheduler_name = str(config.get("scheduler", "cosine"))
    warmup_ratio = float(config.get("warmup_ratio", 0.0 if config.get("backend") == "tiny" else 0.03))
    if scheduler_name not in {"cosine", "constant"} or not 0 <= warmup_ratio < 1:
        raise ValueError("scheduler must be cosine/constant and warmup_ratio must be in [0,1)")
    if min(batch_size, accumulation, epochs, patience) < 1 or not 0 < learning_rate < 1 or not 0 < head_learning_rate < 1:
        raise ValueError("Invalid optimizer/batch/epoch settings")
    if any(not math.isfinite(float(config.get(key, default))) or float(config.get(key, default)) < 0
           for key, default in (("harm_weight", 2), ("cost_weight", 0.05))):
        raise ValueError("Utility weights must be finite and nonnegative")
    all_train = _load_examples(config["train_data"], "train")
    if config.get("dev_data"):
        train = all_train
        validation = _load_examples(config["dev_data"], "dev")
        overlap = {ex.scenario_id for ex in train} & {ex.scenario_id for ex in validation}
        if overlap:
            raise ValueError(f"Training/validation scenario leakage: {sorted(overlap)}")
        validation_source = "official_dev"
    else:
        train, validation = split_scenario_holdout(
            all_train, fold=int(config.get("validation_fold", 0)),
            n_folds=int(config.get("n_folds", 5)), seed=int(config.get("split_seed", 100)),
            holdout_scenario_ids=config.get("holdout_scenario_ids"))
        validation_source = "train_scenario_holdout"
    backend = str(config.get("backend", "hf"))
    expected_origin = "synthetic_fixture" if backend == "tiny" else "natural"
    if any(ex.source != expected_origin for ex in train + validation):
        raise ValueError(f"backend={backend} requires exclusively source={expected_origin}; "
                         "synthetic faults/fixtures cannot be mixed into a natural main experiment")
    torch = require_torch()
    seed = int(config.get("seed", 100))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, tokenizer, metadata = create_model_bundle(config)
    device = str(config.get("device", "cuda" if metadata["backend"] == "hf" else "cpu"))
    prepared_train, excluded_train = _prepare(train, tokenizer, config, kind)
    prepared_validation, excluded_validation = _prepare(validation, tokenizer, config, kind)
    complete_only = kind == "scalar" or bool(config.get("require_complete_labels", True))
    label_counts = Counter(label for ex in train
                           if not complete_only or IGNORE_LABEL not in ex.labels
                           for label in ex.labels if label != IGNORE_LABEL)
    class_weights = None
    weight_setting = config.get("class_weights")
    if weight_setting == "inverse_frequency":
        # Only training labels determine the weighting. Absent classes get zero weight.
        values = [1 / label_counts[c] if label_counts[c] else 0.0 for c in range(4)]
        scale = 4 / sum(values)
        class_weights = torch.tensor([x * scale for x in values], device=device)
    elif weight_setting is not None:
        if not isinstance(weight_setting, list) or len(weight_setting) != 4 or any(float(x) <= 0 for x in weight_setting):
            raise ValueError("class_weights must be four positive values or inverse_frequency")
        class_weights = torch.tensor(weight_setting, dtype=torch.float32, device=device)
    if kind == "scalar" and weight_setting is not None:
        raise ValueError("Class weights apply only to the effect cross-entropy head")
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": learning_rate},
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": head_learning_rate},
    ], weight_decay=float(config.get("weight_decay", 0.01)))
    steps_per_epoch = math.ceil(math.ceil(len(prepared_train) / batch_size) / accumulation)
    total_scheduled_steps = steps_per_epoch * epochs
    warmup_steps = math.ceil(total_scheduled_steps * warmup_ratio)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        if scheduler_name == "constant":
            return 1.0
        progress = min(1.0, (step - warmup_steps) / max(1, total_scheduled_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    output.mkdir(parents=True, exist_ok=True)
    pool_costs = {}
    for name, examples in (("train", train), ("validation", validation)):
        pool_costs[name] = [{"scenario_id": key[0], "task_id": key[1], "state_id": key[2],
                             "denominator": value}
                            for key, value in sorted(candidate_cost_denominators(examples).items())]
    metadata.update({"candidate_cost_denominators_before_label_filter": pool_costs,
                     "training_config": config, "training_scenarios": sorted({x.scenario_id for x in train}),
                     "validation_scenarios": sorted({x.scenario_id for x in validation}),
                     "validation_source": validation_source, "train_digest": _dataset_digest(train),
                     "validation_digest": _dataset_digest(validation), "train_label_counts": dict(label_counts),
                     "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
                     "optimizer": {"name": "AdamW", "encoder_lr": learning_rate, "head_lr": head_learning_rate,
                                   "scheduler": scheduler_name, "warmup_ratio": warmup_ratio,
                                   "warmup_steps": warmup_steps, "total_scheduled_steps": total_scheduled_steps},
                     "early_stopping_metric": "unweighted_validation_loss",
                     "loss": "masked_four_state_cross_entropy" if kind == "effect" else "complete_goal_utility_mse",
                     "require_complete_labels": bool(config.get("require_complete_labels", True)),
                     "unknown_label_policy": "masked" if kind == "effect" and not config.get("require_complete_labels", True) else "exclude_incomplete_rows"})
    atomic_json(output / "training_config.json", config)
    atomic_json(output / "split_manifest.json", {"training_scenarios": metadata["training_scenarios"],
                "validation_scenarios": metadata["validation_scenarios"], "validation_source": validation_source,
                "train_digest": metadata["train_digest"], "validation_digest": metadata["validation_digest"]})
    best = math.inf
    best_epoch = 0
    bad_epochs = 0
    optimizer_steps = 0
    history = []
    processed_tokens = 0
    non_padding_tokens = 0
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    setup_seconds = started - total_started
    rng = random.Random(seed)
    pad_id = int(tokenizer.pad_token_id)
    for epoch in range(1, epochs + 1):
        order = list(range(len(prepared_train)))
        rng.shuffle(order)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        train_count = 0
        pending = 0
        for step, indices in enumerate(batches, 1):
            inputs, target = _batch(torch, [prepared_train[i] for i in indices], pad_id, device, kind)
            loss = _loss(torch, model(**inputs), target, kind, class_weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite training loss; checkpoint is not considered trained")
            (loss / accumulation).backward()
            pending += 1
            train_total += float(loss.detach()) * len(indices)
            train_count += len(indices)
            processed_tokens += inputs["input_ids"].numel()
            non_padding_tokens += int(inputs["attention_mask"].sum())
            if pending == accumulation or step == len(batches):
                if pending < accumulation:
                    # Keep the final short accumulation window at the same gradient scale.
                    for parameter in trainable_parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(accumulation / pending)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, float(config.get("max_grad_norm", 1)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending = 0
        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.inference_mode():
            for start in range(0, len(prepared_validation), batch_size):
                records = prepared_validation[start:start + batch_size]
                inputs, target = _batch(torch, records, pad_id, device, kind)
                # Unweighted validation avoids choosing on a distorted class-prior loss.
                loss = _loss(torch, model(**inputs), target, kind)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite validation loss")
                validation_total += float(loss) * len(records)
                validation_count += len(records)
                processed_tokens += inputs["input_ids"].numel()
                non_padding_tokens += int(inputs["attention_mask"].sum())
        validation_loss = validation_total / validation_count
        entry = {"epoch": epoch, "train_loss": train_total / train_count,
                 "validation_loss": validation_loss, "optimizer_steps": optimizer_steps,
                 "learning_rates_after_epoch": scheduler.get_last_lr()}
        history.append(entry)
        atomic_json(output / "training_history.json", history)
        if validation_loss < best - float(config.get("min_delta", 0)):
            best = validation_loss
            best_epoch = epoch
            bad_epochs = 0
            metadata.update({"best_epoch": epoch, "best_validation_loss": best,
                             "optimizer_steps_at_best": optimizer_steps})
            save_predictor_checkpoint(model, tokenizer, metadata, output / "best")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    summary = {"checkpoint": str(output / "best"), "kind": kind, "backend": metadata["backend"],
               "fixture_backend": metadata["fixture_backend"], "best_epoch": best_epoch,
               "best_validation_loss": best, "epochs_completed": len(history),
               "training_records": len(prepared_train), "validation_records": len(prepared_validation),
               "excluded_unknown_training_rows": excluded_train, "excluded_unknown_validation_rows": excluded_validation,
               "processed_tokens_including_padding": processed_tokens, "non_padding_tokens": non_padding_tokens,
               "seconds": time.perf_counter() - started, "setup_seconds": setup_seconds,
               "total_seconds": time.perf_counter() - total_started,
               "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else None,
               "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else None,
               "optimizer_steps": optimizer_steps,
               "validation_source": validation_source,
               "require_complete_labels": bool(config.get("require_complete_labels", True)),
               "source_counts": dict(Counter(ex.source for ex in train + validation)),
               "trainable_parameters": metadata["trainable_parameters"], "history": history}
    atomic_json(output / "training_summary.json", summary)
    return summary


def evaluate_checkpoint(checkpoint: str | Path, data_path: str | Path,
                        output: str | Path | None = None, *, device: str = "cuda",
                        quantized: bool | None = None) -> dict[str, Any]:
    """Evaluate complete paired pools from held-out train scenarios or official dev."""
    examples = _load_examples(data_path)
    predictor = load_predictor(checkpoint, device=device, quantized=quantized)
    expected_origin = "synthetic_fixture" if predictor.metadata["backend"] == "tiny" else "natural"
    if any(ex.source != expected_origin for ex in examples):
        raise ValueError(f"This checkpoint requires evaluation source={expected_origin}")
    overlap = {ex.scenario_id for ex in examples} & set(predictor.metadata.get("training_scenarios", []))
    if overlap:
        raise ValueError(f"Mechanism evaluation requires held-out scenarios, found trained scenarios {sorted(overlap)}")
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[state_key(ex)].append(i)
    predictions: list[Any] = [None] * len(examples)
    denominators = candidate_cost_denominators(examples)
    for key, indices in sorted(groups.items()):
        first = examples[indices[0]]
        values = predictor.predict(first.instruction, first.history, first.goals,
                                   [examples[i].candidate for i in indices],
                                   cost_denominator=denominators[key])
        for index, value in zip(indices, values):
            predictions[index] = value
    result = evaluate_predictions(examples, predictions, kind=predictor.metadata["kind"],
                                   harm_weight=predictor.metadata["harm_weight"],
                                   cost_weight=predictor.metadata["cost_weight"])
    result.update({"checkpoint": str(checkpoint), "fixture_backend": predictor.metadata["fixture_backend"],
                   "dataset_digest": _dataset_digest(examples), "usage": predictor.usage,
                   "source_counts": dict(Counter(ex.source for ex in examples)),
                   "model_revision": predictor.metadata["revision"]})
    if output:
        atomic_json(output, result)
    return result
