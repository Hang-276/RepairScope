"""Optional Torch/Transformers/PEFT predictors with explicit checkpoint provenance.

Imports are lazy so collecting and evaluating AppWorld does not require CUDA.
The ``tiny`` backend is a random byte-GRU for CPU contract tests, not an
AppWorld baseline and not a substitute for the Qwen experiments.
"""
from __future__ import annotations

import json
import math
import importlib.metadata
import os
import platform
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .io import atomic_json
from .clock import elapsed_time
from .model_inputs import ByteTokenizer, encode_effect_input
from .schemas import Candidate, Goal, LABELS

CHECKPOINT_VERSION = 1


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the 'ml' extra for training/inference (Torch is missing)") from exc
    return torch


def _assert_device(torch: Any, device: str, quantized: bool) -> None:
    if quantized and not str(device).startswith("cuda"):
        raise ValueError("NF4/bitsandbytes loading requires CUDA; use backend=tiny for CPU fixtures")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def _tiny_encoder(torch: Any, hidden_size: int = 32) -> Any:
    class TinyCausalEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(259, hidden_size, padding_idx=0)
            self.recurrent = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)

        def forward(self, input_ids: Any, attention_mask: Any = None, **_: Any) -> Any:
            hidden, _state = self.recurrent(self.embedding(input_ids))
            return SimpleNamespace(last_hidden_state=hidden)
    return TinyCausalEncoder()


def _prediction_model(torch: Any, encoder: Any, hidden_size: int, kind: str) -> Any:
    if kind not in {"effect", "scalar"}:
        raise ValueError("Predictor kind must be effect or scalar")

    class GoalQueryModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Linear(hidden_size, 4 if kind == "effect" else 1)
            self.kind = kind

        def forward(self, input_ids: Any, attention_mask: Any, goal_positions: Any,
                    goal_mask: Any = None) -> Any:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                  use_cache=False, return_dict=True).last_hidden_state
            batch_index = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
            queried = hidden[batch_index, goal_positions]
            # Keep a float32 head and loss even when the encoder uses bf16/NF4.
            queried = queried.float()
            if self.kind == "effect":
                return self.head(queried)
            mask = (goal_mask if goal_mask is not None else torch.ones_like(goal_positions)).float()
            pooled = (queried * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            return self.head(pooled).squeeze(-1)
    return GoalQueryModel()


def _load_hf_base(base_model: str, revision: str | None, *, device: str,
                  quantized: bool, training: bool = False) -> tuple[Any, Any, int, str | None]:
    torch = require_torch()
    _assert_device(torch, device, quantized)
    try:
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Transformers is missing; install the 'ml' extra") from exc
    dtype = torch.bfloat16 if str(device).startswith("cuda") and torch.cuda.is_bf16_supported() else (
        torch.float16 if str(device).startswith("cuda") else torch.float32)
    kwargs: dict[str, Any] = {"revision": revision, "trust_remote_code": False, "torch_dtype": dtype}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
        # Never use auto mapping: it can invisibly offload and change timing/memory.
        kwargs["device_map"] = {"": device}
    encoder = AutoModel.from_pretrained(base_model, **kwargs)
    if not quantized:
        encoder.to(device)
    encoder.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    hidden_size = int(encoder.config.hidden_size)
    resolved = getattr(encoder.config, "_commit_hash", None) or revision
    return encoder, tokenizer, hidden_size, resolved


def create_model_bundle(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Create a trainable LoRA encoder + head (or explicitly synthetic tiny model)."""
    torch = require_torch()
    backend = str(config.get("backend", "hf"))
    kind = str(config.get("kind", "effect"))
    device = str(config.get("device", "cuda" if backend == "hf" else "cpu"))
    quantized = bool(config.get("quantized", backend == "hf"))
    _assert_device(torch, device, quantized)
    metadata: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION, "kind": kind, "backend": backend,
        "label_map": list(LABELS), "max_length": int(config.get("max_length", 4096)),
        "harm_weight": float(config.get("harm_weight", 2)),
        "cost_weight": float(config.get("cost_weight", 0.05)),
        "cost_normalization": "max(1,max_candidate_pool_estimated_api_calls)",
        "trained_quantized": quantized, "input_format": "all_goals_causal_queries_v1",
        "fixture_backend": backend == "tiny", "requested_revision": config.get("revision"),
    }
    if backend == "tiny":
        if quantized:
            raise ValueError("Tiny backend cannot be quantized")
        hidden_size = int(config.get("tiny_hidden_size", 32))
        encoder = _tiny_encoder(torch, hidden_size)
        tokenizer = ByteTokenizer()
        metadata.update({"base_model": "synthetic-byte-gru", "revision": "fixture-v1",
                         "hidden_size": hidden_size})
    elif backend == "hf":
        base_model = str(config.get("base_model", "Qwen/Qwen3-4B"))
        encoder, tokenizer, hidden_size, resolved = _load_hf_base(
            base_model, config.get("revision"), device=device, quantized=quantized, training=True)
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("PEFT is missing; install the 'ml' extra") from exc
        if quantized:
            encoder = prepare_model_for_kbit_training(
                encoder, use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)))
        elif config.get("gradient_checkpointing", True):
            encoder.gradient_checkpointing_enable()
            if hasattr(encoder, "enable_input_require_grads"):
                encoder.enable_input_require_grads()
        lora = {"r": int(config.get("lora_rank", 16)),
                "lora_alpha": int(config.get("lora_alpha", 32)),
                "lora_dropout": float(config.get("lora_dropout", 0.05)),
                "target_modules": list(config.get("lora_target_modules",
                                                   ["q_proj", "k_proj", "v_proj", "o_proj",
                                                    "gate_proj", "up_proj", "down_proj"])),
                "bias": "none", "task_type": "FEATURE_EXTRACTION"}
        encoder = get_peft_model(encoder, LoraConfig(**lora))
        metadata.update({"base_model": base_model, "revision": resolved, "hidden_size": hidden_size,
                         "lora": lora})
        if not resolved:
            raise ValueError("Cannot record the resolved base-model revision; provide an immutable revision")
    else:
        raise ValueError("backend must be hf or tiny")
    model = _prediction_model(torch, encoder, hidden_size, kind)
    model.head.to(device=device, dtype=torch.float32)
    if backend == "tiny":
        model.to(device)
    metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata["library_versions"] = versions
    metadata["python_version"] = platform.python_version()
    metadata["training_device"] = device
    metadata["gpu_name"] = torch.cuda.get_device_name(device) if device.startswith("cuda") else None
    return model, tokenizer, metadata


def save_predictor_checkpoint(model: Any, tokenizer: Any, metadata: Mapping[str, Any],
                              directory: str | Path) -> None:
    """Stage all checkpoint pieces together before replacing the previous best."""
    torch = require_torch()
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + directory.name + "-stage-", dir=directory.parent))
    backup: Path | None = None
    installed = False
    try:
        tokenizer.save_pretrained(stage / "tokenizer")
        if metadata["backend"] == "tiny":
            torch.save(model.encoder.state_dict(), stage / "tiny_encoder.pt")
        else:
            model.encoder.save_pretrained(stage / "adapter", safe_serialization=True)
        torch.save(model.head.state_dict(), stage / "head.pt")
        atomic_json(stage / "metadata.json", dict(metadata))
        if directory.exists():
            backup = Path(tempfile.mkdtemp(prefix="." + directory.name + "-previous-", dir=directory.parent))
            backup.rmdir()
            os.replace(directory, backup)
        try:
            os.replace(stage, directory)
            installed = True
        except BaseException:
            if backup is not None:
                os.replace(backup, directory)
                backup = None
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if installed and backup is not None and backup.exists():
            shutil.rmtree(backup)


def _read_metadata(directory: Path) -> dict[str, Any]:
    with open(directory / "metadata.json", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported predictor checkpoint version")
    if metadata.get("label_map") != list(LABELS):
        raise ValueError("Checkpoint label order differs from 00/01/10/11")
    if metadata.get("kind") not in {"effect", "scalar"}:
        raise ValueError("Unknown checkpoint head kind")
    if metadata.get("input_format") != "all_goals_causal_queries_v1":
        raise ValueError("Checkpoint input encoding is incompatible")
    for field in ("harm_weight", "cost_weight", "max_length", "base_model", "revision"):
        if field not in metadata:
            raise ValueError(f"Checkpoint is missing required provenance field {field}")
    if metadata.get("cost_normalization") != "max(1,max_candidate_pool_estimated_api_calls)":
        raise ValueError("Checkpoint cost normalization is incompatible")
    return metadata


class _Predictor:
    def __init__(self, model: Any, tokenizer: Any, metadata: Mapping[str, Any], device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = dict(metadata)
        self.device = device
        self.last_usage: dict[str, Any] = {}
        self.usage: dict[str, Any] = {"processed_tokens": 0, "seconds": 0.0, "candidates": 0,
                                      "forward_passes": 0, "dropped_history_messages": 0}
        model.eval()

    @staticmethod
    def _cost_denominator(candidates: Sequence[Candidate], declared: float | None = None) -> float:
        observed = max(1.0, max(c.estimated_api_calls for c in candidates))
        if declared is None:
            return observed
        value = float(declared)
        if not math.isfinite(value) or value < observed - 1e-8:
            raise ValueError("Original pool cost denominator cannot be smaller than observed costs")
        return value

    def estimate_tokens(self, instruction: str, history: Any, goals: Sequence[Goal],
                        candidates: Sequence[Candidate], *, cost_denominator: float | None = None) -> int:
        """Exact processed input tokens for the same per-candidate inference path."""
        if not candidates or not goals:
            raise ValueError("At least one goal and candidate are required")
        max_cost = self._cost_denominator(candidates, cost_denominator)
        return sum(len(encode_effect_input(
            self.tokenizer, instruction, history, goals, candidate,
            max_length=int(self.metadata["max_length"]),
            normalized_cost=candidate.estimated_api_calls / max_cost,
            harm_weight=float(self.metadata["harm_weight"]),
            cost_weight=float(self.metadata["cost_weight"])).input_ids) for candidate in candidates)

    def _predict(self, instruction: str, history: Any, goals: Sequence[Goal],
                 candidates: Sequence[Candidate], *, cost_denominator: float | None = None) -> list[Any]:
        torch = require_torch()
        if not goals or not candidates:
            raise ValueError("At least one goal and one candidate are required")
        max_cost = self._cost_denominator(candidates, cost_denominator)
        result = []
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = elapsed_time()
        tokens = dropped = 0
        with torch.inference_mode():
            for candidate in candidates:
                encoded = encode_effect_input(
                    self.tokenizer, instruction, history, goals, candidate,
                    max_length=int(self.metadata["max_length"]),
                    normalized_cost=candidate.estimated_api_calls / max_cost,
                    harm_weight=float(self.metadata["harm_weight"]),
                    cost_weight=float(self.metadata["cost_weight"]))
                inputs = {"input_ids": torch.tensor([encoded.input_ids], device=self.device),
                          "attention_mask": torch.tensor([encoded.attention_mask], device=self.device),
                          "goal_positions": torch.tensor([encoded.goal_positions], device=self.device)}
                output = self.model(**inputs)
                if self.metadata["kind"] == "effect":
                    result.append(output[0].softmax(-1).cpu().tolist())
                else:
                    result.append(float(output[0].cpu()))
                tokens += len(encoded.input_ids)
                dropped += encoded.dropped_history_messages
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.last_usage = {"processed_tokens": tokens, "seconds": elapsed_time() - started,
                           "candidates": len(candidates), "forward_passes": len(candidates),
                           "dropped_history_messages": dropped}
        for key, value in self.last_usage.items():
            self.usage[key] += value
        return result


class EffectPredictor(_Predictor):
    def predict(self, instruction: str, history: Any, goals: Sequence[Goal],
                candidates: Sequence[Candidate], *, cost_denominator: float | None = None) -> list[list[list[float]]]:
        if self.metadata["kind"] != "effect":
            raise ValueError("An effect head is required")
        return self._predict(instruction, history, goals, candidates, cost_denominator=cost_denominator)


class ScalarPredictor(_Predictor):
    def predict(self, instruction: str, history: Any, goals: Sequence[Goal],
                candidates: Sequence[Candidate], *, cost_denominator: float | None = None) -> list[float]:
        if self.metadata["kind"] != "scalar":
            raise ValueError("A scalar head is required")
        return self._predict(instruction, history, goals, candidates, cost_denominator=cost_denominator)


def load_predictor(checkpoint: str | Path, device: str = "cuda",
                   quantized: bool | None = None) -> EffectPredictor | ScalarPredictor:
    torch = require_torch()
    directory = Path(checkpoint)
    metadata = _read_metadata(directory)
    quantized = bool(metadata["trained_quantized"]) if quantized is None else quantized
    _assert_device(torch, device, quantized)
    if metadata["backend"] == "tiny":
        if quantized:
            raise ValueError("Tiny checkpoint cannot be quantized")
        tokenizer = ByteTokenizer()
        encoder = _tiny_encoder(torch, int(metadata["hidden_size"]))
        encoder.load_state_dict(torch.load(directory / "tiny_encoder.pt", map_location="cpu", weights_only=True))
        encoder.to(device)
    elif metadata["backend"] == "hf":
        encoder, _base_tokenizer, hidden_size, resolved = _load_hf_base(
            metadata["base_model"], metadata["revision"], device=device, quantized=quantized)
        if hidden_size != int(metadata["hidden_size"]):
            raise ValueError("Base model hidden dimension no longer matches checkpoint")
        if resolved and resolved != metadata["revision"]:
            raise ValueError("Loaded base model revision differs from checkpoint")
        from transformers import AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", trust_remote_code=False)
        encoder = PeftModel.from_pretrained(encoder, directory / "adapter", is_trainable=False)
    else:
        raise ValueError("Unknown checkpoint backend")
    model = _prediction_model(torch, encoder, int(metadata["hidden_size"]), metadata["kind"])
    model.head.load_state_dict(torch.load(directory / "head.pt", map_location="cpu", weights_only=True))
    model.head.to(device=device, dtype=torch.float32)
    cls = EffectPredictor if metadata["kind"] == "effect" else ScalarPredictor
    predictor = cls(model, tokenizer, metadata, device)
    predictor.metadata["inference_quantized"] = quantized
    return predictor
