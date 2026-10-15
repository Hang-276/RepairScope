"""Label-free, shared model input for effect and scalar prediction.

One causal forward pass produces one hidden query position per goal.  Every
query follows the entire instruction, goal set, candidate and retained history.
Only whole *old* history messages may be dropped; required context overflow is
an error, never silent truncation of the candidate or the goal list.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

from .schemas import Candidate, Goal


class InputTooLongError(ValueError):
    """The full required context or most recent observation cannot fit."""


@dataclass(frozen=True)
class EncodedInput:
    input_ids: list[int]
    attention_mask: list[int]
    goal_positions: list[int]
    retained_history_messages: int
    dropped_history_messages: int
    text: str


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    clean = []
    for message in history:
        # Do not accidentally serialize a replay/evaluation record as an observation.
        extra = set(message) - {"role", "content"}
        if extra:
            raise ValueError(f"History contains non-runtime fields: {sorted(extra)}")
        if message.get("role") not in {"system", "user", "assistant", "tool", "observation"}:
            raise ValueError("History requires a supported role and plain content")
        if not isinstance(message.get("content"), str):
            raise TypeError("History content must be a string")
        clean.append({"role": str(message["role"]), "content": message["content"]})
    return clean


def _parts(
    instruction: str,
    history: Sequence[Mapping[str, Any]],
    goals: Sequence[Goal],
    candidate: Candidate,
    *,
    normalized_cost: float = 1.0,
    harm_weight: float = 2.0,
    cost_weight: float = 0.05,
) -> tuple[str, list[str]]:
    if not instruction or not goals:
        raise ValueError("Instruction and at least one goal are required")
    if len({goal.id for goal in goals}) != len(goals):
        raise ValueError("Duplicate goal IDs")
    if any(goal.evidence_span not in instruction for goal in goals):
        raise ValueError("Goal evidence must occur in the original instruction")
    if not set(candidate.target_goal_ids).issubset({g.id for g in goals}):
        raise ValueError("Candidate targets an unknown goal")
    if not 0 <= normalized_cost <= 1:
        raise ValueError("Normalized candidate cost must be in [0, 1]")
    if any(not math.isfinite(x) or x < 0 for x in (harm_weight, cost_weight)):
        raise ValueError("Decision weights must be finite and nonnegative")
    clean_history = _validate_history(history)
    candidate_record = candidate.to_dict()
    candidate_record["estimated_api_calls"] = float(candidate.estimated_api_calls)
    prefix = (
        "Predict the final goal-state transitions after executing this candidate repair. "
        "Use only the visible task and execution history.\n"
        + "TASK\n" + instruction
        + "\nGOALS\n" + _dump([g.to_dict() for g in goals])
        + "\nCANDIDATE\n" + _dump(candidate_record)
        + "\nDECISION_WEIGHTS\n" + _dump({"harm": float(harm_weight), "cost": float(cost_weight),
                                               "normalized_candidate_cost": float(normalized_cost)})
        + "\nVISIBLE_HISTORY\n" + _dump(clean_history)
        + "\nEND_CONTEXT\n"
    )
    queries = [f"GOAL_QUERY {_dump({'id': g.id, 'text': g.text})}\n" for g in goals]
    return prefix, queries


def build_effect_text(
    instruction: str,
    history: Sequence[Mapping[str, Any]],
    goals: Sequence[Goal],
    candidate: Candidate,
    **kwargs: Any,
) -> str:
    """Human-readable exact input components; never accepts labels or an evaluator."""
    prefix, queries = _parts(instruction, history, goals, candidate, **kwargs)
    return prefix + "".join(queries)


def encode_effect_input(
    tokenizer: Any,
    instruction: str,
    history: Sequence[Mapping[str, Any]],
    goals: Sequence[Goal],
    candidate: Candidate,
    *,
    max_length: int = 4096,
    normalized_cost: float = 1.0,
    harm_weight: float = 2.0,
    cost_weight: float = 0.05,
) -> EncodedInput:
    """Tokenize the shared input, recording the last token of every goal query.

    Segments are tokenized independently with ``add_special_tokens=False``;
    inference and training use precisely this same function. This avoids relying
    on string search for query delimiters that may occur inside quoted code.
    """
    if max_length < 16:
        raise ValueError("max_length must be at least 16")
    clean = _validate_history(history)

    def encode(retained: Sequence[Mapping[str, str]]) -> EncodedInput:
        prefix, queries = _parts(instruction, retained, goals, candidate,
                                  normalized_cost=normalized_cost,
                                  harm_weight=harm_weight, cost_weight=cost_weight)
        ids: list[int] = []
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None:
            ids.append(int(bos))
        ids.extend(tokenizer.encode(prefix, add_special_tokens=False))
        positions = []
        for query in queries:
            encoded = tokenizer.encode(query, add_special_tokens=False)
            if not encoded:
                raise ValueError("Tokenizer produced an empty query segment")
            ids.extend(encoded)
            positions.append(len(ids) - 1)
        return EncodedInput(ids, [1] * len(ids), positions, len(retained),
                            len(clean) - len(retained), prefix + "".join(queries))

    mandatory = encode([])
    if len(mandatory.input_ids) > max_length:
        raise InputTooLongError(
            f"Full task/goals/candidate/query context needs {len(mandatory.input_ids)} tokens; "
            f"max_length={max_length}. Shorten the candidate or increase the budget explicitly."
        )
    if clean:
        newest = encode(clean[-1:])
        if len(newest.input_ids) > max_length:
            raise InputTooLongError(
                f"Required context plus latest history message needs {len(newest.input_ids)} "
                f"tokens; max_length={max_length}. Latest history may not be silently dropped."
            )
    for start in range(len(clean) + 1):
        result = encode(clean[start:])
        if len(result.input_ids) <= max_length:
            return result
    raise AssertionError("Mandatory context was checked to fit")


class ByteTokenizer:
    """Transparent byte tokenizer for local tiny-model tests; never used for Qwen."""
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259
    pad_token = "<pad>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        result = [byte + 3 for byte in text.encode("utf-8")]
        return [1] + result if add_special_tokens else result

    def save_pretrained(self, directory: Any) -> None:
        from pathlib import Path
        from .io import atomic_json
        atomic_json(Path(directory) / "tiny_tokenizer.json", {"type": "byte", "vocab_size": 259})
