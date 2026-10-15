"""Runtime mechanics independently testable without AppWorld or neural models."""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from typing import Any

from .llm import message_token_count


def is_completion(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
               node.func.attr == "complete_task" for node in ast.walk(tree))


def checkpoint_reasons(step: int, last_output: str, code: str, interval: int = 5) -> list[str]:
    if step < 1 or interval < 1:
        raise ValueError("Step and checkpoint interval must be positive")
    reasons = []
    if any(tag in last_output for tag in ("Execution failed.", "Execution failed:",
                                         "Traceback (most recent call last)", "Exception:", "Error:")):
        reasons.append("observed_error")
    if step > 1 and (step - 1) % interval == 0:
        reasons.append("periodic")
    if is_completion(code):
        reasons.append("precompletion")
    return reasons


def trim_actor_messages(messages: list[dict[str, Any]], instruction_messages: int,
                        tokenizer: Any, max_input_tokens: int,
                        chat_template_kwargs: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Preserve upstream task/demo prefix and latest observation; drop oldest pairs.

    This is the same declared token cap for every method, including baseline.
    Mandatory context overflow fails visibly instead of slicing instruction/code.
    """
    clean = [{"role": m["role"], "content": m.get("content") or ""} for m in messages]
    clean = copy.deepcopy(clean)
    prefix, tail = clean[:instruction_messages], clean[instruction_messages:]
    if not 0 <= instruction_messages <= len(clean) or max_input_tokens < 1:
        raise ValueError("Invalid prompt prefix or context budget")
    while message_token_count(prefix + tail, tokenizer, chat_template_kwargs) > max_input_tokens:
        if len(tail) <= 1:
            raise ValueError("Mandatory actor prompt/latest observation exceeds token context cap")
        # Remove an assistant/observation pair together when available.
        tail = tail[2:] if len(tail) > 2 else tail[-1:]
    return prefix + tail


def validate_live_predictor_metadata(metadata: dict[str, Any], method: str) -> None:
    """Reject synthetic checkpoints and scoring mismatches before GPU loading."""
    if method not in {"effect", "scalar"} or metadata.get("kind") != method:
        raise ValueError("Predictor kind does not match the learned online method")
    if metadata.get("backend") != "hf" or metadata.get("fixture_backend") is not False:
        raise ValueError("Only explicitly non-fixture HF checkpoints may run live benchmarks")
    if metadata.get("label_map") != ["00", "01", "10", "11"]:
        raise ValueError("Predictor label order does not match the fixed decision rule")
    for key, expected in (("harm_weight", 2.0), ("cost_weight", 0.05)):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(value, expected):
            raise ValueError(f"Predictor {key} differs from the preregistered selection objective")
    if metadata.get("cost_normalization") != "max(1,max_candidate_pool_estimated_api_calls)":
        raise ValueError("Predictor cost normalization differs from the selection objective")


def count_native_api_calls(path: str | Path) -> int:
    """Count actual RequestTracker records, never statically estimated code calls."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Native API call log was not saved; actual call count is unavailable")
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (not isinstance(record, dict) or record.get("method") not in {"get", "post", "put", "patch", "delete"}
                or not isinstance(record.get("url"), str) or not isinstance(record.get("data"), dict)):
            raise ValueError("Malformed native RequestTracker API call record")
        count += 1
    return count


def visible_history(steps: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for step in steps:
        result.extend([{"role": "assistant", "content": step["code"]},
                       {"role": "user", "content": step["output"]}])
    return result
