"""Dependency-free OpenAI-compatible chat client with bounded, charged retries."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Any
import urllib.error
import urllib.request

from .budget import Budget, BudgetExceeded
from .clock import elapsed_time


class LLMError(RuntimeError):
    pass


class TruncatedResponseError(LLMError):
    pass


class ContextLengthError(LLMError):
    pass


@dataclass(frozen=True)
class ChatResult:
    content: str
    input_tokens: int
    output_tokens: int
    seconds: float
    finish_reason: str = "stop"
    usage_estimated: bool = False


def strict_json_loads(text: str) -> Any:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def count_tokens(text: str, tokenizer: Any = None) -> int:
    """Exact with supplied tokenizer; otherwise conservative UTF-8 byte estimate."""
    if tokenizer is None:
        return len(text.encode("utf-8"))
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text)
    return encoded if isinstance(encoded, int) else len(encoded)


def message_token_count(messages: list[dict[str, str]], tokenizer: Any = None,
                        chat_template_kwargs: dict[str, Any] | None = None) -> int:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            **(chat_template_kwargs or {}))
        return len(ids)
    # Tokenizer-free counts are deliberately conservative, not reported as exact.
    return 32 + sum(16 + count_tokens(m["role"], tokenizer) + count_tokens(m["content"], tokenizer)
                    for m in messages)


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "B2_ACTOR_API_KEY",
                 timeout: float = 120, tokenizer: Any = None, budget: Budget | None = None,
                 max_retries: int = 2, max_response_bytes: int = 4 * 1024 * 1024,
                 chat_template_kwargs: dict[str, Any] | None = None,
                 context_length: int | None = 8192):
        if not base_url.startswith(("http://", "https://")) or not model:
            raise ValueError("A HTTP(S) base URL and nonempty model are required")
        if not math.isfinite(timeout) or timeout <= 0 or not isinstance(max_retries, int) or not 0 <= max_retries <= 5:
            raise ValueError("Positive timeout and 0..5 bounded retries are required")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes < 1:
            raise ValueError("Response byte cap must be a positive integer")
        if context_length is not None and (isinstance(context_length, bool) or
                                           not isinstance(context_length, int) or context_length < 1):
            raise ValueError("Context length must be a positive integer or None")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.tokenizer = tokenizer
        self.budget = budget if budget is not None else Budget()
        self.max_retries = max_retries
        self.max_response_bytes = max_response_bytes
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.context_length = context_length

    @property
    def usage(self) -> dict[str, Any]:
        return self.budget.snapshot()

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 3000,
                 temperature: float = 0, seed: int = 100, *, budget: Budget | None = None,
                 purpose: str = "chat") -> ChatResult:
        if not messages or any(not isinstance(m, dict) or set(m) - {"role", "content"} or
                               not isinstance(m.get("content"), str) or
                               m.get("role") not in {"system", "user", "assistant"} for m in messages):
            raise ValueError("Chat messages must contain only supported role/content strings")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        active = budget if budget is not None else self.budget
        estimate = message_token_count(messages, self.tokenizer, self.chat_template_kwargs)
        if self.context_length is not None:
            max_tokens = min(max_tokens, self.context_length - estimate)
            if max_tokens < 1:
                raise ContextLengthError("Complete input does not fit the configured context; no request sent")
        endpoint = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        for attempt in range(self.max_retries + 1):
            reservation = active.reserve_chat(estimate, max_tokens, purpose)
            payload = dict(model=self.model, messages=messages,
                           max_tokens=reservation.max_output_tokens, temperature=temperature, seed=seed)
            if self.chat_template_kwargs:
                payload["chat_template_kwargs"] = self.chat_template_kwargs
            headers = {"Content-Type": "application/json"}
            api_key = os.environ.get(self.api_key_env, "")
            if api_key:
                headers["Authorization"] = "Bearer " + api_key
            request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                                             headers=headers, method="POST")
            started = elapsed_time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read(self.max_response_bytes + 1)
                elapsed = elapsed_time() - started
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                elapsed = elapsed_time() - started
                active.record_chat(reservation, seconds=elapsed, estimated=True, status="http_error")
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {429, 500, 502, 503, 504}
                if retryable and attempt < self.max_retries:
                    time.sleep(min(0.25 * (2 ** attempt), 2))
                    continue
                # Do not echo server bodies or request headers, which can contain secrets.
                code = exc.code if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
                raise LLMError(f"Chat request failed ({code}) after {attempt + 1} attempt(s)") from None
            usage = None
            try:
                if len(raw) > self.max_response_bytes:
                    raise ValueError("Response exceeds configured byte limit")
                data = strict_json_loads(raw.decode("utf-8"))
                usage = data.get("usage") if isinstance(data, dict) else None
                usage_valid = (isinstance(usage, dict) and all(
                    isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool) and usage[k] >= 0
                    for k in ("prompt_tokens", "completion_tokens")))
                active.record_chat(reservation,
                                   input_tokens=usage["prompt_tokens"] if usage_valid else None,
                                   output_tokens=usage["completion_tokens"] if usage_valid else None,
                                   seconds=elapsed, estimated=not usage_valid)
                choices = data["choices"]
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("Expected exactly one completion choice")
                finish = choices[0].get("finish_reason", "")
                if finish in {"length", "max_tokens"}:
                    raise TruncatedResponseError("Generation hit its output cap; response must not execute")
                if finish != "stop":
                    raise ValueError(f"Unsupported completion finish reason: {finish!r}")
                content = choices[0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Completion content must be a nonempty string")
                return ChatResult(content, usage["prompt_tokens"] if usage_valid else estimate,
                                  usage["completion_tokens"] if usage_valid else reservation.max_output_tokens,
                                  elapsed, finish, not usage_valid)
            except BudgetExceeded:
                raise
            except TruncatedResponseError:
                raise
            except (ValueError, TypeError, KeyError, IndexError, UnicodeError) as exc:
                if reservation.id in active._pending:
                    active.record_chat(reservation, seconds=elapsed, estimated=True, status="invalid_response")
                raise LLMError(f"Malformed chat response: {type(exc).__name__}") from None
        raise AssertionError("Bounded retry loop unexpectedly fell through")
