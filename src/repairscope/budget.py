"""Shared, conservative accounting for actor and auxiliary work.

Usage is task-wide. A failed HTTP request without reliable usage is charged its
reservation, so retries never become free experiment work.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any


class BudgetExceeded(RuntimeError):
    """The next action would exceed a configured task budget."""


@dataclass(frozen=True)
class Reservation:
    id: int
    input_tokens: int
    max_output_tokens: int
    purpose: str


class Budget:
    def __init__(self, max_output_tokens: int = 100000,
                 max_total_tokens: int | None = None, max_calls: int | None = None,
                 max_seconds: float | None = None):
        for name, value in (("max_output_tokens", max_output_tokens),
                            ("max_total_tokens", max_total_tokens),
                            ("max_calls", max_calls)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.seconds = 0.0
        self.estimated_usage_calls = 0
        self.events: list[dict[str, Any]] = []
        self._pending: dict[int, Reservation] = {}
        self._next_id = 0
        self._lock = threading.RLock()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def remaining_output_tokens(self) -> int:
        return max(0, self.max_output_tokens - self.output_tokens - sum(
            r.max_output_tokens for r in self._pending.values()))

    @property
    def remaining_total_tokens(self) -> int | None:
        if self.max_total_tokens is None:
            return None
        return max(0, self.max_total_tokens - self.total_tokens - sum(
            r.input_tokens + r.max_output_tokens for r in self._pending.values()))

    def _check_time(self) -> None:
        if self.max_seconds is not None and self.seconds >= self.max_seconds:
            raise BudgetExceeded("Task inference time budget exhausted")

    def reserve_chat(self, input_tokens: int, max_output_tokens: int,
                     purpose: str = "chat") -> Reservation:
        """Reserve before HTTP; clamp generation to remaining output/total cap."""
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
               for v in (input_tokens, max_output_tokens)) or max_output_tokens == 0:
            raise ValueError("Token counts must be nonnegative integers and generation must be positive")
        with self._lock:
            self._check_time()
            if self.max_calls is not None and self.calls + len(self._pending) >= self.max_calls:
                raise BudgetExceeded("Task HTTP call budget exhausted")
            permitted = min(max_output_tokens, self.remaining_output_tokens)
            total = self.remaining_total_tokens
            if total is not None:
                permitted = min(permitted, total - input_tokens)
            if permitted < 1:
                raise BudgetExceeded("Insufficient task token budget for this request")
            reservation = Reservation(self._next_id, input_tokens, permitted, purpose)
            self._next_id += 1
            self._pending[reservation.id] = reservation
            return reservation

    def record_chat(self, reservation: Reservation, input_tokens: int | None = None,
                    output_tokens: int | None = None, seconds: float = 0.0,
                    estimated: bool = False, status: str = "ok") -> None:
        """Commit actual usage; missing usage conservatively consumes reservation."""
        with self._lock:
            if reservation.id not in self._pending:
                raise ValueError("Unknown or already consumed budget reservation")
            in_count = reservation.input_tokens if input_tokens is None else input_tokens
            out_count = reservation.max_output_tokens if output_tokens is None else output_tokens
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (in_count, out_count)):
                raise ValueError("Reported token usage must be nonnegative integers")
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Reported inference seconds must be finite and nonnegative")
            del self._pending[reservation.id]
            self.calls += 1
            self.input_tokens += in_count
            self.output_tokens += out_count
            self.seconds += seconds
            estimated = estimated or input_tokens is None or output_tokens is None
            self.estimated_usage_calls += int(estimated)
            self.events.append(dict(purpose=reservation.purpose, input_tokens=in_count,
                                    output_tokens=out_count, seconds=seconds,
                                    usage_estimated=estimated, status=status))
            if self.output_tokens > self.max_output_tokens:
                raise BudgetExceeded("Server output usage exceeded task output budget")
            if self.max_total_tokens is not None and self.total_tokens > self.max_total_tokens:
                raise BudgetExceeded("Server reported usage exceeded task total budget")
            if out_count > reservation.max_output_tokens:
                raise BudgetExceeded("Server generated more tokens than the request permitted")
            self._check_time()

    def snapshot(self) -> dict[str, Any]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens, "calls": self.calls,
                "seconds": self.seconds,
                "estimated_usage_calls": self.estimated_usage_calls,
                "remaining_output_tokens": self.remaining_output_tokens,
                "remaining_total_tokens": self.remaining_total_tokens,
                "caps": {"max_output_tokens": self.max_output_tokens,
                         "max_total_tokens": self.max_total_tokens, "max_calls": self.max_calls,
                         "max_seconds": self.max_seconds},
                "events": list(self.events)}
