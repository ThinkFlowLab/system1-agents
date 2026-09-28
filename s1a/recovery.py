# coding: utf-8
"""The bounded recovery budget for one episode: a few attempts and a few active seconds, never reset by progress.

``timeout_s`` counts only the active time spent inside the refresh and planner calls, not the whole episode. Every
stall starts one attempt; progress never gives an attempt or a second back. The budget cancels each call through
``asyncio.wait_for``, so a hung refresh or planner is really stopped at the remaining timeout, not merely noticed
afterwards. The tool front uses this today; the browser front reuses the same object.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


class RecoveryExhausted(RuntimeError):
    """The per-episode recovery budget is spent: no further refresh or plan call is allowed."""


# A recovery error that names a permission wall routes the operator through the normal permission flow.
_PERMISSION_MARKERS = ("permission", "not allowed", "forbidden", "unauthorized", "denied", "accessibility", "403")


def recovery_next_action(*, termination: str | None, stage: str | None = None, error: str | None = None) -> str:
    """One short, actionable next step for a bounded recovery that stopped without settling the task.

    It points at reviewing the fresh observation and the triggering action, and at explicitly starting a new task
    once the blocker is cleared; it never widens the tool set, restarts by itself or retries on its own. A permission
    wall is routed through the existing permission flow, and the task is left blocked when access is not requested.
    ``termination`` and ``stage`` name where the recovery stopped.
    """
    haystack = f"{error or ''} {stage or ''}".lower()
    if any(marker in haystack for marker in _PERMISSION_MARKERS):
        return (
            "Review the blocked action; if appropriate, request the needed access through the existing permission "
            "flow, otherwise leave the task blocked."
        )
    reason = termination or "failure"
    where = f" in the {stage} step" if stage else ""
    return (
        f"Review the fresh observation and the action that triggered recovery ({reason}{where}); clear the "
        "blocker, then explicitly start a new task."
    )


@dataclass(frozen=True)
class RecoveryLimits:
    """How much bounded recovery one episode may spend: whole attempts and active seconds over refresh and planner."""

    max_attempts: int = 3
    timeout_s: float = 15.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError(f"max_attempts must be a positive integer, got {self.max_attempts!r}")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError(f"timeout_s must be a finite positive number, got {self.timeout_s!r}")


class RecoveryBudget:
    """Per-episode recovery state: the attempts started, the active seconds charged, and the calls under them."""

    def __init__(self, limits: RecoveryLimits) -> None:
        self._limits = limits
        self._attempts = 0
        self._spent_s = 0.0

    @property
    def limits(self) -> RecoveryLimits:
        return self._limits

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def spent_s(self) -> float:
        return self._spent_s

    @property
    def remaining_s(self) -> float:
        return max(0.0, self._limits.timeout_s - self._spent_s)

    @property
    def exhausted(self) -> bool:
        return self._attempts >= self._limits.max_attempts or self.remaining_s <= 0.0

    def begin_attempt(self) -> None:
        """One stall starts one attempt; a spent budget refuses it. Progress must never reach this to add back."""
        if self._attempts >= self._limits.max_attempts:
            raise RecoveryExhausted(f"recovery attempts spent: {self._attempts}/{self._limits.max_attempts}")
        if self.remaining_s <= 0.0:
            raise RecoveryExhausted(f"recovery time spent: {self._spent_s:.3f}s of {self._limits.timeout_s}s")
        self._attempts += 1

    async def call(self, awaitable: Awaitable[T]) -> T:
        """Run one refresh or planner call under the remaining timeout and charge its actual active time.

        A call with no time left is refused, not run: its awaitable is closed or cancelled first, so a rejected
        refresh or plan leaks neither a coroutine nor a warning and never executes a body.
        """
        budget = self.remaining_s
        if budget <= 0.0:
            _discard(awaitable)
            raise RecoveryExhausted(f"recovery time spent: {self._spent_s:.3f}s of {self._limits.timeout_s}s")
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(awaitable, timeout=budget)
        finally:
            self._spent_s += time.perf_counter() - started


def _discard(awaitable: Awaitable[Any]) -> None:
    """Drop an awaitable the budget refuses to run: close a coroutine, cancel a future, never await its body."""
    close = getattr(awaitable, "close", None)
    if close is not None:
        close()
        return
    cancel = getattr(awaitable, "cancel", None)
    if cancel is not None:
        cancel()
