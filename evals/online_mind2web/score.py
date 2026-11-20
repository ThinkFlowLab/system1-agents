# coding: utf-8
"""Attempts to the result table: completion with every attempt in the denominator, latency split, failure reasons.

A number that was not measured is "N/A", never zero. Success is a WebJudge verdict at the official threshold (default
3); an episode that finished without a judge is ``unevaluated``, not a pass. Timeouts, agent errors, BLOCKED and
judge failures stay in the denominator.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

NA = "N/A"
JUDGE_THRESHOLD = 3


@dataclass(frozen=True)
class Attempt:
    task_id: str
    arm: str
    passed: bool
    failure: str | None  # None only when passed
    total_ms: int | None = None
    decision_ms: int | None = None  # None when the arm makes no decision-model calls
    judge_score: int | None = None
    judge_model: str | None = None
    agents_sha: str | None = None

    @property
    def environment_ms(self) -> int | None:
        """Episode time outside decision calls: browser, page loads and the chat model."""
        if self.total_ms is None or self.decision_ms is None:
            return None
        return max(self.total_ms - self.decision_ms, 0)

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "environment_ms": self.environment_ms}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Attempt":
        keys = {key: data.get(key) for key in cls.__dataclass_fields__}
        return cls(**keys)


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """The 95% Wilson score interval for ``hits`` of ``total``; None when ``total`` is 0."""
    if total <= 0:
        return None
    if not 0 <= hits <= total:
        raise ValueError(f"hits {hits} outside 0..{total}")
    p = hits / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _median(values: Iterable[int | None]) -> float | None:
    known = [v for v in values if v is not None]
    return statistics.median(known) if known else None


def summarize(attempts: list[Attempt]) -> dict[str, dict[str, Any]]:
    """Per arm, in first-seen order: attempts, passes, rate, Wilson interval, median latencies, failure counts."""
    arms: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        arms.setdefault(attempt.arm, []).append(attempt)
    out: dict[str, dict[str, Any]] = {}
    for arm, rows in arms.items():
        passes = sum(1 for row in rows if row.passed)
        out[arm] = {
            "attempts": len(rows),
            "passed": passes,
            "rate": passes / len(rows),
            "wilson95": wilson(passes, len(rows)),
            "median_total_ms": _median(row.total_ms for row in rows),
            "median_decision_ms": _median(row.decision_ms for row in rows),
            "median_environment_ms": _median(row.environment_ms for row in rows),
            "failures": dict(Counter(row.failure for row in rows if not row.passed and row.failure)),
        }
    return out


def _pct(value: float | None) -> str:
    return NA if value is None else f"{value * 100:.0f}%"


def _ms(value: float | None) -> str:
    return NA if value is None else f"{value / 1000:.1f} s"


def render(
    attempts: list[Attempt],
    *,
    revision: str,
    dataset: str,
    agents_sha: str = NA,
    model: str = NA,
    budget: str = NA,
    date: str = NA,
    judge: str = "WebJudge",
    threshold: int = JUDGE_THRESHOLD,
) -> str:
    """The Markdown result table. Header names the dataset revision, Agents SHA, model, budget and date."""
    lines = [
        f"Online-Mind2Web live slice: `{dataset}` at revision `{revision}`.",
        f"Agents SHA `{agents_sha or NA}`, model `{model or NA}`, budget `{budget or NA}`, date `{date or NA}`.",
        f"Success is {judge} at threshold {threshold}; every attempt (failures, timeouts, errors, BLOCKED, "
        "unevaluated) is in the denominator. This is a live-web suite, not the #109 offline step collector and not "
        "the #104 Anthropic browser_toolset baseline.",
        "",
        "| arm | passed / attempts | rate | Wilson 95% | median total | median decision | median environment |"
        " failures |",
        "|---|---|---|---|---|---|---|---|",
    ]
    summary = summarize(attempts)
    if not summary:
        lines.append(f"| {NA} | 0 / 0 | {NA} | {NA} | {NA} | {NA} | {NA} | no attempts |")
    for arm, row in summary.items():
        interval = row["wilson95"]
        ci = NA if interval is None else f"{_pct(interval[0])}-{_pct(interval[1])}"
        failures = ", ".join(f"{reason} x{count}" for reason, count in sorted(row["failures"].items())) or "none"
        lines.append(
            f"| {arm} | {row['passed']} / {row['attempts']} | {_pct(row['rate'])} | {ci} |"
            f" {_ms(row['median_total_ms'])} | {_ms(row['median_decision_ms'])} | {_ms(row['median_environment_ms'])} |"
            f" {failures} |"
        )
    return "\n".join(lines) + "\n"
