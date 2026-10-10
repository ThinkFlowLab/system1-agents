# coding: utf-8
"""The trial record and the paired on/off summary for the recovery eval.

``jobs.summarize`` scores only the episodes that did not error, so its ``mean_score`` is not a completion rate over
every planned trial. ``paired_summary`` is the authoritative denominator: it counts every planned trial, errors and
timeouts included, and pairs the off and on arms per task and repeat. It also reports the paired mean on-minus-off
deltas for model calls, elapsed seconds and wasted actions.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

ARMS = ("off", "on")


def rate(count: int, planned: int) -> float | None:
    """A rounded completion rate, or ``None`` when nothing was planned (never a division by zero)."""
    return round(count / planned, 3) if planned else None


def mean(values: Iterable[float], digits: int) -> float | None:
    """The rounded arithmetic mean of the values, or ``None`` when there are none."""
    data = list(values)
    return round(statistics.mean(data), digits) if data else None


def coverage(records: Sequence[Any]) -> dict[str, Any]:
    """The shared arm denominator: every planned record counts, verified by its own oracle, errors included."""
    planned = len(records)
    verified = sum(1 for record in records if record.verified)
    return {
        "planned": planned,
        "verified": verified,
        "errored": sum(1 for record in records if record.errored),
        "completion_rate": rate(verified, planned),
    }


def pair_outcomes(records: Sequence[Any]) -> tuple[dict[str, Any], list[tuple[Any, Any]]]:
    """Pair every record by (task, repeat): the shared paired-verification counts and the complete off/on pairs.

    A record whose partner is missing is an incomplete pair, never silently dropped; the returned complete pairs let
    each frontend compute its own on-minus-off deltas without repeating the grouping.
    """
    by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        by_pair.setdefault((record.task, record.repeat), {})[record.arm] = record
    complete: list[tuple[Any, Any]] = []
    both = off_only = on_only = neither = incomplete = 0
    for arms in by_pair.values():
        off, on = arms.get("off"), arms.get("on")
        if off is None or on is None:
            incomplete += 1
            continue
        complete.append((off, on))
        if off.verified and on.verified:
            both += 1
        elif on.verified:
            on_only += 1
        elif off.verified:
            off_only += 1
        else:
            neither += 1
    return {
        "pairs": len(by_pair) - incomplete,
        "incomplete_pairs": incomplete,
        "both_verified": both,
        "off_only_verified": off_only,
        "on_only_verified": on_only,
        "neither_verified": neither,
    }, complete


def on_minus_off_completion_rate(arms: dict[str, Any]) -> float | None:
    """The on arm's completion rate minus the off arm's, or ``None`` unless both arms planned trials."""
    off, on = arms.get("off"), arms.get("on")
    if not off or not on or not off.get("planned") or not on.get("planned"):
        return None
    return round((on["verified"] / on["planned"]) - (off["verified"] / off["planned"]), 3)


def paired_completion_line(paired: dict[str, Any]) -> str:
    """The shared "Paired completion per (task, repeat)" sentence every frontend's markdown carries."""
    return (
        "Paired completion per (task, repeat): "
        f"on-only {paired['on_only_verified']}, off-only {paired['off_only_verified']}, "
        f"both {paired['both_verified']}, neither {paired['neither_verified']}."
    )


def paired_delta_line(paired: dict[str, Any], **named: Any) -> str:
    """The shared paired on-minus-off deltas sentence, over the named ``paired`` keys and their human labels."""
    return (
        "Paired mean on-minus-off deltas: " + ", ".join(f"{label} {paired[key]}" for key, label in named.items()) + "."
    )


def summary_tables(summary: dict[str, Any], metrics: Sequence[tuple[str, str]]) -> list[str]:
    """The arm and task tables shared by the three recovery reports."""
    labels = ["arm", "planned", "verified", "completion", *(label for label, _ in metrics)]
    fields = ["planned", "verified", "completion_rate", *(key for _, key in metrics)]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in labels) + "|"]
    for arm, block in summary["arms"].items():
        lines.append("| " + " | ".join(str(value) for value in [arm, *(block[key] for key in fields)]) + " |")
    lines += ["", "| task | arm | planned | verified | completion |", "|---|---|---|---|---|"]
    for task, arms in summary["by_task"].items():
        for arm in summary["arms"]:
            block = arms[arm]
            lines.append(f"| {task} | {arm} | {block['planned']} | {block['verified']} | {block['completion_rate']} |")
    return lines


@dataclass
class TrialRecord:
    """One planned trial's outcome, whether it verified, failed, timed out or was never supported."""

    task: str
    arm: str
    repeat: int
    verified: bool
    terminal: str
    errored: bool
    scripted_model: str
    platform: str
    recovery_attempts: int = 0
    recovery_spent_s: float = 0.0
    recovery_failed: bool = False
    recovery_termination: str | None = None
    wasted_actions: int = 0
    model_calls: int = 0
    decision_calls: int = 0
    planner_calls: int = 0
    chat_calls: int = 0
    elapsed_s: float = 0.0
    decisions_ms: int = 0
    recorded_probe_ms: int = 0
    cost_usd: float = 0.0
    oracle: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _arm_block(records: list[TrialRecord]) -> dict[str, Any]:
    block = coverage(records)
    costs = [record.cost_usd for record in records]
    return {
        **block,
        "recovery_attempts": sum(record.recovery_attempts for record in records),
        "recovery_spent_s": round(sum(record.recovery_spent_s for record in records), 3),
        "recovery_failed": sum(record.recovery_failed for record in records),
        "wasted_actions": sum(record.wasted_actions for record in records),
        "model_calls": sum(record.model_calls for record in records),
        "decision_calls": sum(record.decision_calls for record in records),
        "planner_calls": sum(record.planner_calls for record in records),
        "chat_calls": sum(record.chat_calls for record in records),
        "mean_model_calls": mean((r.model_calls for r in records), 2),
        "mean_elapsed_s": mean((r.elapsed_s for r in records), 3),
        "mean_wasted_actions": mean((r.wasted_actions for r in records), 2),
        "decisions_ms": sum(record.decisions_ms for record in records),
        "recorded_probe_ms": sum(record.recorded_probe_ms for record in records),
        "cost_usd": round(sum(costs), 6) if costs else 0.0,
    }


def paired_summary(records: list[TrialRecord]) -> dict[str, Any]:
    """Every planned trial counts. Per arm, per task, and the paired off/on outcomes per (task, repeat)."""
    arms = {arm: _arm_block([r for r in records if r.arm == arm]) for arm in ARMS}
    tasks = sorted({record.task for record in records})
    by_task = {
        task: {arm: _arm_block([r for r in records if r.arm == arm and r.task == task]) for arm in ARMS}
        for task in tasks
    }
    paired, complete = pair_outcomes(records)
    paired = {
        **paired,
        "mean_delta_model_calls": mean((on.model_calls - off.model_calls for off, on in complete), 2),
        "mean_delta_elapsed_s": mean((round(on.elapsed_s - off.elapsed_s, 3) for off, on in complete), 3),
        "mean_delta_wasted_actions": mean((on.wasted_actions - off.wasted_actions for off, on in complete), 2),
    }
    return {
        "planned_trials": len(records),
        "arms": arms,
        "by_task": by_task,
        "paired": paired,
        "on_minus_off_completion_rate": on_minus_off_completion_rate(arms),
        "note": (
            "scripted doubles: controlled fault injection to exercise recovery, not a trained model; "
            "cost is 0 by construction, not a real API bill; starts/timeouts/failures count in every denominator."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """A short human-readable companion to the JSON."""
    lines: list[str] = [
        "# Bounded recovery on/off - scripted fixture subset",
        "",
        f"Planned trials: {summary['planned_trials']} (errors and timeouts included, never dropped).",
        "",
    ]
    lines += summary_tables(
        summary,
        [
            ("errored", "errored"),
            ("recovery attempts", "recovery_attempts"),
            ("mean model calls", "mean_model_calls"),
            ("mean elapsed s", "mean_elapsed_s"),
            ("wasted actions", "wasted_actions"),
        ],
    )
    paired = summary["paired"]
    lines += [
        "",
        paired_completion_line(paired),
        f"On minus off completion rate: {summary['on_minus_off_completion_rate']}.",
        paired_delta_line(
            paired,
            **{
                "mean_delta_model_calls": "model calls",
                "mean_delta_elapsed_s": "elapsed s",
                "mean_delta_wasted_actions": "wasted actions",
            },
        ),
        "",
        f"Note: {summary['note']}",
        "",
    ]
    return "\n".join(lines)
