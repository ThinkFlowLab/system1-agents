# coding: utf-8
"""The trial record and the paired on/off summary for the recovery eval.

``jobs.summarize`` scores only the episodes that did not error, so its ``mean_score`` is not a completion rate over
every planned trial. ``paired_summary`` is the authoritative denominator: it counts every planned trial, errors and
timeouts included, and pairs the off and on arms per task and repeat. It also reports the paired mean on-minus-off
deltas for model calls, elapsed seconds and wasted actions.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

ARMS = ("off", "on")


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


def _rate(count: int, planned: int) -> float | None:
    return round(count / planned, 3) if planned else None


def _arm_block(records: list[TrialRecord]) -> dict[str, Any]:
    planned = len(records)
    verified = sum(record.verified for record in records)
    costs = [record.cost_usd for record in records]
    return {
        "planned": planned,
        "verified": verified,
        "errored": sum(record.errored for record in records),
        "completion_rate": _rate(verified, planned),
        "recovery_attempts": sum(record.recovery_attempts for record in records),
        "recovery_spent_s": round(sum(record.recovery_spent_s for record in records), 3),
        "recovery_failed": sum(record.recovery_failed for record in records),
        "wasted_actions": sum(record.wasted_actions for record in records),
        "model_calls": sum(record.model_calls for record in records),
        "decision_calls": sum(record.decision_calls for record in records),
        "planner_calls": sum(record.planner_calls for record in records),
        "chat_calls": sum(record.chat_calls for record in records),
        "mean_model_calls": round(statistics.mean(r.model_calls for r in records), 2) if records else None,
        "mean_elapsed_s": round(statistics.mean(r.elapsed_s for r in records), 3) if records else None,
        "mean_wasted_actions": round(statistics.mean(r.wasted_actions for r in records), 2) if records else None,
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
    pairs: dict[tuple[str, int], dict[str, TrialRecord]] = {}
    for record in records:
        pairs.setdefault((record.task, record.repeat), {})[record.arm] = record
    both = off_only = on_only = neither = incomplete = 0
    delta_model_calls: list[int] = []
    delta_elapsed_s: list[float] = []
    delta_wasted_actions: list[int] = []
    for arm_records in pairs.values():
        off, on = arm_records.get("off"), arm_records.get("on")
        if off is None or on is None:
            incomplete += 1
            continue
        delta_model_calls.append(on.model_calls - off.model_calls)
        delta_elapsed_s.append(round(on.elapsed_s - off.elapsed_s, 3))
        delta_wasted_actions.append(on.wasted_actions - off.wasted_actions)
        if off.verified and on.verified:
            both += 1
        elif on.verified:
            on_only += 1
        elif off.verified:
            off_only += 1
        else:
            neither += 1
    off_rate = arms["off"]["verified"] / arms["off"]["planned"] if arms["off"]["planned"] else None
    on_rate = arms["on"]["verified"] / arms["on"]["planned"] if arms["on"]["planned"] else None
    return {
        "planned_trials": len(records),
        "arms": arms,
        "by_task": by_task,
        "paired": {
            "pairs": len(pairs) - incomplete,
            "incomplete_pairs": incomplete,
            "both_verified": both,
            "off_only_verified": off_only,
            "on_only_verified": on_only,
            "neither_verified": neither,
            "mean_delta_model_calls": round(statistics.mean(delta_model_calls), 2) if delta_model_calls else None,
            "mean_delta_elapsed_s": round(statistics.mean(delta_elapsed_s), 3) if delta_elapsed_s else None,
            "mean_delta_wasted_actions": (
                round(statistics.mean(delta_wasted_actions), 2) if delta_wasted_actions else None
            ),
        },
        "on_minus_off_completion_rate": (
            round(on_rate - off_rate, 3) if on_rate is not None and off_rate is not None else None
        ),
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
        "| arm | planned | verified | completion | errored | recovery attempts | mean model calls | mean elapsed s | wasted actions |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        block = summary["arms"][arm]
        lines.append(
            f"| {arm} | {block['planned']} | {block['verified']} | {block['completion_rate']} | {block['errored']} | "
            f"{block['recovery_attempts']} | {block['mean_model_calls']} | {block['mean_elapsed_s']} | "
            f"{block['wasted_actions']} |"
        )
    lines += ["", "| task | arm | planned | verified | completion |", "|---|---|---|---|---|"]
    for task, arms in summary["by_task"].items():
        for arm in ARMS:
            block = arms[arm]
            lines.append(f"| {task} | {arm} | {block['planned']} | {block['verified']} | {block['completion_rate']} |")
    paired = summary["paired"]
    lines += [
        "",
        "Paired completion per (task, repeat): "
        f"on-only {paired['on_only_verified']}, off-only {paired['off_only_verified']}, "
        f"both {paired['both_verified']}, neither {paired['neither_verified']}.",
        f"On minus off completion rate: {summary['on_minus_off_completion_rate']}.",
        "Paired mean on-minus-off deltas: "
        f"model calls {paired['mean_delta_model_calls']}, elapsed s {paired['mean_delta_elapsed_s']}, "
        f"wasted actions {paired['mean_delta_wasted_actions']}.",
        "",
        f"Note: {summary['note']}",
        "",
    ]
    return "\n".join(lines)
