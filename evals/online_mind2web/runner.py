# coding: utf-8
"""Play every (task, arm) attempt on the live web and record each one, failures included.

There is no site reset (this is the live web, not WebHarbor). Each attempt starts a new browser in its own
``logs_dir``, so profiles are not reused across tasks. A missing judge is ``unevaluated``, never a silent drop.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evals.online_mind2web.score import JUDGE_THRESHOLD, Attempt
from evals.online_mind2web.tasks import OfficialTask

Player = Callable[["OfficialTask", str, Path], Awaitable["Outcome"]]


class Judge(Protocol):
    """Score one finished episode; return a 1–5 WebJudge-style integer, or raise to record a judge error."""

    def __call__(self, task: OfficialTask, outcome: "Outcome", logs_dir: Path) -> int: ...


@dataclass(frozen=True)
class Outcome:
    answer: str
    error: str | None = None  # timeout / BLOCKED / agent: ...
    total_ms: int | None = None
    decision_ms: int | None = None
    screenshot: Path | None = None


def classify(error: str | None) -> str | None:
    """Map a player error to a table reason; None means the episode finished and still needs a judge."""
    if not error:
        return None
    text = error.strip()
    lowered = text.lower()
    if lowered.startswith("timeout"):
        return "timeout"
    if "blocked" in lowered:
        return "BLOCKED"
    return text if text.startswith("agent:") else "agent: " + text[:120]


def judged_attempt(
    task: OfficialTask,
    arm: str,
    outcome: Outcome,
    *,
    judge: Judge | None,
    logs_dir: Path,
    agents_sha: str | None,
    threshold: int = JUDGE_THRESHOLD,
) -> Attempt:
    """An attempt row: player errors never reach the judge; a missing judge is unevaluated, not a pass."""
    failure = classify(outcome.error)
    if failure is not None:
        return Attempt(
            task.id,
            arm,
            False,
            failure,
            outcome.total_ms,
            outcome.decision_ms,
            agents_sha=agents_sha,
        )
    if judge is None:
        return Attempt(
            task.id,
            arm,
            False,
            "unevaluated",
            outcome.total_ms,
            outcome.decision_ms,
            agents_sha=agents_sha,
        )
    try:
        score = judge(task, outcome, logs_dir)
    except Exception as exc:  # a judge bug or missing key must not drop the attempt
        return Attempt(
            task.id,
            arm,
            False,
            "judge: " + str(exc)[:120],
            outcome.total_ms,
            outcome.decision_ms,
            agents_sha=agents_sha,
        )
    passed = score >= threshold
    return Attempt(
        task.id,
        arm,
        passed,
        None if passed else f"judge:{score}",
        outcome.total_ms,
        outcome.decision_ms,
        judge_score=score,
        judge_model=getattr(judge, "model", None),
        agents_sha=agents_sha,
    )


async def attempt(
    task: OfficialTask,
    arm: str,
    play: Player,
    out: Path,
    *,
    judge: Judge | None = None,
    agents_sha: str | None = None,
    threshold: int = JUDGE_THRESHOLD,
) -> Attempt:
    out.mkdir(parents=True, exist_ok=True)
    try:
        outcome = await play(task, arm, out)
    except Exception as exc:
        outcome = Outcome(answer="", error="agent: " + str(exc)[:120])
    row = judged_attempt(task, arm, outcome, judge=judge, logs_dir=out, agents_sha=agents_sha, threshold=threshold)
    (out / "attempt.json").write_text(json.dumps(row.to_json(), indent=1), encoding="utf-8")
    return row


async def run_slice(
    tasks: list[OfficialTask],
    arms: list[str],
    play: Player,
    out_root: Path,
    *,
    repeat: int = 1,
    judge: Judge | None = None,
    agents_sha: str | None = None,
    threshold: int = JUDGE_THRESHOLD,
) -> list[Attempt]:
    """Every attempt in a fixed order; arms alternate which goes first per task so neither always runs first."""
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if not arms:
        raise ValueError("at least one arm is required")
    attempts: list[Attempt] = []
    results = out_root / "attempts.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(tasks):
        for r in range(repeat):
            order = arms if (index + r) % 2 == 0 else list(reversed(arms))
            for arm in order:
                row = await attempt(
                    task,
                    arm,
                    play,
                    out_root / f"{task.id}__{arm}__r{r}",
                    judge=judge,
                    agents_sha=agents_sha,
                    threshold=threshold,
                )
                attempts.append(row)
                with results.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row.to_json()) + "\n")
    return attempts


def load_attempts(path: Path) -> list[Attempt]:
    return [Attempt.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
