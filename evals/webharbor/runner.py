# coding: utf-8
"""Play every (task, arm) attempt of the slice: reset, run the agent, run the verifier, record the attempt.

Nothing is skipped or retried: a failed reset, an agent that raised or timed out, an empty answer and a verifier that
wrote no reward are all recorded as failed attempts with their reason, so the denominator is every planned attempt.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from evals.webharbor.environment import Environment, Episode, SiteError
from evals.webharbor.score import Attempt
from evals.webharbor.tasks import TaskSpec


@dataclass(frozen=True)
class Outcome:
    """One agent episode as the player reports it."""

    answer: str
    error: str | None = None  # a short reason, never provider text; "timeout" for a timed-out episode
    total_ms: int | None = None
    decision_ms: int | None = None
    screenshot: Path | None = None


Player = Callable[[TaskSpec, str, Path], Awaitable[Outcome]]


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


async def attempt(
    environment: Environment, task: TaskSpec, task_dir: Path, arm: str, play: Player, out: Path
) -> Attempt:
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        environment.reset(task)
    except SiteError as exc:
        return Attempt(task.id, arm, False, f"reset: {exc}", reset_ms=_ms(started))
    reset_ms = _ms(started)
    try:
        outcome = await play(task, arm, out)
    except Exception as exc:  # noqa: BLE001 - the type only: a provider error can quote keys
        outcome = Outcome(answer="", error=f"agent raised {type(exc).__name__}")
    status = "completed" if outcome.error is None else ("timeout" if outcome.error == "timeout" else "error")
    episode = Episode(outcome.answer, status, outcome.screenshot, time.time())
    verify_started = time.perf_counter()
    try:
        verdict = environment.verify(task, task_dir, episode, out)
    except SiteError as exc:
        verdict = None
        verify_error = f"verify: {exc}"
    verify_ms = _ms(verify_started)
    if verdict is None:
        failure: str | None = verify_error
    elif verdict.error:
        failure = f"verifier: {verdict.error}"
    elif verdict.passed:
        failure = None
    elif outcome.error:
        failure = outcome.error
    elif not outcome.answer.strip():
        failure = "no answer"
    else:
        failure = "verifier: fail" + (f" ({', '.join(verdict.failed_checks[:3])})" if verdict.failed_checks else "")
    result = Attempt(
        task.id,
        arm,
        passed=verdict is not None and verdict.passed and not verdict.error,
        failure=failure,
        total_ms=outcome.total_ms,
        decision_ms=outcome.decision_ms,
        reset_ms=reset_ms,
        verify_ms=verify_ms,
        checks_total=verdict.checks_total if verdict else None,
        checks_passed=verdict.checks_passed if verdict else None,
    )
    (out / "attempt.json").write_text(json.dumps(result.to_json(), indent=1), encoding="utf-8")
    return result


async def run_slice(
    environment: Environment,
    tasks: list[tuple[TaskSpec, Path]],
    arms: list[str],
    play: Player,
    out_root: Path,
    *,
    repeat: int = 1,
) -> list[Attempt]:
    """Every attempt in a fixed order; arms alternate which goes first per task so neither always runs on a warm host."""
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    attempts: list[Attempt] = []
    results = out_root / "attempts.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        for index, (task, task_dir) in enumerate(tasks):
            for r in range(repeat):
                order = arms if (index + r) % 2 == 0 else list(reversed(arms))
                for arm in order:
                    row = await attempt(environment, task, task_dir, arm, play, out_root / f"{task.id}__{arm}__r{r}")
                    attempts.append(row)
                    with results.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row.to_json()) + "\n")
    finally:
        environment.close()
    return attempts


def load_attempts(path: Path) -> list[Attempt]:
    return [Attempt.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
