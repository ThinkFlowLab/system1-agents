# coding: utf-8
"""The recovery eval runner: one real browser trial per (task, arm, repeat) through the repository's ``browse``.

The browser, the model slot (``BrowserDecisionModel``), the registered ``browser_*`` tools and their generation ids,
the recovery budget and the terminal handling are all the production ones; ``create_browser_agent`` supplies the same
runtime. The decision model and the fallback chat model are the scripted doubles (``evals.recovery.scripted``); there
is no real-backend mode here. Batch actions and ``unsafe_dev`` stay off. The fixture server's recorded POST is the
independent oracle, and every trial gets its own fresh fixture (see ``run_trial``) so no earlier submit can verify it.

Time is reported as separate numbers and never conflated with inference: ``elapsed_s`` is the whole trial wall clock
(it includes harness and browser startup), ``decisions_ms`` is the scripted decision time measured with
``perf_counter`` (near-zero by construction and not a benchmark). ``recorded_probe_ms`` sums recorded probe wall times,
already including action settling. It excludes recovery, final and unticked probes, so it is not total environment
time. Cost is 0 by construction because no paid API is called.
"""

from __future__ import annotations

import asyncio
import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from s1a.browser import browse, prompts
from s1a.browser.decision_model import BrowserPolicy
from s1a.config import HOME
from s1a.decision_models import DecisionModel
from s1a.jobs import RESULTS_DIR, Episode, now_iso, write_job
from s1a.recovery import RecoveryLimits
from s1a.run import started_runner
from s1a.spec import BrowserAgentSpec, Budget

from evals.recovery.fixture import DEFAULT_TASKS, Fixture, FormTask, start_fixture
from evals.recovery.scripted import SCRIPTED_MODEL_ID, ScriptedChatModel, ScriptedRecoveryModel
from evals.recovery.summary import ARMS, TrialRecord, paired_summary, render_markdown

PLAN_HINT = (
    "The field keeps reverting after typing. Click the control labelled 'Enable editing' first, then type the value "
    "and submit the form."
)
GOAL_TEMPLATE = (
    "Open {url} and complete the form: type '{value}' into the field labelled Value and click Submit. "
    "Stop once the form has been submitted."
)


@dataclass(frozen=True)
class EvalConfig:
    """The budgets a trial runs under; defaults keep one repeat cheap for development."""

    timeout_s: float = 120.0
    max_steps: int = 24
    stall_after: int = 3
    max_recovery_attempts: int = 3
    recovery_timeout_s: float = 15.0
    headless: bool = True


@dataclass
class EvalRun:
    summary: dict[str, Any]
    records: list[TrialRecord] = field(default_factory=list)
    job_dirs: list[Path] = field(default_factory=list)
    paired_dir: Path | None = None


def _scripted_models(task: FormTask) -> tuple[DecisionModel, Any]:
    """The controlled doubles: a fault-injecting decision model and a planner/answer chat model, both offline."""
    return ScriptedRecoveryModel(task), ScriptedChatModel(value=task.expected_value, plan=PLAN_HINT)


def _trial_metrics(answer: dict[str, Any], decision_model: DecisionModel, chat: Any) -> dict[str, Any]:
    report = answer.get("report") or {}
    history = report.get("history") or []
    ticks = answer.get("ticks") or []
    decision_calls = int(getattr(decision_model, "decide_calls", 0) or 0)
    planner_calls = int(getattr(chat, "planner_calls", 0) or 0)  # a planner call is a chat call too, never added again
    chat_calls = int(getattr(chat, "invoke_calls", 0) or 0)
    return {
        "wasted_actions": sum(
            1 for entry in history if entry.get("kind") != "wait" and entry.get("page_changed") is False
        ),
        # probe_ms already includes action settling; adding the settle fields would double-count.
        # This is a diagnostic over recorded ticks, not total environment time.
        "recorded_probe_ms": sum(int(tick.get("probe_ms", 0) or 0) for tick in ticks),
        "decisions_ms": sum(int(tick.get("decision_ms", 0) or 0) for tick in ticks),
        "decision_calls": decision_calls,
        "planner_calls": planner_calls,
        "chat_calls": chat_calls,
        "model_calls": decision_calls + chat_calls,  # planner_calls is already inside chat_calls
    }


def _decisions(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": tick.get("tick"),
            "key": tick.get("operation"),
            "target": tick.get("target"),
            "ms": tick.get("decision_ms", 0),
            "source": SCRIPTED_MODEL_ID,
        }
        for tick in ticks
    ]


async def run_trial(
    task: FormTask,
    arm: str,
    repeat: int,
    *,
    config: EvalConfig,
    logs_dir: Path,
    tasks: tuple[FormTask, ...] = DEFAULT_TASKS,
) -> tuple[Episode, TrialRecord]:
    """One real browser trial with its own fresh fixture/oracle; needs a started Runner.

    A new fixture server is opened for this trial alone, so its submit records start empty and no earlier trial's
    success can carry over. Both the CLI and the system test call this function and nothing else.
    """
    with start_fixture(tasks) as fixture:
        return await play_trial(fixture, task, arm, repeat, config=config, logs_dir=logs_dir)


async def play_trial(
    fixture: Fixture,
    task: FormTask,
    arm: str,
    repeat: int,
    *,
    config: EvalConfig,
    logs_dir: Path,
) -> tuple[Episode, TrialRecord]:
    """Play one trial over an already-open fixture; ``run_trial`` owns the per-trial fixture, tests may pass one."""
    started_at = now_iso()
    wall_started = time.perf_counter()
    decision_model, chat = _scripted_models(task)
    spec = BrowserAgentSpec(
        name="recovery_eval",
        description=f"Recovery eval fixture: {task.kind} form.",
        rules=prompts.OPERATION_RULES["en"],
        language="en",
        budget=Budget(max_steps=config.max_steps, timeout_s=config.timeout_s, stall_after=config.stall_after),
        goal=None,
    )
    policy = BrowserPolicy(
        prefetch_values=False,
        batch_actions=False,
        goal_value_cache=False,
        rethink_on=arm == "on",
        recovery_limits=RecoveryLimits(max_attempts=config.max_recovery_attempts, timeout_s=config.recovery_timeout_s),
    )
    goal = GOAL_TEMPLATE.format(url=fixture.task_url(task), value=task.expected_value)
    # The scripted model is handed to browse's decision-model slot; that branch is keyed "jev" but no Jev client
    # exists here, so no API is called. A raise still lets this trial keep its own fixture oracle.
    answer: dict[str, Any] = {}
    run_error: str | None = None
    try:
        answer = await browse.browse(
            spec,
            policy,
            model_name="jev",
            goal=goal,
            timeout_s=config.timeout_s,
            max_steps=config.max_steps,
            logs_dir=logs_dir,
            headless=config.headless,
            chat=chat,
            decision_model=decision_model,
        )
    except Exception as exc:  # noqa: BLE001 - the type is the summary; a provider error can carry headers or keys
        run_error = type(exc).__name__
    status = answer.get("status")
    if run_error is None and status is None and answer.get("error"):
        # browse returns a run-level failure (a timeout, a non-answer result) in ``answer['error']`` rather than
        # raising. Only a missing terminal verdict is a harness failure; a played BLOCKED keeps its descriptive
        # message without becoming an errored trial, while a timeout after a real POST is still kept.
        run_error = str(answer["error"])
    elapsed_s = round(time.perf_counter() - wall_started, 3)
    report = answer.get("report") or {}
    terminal = answer.get("terminal") or {}
    recovery = report.get("recovery") or {}
    ticks = answer.get("ticks") or []
    verified = fixture.verified(task)
    if verified:
        terminal_name = "verified"  # the oracle verified independently; a run error is still kept below
    elif status is not None:
        terminal_name = str(status)  # a played episode that did not submit (e.g. BLOCKED): score 0
    elif run_error is not None:
        terminal_name = "timeout" if "timeout" in run_error.lower() else f"error: {run_error}"
    else:
        run_error = "no terminal verdict"  # the harness could not play the episode at all
        terminal_name = f"error: {run_error}"
    error = run_error
    metrics = _trial_metrics(answer, decision_model, chat)
    record = TrialRecord(
        task=task.name,
        arm=arm,
        repeat=repeat,
        verified=verified,
        terminal=terminal_name,
        errored=error is not None,
        scripted_model=SCRIPTED_MODEL_ID,
        platform=platform.platform(),
        recovery_attempts=int(recovery.get("attempts", 0) or 0),
        recovery_spent_s=float(recovery.get("spent_s", 0.0) or 0.0),
        recovery_failed=bool(recovery.get("failed")),
        recovery_termination=recovery.get("termination"),
        wasted_actions=metrics["wasted_actions"],
        model_calls=metrics["model_calls"],
        decision_calls=metrics["decision_calls"],
        planner_calls=metrics["planner_calls"],
        chat_calls=metrics["chat_calls"],
        elapsed_s=elapsed_s,
        decisions_ms=metrics["decisions_ms"],
        recorded_probe_ms=metrics["recorded_probe_ms"],
        cost_usd=0.0,
        oracle=fixture.oracle_state(task),
    )
    episode = Episode(
        env="recovery",
        policy=f"recovery_{arm}",
        seed=repeat,
        score=1.0 if verified else 0.0,
        steps=len([entry for entry in (report.get("history") or []) if entry.get("kind") != "wait"]),
        elapsed_s=elapsed_s,
        started_at=started_at,
        finished_at=now_iso(),
        final_state={
            "task": task.name,
            "arm": arm,
            "verified": verified,
            "terminal": terminal,
            "recovery": recovery,
            "oracle": fixture.oracle_state(task),
        },
        chat_calls=int(getattr(chat, "invoke_calls", 0)),
        chat_input_tokens=0,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        jev_input_tokens=int(report.get("jev_input_tokens", 0) or 0),
        invalid_keys=0,
        cost_usd=0.0,
        error=error,
        decisions=_decisions(ticks),
        extra=record.as_json(),
    )
    return episode, record


def _failed_episode(
    task: FormTask, arm: str, repeat: int, exc: BaseException, elapsed_s: float
) -> tuple[Episode, TrialRecord]:
    """A trial that raised: recorded with its real reason in the planned denominator, never dropped and never 'ok'."""
    stamp = now_iso()
    reason = f"{type(exc).__name__}: {exc}"
    terminal = "timeout" if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else f"error: {reason}"
    record = TrialRecord(
        task=task.name,
        arm=arm,
        repeat=repeat,
        verified=False,
        terminal=terminal,
        errored=True,
        scripted_model=SCRIPTED_MODEL_ID,
        platform=platform.platform(),
        elapsed_s=round(elapsed_s, 3),
        cost_usd=0.0,
        oracle={"task": task.name, "verified": False, "submissions": []},
    )
    episode = Episode(
        env="recovery",
        policy=f"recovery_{arm}",
        seed=repeat,
        score=0.0,
        steps=0,
        elapsed_s=round(elapsed_s, 3),
        started_at=stamp,
        finished_at=stamp,
        final_state={"task": task.name, "arm": arm, "verified": False, "failed": reason},
        chat_calls=0,
        chat_input_tokens=0,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        jev_input_tokens=0,
        invalid_keys=0,
        cost_usd=0.0,
        error=reason,
        extra=record.as_json(),
    )
    return episode, record


def _validate_run(repeat: int, tasks: tuple[FormTask, ...], arms: tuple[str, ...]) -> None:
    """Reject arguments that would otherwise fabricate a plausible-looking but meaningless report."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if not tasks:
        raise ValueError("at least one fixture task is required")
    if not arms:
        raise ValueError("at least one arm is required")
    if len(set(arms)) != len(arms):
        raise ValueError(f"arms must be distinct, got {arms}")
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; expected any of {ARMS}")
    names = [task.name for task in tasks]
    if len(set(names)) != len(names):
        raise ValueError(f"fixture task names must be distinct, got {names}")


async def run_eval(
    *,
    repeat: int = 1,
    tasks: tuple[FormTask, ...] = DEFAULT_TASKS,
    config: EvalConfig | None = None,
    results_dir: Path = RESULTS_DIR,
    arms: tuple[str, ...] = ARMS,
) -> EvalRun:
    """Run every (task, repeat, arm) once, write the Harbor jobs and the paired summary, and return them.

    The Runner is held in one process; each trial opens its own fixture and tears down its own browser inside
    ``browse``. The arm order alternates with the repeat so the first arm run in a pair is not always the same
    startup; a trial that raises is recorded with its real reason in the planned denominator, never dropped.
    """
    _validate_run(repeat, tasks, arms)
    config = config or EvalConfig()
    # The clock alone has one-second precision, so two runs started in the same second would share every path; the
    # random suffix keeps their logs and paired summaries apart.
    run_id = f"{datetime.now():%Y-%m-%d__%H-%M-%S}-{uuid4().hex[:6]}"
    logs_root = HOME / "runs" / "recovery" / run_id
    episodes_by_arm: dict[str, list[Episode]] = {arm: [] for arm in arms}
    records: list[TrialRecord] = []
    async with started_runner():
        for task in tasks:
            for index in range(repeat):
                order = list(arms) if index % 2 == 0 else list(reversed(arms))
                for arm in order:
                    logs_dir = logs_root / f"{task.name}__{arm}__r{index}"
                    started = time.perf_counter()
                    try:
                        episode, record = await run_trial(
                            task, arm, index, config=config, logs_dir=logs_dir, tasks=tasks
                        )
                    except Exception as exc:  # noqa: BLE001 - a trial that raised is a planned failure, not a drop
                        episode, record = _failed_episode(task, arm, index, exc, time.perf_counter() - started)
                    episodes_by_arm[arm].append(episode)
                    records.append(record)
    job_dirs: list[Path] = []
    for arm in arms:
        if episodes_by_arm[arm]:
            job_dirs.append(write_job("recovery", episodes_by_arm[arm], results_dir=results_dir))
    summary = paired_summary(records)
    paired_dir = results_dir / "recovery_paired" / run_id
    paired_dir.mkdir(parents=True, exist_ok=True)
    (paired_dir / "paired_summary.json").write_text(_json_summary(summary, records), encoding="utf-8")
    (paired_dir / "paired_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return EvalRun(summary=summary, records=records, job_dirs=job_dirs, paired_dir=paired_dir)


def _json_summary(summary: dict[str, Any], records: list[TrialRecord]) -> str:
    return json.dumps(
        {
            "summary": summary,
            "trials": [record.as_json() for record in records],
            "definition": {
                "planned_trials": "every (task, repeat, arm) the run planned; errors and timeouts included",
                "verified": "this trial's own fixture server saw a real submit carrying the task's expected value",
                "oracle": "per-trial fixture namespace; a fresh empty fixture is opened for every trial",
                "wasted_actions": "policy history entries with kind != wait and page_changed == False",
                "elapsed_s": "whole trial wall clock, includes harness and browser startup (not inference)",
                "decisions_ms": "scripted decision time measured with perf_counter, summed over ticks (near-zero, not a benchmark)",
                "recorded_probe_ms": "sum of recorded probe wall times, including action settling; excludes recovery, final and unticked probes; not total environment time",
                "model_calls": "decision_calls + chat_calls (all scripted doubles); planner_calls is a subset of chat_calls",
                "cost_usd": "0 by construction: scripted decisions and planner, no paid API call",
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def format_run(run: EvalRun) -> str:
    lines = [render_markdown(run.summary), "Outputs:"]
    lines += [f"- job: {path}" for path in run.job_dirs]
    if run.paired_dir is not None:
        lines.append(f"- paired summary: {run.paired_dir}")
    return "\n".join(lines)


def run_sync(**kwargs: Any) -> EvalRun:
    return asyncio.run(run_eval(**kwargs))
