# coding: utf-8
"""The live real-model recovery eval: one browser trial per (task, arm, repeat) through the production ``browse``.

Unlike ``evals.recovery`` (scripted doubles), the decision model and the chat/planner model here are the real ones:
``build_model`` (``jev`` over HTTP, ``laya`` or ``cua`` in process) fills the model slot and ``chat_model_from_env``
supplies the chat/planner/answer model. Nothing is injected: no scripted decisions, no standard plan and no hidden
answer reach the run, so a downstream ``DONE`` is only ever the model's own. The browser, the policy, the registered
``browser_*`` tools, the generation ids, the recovery budget and the terminal handling are all the production ones.

Success is the fixture server's recorded form POST carrying the task's expected value, never the model's answer. Each
trial opens its own fresh fixture (unique trial id, empty submit namespace) and its own browser inside ``browse``, so
an earlier trial's submit can never verify a later one. Every planned trial is recorded, errors and timeouts included.

Counting is deliberately split, and never conflated:

- Decision calls are counted by a thin proxy that appends its record at the ``_decide`` entry and fills the wall time
  and outcome in a ``finally``-safe step, so retries (``decide_many`` re-asking after an unusable answer), failures and
  cancellations are counted for real and consistently. A failed call leaves its usage unknown, not zero. The proxy
  preserves the inner model's ``name``, ``question_types``, ``deterministic``, ``supports_images``, ``model``, ``warm``
  and ``close``, so the production policy sees an identical model. Ticks are not used as a call count.
- Planner attempts are counted from the production recovery events (``stage == "planner"``), including planner calls
  that timed out or failed. A planner call is a chat call too, so it is inside ``chat_calls``; the two are never summed.
- Chat call counts and tokens come from the production ``CountingModel`` that ``browse`` wraps around the chat model
  (``answer["usage"]``); when a trial raises before that summary exists they are taken from the outer ``CountingModel``,
  which records the same calls' wall time, tokens and failure status.

A trial's ``verified`` (the fixture's recorded POST) and ``errored`` (the run raised) are independent: a trial that
posted the right value and then raised is both. Only a trial that failed before the browser existed (a fixture or
setup failure) falls back to a fabricated record.

Cost is ``null`` by default: the live eval never reads a provider catalogue, and only an explicit ``CHAT_USD_PER_M_*``
configuration values a trial as an estimate, never a bill. Even then the estimate stays ``null`` when any chat or
decision call did not report usage. (The production browser path's own ``usage_summary`` may consult OpenRouter's
catalogue for a price; the live eval deliberately ignores that and keeps its ``cost_usd`` explicit-config or ``null``.)
Model load time is measured once and listed separately from the per-trial numbers; a setup failure (a model that will
not build, a chat model with no key) exits non-zero without inventing any trial.
"""

from __future__ import annotations

import s1a.entry  # noqa: F401  # routes the harness logs to files before anything imports openjiuwen

import argparse
import asyncio
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from s1a.browser import browse, prompts
from s1a.browser.decision_model import BrowserPolicy
from s1a.config import HOME, chat_model_from_env, first_env
from s1a.counting_model import CountingModel
from s1a.decision_models import Decision, DecisionModel, Observation, Question, Reply, build_model
from s1a.jobs import RESULTS_DIR, now_iso
from s1a.pricing import ChatPrices, cost_usd, env_prices
from s1a.recovery import RecoveryLimits
from s1a.run import started_runner
from s1a.spec import BrowserAgentSpec, Budget

from evals.recovery.fixture import DEFAULT_TASKS, Fixture, FormTask, start_fixture, validated_tasks
from evals.recovery.summary import (
    coverage,
    mean,
    on_minus_off_completion_rate,
    pair_outcomes,
    paired_completion_line,
    paired_delta_line,
    summary_tables,
)

GOAL_TEMPLATE = (
    "Open {url} and complete the form: type '{value}' into the field labelled Value and click Submit. "
    "Stop once the form has been submitted."
)
LIVE_DIRNAME = "recovery_live"
ARMS = ("off", "on")
# A failed or cancelled planner/refresh call, as the production recovery event records it.
PLANNER_FAILURES = ("timeout", "error", "cancelled")
# The decision-model context variables worth recording: weights, subfolder, window and device. Never keys or URLs.
CONTEXT_ENV = (
    "LAYA_MODEL",
    "LAYA_SUBFOLDER",
    "LAYA_MAX_LEN",
    "LAYA_HEAD_MAX_LEN",
    "LAYA_DEVICE",
    "CUA_S1_CHECKPOINT",
    "CUA_S1_SUBFOLDER",
    "CUA_S1_DEVICE",
)


def _context_env() -> dict[str, str]:
    """The named decision-model context variables that are set; never the whole environment, never keys or URLs."""
    return {name: value for name in CONTEXT_ENV if (value := first_env(name))}


def _chat_identity(chat: Any) -> dict[str, str | None]:
    """The chat model's safe identity: its ``model_config``'s model name and its client provider. No key, no base URL.

    The whole config objects are not serializable (the client config carries the api key), so only the safe fields are
    read; a caller-supplied chat model is identified the same way.
    """
    request = getattr(chat, "model_config", None)
    client = getattr(chat, "model_client_config", None)
    return {
        "model_name": getattr(request, "model_name", None),
        "provider": getattr(client, "client_provider", None),
    }


@dataclass(frozen=True)
class LiveConfig:
    """The budgets a live trial runs under; defaults keep one repeat cheap for development.

    ``validate_submission`` is off by default, so the original unconstrained fixtures and their reports are unchanged.
    When on, ``run_live`` uses validated equivalents of the supplied tasks: a mismatched POST gets HTTP 422 and the
    same retryable form, never a success page. This is a server-side variant, not an injected plan or action.
    """

    timeout_s: float = 180.0
    max_steps: int = 24
    stall_after: int = 3
    max_recovery_attempts: int = 3
    recovery_timeout_s: float = 15.0
    headless: bool = True
    validate_submission: bool = False


class CountingDecisionModel(DecisionModel):
    """A proxy that counts every ``_decide`` call, retries and failures included, and changes nothing else.

    ``decide_many`` is the production policy's entry point; the retry loop inside the base class calls ``_decide``
    again on an unusable answer, so counting at ``_decide`` counts the backend calls that actually happened. The
    record is appended before the call runs and updated in a ``finally``-safe step, with wall time measured
    consistently for every outcome, so a failed or cancelled call is never lost. A failed call's tokens are unknown,
    not zero. The inner model's interface is preserved, so ``BrowserDecisionModel`` behaves exactly as with the raw
    model.
    """

    bills_input_tokens = False

    def __init__(self, inner: DecisionModel) -> None:
        self._inner = inner
        self.name = inner.name
        self.question_types = inner.question_types
        self.deterministic = inner.deterministic
        self.supports_images = inner.supports_images
        self.bills_input_tokens = inner.bills_input_tokens
        self.calls: list[dict[str, Any]] = []
        self.decide_many_calls = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def usage_known(self) -> bool:
        """Whether every counted call reported its usage: one failed call makes the total unknown, not zero."""
        return all(bool(call.get("usage_known", True)) for call in self.calls)

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "ms": 0,
            "status": "running",
            "input_tokens": 0,
            "output_tokens": 0,
            "usage_known": False,  # no reply yet: the tokens are unknown, not zero
        }
        self.calls.append(record)
        try:
            reply = await self._inner._decide(observation, questions)
        except asyncio.CancelledError:
            record["ms"] = round((time.perf_counter() - started) * 1000)
            record["status"] = "cancelled"
            raise
        except BaseException as exc:  # noqa: BLE001 - the failure is counted, then re-raised unchanged
            record["ms"] = round((time.perf_counter() - started) * 1000)
            record["status"] = "error"
            record["error"] = type(exc).__name__  # the type only: a provider error can carry headers or keys
            raise
        record["ms"] = round((time.perf_counter() - started) * 1000)
        record["status"] = "ok"
        record["input_tokens"] = int(reply.usage.input_tokens)
        record["output_tokens"] = int(reply.usage.output_tokens)
        # A successful reply may still not report usage: the backend's own completeness flag is authoritative, so an
        # explicit zero stays known while missing or malformed usage keeps the call's tokens unknown, never zero.
        record["usage_known"] = bool(reply.usage.known)
        return reply

    async def decide_many(
        self, observation: Observation, questions: dict[str, Question], *, attempts: int = 1
    ) -> Decision:
        self.decide_many_calls += 1
        return await super().decide_many(observation, questions, attempts=attempts)

    async def warm(self) -> None:
        await self._inner.warm()

    async def close(self) -> None:
        await self._inner.close()


@dataclass
class LiveTrial:
    """One planned trial's outcome, whether it verified, failed, timed out or never reached the browser."""

    task: str
    arm: str
    repeat: int
    verified: bool
    terminal: str
    errored: bool
    error: str | None
    model: str
    platform: str
    elapsed_s: float
    recovery_attempts: int = 0
    recovery_spent_s: float = 0.0
    recovery_failed: bool = False
    recovery_termination: str | None = None
    recovery_events: list[dict[str, Any]] = field(default_factory=list)
    planner_attempts: int = 0
    planner_failures: int = 0
    decision_calls: int = 0
    decision_failures: int = 0
    decision_retries: int = 0
    decision_ms_total: int = 0
    decision_ms_mean: float | None = None
    decision_input_tokens: int = 0
    decision_usage_known: bool = True
    unknown_decision_calls: int = 0
    chat_calls: int = 0
    chat_calls_observed: int = 0
    chat_failures: int = 0
    chat_ms_total: int = 0
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    chat_cache_tokens: int = 0
    usage_known: bool = False
    unknown_chat_calls: int = 0
    jev_input_tokens: int = 0
    cost_usd: float | None = None
    cost_basis: str = ""
    wasted_actions: int = 0
    steps: int = 0
    oracle: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiveRun:
    """A finished run: its manifest, every trial, the summary, and where they were written."""

    manifest: dict[str, Any]
    trials: list[LiveTrial]
    summary: dict[str, Any]
    run_dir: Path


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _planned_trials(repeat: int, tasks: tuple[FormTask, ...], arms: tuple[str, ...]) -> list[tuple[FormTask, str, int]]:
    """Every (task, arm, repeat) the run plans, with the arm order alternating by repeat to blunt startup bias."""
    planned: list[tuple[FormTask, str, int]] = []
    for task in tasks:
        for index in range(repeat):
            order = list(arms) if index % 2 == 0 else list(reversed(arms))
            for arm in order:
                planned.append((task, arm, index))
    return planned


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


def _cost(
    *,
    usage_known: bool,
    decision_usage_known: bool,
    jev_input_tokens: int,
    chat_input_tokens: int,
    chat_output_tokens: int,
    chat_cache_tokens: int,
    prices: ChatPrices | None,
) -> tuple[float | None, str]:
    """A trial's cost: ``None`` unless the user explicitly configured prices, then an estimate, never a bill.

    The estimate stays ``None`` when any call's usage is incomplete: a failed chat or decision call spent tokens that
    cannot be summed, and a partial sum must not be reported as if it were the whole bill.
    """
    if prices is None:
        return None, "unknown: live provider pricing is not verified and CHAT_USD_PER_M_* was not set"
    if not usage_known:
        return None, "unknown: at least one chat call did not report usage"
    if not decision_usage_known:
        return None, "unknown: at least one decision call did not report usage"
    value = cost_usd(jev_input_tokens, chat_input_tokens, chat_output_tokens, chat_cache_tokens, prices)
    if value is None:
        return None, "unknown: chat tokens were spent but the configured price could not value them"
    return value, "estimated from explicit CHAT_USD_PER_M_* config; not a provider bill"


def _recovery_events(recovery: dict[str, Any]) -> list[dict[str, Any]]:
    """The compact recovery events kept in a trial: the trigger, the stage and how it ended, never the page."""
    return [
        {
            "trigger": event.get("trigger"),
            "stage": event.get("stage"),
            "termination": event.get("termination"),
            "tick": event.get("tick"),
            "attempt": event.get("attempt"),
            "spent_s": event.get("spent_s"),
            "error": event.get("error"),
        }
        for event in (recovery.get("events") or [])
        if isinstance(event, dict)
    ]


async def play_trial(
    fixture: Fixture,
    task: FormTask,
    arm: str,
    repeat: int,
    *,
    config: LiveConfig,
    logs_dir: Path,
    tasks: tuple[FormTask, ...],
    model_name: str,
    chat: Any,
    decision_model: DecisionModel,
    prices: ChatPrices | None,
) -> LiveTrial:
    """Play one live trial over an already-open fixture; the caller owns the per-trial fixture and browser."""
    wall_started = time.perf_counter()
    proxy = CountingDecisionModel(decision_model)
    outer_calls: list[dict[str, Any]] = []
    counting_chat = CountingModel(chat, outer_calls)
    spec = BrowserAgentSpec(
        name="recovery_live",
        description=f"Live recovery eval fixture: {task.kind} form.",
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
    answer: dict[str, Any] = {}
    error: str | None = None
    try:
        answer = await browse.browse(
            spec,
            policy,
            model_name=model_name,
            goal=goal,
            timeout_s=config.timeout_s,
            max_steps=config.max_steps,
            logs_dir=logs_dir,
            headless=config.headless,
            chat=counting_chat,
            decision_model=proxy,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the trial keeps its real counters and oracle; the type is the summary
        error = type(exc).__name__  # the type only: a provider error can carry headers or keys in its message
    status = answer.get("status")
    if error is None and status is None and answer.get("error"):
        # browse returns a run-level failure (a timeout, a non-answer result) in ``answer['error']`` instead of
        # raising. Keep it even when the fixture already recorded the correct POST, so a verified trial that then
        # timed out is not reported as error-free; a played BLOCKED keeps its descriptive message, not an error.
        error = str(answer["error"])
    elapsed_s = round(time.perf_counter() - wall_started, 3)
    report = answer.get("report") or {}
    recovery = report.get("recovery") or {}
    usage = answer.get("usage") or {}
    history = report.get("history") or []
    events = _recovery_events(recovery)

    calls = proxy.calls
    decision_calls = len(calls)
    decision_ms_total = sum(int(call.get("ms") or 0) for call in calls)
    decision_input_tokens = sum(int(call.get("input_tokens") or 0) for call in calls)
    unknown_decision_calls = sum(1 for call in calls if not call.get("usage_known", True))
    usage_known = bool(usage.get("usage_known"))
    if usage:
        chat_calls = int(usage.get("chat_calls") or 0)
        chat_input_tokens = int(usage.get("chat_input_tokens") or 0)
        chat_output_tokens = int(usage.get("chat_output_tokens") or 0)
        chat_cache_tokens = int(usage.get("chat_cache_tokens") or 0)
        unknown_chat_calls = int(usage.get("unknown_calls") or 0)
    else:  # the browser raised before producing its own usage summary: use the outer counter's real records
        chat_calls = len(outer_calls)
        chat_input_tokens = sum(int(call.get("input_tokens") or 0) for call in outer_calls)
        chat_output_tokens = sum(int(call.get("output_tokens") or 0) for call in outer_calls)
        chat_cache_tokens = sum(int(call.get("cache_tokens") or 0) for call in outer_calls)
        usage_known = bool(outer_calls) and all(call.get("usage_known", True) for call in outer_calls)
        unknown_chat_calls = sum(1 for call in outer_calls if not call.get("usage_known", True))
    # The report's jev ticks miss validation retries; the proxy saw every actual backend reply, retries included.
    jev_input_tokens = max(int(usage.get("jev_input_tokens") or 0), decision_input_tokens if proxy.name == "jev" else 0)
    cost, basis = _cost(
        usage_known=usage_known,
        decision_usage_known=proxy.usage_known,
        jev_input_tokens=jev_input_tokens,
        chat_input_tokens=chat_input_tokens,
        chat_output_tokens=chat_output_tokens,
        chat_cache_tokens=chat_cache_tokens,
        prices=prices,
    )
    verified = fixture.verified(task)
    if verified:
        terminal = "verified"  # the oracle verified independently; any run error is still kept in ``error``
    elif status is not None:
        terminal = str(status)
    elif error is not None:
        terminal = "timeout" if "timeout" in error.lower() else f"error: {error}"
    else:
        error = "no terminal verdict"
        terminal = f"error: {error}"
    return LiveTrial(
        task=task.name,
        arm=arm,
        repeat=repeat,
        verified=verified,
        terminal=terminal,
        errored=error is not None,
        error=error,
        model=decision_model.model,
        platform=platform.platform(),
        elapsed_s=elapsed_s,
        recovery_attempts=int(recovery.get("attempts", 0) or 0),
        recovery_spent_s=float(recovery.get("spent_s", 0.0) or 0.0),
        recovery_failed=bool(recovery.get("failed")),
        recovery_termination=recovery.get("termination"),
        recovery_events=events,
        planner_attempts=sum(1 for event in events if event.get("stage") == "planner"),
        planner_failures=sum(
            1 for event in events if event.get("stage") == "planner" and event.get("termination") in PLANNER_FAILURES
        ),
        decision_calls=decision_calls,
        decision_failures=sum(1 for call in calls if call.get("status") != "ok"),
        decision_retries=max(0, decision_calls - proxy.decide_many_calls),
        decision_ms_total=decision_ms_total,
        decision_ms_mean=round(decision_ms_total / decision_calls, 1) if decision_calls else None,
        decision_input_tokens=decision_input_tokens,
        decision_usage_known=proxy.usage_known,
        unknown_decision_calls=unknown_decision_calls,
        chat_calls=chat_calls,
        chat_calls_observed=len(outer_calls),
        chat_failures=sum(1 for call in outer_calls if call.get("status") != "ok"),
        chat_ms_total=sum(int(call.get("ms") or 0) for call in outer_calls),
        chat_input_tokens=chat_input_tokens,
        chat_output_tokens=chat_output_tokens,
        chat_cache_tokens=chat_cache_tokens,
        usage_known=usage_known,
        unknown_chat_calls=unknown_chat_calls,
        jev_input_tokens=jev_input_tokens,
        cost_usd=cost,
        cost_basis=basis,
        wasted_actions=sum(
            1 for entry in history if entry.get("kind") != "wait" and entry.get("page_changed") is False
        ),
        steps=len([entry for entry in history if entry.get("kind") != "wait"]),
        oracle=fixture.oracle_state(task),
    )


async def run_trial(
    task: FormTask,
    arm: str,
    repeat: int,
    *,
    config: LiveConfig,
    logs_dir: Path,
    tasks: tuple[FormTask, ...],
    model_name: str,
    chat: Any,
    decision_model: DecisionModel,
    prices: ChatPrices | None,
) -> LiveTrial:
    """One live trial with its own fresh fixture/oracle; needs a started Runner."""
    with start_fixture(tasks) as fixture:
        return await play_trial(
            fixture,
            task,
            arm,
            repeat,
            config=config,
            logs_dir=logs_dir,
            tasks=tasks,
            model_name=model_name,
            chat=chat,
            decision_model=decision_model,
            prices=prices,
        )


def _failed_trial(
    task: FormTask, arm: str, repeat: int, exc: BaseException, *, model_id: str, elapsed_s: float
) -> LiveTrial:
    """A trial that raised before any oracle existed: recorded with its real reason in the planned denominator."""
    reason = f"{type(exc).__name__}: {exc}"
    terminal = "timeout" if isinstance(exc, asyncio.TimeoutError) else f"error: {reason}"
    return LiveTrial(
        task=task.name,
        arm=arm,
        repeat=repeat,
        verified=False,
        terminal=terminal,
        errored=True,
        error=reason,
        model=model_id,
        platform=platform.platform(),
        elapsed_s=round(elapsed_s, 3),
        cost_usd=None,
        cost_basis="unknown: the trial never produced usage",
        oracle={"task": task.name, "verified": False, "submissions": []},
    )


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _arm_block(records: list[LiveTrial]) -> dict[str, Any]:
    costs = [record.cost_usd for record in records]
    decision_means = [record.decision_ms_mean for record in records if record.decision_ms_mean is not None]
    if not costs:
        cost_total: float | None = 0.0
    elif any(cost is None for cost in costs):
        cost_total = None  # one unknown trial makes the arm's cost unknown, not zero
    else:
        cost_total = round(sum(cost for cost in costs if cost is not None), 6)
    return {
        **coverage(records),
        "terminals": _counts(record.terminal for record in records),
        "recovery_attempts": sum(record.recovery_attempts for record in records),
        "recovery_failed": sum(record.recovery_failed for record in records),
        "recovery_terminations": _counts(record.recovery_termination for record in records),
        "planner_attempts": sum(record.planner_attempts for record in records),
        "planner_failures": sum(record.planner_failures for record in records),
        "decision_calls": sum(record.decision_calls for record in records),
        "decision_failures": sum(record.decision_failures for record in records),
        "decision_retries": sum(record.decision_retries for record in records),
        "unknown_decision_calls": sum(record.unknown_decision_calls for record in records),
        "decision_usage_known": all(record.decision_usage_known for record in records) if records else None,
        "chat_calls": sum(record.chat_calls for record in records),
        "chat_failures": sum(record.chat_failures for record in records),
        "wasted_actions": sum(record.wasted_actions for record in records),
        "mean_elapsed_s": mean((record.elapsed_s for record in records), 3),
        "mean_decision_ms": mean(decision_means, 1),
        "mean_recovery_attempts": mean((float(record.recovery_attempts) for record in records), 2),
        "chat_input_tokens": sum(record.chat_input_tokens for record in records),
        "chat_output_tokens": sum(record.chat_output_tokens for record in records),
        "chat_cache_tokens": sum(record.chat_cache_tokens for record in records),
        "jev_input_tokens": sum(record.jev_input_tokens for record in records),
        "usage_known": all(record.usage_known for record in records) if records else None,
        "unknown_chat_calls": sum(record.unknown_chat_calls for record in records),
        "cost_usd": cost_total,
    }


def summarize(trials: list[LiveTrial], arms: tuple[str, ...], *, model_id: str, model_name: str) -> dict[str, Any]:
    """Every planned trial counts. Per arm, per task, and the paired off/on outcomes per (task, repeat)."""
    arms_block = {arm: _arm_block([record for record in trials if record.arm == arm]) for arm in arms}
    tasks = sorted({record.task for record in trials})
    by_task = {
        task: {arm: _arm_block([r for r in trials if r.arm == arm and r.task == task]) for arm in arms}
        for task in tasks
    }
    paired, complete = pair_outcomes(trials)
    paired = {
        **paired,
        "mean_delta_decision_calls": mean((on.decision_calls - off.decision_calls for off, on in complete), 2),
        "mean_delta_chat_calls": mean((on.chat_calls - off.chat_calls for off, on in complete), 2),
        "mean_delta_elapsed_s": mean((round(on.elapsed_s - off.elapsed_s, 3) for off, on in complete), 3),
        "mean_delta_wasted_actions": mean((on.wasted_actions - off.wasted_actions for off, on in complete), 2),
    }
    return {
        "planned_trials": len(trials),
        "model": {"requested": model_name, "resident": model_id},
        "arms": arms_block,
        "by_task": by_task,
        "paired": paired,
        "on_minus_off_completion_rate": on_minus_off_completion_rate(arms_block),
        "note": (
            "real decision and chat models on three small synthetic form pages; success is the fixture's recorded "
            "POST, not the model's DONE. This is not an open-task success rate and recovery may not trigger. "
            "chat_calls/tokens are the production CountingModel's; planner attempts are the production recovery "
            "events with stage == 'planner' and are also inside chat_calls (never summed). cost_usd is null unless "
            "CHAT_USD_PER_M_* was explicitly configured and every chat and decision call reported usage; then it is "
            "that config's estimate, not a provider bill."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """A short human-readable companion to the JSON."""
    lines: list[str] = [
        "# Bounded recovery on/off - live real-model fixture subset",
        "",
        f"Model: {summary['model']['requested']} (resident id: {summary['model']['resident']}).",
        f"Planned trials: {summary['planned_trials']} (errors and timeouts included, never dropped).",
        "",
    ]
    lines += summary_tables(
        summary,
        [
            ("errored", "errored"),
            ("recovery attempts", "recovery_attempts"),
            ("planner attempts", "planner_attempts"),
            ("decision calls", "decision_calls"),
            ("chat calls", "chat_calls"),
            ("mean elapsed s", "mean_elapsed_s"),
            ("wasted actions", "wasted_actions"),
            ("cost usd", "cost_usd"),
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
                "mean_delta_decision_calls": "decision calls",
                "mean_delta_chat_calls": "chat calls",
                "mean_delta_elapsed_s": "elapsed s",
                "mean_delta_wasted_actions": "wasted actions",
            },
        ),
        "",
        f"Note: {summary['note']}",
        "",
    ]
    return "\n".join(lines)


def _write_json(path: Path, data: Any) -> None:
    """Write one JSON artifact, creating its parent: a finished trial survives a batch kill that stops the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_live(
    *,
    model_name: str,
    repeat: int = 1,
    tasks: tuple[FormTask, ...] = DEFAULT_TASKS,
    arms: tuple[str, ...] = ARMS,
    config: LiveConfig | None = None,
    results_dir: Path = RESULTS_DIR,
    chat: Any = None,
) -> LiveRun:
    """Build the real models once, run every planned trial, and write the manifest, trials and summary.

    The decision and chat models are loaded once and their load time is reported separately from the trials. The
    Runner is held in one process; each trial opens its own fixture and ``browse`` tears down its own browser. A trial
    that raises inside ``browse`` is recorded with its real counters and oracle in the planned denominator, never
    dropped. A setup failure (no key, no weights, a half-set price) propagates before any trial, so the caller can
    exit non-zero without a fake run. ``chat`` lets the caller supply the chat/planner model (for example a real model
    with custom headers); when it is ``None`` the environment's chat model is built.

    A unique run directory is created and the planned manifest written before the first trial, and each finished trial
    is written on its own, so a batch scheduler's kill does not erase completed evidence. There is no resume engine.
    """
    _validate_run(repeat, tasks, arms)
    config = config or LiveConfig()
    if config.validate_submission:
        # The supplied tasks are immutable and stay untouched; the run uses validated equivalents of them.
        tasks = validated_tasks(tasks)
    results_dir = Path(results_dir)
    run_id = f"{datetime.now():%Y-%m-%d__%H-%M-%S}-{uuid4().hex[:6]}"
    run_dir = results_dir / LIVE_DIRNAME / run_id
    trials_dir = run_dir / "trials"
    logs_root = HOME / "runs" / "recovery_live" / run_id
    prices = env_prices()  # explicit user config only; a half-set pair raises here, before any trial
    planned = _planned_trials(repeat, tasks, arms)

    started = now_iso()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "kind": "recovery_live",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "started_at": started,
        "status": "planned",
        "tasks": [task.name for task in tasks],
        "arms": list(arms),
        "repeat": repeat,
        "planned_trials": len(planned),
        "plan": [{"task": task.name, "arm": arm, "repeat": index} for task, arm, index in planned],
        "config": asdict(config),
        "results_dir": str(results_dir),
    }
    _write_json(run_dir / "manifest.json", manifest)  # the run id exists and the plan is on disk before any model

    decision_started = time.perf_counter()
    decision_model = build_model(model_name)
    decision_load_ms = round((time.perf_counter() - decision_started) * 1000)
    model_id = decision_model.model
    trials: list[LiveTrial] = []
    try:
        chat_started = time.perf_counter()
        if chat is None:
            chat = chat_model_from_env()
            chat_load_ms = round((time.perf_counter() - chat_started) * 1000)
            chat_load_note = "one-time load for the whole run; excluded from every per-trial number"
        else:
            chat_load_ms = 0
            chat_load_note = "chat model supplied by the caller; its load time is not measured here"
        manifest.update(
            {
                "status": "running",
                "model": {"requested": model_name, "resident": model_id},
                "chat": _chat_identity(chat),
                "context": _context_env(),
                "model_load_ms": {
                    "decision": decision_load_ms,
                    "chat": chat_load_ms,
                    "total": decision_load_ms + chat_load_ms,
                    "note": chat_load_note,
                },
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        async with started_runner():
            for ordinal, (task, arm, index) in enumerate(planned):
                logs_dir = logs_root / f"{task.name}__{arm}__r{index}"
                trial_started = time.perf_counter()
                try:
                    trial = await run_trial(
                        task,
                        arm,
                        index,
                        config=config,
                        logs_dir=logs_dir,
                        tasks=tasks,
                        model_name=model_name,
                        chat=chat,
                        decision_model=decision_model,
                        prices=prices,
                    )
                except Exception as exc:  # noqa: BLE001 - a raised trial is a planned failure, not a drop
                    trial = _failed_trial(
                        task,
                        arm,
                        index,
                        exc,
                        model_id=model_id,
                        elapsed_s=time.perf_counter() - trial_started,
                    )
                trials.append(trial)
                _write_json(
                    trials_dir / f"{ordinal:02d}__{task.name}__{arm}__r{index}.json", trial.as_json()
                )  # a finished trial is on disk before the next one starts
    finally:
        await decision_model.close()

    summary = summarize(trials, arms, model_id=model_id, model_name=model_name)
    manifest["status"] = "finished"
    manifest["finished_at"] = now_iso()
    manifest["completed_trials"] = len(trials)
    manifest["cost"] = {
        "default": "null unless CHAT_USD_PER_M_* is explicitly set",
        "configured": prices is not None,
        "basis": (
            "estimate from explicit CHAT_USD_PER_M_* config; not a provider bill"
            if prices is not None
            else "unknown: live provider pricing is not verified here"
        ),
        "note": (
            "the live eval never reads a provider catalogue; even with explicit prices a trial is null when any "
            "call's usage is incomplete"
        ),
    }
    manifest["note"] = summary["note"]
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "trials.json", [trial.as_json() for trial in trials])
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return LiveRun(manifest=manifest, trials=trials, summary=summary, run_dir=run_dir)


def format_run(run: LiveRun) -> str:
    return "\n".join(
        [
            render_markdown(run.summary),
            "Outputs:",
            f"- run dir: {run.run_dir}",
            f"- manifest: {run.run_dir / 'manifest.json'}",
            f"- trials: {run.run_dir / 'trials.json'}",
            f"- per-trial: {run.run_dir / 'trials'}",
            f"- summary: {run.run_dir / 'summary.json'}",
        ]
    )


def parser() -> argparse.ArgumentParser:
    build = argparse.ArgumentParser(prog="python -m evals.recovery.live", description=__doc__)
    build.add_argument(
        "--model",
        choices=("jev", "laya", "cua"),
        required=True,
        help="the real decision model: jev over HTTP, laya or cua in process",
    )
    build.add_argument("--repeat", type=int, default=1, help="repeats per (task, arm), paired; must be >= 1")
    build.add_argument(
        "--tasks",
        default=",".join(task.name for task in DEFAULT_TASKS),
        help="comma-separated fixture tasks (normal,recoverable,permanently_blocked)",
    )
    build.add_argument("--arms", default=",".join(ARMS), help="comma-separated recovery arms (off,on)")
    build.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="root of the live run folders")
    build.add_argument("--timeout", type=float, default=LiveConfig.timeout_s, help="seconds per trial")
    build.add_argument("--max-steps", type=int, default=LiveConfig.max_steps, help="the subagent's iteration cap")
    build.add_argument("--stall-after", type=int, default=LiveConfig.stall_after, help="actions without a page change")
    build.add_argument(
        "--recovery-attempts", type=int, default=LiveConfig.max_recovery_attempts, help="bounded recovery attempts"
    )
    build.add_argument(
        "--recovery-timeout",
        type=float,
        default=LiveConfig.recovery_timeout_s,
        help="bounded recovery active seconds per trial",
    )
    build.add_argument("--headed", action="store_true", help="show the browser (default: headless)")
    build.add_argument(
        "--validate-submission",
        action="store_true",
        help="reject a mismatched form value with HTTP 422 and the same retryable form (default: off, unchanged)",
    )
    return build


def main(argv: list[str] | None = None) -> int:
    parser_ = parser()
    args = parser_.parse_args(argv)
    task_table = {task.name: task for task in DEFAULT_TASKS}
    try:
        tasks = tuple(task_table[name] for name in _csv(args.tasks))
    except KeyError as exc:
        parser_.error(f"unknown task {exc}; expected any of {sorted(task_table)}")
    arms = tuple(_csv(args.arms))
    config = LiveConfig(
        timeout_s=args.timeout,
        max_steps=args.max_steps,
        stall_after=args.stall_after,
        max_recovery_attempts=args.recovery_attempts,
        recovery_timeout_s=args.recovery_timeout,
        headless=not args.headed,
        validate_submission=args.validate_submission,
    )
    try:
        run = asyncio.run(
            run_live(
                model_name=args.model,
                repeat=args.repeat,
                tasks=tasks,
                arms=arms,
                config=config,
                results_dir=args.results_dir,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a setup failure is reported as such and exits non-zero
        print(f"live recovery eval setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(format_run(run))
    return 1 if any(trial.errored for trial in run.trials) else 0


if __name__ == "__main__":
    raise SystemExit(main())
