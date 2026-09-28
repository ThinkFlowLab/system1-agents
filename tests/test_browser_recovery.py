# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Modifications Copyright 2026 ThinkFlowLab
# SPDX-License-Identifier: Apache-2.0

"""The browser policy's bounded recovery: a stall refreshes the page and plans, under a per-task budget.

The turn loop, the action space and the wire are the scripted doubles of ``test_browser_policy``; these tests only
add the recovery branch, so the off path stays the legacy BLOCKED path and the on path never widens the tool set.

A stall is built from repeated ``TYPE_TEXT`` actions: typing never retires a target the way repeated clicks do, so
the run keeps a stable, answerable action space across the stall and the recovery.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from openjiuwen.core.foundation.llm import Model
from openjiuwen.core.foundation.tool.schema import ToolInfo
from test_browser_policy import (
    _FAILED_PROBE_ENVELOPE,
    _MESSAGES,
    _SNAPSHOT,
    _TOOLS,
    _Wire,
    _answer,
    _answers,
    _fake_model_init,
    _fallback,
    _spec,
    _wire,
)

from s1a.browser import browse
from s1a.browser.decision_model import PROBE_QUIET_MS, PROBE_SETTLE_MS, BrowserDecisionModel, BrowserPolicy
from s1a.decision_models import JevModel
from s1a.recovery import RecoveryLimits
from s1a.tool.rethink import REPLAN_PROMPT

_FULL_OPS = ["CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "WAIT", "DONE", "BLOCKED"]
_NO_SELECT_OPS = ["CLICK", "TYPE_TEXT", "SCROLL_DOWN", "WAIT", "DONE", "BLOCKED"]
_NO_SELECT_TOOLS = [ToolInfo(name=name) for name in ("browser_click", "browser_type", "browser_press_key")]


class _RecoveryProbeRuntime:
    """Stock page on decision and settle probes; the recovery raw probe returns a scripted override, fails or hangs.

    A recovery probe is told apart by its stock settle/quiet window and ``after=None``. The run's first such probe is
    its initial observation; every later one is a recovery probe, because the action-settle probes use the action's
    own wait as their quiet window and the answer probe sets ``all_text``.
    """

    def __init__(self, *, overrides: dict[str, Any] | None = None, fail: bool = False, hang: bool = False) -> None:
        self.overrides = overrides or {}
        self.fail = fail
        self.hang = hang
        self.calls: list[dict[str, Any]] = []
        self._initial = False

    @staticmethod
    def _is_recovery_probe(params: dict[str, Any]) -> bool:
        return (
            params.get("after") is None
            and params.get("settle_ms") == PROBE_SETTLE_MS
            and params.get("quiet_ms") == PROBE_QUIET_MS
            and not params.get("all_text")
        )

    async def probe_for_policy(self, source: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(params)
        if self._is_recovery_probe(params):
            if not self._initial:
                self._initial = True  # the run's first observation, not a recovery probe
            elif self.hang:
                await asyncio.sleep(30)
            elif self.fail:
                return json.loads(json.dumps(_FAILED_PROBE_ENVELOPE))
            else:
                snapshot = json.loads(json.dumps(_SNAPSHOT))
                snapshot.update(self.overrides)
                return snapshot
        return json.loads(json.dumps(_SNAPSHOT))


def _type(operations: list[str] | None = None) -> dict[str, Any]:
    """A TYPE_TEXT answer over the full heads; ``operations`` narrows the operation head to a filtered tool set."""
    payload = _answers("TYPE_TEXT", "none")
    if operations is not None:
        payload["answers"]["operation"] = _answer("TYPE_TEXT", operations)
    return payload


def _planner_fallback(*, plan: str = "use the Search button", error: Exception | None = None, hang: Any = None) -> Any:
    """The stock fallback, with the replan call scripted: a plan, an error or a wait; every other call is stock."""
    fallback = _fallback()
    original = fallback.invoke

    async def invoke(messages: list[dict[str, str]], **kwargs: Any) -> Any:
        if messages and messages[0].get("content") == REPLAN_PROMPT:
            fallback.asked.append(messages)  # the stock invoke records its calls; the replan branch does too
            if error is not None:
                raise error
            if hang is not None:
                await hang.wait()
            return mock.MagicMock(content=plan)
        return await original(messages, **kwargs)

    fallback.invoke = invoke
    return fallback


def _slot(
    scripted: list[dict[str, Any]],
    *,
    runtime: Any,
    rethink: bool = True,
    limits: RecoveryLimits | None = None,
    fallback: Any = None,
) -> BrowserDecisionModel:
    """The slot model with bounded recovery switched by ``rethink`` and its own runtime and planner double."""
    fallback = fallback if fallback is not None else _fallback()
    policy = BrowserPolicy(
        prefetch_values=False,
        batch_actions=False,
        goal_value_cache=False,
        rethink_on=rethink,
        recovery_limits=limits or RecoveryLimits(),
    )
    with mock.patch.object(Model, "__init__", _fake_model_init):
        slot = BrowserDecisionModel(
            _spec(), policy, fallback, decision_model=JevModel(_Wire(scripted)), value_model=None
        )
    slot.bind_runtime(runtime)
    return slot


def _replan_asks(fallback: Any) -> list[list[dict[str, str]]]:
    return [asked for asked in fallback.asked if asked and asked[0].get("content") == REPLAN_PROMPT]


class TestBrowserRecoveryOff(IsolatedAsyncioTestCase):
    """With rethink off the legacy stop points still end BLOCKED, without touching a budget."""

    async def test_off_still_blocks_a_stall_without_a_recovery_event(self) -> None:
        slot = _slot([_type() for _ in range(3)], runtime=_RecoveryProbeRuntime(), rethink=False)

        for _ in range(3):
            self.assertEqual((await slot.invoke(_MESSAGES, tools=_TOOLS)).tool_calls[0].name, "browser_type")
        summary = json.loads((await slot.invoke(_MESSAGES, tools=_TOOLS)).content)

        self.assertEqual(summary["status"], "BLOCKED")
        self.assertEqual(summary["reason"], "3 actions without page change")
        self.assertEqual(slot.report()["recovery"]["attempts"], 0)
        self.assertEqual(slot.report()["recovery"]["events"], [])
        self.assertTrue(slot._run.finished)


class TestBrowserRecoveryPlan(IsolatedAsyncioTestCase):
    """A stall on the on path spends one refresh and one plan, then the normal decision picks the action."""

    async def test_stall_refreshes_plans_then_chooses_a_normal_tool(self) -> None:
        fallback = _planner_fallback(plan="fill the field, then search")
        slot = _slot([_type() for _ in range(4)], runtime=_RecoveryProbeRuntime(), fallback=fallback)

        for _ in range(3):
            self.assertEqual((await slot.invoke(_MESSAGES, tools=_TOOLS)).tool_calls[0].name, "browser_type")
        recovered = await slot.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(recovered.tool_calls[0].name, "browser_type", "the plan never executes a click itself")
        body = _wire(slot).bodies[3]
        self.assertEqual(body["state"]["plan"], "fill the field, then search", "the plan rides into the next turn")
        self.assertNotIn("plan", _wire(slot).bodies[2]["state"], "the plan is not smeared across turns")
        self.assertTrue(_replan_asks(fallback), "the chat model writes the plan")
        recovery = slot.report()["recovery"]
        self.assertEqual((recovery["attempts"], recovery["events"][0]["termination"]), (1, "planned"))
        self.assertEqual(recovery["events"][0]["trigger"], "stalled")
        self.assertTrue(recovery["events"][0]["recent_actions"])
        self.assertIsNotNone(recovery["events"][0]["fresh_obs"])
        self.assertTrue(slot.report()["history"], "history survives an attempt")

    async def test_a_plan_cannot_recreate_a_tool_the_run_was_not_offered(self) -> None:
        fallback = _planner_fallback()
        slot = _slot([_type(_NO_SELECT_OPS) for _ in range(4)], runtime=_RecoveryProbeRuntime(), fallback=fallback)

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_NO_SELECT_TOOLS)
        await slot.invoke(_MESSAGES, tools=_NO_SELECT_TOOLS)

        replans = _replan_asks(fallback)
        self.assertTrue(replans)
        context = json.loads(replans[-1][1]["content"])
        self.assertNotIn("SELECT", context["candidates"]["operations"], "the planner sees the filtered candidates")
        operation = _wire(slot).bodies[3]["questions"]["operation"]["criteria"]
        self.assertNotIn("SELECT", operation, "the turn itself never offers the missing tool back")

    async def test_fresh_page_progress_skips_the_planner(self) -> None:
        fallback = _planner_fallback()
        runtime = _RecoveryProbeRuntime(overrides={"page_key": "k2"})
        slot = _slot([_type() for _ in range(4)], runtime=runtime, fallback=fallback)

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        await slot.invoke(_MESSAGES, tools=_TOOLS)

        event = slot.report()["recovery"]["events"][0]
        self.assertEqual(event["termination"], "delayed_progress")
        self.assertEqual(event["fresh_obs"]["page_key"], "k2")
        self.assertFalse(_replan_asks(fallback), "delayed progress means there is nothing to replan")

    async def test_recovered_action_uses_the_fresh_snapshot(self) -> None:
        elements = [{**item, "target_id": f"t_g2_{index}"} for index, item in enumerate(_SNAPSHOT["elements"], 1)]
        runtime = _RecoveryProbeRuntime(overrides={"generation_id": "g2", "elements": elements})
        slot = _slot([_type() for _ in range(4)], runtime=runtime)

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        recovered = await slot.invoke(_MESSAGES, tools=_TOOLS)

        args = json.loads(recovered.tool_calls[0].arguments)
        self.assertEqual((args["generation_id"], args["target_id"]), ("g2", "t_g2_2"), "a stale target is re-read")


class _ProgressThenStallRuntime(_RecoveryProbeRuntime):
    """The page really moves once, then stays put, so the history has progress and the stall guard still fires.

    The recovery refresh returns the same page the stall saw, so the planner runs instead of reporting delayed
    progress. This is the shape a live run had: replans happened, then the policy answered BLOCKED.
    """

    def __init__(self) -> None:
        super().__init__(overrides={"page_key": "k2"})
        self._decision_probes = 0

    async def probe_for_policy(self, source: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._is_recovery_probe(params):
            return await super().probe_for_policy(source, params)
        self._decision_probes += 1
        snapshot = json.loads(json.dumps(_SNAPSHOT))
        if self._decision_probes >= 1:  # every decision probe after the first action sees the moved page
            snapshot["page_key"] = "k2"
        return snapshot


class TestBrowserRecoveryTerminalBlock(IsolatedAsyncioTestCase):
    """A policy that still answers BLOCKED after a replan is a failure, never a success through a partial answer."""

    async def test_blocked_after_a_replan_keeps_the_partial_answer_as_context_only(self) -> None:
        fallback = _planner_fallback()
        slot = _slot(
            [_type() for _ in range(4)] + [_answers("BLOCKED", "none")],
            runtime=_ProgressThenStallRuntime(),
            limits=RecoveryLimits(max_attempts=3, timeout_s=15.0),
            fallback=fallback,
        )

        for _ in range(4):
            self.assertEqual((await slot.invoke(_MESSAGES, tools=_TOOLS)).tool_calls[0].name, "browser_type")
        summary = json.loads((await slot.invoke(_MESSAGES, tools=_TOOLS)).content)

        recovery = summary["recovery"]
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertFalse(recovery["failed"], "a model BLOCKED is not a failed recovery")
        self.assertFalse(recovery["exhausted"], "a model BLOCKED is not a spent budget")
        self.assertTrue(recovery["blocked_after_recovery"])
        self.assertEqual(recovery["termination"], "planned", "the last recovery did plan, it did not give up")
        self.assertIn("BLOCKED after 1 recovery attempt", recovery["reason"])
        self.assertIn("start a new task", recovery["next_action"])
        self.assertEqual(summary["next_action"], recovery["next_action"])
        self.assertTrue(summary["answer"], "the partial answer is kept as context")
        # No extra calls or retries: one plan for the one attempt, and the one answer call the legacy path makes.
        self.assertEqual(len(_replan_asks(fallback)), 1)
        self.assertEqual(len([asked for asked in fallback.asked if "final page" in asked[0]["content"]]), 1)


class TestBrowserRecoveryBudget(IsolatedAsyncioTestCase):
    """Attempts and active seconds are global to the run; a failure is a clean BLOCKED, never a retry loop."""

    async def test_global_attempts_do_not_reset_across_stalls(self) -> None:
        fallback = _planner_fallback()
        slot = _slot(
            [_type() for _ in range(6)],
            runtime=_RecoveryProbeRuntime(),
            limits=RecoveryLimits(max_attempts=1, timeout_s=15.0),
            fallback=fallback,
        )

        for _ in range(6):
            self.assertEqual((await slot.invoke(_MESSAGES, tools=_TOOLS)).tool_calls[0].name, "browser_type")
        terminal = await slot.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads(terminal.content)

        self.assertEqual(summary["status"], "BLOCKED")
        self.assertTrue(summary["recovery"]["failed"])
        self.assertEqual((summary["recovery"]["attempts"], summary["recovery"]["termination"]), (1, "give_up"))
        self.assertEqual(summary["answer"], "")
        self.assertFalse(any("final page" in asked[0]["content"] for asked in fallback.asked))

        run = slot._run
        again = await slot.invoke(_MESSAGES, tools=_TOOLS)
        self.assertIs(slot._run, run, "a recovery-terminated task never starts a fresh run on a re-call")
        self.assertEqual(again.content, terminal.content)

    async def test_a_refresh_timeout_is_a_clean_terminal(self) -> None:
        fallback = _planner_fallback()
        slot = _slot(
            [_type() for _ in range(3)],
            runtime=_RecoveryProbeRuntime(hang=True),
            limits=RecoveryLimits(max_attempts=2, timeout_s=0.05),
            fallback=fallback,
        )

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads((await slot.invoke(_MESSAGES, tools=_TOOLS)).content)

        recovery = summary["recovery"]
        self.assertEqual((summary["status"], recovery["failed"]), ("BLOCKED", True))
        self.assertEqual((recovery["termination"], recovery["stage"]), ("timeout", "refresh"))
        self.assertGreater(recovery["spent_s"], 0)
        self.assertFalse(any("final page" in asked[0]["content"] for asked in fallback.asked))

    async def test_a_planner_timeout_is_a_clean_terminal(self) -> None:
        gate = asyncio.Event()
        fallback = _planner_fallback(hang=gate)
        slot = _slot(
            [_type() for _ in range(3)],
            runtime=_RecoveryProbeRuntime(),
            limits=RecoveryLimits(max_attempts=2, timeout_s=0.05),
            fallback=fallback,
        )

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads((await slot.invoke(_MESSAGES, tools=_TOOLS)).content)

        recovery = summary["recovery"]
        self.assertEqual((summary["status"], recovery["failed"]), ("BLOCKED", True))
        self.assertEqual((recovery["termination"], recovery["stage"]), ("timeout", "planner"))

    async def test_a_planner_error_is_a_clean_terminal(self) -> None:
        fallback = _planner_fallback(error=RuntimeError("planner down"))
        slot = _slot([_type() for _ in range(3)], runtime=_RecoveryProbeRuntime(), fallback=fallback)

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads((await slot.invoke(_MESSAGES, tools=_TOOLS)).content)

        recovery = summary["recovery"]
        self.assertEqual((summary["status"], recovery["failed"]), ("BLOCKED", True))
        self.assertEqual((recovery["termination"], recovery["stage"]), ("error", "planner"))
        self.assertIn("planner down", recovery["error"])

    async def test_an_empty_or_whitespace_plan_is_a_planner_failure(self) -> None:
        for plan in ("", "   ", "\n\t ", None):
            with self.subTest(plan=plan):
                fallback = _planner_fallback(plan=plan)
                slot = _slot([_type() for _ in range(3)], runtime=_RecoveryProbeRuntime(), fallback=fallback)
                for _ in range(3):
                    self.assertEqual((await slot.invoke(_MESSAGES, tools=_TOOLS)).tool_calls[0].name, "browser_type")
                terminal = await slot.invoke(_MESSAGES, tools=_TOOLS)
                summary = json.loads(terminal.content)

                recovery = summary["recovery"]
                self.assertEqual((summary["status"], recovery["failed"]), ("BLOCKED", True))
                self.assertEqual((recovery["termination"], recovery["stage"]), ("error", "planner"))
                self.assertEqual(recovery["attempts"], 1, "the charged attempt is kept")
                self.assertIn("empty plan", recovery["error"])
                self.assertIn("start a new task", summary["next_action"])
                event = slot.report()["recovery"]["events"][0]
                self.assertEqual((event["termination"], event["stage"]), ("error", "planner"))
                self.assertEqual(event["attempt"], 1)
                self.assertGreaterEqual(event["spent_s"], 0.0, "the consumed active time is kept")
                self.assertIsNotNone(event["fresh_obs"], "the successful refresh is kept")
                self.assertEqual(len(_replan_asks(fallback)), 1, "one plan attempt, never a silent retry")
                self.assertFalse(any("final page" in asked[0]["content"] for asked in fallback.asked))
                decisions = len(_wire(slot).bodies)
                again = await slot.invoke(_MESSAGES, tools=_TOOLS)
                self.assertEqual(again.content, terminal.content)
                self.assertEqual(len(_wire(slot).bodies), decisions, "a stopped task starts no new decision")

    async def test_a_probe_error_stops_without_retrying(self) -> None:
        slot = _slot(
            [_type() for _ in range(3)],
            runtime=_RecoveryProbeRuntime(fail=True),
            limits=RecoveryLimits(max_attempts=2, timeout_s=15.0),
        )

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads((await slot.invoke(_MESSAGES, tools=_TOOLS)).content)

        recovery = summary["recovery"]
        self.assertEqual((summary["status"], recovery["failed"]), ("BLOCKED", True))
        self.assertEqual((recovery["attempts"], recovery["termination"]), (1, "error"))
        self.assertIn("probe failed", recovery["error"])

    async def test_a_spent_budget_escalates_with_a_specific_reason_and_next_action(self) -> None:
        fallback = _planner_fallback()
        slot = _slot(
            [_type() for _ in range(6)],
            runtime=_RecoveryProbeRuntime(),
            limits=RecoveryLimits(max_attempts=1, timeout_s=15.0),
            fallback=fallback,
        )

        for _ in range(6):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads((await slot.invoke(_MESSAGES, tools=_TOOLS)).content)

        recovery = summary["recovery"]
        self.assertEqual((summary["status"], recovery["failed"]), ("BLOCKED", True))
        self.assertIn("recovery attempts spent", recovery["reason"])
        self.assertIn("start a new task", recovery["next_action"])
        self.assertEqual(summary["next_action"], recovery["next_action"])
        event = slot.report()["recovery"]["events"][-1]
        self.assertEqual(event["termination"], "give_up")
        self.assertIn("start a new task", event["next_action"])
        # The escalation is a message, not a fresh call: one plan for the one attempt and no analysis call after.
        self.assertEqual(len(_replan_asks(fallback)), 1)

    async def test_a_refresh_failure_escalation_names_the_reason_and_a_next_step(self) -> None:
        slot = _slot(
            [_type() for _ in range(3)],
            runtime=_RecoveryProbeRuntime(fail=True),
            limits=RecoveryLimits(max_attempts=2, timeout_s=15.0),
        )

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads((await slot.invoke(_MESSAGES, tools=_TOOLS)).content)

        recovery = summary["recovery"]
        self.assertEqual(recovery["termination"], "error")
        self.assertIn("probe failed", recovery["reason"])
        self.assertIn("start a new task", recovery["next_action"])
        self.assertNotIn("unsafe_dev", recovery["next_action"])

    async def test_a_permission_error_escalates_through_the_permission_flow(self) -> None:
        slot = _slot(
            [_type() for _ in range(3)],
            runtime=_RecoveryProbeRuntime(),
            fallback=_planner_fallback(error=RuntimeError("permission denied: Accessibility")),
        )

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads((await slot.invoke(_MESSAGES, tools=_TOOLS)).content)

        recovery = summary["recovery"]
        self.assertEqual(recovery["termination"], "error")
        self.assertIn("permission flow", recovery["next_action"])
        self.assertIn("leave the task blocked", recovery["next_action"])
        self.assertNotIn("unsafe_dev", recovery["next_action"])

    async def test_a_cancelled_recovery_keeps_its_event(self) -> None:
        gate = asyncio.Event()
        fallback = _planner_fallback(hang=gate)
        slot = _slot([_type() for _ in range(4)], runtime=_RecoveryProbeRuntime(), fallback=fallback)

        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        task = asyncio.create_task(slot.invoke(_MESSAGES, tools=_TOOLS))
        await asyncio.sleep(0.05)  # the refresh has finished; the planner is blocked
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        event = slot._run.recovery_events[-1]
        self.assertEqual(event["termination"], "cancelled")
        self.assertIsNotNone(event["fresh_obs"], "the successful refresh is kept even when the plan is cancelled")
        self.assertTrue(slot._run.recovery_stopped)


class TestBrowserRecoveryConfigAndTerminal(TestCase):
    """The flags parse into the budget; a recovery failure never turns a carried answer into a success."""

    def test_rethink_flags_default_off_and_carry_the_budget(self) -> None:
        parser = browse.parser(_spec())
        args = parser.parse_args(["--model", "laya", "--goal", "g"])
        self.assertEqual(args.rethink, "off")
        off = browse.policy_from_args(args)
        self.assertFalse(off.rethink_on)
        self.assertEqual(off.recovery_limits, RecoveryLimits(3, 15.0))

        on = parser.parse_args(
            ["--model", "laya", "--goal", "g", "--rethink", "on", "--rethink-attempts", "4", "--rethink-timeout", "2.5"]
        )
        policy = browse.policy_from_args(on)
        self.assertTrue(policy.rethink_on)
        self.assertEqual(policy.recovery_limits, RecoveryLimits(4, 2.5))

    def test_a_non_finite_rethink_timeout_is_rejected(self) -> None:
        parser = browse.parser(_spec())
        finite = parser.parse_args(["--model", "laya", "--goal", "g", "--rethink", "on", "--rethink-timeout", "inf"])
        self.assertEqual(finite.rethink_timeout, float("inf"))  # argparse alone would accept inf
        with self.assertRaises(ValueError):
            browse.policy_from_args(finite)  # the recovery budget is where inf and NaN are refused
        for text in ("-inf", "nan", "0"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--model", "laya", "--goal", "g", "--rethink", "on", "--rethink-timeout", text])

    def test_recovery_failure_is_a_failure_even_with_an_answer(self) -> None:
        summary = json.dumps(
            {
                "status": "BLOCKED",
                "reason": "no more attempts",
                "answer": "the page already held an answer",
                "recovery": {"failed": True, "termination": "give_up", "error": "recovery attempts spent: 3/3"},
            }
        )
        answer = browse.finish_decision_model(
            {"ok": False, "final": summary, "screenshot": None, "error": None}, model_name="jev"
        )
        self.assertFalse(answer["ok"])
        self.assertEqual(answer["final"], "the page already held an answer")
        self.assertIn("recovery give_up", answer["error"])

    def test_blocked_after_a_replan_is_a_failure_even_with_a_partial_answer(self) -> None:
        summary = json.dumps(
            {
                "status": "BLOCKED",
                "reason": "model reported BLOCKED after 2 recovery attempt(s)",
                "answer": "the page already held a partial answer",
                "recovery": {
                    "attempts": 2,
                    "failed": False,
                    "exhausted": False,
                    "blocked_after_recovery": True,
                    "termination": "planned",
                    "reason": "model reported BLOCKED after 2 recovery attempt(s)",
                    "next_action": "Review the fresh observation and the action that triggered recovery (blocked); "
                    "clear the blocker, then explicitly start a new task.",
                },
            }
        )
        answer = browse.finish_decision_model(
            {"ok": False, "final": summary, "screenshot": None, "error": None}, model_name="laya"
        )
        self.assertFalse(answer["ok"], "a BLOCKED run after a replan never succeeds through its partial answer")
        self.assertEqual(
            answer["final"], "the page already held a partial answer", "the partial answer stays as context"
        )
        self.assertIn("BLOCKED after recovery", answer["error"])
        self.assertIn("start a new task", answer["next_action"])
        self.assertNotIn("give_up", answer["error"])
        self.assertNotIn("exhausted", answer["error"])

    def test_a_failed_recovery_escalation_is_actionable(self) -> None:
        summary = json.dumps(
            {
                "status": "BLOCKED",
                "reason": "recovery planner timed out",
                "answer": "",
                "recovery": {
                    "failed": True,
                    "termination": "timeout",
                    "stage": "planner",
                    "error": "recovery planner timed out",
                    "next_action": "clear the blocker or restore the service, then start a new task",
                },
            }
        )
        answer = browse.finish_decision_model(
            {"ok": False, "final": summary, "screenshot": None, "error": None}, model_name="jev"
        )
        self.assertFalse(answer["ok"])
        self.assertIn("start a new task", answer["next_action"])
        self.assertNotIn("unsafe_dev", answer["next_action"])

    def test_recovery_off_keeps_the_blocked_answer_as_a_success(self) -> None:
        summary = json.dumps({"status": "BLOCKED", "reason": "oscillating", "answer": "Rated 4.6 by 243."})
        answer = browse.finish_decision_model(
            {"ok": False, "final": summary, "screenshot": None, "error": None}, model_name="jev"
        )
        self.assertEqual((answer["ok"], answer["final"]), (True, "Rated 4.6 by 243."))


class TestBrowseRejectsPlainLlmWithRethink(IsolatedAsyncioTestCase):
    async def test_plain_llm_with_rethink_on_is_rejected_before_the_browser(self) -> None:
        policy = BrowserPolicy(prefetch_values=True, batch_actions=False, goal_value_cache=False, rethink_on=True)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as caught:
                await browse.browse(
                    _spec(),
                    policy,
                    model_name="llm",
                    goal="g",
                    timeout_s=1,
                    max_steps=1,
                    logs_dir=Path(tmp),
                    headless=True,
                    chat=mock.MagicMock(),
                    decision_model=None,
                )
        self.assertIn("rethink", str(caught.exception))


class TestRecoveryLifecycle(IsolatedAsyncioTestCase):
    async def test_browser_planner_receives_the_original_goal(self) -> None:
        fallback = _fallback()
        slot = _slot([_type() for _ in range(4)], runtime=_RecoveryProbeRuntime(), fallback=fallback)
        for _ in range(4):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        context = json.loads(_replan_asks(fallback)[0][1]["content"])
        self.assertEqual(context["state"]["goal"], _MESSAGES[0]["content"])

    async def test_large_observations_stay_in_report_not_terminal_message(self) -> None:
        runtime = _RecoveryProbeRuntime(overrides={"diagnostic_blob": "x" * 20000})
        slot = _slot([_type() for _ in range(6)], runtime=runtime, limits=RecoveryLimits(max_attempts=1))
        for _ in range(7):
            terminal = await slot.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads(terminal.content)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertTrue(summary["recovery"]["failed"])
        self.assertLess(len(terminal.content), 8000)
        event = slot.report()["recovery"]["events"][0]
        self.assertEqual(len(event["fresh_obs"]["diagnostic_blob"]), 20000)

    async def test_cancelled_recovery_cannot_resume_actions_for_the_same_goal(self) -> None:
        entered = asyncio.Event()

        fallback = _planner_fallback()
        original = fallback.invoke

        async def planner(messages, **kwargs):
            if messages[0].get("content") == REPLAN_PROMPT:
                entered.set()
                await asyncio.Event().wait()
            return await original(messages, **kwargs)

        fallback.invoke = planner
        runtime = _RecoveryProbeRuntime()
        slot = _slot([_type() for _ in range(4)], runtime=runtime, fallback=fallback)
        for _ in range(3):
            await slot.invoke(_MESSAGES, tools=_TOOLS)
        task = asyncio.create_task(slot.invoke(_MESSAGES, tools=_TOOLS))
        try:
            await asyncio.wait_for(entered.wait(), 1)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        probes = len(runtime.calls)
        decisions = len(_wire(slot).bodies)
        stopped = await asyncio.wait_for(slot.invoke(_MESSAGES, tools=_TOOLS), 1)
        self.assertFalse(stopped.tool_calls)
        self.assertEqual(json.loads(stopped.content)["status"], "BLOCKED")
        self.assertEqual(len(runtime.calls), probes)
        self.assertEqual(len(_wire(slot).bodies), decisions)

    async def test_repeated_waits_consume_the_global_attempt_budget(self) -> None:
        fallback = _fallback()
        slot = _slot(
            [_answers("WAIT", "none"), _answers("WAIT", "none")],
            runtime=_RecoveryProbeRuntime(),
            fallback=fallback,
            limits=RecoveryLimits(max_attempts=1),
        )
        stopped = await asyncio.wait_for(slot.invoke(_MESSAGES, tools=_TOOLS), 1)
        summary = json.loads(stopped.content)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertEqual(summary["recovery"]["attempts"], 1)
        self.assertEqual(summary["recovery"]["termination"], "give_up")
        self.assertEqual(len(_replan_asks(fallback)), 1)
