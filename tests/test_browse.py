# coding: utf-8
"""The browser front: the answer shape, the assembly of the subagent per model."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from openjiuwen.harness.subagents.browser_agent import (
    DEFAULT_BROWSER_AGENT_TEMPERATURE,
    _browser_model_with_temperature,
)

from s1a.agents import flights
from s1a.browser import browse
from s1a.browser.decision_model import BrowserDecisionModel, BrowserPolicy
from s1a.decision_models import JevModel
from s1a.config import chat_model_from_env
from s1a.counting_model import CountingModel
from support import CHAT_ENV as ENV
from support import NoDecisionModel, browse_offline, browser_result

SPEC = flights.SPEC
GOAL = ["--model", "jev", "--goal", "x"]


class TestParser(TestCase):
    def test_the_spec_and_the_policy_defaults_and_their_overrides(self) -> None:
        args = browse.parser(SPEC).parse_args(GOAL)
        self.assertEqual((args.max_steps, args.timeout, args.profile_out), (100, 180.0, None))
        self.assertEqual((args.batch, args.prefetch, args.goal_values), ("off", "on", "off"))
        self.assertEqual(
            browse.policy_from_args(args),
            BrowserPolicy(prefetch_values=True, batch_actions=False, goal_value_cache=False),
        )
        args = browse.parser(SPEC).parse_args(
            [*GOAL, "--max-steps", "7", "--profile-out", "p.json", "--batch", "on", "--prefetch", "off"]
        )
        self.assertEqual((args.max_steps, args.profile_out), (7, Path("p.json")))
        self.assertEqual(
            browse.policy_from_args(args),
            BrowserPolicy(prefetch_values=False, batch_actions=True, goal_value_cache=False),
        )

    def test_a_timeout_or_max_steps_at_or_below_zero_is_a_usage_error(self) -> None:
        for flags in (["--timeout", "0"], ["--timeout", "-1"], ["--max-steps", "0"], ["--max-steps", "-3"]):
            with self.assertRaises(SystemExit) as caught:
                browse.parser(SPEC).parse_args([*GOAL, *flags])
            self.assertEqual(caught.exception.code, 2)

    def test_the_model_flag_takes_a_decision_model_or_the_chat_model(self) -> None:
        self.assertEqual(browse.BROWSER_MODEL_NAMES, ("jev", "laya", "cua", "llm"))
        for model_name in browse.BROWSER_MODEL_NAMES:
            self.assertEqual(browse.parser(SPEC).parse_args(["--model", model_name, "--goal", "x"]).model, model_name)
        with self.assertRaises(SystemExit):
            browse.parser(SPEC).parse_args(["--model", "random", "--goal", "x"])


class TestSolverContract(TestCase):
    def test_terminal_summary_only_for_the_policy_json(self) -> None:
        self.assertEqual(browse.terminal_summary(json.dumps({"status": "DONE", "reason": "x"}))["status"], "DONE")
        self.assertIsNone(browse.terminal_summary("The cheapest flight is 120 CHF."))
        self.assertIsNone(browse.terminal_summary(json.dumps({"answer": 1})))

    def test_terminal_summary_survives_text_the_rail_appends(self) -> None:
        final = (
            json.dumps({"status": "DONE", "reason": "", "answer": "42"})
            + "\n\nThe runtime could not verify completion."
        )
        self.assertEqual(browse.terminal_summary(final)["answer"], "42")
        self.assertEqual(browse.finish_decision_model(_answer(final), model_name="jev")["final"], "42")

    def test_terminal_summary_reads_the_summary_inside_browser_result(self) -> None:
        final = _completed(json.dumps({"status": "BLOCKED", "reason": "", "page_text": "Flights"}))
        self.assertEqual(browse.terminal_summary(final)["status"], "BLOCKED")
        self.assertEqual(browse.browser_result(final)["status"], "completed")
        self.assertIsNone(browse.browser_result("plain text"))

    def test_finish_llm_takes_the_summary_as_the_answer(self) -> None:
        answer = browse.finish_llm(_answer(_completed("The flight costs 120 CHF.")))
        self.assertEqual((answer["ok"], answer["final"], answer["error"]), (True, "The flight costs 120 CHF.", None))
        self.assertEqual(
            answer["browser_result"], {"status": "completed", "terminal_reason": "runtime_completion_validated"}
        )
        blocked = browse.finish_llm(_answer(browser_result("stuck", status="blocked")))
        self.assertFalse(blocked["ok"])
        self.assertIn("blocked", blocked["error"])
        plain = browse.finish_llm(_answer("Not JSON at all"))
        self.assertEqual(plain["final"], "Not JSON at all")

    def test_a_failed_chat_call_makes_the_usage_unknown_not_zero(self) -> None:
        calls = [
            {"status": "ok", "usage_known": True, "input_tokens": 100, "output_tokens": 5, "tool_calls": []},
            {"status": "error", "usage_known": False, "input_tokens": 0, "output_tokens": 0, "tool_calls": []},
        ]
        summary = browse.usage_summary(calls, jev_input_tokens=0, decisions=0)
        self.assertFalse(summary["usage_known"])
        self.assertEqual(summary["unknown_calls"], 1)
        self.assertIsNone(summary["cost_usd"], "an unknown call makes the task's cost unknown, never zero")

    def test_every_call_reported_usage_keeps_the_cost_known(self) -> None:
        calls = [
            {"status": "ok", "usage_known": True, "input_tokens": 100, "output_tokens": 5, "tool_calls": []},
            {"status": "ok", "usage_known": True, "input_tokens": 20, "output_tokens": 1, "tool_calls": []},
        ]
        prices = {"CHAT_USD_PER_M_INPUT": "1", "CHAT_USD_PER_M_OUTPUT": "2"}
        with patch.dict(os.environ, prices):
            summary = browse.usage_summary(calls, jev_input_tokens=0, decisions=0)
        self.assertTrue(summary["usage_known"])
        self.assertEqual(summary["unknown_calls"], 0)
        self.assertIsNotNone(summary["cost_usd"])


def _completed(summary: str) -> str:
    return browser_result(summary, status="completed")


def _answer(final: str) -> dict[str, Any]:
    return {"ok": True, "final": final, "screenshot": None, "error": None}


class TestFinishJev(TestCase):
    def test_blocked_is_a_failure_even_when_the_subagent_says_completed(self) -> None:
        final = _completed(json.dumps({"status": "BLOCKED", "reason": "", "page_text": "Flights", "answer": ""}))
        answer = browse.finish_decision_model(_answer(final), model_name="laya")
        self.assertEqual((answer["ok"], answer["final"], answer["status"]), (False, "", "BLOCKED"))
        self.assertEqual(answer["error"], "laya BLOCKED: no reason given")

    def test_done_takes_the_policy_answer_and_clears_the_harness_verdict(self) -> None:
        summary = {
            "status": "DONE",
            "reason": "",
            "page_text": "ORD-CDG 412 USD",
            "answer": "Three flights from 412 USD.",
        }
        flagged = {
            **_answer(_completed(json.dumps(summary))),
            "ok": False,
            "error": "result_type='error': partial",
        }
        answer = browse.finish_decision_model(flagged, model_name="jev")
        self.assertEqual((answer["ok"], answer["final"], answer["error"]), (True, "Three flights from 412 USD.", None))
        self.assertEqual(answer["terminal"], summary)

    def test_blocked_after_progress_keeps_the_pages_answer(self) -> None:
        summary = {"status": "BLOCKED", "reason": "oscillating between two pages", "answer": "Rated 4.6 by 243."}
        answer = browse.finish_decision_model(_answer(json.dumps(summary)), model_name="jev")
        self.assertEqual(
            (answer["ok"], answer["final"], answer["status"], answer["error"]),
            (True, "Rated 4.6 by 243.", "BLOCKED", None),
        )

    def test_done_without_an_answer_is_a_failure(self) -> None:
        final = _completed(json.dumps({"status": "DONE", "reason": "", "page_text": "x", "answer": ""}))
        answer = browse.finish_decision_model(_answer(final), model_name="cua")
        self.assertEqual((answer["ok"], answer["error"]), (False, "cua DONE without an answer"))


class TestBrowseAssembly(IsolatedAsyncioTestCase):
    """``browse`` builds the real slot model from the spec; the factory and the browser run are faked, no Runner runs."""

    async def _browse(self, model_name: str, *, batch: bool) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        policy = BrowserPolicy(prefetch_values=True, batch_actions=batch, goal_value_cache=False)
        return await browse_offline(SPEC, policy, model_name=model_name, max_steps=7)

    async def test_a_decision_model_puts_the_slot_model_in_the_slot_and_records_its_ticks(self) -> None:
        for model_name in ("jev", "laya"):
            with self.subTest(model_name=model_name):
                answer, seen, files = await self._browse(model_name, batch=False)
                self.assertIsInstance(seen["model"], BrowserDecisionModel)
                self.assertIsInstance(seen["model"]._decision_model, NoDecisionModel)
                self.assertEqual(
                    (seen["max_iterations"], seen["browser_capabilities"], seen["timeout_s"]), (7, None, 30)
                )
                self.assertIn("--headless", seen["browser_instance"].launch_args)
                self.assertEqual((answer["ok"], answer["final"], answer["status"]), (True, "Three flights.", "DONE"))
                self.assertEqual(answer["terminal"]["url"], "https://x")
                self.assertEqual((answer["report"]["decisions"], answer["ticks"]), (0, []))
                self.assertEqual(answer["usage"]["decisions"], 0)
                self.assertEqual(files, ["decision_ticks.json"])

    async def test_batched_actions_ask_for_the_unsafe_dev_capability(self) -> None:
        _answer, seen, _files = await self._browse("jev", batch=True)
        self.assertEqual(seen["browser_capabilities"], ["unsafe_dev"])

    async def test_llm_keeps_the_counting_model_in_the_slot(self) -> None:
        answer, seen, files = await self._browse("llm", batch=False)
        self.assertIsInstance(seen["model"], CountingModel)
        self.assertIs(_browser_model_with_temperature(seen["model"], DEFAULT_BROWSER_AGENT_TEMPERATURE), seen["model"])
        self.assertEqual((answer["ok"], answer["final"], answer["usage"]["chat_calls"]), (True, "42", 0))
        self.assertEqual(answer["usage"]["chat_temperature"], 0.0, "the sampling setting the baseline really ran at")
        self.assertEqual(files, ["chat_calls.json"])

    async def test_jev_and_laya_refuse_to_run_without_a_decision_model(self) -> None:
        policy = BrowserPolicy(prefetch_values=True, batch_actions=False, goal_value_cache=False)
        for model_name in ("jev", "laya"):
            with (
                self.subTest(model_name=model_name),
                patch.dict(os.environ, ENV, clear=True),
                tempfile.TemporaryDirectory() as tmp,
            ):
                with self.assertRaises(RuntimeError) as caught:
                    await browse.browse(
                        SPEC,
                        policy,
                        model_name=model_name,
                        goal="g",
                        timeout_s=1,
                        max_steps=1,
                        logs_dir=Path(tmp),
                        headless=True,
                        chat=chat_model_from_env(),
                        decision_model=None,
                    )
                self.assertIn(model_name, str(caught.exception))


class _FakeAgent:
    def __init__(self, events: list[Any]) -> None:
        self.events = events

    async def ensure_initialized(self) -> None:
        self.events.append("init")

    async def cleanup_task_resources(self) -> None:
        self.events.append("cleanup")


class TestRunTask(IsolatedAsyncioTestCase):
    """``run_task`` releases the browser and the Runner session after every run, finished or timed out."""

    def _runner(self, events: list[Any], *, hang: bool) -> type:
        class FakeRunner:
            @staticmethod
            async def run_agent(agent: Any, inputs: dict[str, Any]) -> dict[str, Any]:
                events.append(("run", inputs["conversation_id"]))
                if hang:
                    await asyncio.sleep(60)
                return {"output": "done", "result_type": "answer"}

            @staticmethod
            async def release(session_id: str) -> None:
                events.append(("release", session_id))

        return FakeRunner

    async def test_a_finished_run_releases_the_browser_and_the_session(self) -> None:
        events: list[Any] = []
        with patch.object(browse, "Runner", self._runner(events, hang=False)):
            answer = await browse.run_task(_FakeAgent(events), "g", timeout_s=1)
        self.assertEqual((answer["ok"], answer["final"]), (True, "done"))
        conversation = events[1][1]
        self.assertEqual(events, ["init", ("run", conversation), "cleanup", ("release", conversation)])

    async def test_a_timed_out_run_is_released_too(self) -> None:
        events: list[Any] = []
        with patch.object(browse, "Runner", self._runner(events, hang=True)):
            answer = await browse.run_task(_FakeAgent(events), "g", timeout_s=0.01)
        self.assertIn("timeout", answer["error"])
        self.assertEqual(events[2:], ["cleanup", ("release", events[1][1])])


class TestPlay(IsolatedAsyncioTestCase):
    """``play`` builds the decision model ``--model`` names, hands the flags to ``browse`` and returns the answer without the ticks;
    no profiler without a path."""

    async def test_the_answer_comes_back_without_ticks_and_without_a_profile(self) -> None:
        seen: dict[str, Any] = {}

        async def fake_browse(spec: Any, policy: Any, **kwargs: Any) -> dict[str, Any]:
            seen.update(spec=spec, policy=policy, **kwargs)
            return {"ok": True, "final": "42", "error": None, "ticks": [{"tick": 1}], "report": {}, "usage": {}}

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV, clear=True):
            args = browse.parser(SPEC).parse_args(["--model", "llm", "--goal", "g", "--logs-dir", tmp, "--batch", "on"])
            with patch.object(browse, "browse", fake_browse), patch.object(browse, "BrowserProfiler", None):
                answer = await browse.play(SPEC, args)
            written = json.loads((Path(tmp) / "answer.json").read_text(encoding="utf-8"))
        self.assertEqual(answer, {"ok": True, "final": "42", "error": None, "report": {}, "usage": {}})
        self.assertEqual(
            written, {**answer, "agent": "flights", "model": "llm"}, "the result lands next to the run's records"
        )
        self.assertEqual((seen["model_name"], seen["max_steps"], seen["policy"].batch_actions), ("llm", 100, True))
        self.assertEqual((seen["goal"], seen["timeout_s"], seen["headless"]), ("g", 180.0, True))
        self.assertIsNone(seen["decision_model"], "the chat model needs no model")

    async def test_a_decision_model_is_built_from_the_environment_and_closed(self) -> None:
        seen: dict[str, Any] = {}
        closed: list[str] = []

        async def fake_browse(spec: Any, policy: Any, **kwargs: Any) -> dict[str, Any]:
            seen.update(**kwargs)
            return {"ok": True, "final": "42", "error": None, "ticks": [], "report": {}, "usage": {}}

        async def close(self: Any) -> None:
            closed.append(self.name)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV, clear=True):
            args = browse.parser(SPEC).parse_args(["--model", "jev", "--goal", "g", "--logs-dir", tmp])
            with patch.object(browse, "browse", fake_browse), patch.object(JevModel, "close", close):
                await browse.play(SPEC, args)
        self.assertIsInstance(seen["decision_model"], JevModel)
        self.assertEqual(closed, ["jev"], "the model is closed once the task is over")

    async def test_the_profiler_needs_a_decision_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV, clear=True):
            args = browse.parser(SPEC).parse_args(["--model", "llm", "--goal", "g", "--profile-out", f"{tmp}/p.json"])
            with patch.object(browse, "browse", None), self.assertRaises(RuntimeError):
                await browse.play(SPEC, args)

    def test_the_runs_dir_lives_under_the_configured_home(self) -> None:
        from s1a import config

        self.assertEqual(browse.RUNS_DIR, config.HOME / "runs" / "browser")


class TestAllrecipesGoal(TestCase):
    def test_the_goal_is_the_webvoyager_task_on_its_site(self) -> None:
        from s1a.agents import allrecipes

        self.assertTrue(allrecipes.SPEC.goal.startswith(f"Go to {allrecipes.SITE} and complete this task:"))
        self.assertIn(allrecipes.TASK, allrecipes.SPEC.goal)
        for condition in ("more than 100 reviews", "at least 4.5 stars", "suitable for 6 people"):
            self.assertIn(condition, allrecipes.TASK)
        self.assertEqual(allrecipes.SPEC.language, "en")


class TestFlightsGoal(TestCase):
    def test_the_goal_date_is_a_sunday_at_least_four_weeks_out(self) -> None:
        day = datetime.strptime(flights.GOAL_DATE, "%B %d, %Y").date()
        self.assertEqual(day.weekday(), 6)
        self.assertGreaterEqual(day, date.today() + timedelta(days=28))
        self.assertLess(day, date.today() + timedelta(days=35))
        self.assertIn(flights.GOAL_DATE, flights.SPEC.goal)
        self.assertIn(flights.GOAL_DATE, flights.SPEC.description)
