# coding: utf-8
"""The desktop agent offline: the plan baseline plays 1 2 × 7 = to 84 in the fake Calculator, a dry run plans one click."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk, Model
from openjiuwen.core.runner import Runner

from s1a.agents import desktop
from s1a.decision_models import RuleModel
from s1a.desktop.driver import DriverError, Element, Snapshot, Window
from s1a.desktop.env import WindowEnv
from s1a.recovery import RecoveryLimits
from s1a.run import started_runner
from s1a.tool import loop, series
from s1a.tool.models import placeholder_model
from support_desktop import FakeCalculator, FakeWindowsCalculator

CALCULATOR = ["--app", "Calculator", "--goal", "compute 12 times 7", "--expect", "84"]
PLAN = ["--plan", "1,2,Multiply|×,7,Equals|=", "--clear", "All Clear"]


class TestPieces(TestCase):
    def test_windows_result_matches_the_full_label_without_app_specific_parsing(self) -> None:
        window = Window(42, 7, "ApplicationFrameHost.exe", "Calculator")
        display = Element(8, "Text", "Display is 84", "", "s1:8", ("invoke",))
        snapshot = Snapshot(window, "s1", (display,), {})
        self.assertTrue(desktop.shows(snapshot, "Display is 84"))
        self.assertFalse(desktop.shows(snapshot, "84"))
        self.assertFalse(desktop.shows(snapshot, "8"))
        self.assertFalse(desktop.shows(snapshot, ""))
        for element in (
            Element(1, "Button", "Display is 84", "", "s1:1", ("invoke",)),
            Element(2, "Text", "Expression is 84", "", "s1:2", ("text",)),
        ):
            self.assertFalse(desktop.shows(Snapshot(window, "s1", (element,), {}), "Display is 84"))

    def test_shows_reads_displays_and_labels_of_non_clickable_elements_only(self) -> None:
        window = Window(1, 1, "Calculator", "Calculator")
        elements = (
            Element(1, "AXButton", "84", "", "t", ("AXPress",)),
            Element(0, "AXStaticText", "", " 84 ", None, ()),
        )
        self.assertTrue(desktop.shows(Snapshot(window, None, elements, {}), "84"))
        self.assertFalse(desktop.shows(Snapshot(window, None, elements[:1], {}), "84"))
        self.assertTrue(
            desktop.shows(Snapshot(window, None, (Element(2, "AXStaticText", "PASS", "", None, ()),), {}), "PASS")
        )

    def test_the_plan_rule_follows_the_labels_with_variants_then_says_done_or_abstains(self) -> None:
        rule = desktop.plan_rule(desktop.parse_plan("1, 2,Multiply|×,7,Equals|="))
        offered = {f"click:{label}": "" for label in ("1", "2", "×", "7", "=")}
        self.assertEqual(
            [rule({"presses": ["x"] * n}, offered) for n in range(6)],
            ["click:1", "click:2", "click:×", "click:7", "click:=", "done"],
        )
        self.assertEqual(rule({"presses": []}, {"click:9": ""}), "abstain")

    def test_the_flags_require_app_goal_and_expect_and_default_to_a_dry_run(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            ["--model", "jev", "--rethink", "off", "--episodes", "1", *CALCULATOR]
        )
        self.assertEqual((args.execute, args.plan, args.clear), (False, "", ""))
        with self.assertRaises(SystemExit):
            series.parser(desktop.SPEC).parse_args(["--model", "jev", "--rethink", "off", "--episodes", "1"])

    def test_the_stall_default_and_the_bounded_rethink_flags(self) -> None:
        self.assertEqual(desktop.SPEC.budget.stall_after, 3)
        args = series.parser(desktop.SPEC).parse_args(
            ["--model", "jev", "--rethink", "on", "--episodes", "1", *CALCULATOR]
        )
        self.assertEqual((args.rethink_attempts, args.rethink_timeout), (3, 15.0))
        tuned = series.parser(desktop.SPEC).parse_args(
            [
                "--model",
                "jev",
                "--rethink",
                "on",
                "--episodes",
                "1",
                "--rethink-attempts",
                "5",
                "--rethink-timeout",
                "2.5",
                *CALCULATOR,
            ]
        )
        self.assertEqual((tuned.rethink_attempts, tuned.rethink_timeout), (5, 2.5))

    def test_without_a_plan_there_is_no_rule_baseline(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            ["--model", "rule", "--rethink", "off", "--episodes", "1", *CALCULATOR]
        )
        with patch.object(desktop, "driver_from_env", lambda label: FakeCalculator()):
            self.assertIsNone(desktop.make_series(args).baseline)


async def _noop(app: str, driver: object) -> None:
    return None


class _Exited:
    def __init__(self, code: int) -> None:
        self.returncode = code

    async def wait(self) -> int:
        return self.returncode


class TestLaunch(IsolatedAsyncioTestCase):
    async def test_windows_launches_through_driver_without_running_macos_open(self) -> None:
        fake = FakeCalculator()
        with (
            patch("sys.platform", "win32"),
            patch.object(desktop.asyncio, "create_subprocess_exec", AsyncMock()) as spawn,
        ):
            await desktop.launch_app("Windows Calculator", fake)
        self.assertEqual(fake.launched_apps, ["Windows Calculator"])
        spawn.assert_not_called()

    async def test_macos_retains_open_a(self) -> None:
        process = AsyncMock()
        process.wait.return_value = 0
        with (
            patch("sys.platform", "darwin"),
            patch.object(desktop.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)) as spawn,
            patch.object(desktop.asyncio, "sleep", AsyncMock()),
        ):
            await desktop.launch_app("Calculator", FakeCalculator())
        spawn.assert_awaited_once_with("open", "-a", "Calculator")

    async def test_a_failed_macos_open_raises_with_the_exit_code(self) -> None:
        argv: list[str] = []

        async def spawn(*args: str) -> _Exited:
            argv.extend(args)
            return _Exited(1)

        with patch("sys.platform", "darwin"), patch.object(desktop.asyncio, "create_subprocess_exec", spawn):
            with self.assertRaises(RuntimeError) as caught:
                await desktop.launch_app("Nope", FakeCalculator())
        self.assertEqual((argv, str(caught.exception)), (["open", "-a", "Nope"], "open -a 'Nope' exited 1"))


class TestThroughTheSeries(IsolatedAsyncioTestCase):
    async def _play(self, fake: FakeCalculator, *flags: str) -> tuple[dict, dict]:
        args = series.parser(desktop.SPEC).parse_args(
            ["--model", "rule", "--rethink", "off", "--episodes", "1", *CALCULATOR, *PLAN, *flags]
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(loop, "WORKSPACE", Path(tmp) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
            patch.object(desktop, "driver_from_env", lambda label: fake),
            patch.object(desktop, "launch_app", _noop),
        ):
            async with started_runner():
                result = await series.play(desktop.SPEC, args, results_dir=Path(tmp) / "results")
            (episode_file,) = Path(result["job_dir"]).glob("*/agent/episode.json")
            episode = json.loads(episode_file.read_text(encoding="utf-8"))
        return result, episode

    async def test_the_plan_presses_the_buttons_and_the_display_reads_84(self) -> None:
        fake = FakeCalculator()
        result, episode = await self._play(fake, "--execute")
        self.assertEqual((result["policy"], result["mean_score"], result["errors"]), ("plan", 1.0, 0))
        self.assertEqual(
            [d["key"] for d in episode["decisions"]],
            ["click:1", "click:2", "click:Multiply", "click:7", "click:Equals"],
        )
        self.assertEqual((fake.display, fake.opened), ("84", 0))  # the session opened and closed the driver
        self.assertEqual(episode["final_state"]["goal"], "compute 12 times 7")

    async def test_without_execute_one_click_is_planned_and_nothing_is_touched(self) -> None:
        fake = FakeCalculator()
        result, episode = await self._play(fake)
        self.assertEqual((result["mean_score"], len(episode["decisions"]), fake.clicks), (0.0, 1, []))
        self.assertEqual(episode["final_state"]["planned"]["label"], "1")

    async def test_windows_plan_finishes_when_uia_exposes_the_result_in_the_label(self) -> None:
        fake = FakeWindowsCalculator()
        result, episode = await self._play(
            fake,
            "--execute",
            "--expect",
            "Display is 84",
            "--plan",
            "One,Two,Multiply by,Seven,Equals",
            "--clear",
            "Clear",
        )
        self.assertEqual((result["mean_score"], result["errors"]), (1.0, 0))
        self.assertEqual(
            [d["key"] for d in episode["decisions"]],
            ["click:One", "click:Two", "click:Multiply by", "click:Seven", "click:Equals"],
        )
        self.assertEqual(fake.display, "84")


class TestBoundedRethinkRejection(IsolatedAsyncioTestCase):
    async def test_plain_llm_with_rethink_on_is_rejected_before_the_driver(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            ["--model", "llm", "--rethink", "on", "--episodes", "1", *CALCULATOR]
        )
        touched: list[str] = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(series, "optional_chat_model", lambda: object()),
            patch.object(desktop, "driver_from_env", lambda label: touched.append(label)),
        ):
            with self.assertRaises(RuntimeError) as caught:
                await series.play(desktop.SPEC, args, results_dir=Path(tmp))
        self.assertIn("decision model", str(caught.exception))
        self.assertEqual(touched, [])  # rejected before the driver or an env was built


class PlanChat(Model):
    """A chat model that answers the replan request with one useful line and counts its calls."""

    def __init__(self) -> None:
        source = placeholder_model()
        super().__init__(source.model_client_config, source.model_config)
        self.calls = 0

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        self.calls += 1
        return AssistantMessage(content="Click 1, 2, Multiply, 7, Equals.")

    async def stream(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AsyncIterator[AssistantMessageChunk]:
        self.calls += 1
        yield AssistantMessageChunk(content="Click 1, 2, Multiply, 7, Equals.")


def _window_env(fake: FakeCalculator, *, clear_labels: tuple[str, ...] = ("All Clear",)) -> WindowEnv:
    return WindowEnv(
        fake,
        app_name="Calculator",
        goal="compute 12 times 7",
        done_when=lambda snapshot: snapshot.elements[0].value == "84",
        execute=True,
        clear_labels=clear_labels,
    )


async def _bounded_play(
    env: WindowEnv, decision_model: RuleModel, chat: Model, *, max_acts: int, max_attempts: int = 3
) -> Any:
    with tempfile.TemporaryDirectory() as tmp, patch.object(loop, "WORKSPACE", Path(tmp)):
        await Runner.start()
        try:
            return await loop.run_episode(
                desktop.SPEC,
                env,
                model_name="rule",
                seed=0,
                chat=chat,
                decision_model=decision_model,
                rethink_on=True,
                max_acts=max_acts,
                timeout_s=60.0,
                prices=None,
                log=False,
                limits=RecoveryLimits(max_attempts=max_attempts, timeout_s=5.0),
            )
        finally:
            await Runner.stop()


CALC_SOLUTION = ["click:1", "click:2", "click:Multiply", "click:7", "click:Equals"]


def _solution_rule(solution: list[str]) -> RuleModel:
    """A rule that no-ops until it sees a plan, then plays the solution to 84 and finishes."""
    cursor = {"planned": False, "i": 0}

    def rule(state: dict[str, Any], offered: dict[str, str]) -> str:
        if state.get("plan"):
            cursor["planned"] = True
        if not cursor["planned"]:
            return "click:Multiply"  # a real no-op on this window: the display never moves
        if cursor["i"] >= len(solution):
            return "done"
        key = solution[cursor["i"]]
        cursor["i"] += 1
        return key

    return RuleModel("solution", rule)


class _DeniedCalculator(FakeCalculator):
    """A driver that refuses every click, as when Accessibility permission was never granted."""

    async def click(self, window: Window, token: str) -> dict[str, Any]:
        raise DriverError("click: refused (accessibility permission denied)")


class TestBoundedRethinkChain(IsolatedAsyncioTestCase):
    """WindowEnv, the real ToolDecisionModel and ActTool, and the bounded rail, with no real GUI."""

    async def test_a_no_op_stall_plans_and_the_next_click_uses_the_refreshed_snapshot(self) -> None:
        fake = FakeCalculator()
        env = _window_env(fake)
        chat = PlanChat()
        episode = await _bounded_play(env, _solution_rule(CALC_SOLUTION), chat, max_acts=12)
        self.assertEqual((episode.score, fake.display), (1.0, "84"))
        self.assertEqual([decision["key"] for decision in episode.decisions], ["click:Multiply"] * 3 + CALC_SOLUTION)
        self.assertEqual(fake.clicks[4][2], "tok-1-6")  # the planned click names the refreshed snapshot's token
        events = [event for event in episode.extra["rethinks"] if event["kind"] == "stall"]
        self.assertEqual([event["termination"] for event in events], ["planned"])
        self.assertEqual(events[0]["trigger_reason"], "stalled")
        self.assertIn("click:1", events[0]["fresh_obs"]["candidates"])
        self.assertEqual(chat.calls, 1)

    async def test_the_only_plan_of_a_one_attempt_budget_drives_a_normal_act_success(self) -> None:
        fake = FakeCalculator()
        env = _window_env(fake)
        chat = PlanChat()
        episode = await _bounded_play(env, _solution_rule(CALC_SOLUTION), chat, max_acts=12, max_attempts=1)
        self.assertEqual((episode.score, fake.display), (1.0, "84"))
        stalls = [event for event in episode.extra["rethinks"] if event["kind"] == "stall"]
        self.assertEqual((len(stalls), stalls[0]["termination"], stalls[0]["attempt"]), (1, "planned", 1))
        self.assertNotIn("give_up", [event.get("termination") for event in episode.extra["rethinks"]])
        self.assertIsNone(episode.error)

    async def test_a_permission_denied_click_ends_the_episode_without_a_planner_retry(self) -> None:
        fake = _DeniedCalculator()
        env = _window_env(fake, clear_labels=())
        chat = PlanChat()
        episode = await _bounded_play(env, RuleModel("first", lambda state, offered: "click:1"), chat, max_acts=12)
        self.assertEqual(episode.error, "act failed: click: refused (accessibility permission denied)")
        self.assertEqual((chat.calls, fake.clicks, fake.snapshots), (0, [], 1))  # no refresh, no plan, no retry
        self.assertEqual(episode.steps, 0)
        self.assertEqual(episode.extra["rethinks"], [])
