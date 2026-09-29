# coding: utf-8
"""The desktop agent offline: the plan baseline plays 1 2 × 7 = to 84 in the fake Calculator, a dry run plans one click."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from s1a.agents import desktop
from s1a.desktop.driver import Element, Snapshot, Window
from s1a.run import started_runner
from s1a.tool import loop, series
from support_desktop import FakeCalculator, FakeWindowsCalculator
from test_desktop_driver import _process, _result

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

    def test_without_a_plan_there_is_no_rule_baseline(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            ["--model", "rule", "--rethink", "off", "--episodes", "1", *CALCULATOR]
        )
        with patch.object(desktop, "driver_from_env", lambda label: FakeCalculator()):
            self.assertIsNone(desktop.make_series(args).baseline)


async def _noop(app: str, driver: object, window_title: str = "") -> None:
    return None


class _Exited:
    def __init__(self, code: int) -> None:
        self.returncode = code

    async def wait(self) -> int:
        return self.returncode


class TestLaunch(IsolatedAsyncioTestCase):
    async def test_windows_title_selects_at_launch_and_keeps_the_window_binding(self) -> None:
        windows = [
            {"pid": 41, "window_id": 9, "title": "Other"},
            {"pid": 42, "window_id": 7, "title": "Draft", "app_name": "ApplicationFrameHost.exe"},
            {"pid": 42, "window_id": 8, "title": "Draft", "is_on_screen": False},
        ]
        listed = [*windows, {"pid": 50, "window_id": 7, "title": "Draft", "app_name": "Editor"}]
        args = series.parser(desktop.SPEC).parse_args(
            ["--model", "rule", "--rethink", "off", "--episodes", "1"]
            + ["--app", "Editor", "--window-title", "Draft", "--goal", "Save", "--expect", "Saved"]
        )
        with (
            patch("sys.platform", "win32"),
            _process(
                _result({}),
                _result({"apps": []}),
                _result({"pid": 100, "windows": windows}),
                _result({"windows": listed}),
                _result({"elements": []}),
                _result({}),
            ) as (driver, session, _),
            patch.object(desktop, "driver_from_env", return_value=driver),
        ):
            task = desktop.make_series(args)
            async with task.session:
                env = task.env_for(0)
                await env.reset()
                self.assertEqual((await env.observe())["title"], "Draft")
        name, target = session.calls[4]
        self.assertEqual((name, target["pid"], target["window_id"]), ("get_window_state", 42, 7))

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
            await desktop.launch_app("Calculator", FakeCalculator(), "Paper Tape")
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
