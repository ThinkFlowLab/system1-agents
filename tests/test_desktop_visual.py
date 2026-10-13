"""Screenshot bytes reach the decision model; pixel input remains capture-bound."""

import base64
import io
import json
import os
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, call, patch

from mcp.types import CallToolResult, ImageContent
from PIL import Image as PILImage
from PIL import PngImagePlugin

from s1a.agents import desktop
from s1a.agents.desktop import parse_pixel_targets
from s1a.decision_models import Image, ScriptedModel
from s1a.desktop.driver import Capture, DriverError, Element, Snapshot, Window
from s1a.desktop.env import WindowEnv
from s1a.recovery import RecoveryLimits
from s1a.tool import series
from s1a.tool.loop import view_of
from s1a.tool.models import EvalState, ToolDecisionModel
from s1a.tool.rethink import RethinkRail
from test_desktop_driver import _driver, _result
from test_rethink_rail import FakePlanner

WINDOW = Window(42, 7, "Canvas", "Canvas task")


def _png(colour: tuple[int, int, int], *, stamp: str = "") -> bytes:
    """Valid PNG bytes for one colour; ``stamp`` only varies the file's metadata, never its pixels."""
    info = PngImagePlugin.PngInfo()
    if stamp:
        info.add_text("stamp", stamp)
    buffer = io.BytesIO()
    PILImage.new("RGB", (8, 8), colour).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


class PixelWindow:
    """A static window whose screenshot pixels the test sets; every read gets a fresh capture id and metadata."""

    def __init__(self, colour: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.colour = colour
        self.reads = 0
        self.pngs: list[bytes] = []

    async def find_window(self, app_name: str, window_title: str = "") -> Window:
        return WINDOW

    async def window_state(self, window: Window, *, screenshot: bool = False) -> Snapshot:
        self.reads += 1
        capture = None
        if screenshot:
            png = _png(self.colour, stamp=f"read-{self.reads}")
            self.pngs.append(png)
            capture = Capture(f"capture-{self.reads}", Image(png), 8, 8)
        elements = (Element(1, "AXStaticText", "", "0", None, ()),)
        return Snapshot(window, f"snap-{self.reads}", elements, {}, capture)

    async def click_at(self, window: Window, capture: Capture, x: float, y: float) -> dict[str, Any]:
        return {"effect": "confirmed"}


class TestVisualTargets(IsolatedAsyncioTestCase):
    async def test_disabled_result_is_readable_and_menus_are_not_click_candidates(self) -> None:
        driver, _ = _driver(
            _result(
                {
                    "elements": [
                        {
                            "element_index": 1,
                            "role": "AXButton",
                            "label": "Saved",
                            "enabled": False,
                            "element_token": "status",
                            "actions": ["AXPress"],
                        },
                        {
                            "element_index": 2,
                            "role": "AXMenuItem",
                            "label": "Recent document",
                            "element_token": "menu",
                            "actions": ["AXPress"],
                        },
                    ]
                }
            )
        )
        snapshot = await driver.window_state(WINDOW)
        self.assertTrue(desktop.shows(snapshot, "Saved"))
        fake = AsyncMock()
        fake.find_window.return_value = WINDOW
        fake.window_state.return_value = snapshot
        env = WindowEnv(fake, app_name="Canvas", goal="Save", done_when=lambda s: False, execute=False, clear_labels=())
        await env.reset()
        self.assertEqual(set(await env.candidates()), {"abstain"})
        self.assertEqual([e["label"] for e in (await env.observe())["elements"]], ["Saved"])

    async def test_window_title_selects_the_canvas_among_auxiliary_windows(self) -> None:
        windows = {
            "windows": [
                {"pid": 42, "window_id": 7, "app_name": "Canvas", "title": "Canvas task"},
                {"pid": 42, "window_id": 9, "app_name": "Canvas", "title": "Window"},
            ]
        }
        driver, _ = _driver(_result(windows), _result(windows))
        self.assertEqual(await driver.find_window("Canvas", "Canvas task"), WINDOW)
        with self.assertRaises(DriverError):
            await driver.find_window("Canvas", "Other")

    async def test_explicit_app_path_launches_bundle_and_discovers_window_by_app_name(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            [
                "--model",
                "rule",
                "--rethink",
                "off",
                "--episodes",
                "1",
                "--app",
                "Canvas",
                "--app-path",
                "/tmp/Canvas.app",
                "--window-title",
                "Canvas task",
                "--goal",
                "click Save",
                "--expect",
                "Saved",
            ]
        )
        driver = AsyncMock()
        driver.find_window.return_value = WINDOW
        driver.window_state.return_value = Snapshot(WINDOW, "s1", (), {})
        with (
            patch.object(desktop, "driver_from_env", return_value=driver),
            patch.object(desktop, "launch_app", AsyncMock()) as launch,
            patch("sys.platform", "darwin"),
        ):
            task = desktop.make_series(args)
            async with task.session:
                await task.env_for(0).reset()
        launch.assert_awaited_once_with("/tmp/Canvas.app", driver, "Canvas task")
        self.assertEqual(driver.find_window.await_args_list, [call("Canvas", "Canvas task")] * 2)

    async def test_multimodal_ax_task_receives_current_window_image(self) -> None:
        driver = AsyncMock()
        driver.find_window.return_value = WINDOW

        picture = Image(_png((10, 20, 30)))

        async def snapshot(window, *, screenshot=False):
            self.assertTrue(screenshot)
            return Snapshot(window, "s1", (), {}, capture=Capture(str(window.window_id), picture, 100, 100))

        driver.window_state.side_effect = snapshot
        env = WindowEnv(
            driver,
            app_name="Canvas",
            goal="save",
            done_when=lambda s: False,
            execute=False,
            clear_labels=(),
            screenshot=True,
        )
        await env.reset()
        self.assertEqual(await env.images(), (picture,))
        self.assertEqual(set(await env.candidates()), {"abstain"})

    async def test_screenshot_goes_to_model_and_selected_point_uses_its_capture(self) -> None:
        picture = Image(_png((40, 50, 60)))
        capture = Capture("capture-1", picture, 800, 600)
        snapshot = Snapshot(WINDOW, "s1", (), {}, capture=capture)
        driver = AsyncMock()
        driver.find_window.return_value = WINDOW
        driver.window_state.return_value = snapshot
        env = WindowEnv(
            driver,
            app_name="Canvas",
            goal="click the tile marked Save",
            done_when=lambda s: False,
            execute=True,
            clear_labels=(),
            pixel_targets={"left": (0.25, 0.5), "right": (0.75, 0.5)},
        )
        await env.reset()
        model = ScriptedModel(choose="pixel:right")
        slot = ToolDecisionModel(env, EvalState(), rules="look at the image", decision_model=model, fallback=None)
        msg = await slot._decide()
        self.assertEqual(model.calls[0][0].images, (picture,))
        self.assertEqual(json.loads(msg.tool_calls[0].arguments), {"key": "pixel:right"})
        await env.step("pixel:right")
        driver.click_at.assert_awaited_once_with(WINDOW, capture, 600.0, 300.0)
        self.assertNotIn("png", json.dumps(await env.observe()))

    async def test_dry_run_never_clicks_and_missing_capture_is_an_error(self) -> None:
        driver = AsyncMock()
        driver.find_window.return_value = WINDOW
        driver.window_state.return_value = Snapshot(
            WINDOW, "s1", (), {}, capture=Capture("c", Image(_png((70, 80, 90))), 100, 100)
        )
        env = WindowEnv(
            driver,
            app_name="Canvas",
            goal="click",
            done_when=lambda s: False,
            execute=False,
            clear_labels=(),
            pixel_targets={"left": (0.25, 0.5)},
        )
        await env.reset()
        await env.step("pixel:left")
        driver.click_at.assert_not_called()
        self.assertEqual((await env.observe())["planned"]["key"], "pixel:left")
        driver.window_state.return_value = Snapshot(WINDOW, "s2", (), {})
        with self.assertRaisesRegex(DriverError, "capture"):
            await env.reset()

    async def test_driver_preserves_image_content_and_passes_capture_id_to_pixel_click(self) -> None:
        response = CallToolResult(
            content=[ImageContent(type="image", data=base64.b64encode(b"png").decode(), mimeType="image/png")],
            structuredContent={
                "elements": [],
                "capture_id": "c1",
                "screenshot_frame_valid": True,
                "screenshot_width": 800,
                "screenshot_height": 600,
            },
        )
        driver, session = _driver(
            response, _result({"effect": "confirmed"}), _result({"effect": "refused", "escalation": "stale_capture"})
        )
        snap = await driver.window_state(WINDOW, screenshot=True)
        self.assertEqual(snap.capture, Capture("c1", Image(b"png"), 800, 600))
        await driver.click_at(WINDOW, snap.capture, 300, 400)
        self.assertEqual(session.calls[-1][1]["capture_id"], "c1")
        self.assertEqual(session.calls[-1][1]["target"], WINDOW.target)
        with self.assertRaisesRegex(DriverError, "stale_capture"):
            await driver.click_at(WINDOW, snap.capture, 300, 400)
        with self.assertRaisesRegex(ValueError, "bounds"):
            await driver.click_at(WINDOW, snap.capture, 800, 0)

    async def test_unresolved_window_is_a_driver_error_not_a_model_abstention(self) -> None:
        driver, _ = _driver(
            _result(
                {"elements": [], "degraded": True, "degraded_reason": "ax_window_unresolved: no matching AX window"}
            )
        )
        with self.assertRaisesRegex(DriverError, "ax_window_unresolved"):
            await driver.window_state(WINDOW, screenshot=True)


class TestPixelArguments(TestCase):
    def test_runs_use_separate_driver_sessions_and_app_paths_are_mac_only(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            [
                "--model",
                "rule",
                "--rethink",
                "off",
                "--episodes",
                "1",
                "--app",
                "Canvas",
                "--goal",
                "click Save",
                "--expect",
                "Saved",
            ]
        )
        with patch.object(desktop, "driver_from_env", return_value=AsyncMock()) as factory:
            desktop.make_series(args)
            desktop.make_series(args)
        labels = [call.args[0] for call in factory.call_args_list]
        self.assertNotEqual(*labels)
        args.app_path = "/tmp/Canvas.app"
        with patch("sys.platform", "win32"), self.assertRaisesRegex(ValueError, "macOS"):
            desktop.make_series(args)

    def test_text_only_model_cannot_silently_ignore_required_screenshot(self) -> None:
        for model in ("laya", "jev", "llm", "cua"):
            args = series.parser(desktop.SPEC).parse_args(
                [
                    "--model",
                    model,
                    "--rethink",
                    "off",
                    "--episodes",
                    "1",
                    "--app",
                    "Canvas",
                    "--goal",
                    "click Save",
                    "--expect",
                    "Saved",
                    "--pixel-target",
                    "left=0.25,0.5",
                ]
            )
            with self.subTest(model=model), patch.dict(os.environ, {"CUA_S1_VARIANT": "nano"}):
                with self.assertRaisesRegex(ValueError, "screenshot"):
                    desktop.make_series(args)

    def test_points_are_explicit_unique_and_finite_fractions(self) -> None:
        self.assertEqual(parse_pixel_targets(["left=0.25,0.5"]), {"left": (0.25, 0.5)})
        for entries in (["bad"], ["x=nan,0.5"], ["x=1,0"], ["x=-0.1,0"], ["x=0,0", "x=0.5,0.5"]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                parse_pixel_targets(entries)


class TestVisualProgress(IsolatedAsyncioTestCase):
    """Drive pixel actions through the environment and recovery hook."""

    async def _drive(self, colours: list[tuple[int, int, int]]):
        driver = PixelWindow()
        env = WindowEnv(
            driver,
            app_name="Canvas",
            goal="paint the canvas",
            done_when=lambda s: False,
            execute=True,
            clear_labels=(),
            pixel_targets={"left": (0.5, 0.5)},
        )
        state, planner, views = EvalState(), FakePlanner(), []

        async def refresh() -> dict[str, Any]:
            await env.refresh()
            return await view_of(env, state)

        rail = RethinkRail(
            state,
            rules="paint",
            initial_score=0.0,
            planner=planner,
            stall_after=3,
            repeat_after=0,
            give_up_after=3,
            refresh=refresh,
            limits=RecoveryLimits(max_attempts=1, timeout_s=5.0),
        )
        await env.reset()
        for colour in colours:
            driver.colour = colour
            await env.step("pixel:left")
            view = await view_of(env, state)
            views.append(view["state"])
            result = json.dumps(view)
            await rail.after_tool_call(
                SimpleNamespace(
                    inputs=SimpleNamespace(tool_name="act", tool_args={"key": "pixel:left"}, tool_result=result)
                )
            )
        return state, planner, driver, views

    async def test_changed_pixels_are_progress_while_unchanged_pixels_stall_once(self) -> None:
        changed = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]
        state, planner, _driver, views = await self._drive(changed)
        # Same AX tree and title on every step; only the canvas pixels move.
        for view in views:
            self.assertEqual(view["progress"]["title"], WINDOW.title)
            self.assertEqual(view["progress"]["elements"], views[0]["progress"]["elements"])
        self.assertEqual((state.rethinks, planner.calls, state.give_up), ([], [], False))
        state, planner, driver, _views = await self._drive([(255, 0, 0)] * 6)
        self.assertEqual(len(set(driver.pngs)), len(driver.pngs), "every read is a fresh capture with new metadata")
        self.assertEqual([event["termination"] for event in state.rethinks], ["planned", "give_up"])
        self.assertEqual([event["attempt"] for event in state.rethinks], [1, 1])
        self.assertEqual((state.give_up, len(planner.calls)), (True, 1))
