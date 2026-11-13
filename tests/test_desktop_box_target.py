# coding: utf-8
"""Desktop --box-target: Jev Box locate + gated offers (#98), offline with a fake transport/model."""

from __future__ import annotations

import json
import os
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from s1a.agents import desktop
from s1a.decision_models import (
    Box,
    BoxQuestion,
    Image,
    JevModel,
    NoulQuestion,
    Observation,
    ScriptedModel,
    ScriptedTransport,
    validate_box,
)
from s1a.desktop.driver import Capture, Snapshot, Window
from s1a.desktop.env import WindowEnv
from s1a.tool.models import EvalState, ToolDecisionModel
from s1a.tool import series
from test_desktop_visual import _png

WINDOW = Window(1, 2, "Canvas", "Canvas")


def _snapshot(capture_id: str = "c1", size: tuple[int, int] = (800, 600)) -> Snapshot:
    return Snapshot(
        WINDOW,
        "s1",
        (),
        {},
        capture=Capture(capture_id, Image(_png((10, 20, 30))), *size),
    )


class TestBoxTypes(TestCase):
    def test_validate_box_and_centre(self) -> None:
        box = validate_box(
            {"type": "box", "box": {"top": 0.2, "bottom": 0.6, "left": 0.1, "right": 0.5}, "confidence": 0.8}
        )
        self.assertEqual(box.centre, (0.3, 0.4))
        self.assertEqual(box.confidence, 0.8)
        with self.assertRaises(Exception):
            validate_box({"box": {"top": 0.6, "bottom": 0.2, "left": 0.1, "right": 0.5}, "confidence": 0.8})

    def test_jev_wire_box_question(self) -> None:
        from s1a.decision_models.jev import jev_question

        self.assertEqual(
            jev_question(BoxQuestion("the Save tile in `screenshot`")),
            {"type": "box", "instructions": "the Save tile in `screenshot`"},
        )


class TestBoxArguments(TestCase):
    def test_parse_and_reject_bad_entries(self) -> None:
        self.assertEqual(desktop.parse_box_targets(["Save=the Save tile"]), {"Save": "the Save tile"})
        for entries in (["bad"], ["=desc"], ["k="], ["a=one", "a=two"]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                desktop.parse_box_targets(entries)

    def test_requires_jev_114_with_images(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            [
                "--model",
                "jev",
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
                "--box-target",
                "Save=the Save tile",
            ]
        )
        clean = {
            "TYPESAFE_API_KEY": "k",
            "TYPESAFE_API_URL": "",
            "OPENROUTER_API_KEY": "",
            "S1A_JEV_IMAGES": "",
            "TYPESAFE_MODEL": "",
        }
        with patch.dict(os.environ, clean), self.assertRaisesRegex(ValueError, "S1A_JEV_IMAGES"):
            desktop.make_series(args)
        with patch.dict(os.environ, {**clean, "S1A_JEV_IMAGES": "1", "TYPESAFE_MODEL": "jev-1.13.0"}):
            with self.assertRaisesRegex(ValueError, "1.14"):
                desktop.make_series(args)
        with patch.dict(os.environ, {**clean, "S1A_JEV_IMAGES": "1", "TYPESAFE_MODEL": "jev-1.14.0"}):
            with patch.object(desktop, "driver_from_env", return_value=AsyncMock()):
                series_obj = desktop.make_series(args)
                env = series_obj.env_for(0)
        self.assertEqual(env.box_targets, {"Save": "the Save tile"})


class TestBoxLocate(IsolatedAsyncioTestCase):
    async def test_absent_and_low_confidence_not_offered_centre_click_and_outside_rejected(self) -> None:
        from s1a.desktop.env import BoxHit

        driver = AsyncMock()
        driver.find_window.return_value = WINDOW
        driver.window_state.return_value = _snapshot()
        import math

        async def bounds_check(window, capture, x, y):
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < capture.width and 0 <= y < capture.height):
                raise ValueError("pixel click outside capture bounds")
            return {"effect": "confirmed"}

        driver.click_at.side_effect = bounds_check
        env = WindowEnv(
            driver,
            app_name="Canvas",
            goal="click Save",
            done_when=lambda s: False,
            execute=True,
            clear_labels=(),
            box_targets={"Save": "the Save tile", "Missing": "a unicorn", "Weak": "a faint ghost"},
            presence_threshold=0.5,
            box_confidence_threshold=0.5,
        )
        await env.reset()
        model = ScriptedModel(
            answers=[
                {
                    "present:Save": {"noul": 0.9},
                    "box:Save": {
                        "type": "box",
                        "box": {"top": 0.4, "bottom": 0.6, "left": 0.7, "right": 0.9},
                        "confidence": 0.85,
                    },
                    "present:Missing": {"noul": 0.1},
                    "box:Missing": {
                        "type": "box",
                        "box": {"top": 0.1, "bottom": 0.2, "left": 0.1, "right": 0.2},
                        "confidence": 0.9,
                    },
                    "present:Weak": {"noul": 0.9},
                    "box:Weak": {
                        "type": "box",
                        "box": {"top": 0.1, "bottom": 0.2, "left": 0.1, "right": 0.2},
                        "confidence": 0.2,
                    },
                }
            ]
        )
        record = await env.locate_box_targets(model)
        assert record is not None
        self.assertEqual(set(record["offered"]), {"Save"})
        reasons = {row["key"]: row["reason"] for row in record["rejected"]}
        self.assertEqual(reasons["Missing"], "low_presence")
        self.assertEqual(reasons["Weak"], "low_box_confidence")
        candidates = await env.candidates()
        self.assertIn("box:Save", candidates)
        self.assertNotIn("box:Missing", candidates)
        self.assertNotIn("box:Weak", candidates)
        await env.step("box:Save")
        # centre of left=0.7,right=0.9,top=0.4,bottom=0.6 → (0.8, 0.5) → (640, 300) on 800x600
        self.assertEqual(driver.click_at.await_args.args[2:], (640.0, 300.0))
        # a box whose centre maps outside the capture is rejected by the same bounds check as pixel targets
        env._box_hits["Off"] = BoxHit(
            key="Off",
            description="off",
            box=Box(-0.4, -0.1, -0.5, -0.2, 0.99),  # centre outside [0, 1)
            presence=0.99,
            capture_id="c1",
        )
        with self.assertRaisesRegex(ValueError, "bounds"):
            await env.step("box:Off")

    async def test_tool_loop_records_box_on_tick(self) -> None:
        driver = AsyncMock()
        driver.find_window.return_value = WINDOW
        capture = Capture("cap-9", Image(_png((1, 2, 3))), 400, 400)
        driver.window_state.return_value = Snapshot(WINDOW, "s1", (), {}, capture=capture)
        env = WindowEnv(
            driver,
            app_name="Canvas",
            goal="click",
            done_when=lambda s: False,
            execute=False,
            clear_labels=(),
            box_targets={"Save": "the Save tile"},
        )
        await env.reset()
        model = ScriptedModel(
            choose="box:Save",
            answers=[
                {
                    "present:Save": {"noul": 0.95},
                    "box:Save": {
                        "type": "box",
                        "box": {"top": 0.25, "bottom": 0.75, "left": 0.25, "right": 0.75},
                        "confidence": 0.9,
                    },
                },
                # second decide_many is the choice; ScriptedModel uses choose=
            ],
        )
        # After first answers dict is consumed, choice uses _answer / choose
        state = EvalState()
        slot = ToolDecisionModel(env, state, rules="look", decision_model=model, fallback=None)
        msg = await slot._decide()
        self.assertEqual(json.loads(msg.tool_calls[0].arguments), {"key": "box:Save"})
        self.assertEqual(state.ticks[0]["box"]["capture_id"], "cap-9")
        self.assertEqual(state.ticks[0]["box"]["centre"], [0.5, 0.5])
        self.assertIn("box_locate", state.ticks[0])
        await env.step("box:Save")
        self.assertEqual((await env.observe())["planned"]["box"]["capture_id"], "cap-9")

    async def test_jev_model_sends_box_with_images(self) -> None:
        import io
        from PIL import Image as PILImage

        transport = ScriptedTransport(
            answers=[
                {
                    "present:Save": {"noul": 0.8},
                    "box:Save": {
                        "type": "box",
                        "box": {"top": 0.1, "bottom": 0.3, "left": 0.2, "right": 0.4},
                        "confidence": 0.7,
                    },
                }
            ]
        )
        transport.images = True
        model = JevModel(transport)
        buf = io.BytesIO()
        PILImage.new("RGB", (320, 320), (40, 50, 60)).save(buf, format="PNG")
        decision = await model.decide_many(
            Observation({"goal": "click"}, images=(Image(buf.getvalue()),)),
            {
                "present:Save": NoulQuestion("Is the Save tile visible in `screenshot`?"),
                "box:Save": BoxQuestion("the Save tile in `screenshot`"),
            },
        )
        self.assertAlmostEqual(decision.box("box:Save").centre[0], 0.3)
        self.assertAlmostEqual(decision.box("box:Save").centre[1], 0.2)
        self.assertEqual(transport.bodies[0]["questions"]["box:Save"]["type"], "box")
        self.assertIn("TypeSafe-Danger-Zone-Images", transport.headers[0])
