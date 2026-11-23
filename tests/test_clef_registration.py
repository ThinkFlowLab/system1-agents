# coding: utf-8
"""``clef`` is registered on the text fronts, the browser front, and the model factory."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from s1a.browser import browse
from s1a.browser.decision_model import BrowserPolicy
import typing

from s1a import cli, mcp_server, rails
from s1a.decision_models import DECISION_MODEL_NAMES, build_model
from s1a.decision_models.fakes import ScriptedModel
from s1a.spec import BrowserAgentSpec, Budget
from s1a.tool import loop, series


def spec() -> BrowserAgentSpec:
    return BrowserAgentSpec(
        name="fixture",
        description="A browser fixture",
        rules="Choose an offered action",
        language="en",
        budget=Budget(max_steps=2, timeout_s=10, stall_after=0),
        goal="Open the fixture",
    )


class TestClefRegistration(TestCase):
    def test_factory_and_fronts_offer_the_backend(self) -> None:
        self.assertIn("clef", DECISION_MODEL_NAMES)
        self.assertIn("clef", browse.BROWSER_MODEL_NAMES)
        self.assertIn("clef", cli.DECIDE_MODEL_NAMES)
        self.assertIn("clef", loop.MODEL_NAMES)
        self.assertIn("clef", rails.RAIL_MODEL_NAMES)
        decide_model = typing.get_type_hints(
            mcp_server.decide.__wrapped__ if hasattr(mcp_server.decide, "__wrapped__") else mcp_server.decide
        )["model"]
        self.assertIn("clef", typing.get_args(decide_model))
        args = browse.parser(spec()).parse_args(["--model", "clef"])
        self.assertEqual(args.model, "clef")


class TestClefEntrypoints(IsolatedAsyncioTestCase):
    async def test_factory_builds_the_served_model_from_the_environment(self) -> None:
        with mock.patch.dict(os.environ, {"CLEF_URL": "http://127.0.0.1:30000", "CLEF_TIMEOUT_S": "5"}):
            model = build_model("clef")
        try:
            self.assertEqual(type(model).__name__, "ClefModel")
            self.assertEqual(model.name, "clef")
            self.assertEqual(model.model, "clef-flash")
            self.assertTrue(model.supports_images)
            self.assertTrue(model.deterministic)
        finally:
            await model.close()

    async def test_browser_routes_the_backend_to_the_decision_model_slot(self) -> None:
        decision_model = ScriptedModel()
        decision_model.name = "clef"
        slot = mock.MagicMock()
        slot.report.return_value = {"jev_input_tokens": 0, "decisions": 1}
        slot.ticks = []
        final = json.dumps({"status": "DONE", "answer": "Fixture answer"})
        with (
            mock.patch.object(browse, "CountingModel") as counting,
            mock.patch.object(browse, "BrowserDecisionModel", return_value=slot) as slot_class,
            mock.patch.object(browse, "create_browser_agent") as create_agent,
            mock.patch.object(
                browse,
                "run_task",
                new=mock.AsyncMock(return_value={"ok": True, "final": final, "error": None}),
            ) as run_task,
            mock.patch.object(browse, "usage_summary", return_value={"decisions": 1}),
        ):
            with tempfile.TemporaryDirectory() as directory:
                answer = await browse.browse(
                    spec(),
                    BrowserPolicy(prefetch_values=False, batch_actions=False, goal_value_cache=False),
                    model_name="clef",
                    goal="Open the fixture",
                    timeout_s=10,
                    max_steps=2,
                    logs_dir=Path(directory),
                    headless=True,
                    chat=mock.MagicMock(),
                    decision_model=decision_model,
                )
                self.assertTrue((Path(directory) / "decision_ticks.json").is_file())
        self.assertIs(slot_class.call_args.args[2], counting.return_value)
        self.assertIs(slot_class.call_args.kwargs["decision_model"], decision_model)
        self.assertEqual(slot_class.call_args.kwargs["max_steps"], 2)
        self.assertIs(create_agent.call_args.args[0], slot)
        run_task.assert_awaited_once()
        self.assertEqual(answer["final"], "Fixture answer")


class TestSeriesSharesTheModel(TestCase):
    def test_series_treats_clef_as_a_shared_decision_model(self) -> None:
        source = Path(series.__file__).read_text(encoding="utf-8")
        self.assertIn('("jev", "clm", "laya", "laya-served", "cua", "clef")', source)
