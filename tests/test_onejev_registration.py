# coding: utf-8
"""``onejev`` is registered for image-bearing browser decisions and the model factory."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from s1a.browser import browse
from s1a.browser.decision_model import BrowserPolicy
from s1a.decision_models import DECISION_MODEL_NAMES, build_model
from s1a.decision_models.fakes import ScriptedModel
from s1a.spec import BrowserAgentSpec, Budget


def spec() -> BrowserAgentSpec:
    return BrowserAgentSpec(
        name="fixture",
        description="A browser fixture",
        rules="Choose an offered action",
        language="en",
        budget=Budget(max_steps=2, timeout_s=10, stall_after=0),
        goal="Open the fixture",
    )


class TestOneJevRegistration(TestCase):
    def test_factory_and_browser_offer_the_backend(self) -> None:
        self.assertIn("onejev", DECISION_MODEL_NAMES)
        args = browse.parser(spec()).parse_args(["--model", "onejev"])
        self.assertEqual(args.model, "onejev")


class TestOneJevEntrypoints(IsolatedAsyncioTestCase):
    async def test_factory_builds_the_served_vision_model_from_the_environment(self) -> None:
        with mock.patch.dict(os.environ, {"ONEJEV_URL": "http://127.0.0.1:8000", "ONEJEV_TIMEOUT_S": "5"}):
            model = build_model("onejev")
        try:
            self.assertEqual(type(model).__name__, "OneJevModel")
            self.assertEqual(model.name, "onejev")
            self.assertTrue(model.supports_images)
            self.assertTrue(model.deterministic)
        finally:
            await model.close()

    async def test_browser_routes_the_backend_to_the_decision_model_slot(self) -> None:
        decision_model = ScriptedModel()
        decision_model.name = "onejev"
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
                    model_name="onejev",
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
