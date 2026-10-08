"""Cua-S1 4B reads the current screenshot and preserves the offered option keys."""

from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from openjiuwen.core.common.exception.errors import BaseError

from s1a.decision_models import ChoiceQuestion, Image, Observation
from s1a.decision_models.cua_four_b import CuaFourBModel


class FakeFourB:
    def __init__(self) -> None:
        self.seen = []
        self.bad = False

    def load(self) -> None:
        pass

    def forward(self, options, **kwargs):
        self.seen.append(
            (options, kwargs, Path(kwargs["screenshot"]).read_bytes() if kwargs.get("screenshot") else None)
        )
        return [
            SimpleNamespace(
                element_id="unknown" if self.bad else option.element_id,
                probability=0.8 / (len(options) - 1) if i else 0.2,
            )
            for i, option in reversed(list(enumerate(options)))
        ]


class TestFourB(IsolatedAsyncioTestCase):
    async def test_desktop_options_keep_native_click_and_fill_semantics(self) -> None:
        scorer = FakeFourB()
        model = CuaFourBModel(scorer, model="test/text", modality="text")
        observation = Observation(
            {
                "app": "Editor",
                "elements": [
                    {"role": "AXTextField", "label": "Untitled.txt", "identifier": "filename"},
                    {"role": "AXButton", "label": "Save"},
                ],
            }
        )
        await model.decide_many(
            observation,
            {
                "pick": ChoiceQuestion(
                    {
                        "type:filename": 'Replace AXTextField "filename" with "report.txt"',
                        "click:Save": 'AXButton "Save"',
                        "abstain": "Stop",
                    }
                )
            },
        )
        options = scorer.seen[0][0]
        self.assertEqual(
            [(o.role, o.action) for o in options], [("TextField", "fill"), ("Button", "click"), ("action", "select")]
        )
        self.assertIn("report.txt", options[0].entity_id)
        self.assertEqual([o.element_id for o in options], ["type:filename", "click:Save", "abstain"])

    async def test_multimodal_reads_bytes_and_maps_scores_by_key_not_return_order(self) -> None:
        scorer = FakeFourB()
        model = CuaFourBModel(scorer, model="test/mm", modality="multimodal")
        obs = Observation({"goal": "click Save"}, (Image(b"image bytes"),))
        decision = await model.decide_many(
            obs, {"pick": ChoiceQuestion({"pixel:left": "left point", "pixel:right": "right point"})}
        )
        self.assertEqual(decision.choice("pick").key, "pixel:right")
        self.assertEqual(decision.choice("pick").probabilities, {"pixel:left": 0.2, "pixel:right": 0.8})
        self.assertEqual(scorer.seen[0][2], b"image bytes")
        self.assertFalse(Path(scorer.seen[0][1]["screenshot"]).exists())
        scorer.bad = True
        with self.assertRaises(BaseError):
            await model.decide_many(obs, {"pick": ChoiceQuestion({"a": "A", "b": "B"})})

    async def test_missing_image_and_too_many_options_fail_before_forward(self) -> None:
        scorer = FakeFourB()
        model = CuaFourBModel(scorer, model="test/mm", modality="multimodal")
        with self.assertRaises(BaseError):
            await model.decide_many(Observation({}), {"pick": ChoiceQuestion({"a": "A"})})
        with self.assertRaises(BaseError):
            await model.decide_many(
                Observation({}, (Image(b"p"),)), {"pick": ChoiceQuestion({str(n): str(n) for n in range(27)})}
            )
        self.assertEqual(scorer.seen, [])

    async def test_text_adapter_uses_state_without_requesting_an_image(self) -> None:
        scorer = FakeFourB()
        model = CuaFourBModel(scorer, model="test/text", modality="text")
        await model.decide_many(Observation({"goal": "choose B"}), {"pick": ChoiceQuestion({"a": "A", "b": "B"})})
        self.assertIn("choose B", scorer.seen[0][1]["ax_tree"])
        self.assertIsNone(scorer.seen[0][2])
