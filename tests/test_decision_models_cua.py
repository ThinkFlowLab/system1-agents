# coding: utf-8
"""``CuaS1Model`` over a fake ``NanoScorer`` (no torch): the contract, the rendering, the answer mapping, the window
warning, the error wrap, and ``from_env`` with and without the extra."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

from decision_model_contract import OBSERVATION, PICK, DecisionModelContract
from s1a.decision_models import DecisionModel, ChoiceQuestion, CuaS1Model, Observation, build_model
from s1a.decision_models import cua as cua_module
from s1a.decision_models.cua import Element, cua_context, cua_option


class FakeNanoScorer:
    """``score_elements`` in Nano's exact shape: per element id, option index to probability; every call kept."""

    def __init__(self, *, peak: int = 0, error: Exception | None = None) -> None:
        self.peak = peak
        self.error = error
        self.calls: list[list[Element]] = []

    def score_elements(
        self, elements: Sequence[Element], collator: Any, device: Any = None
    ) -> dict[str, dict[int, float]]:
        self.calls.append(list(elements))
        if self.error is not None:
            raise self.error
        scores: dict[str, dict[int, float]] = {}
        for element in elements:
            n = len(element.options)
            peak = min(self.peak, n - 1)
            rest = 0.2 / (n - 1) if n > 1 else 0.0
            scores[element.element_id] = {i: (0.8 if i == peak else rest) for i in range(n)}
        return scores


def _model(scorer: FakeNanoScorer, *, context_bytes: int = 256, option_bytes: int = 96) -> CuaS1Model:
    return CuaS1Model(
        scorer, object(), model="cua-ai/cua-s1-nano-0.1/text", context_bytes=context_bytes, option_bytes=option_bytes
    )


class TestCuaContract(DecisionModelContract, IsolatedAsyncioTestCase):
    scriptable = False  # a scorer answers over the offered options only; it cannot be made to name a foreign key
    supports_noul = False

    def make(self) -> DecisionModel:
        return _model(FakeNanoScorer())


class TestRendering(TestCase):
    def test_the_context_is_header_goal_state_then_rules_so_a_cut_drops_the_rules_first(self) -> None:
        question = ChoiceQuestion({"a": ""}, goal="Fly to London", rules=["r1", "r2"])
        text = cua_context(Observation({"n": 1}), question, "pick")
        self.assertEqual(text.splitlines(), ["# s1a / pick", "goal: Fly to London", '{"n": 1}', "r1", "r2"])
        self.assertEqual(cua_context(Observation("plain"), ChoiceQuestion({"a": ""}), "q"), "# s1a / q\nplain")

    def test_an_option_is_the_trained_shape_with_the_key_before_its_description(self) -> None:
        self.assertEqual(cua_option("stand", "keep the hand"), "click:Button:stand keep the hand")
        self.assertEqual(cua_option("done", ""), "click:Button:done")
        self.assertEqual(cua_option("3", {"element": "[3] Search"}), 'click:Button:3 {"element": "[3] Search"}')


class TestDecide(IsolatedAsyncioTestCase):
    async def test_the_peak_is_the_choice_with_its_probability_as_confidence(self) -> None:
        scorer = FakeNanoScorer(peak=1)
        decision = await _model(scorer).decide_many(OBSERVATION, {"pick": PICK})
        choice = decision.choice("pick")
        self.assertEqual((choice.key, choice.confidence), ("stand", 0.8))
        self.assertEqual(choice.probabilities, {"hit": 0.2, "stand": 0.8})
        self.assertEqual((decision.usage.input_tokens, decision.model), (0, "cua-ai/cua-s1-nano-0.1/text"))
        self.assertGreaterEqual(decision.latency_ms, 0)
        self.assertEqual(decision.raw["scores"]["pick"], {0: 0.2, 1: 0.8})
        ((element,),) = scorer.calls
        self.assertEqual(
            (element.element_id, element.options),
            ("pick", ("click:Button:hit take a card", "click:Button:stand keep the hand")),
        )
        self.assertTrue(element.context.startswith("# s1a / pick\n"))

    async def test_several_questions_are_one_forward_pass(self) -> None:
        scorer = FakeNanoScorer()
        questions = {"pick": PICK, "again": ChoiceQuestion({"x": "", "y": "", "z": ""})}
        decision = await _model(scorer).decide_many(OBSERVATION, questions)
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual([e.element_id for e in scorer.calls[0]], ["pick", "again"])
        self.assertEqual(decision.choice("again").key, "x")

    async def test_torch_and_batch_errors_are_model_call_failures(self) -> None:
        for error in (ValueError("empty batch"), RuntimeError("MPS out of memory")):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(BaseError) as caught:
                    await _model(FakeNanoScorer(error=error)).decide_many(OBSERVATION, {"pick": PICK})
                self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
                self.assertIn(str(error), str(caught.exception))

    async def test_an_overflowing_request_is_warned_about_once_per_instance(self) -> None:
        decision_model = _model(FakeNanoScorer(), context_bytes=40, option_bytes=24)
        small = ChoiceQuestion({"a": "", "b": ""})
        wide = ChoiceQuestion({"a": "x" * 30, "b": ""}, rules="r" * 50)
        with patch.object(cua_module, "logger") as log:
            await decision_model.decide_many(Observation({"n": 1}), {"pick": small})
            log.warning.assert_not_called()
            await decision_model.decide_many(Observation({"n": 1}), {"pick": wide})
            await decision_model.decide_many(Observation({"n": 1}), {"pick": wide})
            log.warning.assert_called_once()
        message, name, context_over, options_over, budget = log.warning.call_args.args
        self.assertIn("overflows the window", message)
        self.assertEqual((name, options_over, budget), ("pick", 1, 24))
        self.assertGreater(context_over, 0)

    async def test_the_model_is_deterministic_text_only_and_choice_only(self) -> None:
        decision_model = _model(FakeNanoScorer())
        self.assertTrue(decision_model.deterministic)
        self.assertFalse(decision_model.supports_images)
        self.assertEqual((decision_model.name, decision_model.question_types), ("cua", frozenset({"choice"})))


def _fake_nano(config: dict[str, Any], loads: list[tuple[str, str]]) -> SimpleNamespace:
    def load_nano_checkpoint(directory: Path, device: str) -> tuple[FakeNanoScorer, object, dict[str, Any]]:
        loads.append((str(directory), device))
        return FakeNanoScorer(), object(), config

    return SimpleNamespace(load_nano_checkpoint=load_nano_checkpoint)


class TestFromEnv(TestCase):
    def test_without_the_extra_it_is_a_config_error_naming_the_extra(self) -> None:
        with patch.dict(sys.modules, {"cua_s1": None, "cua_s1.nano": None}):
            with self.assertRaises(BaseError) as caught:
                CuaS1Model.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("--extra cua", str(caught.exception))

    def test_a_local_directory_is_loaded_as_it_is_with_the_window_from_its_config(self) -> None:
        loads: list[tuple[str, str]] = []
        nano = _fake_nano({"context_tokens": 512, "option_tokens": 64}, loads)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "text").mkdir()
            env = {"CUA_S1_CHECKPOINT": tmp, "CUA_S1_SUBFOLDER": "", "CUA_S1_DEVICE": "cpu"}
            with patch.dict(sys.modules, {"cua_s1.nano": nano}), patch.dict(os.environ, env):
                decision_model = CuaS1Model.from_env()
            self.assertEqual(loads, [(str(Path(tmp) / "text"), "cpu")])
        self.assertEqual(decision_model.model, f"{tmp}/text")
        self.assertEqual((decision_model._context_bytes, decision_model._option_bytes), (512, 64))

    def test_a_hub_id_is_fetched_once_subfolder_only_then_loaded_with_the_defaults(self) -> None:
        loads: list[tuple[str, str]] = []
        fetched: list[tuple[str, list[str]]] = []

        def snapshot_download(repo_id: str, allow_patterns: list[str]) -> str:
            fetched.append((repo_id, allow_patterns))
            return "/cache/cua-s1-nano-0.1"

        hub = SimpleNamespace(snapshot_download=snapshot_download)
        env = {"CUA_S1_CHECKPOINT": "", "CUA_S1_SUBFOLDER": "", "CUA_S1_DEVICE": ""}
        with (
            patch.dict(sys.modules, {"cua_s1.nano": _fake_nano({}, loads), "huggingface_hub": hub}),
            patch.dict(os.environ, env),
        ):
            decision_model = CuaS1Model.from_env()
        self.assertEqual(fetched, [("cua-ai/cua-s1-nano-0.1", ["text/*"])])
        self.assertEqual(loads, [(str(Path("/cache/cua-s1-nano-0.1") / "text"), "auto")])
        self.assertEqual(decision_model.model, "cua-ai/cua-s1-nano-0.1/text")
        self.assertEqual((decision_model._context_bytes, decision_model._option_bytes), (256, 96))

    def test_the_factory_builds_it_for_cua(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "text").mkdir()
            with (
                patch.dict(sys.modules, {"cua_s1.nano": _fake_nano({}, [])}),
                patch.dict(os.environ, {"CUA_S1_CHECKPOINT": tmp}),
            ):
                self.assertIsInstance(build_model("cua"), CuaS1Model)
