# coding: utf-8
"""``LayaModel`` over a fake ``laya.Agent`` (no torch): the contract, the question mapping, the error wrap,
the filled-window error, and ``from_env`` with and without the extra."""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

from decision_model_contract import CHECK, OBSERVATION, PICK, DecisionModelContract
from s1a.decision_models import (
    DecisionModel,
    Choice,
    ChoiceQuestion,
    LayaModel,
    Noul,
    NoulQuestion,
    Observation,
    jev_question,
)
from s1a.decision_models import laya as laya_module

ACTION = {"act_probability": 0.5}


def _laya_shape(answer: dict[str, Any]) -> dict[str, Any]:
    """Laya's exact answer dicts: ``type`` and ``action`` alongside the Jev fields, a confidence always present."""
    if "noul" in answer:
        p = answer["noul"]
        return {"type": "noul", "noul": p, "confidence": answer.get("confidence", max(p, 1 - p)), "action": ACTION}
    return {"type": "choice", **answer, "action": ACTION}


class FakeLayaAgent:
    """``laya.Agent`` without torch: the same ``cfg`` keys, ``system_one`` in Laya's exact shapes, every call kept."""

    def __init__(
        self,
        *,
        answers: list[dict[str, Any]] | None = None,
        input_tokens: int = 100,
        error: Exception | None = None,
        sleep_s: float = 0.001,
    ) -> None:
        self.cfg = {"max_len": 512, "head_max_len": 192}
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self._answers = list(answers or [])
        self._input_tokens = input_tokens
        self._error = error
        self._sleep_s = sleep_s

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.calls.append((state, questions))
        if self._error is not None:
            raise self._error
        time.sleep(self._sleep_s)
        if self._answers:
            answers = self._answers.pop(0)
        else:
            answers = {name: self._answer(question) for name, question in questions.items()}
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": self._input_tokens, "output_tokens": 0},
        }

    @staticmethod
    def _answer(question: dict[str, Any]) -> dict[str, Any]:
        if question["type"] == "noul":
            return {"type": "noul", "noul": 0.5, "confidence": 0.5, "action": ACTION}
        keys = list(question["criteria"])
        probabilities = {key: (1.0 if key == keys[0] else 0.0) for key in keys}
        return {
            "type": "choice",
            "choice": keys[0],
            "probabilities": probabilities,
            "confidence": 1.0,
            "action": ACTION,
        }


def _model(agent: FakeLayaAgent | None = None) -> LayaModel:
    return LayaModel(agent or FakeLayaAgent(), model="convaiinnovations/laya")


class TestLayaContract(DecisionModelContract, IsolatedAsyncioTestCase):
    def make(self) -> DecisionModel:
        return _model()

    def make_scripted(self, answers: list[dict[str, Any]]) -> DecisionModel:
        scripted = [{name: _laya_shape(answer) for name, answer in call.items()} for call in answers]
        return _model(FakeLayaAgent(answers=scripted))


class TestMapping(IsolatedAsyncioTestCase):
    async def test_a_choice_question_and_the_state_pass_through_as_the_jev_question(self) -> None:
        agent = FakeLayaAgent()
        question = ChoiceQuestion({"1": {"element": "[1] Search", "role": "button"}}, goal="g", rules=["a", "b"])
        await _model(agent).decide_many(Observation({"page": "x"}), {"click_target": question})
        ((state, asked),) = agent.calls
        self.assertEqual(state, {"page": "x"})
        self.assertEqual(asked, {"click_target": jev_question(question)})
        self.assertEqual(asked["click_target"]["instructions"], {"goal": "g", "rules": ["a", "b"]})

    async def test_a_noul_question_sends_a_string_instruction_and_native_criteria(self) -> None:
        agent = FakeLayaAgent()
        await _model(agent).decide_many(Observation("plain text"), {"check": CHECK})
        self.assertEqual(
            agent.calls[0][1]["check"],
            {"type": "noul", "instructions": CHECK.question, "criteria": CHECK.criteria},
        )
        self.assertEqual(laya_module.laya_question(NoulQuestion("q")), {"type": "noul", "instructions": "q"})

    async def test_the_answers_come_back_typed_with_laya_usage_model_and_a_measured_latency(self) -> None:
        agent = FakeLayaAgent(input_tokens=77)
        decision = await _model(agent).decide_many(OBSERVATION, {"pick": PICK, "check": CHECK})
        self.assertEqual(decision.choice("pick"), Choice("hit", {"hit": 1.0, "stand": 0.0}, 1.0))
        self.assertEqual(decision.noul("check"), Noul(0.5, 0.5))
        self.assertEqual((decision.usage.input_tokens, decision.usage.output_tokens), (77, 0))
        self.assertEqual(decision.model, "laya-rl-agent")
        self.assertGreater(decision.latency_ms, 0)
        self.assertEqual(decision.raw["answers"]["pick"]["action"], ACTION)

    async def test_the_model_is_deterministic_and_text_only(self) -> None:
        decision_model = _model()
        self.assertTrue(decision_model.deterministic)
        self.assertFalse(decision_model.supports_images)
        self.assertEqual((decision_model.name, decision_model.model), ("laya", "convaiinnovations/laya"))


_BROWSER_STATE = {
    "page": {"url": "https://example.com/flights", "title": "Google Flights" + "!" * 100, "text": "x" * 5000},
    "elements": [
        {
            "index": "1",
            "role": "textbox",
            "label": "Where from?" + " padding" * 20,
            "value": "",
            "operations": ["TYPE_TEXT"],
        },
        {
            "index": "2",
            "role": "button",
            "label": "Search",
            "value": "",
            "checked": True,
            "operations": ["CLICK"],
        },
    ],
    "recent_actions": [
        {"action": f"step-{i}", "kind": "click", "text": "", "page_changed": i % 2 == 0} for i in range(10)
    ],
}


class TestLayaState(TestCase):
    """``laya_state`` folds the browser front's state to fit Laya's window; anything else passes through."""

    def test_a_non_browser_state_passes_through(self) -> None:
        for state in ({"page": "x"}, "plain text", {"score": 1}, {"page": {"url": "u"}, "elements": "not a list"}):
            self.assertEqual(laya_module.laya_state(state), state)

    def test_page_text_is_dropped_and_the_title_is_capped(self) -> None:
        compact = laya_module.laya_state(_BROWSER_STATE)
        self.assertEqual(compact["page"]["url"], "https://example.com/flights")
        self.assertNotIn("text", compact["page"])
        self.assertLessEqual(len(compact["page"]["title"]), laya_module.LAYA_BROWSER_TITLE_CHARS)

    def test_each_element_row_becomes_one_short_line_not_a_json_object(self) -> None:
        compact = laya_module.laya_state(_BROWSER_STATE)
        self.assertEqual(len(compact["elements"]), 2)
        self.assertTrue(all(isinstance(row, str) for row in compact["elements"]))
        self.assertLess(len(compact["elements"][0]), len("label") * 20)  # far short of the padded label
        self.assertIn("[C]", compact["elements"][1])  # the checked flag survives as a letter, not a key

    def test_history_is_capped_at_the_last_few_actions(self) -> None:
        compact = laya_module.laya_state(_BROWSER_STATE)
        self.assertEqual(len(compact["recent_actions"]), laya_module.LAYA_BROWSER_HISTORY_KEPT)
        self.assertEqual(compact["recent_actions"][-1], "click:step-9 (no change)")

    def test_compaction_shrinks_the_json_size_by_an_order_of_magnitude(self) -> None:
        import json

        raw = json.dumps(_BROWSER_STATE)
        compact = json.dumps(laya_module.laya_state(_BROWSER_STATE))
        self.assertGreater(len(raw) / len(compact), 8)


class TestLayaModelCompaction(IsolatedAsyncioTestCase):
    async def test_the_browser_state_reaching_the_agent_is_compacted_by_default(self) -> None:
        agent = FakeLayaAgent()
        question = ChoiceQuestion({"1": {"element": "[1] Search"}})
        await _model(agent).decide_many(Observation(_BROWSER_STATE), {"operation": question})
        ((state, _asked),) = agent.calls
        self.assertEqual(state, laya_module.laya_state(_BROWSER_STATE))
        self.assertNotIn("text", state["page"])

    async def test_compaction_turns_off_with_compact_browser_state_false(self) -> None:
        agent = FakeLayaAgent()
        question = ChoiceQuestion({"1": {"element": "[1] Search"}})
        model = LayaModel(agent, model="convaiinnovations/laya", compact_browser_state=False)
        await model.decide_many(Observation(_BROWSER_STATE), {"operation": question})
        ((state, _asked),) = agent.calls
        self.assertEqual(state, _BROWSER_STATE)


class TestFailures(IsolatedAsyncioTestCase):
    async def test_option_overflow_and_torch_errors_are_model_call_failures(self) -> None:
        for error in (ValueError("question 'pick' options exceed head_max_len=192"), RuntimeError("CUDA error")):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(BaseError) as caught:
                    await _model(FakeLayaAgent(error=error)).decide_many(OBSERVATION, {"pick": PICK})
                self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
                self.assertIn(str(error), str(caught.exception))

    async def test_a_filled_window_is_a_config_error_naming_the_window_and_the_knobs(self) -> None:
        await _model(FakeLayaAgent(input_tokens=511)).decide_many(OBSERVATION, {"pick": PICK})
        with self.assertRaises(BaseError) as caught:
            await _model(FakeLayaAgent(input_tokens=512)).decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("512-token window filled", str(caught.exception))
        self.assertIn("LAYA_MAX_LEN", str(caught.exception))

    async def test_the_window_scales_with_the_number_of_questions(self) -> None:
        questions = {"pick": PICK, "check": CHECK}
        await _model(FakeLayaAgent(input_tokens=600)).decide_many(OBSERVATION, questions)
        with self.assertRaises(BaseError):
            await _model(FakeLayaAgent(input_tokens=1024)).decide_many(OBSERVATION, questions)


class TestFromEnv(TestCase):
    def test_without_the_extra_it_is_a_config_error_naming_the_extra(self) -> None:
        with patch.dict(sys.modules, {"laya": None}):
            with self.assertRaises(BaseError) as caught:
                LayaModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("--extra laya", str(caught.exception))

    def test_an_agent_without_system_one_is_a_config_error_naming_the_method(self) -> None:
        fake_laya = SimpleNamespace(load=lambda *a, **k: SimpleNamespace(cfg={}))
        with patch.dict(sys.modules, {"laya": fake_laya}), patch.dict(os.environ, {"LAYA_SUBFOLDER": ""}):
            with self.assertRaises(BaseError) as caught:
                LayaModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("system_one", str(caught.exception))
        self.assertIn("laya unknown", str(caught.exception))

    def test_the_env_names_the_checkpoint_and_overrides_the_window(self) -> None:
        loads: list[tuple[Any, ...]] = []

        def load(model: str, device: Any = None, token: Any = None, subfolder: Any = None) -> FakeLayaAgent:
            loads.append((model, device, subfolder))
            return FakeLayaAgent()

        env = {
            "LAYA_MODEL": "convaiinnovations/laya",
            "LAYA_SUBFOLDER": "multilingual",
            "LAYA_DEVICE": "cpu",
            "LAYA_MAX_LEN": "1024",
            "LAYA_HEAD_MAX_LEN": "512",
        }
        with patch.dict(sys.modules, {"laya": SimpleNamespace(load=load)}), patch.dict(os.environ, env):
            decision_model = LayaModel.from_env()
        self.assertEqual(loads, [("convaiinnovations/laya", "cpu", "multilingual")])
        self.assertEqual(decision_model.model, "convaiinnovations/laya/multilingual")
        self.assertEqual(decision_model._agent.cfg, {"max_len": 1024, "head_max_len": 512})

    def test_the_defaults_when_the_env_is_empty(self) -> None:
        env = {"LAYA_MODEL": "", "LAYA_SUBFOLDER": "", "LAYA_DEVICE": "", "LAYA_MAX_LEN": "", "LAYA_HEAD_MAX_LEN": ""}
        with patch.dict(sys.modules, {"laya": SimpleNamespace(load=lambda *a, **k: FakeLayaAgent())}):
            with patch.dict(os.environ, env):
                decision_model = LayaModel.from_env()
        self.assertEqual(decision_model.model, laya_module.LAYA_DEFAULT_MODEL)
        self.assertEqual(decision_model._agent.cfg, {"max_len": 512, "head_max_len": 192})
        self.assertTrue(decision_model._compact_browser_state)

    def test_laya_compact_browser_state_env_var_turns_compaction_off(self) -> None:
        for off in ("0", "false", "False", "no"):
            with self.subTest(off=off):
                env = {"LAYA_COMPACT_BROWSER_STATE": off}
                with patch.dict(sys.modules, {"laya": SimpleNamespace(load=lambda *a, **k: FakeLayaAgent())}):
                    with patch.dict(os.environ, env):
                        decision_model = LayaModel.from_env()
                self.assertFalse(decision_model._compact_browser_state)
