# coding: utf-8
"""``OmniJevModel`` over a fake ``mso.infer.MSO1`` (no torch): the contract, the question and answer mapping, the
screenshot hand-off, the errors, and ``from_env`` without a clone."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

import decision_model_contract as contract
from s1a.decision_models import ChoiceQuestion, DecisionModel, Image, Observation, OmniJevModel
from s1a.decision_models import omnijev as omnijev_module

SCREEN = Image(b"\x89PNG fake screenshot")
PICTURED = Observation(contract.OBSERVATION.state, images=(SCREEN,))


class FakeMSO1:
    """``MSO1`` without torch: ``system_one`` in OmniJev's shapes (a choice leaves ``abstain`` out of its sum)."""

    def __init__(self, *, answers: list[dict[str, Any]] | None = None, error: Exception | None = None) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any], list[bytes]]] = []
        self._answers = list(answers or [])
        self._error = error

    def system_one(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        images = [Path(path).read_bytes() for path in state["images"]]  # the files exist during the call
        self.calls.append((state, questions, images))
        if self._error is not None:
            raise self._error
        if self._answers:
            return self._answers.pop(0)
        return {name: self._answer(question) for name, question in questions.items()}

    @staticmethod
    def _answer(question: dict[str, Any]) -> dict[str, Any]:
        if question["type"] == "noul":
            return {"noul": 0.3, "latency_s": 0.01}
        keys = list(question["criteria"])
        probabilities = {key: (0.6 if key == keys[-1] else 0.3 / max(1, len(keys) - 1)) for key in keys}
        return {"choice": keys[-1], "probabilities": probabilities, "abstain": 0.1, "valid": True, "confidence": 0.5}


def _model(agent: FakeMSO1 | None = None) -> OmniJevModel:
    return OmniJevModel(agent or FakeMSO1(), model="tinnel123/OmniJev-0.8B@v1.1")


class TestOmniJevContract(contract.DecisionModelContract, IsolatedAsyncioTestCase):
    """The shared contract, over an observation that carries a screenshot: OmniJev decides over one."""

    def setUp(self) -> None:
        patcher = patch.object(contract, "OBSERVATION", PICTURED)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self) -> DecisionModel:
        return _model()

    def make_scripted(self, answers: list[dict[str, Any]]) -> DecisionModel:
        return _model(FakeMSO1(answers=answers))


class TestMapping(IsolatedAsyncioTestCase):
    async def test_the_screenshot_reaches_omnijev_as_a_file_and_is_removed_after(self) -> None:
        agent = FakeMSO1()
        await _model(agent).decide_many(PICTURED, {"pick": contract.PICK})
        ((state, _questions, images),) = agent.calls
        self.assertEqual(images, [SCREEN.data])
        self.assertFalse(any(Path(path).exists() for path in state["images"]))

    async def test_the_text_state_goal_and_rules_come_before_the_ask(self) -> None:
        agent = FakeMSO1()
        question = ChoiceQuestion({"1": "a"}, goal="find flights", rules="one way only", operation="CLICK")
        await _model(agent).decide_many(PICTURED, {"click_target": question})
        instructions = agent.calls[0][1]["click_target"]["instructions"].splitlines()
        self.assertTrue(instructions[0].startswith("State: "))
        self.assertEqual(instructions[1:], ["Task: find flights", "one way only", "Which element should CLICK act on?"])

    def test_a_long_page_text_does_not_push_out_the_recent_actions_or_the_elements(self) -> None:
        state = {
            "page": {"url": "https://flights", "title": "Flights", "text": "x" * 10_000},
            "elements": [{"index": "19", "role": "button", "label": "Search"}],
            "recent_actions": [{"action": "Search", "kind": "click", "page_changed": False}],
        }
        context = omnijev_module.omnijev_context(Observation(state, images=(SCREEN,)))
        self.assertLessEqual(len(context), omnijev_module.OMNIJEV_STATE_CHARS)
        self.assertTrue(context.startswith('{"recent_actions":[{"action":"Search"'))
        self.assertIn('"label":"Search"', context)
        self.assertLess(context.count("x"), omnijev_module.OMNIJEV_PAGE_TEXT_CHARS + 10)

    def test_a_non_browser_state_is_sent_as_it_is(self) -> None:
        context = omnijev_module.omnijev_context(Observation({"score": 1}, images=(SCREEN,)))
        self.assertEqual(context, '{"score":1}')

    async def test_the_short_prompt_keeps_the_goal_and_the_ask_only(self) -> None:
        agent = FakeMSO1()
        question = ChoiceQuestion({"1": "a"}, goal="find flights", rules="one way only", operation="CLICK")
        model = OmniJevModel(agent, model="m", prompt="short")
        await model.decide_many(PICTURED, {"click_target": question})
        instructions = agent.calls[0][1]["click_target"]["instructions"]
        self.assertEqual(instructions, "Task: find flights\nWhich element should CLICK act on?")

    def test_an_unknown_prompt_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            OmniJevModel(FakeMSO1(), model="m", prompt="long")

    async def test_a_browser_element_row_becomes_its_label_and_value(self) -> None:
        agent = FakeMSO1()
        question = ChoiceQuestion(
            {"12": {"element": "[12] Where from?", "current_value": "Zurich", "role": "combobox"}}
        )
        await _model(agent).decide_many(PICTURED, {"click_target": question})
        self.assertEqual(agent.calls[0][1]["click_target"]["criteria"], {"12": "Where from? = Zurich"})

    async def test_noul_criteria_are_spelled_out_after_the_question(self) -> None:
        agent = FakeMSO1()
        await _model(agent).decide_many(PICTURED, {"check": contract.CHECK})
        asked = agent.calls[0][1]["check"]
        self.assertEqual(asked["type"], "noul")
        self.assertTrue(asked["instructions"].endswith("true: the player stands\nfalse: the player hits"))

    async def test_a_choice_is_renormalised_over_the_offered_keys(self) -> None:
        choice = (await _model().decide_many(PICTURED, {"pick": contract.PICK})).choice("pick")
        self.assertEqual(choice.key, "stand")
        self.assertAlmostEqual(sum(choice.probabilities.values()), 1.0, places=6)
        self.assertAlmostEqual(choice.probabilities["stand"], 0.6 / 0.9, places=6)  # abstain 0.1 left out
        self.assertAlmostEqual(choice.confidence, (2 * 0.6 / 0.9 - 1) / 1, places=6)

    async def test_abstain_stays_in_the_raw_reply(self) -> None:
        decision = await _model().decide_many(PICTURED, {"pick": contract.PICK})
        self.assertEqual(decision.raw["pick"]["abstain"], 0.1)

    async def test_a_noul_answer_is_the_probability_omnijev_gave(self) -> None:
        noul = (await _model().decide_many(PICTURED, {"check": contract.CHECK})).noul("check")
        self.assertEqual(noul.p, 0.3)


class TestFailures(IsolatedAsyncioTestCase):
    async def test_an_observation_without_a_screenshot_is_a_call_failure(self) -> None:
        with self.assertRaises(BaseError) as caught:
            await _model().decide_many(Observation({"page": "x"}), {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("screenshot", str(caught.exception))

    async def test_a_runtime_error_becomes_model_call_failed(self) -> None:
        with self.assertRaises(BaseError) as caught:
            await _model(FakeMSO1(error=RuntimeError("CUDA out of memory"))).decide_many(
                PICTURED, {"pick": contract.PICK}
            )
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("CUDA out of memory", str(caught.exception))


class TestFromEnv(TestCase):
    def test_without_omnijev_repo_it_is_a_config_error_naming_the_variable(self) -> None:
        with patch.dict(os.environ, {"OMNIJEV_REPO": ""}):
            with self.assertRaises(BaseError) as caught:
                OmniJevModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("OMNIJEV_REPO", str(caught.exception))

    def test_an_unknown_omnijev_prompt_is_a_config_error_before_any_load(self) -> None:
        with patch.dict(os.environ, {"OMNIJEV_PROMPT": "long", "OMNIJEV_REPO": ""}):
            with self.assertRaises(BaseError) as caught:
                OmniJevModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("OMNIJEV_PROMPT", str(caught.exception))

    def test_a_folder_that_is_not_an_omnijev_clone_is_a_config_error(self) -> None:
        with patch.dict(os.environ, {"OMNIJEV_REPO": str(Path(__file__).parent)}):
            with self.assertRaises(BaseError) as caught:
                OmniJevModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)

    def test_the_default_checkpoint_is_the_cpu_sized_release(self) -> None:
        self.assertEqual(
            (omnijev_module.OMNIJEV_DEFAULT_CHECKPOINT, omnijev_module.OMNIJEV_DEFAULT_REVISION),
            ("tinnel123/OmniJev-0.8B", "v1.1"),
        )
