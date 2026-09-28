# coding: utf-8
"""``DecisionModel`` through the scripted double: the contract, re-asks, the backend-error path, images."""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError, build_error

from decision_model_contract import CHECK, OBSERVATION, PICK, DecisionModelContract
from s1a.decision_models import (
    CuaS1Model,
    DecisionModel,
    JevModel,
    LayaModel,
    Choice,
    Image,
    Json,
    Noul,
    Observation,
    RandomModel,
    RuleModel,
    ScriptedModel,
)


class TestBillingContract(TestCase):
    def test_every_backend_declares_whether_input_tokens_use_jev_pricing(self) -> None:
        expected = {
            JevModel: True,
            LayaModel: False,
            CuaS1Model: False,
            RandomModel: False,
            RuleModel: False,
            ScriptedModel: False,
        }
        backends = set(DecisionModel.__subclasses__())
        self.assertLessEqual(set(expected), backends)
        for backend in backends:
            with self.subTest(backend=backend.__name__):
                declared = vars(backend).get("bills_input_tokens")
                self.assertIsInstance(declared, bool)
                if backend in expected:
                    self.assertIs(declared, expected[backend])


class TestScriptedContract(DecisionModelContract, IsolatedAsyncioTestCase):
    def make(self) -> DecisionModel:
        return ScriptedModel()

    def make_scripted(self, answers: list[dict[str, Json]]) -> DecisionModel:
        return ScriptedModel(answers=answers)


class TestScriptedModel(IsolatedAsyncioTestCase):
    async def test_the_knobs_script_the_answers_and_the_calls_are_recorded(self) -> None:
        decision_model = ScriptedModel(choose="hit", noul=[0.3], latency_ms=4, model="s-1")
        decision = await decision_model.decide_many(OBSERVATION, {"pick": PICK, "check": CHECK})
        self.assertEqual(decision.choice("pick"), Choice("hit", {"hit": 1.0, "stand": 0.0}, 1.0))
        self.assertEqual(decision.noul("check"), Noul(0.3, 0.7))
        self.assertEqual((decision.latency_ms, decision.model), (4, "s-1"))
        self.assertEqual(decision_model.calls, [(OBSERVATION, {"pick": PICK, "check": CHECK})])

    async def test_an_image_passes_through_to_a_model_that_reads_them(self) -> None:
        decision_model = ScriptedModel()
        pictured = Observation({"n": 1}, images=(Image(b"png"),))
        await decision_model.decide_many(pictured, {"pick": PICK})
        self.assertEqual(decision_model.calls[0][0].images[0].data, b"png")

    async def test_a_backend_error_is_raised_at_once_and_not_re_asked(self) -> None:
        error = build_error(StatusCode.MODEL_CALL_FAILED, error_msg="decisions endpoint unavailable")
        decision_model = ScriptedModel(error=error)
        with self.assertRaises(BaseError) as caught:
            await decision_model.decide_many(OBSERVATION, {"pick": PICK}, attempts=3)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(decision_model.calls), 1)

    async def test_the_shorthands_build_the_questions(self) -> None:
        decision_model = ScriptedModel(noul=[0.9])
        choice = await decision_model.choose(OBSERVATION, {"a": "one", "b": "two"}, rules="pick a")
        self.assertEqual(choice.key, "a")
        noul = await decision_model.ask(OBSERVATION, "Does it hold?", criteria={"true": "y", "false": "n"})
        self.assertEqual(noul.p, 0.9)
        (_, first), (_, second) = decision_model.calls
        self.assertEqual((list(first), first["pick"].rules), (["pick"], ("pick a",)))
        self.assertEqual((list(second), second["check"].criteria), (["check"], {"true": "y", "false": "n"}))
