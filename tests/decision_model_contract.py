# coding: utf-8
"""The contract every decision-model backend passes: mix into an ``IsolatedAsyncioTestCase`` and provide ``make``.

``make()`` returns a decision model that answers any question; ``make_scripted(answers)`` one that answers the calls
in turn from Jev-shaped answer dicts (for backends that can script a wrong answer). Not a test module itself: no ``test_``
prefix, so pytest collects it only through the concrete classes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

from s1a.decision_models import base as base_module
from s1a.decision_models import (
    DecisionModel,
    ChoiceQuestion,
    Image,
    Json,
    NoulQuestion,
    Observation,
)

OBSERVATION = Observation({"player_total": 18, "dealer_card": 9})
OPTIONS = {"hit": "take a card", "stand": "keep the hand"}
PICK = ChoiceQuestion(OPTIONS, rules="stand on 17 or more")
CHECK = NoulQuestion("Does the player stand?", {"true": "the player stands", "false": "the player hits"})
GOOD = {"pick": {"choice": "stand", "probabilities": {"hit": 0.1, "stand": 0.9}, "confidence": 0.9}}
FOREIGN = {"pick": {"choice": "split", "probabilities": {"hit": 0.1, "stand": 0.9}, "confidence": 0.9}}
SHORT = {"pick": {"choice": "stand", "probabilities": {"stand": 1.0}, "confidence": 1.0}}


class DecisionModelContract:
    scriptable = True  # False when the backend cannot be made to answer a scripted distribution
    supports_noul = True  # False for the choice-only baselines

    def make(self) -> DecisionModel:
        raise NotImplementedError

    def make_scripted(self, answers: list[dict[str, Json]]) -> DecisionModel:
        raise NotImplementedError

    # -- shape, every backend ------------------------------------------------

    async def test_choice_distribution_covers_the_offered_keys_and_sums_to_one(self) -> None:
        choice = (await self.make().decide_many(OBSERVATION, {"pick": PICK})).choice("pick")
        self.assertEqual(sorted(choice.probabilities), ["hit", "stand"])
        self.assertAlmostEqual(sum(choice.probabilities.values()), 1.0, delta=0.02)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in choice.probabilities.values()))

    async def test_the_chosen_key_is_the_peak_of_the_distribution(self) -> None:
        choice = (await self.make().decide_many(OBSERVATION, {"pick": PICK})).choice("pick")
        self.assertIn(choice.key, OPTIONS)
        self.assertGreaterEqual(choice.probabilities[choice.key], max(choice.probabilities.values()) - 1e-6)

    async def test_confidence_and_latency_are_in_range(self) -> None:
        decision = await self.make().decide_many(OBSERVATION, {"pick": PICK})
        self.assertTrue(0.0 <= decision.choice("pick").confidence <= 1.0)
        self.assertGreaterEqual(decision.latency_ms, 0)
        self.assertGreaterEqual(decision.usage.input_tokens, 0)

    async def test_choice_confidence_is_the_peak_probability(self) -> None:
        """Every backend reports ``max_k p_k`` as ``Choice.confidence`` (issue #89)."""
        choice = (await self.make().decide_many(OBSERVATION, {"pick": PICK})).choice("pick")
        self.assertAlmostEqual(choice.confidence, max(choice.probabilities.values()), places=6)

    async def test_decide_equals_decide_many_with_one_question(self) -> None:
        one = await self.make().decide(OBSERVATION, PICK, name="pick")
        many = (await self.make().decide_many(OBSERVATION, {"pick": PICK})).answers["pick"]
        self.assertEqual(one, many)

    async def test_multiple_questions_return_one_answer_each(self) -> None:
        questions: dict[str, Any] = {"pick": PICK, "again": ChoiceQuestion({"a": "", "b": "", "c": ""})}
        if self.supports_noul:
            questions["check"] = CHECK
        decision = await self.make().decide_many(OBSERVATION, questions)
        self.assertEqual(list(decision.answers), list(questions))
        self.assertEqual(sorted(decision.choice("again").probabilities), ["a", "b", "c"])
        if self.supports_noul:
            self.assertTrue(0.0 <= decision.noul("check").p <= 1.0)

    async def test_empty_questions_is_a_config_error(self) -> None:
        with self.assertRaises(BaseError) as caught:
            await self.make().decide_many(OBSERVATION, {})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)

    async def test_an_image_reaches_the_backend_only_when_the_model_reads_images(self) -> None:
        """A text-only model decides on the observation without its images and warns once per instance."""
        decision_model = self.make()
        seen = _seen_observations(decision_model)
        pictured = Observation(OBSERVATION.state, images=(Image(b"png"),))
        with patch.object(base_module, "logger") as log:
            await decision_model.decide_many(pictured, {"pick": PICK})
            await decision_model.decide_many(pictured, {"pick": PICK})
        self.assertEqual([observation.state for observation in seen], [OBSERVATION.state] * 2)
        if decision_model.supports_images:
            self.assertEqual([observation.images for observation in seen], [pictured.images] * 2)
            log.warning.assert_not_called()
        else:
            self.assertEqual([observation.images for observation in seen], [(), ()])
            log.warning.assert_called_once()
            self.assertIn(decision_model.name, log.warning.call_args.args[1])

    async def test_unsupported_question_type_is_a_config_error(self) -> None:
        decision_model = self.make()
        if "noul" in decision_model.question_types:
            self.skipTest("this model answers noul questions")
        with self.assertRaises(BaseError) as caught:
            await decision_model.decide_many(OBSERVATION, {"check": CHECK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)

    async def test_noul_answer_is_a_probability_with_confidence(self) -> None:
        if not self.supports_noul:
            self.skipTest("choice only")
        noul = (await self.make().decide_many(OBSERVATION, {"check": CHECK})).noul("check")
        self.assertTrue(0.0 <= noul.p <= 1.0)
        self.assertTrue(0.0 <= noul.confidence <= 1.0)

    async def test_name_and_model_are_set(self) -> None:
        decision_model = self.make()
        self.assertTrue(decision_model.name and decision_model.name == decision_model.name.lower())
        self.assertTrue(decision_model.model)

    async def test_close_is_idempotent(self) -> None:
        decision_model = self.make()
        await decision_model.close()
        await decision_model.close()

    # -- scripted, backends that can answer wrongly on purpose ----------------

    async def test_an_answer_outside_the_menu_fails_validation(self) -> None:
        if not self.scriptable:
            self.skipTest("not scriptable")
        with self.assertRaises(BaseError) as caught:
            await self.make_scripted([FOREIGN]).decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("not one of", str(caught.exception))

    async def test_a_short_distribution_fails_validation(self) -> None:
        if not self.scriptable:
            self.skipTest("not scriptable")
        with self.assertRaises(BaseError) as caught:
            await self.make_scripted([SHORT]).decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)

    async def test_attempts_re_asks_the_same_request_then_raises(self) -> None:
        """``_decide`` runs ``attempts`` times on unusable answers, once when the backend is deterministic."""
        if not self.scriptable:
            self.skipTest("not scriptable")
        decision_model = self.make_scripted([FOREIGN, FOREIGN])
        calls = _count_decides(decision_model)
        with self.assertRaises(BaseError):
            await decision_model.decide_many(OBSERVATION, {"pick": PICK}, attempts=2)
        self.assertEqual(calls, [1] if decision_model.deterministic else [1, 1])

    async def test_a_good_answer_is_not_re_asked(self) -> None:
        if not self.scriptable:
            self.skipTest("not scriptable")
        decision_model = self.make_scripted([GOOD, FOREIGN])
        calls = _count_decides(decision_model)
        decision = await decision_model.decide_many(OBSERVATION, {"pick": PICK}, attempts=2)
        self.assertEqual((decision.choice("pick").key, calls), ("stand", [1]))

    async def test_a_bad_answer_then_a_good_one_is_the_good_one(self) -> None:
        if not self.scriptable:
            self.skipTest("not scriptable")
        decision_model = self.make_scripted([FOREIGN, GOOD])
        if decision_model.deterministic:
            self.skipTest("a deterministic backend is not re-asked")
        decision = await decision_model.decide_many(OBSERVATION, {"pick": PICK}, attempts=2)
        self.assertEqual(decision.choice("pick").key, "stand")

    async def test_a_scripted_noul_comes_back_as_given(self) -> None:
        if not (self.scriptable and self.supports_noul):
            self.skipTest("not scriptable, or choice only")
        decision = await self.make_scripted([{"check": {"noul": 0.25}}]).decide_many(OBSERVATION, {"check": CHECK})
        self.assertEqual((decision.noul("check").p, decision.noul("check").confidence), (0.25, 0.75))

    async def test_scripted_backend_confidence_is_kept_apart_from_contract_confidence(self) -> None:
        """A backend-native confidence must not become ``Choice.confidence`` (issue #89).

        Some backends rewrite the number before validation (OmniJev recomputes Jev's margin); whatever lands in
        the reply's ``confidence`` field is kept as ``backend_confidence``, while the contract field is always
        ``max_k p_k``.
        """
        if not self.scriptable:
            self.skipTest("not scriptable")
        # peak p = 0.9, with a Laya-like entropy confidence of 0.4 in the scripted dict
        skewed = {"pick": {"choice": "stand", "probabilities": {"hit": 0.1, "stand": 0.9}, "confidence": 0.4}}
        decision = await self.make_scripted([skewed]).decide_many(OBSERVATION, {"pick": PICK})
        choice = decision.choice("pick")
        self.assertEqual(choice.key, "stand")
        self.assertAlmostEqual(choice.confidence, 0.9, places=6)
        self.assertIsNotNone(choice.backend_confidence)
        self.assertEqual(decision.raw["backend_confidence"]["pick"], choice.backend_confidence)


def _seen_observations(decision_model: DecisionModel) -> list[Observation]:
    """Wrap the instance's ``_decide``; the returned list grows by the observation of every call."""
    seen: list[Observation] = []
    original = decision_model._decide

    async def recording(observation: Observation, questions: dict[str, Any]) -> Any:
        seen.append(observation)
        return await original(observation, questions)

    decision_model._decide = recording  # type: ignore[method-assign]
    return seen


def _count_decides(decision_model: DecisionModel) -> list[int]:
    """Wrap the instance's ``_decide``; the returned list grows by one per call."""
    calls: list[int] = []
    original = decision_model._decide

    async def counting(observation: Observation, questions: dict[str, Any]) -> Any:
        calls.append(1)
        return await original(observation, questions)

    decision_model._decide = counting  # type: ignore[method-assign]
    return calls
