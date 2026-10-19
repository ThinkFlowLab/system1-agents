# coding: utf-8
"""The baselines: chance is uniform and reproducible per seed, a rule is one-hot and certain."""

from __future__ import annotations

import random
from unittest import IsolatedAsyncioTestCase

from openjiuwen.core.common.exception.errors import BaseError

from decision_model_contract import OBSERVATION, OPTIONS, PICK, DecisionModelContract
from s1a.decision_models import (
    DecisionModel,
    Choice,
    ChoiceQuestion,
    Observation,
    RandomModel,
    RuleModel,
)


class TestRandomContract(DecisionModelContract, IsolatedAsyncioTestCase):
    scriptable = False
    supports_noul = False

    def make(self) -> DecisionModel:
        return RandomModel(1)


class TestRuleContract(DecisionModelContract, IsolatedAsyncioTestCase):
    scriptable = False
    supports_noul = False

    def make(self) -> DecisionModel:
        return RuleModel("last-option", lambda state, options: list(options)[-1])


class TestRandomModel(IsolatedAsyncioTestCase):
    async def _picks(self, seed: int) -> list[str]:
        decision_model = RandomModel(seed)
        return [(await decision_model.decide(OBSERVATION, PICK)).key for _ in range(20)]  # type: ignore[union-attr]

    async def test_uniform_with_peak_confidence_and_no_cost(self) -> None:
        decision = await RandomModel(0).decide_many(OBSERVATION, {"pick": PICK})
        choice = decision.choice("pick")
        # contract confidence is max p (= 0.5 for uniform two-way); backend sent 0.0 for provenance
        self.assertEqual(choice.probabilities, {"hit": 0.5, "stand": 0.5})
        self.assertEqual((choice.confidence, choice.backend_confidence), (0.5, 0.0))
        self.assertEqual((decision.latency_ms, decision.usage.input_tokens, decision.model), (0, 0, "random"))

    async def test_reproducible_per_seed_and_not_the_stream_of_random_seeded_with_the_integer(self) -> None:
        self.assertEqual(await self._picks(1), await self._picks(1))
        self.assertNotEqual(await self._picks(1), await self._picks(2))
        rng = random.Random(1)
        self.assertNotEqual(await self._picks(1), [rng.choice(list(OPTIONS)) for _ in range(20)])


class TestRuleModel(IsolatedAsyncioTestCase):
    async def test_one_hot_on_the_rule_with_full_confidence_and_the_rule_name_as_source(self) -> None:
        seen = []
        decision_model = RuleModel("always-inc", lambda state, options: seen.append((state, options)) or "inc")
        options = {"inc": "add one", "noop": "do nothing"}
        decision = await decision_model.decide_many(Observation({"n": 1}), {"pick": ChoiceQuestion(options)})
        self.assertEqual(decision.choice("pick"), Choice("inc", {"inc": 1.0, "noop": 0.0}, 1.0, backend_confidence=1.0))
        self.assertEqual(
            (decision_model.name, decision_model.model, decision.latency_ms), ("always-inc", "always-inc", 0)
        )
        self.assertEqual(seen, [({"n": 1}, options)])

    async def test_a_text_observation_reaches_the_rule_as_a_dict(self) -> None:
        decision_model = RuleModel("first", lambda state, options: state["text"] and next(iter(options)))
        choice = await decision_model.decide(Observation("plain"), PICK)
        self.assertEqual(choice.key, "hit")  # type: ignore[union-attr]

    async def test_a_foreign_key_is_a_runtime_error_not_a_decision_failure(self) -> None:
        broken = RuleModel("broken", lambda state, options: "dec")
        with self.assertRaises(RuntimeError) as caught:
            await broken.decide_many(OBSERVATION, {"pick": PICK})
        self.assertNotIsInstance(caught.exception, BaseError)
        self.assertIn("'dec'", str(caught.exception))
