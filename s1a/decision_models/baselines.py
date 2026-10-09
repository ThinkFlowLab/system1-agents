# coding: utf-8
"""The two baselines every eval compares against: chance (uniform) and a hand-written rule (one-hot)."""

from __future__ import annotations

import random
from typing import Any, Callable

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.types import ChoiceQuestion, Json, Observation, Question, Reply

Rule = Callable[[dict[str, Any], dict[str, str]], str]  # (state, options) -> the chosen key


class RandomModel(DecisionModel):
    """Uniform over the offered keys: the loop-overhead arm. Answers choice questions only."""

    name = "random"
    bills_input_tokens = False
    question_types = frozenset({"choice"})

    def __init__(self, seed: int) -> None:
        # a string seed: an env seeded with the plain integer draws its own stream
        self._rng = random.Random(f"random-slot-{seed}")

    @property
    def model(self) -> str:
        return "random"

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        answers: dict[str, Json] = {}
        for name, question in questions.items():
            assert isinstance(question, ChoiceQuestion)
            ids = question.ids
            # the entropy of a uniform pick is total, so its confidence is none
            answers[name] = _choice(self._rng.choice(ids), {key: 1.0 / len(ids) for key in ids}, 0.0)
        return Reply(answers=answers, latency_ms=0, model=self.model)


class RuleModel(DecisionModel):
    """A plain function ``(state, options) -> key`` in the slot: the hand-written baseline, one-hot and certain.

    A key outside the options is a bug in the rule, so it raises ``RuntimeError`` rather than becoming a recorded
    decision failure.
    """

    bills_input_tokens = False
    question_types = frozenset({"choice"})
    deterministic = True

    def __init__(self, name: str, rule: Rule) -> None:
        self.name = name
        self._rule = rule

    @property
    def model(self) -> str:
        return self.name

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        state = observation.state if isinstance(observation.state, dict) else {"text": observation.state}
        answers: dict[str, Json] = {}
        for name, question in questions.items():
            assert isinstance(question, ChoiceQuestion)
            options = {key: value for key, value in question.options.items() if isinstance(value, str)}
            assert len(options) == len(question.options), "a rule reads text options only"
            key = self._rule(state, options)
            if key not in question.options:
                raise RuntimeError(f"rule {self.name!r} chose {key!r}, not among {sorted(question.options)}")
            answers[name] = _choice(key, {k: 1.0 if k == key else 0.0 for k in question.ids}, 1.0)
        return Reply(answers=answers, latency_ms=0, model=self.model)


def _choice(key: str, probabilities: dict[str, float], confidence: float) -> Json:
    return {"choice": key, "probabilities": probabilities, "confidence": confidence}
