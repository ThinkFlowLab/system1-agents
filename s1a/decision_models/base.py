# coding: utf-8
"""The decision-model interface: one backend call per ``_decide`` returns a ``Reply``; ``decide_many`` validates it into a
``Decision`` and re-asks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Protocol

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError, build_error
from openjiuwen.core.common.logging import logger

from s1a.decision_models.types import (
    Answer,
    Choice,
    ChoiceQuestion,
    Decision,
    Json,
    Noul,
    NoulQuestion,
    Observation,
    Question,
    Reply,
)
from s1a.decision_models.validation import validate_answers


class JevTransport(Protocol):
    """The wire seam: post one Jev-shaped body, get the payload and the round-trip in ms."""

    model: str

    async def warm(self) -> None: ...
    async def decide(self, body: Json) -> tuple[Json, int]: ...
    async def close(self) -> None: ...


class DecisionModel(ABC):
    """A model that reads an observation and a discrete action space and returns a distribution over it."""

    name: str = "decision_model"  # the ``--model`` value; lands in every tick's ``source``
    bills_input_tokens: bool  # True when input tokens are priced at JEV_USD_PER_INPUT_TOKEN
    supports_images: bool = False
    question_types: frozenset[str] = frozenset({"choice", "noul"})
    deterministic: bool = False  # the same request always gets the same answer, so a re-ask is a wasted call
    _warned_images: bool = False

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier sent or loaded."""

    @abstractmethod
    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        """One backend call: translate the questions, return the backend's answers as the dicts it sent."""

    async def decide_many(
        self, observation: Observation, questions: dict[str, Question], *, attempts: int = 1
    ) -> Decision:
        """Ask every question over one observation, validate, re-ask up to ``attempts`` times on an unusable answer.

        A text-only model reads the observation without its images and warns once per instance.
        ``MODEL_SERVICE_CONFIG_ERROR`` when there is no question or a question's type is not supported.
        ``MODEL_CALL_FAILED`` from the backend at once, and from validation after the last attempt.
        """
        self._check(questions)
        observation = self._readable(observation)
        tries = 1 if self.deterministic else max(1, attempts)
        for attempt in range(1, tries + 1):
            reply = await self._decide(observation, questions)
            try:
                return validate_answers(reply, questions)
            except BaseError as exc:
                if attempt == tries:
                    raise
                logger.warning("[%s] re-asking after an unusable answer: %s", self.name, exc)
        raise AssertionError("unreachable")

    async def decide(
        self, observation: Observation, question: Question, *, name: str = "pick", attempts: int = 1
    ) -> Answer:
        """One question; its answer."""
        return (await self.decide_many(observation, {name: question}, attempts=attempts)).answers[name]

    async def choose(
        self,
        observation: Observation,
        options: dict[str, str | Json],
        *,
        goal: str = "",
        rules: str | tuple[str, ...] = (),
        attempts: int = 1,
    ) -> Choice:
        """Shorthand for one choice question."""
        question = ChoiceQuestion(options, goal=goal, rules=rules)
        answer = await self.decide(observation, question, attempts=attempts)
        assert isinstance(answer, Choice)
        return answer

    async def ask(self, observation: Observation, question: str, *, criteria: dict[str, str] | None = None) -> Noul:
        """Shorthand for one noul question."""
        answer = await self.decide(observation, NoulQuestion(question, criteria), name="check")
        assert isinstance(answer, Noul)
        return answer

    async def warm(self) -> None:
        """Open connections or load weights ahead of the first decision."""
        return None

    async def close(self) -> None:
        return None

    def _readable(self, observation: Observation) -> Observation:
        """The observation as this model can read it: without its images when it reads text only."""
        if not observation.images or self.supports_images:
            return observation
        if not self._warned_images:
            logger.warning(
                "[%s] reads text only: %d image(s) dropped from this and every later observation",
                self.name,
                len(observation.images),
            )
            self._warned_images = True
        return replace(observation, images=())

    def _check(self, questions: dict[str, Question]) -> None:
        if not questions:
            raise build_error(StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="no question to decide")
        for name, question in questions.items():
            if question.type not in self.question_types:
                raise build_error(
                    StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                    error_msg=f"the {self.name} model answers no {question.type} question ({name!r})",
                )
