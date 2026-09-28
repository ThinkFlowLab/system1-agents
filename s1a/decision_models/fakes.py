# coding: utf-8
"""Scripted doubles at both levels: a transport that fakes the wire, a decision model that fakes the interface."""

from __future__ import annotations

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.types import ChoiceQuestion, Json, Observation, Question, Reply, Usage


class ScriptedTransport:
    """A ``JevTransport`` answering from a script; records every body it was sent.

    Per call: ``error`` is raised; else the next dict of ``answers`` is returned verbatim (for malformed-answer
    tests); else every question gets a one-hot choice on ``choose`` (the first offered key when None) or the next
    ``noul`` probability (0.5 when the list runs out).
    """

    model = "typesafe/jev-test"

    def __init__(
        self,
        *,
        choose: str | None = None,
        noul: list[float] | None = None,
        error: Exception | None = None,
        answers: list[Json] | None = None,
        usage: Json | None = None,
        latency_ms: int = 9,
    ) -> None:
        self._choose = choose
        self._noul = list(noul or [])
        self._error = error
        self._answers = list(answers or [])
        self._usage = usage or {"input_tokens": 0, "output_tokens": 0}
        self._latency_ms = latency_ms
        self.bodies: list[Json] = []

    async def warm(self) -> None:
        return None

    async def decide(self, body: Json) -> tuple[Json, int]:
        self.bodies.append(body)
        if self._error is not None:
            raise self._error
        if self._answers:
            answers = self._answers.pop(0)
        else:
            answers = {name: self._answer(question) for name, question in body["questions"].items()}
        return {"answers": answers, "usage": dict(self._usage)}, self._latency_ms

    async def close(self) -> None:
        return None

    def _answer(self, question: Json) -> Json:
        if question["type"] == "noul":
            return {"noul": self._noul.pop(0) if self._noul else 0.5}
        ids = list(question["criteria"])
        key = self._choose or ids[0]
        return {"choice": key, "probabilities": {i: 1.0 if i == key else 0.0 for i in ids}, "confidence": 1.0}


class ScriptedModel(DecisionModel):
    """The interface faked directly with the same knobs; ``answers`` holds one dict of answer dicts per call. Reads
    images, so a pass-through can be tested."""

    name = "scripted"
    bills_input_tokens = False
    supports_images = True

    def __init__(
        self,
        *,
        choose: str | None = None,
        noul: list[float] | None = None,
        error: Exception | None = None,
        answers: list[dict[str, Json]] | None = None,
        usage: Usage = Usage(),
        latency_ms: int = 9,
        model: str = "scripted",
    ) -> None:
        self._choose = choose
        self._noul = list(noul or [])
        self._error = error
        self._answers = list(answers or [])
        self._usage = usage
        self._latency_ms = latency_ms
        self._model = model
        self.calls: list[tuple[Observation, dict[str, Question]]] = []

    @property
    def model(self) -> str:
        return self._model

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        self.calls.append((observation, dict(questions)))
        if self._error is not None:
            raise self._error
        if self._answers:
            answers = self._answers.pop(0)
        else:
            answers = {name: self._answer(question) for name, question in questions.items()}
        return Reply(answers=answers, latency_ms=self._latency_ms, usage=self._usage, model=self._model)

    def _answer(self, question: Question) -> Json:
        if not isinstance(question, ChoiceQuestion):
            return {"noul": self._noul.pop(0) if self._noul else 0.5}
        key = self._choose or question.ids[0]
        return {"choice": key, "probabilities": {i: 1.0 if i == key else 0.0 for i in question.ids}, "confidence": 1.0}
