# coding: utf-8
"""TypeSafe Jev behind a ``JevTransport``: the wire body the repo has always sent, the answers as they came."""

from __future__ import annotations

from s1a.decision_models.base import DecisionModel, JevTransport
from s1a.decision_models.types import ChoiceQuestion, Json, NoulQuestion, Observation, Question, Reply, Usage
from s1a.decision_models.wire import DECISIONS_TIMEOUT_S, client_from_env, decisions_backend_from_env


def jev_question(question: Question) -> Json:
    """One question in the decisions request shape.

    A choice question sends ``goal``, ``operation`` and ``rules`` only when set, one rule as a string and several as
    a list, so the bodies match what every front sent before the layer existed.
    """
    match question:
        case ChoiceQuestion():
            instructions: Json = {}
            if question.goal:
                instructions["goal"] = question.goal
            if question.operation:
                instructions["operation"] = question.operation
            if question.rules:
                instructions["rules"] = question.rules[0] if len(question.rules) == 1 else list(question.rules)
            return {"type": "choice", "criteria": dict(question.options), "instructions": instructions}
        case NoulQuestion():
            criteria = {"criteria": dict(question.criteria)} if question.criteria else {}
            return {"type": "noul", **criteria, "instructions": {"question": question.question}}
    raise TypeError(f"not a question: {question!r}")


class JevModel(DecisionModel):
    """TypeSafe Jev over HTTP; the transport owns the connection, its retries and the round-trip clock."""

    name = "jev"
    bills_input_tokens = True

    def __init__(self, transport: JevTransport) -> None:
        self._transport = transport

    @property
    def model(self) -> str:
        return self._transport.model

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        body = {
            "model": self.model,
            "state": observation.state,
            "questions": {name: jev_question(question) for name, question in questions.items()},
        }
        payload, ms = await self._transport.decide(body)
        fields = payload if isinstance(payload, dict) else {}
        answers = fields.get("answers")
        answers = answers if isinstance(answers, dict) else {}
        return Reply(
            answers={name: answers[name] for name in questions if name in answers},
            latency_ms=ms,
            usage=Usage.from_payload(fields.get("usage")),
            model=str(fields.get("model") or self.model),
            raw=fields,
        )

    async def warm(self) -> None:
        await self._transport.warm()

    async def close(self) -> None:
        await self._transport.close()

    @classmethod
    def from_env(cls, *, timeout_s: float = DECISIONS_TIMEOUT_S) -> "JevModel":
        """TypeSafe directly or the OpenRouter proxy, whichever the environment names."""
        return cls(client_from_env(decisions_backend_from_env(), timeout_s=timeout_s))
