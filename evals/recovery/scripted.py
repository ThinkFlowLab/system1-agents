# coding: utf-8
"""Clearly-marked scripted doubles for the recovery eval: a faulted decision model and a planner/answer model.

These are **not** a trained decision model and **not** a real benchmark. They are controlled doubles that only
exercise the recovery wiring:

- ``ScriptedRecoveryModel`` is a deterministic state machine that reads the real observation and picks only offered
  candidates. On the ``recoverable`` and ``blocked`` tasks it *deliberately keeps typing* (the injected fault) until
  the production ``draft_plan`` puts a plan in the observation; only then may it choose the page's unlock control.
  The same instance logic runs for the recovery on and off arms: with recovery off no plan ever appears, so it never
  leaves the fault. It can never complete a task by itself -- every non-terminal turn returns one real ``browser_*``
  tool call, and completion is judged by the fixture server, never by the model.
- ``ScriptedChatModel`` answers only the fallback calls the browser policy makes: the replan prompt (``draft_plan``),
  value generation, goal-value extraction and the final answer. It makes no network call.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk, Model, ModelClientConfig

from s1a.browser import prompts
from s1a.decision_models import DecisionModel, Observation, Question, Reply, Usage
from s1a.tool.rethink import REPLAN_PROMPT

from evals.recovery.fixture import FormTask

SCRIPTED_MODEL_ID = "scripted-fault-v1"
SCRIPTED_NOTE = "scripted double: controlled fault injection, not a trained model and not a real API call"

_UNUSED_CLIENT = ModelClientConfig(
    client_provider="OpenAI",
    api_key="scripted-not-a-secret",
    api_base="http://127.0.0.1:1/scripted",
    verify_ssl=False,
)


def _one_hot(key: str, ids: list[str]) -> dict[str, Any]:
    """A valid choice answer over ``ids``: the chosen key peaks, the distribution sums to one."""
    chosen = key if key in ids else ids[0]
    return {
        "choice": chosen,
        "probabilities": {candidate: (1.0 if candidate == chosen else 0.0) for candidate in ids},
        "confidence": 1.0,
    }


class ScriptedRecoveryModel(DecisionModel):
    """The faulted form-filler: type until a plan arrives, then act on the offered unlock control, then submit."""

    name = "scripted"
    bills_input_tokens = False
    supports_images = False
    deterministic = True

    def __init__(self, task: FormTask) -> None:
        self._task = task
        self._unlocked = False  # set only after a production plan offered the page's unlock control
        self.decide_calls = 0
        self.operations: list[str] = []

    @property
    def model(self) -> str:
        return SCRIPTED_MODEL_ID

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        started = time.perf_counter()
        self.decide_calls += 1
        state = observation.state if isinstance(observation.state, dict) else {}
        plan = str(state.get("plan") or "")
        elements = [row for row in (state.get("elements") or []) if isinstance(row, dict)]
        operation, target = self._choose(plan, elements)
        self.operations.append(operation)
        latency_ms = int(round((time.perf_counter() - started) * 1000))  # measured, not a fabricated constant
        return Reply(
            answers=self._answers(operation, target, questions),
            latency_ms=latency_ms,
            usage=Usage(),
            model=SCRIPTED_MODEL_ID,
        )

    # -- the faulted strategy -------------------------------------------------

    @staticmethod
    def _row(elements: list[dict[str, Any]], *, operation: str, label: str | None = None) -> dict[str, Any] | None:
        for row in elements:
            if operation not in (row.get("operations") or []):
                continue
            if label is not None and label not in str(row.get("label") or "").lower():
                continue
            return row
        return None

    def _choose(self, plan: str, elements: list[dict[str, Any]]) -> tuple[str, str | None]:
        """The operation and target index for this turn; only offered candidates, DONE only when the page shows it."""
        input_row = self._row(elements, operation="TYPE_TEXT")
        submit = self._row(elements, operation="CLICK", label="submit")
        enable = self._row(elements, operation="CLICK", label="enable")
        expected = self._task.expected_value
        if self._task.kind == "normal":
            if input_row is not None and str(input_row.get("value") or "") != expected:
                return "TYPE_TEXT", str(input_row["index"])
            if submit is not None:
                return "CLICK", str(submit["index"])
            return "DONE", None
        # recoverable/blocked: the fault keeps typing until the production plan names the unlock.
        if plan and enable is not None and not self._unlocked:
            self._unlocked = True
            return "CLICK", str(enable["index"])
        if plan or self._unlocked:
            if input_row is not None and str(input_row.get("value") or "") != expected:
                return "TYPE_TEXT", str(input_row["index"])
            if submit is not None:
                return "CLICK", str(submit["index"])
            return "DONE", None
        if input_row is not None:  # the injected fault: keep filling the field the page restores
            return "TYPE_TEXT", str(input_row["index"])
        return "DONE", None

    @staticmethod
    def _answers(operation: str, target: str | None, questions: dict[str, Question]) -> dict[str, Any]:
        """Answer every asked head with an offered key; the chosen operation's target head carries the pick."""
        answers: dict[str, Any] = {}
        for name, question in questions.items():
            ids = list(question.ids)
            if name == "operation":
                key = operation
            elif target is not None and name == f"{operation.lower()}_target":
                key = target
            else:
                key = ids[0]
            answers[name] = _one_hot(key, ids)
        return answers


class ScriptedChatModel(Model):
    """The fallback chat model as a scripted double: replan, values and the final answer, with no network call."""

    def __init__(self, *, value: str, plan: str, answer: str = "done") -> None:
        super().__init__(_UNUSED_CLIENT, None)
        self._value = value
        self._plan = plan
        self._answer = answer
        self.invoke_calls = 0
        self.planner_calls = 0  # the replan prompt's share of invoke_calls, the rest is chat fallback

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        self.invoke_calls += 1
        return AssistantMessage(content=self._reply(messages), finish_reason="stop")

    async def stream(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AsyncIterator[AssistantMessageChunk]:
        self.invoke_calls += 1
        yield AssistantMessageChunk(content=self._reply(messages), finish_reason="stop")

    def _reply(self, messages: Any) -> str:
        system = ""
        if messages:
            first = messages[0]
            system = first.get("content", "") if isinstance(first, dict) else str(getattr(first, "content", ""))
        if system == REPLAN_PROMPT:
            self.planner_calls += 1
            return self._plan
        if system in prompts.VALUE_GENERATION.values():
            return json.dumps({"text": self._value})
        if system in prompts.VALUE_EXTRACTION.values():
            return json.dumps({"values": [self._value]})
        if system in prompts.ANSWER_RULES.values():
            return self._answer
        return self._answer
