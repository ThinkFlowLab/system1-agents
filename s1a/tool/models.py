# coding: utf-8
"""The model in the eval loop's model slot: each turn is one ``act(key)`` tool call chosen by the model behind it.

jiuwen's DeepAgent calls ``invoke`` or ``stream`` with the turn's tool list. A turn whose tools include ``act``
is a decision turn: the model reads the environment directly and answers with exactly one ``act`` call. Any
other turn goes to the fallback chat model. Jev, Laya, chance and a rule are all decision models; the slot model does not
know which one it holds. ``Model`` is openJiuwen's chat-model base class; ``DecisionModel`` is this package's own interface.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk, Model, ToolCall, init_model

from s1a.decision_models import DecisionModel, ChoiceQuestion, Observation
from s1a.env import Env, VisualEnv
from s1a.recovery import recovery_next_action

ACT_TOOL = "act"
OBSERVE_TOOL = "observe"


def placeholder_model() -> Model:
    """A config the harness accepts for a model that never makes a chat call (``Model`` refuses a missing one)."""
    return init_model(provider="openai", model_name="none", api_key="unused", api_base="https://api.openai.com/v1")


@dataclass
class EvalState:
    """Per-episode facts shared by the slot model, the ``act`` tool and the rethink rail."""

    plan: str = ""
    notices: list[str] = field(default_factory=list)
    blocked: set[str] = field(default_factory=set)
    acts: list[dict[str, Any]] = field(default_factory=list)
    act_calls: list[dict[str, Any]] = field(default_factory=list)  # every act call, {key, accepted}, in order
    views: list[dict[str, Any]] = field(default_factory=list)  # the act tool's result after every act
    ticks: list[dict[str, Any]] = field(default_factory=list)
    rethinks: list[dict[str, Any]] = field(default_factory=list)
    chat: list[dict[str, Any]] = field(default_factory=list)  # every chat-model call, from CountingModel
    invalid_keys: int = 0  # act calls with a key that was not offered
    max_acts: int = 0  # the episode's act budget, positive in every run; 0 leaves it unbounded in tests
    give_up: bool = False
    error: str | None = None  # a decision or an act that failed; the episode is an errored trial
    bounded_recovery: bool = False  # a bounded RethinkRail is active, so give-up/error outrank a spent act budget
    terminal: dict[str, Any] | None = None  # the structured stop summary, kept whole when the text output is truncated

    @property
    def budget_spent(self) -> bool:
        return 0 < self.max_acts <= len(self.acts)


def tool_name(tools: Any, wanted: str) -> str | None:
    """The registered name of ``wanted`` in a turn's tool list, matched exactly or by ``_`` suffix."""
    for tool in tools or []:
        name = str(getattr(tool, "name", "") or "")
        if name == wanted or name.endswith("_" + wanted):
            return name
    return None


class ToolDecisionModel(Model):
    """One ``act`` call per decision turn, a stop message when the episode ends, any decision model behind it.

    The plan and the notices ride along in the observation; the question is one ``choice`` over the offered keys.
    A decision failure (``BaseError``) ends the episode as BLOCKED and is the episode's error.
    """

    def __init__(
        self, env: Env, state: EvalState, *, rules: str, decision_model: DecisionModel, fallback: Model | None
    ) -> None:
        source = fallback if fallback is not None and fallback.model_client_config is not None else placeholder_model()
        super().__init__(source.model_client_config, source.model_config)
        self._env = env
        self._rules = rules
        self._state = state
        self._decision_model = decision_model
        self._fallback = fallback
        self._act_name = ACT_TOOL
        self.name = decision_model.name  # lands in every tick's ``source`` and in ``Episode.policy``
        self.bills_input_tokens = decision_model.bills_input_tokens

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        act_name = tool_name(tools, ACT_TOOL)
        if act_name is None:
            if self._over():
                return self._stop_message()
            return await self._require_fallback().invoke(messages, tools=tools, **kwargs)
        self._act_name = act_name
        return await self._decide()

    async def stream(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AsyncIterator[AssistantMessageChunk]:
        act_name = tool_name(tools, ACT_TOOL)
        if act_name is None and not self._over():
            async for chunk in self._require_fallback().stream(messages, tools=tools, **kwargs):
                yield chunk
            return
        if act_name is not None:
            self._act_name = act_name
        message = self._stop_message() if act_name is None else await self._decide()
        yield AssistantMessageChunk(
            content=message.content, tool_calls=message.tool_calls, finish_reason=message.finish_reason
        )

    def _over(self) -> bool:
        state = self._state
        return self._env.done or state.budget_spent or state.give_up or state.error is not None

    def _stop_message(self) -> AssistantMessage:
        """The turn after the last act, with no ``act`` tool offered: the summary line, no chat model."""
        return self._stop("DONE" if self._env.done else "BLOCKED", "episode over")

    def _require_fallback(self) -> Model:
        if self._fallback is None:
            raise RuntimeError(f"{type(self).__name__}: a turn without the {ACT_TOOL} tool and no fallback model")
        return self._fallback

    async def _decide(self) -> AssistantMessage:
        env, state = self._env, self._state
        if env.done:
            return self._stop("DONE", "environment done")
        # Legacy games keep DONE at their act limit; bounded recovery keeps its failure reason.
        if state.budget_spent and not (state.bounded_recovery and (state.give_up or state.error is not None)):
            return self._stop("BLOCKED" if state.bounded_recovery else "DONE", "act budget spent")
        if state.give_up:
            return self._stop("BLOCKED", "rethink give-up ceiling")
        if state.error is not None:
            return self._stop("BLOCKED", state.error)
        observation = await env.observe()
        candidates = await env.candidates()
        if not candidates:
            return self._stop("BLOCKED", "no candidates")
        offered = {key: text for key, text in candidates.items() if key not in state.blocked} or candidates
        request_state = dict(observation)
        if state.plan:
            request_state["plan"] = state.plan
        if state.notices:
            request_state["harness_notices"] = list(state.notices)
        started = time.perf_counter()
        images = await env.images() if isinstance(env, VisualEnv) else ()
        try:
            decision = await self._decision_model.decide_many(
                Observation(request_state, images), {"pick": ChoiceQuestion(offered, rules=self._rules)}
            )
        except BaseError as exc:  # any decisions failure ends the episode as BLOCKED, recorded
            state.error = f"decision failed: {exc}"
            return self._stop("BLOCKED", state.error)
        choice = decision.choice("pick")
        state.ticks.append(
            {
                "step": len(state.ticks) + 1,
                "key": choice.key,
                "confidence": round(choice.confidence, 3),
                "probabilities": choice.probabilities,
                "ms": round((time.perf_counter() - started) * 1000),
                "input_tokens": decision.usage.input_tokens,
                "output_tokens": decision.usage.output_tokens,
                "usage_known": decision.usage.known,
                "plan": bool(state.plan),
                "blocked": sorted(state.blocked),
                "source": self.name,
                "model": decision.model,
                **decision.provenance,
            }
        )
        state.blocked = set()  # a block, the notices and the plan last one turn
        state.notices = []
        state.plan = ""
        call = ToolCall(
            id=f"{self.name}-{len(state.ticks)}",
            type="function",
            name=self._act_name,
            arguments=json.dumps({"key": choice.key}),
        )
        return AssistantMessage(content="", tool_calls=[call], finish_reason="tool_calls")

    def _stop(self, status: str, reason: str) -> AssistantMessage:
        state = self._state
        summary: dict[str, Any] = {
            "status": status,
            "reason": reason,
            "score": self._env.score,
            "steps": len(state.acts),
        }
        # Bounded stops include failure guidance; legacy game output stays unchanged.
        if state.bounded_recovery and status == "BLOCKED":
            event = next((e for e in reversed(state.rethinks) if e.get("next_action")), None)
            if event is not None:
                next_action = event["next_action"]
                recovery = {
                    "failed": True,
                    "attempt": event.get("attempt"),
                    "termination": event.get("termination"),
                    "phase": event.get("phase"),
                    "reason": event.get("error") or reason,
                    "next_action": next_action,
                }
            else:
                # An act cap can follow a valid plan, so it needs its own stop guidance.
                # Pass the reason through the shared permission guidance too.
                last = next((e for e in reversed(state.rethinks) if e.get("attempt") is not None), None)
                next_action = recovery_next_action(termination=reason, error=reason)
                recovery = {
                    "failed": True,
                    "attempt": None if last is None else last.get("attempt"),
                    "termination": None,
                    "phase": None,
                    "reason": reason,
                    "next_action": next_action,
                }
            summary["next_action"] = next_action
            summary["recovery"] = recovery
        # Keep the full terminal record separately from shortened display output.
        state.terminal = summary
        return AssistantMessage(content=json.dumps(summary, ensure_ascii=False), finish_reason="stop")
