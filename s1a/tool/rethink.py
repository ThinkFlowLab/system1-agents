# coding: utf-8
"""RethinkRail: after every ``act``, notice a stalled episode and hand the model a plan and a blocked key."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.llm import Model
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext
from openjiuwen.harness.rails.base import DeepAgentRail

from s1a.recovery import RecoveryBudget, RecoveryExhausted, RecoveryLimits, recovery_next_action
from s1a.tool.models import ACT_TOOL, EvalState

REPLAN_PROMPT = (
    "The player is stuck. From the rules, the recent steps, the current state and the candidates, write a plan of "
    "at most 60 words for the next few moves. Name concrete candidate keys."
)
REPLAN_MAX_TOKENS = 120
RECENT_STEPS = 12


def parse_tool_result(raw: Any) -> dict[str, Any] | None:
    """The ``act`` tool's JSON result, from the string or object the callback hands over."""
    text = raw if isinstance(raw, str) else getattr(raw, "content", None)
    if isinstance(raw, dict):
        return raw
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def progress_digest(state: Any) -> str:
    """What "the same state" means for a stall: the observation's ``progress`` part when the adapter names one
    (histories and step counters change every step), otherwise the whole observation."""
    if isinstance(state, dict) and "progress" in state:
        state = state["progress"]
    return json.dumps(state, sort_keys=True, default=str)


def parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def draft_plan(
    planner: Model,
    *,
    rules: str,
    recent_steps: list[dict[str, Any]],
    state: Any,
    candidates: Any,
) -> str:
    """The replan request as a standalone function: rules, recent steps, a fresh state and its candidates to a plan.

    RethinkRail and the bounded branch both call it; the browser front will call the same function later.
    """
    context = {
        "rules": rules,
        "recent_steps": recent_steps,
        "state": state,
        "candidates": candidates,
    }
    reply = await planner.invoke(
        [
            {"role": "system", "content": REPLAN_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
        ],
        max_tokens=REPLAN_MAX_TOKENS,
        temperature=0,
    )
    return str(reply.content or "").strip()


class RethinkRail(DeepAgentRail):
    """Three signals, cheapest first: an exact repeat blocks a key, a semantic stall asks for a plan, too many stalls give up.

    ``repeat_after``: the same key for that many consecutive acts without a score change blocks it for one turn.
    ``stall_after``: that many acts without a score change, with a state seen before in that window, trigger a plan.
    ``give_up_after``: a stall after that many plans, in the same state and without a score change, ends the episode
    (a fourth stall after three plans without a score change ends the episode).
    A plan the chat model fails to write is the episode's error. The plan and the block last one turn.

    ``refresh`` plus ``limits`` turn on the bounded branch instead: a stall starts one attempt, re-reads the window
    through ``refresh`` (a done window or delayed progress skips the plan), then writes the plan under the remaining
    ``RecoveryLimits`` budget. A refresh or planner failure is the episode's error; the last stall past the attempts
    gives up. The plan only rides into the next turn's observation, never clicks. Without both the game branch above
    runs unchanged. The harness registers the overridden ``after_tool_call`` hook by itself.
    """

    def __init__(
        self,
        state: EvalState,
        *,
        rules: str,
        initial_score: float,
        planner: Model | None,
        stall_after: int,
        repeat_after: int,
        give_up_after: int,
        refresh: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        limits: RecoveryLimits | None = None,
    ) -> None:
        super().__init__()
        self._state = state
        self._rules = rules
        self._planner = planner
        self._stall_after = stall_after
        self._repeat_after = repeat_after
        self._give_up_after = give_up_after
        self._keys: list[str] = []
        self._scores: list[float] = []
        self._digests: list[str] = []
        self._last_score = initial_score
        self._since_progress = 0
        self._stalls_without_progress = 0
        self._stall_digest = ""  # the state of the last stall; a stall elsewhere restarts the give-up count
        # Both a refresh callback and limits turn the bounded branch on; one alone is a contract error, never a
        # silent fall back to the unbounded game branch. Neither keeps the game branch exactly as before.
        if (refresh is None) != (limits is None):
            raise ValueError("bounded rethink needs refresh and limits together; neither turns on the game branch")
        self._refresh = refresh
        self._recovery = RecoveryBudget(limits) if limits is not None else None
        if self._recovery is not None:
            state.bounded_recovery = True

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = ctx.inputs
        name = str(getattr(inputs, "tool_name", "") or "")
        if not (name == ACT_TOOL or name.endswith("_" + ACT_TOOL)):
            return
        result = parse_tool_result(getattr(inputs, "tool_result", None))
        if result is None or "score" not in result or self._state.error is not None:
            return
        key = str(parse_tool_args(getattr(inputs, "tool_args", None)).get("key", ""))
        if self._recovery is not None:
            if result.get("done") or self._state.budget_spent or self._state.give_up:
                return  # the episode is over: never replan
            await self._bounded_after(key, result)
            return
        score = float(result["score"])
        progressed = score != self._last_score
        self._last_score = score
        self._keys.append(key)
        self._scores.append(score)
        self._digests.append(progress_digest(result.get("state")))
        if progressed:
            self._since_progress = 0
            self._stalls_without_progress = 0
            return
        self._since_progress += 1
        step = len(self._keys)
        state = self._state
        tail = self._keys[-self._repeat_after :]
        repeated = self._repeat_after and self._since_progress >= self._repeat_after and len(set(tail)) == 1
        if repeated:
            state.blocked.add(key)
            state.notices.append(f"{key} repeated {self._repeat_after} times without a score change")
            state.rethinks.append({"kind": "repeat", "step": step, "key": key})
        window = self._digests[-self._stall_after :]
        stalled = self._stall_after > 0 and self._since_progress >= self._stall_after and window.count(window[-1]) > 1
        if not stalled:
            return
        if window[-1] != self._stall_digest:
            self._stalls_without_progress = 0
            self._stall_digest = window[-1]
        self._stalls_without_progress += 1
        if self._stalls_without_progress > self._give_up_after:
            state.give_up = True
            state.rethinks.append({"kind": "give_up", "step": step})
            return
        most_repeated = Counter(self._keys[-self._stall_after :]).most_common(1)[0][0]
        since_progress, self._since_progress = self._since_progress, 0
        try:
            plan = await self._plan(self._planner, result) if self._planner is not None else ""
        except BaseError as exc:  # the callback framework swallows a raise; the error field is what stops the episode
            state.error = f"plan failed: {exc}"
            state.rethinks.append({"kind": "plan_failed", "step": step, "error": str(exc)})
            return
        state.plan = plan
        state.blocked.add(most_repeated)
        state.notices.append(f"no score change for {since_progress} steps; {most_repeated} blocked this turn")
        state.rethinks.append({"kind": "stall", "step": step, "blocked": most_repeated, "plan": plan})

    async def _bounded_after(self, key: str, result: dict[str, Any]) -> None:
        """The bounded branch: a stall starts an attempt, a refresh, then a plan for the next turn or a stop."""
        state = self._state
        score = float(result["score"])
        progressed = score != self._last_score
        self._last_score = score
        self._keys.append(key)
        self._scores.append(score)
        self._digests.append(progress_digest(result.get("state")))
        if progressed:
            self._since_progress = 0  # progress clears the window; it never returns attempts or seconds
            return
        self._since_progress += 1
        step = len(self._keys)
        # A bounded repeat is a real no-op: the same key and an unchanged window. The score never moves on desktop,
        # so a click whose display changed is progress even when the key repeats, and must not be blocked.
        tail = self._keys[-self._repeat_after :]
        tail_digests = self._digests[-self._repeat_after :]
        if (
            self._repeat_after
            and self._since_progress >= self._repeat_after
            and len(set(tail)) == 1
            and len(set(tail_digests)) == 1
        ):
            state.blocked.add(key)
            state.notices.append(f"{key} repeated {self._repeat_after} times without a window change")
            state.rethinks.append({"kind": "repeat", "step": step, "key": key})
        window = self._digests[-self._stall_after :]
        stalled = self._stall_after > 0 and self._since_progress >= self._stall_after and window.count(window[-1]) > 1
        if stalled:
            await self._recover(step, key)

    async def _recover(self, step: int, key: str) -> None:
        """One stall, one attempt: record the event first, then refresh and plan, updating it as each phase ends.

        The event is appended before any await, so an external cancellation or timeout still leaves its trigger, its
        recent actions and the phase it died in. ``spent_s`` is refreshed in one ``finally``; a cancellation is never
        swallowed. An effective plan resets only the detection window, never the attempts or the active seconds.
        """
        state, budget, refresh = self._state, self._recovery, self._refresh
        assert budget is not None and refresh is not None
        recent = [
            {"key": recent_key, "score": score}
            for recent_key, score in zip(self._keys[-RECENT_STEPS:], self._scores[-RECENT_STEPS:])
        ]
        event: dict[str, Any] = {
            "kind": "stall",
            "step": step,
            "trigger": key,
            "trigger_reason": "stalled",
            "recent_actions": recent,
            "fresh_obs": None,
            "plan": "",
            "attempt": budget.attempts,
            "phase": None,
            "spent_s": round(budget.spent_s, 3),
            "termination": None,
        }
        state.rethinks.append(event)  # before the awaits: an external cancel must not lose the stall
        try:
            budget.begin_attempt()  # a stall starts one attempt; a spent budget gives up, never a success
        except RecoveryExhausted as exc:
            state.give_up = True
            event.update(
                termination="give_up",
                error=str(exc),
                next_action=recovery_next_action(termination="give_up", error=str(exc)),
            )
            return
        event["attempt"] = budget.attempts
        try:
            event["phase"] = "refresh"
            fresh = await budget.call(refresh())
            event["fresh_obs"] = fresh
            if fresh.get("done"):
                event["termination"] = "done"
                return
            fresh_digest = progress_digest(fresh.get("state"))
            if fresh_digest != self._digests[-1]:
                self._digests.append(fresh_digest)  # the window moved after all: delayed progress, no plan
                self._since_progress = 0
                event["termination"] = "delayed_progress"
                return
            if self._planner is None:
                event["termination"] = "no_planner"
                return
            event["phase"] = "planner"
            plan = await budget.call(self._draft(recent, fresh))
            if not (plan or "").strip():
                # A blank answer is a planner failure, not a plan: the next turn would get no guidance and the event
                # would read as planned. ``draft_plan`` strips, so None/whitespace arrives here as "".
                raise ValueError("planner returned an empty plan")
        except asyncio.CancelledError:
            event["termination"] = "cancelled"
            event["next_action"] = recovery_next_action(
                termination="cancelled", stage=event["phase"], error="recovery cancelled"
            )
            raise  # the harness cancelled the episode: record it and keep unwinding
        except asyncio.TimeoutError:
            state.error = f"rethink {event['phase']} timed out"
            event.update(
                termination="error",
                error=state.error,
                next_action=recovery_next_action(termination="timeout", stage=event["phase"], error=state.error),
            )
            return
        except BaseError as exc:  # the callback framework swallows a raise; the error field stops the episode
            prefix = "plan failed" if event["phase"] == "planner" else f"rethink {event['phase']} failed"
            state.error = f"{prefix}: {exc}"
            event.update(
                termination="error",
                error=str(exc),
                next_action=recovery_next_action(termination="error", stage=event["phase"], error=str(exc)),
            )
            return
        except Exception as exc:
            state.error = f"rethink {event['phase']} failed: {exc}"
            event.update(
                termination="error",
                error=str(exc),
                next_action=recovery_next_action(termination="error", stage=event["phase"], error=str(exc)),
            )
            return
        finally:
            event["spent_s"] = round(budget.spent_s, 3)
        event.update(plan=plan, termination="planned")
        state.plan = plan  # the next observation offers the candidates; the plan never executes a click itself
        self._since_progress = 0  # a fresh plan gets stall_after new actions before the next stall is judged

    def _draft(self, recent: list[dict[str, Any]], fresh: dict[str, Any]) -> Awaitable[str]:
        assert self._planner is not None
        return draft_plan(
            self._planner,
            rules=self._rules,
            recent_steps=recent,
            state=fresh.get("state"),
            candidates=fresh.get("candidates"),
        )

    async def _plan(self, planner: Model, result: dict[str, Any]) -> str:
        return await draft_plan(
            planner,
            rules=self._rules,
            recent_steps=[
                {"key": key, "score": score}
                for key, score in zip(self._keys[-RECENT_STEPS:], self._scores[-RECENT_STEPS:])
            ],
            state=result.get("state"),
            candidates=result.get("candidates"),
        )
