# coding: utf-8
"""The tool-front loop: jiuwen's DeepAgent over two tools per environment, with a decision model or a chat model in the model slot."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, AsyncIterator
from uuid import uuid4

from openjiuwen.core.foundation.llm import Model
from openjiuwen.core.foundation.tool import Tool, ToolCard
from openjiuwen.core.runner import Runner
from openjiuwen.harness.deep_agent import DeepAgent
from openjiuwen.harness.factory import create_deep_agent

from s1a.decision_models import DecisionModel
from s1a.env import Env, RequiresImages
from s1a.jobs import Episode, now_iso
from s1a.config import HOME
from s1a.counting_model import CountingModel
from s1a.pricing import ChatPrices, cost_usd
from s1a.spec import ToolAgentSpec
from s1a.tool.rethink import RethinkRail
from s1a.tool.models import ACT_TOOL, OBSERVE_TOOL, EvalState, ToolDecisionModel

MODEL_NAMES = ("jev", "clm", "llm", "random", "rule", "laya", "laya-served", "cua")
EVAL_PROMPT = (
    "You play a game through two tools. Call observe first. Then call act with exactly one of the candidate keys the "
    "last tool result offered, one act per turn, until done is true. Then reply with one line: the final score."
)
REPEAT_AFTER = 3
GIVE_UP_AFTER = 3
LLM_ITERATION_HEADROOM = 2  # the chat model spends turns on malformed or unknown keys; its cap is this many budgets
ITERATION_HEADROOM = 2  # observe and the final turn, beyond the act budget
WORKSPACE = HOME / "runs" / "evals"  # the DeepAgent scaffolds SOUL.md, memory/ and friends here, not in the repo root


async def view_of(env: Env, state: EvalState) -> dict[str, Any]:
    """What the model sees after a tool call: the state, the candidate keys, done and the score."""
    done = env.done or state.budget_spent or state.give_up or state.error is not None
    return {
        "state": await env.observe(),
        "candidates": {} if done else await env.candidates(),
        "done": done,
        "score": env.score,
    }


async def snapshot(env: Env, state: EvalState) -> str:
    return json.dumps(await view_of(env, state), ensure_ascii=False)


async def _once(result: str) -> AsyncIterator[str]:
    yield result


class ObserveTool(Tool):
    def __init__(self, env: Env, state: EvalState, *, agent_name: str) -> None:
        super().__init__(
            ToolCard(
                id=f"s1a.{agent_name}.{OBSERVE_TOOL}",
                name=OBSERVE_TOOL,
                description="The current state, the candidate keys, whether the game is done, and the score.",
                input_params={"type": "object", "properties": {}},
                parallel_safe=False,
            )
        )
        self._env = env
        self._state = state

    async def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> str:
        return await snapshot(self._env, self._state)

    async def stream(self, inputs: dict[str, Any], **kwargs: Any) -> AsyncIterator[str]:
        return _once(await self.invoke(inputs, **kwargs))


class ActTool(Tool):
    def __init__(self, env: Env, state: EvalState, *, agent_name: str) -> None:
        super().__init__(
            ToolCard(
                id=f"s1a.{agent_name}.{ACT_TOOL}",
                name=ACT_TOOL,
                description="Play one candidate key. Returns the new state, the candidate keys, done, and the score.",
                input_params={
                    "type": "object",
                    "properties": {"key": {"type": "string", "description": "one of the candidate keys"}},
                    "required": ["key"],
                },
                parallel_safe=False,
            )
        )
        self._env = env
        self._state = state

    async def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> str:
        key = str((inputs or {}).get("key", ""))
        state = self._state
        if state.budget_spent or state.error is not None:
            state.act_calls.append({"key": key, "accepted": False})
            return await snapshot(self._env, state)
        candidates = await self._env.candidates()
        if key not in candidates:
            state.invalid_keys += 1
            state.act_calls.append({"key": key, "accepted": False})
            return json.dumps({"error": f"unknown key {key!r}", "candidates": candidates}, ensure_ascii=False)
        try:
            await self._env.step(key)
        except Exception as exc:
            state.error = f"act failed: {exc}"  # the harness feeds the raise back to the model; every model then stops
            state.act_calls.append({"key": key, "accepted": False})
            raise
        state.act_calls.append({"key": key, "accepted": True})
        state.acts.append({"key": key, "score": self._env.score, "done": self._env.done})
        view = await view_of(self._env, self._state)  # after the act is counted: the budget may be spent now
        self._state.views.append(view)
        return json.dumps(view, ensure_ascii=False)

    async def stream(self, inputs: dict[str, Any], **kwargs: Any) -> AsyncIterator[str]:
        return _once(await self.invoke(inputs, **kwargs))


def build_env_tools(env: Env, state: EvalState, *, agent_name: str) -> list[Tool | ToolCard]:
    return [ObserveTool(env, state, agent_name=agent_name), ActTool(env, state, agent_name=agent_name)]


def build_slot_model(
    model_name: str,
    env: Env,
    state: EvalState,
    *,
    rules: str,
    chat: Model | None,
    decision_model: DecisionModel | None,
) -> Model:
    """The chat model for ``llm``; the one slot model over the decision model for every other name."""
    match model_name:
        case "llm":
            if chat is None:
                raise RuntimeError("--model llm needs the chat model: OPENAI_API_KEY or LLM_API_KEY, and MODEL_NAME")
            return chat
        case "jev" | "clm" | "laya" | "laya-served" | "cua" | "random" | "rule":
            if decision_model is None:
                raise RuntimeError(f"--model {model_name} needs a decision model")
            return ToolDecisionModel(env, state, rules=rules, decision_model=decision_model, fallback=chat)
        case _:
            raise ValueError(f"unknown model {model_name!r}; one of {MODEL_NAMES}")


def create_eval_agent(
    spec: ToolAgentSpec, env: Env, model: Model, state: EvalState, *, rethink: RethinkRail | None, max_iterations: int
) -> DeepAgent:
    """The one assembly path: ``create_deep_agent`` with the two env tools and, when asked, the rethink rail."""
    workspace = WORKSPACE / spec.name
    workspace.mkdir(parents=True, exist_ok=True)
    # The anomaly rail stays off: its bailout ends a run on the third identical call, and act(LEFT) three times is
    # a legal 2048 line. Exact repeats are RethinkRail's first layer instead. The image probe stays off: the games
    # read no images, and the probe would spend one chat call per episode on the fallback model.
    return create_deep_agent(
        model,
        system_prompt=f"{EVAL_PROMPT}\n\nRules: {spec.rules}",
        tools=build_env_tools(env, state, agent_name=spec.name),
        rails=[rethink] if rethink is not None else None,
        max_iterations=max_iterations,
        workspace=str(workspace),
        language="en",
        parallel_tool_calls=False,
        enable_model_anomaly_detection_rail=False,
        enable_read_image_multimodal=False,
        enable_sys_operation=False,  # the games own no filesystem, shell or code tools
    )


def llm_decisions(calls: list[dict[str, Any]], act_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The chat model's decisions: every ``act`` tool call it wrote, in the ticks' shape, ``accepted`` per the act tool.

    A reply with several act calls splits its latency and tokens evenly; a call the harness never invoked is not accepted."""
    ticks: list[dict[str, Any]] = []
    pending = list(act_calls)
    for call in calls:
        acts = [
            arguments
            for name, arguments in zip(call["tool_calls"], call["tool_args"])
            if name == ACT_TOOL or name.endswith("_" + ACT_TOOL)
        ]
        for arguments in acts:
            try:
                key = str(json.loads(arguments).get("key", ""))
            except (ValueError, AttributeError):
                key = ""
            accepted = False
            if pending and pending[0]["key"] == key:
                accepted = bool(pending.pop(0)["accepted"])
            ticks.append(
                {
                    "step": len(ticks) + 1,
                    "key": key,
                    "confidence": 0.0,
                    "probabilities": {},
                    "ms": round(call["ms"] / len(acts)),
                    "input_tokens": round(call["input_tokens"] / len(acts)),
                    "output_tokens": round(call["output_tokens"] / len(acts)),
                    "plan": False,
                    "blocked": [],
                    "source": "llm",
                    "accepted": accepted,
                }
            )
    return ticks


async def run_episode(
    spec: ToolAgentSpec,
    env: Env,
    *,
    model_name: str,
    seed: int,
    chat: Model | None,
    decision_model: DecisionModel | None,
    rethink_on: bool,
    max_acts: int,
    timeout_s: float,
    prices: ChatPrices | None,
    log: bool,
) -> Episode:
    """One episode through the agent: reset, one conversation, the ticks, rethinks, tokens and dollars into the Episode.

    ``max_acts`` bounds the acts for every model; ``timeout_s`` bounds the wall clock, and a timed-out episode
    keeps its score so far with ``result_type: timeout``."""
    if isinstance(env, RequiresImages) and env.requires_images() and model_name != "random":
        # A ``RequiresImages`` environment refuses a text-only backend up front — deciding 
        # over an observation whose images were stripped would measure a blind run (``random``
        # never reads the observation and stays the offline smoke)
        if decision_model is None or not decision_model.supports_images:
            raise ValueError(
                f"{spec.name}: this environment decides over its images; --model {model_name} reads text only"
            )
    state = EvalState(max_acts=max_acts)
    await env.reset()
    first_view = await view_of(env, state)
    counted = CountingModel(chat, state.chat) if chat is not None else None
    model = build_slot_model(model_name, env, state, rules=spec.rules, chat=counted, decision_model=decision_model)
    rail = None
    if rethink_on and spec.budget.stall_after > 0 and model_name != "llm":  # the chat model reads no plan or block
        rail = RethinkRail(
            state,
            rules=spec.rules,
            initial_score=env.score,
            planner=counted,
            stall_after=spec.budget.stall_after,
            repeat_after=REPEAT_AFTER,
            give_up_after=GIVE_UP_AFTER,
        )
    cap = (max_acts + ITERATION_HEADROOM) * (LLM_ITERATION_HEADROOM if model_name == "llm" else 1)
    agent = create_eval_agent(spec, env, model, state, rethink=rail, max_iterations=cap)
    started_at = now_iso()
    started = time.perf_counter()
    await agent.ensure_initialized()
    conversation_id = f"evals-{spec.name}-{seed}-{uuid4().hex[:6]}"
    try:
        result = await asyncio.wait_for(
            Runner.run_agent(agent, {"query": "Play the game.", "conversation_id": conversation_id}), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        result = {"result_type": "timeout", "output": f"episode stopped after {timeout_s:.0f} s"}
    finally:
        await agent.cleanup_task_resources()
        agent.ability_manager.teardown_tools()  # the act and observe cards hold the env and the state
        await Runner.release(conversation_id)
    if model_name == "llm":
        state.ticks = llm_decisions(state.chat, state.act_calls)
    if log:
        for tick in state.ticks:
            print(
                f"  {tick['step']:>3} {tick['key']:<22} conf={tick['confidence']:.2f} {tick['ms']:>5} ms",
                file=sys.stderr,
            )
        for event in state.rethinks:
            print(f"  rethink {event}", file=sys.stderr)
    policy = model.name if isinstance(model, ToolDecisionModel) else "llm"
    jev_input_tokens = (
        sum(tick["input_tokens"] for tick in state.ticks if tick["source"] != "llm")
        if isinstance(model, ToolDecisionModel) and model.bills_input_tokens
        else 0
    )
    chat_input_tokens = sum(call["input_tokens"] for call in state.chat)
    chat_output_tokens = sum(call["output_tokens"] for call in state.chat)
    chat_cache_tokens = sum(call["cache_tokens"] for call in state.chat)
    return Episode(
        env=spec.name,
        policy=policy,
        seed=seed,
        score=env.score,
        steps=len(state.acts),
        elapsed_s=round(time.perf_counter() - started, 1),
        started_at=started_at,
        finished_at=now_iso(),
        final_state=await env.observe(),
        chat_calls=len(state.chat),
        chat_input_tokens=chat_input_tokens,
        chat_output_tokens=chat_output_tokens,
        chat_cache_tokens=chat_cache_tokens,
        jev_input_tokens=jev_input_tokens,
        invalid_keys=state.invalid_keys,
        cost_usd=cost_usd(jev_input_tokens, chat_input_tokens, chat_output_tokens, chat_cache_tokens, prices),
        error=state.error,
        decisions=state.ticks,
        views=[first_view, *state.views],
        extra={
            "rethink": rail is not None,
            "rethinks": state.rethinks,
            "result_type": result.get("result_type"),
            "output": str(result.get("output") or "")[:300],
        },
    )
