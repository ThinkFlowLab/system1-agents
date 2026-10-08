# coding: utf-8
"""The eval agent's two tools, the slot-model factory, and one episode through the DeepAgent offline."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk, Model, ToolCall
from openjiuwen.core.runner import Runner
from openjiuwen.core.session.checkpointer import CheckpointerFactory

from s1a.decision_models import (
    DecisionModel,
    JevModel,
    RandomModel,
    RuleModel,
    ScriptedModel,
    ScriptedTransport,
    Usage,
)
from s1a.spec import Budget, ToolAgentSpec
from s1a.tool import loop as agent
from s1a.tool.loop import (
    ActTool,
    ObserveTool,
    build_env_tools,
    build_slot_model,
    llm_decisions,
    run_episode,
    view_of,
)
from s1a.tool.models import EvalState, ToolDecisionModel, placeholder_model, tool_name

RULES = "count to three"
ALWAYS_INC = RuleModel("always-inc", lambda observation, offered: "inc")


class CountingEnv:
    def __init__(self) -> None:
        self.n = 0

    async def reset(self) -> None:
        self.n = 0

    async def observe(self) -> dict[str, Any]:
        return {"n": self.n}

    async def candidates(self) -> dict[str, str]:
        return {} if self.done else {"inc": "add one", "noop": "do nothing"}

    async def step(self, key: str) -> None:
        self.n += key == "inc"

    @property
    def done(self) -> bool:
        return self.n >= 3

    @property
    def score(self) -> float:
        return float(self.n)


class SlowEnv(CountingEnv):
    async def step(self, key: str) -> None:
        await asyncio.sleep(0.3)
        await super().step(key)


class DyingEnv(CountingEnv):
    async def step(self, key: str) -> None:
        raise RuntimeError("playwright died")


class ScriptedChatModel(Model):
    """The chat model offline: one act call per decision turn from a script of keys, then a final line."""

    def __init__(self, keys: list[str]) -> None:
        source = placeholder_model()
        super().__init__(source.model_client_config, source.model_config)
        self._keys = list(keys)

    def _answer(self, tools: Any) -> AssistantMessage:
        act = tool_name(tools, "act")
        if act is None or not self._keys:
            return AssistantMessage(content="score", finish_reason="stop")
        call = ToolCall(
            id=f"c{len(self._keys)}", type="function", name=act, arguments=json.dumps({"key": self._keys.pop(0)})
        )
        return AssistantMessage(content="", tool_calls=[call], finish_reason="tool_calls")

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        return self._answer(tools)

    async def stream(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AsyncIterator[AssistantMessageChunk]:
        message = self._answer(tools)
        yield AssistantMessageChunk(
            content=message.content, tool_calls=message.tool_calls, finish_reason=message.finish_reason
        )


SPEC = ToolAgentSpec(
    name="counter",
    description="Count to three.",
    rules=RULES,
    budget=Budget(max_steps=10, timeout_s=60, stall_after=0),
    flags=lambda parser: None,
    series=lambda flags: None,  # run_episode never builds a series
)


class TestEnvTools(IsolatedAsyncioTestCase):
    async def test_observe_returns_the_snapshot(self) -> None:
        snapshot = json.loads(await ObserveTool(CountingEnv(), EvalState(), agent_name="counter").invoke({}))
        self.assertEqual(
            snapshot,
            {"state": {"n": 0}, "candidates": {"inc": "add one", "noop": "do nothing"}, "done": False, "score": 0.0},
        )

    async def test_act_steps_and_records(self) -> None:
        env, state = CountingEnv(), EvalState()
        snapshot = json.loads(await ActTool(env, state, agent_name="counter").invoke({"key": "inc"}))
        self.assertEqual((env.n, snapshot["state"], snapshot["score"]), (1, {"n": 1}, 1.0))
        self.assertEqual(state.acts, [{"key": "inc", "score": 1.0, "done": False}])
        self.assertEqual(state.views, [snapshot])  # the view the next decision sees, recorded as returned

    async def test_act_rejects_an_unknown_key_without_stepping(self) -> None:
        env, state = CountingEnv(), EvalState()
        reply = json.loads(await ActTool(env, state, agent_name="counter").invoke({"key": "dec"}))
        self.assertIn("unknown key", reply["error"])
        self.assertEqual(reply["candidates"], {"inc": "add one", "noop": "do nothing"})
        self.assertEqual((env.n, state.acts, state.invalid_keys), (0, [], 1))

    async def test_act_refuses_a_key_once_the_budget_is_spent(self) -> None:
        env, state = CountingEnv(), EvalState(max_acts=2)
        tool = ActTool(env, state, agent_name="counter")
        await tool.invoke({"key": "inc"})
        second = json.loads(await tool.invoke({"key": "inc"}))
        self.assertEqual((second["done"], second["candidates"], env.n), (True, {}, 2))
        third = json.loads(await tool.invoke({"key": "inc"}))
        self.assertEqual((third["done"], env.n, len(state.acts)), (True, 2, 2))

    async def test_a_given_up_episode_is_done_with_no_candidates(self) -> None:
        view = await view_of(CountingEnv(), EvalState(give_up=True))
        self.assertEqual((view["done"], view["candidates"]), (True, {}))

    def test_tool_names(self) -> None:
        tools = build_env_tools(CountingEnv(), EvalState(), agent_name="counter")
        self.assertEqual([tool.card.name for tool in tools], ["observe", "act"])

    def test_llm_decisions_are_the_chat_calls_that_acted(self) -> None:
        calls = [
            {"ms": 900, "input_tokens": 700, "output_tokens": 12, "tool_calls": ["observe"], "tool_args": ["{}"]},
            {
                "ms": 1200,
                "input_tokens": 900,
                "output_tokens": 14,
                "tool_calls": ["act"],
                "tool_args": ['{"key": "inc"}'],
            },
            {
                "ms": 1100,
                "input_tokens": 950,
                "output_tokens": 10,
                "tool_calls": ["evals_act"],
                "tool_args": ['{"key": "noop"}'],
            },
            {"ms": 300, "input_tokens": 100, "output_tokens": 8, "tool_calls": [], "tool_args": []},
        ]
        ticks = llm_decisions(calls, [{"key": "inc", "accepted": True}, {"key": "noop", "accepted": False}])
        self.assertEqual(
            [(t["step"], t["key"], t["ms"], t["input_tokens"], t["source"], t["accepted"]) for t in ticks],
            [(1, "inc", 1200, 900, "llm", True), (2, "noop", 1100, 950, "llm", False)],
        )

    def test_llm_ticks_split_one_reply_with_two_act_calls_and_survive_an_uninvoked_call(self) -> None:
        calls = [
            {
                "ms": 1000,
                "input_tokens": 800,
                "output_tokens": 20,
                "tool_calls": ["act", "act"],
                "tool_args": ['{"key": "inc"}', '{"key": "inc"}'],
            },
            {"ms": 700, "input_tokens": 600, "output_tokens": 9, "tool_calls": ["act"], "tool_args": ["not json"]},
        ]
        ticks = llm_decisions(calls, [{"key": "inc", "accepted": True}, {"key": "inc", "accepted": True}])
        self.assertEqual(
            [(t["ms"], t["input_tokens"], t["accepted"]) for t in ticks],
            [(500, 400, True), (500, 400, True), (700, 600, False)],
        )


class TestSlotFactory(TestCase):
    def test_every_name_but_llm_is_the_one_slot_model_and_the_errors(self) -> None:
        env, state = CountingEnv(), EvalState()
        chance = build_slot_model("random", env, state, rules=RULES, chat=None, decision_model=RandomModel(0))
        self.assertIsInstance(chance, ToolDecisionModel)
        self.assertEqual(chance.name, "random")
        rule = build_slot_model("rule", env, state, rules=RULES, chat=None, decision_model=ALWAYS_INC)
        self.assertIsInstance(rule, ToolDecisionModel)
        self.assertEqual(rule.name, "always-inc")
        for model_name, error in (
            ("llm", RuntimeError),
            ("rule", RuntimeError),
            ("jev", RuntimeError),
            ("laya", RuntimeError),
            ("oracle", ValueError),
        ):
            with self.assertRaises(error):
                build_slot_model(model_name, env, state, rules=RULES, chat=None, decision_model=None)


async def _play(
    env: CountingEnv,
    *,
    max_acts: int,
    timeout_s: float,
    model_name: str,
    decision_model: DecisionModel | None,
    chat: Model | None = None,
    rethink_on: bool = False,
) -> Any:
    with tempfile.TemporaryDirectory() as tmp, patch.object(agent, "WORKSPACE", Path(tmp)):
        await Runner.start()
        try:
            return await run_episode(
                SPEC,
                env,
                model_name=model_name,
                seed=0,
                chat=chat,
                decision_model=decision_model,
                rethink_on=rethink_on,
                max_acts=max_acts,
                timeout_s=timeout_s,
                prices=None,
                log=False,
            )
        finally:
            await Runner.stop()


def _refusing() -> JevModel:
    error = build_error(StatusCode.MODEL_CALL_FAILED, error_msg="decisions endpoint returned HTTP 401")
    return JevModel(ScriptedTransport(error=error))


class TestEpisodeThroughTheAgent(IsolatedAsyncioTestCase):
    """Episodes through ``create_deep_agent`` and the Runner, offline: a rule in the slot, no chat model."""

    async def test_a_priced_backend_counts_its_tokens_without_relying_on_its_name(self) -> None:
        decision_model = ScriptedModel(choose="inc", usage=Usage(input_tokens=300))
        decision_model.bills_input_tokens = True
        episode = await _play(
            CountingEnv(), max_acts=1, timeout_s=60.0, model_name="random", decision_model=decision_model
        )
        self.assertEqual((episode.decisions[0]["source"], episode.decisions[0]["input_tokens"]), ("scripted", 300))
        self.assertEqual((episode.jev_input_tokens, episode.cost_usd), (300, 0.000013))

    async def test_a_refused_decision_is_the_episodes_error_with_no_decisions(self) -> None:
        episode = await _play(CountingEnv(), max_acts=10, timeout_s=60.0, model_name="jev", decision_model=_refusing())
        self.assertTrue(episode.error.startswith("decision failed: "), episode.error)
        self.assertIn("decisions endpoint returned HTTP 401", episode.error)
        self.assertEqual((episode.decisions, episode.steps, episode.score), ([], 0, 0.0))
        self.assertIn("BLOCKED", episode.extra["output"])

    async def test_rule_plays_to_the_end_and_every_act_is_a_recorded_decision(self) -> None:
        episode = await _play(CountingEnv(), max_acts=10, timeout_s=60.0, model_name="rule", decision_model=ALWAYS_INC)
        self.assertEqual((episode.policy, episode.score, episode.steps), ("always-inc", 3.0, 3))
        self.assertEqual([decision["key"] for decision in episode.decisions], ["inc", "inc", "inc"])
        self.assertEqual(episode.final_state, {"n": 3})
        self.assertEqual(episode.extra["result_type"], "answer")
        self.assertEqual((episode.chat_calls, episode.jev_input_tokens, episode.invalid_keys), (0, 0, 0))
        self.assertEqual(episode.cost_usd, 0.0)
        # One view per decision plus the final one; view i is what act i chose from.
        self.assertEqual(len(episode.views), episode.steps + 1)
        self.assertEqual([view["state"] for view in episode.views], [{"n": 0}, {"n": 1}, {"n": 2}, {"n": 3}])
        for view, decision in zip(episode.views, episode.decisions):
            self.assertIn(decision["key"], view["candidates"])
        self.assertEqual(episode.views[-1], {"state": {"n": 3}, "candidates": {}, "done": True, "score": 3.0})
        self.assertIsNone(episode.frames_dir)

    async def test_two_random_runs_with_the_same_seed_pick_the_same_keys(self) -> None:
        first = await _play(
            CountingEnv(), max_acts=10, timeout_s=60.0, model_name="random", decision_model=RandomModel(7)
        )
        second = await _play(
            CountingEnv(), max_acts=10, timeout_s=60.0, model_name="random", decision_model=RandomModel(7)
        )
        self.assertEqual([d["key"] for d in first.decisions], [d["key"] for d in second.decisions])
        self.assertEqual((first.policy, first.decisions[0]["probabilities"]), ("random", {"inc": 0.5, "noop": 0.5}))

    async def test_the_act_budget_stops_the_agent_before_the_game_ends(self) -> None:
        episode = await _play(CountingEnv(), max_acts=2, timeout_s=60.0, model_name="rule", decision_model=ALWAYS_INC)
        self.assertEqual((episode.score, episode.steps), (2.0, 2))
        self.assertEqual(episode.extra["result_type"], "answer")

    async def test_an_env_failure_inside_act_ends_the_episode_as_an_error(self) -> None:
        episode = await _play(DyingEnv(), max_acts=5, timeout_s=60.0, model_name="rule", decision_model=ALWAYS_INC)
        self.assertEqual(episode.error, "act failed: playwright died")
        self.assertEqual((episode.steps, len(episode.decisions), episode.score), (0, 1, 0.0))
        self.assertIn("BLOCKED", episode.extra["output"])

    async def test_llm_marks_rejected_keys_and_runs_without_the_rail(self) -> None:
        chat = ScriptedChatModel(["dec", "inc", "inc", "inc"])
        episode = await _play(
            CountingEnv(),
            max_acts=10,
            timeout_s=60.0,
            model_name="llm",
            decision_model=None,
            chat=chat,
            rethink_on=True,
        )
        self.assertEqual((episode.policy, episode.score, episode.steps, episode.invalid_keys), ("llm", 3.0, 3, 1))
        self.assertEqual(
            [(t["key"], t["accepted"]) for t in episode.decisions],
            [("dec", False), ("inc", True), ("inc", True), ("inc", True)],
        )
        self.assertEqual(len(episode.views), episode.steps + 1)
        self.assertEqual((episode.extra["rethink"], episode.chat_cache_tokens), (False, 0))

    async def test_an_episode_leaves_no_system_tools_or_conversation_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(agent, "WORKSPACE", Path(tmp)):
            await Runner.start()
            try:
                for _ in range(2):
                    await run_episode(
                        SPEC,
                        CountingEnv(),
                        model_name="rule",
                        seed=0,
                        chat=None,
                        decision_model=ALWAYS_INC,
                        rethink_on=False,
                        max_acts=10,
                        timeout_s=60.0,
                        prices=None,
                        log=False,
                    )
                cards = list(Runner.resource_mgr._id_to_card)
                stores = list(CheckpointerFactory.get_checkpointer()._agent_stores)
            finally:
                await Runner.stop()
        self.assertEqual([card for card in cards if ".fs." in card or ".shell." in card or ".code." in card], [])
        self.assertEqual([card for card in cards if card.startswith(("act_", "observe_"))], [])
        self.assertEqual([store for store in stores if store.startswith("evals-counter")], [])

    async def test_a_slow_episode_stops_at_the_timeout_and_keeps_its_score(self) -> None:
        episode = await _play(SlowEnv(), max_acts=10, timeout_s=0.5, model_name="rule", decision_model=ALWAYS_INC)
        self.assertEqual(episode.extra["result_type"], "timeout")
        self.assertLess(episode.score, 3.0)
        self.assertIn(len(episode.decisions) - episode.steps, (0, 1), "the decision in flight at the cut has no act")
