# coding: utf-8
"""The slot model: one act call per decision turn, a stop message at the end, the fallback for other turns."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk

from s1a.decision_models import (
    DecisionModel,
    JevModel,
    Observation,
    Question,
    Reply,
    RandomModel,
    RuleModel,
    ScriptedTransport,
)
from s1a.tool.models import EvalState, ToolDecisionModel, tool_name

TOOLS = [SimpleNamespace(name="observe"), SimpleNamespace(name="act")]
OBSERVE_ONLY = [SimpleNamespace(name="observe")]
RULES = "count to three"
USAGE = {"input_tokens": 315, "output_tokens": 31}


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


class FakeFallback:
    model_client_config = None
    model_config = None

    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        self.calls.append(messages)
        return AssistantMessage(content="fallback")

    async def stream(self, messages: Any, *, tools: Any = None, **kwargs: Any):
        self.calls.append(messages)
        yield AssistantMessageChunk(content="fallback")


def _transport(**kwargs: Any) -> ScriptedTransport:
    return ScriptedTransport(usage=USAGE, latency_ms=5, **kwargs)


def _model(
    env: CountingEnv, state: EvalState, decision_model: DecisionModel, fallback: Any = None
) -> ToolDecisionModel:
    return ToolDecisionModel(env, state, rules=RULES, decision_model=decision_model, fallback=fallback)


def _jev(env: CountingEnv, state: EvalState, transport: ScriptedTransport, fallback: Any = None) -> ToolDecisionModel:
    return _model(env, state, JevModel(transport), fallback)


class TestToolDecisionModelOverJev(IsolatedAsyncioTestCase):
    async def test_decision_turn_returns_one_act_call(self) -> None:
        env, state, transport = CountingEnv(), EvalState(), _transport()
        message = await _jev(env, state, transport).invoke([], tools=TOOLS)
        self.assertEqual(message.finish_reason, "tool_calls")
        (call,) = message.tool_calls
        self.assertEqual((call.name, json.loads(call.arguments)), ("act", {"key": "inc"}))
        self.assertEqual(state.ticks[0]["key"], "inc")
        self.assertEqual(state.ticks[0]["source"], "jev")
        self.assertEqual((state.ticks[0]["input_tokens"], state.ticks[0]["output_tokens"]), (315, 31))
        self.assertEqual(state.ticks[0]["probabilities"], {"inc": 1.0, "noop": 0.0})
        body = transport.bodies[0]
        self.assertEqual(body["state"], {"n": 0})
        self.assertEqual(body["questions"]["pick"]["criteria"], {"inc": "add one", "noop": "do nothing"})
        self.assertEqual(body["questions"]["pick"]["instructions"], {"rules": "count to three"})

    async def test_the_registered_tool_name_is_used(self) -> None:
        self.assertEqual(tool_name([SimpleNamespace(name="evals_act")], "act"), "evals_act")
        self.assertIsNone(tool_name(OBSERVE_ONLY, "act"))
        message = await _jev(CountingEnv(), EvalState(), _transport()).invoke(
            [], tools=[SimpleNamespace(name="evals_act")]
        )
        self.assertEqual(message.tool_calls[0].name, "evals_act")

    async def test_a_blocked_key_is_left_out_for_one_turn(self) -> None:
        env, state, transport = CountingEnv(), EvalState(), _transport()
        state.blocked = {"inc"}
        message = await _jev(env, state, transport).invoke([], tools=TOOLS)
        self.assertEqual(json.loads(message.tool_calls[0].arguments), {"key": "noop"})
        self.assertEqual(list(transport.bodies[0]["questions"]["pick"]["criteria"]), ["noop"])
        self.assertEqual(state.ticks[0]["blocked"], ["inc"])
        self.assertEqual(state.blocked, set())

    async def test_blocking_every_key_offers_all_of_them(self) -> None:
        state = EvalState()
        state.blocked = {"inc", "noop"}
        transport = _transport()
        await _jev(CountingEnv(), state, transport).invoke([], tools=TOOLS)
        self.assertEqual(sorted(transport.bodies[0]["questions"]["pick"]["criteria"]), ["inc", "noop"])

    async def test_plan_and_notices_ride_in_the_state_once(self) -> None:
        state = EvalState(plan="add one twice", notices=["inc repeated"])
        transport = _transport()
        await _jev(CountingEnv(), state, transport).invoke([], tools=TOOLS)
        self.assertEqual(transport.bodies[0]["state"]["plan"], "add one twice")
        self.assertEqual(transport.bodies[0]["state"]["harness_notices"], ["inc repeated"])
        self.assertTrue(state.ticks[0]["plan"])
        self.assertEqual((state.plan, state.notices), ("", []))
        await _jev(CountingEnv(), state, transport).invoke([], tools=TOOLS)
        self.assertFalse(state.ticks[1]["plan"])
        self.assertNotIn("plan", transport.bodies[1]["state"])

    async def test_done_returns_a_stop_message(self) -> None:
        env = CountingEnv()
        env.n = 3
        message = await _jev(env, EvalState(), _transport()).invoke([], tools=TOOLS)
        self.assertEqual(message.finish_reason, "stop")
        self.assertFalse(message.tool_calls)
        summary = json.loads(message.content)
        self.assertEqual((summary["status"], summary["score"]), ("DONE", 3.0))

    async def test_give_up_returns_blocked(self) -> None:
        state = EvalState(give_up=True)
        message = await _jev(CountingEnv(), state, _transport()).invoke([], tools=TOOLS)
        self.assertEqual(json.loads(message.content)["status"], "BLOCKED")

    async def test_a_decisions_failure_ends_the_episode_as_blocked_and_is_the_episodes_error(self) -> None:
        state = EvalState()
        error = build_error(StatusCode.MODEL_CALL_FAILED, error_msg="decisions endpoint returned HTTP 401")
        message = await _jev(CountingEnv(), state, _transport(error=error)).invoke([], tools=TOOLS)
        summary = json.loads(message.content)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertIn("HTTP 401", summary["reason"])
        self.assertTrue(state.error.startswith("decision failed: "))
        self.assertIn("HTTP 401", state.error)

    async def test_an_answer_outside_the_offered_keys_ends_as_blocked(self) -> None:
        state = EvalState()
        message = await _jev(CountingEnv(), state, _transport(choose="dec")).invoke([], tools=TOOLS)
        self.assertEqual(json.loads(message.content)["status"], "BLOCKED")
        self.assertIn("not one of", state.error)

    async def test_other_turns_go_to_the_fallback(self) -> None:
        fallback = FakeFallback()
        message = await _jev(CountingEnv(), EvalState(), _transport(), fallback).invoke(["hi"], tools=OBSERVE_ONLY)
        self.assertEqual((message.content, fallback.calls), ("fallback", [["hi"]]))
        with self.assertRaises(RuntimeError):
            await _jev(CountingEnv(), EvalState(), _transport()).invoke(["hi"], tools=OBSERVE_ONLY)

    async def test_the_turn_after_the_episode_ends_needs_no_chat_model(self) -> None:
        env, fallback = CountingEnv(), FakeFallback()
        env.n = 3
        message = await _jev(env, EvalState(), _transport(), fallback).invoke(["hi"], tools=OBSERVE_ONLY)
        self.assertEqual((message.finish_reason, fallback.calls), ("stop", []))
        self.assertEqual(json.loads(message.content)["status"], "DONE")
        chunks = [chunk async for chunk in _jev(env, EvalState(), _transport()).stream(["hi"], tools=[])]
        self.assertEqual((len(chunks), chunks[0].finish_reason), (1, "stop"))

    async def test_stream_yields_the_act_call_as_one_chunk(self) -> None:
        chunks = [chunk async for chunk in _jev(CountingEnv(), EvalState(), _transport()).stream([], tools=TOOLS)]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].tool_calls[0].name, "act")
        self.assertEqual(chunks[0].finish_reason, "tool_calls")


SERVED_BY = {"checkpoint": "convaiinnovations/laya", "revision": "55cf4c4", "device": "mps", "source": "health"}


class ServedStub(DecisionModel):
    """Answers the way the served model does: an identity in ``model``, the server's facts in ``raw``."""

    name = "laya-served"
    bills_input_tokens = False

    @property
    def model(self) -> str:
        return "convaiinnovations/laya@55cf4c4"

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        answers = {"pick": {"choice": "inc", "confidence": 0.8, "probabilities": {"inc": 0.9, "noop": 0.1}}}
        raw = {"answers": answers, "served_by": SERVED_BY, "url": "http://127.0.0.1:8000", "server_timing": {}}
        return Reply(answers, latency_ms=7, model=self.model, raw=raw)


class TestOtherModels(IsolatedAsyncioTestCase):
    async def test_a_tick_names_the_answering_model_and_where_a_served_answer_came_from(self) -> None:
        state = EvalState()
        await _model(CountingEnv(), state, ServedStub()).invoke([], tools=TOOLS)
        tick = state.ticks[0]
        self.assertEqual((tick["source"], tick["model"]), ("laya-served", "convaiinnovations/laya@55cf4c4"))
        self.assertEqual(
            (tick["served_by"], tick["url"], tick["server_timing"]), (SERVED_BY, "http://127.0.0.1:8000", {})
        )
        self.assertNotIn("answers", tick)
        in_process = EvalState()
        await _model(CountingEnv(), in_process, RandomModel(1)).invoke([], tools=TOOLS)
        self.assertNotIn("served_by", in_process.ticks[0])

    async def test_random_picks_an_offered_key_with_a_uniform_distribution(self) -> None:
        state = EvalState()
        message = await _model(CountingEnv(), state, RandomModel(1)).invoke([], tools=TOOLS)
        self.assertIn(json.loads(message.tool_calls[0].arguments)["key"], {"inc", "noop"})
        self.assertEqual((state.ticks[0]["source"], state.ticks[0]["input_tokens"]), ("random", 0))
        self.assertEqual(
            (state.ticks[0]["probabilities"], state.ticks[0]["confidence"]), ({"inc": 0.5, "noop": 0.5}, 0.0)
        )

    async def test_random_is_reproducible_per_seed_and_not_the_stream_of_random_seeded_with_the_integer(self) -> None:
        async def picks(seed: int) -> list[str]:
            state = EvalState()
            model = _model(CountingEnv(), state, RandomModel(seed))
            for _ in range(20):
                await model.invoke([], tools=TOOLS)
            return [tick["key"] for tick in state.ticks]

        self.assertEqual(await picks(1), await picks(1))
        self.assertNotEqual(await picks(1), await picks(2))
        rng = random.Random(1)
        env_stream = [rng.choice(["inc", "noop"]) for _ in range(20)]
        self.assertNotEqual(await picks(1), env_stream)

    async def test_rule_uses_the_rule_one_hot_and_a_foreign_key_raises(self) -> None:
        state = EvalState()
        model = _model(CountingEnv(), state, RuleModel("always-inc", lambda observation, offered: "inc"))
        message = await model.invoke([], tools=TOOLS)
        self.assertEqual(json.loads(message.tool_calls[0].arguments), {"key": "inc"})
        self.assertEqual((state.ticks[0]["source"], state.ticks[0]["confidence"]), ("always-inc", 1.0))
        self.assertEqual(state.ticks[0]["probabilities"], {"inc": 1.0, "noop": 0.0})
        broken = _model(CountingEnv(), EvalState(), RuleModel("broken", lambda o, c: "dec"))
        with self.assertRaises(RuntimeError):
            await broken.invoke([], tools=TOOLS)
