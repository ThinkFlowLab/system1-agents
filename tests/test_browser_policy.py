# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Modifications Copyright 2026 ThinkFlowLab
# SPDX-License-Identifier: Apache-2.0

"""The browser policy: action space and question shape, the decision through the decision-model layer, and the
decision-model turns. The wire is scripted under ``JevModel`` so every request body stays asserted."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase, mock

import httpx

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError, build_error
from openjiuwen.core.foundation.llm import Model, ModelClientConfig
from openjiuwen.core.foundation.llm.model_clients.base_model_client import BaseModelClient
from openjiuwen.core.foundation.tool.schema import ToolInfo
from openjiuwen.harness.schema.decision_policy import DecisionPolicyModel
from s1a.decision_models import wire
from s1a.browser import action_space, prompts
from s1a.decision_models import DecisionModel, JevModel, ScriptedTransport, Usage
from s1a.spec import BrowserAgentSpec, Budget
from s1a.browser.decision_model import (
    ACTION_SETTLE_BUDGET_MS,
    ACTION_SETTLE_START_MS,
    DECISION_ATTEMPTS,
    FINAL_TEXT_CHARS,
    BrowserDecisionModel,
    BrowserPolicy,
    MAX_CONSECUTIVE_WAITS,
    MAX_PROBE_SETTLE_MS,
    PROBE_QUIET_MS,
    PROBE_SETTLE_MS,
    WAIT_SETTLE_BUDGET_MS,
)

_OPERATIONS = ["CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "WAIT", "DONE", "BLOCKED"]
_CONTROL_OPERATIONS = ["WAIT", "DONE", "BLOCKED"]
_SNAPSHOT: dict[str, Any] = {
    "url": "https://flights.test/",
    "title": "Flights",
    "text": "Where from? Where to?",
    "can_scroll_down": True,
    "can_scroll_up": False,
    "page_key": "k1",
    "generation_id": "g1",
    "elements": [
        {"node": 1, "target_id": "t_g1_1", "role": "button", "label": "Search", "value": "", "editable": False},
        {
            "node": 2,
            "target_id": "t_g1_2",
            "role": "combobox",
            "label": "Where to?",
            "value": "",
            "editable": True,
            "expanded": "false",
        },
        {
            "node": 3,
            "target_id": "t_g1_3",
            "role": "combobox",
            "label": "Class",
            "value": "Economy",
            "editable": False,
            "options": [{"label": "Business", "value": "b"}, {"label": "First", "value": "f"}],
        },
    ],
}
_FAILED_PROBE_ENVELOPE: dict[str, Any] = {
    "ok": False,
    "error": "policy probe failed: boom",
    "elements": [],
    "page_state": {},
}


def _answer(choice: str, ids: list[str], confidence: float = 0.9) -> dict[str, Any]:
    top = confidence if len(ids) > 1 else 1.0
    rest = (1.0 - top) / max(1, len(ids) - 1)
    probabilities = {i: (top if i == choice else rest) for i in ids}
    return {"choice": choice, "confidence": confidence, "probabilities": probabilities}


def _answers(operation: str, value: str) -> dict[str, Any]:
    """A payload answering every head, as Jev does; ``decide_many`` validates them all, ``interpret`` uses one."""
    return {
        "answers": {
            "operation": _answer(operation, _OPERATIONS),
            "type_text_target": _answer("2", ["2"]),
            "click_target": _answer("1", ["1", "2"]),
            "select_target": _answer("3:1", ["3:1", "3:2"]),
            "text_value": _answer(value, ["London", "none"]),
        }
    }


def _control_answer(operation: str) -> dict[str, Any]:
    """An ``operation`` answer scoped to an action space with no addressable elements."""
    return {"answers": {"operation": _answer(operation, _CONTROL_OPERATIONS)}}


def _op_answer(operation: str) -> dict[str, Any]:
    """A control operation (``WAIT`` / ``DONE`` / ``BLOCKED`` / scroll) over the full ``_SNAPSHOT`` action space.

    Every head is still answered: the decision-model layer validates the whole answer set, and a real Jev response
    always carries every head. Only the operation is used.
    """
    return _answers(operation, "none")


def _malformed_operation_answer() -> dict[str, Any]:
    """An ``operation`` answer whose probabilities do not sum to 1, over the full ``_OPERATIONS`` set."""
    return {
        "answers": {
            "operation": {
                "choice": "WAIT",
                "confidence": 0.9,
                "probabilities": {op: 0.3 for op in _OPERATIONS},
            }
        }
    }


class TestActionSpaceAndQuestions(TestCase):
    def test_heads_and_questions_follow_the_probe(self) -> None:
        space = action_space.build_action_space(_SNAPSHOT, [])
        self.assertEqual(space.operations, _OPERATIONS)
        self.assertEqual(sorted(space.heads["CLICK"]), ["1", "2"])
        self.assertEqual(list(space.heads["TYPE_TEXT"]), ["2"])
        self.assertEqual(list(space.heads["SELECT"]), ["3:1", "3:2"])
        rules = prompts.OPERATION_RULES["en"]
        questions = action_space.build_questions(
            space, goal="Fly to London", values=["London"], rules=rules, language="en"
        )
        self.assertEqual(
            sorted(questions), ["click_target", "operation", "select_target", "text_value", "type_text_target"]
        )
        self.assertEqual((questions["operation"].goal, questions["operation"].rules), ("Fly to London", (rules,)))
        self.assertEqual(questions["type_text_target"].options["2"]["element"], "[2] Where to?")
        self.assertEqual(questions["click_target"].options["2"]["element"], "[2] Where to?")
        self.assertEqual(questions["select_target"].options["3:2"]["option"], "First")
        self.assertEqual(questions["select_target"].operation, "SELECT")
        self.assertEqual(questions["select_target"].rules, (rules, prompts.TARGET_RULES["en"]))
        self.assertIn("none", questions["text_value"].options)
        self.assertEqual(questions["text_value"].rules, (prompts.VALUE_RULES["en"],))
        observation = action_space.build_observation(space, _SNAPSHOT, [])
        self.assertEqual(observation.state["elements"][1]["operations"], ["TYPE_TEXT", "CLICK"])
        self.assertEqual(observation.state["page"]["url"], "https://flights.test/")
        self.assertEqual(observation.state["recent_actions"], [])

    def test_no_value_head_without_offered_values(self) -> None:
        space = action_space.build_action_space(_SNAPSHOT, [])
        questions = action_space.build_questions(space, goal="g", values=[], rules="r", language="en")
        self.assertNotIn("text_value", questions)

    def test_the_observation_keeps_the_last_ten_actions_with_four_keys(self) -> None:
        history = [
            {"action": f"a{i}", "kind": "click", "text": None, "page_changed": True, "elapsed_ms": i} for i in range(12)
        ]
        observation = action_space.build_observation(action_space.ActionSpace(), _SNAPSHOT, history)
        recent = observation.state["recent_actions"]
        self.assertEqual([entry["action"] for entry in recent], [f"a{i}" for i in range(2, 12)])
        self.assertEqual(set(recent[0]), {"action", "kind", "text", "page_changed"})

    def test_empty_elements_yield_control_only_action_space(self) -> None:
        """B3: the probe's failure envelope must still fold into a decidable (WAIT/DONE/BLOCKED) space."""
        space = action_space.build_action_space(_FAILED_PROBE_ENVELOPE, [])
        self.assertEqual(space.operations, _CONTROL_OPERATIONS)
        self.assertEqual(space.heads, {})

    def test_an_occluded_element_gets_a_row_but_no_candidate_head(self) -> None:
        """A cookie-consent overlay sitting on a control (probe_js's occlusion check) must not be offered
        as a CLICK/TYPE_TEXT/SELECT target: a real click there would not reach it. Its own dismiss button,
        never itself occluded, stays a normal candidate."""
        snapshot = {
            **_SNAPSHOT,
            "elements": [
                {
                    "node": 1,
                    "target_id": "t_g1_1",
                    "role": "button",
                    "label": "Search",
                    "value": "",
                    "editable": False,
                    "actionable": False,
                    "clickable": False,
                    "blocked_by": "We Care About Your Privacy",
                },
                {
                    "node": 2,
                    "target_id": "t_g1_2",
                    "role": "combobox",
                    "label": "Where to?",
                    "value": "",
                    "editable": True,
                    "expanded": "false",
                    "actionable": False,
                    "clickable": False,
                    "blocked_by": "We Care About Your Privacy",
                },
                {
                    "node": 4,
                    "target_id": "t_g1_4",
                    "role": "button",
                    "label": "Accept All",
                    "value": "",
                    "editable": False,
                    "region": "dialog:We Care About Your Privacy",
                },
            ],
        }
        space = action_space.build_action_space(snapshot, [])
        self.assertEqual(
            space.heads.get("CLICK", {}), {"3": action_space.Candidate("CLICK", "3", snapshot["elements"][2])}
        )
        self.assertNotIn("TYPE_TEXT", space.heads)
        self.assertEqual(space.elements[0]["operations"], [])
        self.assertEqual(space.elements[0]["blocked_by"], "We Care About Your Privacy")
        self.assertEqual(space.elements[2]["operations"], ["CLICK"])
        self.assertNotIn("blocked_by", space.elements[2])

    def test_a_blocked_select_offers_no_option_candidates(self) -> None:
        snapshot = {
            **_SNAPSHOT,
            "elements": [
                {
                    "node": 3,
                    "target_id": "t_g1_3",
                    "role": "combobox",
                    "label": "Class",
                    "value": "Economy",
                    "editable": False,
                    "options": [{"label": "Business", "value": "b"}],
                    "actionable": False,
                    "blocked_by": "Sign-in required",
                },
            ],
        }
        space = action_space.build_action_space(snapshot, [])
        self.assertNotIn("SELECT", space.heads)
        self.assertEqual(space.elements[0]["operations"], [])


def _fake_model_init(self: Model, model_client_config: Any, model_config: Any) -> None:
    self.model_client_config = model_client_config
    self.model_config = model_config
    self._client = None


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def probe_for_policy(self, source: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(params)
        return json.loads(json.dumps(_SNAPSHOT))


class _ScriptedPageKeyRuntime:
    """A runtime that returns ``_SNAPSHOT`` with ``page_key`` overridden per probe call, in order.

    Once the scripted sequence is exhausted, the last key repeats -- this only matters for the
    settle loop's clamp behaviour, which the escalation tests exercise explicitly.
    """

    def __init__(self, page_keys: list[str | None]) -> None:
        self.page_keys = page_keys
        self.calls: list[dict[str, Any]] = []

    async def probe_for_policy(self, source: str, params: dict[str, Any]) -> dict[str, Any]:
        index = min(len(self.calls), len(self.page_keys) - 1)
        self.calls.append(params)
        if self.page_keys[index] is None:
            return json.loads(json.dumps(_FAILED_PROBE_ENVELOPE))  # a transport hiccup on this probe
        snapshot = json.loads(json.dumps(_SNAPSHOT))
        snapshot["page_key"] = self.page_keys[index]
        return snapshot


class _ScriptedUrlRuntime:
    """A runtime whose probes land on the scripted URLs in order, every probe on a fresh ``page_key``."""

    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.calls: list[dict[str, Any]] = []

    async def probe_for_policy(self, source: str, params: dict[str, Any]) -> dict[str, Any]:
        index = min(len(self.calls), len(self.urls) - 1)
        self.calls.append(params)
        snapshot = json.loads(json.dumps(_SNAPSHOT))
        snapshot["url"] = self.urls[index]
        snapshot["page_key"] = f"{self.urls[index]}#{len(self.calls)}"
        return snapshot


class _AlwaysFailingRuntime:
    """A runtime whose probe never recovers, mirroring the B3-fixed ``probe_for_policy`` failure envelope."""

    async def probe_for_policy(self, source: str, params: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(_FAILED_PROBE_ENVELOPE))


class _Wire(ScriptedTransport):
    """The scripted wire with its queue exposed: a test appends the next turn's ``{"answers": ...}`` payload.

    An under-scripted test fails loudly on the pop instead of getting a made-up answer.
    """

    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        super().__init__(latency_ms=7)
        self.scripted = scripted

    async def decide(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        self.bodies.append(body)
        return self.scripted.pop(0), 7


def _wire(slot_model: BrowserDecisionModel) -> _Wire:
    """The scripted transport under the slot model's decision model: the bodies it was sent and its answer queue."""
    return slot_model._decision_model._transport


def _failing_model() -> DecisionModel:
    """a decision model whose transport always fails, as B4 requires the model to tolerate."""
    error = build_error(StatusCode.MODEL_CALL_FAILED, error_msg="decisions endpoint unavailable")
    return JevModel(ScriptedTransport(error=error))


def _fallback() -> mock.MagicMock:
    fallback = mock.MagicMock(spec=Model)
    fallback.model_client_config = None
    fallback.model_config = None

    async def fake_invoke(messages: list[dict[str, str]], **_kwargs: Any) -> mock.MagicMock:
        fallback.asked.append(messages)
        if "final page" in messages[0]["content"]:
            return mock.MagicMock(content="  Flights from 120 CHF.  ")
        asks_for_values = "values" in messages[0]["content"]
        payload = {"values": ["London"]} if asks_for_values else {"text": "London"}
        return mock.MagicMock(content=json.dumps(payload))

    fallback.asked = []

    fallback.invoke = fake_invoke
    return fallback


def _spec() -> BrowserAgentSpec:
    return BrowserAgentSpec(
        name="test",
        description="A test agent.",
        rules=prompts.OPERATION_RULES["en"],
        language="en",
        budget=Budget(max_steps=100, timeout_s=120, stall_after=3),
        goal=None,
    )


def _policy(*, goal_value_cache: bool, prefetch_values: bool, batch_actions: bool) -> BrowserPolicy:
    return BrowserPolicy(
        prefetch_values=prefetch_values, batch_actions=batch_actions, goal_value_cache=goal_value_cache
    )


def _slot_model(
    scripted: list[dict[str, Any]],
    *,
    goal_value_cache: bool = True,
    prefetch_values: bool = True,
    decision_model: DecisionModel | None = None,
    runtime: Any = None,
    batch_actions: bool = False,
) -> BrowserDecisionModel:
    with mock.patch.object(Model, "__init__", _fake_model_init):
        slot_model = BrowserDecisionModel(
            _spec(),
            _policy(goal_value_cache=goal_value_cache, prefetch_values=prefetch_values, batch_actions=batch_actions),
            _fallback(),
            decision_model=decision_model or JevModel(_Wire(scripted)),
            value_model=None,
        )
    slot_model.bind_runtime(runtime or _FakeRuntime())
    return slot_model


def _decision_probes(runtime: Any) -> list[dict[str, Any]]:
    """The runtime's probe calls without the whole-page answer probe DONE adds."""
    return [call for call in runtime.calls if not call.get("all_text")]


_TOOLS = [
    ToolInfo(name=name) for name in ("browser_click", "browser_type", "browser_select_option", "browser_press_key")
]
_MESSAGES = [{"role": "user", "content": "Fly to London"}]


class TestBrowserDecisionModel(IsolatedAsyncioTestCase):
    async def test_first_tick_types_the_prefetched_value_then_done(self) -> None:
        slot_model = _slot_model([_answers("TYPE_TEXT", "none"), _answers("DONE", "none")])

        first = await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        self.assertEqual(first.tool_calls[0].name, "browser_type")
        self.assertEqual(
            json.loads(first.tool_calls[0].arguments),
            {"generation_id": "g1", "target_id": "t_g1_2", "text": "London"},
        )
        self.assertEqual(slot_model.ticks[0]["value_source"], "prefetch", "empty fields are prefetched at probe time")
        self.assertNotIn("text_value", _wire(slot_model).bodies[0]["questions"])

        second = await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        self.assertIsNone(second.tool_calls)
        self.assertEqual(json.loads(second.content)["status"], "DONE")
        self.assertEqual(slot_model._runtime.calls[1]["after"], {"kind": "fill", "node": 2})
        self.assertEqual(slot_model.report()["interactions"], 1)

    async def test_the_tick_records_the_decisions_latency_tokens_and_the_answering_model(self) -> None:
        payload = {**_answers("CLICK", "none"), "usage": {"input_tokens": 315, "output_tokens": 0}, "model": "jev-9"}
        slot_model = _slot_model([payload], goal_value_cache=False)

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        tick = slot_model.ticks[0]
        self.assertEqual((tick["decision_ms"], tick["input_tokens"], tick["output_tokens"]), (7, 315, 0))
        self.assertEqual((tick["operation"], tick["target"], tick["confidence"]), ("CLICK", "Search", 0.9))
        self.assertEqual(tick["probabilities"], {"1": 0.9, "2": 0.1}, "the click head's probabilities, for the replay")
        self.assertEqual(tick["candidates"]["1"], "Search")
        report = slot_model.report()
        self.assertEqual((report["decisions"], report["median_decision_ms"], report["jev_input_tokens"]), (1, 7, 315))

    async def test_report_prices_input_tokens_by_backend_flag(self) -> None:
        """The scripted backend's token usage is priced only when it opts into the Jev input rate."""
        from s1a.decision_models import ScriptedModel

        decision_model = ScriptedModel(latency_ms=3, usage=Usage(11, 0), model="laya-rl-agent")
        slot_model = _slot_model([], goal_value_cache=False, decision_model=decision_model)

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(slot_model.ticks[0]["input_tokens"], 11, "the tick itself still records what the model spent")
        self.assertEqual(slot_model.report()["jev_input_tokens"], 0)
        decision_model.bills_input_tokens = True
        self.assertEqual(slot_model.report()["jev_input_tokens"], 11)

    async def test_a_laya_shaped_model_fills_the_slot_the_same_way(self) -> None:
        """The policy asks any decision_model: a scripted one at the interface, no wire at all, decides a tick."""
        from s1a.decision_models import ScriptedModel

        # the double answers every head with its first offered key: CLICK on element 1
        decision_model = ScriptedModel(latency_ms=3, usage=Usage(11, 0), model="laya-rl-agent")
        slot_model = _slot_model([], goal_value_cache=False, decision_model=decision_model)

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(message.tool_calls[0].name, "browser_click")
        self.assertTrue(message.tool_calls[0].id.startswith("scripted-"), "the call id names the slot model, not Jev")
        ((observation, questions),) = decision_model.calls
        self.assertEqual(sorted(questions), ["click_target", "operation", "select_target", "type_text_target"])
        self.assertEqual(observation.state["page"]["title"], "Flights")
        self.assertEqual((slot_model.ticks[0]["decision_ms"], slot_model.ticks[0]["input_tokens"]), (3, 11))

    async def test_cached_value_is_chosen_by_jev(self) -> None:
        slot_model = _slot_model([_answers("TYPE_TEXT", "none")])
        await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        await slot_model._run.values_task
        slot_model._run.pending = None
        _wire(slot_model).scripted.append(_answers("TYPE_TEXT", "London"))

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        self.assertEqual(json.loads(message.tool_calls[0].arguments)["text"], "London")
        self.assertIn("text_value", _wire(slot_model).bodies[-1]["questions"])
        self.assertEqual(slot_model.ticks[-1]["value_source"], "cache")

    async def test_cache_off_means_no_value_head_and_prefetch_source(self) -> None:
        slot_model = _slot_model([_answers("TYPE_TEXT", "none")], goal_value_cache=False)
        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        self.assertEqual(json.loads(message.tool_calls[0].arguments)["text"], "London")
        self.assertNotIn("text_value", _wire(slot_model).bodies[0]["questions"])
        self.assertIsNone(slot_model._run.values_task)
        self.assertEqual(slot_model.ticks[0]["value_source"], "prefetch")
        self.assertEqual(slot_model.report()["values"], {"cache": 0, "prefetch": 1, "llm": 0})

    async def test_non_browser_turn_goes_to_fallback(self) -> None:
        fallback = _fallback()
        fallback.invoke = mock.AsyncMock(return_value="chat-reply")
        with mock.patch.object(Model, "__init__", _fake_model_init):
            slot_model = BrowserDecisionModel(
                _spec(),
                _policy(goal_value_cache=False, prefetch_values=True, batch_actions=False),
                fallback,
                decision_model=JevModel(_Wire([])),
                value_model=None,
            )
        reply = await slot_model.invoke([{"role": "user", "content": "summarise"}], tools=[ToolInfo(name="read_file")])
        self.assertEqual(reply, "chat-reply")

    async def test_report_before_any_turn_returns_the_zeroed_shape(self) -> None:
        slot_model = _slot_model([])
        self.assertEqual(
            slot_model.report(),
            {
                "elapsed_ms": 0,
                "decisions": 0,
                "interactions": 0,
                "waits": 0,
                "median_decision_ms": 0,
                "jev_input_tokens": 0,
                "median_probe_ms": 0,
                "settle_probes": 0,
                "settle_ms": 0,
                "values": {"cache": 0, "prefetch": 0, "llm": 0},
                "history": [],
            },
        )
        self.assertEqual(slot_model.ticks, [])
        self.assertEqual(slot_model.started_at, 0.0)


class TestJevRunIsolation(IsolatedAsyncioTestCase):
    """B1: per-task state must not leak into the task that follows it."""

    async def test_second_task_with_a_new_goal_starts_a_clean_run(self) -> None:
        slot_model = _slot_model([_answers("DONE", "none")], goal_value_cache=False)

        first_messages = [{"role": "user", "content": "Fly to London"}]
        first = await slot_model.invoke(first_messages, tools=_TOOLS)
        self.assertIsNone(first.tool_calls)
        self.assertEqual(json.loads(first.content)["status"], "DONE")
        self.assertTrue(slot_model._run.finished)

        _wire(slot_model).scripted.append(_answers("TYPE_TEXT", "none"))
        second_messages = [{"role": "user", "content": "Book a hotel in https://hotels.test/berlin"}]
        navigate = await slot_model.invoke(second_messages, tools=_TOOLS)
        self.assertEqual(navigate.tool_calls[0].name, "browser_navigate", "the URL shortcut must fire again")
        self.assertEqual(json.loads(navigate.tool_calls[0].arguments)["url"], "https://hotels.test/berlin")
        self.assertEqual(slot_model._run.goal, second_messages[0]["content"])

        second = await slot_model.invoke(second_messages, tools=_TOOLS)
        self.assertEqual(second.tool_calls[0].name, "browser_type")
        body = _wire(slot_model).bodies[-1]
        self.assertEqual(body["questions"]["operation"]["instructions"]["goal"], second_messages[0]["content"])
        self.assertEqual(body["state"]["recent_actions"], [], "the second task must not inherit the first history")

    async def test_starting_a_new_run_cancels_the_previous_runs_background_tasks(self) -> None:
        slot_model = _slot_model([_answers("CLICK", "none")])

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        first_run = slot_model._run
        self.assertIsNotNone(first_run.values_task)
        self.assertTrue(first_run.prefetched, "the empty 'Where to?' field is prefetched on the first probe")
        stale_values_task = first_run.values_task
        stale_prefetch_task = next(iter(first_run.prefetched.values()))

        _wire(slot_model).scripted.append(_answers("DONE", "none"))
        await slot_model.invoke([{"role": "user", "content": "Fly to Berlin"}], tools=_TOOLS)

        # Cancellation is cooperative: a task that never got a chance to run only reflects the
        # cancel() request once the event loop actually delivers it.
        for task in (stale_values_task, stale_prefetch_task):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.assertTrue(stale_values_task.cancelled() or stale_values_task.done())
        self.assertTrue(stale_prefetch_task.cancelled() or stale_prefetch_task.done())
        self.assertIsNot(slot_model._run, first_run)

    async def test_a_terminal_verdict_cancels_the_runs_background_tasks(self) -> None:
        slot_model = _slot_model([_answers("DONE", "none")])

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertTrue(slot_model._run.finished)
        self.assertEqual(slot_model._run.prefetched, {})
        background = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        self.assertEqual(len(background), 2, "the goal-value task and the prefetched 'Where to?' value")
        for task in background:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.assertTrue(all(task.cancelled() for task in background))

    async def test_a_later_user_message_does_not_restart_the_run(self) -> None:
        slot_model = _slot_model([_answers("CLICK", "none"), _answers("DONE", "none")], goal_value_cache=False)

        first = await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        run = slot_model._run
        messages = [
            *_MESSAGES,
            {"role": "tool", "tool_call_id": first.tool_calls[0].id, "content": "clicked"},
            {"role": "user", "content": [{"type": "text", "text": "A screenshot was attached."}]},
        ]
        second = await slot_model.invoke(messages, tools=_TOOLS)

        self.assertIs(slot_model._run, run, "the goal is the first user message; later ones are data")
        self.assertEqual(json.loads(second.content)["status"], "DONE")
        self.assertEqual(_wire(slot_model).bodies[-1]["state"]["recent_actions"][0]["action"], "Search")

    async def test_tool_call_ids_differ_between_runs(self) -> None:
        slot_model = _slot_model([_answers("CLICK", "none"), _answers("CLICK", "none")], goal_value_cache=False)

        first = await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        second = await slot_model.invoke([{"role": "user", "content": "Fly to Berlin"}], tools=_TOOLS)

        self.assertNotEqual(first.tool_calls[0].id, second.tool_calls[0].id)


class TestDecisionModelAttributeIsolation(TestCase):
    """B2: the model must not collide with the base ``Model._client`` slot."""

    def test_base_client_survives_and_the_model_is_the_injected_double(self) -> None:
        client_config = ModelClientConfig(
            client_provider="OpenAI",
            api_key="sk-test-key",
            api_base="https://api.openai.com/v1",
            verify_ssl=False,
        )
        fallback = Model(client_config, None)
        decision_model = JevModel(_Wire([]))

        slot_model = BrowserDecisionModel(
            _spec(),
            _policy(goal_value_cache=False, prefetch_values=True, batch_actions=False),
            fallback,
            decision_model=decision_model,
            value_model=None,
        )

        self.assertIsInstance(slot_model._client, BaseModelClient)
        self.assertIsNot(slot_model._client, decision_model)
        self.assertIs(slot_model._decision_model, decision_model)


class TestJevProbeFailureDegradesToBlocked(IsolatedAsyncioTestCase):
    """B3: a probe that never recovers must not spin or raise — it must reach BLOCKED."""

    async def test_always_failing_probe_terminates_blocked_within_the_wait_budget(self) -> None:
        scripted = [_control_answer("WAIT") for _ in range(MAX_CONSECUTIVE_WAITS + 1)]
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=_AlwaysFailingRuntime())

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertIsNone(message.tool_calls)
        summary = json.loads(message.content)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertEqual(
            len(_wire(slot_model).bodies),
            1,
            "a probe that never recovers exhausts the streak budget on the first WAIT, and an exhausted "
            "budget is terminal -- the model is never re-asked about a snapshot it already answered",
        )


class TestJevWaitCollapsesIntoInPageSettling(IsolatedAsyncioTestCase):
    """A WAIT verdict must be spent re-probing in-page, not by paying another decisions request."""

    async def test_wait_is_absorbed_by_the_probe_not_the_wire(self) -> None:
        # The page is still loading and settles on the second settle probe -- a constant page_key would
        # instead mean "genuinely stuck", which is terminal rather than re-asked.
        runtime = _ScriptedPageKeyRuntime(["k1", "k1", "k2"])
        slot_model = _slot_model(
            [_op_answer("WAIT"), _answers("CLICK", "none")], goal_value_cache=False, runtime=runtime
        )

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(message.tool_calls[0].name, "browser_click")
        self.assertEqual(len(_wire(slot_model).bodies), 2, "one WAIT verdict must cost exactly one re-ask")
        self.assertGreater(
            len(runtime.calls), 2, "the extra waiting must show up as extra probes, not extra decide() calls"
        )

    async def test_a_wait_that_moved_the_page_is_recorded_as_progress(self) -> None:
        runtime = _ScriptedPageKeyRuntime(["k1", "k1", "k2"])
        slot_model = _slot_model(
            [_op_answer("WAIT"), _answers("CLICK", "none")], goal_value_cache=False, runtime=runtime
        )

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        wait_entry = next(entry for entry in slot_model._run.history if entry["kind"] == "wait")
        self.assertIs(wait_entry["page_changed"], True, "the next state must not show the wait as '(no change)'")

    async def test_escalating_settle_doubles_and_clamps_at_the_probe(self) -> None:
        runtime = _ScriptedPageKeyRuntime(["k1"] * 8)
        slot_model = _slot_model([_op_answer("WAIT"), _op_answer("DONE")], goal_value_cache=False, runtime=runtime)

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        settle_calls = [call["settle_ms"] for call in runtime.calls[1:]]
        self.assertEqual(settle_calls, [PROBE_SETTLE_MS, PROBE_SETTLE_MS * 2, MAX_PROBE_SETTLE_MS])
        self.assertLessEqual(sum(settle_calls), WAIT_SETTLE_BUDGET_MS)
        self.assertTrue(all(ms <= MAX_PROBE_SETTLE_MS for ms in settle_calls))

    async def test_never_changing_page_stays_bounded_by_both_budgets(self) -> None:
        scripted = [_op_answer("WAIT") for _ in range(MAX_CONSECUTIVE_WAITS + 1)]
        runtime = _ScriptedPageKeyRuntime(["k1"] * 64)
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=runtime)

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertIsNone(message.tool_calls)
        summary = json.loads(message.content)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertIn("waited without progress", summary["reason"])
        self.assertEqual(
            len(_wire(slot_model).bodies),
            1,
            "an exhausted settle budget is terminal: re-asking about an unchanged snapshot buys the same "
            "WAIT verdict at full request cost",
        )

        settle_calls = [call["settle_ms"] for call in runtime.calls[1:]]
        self.assertEqual(settle_calls, [PROBE_SETTLE_MS, PROBE_SETTLE_MS * 2, MAX_PROBE_SETTLE_MS])
        self.assertLessEqual(
            sum(settle_calls),
            WAIT_SETTLE_BUDGET_MS,
            "the budget spans the whole WAIT streak, not one verdict -- a per-verdict budget would be "
            f"re-granted {MAX_CONSECUTIVE_WAITS + 1}x and stretch one stuck step by that factor",
        )

    async def test_settle_budget_resets_once_the_page_actually_progresses(self) -> None:
        # calls: [0] initial probe, [1]/[2] settle probes -- the page moves on the second, so the turn
        # ends in a CLICK. Second turn: [3] initial probe already shows the click's effect (k3, no action
        # settle), then [4]/[5]/[6] a full, freshly granted budget.
        runtime = _ScriptedPageKeyRuntime(["k1", "k1", "k2", "k3", "k3", "k3", "k3"])
        slot_model = _slot_model(
            [_op_answer("WAIT"), _answers("CLICK", "none"), _op_answer("WAIT")],
            goal_value_cache=False,
            runtime=runtime,
        )

        first = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(first.tool_calls[0].name, "browser_click")
        self.assertEqual(slot_model._run.settle_spent_ms, 0, "a real action ends the streak and clears its budget")

        second = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(json.loads(second.content)["status"], "BLOCKED")
        self.assertEqual(
            [call["settle_ms"] for call in _decision_probes(runtime)[4:]],
            [PROBE_SETTLE_MS, PROBE_SETTLE_MS * 2, MAX_PROBE_SETTLE_MS],
            "the next streak must start from a full budget, not the remainder of the previous one",
        )

    async def test_progress_inside_a_wait_streak_regrants_the_settle_budget(self) -> None:
        # The first WAIT spends the whole budget before the page moves (k2); the second WAIT must get a
        # the next stage of the render (k3) gets a fresh budget.
        runtime = _ScriptedPageKeyRuntime(["k1", "k1", "k1", "k2", "k2", "k3"])
        slot_model = _slot_model(
            [_op_answer("WAIT"), _op_answer("WAIT"), _op_answer("DONE")], goal_value_cache=False, runtime=runtime
        )

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(json.loads(message.content)["status"], "DONE")
        self.assertEqual(
            [call["settle_ms"] for call in _decision_probes(runtime)[4:]], [PROBE_SETTLE_MS, PROBE_SETTLE_MS * 2]
        )

    async def test_a_failed_settle_probe_is_not_progress(self) -> None:
        runtime = _ScriptedPageKeyRuntime(["k1", "k1", None, "k2"])
        slot_model = _slot_model(
            [_op_answer("WAIT"), _answers("CLICK", "none")], goal_value_cache=False, runtime=runtime
        )

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(message.tool_calls[0].name, "browser_click")
        self.assertEqual(len(runtime.calls), 4, "the failed probe is retried, not answered")

    async def test_a_wait_streak_that_ends_blocked_keeps_its_settle_metrics(self) -> None:
        runtime = _ScriptedPageKeyRuntime(["k1"] * 8)
        slot_model = _slot_model([_op_answer("WAIT")], goal_value_cache=False, runtime=runtime)

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(json.loads(message.content)["status"], "BLOCKED")
        self.assertEqual(slot_model.report()["settle_probes"], 3)
        self.assertEqual(slot_model.ticks[-1]["settle_probes"], 3)

    async def test_non_wait_path_is_unchanged_one_probe_one_decide_one_action(self) -> None:
        slot_model = _slot_model([_answers("CLICK", "none")], goal_value_cache=False)

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(message.tool_calls[0].name, "browser_click")
        self.assertEqual(len(_wire(slot_model).bodies), 1)
        self.assertEqual(len(slot_model._runtime.calls), 1)
        self.assertEqual(slot_model.ticks[-1]["settle_probes"], 0)
        self.assertEqual(slot_model.ticks[-1]["settle_ms"], 0)


class TestJevActionSettle(IsolatedAsyncioTestCase):
    """An action whose effect has not shown yet is settled in-page before the model is asked."""

    async def test_unchanged_page_after_an_action_is_settled_before_the_model_is_asked(self) -> None:
        # call#0: turn 1's probe (k1) -> CLICK. call#1: turn 2's probe still shows k1, so the action settle
        # re-probes at 250 ms (call#2, still k1) and 500 ms (call#3, k2): the page moved, Jev sees k2.
        runtime = _ScriptedPageKeyRuntime(["k1", "k1", "k1", "k2"])
        slot_model = _slot_model(
            [_answers("CLICK", "none"), _op_answer("DONE")], goal_value_cache=False, runtime=runtime
        )

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        second = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(json.loads(second.content)["status"], "DONE")
        self.assertEqual(len(_wire(slot_model).bodies), 2, "the delayed effect must not cost a decision request")
        self.assertEqual(len(_decision_probes(runtime)), 4)
        self.assertEqual(
            [(call["settle_ms"], call["quiet_ms"]) for call in _decision_probes(runtime)[2:]],
            [(ACTION_SETTLE_START_MS,) * 2, (ACTION_SETTLE_START_MS * 2,) * 2],
            "action settle waits are wall-clock waits: the quiet window equals the settle window",
        )
        click_entry = next(h for h in slot_model._run.history if h["kind"] == "click")
        self.assertIs(click_entry["page_changed"], True, "page_changed reflects the settled post-action probe")
        self.assertEqual(slot_model.ticks[-1]["action_settle_probes"], 2)
        self.assertEqual(slot_model.ticks[-1]["action_settle_ms"], ACTION_SETTLE_START_MS * 3)

    async def test_action_settle_is_bounded_and_shares_the_wait_budget(self) -> None:
        runtime = _ScriptedPageKeyRuntime(["k1"] * 64)
        slot_model = _slot_model(
            [_answers("CLICK", "none"), _op_answer("WAIT")], goal_value_cache=False, runtime=runtime
        )

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        second = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(json.loads(second.content)["status"], "BLOCKED")
        self.assertEqual(len(_wire(slot_model).bodies), 2)
        settle_calls = [call["settle_ms"] for call in runtime.calls[2:]]
        self.assertEqual(settle_calls, [250, 500, 250, 500, 1000, 500])
        self.assertEqual(sum(settle_calls[:3]), ACTION_SETTLE_BUDGET_MS)
        self.assertEqual(
            sum(settle_calls),
            WAIT_SETTLE_BUDGET_MS,
            "the action settle and the WAIT streak share one per-step budget",
        )
        click_entry = next(h for h in slot_model._run.history if h["kind"] == "click")
        self.assertIs(click_entry["page_changed"], False)

    async def test_a_visible_effect_skips_the_action_settle(self) -> None:
        runtime = _ScriptedPageKeyRuntime(["k1", "k2"])
        slot_model = _slot_model(
            [_answers("CLICK", "none"), _op_answer("DONE")], goal_value_cache=False, runtime=runtime
        )

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(len(_decision_probes(runtime)), 2)
        self.assertEqual(_decision_probes(runtime)[1]["quiet_ms"], PROBE_QUIET_MS)
        self.assertEqual(slot_model.ticks[-1]["action_settle_probes"], 0)

    async def test_a_failed_probe_after_an_action_is_settled_like_an_unchanged_page(self) -> None:
        # call#1 is the runtime's failure envelope (no page_key): it shows nothing, so the action is settled
        # in-page and the model decides over the recovered page, not over zero elements.
        runtime = _ScriptedPageKeyRuntime(["k1", None, "k1"])
        slot_model = _slot_model(
            [_answers("CLICK", "none"), _op_answer("DONE")], goal_value_cache=False, runtime=runtime
        )

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        second = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(json.loads(second.content)["status"], "DONE")
        click_entry = next(h for h in slot_model._run.history if h["kind"] == "click")
        self.assertIs(click_entry["page_changed"], False)
        self.assertEqual(slot_model.ticks[-1]["action_settle_probes"], 3)
        self.assertEqual(slot_model.ticks[-1]["elements"], len(_SNAPSHOT["elements"]))


class TestDecisionFailureDegradesToBlocked(IsolatedAsyncioTestCase):
    """B4: a decisions-endpoint failure must degrade the turn, not crash it; an unusable answer is re-asked once."""

    async def test_transport_failure_returns_blocked_summary_without_a_re_ask(self) -> None:
        slot_model = _slot_model([], decision_model=_failing_model())

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertIsNone(message.tool_calls)
        summary = json.loads(message.content)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertIn("decision failed", summary["reason"])
        self.assertEqual(len(_wire(slot_model).bodies), 1, "the transport has its own retries; the layer adds none")

    async def test_malformed_answers_are_re_asked_once_then_blocked(self) -> None:
        slot_model = _slot_model([_malformed_operation_answer() for _ in range(DECISION_ATTEMPTS)])

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertIsNone(message.tool_calls)
        summary = json.loads(message.content)
        self.assertEqual(summary["status"], "BLOCKED")
        bodies = _wire(slot_model).bodies
        self.assertEqual(len(bodies), DECISION_ATTEMPTS)
        self.assertEqual(bodies[0], bodies[1], "the re-ask sends the same request")

    async def test_a_usable_answer_on_the_re_ask_decides_the_tick(self) -> None:
        slot_model = _slot_model([_malformed_operation_answer(), _answers("CLICK", "none")], goal_value_cache=False)

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(message.tool_calls[0].name, "browser_click")
        self.assertEqual(len(_wire(slot_model).bodies), 2)
        self.assertEqual(len(slot_model.ticks), 1)

    async def test_a_malformed_unused_head_is_still_an_unusable_answer(self) -> None:
        """Every head is validated, not only the chosen one: a broken select head on a CLICK answer is re-asked."""
        broken = _answers("CLICK", "none")
        broken["answers"]["select_target"] = _answer("3:9", ["3:1", "3:2"])
        slot_model = _slot_model([broken, _answers("CLICK", "none")], goal_value_cache=False)

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(message.tool_calls[0].name, "browser_click")
        self.assertEqual(len(_wire(slot_model).bodies), 2)


class TestJevDecisionsClientMalformedBody(IsolatedAsyncioTestCase):
    """Hole 1 (completes B4): a 200 response with a non-JSON body must not crash the run."""

    @staticmethod
    def _client_with_transport(handler: Any) -> wire.JevDecisionsClient:
        client = wire.JevDecisionsClient(
            api_key="test-key",
            url="https://decisions.test/api/alpha/decisions",
            model="typesafe/jev-test",
            timeout_s=5.0,
        )
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return client

    async def test_decide_converts_a_malformed_body_into_model_call_failed(self) -> None:
        secret = "sk-should-not-leak-1234567890"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=f"not-json-at-all {secret}".encode())

        client = self._client_with_transport(handler)

        with self.assertRaises(BaseError) as ctx:
            await client.decide({"model": "m", "questions": {}})

        self.assertEqual(ctx.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertNotIn(secret, str(ctx.exception))

    async def test_decide_converts_a_body_in_a_bad_encoding_into_model_call_failed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b'\xff\xfe{"answers": {}}')

        with self.assertRaises(BaseError) as ctx:
            await self._client_with_transport(handler).decide({"model": "m", "questions": {}})

        self.assertEqual(ctx.exception.status, StatusCode.MODEL_CALL_FAILED)

    async def test_a_malformed_body_degrades_the_turn_to_blocked_without_leaking_the_raw_body(self) -> None:
        secret = "sk-should-not-leak-1234567890"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=f"not-json-at-all {secret}".encode())

        real_client = self._client_with_transport(handler)
        slot_model = _slot_model([], goal_value_cache=False, decision_model=JevModel(real_client))

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertIsNone(message.tool_calls)
        summary = json.loads(message.content)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertNotIn(secret, message.content)


class TestDecisionsClientFromEnv(TestCase):
    """The two decisions backends read distinct keys and never send one backend's key to the other."""

    def test_typesafe_uses_its_own_key_url_and_model(self) -> None:
        env = {"TYPESAFE_API_KEY": "ts-key", "OPENROUTER_API_KEY": "or-key", "TYPESAFE_API_URL": "https://proxy.test/"}
        with mock.patch.dict("os.environ", env, clear=False):
            client = wire.client_from_env("typesafe", timeout_s=5.0)

        self.assertEqual(client.url, wire.TYPESAFE_DECISIONS_URL, "the proxy URL override does not apply")
        self.assertEqual(client.model, wire.TYPESAFE_DEFAULT_MODEL)
        self.assertEqual(client._client.headers["Authorization"], "Bearer ts-key")

    def test_openrouter_uses_the_openrouter_key_and_honours_the_dotenv_overrides(self) -> None:
        env = {
            "TYPESAFE_API_KEY": "ts-key",
            "OPENROUTER_API_KEY": "or-key",
            "TYPESAFE_MODEL": "typesafe/jev-9",
            "TYPESAFE_API_URL": "",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            client = wire.client_from_env("openrouter", timeout_s=5.0)

        self.assertEqual(client.url, wire.DEFAULT_DECISIONS_URL)
        self.assertEqual(client.model, "typesafe/jev-9")
        self.assertEqual(client._client.headers["Authorization"], "Bearer or-key")

    def test_backend_from_env_picks_typesafe_only_without_a_proxy_url(self) -> None:
        with mock.patch.dict("os.environ", {"TYPESAFE_API_KEY": "ts-key", "TYPESAFE_API_URL": ""}, clear=False):
            self.assertEqual(wire.decisions_backend_from_env(), "typesafe")
        proxied = {"TYPESAFE_API_KEY": "ts-key", "TYPESAFE_API_URL": "https://openrouter.ai/api/alpha/decisions"}
        with mock.patch.dict("os.environ", proxied, clear=False):
            self.assertEqual(wire.decisions_backend_from_env(), "openrouter")

    def test_a_missing_key_is_a_config_error(self) -> None:
        with mock.patch.dict("os.environ", {"TYPESAFE_API_KEY": ""}, clear=False):
            with self.assertRaises(BaseError) as ctx:
                wire.client_from_env("typesafe", timeout_s=5.0)
        self.assertEqual(ctx.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)


class TestJevDecisionsClientTransportRetry(IsolatedAsyncioTestCase):
    """A dead connection is retried once and then fails the request, instead of stalling for the full timeout."""

    @staticmethod
    def _client_with_transport(handler: Any) -> wire.JevDecisionsClient:
        client = wire.JevDecisionsClient(
            api_key="test-key",
            url="https://decisions.test/api/alpha/decisions",
            model="typesafe/jev-test",
            timeout_s=5.0,
        )
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return client

    async def test_one_transport_failure_is_retried(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise httpx.ConnectTimeout("dead connection", request=request)
            return httpx.Response(200, json={"answers": {}})

        payload, _latency = await self._client_with_transport(handler).decide({"model": "m", "questions": {}})

        self.assertEqual(payload, {"answers": {}})
        self.assertEqual(len(attempts), 2)

    async def test_a_second_connect_failure_fails_the_request(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            raise httpx.ConnectError("still dead", request=request)

        with self.assertRaises(BaseError) as ctx:
            await self._client_with_transport(handler).decide({"model": "m", "questions": {}})

        self.assertEqual(ctx.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertEqual(len(attempts), 2)

    async def test_a_read_timeout_is_not_retried(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            raise httpx.ReadTimeout("the server may have answered", request=request)

        with self.assertRaises(BaseError):
            await self._client_with_transport(handler).decide({"model": "m", "questions": {}})

        self.assertEqual(len(attempts), 1, "a request the server may have received is never sent twice")

    async def test_a_rate_limit_waits_for_retry_after_within_the_deadline(self) -> None:
        slept: list[float] = []
        attempts: list[int] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                return httpx.Response(429, headers={"Retry-After": "2"})
            return httpx.Response(200, json={"answers": {}})

        with mock.patch.object(wire.asyncio, "sleep", fake_sleep):
            payload, _latency = await self._client_with_transport(handler).decide({"model": "m", "questions": {}})

        self.assertEqual((payload, slept), ({"answers": {}}, [2.0]))

    async def test_a_retry_after_beyond_the_deadline_fails_at_once(self) -> None:
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "60"}, text="slow down")

        with mock.patch.object(wire.asyncio, "sleep", fake_sleep):
            with self.assertRaises(BaseError) as ctx:
                await self._client_with_transport(handler).decide({"model": "m", "questions": {}})

        self.assertIn("HTTP 429", str(ctx.exception))
        self.assertEqual(slept, [])

    async def test_a_non_object_body_is_a_model_call_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[])

        with self.assertRaises(BaseError) as ctx:
            await self._client_with_transport(handler).decide({"model": "m", "questions": {}})

        self.assertEqual(ctx.exception.status, StatusCode.MODEL_CALL_FAILED)

    async def test_an_error_response_names_its_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="unknown criteria key 'x'")

        with self.assertRaises(BaseError) as ctx:
            await self._client_with_transport(handler).decide({"model": "m", "questions": {}})

        self.assertIn("unknown criteria key", str(ctx.exception))

    def test_the_model_uses_the_short_decisions_timeout(self) -> None:
        self.assertEqual(wire.DECISIONS_TIMEOUT_S, 5.0)


class TestJevPrefetchedValueIsolation(IsolatedAsyncioTestCase):
    """B5: prefetched values must not cross pages, and a failed prefetch must degrade gracefully."""

    async def test_failed_prefetch_falls_through_to_the_blocked_path(self) -> None:
        slot_model = _slot_model([_answers("TYPE_TEXT", "none")], goal_value_cache=False)

        async def _raise(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("value generation blew up")

        with mock.patch.object(BrowserDecisionModel, "_generate_value", _raise):
            message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertIsNone(message.tool_calls)
        summary = json.loads(message.content)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertIn("no value for field", summary["reason"])
        # The prefetch task's own exception must reach the job's artifacts, not just a WARNING log line
        # (a bare "no value for field" told us nothing about *why* the eval-time failures kept happening).
        self.assertIn("RuntimeError: value generation blew up", summary["reason"])
        self.assertEqual(slot_model.ticks[-1]["value_error"], "RuntimeError: value generation blew up")

    async def test_a_value_model_answer_with_no_usable_text_is_diagnosed_too(self) -> None:
        """Not every empty value is an exception: the value model can also just answer with nothing usable."""
        slot_model = _slot_model([_answers("TYPE_TEXT", "none")], goal_value_cache=False)

        async def _empty(*_args: Any, **_kwargs: Any) -> None:
            return None

        with mock.patch.object(BrowserDecisionModel, "_generate_value", _empty):
            message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        summary = json.loads(message.content)
        self.assertEqual(summary["status"], "BLOCKED")
        self.assertIn("value model returned no usable text", summary["reason"])

    async def test_prefetched_value_does_not_cross_a_document_boundary(self) -> None:
        slot_model = _slot_model([_answers("CLICK", "none")], goal_value_cache=False)
        await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        run = slot_model._run
        self.assertTrue(run.prefetched, "the empty 'Where to?' field is prefetched on the first probe")
        self.assertTrue(all(key.startswith("flights.test/|") for key in run.prefetched))
        stale_task = next(iter(run.prefetched.values()))

        other_page = json.loads(json.dumps(_SNAPSHOT))
        other_page["url"] = "https://flights.test/results?q=1"
        other_page["page_key"] = "k2"
        slot_model._prefetch_values(run, other_page)

        self.assertTrue(run.prefetched, "the new document's editable field is prefetched")
        self.assertTrue(
            all(key.startswith("flights.test/results|") for key in run.prefetched),
            "the previous document's prefetches must be gone",
        )
        with contextlib.suppress(asyncio.CancelledError):
            await stale_task
        self.assertTrue(stale_task.cancelled() or stale_task.done())


class TestJevPrefetchKey(IsolatedAsyncioTestCase):
    async def test_a_re_stamped_field_keeps_its_prefetched_value(self) -> None:
        class Runtime:
            calls: list[dict[str, Any]] = []

            async def probe_for_policy(self, source: str, params: dict[str, Any]) -> dict[str, Any]:
                self.calls.append(params)
                snapshot = json.loads(json.dumps(_SNAPSHOT))
                if len(self.calls) > 1:  # a re-render dropped the stamps: the same field under a new node
                    snapshot["page_key"] = "k2"
                    snapshot["elements"][1].update(node=9, target_id="t_g1_9")
                return snapshot

        slot_model = _slot_model(
            [_answers("CLICK", "none"), _answers("CLICK", "none")], goal_value_cache=False, runtime=Runtime()
        )

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        task = next(iter(slot_model._run.prefetched.values()))
        await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(list(slot_model._run.prefetched.values()), [task])


class TestJevPrefetchSwitch(IsolatedAsyncioTestCase):
    """``prefetch_values`` decides whether typed values are generated ahead of time or when TYPE_TEXT is chosen."""

    @staticmethod
    def _counting_generate(calls: list[str]) -> Any:
        original = BrowserDecisionModel._generate_value

        async def counting(self: BrowserDecisionModel, run: Any, item: dict[str, Any], snapshot: dict[str, Any]) -> Any:
            calls.append(str(item.get("label")))
            return await original(self, run, item, snapshot)

        return counting

    async def test_off_generates_the_value_only_when_type_text_is_chosen(self) -> None:
        runtime = _ScriptedPageKeyRuntime(["k1", "k2"])
        slot_model = _slot_model(
            [_answers("CLICK", "none"), _answers("TYPE_TEXT", "none")],
            goal_value_cache=False,
            prefetch_values=False,
            runtime=runtime,
        )
        calls: list[str] = []

        with mock.patch.object(BrowserDecisionModel, "_generate_value", self._counting_generate(calls)):
            await slot_model.invoke(_MESSAGES, tools=_TOOLS)
            self.assertEqual(calls, [], "no background value calls while prefetch is off")
            message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(message.tool_calls[0].name, "browser_type")
        self.assertEqual(calls, ["Where to?"], "the value is generated once, when TYPE_TEXT is chosen")
        self.assertEqual(slot_model.ticks[-1]["value_source"], "llm")

    async def test_on_reuses_the_value_across_page_key_changes_within_one_document(self) -> None:
        runtime = _ScriptedPageKeyRuntime(["k1", "k2"])
        slot_model = _slot_model(
            [_answers("CLICK", "none"), _answers("TYPE_TEXT", "none")], goal_value_cache=False, runtime=runtime
        )
        calls: list[str] = []

        with mock.patch.object(BrowserDecisionModel, "_generate_value", self._counting_generate(calls)):
            await slot_model.invoke(_MESSAGES, tools=_TOOLS)
            message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(message.tool_calls[0].name, "browser_type")
        self.assertEqual(calls, ["Where to?"], "the value from the first probe is reused, not regenerated per page_key")
        self.assertEqual(slot_model.ticks[-1]["value_source"], "prefetch")


class TestJevGoalValueExtractionDegradesGracefully(IsolatedAsyncioTestCase):
    """Hole 2 (completes B5): an unguarded ``values_task.result()`` must not crash the run."""

    async def test_a_failed_values_task_degrades_to_an_empty_value_list(self) -> None:
        slot_model = _slot_model([_answers("TYPE_TEXT", "none")], goal_value_cache=True)

        async def _raise(self: BrowserDecisionModel, goal: str) -> list[str]:
            raise RuntimeError("value extraction blew up")

        with mock.patch.object(BrowserDecisionModel, "_extract_values", _raise):
            message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        # `_raise` has no internal `await`, so by the time `_decide_message` reads `.result()` the
        # task -- scheduled ahead of several other awaits in the same call -- has already failed.
        self.assertTrue(slot_model._run.values_task.done())
        self.assertEqual(slot_model._run.values, [])
        self.assertEqual(message.tool_calls[0].name, "browser_type")
        self.assertEqual(
            json.loads(message.tool_calls[0].arguments)["text"], "London", "the per-field prefetch path still fills"
        )


class TestDecisionPolicyModelProtocol(TestCase):
    """B6: the subagent layer recognises decision policies structurally, not by concrete class."""

    def test_the_browser_decision_model_satisfies_the_protocol(self) -> None:
        slot_model = _slot_model([])
        self.assertIsInstance(slot_model, DecisionPolicyModel)

    def test_a_minimal_stub_also_satisfies_the_protocol(self) -> None:
        class _Stub:
            def bind_runtime(self, runtime: Any) -> None:
                return None

        self.assertIsInstance(_Stub(), DecisionPolicyModel)

    def test_an_object_without_bind_runtime_does_not_satisfy_the_protocol(self) -> None:
        class _NotAPolicy:
            pass

        self.assertNotIsInstance(_NotAPolicy(), DecisionPolicyModel)


class TestBrowserToolNamesResolveBySuffix(IsolatedAsyncioTestCase):
    async def test_prefixed_mcp_tool_names_gate_the_turn_and_name_the_call(self) -> None:
        tools = [
            ToolInfo(name="mcp_playwright-official_browser_click"),
            ToolInfo(name="mcp_playwright-official_browser_type"),
            ToolInfo(name="mcp_playwright-official_browser_select_option"),
            ToolInfo(name="mcp_playwright-official_browser_press_key"),
        ]
        slot_model = _slot_model([_answers("TYPE_TEXT", "none")])

        message = await slot_model.invoke(_MESSAGES, tools=tools)

        self.assertEqual(message.tool_calls[0].name, "mcp_playwright-official_browser_type")

    async def test_scroll_and_select_are_not_offered_without_their_tools(self) -> None:
        offered = ["CLICK", "TYPE_TEXT", "WAIT", "DONE", "BLOCKED"]
        heads = {"click_target": _answer("1", ["1", "2"]), "type_text_target": _answer("2", ["2"])}
        slot_model = _slot_model(
            [{"answers": {"operation": _answer("DONE", offered), **heads}}], goal_value_cache=False
        )

        message = await slot_model.invoke(
            _MESSAGES, tools=[ToolInfo(name="browser_click"), ToolInfo(name="browser_type")]
        )

        self.assertEqual(json.loads(message.content)["status"], "DONE")
        questions = _wire(slot_model).bodies[0]["questions"]
        self.assertEqual(list(questions["operation"]["criteria"]), offered)
        self.assertNotIn("select_target", questions)

    async def test_type_text_is_not_offered_without_the_type_tool(self) -> None:
        offered = ["CLICK", "WAIT", "DONE", "BLOCKED"]
        scripted = [{"answers": {"operation": _answer("DONE", offered), "click_target": _answer("1", ["1", "2"])}}]
        slot_model = _slot_model(scripted, goal_value_cache=False)

        message = await slot_model.invoke(_MESSAGES, tools=[ToolInfo(name="mcp_pw_browser_click")])

        self.assertEqual(json.loads(message.content)["status"], "DONE")
        questions = _wire(slot_model).bodies[0]["questions"]
        self.assertEqual(list(questions["operation"]["criteria"]), offered)
        self.assertNotIn("type_text_target", questions)


class TestBatchedActions(IsolatedAsyncioTestCase):
    _TOOLS = [
        ToolInfo(name="mcp_playwright-official_browser_click"),
        ToolInfo(name="mcp_playwright-official_browser_type"),
        ToolInfo(name="mcp_playwright-official_browser_select_option"),
        ToolInfo(name="mcp_playwright-official_browser_press_key"),
        ToolInfo(name="mcp_playwright-official_browser_run_code_unsafe"),
    ]

    async def test_a_batched_scroll_presses_the_key_without_a_target(self) -> None:
        slot_model = _slot_model([_op_answer("SCROLL_DOWN")], goal_value_cache=False, batch_actions=True)

        message = await slot_model.invoke(_MESSAGES, tools=self._TOOLS)

        code = json.loads(message.tool_calls[0].arguments)["code"]
        self.assertIn('page.keyboard.press("PageDown")', code)
        self.assertNotIn("target not found after re-stamp", code)
        self.assertNotIn('="None"', code)

    async def test_action_and_next_probe_travel_in_one_run_code_call(self) -> None:
        slot_model = _slot_model([_answers("TYPE_TEXT", "none"), _answers("DONE", "none")], batch_actions=True)

        first = await slot_model.invoke(_MESSAGES, tools=self._TOOLS)
        call = first.tool_calls[0]
        code = json.loads(call.arguments)["code"]
        self.assertEqual(call.name, "mcp_playwright-official_browser_run_code_unsafe")
        self.assertIn('page.locator("[data-opens1a=\\"2\\"]")', code)
        self.assertIn('await target.fill("London"', code)
        self.assertIn("target not found after re-stamp", code)
        self.assertIn("page.evaluate(", code)
        self.assertIn('"after": {"kind": "fill", "node": 2}', code)
        probes_before = len(_decision_probes(slot_model._runtime))

        returned = json.loads(json.dumps(_SNAPSHOT))
        returned["page_key"] = "changed"
        for item in returned["elements"]:
            item.pop("target_id", None)
        messages = [*_MESSAGES, {"role": "tool", "tool_call_id": call.id, "content": json.dumps(returned)}]
        second = await slot_model.invoke(messages, tools=self._TOOLS)

        self.assertEqual(
            len(_decision_probes(slot_model._runtime)), probes_before, "the returned probe replaces a runtime probe"
        )
        self.assertEqual(json.loads(second.content)["status"], "DONE")
        self.assertTrue(slot_model._run.history[0]["page_changed"])

    async def test_without_the_run_code_tool_actions_stay_separate(self) -> None:
        slot_model = _slot_model([_answers("TYPE_TEXT", "none")], batch_actions=True)
        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        self.assertEqual(message.tool_calls[0].name, "browser_type")


class TestJevAnswerOnDone(IsolatedAsyncioTestCase):
    """DONE ends with one whole-page probe and one chat call whose reply is the task's answer."""

    async def test_done_answers_from_the_whole_page(self) -> None:
        runtime = _FakeRuntime()
        slot_model = _slot_model([_op_answer("DONE")], goal_value_cache=False, runtime=runtime)

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        summary = json.loads(message.content)
        self.assertEqual((summary["status"], summary["answer"]), ("DONE", "Flights from 120 CHF."))
        answer_probe = runtime.calls[-1]
        self.assertTrue(answer_probe["all_text"])
        self.assertEqual(answer_probe["text_chars"], FINAL_TEXT_CHARS)
        asked = [m for m in slot_model._fallback.asked if "final page" in m[0]["content"]]
        self.assertEqual(len(asked), 1)
        payload = json.loads(asked[0][1]["content"])
        self.assertEqual(payload, {"task": "Fly to London", "page_text": _SNAPSHOT["text"]})

    async def test_a_failed_answer_call_still_returns_the_summary(self) -> None:
        slot_model = _slot_model([_op_answer("DONE")], goal_value_cache=False)
        original = slot_model._fallback.invoke

        async def invoke(messages: list[dict[str, str]], **kwargs: Any) -> Any:
            if "final page" in messages[0]["content"]:
                raise RuntimeError("chat model down")
            return await original(messages, **kwargs)

        slot_model._fallback.invoke = invoke

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        summary = json.loads(message.content)
        self.assertEqual((summary["status"], summary["answer"]), ("DONE", ""))
        self.assertTrue(slot_model._run.finished)

    async def test_blocked_makes_no_answer_call(self) -> None:
        runtime = _FakeRuntime()
        slot_model = _slot_model([_op_answer("BLOCKED")], goal_value_cache=False, runtime=runtime)

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(
            (json.loads(message.content)["status"], json.loads(message.content)["answer"]), ("BLOCKED", "")
        )
        self.assertEqual([c for c in runtime.calls if c.get("all_text")], [])
        self.assertEqual([m for m in slot_model._fallback.asked if "final page" in m[0]["content"]], [])


class TestJevLoopGuards(IsolatedAsyncioTestCase):
    """A scroll that moved nothing leaves the offer; an A-B-A-B loop ends BLOCKED; BLOCKED after progress still answers."""

    async def test_a_scroll_that_moved_nothing_is_not_offered_again(self) -> None:
        offered_after = [op for op in _OPERATIONS if op != "SCROLL_DOWN"]
        second = _answers("DONE", "none")
        second["answers"]["operation"] = _answer("DONE", offered_after)  # the offer shrank: so does the head
        scripted = [_op_answer("SCROLL_DOWN"), second]
        runtime = _ScriptedPageKeyRuntime(["k1"] * 64)  # the page never changes: the scroll hit the bottom
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=runtime)

        first = await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        self.assertEqual(first.tool_calls[0].name, "browser_press_key")
        second = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        self.assertEqual(json.loads(second.content)["status"], "DONE")
        self.assertIn("SCROLL_DOWN", _wire(slot_model).bodies[0]["questions"]["operation"]["criteria"])
        self.assertNotIn("SCROLL_DOWN", _wire(slot_model).bodies[1]["questions"]["operation"]["criteria"])

    async def test_alternating_between_two_targets_ends_blocked_with_the_pages_answer(self) -> None:
        def click(target: str) -> dict[str, Any]:
            payload = _answers("CLICK", "none")
            payload["answers"]["click_target"] = _answer(target, ["1", "2"])
            return payload

        scripted = [click("1"), click("2"), click("1"), click("2")]
        runtime = _ScriptedPageKeyRuntime(["k1", "k2", "k3", "k3"])  # the repeated pair moves nothing
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=runtime)

        for _ in range(4):
            message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)
            self.assertEqual(message.tool_calls[0].name, "browser_click")
        final = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        summary = json.loads(final.content)
        self.assertEqual((summary["status"], summary["reason"]), ("BLOCKED", "oscillating between two pages"))
        self.assertEqual(summary["answer"], "Flights from 120 CHF.")
        self.assertEqual(len(_wire(slot_model).bodies), 4, "the guard fires before another decision is paid")

    async def test_alternating_targets_that_keep_changing_the_page_are_not_oscillation(self) -> None:
        def click(target: str) -> dict[str, Any]:
            return {"answers": {**_answers("CLICK", "none")["answers"], "click_target": _answer(target, ["1", "2"])}}

        scripted = [click("1"), click("2"), click("1"), click("2"), click("1")]
        runtime = _ScriptedPageKeyRuntime([f"k{i}" for i in range(1, 40)])  # a corrected search: every step lands
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=runtime)

        for _ in range(5):
            self.assertEqual((await slot_model.invoke(_MESSAGES, tools=_TOOLS)).tool_calls[0].name, "browser_click")

    async def test_a_blocked_verdict_after_progress_answers_from_the_page_reached(self) -> None:
        scripted = [_answers("CLICK", "none"), _op_answer("BLOCKED")]
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=_ScriptedPageKeyRuntime(["k1", "k2", "k2"]))

        await slot_model.invoke(_MESSAGES, tools=_TOOLS)
        summary = json.loads((await slot_model.invoke(_MESSAGES, tools=_TOOLS)).content)

        self.assertEqual((summary["status"], summary["answer"]), ("BLOCKED", "Flights from 120 CHF."))

    async def test_blocked_before_any_page_change_has_no_answer(self) -> None:
        slot_model = _slot_model([_op_answer("BLOCKED")], goal_value_cache=False)

        summary = json.loads((await slot_model.invoke(_MESSAGES, tools=_TOOLS)).content)

        self.assertEqual((summary["status"], summary["answer"]), ("BLOCKED", ""))
        self.assertFalse(any("final page" in asked[0]["content"] for asked in slot_model._fallback.asked))

    async def test_filling_five_fields_on_one_url_is_not_a_revisit(self) -> None:
        scripted = [_answers("TYPE_TEXT", "none") for _ in range(5)]
        runtime = _ScriptedPageKeyRuntime([f"k{i}" for i in range(1, 40)])  # every fill changes the page key
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=runtime)

        for _ in range(5):
            self.assertEqual((await slot_model.invoke(_MESSAGES, tools=_TOOLS)).tool_calls[0].name, "browser_type")

    async def test_arriving_at_the_same_url_a_fourth_time_ends_blocked_with_the_pages_answer(self) -> None:
        scripted = [_answers("CLICK", "none") for _ in range(6)]
        runtime = _ScriptedUrlRuntime(["https://a.test/", "https://b.test/"] * 4)
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=runtime)

        for _ in range(6):
            self.assertEqual((await slot_model.invoke(_MESSAGES, tools=_TOOLS)).tool_calls[0].name, "browser_click")
        summary = json.loads((await slot_model.invoke(_MESSAGES, tools=_TOOLS)).content)

        self.assertEqual((summary["status"], summary["reason"]), ("BLOCKED", "reached the same page four times"))
        self.assertEqual(summary["answer"], "Flights from 120 CHF.")
        self.assertEqual(len(_wire(slot_model).bodies), 6, "the guard fires before another decision is paid")

    async def test_a_fourth_arrival_inside_a_wait_streak_ends_blocked_too(self) -> None:
        # The sixth turn answers WAIT on b; its settle probe lands on a for the fourth time.
        scripted = [*(_answers("CLICK", "none") for _ in range(5)), _op_answer("WAIT")]
        runtime = _ScriptedUrlRuntime(["https://a.test/", "https://b.test/"] * 3 + ["https://a.test/"] * 2)
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=runtime)

        for _ in range(5):
            self.assertEqual((await slot_model.invoke(_MESSAGES, tools=_TOOLS)).tool_calls[0].name, "browser_click")
        summary = json.loads((await slot_model.invoke(_MESSAGES, tools=_TOOLS)).content)

        self.assertEqual((summary["status"], summary["reason"]), ("BLOCKED", "reached the same page four times"))
        self.assertEqual(len(_wire(slot_model).bodies), 6, "the guard fires before another decision is paid")

    async def test_transient_probe_failures_on_one_url_are_not_visits(self) -> None:
        # Every WAIT's first settle probe fails and its second lands on the same URL with a fresh page_key.
        runtime = _ScriptedPageKeyRuntime(["k1", None, "k2", None, "k3", None, "k4"])
        scripted = [_op_answer("WAIT"), _op_answer("WAIT"), _op_answer("WAIT"), _op_answer("DONE")]
        slot_model = _slot_model(scripted, goal_value_cache=False, runtime=runtime)

        summary = json.loads((await slot_model.invoke(_MESSAGES, tools=_TOOLS)).content)

        self.assertEqual(summary["status"], "DONE")
        self.assertEqual(slot_model._run.url_visits, {"https://flights.test/": 1})
        self.assertEqual(slot_model._run.last_url, "https://flights.test/")


class _SnapshotRuntime:
    """A runtime whose every probe returns one given snapshot."""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.calls: list[dict[str, Any]] = []

    async def probe_for_policy(self, source: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(params)
        return json.loads(json.dumps(self.snapshot))


def _filled_search(value: str = "vegetarian lasagna", **overrides: Any) -> dict[str, Any]:
    """The stock page with one element: a search box holding ``value``."""
    item = {"node": 2, "target_id": "t_g1_2", "role": "searchbox", "label": "Search", "value": value, "editable": True}
    return {**_SNAPSHOT, "elements": [{**item, **overrides}]}


_SUBMIT_OPERATIONS = ["CLICK", "TYPE_TEXT", "PRESS_ENTER", "SCROLL_DOWN", "WAIT", "DONE", "BLOCKED"]


def _press_enter_answers() -> dict[str, Any]:
    return {
        "answers": {
            "operation": _answer("PRESS_ENTER", _SUBMIT_OPERATIONS),
            "press_enter_target": _answer("1", ["1"]),
            "click_target": _answer("1", ["1"]),
            "type_text_target": _answer("1", ["1"]),
        }
    }


class TestPressEnter(IsolatedAsyncioTestCase):
    def test_press_enter_is_offered_for_a_filled_text_field_only(self) -> None:
        space = action_space.build_action_space(_filled_search(), [])
        self.assertIn("PRESS_ENTER", space.operations)
        self.assertEqual(list(space.heads["PRESS_ENTER"]), ["1"])
        self.assertEqual(space.elements[0]["operations"], ["TYPE_TEXT", "PRESS_ENTER", "CLICK"])
        self.assertEqual(action_space.build_action_space(_SNAPSHOT, []).operations, _OPERATIONS, "empty combobox")

    def test_press_enter_needs_an_editable_text_role_and_a_value(self) -> None:
        def offered(**overrides: Any) -> bool:
            return "PRESS_ENTER" in action_space.build_action_space(_filled_search(**overrides), []).operations

        self.assertTrue(offered())
        self.assertTrue(offered(role="textbox"))
        self.assertTrue(offered(role="combobox"))
        self.assertFalse(offered(value=""), "an empty field has nothing to submit")
        self.assertFalse(offered(value="   "), "whitespace is not a value")
        self.assertFalse(offered(editable=False), "a read-only field is not typed into")
        self.assertFalse(offered(role="button"), "Enter on a button is a click")
        self.assertFalse(offered(actionable=False), "an occluded field is not offered at all")

    def test_press_enter_asks_for_its_target(self) -> None:
        space = action_space.build_action_space(_filled_search(), [])
        questions = action_space.build_questions(space, goal="Find pasta", values=[], rules="r", language="en")
        self.assertIn("PRESS_ENTER", questions["operation"].options)
        self.assertEqual(questions["press_enter_target"].options["1"]["element"], "[1] Search")

    async def test_unbatched_press_enter_sends_the_key_alone(self) -> None:
        """@playwright/mcp's browser_press_key takes a key and nothing else; it lands on the focused field."""
        slot_model = _slot_model(
            [_press_enter_answers()], goal_value_cache=False, runtime=_SnapshotRuntime(_filled_search())
        )

        message = await slot_model.invoke(_MESSAGES, tools=_TOOLS)

        call = message.tool_calls[0]
        self.assertEqual((call.name, json.loads(call.arguments)), ("browser_press_key", {"key": "Enter"}))
        self.assertEqual((slot_model.ticks[0]["operation"], slot_model.ticks[0]["target"]), ("PRESS_ENTER", "Search"))
        self.assertEqual(slot_model._run.history[-1]["kind"], "submit")

    async def test_batched_press_enter_presses_on_the_field_itself(self) -> None:
        """Batched, Enter goes to the field's own locator, whatever holds focus."""
        slot_model = _slot_model(
            [_press_enter_answers()],
            goal_value_cache=False,
            batch_actions=True,
            runtime=_SnapshotRuntime(_filled_search()),
        )

        message = await slot_model.invoke(_MESSAGES, tools=TestBatchedActions._TOOLS)

        code = json.loads(message.tool_calls[0].arguments)["code"]
        self.assertIn('await target.press("Enter"', code)
        self.assertNotIn("page.keyboard.press", code)
        self.assertIn('"after": {"kind": "submit", "node": 2}', code)

    async def test_press_enter_is_dropped_without_the_key_tool(self) -> None:
        # no press_key tool: no PRESS_ENTER and no scroll either; no select element on the page
        answers = {
            "answers": {
                "operation": _answer("DONE", ["CLICK", "TYPE_TEXT", "WAIT", "DONE", "BLOCKED"]),
                "click_target": _answer("1", ["1"]),
                "type_text_target": _answer("1", ["1"]),
            }
        }
        slot_model = _slot_model([answers], goal_value_cache=False, runtime=_SnapshotRuntime(_filled_search()))
        tools = [ToolInfo(name=name) for name in ("browser_click", "browser_type", "browser_select_option")]

        await slot_model.invoke(_MESSAGES, tools=tools)

        questions = _wire(slot_model).bodies[0]["questions"]
        self.assertNotIn("PRESS_ENTER", questions["operation"]["criteria"])
        self.assertNotIn("press_enter_target", questions)


class TestDeadTargets(TestCase):
    def test_a_twice_dead_click_target_stops_being_offered(self) -> None:
        history = [
            {"action": "Search the site", "kind": "click", "text": None, "page_changed": False},
            {"action": "Search the site", "kind": "click", "text": None, "page_changed": False},
        ]
        self.assertEqual(action_space.dead_targets(history), {"Search the site"})
        snapshot = {
            **_SNAPSHOT,
            "elements": [
                {
                    "node": 1,
                    "target_id": "t_g1_1",
                    "role": "button",
                    "label": "Search the site",
                    "value": "",
                    "editable": False,
                },
                {
                    "node": 2,
                    "target_id": "t_g1_2",
                    "role": "link",
                    "label": "Quick Pasta Sauce",
                    "value": "",
                    "editable": False,
                },
            ],
        }
        space = action_space.build_action_space(snapshot, history)
        self.assertEqual(list(space.heads["CLICK"]), ["2"], "the dead target is not a click candidate")
        self.assertTrue(space.elements[0]["click_did_nothing"], "its row stays, and says why")
        self.assertEqual(space.elements[0]["operations"], [])
        self.assertNotIn("click_did_nothing", space.elements[1])

    def test_a_dead_target_returns_once_a_click_works_again(self) -> None:
        history = [
            {"action": "Search the site", "kind": "click", "text": None, "page_changed": False},
            {"action": "Search the site", "kind": "click", "text": None, "page_changed": False},
            {"action": "Search the site", "kind": "click", "text": None, "page_changed": True},
        ]
        self.assertEqual(action_space.dead_targets(history), set())

    def test_dead_target_detection_ignores_other_kinds_and_short_streaks(self) -> None:
        one_failure = [{"action": "Search", "kind": "click", "text": None, "page_changed": False}]
        self.assertEqual(action_space.dead_targets(one_failure), set(), "one failure is not evidence")
        typing = [
            {"action": "Search", "kind": "fill", "text": "x", "page_changed": False},
            {"action": "Search", "kind": "fill", "text": "y", "page_changed": False},
        ]
        self.assertEqual(action_space.dead_targets(typing), set(), "a fill that changes nothing is normal")
        self.assertEqual(action_space.dead_targets([]), set())
