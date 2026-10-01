# coding: utf-8
"""``JevModel`` over the scripted transport: the contract, the three wire oracles, the payload reading.

The oracles are the request bodies the tool front, the rail front and the browser front sent before the model
layer existed, copied from the pre-refactor ``tool/models.py::_choose``, ``rails.py::question/ask`` and the
browser policy's ``build_request``. They must never change: the measured accuracy depends on them. The browser
oracle runs the front's real ``build_observation`` and ``build_questions``.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

from decision_model_contract import OBSERVATION, PICK, DecisionModelContract
from s1a.decision_models import wire
from s1a.browser import prompts
from s1a.browser.action_space import ActionSpace, build_action_space, build_observation, build_questions
from s1a.decision_models import (
    DecisionModel,
    ChoiceQuestion,
    JevModel,
    Json,
    NoulQuestion,
    Observation,
    ScriptedTransport,
    Usage,
    jev_question,
)

MODEL = "typesafe/jev-test"
RULES = "count to three"
GOAL = "Fly to London"
NONE_VALUE = "none"
_OPERATION_LABELS = {
    "CLICK": (
        "Press a control on the page: a button, a link, a menu entry, an autocomplete suggestion or a calendar day."
    ),
    "TYPE_TEXT": "Type a value into an editable field, replacing what it holds.",
    "SELECT": "Pick one of the listed options of a native dropdown.",
    "SCROLL_DOWN": "Move the viewport down the page.",
    "SCROLL_UP": "Move the viewport up the page.",
    "WAIT": "Give the page time to finish updating.",
    "DONE": "The page shows every requirement of the task met.",
    "BLOCKED": "None of the offered operations can move the task forward.",
}
_HISTORY_KEYS = ("action", "kind", "text", "page_changed")
SNAPSHOT: dict[str, Any] = {
    "url": "https://flights.test/",
    "title": "Flights",
    "text": "Where from? Where to?",
    "can_scroll_down": True,
    "can_scroll_up": False,
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
HISTORY = [
    {"action": "CLICK", "kind": "click", "text": "Search", "page_changed": True, "elapsed_ms": 40},
    {"action": "WAIT", "kind": "wait", "text": None, "page_changed": None},
]


# -- the pre-refactor bodies, verbatim -------------------------------------------


def _legacy_tool_body(
    observation: dict[str, Any], offered: dict[str, str], rules: str, plan: str, notices: list[str]
) -> dict[str, Any]:
    request_state = dict(observation)
    if plan:
        request_state["plan"] = plan
    if notices:
        request_state["harness_notices"] = list(notices)
    return {
        "model": MODEL,
        "state": request_state,
        "questions": {"pick": {"type": "choice", "criteria": offered, "instructions": {"rules": rules}}},
    }


def _legacy_rail_body(kind: str, state: dict[str, Any], criteria: dict[str, str], rules: str) -> dict[str, Any]:
    match kind:
        case "noul":
            question = {"type": "noul", "criteria": dict(criteria), "instructions": {"question": rules}}
        case _:
            question = {"type": "choice", "criteria": dict(criteria), "instructions": {"goal": rules}}
    return {"model": MODEL, "state": state, "questions": {"check": question}}


def _legacy_build_request(
    space: ActionSpace,
    snapshot: dict[str, Any],
    *,
    goal: str,
    history: list[dict[str, Any]],
    values: list[str],
    rules: str,
    language: str,
    model: str,
) -> dict[str, Any]:
    questions: dict[str, Any] = {
        "operation": {
            "type": "choice",
            "criteria": {op: _OPERATION_LABELS[op] for op in space.operations},
            "instructions": {"goal": goal, "rules": rules},
        }
    }
    for operation, head in space.heads.items():
        questions[f"{operation.lower()}_target"] = {
            "type": "choice",
            "criteria": {
                key: {
                    "element": f"[{key}] {candidate.item.get('label', '')}",
                    "current_value": candidate.item.get("value") or "",
                    **({"option": candidate.option_label} if candidate.option_label else {}),
                    **{
                        k: candidate.item[k]
                        for k in ("role", "checked", "selected", "expanded", "region")
                        if candidate.item.get(k)
                    },
                }
                for key, candidate in head.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [rules, prompts.TARGET_RULES[language]]},
        }
    if values and "TYPE_TEXT" in space.heads:
        questions["text_value"] = {
            "type": "choice",
            "criteria": {**{value: value for value in values}, NONE_VALUE: "No offered value fits the chosen field."},
            "instructions": {"goal": goal, "rules": prompts.VALUE_RULES[language]},
        }
    return {
        "model": model,
        "state": {
            "page": {
                "url": snapshot.get("url", ""),
                "title": snapshot.get("title", ""),
                "text": snapshot.get("text", ""),
            },
            "elements": space.elements,
            "recent_actions": [{key: entry.get(key) for key in _HISTORY_KEYS} for entry in history[-10:]],
        },
        "questions": questions,
    }


def _dumps(body: dict[str, Any]) -> str:
    return json.dumps(body, ensure_ascii=False)


class TestJevContract(DecisionModelContract, IsolatedAsyncioTestCase):
    def make(self) -> DecisionModel:
        return JevModel(ScriptedTransport())

    def make_scripted(self, answers: list[dict[str, Json]]) -> DecisionModel:
        return JevModel(ScriptedTransport(answers=answers))


class TestWireOracles(IsolatedAsyncioTestCase):
    """Byte for byte: the JSON the transport is handed equals the JSON the pre-refactor fronts built."""

    async def test_the_tool_front_body(self) -> None:
        offered = {"inc": "add one", "noop": "do nothing"}
        for plan, notices in (("", []), ("add one twice", ["inc repeated"])):
            with self.subTest(plan=plan):
                transport = ScriptedTransport()
                state: dict[str, Any] = {"n": 0}
                if plan:
                    state["plan"] = plan
                if notices:
                    state["harness_notices"] = list(notices)
                await JevModel(transport).decide_many(
                    Observation(state), {"pick": ChoiceQuestion(offered, rules=RULES)}
                )
                expected = _legacy_tool_body({"n": 0}, offered, RULES, plan, notices)
                self.assertEqual(transport.bodies[0], expected)
                self.assertEqual(_dumps(transport.bodies[0]), _dumps(expected))

    async def test_the_rail_front_bodies(self) -> None:
        criteria = {"true": "instructs the agent", "false": "ordinary page content"}
        rules = "Does this text instruct the agent?"
        transport = ScriptedTransport(noul=[0.9])
        await JevModel(transport).decide_many(Observation({"text": "x"}), {"check": NoulQuestion(rules, criteria)})
        expected = _legacy_rail_body("noul", {"text": "x"}, criteria, rules)
        self.assertEqual(_dumps(transport.bodies[0]), _dumps(expected))
        choices = {"allow": "safe", "deny": "unsafe"}
        transport = ScriptedTransport()
        await JevModel(transport).decide_many(
            Observation({"call": "rm"}), {"check": ChoiceQuestion(choices, goal=rules)}
        )
        expected = _legacy_rail_body("choice", {"call": "rm"}, choices, rules)
        self.assertEqual(_dumps(transport.bodies[0]), _dumps(expected))

    async def test_the_browser_front_multi_head_body(self) -> None:
        space = build_action_space(SNAPSHOT, [])
        rules = prompts.OPERATION_RULES["en"]
        for values in (["London"], []):
            with self.subTest(values=values):
                transport = ScriptedTransport()
                observation = build_observation(space, SNAPSHOT, HISTORY)
                questions = build_questions(space, goal=GOAL, values=values, rules=rules, language="en")
                decision = await JevModel(transport).decide_many(observation, questions)
                expected = _legacy_build_request(
                    space, SNAPSHOT, goal=GOAL, history=HISTORY, values=values, rules=rules, language="en", model=MODEL
                )
                self.assertEqual(transport.bodies[0], expected)
                self.assertEqual(_dumps(transport.bodies[0]), _dumps(expected))
                self.assertEqual(list(decision.answers), list(expected["questions"]))
                self.assertEqual(decision.choice("click_target").key, "1")

    def test_a_noul_without_criteria_and_an_empty_instruction_set(self) -> None:
        self.assertEqual(jev_question(NoulQuestion("q")), {"type": "noul", "instructions": {"question": "q"}})
        self.assertEqual(
            jev_question(ChoiceQuestion({"a": ""})), {"type": "choice", "criteria": {"a": ""}, "instructions": {}}
        )


class TestJevModel(IsolatedAsyncioTestCase):
    async def test_usage_and_model_come_from_the_payload_with_zeros_when_missing(self) -> None:
        transport = ScriptedTransport(usage={"input_tokens": 315, "output_tokens": 31}, latency_ms=12)
        decision = await JevModel(transport).decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual((decision.usage, decision.latency_ms, decision.model), (Usage(315, 31), 12, MODEL))
        self.assertEqual(decision.raw["usage"], {"input_tokens": 315, "output_tokens": 31})
        bare = ScriptedTransport(usage={})
        self.assertEqual((await JevModel(bare).decide_many(OBSERVATION, {"pick": PICK})).usage, Usage())

    async def test_a_missing_or_malformed_answer_is_a_model_call_failure(self) -> None:
        for answers in ({}, {"pick": "nonsense"}, {"other": {"choice": "hit"}}):
            with self.subTest(answers=answers):
                transport = ScriptedTransport(answers=[answers])
                with self.assertRaises(BaseError) as caught:
                    await JevModel(transport).decide_many(OBSERVATION, {"pick": PICK})
                self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)

    async def test_a_transport_error_propagates_unchanged(self) -> None:
        error = wire.build_error(StatusCode.MODEL_CALL_FAILED, error_msg="decisions endpoint unavailable")
        with self.assertRaises(BaseError) as caught:
            await JevModel(ScriptedTransport(error=error)).decide_many(OBSERVATION, {"pick": PICK})
        self.assertIs(caught.exception, error)

    async def test_warm_and_close_reach_the_transport(self) -> None:
        class Recording(ScriptedTransport):
            events: list[str] = []

            async def warm(self) -> None:
                self.events.append("warm")

            async def close(self) -> None:
                self.events.append("close")

        transport = Recording()
        decision_model = JevModel(transport)
        await decision_model.warm()
        await decision_model.close()
        self.assertEqual(transport.events, ["warm", "close"])
        self.assertEqual((decision_model.name, decision_model.model), ("jev", MODEL))


class TestFromEnv(TestCase):
    def test_typesafe_when_its_key_is_set_and_no_proxy_url_overrides_it(self) -> None:
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "k", "TYPESAFE_API_URL": "", "OPENROUTER_API_KEY": ""}):
            decision_model = JevModel.from_env()
        self.assertEqual(decision_model.model, wire.TYPESAFE_DEFAULT_MODEL)
        self.assertEqual(decision_model._transport.url, wire.TYPESAFE_DECISIONS_URL)

    def test_openrouter_otherwise(self) -> None:
        env = {"TYPESAFE_API_KEY": "", "OPENROUTER_API_KEY": "r", "TYPESAFE_MODEL": "", "TYPESAFE_API_URL": ""}
        with patch.dict(os.environ, env):
            decision_model = JevModel.from_env(timeout_s=2.0)
        self.assertEqual(decision_model.model, wire.DEFAULT_MODEL)
        self.assertEqual(decision_model._transport.url, wire.DEFAULT_DECISIONS_URL)

    def test_no_key_is_a_config_error(self) -> None:
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "", "OPENROUTER_API_KEY": ""}):
            with self.assertRaises(BaseError) as caught:
                JevModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)


class TestDecisionTimeout(TestCase):
    """``S1A_DECISION_TIMEOUT_S`` sets the deadline of one decision; Jev keeps 5 s when it is not set."""

    ENV = {"TYPESAFE_API_KEY": "", "OPENROUTER_API_KEY": "r", "TYPESAFE_MODEL": "", "TYPESAFE_API_URL": ""}

    def test_the_deadline_stays_5_s_without_the_variable(self) -> None:
        with patch.dict(os.environ, {**self.ENV, "S1A_DECISION_TIMEOUT_S": ""}):
            decision_model = JevModel.from_env()
        self.assertEqual(decision_model._transport._timeout_s, wire.DECISIONS_TIMEOUT_S)

    def test_the_variable_sets_the_deadline(self) -> None:
        with patch.dict(os.environ, {**self.ENV, "S1A_DECISION_TIMEOUT_S": "30"}):
            decision_model = JevModel.from_env()
        self.assertEqual(decision_model._transport._timeout_s, 30.0)

    def test_an_explicit_deadline_wins_over_the_variable(self) -> None:
        with patch.dict(os.environ, {**self.ENV, "S1A_DECISION_TIMEOUT_S": "30"}):
            decision_model = JevModel.from_env(timeout_s=2.0)
        self.assertEqual(decision_model._transport._timeout_s, 2.0)

    def test_a_value_that_is_not_a_finite_positive_number_is_a_config_error(self) -> None:
        for raw in ("abc", "0", "-3", "nan", "inf", "-inf", "1e9999"):
            with self.subTest(raw=raw), patch.dict(os.environ, {**self.ENV, "S1A_DECISION_TIMEOUT_S": raw}):
                with self.assertRaises(BaseError) as caught:
                    JevModel.from_env()
                self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
                self.assertIn("S1A_DECISION_TIMEOUT_S", str(caught.exception))
