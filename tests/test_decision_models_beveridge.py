# coding: utf-8
"""``beveridge`` adapter: contract tests against a stub /v1/decide server and a fake bev.load (no weights, no GPU)."""

from __future__ import annotations

import json
import os
import types
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

import decision_model_contract as contract
from s1a.decision_models import ChoiceQuestion, DecisionModel, Image, NoulQuestion, Observation, build_model
from s1a.decision_models.beveridge import (
    DEFAULT_BEV_MODEL,
    MAX_BEV_QUESTIONS,
    PRO_BEV_MODEL,
    WEB_BEV_MODEL,
    BeveridgeClient,
    BeveridgeModel,
    beveridge_question,
    resolve_bev_model,
)

SCREEN = Image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24, media_type="image/png")
URL = "http://bev.test"


def _choice_answer(question: dict[str, Any]) -> dict[str, Any]:
    keys = list(question["criteria"])
    if not keys:
        return {"choice": "", "probabilities": {}, "confidence": 0.0}
    if len(keys) == 1:
        probabilities = {keys[0]: 1.0}
    else:
        rest = 0.2 / (len(keys) - 1)
        probabilities = {key: (0.8 if i == 0 else rest) for i, key in enumerate(keys)}
    return {"choice": keys[0], "probabilities": probabilities, "confidence": probabilities[keys[0]]}


def _noul_answer(_question: dict[str, Any]) -> dict[str, Any]:
    return {"noul": 0.65, "probabilities": {"true": 0.65, "false": 0.35}}


def _decision_body(request: httpx.Request) -> dict[str, Any]:
    body = json.loads(request.content)
    answers = {}
    for name, question in body["questions"].items():
        answers[name] = _noul_answer(question) if question["type"] == "noul" else _choice_answer(question)
    return {"answers": answers, "usage": {"input_tokens": 12, "output_tokens": 0}}


class Server:
    """Scripted local ``/v1/decide`` + ``/health``."""

    def __init__(self, *, health: Any = None, script: list[Any] | None = None) -> None:
        self.health = health if health is not None else {"ok": True}
        self.script = list(script or [])
        self.decisions: list[httpx.Request] = []
        self.health_calls = 0

    def answer(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_decision_body(request))

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            self.health_calls += 1
            if isinstance(self.health, Exception):
                raise self.health
            if isinstance(self.health, httpx.Response):
                return self.health
            return httpx.Response(200, json=self.health)
        self.decisions.append(request)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if callable(step):
                return step(request)
            return step
        return self.answer(request)


def make_served(server: Server, *, timeout_s: float = 5.0, model: str = DEFAULT_BEV_MODEL) -> BeveridgeModel:
    client = BeveridgeClient(
        url=URL + "/",
        endpoint=URL + "/v1/decide",
        timeout_s=timeout_s,
        transport=httpx.MockTransport(server.handler),
        health_url=URL + "/health",
    )
    return BeveridgeModel(client=client, model=model)


class FakeBev:
    """Stand-in for ``bev.load()``: same ``decide(state, questions)`` return shape, no weights."""

    def __init__(self, *, model_name: str = DEFAULT_BEV_MODEL, script: list[Any] | None = None) -> None:
        self.model_name = model_name
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.script = list(script or [])

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.calls.append((state, questions))
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step
        answers = {}
        for name, question in questions.items():
            answers[name] = _noul_answer(question) if question["type"] == "noul" else _choice_answer(question)
        return answers


def make_in_process(agent: FakeBev | None = None, *, model: str = DEFAULT_BEV_MODEL) -> BeveridgeModel:
    return BeveridgeModel(agent=agent or FakeBev(model_name=model), model=model)


class TestBeveridgeServedContract(contract.DecisionModelContract, IsolatedAsyncioTestCase):
    def make(self) -> DecisionModel:
        return make_served(Server())

    def make_scripted(self, answers: list[dict[str, Any]]) -> DecisionModel:
        script = [httpx.Response(200, json={"answers": a, "usage": {"input_tokens": 8}}) for a in answers]
        return make_served(Server(script=script))


class TestBeveridgeInProcessContract(contract.DecisionModelContract, IsolatedAsyncioTestCase):
    def make(self) -> DecisionModel:
        return make_in_process()

    def make_scripted(self, answers: list[dict[str, Any]]) -> DecisionModel:
        return make_in_process(FakeBev(script=list(answers)))


class TestRequestBody(IsolatedAsyncioTestCase):
    async def test_served_posts_v1_decide_without_images_or_systemone_fields(self) -> None:
        server = Server()
        pictured = Observation(contract.OBSERVATION.state, images=(SCREEN,))
        await make_served(server).decide_many(pictured, {"pick": contract.PICK, "check": contract.CHECK})
        request = server.decisions[0]
        self.assertEqual(request.url, httpx.URL(URL + "/v1/decide"))
        body = json.loads(request.content)
        self.assertEqual(body["state"], contract.OBSERVATION.state)
        self.assertNotIn("model", body)
        self.assertNotIn("images", body)
        self.assertNotIn("media", body)
        self.assertEqual(body["questions"]["pick"]["type"], "choice")
        self.assertEqual(body["questions"]["check"]["type"], "noul")
        self.assertIsInstance(body["questions"]["pick"]["instructions"], str)
        self.assertIn("stand on 17 or more", body["questions"]["pick"]["instructions"])
        self.assertEqual(body["questions"]["check"]["instructions"], "Does the player stand?")
        self.assertEqual(
            body["questions"]["check"]["criteria"], {"true": "the player stands", "false": "the player hits"}
        )
        self.assertIn("X-Request-Id", request.headers)

    async def test_provenance_records_served_path_and_requested_model(self) -> None:
        decision = await make_served(Server()).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(decision.raw["url"], URL)
        self.assertIsInstance(decision.raw["request_id"], str)
        self.assertEqual(decision.raw["requested_model"], DEFAULT_BEV_MODEL)
        self.assertEqual(decision.raw["served_by"]["path"], "served")
        self.assertEqual(decision.raw["served_by"]["checkpoint"], DEFAULT_BEV_MODEL)
        self.assertEqual(decision.model, DEFAULT_BEV_MODEL)
        self.assertEqual(decision.usage.input_tokens, 12)
        self.assertEqual(decision.choice("pick").key, "hit")

    async def test_served_echoes_the_model_name_from_the_payload(self) -> None:
        def with_model(request: httpx.Request) -> httpx.Response:
            payload = _decision_body(request)
            payload["model"] = "sathiskumarjothi0/Beveridge-Pro-C"
            return httpx.Response(200, json=payload)

        decision = await make_served(Server(script=[with_model])).decide_many(
            contract.OBSERVATION, {"pick": contract.PICK}
        )
        self.assertEqual(decision.model, "sathiskumarjothi0/Beveridge-Pro-C")

    async def test_in_process_calls_decide_and_records_path(self) -> None:
        agent = FakeBev(model_name="echoed-large-c")
        decision = await make_in_process(agent).decide_many(
            contract.OBSERVATION, {"pick": contract.PICK, "check": contract.CHECK}
        )
        self.assertEqual(len(agent.calls), 1)
        state, asked = agent.calls[0]
        self.assertEqual(state, contract.OBSERVATION.state)
        self.assertEqual(asked["pick"]["type"], "choice")
        self.assertEqual(asked["check"]["type"], "noul")
        self.assertNotIn("images", asked)
        self.assertEqual(decision.raw["served_by"]["path"], "in-process")
        self.assertEqual(decision.raw["requested_model"], DEFAULT_BEV_MODEL)
        self.assertEqual(decision.model, "echoed-large-c")
        self.assertEqual(decision.choice("pick").key, "hit")

    async def test_browser_element_row_becomes_label(self) -> None:
        server = Server()
        question = ChoiceQuestion(
            {"12": {"element": "[12] Where from?", "current_value": "Zurich", "role": "combobox"}},
            goal="book a flight",
        )
        await make_served(server).decide_many(contract.OBSERVATION, {"click_target": question})
        asked = json.loads(server.decisions[0].content)["questions"]["click_target"]
        self.assertEqual(asked["criteria"], {"12": "Where from? = Zurich"})
        self.assertIn("book a flight", asked["instructions"])


class TestFailures(IsolatedAsyncioTestCase):
    async def test_more_than_fifteen_questions_is_a_config_error(self) -> None:
        questions = {f"q{i}": NoulQuestion(f"Is {i} ok?") for i in range(MAX_BEV_QUESTIONS + 1)}
        with self.assertRaises(BaseError) as caught:
            await make_served(Server()).decide_many(contract.OBSERVATION, questions)
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("15", str(caught.exception))

    async def test_fifteen_questions_are_accepted(self) -> None:
        questions = {f"q{i}": NoulQuestion(f"Is {i} ok?") for i in range(MAX_BEV_QUESTIONS)}
        decision = await make_served(Server()).decide_many(contract.OBSERVATION, questions)
        self.assertEqual(len(decision.answers), MAX_BEV_QUESTIONS)

    async def test_transport_error_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.ConnectError("refused"), httpx.ConnectError("refused")])
        with self.assertRaises(BaseError) as caught:
            await make_served(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("beveridge", str(caught.exception).lower())

    async def test_malformed_answers_object_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(200, json={"answers": "nope"})])
        with self.assertRaises(BaseError) as caught:
            await make_served(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("answers", str(caught.exception).lower())

    async def test_missing_answers_object_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(200, json={"model": "x"})])
        with self.assertRaises(BaseError) as caught:
            await make_served(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("answers", str(caught.exception).lower())

    async def test_http_error_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(500, text="boom")])
        with self.assertRaises(BaseError) as caught:
            await make_served(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("500", str(caught.exception))

    async def test_in_process_malformed_answers_fail_closed(self) -> None:
        with self.assertRaises(BaseError) as caught:
            await make_in_process(FakeBev(script=["nope"])).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("answers", str(caught.exception).lower())

    async def test_in_process_runtime_error_is_a_call_failure(self) -> None:
        with self.assertRaises(BaseError) as caught:
            await make_in_process(FakeBev(script=[RuntimeError("cuda")])).decide_many(
                contract.OBSERVATION, {"pick": contract.PICK}
            )
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("forward pass", str(caught.exception))

    async def test_images_are_dropped_and_not_sent(self) -> None:
        server = Server()
        pictured = Observation({"task": "Pay the invoice"}, images=(SCREEN,))
        with patch("s1a.decision_models.base.logger.warning") as warned:
            await make_served(server).decide_many(pictured, {"pick": contract.PICK})
        body = json.loads(server.decisions[0].content)
        self.assertNotIn("images", body)
        warned.assert_called()


class TestMappingHelpers(TestCase):
    def test_aliases_resolve_to_released_checkpoints(self) -> None:
        self.assertEqual(resolve_bev_model(None), DEFAULT_BEV_MODEL)
        self.assertEqual(resolve_bev_model(""), DEFAULT_BEV_MODEL)
        self.assertEqual(resolve_bev_model("large-c"), DEFAULT_BEV_MODEL)
        self.assertEqual(resolve_bev_model("PRO-C"), PRO_BEV_MODEL)
        self.assertEqual(resolve_bev_model("pro-c-web"), WEB_BEV_MODEL)
        self.assertEqual(resolve_bev_model(DEFAULT_BEV_MODEL), DEFAULT_BEV_MODEL)
        self.assertEqual(resolve_bev_model("/models/Beveridge-Large-C"), "/models/Beveridge-Large-C")

    def test_choice_instructions_are_a_string(self) -> None:
        asked = beveridge_question(ChoiceQuestion({"a": "one", "b": "two"}, goal="pick a side", rules="be fair"))
        self.assertEqual(asked["type"], "choice")
        self.assertIsInstance(asked["instructions"], str)
        self.assertIn("pick a side", asked["instructions"])
        self.assertIn("be fair", asked["instructions"])
        self.assertEqual(asked["criteria"], {"a": "one", "b": "two"})


class TestFromEnv(TestCase):
    def test_url_selects_the_served_path(self) -> None:
        with patch.dict(os.environ, {"BEV_URL": "http://127.0.0.1:8000", "BEV_TIMEOUT_S": "5"}, clear=False):
            os.environ.pop("BEV_MODEL", None)
            os.environ.pop("BEV_API_KEY", None)
            model = build_model("beveridge")
        try:
            self.assertIsInstance(model, BeveridgeModel)
            self.assertEqual(model.name, "beveridge")
            self.assertEqual(model.model, DEFAULT_BEV_MODEL)
            self.assertEqual(model.path, "served")
            self.assertFalse(model.supports_images)
            self.assertTrue(model.deterministic)
            self.assertEqual(model._client.endpoint, "http://127.0.0.1:8000/v1/decide")
        finally:
            import asyncio

            asyncio.run(model.close())

    def test_url_wins_over_an_installed_package(self) -> None:
        with patch.dict(os.environ, {"BEV_URL": "http://127.0.0.1:8000", "BEV_MODEL": "pro-c"}, clear=False):
            with patch.dict(
                "sys.modules",
                {
                    "bev": types.SimpleNamespace(
                        load=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load"))
                    )
                },
            ):
                model = build_model("beveridge")
        try:
            self.assertEqual(model.path, "served")
            self.assertEqual(model.model, PRO_BEV_MODEL)
        finally:
            import asyncio

            asyncio.run(model.close())

    def test_missing_package_without_url_is_a_config_error(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BEV_URL", None)
            os.environ.pop("BEV_MODEL", None)
            with patch.dict("sys.modules", {"bev": None}):  # type: ignore[dict-item]
                with self.assertRaises(BaseError) as caught:
                    build_model("beveridge")
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("BEV_URL", str(caught.exception))
        self.assertIn("beveridge", str(caught.exception))

    def test_in_process_uses_fake_load(self) -> None:
        agent = FakeBev(model_name="loaded")

        def fake_load(path_or_repo: str = DEFAULT_BEV_MODEL, device: str | None = None) -> FakeBev:
            self.assertEqual(path_or_repo, WEB_BEV_MODEL)
            self.assertEqual(device, "cpu")
            return agent

        with patch.dict(os.environ, {"BEV_MODEL": "pro-c-web", "BEV_DEVICE": "cpu"}, clear=False):
            os.environ.pop("BEV_URL", None)
            with patch.dict("sys.modules", {"bev": types.SimpleNamespace(load=fake_load)}):
                model = build_model("beveridge")
        self.assertEqual(model.path, "in-process")
        self.assertEqual(model.model, WEB_BEV_MODEL)
        self.assertIs(model._agent, agent)

    def test_bad_timeout_is_a_config_error(self) -> None:
        with patch.dict(os.environ, {"BEV_URL": "http://127.0.0.1:8000", "BEV_TIMEOUT_S": "nope"}):
            with self.assertRaises(BaseError) as caught:
                build_model("beveridge")
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("BEV_TIMEOUT_S", str(caught.exception))
