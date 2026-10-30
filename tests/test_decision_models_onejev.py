# coding: utf-8
"""``onejev`` served client: contract tests against a fake HTTP server (no GPU / weights)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

import decision_model_contract as contract
from s1a.decision_models import ChoiceQuestion, DecisionModel, Image, Observation, build_model
from s1a.decision_models.onejev import (
    DEFAULT_ONEJEV_MODEL,
    OneJevClient,
    OneJevModel,
    onejev_media,
    onejev_question,
    onejev_state,
)

# Minimal valid-looking PNG header bytes for media encoding tests (not decoded by the stub).
SCREEN = Image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24, media_type="image/png")
PICTURED = Observation({"task": "Pay the invoice"}, images=(SCREEN,))
URL = "http://onejev.test"


def _choice_answer(question: dict[str, Any]) -> dict[str, Any]:
    keys = list(question["criteria"])
    rest = 0.2 / max(1, len(keys) - 1)
    probabilities = {key: (0.8 if i == 0 else rest) for i, key in enumerate(keys)} if keys else {}
    return {"choice": keys[0], "probabilities": probabilities, "confidence": 0.8}


def _noul_answer(_question: dict[str, Any]) -> dict[str, Any]:
    return {"noul": 0.65, "confidence": 0.65}


class Server:
    """Scripted ``/v1/systemone`` + ``/health`` stand-in for qev/flide."""

    def __init__(self, *, health: Any = None, script: list[Any] | None = None) -> None:
        self.health = health if health is not None else {"model": DEFAULT_ONEJEV_MODEL, "device": "cuda"}
        self.script = list(script or [])
        self.decisions: list[httpx.Request] = []
        self.health_calls = 0

    def answer(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        answers = {}
        for name, question in body["questions"].items():
            answers[name] = _noul_answer(question) if question["type"] == "noul" else _choice_answer(question)
        return httpx.Response(
            200,
            json={
                "model": body.get("model") or DEFAULT_ONEJEV_MODEL,
                "answers": answers,
                "usage": {"input_tokens": 12, "output_tokens": 0},
            },
        )

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


def make_model(server: Server, *, timeout_s: float = 5.0, model: str = DEFAULT_ONEJEV_MODEL) -> OneJevModel:
    client = OneJevClient(url=URL + "/", timeout_s=timeout_s, transport=httpx.MockTransport(server.handler))
    return OneJevModel(client, model=model)


class TestOneJevContract(contract.DecisionModelContract, IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = patch.object(contract, "OBSERVATION", PICTURED)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self) -> DecisionModel:
        return make_model(Server())

    def make_scripted(self, answers: list[dict[str, Any]]) -> DecisionModel:
        script = [
            httpx.Response(200, json={"model": DEFAULT_ONEJEV_MODEL, "answers": a, "usage": {"input_tokens": 8}})
            for a in answers
        ]
        return make_model(Server(script=script))


class TestRequestBody(IsolatedAsyncioTestCase):
    async def test_posts_media_and_image_marker(self) -> None:
        server = Server()
        await make_model(server).decide_many(PICTURED, {"pick": contract.PICK, "check": contract.CHECK})
        request = server.decisions[0]
        self.assertEqual(request.url, httpx.URL(URL + "/v1/systemone"))
        body = json.loads(request.content)
        self.assertEqual(body["model"], DEFAULT_ONEJEV_MODEL)
        self.assertEqual(body["state"]["screen"], "<image:1>")
        self.assertEqual(body["state"]["task"], "Pay the invoice")
        self.assertEqual(len(body["media"]), 1)
        self.assertEqual(body["media"][0]["type"], "image")
        self.assertTrue(body["media"][0]["data"].startswith("data:image/png;base64,"))
        decoded = base64.b64decode(body["media"][0]["data"].split(",", 1)[1])
        self.assertEqual(decoded, SCREEN.data)
        self.assertEqual(body["questions"]["pick"]["type"], "choice")
        self.assertEqual(body["questions"]["check"]["type"], "noul")
        self.assertIsInstance(body["questions"]["pick"]["instructions"], str)
        self.assertIn("X-Request-Id", request.headers)

    async def test_provenance_fields_and_no_image_bytes_in_raw(self) -> None:
        decision = await make_model(Server()).decide_many(PICTURED, {"pick": contract.PICK})
        self.assertEqual(decision.raw["url"], URL)
        self.assertIsInstance(decision.raw["request_id"], str)
        self.assertEqual(decision.raw["served_by"]["checkpoint"], DEFAULT_ONEJEV_MODEL)
        self.assertEqual(decision.raw["sent_images"][0]["bytes"], len(SCREEN.data))
        dumped = json.dumps(decision.raw)
        self.assertNotIn(base64.b64encode(SCREEN.data).decode("ascii"), dumped)

    async def test_browser_element_row_becomes_label(self) -> None:
        server = Server()
        question = ChoiceQuestion(
            {"12": {"element": "[12] Where from?", "current_value": "Zurich", "role": "combobox"}},
            goal="book a flight",
        )
        await make_model(server).decide_many(PICTURED, {"click_target": question})
        asked = json.loads(server.decisions[0].content)["questions"]["click_target"]
        self.assertEqual(asked["criteria"], {"12": "Where from? = Zurich"})
        self.assertIn("book a flight", asked["instructions"])


class TestFailures(IsolatedAsyncioTestCase):
    async def test_missing_image_is_a_call_failure(self) -> None:
        with self.assertRaises(BaseError) as caught:
            await make_model(Server()).decide_many(Observation({"page": "x"}), {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("no image", str(caught.exception).lower())

    async def test_transport_error_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.ConnectError("refused"), httpx.ConnectError("refused")])
        with self.assertRaises(BaseError) as caught:
            await make_model(server).decide_many(PICTURED, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("onejev", str(caught.exception).lower())

    async def test_malformed_answers_object_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(200, json={"model": "x", "answers": "nope"})])
        with self.assertRaises(BaseError) as caught:
            await make_model(server).decide_many(PICTURED, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("answers", str(caught.exception).lower())

    async def test_http_error_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(500, text="boom")])
        with self.assertRaises(BaseError) as caught:
            await make_model(server).decide_many(PICTURED, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("500", str(caught.exception))


class TestDecisionDeadline(IsolatedAsyncioTestCase):
    async def test_retry_uses_remaining_budget_for_httpx_and_wait_for(self) -> None:
        for failure in (httpx.ConnectError("refused"), httpx.Response(502), httpx.Response(504)):
            with self.subTest(failure=failure):
                now = 0.0
                requests = []

                def handler(request: httpx.Request) -> httpx.Response:
                    nonlocal now
                    requests.append(request)
                    if len(requests) == 1:
                        now += 1.25
                        if isinstance(failure, Exception):
                            raise failure
                        return failure
                    return httpx.Response(200, json={"answers": {}})

                client = OneJevClient(url=URL, timeout_s=2.0, transport=httpx.MockTransport(handler), clock=lambda: now)
                self.addAsyncCleanup(client.close)
                with patch("s1a.decision_models.onejev.asyncio.wait_for", wraps=asyncio.wait_for) as wait_for:
                    await client.decide({"questions": {}}, "same-request")
                self.assertEqual([call.args[1] for call in wait_for.call_args_list], [2.0, 0.75])
                for request, timeout in zip(requests, (2.0, 0.75), strict=True):
                    self.assertEqual(set(request.extensions["timeout"].values()), {timeout})
                    self.assertEqual(request.headers["X-Request-Id"], "same-request")

    async def test_exhausted_budget_prevents_retry(self) -> None:
        for failure in (httpx.ConnectError("refused"), httpx.Response(502), httpx.Response(504)):
            with self.subTest(failure=failure):
                now = 0.0
                requests = []

                def handler(request: httpx.Request) -> httpx.Response:
                    nonlocal now
                    requests.append(request)
                    now += 2.0
                    if isinstance(failure, Exception):
                        raise failure
                    return failure

                client = OneJevClient(url=URL, timeout_s=2.0, transport=httpx.MockTransport(handler), clock=lambda: now)
                self.addAsyncCleanup(client.close)
                with self.assertRaises(BaseError) as caught:
                    await client.decide({"questions": {}}, "exhausted")
                self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
                self.assertIn("within 2 s", str(caught.exception))
                self.assertEqual(len(requests), 1)

    async def test_stalled_retry_is_cancelled_at_remaining_deadline(self) -> None:
        now = 0.0
        attempts = 0
        stopped = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal now, attempts
            attempts += 1
            if attempts == 1:
                now = 0.99
                raise httpx.ConnectError("refused", request=request)
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            raise AssertionError("stalled retry was not cancelled")

        client = OneJevClient(url=URL, timeout_s=1.0, transport=httpx.MockTransport(handler), clock=lambda: now)
        self.addAsyncCleanup(client.close)
        with patch("s1a.decision_models.onejev.asyncio.wait_for", wraps=asyncio.wait_for) as wait_for:
            with self.assertRaises(BaseError) as caught:
                await client.decide({"questions": {}}, "stalled")
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
        self.assertAlmostEqual(wait_for.call_args_list[1].args[1], 0.01)
        self.assertEqual(attempts, 2)
        self.assertTrue(stopped.is_set())


class TestMappingHelpers(TestCase):
    def test_state_wraps_text_and_marks_one_image(self) -> None:
        self.assertEqual(
            onejev_state(Observation("plain", images=(SCREEN,))),
            {"text": "plain", "screen": "<image:1>"},
        )

    def test_state_marks_multiple_images(self) -> None:
        second = Image(b"\x89PNG second", media_type="image/png")
        state = onejev_state(Observation({"goal": "g"}, images=(SCREEN, second)))
        self.assertEqual(state["image_1"], "<image:1>")
        self.assertEqual(state["image_2"], "<image:2>")

    def test_media_rejects_empty_and_unknown_types(self) -> None:
        with self.assertRaises(BaseError):
            onejev_media(Image(b"", media_type="image/png"), 1)
        with self.assertRaises(BaseError):
            onejev_media(Image(b"x", media_type="image/gif"), 1)

    def test_question_uses_string_instructions(self) -> None:
        asked = onejev_question(ChoiceQuestion({"a": "A"}, goal="g", rules="r"))
        self.assertEqual(asked["type"], "choice")
        self.assertIsInstance(asked["instructions"], str)
        self.assertIn("g", asked["instructions"])


class TestFromEnv(TestCase):
    def test_missing_url_is_a_config_error(self) -> None:
        with patch.dict(os.environ, {"ONEJEV_URL": ""}, clear=False):
            os.environ.pop("ONEJEV_URL", None)
            with self.assertRaises(BaseError) as caught:
                OneJevModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("ONEJEV_URL", str(caught.exception))

    def test_build_model_onejev_from_env(self) -> None:
        with patch.dict(os.environ, {"ONEJEV_URL": "http://127.0.0.1:8000", "ONEJEV_MODEL": "OmniJev/OneJev-0.8B"}):
            model = build_model("onejev")
        self.assertIsInstance(model, OneJevModel)
        self.assertEqual(model.name, "onejev")
        self.assertEqual(model.model, "OmniJev/OneJev-0.8B")
        self.assertTrue(model.supports_images)

    def test_bad_timeout_is_a_config_error(self) -> None:
        with patch.dict(os.environ, {"ONEJEV_URL": "http://127.0.0.1:8000", "ONEJEV_TIMEOUT_S": "0"}):
            with self.assertRaises(BaseError) as caught:
                OneJevModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
