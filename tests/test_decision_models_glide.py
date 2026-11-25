# coding: utf-8
"""``glide`` served client: contract tests against a fake HTTP server (no Fastino / GPU)."""

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
from s1a.decision_models.glide import (
    DEFAULT_GLIDE_MODEL,
    HOSTED_GLIDE_URL,
    MAX_GLIDE_IMAGE_BYTES,
    GlideClient,
    GlideModel,
    glide_image,
    glide_images,
    glide_payload,
    glide_question,
    glide_state,
    retry_after_s,
)

SCREEN = Image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24, media_type="image/png")
URL = "http://glide.test"


def _choice_answer(question: dict[str, Any]) -> dict[str, Any]:
    keys = list(question["criteria"])
    if not keys:
        return {"type": "choice", "choice": "", "probabilities": {}, "confidence": 0.0}
    if len(keys) == 1:
        probabilities = {keys[0]: 1.0}
    else:
        rest = 0.2 / (len(keys) - 1)
        probabilities = {key: (0.8 if i == 0 else rest) for i, key in enumerate(keys)}
    return {"type": "choice", "choice": keys[0], "probabilities": probabilities, "confidence": probabilities[keys[0]]}


def _noul_answer(_question: dict[str, Any]) -> dict[str, Any]:
    return {"type": "noul", "noul": 0.65, "confidence": 0.3}


def _decision_body(request: httpx.Request) -> dict[str, Any]:
    body = json.loads(request.content)
    answers = {}
    for name, question in body["questions"].items():
        answers[name] = _noul_answer(question) if question["type"] == "noul" else _choice_answer(question)
    return {
        "model": "glide",
        "answers": answers,
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }


class Server:
    """Scripted local ``/v1/systemone``, or the hosted Fastino endpoint.

    Rejects unsupported top-level fields the way Fastino's 422 would.
    """

    allowed = frozenset({"model", "state", "questions", "store"})

    def __init__(self, *, script: list[Any] | None = None, reject_extra: bool = True) -> None:
        self.script = list(script or [])
        self.reject_extra = reject_extra
        self.decisions: list[httpx.Request] = []

    def answer(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        extra = sorted(set(body) - self.allowed)
        if self.reject_extra and extra:
            return httpx.Response(422, json={"detail": f"unsupported fields: {extra}"})
        return httpx.Response(200, json=_decision_body(request))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.decisions.append(request)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if callable(step):
                return step(request)
            return step
        return self.answer(request)


def make_local(server: Server, *, timeout_s: float = 5.0, model: str = DEFAULT_GLIDE_MODEL) -> GlideModel:
    client = GlideClient(
        url=URL + "/",
        endpoint=URL + "/v1/systemone",
        timeout_s=timeout_s,
        transport=httpx.MockTransport(server.handler),
        path="local",
    )
    return GlideModel(client, model=model)


def make_hosted(server: Server, *, timeout_s: float = 5.0, model: str = DEFAULT_GLIDE_MODEL) -> GlideModel:
    client = GlideClient(
        url=HOSTED_GLIDE_URL,
        endpoint=HOSTED_GLIDE_URL,
        api_key="fast_sk_test",
        timeout_s=timeout_s,
        transport=httpx.MockTransport(server.handler),
        path="hosted",
    )
    return GlideModel(client, model=model)


class TestGlideContract(contract.DecisionModelContract, IsolatedAsyncioTestCase):
    def make(self) -> DecisionModel:
        return make_local(Server())

    def make_scripted(self, answers: list[dict[str, Any]]) -> DecisionModel:
        script = [
            httpx.Response(200, json={"model": "glide", "answers": a, "usage": {"input_tokens": 8}}) for a in answers
        ]
        return make_local(Server(script=script))


class TestRequestBody(IsolatedAsyncioTestCase):
    async def test_text_path_posts_systemone_without_images_media_or_store(self) -> None:
        server = Server()
        await make_local(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK, "check": contract.CHECK})
        request = server.decisions[0]
        self.assertEqual(request.url, httpx.URL(URL + "/v1/systemone"))
        self.assertNotIn("X-API-Key", request.headers)
        self.assertNotIn("Authorization", request.headers)
        self.assertIn("X-Request-Id", request.headers)
        body = json.loads(request.content)
        self.assertEqual(body["model"], DEFAULT_GLIDE_MODEL)
        self.assertEqual(body["state"], contract.OBSERVATION.state)
        self.assertNotIn("store", body)
        self.assertNotIn("images", body)
        self.assertNotIn("media", body)
        self.assertNotIn("videos", body)
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(body["questions"]["pick"]["type"], "choice")
        self.assertEqual(body["questions"]["check"]["type"], "noul")
        self.assertIsInstance(body["questions"]["pick"]["instructions"], dict)
        self.assertIsInstance(body["questions"]["check"]["instructions"], dict)

    async def test_hosted_sends_api_key_and_store_false(self) -> None:
        server = Server()
        decision = await make_hosted(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        request = server.decisions[0]
        self.assertEqual(str(request.url), HOSTED_GLIDE_URL)
        self.assertEqual(request.headers["X-API-Key"], "fast_sk_test")
        self.assertNotIn("Authorization", request.headers)
        body = json.loads(request.content)
        self.assertEqual(body["store"], False)
        self.assertEqual(set(body), {"model", "state", "questions", "store"})
        self.assertEqual(decision.model, "glide")
        self.assertEqual(decision.raw["served_by"]["path"], "hosted")
        self.assertEqual(decision.usage.input_tokens, 12)

    async def test_images_are_state_parts_not_clef_or_onejev_layout(self) -> None:
        server = Server()
        pictured = Observation({"task": "Pay the invoice"}, images=(SCREEN,))
        await make_local(server).decide_many(pictured, {"pick": contract.PICK})
        body = json.loads(server.decisions[0].content)
        self.assertNotIn("images", body)
        self.assertNotIn("media", body)
        state = body["state"]
        self.assertIsInstance(state, list)
        self.assertEqual(
            state[0], {"type": "text", "text": json.dumps({"task": "Pay the invoice"}, separators=(",", ":"))}
        )
        part = state[1]
        self.assertEqual(part["type"], "image_url")
        url = part["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), SCREEN.data)

    async def test_string_state_with_images_keeps_a_text_part(self) -> None:
        server = Server()
        pictured = Observation("a receipt", images=(SCREEN,))
        await make_local(server).decide_many(pictured, {"pick": contract.PICK})
        state = json.loads(server.decisions[0].content)["state"]
        self.assertEqual(state[0], {"type": "text", "text": "a receipt"})
        self.assertEqual(state[1]["type"], "image_url")

    async def test_provenance_fields_and_no_image_bytes_in_raw(self) -> None:
        pictured = Observation({"task": "Pay the invoice"}, images=(SCREEN,))
        decision = await make_local(Server()).decide_many(pictured, {"pick": contract.PICK})
        self.assertEqual(decision.raw["url"], URL)
        self.assertIsInstance(decision.raw["request_id"], str)
        self.assertEqual(decision.raw["served_by"]["path"], "local")
        self.assertEqual(decision.raw["served_by"]["checkpoint"], DEFAULT_GLIDE_MODEL)
        self.assertEqual(decision.raw["sent_images"][0]["bytes"], len(SCREEN.data))
        dumped = json.dumps(decision.raw)
        self.assertNotIn(base64.b64encode(SCREEN.data).decode("ascii"), dumped)

    async def test_hosted_records_fastino_headers(self) -> None:
        def with_headers(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_decision_body(request),
                headers={"X-Fastino-Data-Retention": "not-stored", "X-Fastino-Request-Id": "fn-1"},
            )

        decision = await make_hosted(Server(script=[with_headers])).decide_many(
            contract.OBSERVATION, {"pick": contract.PICK}
        )
        self.assertEqual(decision.raw["data_retention"], "not-stored")
        self.assertEqual(decision.raw["fastino_request_id"], "fn-1")

    async def test_browser_element_row_becomes_a_description_string(self) -> None:
        server = Server()
        question = ChoiceQuestion(
            {"12": {"element": "[12] Where from?", "current_value": "Zurich", "role": "combobox"}},
            goal="book a flight",
        )
        await make_local(server).decide_many(contract.OBSERVATION, {"click_target": question})
        asked = json.loads(server.decisions[0].content)["questions"]["click_target"]
        self.assertEqual(asked["criteria"], {"12": "Where from? = Zurich"})
        self.assertEqual(asked["instructions"]["goal"], "book a flight")


class TestFailures(IsolatedAsyncioTestCase):
    async def test_too_many_images_is_a_call_failure(self) -> None:
        pictured = Observation("x", images=tuple(SCREEN for _ in range(5)))
        with self.assertRaises(BaseError) as caught:
            await make_local(Server()).decide_many(pictured, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("at most 4", str(caught.exception))

    async def test_transport_error_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.ConnectError("refused"), httpx.ConnectError("refused")])
        with self.assertRaises(BaseError) as caught:
            await make_local(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("glide", str(caught.exception).lower())

    async def test_malformed_answers_object_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(200, json={"model": "x", "answers": "nope"})])
        with self.assertRaises(BaseError) as caught:
            await make_local(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("answers", str(caught.exception).lower())

    async def test_http_error_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(500, text="boom")])
        with self.assertRaises(BaseError) as caught:
            await make_local(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("500", str(caught.exception))

    async def test_422_on_extra_fields_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(422, json={"detail": "unsupported field: foo"})])
        with self.assertRaises(BaseError) as caught:
            await make_local(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("422", str(caught.exception))


class TestFastinoRetries(IsolatedAsyncioTestCase):
    async def test_425_429_503_retry_after_zero_then_succeed(self) -> None:
        for status in (425, 429, 503):
            with self.subTest(status=status):
                sleeps: list[float] = []

                async def sleeper(delay: float, recorded=sleeps) -> None:
                    recorded.append(delay)

                def first_then_ok(status=status):
                    calls = {"n": 0}

                    def handler(request: httpx.Request) -> httpx.Response:
                        calls["n"] += 1
                        if calls["n"] == 1:
                            return httpx.Response(
                                status, json={"error": {"code": "model_warming"}}, headers={"Retry-After": "0"}
                            )
                        return httpx.Response(200, json=_decision_body(request))

                    return handler

                client = GlideClient(
                    url=URL,
                    endpoint=URL + "/v1/systemone",
                    timeout_s=5.0,
                    transport=httpx.MockTransport(first_then_ok()),
                    sleeper=sleeper,
                    path="hosted",
                )
                self.addAsyncCleanup(client.close)
                payload, _ms, _headers = await client.decide(
                    {
                        "model": DEFAULT_GLIDE_MODEL,
                        "questions": {"pick": {"type": "choice", "criteria": {"hit": "Hit"}, "instructions": "x"}},
                    },
                    "rid",
                )
                self.assertIn("answers", payload)
                self.assertEqual(sleeps, [0.0])

    async def test_exhausted_budget_skips_425_retry(self) -> None:
        now = 0.0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal now
            now += 2.0
            return httpx.Response(425, json={"error": {"code": "model_warming"}}, headers={"Retry-After": "30"})

        client = GlideClient(
            url=URL,
            endpoint=URL + "/v1/systemone",
            timeout_s=2.0,
            transport=httpx.MockTransport(handler),
            clock=lambda: now,
            path="hosted",
        )
        self.addAsyncCleanup(client.close)
        with self.assertRaises(BaseError) as caught:
            await client.decide({"questions": {}}, "warm")
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("425", str(caught.exception))


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

                client = GlideClient(
                    url=URL,
                    endpoint=URL + "/v1/systemone",
                    timeout_s=2.0,
                    transport=httpx.MockTransport(handler),
                    clock=lambda: now,
                    path="local",
                )
                self.addAsyncCleanup(client.close)
                with patch("s1a.decision_models.glide.asyncio.wait_for", wraps=asyncio.wait_for) as wait_for:
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

                client = GlideClient(
                    url=URL,
                    endpoint=URL + "/v1/systemone",
                    timeout_s=2.0,
                    transport=httpx.MockTransport(handler),
                    clock=lambda: now,
                    path="local",
                )
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

        client = GlideClient(
            url=URL,
            endpoint=URL + "/v1/systemone",
            timeout_s=1.0,
            transport=httpx.MockTransport(handler),
            clock=lambda: now,
            path="local",
        )
        self.addAsyncCleanup(client.close)
        with patch("s1a.decision_models.glide.asyncio.wait_for", wraps=asyncio.wait_for) as wait_for:
            with self.assertRaises(BaseError) as caught:
                await client.decide({"questions": {}}, "stalled")
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
        self.assertAlmostEqual(wait_for.call_args_list[1].args[1], 0.01)
        self.assertEqual(attempts, 2)
        self.assertTrue(stopped.is_set())


class TestMappingHelpers(TestCase):
    def test_payload_requires_an_answers_object(self) -> None:
        inner = {"model": "glide", "answers": {"pick": {"noul": 0.2}}, "usage": {"input_tokens": 1}}
        self.assertEqual(glide_payload(inner), inner)
        with self.assertRaises(BaseError):
            glide_payload({"model": "glide"})
        with self.assertRaises(BaseError):
            glide_payload({"answers": "nope"})

    def test_image_rejects_empty_unknown_types_and_oversize(self) -> None:
        with self.assertRaises(BaseError):
            glide_image(Image(b"", media_type="image/png"))
        with self.assertRaises(BaseError):
            glide_image(Image(b"x", media_type="image/tiff"))
        with self.assertRaises(BaseError):
            glide_images((SCREEN,) * 5)
        huge = Image(b"x" * (MAX_GLIDE_IMAGE_BYTES + 1), media_type="image/png")
        with self.assertRaises(BaseError) as caught:
            glide_image(huge)
        self.assertIn(str(MAX_GLIDE_IMAGE_BYTES), str(caught.exception))

    def test_gif_is_accepted(self) -> None:
        part, record = glide_image(Image(b"GIF89a", media_type="image/gif"))
        self.assertEqual(part["type"], "image_url")
        self.assertTrue(part["image_url"]["url"].startswith("data:image/gif;base64,"))
        self.assertEqual(record["bytes"], 6)

    def test_question_keeps_jev_instruction_objects(self) -> None:
        asked = glide_question(ChoiceQuestion({"a": "A"}, goal="g", rules="r"))
        self.assertEqual(asked["type"], "choice")
        self.assertEqual(asked["instructions"], {"goal": "g", "rules": "r"})
        self.assertEqual(asked["criteria"], {"a": "A"})

    def test_empty_instructions_become_a_string(self) -> None:
        asked = glide_question(ChoiceQuestion({"a": "A"}))
        self.assertIsInstance(asked["instructions"], str)
        self.assertTrue(asked["instructions"])

    def test_text_state_without_images_is_unchanged(self) -> None:
        self.assertEqual(glide_state(contract.OBSERVATION), contract.OBSERVATION.state)

    def test_retry_after_parses_seconds_and_falls_back(self) -> None:
        self.assertEqual(retry_after_s(httpx.Response(425, headers={"Retry-After": "12"}), default=30.0), 12.0)
        self.assertEqual(retry_after_s(httpx.Response(425), default=30.0), 30.0)
        self.assertEqual(retry_after_s(httpx.Response(425, headers={"Retry-After": "nope"}), default=30.0), 30.0)


class TestFromEnv(TestCase):
    def test_missing_url_and_key_is_a_config_error(self) -> None:
        with patch.dict(os.environ, {"GLIDE_URL": "", "FASTINO_API_KEY": "", "GLIDE_API_KEY": ""}, clear=False):
            os.environ.pop("GLIDE_URL", None)
            os.environ.pop("FASTINO_API_KEY", None)
            os.environ.pop("GLIDE_API_KEY", None)
            with self.assertRaises(BaseError) as caught:
                GlideModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("GLIDE_URL", str(caught.exception))
        self.assertIn("FASTINO_API_KEY", str(caught.exception))

    def test_build_model_glide_from_local_url(self) -> None:
        with patch.dict(os.environ, {"GLIDE_URL": "http://127.0.0.1:8000", "GLIDE_TIMEOUT_S": "5"}):
            model = build_model("glide")
        self.addCleanup(lambda: asyncio.run(model.close()))
        self.assertIsInstance(model, GlideModel)
        self.assertEqual(model.name, "glide")
        self.assertEqual(model.model, DEFAULT_GLIDE_MODEL)
        self.assertTrue(model.supports_images)
        self.assertTrue(model.deterministic)
        self.assertEqual(model._client.path, "local")
        self.assertEqual(model._client.endpoint, "http://127.0.0.1:8000/v1/systemone")

    def test_build_model_glide_from_hosted_key(self) -> None:
        with patch.dict(os.environ, {"GLIDE_URL": "", "FASTINO_API_KEY": "fast_sk_secret"}, clear=False):
            os.environ.pop("GLIDE_URL", None)
            model = build_model("glide")
        self.addCleanup(lambda: asyncio.run(model.close()))
        self.assertIsInstance(model, GlideModel)
        self.assertEqual(model._client.path, "hosted")
        self.assertEqual(model._client.endpoint, HOSTED_GLIDE_URL)

    def test_local_url_wins_over_hosted_key(self) -> None:
        with patch.dict(os.environ, {"GLIDE_URL": "http://127.0.0.1:8000", "FASTINO_API_KEY": "fast_sk_secret"}):
            model = GlideModel.from_env()
        self.addCleanup(lambda: asyncio.run(model.close()))
        self.assertEqual(model._client.path, "local")

    def test_custom_model_id_and_bad_timeout_are_handled(self) -> None:
        with patch.dict(os.environ, {"GLIDE_URL": "http://127.0.0.1:8000", "GLIDE_MODEL": "fastino/GLiDE-deployed"}):
            model = GlideModel.from_env()
        self.addCleanup(lambda: asyncio.run(model.close()))
        self.assertEqual(model.model, "fastino/GLiDE-deployed")
        with patch.dict(os.environ, {"GLIDE_URL": "http://127.0.0.1:8000", "GLIDE_TIMEOUT_S": "0"}):
            with self.assertRaises(BaseError) as caught:
                GlideModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
