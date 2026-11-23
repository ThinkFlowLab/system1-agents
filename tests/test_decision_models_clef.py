# coding: utf-8
"""``clef`` served client: contract tests against a fake HTTP server (no Cloudflare / GPU)."""

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
from s1a.decision_models.clef import (
    DEFAULT_CLEF_MODEL,
    ClefClient,
    ClefModel,
    clef_image,
    clef_images,
    clef_payload,
    clef_question,
    workers_ai_url,
)

SCREEN = Image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24, media_type="image/png")
URL = "http://clef.test"
ACCOUNT = "acct-test"
WORKERS = workers_ai_url(ACCOUNT, DEFAULT_CLEF_MODEL)


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
    return {"type": "noul", "noul": 0.65}


def _decision_body(request: httpx.Request) -> dict[str, Any]:
    body = json.loads(request.content)
    answers = {}
    for name, question in body["questions"].items():
        answers[name] = _noul_answer(question) if question["type"] == "noul" else _choice_answer(question)
    return {
        "model": body.get("model") or DEFAULT_CLEF_MODEL,
        "answers": answers,
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }


class Server:
    """Scripted local ``/v1/systemone`` + ``/health``, or a Workers AI run endpoint."""

    def __init__(
        self,
        *,
        health: Any = None,
        script: list[Any] | None = None,
        envelope: bool = False,
    ) -> None:
        self.health = health if health is not None else {"model": DEFAULT_CLEF_MODEL, "device": "cuda"}
        self.script = list(script or [])
        self.envelope = envelope
        self.decisions: list[httpx.Request] = []
        self.health_calls = 0

    def answer(self, request: httpx.Request) -> httpx.Response:
        payload = _decision_body(request)
        if self.envelope:
            return httpx.Response(200, json={"success": True, "result": payload, "errors": [], "messages": []})
        return httpx.Response(200, json=payload)

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


def make_local(server: Server, *, timeout_s: float = 5.0, model: str = DEFAULT_CLEF_MODEL) -> ClefModel:
    client = ClefClient(
        url=URL + "/",
        endpoint=URL + "/v1/systemone",
        timeout_s=timeout_s,
        transport=httpx.MockTransport(server.handler),
        health_url=URL + "/health",
        path="local",
    )
    return ClefModel(client, model=model)


def make_workers(server: Server, *, timeout_s: float = 5.0, model: str = DEFAULT_CLEF_MODEL) -> ClefModel:
    client = ClefClient(
        url=WORKERS,
        endpoint=WORKERS,
        api_key="token",
        timeout_s=timeout_s,
        transport=httpx.MockTransport(server.handler),
        path="workers-ai",
    )
    return ClefModel(client, model=model)


class TestClefContract(contract.DecisionModelContract, IsolatedAsyncioTestCase):
    def make(self) -> DecisionModel:
        return make_local(Server())

    def make_scripted(self, answers: list[dict[str, Any]]) -> DecisionModel:
        script = [
            httpx.Response(200, json={"model": DEFAULT_CLEF_MODEL, "answers": a, "usage": {"input_tokens": 8}})
            for a in answers
        ]
        return make_local(Server(script=script))


class TestRequestBody(IsolatedAsyncioTestCase):
    async def test_text_path_omits_images_and_posts_systemone(self) -> None:
        server = Server()
        await make_local(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK, "check": contract.CHECK})
        request = server.decisions[0]
        self.assertEqual(request.url, httpx.URL(URL + "/v1/systemone"))
        body = json.loads(request.content)
        self.assertEqual(body["model"], DEFAULT_CLEF_MODEL)
        self.assertEqual(body["state"], contract.OBSERVATION.state)
        self.assertNotIn("images", body)
        self.assertNotIn("media", body)
        self.assertNotIn("videos", body)
        self.assertEqual(body["questions"]["pick"]["type"], "choice")
        self.assertEqual(body["questions"]["check"]["type"], "noul")
        self.assertIsInstance(body["questions"]["pick"]["instructions"], str)
        self.assertIn("X-Request-Id", request.headers)

    async def test_images_are_clef_data_urls_not_onejev_media(self) -> None:
        server = Server()
        pictured = Observation({"task": "Pay the invoice"}, images=(SCREEN,))
        await make_local(server).decide_many(pictured, {"pick": contract.PICK})
        body = json.loads(server.decisions[0].content)
        self.assertNotIn("media", body)
        self.assertEqual(len(body["images"]), 1)
        self.assertTrue(body["images"][0].startswith("data:image/png;base64,"))
        decoded = base64.b64decode(body["images"][0].split(",", 1)[1])
        self.assertEqual(decoded, SCREEN.data)
        self.assertEqual(body["state"], {"task": "Pay the invoice"})

    async def test_provenance_fields_and_no_image_bytes_in_raw(self) -> None:
        pictured = Observation({"task": "Pay the invoice"}, images=(SCREEN,))
        decision = await make_local(Server()).decide_many(pictured, {"pick": contract.PICK})
        self.assertEqual(decision.raw["url"], URL)
        self.assertIsInstance(decision.raw["request_id"], str)
        self.assertEqual(decision.raw["served_by"]["path"], "local")
        self.assertEqual(decision.raw["served_by"]["checkpoint"], DEFAULT_CLEF_MODEL)
        self.assertEqual(decision.raw["sent_images"][0]["bytes"], len(SCREEN.data))
        dumped = json.dumps(decision.raw)
        self.assertNotIn(base64.b64encode(SCREEN.data).decode("ascii"), dumped)

    async def test_workers_ai_unwraps_the_rest_envelope(self) -> None:
        server = Server(envelope=True)
        decision = await make_workers(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        request = server.decisions[0]
        self.assertEqual(str(request.url), WORKERS)
        self.assertTrue(request.headers["Authorization"].startswith("Bearer "))
        self.assertEqual(decision.choice("pick").key, "hit")
        self.assertEqual(decision.model, DEFAULT_CLEF_MODEL)
        self.assertEqual(decision.raw["served_by"]["path"], "workers-ai")
        self.assertEqual(decision.usage.input_tokens, 12)
        self.assertEqual(server.health_calls, 0)

    async def test_workers_ai_records_cf_ray(self) -> None:
        def with_ray(request: httpx.Request) -> httpx.Response:
            payload = _decision_body(request)
            return httpx.Response(
                200,
                json={"success": True, "result": payload, "errors": [], "messages": []},
                headers={"cf-ray": "abc-DFW"},
            )

        decision = await make_workers(Server(script=[with_ray])).decide_many(
            contract.OBSERVATION, {"pick": contract.PICK}
        )
        self.assertEqual(decision.raw["cf_ray"], "abc-DFW")

    async def test_browser_element_row_becomes_label(self) -> None:
        server = Server()
        question = ChoiceQuestion(
            {"12": {"element": "[12] Where from?", "current_value": "Zurich", "role": "combobox"}},
            goal="book a flight",
        )
        await make_local(server).decide_many(contract.OBSERVATION, {"click_target": question})
        asked = json.loads(server.decisions[0].content)["questions"]["click_target"]
        self.assertEqual(asked["criteria"], {"12": "Where from? = Zurich"})
        self.assertIn("book a flight", asked["instructions"])


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
        self.assertIn("clef", str(caught.exception).lower())

    async def test_malformed_answers_object_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(200, json={"model": "x", "answers": "nope"})])
        with self.assertRaises(BaseError) as caught:
            await make_local(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("answers", str(caught.exception).lower())

    async def test_workers_ai_error_envelope_is_a_call_failure(self) -> None:
        server = Server(
            script=[
                httpx.Response(
                    200, json={"success": False, "errors": [{"code": 10000, "message": "nope"}], "result": None}
                )
            ]
        )
        with self.assertRaises(BaseError) as caught:
            await make_workers(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("nope", str(caught.exception))

    async def test_http_error_is_a_call_failure(self) -> None:
        server = Server(script=[httpx.Response(500, text="boom")])
        with self.assertRaises(BaseError) as caught:
            await make_local(server).decide_many(contract.OBSERVATION, {"pick": contract.PICK})
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

                client = ClefClient(
                    url=URL,
                    endpoint=URL + "/v1/systemone",
                    timeout_s=2.0,
                    transport=httpx.MockTransport(handler),
                    clock=lambda: now,
                    path="local",
                )
                self.addAsyncCleanup(client.close)
                with patch("s1a.decision_models.clef.asyncio.wait_for", wraps=asyncio.wait_for) as wait_for:
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

                client = ClefClient(
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

        client = ClefClient(
            url=URL,
            endpoint=URL + "/v1/systemone",
            timeout_s=1.0,
            transport=httpx.MockTransport(handler),
            clock=lambda: now,
            path="local",
        )
        self.addAsyncCleanup(client.close)
        with patch("s1a.decision_models.clef.asyncio.wait_for", wraps=asyncio.wait_for) as wait_for:
            with self.assertRaises(BaseError) as caught:
                await client.decide({"questions": {}}, "stalled")
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
        self.assertAlmostEqual(wait_for.call_args_list[1].args[1], 0.01)
        self.assertEqual(attempts, 2)
        self.assertTrue(stopped.is_set())


class TestMappingHelpers(TestCase):
    def test_payload_unwraps_workers_ai_envelope_and_rejects_bad_bodies(self) -> None:
        inner = {
            "model": "clef-flash",
            "answers": {"pick": {"noul": 0.2}},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }
        self.assertEqual(clef_payload(inner), inner)
        self.assertEqual(clef_payload({"success": True, "result": inner}), inner)
        with self.assertRaises(BaseError):
            clef_payload({"success": False, "errors": [{"message": "denied"}], "result": None})
        with self.assertRaises(BaseError):
            clef_payload({"result": {"nope": True}})

    def test_image_rejects_empty_and_unknown_types(self) -> None:
        with self.assertRaises(BaseError):
            clef_image(Image(b"", media_type="image/png"))
        with self.assertRaises(BaseError):
            clef_image(Image(b"x", media_type="image/gif"))
        with self.assertRaises(BaseError):
            clef_images((SCREEN,) * 5)

    def test_question_uses_string_instructions(self) -> None:
        asked = clef_question(ChoiceQuestion({"a": "A"}, goal="g", rules="r"))
        self.assertEqual(asked["type"], "choice")
        self.assertIsInstance(asked["instructions"], str)
        self.assertIn("g", asked["instructions"])
        self.assertEqual(asked["criteria"], {"a": "A"})

    def test_workers_ai_url_matches_the_official_path(self) -> None:
        self.assertEqual(
            workers_ai_url("abc", "clef"),
            "https://api.cloudflare.com/client/v4/accounts/abc/ai/run/@cf/cloudflare/clef",
        )
        self.assertEqual(
            workers_ai_url("abc", "clef-flash"),
            "https://api.cloudflare.com/client/v4/accounts/abc/ai/run/@cf/cloudflare/clef-flash",
        )


class TestFromEnv(TestCase):
    def test_missing_url_and_token_is_a_config_error(self) -> None:
        with patch.dict(
            os.environ,
            {"CLEF_URL": "", "CLOUDFLARE_ACCOUNT_ID": "", "CLOUDFLARE_AUTH_TOKEN": "", "CLOUDFLARE_API_TOKEN": ""},
            clear=False,
        ):
            os.environ.pop("CLEF_URL", None)
            os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)
            os.environ.pop("CLOUDFLARE_AUTH_TOKEN", None)
            os.environ.pop("CLOUDFLARE_API_TOKEN", None)
            with self.assertRaises(BaseError) as caught:
                ClefModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("CLEF_URL", str(caught.exception))
        self.assertIn("CLOUDFLARE_ACCOUNT_ID", str(caught.exception))

    def test_build_model_clef_from_local_url(self) -> None:
        with patch.dict(os.environ, {"CLEF_URL": "http://127.0.0.1:30000", "CLEF_MODEL": "clef"}):
            model = build_model("clef")
        self.assertIsInstance(model, ClefModel)
        self.assertEqual(model.name, "clef")
        self.assertEqual(model.model, "clef")
        self.assertTrue(model.supports_images)
        self.assertEqual(model._client.path, "local")
        self.assertEqual(model._client.endpoint, "http://127.0.0.1:30000/v1/systemone")

    def test_build_model_clef_from_workers_ai_env(self) -> None:
        env = {
            "CLEF_URL": "",
            "CLOUDFLARE_ACCOUNT_ID": ACCOUNT,
            "CLOUDFLARE_AUTH_TOKEN": "secret",
            "CLEF_MODEL": "clef-flash",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CLEF_URL", None)
            model = build_model("clef")
        self.assertIsInstance(model, ClefModel)
        self.assertEqual(model.model, DEFAULT_CLEF_MODEL)
        self.assertEqual(model._client.path, "workers-ai")
        self.assertEqual(model._client.endpoint, WORKERS)
        self.assertIsNone(model._client.health_url)

    def test_local_url_wins_over_cloudflare_credentials(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CLEF_URL": "http://127.0.0.1:30000",
                "CLOUDFLARE_ACCOUNT_ID": ACCOUNT,
                "CLOUDFLARE_AUTH_TOKEN": "secret",
            },
        ):
            model = ClefModel.from_env()
        self.assertEqual(model._client.path, "local")

    def test_api_token_alias_is_accepted(self) -> None:
        with patch.dict(
            os.environ,
            {"CLOUDFLARE_ACCOUNT_ID": ACCOUNT, "CLOUDFLARE_API_TOKEN": "alias", "CLOUDFLARE_AUTH_TOKEN": ""},
            clear=False,
        ):
            os.environ.pop("CLEF_URL", None)
            os.environ.pop("CLOUDFLARE_AUTH_TOKEN", None)
            model = ClefModel.from_env()
        self.assertEqual(model._client.path, "workers-ai")

    def test_bad_model_and_timeout_are_config_errors(self) -> None:
        with patch.dict(os.environ, {"CLEF_URL": "http://127.0.0.1:30000", "CLEF_MODEL": "clef-omni"}):
            with self.assertRaises(BaseError) as caught:
                ClefModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("clef-flash", str(caught.exception))
        with patch.dict(os.environ, {"CLEF_URL": "http://127.0.0.1:30000", "CLEF_TIMEOUT_S": "0"}):
            with self.assertRaises(BaseError) as caught:
                ClefModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
