# coding: utf-8
"""``laya-served``: the shared contract, the error mapping, the window check and who answered.

A ``httpx.MockTransport`` stands in for the server; its answers follow the responses recorded in
``tests/data/served_laya/``. No torch, laya or network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

from s1a.decision_models import ChoiceQuestion, NoulQuestion, Observation
from s1a.decision_models.laya import laya_question
from s1a.decision_models.served import (
    HEALTH_MAX_AGE_S,
    ServedLayaClient,
    ServedLayaModel,
    parse_server_timing,
)
from tests.decision_model_contract import DecisionModelContract

FIXTURES = Path(__file__).resolve().parent / "data" / "served_laya"
URL = "http://laya.test"
OBSERVATION = Observation({"ticket": "I was charged twice. Please refund the duplicate."})
PICK = ChoiceQuestion({"billing": "Charges and refunds", "technical": "Software problems"}, rules="route it")
CHECK = NoulQuestion("Does the customer ask for a refund?")


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


WORKER_HEALTH = fixture("worker.health")["body"]
LAYA_SERVE_HEALTH = fixture("laya-serve.health")["body"]
ROUTING = {"model": "english", "repo": "convaiinnovations/laya", "reason": "explicit model='english'"}


def laya_answer(question: dict[str, Any]) -> dict[str, Any]:
    """An answer the way laya-serve shapes it: the first option wins."""
    if question["type"] == "noul":
        return {
            "type": "noul",
            "noul": 0.7,
            "confidence": 0.7,
            "answer_confidence": 0.7,
            "action": {"act_probability": 1.0},
        }
    keys = list(question["criteria"])
    rest = 0.2 / max(1, len(keys) - 1)
    probabilities = {key: (0.8 if i == 0 else rest) for i, key in enumerate(keys)} if len(keys) > 1 else {keys[0]: 1.0}
    return {
        "type": "choice",
        "choice": keys[0],
        "probabilities": probabilities,
        "confidence": 0.8,
        "answer_confidence": 0.8,
        "action": {"act_probability": 1.0},
    }


def ok(payload: dict[str, Any], headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(200, json=payload, headers=headers)


class Server:
    """A scripted server: ``/health`` returns ``health``; each decision takes the next entry of ``script``
    (a response, an exception, or a callable of the request), then answers like laya-serve."""

    def __init__(self, *, health: Any = WORKER_HEALTH, script: list[Any] | None = None, usage: int = 40) -> None:
        self.health = health
        self.script = list(script or [])
        self.usage = usage
        self.decisions: list[httpx.Request] = []
        self.health_calls = 0

    def answer(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return ok(
            {
                "model": "laya-rl-agent",
                "answers": {name: laya_answer(q) for name, q in body["questions"].items()},
                "usage": {"input_tokens": self.usage, "output_tokens": 0},
                "routing": ROUTING,
            }
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            self.health_calls += 1
            if isinstance(self.health, Exception):
                raise self.health
            if callable(self.health):
                return self.health(request)
            if isinstance(self.health, httpx.Response):
                return self.health
            return ok(self.health)
        self.decisions.append(request)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if callable(step):
                return step(request)
            return step
        return self.answer(request)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_model(
    server: Server,
    *,
    api_key: str | None = None,
    timeout_s: float = 5.0,
    max_len: int = 512,
    clock: Clock | None = None,
) -> tuple[ServedLayaModel, list[float]]:
    clock = clock or Clock()
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.now += seconds

    client = ServedLayaClient(
        url=URL + "/",
        api_key=api_key,
        timeout_s=timeout_s,
        transport=httpx.MockTransport(server.handler),
        clock=clock,
        sleep=sleep,
    )
    return ServedLayaModel(client, max_len=max_len, clock=clock), slept


class ServedContract(DecisionModelContract, IsolatedAsyncioTestCase):
    def make(self) -> ServedLayaModel:
        return make_model(Server())[0]

    def make_scripted(self, answers: list[dict[str, Any]]) -> ServedLayaModel:
        script = [
            ok({"model": "laya-rl-agent", "answers": a, "usage": {"input_tokens": 30}, "routing": ROUTING})
            for a in answers
        ]
        return make_model(Server(script=script))[0]


class RequestTests(IsolatedAsyncioTestCase):
    async def test_the_body_is_what_in_process_laya_reads(self) -> None:
        server = Server()
        model, _ = make_model(server)
        await model.decide_many(OBSERVATION, {"pick": PICK, "check": CHECK})
        request = server.decisions[0]
        self.assertEqual(request.url, httpx.URL(URL + "/v1/systemone"))
        self.assertEqual(
            json.loads(request.content),
            {
                "model": "english",
                "state": OBSERVATION.state,
                "questions": {"pick": laya_question(PICK), "check": laya_question(CHECK)},
            },
        )
        self.assertIsInstance(json.loads(request.content)["questions"]["check"]["instructions"], str)

    async def test_the_bearer_token_is_sent_only_when_set(self) -> None:
        server = Server()
        await make_model(server, api_key="secret")[0].decide_many(OBSERVATION, {"pick": PICK})
        await make_model(server)[0].decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(server.decisions[0].headers.get("authorization"), "Bearer secret")
        self.assertNotIn("authorization", server.decisions[1].headers)

    async def test_only_the_asked_questions_come_back(self) -> None:
        extra = lambda request: ok(  # noqa: E731
            {
                "answers": {"pick": laya_answer(laya_question(PICK)), "stray": {"noul": 0.1}},
                "usage": {"input_tokens": 9},
            }
        )
        decision = await make_model(Server(script=[extra]))[0].decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(list(decision.answers), ["pick"])


class ErrorTests(IsolatedAsyncioTestCase):
    async def assert_fails(self, server: Server, status: StatusCode, text: str, **kwargs: Any) -> BaseError:
        model, _ = make_model(server, **kwargs)
        with self.assertRaises(BaseError) as caught:
            await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.status, status)
        self.assertIn(text, str(caught.exception))
        return caught.exception

    async def test_a_dropped_connection_is_retried_once(self) -> None:
        server = Server(script=[httpx.ConnectError("refused")])
        decision = await make_model(server)[0].decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual((len(server.decisions), decision.choice("pick").key), (2, "billing"))

    async def test_one_request_id_per_decision_kept_across_the_retry(self) -> None:
        server = Server(script=[httpx.ConnectError("refused")])
        model = make_model(server)[0]
        first = await model.decide_many(OBSERVATION, {"pick": PICK})
        await model.decide_many(OBSERVATION, {"pick": PICK})
        sent = [request.headers["X-Request-Id"] for request in server.decisions]
        self.assertEqual((sent[0], sent[1], len(set(sent))), (first.raw["request_id"], first.raw["request_id"], 2))

    async def test_a_second_dropped_connection_fails(self) -> None:
        server = Server(script=[httpx.ConnectError("refused"), httpx.ConnectError("refused")])
        await self.assert_fails(server, StatusCode.MODEL_CALL_FAILED, "no served Laya at http://laya.test")
        self.assertEqual(len(server.decisions), 2)
        dropped = Server(script=[httpx.ReadError("reset"), httpx.ReadError("reset")])
        await self.assert_fails(dropped, StatusCode.MODEL_CALL_FAILED, "unreachable at http://laya.test")

    async def test_502_and_504_from_the_frontend_are_retried_once(self) -> None:
        for status in (502, 504):
            server = Server(script=[httpx.Response(status, text="backend unavailable\n")])
            await make_model(server)[0].decide_many(OBSERVATION, {"pick": PICK})
            self.assertEqual(len(server.decisions), 2, status)
            failing = Server(script=[httpx.Response(status, text="backend timed out\n")] * 2)
            await self.assert_fails(failing, StatusCode.MODEL_CALL_FAILED, f"HTTP {status}")

    async def test_503_waits_retry_after_then_retries(self) -> None:
        server = Server(script=[httpx.Response(503, headers={"Retry-After": "1"}, json={"detail": "busy"})])
        model, slept = make_model(server)
        await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual((slept, len(server.decisions)), ([1.0], 2))

    async def test_503_past_the_deadline_fails_without_waiting(self) -> None:
        server = Server(script=[httpx.Response(503, headers={"Retry-After": "30"}, json={"detail": "busy"})])
        model, slept = make_model(server)
        with self.assertRaises(BaseError) as caught:
            await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertIn("overloaded", str(caught.exception))
        self.assertEqual((slept, len(server.decisions)), ([], 1))

    async def test_request_errors_fail_at_once_with_the_servers_reason(self) -> None:
        for name in (
            "worker.malformed_json",
            "worker.no_questions",
            "worker.invalid_question",
            "worker.too_many_questions",
        ):
            record = fixture(name)
            server = Server(script=[httpx.Response(record["status"], json=record["body"])])
            await self.assert_fails(server, StatusCode.MODEL_CALL_FAILED, record["body"]["detail"])
            self.assertEqual(len(server.decisions), 1, name)

    async def test_problem_json_reasons_carry_the_code(self) -> None:
        problem = {
            "type": "urn:laya:problem:invalid-question",
            "title": "Invalid question",
            "status": 422,
            "code": "invalid_question",
            "detail": "question 'q': unknown type",
        }
        response = httpx.Response(422, json=problem, headers={"content-type": "application/problem+json"})
        await self.assert_fails(
            Server(script=[response]), StatusCode.MODEL_CALL_FAILED, "invalid_question: question 'q'"
        )

    async def test_401_is_a_configuration_error(self) -> None:
        record = fixture("worker.unauthorized")
        server = Server(script=[httpx.Response(401, json=record["body"])])
        await self.assert_fails(server, StatusCode.MODEL_SERVICE_CONFIG_ERROR, "LAYA_SERVED_API_KEY")

    async def test_500_fails_at_once(self) -> None:
        server = Server(script=[httpx.Response(500, json={"detail": "inference failed"})])
        await self.assert_fails(server, StatusCode.MODEL_CALL_FAILED, "inference failed")
        self.assertEqual(len(server.decisions), 1)

    async def test_a_read_timeout_fails_with_the_deadline(self) -> None:
        server = Server(script=[httpx.ReadTimeout("slow")])
        await self.assert_fails(server, StatusCode.MODEL_CALL_FAILED, "within 5 s")

    async def test_the_deadline_covers_the_retry(self) -> None:
        clock = Clock()

        def slow_refusal(request: httpx.Request) -> httpx.Response:
            clock.now += 6.0  # the first attempt used the whole deadline
            raise httpx.ConnectError("refused")

        server = Server(script=[slow_refusal])
        await self.assert_fails(server, StatusCode.MODEL_CALL_FAILED, "within 5 s", clock=clock)
        self.assertEqual(len(server.decisions), 1)

    async def test_a_stalled_identity_refresh_stays_inside_the_deadline(self) -> None:
        clock = Clock()
        given: dict[str, float] = {}

        def stalled_health(request: httpx.Request) -> httpx.Response:
            given["health"] = request.extensions["timeout"]["read"]
            clock.now += given["health"]  # waits out its whole share
            raise httpx.ReadTimeout("stalled")

        def answer(request: httpx.Request) -> httpx.Response:
            given["decide"] = request.extensions["timeout"]["read"]
            return server.answer(request)

        server = Server(health=stalled_health, script=[answer])
        model, _ = make_model(server, timeout_s=3.0, clock=clock)
        await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(given, {"health": 1.5, "decide": 1.5})  # half the budget each, 3 s in all

    async def test_a_body_without_answers_is_malformed(self) -> None:
        for response in (httpx.Response(200, text="not json"), ok({"model": "x"}), httpx.Response(200, json=[1])):
            await self.assert_fails(Server(script=[response]), StatusCode.MODEL_CALL_FAILED, "served Laya returned")


class WindowTests(IsolatedAsyncioTestCase):
    async def test_a_filled_window_is_an_error(self) -> None:
        model, _ = make_model(Server(usage=512), max_len=512)
        with self.assertRaises(BaseError) as caught:
            await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("LAYA_SERVED_MAX_LEN", str(caught.exception))

    async def test_the_window_scales_with_the_questions(self) -> None:
        model, _ = make_model(Server(usage=600), max_len=512)
        await model.decide_many(OBSERVATION, {"pick": PICK, "check": CHECK})  # 600 < 2 * 512


class WarmTests(IsolatedAsyncioTestCase):
    async def test_warm_fails_early_when_no_server_listens(self) -> None:
        model, _ = make_model(Server(health=httpx.ConnectError("refused")))
        with self.assertRaises(BaseError) as caught:
            await model.warm()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("listens only once it has loaded and warmed up", str(caught.exception))

    async def test_an_unusable_health_is_a_warning_not_an_error(self) -> None:
        for health in (httpx.Response(404), httpx.Response(200, text="ok")):
            model, _ = make_model(Server(health=health))
            with self.assertLogs("s1a.decision_models.served", level="WARNING"):
                await model.warm()
            decision = await model.decide_many(OBSERVATION, {"pick": PICK})
            self.assertEqual(decision.model, "convaiinnovations/laya")  # from the response's routing


class IdentityTests(IsolatedAsyncioTestCase):
    async def raw(self, server: Server, clock: Clock | None = None) -> tuple[Any, dict[str, Any]]:
        model, _ = make_model(server, clock=clock)
        await model.warm()
        decision = await model.decide_many(OBSERVATION, {"pick": PICK})
        return decision, decision.raw

    async def test_the_worker_names_checkpoint_revision_and_device(self) -> None:
        decision, raw = await self.raw(Server())
        self.assertEqual(decision.model, "convaiinnovations/laya@55cf4c4ebb4e")
        served_by = raw["served_by"]
        self.assertEqual(
            {k: served_by[k] for k in ("checkpoint", "device", "weights_dtype", "compiled", "source")},
            {
                "checkpoint": "convaiinnovations/laya",
                "device": "mps",
                "weights_dtype": "torch.float16",
                "compiled": True,
                "source": "health",
            },
        )
        self.assertTrue(served_by["read_at"])
        self.assertEqual(raw["url"], URL)

    async def test_plain_laya_serve_falls_back_to_the_routed_repo(self) -> None:
        decision, raw = await self.raw(Server(health=LAYA_SERVE_HEALTH))
        self.assertEqual(decision.model, "convaiinnovations/laya")
        self.assertEqual(
            raw["served_by"],
            {"checkpoint": "convaiinnovations/laya", "revision": None, "device": None, "source": "routing"},
        )

    async def test_the_answering_model_is_looked_up_by_routing(self) -> None:
        health = json.loads(json.dumps(WORKER_HEALTH))
        health["models"]["multilingual"] = {
            **health["models"]["english"],
            "device": "cpu",
            "checkpoint": "convaiinnovations/laya-multilingual",
        }
        routed = lambda request: ok(  # noqa: E731
            {
                "answers": {"pick": laya_answer(laya_question(PICK))},
                "usage": {"input_tokens": 9},
                "routing": {"model": "multilingual", "repo": "convaiinnovations/laya"},
            }
        )
        _, raw = await self.raw(Server(health=health, script=[routed]))
        self.assertEqual(
            (raw["served_by"]["device"], raw["served_by"]["checkpoint"]), ("cpu", "convaiinnovations/laya-multilingual")
        )

    async def test_served_by_in_the_response_wins(self) -> None:
        sent = {
            "checkpoint": "convaiinnovations/laya",
            "revision": "abcdef1234567890",
            "device": "mps",
            "weights_dtype": "float16",
        }
        with_served_by = lambda request: ok(  # noqa: E731
            {
                "model": "english",
                "served_by": sent,
                "answers": {"pick": laya_answer(laya_question(PICK))},
                "usage": {"input_tokens": 9},
            }
        )
        decision, raw = await self.raw(Server(script=[with_served_by]))
        self.assertEqual(
            (decision.model, raw["served_by"]["source"]), ("convaiinnovations/laya@abcdef123456", "response")
        )

    async def test_server_timing_is_kept(self) -> None:
        timed = lambda request: ok(  # noqa: E731
            {"answers": {"pick": laya_answer(laya_question(PICK))}, "usage": {"input_tokens": 9}},
            headers={"Server-Timing": "queue;dur=0.2, infer;dur=27.4"},
        )
        _, raw = await self.raw(Server(script=[timed]))
        self.assertEqual(raw["server_timing"], {"queue": 0.2, "infer": 27.4})

    async def test_a_reading_older_than_the_limit_is_refreshed(self) -> None:
        clock = Clock()
        server = Server()
        model, _ = make_model(server, clock=clock)
        await model.warm()
        await model.decide_many(OBSERVATION, {"pick": PICK})
        clock.now += HEALTH_MAX_AGE_S - 1
        await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(server.health_calls, 1)
        server.health = {
            **WORKER_HEALTH,
            "models": {"english": {**WORKER_HEALTH["models"]["english"], "device": "cpu"}},
        }
        clock.now += 2
        decision = await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual((server.health_calls, decision.raw["served_by"]["device"]), (2, "cpu"))

    async def test_a_failed_refresh_keeps_the_previous_reading(self) -> None:
        clock = Clock()
        server = Server()
        model, _ = make_model(server, clock=clock)
        await model.warm()
        server.health = httpx.ReadTimeout("slow")
        clock.now += HEALTH_MAX_AGE_S + 1
        with self.assertLogs("s1a.decision_models.served", level="WARNING"):
            decision = await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(decision.raw["served_by"]["device"], "mps")


class RecordedResponseTests(IsolatedAsyncioTestCase):
    async def test_recorded_worker_and_laya_serve_answers_validate(self) -> None:
        for server_name in ("worker", "frontend", "laya-serve"):
            record = fixture(f"{server_name}.combined")
            request = record["request"]
            questions = {
                "department": ChoiceQuestion(request["questions"]["department"]["criteria"], rules="route"),
                "refund": NoulQuestion(request["questions"]["refund"]["instructions"]),
            }
            payload = dict(record["body"])
            payload["answers"] = {k: payload["answers"][k] for k in questions}
            model, _ = make_model(Server(script=[ok(payload)]))
            decision = await model.decide_many(Observation(request["state"]), questions)
            self.assertEqual(decision.choice("department").key, "billing", server_name)
            self.assertGreater(decision.noul("refund").p, 0.5, server_name)


class ConfigTests(IsolatedAsyncioTestCase):
    def env(self, **values: str) -> Any:
        keys = (
            "LAYA_SERVED_URL",
            "LAYA_SERVED_MODEL",
            "LAYA_SERVED_API_KEY",
            "LAYA_SERVED_TIMEOUT_S",
            "LAYA_SERVED_MAX_LEN",
        )
        return patch.dict(os.environ, {**{k: "" for k in keys}, **values})

    async def test_the_url_is_required(self) -> None:
        with self.env(), self.assertRaises(BaseError) as caught:
            ServedLayaModel.from_env()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
        self.assertIn("LAYA_SERVED_URL", str(caught.exception))

    async def test_defaults_and_overrides(self) -> None:
        with self.env(LAYA_SERVED_URL="http://127.0.0.1:8000"):
            model = ServedLayaModel.from_env()
            self.assertEqual(
                (model.model, model._max_len, model._client.url), ("english", 512, "http://127.0.0.1:8000")
            )
            await model.close()
        with self.env(
            LAYA_SERVED_URL="http://h:8080/",
            LAYA_SERVED_MODEL="multilingual",
            LAYA_SERVED_MAX_LEN="1024",
            LAYA_SERVED_TIMEOUT_S="2.5",
            LAYA_SERVED_API_KEY="k",
        ):
            model = ServedLayaModel.from_env()
            self.assertEqual((model.model, model._max_len, model._client.url), ("multilingual", 1024, "http://h:8080"))
            self.assertEqual(model._client._client.headers["authorization"], "Bearer k")
            await model.close()

    async def test_bad_numbers_are_configuration_errors(self) -> None:
        for values in ({"LAYA_SERVED_TIMEOUT_S": "soon"}, {"LAYA_SERVED_MAX_LEN": "0"}):
            with self.env(LAYA_SERVED_URL="http://h", **values), self.assertRaises(BaseError) as caught:
                ServedLayaModel.from_env()
            self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)

    async def test_no_cloud_key_is_read(self) -> None:
        with (
            self.env(LAYA_SERVED_URL="http://h"),
            patch.dict(os.environ, {"TYPESAFE_API_KEY": "", "OPENROUTER_API_KEY": ""}),
        ):
            await ServedLayaModel.from_env().close()


def test_server_timing_parsing() -> None:
    assert parse_server_timing(None) == {}
    assert parse_server_timing('queue;dur=0.2, infer;desc="x";dur=27.4, proxy, bad;dur=x') == {
        "queue": 0.2,
        "infer": 27.4,
    }
