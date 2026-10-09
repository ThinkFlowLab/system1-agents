# coding: utf-8
"""``clm``: the question shape CLM's schema takes, the error mapping, and every place that must offer it.

A ``httpx.MockTransport`` stands in for ``clm-serve``; no network and no CUDA. The shape asserted here is the one
``clm.client`` builds, which is what ``clm-serve`` documents in its own ``POST /v1/systemone`` docstring.
"""

from __future__ import annotations

import ast
import json
import os
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import httpx
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

from s1a import mcp_server
from s1a.browser.browse import BROWSER_MODEL_NAMES
from s1a.cli import DECIDE_MODEL_NAMES
from s1a.decision_models import DECISION_MODEL_NAMES, ChoiceQuestion, NoulQuestion, Observation
from s1a.decision_models.clm import ClmClient, ClmModel, clm_question
from s1a.rails import RAIL_MODEL_NAMES
from s1a.agents.ticket_router import RULES, TicketRouterEnv
from s1a.tool.loop import MODEL_NAMES
from tests.decision_model_contract import DecisionModelContract

SOURCE = Path(__file__).resolve().parents[1] / "s1a"
URL = "http://clm.test"
OBSERVATION = Observation({"ticket": "I was charged twice. Please refund the duplicate."})
PICK = ChoiceQuestion({"billing": "Charges and refunds", "technical": "Software problems"}, rules="route it")
CHECK = NoulQuestion("Does the customer ask for a refund?")
# What `s1a/browser/action_space.py` builds for a `<op>_target` head: object criteria, strings and bools alike.
BROWSER_PICK = ChoiceQuestion(
    {
        "e1": {"element": "[e1] Sign in", "current_value": "", "role": "button"},
        "e2": {"element": "[e2] Apply coupon", "current_value": "", "role": "button"},
        "e3": {"element": "[e3] Gift wrap", "current_value": "", "role": "checkbox", "checked": True},
    },
    goal="Pick the element to click.",
)


def clm_answer(question: dict[str, Any]) -> dict[str, Any]:
    """An answer the way ``clm-serve`` shapes it, for whatever question it was asked."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.7, "confidence": 0.7}
    keys = list(question["criteria"])
    return {
        "type": "choice",
        "choice": keys[0],
        "confidence": 0.9,
        "probabilities": {key: (0.9 if i == 0 else 0.1 / max(1, len(keys) - 1)) for i, key in enumerate(keys)},
    }


def answer_for(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": body.get("model", "clm-latest"),
        "answers": {name: clm_answer(q) for name, q in body["questions"].items()},
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }


ANSWER = answer_for({"model": "clm-latest", "questions": {"pick": clm_question(PICK)}})


def ok(payload: dict[str, Any], headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(200, json=payload, headers=headers)


class Server:
    """One scripted ``clm-serve``: the answers it gives, and the requests it saw."""

    def __init__(self, script: list[httpx.Response] | None = None, models: dict[str, Any] | None = None) -> None:
        self.script = list(script or [])
        # /health answers from here, which is also how a test makes the readiness read itself fail.
        self.models: Any = (
            models
            if models is not None
            else {"ok": True, "embedder": True, "models": ["clm-latest", "clm-raw"], "cache": {"device": "cuda"}}
        )
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/health":
            if isinstance(self.models, Exception):
                raise self.models
            return ok(self.models)
        if self.script:  # the decisions, in the order the test listed them
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step
        return ok(answer_for(json.loads(request.content)))

    def client(self, **kwargs: Any) -> ClmClient:
        return ClmClient(url=URL, transport=httpx.MockTransport(self.handler), **kwargs)

    def model(self, **kwargs: Any) -> ClmModel:
        return ClmModel(self.client(), **kwargs)


class ClmContract(DecisionModelContract, IsolatedAsyncioTestCase):
    """The contract every backend passes: shape, validation, re-asks, images, close."""

    def make(self) -> ClmModel:
        return Server().model()

    def make_scripted(self, answers: list[dict[str, Any]]) -> ClmModel:
        script = [ok({"model": "clm-latest", "answers": a, "usage": {"input_tokens": 12}}) for a in answers]
        return Server(script=script).model()


class TestClmQuestion(TestCase):
    def test_a_choice_sends_its_options_as_criteria(self) -> None:
        self.assertEqual(
            clm_question(PICK),
            {
                "type": "choice",
                "instructions": "route it",
                "criteria": {"billing": "Charges and refunds", "technical": "Software problems"},
            },
        )

    def test_the_instructions_are_text_and_not_the_object_jev_wants(self) -> None:
        """CLM's state head embeds ``state + instructions`` as prose; a Jev-shaped object is not what it reads."""
        question = ChoiceQuestion({"a": "one"}, goal="Pick one.", operation="choose", rules=("r1", "r2"))
        self.assertEqual(clm_question(question)["instructions"], "Pick one.\nchoose\nr1\nr2")

    def test_a_question_with_nothing_to_say_sends_no_instructions(self) -> None:
        self.assertIsNone(clm_question(ChoiceQuestion({"a": "one"}))["instructions"])

    def test_a_noul_carries_criteria_only_when_the_caller_gave_them(self) -> None:
        self.assertEqual(clm_question(CHECK), {"type": "noul", "instructions": "Does the customer ask for a refund?"})
        spelled = NoulQuestion("Is it so?", {"true": "yes", "false": "no"})
        self.assertEqual(clm_question(spelled)["criteria"], {"true": "yes", "false": "no"})

    def test_a_single_rule_is_a_string_and_several_are_joined(self) -> None:
        self.assertEqual(clm_question(ChoiceQuestion({"a": "one"}, rules="just this"))["instructions"], "just this")
        self.assertEqual(clm_question(ChoiceQuestion({"a": "one"}, rules=("a", "b")))["instructions"], "a\nb")


class TestClmModel(IsolatedAsyncioTestCase):
    async def test_the_body_is_the_one_clm_serve_documents(self) -> None:
        server = Server()
        await server.model().decide_many(OBSERVATION, {"pick": PICK})
        post = server.requests[-1]
        self.assertEqual(post.url, httpx.URL(URL + "/v1/systemone"))
        body = json.loads(post.content)
        self.assertEqual(body["model"], "clm-latest")
        self.assertEqual(body["state"], OBSERVATION.state)
        self.assertEqual(body["questions"]["pick"]["criteria"], dict(PICK.options))

    async def test_the_answer_and_who_served_it_are_recorded(self) -> None:
        server = Server()
        model = server.model()
        await model.warm()
        decision = await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(decision.choice("pick").key, "billing")
        self.assertEqual(decision.model, "clm-latest")
        self.assertEqual(decision.usage.input_tokens, 12)
        self.assertEqual(decision.provenance["served_by"]["url"], URL)

    async def test_a_server_that_is_not_there_names_the_recipe(self) -> None:
        server = Server(script=[httpx.ConnectError("refused"), httpx.ConnectError("refused")])
        with self.assertRaises(BaseError) as caught:
            await server.model().decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("recipe/clm/README.md", str(caught.exception))

    async def test_the_readiness_read_is_the_path_the_omni_frontend_routes(self) -> None:
        """clm-serve also serves /v1/models, but omni-jev routes only /v1/systemone and /health."""
        server = Server()
        await server.model().warm()
        self.assertEqual([r.url.path for r in server.requests], ["/health"])

    async def test_warm_reports_a_server_that_is_not_listening(self) -> None:
        server = Server(models=httpx.ConnectError("refused"))
        with self.assertRaises(BaseError) as caught:
            await server.model().warm()
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)

    async def test_an_error_status_is_a_failed_call_and_keeps_the_server_s_words(self) -> None:
        server = Server(script=[httpx.Response(500, text="cache exhausted")])
        with self.assertRaises(BaseError) as caught:
            await server.model().decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("cache exhausted", str(caught.exception))

    async def test_a_body_without_answers_is_a_failed_call(self) -> None:
        server = Server(script=[ok({"model": "clm-latest"})])
        with self.assertRaises(BaseError) as caught:
            await server.model().decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)

    async def test_a_decision_records_the_url_and_request_id_without_a_warm_call(self) -> None:
        """`decide` and `probe` never call `warm()`; their record still has to say who answered."""
        server = Server()
        decision = await server.model().decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(decision.provenance["url"], URL)
        self.assertEqual(len(decision.provenance["request_id"]), 32)
        self.assertEqual(decision.provenance["served_by"]["models"], server.models["models"])
        self.assertEqual(decision.provenance["served_by"]["device"], "cuda")
        self.assertIs(decision.provenance["served_by"]["embedder"], True)

    async def test_the_readiness_read_happens_once(self) -> None:
        server = Server()
        model = server.model()
        for _ in range(3):
            await model.decide_many(OBSERVATION, {"pick": PICK})
        reads = [r for r in server.requests if r.url.path == "/health"]
        self.assertEqual(len(reads), 1)

    async def test_a_readiness_read_that_fails_does_not_fail_the_decision(self) -> None:
        server = Server(models=httpx.ConnectError("refused"))
        decision = await server.model().decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(decision.choice("pick").key, "billing")
        self.assertIsNone(decision.provenance["served_by"]["models"])
        self.assertIsNone(decision.provenance["served_by"]["read_at"])  # nothing was read

    async def test_the_served_by_reading_says_when_it_was_taken(self) -> None:
        """One read is kept for the model's lifetime, so the record has to date itself."""
        decision = await Server().model().decide_many(OBSERVATION, {"pick": PICK})
        read_at = decision.provenance["served_by"]["read_at"]
        self.assertIsNotNone(read_at)
        age = datetime.now(timezone.utc) - datetime.fromisoformat(read_at)
        self.assertLess(abs(age.total_seconds()), 60)

    async def test_the_engine_s_own_latency_surfaces_as_server_timing(self) -> None:
        """`X-CLM-Latency-Ms` is the engine's scoring time; `server_timing` is the provenance key for durations."""
        server = Server(script=[ok(ANSWER, {"X-CLM-Latency-Ms": "41.5", "Server-Timing": "queue;dur=1.5"})])
        decision = await server.model().decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(decision.provenance["server_timing"], {"queue": 1.5, "clm": 41.5})

    async def test_an_engine_latency_that_is_not_a_number_is_dropped(self) -> None:
        server = Server(script=[ok(ANSWER, {"X-CLM-Latency-Ms": "later"})])
        decision = await server.model().decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(decision.provenance["server_timing"], {})
        self.assertEqual(decision.choice("pick").key, "billing")  # a bad header is not a failed decision

    async def test_a_retry_happens_once_on_a_refused_connection(self) -> None:
        server = Server(
            script=[
                httpx.ConnectError("refused"),
                ok(answer_for({"model": "clm-latest", "questions": {"pick": clm_question(PICK)}})),
            ]
        )
        model = server.model()
        await model.warm()  # the readiness read is one request of its own; count only the decision's
        decision = await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(decision.choice("pick").key, "billing")
        posts = [r for r in server.requests if r.url.path == "/v1/systemone"]
        self.assertEqual(len(posts), 2)

    def test_from_env_needs_a_url_and_reads_the_rest(self) -> None:
        with patch.dict(os.environ, {"CLM_URL": ""}, clear=False):
            with self.assertRaises(BaseError) as caught:
                ClmModel.from_env()
            self.assertEqual(caught.exception.status, StatusCode.MODEL_SERVICE_CONFIG_ERROR)
            self.assertIn("CLM_URL", str(caught.exception))
        with patch.dict(os.environ, {"CLM_URL": URL, "CLM_MODEL": "clm-raw"}, clear=False):
            self.assertEqual(ClmModel.from_env().model, "clm-raw")
        with patch.dict(os.environ, {"CLM_URL": URL, "CLM_TIMEOUT_S": "0"}, clear=False):
            with self.assertRaises(BaseError):
                ClmModel.from_env()


def places_missing_clm() -> list[str]:
    """Every tuple, list, set, ``match`` or ``Literal`` that offers an HTTP decision model and not ``clm``."""
    missing = []
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                values = {e.value for e in node.elts if isinstance(e, ast.Constant)}
            elif isinstance(node, ast.Match):
                values = set().union(
                    *({n.value for n in ast.walk(case.pattern) if isinstance(n, ast.Constant)} for case in node.cases)
                )
            else:
                continue
            if "jev" in values and "clm" not in values:
                missing.append(f"{path.relative_to(SOURCE.parent).as_posix()}:{node.lineno}")
    return missing


def test_every_place_that_offers_jev_offers_clm() -> None:
    assert places_missing_clm() == []


def test_the_named_lists() -> None:
    for names in (DECISION_MODEL_NAMES, DECIDE_MODEL_NAMES, RAIL_MODEL_NAMES, MODEL_NAMES, BROWSER_MODEL_NAMES):
        assert "clm" in names
    decide_model = typing.get_type_hints(
        mcp_server.decide.__wrapped__ if hasattr(mcp_server.decide, "__wrapped__") else mcp_server.decide
    )["model"]
    assert "clm" in typing.get_args(decide_model)


def test_the_check_would_catch_a_missing_place(tmp_path: Path, monkeypatch: Any) -> None:
    (tmp_path / "s1a").mkdir()
    (tmp_path / "s1a" / "new_front.py").write_text('NAMES = ("jev", "laya")\n', encoding="utf-8")
    monkeypatch.setattr(__name__ + ".SOURCE", tmp_path / "s1a")
    assert places_missing_clm() == ["s1a/new_front.py:1"]


FIXTURES = Path(__file__).resolve().parent / "data" / "clm"


def recorded(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class TestRecordedResponses(IsolatedAsyncioTestCase):
    """Replay what a real clm-serve sent, recorded by ``tests/data/clm/record.py``.

    The bodies are the server's, not ours: if it changes shape, these fail rather than the hand-written mock
    quietly agreeing with itself.
    """

    def server(self, name: str) -> Server:
        entry = recorded(name)
        response = httpx.Response(
            entry["status"],
            json=entry["body"],
            headers={"Content-Type": entry.get("content_type", "application/json")},
        )
        return Server(script=[response])

    async def test_a_recorded_choice_answer_is_read(self) -> None:
        decision = (
            await self.server("choice")
            .model()
            .decide_many(Observation({"ticket": "I was charged twice for order 4411."}), {"pick": PICK})
        )
        choice = decision.choice("pick")
        self.assertIn(choice.key, PICK.options)
        self.assertAlmostEqual(sum(choice.probabilities.values()), 1.0, delta=0.02)
        self.assertEqual(decision.model, "clm-latest")

    async def test_a_recorded_noul_answer_is_read(self) -> None:
        decision = await self.server("noul").model().decide_many(OBSERVATION, {"check": CHECK})
        answer = decision.answers["check"]
        self.assertGreaterEqual(answer.p, 0.0)
        self.assertLessEqual(answer.p, 1.0)
        self.assertAlmostEqual(answer.p, 0.9080319883987279, places=9)

    async def test_a_recorded_rejection_keeps_the_server_s_reason(self) -> None:
        with self.assertRaises(BaseError) as caught:
            await self.server("bad_question").model().decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.status, StatusCode.MODEL_CALL_FAILED)
        self.assertIn("criteria", str(caught.exception))

    async def test_a_recorded_models_list_names_what_is_served(self) -> None:
        entry = recorded("models")
        served = await Server(models=entry["body"]).model()._client.warm()
        self.assertEqual([m["name"] for m in served["models"]], ["clm-latest", "clm-raw"])

    async def test_a_recorded_browser_target_answer_is_read(self) -> None:
        """The browser front's `<op>_target` heads send object criteria, not `{key: description}`. clm-serve takes
        them (recorded: 200, not 422) and answers over the keys it was given. Only the shape is asserted -- the
        fixture was recorded against a stub encoder, so its numbers are the stub's."""
        server = self.server("browser_target")
        decision = await server.model().decide_many(
            Observation({"url": "https://example.test/checkout"}), {"click_target": BROWSER_PICK}
        )
        choice = decision.choice("click_target")
        self.assertIn(choice.key, BROWSER_PICK.options)
        self.assertAlmostEqual(sum(choice.probabilities.values()), 1.0, delta=0.02)
        sent = json.loads(server.requests[-1].content)["questions"]["click_target"]["criteria"]
        self.assertEqual(sent["e3"], BROWSER_PICK.options["e3"])  # sent as the object it was built as
        self.assertIs(sent["e3"]["checked"], True)


class TestWhatTheLoopSends(IsolatedAsyncioTestCase):
    """The claim the ticket-router evidence rests on: the backend is a faithful translation.

    Nothing the loop produced is dropped or rewritten on the way to the engine — the state is the observation, the
    criteria are the environment's candidates, and the rules survive in full. This was a one-off script while the
    question was open; it is a test now, because it is the thing a reviewer has to take on trust otherwise.
    """

    async def test_the_loops_own_request_reaches_the_wire_unchanged(self) -> None:
        env = TicketRouterEnv(seed=0, batch_size=1)
        await env.reset()
        observation = await env.observe()
        candidates = await env.candidates()
        question = ChoiceQuestion(candidates, rules=RULES)  # exactly what s1a/tool/models.py builds

        server = Server()
        await server.model().decide_many(Observation(observation), {"pick": question})
        body = json.loads(server.requests[-1].content)

        self.assertEqual(body["state"], observation)
        self.assertEqual(body["questions"]["pick"]["criteria"], candidates)
        self.assertIn(RULES, body["questions"]["pick"]["instructions"])
        self.assertEqual(set(body["questions"]["pick"]["criteria"]), set(candidates))

    async def test_a_transient_gateway_status_is_retried_once(self) -> None:
        """omni-jev answers 502 when it cannot reach the engine behind it and 504 when it was too slow."""
        for status in (502, 504):
            with self.subTest(status=status):
                server = Server(
                    script=[
                        httpx.Response(status, text="backend unavailable"),
                        ok(answer_for({"model": "clm-latest", "questions": {"pick": clm_question(PICK)}})),
                    ]
                )
                decision = await server.model().decide_many(OBSERVATION, {"pick": PICK})
                self.assertEqual(decision.choice("pick").key, "billing")
                self.assertEqual(len([r for r in server.requests if r.url.path == "/v1/systemone"]), 2)
