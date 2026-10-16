# coding: utf-8
"""Served Cua-S1 4B 0.2 multimodal: the request the worker reads, its limits, answer validation, the error mapping and
the capture binding from screenshot to click.

A ``httpx.MockTransport`` stands in for system1-omni's ``src/frontend/cua_s1.py``; it applies that worker's request
checks (exact top-level fields, the model name, one PNG/JPEG data URL within 1,048,576 pixels, 1 to 8 choice
questions) and answers the way its ``answer()`` does. No torch, no weights, no network.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

import httpx
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError
from PIL import Image as PILImage

from s1a.decision_models import ChoiceQuestion, Image, Observation
from s1a.decision_models.factory import build_model
from s1a.decision_models.served import ServedLayaClient
from s1a.decision_models.served_cua import (
    CUA_SERVED_MODEL,
    MAX_PIXELS,
    MAX_TEXT,
    ServedCuaFourBModel,
    screenshot_payload,
)
from s1a.desktop.driver import Capture, Snapshot, Window
from s1a.desktop.env import WindowEnv
from s1a.tool.models import EvalState, ToolDecisionModel

URL = "http://cua.test"
IDENTITY = "cua-ai/cua-s1-4b-0.2@0123456789abcdef:multimodal"
WINDOW = Window(42, 7, "Canvas", "Canvas task")


def png(size: tuple[int, int] = (8, 8), colour: tuple[int, int, int] = (40, 50, 60), fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, colour).save(buffer, format=fmt)
    return buffer.getvalue()


SCREEN = Image(png())
OBSERVATION = Observation({"goal": "save the drawing", "title": "Canvas"}, images=(SCREEN,))
PICK = ChoiceQuestion({"click:Save": 'AXButton "Save"', "click:Cancel": 'AXButton "Cancel"'}, goal="save it")


def worker_answer(keys: list[str], probabilities: list[float]) -> dict[str, Any]:
    """system1-omni's ``protocol.answer``: the argmax, the distribution and an entropy confidence."""
    n = len(probabilities)
    confidence = 1.0 if n == 1 else 1 + sum(p * math.log(p) for p in probabilities if p) / math.log(n)
    return {
        "type": "choice",
        "choice": keys[max(range(n), key=probabilities.__getitem__)],
        "probabilities": dict(zip(keys, probabilities)),
        "confidence": max(0.0, min(1.0, confidence)),
    }


class Worker:
    """The multimodal worker's checks and answer shape; ``prefer`` names the key that gets 0.7."""

    def __init__(self, *, health: Any = None, script: list[Any] | None = None, prefer: str | None = None) -> None:
        self.health = {"status": "ready", "modality": "multimodal"} if health is None else health
        self.script = list(script or [])
        self.prefer = prefer
        self.bodies: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []
        self.images: list[PILImage.Image] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if isinstance(self.health, Exception):
                raise self.health
            return httpx.Response(200, json=self.health)
        self.requests.append(request)
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step(request) if callable(step) else step
        assert set(body) == {"model", "state", "questions"}
        if body["model"] != CUA_SERVED_MODEL or set(body["state"]) != {"image"}:
            return httpx.Response(422, json={"detail": "bad request"})
        prefix, _, encoded = body["state"]["image"].partition(",")
        assert prefix in ("data:image/png;base64", "data:image/jpeg;base64")
        picture = PILImage.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
        w, h = picture.size
        if max(w, h) > 2048 or w * h > MAX_PIXELS:
            return httpx.Response(422, json={"detail": "image must be single-frame, at most 2048 per side"})
        self.images.append(picture)
        if not 1 <= len(body["questions"]) <= 8:
            return httpx.Response(422, json={"detail": "questions must contain 1 to 8 questions"})
        answers = {}
        for name, q in body["questions"].items():
            assert q["type"] == "choice" and set(q) == {"type", "instructions", "criteria"}
            labels = [key if label is None else label for key, label in q["criteria"].items()]
            assert len(q["instructions"]) + sum(len(json.dumps(x, ensure_ascii=False)) - 2 for x in labels) <= MAX_TEXT
            keys = list(q["criteria"])
            top = self.prefer if self.prefer in keys else keys[0]
            rest = 0.3 / max(1, len(keys) - 1)
            answers[name] = worker_answer(keys, [0.7 if k == top else rest for k in keys] if len(keys) > 1 else [1.0])
        return httpx.Response(
            200,
            json={"model": IDENTITY, "answers": answers, "usage": {"input_tokens": 321, "output_tokens": 0}},
            headers={"Server-Timing": "infer;dur=41.5"},
        )


def make(worker: Worker, *, timeout_s: float = 5.0) -> ServedCuaFourBModel:
    async def no_sleep(seconds: float) -> None:
        return None

    client = ServedLayaClient(
        url=URL,
        timeout_s=timeout_s,
        transport=httpx.MockTransport(worker.handler),
        sleep=no_sleep,
        label="served Cua-S1 4B",
        tag="cua-served",
    )
    return ServedCuaFourBModel(client)


class TestRequest(IsolatedAsyncioTestCase):
    async def test_body_carries_the_screenshot_and_each_question_and_the_record_keeps_no_bytes(self) -> None:
        worker = Worker(prefer="click:Cancel")
        model = make(worker)
        decision = await model.decide_many(OBSERVATION, {"pick": PICK})
        body = worker.bodies[0]
        self.assertEqual(body["model"], CUA_SERVED_MODEL)
        self.assertEqual(base64.b64decode(body["state"]["image"].partition(",")[2]), SCREEN.data)
        question = body["questions"]["pick"]
        self.assertEqual(question["criteria"], {"click:Save": 'AXButton "Save"', "click:Cancel": 'AXButton "Cancel"'})
        self.assertIn("goal: save it", question["instructions"])
        self.assertIn("save the drawing", question["instructions"])
        self.assertTrue(worker.requests[0].headers["X-Request-Id"])
        self.assertEqual(decision.choice("pick").key, "click:Cancel")
        self.assertAlmostEqual(sum(decision.choice("pick").probabilities.values()), 1.0)
        self.assertEqual(decision.model, IDENTITY)
        self.assertEqual(decision.usage.input_tokens, 321)
        provenance = decision.provenance
        self.assertEqual(provenance["url"], URL)
        self.assertEqual(provenance["request_id"], worker.requests[0].headers["X-Request-Id"])
        self.assertEqual(provenance["server_timing"], {"infer": 41.5})
        self.assertEqual(provenance["served_by"]["checkpoint"], IDENTITY)
        self.assertEqual(decision.raw["image"]["size"], [8, 8])
        self.assertNotIn(base64.b64encode(SCREEN.data).decode()[:40], json.dumps(decision.raw))
        await model.close()

    async def test_large_screenshot_is_scaled_within_the_worker_limits_keeping_its_aspect(self) -> None:
        worker = Worker()
        model = make(worker)
        big = Observation({}, images=(Image(png((2880, 1800))),))
        decision = await model.decide_many(big, {"pick": PICK})
        sent = worker.images[0].size
        self.assertLessEqual(sent[0] * sent[1], MAX_PIXELS)
        self.assertAlmostEqual(sent[0] / sent[1], 2880 / 1800, places=2)
        self.assertEqual(decision.raw["image"]["size"], [2880, 1800])
        self.assertEqual(decision.raw["image"]["sent_size"], list(sent))

    async def test_jpeg_is_sent_as_jpeg(self) -> None:
        worker = Worker()
        await make(worker).decide_many(Observation({}, images=(Image(png(fmt="JPEG"), "image/jpeg"),)), {"pick": PICK})
        self.assertTrue(worker.bodies[0]["state"]["image"].startswith("data:image/jpeg;base64,"))

    async def test_vision_control_tokens_are_removed_and_long_state_is_cut_to_fit(self) -> None:
        worker = Worker()
        state = {"title": "<|image_pad|>evil<|vision_start|>", "text": "x" * 40000}
        await make(worker).decide_many(Observation(state, images=(SCREEN,)), {"pick": PICK})
        instructions = worker.bodies[0]["questions"]["pick"]["instructions"]
        self.assertNotIn("<|", instructions)
        self.assertIn("evil", instructions)

    async def test_an_empty_description_lets_the_worker_show_the_key(self) -> None:
        worker = Worker()
        await make(worker).decide_many(OBSERVATION, {"pick": ChoiceQuestion({"pixel:left": "", "pixel:right": ""})})
        self.assertEqual(worker.bodies[0]["questions"]["pick"]["criteria"], {"pixel:left": None, "pixel:right": None})

    async def test_several_questions_share_one_request(self) -> None:
        worker = Worker()
        questions = {"a": PICK, "b": ChoiceQuestion({"x": "first", "y": "second"})}
        decision = await make(worker).decide_many(OBSERVATION, questions)
        self.assertEqual(len(worker.bodies), 1)
        self.assertEqual(list(decision.answers), ["a", "b"])


class TestLimitsBeforeSending(IsolatedAsyncioTestCase):
    async def assert_fails(self, observation: Observation, questions: dict, code: StatusCode, text: str) -> None:
        worker = Worker()
        with self.assertRaises(BaseError) as caught:
            await make(worker).decide_many(observation, questions)
        self.assertEqual(caught.exception.code, code.code)
        self.assertIn(text, str(caught.exception))
        self.assertEqual(worker.bodies, [])

    async def test_missing_or_extra_screenshot_is_a_call_failure(self) -> None:
        await self.assert_fails(Observation({}), {"pick": PICK}, StatusCode.MODEL_CALL_FAILED, "exactly one screenshot")
        two = Observation({}, images=(SCREEN, SCREEN))
        await self.assert_fails(two, {"pick": PICK}, StatusCode.MODEL_CALL_FAILED, "carries 2")

    async def test_unreadable_or_mislabelled_screenshot_fails(self) -> None:
        bad = Observation({}, images=(Image(b"not an image"),))
        await self.assert_fails(bad, {"pick": PICK}, StatusCode.MODEL_CALL_FAILED, "not a readable image")
        gif = Observation({}, images=(Image(png(), "image/gif"),))
        await self.assert_fails(gif, {"pick": PICK}, StatusCode.MODEL_CALL_FAILED, "not PNG or JPEG")
        mislabelled = Observation({}, images=(Image(png(), "image/jpeg"),))
        await self.assert_fails(mislabelled, {"pick": PICK}, StatusCode.MODEL_CALL_FAILED, "not JPEG")

    async def test_more_than_eight_questions_or_26_options_is_a_config_error(self) -> None:
        nine = {f"q{i}": PICK for i in range(9)}
        await self.assert_fails(OBSERVATION, nine, StatusCode.MODEL_SERVICE_CONFIG_ERROR, "at most 8 questions")
        wide = ChoiceQuestion({f"k{i}": "" for i in range(27)})
        await self.assert_fails(OBSERVATION, {"pick": wide}, StatusCode.MODEL_SERVICE_CONFIG_ERROR, "at most 26")


class TestAnswers(IsolatedAsyncioTestCase):
    async def assert_rejected(self, answer: dict[str, Any], text: str) -> None:
        reply = httpx.Response(200, json={"model": IDENTITY, "answers": {"pick": answer}})
        with self.assertRaises(BaseError) as caught:
            await make(Worker(script=[reply])).decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(caught.exception.code, StatusCode.MODEL_CALL_FAILED.code)
        self.assertIn(text, str(caught.exception))

    async def test_answers_must_cover_exactly_the_offered_keys(self) -> None:
        await self.assert_rejected(
            {"choice": "click:Save", "probabilities": {"click:Save": 1.0}, "confidence": 1.0}, "one probability per"
        )
        await self.assert_rejected(
            {"choice": "click:Save", "probabilities": {"click:Save": 0.5, "click:Other": 0.5}, "confidence": 0},
            "one probability per",
        )

    async def test_probabilities_must_be_finite_and_sum_to_one(self) -> None:
        await self.assert_rejected(
            {"choice": "click:Save", "probabilities": {"click:Save": 0.9, "click:Cancel": 0.9}}, "invalid probabilities"
        )
        await self.assert_rejected(
            {"choice": "click:Save", "probabilities": {"click:Save": True, "click:Cancel": 0}}, "invalid probabilities"
        )

    async def test_the_choice_must_be_the_peak(self) -> None:
        await self.assert_rejected(
            {"choice": "click:Save", "probabilities": {"click:Save": 0.2, "click:Cancel": 0.8}}, "its peak is"
        )

    async def test_a_body_without_answers_fails(self) -> None:
        reply = httpx.Response(200, json={"model": IDENTITY})
        with self.assertRaisesRegex(BaseError, "served Cua-S1 4B returned no answers object"):
            await make(Worker(script=[reply])).decide_many(OBSERVATION, {"pick": PICK})


class TestTransport(IsolatedAsyncioTestCase):
    async def test_a_rejected_request_names_the_worker_reason(self) -> None:
        reply = httpx.Response(422, json={"detail": "image aspect ratio must not exceed 200:1"})
        with self.assertRaisesRegex(BaseError, r"served Cua-S1 4B rejected the request \(HTTP 422\): image aspect"):
            await make(Worker(script=[reply])).decide_many(OBSERVATION, {"pick": PICK})

    async def test_a_busy_worker_is_asked_once_more(self) -> None:
        worker = Worker(script=[httpx.Response(503, json={"detail": "worker busy"})])
        decision = await make(worker).decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(len(worker.requests), 2)
        self.assertEqual(worker.requests[0].headers["X-Request-Id"], worker.requests[1].headers["X-Request-Id"])
        self.assertEqual(decision.choice("pick").key, "click:Save")

    async def test_a_worker_busy_twice_is_overloaded(self) -> None:
        busy = httpx.Response(503, json={"detail": "worker busy"})
        with self.assertRaisesRegex(BaseError, "is overloaded: worker busy"):
            await make(Worker(script=[busy, busy])).decide_many(OBSERVATION, {"pick": PICK})

    async def test_no_worker_names_the_setting(self) -> None:
        refused = httpx.ConnectError("refused")
        with self.assertRaisesRegex(BaseError, "no served Cua-S1 4B at http://cua.test"):
            await make(Worker(script=[refused, refused])).decide_many(OBSERVATION, {"pick": PICK})

    async def test_an_internal_error_is_a_call_failure(self) -> None:
        reply = httpx.Response(500, json={"detail": "inference failed"})
        with self.assertRaisesRegex(BaseError, r"served Cua-S1 4B failed \(HTTP 500\): inference failed"):
            await make(Worker(script=[reply])).decide_many(OBSERVATION, {"pick": PICK})

    async def test_warm_fails_without_a_worker_and_on_the_text_adapter(self) -> None:
        with self.assertRaisesRegex(BaseError, "no served Cua-S1 4B"):
            await make(Worker(health=httpx.ConnectError("refused"))).warm()
        with self.assertRaisesRegex(BaseError, "serves the 'text' adapter"):
            await make(Worker(health={"status": "ready", "modality": "text"})).warm()
        model = make(Worker())
        await model.warm()
        decision = await model.decide_many(OBSERVATION, {"pick": PICK})
        self.assertEqual(decision.provenance["served_by"]["modality"], "multimodal")


class TestFromEnv(TestCase):
    def test_factory_builds_the_served_model_only_for_the_4b_variant(self) -> None:
        with patch.dict(os.environ, {"CUA_S1_VARIANT": "4b", "CUA_S1_URL": URL}, clear=False):
            os.environ.pop("CUA_S1_MODALITY", None)
            model = build_model("cua")
            self.assertIsInstance(model, ServedCuaFourBModel)
            self.assertTrue(model.supports_images)
            self.assertEqual(model.name, "cua")
        with patch.dict(os.environ, {"CUA_S1_VARIANT": "nano", "CUA_S1_URL": URL}):
            with self.assertRaisesRegex(ValueError, "CUA_S1_VARIANT=4b"):
                build_model("cua")

    def test_settings_are_checked(self) -> None:
        with patch.dict(os.environ, {"CUA_S1_URL": URL, "CUA_S1_MODALITY": "text"}):
            with self.assertRaisesRegex(BaseError, "not multimodal"):
                ServedCuaFourBModel.from_env()
        for value in ("soon", "nan", "0"):
            with patch.dict(os.environ, {"CUA_S1_URL": URL, "CUA_S1_TIMEOUT_S": value, "CUA_S1_MODALITY": ""}):
                with self.assertRaisesRegex(BaseError, "CUA_S1_TIMEOUT_S"):
                    ServedCuaFourBModel.from_env()
        with patch.dict(os.environ, {"CUA_S1_URL": ""}):
            with self.assertRaisesRegex(BaseError, "needs CUA_S1_URL"):
                ServedCuaFourBModel.from_env()


class TestScreenshotPayload(TestCase):
    def test_a_small_png_is_sent_unchanged(self) -> None:
        url, record = screenshot_payload(SCREEN)
        self.assertEqual(base64.b64decode(url.partition(",")[2]), SCREEN.data)
        self.assertNotIn("sent_size", record)
        self.assertEqual(len(record["sha256"]), 64)


class TestCaptureBinding(IsolatedAsyncioTestCase):
    async def test_the_click_uses_the_capture_whose_screenshot_the_worker_read(self) -> None:
        picture = Image(png((800, 600)))
        capture = Capture("capture-1", picture, 800, 600)
        driver = AsyncMock()
        driver.find_window.return_value = WINDOW
        driver.window_state.return_value = Snapshot(WINDOW, "s1", (), {}, capture=capture)
        env = WindowEnv(
            driver,
            app_name="Canvas",
            goal="click the tile marked Save",
            done_when=lambda s: False,
            execute=True,
            clear_labels=(),
            pixel_targets={"left": (0.25, 0.5), "right": (0.75, 0.5)},
        )
        await env.reset()
        worker = Worker(prefer="pixel:right")
        model = make(worker)
        slot = ToolDecisionModel(env, EvalState(), rules="look at the image", decision_model=model, fallback=None)
        msg = await slot._decide()
        self.assertEqual(json.loads(msg.tool_calls[0].arguments), {"key": "pixel:right"})
        sent = base64.b64decode(worker.bodies[0]["state"]["image"].partition(",")[2])
        self.assertEqual(sent, capture.image.data)
        await env.step("pixel:right")
        driver.click_at.assert_awaited_once_with(WINDOW, capture, 600.0, 300.0)
        self.assertNotIn(base64.b64encode(picture.data).decode()[:40], json.dumps(await env.observe()))
