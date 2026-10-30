# coding: utf-8
"""Hosted Jev screenshots (#97, experimental): off by default, direct TypeSafe only, images prepared to the
documented limits before anything is sent."""

from __future__ import annotations

import base64
import io
import os
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image as PILImage

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError

from s1a.agents import desktop
from s1a.decision_models import ChoiceQuestion, Image, JevModel, NoulQuestion, Observation, ScriptedTransport
from s1a.decision_models import wire
from s1a.decision_models.jev import JEV_IMAGE_MAX_SIDE, jev_image_asset
from s1a.tool import series

HEADER = "TypeSafe-Danger-Zone-Images"
QUESTIONS = {
    "pick": ChoiceQuestion({"a": "the left tile", "b": "the right tile"}, goal="click the red tile"),
    "done": NoulQuestion("Is the task done?"),
}
_CLEAN_ENV = {"TYPESAFE_API_KEY": "", "TYPESAFE_API_URL": "", "OPENROUTER_API_KEY": "", "S1A_JEV_IMAGES": ""}


def _picture(size: tuple[int, int], fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, (200, 30, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


def _image_transport() -> ScriptedTransport:
    transport = ScriptedTransport()
    transport.images = True
    return transport


class TestSetting(TestCase):
    def test_unset_is_off_and_text_only(self) -> None:
        with patch.dict(os.environ, {**_CLEAN_ENV, "TYPESAFE_API_KEY": "k"}):
            client = wire.client_from_env("typesafe", timeout_s=5.0)
            self.assertFalse(client.images)
            self.assertFalse(JevModel(client).supports_images)

    def test_on_for_direct_typesafe(self) -> None:
        for raw in ("1", "true", "YES", " on "):
            with (
                self.subTest(raw=raw),
                patch.dict(os.environ, {**_CLEAN_ENV, "TYPESAFE_API_KEY": "k", "S1A_JEV_IMAGES": raw}),
            ):
                model = JevModel.from_env()
                self.assertTrue(model.supports_images)

    def test_proxy_refuses_the_setting(self) -> None:
        env = {**_CLEAN_ENV, "OPENROUTER_API_KEY": "k", "S1A_JEV_IMAGES": "1"}
        with patch.dict(os.environ, env), self.assertRaises(BaseError) as caught:
            JevModel.from_env()
        self.assertEqual(caught.exception.code, StatusCode.MODEL_SERVICE_CONFIG_ERROR.code)
        self.assertIn("TypeSafe", str(caught.exception))
        # a custom proxy URL next to a TypeSafe key is the proxy backend too
        env = {**_CLEAN_ENV, "TYPESAFE_API_KEY": "k", "TYPESAFE_API_URL": "http://proxy.test/", "S1A_JEV_IMAGES": "1"}
        with patch.dict(os.environ, env), self.assertRaises(BaseError):
            JevModel.from_env()

    def test_unreadable_value_is_a_config_error(self) -> None:
        with patch.dict(os.environ, {**_CLEAN_ENV, "TYPESAFE_API_KEY": "k", "S1A_JEV_IMAGES": "maybe"}):
            with self.assertRaises(BaseError):
                JevModel.from_env()


class TestRequests(IsolatedAsyncioTestCase):
    async def test_text_only_request_is_unchanged_when_off(self) -> None:
        transport = ScriptedTransport()
        model = JevModel(transport)
        observation = Observation({"title": "Canvas"}, images=(Image(_picture((400, 300))),))
        decision = await model.decide_many(observation, QUESTIONS)
        self.assertEqual(set(transport.bodies[0]), {"model", "state", "questions"})
        self.assertEqual(transport.bodies[0]["state"], {"title": "Canvas"})
        self.assertEqual(transport.headers, [{}])
        self.assertNotIn("sent_image", decision.provenance)

    async def test_screenshot_rides_in_assets_with_the_header(self) -> None:
        transport = _image_transport()
        model = JevModel(transport)
        png = _picture((640, 400))
        decision = await model.decide_many(Observation({"title": "Canvas"}, images=(Image(png),)), QUESTIONS)
        body = transport.bodies[0]
        self.assertEqual(transport.headers, [{HEADER: "true"}])
        (reference,) = body["assets"]
        self.assertEqual(body["state"], {"title": "Canvas", "screenshot": reference})
        self.assertEqual(
            body["assets"][reference], {"type": "image", "format": "png", "data_base64": base64.b64encode(png).decode()}
        )
        # referenced exactly once (in the state), never per question
        self.assertNotIn(reference, str(body["questions"]))
        self.assertEqual(
            decision.provenance["sent_image"],
            {"width": 640, "height": 400, "bytes": len(png), "format": "png", "resized": False},
        )

    async def test_fresh_reference_per_request(self) -> None:
        transport = _image_transport()
        model = JevModel(transport)
        observation = Observation({}, images=(Image(_picture((300, 300))),))
        await model.decide_many(observation, QUESTIONS)
        await model.decide_many(observation, QUESTIONS)
        self.assertNotEqual(list(transport.bodies[0]["assets"]), list(transport.bodies[1]["assets"]))

    async def test_text_state_is_wrapped(self) -> None:
        transport = _image_transport()
        await JevModel(transport).decide_many(Observation("a page", images=(Image(_picture((300, 300))),)), QUESTIONS)
        state = transport.bodies[0]["state"]
        self.assertEqual(state["text"], "a page")
        self.assertIn(state["screenshot"], transport.bodies[0]["assets"])

    async def test_no_screenshot_means_a_plain_request(self) -> None:
        transport = _image_transport()
        await JevModel(transport).decide_many(Observation({"title": "x"}), QUESTIONS)
        self.assertNotIn("assets", transport.bodies[0])
        self.assertEqual(transport.headers, [{}])

    async def test_more_than_one_screenshot_fails_before_sending(self) -> None:
        transport = _image_transport()
        image = Image(_picture((300, 300)))
        with self.assertRaises(BaseError) as caught:
            await JevModel(transport).decide_many(Observation({}, images=(image, image)), QUESTIONS)
        self.assertEqual(caught.exception.code, StatusCode.MODEL_CALL_FAILED.code)
        self.assertEqual(transport.bodies, [])

    async def test_oversize_is_resized_and_undersize_rejected(self) -> None:
        transport = _image_transport()
        decision = await JevModel(transport).decide_many(
            Observation({}, images=(Image(_picture((3136, 1960))),)), QUESTIONS
        )
        sent = decision.provenance["sent_image"]
        self.assertEqual((sent["width"], sent["height"], sent["resized"]), (JEV_IMAGE_MAX_SIDE, 980, True))
        (asset,) = transport.bodies[0]["assets"].values()
        with PILImage.open(io.BytesIO(base64.b64decode(asset["data_base64"]))) as picture:
            self.assertEqual(picture.size, (JEV_IMAGE_MAX_SIDE, 980))
        for size in ((279, 600), (4000, 500)):  # too small, or too thin once the long side fits
            with self.subTest(size=size), self.assertRaises(BaseError) as caught:
                await JevModel(_image_transport()).decide_many(
                    Observation({}, images=(Image(_picture(size)),)), QUESTIONS
                )
            self.assertEqual(caught.exception.code, StatusCode.MODEL_SERVICE_CONFIG_ERROR.code)

    async def test_bad_images_are_rejected(self) -> None:
        cases = (
            Image(_picture((300, 300), "GIF"), "image/gif"),
            Image(_picture((300, 300), "JPEG"), "image/png"),  # bytes do not match the media type
            Image(b"not an image"),
        )
        for image in cases:
            with self.subTest(media_type=image.media_type), self.assertRaises(BaseError):
                jev_image_asset(image)
        asset, sent = jev_image_asset(Image(_picture((300, 300), "JPEG"), "image/jpeg"))
        self.assertEqual((asset["format"], sent["format"]), ("jpg", "jpg"))


class TestWireHeader(IsolatedAsyncioTestCase):
    async def test_header_only_on_the_request_that_asks_for_it(self) -> None:
        seen: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers)
            return httpx.Response(200, json={"answers": {}})

        client = wire.JevDecisionsClient(api_key="k", url="https://api.test/v1/systemone", model="m", timeout_s=5.0)
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer k"}
        )
        await client.decide({"model": "m"}, headers={HEADER: "true"})
        await client.decide({"model": "m"})
        await client.close()
        self.assertEqual(seen[0][HEADER], "true")
        self.assertNotIn(HEADER, seen[1])
        self.assertEqual(seen[0]["Authorization"], "Bearer k")


class TestDesktopPixelTargets(TestCase):
    def _args(self):
        return series.parser(desktop.SPEC).parse_args(
            [
                "--model", "jev", "--rethink", "off", "--episodes", "1", "--app", "Canvas",
                "--goal", "click Save", "--expect", "Saved", "--pixel-target", "left=0.25,0.5",
            ]
        )  # fmt: skip

    def test_jev_with_images_may_use_pixel_targets(self) -> None:
        env = {**_CLEAN_ENV, "TYPESAFE_API_KEY": "k", "S1A_JEV_IMAGES": "1"}
        with patch.dict(os.environ, env), patch.object(desktop, "driver_from_env", return_value=AsyncMock()):
            desktop.make_series(self._args())

    def test_jev_without_images_still_refuses(self) -> None:
        with (
            patch.dict(os.environ, {**_CLEAN_ENV, "TYPESAFE_API_KEY": "k"}),
            self.assertRaisesRegex(ValueError, "S1A_JEV_IMAGES"),
        ):
            desktop.make_series(self._args())
