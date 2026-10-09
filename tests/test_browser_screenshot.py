# coding: utf-8
"""``BrowserDecisionModel._screenshot``: the viewport PNG an image-reading decision model decides over."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase

from s1a.browser.decision_model import SCREENSHOT_JS, BrowserDecisionModel

PNG = b"\x89PNG\r\n\x1a\n fake"
ENCODED = base64.b64encode(PNG).decode()


def _runtime(result: Any = None, error: Exception | None = None, failures: int | None = None) -> SimpleNamespace:
    """``error`` is raised on every call, or on the first ``failures`` calls only."""
    calls: list[str] = []

    async def executor(code: str) -> Any:
        calls.append(code)
        if error is not None and (failures is None or len(calls) <= failures):
            raise error
        return result

    return SimpleNamespace(code_executor=executor, calls=calls)


async def _shot(runtime: Any) -> tuple[Any, int]:
    return await BrowserDecisionModel._screenshot(SimpleNamespace(_runtime=runtime))  # type: ignore[arg-type]


class TestScreenshot(IsolatedAsyncioTestCase):
    async def test_the_png_comes_back_from_the_run_code_payload(self) -> None:
        payload = {"__browser_compact_rpc__": True, "payload": {"result": json.dumps({"png": ENCODED})}}
        runtime = _runtime(payload)
        image, ms = await _shot(runtime)
        self.assertEqual(runtime.calls, [SCREENSHOT_JS])
        self.assertEqual((image.data, image.media_type), (PNG, "image/png"))
        self.assertGreaterEqual(ms, 0)

    async def test_a_failed_capture_is_no_image_not_an_error(self) -> None:
        runtime = _runtime(error=RuntimeError("page closed"))
        image, _ms = await _shot(runtime)
        self.assertIsNone(image)
        self.assertEqual(len(runtime.calls), 2)

    async def test_one_failed_capture_is_retried_once(self) -> None:
        payload = {"payload": {"result": json.dumps({"png": ENCODED})}}
        runtime = _runtime(payload, error=TimeoutError("bringToFront"), failures=1)
        image, _ms = await _shot(runtime)
        self.assertEqual(image.data, PNG)
        self.assertEqual(runtime.calls, [SCREENSHOT_JS, SCREENSHOT_JS])

    async def test_a_result_without_a_png_is_no_image(self) -> None:
        image, _ms = await _shot(_runtime({"payload": {"result": "{}"}}))
        self.assertIsNone(image)
