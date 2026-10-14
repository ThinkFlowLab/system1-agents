# coding: utf-8
"""``BrowserDecisionModel._screenshot``: the viewport PNG an image-reading decision model decides over."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from s1a.browser.decision_model import SCREENSHOT_JS, BrowserDecisionModel, _png_from_run_code

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


def _mcp_text(result_json: str, page_text: str = "- generic [ref=e1]") -> dict[str, Any]:
    """A playwright-mcp run-code result: the returned value as JSON text after ``### Result``, then the page state."""
    text = f"### Result\n{result_json}\n\n### Ran Playwright code\n```js\n...\n```\n\n### Page state\n{page_text}"
    return {"content": [{"type": "text", "text": text}]}


class TestPngFromRunCode(TestCase):
    def test_every_wrapping_of_the_result_object_is_opened(self) -> None:
        returned = json.dumps({"png": ENCODED})
        shapes = {
            "object": {"png": ENCODED},
            "json text": returned,
            "result field": {"result": returned},
            "compact rpc": {"__browser_compact_rpc__": True, "payload": {"result": returned}},
            "mcp text, encoded twice": _mcp_text(json.dumps(returned)),
            "compact rpc around mcp": {"__browser_compact_rpc__": True, "payload": _mcp_text(returned)},
        }
        for name, raw in shapes.items():
            with self.subTest(name):
                self.assertEqual(_png_from_run_code(raw), PNG)

    def test_png_like_text_outside_the_result_object_is_not_the_picture(self) -> None:
        # The capture timed out; the page itself shows text that a search over the whole result would take for it.
        raw = _mcp_text(json.dumps({"error": "timeout"}), page_text=f'- text: png": "{ENCODED}"')
        self.assertIsNone(_png_from_run_code(raw))

    def test_a_png_field_that_is_not_base64_of_a_png_is_no_image(self) -> None:
        for name, png in {
            "not base64": "not base64!",
            "base64 of other bytes": base64.b64encode(b"GIF89a").decode(),
            "not a string": 42,
        }.items():
            with self.subTest(name):
                self.assertIsNone(_png_from_run_code({"png": png}))

    def test_an_unknown_shape_is_no_image(self) -> None:
        for raw in (None, 7, ["png", ENCODED], {"other": ENCODED}):
            with self.subTest(raw=raw):
                self.assertIsNone(_png_from_run_code(raw))
