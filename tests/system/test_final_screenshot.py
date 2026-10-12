# coding: utf-8
"""Opt-in check that a browser run leaves the page it ended on as ``final.png``: ``browse`` end to end in a headless
Chromium through @playwright/mcp, on a local page, with a scripted decision model and a chat model on a closed port.

Run with ``S1A_BROWSER_TESTS=1 pytest tests/system``; needs Node (``npx``) and no key.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.foundation.llm import init_model
from openjiuwen.core.runner import Runner

from s1a.agents import flights
from s1a.browser import browse
from s1a.browser.decision_model import BrowserPolicy
from s1a.decision_models import ScriptedModel
from s1a.tool.hands import serve_static

PAGE = "<!doctype html><title>Results</title><body><h1>Three flights from 412 USD</h1><button>Search</button></body>"

pytestmark = pytest.mark.skipif(
    not os.getenv("S1A_BROWSER_TESTS") or not shutil.which("npx"),
    reason="Set S1A_BROWSER_TESTS=1 with Node installed to run the final screenshot check.",
)


async def _browse(site: Path, logs_dir: Path) -> dict[str, Any]:
    """The policy navigates to the page, then its decision model fails and the run ends BLOCKED on that page."""
    server = serve_static(site)
    await Runner.start()
    try:
        return await browse.browse(
            flights.SPEC,
            BrowserPolicy(prefetch_values=False, batch_actions=False, goal_value_cache=False),
            model_name="jev",
            goal=f"Go to http://127.0.0.1:{server.server_address[1]}/ and report the cheapest flight.",
            timeout_s=60,
            max_steps=4,
            logs_dir=logs_dir,
            headless=True,
            chat=init_model(provider="openai", model_name="m", api_key="k", api_base="http://127.0.0.1:9/v1"),
            decision_model=ScriptedModel(error=build_error(StatusCode.MODEL_CALL_FAILED, error_msg="scripted stop")),
        )
    finally:
        await Runner.stop()
        server.shutdown()
        server.server_close()


def test_a_browser_run_leaves_the_page_it_ended_on_as_final_png() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        site, logs_dir = Path(tmp) / "site", Path(tmp) / "logs"
        site.mkdir()
        (site / "index.html").write_text(PAGE, encoding="utf-8")
        answer = asyncio.run(_browse(site, logs_dir))
        png = (logs_dir / "final.png").read_bytes()
    assert answer.get("status") == "BLOCKED" and "scripted stop" in str(answer["error"]), answer
    assert answer["screenshot"] == str(logs_dir / "final.png") and "screenshot_error" not in answer, answer
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 1_000, "a viewport PNG of the page"
