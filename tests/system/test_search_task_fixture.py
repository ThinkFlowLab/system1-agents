# coding: utf-8
"""One repeatable local search task through the browser harness, with no live model calls.

Run with ``S1A_BROWSER_TESTS=1 pytest -q tests/system/test_search_task_fixture.py``; needs Node and
the Playwright MCP package used by the browser harness. Each run starts a fresh local server.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from openjiuwen.core.foundation.llm import AssistantMessage, Model
from openjiuwen.core.runner import Runner

from s1a.agents import flights
from s1a.browser import browse
from s1a.browser.decision_model import BrowserPolicy
from s1a.decision_models import ChoiceQuestion, DecisionModel, Observation, Reply, Usage
from s1a.tool.models import placeholder_model

pytestmark = pytest.mark.skipif(
    not os.getenv("S1A_BROWSER_TESTS") or not shutil.which("npx"),
    reason="Set S1A_BROWSER_TESTS=1 with Node installed to run the local search browser check.",
)


class SearchHandler(BaseHTTPRequestHandler):
    requests: list[str] = []

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query).get("q", [""])[0]
        type(self).requests.append(self.path)
        if parsed.path == "/search" and query == "Riverton":
            body = "<!doctype html><title>Search results</title><h1>Riverton results</h1><p>Riverton Park</p>"
        else:
            body = (
                "<!doctype html><title>Directory search</title><h1>Directory</h1>"
                '<form action="/search" method="get"><label>Search directory '
                '<input name="q" value="Riverton"></label><button type="submit">Search</button></form>'
            )
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


class SearchDecisions(DecisionModel):
    name = "scripted-local-search"
    bills_input_tokens = False
    supports_images = False

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model(self) -> str:
        return self.name

    async def _decide(self, observation: Observation, questions: dict[str, Any]) -> Reply:
        self.calls += 1
        state = observation.state
        on_results = "Riverton results" in str(state)
        answers: dict[str, dict[str, Any]] = {}
        for name, question in questions.items():
            if not isinstance(question, ChoiceQuestion):
                answers[name] = {"noul": 0.0}
                continue
            if name == "operation":
                chosen = "DONE" if on_results else "CLICK"
            elif name == "click_target":
                chosen = next(
                    key
                    for key, option in question.options.items()
                    if isinstance(option, dict) and option.get("role") == "button" and "Search" in str(option)
                )
            else:
                chosen = question.ids[0]
            answers[name] = {
                "choice": chosen,
                "probabilities": {key: float(key == chosen) for key in question.ids},
                "confidence": 1.0,
            }
        return Reply(answers=answers, latency_ms=0, usage=Usage(), model=self.name)


class LocalAnswerModel(Model):
    def __init__(self) -> None:
        source = placeholder_model()
        super().__init__(source.model_client_config, source.model_config)

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        return AssistantMessage(content="Found Riverton Park.", finish_reason="stop")


async def _run(logs_dir: Path) -> tuple[dict[str, Any], SearchDecisions]:
    SearchHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), SearchHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    decisions = SearchDecisions()
    await Runner.start()
    try:
        result = await browse.browse(
            flights.SPEC,
            BrowserPolicy(prefetch_values=False, batch_actions=False, goal_value_cache=False),
            model_name="jev",
            goal=f"Search the directory for Riverton and report the matching result at http://127.0.0.1:{server.server_address[1]}/.",
            timeout_s=60,
            max_steps=6,
            logs_dir=logs_dir,
            headless=True,
            chat=LocalAnswerModel(),
            decision_model=decisions,
        )
        return result, decisions
    finally:
        await Runner.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def test_browser_search_task_has_a_repeatable_fixture_and_independent_verifier() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        answer, decisions = asyncio.run(_run(Path(tmp) / "logs"))

    # Verify the application-level effect independently of the browser agent's final answer.
    submitted = [urlsplit(path) for path in SearchHandler.requests if urlsplit(path).path == "/search"]
    assert len(submitted) == 1, SearchHandler.requests
    assert parse_qs(submitted[0].query) == {"q": ["Riverton"]}
    assert answer.get("status") == "DONE", answer
    assert answer.get("ok") is True, answer
    assert answer.get("final") == "Found Riverton Park.", answer
    assert "Riverton results" in str(answer.get("terminal", {}).get("page_text", "")), answer
    assert decisions.calls >= 2, decisions.calls
