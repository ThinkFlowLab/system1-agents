# coding: utf-8
"""Shared fixture and independent verifier for the bounded local browser-search task."""

from __future__ import annotations

import sys
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


async def run_search_task(logs_dir: Path) -> tuple[dict[str, Any], SearchDecisions]:
    SearchHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), SearchHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    thread_started = False
    try:
        server_thread.start()
        thread_started = True
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
        finally:
            await Runner.stop()
        return result, decisions
    finally:
        try:
            if thread_started:
                server.shutdown()
        finally:
            try:
                server.server_close()
            finally:
                if thread_started:
                    server_thread.join(timeout=2)


def verify_search_result(answer: dict[str, Any], requests: list[str]) -> None:
    """Check the submitted query and visible result independently of the scripted final answer."""
    submitted = [urlsplit(path) for path in requests if urlsplit(path).path == "/search"]
    assert len(submitted) == 1, requests
    assert parse_qs(submitted[0].query) == {"q": ["Riverton"]}
    assert answer.get("status") == "DONE", answer
    assert answer.get("ok") is True, answer
    assert answer.get("final") == "Found Riverton Park.", answer
    page_text = str(answer.get("terminal", {}).get("page_text", ""))
    assert "Riverton results" in page_text, answer
    assert "Riverton Park" in page_text, answer


class _FakeServer:
    server_address = ("127.0.0.1", 12345)

    def __init__(self) -> None:
        self.events: list[str] = []

    def serve_forever(self) -> None:
        self.events.append("serve")

    def shutdown(self) -> None:
        self.events.append("shutdown")

    def server_close(self) -> None:
        self.events.append("close")


class _FakeThread:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def join(self, timeout: float | None = None) -> None:
        self.events.append("join")


def _mock_server(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeServer, _FakeThread]:
    server, thread = _FakeServer(), _FakeThread()
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ThreadingHTTPServer", lambda *args: server)
    monkeypatch.setattr(module, "Thread", lambda *args, **kwargs: thread)
    return server, thread
