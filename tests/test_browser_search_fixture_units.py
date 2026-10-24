# coding: utf-8
"""Fast checks for the browser search fixture's independent verifier and cleanup paths."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tests import search_task_fixture as fixture


def test_independent_verifier_rejects_missing_result_content() -> None:
    answer = {
        "status": "DONE",
        "ok": True,
        "final": "Found Riverton Park.",
        "terminal": {"page_text": "Riverton results"},
    }
    with pytest.raises(AssertionError, match="Riverton Park"):
        fixture.verify_search_result(answer, ["/search?q=Riverton"])


def test_fixture_server_is_closed_when_runner_start_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, thread = fixture._mock_server(monkeypatch)
    start = AsyncMock(side_effect=RuntimeError("runner start failed"))
    stop = AsyncMock()
    monkeypatch.setattr(fixture.Runner, "start", start)
    monkeypatch.setattr(fixture.Runner, "stop", stop)

    with pytest.raises(RuntimeError, match="runner start failed"):
        asyncio.run(fixture.run_search_task(tmp_path / "logs"))

    assert server.events == ["shutdown", "close"]
    assert thread.events == ["start", "join"]
    stop.assert_not_awaited()


def test_fixture_server_is_closed_when_runner_stop_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, thread = fixture._mock_server(monkeypatch)
    monkeypatch.setattr(fixture.Runner, "start", AsyncMock())
    monkeypatch.setattr(fixture.Runner, "stop", AsyncMock(side_effect=RuntimeError("runner stop failed")))
    monkeypatch.setattr(fixture.browse, "browse", AsyncMock(return_value={"status": "DONE"}))

    with pytest.raises(RuntimeError, match="runner stop failed"):
        asyncio.run(fixture.run_search_task(tmp_path / "logs"))

    assert server.events == ["shutdown", "close"]
    assert thread.events == ["start", "join"]
