# coding: utf-8
"""Opt-in end-to-end browser task through @playwright/mcp, with no live model calls."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.search_task_fixture import SearchHandler, run_search_task, verify_search_result


@pytest.mark.skipif(
    not os.getenv("S1A_BROWSER_TESTS") or not shutil.which("npx"),
    reason="Set S1A_BROWSER_TESTS=1 with Node installed to run the local search browser check.",
)
def test_browser_search_task_has_a_repeatable_fixture_and_independent_verifier() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        answer, decisions = asyncio.run(run_search_task(Path(tmp) / "logs"))

    verify_search_result(answer, SearchHandler.requests)
    assert decisions.calls >= 2, decisions.calls
