# coding: utf-8
"""Opt-in end-to-end check of the recovery eval in a real headless Chromium through @playwright/mcp.

Run with ``S1A_BROWSER_TESTS=1 pytest tests/system``; needs Node (``npx``) and no chat key. It drives the local
fixture pages through the repository's ``browse`` front with the scripted doubles: the normal task completes, the
recoverable task completes only with recovery on (the off arm hits the legacy BLOCKED), and the permanently blocked
task stops clearly with the recovery budget spent. Success is the fixture server's recorded submit, never the model.

Every call to ``run_trial`` opens its own fresh fixture, so the test runs the recoverable ``on`` arm *before* the
``off`` arm: the off trial's empty oracle proving it not verified is the end-to-end isolation check.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from s1a.config import HOME
from s1a.run import started_runner

from evals.recovery.fixture import BLOCKED, NORMAL, RECOVERABLE
from evals.recovery.runner import EvalConfig, run_trial

pytestmark = pytest.mark.skipif(
    not os.getenv("S1A_BROWSER_TESTS") or not shutil.which("npx"),
    reason="Set S1A_BROWSER_TESTS=1 with Node installed to run the browser recovery check.",
)


def _exercise() -> None:
    async def main() -> dict:
        config = EvalConfig(timeout_s=120.0, max_steps=24, headless=True)
        logs = HOME / "runs" / "recovery_browser_test"

        async def trial(task, arm, name):
            return await run_trial(task, arm, 0, config=config, logs_dir=Path(logs) / name)

        async with started_runner():
            normal_off, normal_off_record = await trial(NORMAL, "off", "normal_off")
            rec_on, rec_on_record = await trial(RECOVERABLE, "on", "recoverable_on")
            rec_off, rec_off_record = await trial(RECOVERABLE, "off", "recoverable_off")
            _, blocked_record = await trial(BLOCKED, "on", "blocked_on")
        return {
            "normal_off": normal_off,
            "normal_off_record": normal_off_record,
            "rec_on_record": rec_on_record,
            "rec_off": rec_off,
            "rec_off_record": rec_off_record,
            "blocked_record": blocked_record,
        }

    result = asyncio.run(main())
    normal_off = result["normal_off"]
    normal_off_record = result["normal_off_record"]
    rec_on_record = result["rec_on_record"]
    rec_off = result["rec_off"]
    rec_off_record = result["rec_off_record"]
    blocked_record = result["blocked_record"]

    assert normal_off.error is None and normal_off.score == 1.0, "the normal form completes without recovery"
    assert normal_off_record.verified, "the normal form's own fixture saw the submit"
    assert any(
        record["fields"].get("value") == NORMAL.expected_value for record in normal_off_record.oracle["submissions"]
    )

    assert rec_on_record.verified, "with recovery the plan unlocks the field and the form is submitted"
    assert rec_on_record.recovery_attempts >= 1, "the recoverable task spends at least one recovery attempt"

    assert rec_off.score == 0.0 and rec_off.error is None, "without recovery the reverted field stalls to BLOCKED"
    assert rec_off.final_state["terminal"]["status"] == "BLOCKED"
    assert not rec_off_record.verified, "the off trial's fresh oracle must not inherit the on trial's submit"
    assert rec_off_record.oracle["submissions"] == [], "each trial starts with an empty record namespace"

    assert not blocked_record.verified, "the blocked form can never be submitted"
    assert blocked_record.recovery_failed, "the blocked task ends because the recovery budget is spent"
    assert blocked_record.recovery_attempts >= 1


def test_recovery_on_off_against_the_local_fixture() -> None:
    _exercise()
