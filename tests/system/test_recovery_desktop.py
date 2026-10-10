# coding: utf-8
"""Opt-in end-to-end check of the desktop recovery eval on a native desktop through the real cua-driver.

Run with ``S1A_DESKTOP_TESTS=1 S1A_DESKTOP_DRIVER=<path-to-cua-driver> pytest tests/system``; it needs a native
desktop session (Windows today) and the system .NET C# compiler to build the fixture. No chat key is used. It drives
the self-built fixture through the repository's ``run_episode`` with the scripted doubles: the recoverable task
completes only with recovery on, and the permanent task ends with the bounded budget spent and a next action. Success
is the fixture's own ``result.json``, never the model's ``DONE``.
"""

from __future__ import annotations

import os

import pytest

from evals.desktop_recovery import (
    TrialPlan,
    default_csc,
    default_driver,
    run_sync,
)


def _ready() -> bool:
    return bool(os.getenv("S1A_DESKTOP_TESTS")) and bool(default_driver()) and bool(default_csc())


pytestmark = pytest.mark.skipif(
    not _ready(),
    reason="Set S1A_DESKTOP_TESTS=1 and S1A_DESKTOP_DRIVER (plus a system .NET csc) to run the desktop check.",
)


def _exercise() -> None:
    run = run_sync(
        trials=[
            TrialPlan("recoverable", "off"),
            TrialPlan("recoverable", "on"),
            TrialPlan("permanent", "on"),
        ]
    )
    records = {f"{record.task}-{record.arm}": record for record in run.records}

    recoverable_off = records["recoverable-off"]
    recoverable_on = records["recoverable-on"]
    permanent_on = records["permanent-on"]

    assert run.summary["planned_trials"] == 3
    assert recoverable_on.verified, "with recovery the plan unlocks the fixture and the oracle says finished"
    assert recoverable_on.recovery_attempts >= 1, "the recoverable task spends at least one recovery attempt"
    assert not recoverable_off.verified, "without recovery the fixture is never solved, whatever the model reports"
    assert not recoverable_off.errored

    assert not permanent_on.verified, "the permanent fixture can never be solved"
    assert permanent_on.recovery_failed, "the permanent task ends because the bounded recovery is spent"
    assert permanent_on.recovery_attempts >= 1
    assert permanent_on.recovery_next_action, "a bounded BLOCKED terminal must carry a next action"
    assert "start a new task" in permanent_on.recovery_next_action
    assert permanent_on.next_action_source in ("event", "terminal"), "the runtime must supply the next action"

    # Only this run's fixture pid is ever touched: every oracle event and the final result name the launched pid.
    for record in run.records:
        pids = record.oracle.get("distinct_pids") or []
        assert pids, f"{record.task}-{record.arm}: no fixture pid recorded"
        assert record.oracle["final"].get("pid") == pids[0]


def test_desktop_recovery_on_off_against_the_native_fixture() -> None:
    _exercise()
