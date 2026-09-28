# coding: utf-8
"""A Windows or macOS app window through Cua Driver: the goal, app and expected result come as flags.

The Calculator example::

    s1a run desktop --app Calculator --goal "compute 12 times 7" --expect 84 --execute \\
        --plan "1,2,Multiply|×,7,Equals|=" --clear "All Clear" --model jev --rethink off --episodes 1

On Windows use ``--app "Windows Calculator" --expect "Display is 84"`` and match the UIA button labels with
``--plan "One,Two,Multiply by,Seven,Equals" --clear Clear``. Result text is matched exactly, in the app's language.

Without ``--execute`` the run is a dry run: one decision, recorded as ``planned``, nothing clicked. ``--plan`` is the
rule baseline (``--model rule``): button labels in order, ``|`` between variants of one label. Needs ``cua-driver`` on
PATH; macOS also needs Accessibility and Screen Recording granted.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from s1a.desktop.driver import CuaDriver, Snapshot, driver_from_env, opened
from s1a.desktop.env import ABSTAIN, DONE, WindowEnv, clickable
from s1a.spec import Budget, Series, ToolAgentSpec, positive_float, positive_int

RULES = (
    "A desktop app window. goal says what to do; elements lists the window's controls with their labels and values; "
    "presses lists what was clicked so far. Click the one control that moves the goal forward, one click per turn. "
    "When the window shows the goal's result, pick done. Pick abstain only when no offered click helps."
)


def shows(snapshot: Snapshot, text: str) -> bool:
    """Whether a non-clickable element (a display, a label) shows ``text``: the task's finish line."""
    wanted = text.strip()
    if not wanted:
        return False
    return any((e.value.strip() == wanted or e.label.strip() == wanted) for e in snapshot.elements if not clickable(e))


def parse_plan(text: str) -> tuple[tuple[str, ...], ...]:
    """``"1,2,Multiply|×"`` to ``(("1",), ("2",), ("Multiply", "×"))``."""
    return tuple(tuple(v.strip() for v in step.split("|")) for step in text.split(",") if step.strip())


def plan_rule(plan: tuple[tuple[str, ...], ...]) -> Any:
    """The baseline: the next button of the plan by how many presses were made, then done."""

    def rule(state: dict[str, Any], candidates: dict[str, str]) -> str:
        step = len(state["presses"])
        if step >= len(plan):
            return DONE
        return next((f"click:{label}" for label in plan[step] if f"click:{label}" in candidates), ABSTAIN)

    return rule


async def launch_app(app: str, driver: CuaDriver) -> None:
    """Windows uses the driver's launch result; macOS keeps ``open -a`` and name-based discovery."""
    if sys.platform == "win32":
        await driver.launch_app(app)
        return
    process = await asyncio.create_subprocess_exec("open", "-a", app)
    returncode = await process.wait()
    if returncode != 0:
        raise RuntimeError(f"open -a {app!r} exited {returncode}")
    await asyncio.sleep(1.0)  # the window appears after open returns


@asynccontextmanager
async def _session(driver: CuaDriver, app: str) -> AsyncIterator[None]:
    async with opened(driver):
        await launch_app(app, driver)
        yield


def make_series(flags: argparse.Namespace) -> Series:
    driver = driver_from_env("s1a-desktop")  # raises before the series starts when the driver is missing
    plan = parse_plan(flags.plan) if flags.plan else ()
    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=lambda seed: WindowEnv(
            driver,
            app_name=flags.app,
            goal=flags.goal,
            done_when=lambda snapshot: shows(snapshot, flags.expect),
            execute=flags.execute,
            clear_labels=tuple(v.strip() for v in flags.clear.split(",") if v.strip()),
        ),
        session=_session(driver, flags.app),
        baseline=("plan", plan_rule(plan)) if plan else None,
        annotate=lambda env, episode: None,
    )


def flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--app", required=True, help="app name (Windows Calculator / Calculator), or a Windows AUMID")
    parser.add_argument("--goal", required=True, help="what to do in the window, read by the model on every turn")
    parser.add_argument("--expect", required=True, help="the text a display or label shows when the goal is met")
    parser.add_argument("--execute", action="store_true", help="click for real; without it one decision is planned")
    parser.add_argument("--plan", default="", help="the rule baseline: button labels in order, | between variants")
    parser.add_argument("--clear", default="", help="button labels pressed on reset when the window has one")
    parser.add_argument(
        "--rethink-attempts",
        type=positive_int,
        default=3,
        help="bounded rethink: stalls handled by a refresh and a plan before the episode gives up",
    )
    parser.add_argument(
        "--rethink-timeout",
        type=positive_float,
        default=15.0,
        help="bounded rethink: seconds across all refreshes and plans in one episode",
    )


SPEC = ToolAgentSpec(
    name="desktop",
    description="A Windows or macOS app window through Cua Driver: click controls toward --goal until --expect appears.",
    rules=RULES,
    budget=Budget(max_steps=12, timeout_s=90, stall_after=3),
    flags=flags,
    series=make_series,
)
