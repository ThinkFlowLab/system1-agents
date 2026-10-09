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
import math
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from s1a.desktop.driver import CuaDriver, DriverError, Snapshot, driver_from_env, opened
from s1a.desktop.env import ABSTAIN, WindowEnv, clickable, observable
from s1a.spec import Budget, Series, ToolAgentSpec

RULES = (
    "A desktop app window. goal says what to do; elements lists current controls and values; presses lists past "
    "actions. Pick one offered click or type action that moves the goal forward. Type uses the task's "
    "supplied text, never text invented by this decision. After each action the window is observed again. "
    "The environment stops automatically when the result is verified. Pick abstain only when no offered action helps."
)


def shows(snapshot: Snapshot, text: str) -> bool:
    """Whether a non-clickable element (a display, a label) shows ``text``: the task's finish line."""
    wanted = text.strip()
    if not wanted:
        return False
    return any(
        (e.value.strip() == wanted or e.label.strip() == wanted)
        for e in snapshot.elements
        if observable(e) and not clickable(e)
    )


def parse_plan(text: str) -> tuple[tuple[str, ...], ...]:
    """``"1,2,Multiply|×"`` to ``(("1",), ("2",), ("Multiply", "×"))``."""
    return tuple(tuple(v.strip() for v in step.split("|")) for step in text.split(",") if step.strip())


def parse_pixel_targets(entries: list[str]) -> dict[str, tuple[float, float]]:
    """Task-defined points in screenshot fractions; the model chooses among these bounded targets."""
    points: dict[str, tuple[float, float]] = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        try:
            x, y = map(float, value.split(","))
        except ValueError as exc:
            raise ValueError("--pixel-target requires KEY=X,Y with screenshot fractions") from exc
        if not sep or not key.strip() or key in points or not all(math.isfinite(p) and 0 <= p < 1 for p in (x, y)):
            raise ValueError("--pixel-target requires unique keys and finite coordinates in [0, 1)")
        points[key] = (x, y)
    return points


def plan_rule(plan: tuple[tuple[str, ...], ...]) -> Any:
    """The baseline: the next action of the plan by how many actions were made."""

    def rule(state: dict[str, Any], candidates: dict[str, str]) -> str:
        step = len(state["presses"])
        if step >= len(plan):
            return ABSTAIN
        keys = (label if label.startswith(("click:", "type:", "pixel:")) else f"click:{label}" for label in plan[step])
        return next((key for key in keys if key in candidates), ABSTAIN)

    return rule


async def launch_app(app: str, driver: CuaDriver, window_title: str = "") -> None:
    """Windows uses the driver's launch result; macOS keeps ``open -a`` and name-based discovery."""
    if sys.platform == "win32":
        await driver.launch_app(app, window_title)
        return
    process = await asyncio.create_subprocess_exec("open", "-a", app)
    returncode = await process.wait()
    if returncode != 0:
        raise RuntimeError(f"open -a {app!r} exited {returncode}")
    await asyncio.sleep(1.0)  # the window appears after open returns


@asynccontextmanager
async def _session(driver: CuaDriver, app: str, *, owner: str, title: str) -> AsyncIterator[None]:
    async with opened(driver):
        await launch_app(app, driver, title)
        for attempt in range(20):
            try:
                await driver.find_window(owner, title) if title else await driver.find_window(owner)
                break
            except DriverError as exc:
                if not str(exc).startswith("list_windows: 0 on-screen") or attempt == 19:
                    raise
                await asyncio.sleep(0.25)
        yield


def make_series(flags: argparse.Namespace) -> Series:
    if flags.app_path and sys.platform != "darwin":
        raise ValueError("--app-path is supported only on macOS")
    pixel_targets = parse_pixel_targets(flags.pixel_target)
    visual_model = (
        flags.model == "cua"
        and os.getenv("CUA_S1_VARIANT", "nano") == "4b"
        and os.getenv("CUA_S1_MODALITY", "multimodal") == "multimodal"
    )
    if pixel_targets and flags.model not in {"rule", "random"} and not visual_model:
        raise ValueError(
            "--pixel-target requires a screenshot model: use --model cua with CUA_S1_VARIANT=4b and "
            "CUA_S1_MODALITY=multimodal, or an explicit rule/random baseline"
        )
    plan = parse_plan(flags.plan) if flags.plan else ()
    if flags.verify_file and not flags.text:
        raise ValueError("--verify-file requires --text")

    def file_version() -> tuple[int, int, int] | None:
        try:
            stat = Path(flags.verify_file).stat()
            return stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns
        except OSError:
            return None

    def finished(snapshot: Snapshot, initial_file: tuple[int, int, int] | None) -> bool:
        if not shows(snapshot, flags.expect):
            return False
        if not flags.verify_file:
            return True
        try:
            return file_version() != initial_file and Path(flags.verify_file).read_text(encoding="utf-8") == flags.text
        except (OSError, UnicodeError):
            return False

    driver = driver_from_env(f"s1a-desktop-{uuid.uuid4().hex[:12]}")  # one public session per run/transport

    def env_for(seed: int) -> WindowEnv:
        initial_file = file_version() if flags.verify_file else None
        return WindowEnv(
            driver,
            app_name=flags.app,
            window_title=flags.window_title,
            goal=flags.goal,
            done_when=lambda snapshot: finished(snapshot, initial_file),
            execute=flags.execute,
            clear_labels=tuple(v.strip() for v in flags.clear.split(",") if v.strip()),
            text=flags.text,
            text_target=flags.text_target,
            text_mode=flags.text_mode,
            pixel_targets=pixel_targets,
            screenshot=visual_model,
        )

    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=env_for,
        session=_session(driver, flags.app_path or flags.app, owner=flags.app, title=flags.window_title),
        baseline=("plan", plan_rule(plan)) if plan else None,
        annotate=lambda env, episode: None,
    )


def flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--app", required=True, help="app name (Windows Calculator / Calculator), or a Windows AUMID")
    parser.add_argument("--window-title", default="", help="exact window title when the app has multiple windows")
    parser.add_argument(
        "--app-path", default="", help="optional macOS .app bundle path; --app remains the window owner name"
    )
    parser.add_argument("--goal", required=True, help="what to do in the window, read by the model on every turn")
    parser.add_argument("--expect", required=True, help="the text a display or label shows when the goal is met")
    parser.add_argument("--execute", action="store_true", help="act for real; without it one decision is planned")
    parser.add_argument(
        "--plan", default="", help="the rule baseline: button labels or action keys in order, | between variants"
    )
    parser.add_argument("--clear", default="", help="button labels pressed on reset when the window has one")
    parser.add_argument("--text", default="", help="task-supplied text to enter and verify in an editable element")
    parser.add_argument("--text-target", default="", help="restrict --text to this exact field label or identifier")
    parser.add_argument(
        "--text-mode",
        choices=("insert", "replace"),
        default="insert",
        help="insert at selection, or replace a native field's entire value; both require fresh readback",
    )
    parser.add_argument("--verify-file", default="", help="also require this file's UTF-8 content to equal --text")
    parser.add_argument(
        "--pixel-target",
        action="append",
        default=[],
        metavar="KEY=X,Y",
        help="task-defined screenshot point (fractions in [0,1)); repeat for closed visual choices",
    )


SPEC = ToolAgentSpec(
    name="desktop",
    description="A Windows or macOS app window through Cua Driver: choose grounded click and type actions.",
    rules=RULES,
    budget=Budget(max_steps=12, timeout_s=90, stall_after=0),
    flags=flags,
    series=make_series,
)
