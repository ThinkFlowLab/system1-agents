# coding: utf-8
"""Native desktop bounded-recovery on/off eval on a self-built, per-run fixture.

This is the reproducible subset of the 2026-09-27 native Windows experiment: ``normal``, ``recoverable`` and
``permanent`` tasks, each with recovery ``off`` and ``on``. It reuses the repository's real machinery unchanged:
the real ``cua-driver`` over its MCP stdio bridge, the real ``CuaDriver`` adapter, the real ``WindowEnv``, the real
``run_episode`` loop and the real ``RethinkRail`` (bounded branch). Only the decision model and the planner are
scripted, for controlled fault injection: this measures the mechanism wiring and the budget, **not** a trained model
and **not** a real app success rate. There is no API call and the cost is 0 by construction.

The fixture source lives under ``tests/fixtures/desktop_recovery`` and is compiled by the system .NET compiler; no
installer is run. Every run gets a unique id, so the process image name and the window title are unique and only this
run's window is ever targeted. The fixture never activates its window and the driver clicks in the background, so the
mouse is not moved and no other app is read or clicked. Only the fixture pid this run started is terminated.

The independent oracle is the fixture's own ``result.json``: success is ``finished`` true, never the model's ``DONE``.
Each trial gets its own fresh fixture and output directory, so no earlier trial's success can verify a later one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk, Model

from s1a.agents import desktop
from s1a.decision_models import RuleModel
from s1a.desktop.driver import CuaDriver, DriverError, opened
from s1a.desktop.env import WindowEnv
from s1a.jobs import RESULTS_DIR, Episode, now_iso, write_job
from s1a.recovery import RecoveryLimits
from s1a.run import started_runner
from s1a.tool import loop
from s1a.tool.models import placeholder_model
from s1a.tool.rethink import progress_digest

from evals.recovery.summary import (
    coverage,
    mean,
    on_minus_off_completion_rate,
    pair_outcomes,
    paired_completion_line,
    summary_tables,
)

REPO = Path(__file__).resolve().parent.parent
FIXTURE_SOURCE = REPO / "tests" / "fixtures" / "desktop_recovery" / "S1AFixture.cs"
TASKS = ("normal", "recoverable", "permanent")
ARMS = ("off", "on")
SCRIPTED_MODEL = "scripted-rule"  # a hand-written rule, not a neural decision model
GOAL = "press unlock then finish so the window title shows finished OK"
PLAN = "Click unlock, then click finish."
DONE_TEXT = "finished OK"
DRIVER_CHANNEL = "not inferred from the executable path"
DEFAULT_DRIVER_VERSION = "unknown"
ORACLE_WAIT_S = 25.0
WINDOW_WAIT_S = 15.0  # the fixture writes its oracle before its window is on screen; wait for the window
CSC_CANDIDATES = (
    r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
    "/mnt/c/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe",
)
TASKKILL = "/mnt/c/Windows/System32/taskkill.exe"


@dataclass(frozen=True)
class TrialPlan:
    """One planned trial: a task, an arm and a repeat index."""

    task: str
    arm: str
    repeat: int = 0


@dataclass(frozen=True)
class DesktopEvalConfig:
    """The fixed budgets and binaries one run uses; on/off share every value but the arm itself."""

    driver_bin: str
    csc: str
    fixture_source: Path = FIXTURE_SOURCE
    max_acts: int = 12
    timeout_s: float = 60.0
    max_recovery_attempts: int = 3
    recovery_timeout_s: float = 15.0
    session_label: str = "s1a-desktop-recovery-eval"
    driver_version: str = DEFAULT_DRIVER_VERSION


@dataclass
class DesktopTrialRecord:
    """One trial's outcome: the independent oracle plus the recovery accounting, all scripted and cost-free."""

    task: str
    arm: str
    repeat: int
    seed: int
    verified: bool
    terminal: str
    errored: bool
    scripted_model: str
    driver: dict[str, Any]
    platform: str
    recovery_attempts: int
    recovery_spent_s: float
    recovery_failed: bool
    recovery_termination: str | None
    recovery_next_action: str | None
    next_action_source: str | None
    wasted_actions: int
    noop_clicks: int
    steps: int
    model_calls: int
    decision_calls: int
    planner_calls: int
    chat_calls: int
    elapsed_s: float
    cost_usd: float
    oracle: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DesktopEvalRun:
    summary: dict[str, Any]
    records: list[DesktopTrialRecord] = field(default_factory=list)
    run_dir: Path | None = None
    job_dirs: list[Path] = field(default_factory=list)


class PlanChat(Model):
    """A scripted planner: one useful plan line per stall, counting its calls. No API, no tokens, no cost."""

    def __init__(self) -> None:
        source = placeholder_model()
        super().__init__(source.model_client_config, source.model_config)
        self.calls = 0

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        self.calls += 1
        return AssistantMessage(content=PLAN)

    async def stream(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AsyncIterator[AssistantMessageChunk]:
        self.calls += 1
        yield AssistantMessageChunk(content=PLAN)


def default_driver() -> str | None:
    return os.getenv("S1A_DESKTOP_DRIVER") or os.getenv("CUA_DRIVER_BIN") or shutil.which("cua-driver")


def default_csc() -> str | None:
    override = os.getenv("S1A_DESKTOP_CSC")
    if override:
        return override
    for candidate in CSC_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return shutil.which("csc") or shutil.which("csc.exe")


def detect_driver_version(driver: str) -> str:
    """Read the supplied binary's version; do not label an arbitrary binary as the tested release."""
    try:
        result = subprocess.run([driver, "--version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    match = re.search(r"\b\d+\.\d+\.\d+(?:[-+][\w.-]+)?\b", result.stdout)
    return match.group(0) if result.returncode == 0 and match else "unknown"


def default_config(*, driver_bin: str | None = None, csc: str | None = None) -> DesktopEvalConfig:
    driver = driver_bin or default_driver()
    if not driver:
        raise RuntimeError("no cua-driver binary; pass --driver / S1A_DESKTOP_DRIVER, or install cua-driver")
    compiler = csc or default_csc()
    if not compiler:
        raise RuntimeError("no C# compiler; pass --csc / S1A_DESKTOP_CSC (the installed system .NET csc.exe)")
    return DesktopEvalConfig(driver_bin=driver, csc=compiler, driver_version=detect_driver_version(driver))


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex()


def default_trials(repeat: int = 1, tasks: tuple[str, ...] = TASKS, arms: tuple[str, ...] = ARMS) -> list[TrialPlan]:
    return [TrialPlan(task, arm, index) for task in tasks for index in range(repeat) for arm in arms]


def to_windows_path(path: Path) -> str:
    """A path the Windows fixture/compiler can write to; on native Windows the path is already native."""
    if os.name == "nt":
        return str(path)
    try:
        result = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
        return result or str(path)
    except (OSError, subprocess.CalledProcessError):
        return str(path)


def fixture_mode(task: str) -> str:
    return "permanent" if task == "permanent" else "normal"


def scripted_rule(task: str, arm: str) -> RuleModel:
    """The controlled fault: a hand-written rule per task; it reads the plan the rail leaves in the observation."""
    state = {"unlocked": False, "go": False}

    def pick(offered: dict[str, str], preferences: list[str]) -> str:
        for preference in preferences:
            if preference in offered:
                return preference
        return next(iter(offered))

    def rule(observation: dict[str, Any], offered: dict[str, str]) -> str:
        if task == "normal":
            if not state["unlocked"] and "click:unlock" in offered:
                state["unlocked"] = True
                return "click:unlock"
            if "click:finish" in offered:
                return "click:finish"
            return pick(offered, ["click:unlock", "click:finish", "click:noop"])
        if task == "recoverable":
            if observation.get("plan"):
                state["go"] = True
            if not state["go"]:
                return pick(offered, ["click:noop", "click:unlock"])
            if not state["unlocked"] and "click:unlock" in offered:
                state["unlocked"] = True
                return "click:unlock"
            if "click:finish" in offered:
                return "click:finish"
            return pick(offered, ["click:noop"])
        return pick(offered, ["click:noop", "click:unlock", "click:finish"])

    return RuleModel(f"scripted-{arm}", rule)


def build_fixture(source: Path, out_exe: Path, *, csc: str, log: Path) -> Path:
    """Compile the committed fixture source with the system .NET compiler; no installer, no new package."""
    out_exe.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [csc, "/nologo", "/target:winexe", f"/out:{to_windows_path(out_exe)}", to_windows_path(source)]
    with log.open("wb") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not out_exe.exists():
        raise RuntimeError(f"csc failed (exit {result.returncode}); see {log}")
    out_exe.chmod(out_exe.stat().st_mode | 0o111)  # csc writes from Windows: make it runnable through WSL interop
    return out_exe


def launch_fixture(
    exe: Path, mode: str, run_id: str, outdir: Path, log: Path
) -> tuple[subprocess.Popen, dict[str, Any]]:
    """Start one fixture process and wait for its startup oracle; the caller cleans up only this pid."""
    outdir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    oracle_file = outdir / "result.json"
    if oracle_file.exists():
        oracle_file.unlink()
    with log.open("wb") as handle:
        proc = subprocess.Popen(
            [str(exe), "--mode", mode, "--id", run_id, "--outdir", to_windows_path(outdir)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.monotonic() + ORACLE_WAIT_S
    while time.monotonic() < deadline:
        if oracle_file.exists():
            try:
                return proc, json.loads(oracle_file.read_text(encoding="utf-8-sig"))
            except ValueError:
                pass  # the fixture is still writing it; keep waiting until the deadline
        if proc.poll() is not None:
            raise RuntimeError(f"fixture {mode!r} exited (returncode={proc.poll()}) without writing {oracle_file}")
        time.sleep(0.25)
    if proc.poll() is None:  # alive but never wrote: kill it rather than leak a fixture
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    raise RuntimeError(f"fixture {mode!r} never wrote {oracle_file}")


async def wait_for_window(driver: CuaDriver, app_name: str, pid: int) -> Any:
    """Wait for this trial's fixture window to be on screen; the fixture writes its oracle before the window appears."""
    deadline = time.monotonic() + WINDOW_WAIT_S
    last_error = "no window seen"
    while time.monotonic() < deadline:
        try:
            window = await asyncio.wait_for(
                driver.find_window(app_name), timeout=max(0.01, deadline - time.monotonic())
            )
        except DriverError as exc:
            last_error = str(exc)
        else:
            if window.pid == pid:
                return window
            last_error = f"found window pid {window.pid} but launched fixture pid {pid}"
        await asyncio.sleep(0.25)
    raise RuntimeError(f"fixture window for {app_name!r} not ready: {last_error}")


def kill_pid(pid: int) -> None:
    """Terminate exactly this run's fixture pid; never a generic image name, never another process."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif Path(TASKKILL).exists():  # WSL: the fixture pid is a Windows pid
        subprocess.run([TASKKILL, "/PID", str(pid), "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["kill", "-9", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_oracle(outdir: Path) -> dict[str, Any]:
    try:
        return json.loads((outdir / "result.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def read_events(outdir: Path) -> list[dict[str, Any]]:
    path = outdir / "app_events.jsonl"
    events: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    return events


def _terminal(episode: Episode) -> dict[str, Any]:
    """The runtime's structured terminal, consumed directly; never regex-parsed from the truncated output.

    The runtime records the whole stop summary in ``extra['terminal']`` so a long reason or next action survives the
    terminal output's short budget. An episode that never reached a stop message carries ``None`` there, which is an
    empty summary here, not a fabricated one.
    """
    terminal = episode.extra.get("terminal")
    return terminal if isinstance(terminal, dict) else {}


def count_wasted_actions(views: list[dict[str, Any]], steps: int) -> int:
    """Accepted acts whose window progress digest did not change: the desktop no-op/wasted action count."""
    wasted = 0
    for index in range(min(steps, max(0, len(views) - 1))):
        before = progress_digest(views[index].get("state"))
        after = progress_digest(views[index + 1].get("state"))
        wasted += before == after
    return wasted


def _recovery_fields(episode: Episode, *, bounded: bool) -> dict[str, Any]:
    """Read recovery accounting from structured terminal fields and recorded events."""
    events = [event for event in (episode.extra.get("rethinks") or []) if event.get("kind") == "stall"]
    attempts = [int(event.get("attempt") or 0) for event in events]
    spent = [float(event.get("spent_s") or 0.0) for event in events]
    terminal = _terminal(episode)
    status = terminal.get("status")
    reason = terminal.get("reason") or episode.error
    negative = {"give_up", "error", "timeout", "cancelled"}
    failed = bounded and (
        episode.error is not None
        or status == "BLOCKED"
        or episode.extra.get("result_type") == "timeout"
        or any(event.get("termination") in negative for event in events)
    )
    source: str | None = None
    # The next action is retained from a recovery event or the runtime's terminal; the evaluator never invents one.
    action = next((event["next_action"] for event in reversed(events) if event.get("next_action")), None)
    if action is not None:
        source = "event"
    else:
        terminal_action = terminal.get("next_action")
        if terminal_action:
            action, source = terminal_action, "terminal"
    return {
        "recovery_attempts": max(attempts or [0]),
        "recovery_spent_s": round(max(spent or [0.0]), 3),
        "recovery_failed": failed,
        "recovery_termination": (
            "task_timeout"
            if episode.extra.get("result_type") == "timeout"
            else "act_budget"
            if failed and reason == "act budget spent"
            else events[-1].get("termination")
            if events
            else None
        ),
        "recovery_next_action": action,
        "next_action_source": source,
    }


def _episode_to_dict(episode: Episode) -> dict[str, Any]:
    data = asdict(episode)
    data["frames_dir"] = None if episode.frames_dir is None else str(episode.frames_dir)
    return data


async def run_trial(
    driver: CuaDriver,
    exe: Path,
    plan: TrialPlan,
    *,
    config: DesktopEvalConfig,
    run_dir: Path,
    run_id: str,
) -> tuple[Episode, DesktopTrialRecord]:
    """One real desktop trial over its own fresh fixture; only this trial's pid is ever touched."""
    name = f"{plan.task}-rethink_{plan.arm}-r{plan.repeat}"
    outdir = run_dir / "trials" / name
    outdir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "logs" / f"trial-{name}.out"
    expect = f"S1AFixture-{run_id} {DONE_TEXT}"
    app_name = exe.name
    planner = PlanChat()
    rule = scripted_rule(plan.task, plan.arm)
    started = time.perf_counter()
    proc: subprocess.Popen | None = None
    pid: int | None = None
    try:
        proc, oracle_start = launch_fixture(exe, fixture_mode(plan.task), run_id, outdir, log)
        pid = int(oracle_start["pid"])
        await wait_for_window(driver, app_name, pid)  # only this trial's verified pid is ever used
        env = WindowEnv(
            driver,
            app_name=app_name,
            goal=GOAL,
            done_when=lambda snapshot: desktop.shows(snapshot, expect),
            execute=True,
            clear_labels=(),
        )
        episode = await loop.run_episode(
            desktop.SPEC,
            env,
            model_name="rule",
            seed=plan.repeat,
            chat=planner,
            decision_model=rule,
            rethink_on=(plan.arm == "on"),
            max_acts=config.max_acts,
            timeout_s=config.timeout_s,
            prices=None,
            log=False,
            limits=(
                RecoveryLimits(max_attempts=config.max_recovery_attempts, timeout_s=config.recovery_timeout_s)
                if plan.arm == "on"
                else None
            ),
        )
        elapsed_s = round(time.perf_counter() - started, 3)
        # The models here are entirely scripted: no paid call or token usage occurred.
        episode.cost_usd = 0.0
        episode.usage_known = True
        final_oracle = read_oracle(outdir)
        events = read_events(outdir)
        event_counts: dict[str, int] = {}
        for event in events:
            event_counts[event.get("event", "?")] = event_counts.get(event.get("event", "?"), 0) + 1
        terminal_summary = _terminal(episode)
        if episode.extra.get("result_type") == "timeout":
            terminal = "timeout"
        else:
            terminal = str(terminal_summary.get("status") or "unknown")
        record = DesktopTrialRecord(
            task=plan.task,
            arm=plan.arm,
            repeat=plan.repeat,
            seed=plan.repeat,
            verified=final_oracle.get("finished") is True,
            terminal=terminal,
            errored=episode.error is not None or episode.extra.get("result_type") == "timeout",
            scripted_model=SCRIPTED_MODEL,
            driver={"bin": config.driver_bin, "version": config.driver_version, "channel": DRIVER_CHANNEL},
            platform=platform.platform(),
            wasted_actions=count_wasted_actions(episode.views, episode.steps),
            noop_clicks=event_counts.get("noop", 0),
            steps=episode.steps,
            decision_calls=len(episode.decisions),
            planner_calls=planner.calls,  # the planner is a chat model, so its calls are a subset of chat_calls
            chat_calls=episode.chat_calls,
            model_calls=len(episode.decisions) + episode.chat_calls,
            elapsed_s=elapsed_s,
            cost_usd=0.0,
            oracle={
                "start": oracle_start,
                "final": final_oracle,
                "events": event_counts,
                "distinct_pids": sorted({event.get("pid") for event in events if event.get("pid")}),
            },
            **_recovery_fields(episode, bounded=(plan.arm == "on")),
        )
        (outdir / "trial.json").write_text(
            json.dumps(
                {"record": record.as_json(), "episode": _episode_to_dict(episode)}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        return episode, record
    finally:
        if pid is not None:
            kill_pid(pid)
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass


def _failed_episode(
    plan: TrialPlan, config: DesktopEvalConfig, reason: str, elapsed_s: float
) -> tuple[Episode, DesktopTrialRecord]:
    """A trial that raised: recorded in the planned denominator, never dropped and never counted as success."""
    stamp = now_iso()
    record = DesktopTrialRecord(
        task=plan.task,
        arm=plan.arm,
        repeat=plan.repeat,
        seed=plan.repeat,
        verified=False,
        terminal="timeout" if "timeout" in reason.lower() else f"error: {reason}",
        errored=True,
        scripted_model=SCRIPTED_MODEL,
        driver={"bin": config.driver_bin, "version": config.driver_version, "channel": DRIVER_CHANNEL},
        platform=platform.platform(),
        recovery_attempts=0,
        recovery_spent_s=0.0,
        recovery_failed=False,
        recovery_termination=None,
        recovery_next_action=None,
        next_action_source=None,
        wasted_actions=0,
        noop_clicks=0,
        steps=0,
        model_calls=0,
        decision_calls=0,
        planner_calls=0,
        chat_calls=0,
        elapsed_s=elapsed_s,
        cost_usd=0.0,
        oracle={"verified": False, "failed": reason},
    )
    episode = Episode(
        env="desktop_recovery",
        policy=f"scripted-{plan.arm}",
        seed=plan.repeat,
        score=0.0,
        steps=0,
        elapsed_s=elapsed_s,
        started_at=stamp,
        finished_at=stamp,
        final_state={"task": plan.task, "arm": plan.arm, "verified": False, "failed": reason},
        chat_calls=0,
        chat_input_tokens=0,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        jev_input_tokens=0,
        invalid_keys=0,
        cost_usd=0.0,
        error=reason,
        extra=record.as_json(),
    )
    return episode, record


def _validate(trials: list[TrialPlan]) -> None:
    if not trials:
        raise ValueError("at least one trial is required")
    seen: set[tuple[str, str, int]] = set()
    for plan in trials:
        if plan.task not in TASKS:
            raise ValueError(f"unknown task {plan.task!r}; expected any of {TASKS}")
        if plan.arm not in ARMS:
            raise ValueError(f"unknown arm {plan.arm!r}; expected any of {ARMS}")
        if plan.repeat < 0:
            raise ValueError(f"repeat must be >= 0, got {plan.repeat}")
        key = (plan.task, plan.arm, plan.repeat)
        if key in seen:
            raise ValueError(f"duplicate trial {key}")
        seen.add(key)


async def run_eval(
    *,
    trials: list[TrialPlan] | None = None,
    config: DesktopEvalConfig | None = None,
    repeat: int = 1,
    run_id: str | None = None,
    results_dir: Path = RESULTS_DIR,
) -> DesktopEvalRun:
    """Run every planned trial, write the job folders and the paired summary. Every run starts fresh."""
    trials = default_trials(repeat) if trials is None else list(trials)
    _validate(trials)
    config = config or default_config()
    if not config.fixture_source.exists():
        raise RuntimeError(f"fixture source not found: {config.fixture_source}")
    run_id = run_id or new_run_id()
    run_dir = results_dir / "desktop_recovery" / run_id
    run_dir.mkdir(parents=True)
    previous_workspace = loop.WORKSPACE
    loop.WORKSPACE = run_dir / "agent-workspaces"
    exe = run_dir / "build" / f"S1AFixture-{run_id}.exe"
    pairs: list[tuple[Episode, DesktopTrialRecord]] = []
    try:
        async with started_runner():
            driver = CuaDriver(config.driver_bin, session_label=config.session_label)
            async with opened(driver):
                if not exe.exists():
                    build_fixture(config.fixture_source, exe, csc=config.csc, log=run_dir / "logs" / "csc.log")
                for plan in trials:
                    name = f"{plan.task}-rethink_{plan.arm}-r{plan.repeat}"
                    started = time.perf_counter()
                    try:
                        pair = await run_trial(driver, exe, plan, config=config, run_dir=run_dir, run_id=run_id)
                    except Exception as exc:  # noqa: BLE001 - a raised trial is a planned failure, not a drop
                        pair = _failed_episode(
                            plan, config, f"{type(exc).__name__}: {exc}", time.perf_counter() - started
                        )
                    pairs.append(pair)
                    print(
                        f"  [done] {name}: verified={pair[1].verified} terminal={pair[1].terminal} "
                        f"steps={pair[1].steps} planner_calls={pair[1].planner_calls} "
                        f"next_action={pair[1].next_action_source}"
                    )
    finally:
        loop.WORKSPACE = previous_workspace
    records = [record for _, record in pairs]
    job_dirs: list[Path] = []
    for arm in ARMS:
        episodes = [episode for episode, record in pairs if record.arm == arm]
        if episodes:
            job_dirs.append(write_job("desktop_recovery", episodes, results_dir=results_dir))
    summary = paired_summary(records)
    (run_dir / "summary.json").write_text(
        json.dumps(_json_summary(summary, records, run_id, config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return DesktopEvalRun(summary=summary, records=records, run_dir=run_dir, job_dirs=job_dirs)


def run_sync(**kwargs: Any) -> DesktopEvalRun:
    return asyncio.run(run_eval(**kwargs))


def _arm_block(records: list[DesktopTrialRecord]) -> dict[str, Any]:
    return {
        **coverage(records),
        "recovery_attempts": sum(record.recovery_attempts for record in records),
        "recovery_spent_s": round(sum(record.recovery_spent_s for record in records), 3),
        "recovery_failed": sum(record.recovery_failed for record in records),
        "wasted_actions": sum(record.wasted_actions for record in records),
        "noop_clicks": sum(record.noop_clicks for record in records),
        "mean_model_calls": mean((r.model_calls for r in records), 2),
        "mean_elapsed_s": mean((r.elapsed_s for r in records), 3),
        "cost_usd": 0.0,
    }


def paired_summary(records: list[DesktopTrialRecord]) -> dict[str, Any]:
    """Every planned trial counts, errors included. Success is each trial's own oracle, never the model verdict."""
    arms = {arm: _arm_block([r for r in records if r.arm == arm]) for arm in ARMS}
    tasks = sorted({record.task for record in records})
    by_task = {
        task: {arm: _arm_block([r for r in records if r.arm == arm and r.task == task]) for arm in ARMS}
        for task in tasks
    }
    paired, complete = pair_outcomes(records)
    paired = {
        **paired,
        "mean_delta_elapsed_s": mean((round(on.elapsed_s - off.elapsed_s, 3) for off, on in complete), 3),
        "mean_delta_wasted_actions": mean((on.wasted_actions - off.wasted_actions for off, on in complete), 2),
    }
    return {
        "planned_trials": len(records),
        "arms": arms,
        "by_task": by_task,
        "paired": paired,
        "on_minus_off_completion_rate": on_minus_off_completion_rate(arms),
        "note": (
            "controlled fault injection with a scripted decision model and a scripted planner on a self-built "
            "fixture: it exercises the recovery mechanism and budget, not a trained model and not a real app success "
            "rate. Success is the fixture's own result.json, never the model's DONE. Errors count in every "
            "denominator; cost is 0 by construction because no paid API is called."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Desktop bounded recovery on/off - scripted fixture subset",
        "",
        f"Planned trials: {summary['planned_trials']} (errors included).",
        "",
    ]
    lines += summary_tables(
        summary,
        [
            ("errored", "errored"),
            ("recovery attempts", "recovery_attempts"),
            ("recovery failed", "recovery_failed"),
            ("wasted actions", "wasted_actions"),
            ("mean elapsed s", "mean_elapsed_s"),
        ],
    )
    paired = summary["paired"]
    lines += [
        "",
        paired_completion_line(paired),
        f"On minus off completion rate: {summary['on_minus_off_completion_rate']}.",
        "",
        f"Note: {summary['note']}",
        "",
    ]
    return "\n".join(lines)


def _json_summary(
    summary: dict[str, Any], records: list[DesktopTrialRecord], run_id: str, config: DesktopEvalConfig
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "summary": summary,
        "trials": [record.as_json() for record in records],
        "definition": {
            "planned_trials": "every (task, arm, repeat) the run planned; errors included",
            "verified": "this trial's own fixture result.json has finished true (never the model's DONE)",
            "oracle": "per-trial fixture output directory; a fresh fixture is launched for every trial",
            "wasted_actions": "accepted acts whose window progress digest did not change (the desktop no-op count)",
            "noop_clicks": "the fixture's own logged clicks on its no-op control",
            "elapsed_s": "trial wall clock including fixture startup and the agent episode; excludes shared driver and compiler startup",
            "model_calls": "decision_calls + chat_calls (all scripted doubles); planner_calls is a subset of chat_calls",
            "next_action": "the actionable escalation the runtime terminal carried for a failed bounded recovery",
            "next_action_source": "where a recorded next_action came from: a recovery event or the terminal; never generated by the evaluator",
            "cost_usd": "0 by construction: scripted decisions and planner, no paid API call",
        },
        "budget": {
            "max_acts": config.max_acts,
            "timeout_s": config.timeout_s,
            "max_recovery_attempts": config.max_recovery_attempts,
            "recovery_timeout_s": config.recovery_timeout_s,
        },
        "driver": {"bin": config.driver_bin, "version": config.driver_version, "channel": DRIVER_CHANNEL},
        "scripted_model": SCRIPTED_MODEL,
    }


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="Native desktop bounded-recovery on/off subset eval.")
    argument_parser.add_argument("--driver", default=None, help="path to cua-driver (or S1A_DESKTOP_DRIVER)")
    argument_parser.add_argument("--csc", default=None, help="path to the system .NET C# compiler (or S1A_DESKTOP_CSC)")
    argument_parser.add_argument("--repeat", type=int, default=1, help="repeats per (task, arm); >= 1")
    argument_parser.add_argument("--tasks", default=",".join(TASKS), help=f"comma-separated subset of {TASKS}")
    argument_parser.add_argument("--arms", default=",".join(ARMS), help=f"comma-separated subset of {ARMS}")
    argument_parser.add_argument("--run-id", default=None, help="name this run explicitly (default: a fresh unique id)")
    argument_parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="where the run directory is created")
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    tasks = tuple(part.strip() for part in args.tasks.split(",") if part.strip())
    arms = tuple(part.strip() for part in args.arms.split(",") if part.strip())
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    config = default_config(driver_bin=args.driver, csc=args.csc)
    run = run_sync(
        trials=default_trials(args.repeat, tasks, arms),
        config=config,
        run_id=args.run_id,
        results_dir=Path(args.results_dir),
    )
    print(render_markdown(run.summary))
    print(f"run dir: {run.run_dir}")
    for job_dir in run.job_dirs:
        print(f"job: {job_dir}")
    return 1 if any(record.errored for record in run.records) else 0


if __name__ == "__main__":
    sys.exit(main())
