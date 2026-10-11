# coding: utf-8
"""``python -m evals.recovery [--repeat 1] [--arms off on] [--tasks normal recoverable permanently_blocked]``.

Runs the repeatable bounded-recovery on/off subset against the local fixture pages in a real headless Chromium,
writes the Harbor job folders and a paired JSON + Markdown summary, and prints the summary. Defaults to one repeat;
raise ``--repeat`` for more pairs, which are then averaged. Needs Node/``npx`` and the Playwright MCP package, the
same dependency ``tests/system`` uses. The decision model and planner are scripted doubles, so this is a controlled
mechanism test, not a real model benchmark. Exits non-zero when any planned trial raised (a harness error or timeout);
expected terminal scores such as BLOCKED do not make the process fail.
"""

from __future__ import annotations

import s1a.entry  # noqa: F401  # routes the harness logs to files before anything imports openjiuwen
import argparse
from pathlib import Path

from s1a.jobs import RESULTS_DIR

from evals.recovery.fixture import DEFAULT_TASKS
from evals.recovery.runner import EvalConfig, format_run, run_sync

TASKS = {task.name: task for task in DEFAULT_TASKS}


def parser() -> argparse.ArgumentParser:
    build = argparse.ArgumentParser(prog="python -m evals.recovery", description=__doc__)
    build.add_argument("--repeat", type=int, default=1, help="repeats per (task, arm), paired; must be >= 1")
    build.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS), help="which fixture tasks")
    build.add_argument("--arms", nargs="+", choices=("off", "on"), default=["off", "on"], help="recovery arms")
    build.add_argument("--headed", action="store_true", help="show the browser (default: headless)")
    build.add_argument("--timeout", type=float, default=120.0, help="seconds per trial")
    build.add_argument("--max-steps", type=int, default=24, help="the subagent's iteration cap")
    build.add_argument("--stall-after", type=int, default=3, help="no-op actions before a stall")
    build.add_argument("--recovery-attempts", type=int, default=3, help="bounded recovery attempts per trial")
    build.add_argument("--recovery-timeout", type=float, default=15.0, help="bounded recovery active seconds per trial")
    build.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="root of the Harbor job folders")
    return build


def main() -> int:
    args = parser().parse_args()
    config = EvalConfig(
        timeout_s=args.timeout,
        max_steps=args.max_steps,
        stall_after=args.stall_after,
        max_recovery_attempts=args.recovery_attempts,
        recovery_timeout_s=args.recovery_timeout,
        headless=not args.headed,
    )
    run = run_sync(
        repeat=args.repeat,
        tasks=tuple(TASKS[name] for name in args.tasks),
        arms=tuple(args.arms),
        config=config,
        results_dir=args.results_dir,
    )
    print(format_run(run))
    return 1 if any(record.errored for record in run.records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
