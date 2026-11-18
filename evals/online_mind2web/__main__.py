# coding: utf-8
"""``python -m evals.online_mind2web {show,select,fetch,run,score}``: the seeded Online-Mind2Web live slice (#116).

- ``show`` prints the committed slice (ids, levels, websites and the pinned revision).
- ``select --official Online_Mind2Web.json`` re-derives the ids from the official file and the slice's seed, to check them.
- ``fetch --dest FILE`` downloads the official file at the pinned revision (network, gated; needs ``HF_TOKEN``).
  Nothing is vendored.
- ``run --official FILE --arms jev`` plays each task through ``browse`` with a decision model or the chat model under
  the same ``--max-steps`` and ``--timeout``, writes ``attempts.jsonl`` and ``results.md``. Needs the models' keys and
  a live network. Without a judge, finished episodes are recorded as ``unevaluated`` (not a pass). ``--judge-scores``
  applies WebJudge-style integer scores from a JSON object keyed by ``task_id`` (or ``task_id__arm``).
- ``score attempts.jsonl`` re-renders the table from recorded attempts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from evals.online_mind2web import score, tasks

RESULTS = Path("evals/results/online_mind2web")


def parser() -> argparse.ArgumentParser:
    build = argparse.ArgumentParser(
        prog="python -m evals.online_mind2web",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    build.add_argument("--slice", type=Path, default=tasks.SLICE_PATH)
    commands = build.add_subparsers(dest="command", required=True)
    commands.add_parser("show")
    sel = commands.add_parser("select")
    sel.add_argument("--official", type=Path, required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--dest", type=Path, default=Path(".external/online_mind2web/Online_Mind2Web.json"))
    run = commands.add_parser("run")
    run.add_argument("--official", type=Path, default=Path(".external/online_mind2web/Online_Mind2Web.json"))
    run.add_argument("--arms", nargs="+", default=["jev"])
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--max-steps", type=int, default=40)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--headed", action="store_true")
    run.add_argument("--judge-scores", type=Path, default=None, help="JSON object of WebJudge integer scores")
    run.add_argument("--judge-model", default=None, help="name recorded next to --judge-scores")
    run.add_argument("--threshold", type=int, default=score.JUDGE_THRESHOLD)
    run.add_argument("--out", type=Path, default=None)
    sc = commands.add_parser("score")
    sc.add_argument("attempts", type=Path)
    sc.add_argument("--agents-sha", default=None)
    sc.add_argument("--model", default=None)
    sc.add_argument("--budget", default=None)
    sc.add_argument("--date", default=None)
    return build


def agents_sha() -> str:
    """The checkout's HEAD, or N/A when this is not a git work tree."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return score.NA
    sha = result.stdout.strip()
    return sha if sha else score.NA


def load_score_judge(path: Path, model: str | None):
    """A judge that reads integer scores from a JSON object; used in tests and for applying an offline WebJudge run."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise tasks.SliceError(f"{path} must be a JSON object of task ids to integer scores")

    def judge(task, outcome, logs_dir):
        del outcome, logs_dir
        value = raw.get(f"{task.id}__{getattr(judge, 'arm', '')}", raw.get(task.id))
        if not isinstance(value, int) or isinstance(value, bool):
            raise tasks.SliceError(f"no integer WebJudge score for {task.id} in {path}")
        return value

    judge.model = model
    return judge


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        slice_ = tasks.load_slice(args.slice)
    except tasks.SliceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "show":
        print(f"{slice_.name}: {slice_.dataset} {slice_.source_file} @ {slice_.revision} (seed {slice_.seed})")
        for task in slice_.tasks:
            print(f"{task.id}\t{task.level}\t{task.reference_length}\t{task.website}")
        return 0
    if args.command == "select":
        try:
            official = tasks.load_official(args.official)
            tasks.check_selection(slice_, official)
        except tasks.SliceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for task_id in slice_.ids:
            print(task_id)
        return 0
    if args.command == "fetch":
        try:
            path = tasks.fetch_official(slice_, args.dest)
        except tasks.SliceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {path} ({slice_.expected_tasks} tasks @ {slice_.revision})")
        return 0
    if args.command == "score":
        from evals.online_mind2web.runner import load_attempts

        rows = load_attempts(args.attempts)
        sha = args.agents_sha or next((row.agents_sha for row in rows if row.agents_sha), score.NA)
        print(
            score.render(
                rows,
                revision=slice_.revision,
                dataset=slice_.dataset,
                agents_sha=sha,
                model=args.model or score.NA,
                budget=args.budget or score.NA,
                date=args.date or score.NA,
            ),
            end="",
        )
        return 0
    return asyncio.run(_run(args, slice_))


async def _run(args: argparse.Namespace, slice_: tasks.Slice) -> int:
    import s1a.entry  # noqa: F401  # routes the harness logs to files before anything imports openjiuwen
    from s1a.run import started_runner

    from evals.online_mind2web.live import LiveBudget, make_player
    from evals.online_mind2web.runner import run_slice

    try:
        official = tasks.bind(slice_, tasks.load_official(args.official))
    except (tasks.SliceError, FileNotFoundError) as exc:
        print(f"error: {exc} (run `python -m evals.online_mind2web fetch` first)", file=sys.stderr)
        return 2
    judge = load_score_judge(args.judge_scores, args.judge_model) if args.judge_scores else None
    out = args.out or RESULTS / f"{datetime.now():%Y-%m-%d__%H-%M-%S}"
    budget = LiveBudget(args.max_steps, args.timeout, not args.headed)
    sha = agents_sha()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    budget_label = f"max_steps={budget.max_steps} timeout_s={budget.timeout_s:g}"
    play = make_player(budget)
    async with started_runner():
        attempts = await run_slice(
            official, args.arms, play, out, repeat=args.repeat, judge=judge, agents_sha=sha, threshold=args.threshold
        )
    table = score.render(
        attempts,
        revision=slice_.revision,
        dataset=slice_.dataset,
        agents_sha=sha,
        model=",".join(args.arms),
        budget=budget_label,
        date=stamp,
        threshold=args.threshold,
    )
    (out / "results.md").write_text(table, encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(
            {
                "dataset": slice_.dataset,
                "revision": slice_.revision,
                "agents_sha": sha,
                "arms": args.arms,
                "budget": {"max_steps": budget.max_steps, "timeout_s": budget.timeout_s, "headless": budget.headless},
                "date": stamp,
                "threshold": args.threshold,
                "tasks": list(slice_.ids),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(table, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
