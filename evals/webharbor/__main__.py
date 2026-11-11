# coding: utf-8
"""``python -m evals.webharbor {show,select,fetch,run,score}``: the seeded WebHarbor slice (issue 118).

- ``show`` prints the committed slice (ids and revisions).
- ``select --registry registry.json`` re-derives the ids from the registry and the slice's seed, to check them.
- ``fetch --dest DIR`` downloads the slice's task folders at the pinned revision (network; nothing is vendored).
- ``run --tasks-dir DIR --arms jev llm`` resets the site for every attempt, plays it, runs the site verifier inside
  the container and writes ``attempts.jsonl`` and ``results.md``. Needs Docker, the per-site image and the models'
  keys; ``--control-url`` resets through the all-sites image's control plane (``WEBSYN_CONTROL_TOKEN``) instead.
- ``score attempts.jsonl`` re-renders the table from recorded attempts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from evals.webharbor import score, tasks

RESULTS = Path("evals/results/webharbor")


def parser() -> argparse.ArgumentParser:
    build = argparse.ArgumentParser(
        prog="python -m evals.webharbor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    build.add_argument("--slice", type=Path, default=tasks.SLICE_PATH)
    commands = build.add_subparsers(dest="command", required=True)
    commands.add_parser("show")
    sel = commands.add_parser("select")
    sel.add_argument("--registry", type=Path, required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--dest", type=Path, default=Path(".external/webharbor"))
    run = commands.add_parser("run")
    run.add_argument("--tasks-dir", type=Path, default=Path(".external/webharbor"))
    run.add_argument("--arms", nargs="+", default=["jev", "llm"])
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--max-steps", type=int, default=40)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--headed", action="store_true")
    run.add_argument("--control-url", default=None, help="reset through the control plane instead of a new container")
    run.add_argument("--out", type=Path, default=None)
    sc = commands.add_parser("score")
    sc.add_argument("attempts", type=Path)
    sc.add_argument("--image-digest", default=score.NA)
    return build


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        slice_ = tasks.load_slice(args.slice)
    except tasks.SliceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "show":
        print(f"{slice_.name}: {slice_.repo} {slice_.dataset_ref} @ {slice_.registry_revision}")
        for task in slice_.tasks:
            print(f"{task.id}\t{task.state_change}")
        return 0
    if args.command == "select":
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        tasks.check_against_registry(slice_, registry)
        picked = tasks.select(
            registry, slice_.dataset, slice_.dataset_version, slice_.site, len(slice_.tasks), slice_.seed
        )
        same = picked == [task.id for task in slice_.tasks]
        print("\n".join(picked))
        print("matches slice.json" if same else "DIFFERS from slice.json", file=sys.stderr)
        return 0 if same else 1
    if args.command == "fetch":
        root = tasks.download(slice_, args.dest)
        print(root)
        return 0
    if args.command == "score":
        from evals.webharbor.runner import load_attempts

        print(
            score.render(
                load_attempts(args.attempts),
                image_digest=args.image_digest,
                task_revision=slice_.registry_revision,
                dataset=slice_.dataset_ref,
            ),
            end="",
        )
        return 0
    return asyncio.run(_run(args, slice_))


async def _run(args: argparse.Namespace, slice_: tasks.Slice) -> int:
    import s1a.entry  # noqa: F401  # routes the harness logs to files before anything imports openjiuwen
    from s1a.run import started_runner

    from evals.webharbor.control import ControlPlane
    from evals.webharbor.environment import DockerSite
    from evals.webharbor.live import LiveBudget, make_player
    from evals.webharbor.runner import run_slice

    try:
        specs = [
            (tasks.load_task(slice_, args.tasks_dir / "tasks" / t.id), args.tasks_dir / "tasks" / t.id)
            for t in slice_.tasks
        ]
    except tasks.SliceError as exc:
        print(f"error: {exc} (run `python -m evals.webharbor fetch` first)", file=sys.stderr)
        return 2
    control = ControlPlane.from_env(args.control_url) if args.control_url else None
    environment = DockerSite(control=control)
    out = args.out or RESULTS / f"{datetime.now():%Y-%m-%d__%H-%M-%S}"
    play = make_player(LiveBudget(args.max_steps, args.timeout, not args.headed))
    async with started_runner():
        attempts = await run_slice(environment, specs, args.arms, play, out, repeat=args.repeat)
    table = score.render(
        attempts,
        image_digest=environment.image_digest(),
        task_revision=slice_.registry_revision,
        dataset=slice_.dataset_ref,
    )
    (out / "results.md").write_text(table, encoding="utf-8")
    print(table, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
