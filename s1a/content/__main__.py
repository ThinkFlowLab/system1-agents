# coding: utf-8
"""``python -m s1a.content``: the five-stage desk run, and the later latency-gate command.

Stdout is one JSON object on success. A usage or gate error is one line on stderr and exit 2.
An Omni or feed failure is one line on stderr and exit 1. A completed run never publishes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from s1a.content.ingest import load_inputs
from s1a.content.latency import gate, omni_call_latencies
from s1a.content.omni_client import decider_from_env, require_worker, worker_record
from s1a.content.rules import RuleDecider
from s1a.content.workflow import GateRequired, run_workflow

ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    build = argparse.ArgumentParser(prog="python -m s1a.content", description=__doc__)
    commands = build.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="ingest, screen, pick, fact-check, and record a publish decision")
    run.add_argument("--items", type=Path, help="labelled or unlabelled news items in JSONL")
    run.add_argument("--rss", type=Path, action="append", default=[], help="a local RSS or Atom file; repeatable")
    run.add_argument(
        "--rss-url", action="append", default=[], help="an http(s) feed URL; repeatable; not used by the fixture"
    )
    run.add_argument("--x-fixture", type=Path, action="append", default=[], help="X posts in JSONL; repeatable")
    run.add_argument("--picks", type=Path, help='JSON object {"ids": [...]} of 3 to 5 kept items')
    run.add_argument("--publish", type=Path, help='JSON object {"decisions": {id: publish|hold}}')
    run.add_argument("--backend", choices=("auto", "rule", "omni"), default="auto")
    run.add_argument(
        "--fact-check-worker",
        choices=("startlux", "pplx"),
        default="startlux",
        help="draft worker for fact-check when the backend is omni; pplx needs OMNI_FACTCHECK_BASE_URL",
    )
    run.add_argument(
        "--allow-unsafe-skip-gates",
        action="store_true",
        help="UNSAFE. Invent a pick and hold every item so a run can finish without a person. Does not publish.",
    )
    run.add_argument("--packet", type=Path, help="write the pre-gate packet here when a human gate stops the run")
    latency = commands.add_parser(
        "latency-report",
        help="summarize Omni request latencies already in a report; optional ceilings fail closed",
    )
    latency.add_argument("--report", type=Path, required=True)
    latency.add_argument("--p50-max-ms", type=float)
    latency.add_argument("--p95-max-ms", type=float)
    latency.add_argument("--min-samples", type=int, default=1)
    return build


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env", override=False)
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            report = _run(args)
        else:
            report = _latency(args)
    except GateRequired as exc:
        if args.command == "run" and args.packet is not None and exc.packet is not None:
            _write_json(args.packet, exc.packet)
        print(exc, file=sys.stderr)
        return 2
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    except (RuntimeError, OSError, httpx.HTTPError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False))
    if args.command == "latency-report" and report.get("enforced") and not report.get("passed"):
        return 1
    return 0


def _run(args: argparse.Namespace) -> dict[str, Any]:
    http = httpx.Client(timeout=15.0, follow_redirects=False, trust_env=False) if args.rss_url else None
    try:
        items = load_inputs(
            items=args.items,
            rss_files=args.rss,
            rss_urls=args.rss_url,
            x_files=args.x_fixture,
            http=http,
        )
    finally:
        if http is not None:
            http.close()
    backend = _resolve_backend(args.backend, os.environ)
    picks = _load_picks(args.picks) if args.picks else None
    publish = _load_publish(args.publish) if args.publish else None
    interactive = sys.stdin.isatty() and not args.allow_unsafe_skip_gates
    pick_hook = _prompt_picks if picks is None and interactive else None
    publish_hook = _prompt_publish if publish is None and interactive else None
    if backend == "rule":
        return run_workflow(
            items,
            decider=RuleDecider(),
            picks=picks,
            publish=publish,
            pick_hook=pick_hook,
            publish_hook=publish_hook,
            allow_unsafe_skip_gates=args.allow_unsafe_skip_gates,
            workers={"screen": None, "fact_check": None},
        )
    decider = decider_from_env(os.environ, fact_check_worker=args.fact_check_worker)
    workers = {
        "screen": worker_record(
            decider.screen_client.profile, decider.screen_client.base_url, decider.screen_client.model
        ),
        "fact_check": worker_record(
            decider.fact_client.profile, decider.fact_client.base_url, decider.fact_client.model
        ),
    }
    try:
        return run_workflow(
            items,
            decider=decider,
            picks=picks,
            publish=publish,
            pick_hook=pick_hook,
            publish_hook=publish_hook,
            allow_unsafe_skip_gates=args.allow_unsafe_skip_gates,
            workers=workers,
        )
    finally:
        decider.close()


def _resolve_backend(name: str, env: Any) -> str:
    if name == "auto":
        return "omni" if (env.get("OMNI_BASE_URL") or "").strip() else "rule"
    if name == "omni" and not (env.get("OMNI_BASE_URL") or "").strip():
        raise ValueError("backend omni needs OMNI_BASE_URL")
    if name == "omni":
        require_worker("startlux", fact_check=False)
    return name


def _load_picks(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("ids"), list):
        raise ValueError("--picks must be a JSON object with an ids array")
    return data["ids"]


def _load_publish(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), dict):
        raise ValueError("--publish must be a JSON object with a decisions object")
    return data["decisions"]


def _prompt_picks(kept: list[dict[str, Any]]) -> list[str]:
    print("Human pick gate. Choose 3 to 5 ids. This gate does not publish.", file=sys.stderr)
    for item in kept:
        print(f"{item['id']}\t{item['title']}", file=sys.stderr)
    try:
        raw = input("ids: ")
    except EOFError as exc:
        raise ValueError("human pick gate closed without a decision") from exc
    return [part.strip() for part in raw.split(",") if part.strip()]


def _prompt_publish(checked: list[dict[str, Any]]) -> dict[str, str]:
    print("Human publish gate. Type publish or hold for each id. Nothing is posted.", file=sys.stderr)
    decisions = {}
    for row in checked:
        print(f"{row['id']} fact-check={row['choice']}", file=sys.stderr)
        try:
            raw = input("publish or hold: ").strip()
        except EOFError as exc:
            raise ValueError("human publish gate closed without a decision") from exc
        decisions[row["id"]] = raw
    return decisions


def _latency(args: argparse.Namespace) -> dict[str, Any]:
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"--report: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("--report must be a JSON object")
    return gate(
        omni_call_latencies(report),
        p50_max_ms=args.p50_max_ms,
        p95_max_ms=args.p95_max_ms,
        min_samples=args.min_samples,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
