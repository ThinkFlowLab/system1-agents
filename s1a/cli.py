# coding: utf-8
"""``s1a``: list the agents, run one by name with its own flags, or ask a decision model one choice question."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import s1a.console  # first: routes the harness logs to files before openjiuwen loads and logs to the console
from openjiuwen.core.common.exception.errors import BaseError

from s1a import config, probe
from s1a.decision_models import build_model
from s1a import run as agents
from s1a.run import started_runner
from s1a.spec import Json

DECIDE_MODEL_NAMES = (
    "jev",
    "clm",
    "laya",
    "laya-served",
    "cua",
)  # the decision models that answer one question on their own: no env, no rule, no chance


def parser() -> argparse.ArgumentParser:
    build = argparse.ArgumentParser(
        prog="s1a",
        description="S1A agents: System 1 decision models (TypeSafe Jev, Laya, Cua-S1) in the model slot of openJiuwen agents.",
    )
    commands = build.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="the registered agents, one name per line")
    run = commands.add_parser("run", help="run one agent; its own flags follow the name (--help lists them)")
    run.add_argument("agent", help=f"one of: {', '.join(agents.names())}")
    run.add_argument("flags", nargs=argparse.REMAINDER)
    decide = commands.add_parser(
        "decide",
        help="one choice question to a decision model: jev, clm or laya-served over HTTP, or laya and cua in process",
    )
    decide.add_argument("--state", required=True, help="a JSON object, or @path to a file holding one")
    decide.add_argument(
        "--option", action="append", required=True, metavar="KEY=DESCRIPTION", help="one candidate; repeat per option"
    )
    decide.add_argument("--rules", required=True, help="the facts the model applies when it picks")
    decide.add_argument(
        "--model",
        choices=DECIDE_MODEL_NAMES,
        default="jev",
        help="who answers: jev, clm or laya-served over HTTP, laya and cua in process",
    )
    fit = commands.add_parser("probe", help="the fit probe: hand-written choice cases from a JSONL file")
    fit.add_argument(
        "cases",
        type=Path,
        help="JSONL, one case per line: state (object), options (key to text), rules, accept (list of right keys), note",
    )
    fit.add_argument(
        "--model",
        choices=DECIDE_MODEL_NAMES,
        default="jev",
        help="who answers: jev, clm or laya-served over HTTP, laya and cua in process",
    )
    return build


def parse_state(text: str) -> dict[str, Any]:
    try:
        raw = Path(text[1:]).read_text(encoding="utf-8") if text.startswith("@") else text
        state = json.loads(raw)
    except (ValueError, OSError) as exc:
        raise ValueError(f"--state: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError("--state must be a JSON object")
    return state


def parse_options(items: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for item in items:
        key, sep, description = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--option needs KEY=DESCRIPTION, got {item!r}")
        options[key] = description
    return options


async def run_agent(name: str, flags: list[str]) -> Json:
    """One run with the Runner held for the process; the result is the one JSON line on stdout."""
    async with started_runner():
        result = await agents.run_by_name(name, flags)
    print(json.dumps(result, ensure_ascii=False))
    return result


async def decide(args: argparse.Namespace) -> dict[str, Any]:
    """One question through the model ``--model`` names; prints ``{"choice", "probabilities", "confidence", "ms"}``."""
    state, options = parse_state(args.state), parse_options(args.option)  # bad input is reported before any key check
    decision_model = build_model(args.model)
    try:
        answer = await probe.pick(decision_model, state=state, options=options, rules=args.rules)
    finally:
        await decision_model.close()
    print(json.dumps(answer, ensure_ascii=False))
    return answer


async def run_probe(cases: Path, model_name: str) -> dict[str, Any]:
    read = probe.read_cases(cases)
    decision_model = build_model(model_name)
    try:
        summary = await probe.run(read, decision_model)
    finally:
        await decision_model.close()
    print(probe.render(summary))
    return summary


def main(argv: list[str]) -> int:
    """Exit 0 when the command ran (a run that returns ok:false included), 1 for a run, key, model or file error or a
    probe verdict other than fits, 2 for a usage error: an unknown agent, bad flags, a malformed --state, --option,
    @file or cases file. A flag argparse rejects prints its usage block; every other non-zero exit is one line on
    stderr."""
    config.load_env()
    args = parser().parse_args(argv)
    try:
        match args.command:
            case "list":
                print("\n".join(agents.names()))
            case "run":
                asyncio.run(run_agent(args.agent, args.flags))
            case "decide":
                asyncio.run(decide(args))
            case "probe":
                return 0 if asyncio.run(run_probe(args.cases, args.model))["verdict"] == "fits" else 1
    except (agents.UnknownAgent, ValueError) as exc:  # JSONDecodeError is a ValueError
        print(exc, file=sys.stderr)
        return 2
    except (RuntimeError, OSError, BaseError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def entry() -> None:
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    s1a.console.utf8_console()
    entry()
