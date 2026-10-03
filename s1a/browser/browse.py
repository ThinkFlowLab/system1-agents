# coding: utf-8
"""The browser front: one task through openJiuwen's browser subagent with a decision model, or the chat model, in its model slot."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from openjiuwen.core.foundation.llm import Model
from openjiuwen.core.runner import Runner
from openjiuwen.harness.deep_agent import DeepAgent
from openjiuwen.harness.subagents import create_browser_agent
from openjiuwen.harness.tools.browser_move.playwright_runtime.config import BrowserInstanceConfig
from openjiuwen.harness.subagents.browser_agent import (
    _BROWSER_MODEL_TEMPERATURE_MARKER,
    DEFAULT_BROWSER_AGENT_TEMPERATURE,
)
from openjiuwen.harness.tools.browser_move.utils.parsing import extract_json_object

from s1a.browser.decision_model import URL_RE, BrowserDecisionModel, BrowserPolicy
from s1a.decision_models import DecisionModel, build_model
from s1a.config import HOME, browser_launch_args, chat_model_from_env, first_env
from s1a.counting_model import CountingModel
from s1a.pricing import chat_prices, cost_usd
from s1a.browser.profiler import BrowserProfiler, render
from s1a.spec import BrowserAgentSpec, positive_float, positive_int

Answer = dict[str, Any]
BROWSER_MODEL_NAMES = (
    "jev",
    "laya",
    "laya-served",
    "cua",
    "llm",
)  # a decision model (Jev or served Laya over HTTP, Laya or Cua-S1 in process) or the chat model
RUNS_DIR = HOME / "runs" / "browser"


def browser_result(final: str) -> dict[str, Any] | None:
    """The subagent's structured completion (``browser_result``) when the final text holds one; None otherwise."""
    result = extract_json_object(final).get("browser_result")
    return result if isinstance(result, dict) else None


def terminal_summary(final: str) -> dict[str, Any] | None:
    """The JSON the Jev policy returns on DONE or BLOCKED, bare or inside ``browser_result.summary``, whatever
    text the runtime's rail appends after it."""
    result = browser_result(final)
    text = str(result.get("summary") or "") if result is not None else final
    parsed = extract_json_object(text)
    return parsed if "status" in parsed else None


def finish_llm(answer: Answer) -> Answer:
    """The chat model's final message is ``browser_result.summary``; the subagent's status decides ``ok``."""
    result = browser_result(answer["final"])
    if result is None:
        return answer
    answer["browser_result"] = {
        key: value for key, value in result.items() if key != "summary"
    }  # the runtime's verdict
    answer["final"] = str(result.get("summary") or "")
    answer["ok"] = result.get("status") == "completed" and bool(answer["final"])
    if not answer["ok"]:
        answer["error"] = f"browser_result {result.get('status')}: {result.get('terminal_reason')}"
    return answer


def finish_decision_model(answer: Answer, *, model_name: str) -> Answer:
    """The policy's answer from the page it reached (on DONE, or on BLOCKED after progress) is the result; none fails.
    ``model_name`` names the decision model in the error."""
    summary = terminal_summary(answer["final"])
    if summary is None:
        return answer
    status = str(summary.get("status") or "")
    answer["status"] = status
    answer["terminal"] = summary
    answer["final"] = str(summary.get("answer") or "")
    answer["ok"] = bool(answer["final"])
    if answer["ok"]:
        answer["error"] = None  # run_task flagged the harness's own verdict; the policy's answer is the one that counts
    elif status == "DONE":
        answer["error"] = f"{model_name} DONE without an answer"
    else:
        answer["error"] = f"{model_name} {status}: {summary.get('reason') or 'no reason given'}"
    return answer


def usage_summary(calls: list[dict[str, Any]], *, jev_input_tokens: int, decisions: int) -> dict[str, Any]:
    """Counts and dollars for one task: decisions, the chat model's calls and tokens, Jev's input tokens, the sum in USD."""
    chat_in = sum(int(call["input_tokens"]) for call in calls)
    chat_out = sum(int(call["output_tokens"]) for call in calls)
    chat_cached = sum(int(call.get("cache_tokens") or 0) for call in calls)
    prices = chat_prices(first_env("MODEL_NAME")) if chat_in + chat_out else None
    return {
        "decisions": decisions,
        "chat_calls": len(calls),
        "chat_input_tokens": chat_in,
        "chat_output_tokens": chat_out,
        "chat_cache_tokens": chat_cached,
        "jev_input_tokens": jev_input_tokens,
        "cost_usd": cost_usd(jev_input_tokens, chat_in, chat_out, chat_cached, prices),
    }


async def run_task(agent: DeepAgent, goal: str, *, timeout_s: float) -> Answer:
    """One conversation through the started Runner; a timed-out task keeps the harness's partial output.

    The browser (the agent's task resources) and the Runner session are released however the task ends, so the
    next run in the same process starts its own browser with its own launch args and cookies.
    """
    answer: Answer = {"ok": False, "final": "", "screenshot": None, "error": None, "elapsed_ms": 0}
    conversation_id = f"s1a-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    await agent.ensure_initialized()
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            Runner.run_agent(agent, {"query": goal, "conversation_id": conversation_id}), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        answer["error"] = f"timeout after {timeout_s:.0f} s"
        return answer
    finally:
        answer["elapsed_ms"] = round((time.perf_counter() - started) * 1000)  # the task's wall clock
        await agent.cleanup_task_resources()
        await Runner.release(conversation_id)
    answer["final"] = str(result.get("output") or "")
    answer["ok"] = result.get("result_type") == "answer" and bool(answer["final"])
    if not answer["ok"]:
        answer["error"] = f"result_type={result.get('result_type')!r}: {answer['final'][:200]}"
    return answer


async def browse(
    spec: BrowserAgentSpec,
    policy: BrowserPolicy,
    *,
    model_name: str,
    goal: str,
    timeout_s: float,
    max_steps: int,
    logs_dir: Path,
    headless: bool,
    chat: Model,
    decision_model: DecisionModel | None,
) -> Answer:
    """One task with a decision model (``jev``, ``laya``, ``cua``) or the chat model (``llm``) deciding every browser step.
    Needs a started Runner.

    A decision model writes ``decision_ticks.json`` under ``logs_dir`` and returns the ticks and the policy's report with
    the answer; ``llm`` writes ``chat_calls.json``. ``decision_model`` is required by every other name and unused
    by ``llm``.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    calls: list[dict[str, Any]] = []
    counted = CountingModel(chat, calls)
    workspace = str(logs_dir / "workspace")  # the harness scaffolds SOUL.md, memory/ and friends here, not in the cwd
    instance = BrowserInstanceConfig(launch_args=browser_launch_args(headless))
    match model_name:
        case "jev" | "laya" | "laya-served" | "cua":
            if decision_model is None:
                raise RuntimeError(f"--model {model_name} needs a decision model")
            slot_model = BrowserDecisionModel(spec, policy, counted, decision_model=decision_model, value_model=None)
            agent = create_browser_agent(
                slot_model,
                language=spec.language,
                max_iterations=max_steps,
                workspace=workspace,
                browser_instance=instance,
                browser_capabilities=["unsafe_dev"] if policy.batch_actions else None,
            )
            answer = await run_task(agent, goal, timeout_s=timeout_s)
            report = slot_model.report()
            answer["usage"] = await asyncio.to_thread(  # the price catalogue fetch is a blocking HTTP call
                usage_summary, calls, jev_input_tokens=report["jev_input_tokens"], decisions=report["decisions"]
            )
            answer["report"], answer["ticks"] = report, slot_model.ticks
            (logs_dir / "decision_ticks.json").write_text(
                json.dumps(
                    {
                        "report": report,
                        "ticks": slot_model.ticks,
                        "final": terminal_summary(answer["final"]),
                        "usage": answer["usage"],
                    },
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
            return finish_decision_model(answer, model_name=decision_model.name)
        case "llm":
            # create_browser_agent swaps a plain Model for a fresh copy at its own temperature; the marker keeps the counter in
            setattr(counted, _BROWSER_MODEL_TEMPERATURE_MARKER, DEFAULT_BROWSER_AGENT_TEMPERATURE)
            agent = create_browser_agent(
                counted,
                language=spec.language,
                max_iterations=max_steps,
                workspace=workspace,
                browser_instance=instance,
            )
            answer = finish_llm(await run_task(agent, goal, timeout_s=timeout_s))
            answer["usage"] = await asyncio.to_thread(
                usage_summary, calls, jev_input_tokens=0, decisions=sum(1 for call in calls if call["tool_calls"])
            )
            answer["usage"]["chat_temperature"] = chat.model_config.temperature  # the marker above hides it
            (logs_dir / "chat_calls.json").write_text(json.dumps(calls, ensure_ascii=False, indent=1), encoding="utf-8")
            return answer
        case _:
            raise ValueError(f"unknown model {model_name!r}; one of {BROWSER_MODEL_NAMES}")


def parser(spec: BrowserAgentSpec) -> argparse.ArgumentParser:
    """The shared browser flags: the spec's budget and goal as defaults, the policy switches off except prefetch."""
    build = argparse.ArgumentParser(prog=f"s1a run {spec.name}", description=spec.description)
    build.add_argument(
        "--model",
        choices=BROWSER_MODEL_NAMES,
        required=True,
        help="who decides each browser step: jev or laya-served (over HTTP), laya or cua (in process), or llm (the chat model in MODEL_NAME)",
    )
    build.add_argument(
        "--goal", default=spec.goal, required=spec.goal is None, help="the task; the spec's goal when it has one"
    )
    build.add_argument("--headed", action="store_true", help="show the browser (default: headless)")
    build.add_argument(
        "--timeout", type=positive_float, default=spec.budget.timeout_s, help="seconds for the whole task"
    )
    build.add_argument(
        "--max-steps",
        type=positive_int,
        default=spec.budget.max_steps,
        help="the subagent's iteration cap for the task",
    )
    build.add_argument(
        "--batch",
        choices=("on", "off"),
        default="off",
        help="decision models only: one browser_run_code_unsafe call per step (on, needs unsafe_dev) or the browser_* tools (off)",
    )
    build.add_argument(
        "--prefetch",
        choices=("on", "off"),
        default="on",
        help="decision models only: generate a typed value for every editable field as soon as a probe shows it",
    )
    build.add_argument(
        "--goal-values",
        choices=("on", "off"),
        default="off",
        help="decision models only: offer values extracted from the goal as a choice head",
    )
    build.add_argument(
        "--logs-dir",
        type=Path,
        default=RUNS_DIR / spec.name / f"{datetime.now():%Y-%m-%d__%H-%M-%S}",
        help="where the ticks, the chat calls and the workspace go",
    )
    build.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="decision models only: attach the profiler and write its JSON here",
    )
    return build


def policy_from_args(args: argparse.Namespace) -> BrowserPolicy:
    return BrowserPolicy(
        prefetch_values=args.prefetch == "on",
        batch_actions=args.batch == "on",
        goal_value_cache=args.goal_values == "on",
    )


async def play(spec: BrowserAgentSpec, args: argparse.Namespace) -> Answer:
    """One task inside an already started Runner: the answer without the ticks, which ``decision_ticks.json`` holds."""
    if args.profile_out is not None and args.model == "llm":
        raise RuntimeError("--profile-out needs a decision model: the profiler times the slot model's turns")
    chat = chat_model_from_env()
    decision_model = None if args.model == "llm" else build_model(args.model)
    profiler = BrowserProfiler().attach() if args.profile_out is not None else None
    try:
        answer = await browse(
            spec,
            policy_from_args(args),
            model_name=args.model,
            goal=args.goal,
            timeout_s=args.timeout,
            max_steps=args.max_steps,
            logs_dir=args.logs_dir,
            headless=not args.headed,
            chat=chat,
            decision_model=decision_model,
        )
    finally:
        if decision_model is not None:
            await decision_model.close()
        if profiler is not None:
            profiler.detach()
    result = {key: value for key, value in answer.items() if key != "ticks"}
    (
        args.logs_dir / "answer.json"
    ).write_text(  # the printed result, with the agent and the model, next to the run's records
        json.dumps({**result, "agent": spec.name, "model": args.model}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if profiler is not None:
        final_url = str((answer.get("terminal") or {}).get("url") or "")
        goal_url = URL_RE.search(args.goal)
        site = urlparse(goal_url.group(0))._replace(query="", fragment="").geturl() if goal_url else ""
        profile = profiler.report(
            ticks=answer.get("ticks") or [],
            waits=(answer.get("report") or {}).get("waits", 0),
            final_url=final_url,
            verified_prefix=site,
        )
        profile["model"], profile["usage"] = args.model, answer["usage"]
        print(render(profile), file=sys.stderr)
        args.profile_out.write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
