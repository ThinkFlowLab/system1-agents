# coding: utf-8
"""One command line and one lifecycle for a tool-front agent: parse the flags, play the seeds, write the job folder."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from s1a.decision_models import DecisionModel, build_model
from s1a.config import first_env, optional_chat_model
from s1a.jobs import Episode, summarize, write_job
from s1a.pricing import chat_prices, cost_usd, env_prices
from s1a.spec import Json, ToolAgentSpec, positive_float, positive_int
from s1a.tool.loop import MODEL_NAMES, run_episode


def parser(spec: ToolAgentSpec) -> argparse.ArgumentParser:
    """The shared flags, with the spec's budget as their defaults, then the agent's own flags."""
    build = argparse.ArgumentParser(prog=f"s1a run {spec.name}", description=spec.description)
    build.add_argument(
        "--model",
        choices=MODEL_NAMES,
        required=True,
        help="who decides: jev or laya-served (over HTTP), laya or cua (in process), llm (the chat model in MODEL_NAME), random, or rule (the agent's baseline)",
    )
    build.add_argument(
        "--rethink",
        choices=("on", "off"),
        required=True,
        help="the rethink rail: a plan from the chat model after a stall, a block after a repeat (evals/README.md)",
    )
    build.add_argument(
        "--episodes", type=positive_int, required=True, help="seeds --seed to --seed+N-1, one episode each"
    )
    build.add_argument("--seed", type=int, default=0, help="the first seed")
    build.add_argument("--max-steps", type=positive_int, default=spec.budget.max_steps, help="acts per episode")
    build.add_argument(
        "--timeout",
        type=positive_float,
        default=spec.budget.timeout_s,
        help="seconds per episode; the score so far counts",
    )
    build.add_argument(
        "--headed", action="store_true", help="page games only (game2048, millionaire): show the browser"
    )
    build.add_argument("--log", action="store_true", help="print each act and decision to stderr")
    build.add_argument(
        "--showcase",
        action="store_true",
        help="write the job under evals/showcase/ (outside the matrix) and, for the browser games, one PNG per move",
    )
    spec.flags(build)
    return build


def price_episodes(episodes: list[Episode]) -> None:
    """Dollars for the episodes that spent chat tokens: one catalogue lookup, and none when nothing called the chat model."""
    spent = [e for e in episodes if e.chat_input_tokens + e.chat_output_tokens]
    if not spent:
        return
    prices = chat_prices(first_env("MODEL_NAME"))
    for episode in spent:
        episode.cost_usd = cost_usd(
            episode.jev_input_tokens,
            episode.chat_input_tokens,
            episode.chat_output_tokens,
            episode.chat_cache_tokens,
            prices,
        )


async def play(spec: ToolAgentSpec, args: argparse.Namespace, *, results_dir: Path) -> Json:
    """Play the series inside an already started Runner; the summary with its job folder under ``results_dir``.

    Jev, Laya, Cua and the rule share one model for the series (warmed once, closed at the end); chance gets a fresh
    stream per episode from that episode's seed. The keys and the price variables are checked before the series is
    built. Whatever ends the seed loop, the finished episodes are written; a series whose every episode errored writes
    its job folder, then raises."""
    env_prices()
    chat = optional_chat_model()
    if args.model == "llm" and chat is None:
        raise RuntimeError("--model llm needs the chat model: OPENAI_API_KEY or LLM_API_KEY, and MODEL_NAME")
    if args.rethink == "on" and spec.budget.stall_after > 0 and chat is None:
        raise RuntimeError("--rethink on needs the chat model for plans: OPENAI_API_KEY or LLM_API_KEY, and MODEL_NAME")
    shared = build_model(args.model) if args.model in ("jev", "laya", "laya-served", "cua") else None
    run = await asyncio.to_thread(spec.series, args)  # question fetches, game file parsing: seconds of blocking I/O
    if args.model == "rule":
        shared = build_model("rule", rule=run.baseline)

    def model_for(seed: int) -> DecisionModel | None:
        if args.model == "llm":
            return None
        if args.model == "random":
            return build_model("random", seed=seed)
        return shared

    episodes: list[Episode] = []
    try:
        if shared is not None:
            await shared.warm()
        async with run.session:
            for seed in run.seeds:
                env = run.env_for(seed)
                episode = await run_episode(
                    spec,
                    env,
                    model_name=args.model,
                    seed=seed,
                    chat=chat,
                    decision_model=model_for(seed),
                    rethink_on=args.rethink == "on",
                    max_acts=args.max_steps,
                    timeout_s=float(args.timeout),
                    prices=None,
                    log=args.log,
                )
                run.annotate(env, episode)
                episodes.append(episode)
    finally:
        if shared is not None:
            await shared.close()
        if episodes:
            await asyncio.to_thread(price_episodes, episodes)  # the price catalogue fetch is a blocking HTTP call
            job_dir = write_job(spec.name, episodes, results_dir=results_dir)
    if not episodes:
        raise RuntimeError("the series selected no episodes")
    if all(episode.error is not None for episode in episodes):
        raise RuntimeError(f"every episode failed: {episodes[0].error}; job folder {job_dir}")
    return {**summarize(episodes), "job_dir": str(job_dir)}
