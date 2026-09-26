# coding: utf-8
"""``s1a-mcp``: the agents and the choice primitive as MCP tools, for hosts that pull tools instead of running a shell."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator
from typing import Any, Literal

import s1a.console  # first: routes the harness logs to files before openjiuwen loads and logs to the console
from mcp.server.fastmcp import FastMCP

from s1a import config, probe
from s1a.decision_models import build_model
from s1a import run as agents
from s1a.run import started_runner

INSTRUCTIONS = (
    "S1A agents: System 1 decision models (TypeSafe Jev, Laya, Cua-S1) in the model slot of openJiuwen agents. Use run_agent for a page task with enumerable "
    "controls, a game or a quiz, and decide for one selection over options you enumerate. Not for arithmetic, "
    "constraint puzzles or free-text generation. list_agents gives every agent's flags: a browser agent takes "
    "--model jev --goal '...' and needs a chat-model key (OPENAI_API_KEY or LLM_API_KEY plus MODEL_NAME), a Jev key "
    "(TYPESAFE_API_KEY or OPENROUTER_API_KEY) and Node for @playwright/mcp; a tool agent takes --model, --rethink and "
    "--episodes; a rail takes --labelled-set. decide accepts model jev (default), laya or cua; local models need "
    "their optional extra and no Jev key. Runs and decisions go one at a time per server."
)


@contextlib.asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """One Runner for the server's lifetime."""
    async with started_runner():
        yield


server = FastMCP("s1a", instructions=INSTRUCTIONS, lifespan=lifespan)
# FastMCP puts a RichHandler on the root logger and the harness loggers propagate to it; only warnings pass.
for _handler in logging.getLogger().handlers:
    _handler.setLevel(logging.WARNING)
_ONE_RUN = (
    asyncio.Lock()
)  # ponytail: runs are serialized; per-run browser and workspace isolation before parallel tasks


@server.tool()
async def list_agents() -> list[dict[str, Any]]:
    """Every S1A agent: its name, its front (tool, browser or rail), what it does and the flags run_agent takes for it
    (flag, takes_value, required, choices, default, help; a flag with takes_value false is passed alone); an agent
    that fails to import, a missing optional dependency included, is listed as unavailable with the error instead."""
    return await asyncio.to_thread(_agent_rows)  # importing the agent modules is slow and must not stall the loop


def _agent_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in agents.names():
        try:
            spec = agents.load(name)
        except Exception as exc:  # one agent's import failure is its own row, never the whole listing's
            rows.append({"name": name, "front": "unavailable", "description": str(exc)})
            continue
        rows.append(
            {
                "name": name,
                "front": agents.FRONTS[type(spec)].label,
                "description": spec.description,
                "flags": agents.flags(spec),
            }
        )
    return rows


@server.tool()
async def run_agent(name: str, flags: list[str]) -> dict[str, Any]:
    """Run one agent with the flags `s1a run <name>` takes, e.g. ["--model", "jev", "--goal", "..."].

    A browser agent returns its answer, a tool agent its series summary with the job folder it wrote, a rail its
    evaluation summary.
    """
    # argparse writes help to stdout and any stray print would too; under the stdio transport that stream is the protocol
    async with _ONE_RUN:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                return await agents.run_by_name(name, flags)
            except SystemExit as exc:
                raise ValueError(f"{name}: bad flags {flags}; `s1a run {name} --help` lists them") from exc


@server.tool()
async def decide(
    state: dict[str, Any], options: dict[str, str], rules: str, model: Literal["jev", "laya", "cua"] = "jev"
) -> dict[str, Any]:
    """One choice question: the chosen key, probabilities, confidence and decision latency in ms.

    Use jev over HTTP (default), or laya/cua locally after installing the matching extra. Local model loading
    is excluded from the reported latency.
    """
    async with _ONE_RUN:
        with contextlib.redirect_stdout(sys.stderr):  # local SDK loading must not write to the stdio protocol
            decision_model = await asyncio.to_thread(build_model, model)
            try:
                return await probe.pick(decision_model, state=state, options=options, rules=rules)
            finally:
                await decision_model.close()


def main() -> None:
    """Serve over stdio, inside the host's own process."""
    config.load_env()
    argparse.ArgumentParser(prog="s1a-mcp", description=INSTRUCTIONS).parse_args()
    server.run(transport="stdio")


if __name__ == "__main__":
    s1a.console.utf8_console()
    main()
