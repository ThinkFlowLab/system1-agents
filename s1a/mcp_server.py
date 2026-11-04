# coding: utf-8
"""``s1a-mcp``: the agents and the choice primitive as MCP tools, for hosts that pull tools instead of running a shell."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import json
import logging
import sys
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

import anyio

import s1a.console  # first: routes the harness logs to files before openjiuwen loads and logs to the console
from mcp.server.experimental.task_context import ServerTaskContext
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.exceptions import McpError
from mcp.shared.experimental.tasks.in_memory_task_store import InMemoryTaskStore
from mcp.types import (
    METHOD_NOT_FOUND,
    TASK_OPTIONAL,
    TASK_STATUS_CANCELLED,
    CallToolRequest,
    CallToolResult,
    CancelTaskRequest,
    ContentBlock,
    ErrorData,
    Result,
    ServerResult,
    Task,
    TaskStatus,
    TextContent,
    Tool,
    ToolAnnotations,
    ToolExecution,
)
from typing_extensions import NotRequired, TypedDict  # pydantic, under mcp, needs this TypedDict before Python 3.12

from s1a import config, probe, progress
from s1a.decision_models import build_model
from s1a import run as agents
from s1a.run import started_runner

INSTRUCTIONS = (
    "S1A agents: System 1 decision models (TypeSafe Jev, Laya, Cua-S1) in the model slot of openJiuwen agents. Use run_agent for a page task with enumerable "
    "controls, a game or a quiz, and decide for one selection over options you enumerate. One run or decision at a "
    "time per server; a second call fails at once, naming the first. A run can take minutes: raise the host's tool "
    "timeout to 600 s (Codex tool_timeout_sec, Claude Code MCP_TOOL_TIMEOUT). Not for arithmetic, "
    "constraint puzzles or free-text generation. list_agents gives every agent's flags: a browser agent takes "
    "--model jev --goal '...' and needs a chat-model key (OPENAI_API_KEY or LLM_API_KEY plus MODEL_NAME), a Jev key "
    "(TYPESAFE_API_KEY or OPENROUTER_API_KEY) and Node for @playwright/mcp; a tool agent takes --model, --rethink and "
    "--episodes; a rail takes --labelled-set. decide accepts model jev (default), clm, laya, cua or laya-served; local models "
    "need their optional extra and no Jev key, laya-served needs LAYA_SERVED_URL. run_agent reports progress per "
    "episode or browser step when the request carries a progressToken. A client that supports MCP tasks may call "
    "run_agent as a task: the call returns a task id at once, tasks/get reports the latest step and tasks/cancel "
    "stops the run."
)


@contextlib.asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """One Runner for the server's lifetime."""
    async with started_runner():
        yield


TASK_TOOLS = frozenset({"run_agent"})  # the one long call; list_agents and decide stay plain requests (#102)
TASK_TTL_MS = 3_600_000  # the shortest task ttl granted: longer than any run's budget, reported back to the client
_TASK_REPORTER: contextvars.ContextVar[progress.Reporter | None] = contextvars.ContextVar(
    "s1a_task_reporter", default=None
)
_TASK_SCOPES: dict[str, anyio.CancelScope] = {}  # task id: the scope tasks/cancel cancels


class _TaskServer(FastMCP):
    """FastMCP with ``run_agent`` callable as an MCP task (2025-11-25 experimental tasks in the 1.x SDK, #102).

    A request without task metadata takes the unchanged synchronous path. A task-augmented ``run_agent`` returns a
    ``CreateTaskResult`` at once; the run continues in the session's task group, its progress messages become the
    task's ``statusMessage``, and ``tasks/cancel`` cancels it through the same path as ``notifications/cancelled``.
    A tool error, the busy refusal included, completes the task with an ``isError`` result, as it is the result of
    the synchronous call.
    Tasks live in memory: they outlive the request, not the server process or the session."""

    async def list_tools(self) -> list[Tool]:
        tools = await super().list_tools()
        return [
            tool.model_copy(update={"execution": ToolExecution(taskSupport=TASK_OPTIONAL)})
            if tool.name in TASK_TOOLS
            else tool
            for tool in tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Sequence[ContentBlock] | dict[str, Any]:
        experimental = self.get_context().request_context.experimental
        if not experimental.is_task:
            return await super().call_tool(name, arguments)
        metadata = experimental.task_metadata
        if metadata is not None:  # the store counts ttl from creation, so a shorter one would drop a run mid-way
            metadata.ttl = max(metadata.ttl or 0, TASK_TTL_MS)
        created = await experimental.run_task(lambda task: self._run_as_task(task, name, arguments))
        return created  # the low-level handler passes a CreateTaskResult through

    async def _run_as_task(self, task: ServerTaskContext, name: str, arguments: dict[str, Any]) -> CallToolResult:
        async def status(done: float, total: float | None, message: str) -> None:
            await task.update_status(message)  # the task's statusMessage, and notifications/tasks/status

        token = _TASK_REPORTER.set(_rate_bounded(status))
        with anyio.CancelScope() as scope:
            _TASK_SCOPES[task.task_id] = scope
            try:
                converted = await FastMCP.call_tool(self, name, arguments)
            except Exception as exc:  # a tool error is the task's result, as it is the synchronous call's
                return CallToolResult(content=[TextContent(type="text", text=str(exc))], isError=True)
            finally:
                _TASK_SCOPES.pop(task.task_id, None)
                _TASK_REPORTER.reset(token)
        if scope.cancelled_caught:  # tasks/cancel already set the status; this result is never delivered
            return CallToolResult(content=[TextContent(type="text", text=f"{name} cancelled")], isError=True)
        return _call_tool_result(converted)


def _call_tool_result(converted: Any) -> CallToolResult:
    """What the low-level handler would have built from FastMCP's converted result."""
    if isinstance(converted, CallToolResult):
        return converted
    if isinstance(converted, tuple):
        content, structured = converted
        return CallToolResult(content=list(content), structuredContent=structured)
    if isinstance(converted, dict):
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(converted, indent=2))], structuredContent=converted
        )
    return CallToolResult(content=list(converted))


class _TaskStore(InMemoryTaskStore):
    """A cancelled task stays cancelled: a status or result the run writes while it unwinds is dropped, not raised."""

    async def update_task(
        self, task_id: str, status: TaskStatus | None = None, status_message: str | None = None
    ) -> Task:
        current = await self.get_task(task_id)
        if current is not None and current.status == TASK_STATUS_CANCELLED:
            return current
        return await super().update_task(task_id, status=status, status_message=status_message)

    async def store_result(self, task_id: str, result: Result) -> None:
        current = await self.get_task(task_id)
        if current is None or current.status != TASK_STATUS_CANCELLED:
            await super().store_result(task_id, result)


server = _TaskServer("s1a", instructions=INSTRUCTIONS, lifespan=lifespan)
# In memory, one store per server process; registers tasks/get, tasks/result, tasks/list and tasks/cancel.
# Deprecated with SEP-1686 leaving the spec for an extension; the only task API 1.x has (docs/architecture.md).
_TASKS = server._mcp_server.experimental.enable_tasks(store=_TaskStore())  # ty: ignore[deprecated]
_default_cancel = server._mcp_server.request_handlers[CancelTaskRequest]


async def _cancel_task(req: CancelTaskRequest) -> ServerResult:
    """The SDK's tasks/cancel (scope check, terminal-state check, status) and then stop the run itself."""
    result = await _default_cancel(req)
    scope = _TASK_SCOPES.get(req.params.taskId)
    if scope is not None:
        scope.cancel()
    return result


server._mcp_server.request_handlers[CancelTaskRequest] = _cancel_task
_call_tool_handler = server._mcp_server.request_handlers[CallToolRequest]


async def _call_tool_request(req: CallToolRequest) -> ServerResult:
    """A task-augmented call of a tool that takes none is a protocol error (-32601), not a tool result."""
    if req.params.task is not None and req.params.name not in TASK_TOOLS:
        raise McpError(
            ErrorData(code=METHOD_NOT_FOUND, message=f"{req.params.name} does not support task-augmented calls")
        )
    return await _call_tool_handler(req)


server._mcp_server.request_handlers[CallToolRequest] = _call_tool_request
# FastMCP puts a RichHandler on the root logger and the harness loggers propagate to it; only warnings pass.
for _handler in logging.getLogger().handlers:
    _handler.setLevel(logging.WARNING)
_ONE_RUN = (
    asyncio.Lock()
)  # ponytail: runs are serialized; per-run browser and workspace isolation before parallel tasks
_RUNNING: list[tuple[str, float]] = []  # the call holding _ONE_RUN and its monotonic start, for the busy error
PROGRESS_INTERVAL_S = 1.0  # at most one progress notification per second, the last point of a run always sent
log = logging.getLogger(__name__)


def _refuse_if_busy(label: str) -> None:
    """A tool error naming the call that holds the run slot, if one does."""
    if _ONE_RUN.locked():
        running, started = _RUNNING[-1] if _RUNNING else ("another call", time.monotonic())
        raise ToolError(
            f"{label} refused: {running} has been running for {time.monotonic() - started:.0f} s and this server "
            "runs one call at a time; call again when it finishes, or cancel it"
        )


@contextlib.asynccontextmanager
async def _one_at_a_time(label: str) -> AsyncIterator[None]:
    """Hold the server's one run slot, or fail at once with a tool error naming the call that holds it.

    Queueing would outlast a host's tool timeout (Codex's default is 60 s) while the earlier run kept going, so a
    second call is refused instead of waiting. Checking and taking the lock happen without a suspension point
    between them, so two calls cannot both pass the check."""
    _refuse_if_busy(label)
    async with _ONE_RUN:
        _RUNNING.append((label, time.monotonic()))
        try:
            yield
        except asyncio.CancelledError:
            log.warning("%s cancelled after %.0f s", label, time.monotonic() - _RUNNING[-1][1])
            raise
        finally:
            _RUNNING.pop()


def _progress_reporter(ctx: Context) -> progress.Reporter | None:
    """Forward progress to the host when its request carries a progressToken: increasing values, rate-bounded."""
    meta = ctx.request_context.meta
    if meta is None or meta.progressToken is None:
        return None
    return _rate_bounded(ctx.report_progress)


def _rate_bounded(report: progress.Reporter) -> progress.Reporter:
    """Increasing values only, at most one per PROGRESS_INTERVAL_S, the final point always passed on."""
    last: list[float] = [float("-inf"), float("-inf")]  # the last value sent and when

    async def send(done: float, total: float | None, message: str) -> None:
        now = time.monotonic()
        final = total is not None and done >= total
        if done <= last[0] or (not final and now - last[1] < PROGRESS_INTERVAL_S):
            return  # the spec wants increasing values; a burst of fast steps sends one notification per second
        last[0], last[1] = done, now
        await report(done, total, message)

    return send


class AgentRow(TypedDict):
    """One ``list_agents`` row; an agent that fails to import has no flags."""

    name: str
    front: str
    description: str
    flags: NotRequired[list[dict[str, Any]]]


class Decision(TypedDict):
    """What ``decide`` returns: ``probe.pick``'s four keys."""

    choice: str
    probabilities: dict[str, float]
    confidence: float
    ms: int


# Hints for hosts that auto-approve by them (Codex approval modes, Claude permissions); they are hints, not enforcement.
LIST_AGENTS_HINTS = ToolAnnotations(  # imports local agent modules; no I/O beyond the checkout
    readOnlyHint=True, idempotentHint=True, openWorldHint=False
)
DECIDE_HINTS = ToolAnnotations(  # changes nothing, but jev, clm and laya-served call a remote service that may bill
    readOnlyHint=True, openWorldHint=True
)
RUN_AGENT_HINTS = ToolAnnotations(  # a browser agent clicks and submits on live sites; a tool agent writes job folders
    readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
)


@server.tool(title="List S1A agents", annotations=LIST_AGENTS_HINTS)
async def list_agents() -> list[AgentRow]:
    """Every S1A agent: its name, its front (tool, browser or rail), what it does and the flags run_agent takes for it
    (flag, takes_value, required, choices, default, help; a flag with takes_value false is passed alone); an agent
    that fails to import, a missing optional dependency included, is listed as unavailable with the error instead."""
    return await asyncio.to_thread(_agent_rows)  # importing the agent modules is slow and must not stall the loop


def _agent_rows() -> list[AgentRow]:
    rows: list[AgentRow] = []
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


@server.tool(title="Run an S1A agent", annotations=RUN_AGENT_HINTS)
async def run_agent(name: str, flags: list[str], ctx: Context) -> dict[str, Any]:
    """Run one agent with the flags `s1a run <name>` takes, e.g. ["--model", "jev", "--goal", "..."].

    A browser agent returns its answer, a tool agent its series summary with the job folder it wrote, a rail its
    evaluation summary. The result stays an untyped object because its shape depends on the agent's front. A run can
    take minutes: progress goes out per episode or browser step when the request has a progressToken, a call made
    while another runs is refused at once, and a cancelled run releases its browser and records the cancellation.
    A client with MCP task support may call it as a task and poll tasks/get instead of holding the request open.
    """
    reporter = _TASK_REPORTER.get() or _progress_reporter(ctx)
    # argparse writes help to stdout and any stray print would too; under the stdio transport that stream is the protocol
    async with _one_at_a_time(f"run_agent {name}"):
        with (
            contextlib.redirect_stdout(sys.stderr),
            progress.reporting(reporter) if reporter is not None else contextlib.nullcontext(),
        ):
            try:
                return await agents.run_by_name(name, flags)
            except SystemExit as exc:
                raise ValueError(f"{name}: bad flags {flags}; `s1a run {name} --help` lists them") from exc


@server.tool(title="Decide one choice", annotations=DECIDE_HINTS)
async def decide(
    state: dict[str, Any],
    options: dict[str, str],
    rules: str,
    model: Literal["jev", "clm", "laya", "laya-served", "cua"] = "jev",
) -> Decision:
    """One choice question: the chosen key, probabilities, confidence (= max p) and decision latency in ms.

    Use jev over HTTP (default), laya/cua locally after installing the matching extra, or laya-served against a
    running Laya server (LAYA_SERVED_URL). Local model loading
    is excluded from the reported latency.
    """
    async with _one_at_a_time(f"decide ({model})"):
        with contextlib.redirect_stdout(sys.stderr):  # local SDK loading must not write to the stdio protocol
            decision_model = await asyncio.to_thread(build_model, model)
            try:
                answer = await probe.pick(decision_model, state=state, options=options, rules=rules)
                return Decision(
                    choice=answer["choice"],
                    probabilities=answer["probabilities"],
                    confidence=answer["confidence"],
                    ms=answer["ms"],
                )
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
