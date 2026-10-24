# coding: utf-8
"""The MCP server over an in-memory session: the three tools, decide against a fake client, the agent listing, and
long runs (progress, the busy refusal, cancellation)."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import jsonschema
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import AsyncIterator
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase, skipUnless
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CancelledNotification, CancelledNotificationParams, ClientNotification

from s1a import mcp_server
from s1a.browser import browse
from s1a import run as agents
from s1a.decision_models import JevModel, ScriptedModel, ScriptedTransport
from s1a.tool import loop, series
from s1a.agents.flights import SPEC as FLIGHTS
from support import COUNTER

RANDOM_RUN = ["--model", "random", "--rethink", "off", "--episodes", "1", "--showcase"]
NO_KEYS = {"TYPESAFE_API_KEY": "", "OPENROUTER_API_KEY": "", "MODEL_NAME": "", "OPENAI_API_KEY": "", "LLM_API_KEY": ""}
HAVE_RLCARD = importlib.util.find_spec("rlcard") is not None  # the one tool agent that runs offline
ROOT = Path(__file__).resolve().parents[1]
HARNESS_LOG = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ")  # the console entry routes these to files


def _rows(result: Any) -> Any:
    if result.structuredContent is not None:
        return result.structuredContent.get("result", result.structuredContent)
    return json.loads(result.content[0].text)


class TestRunAgentInMemory(IsolatedAsyncioTestCase):
    """``run_agent`` on the counter spec: the summary comes back as the result and nothing reaches sys.stdout."""

    async def test_a_tool_agent_returns_its_summary_and_help_is_a_tool_error(self) -> None:
        captured = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(agents, "load", lambda name: COUNTER),
            patch.object(agents, "SHOWCASE_DIR", Path(tmp) / "showcase"),
            patch.object(loop, "WORKSPACE", Path(tmp) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
            contextlib.redirect_stdout(captured),
        ):
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                result = await session.call_tool("run_agent", {"name": "counter", "flags": RANDOM_RUN})
                help_result = await session.call_tool("run_agent", {"name": "counter", "flags": ["--help"]})
            self.assertFalse(result.isError)
            summary = _rows(result)
            self.assertEqual((summary["env"], summary["policy"], summary["episodes"]), ("counter", "random", 1))
            self.assertTrue(Path(summary["job_dir"]).is_dir())
        self.assertTrue(help_result.isError)
        self.assertIn("bad flags", help_result.content[0].text)
        stray = [line for line in captured.getvalue().splitlines() if line.strip() and not HARNESS_LOG.match(line)]
        self.assertEqual(stray, [])


class TestStdioTransport(IsolatedAsyncioTestCase):
    """The advertised entry over a real stdio pipe: the listing, argparse help and a tool-agent run leave the protocol intact."""

    def setUp(self) -> None:
        self.strays: list[Exception] = []

    @contextlib.asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        async def record(message: Any) -> None:
            if isinstance(message, Exception):
                self.strays.append(message)

        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", "from s1a import entry; entry.s1a_mcp()"],
            env={**os.environ, **NO_KEYS, "PYTHONUNBUFFERED": "1"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, message_handler=record) as session:
                await session.initialize()
                yield session

    async def test_the_listing_and_help_leave_the_stream_clean(self) -> None:
        async with self._session() as session:
            listed = await session.call_tool("list_agents", {})
            help_result = await session.call_tool("run_agent", {"name": "game2048", "flags": ["--help"]})
            again = await session.call_tool("list_agents", {})
        self.assertEqual(self.strays, [])
        self.assertFalse(listed.isError)
        self.assertEqual([row["name"] for row in _rows(listed)], agents.names())
        self.assertTrue(help_result.isError)
        self.assertIn("bad flags", help_result.content[0].text)
        self.assertEqual([row["name"] for row in _rows(again)], agents.names())

    @skipUnless(HAVE_RLCARD, "the blackjack extra (rlcard) is not installed")
    async def test_a_blackjack_run_leaves_the_stream_clean_and_writes_under_s1a_home(self) -> None:
        async with self._session() as session:
            run = await session.call_tool(
                "run_agent", {"name": "blackjack", "flags": ["--model", "rule", *RANDOM_RUN[2:]]}
            )
        self.assertEqual(self.strays, [])
        self.assertFalse(run.isError, run.content[0].text if run.content else run)
        self.assertEqual((_rows(run)["env"], _rows(run)["episodes"]), ("blackjack", 1))
        self.assertTrue(Path(_rows(run)["job_dir"]).is_relative_to(os.environ["S1A_HOME"]), _rows(run)["job_dir"])


class TestImportAlone(TestCase):
    """``python -m s1a.mcp_server`` or ``mcp run s1a/mcp_server.py:server`` import the module without the console entry."""

    def test_importing_the_server_module_alone_prints_nothing(self) -> None:
        done = subprocess.run(
            [sys.executable, "-c", "import s1a.mcp_server"], capture_output=True, text=True, timeout=120, check=True
        )
        self.assertEqual(done.stdout, "")

    def test_running_the_server_module_prints_its_help(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "s1a.mcp_server", "--help"], capture_output=True, text=True, timeout=120, check=True
        )
        self.assertTrue(done.stdout.startswith("usage: s1a-mcp"), done.stdout[:80])


class TestMcpServer(IsolatedAsyncioTestCase):
    async def test_exposes_the_three_tools(self) -> None:
        async with create_connected_server_and_client_session(mcp_server.server) as session:
            tools = await session.list_tools()
        self.assertEqual(sorted(tool.name for tool in tools.tools), ["decide", "list_agents", "run_agent"])

    async def test_decide_returns_the_validated_choice(self) -> None:
        transport = ScriptedTransport(choose="inc")
        with patch.object(mcp_server, "build_model", return_value=JevModel(transport)) as build:
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                result = await session.call_tool(
                    "decide", {"state": {"n": 1}, "options": {"inc": "add one", "noop": "do nothing"}, "rules": "count"}
                )
        self.assertFalse(result.isError)
        answer = _rows(result)
        self.assertEqual((answer["choice"], answer["probabilities"]["inc"], answer["ms"]), ("inc", 1.0, 9))
        self.assertEqual(transport.bodies[0]["questions"]["pick"]["instructions"]["rules"], "count")
        build.assert_called_once_with("jev")

    async def test_decide_advertises_optional_model_choices(self) -> None:
        async with create_connected_server_and_client_session(mcp_server.server) as session:
            tools = await session.list_tools()
        schema = next(tool.inputSchema for tool in tools.tools if tool.name == "decide")
        self.assertEqual(schema["properties"]["model"]["enum"], ["jev", "clm", "laya", "laya-served", "cua"])
        self.assertEqual(schema["properties"]["model"]["default"], "jev")
        self.assertNotIn("model", schema["required"])

    async def test_decide_selects_and_closes_each_local_model_without_api_keys(self) -> None:
        for model_name in ("laya", "laya-served", "cua"):
            with self.subTest(model=model_name):
                decision_model = ScriptedModel(choose="inc")
                with (
                    patch.dict(os.environ, NO_KEYS),
                    patch.object(mcp_server, "build_model", return_value=decision_model) as build,
                    patch.object(decision_model, "close", new_callable=AsyncMock) as close,
                ):
                    async with create_connected_server_and_client_session(mcp_server.server) as session:
                        result = await session.call_tool(
                            "decide",
                            {"state": {"n": 1}, "options": {"inc": "add one"}, "rules": "count", "model": model_name},
                        )
                self.assertFalse(result.isError)
                self.assertEqual(
                    _rows(result), {"choice": "inc", "probabilities": {"inc": 1.0}, "confidence": 1.0, "ms": 9}
                )
                build.assert_called_once_with(model_name)
                close.assert_awaited_once()

    async def test_decide_rejects_unsupported_models_before_building_one(self) -> None:
        with patch.object(mcp_server, "build_model") as build:
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                for model_name in ("llm", "random", "rule", "unknown"):
                    with self.subTest(model=model_name):
                        result = await session.call_tool(
                            "decide", {"state": {}, "options": {"a": "a"}, "rules": "pick", "model": model_name}
                        )
                        self.assertTrue(result.isError)
        build.assert_not_called()

    async def test_decide_closes_a_local_model_when_the_decision_fails(self) -> None:
        decision_model = ScriptedModel(error=RuntimeError("local inference failed"))
        with (
            patch.object(mcp_server, "build_model", return_value=decision_model),
            patch.object(decision_model, "close", new_callable=AsyncMock) as close,
        ):
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                result = await session.call_tool(
                    "decide", {"state": {}, "options": {"a": "a"}, "rules": "pick", "model": "cua"}
                )
        self.assertTrue(result.isError)
        self.assertIn("local inference failed", result.content[0].text)
        close.assert_awaited_once()

    async def test_decide_reports_how_to_install_a_missing_local_model(self) -> None:
        for model_name, module_name in (("laya", "laya"), ("cua", "cua_s1.nano")):
            with (
                self.subTest(model=model_name),
                patch.dict(os.environ, NO_KEYS),
                patch.dict(sys.modules, {module_name: None}),
            ):
                async with create_connected_server_and_client_session(mcp_server.server) as session:
                    result = await session.call_tool(
                        "decide", {"state": {}, "options": {"a": "a"}, "rules": "pick", "model": model_name}
                    )
                self.assertTrue(result.isError)
                self.assertIn(f"uv sync --extra {model_name}", result.content[0].text)

    async def test_local_model_loading_keeps_stdout_clean(self) -> None:
        def build(model_name: str) -> ScriptedModel:
            print("loading local weights")
            return ScriptedModel(choose="a")

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(mcp_server, "build_model", build),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                result = await session.call_tool(
                    "decide", {"state": {}, "options": {"a": "a"}, "rules": "pick", "model": "laya"}
                )
        self.assertFalse(result.isError)
        self.assertNotIn("loading local weights", stdout.getvalue())
        self.assertIn("loading local weights", stderr.getvalue())

    async def test_list_agents_names_every_front_with_its_flags(self) -> None:
        async with create_connected_server_and_client_session(mcp_server.server) as session:
            result = await session.call_tool("list_agents", {})
        self.assertFalse(result.isError)
        rows = {row["name"]: row for row in _rows(result)}
        fronts = tuple(rows[name]["front"] for name in ("game2048", "flights", "injection_guard"))
        self.assertEqual(fronts, ("tool", "browser", "rail"))
        for name in ("blackjack", "alfworld"):  # their extras may be absent; the row stays, with the install line
            self.assertIn(rows[name]["front"], ("tool", "unavailable"))
            if rows[name]["front"] == "unavailable":
                self.assertIn(f"uv sync --extra {name}", rows[name]["description"])
        flags = {name: {row["flag"]: row for row in rows[name]["flags"]} for name in ("game2048", "flights")}
        self.assertEqual(
            (flags["game2048"]["--model"]["required"], flags["game2048"]["--model"]["choices"]),
            (True, ["jev", "clm", "llm", "random", "rule", "laya", "laya-served", "cua"]),
        )
        max_steps = str(agents.load("game2048").budget.max_steps)
        self.assertEqual(
            (flags["game2048"]["--max-steps"]["required"], flags["game2048"]["--max-steps"]["default"]),
            (False, max_steps),
        )
        self.assertEqual(
            (flags["game2048"]["--model"]["takes_value"], flags["game2048"]["--headed"]["takes_value"]), (True, False)
        )
        self.assertIsNone(flags["game2048"]["--headed"]["default"])  # a switch is passed alone: no value to send
        self.assertEqual(
            (flags["flights"]["--goal"]["required"], flags["flights"]["--batch"]["default"]), (False, "off")
        )
        self.assertNotIn("--help", flags["flights"])

    async def test_one_agent_failing_to_import_is_its_own_row_and_the_others_stay(self) -> None:
        real = agents.load

        def load(name: str) -> agents.Spec:
            if name == "alfworld":
                raise FileNotFoundError("ALFWORLD_DATA is empty")
            return real(name)

        with patch.object(agents, "load", load):
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                result = await session.call_tool("list_agents", {})
        self.assertFalse(result.isError)
        rows = {row["name"]: row for row in _rows(result)}
        self.assertEqual(
            rows["alfworld"], {"name": "alfworld", "front": "unavailable", "description": "ALFWORLD_DATA is empty"}
        )
        self.assertEqual(rows["game2048"]["front"], "tool")


async def _cancel(session: ClientSession, request_id: int) -> None:
    params = CancelledNotificationParams(requestId=request_id, reason="host gave up")
    await session.send_notification(ClientNotification(CancelledNotification(params=params)))


class TestLongRuns(IsolatedAsyncioTestCase):
    """What a host with a tool timeout needs: progress, a fast refusal while busy, and a clean cancellation."""

    def setUp(self) -> None:
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(agents, "load", lambda name: COUNTER))
        self.enterContext(patch.object(agents, "SHOWCASE_DIR", self.tmp / "showcase"))
        self.enterContext(patch.object(loop, "WORKSPACE", self.tmp / "ws"))
        self.enterContext(patch.object(series, "optional_chat_model", lambda: None))

    def test_the_first_512_characters_say_runs_are_serialized_and_need_a_raised_timeout(self) -> None:
        head = mcp_server.INSTRUCTIONS[:512]
        self.assertIn("One run or decision at a time per server", head)
        for setting in ("tool timeout", "tool_timeout_sec", "MCP_TOOL_TIMEOUT"):
            self.assertIn(setting, head)

    async def test_a_progress_token_gets_increasing_progress_per_episode(self) -> None:
        points: list[tuple[float, float | None, str | None]] = []

        async def heard(done: float, total: float | None, message: str | None) -> None:
            points.append((done, total, message))

        flags = [*RANDOM_RUN[:5], "3", "--showcase"]
        with patch.object(mcp_server, "PROGRESS_INTERVAL_S", 0.0):
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                result = await session.call_tool(
                    "run_agent", {"name": "counter", "flags": flags}, progress_callback=heard
                )
        self.assertFalse(result.isError, result.content)
        self.assertEqual(_rows(result)["episodes"], 3)
        self.assertEqual(points, [(1, 3, "episode 1/3"), (2, 3, "episode 2/3"), (3, 3, "episode 3/3")])

    async def test_without_a_token_no_progress_is_sent(self) -> None:
        flags = [*RANDOM_RUN[:5], "3", "--showcase"]
        with patch.object(mcp_server.Context, "report_progress", new_callable=AsyncMock) as sent:
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                result = await session.call_tool("run_agent", {"name": "counter", "flags": flags})
        self.assertFalse(result.isError, result.content)
        sent.assert_not_awaited()

    async def test_the_rate_bound_keeps_the_last_point_and_drops_repeats(self) -> None:
        ctx = SimpleNamespace(
            request_context=SimpleNamespace(meta=SimpleNamespace(progressToken="t")), report_progress=AsyncMock()
        )
        send = mcp_server._progress_reporter(ctx)  # ty: ignore[invalid-argument-type]
        assert send is not None
        for done in (1, 1, 2, 3, 4):
            await send(done, 4, f"step {done}/4")
        sent = [call.args for call in ctx.report_progress.await_args_list]
        self.assertEqual(sent, [(1, 4, "step 1/4"), (4, 4, "step 4/4")])  # 2 and 3 fell inside the interval

    async def test_a_call_while_a_run_holds_the_slot_is_refused_at_once(self) -> None:
        started = asyncio.Event()

        async def slow(name: str, flags: list[str]) -> dict[str, Any]:
            started.set()
            await asyncio.Event().wait()
            return {}

        with (
            patch.object(agents, "run_by_name", slow),
            patch.object(mcp_server, "build_model", return_value=ScriptedModel(choose="a")),
        ):
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                request_id = session._request_id  # the id the next request takes
                first = asyncio.create_task(session.call_tool("run_agent", {"name": "counter", "flags": []}))
                await asyncio.wait_for(started.wait(), 5)
                busy = await asyncio.wait_for(
                    session.call_tool("decide", {"state": {}, "options": {"a": "a"}, "rules": "pick"}), 1
                )
                again = await asyncio.wait_for(
                    session.call_tool("run_agent", {"name": "counter", "flags": RANDOM_RUN}), 1
                )
                await _cancel(session, request_id)
                with contextlib.suppress(McpError):
                    await asyncio.wait_for(first, 5)
        for refused in (busy, again):
            self.assertTrue(refused.isError)
            self.assertIn("run_agent counter has been running for", refused.content[0].text)
            self.assertIn("one call at a time", refused.content[0].text)

    async def test_a_cancelled_browser_run_tears_down_and_frees_the_slot(self) -> None:
        started, torn_down = asyncio.Event(), []

        async def run_agent(agent: Any, inputs: dict[str, Any]) -> dict[str, Any]:
            started.set()
            await asyncio.Event().wait()
            return {}

        async def cleanup() -> None:
            await asyncio.sleep(0.01)  # a real teardown awaits; the cancelled task must not cut it short
            torn_down.append("browser")

        async def release(conversation_id: str) -> None:
            await asyncio.sleep(0)
            torn_down.append("session")

        agent = SimpleNamespace(ensure_initialized=AsyncMock(), cleanup_task_resources=cleanup)
        runner = SimpleNamespace(run_agent=run_agent, release=release)

        async def browser_run(name: str, flags: list[str]) -> dict[str, Any]:
            return dict(await browse.run_task(agent, "goal", timeout_s=60, logs_dir=self.tmp))  # ty: ignore[invalid-argument-type]

        with (
            patch.object(browse, "Runner", runner),
            patch.object(browse, "save_final_screenshot", AsyncMock()),
            patch.object(mcp_server, "build_model", return_value=ScriptedModel(choose="a")),
        ):
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                with patch.object(agents, "run_by_name", browser_run):
                    request_id = session._request_id
                    first = asyncio.create_task(session.call_tool("run_agent", {"name": "flights", "flags": []}))
                    await asyncio.wait_for(started.wait(), 5)
                    await _cancel(session, request_id)
                    with contextlib.suppress(McpError):
                        await asyncio.wait_for(first, 5)
                    for _ in range(100):
                        if not mcp_server._ONE_RUN.locked():
                            break
                        await asyncio.sleep(0.01)
                self.assertFalse(mcp_server._ONE_RUN.locked())
                after = await asyncio.wait_for(
                    session.call_tool("decide", {"state": {}, "options": {"a": "a"}, "rules": "pick"}), 5
                )
        self.assertEqual(torn_down, ["browser", "session"])
        self.assertFalse(after.isError, after.content)
        self.assertEqual(_rows(after)["choice"], "a")

    async def test_a_cancelled_series_writes_its_job_as_cancelled(self) -> None:
        real, blocked = loop.run_episode, asyncio.Event()

        async def second_blocks(*args: Any, **kwargs: Any) -> Any:
            if kwargs["seed"] > 0:
                blocked.set()
                await asyncio.Event().wait()
            return await real(*args, **kwargs)

        parsed = series.parser(COUNTER).parse_args([*RANDOM_RUN[:5], "3"])
        with patch.object(series, "run_episode", second_blocks):
            task = asyncio.create_task(series.play(COUNTER, parsed, results_dir=self.tmp / "results"))
            await asyncio.wait_for(blocked.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        (job_dir,) = (self.tmp / "results" / COUNTER.name).iterdir()
        job = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
        self.assertTrue(job["cancelled"])
        self.assertEqual((job["n_total_trials"], job["stats"]["n_cancelled_trials"]), (3, 2))

    async def test_a_cancelled_browser_play_writes_a_cancelled_answer(self) -> None:
        async def blocks(*args: Any, **kwargs: Any) -> Any:
            await asyncio.Event().wait()

        args = SimpleNamespace(
            profile_out=None, model="llm", goal="g", timeout=60.0, max_steps=5, logs_dir=self.tmp / "logs", headed=False
        )
        with (
            patch.object(browse, "chat_model_from_env", lambda: None),
            patch.object(browse, "policy_from_args", lambda args: None),
            patch.object(browse, "browse", blocks),
        ):
            task = asyncio.create_task(browse.play(FLIGHTS, args))  # ty: ignore[invalid-argument-type]
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        answer = json.loads((self.tmp / "logs" / "answer.json").read_text(encoding="utf-8"))
        self.assertEqual((answer["ok"], answer["cancelled"], answer["model"]), (False, True, "llm"))

    async def test_a_cancel_before_any_episode_still_writes_a_cancelled_job(self) -> None:
        """Cancel during the first episode: no finished trial, but every planned seed is recorded."""
        real, blocked = series.run_episode, asyncio.Event()

        async def blocks_first(*args: Any, **kwargs: Any) -> Any:
            blocked.set()
            await asyncio.Event().wait()
            return await real(*args, **kwargs)

        parsed = series.parser(COUNTER).parse_args([*RANDOM_RUN[:5], "3"])
        with patch.object(series, "run_episode", blocks_first):
            task = asyncio.create_task(series.play(COUNTER, parsed, results_dir=self.tmp / "results"))
            await asyncio.wait_for(blocked.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        (job_dir,) = (self.tmp / "results" / COUNTER.name).iterdir()
        job = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
        self.assertTrue(job["cancelled"])
        self.assertEqual((job["n_total_trials"], job["stats"]["n_cancelled_trials"]), (3, 3))
        self.assertEqual(job["stats"]["n_completed_trials"], 0)
        config = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["datasets"], [{"name": COUNTER.name, "n_tasks": 3}])


class TestSdkBound(TestCase):
    """mcp stays on 1.x until openjiuwen's fastmcp allows 2.x; docs/architecture.md § MCP SDK 2.x tracks the port."""

    NOTE = "### MCP SDK 2.x"
    ONE_X_ONLY = re.compile(r"^\s*from mcp(?:\.server\.fastmcp|\.client\.stdio|\.shared\.memory)? import ", re.M)

    def test_the_declared_bound_and_the_installed_sdk_stay_below_2(self) -> None:
        declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
        self.assertIn("mcp>=1.26,<2", declared)
        self.assertEqual(version("mcp").split(".")[0], "1", version("mcp"))

    def test_the_migration_note_names_every_file_importing_1x_only_names(self) -> None:
        architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
        self.assertIn(self.NOTE, architecture)
        note = architecture.split(self.NOTE, 1)[1]
        sources = [*ROOT.glob("s1a/**/*.py"), *ROOT.glob("tests/**/*.py"), *ROOT.glob("scripts/*.sh")]
        importing = sorted(
            path.relative_to(ROOT).as_posix()
            for path in sources
            if self.ONE_X_ONLY.search(path.read_text(encoding="utf-8"))
        )
        self.assertIn("s1a/mcp_server.py", importing)
        self.assertEqual([name for name in importing if f"`{name}`" not in note], [])


class TestAdvertisedTools(IsolatedAsyncioTestCase):
    """What ``tools/list`` tells a host: titles, behavior hints and typed result schemas (#82)."""

    async def _tools(self) -> dict[str, Any]:
        async with create_connected_server_and_client_session(mcp_server.server) as session:
            listed = await session.list_tools()
        return {tool.name: tool for tool in listed.tools}

    async def test_each_tool_has_a_title_and_its_behavior_hints(self) -> None:
        tools = await self._tools()
        expected = {  # name: (title, readOnly, destructive, idempotent, openWorld)
            "list_agents": ("List S1A agents", True, None, True, False),
            "decide": ("Decide one choice", True, None, None, True),
            "run_agent": ("Run an S1A agent", False, True, False, True),
        }
        for name, (title, read_only, destructive, idempotent, open_world) in expected.items():
            with self.subTest(tool=name):
                hints = tools[name].annotations
                self.assertEqual(tools[name].title, title)
                self.assertIsNotNone(hints)
                self.assertEqual(
                    (hints.readOnlyHint, hints.destructiveHint, hints.idempotentHint, hints.openWorldHint),
                    (read_only, destructive, idempotent, open_world),
                )

    async def test_decide_advertises_its_four_keys_and_its_result_validates(self) -> None:
        schema = (await self._tools())["decide"].outputSchema
        self.assertIsNotNone(schema)
        self.assertEqual(sorted(schema["required"]), ["choice", "confidence", "ms", "probabilities"])
        with patch.object(mcp_server, "build_model", return_value=JevModel(ScriptedTransport(choose="inc"))):
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                result = await session.call_tool(
                    "decide", {"state": {"n": 1}, "options": {"inc": "add one", "dec": "take one"}, "rules": "count"}
                )
        self.assertFalse(result.isError)
        jsonschema.validate(result.structuredContent, schema)
        self.assertEqual(json.loads(result.content[0].text), result.structuredContent)

    async def test_list_agents_rows_are_typed_and_an_unavailable_row_still_validates(self) -> None:
        schema = (await self._tools())["list_agents"].outputSchema
        self.assertIsNotNone(schema)
        row = schema["properties"]["result"]["items"]
        row = schema["$defs"][row["$ref"].rsplit("/", 1)[-1]] if "$ref" in row else row
        self.assertEqual(sorted(row["required"]), ["description", "front", "name"])
        self.assertIn("flags", row["properties"])
        real = agents.load

        def load(name: str) -> agents.Spec:
            if name == "alfworld":
                raise FileNotFoundError("ALFWORLD_DATA is empty")
            return real(name)

        with patch.object(agents, "load", load):
            async with create_connected_server_and_client_session(mcp_server.server) as session:
                result = await session.call_tool("list_agents", {})
        self.assertFalse(result.isError)
        jsonschema.validate(result.structuredContent, schema)
        rows = {row["name"]: row for row in _rows(result)}
        self.assertNotIn("flags", rows["alfworld"])
        self.assertIn("flags", rows["game2048"])

    async def test_run_agent_stays_an_untyped_object(self) -> None:
        schema = (await self._tools())["run_agent"].outputSchema
        self.assertEqual((schema or {}).get("type"), "object")
        self.assertNotIn("required", schema or {})
