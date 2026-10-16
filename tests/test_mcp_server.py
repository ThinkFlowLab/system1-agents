# coding: utf-8
"""The MCP server over an in-memory session: the three tools, decide against a fake client, and the agent listing."""

from __future__ import annotations

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
from unittest.mock import AsyncMock, patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from s1a import mcp_server
from s1a import run as agents
from s1a.decision_models import JevModel, ScriptedModel, ScriptedTransport
from s1a.tool import loop, series
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
