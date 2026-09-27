# coding: utf-8
"""Cua Driver over MCP stdio: one process per series, exact windows, snapshot-bound element actions.

The driver runs in ``standard`` permission mode; every action names the pid and window id the env was bound to.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, AsyncIterator, Protocol

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

Json = dict[str, Any]
INSTALL_HINT = (
    "cua-driver is not on PATH (or CUA_DRIVER_BIN): install it from https://cua.ai/driver "
    "for your platform; on macOS grant Accessibility and Screen Recording in System Settings"
)
# ponytail: standard mode and a pinned window; a bounded capability manifest (CUA_DRIVER_PERMISSION_MODE=bounded) when
# the agent must be kept out of every other app by the driver itself
_MODE_ENV = "CUA_DRIVER_PERMISSION_MODE"
DRIVER_CALL_TIMEOUT_S = 30.0  # a hung driver fails the call; reset() runs before the episode's own wait_for


class DriverError(RuntimeError):
    """The driver refused or failed a tool; the message names the tool and quotes the driver."""


@dataclass(frozen=True)
class Window:
    pid: int
    window_id: int
    app_name: str
    title: str

    @property
    def target(self) -> Json:
        return {"kind": "window", "pid": self.pid, "window_id": self.window_id}


@dataclass(frozen=True)
class Element:
    """One accessibility element of a snapshot; ``token`` is what an action names, valid for that snapshot only."""

    index: int
    role: str
    label: str
    value: str
    token: str | None
    actions: tuple[str, ...]
    enabled: bool = True
    identifier: str = ""


@dataclass(frozen=True)
class Snapshot:
    window: Window
    snapshot_id: str | None
    elements: tuple[Element, ...]
    raw: Json = field(repr=False, compare=False)


class Driver(Protocol):
    """Window-scoped observation and input used by the desktop environment."""

    async def find_window(self, app_name: str, window_title: str = "") -> Window: ...
    async def window_state(self, window: Window) -> Snapshot: ...
    async def click(self, window: Window, token: str) -> Json: ...
    async def type_text(self, window: Window, token: str, text: str) -> Json: ...
    async def set_value(self, window: Window, token: str, text: str) -> Json: ...


class CuaDriver:
    """The MCP client behind ``Driver``; ``open`` starts the process and a named session, ``close`` ends both."""

    def __init__(self, binary: str, *, session_label: str) -> None:
        self._binary = binary
        self._label = session_label
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._launched_windows: dict[str, Window] = {}

    async def open(self) -> None:
        env = {_MODE_ENV: os.environ.get(_MODE_ENV, "standard")}  # the driver gets no API keys
        params = StdioServerParameters(command=self._binary, args=["mcp"], env=env)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        try:
            timeout = timedelta(seconds=DRIVER_CALL_TIMEOUT_S)
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timeout)
            )
            await self._session.initialize()
            await self.call("start_session", session=self._label)
        except BaseException:
            await self._stack.aclose()
            self._session = None
            raise

    async def close(self) -> None:
        try:
            if self._session is not None:
                await self.call("end_session", session=self._label)
        except Exception:
            pass  # the session is gone either way; the process exits with the stack
        finally:
            await self._stack.aclose()
            self._session = None
            self._launched_windows.clear()

    async def call(self, tool: str, **args: Any) -> Json:
        """One tool call; the driver's structured result, or its JSON text, as a dict."""
        if self._session is None:
            raise DriverError(f"{tool}: the driver is not open")
        result = await self._session.call_tool(
            tool, args, read_timeout_seconds=timedelta(seconds=DRIVER_CALL_TIMEOUT_S)
        )
        text = " ".join(part.text for part in result.content if isinstance(part, TextContent))
        if result.isError:
            raise DriverError(f"{tool}: {text or 'the driver returned an error'}")
        if isinstance(result.structuredContent, dict):
            return result.structuredContent
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise DriverError(f"{tool}: the driver returned no JSON object: {text[:200]!r}") from exc
        if not isinstance(payload, dict):
            raise DriverError(f"{tool}: the driver returned no JSON object: {text[:200]!r}")
        return payload

    async def launch_app(self, app_name: str) -> None:
        """Launch through the driver and pin the returned window, including Windows shared-host apps."""
        launched = await self.call("launch_app", **await self._launch_target(app_name))
        windows = launched.get("windows")
        if not isinstance(windows, list):
            raise DriverError(f"launch_app: no windows array for {app_name!r}")
        visible = [w for w in windows if isinstance(w, dict) and w.get("is_on_screen", True)]
        if len(visible) != 1:
            raise DriverError(f"launch_app: expected one on-screen window of {app_name!r}, got {len(visible)}")
        window = visible[0]
        try:
            pid = int(window.get("pid", launched.get("pid")))
            window_id = int(window["window_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DriverError(f"launch_app: no identified window for {app_name!r}") from exc
        self._launched_windows[app_name.casefold()] = Window(pid, window_id, app_name, str(window.get("title") or ""))

    async def _launch_target(self, app_name: str) -> Json:
        if "!" in app_name:
            return {"aumid": app_name}
        listed = await self.call("list_apps")
        apps = listed.get("apps")
        if not isinstance(apps, list):
            raise DriverError("list_apps: no apps array")
        matches = [app for app in apps if str(app.get("name") or "").casefold() == app_name.casefold()]
        if len(matches) > 1:
            raise DriverError(f"list_apps: {len(matches)} apps named {app_name!r}; use an explicit AUMID")
        path = str(matches[0].get("launch_path") or "") if matches else ""
        prefix = "shell:appsFolder\\"
        if path.casefold().startswith(prefix.casefold()):
            # list_apps' friendly name can differ from the shell index used by launch_app(name).
            return {"aumid": path[len(prefix) :]}
        return {"launch_path": path} if path else {"name": app_name}

    async def find_window(self, app_name: str, window_title: str = "") -> Window:
        """Find one on-screen app window, using an exact title when several exist."""
        listed = await self.call("list_windows", on_screen_only=True)
        windows = listed.get("windows")
        if not isinstance(windows, list):
            raise DriverError(f"list_windows: no windows array in {str(listed)[:200]!r}")
        wanted = app_name.casefold()
        pinned = self._launched_windows.get(wanted)
        if pinned is not None:
            matches = [
                w
                for w in windows
                if w.get("pid") == pinned.pid and w.get("window_id") == pinned.window_id and w.get("is_on_screen", True)
            ]
        else:
            matches = [
                w for w in windows if str(w.get("app_name") or "").casefold() == wanted and w.get("is_on_screen", True)
            ]
        if window_title:
            matches = [w for w in matches if str(w.get("title") or "") == window_title]
        if len(matches) != 1:
            titles = [str(w.get("title")) for w in matches]
            choice = f" titled {window_title!r}" if window_title else ""
            raise DriverError(f"list_windows: {len(matches)} on-screen window(s) of {app_name!r}{choice}: {titles}")
        window = matches[0]
        owner = str(window.get("app_name") or app_name) if pinned is not None else app_name
        return Window(int(window["pid"]), int(window["window_id"]), owner, str(window.get("title") or ""))

    async def window_state(self, window: Window) -> Snapshot:
        state = await self.call(
            "get_window_state",
            pid=window.pid,
            window_id=window.window_id,
            session=self._label,
            include_accessibility_tree=True,
            include_screenshot=False,
        )
        raw_elements = state.get("elements")
        if str(state.get("degraded_reason", "")).startswith("ax_window_unresolved"):
            raise DriverError(f"get_window_state: {state['degraded_reason']}")
        if not isinstance(raw_elements, list):
            raise DriverError(f"get_window_state: no elements in the snapshot ({state.get('degradation')!r})")
        elements = tuple(_element(raw, tree=str(state.get("tree_markdown") or "")) for raw in raw_elements)
        snapshot_id = state.get("snapshot_id")
        return Snapshot(window, str(snapshot_id) if snapshot_id else None, elements, state)

    async def click(self, window: Window, token: str) -> Json:
        """Click a snapshot-bound element (background by default); a refused action is an error."""
        return await self._input(
            "click", target=window.target, element_token=token, delivery_mode="background", session=self._label
        )

    async def type_text(self, window: Window, token: str, text: str) -> Json:
        """Insert task-supplied text into one element from the latest snapshot."""
        return await self._input(
            "type_text",
            target=window.target,
            element_token=token,
            text=text,
            delivery_mode="background",
            session=self._label,
        )

    async def set_value(self, window: Window, token: str, text: str) -> Json:
        """Replace a native field through AXValue, with the driver's exact-window admission and readback."""
        return await self._input(
            "set_value",
            pid=window.pid,
            window_id=window.window_id,
            element_token=token,
            value=text,
            session=self._label,
        )

    async def _input(self, tool: str, **args: Any) -> Json:
        result = await self.call(tool, **args)
        if result.get("effect") == "refused":
            raise DriverError(f"{tool}: refused ({result.get('escalation')!r})")
        return result


def _tree_identifier(raw: Json, tree: str) -> str:
    """Cua Driver 0.30.1 omits AXIdentifier from JSON but includes it in indexed tree rows.

    Consume the known label/value first: text inside a field must never be parsed as metadata.
    Ambiguous or unfamiliar rows get no fallback identifier. New drivers can supply it directly.
    """
    label, value = str(raw.get("label") or ""), str(raw.get("value") or "")
    prefix = rf"^[ \t]*- \[{int(raw['element_index'])}\] {re.escape(str(raw.get('role') or ''))}(?= |$)"
    rows = list(re.finditer(prefix, tree, re.MULTILINE))
    if len(rows) != 1:
        return ""
    rest = tree[rows[0].end() :]
    if label and rest.startswith(f' "{label}"'):
        rest = rest[len(label) + 3 :]
    if rest.startswith(f' = "{value}"'):
        rest = rest[len(value) + 5 :]
    elif value:
        return ""
    if label and rest.startswith(f" ({label})"):
        rest = rest[len(label) + 3 :]
    rest = rest.split("\n", 1)[0]
    match = re.fullmatch(r' \[id=([^\[\]\r\n]+?)(?: help="[^"\n]*")?(?: actions=\[[\w, ]*\])?\]', rest)
    return match[1] if match else ""


def _element(raw: Any, *, tree: str = "") -> Element:
    if not isinstance(raw, dict) or "element_index" not in raw:
        raise DriverError(f"get_window_state: an element without element_index: {str(raw)[:120]!r}")
    token = raw.get("element_token")
    return Element(
        index=int(raw["element_index"]),
        role=str(raw.get("role") or ""),
        label=str(raw.get("label") or ""),
        identifier=str(raw.get("identifier") or "") or _tree_identifier(raw, tree),
        value=str(raw.get("value") or ""),
        token=str(token) if token else None,
        actions=tuple(str(action) for action in raw.get("actions") or ()),
        enabled=raw.get("enabled") is not False,
    )


def driver_from_env(session_label: str) -> CuaDriver:
    """``CUA_DRIVER_BIN`` or ``cua-driver`` on PATH; a missing binary names the installer."""
    binary = os.getenv("CUA_DRIVER_BIN") or shutil.which("cua-driver")
    if not binary:
        raise RuntimeError(INSTALL_HINT)
    return CuaDriver(binary, session_label=session_label)


@asynccontextmanager
async def opened(driver: CuaDriver) -> AsyncIterator[CuaDriver]:
    await driver.open()
    try:
        yield driver
    finally:
        await driver.close()
