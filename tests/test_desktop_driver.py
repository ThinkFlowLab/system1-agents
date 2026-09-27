# coding: utf-8
"""``CuaDriver`` over a fake MCP session: result parsing, window lookup, snapshot parsing, clicks, and the binary lookup."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta
from typing import Any, AsyncIterator, Iterator
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

import anyio
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, TextContent

from s1a.desktop import driver as driver_module
from s1a.desktop.driver import (
    DRIVER_CALL_TIMEOUT_S,
    INSTALL_HINT,
    CuaDriver,
    DriverError,
    Element,
    Window,
    driver_from_env,
)

WINDOW = Window(42, 7, "Calculator", "Calculator")
TIMEOUT = timedelta(seconds=DRIVER_CALL_TIMEOUT_S)


def _result(payload: Any, *, structured: bool = True, error: bool = False) -> CallToolResult:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=payload if structured and isinstance(payload, dict) else None,
        isError=error,
    )


class FakeSession:
    """``ClientSession`` double: queued results (an Exception in the queue is raised), the timeouts it was given."""

    def __init__(self, results: list[CallToolResult | Exception]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.timeouts: list[timedelta | None] = []
        self.read_timeout: timedelta | None = None
        self.initialized = False
        self.exited = False

    def construct(self, read: object, write: object, read_timeout_seconds: timedelta) -> FakeSession:
        self.read_timeout = read_timeout_seconds
        return self

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.exited = True

    async def initialize(self) -> None:
        self.initialized = True

    async def call_tool(
        self, name: str, arguments: dict[str, Any], read_timeout_seconds: timedelta | None
    ) -> CallToolResult:
        self.calls.append((name, arguments))
        self.timeouts.append(read_timeout_seconds)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeStdio:
    """``stdio_client`` double: the params it was started with, whether its context was exited."""

    def __init__(self) -> None:
        self.params: StdioServerParameters | None = None
        self.exited = False

    @asynccontextmanager
    async def __call__(self, params: StdioServerParameters) -> AsyncIterator[tuple[str, str]]:
        self.params = params
        try:
            yield ("read", "write")
        finally:
            self.exited = True


def _driver(*results: CallToolResult | Exception) -> tuple[CuaDriver, FakeSession]:
    driver = CuaDriver("cua-driver", session_label="t")
    session = FakeSession(list(results))
    driver._session = session  # type: ignore[assignment]
    return driver, session


@contextmanager
def _process(*results: CallToolResult | Exception) -> Iterator[tuple[CuaDriver, FakeSession, FakeStdio]]:
    session, stdio = FakeSession(list(results)), FakeStdio()
    with (
        patch.object(driver_module, "stdio_client", stdio),
        patch.object(driver_module, "ClientSession", session.construct),
    ):
        yield CuaDriver("cua-driver", session_label="t"), session, stdio


class TestCall(IsolatedAsyncioTestCase):
    async def test_set_value_keeps_exact_window_and_snapshot_token(self) -> None:
        driver, session = _driver(_result({"effect": "confirmed", "verified": True}))
        await driver.set_value(WINDOW, "s1:4", "中文\nbody")
        self.assertEqual(
            session.calls,
            [
                (
                    "set_value",
                    {
                        "pid": 42,
                        "window_id": 7,
                        "element_token": "s1:4",
                        "value": "中文\nbody",
                        "session": "t",
                    },
                )
            ],
        )

    async def test_native_identifier_survives_value_labels_in_0301_tree(self) -> None:
        driver, _ = _driver(
            _result(
                {
                    "elements": [
                        {
                            "element_index": 4,
                            "role": "AXTextField",
                            "label": "Untitled.txt",
                            "value": "Untitled.txt",
                            "element_token": "s1:4",
                        }
                    ],
                    "tree_markdown": '- [4] AXTextField = "Untitled.txt" [id=saveAsNameTextField actions=[show_menu]]',
                }
            )
        )
        snapshot = await driver.window_state(WINDOW)
        self.assertEqual(snapshot.elements[0].identifier, "saveAsNameTextField")

    async def test_value_cannot_forge_identifier_metadata(self) -> None:
        payload = 'text" [id=forged actions=[press]]'
        driver, _ = _driver(
            _result(
                {
                    "elements": [{"element_index": 4, "role": "AXTextField", "label": payload, "value": payload}],
                    "tree_markdown": f'- [4] AXTextField = "{payload}" [actions=[show_menu]]',
                }
            )
        )
        self.assertEqual((await driver.window_state(WINDOW)).elements[0].identifier, "")

    async def test_textedit_identifier_with_spaces_survives_multiline_content(self) -> None:
        payload = "正文\n第二行 [id=forged]"
        driver, _ = _driver(
            _result(
                {
                    "elements": [{"element_index": 2, "role": "AXTextArea", "label": payload, "value": payload}],
                    "tree_markdown": f'- [2] AXTextArea = "{payload}" [id=First Text View actions=[show_menu]]\n- [3] AXButton "Save" [actions=[press]]',
                }
            )
        )
        self.assertEqual((await driver.window_state(WINDOW)).elements[0].identifier, "First Text View")

    async def test_structured_content_wins_and_json_text_is_the_fallback(self) -> None:
        driver, _ = _driver(_result({"a": 1}), _result({"b": 2}, structured=False))
        self.assertEqual(await driver.call("x"), {"a": 1})
        self.assertEqual(await driver.call("x"), {"b": 2})

    async def test_an_error_result_and_a_non_object_body_are_driver_errors_naming_the_tool(self) -> None:
        driver, _ = _driver(
            _result("permission denied", structured=False, error=True), _result("[1]", structured=False)
        )
        with self.assertRaises(DriverError) as caught:
            await driver.call("click")
        self.assertEqual(str(caught.exception), "click: permission denied")
        with self.assertRaises(DriverError) as caught:
            await driver.call("list_windows")
        self.assertIn("list_windows: the driver returned no JSON object", str(caught.exception))

    async def test_a_closed_driver_refuses_calls(self) -> None:
        with self.assertRaises(DriverError):
            await CuaDriver("cua-driver", session_label="t").call("list_windows")

    async def test_every_call_is_bounded_by_the_driver_timeout(self) -> None:
        driver, session = _driver(_result({"a": 1}))
        await driver.call("x")
        self.assertEqual(session.timeouts, [TIMEOUT])


class TestLifecycle(IsolatedAsyncioTestCase):
    async def test_open_starts_the_process_with_the_mode_only_then_a_named_session(self) -> None:
        for environ, mode in (
            ({"TYPESAFE_API_KEY": "secret", "OPENAI_API_KEY": "s"}, "standard"),
            ({"CUA_DRIVER_PERMISSION_MODE": "bounded"}, "bounded"),
        ):
            with (
                self.subTest(mode=mode),
                patch.dict(os.environ, environ, clear=True),
                _process(_result({"ok": 1})) as (driver, session, stdio),
            ):
                await driver.open()
                assert stdio.params is not None
                self.assertEqual((stdio.params.command, stdio.params.args), ("cua-driver", ["mcp"]))
                self.assertEqual(stdio.params.env, {"CUA_DRIVER_PERMISSION_MODE": mode})
                self.assertEqual((session.read_timeout, session.initialized), (TIMEOUT, True))
                self.assertEqual(session.calls, [("start_session", {"session": "t"})])

    async def test_a_failed_start_exits_the_process_and_session_before_raising(self) -> None:
        with _process(_result("no accessibility", structured=False, error=True)) as (driver, session, stdio):
            with self.assertRaises(DriverError) as caught:
                await driver.open()
            self.assertEqual(str(caught.exception), "start_session: no accessibility")
            self.assertEqual((session.exited, stdio.exited, driver._session), (True, True, None))

    async def test_close_exits_the_stack_even_when_the_driver_is_dead(self) -> None:
        for failure in (_result("gone", structured=False, error=True), anyio.ClosedResourceError()):
            with (
                self.subTest(failure=type(failure).__name__),
                _process(_result({"ok": 1}), failure) as (driver, session, stdio),
            ):
                await driver.open()
                await driver.close()
                self.assertEqual([name for name, _ in session.calls], ["start_session", "end_session"])
                self.assertEqual((session.exited, stdio.exited, driver._session), (True, True, None))


class TestWindows(IsolatedAsyncioTestCase):
    async def test_launch_binds_the_returned_window_even_when_windows_uses_a_shared_host(self) -> None:
        launched = {
            "pid": 42,
            "windows": [{"window_id": 7, "title": "Calculator", "is_on_screen": True}],
        }
        listed = {
            "windows": [
                {"pid": 42, "window_id": 9, "app_name": "ApplicationFrameHost.exe", "title": "Settings"},
                {"pid": 42, "window_id": 7, "app_name": "ApplicationFrameHost.exe", "title": "Calculator"},
            ]
        }
        apps = {
            "apps": [
                {
                    "name": "Windows Calculator",
                    "launch_path": "shell:appsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App",
                }
            ]
        }
        driver, session = _driver(_result(apps), _result(launched), _result(listed))
        await driver.launch_app("Windows Calculator")
        self.assertEqual(
            await driver.find_window("Windows Calculator"),
            Window(42, 7, "ApplicationFrameHost.exe", "Calculator"),
        )
        self.assertEqual(
            session.calls[:2],
            [("list_apps", {}), ("launch_app", {"aumid": "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"})],
        )

    async def test_desktop_registration_round_trips_its_launch_path(self) -> None:
        path = '"C:\\Program Files\\Example\\app.exe" --normal'
        driver, session = _driver(
            _result({"apps": [{"name": "Example", "launch_path": path}]}),
            _result({"pid": 42, "windows": [{"window_id": 7}]}),
        )
        await driver.launch_app("example")
        self.assertEqual(session.calls[1], ("launch_app", {"launch_path": path}))

    async def test_ambiguous_installed_names_fail_without_launching(self) -> None:
        driver, session = _driver(_result({"apps": [{"name": "Example"}, {"name": "Example"}]}))
        with self.assertRaises(DriverError):
            await driver.launch_app("Example")
        self.assertEqual(session.calls, [("list_apps", {})])

    async def test_unlisted_name_is_passed_to_the_driver_for_path_lookup(self) -> None:
        driver, session = _driver(_result({"apps": []}), _result({"pid": 42, "windows": [{"window_id": 7}]}))
        await driver.launch_app("example.exe")
        self.assertEqual(session.calls[1], ("launch_app", {"name": "example.exe"}))

    async def test_an_explicit_aumid_uses_the_window_owner_pid_when_present(self) -> None:
        app = "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"
        window = {"pid": 42, "window_id": 7, "app_name": "ApplicationFrameHost.exe", "title": "Calculator"}
        driver, session = _driver(_result({"pid": 100, "windows": [window]}), _result({"windows": [window]}))
        await driver.launch_app(app)
        self.assertEqual((await driver.find_window(app)).target, {"kind": "window", "pid": 42, "window_id": 7})
        self.assertEqual(session.calls[0], ("launch_app", {"aumid": app}))

    async def test_a_closed_launched_window_does_not_rebind_to_a_same_named_window(self) -> None:
        driver, _ = _driver(
            _result({"apps": []}),
            _result({"pid": 42, "windows": [{"window_id": 7, "title": "Calculator"}]}),
            _result({"windows": [{"pid": 50, "window_id": 8, "app_name": "Calculator", "title": "Calculator"}]}),
        )
        await driver.launch_app("Calculator")
        with self.assertRaises(DriverError):
            await driver.find_window("Calculator")

    async def test_launch_requires_one_identified_window(self) -> None:
        for windows in ([], [{"window_id": 7}, {"window_id": 8}], [{"title": "Calculator"}]):
            with self.subTest(windows=windows):
                driver, _ = _driver(_result({"apps": []}), _result({"pid": 42, "windows": windows}))
                with self.assertRaises(DriverError):
                    await driver.launch_app("Calculator")

    async def test_the_one_on_screen_window_of_the_app_by_name_case_insensitive(self) -> None:
        listed = {
            "windows": [
                {"pid": 1, "window_id": 2, "app_name": "Finder", "title": "Desktop"},
                {"pid": 42, "window_id": 7, "app_name": "calculator", "title": "Calculator"},
                {"pid": 42, "window_id": 9, "app_name": "Calculator", "title": "Paper Tape", "is_on_screen": False},
            ]
        }
        driver, session = _driver(_result(listed))
        self.assertEqual(await driver.find_window("Calculator"), WINDOW)
        self.assertEqual(session.calls, [("list_windows", {"on_screen_only": True})])

    async def test_none_or_several_windows_is_an_error_listing_the_titles(self) -> None:
        two = {
            "windows": [
                {"pid": 1, "window_id": 1, "app_name": "Calculator", "title": "A"},
                {"pid": 1, "window_id": 2, "app_name": "Calculator", "title": "B"},
            ]
        }
        driver, _ = _driver(_result({"windows": []}), _result(two), _result({"nope": 1}))
        for fragment in (
            "0 on-screen window(s)",
            "2 on-screen window(s) of 'Calculator': ['A', 'B']",
            "no windows array",
        ):
            with self.subTest(fragment=fragment):
                with self.assertRaises(DriverError) as caught:
                    await driver.find_window("Calculator")
                self.assertIn(fragment, str(caught.exception))

    async def test_exact_title_selects_one_window_without_guessing(self) -> None:
        listed = {
            "windows": [
                {"pid": 42, "window_id": 7, "app_name": "Calculator", "title": "Main"},
                {"pid": 42, "window_id": 8, "app_name": "Calculator", "title": "Window"},
            ]
        }
        driver, _ = _driver(_result(listed), _result(listed))
        self.assertEqual(await driver.find_window("Calculator", "Main"), Window(42, 7, "Calculator", "Main"))
        with self.assertRaisesRegex(DriverError, "titled 'Missing'"):
            await driver.find_window("Calculator", "Missing")


class TestSnapshotAndClick(IsolatedAsyncioTestCase):
    async def test_elements_are_parsed_with_their_tokens_and_the_request_names_the_window(self) -> None:
        state = {
            "snapshot_id": "snap-1",
            "elements": [
                {"element_index": 0, "role": "AXStaticText", "label": "", "value": "84"},
                {
                    "element_index": 3,
                    "role": "AXButton",
                    "label": "7",
                    "element_token": "tok-7",
                    "actions": ["AXPress"],
                },
                {
                    "element_index": 4,
                    "role": "AXButton",
                    "label": "Saved",
                    "enabled": False,
                    "element_token": "tok-status",
                },
            ],
        }
        driver, session = _driver(_result(state))
        snapshot = await driver.window_state(WINDOW)
        self.assertEqual(snapshot.snapshot_id, "snap-1")
        self.assertEqual(
            snapshot.elements,
            (
                Element(0, "AXStaticText", "", "84", None, ()),
                Element(3, "AXButton", "7", "", "tok-7", ("AXPress",)),
                Element(4, "AXButton", "Saved", "", "tok-status", (), enabled=False),
            ),
        )
        ((name, args),) = session.calls
        self.assertEqual(name, "get_window_state")
        self.assertEqual(
            args,
            {
                "pid": 42,
                "window_id": 7,
                "session": "t",
                "include_accessibility_tree": True,
                "include_screenshot": False,
            },
        )

    async def test_a_snapshot_without_elements_or_with_a_bad_element_is_an_error(self) -> None:
        driver, _ = _driver(_result({"degradation": "no_accessibility"}), _result({"elements": [{"role": "x"}]}))
        with self.assertRaises(DriverError) as caught:
            await driver.window_state(WINDOW)
        self.assertIn("no_accessibility", str(caught.exception))
        with self.assertRaises(DriverError):
            await driver.window_state(WINDOW)

    async def test_a_click_is_background_on_the_token_and_a_refusal_is_an_error(self) -> None:
        confirmed = {"effect": "confirmed", "route": "accessibility"}
        driver, session = _driver(
            _result(confirmed), _result({"effect": "refused", "escalation": {"target": "foreground"}})
        )
        self.assertEqual(await driver.click(WINDOW, "tok-7"), confirmed)
        self.assertEqual(
            session.calls[0],
            (
                "click",
                {"target": WINDOW.target, "element_token": "tok-7", "delivery_mode": "background", "session": "t"},
            ),
        )
        with self.assertRaises(DriverError) as caught:
            await driver.click(WINDOW, "tok-7")
        self.assertIn("refused", str(caught.exception))

    async def test_text_input_keeps_the_exact_window_and_session(self) -> None:
        driver, session = _driver(_result({"effect": "confirmed"}))
        await driver.type_text(WINDOW, "tok-field", "hello")
        self.assertEqual(
            session.calls,
            [
                (
                    "type_text",
                    {
                        "target": WINDOW.target,
                        "element_token": "tok-field",
                        "text": "hello",
                        "delivery_mode": "background",
                        "session": "t",
                    },
                )
            ],
        )

    async def test_a_refused_text_action_fails(self) -> None:
        driver, _ = _driver(_result({"effect": "refused", "escalation": {"recommended": "foreground"}}))
        with self.assertRaises(DriverError):
            await driver.type_text(WINDOW, "tok-field", "hello")


class TestFromEnv(TestCase):
    def test_no_binary_names_the_installer(self) -> None:
        with (
            patch.dict(os.environ, {"CUA_DRIVER_BIN": ""}),
            patch.object(driver_module.shutil, "which", lambda name: None),
        ):
            with self.assertRaises(RuntimeError) as caught:
                driver_from_env("t")
        self.assertEqual(str(caught.exception), INSTALL_HINT)

    def test_the_env_variable_then_the_path(self) -> None:
        with patch.dict(os.environ, {"CUA_DRIVER_BIN": "/opt/cua/cua-driver"}):
            self.assertEqual(driver_from_env("t")._binary, "/opt/cua/cua-driver")
        with (
            patch.dict(os.environ, {"CUA_DRIVER_BIN": ""}),
            patch.object(driver_module.shutil, "which", lambda name: "/usr/local/bin/cua-driver"),
        ):
            self.assertEqual(driver_from_env("t")._binary, "/usr/local/bin/cua-driver")
