# coding: utf-8
"""Desktop input actions through a snapshot-bound fake document window."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from s1a.agents import desktop
from s1a.decision_models import Image
from s1a.desktop.driver import Capture, DriverError, Element, Snapshot, Window
from s1a.desktop.env import WindowEnv
from s1a.run import started_runner
from s1a.tool import loop, series
from test_desktop_visual import _png


class FakeDocument:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.window = Window(91, 3, "Document", "Document")
        self.snapshots = 0
        self.text = ""
        self.status = "Editing"
        self.actions: list[tuple[str, Any]] = []
        self.opened = 0

    async def open(self) -> None:
        self.opened += 1

    async def close(self) -> None:
        self.opened -= 1

    async def find_window(self, app_name: str) -> Window:
        assert app_name == "Document"
        return self.window

    async def window_state(self, window: Window, *, screenshot: bool = False) -> Snapshot:
        assert window == self.window
        self.snapshots += 1
        n = self.snapshots
        return Snapshot(
            window,
            f"snap-{n}",
            (
                Element(1, "AXTextArea", "Body", self.text, f"body-{n}", ("AXSetValue",)),
                Element(2, "AXButton", "Save", "", f"save-{n}", ("AXPress",)),
                Element(3, "AXStaticText", "Status", self.status, None, ()),
            ),
            {},
            capture=Capture(f"capture-{n}", Image(_png((1, 2, 3), stamp=str(n))), 800, 600) if screenshot else None,
        )

    async def click(self, window: Window, token: str) -> dict[str, Any]:
        self._check(window, token)
        self.actions.append(("click", token))
        if token.startswith("save-"):
            self.path.write_text(self.text, encoding="utf-8")
            self.status = "Saved"
        return {"effect": "confirmed"}

    async def type_text(self, window: Window, token: str, text: str) -> dict[str, Any]:
        self._check(window, token)
        self.actions.append(("type", token))
        self.text = text
        return {"effect": "confirmed"}

    async def click_at(self, window: Window, capture: Capture, x: float, y: float) -> dict[str, Any]:
        self._check(window, capture.capture_id)
        assert (x, y) in ((600, 300), (200, 300))
        self.actions.append(("pixel", capture.capture_id))
        if x == 600:
            self.path.write_text(self.text, encoding="utf-8")
            self.status = "Saved"
        return {"effect": "confirmed"}

    async def set_value(self, window: Window, token: str, text: str) -> dict[str, Any]:
        self._check(window, token)
        self.actions.append(("replace", token))
        self.text = text
        return {"effect": "confirmed", "verified": True}

    def _check(self, window: Window, token: str) -> None:
        if window != self.window or not token.endswith(f"-{self.snapshots}"):
            raise DriverError(f"stale token {token}")


class TestDocumentActions(IsolatedAsyncioTestCase):
    async def test_ambiguous_text_fields_are_not_offered_or_written(self) -> None:
        original = self.fake.window_state
        for identifier, enabled in (("", True), ("shared", True), ("", False)):
            with self.subTest(identifier=identifier, enabled=enabled):

                async def duplicate_fields(window: Window) -> Snapshot:
                    snapshot = await original(window)
                    field = replace(snapshot.elements[0], index=0, identifier=identifier)
                    duplicate = replace(field, index=1, token="other", enabled=enabled)
                    return replace(snapshot, elements=(field, duplicate, *snapshot.elements[1:]))

                with patch.object(self.fake, "window_state", side_effect=duplicate_fields):
                    env = self.env()
                    await env.reset()
                    key = f"type:{identifier or 'Body'}"
                    for candidate in (key, f"{key}#1"):
                        with self.assertRaises(KeyError):
                            await env.step(candidate)
                    self.assertFalse(any(k.startswith("type:") for k in await env.candidates()))
                    self.assertTrue((await env.observe())["text_pending"])
                    self.assertEqual((self.fake.text, self.fake.actions), ("", []))

    async def test_same_label_with_distinct_identifiers_keeps_readback_identity(self) -> None:
        original = self.fake.window_state

        async def distinct_fields(window: Window) -> Snapshot:
            snapshot = await original(window)
            field = replace(snapshot.elements[0], identifier="body")
            other = replace(field, index=4, identifier="other", value="untouched", token="other")
            return replace(snapshot, elements=(field, other, *snapshot.elements[1:]))

        with patch.object(self.fake, "window_state", side_effect=distinct_fields):
            env = self.env()
            await env.reset()
            self.assertIn("type:body", await env.candidates())
            self.assertIn("type:other", await env.candidates())
            await env.step("type:body")
            self.assertEqual(self.fake.text, "hello")
            self.assertFalse((await env.observe())["text_pending"])
            self.assertEqual((await env.observe())["elements"][1]["value"], "untouched")

    async def test_saved_status_requires_verified_text_without_file_check(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            ["--model", "rule", "--rethink", "off", "--episodes", "1"]
            + ["--app", "Document", "--goal", "write and save", "--expect", "Saved", "--text", "hello", "--execute"]
        )
        with patch.object(desktop, "driver_from_env", return_value=self.fake):
            env = desktop.make_series(args).env_for(0)
        await env.reset()
        await env.step("click:Save")
        self.assertEqual((env.done, env.score), (False, 0.0))
        self.assertTrue((await env.observe())["text_pending"])
        self.assertIn("type:Body", await env.candidates())

        await env.step("type:Body")
        self.assertEqual((env.done, env.score), (False, 0.0))
        await env.step("click:Save")
        self.assertEqual((env.done, env.score), (True, 1.0))
        self.fake.text = "changed by app"
        await env.step("click:Save")
        self.assertEqual((env.done, env.score), (False, 0.0))
        self.assertTrue((await env.observe())["text_pending"])

    async def test_insert_accepts_replacing_selected_payload_with_identical_text(self) -> None:
        self.fake.text = "hello world"

        async def replace_selection(window: Window, token: str, text: str) -> dict:
            self.fake._check(window, token)
            self.fake.text = text + self.fake.text[len("hello") :]
            return {"effect": "confirmed"}

        self.fake.type_text = replace_selection
        env = self.env()
        await env.reset()
        await env.step("type:Body")
        self.assertEqual(self.fake.text, "hello world")
        self.assertFalse((await env.observe())["text_pending"])
        self.assertEqual((await env.observe())["presses"], ["type:Body"])
        self.assertNotIn("type:Body", await env.candidates())

    async def test_unchanged_insert_requires_driver_confirmation_and_matching_text(self) -> None:
        for value, effect in (("hello world", "unverifiable"), ("other text", "confirmed")):
            with self.subTest(value=value, effect=effect):
                self.fake.text = value
                self.fake.type_text = AsyncMock(return_value={"effect": effect})
                env = self.env()
                await env.reset()
                with self.assertRaisesRegex(DriverError, "text verification failed"):
                    await env.step("type:Body")
                self.assertTrue((await env.observe())["text_pending"])
                self.assertEqual((await env.observe())["presses"], [])

    async def test_claimed_success_with_incomplete_text_is_not_recorded_as_complete(self) -> None:
        async def truncated(window: Window, token: str, text: str) -> dict:
            self.fake.text = text[:2]
            return {"effect": "confirmed"}

        self.fake.type_text = truncated
        env = self.env()
        await env.reset()
        with self.assertRaisesRegex(DriverError, "text verification failed"):
            await env.step("type:Body")
        self.assertEqual((await env.observe())["presses"], [])
        self.assertTrue((await env.observe())["text_pending"])

    async def test_insert_does_not_verify_against_a_payload_already_in_the_field(self) -> None:
        self.fake.text = "hello world"

        async def partial(window: Window, token: str, text: str) -> dict:
            self.fake.text = "hhello world"
            return {"effect": "confirmed"}

        self.fake.type_text = partial
        env = self.env()
        await env.reset()
        with self.assertRaisesRegex(DriverError, "text verification failed"):
            await env.step("type:Body")

    async def test_replace_handles_existing_chinese_and_multiline_text_and_detects_later_edits(self) -> None:
        self.fake.text = "old default"
        payload = "本地桌面代理\n第二行：验证保存。"
        env = WindowEnv(
            self.fake,
            app_name="Document",
            goal="write",
            done_when=lambda s: False,
            execute=True,
            clear_labels=(),
            text=payload,
            text_target="Body",
            text_mode="replace",
        )
        await env.reset()
        await env.step("type:Body")
        self.assertEqual(self.fake.text, payload)
        self.assertEqual(self.fake.actions[0][0], "replace")
        self.assertNotIn("type:Body", await env.candidates())
        self.fake.text = "changed by app"
        await env.step("click:Save")
        self.assertIn("type:Body", await env.candidates())
        self.assertTrue((await env.observe())["text_pending"])

    async def test_unverifiable_replace_is_not_accepted_even_if_ax_value_matches(self) -> None:
        original = self.fake.set_value

        async def unverifiable(window: Window, token: str, text: str) -> dict:
            await original(window, token, text)
            return {"effect": "unverifiable"}

        self.fake.set_value = unverifiable
        env = WindowEnv(
            self.fake,
            app_name="Document",
            goal="write",
            done_when=lambda s: False,
            execute=True,
            clear_labels=(),
            text="hello",
            text_target="Body",
            text_mode="replace",
        )
        await env.reset()
        with self.assertRaisesRegex(DriverError, "unverifiable"):
            await env.step("type:Body")
        self.assertEqual((await env.observe())["presses"], [])

    async def test_session_waits_for_new_window_but_does_not_retry_ambiguity(self) -> None:
        driver = AsyncMock()
        driver.find_window.side_effect = [DriverError("list_windows: 0 on-screen window(s)"), self.fake.window]
        with patch.object(desktop, "launch_app", AsyncMock()), patch.object(desktop.asyncio, "sleep", AsyncMock()):
            async with desktop._session(driver, "Document", owner="Document", title="Document"):
                pass
        self.assertEqual(driver.find_window.await_count, 2)
        driver.find_window.reset_mock(side_effect=True)
        driver.find_window.side_effect = DriverError("list_windows: 2 on-screen window(s)")
        with patch.object(desktop, "launch_app", AsyncMock()), self.assertRaises(DriverError):
            async with desktop._session(driver, "Document", owner="Document", title="Document"):
                pass
        self.assertEqual(driver.find_window.await_count, 1)

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "document.txt"
        self.fake = FakeDocument(self.path)

    def env(self, *, execute: bool = True) -> WindowEnv:
        return WindowEnv(
            self.fake,
            app_name="Document",
            goal="write and save the document",
            done_when=lambda snapshot: snapshot.elements[2].value == "Saved" and self.path.read_text() == "hello",
            execute=execute,
            clear_labels=(),
            text="hello",
        )

    async def test_typing_and_save_refresh_the_snapshot_and_verify_the_file(self) -> None:
        env = self.env()
        await env.reset()
        self.assertIn("type:Body", await env.candidates())
        await env.step("type:Body")
        self.assertNotIn("type:Body", await env.candidates())
        await env.step("click:Save")
        self.assertEqual(self.path.read_text(), "hello")
        self.assertEqual((env.done, env.score), (True, 1.0))
        self.assertEqual(self.fake.snapshots, 3)
        self.assertEqual([name for name, _ in self.fake.actions], ["type", "click"])
        self.assertEqual((await env.observe())["presses"], ["type:Body", "Save"])

    async def test_dry_run_records_the_selected_action_without_mutation(self) -> None:
        env = self.env(execute=False)
        await env.reset()
        await env.step("type:Body")
        self.assertEqual((env.done, env.score, self.fake.actions, self.path.exists()), (True, 0.0, [], False))
        self.assertEqual((await env.observe())["planned"]["key"], "type:Body")

    async def test_explicit_text_target_excludes_other_text_areas(self) -> None:
        original = self.fake.window_state

        async def with_status(window: Window) -> Snapshot:
            snapshot = await original(window)
            return Snapshot(
                window,
                snapshot.snapshot_id,
                (*snapshot.elements, Element(4, "AXTextArea", "Status", "Editing", "status", ())),
                {},
            )

        self.fake.window_state = with_status
        env = WindowEnv(
            self.fake,
            app_name="Document",
            goal="write",
            done_when=lambda s: False,
            execute=True,
            clear_labels=(),
            text="hello",
            text_target="Body",
        )
        await env.reset()
        self.assertIn("type:Body", await env.candidates())
        self.assertNotIn("type:Status", await env.candidates())
        await env.step("type:Body")
        self.assertEqual(self.fake.text, "hello")

    async def test_stale_target_and_driver_failure_do_not_count_as_a_successful_step(self) -> None:
        env = self.env()
        await env.reset()
        await self.fake.window_state(self.fake.window)  # invalidates the token given to the model
        with self.assertRaises(DriverError):
            await env.step("type:Body")
        self.assertEqual((await env.observe())["presses"], [])
        self.assertFalse(self.path.exists())

    async def test_full_agent_rule_run_writes_and_checks_the_document(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            [
                "--model",
                "rule",
                "--rethink",
                "off",
                "--episodes",
                "1",
                "--app",
                "Document",
                "--goal",
                "write hello in the Body and save it",
                "--expect",
                "Saved",
                "--text",
                "hello",
                "--verify-file",
                str(self.path),
                "--plan",
                "type:Body,Save",
                "--execute",
            ]
        )
        with (
            patch.object(loop, "WORKSPACE", Path(self.tmp.name) / "ws"),
            patch.object(series, "optional_chat_model", lambda: None),
            patch.object(desktop, "driver_from_env", lambda label: self.fake),
            patch.object(desktop, "launch_app", AsyncMock()),
        ):
            async with started_runner():
                result = await series.play(desktop.SPEC, args, results_dir=Path(self.tmp.name) / "results")
        self.assertEqual((result["mean_score"], result["errors"], self.fake.opened), (1.0, 0, 0))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "hello")

    async def test_full_agent_does_not_finish_on_the_status_from_an_early_save(self) -> None:
        for mode in ("insert", "replace"):
            for plan, score, saved in (
                ("Save,type:Body,Save", 1.0, "hello"),
                ("Save,type:Body", 0.0, ""),
                ("Save,type:Body,Other,Save", 1.0, "hello"),
                ("Save,type:Body,Other", 0.0, ""),
                ("pixel:Save,type:Body,pixel:Save", 1.0, "hello"),
                ("pixel:Save,type:Body", 0.0, ""),
                ("pixel:Save,type:Body,pixel:Other,pixel:Save", 1.0, "hello"),
                ("pixel:Save,type:Body,pixel:Other", 0.0, ""),
            ):
                with self.subTest(mode=mode, plan=plan):
                    self.fake = FakeDocument(self.path)
                    original = self.fake.window_state

                    async def with_other_button(window: Window, *, screenshot: bool = False) -> Snapshot:
                        snapshot = await original(window, screenshot=screenshot)
                        other = Element(4, "AXButton", "Other", "", f"other-{self.fake.snapshots}", ("AXPress",))
                        return replace(snapshot, elements=(*snapshot.elements, other))

                    self.fake.window_state = with_other_button
                    args = series.parser(desktop.SPEC).parse_args(
                        ["--model", "rule", "--rethink", "off", "--episodes", "1"]
                        + ["--app", "Document", "--goal", "write hello and save", "--expect", "Saved"]
                        + ["--text", "hello", "--text-mode", mode, "--plan", plan, "--execute"]
                        + (
                            ["--pixel-target", "Save=0.75,0.5", "--pixel-target", "Other=0.25,0.5"]
                            if "pixel:" in plan
                            else []
                        )
                    )
                    with (
                        patch.object(loop, "WORKSPACE", Path(self.tmp.name) / "ws"),
                        patch.object(series, "optional_chat_model", lambda: None),
                        patch.object(desktop, "driver_from_env", return_value=self.fake),
                        patch.object(desktop, "launch_app", AsyncMock()),
                    ):
                        async with started_runner():
                            result = await series.play(desktop.SPEC, args, results_dir=Path(self.tmp.name) / "results")
                    self.assertEqual((result["mean_score"], result["errors"], self.fake.opened), (score, 0, 0))
                    self.assertEqual(self.fake.text, "hello")
                    self.assertEqual(self.path.read_text(encoding="utf-8"), saved)
                    actions = [
                        ("type" if mode == "insert" else "replace")
                        if action == "type:Body"
                        else "pixel"
                        if action.startswith("pixel:")
                        else "click"
                        for action in plan.split(",")
                    ]
                    self.assertEqual([name for name, _ in self.fake.actions], actions)

    async def test_text_input_can_produce_a_new_completion_state(self) -> None:
        env = WindowEnv(
            self.fake,
            app_name="Document",
            goal="write hello",
            done_when=lambda snapshot: snapshot.elements[0].value == "hello",
            execute=True,
            clear_labels=(),
            text="hello",
        )
        await env.reset()
        await env.step("type:Body")
        self.assertEqual((env.done, env.score), (True, 1.0))

    async def test_file_verification_does_not_trust_saved_label(self) -> None:
        args = series.parser(desktop.SPEC).parse_args(
            [
                "--model",
                "rule",
                "--rethink",
                "off",
                "--episodes",
                "1",
                "--app",
                "Document",
                "--goal",
                "save hello",
                "--expect",
                "Saved",
                "--text",
                "hello",
                "--verify-file",
                str(self.path),
                "--execute",
            ]
        )
        with patch.object(desktop, "driver_from_env", lambda label: self.fake):
            env = desktop.make_series(args).env_for(0)
        await env.reset()
        self.assertFalse(env.done)
        self.fake.status = "Saved"
        self.path.write_text("wrong contents", encoding="utf-8")
        await env.step("type:Body")
        self.assertFalse(env.done)
        await env.step("click:Save")
        self.assertTrue(env.done)

    async def test_existing_matching_file_does_not_count_as_a_new_save(self) -> None:
        self.path.write_text("hello", encoding="utf-8")
        # Model a previous run without requiring two writes to advance the filesystem clock.
        os.utime(self.path, (1_000_000_000, 1_000_000_000))
        args = series.parser(desktop.SPEC).parse_args(
            [
                "--model",
                "rule",
                "--rethink",
                "off",
                "--episodes",
                "1",
                "--app",
                "Document",
                "--goal",
                "save hello",
                "--expect",
                "Saved",
                "--text",
                "hello",
                "--verify-file",
                str(self.path),
                "--execute",
            ]
        )
        with patch.object(desktop, "driver_from_env", lambda label: self.fake):
            env = desktop.make_series(args).env_for(0)
        await env.reset()
        self.fake.status = "Saved"  # stale status and bytes from a previous trial
        await env._refresh()
        self.assertFalse(env.done)
        await env.step("type:Body")
        self.assertFalse(env.done)
        await env.step("click:Save")
        self.assertTrue(env.done)
