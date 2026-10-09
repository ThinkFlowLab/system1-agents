# coding: utf-8
"""``WindowEnv`` over the fake calculator: candidates, snapshot-bound clicks, done and score, dry run."""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from s1a.desktop.driver import Element, Snapshot
from s1a.desktop.env import ABSTAIN, DONE, WindowEnv, clickable, observable
from support_desktop import FakeCalculator


def _display(snapshot: Snapshot) -> str:
    return snapshot.elements[0].value


def _env(fake: FakeCalculator, *, execute: bool = True) -> WindowEnv:
    return WindowEnv(
        fake,
        app_name="Calculator",
        goal="compute 12 times 7",
        done_when=lambda s: _display(s) == "84",
        execute=execute,
        clear_labels=("All Clear",),
    )


class TestWindowEnv(IsolatedAsyncioTestCase):
    async def test_menu_items_and_unlabelled_window_controls_are_not_candidates(self) -> None:
        self.assertFalse(clickable(Element(1, "AXMenuItem", "Shut Down", "", "token", ("AXPress",))))
        self.assertFalse(clickable(Element(2, "AXButton", "", "", "token", ("AXPress",))))
        self.assertFalse(clickable(Element(4, "AXButton", "Saved", "", "token", ("AXPress",), enabled=False)))
        self.assertTrue(clickable(Element(3, "AXButton", "Save", "", "token", ("AXPress",))))
        self.assertFalse(observable(Element(1, "AXMenuItem", "Shut Down", "", "token", ("AXPress",))))
        self.assertFalse(observable(Element(2, "AXButton", "", "", "token", ("AXPress",))))
        self.assertTrue(observable(Element(4, "AXTextArea", "Status", "Saved", "token", ())))

    async def test_candidates_are_the_clickable_elements_plus_abstain_until_success(self) -> None:
        env = _env(FakeCalculator())
        await env.reset()
        offered = await env.candidates()
        self.assertEqual(offered["click:7"], 'AXButton "7"')
        self.assertNotIn(DONE, offered)  # a premature done cannot be a valid model choice
        self.assertEqual(offered[ABSTAIN], "No offered action moves the task forward; stop without acting.")
        with self.assertRaises(KeyError):
            await env.step(DONE)
        self.assertNotIn("click:keypad", offered)  # no press action
        self.assertEqual(len(offered), 13 + 1)

    async def test_a_step_clicks_the_token_of_the_latest_snapshot_in_the_bound_window_and_re_reads(self) -> None:
        fake = FakeCalculator()
        env = _env(fake)
        await env.reset()  # one snapshot, All Clear clicked, one more snapshot
        self.assertEqual([c[2] for c in fake.clicks], ["tok-All Clear-1"])
        for key in ("click:1", "click:2", "click:Multiply", "click:7"):
            await env.step(key)
        self.assertEqual([c[:2] for c in fake.clicks], [(42, 7)] * 5)
        self.assertEqual(fake.clicks[-1][2], "tok-7-5")  # the token of the snapshot the model saw
        state = await env.observe()
        self.assertEqual((state["goal"], state["app"]), ("compute 12 times 7", "Calculator"))
        self.assertEqual(
            (state["presses"], state["progress"]), (["1", "2", "Multiply", "7"], {"values": ["7"], "presses": 4})
        )
        self.assertEqual(state["elements"][0], {"role": "AXStaticText", "label": "", "value": "7"})
        self.assertEqual((env.done, env.score), (False, 0.0))
        await env.step("click:Equals")
        self.assertEqual((env.done, env.score, await env.candidates()), (True, 1.0, {}))

    async def test_abstain_ends_the_episode_without_a_click(self) -> None:
        fake = FakeCalculator()
        env = _env(fake)
        await env.reset()
        await env.step(ABSTAIN)
        self.assertEqual((env.done, env.score, len(fake.clicks)), (True, 0.0, 1))

    async def test_a_dry_run_plans_one_click_and_ends_without_touching_the_window(self) -> None:
        fake = FakeCalculator()
        env = _env(fake, execute=False)
        await env.reset()
        self.assertEqual(fake.clicks, [])  # no All Clear either
        await env.step("click:1")
        self.assertEqual(fake.clicks, [])
        self.assertEqual((env.done, env.score), (True, 0.0))
        planned = (await env.observe())["planned"]
        self.assertEqual((planned["key"], planned["label"], planned["role"]), ("click:1", "1", "AXButton"))
        self.assertTrue(planned["token"].startswith("tok-1-"))

    async def test_observing_before_reset_is_an_error_and_a_missing_window_surfaces(self) -> None:
        with self.assertRaises(RuntimeError):
            await _env(FakeCalculator()).observe()
        with self.assertRaises(Exception) as caught:
            await _env(FakeCalculator(windows=2)).reset()
        self.assertIn("2 on-screen window(s)", str(caught.exception))

    async def test_a_window_that_already_shows_the_result_refuses_to_start(self) -> None:
        fake = FakeCalculator()
        fake.display = "84"
        env = WindowEnv(
            fake,
            app_name="Calculator",
            goal="compute 12 times 7",
            done_when=lambda s: _display(s) == "84",
            execute=True,
            clear_labels=(),
        )
        with self.assertRaises(RuntimeError) as caught:
            await env.reset()
        self.assertIn("--clear", str(caught.exception))
        cleared = _env(fake)
        await cleared.reset()  # All Clear wipes the display first, so the same window starts
        self.assertEqual((fake.display, cleared.done), ("", False))
