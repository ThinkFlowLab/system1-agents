# coding: utf-8
"""Frame capture in the browser games: off by default, one PNG per move when a frames directory is given."""

from __future__ import annotations

import json
import socket
import tempfile
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any, AsyncIterator
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from s1a.agents.game2048 import Game2048Env
from s1a.agents.millionaire import MillionaireEnv
from s1a.tool import hands as hands_module
from s1a.tool.hands import BrowserHands, serve_static, serving


class FakeHands:
    """Answers every page read with the next canned state and records the frames it was asked for."""

    def __init__(self, states: list[dict[str, Any]]) -> None:
        self._states = list(states)
        self.frames: list[Path] = []
        self.pressed: list[str] = []

    async def wait_ready(self, selector: str, timeout_s: float) -> None:
        return None

    async def read(self, source: str, args: Any) -> Any:
        if "__state" in source or "gameState" in source:
            return self._states.pop(0)
        return "ok"

    async def press(self, key: str) -> None:
        self.pressed.append(key)

    async def frame(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        self.frames.append(path)


def _grid_state(score: int) -> dict[str, Any]:
    return {
        "grid": [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, score]],
        "score": score,
        "over": False,
        "won": False,
    }


class TestGame2048Frames(IsolatedAsyncioTestCase):
    async def test_no_frames_dir_means_no_frames(self) -> None:
        hands = FakeHands([_grid_state(0), _grid_state(4)])
        env = Game2048Env(hands, seed=0, frames_dir=None)
        await env.reset()
        await env.step("LEFT")
        self.assertEqual(hands.frames, [])
        self.assertIsNone(env.frames_dir)

    async def test_one_frame_at_reset_and_one_per_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "2048-0"
            hands = FakeHands([_grid_state(0), _grid_state(4), _grid_state(8)])
            env = Game2048Env(hands, seed=0, frames_dir=frames_dir)
            await env.reset()
            await env.step("LEFT")
            await env.step("DOWN")
            self.assertEqual([p.name for p in hands.frames], ["000.png", "001.png", "002.png"])
            self.assertEqual(env.frames_dir, frames_dir)
            self.assertEqual(sorted(p.name for p in frames_dir.iterdir()), ["000.png", "001.png", "002.png"])

    async def test_the_2048_tile_ends_the_game_and_the_observation_names_the_corner_tie(self) -> None:
        won = {
            "grid": [[512, 0, 0, 512], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "score": 9,
            "over": False,
            "won": True,
        }
        tied = {
            "grid": [[0, 512, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 512]],
            "score": 9,
            "over": False,
            "won": False,
        }
        env = Game2048Env(FakeHands([tied, won]), seed=0, frames_dir=None)
        await env.reset()
        state = await env.observe()
        self.assertEqual(
            (state["largest_tile_row_col"], state["progress"]), ([3, 3], {"score": 9, "largest_tile": 512})
        )
        self.assertFalse(env.done)
        await env.step("UP")
        self.assertTrue(env.done)
        self.assertEqual(await env.candidates(), {})

    async def test_reset_clears_frames_left_by_an_earlier_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "2048-0"
            frames_dir.mkdir()
            (frames_dir / "007.png").write_bytes(b"old")
            env = Game2048Env(FakeHands([_grid_state(0)]), seed=0, frames_dir=frames_dir)
            await env.reset()
            self.assertEqual(sorted(p.name for p in frames_dir.iterdir()), ["000.png"])


def _quiz_state(level: int, status: str) -> dict[str, Any]:
    return {
        "level": level,
        "prize_for_this_question": 100 * level,
        "question": "Q?",
        "answers": {"1": "a", "2": "b", "3": "c", "4": "d"},
        "fifty_fifty_available": True,
        "status": status,
        "winnings": 0,
        "history": [],
    }


class TestMillionaireFrames(IsolatedAsyncioTestCase):
    async def test_a_correct_answer_waits_for_the_next_question_to_load(self) -> None:
        hands = FakeHands([_quiz_state(1, "playing"), _quiz_state(2, "loading"), _quiz_state(2, "playing")])
        env = MillionaireEnv(hands, "http://127.0.0.1:1/index.html", seed=0, frames_dir=None)
        await env.reset()
        await env.step("1")
        self.assertEqual(((await env.observe())["status"], hands.frames, hands._states), ("playing", [], []))

    async def test_one_frame_per_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "millionaire-0"
            hands = FakeHands([_quiz_state(1, "playing"), _quiz_state(2, "playing")])
            env = MillionaireEnv(hands, "http://127.0.0.1:1/index.html", seed=0, frames_dir=frames_dir)
            await env.reset()
            await env.step("1")
            self.assertEqual([p.name for p in hands.frames], ["000.png", "001.png"])
            self.assertEqual(hands.pressed, ["1"])
            self.assertEqual(json.loads(json.dumps(await env.observe()))["level"], 2)


class FakeRuntime:
    """The Playwright runtime offline: navigate fails, every probe answers without a value, stop is recorded."""

    def __init__(self, **kwargs: Any) -> None:
        self.stopped: list[str] = []

    async def acquire_task_resources(self) -> None:
        return None

    async def ensure_started(self) -> None:
        return None

    async def _call_playwright_tool(self, name: str, args: dict[str, Any]) -> None:
        raise RuntimeError(f"{name} failed")

    async def _execute_probe_json(self, code: str, artifact_kind: str) -> tuple[dict[str, Any], None, int]:
        return {"error": "page crashed"}, None, 0

    async def release_task_resources(self) -> None:
        self.stopped.append("release")

    async def shutdown(self) -> None:
        self.stopped.append("shutdown")


class TestBrowserHandsLifecycle(IsolatedAsyncioTestCase):
    async def test_a_failed_start_still_stops_the_runtime(self) -> None:
        with patch.object(hands_module, "BrowserAgentRuntime", FakeRuntime):
            hands = BrowserHands(headless=True)
        with self.assertRaisesRegex(RuntimeError, "browser_navigate failed"):
            async with hands.session("http://127.0.0.1:1/index.html", ready_selector="body", timeout_s=1.0):
                raise AssertionError("the session must not open")
        self.assertEqual(hands.runtime.stopped, ["release", "shutdown"])

    async def test_page_not_ready_names_the_last_probe_error(self) -> None:
        with patch.object(hands_module, "BrowserAgentRuntime", FakeRuntime):
            hands = BrowserHands(headless=True)
        with self.assertRaisesRegex(RuntimeError, "page not ready") as caught:
            await hands.wait_ready("body", 0.0)
        self.assertIn("page crashed", str(caught.exception.__cause__))

    async def test_serving_shuts_the_static_server_down_when_the_session_ends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            site = serve_static(Path(tmp))
            self.assertGreaterEqual(site.fileno(), 0)
            async with serving(site, nullcontext()):
                with socket.create_connection(site.server_address, timeout=1.0):
                    pass
            # Check our socket directly: a closed-port connection can time out on Windows.
            self.assertEqual(site.fileno(), -1)

    async def test_serving_shuts_the_static_server_down_when_the_session_fails_to_open(self) -> None:
        @asynccontextmanager
        async def failing() -> AsyncIterator[None]:
            raise RuntimeError("no browser")
            yield

        with tempfile.TemporaryDirectory() as tmp:
            site = serve_static(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "no browser"):
                async with serving(site, failing()):
                    pass
            self.assertEqual(site.fileno(), -1)
