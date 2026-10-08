# coding: utf-8
"""Regressions for the snake client: multi-grid raster stability, warmup terminal states,
prompt forwarding, concurrent failure stop, out-of-order recording timelines, custom
board sizes, and backend captions derived from recorded provenance."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals" / "snake"))

from snake.game import SnakeGame  # noqa: E402
from snake.multi import run_multi  # noqa: E402
from snake.multi_replay import backend_caption, load_multi  # noqa: E402
from snake.multi_ui import CELL_H, CELL_W, TOP_BAR, cell_size, compose_multi  # noqa: E402
from snake.replay import TerminalRaster  # noqa: E402
from snake.warmup import warm_up  # noqa: E402


def a_frame(seed, at=0.0):
    game = SnakeGame(24, 16, seed, 6)
    return {
        "type": "frame",
        "at": at,
        "seed": seed,
        "round": 1,
        "game": game.snapshot(),
        "decision": {},
        "steps": 0,
        "deaths": 0,
        "interventions": 0,
    }


def pending(seed):
    return {"seed": seed, "pending": True, "width": 24, "height": 16}


def stats(**overrides):
    base = {"alive": 1, "waiting": 15, "dps": 0, "mean_ms": 0, "score": 0, "clock": "00:00"}
    base.update(overrides)
    return base


class ExplodingPolicy:
    def decide(self, game):  # pragma: no cover - must never run
        raise AssertionError("decide called on a game with no moves")


class CountingPolicy:
    def __init__(self):
        self.calls = 0

    def decide(self, game):
        self.calls += 1
        moves = [m for m in game.moves() if m.legal]
        return SimpleNamespace(executed=(moves or game.moves())[0].direction)


class RightPolicy:
    def __init__(self):
        self.calls = 0

    def decide(self, game):
        self.calls += 1
        return SimpleNamespace(executed="RIGHT")


class TestWarmup(unittest.TestCase):
    def test_warmup_stops_when_the_warm_game_wins(self):
        # 4x4 board, length 15, seed 0, RIGHT: eating the last cell wins in one step,
        # and a won game stays "alive" while its move list is empty.
        game = SnakeGame(4, 4, 0, 15)
        self.assertEqual(game.legal_reason("RIGHT"), "legal")
        self.assertEqual(game.target("RIGHT"), game.food)
        game.step("RIGHT")
        self.assertTrue(game.won)
        self.assertTrue(game.alive)
        self.assertEqual(game.moves(), [])
        warm_up(ExplodingPolicy(), game)

    def test_warmup_stops_after_winning_during_warmup(self):
        # The same 4x4 seed-0 game wins on the very first RIGHT decision; the
        # warmup must not ask for another decision after the win.
        policy = RightPolicy()
        warm_up(policy, SnakeGame(4, 4, 0, 15))
        self.assertEqual(policy.calls, 1)

    def test_warmup_respects_the_step_budget(self):
        policy = CountingPolicy()
        warm_up(policy, SnakeGame(24, 16, 7, 6), steps=3)
        self.assertEqual(policy.calls, 3)

    def test_warmup_runs_on_a_normal_board(self):
        policy = CountingPolicy()
        warm_up(policy, SnakeGame(24, 16, 7, 6))
        self.assertEqual(policy.calls, 6)


class TestMultiGridRaster(unittest.TestCase):
    def test_canvas_size_is_constant_with_pending_slots(self):
        frame = a_frame(1000)
        partial = compose_multi([frame] + [pending(s) for s in range(1, 16)], stats(), 4, "C")
        full = compose_multi([frame] * 16, stats(waiting=0), 4, "C")
        self.assertEqual((partial.width, partial.height), (full.width, full.height))
        self.assertEqual(full.height, TOP_BAR + 4 * CELL_H + 2)

    def test_raster_keeps_every_row_inside_the_image(self):
        for width, height in ((24, 16), (32, 24), (24, 40)):
            with self.subTest(width=width, height=height):
                frames = [a_frame(1000 + i) for i in range(16)]
                for frame in frames:
                    frame["game"] = SnakeGame(width, height, frame["seed"], 6).snapshot()
                canvas = compose_multi(frames, stats(waiting=0), 4, "C")
                try:
                    raster = TerminalRaster(canvas.width, canvas.height, width=1920, height=1080)
                except FileNotFoundError:
                    self.skipTest("no monospace font available on this host")
                self.assertGreaterEqual(raster.y - 42, 0)
                self.assertLessEqual(raster.y + canvas.height * raster.ch + 14, 1080)

    def test_pending_slots_render_a_placeholder(self):
        canvas = compose_multi([a_frame(1000), pending(1001)], stats(waiting=1), 2, "C")
        text = "".join("".join(row) for row in canvas.chars)
        self.assertIn("pending", text)
        self.assertIn("#1001", text)

    def test_width_matches_the_grid(self):
        canvas = compose_multi([pending(s) for s in range(4)], stats(), 4, "C")
        self.assertEqual(canvas.width, 6 + 4 * CELL_W)

    def test_large_boards_fit_their_slot(self):
        # 32x24 board: the slot must grow from the board size; the fixed
        # 44x22 slot used to overlap columns and silently clip the bottom border.
        frame = a_frame(1000)
        frame["game"] = SnakeGame(32, 24, 1000, 6).snapshot()
        canvas = compose_multi([frame] * 4, stats(waiting=0), 4, "C")
        cell_w, cell_h = cell_size(32, 24)
        self.assertEqual(canvas.width, 6 + 4 * cell_w)
        self.assertEqual(canvas.height, TOP_BAR + cell_h + 2)
        text = "".join("".join(row) for row in canvas.chars)
        self.assertIn("└" + "─" * 32 + "┘", text)


class TestPromptForwarding(unittest.TestCase):
    def test_multi_forwards_the_selected_prompt(self):
        captured = {}

        class FakePolicy:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                self.metadata = {}

            def decide(self, game):
                moves = [m for m in game.moves() if m.legal]
                return SimpleNamespace(
                    executed=(moves or game.moves())[0].direction,
                    to_dict=lambda: {},
                    inference_ms=0.1,
                    intervened=False,
                )

        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "rec.jsonl"
            with mock.patch("snake.multi.LayaPolicy", FakePolicy):
                run_multi(
                    [
                        "--games",
                        "1",
                        "--steps",
                        "1",
                        "--fps",
                        "100",
                        "--prompt",
                        "detailed",
                        "--record",
                        str(record),
                    ]
                )
            first = json.loads(record.read_text().splitlines()[0])
            self.assertEqual(first["settings"]["prompt"], "detailed")

        self.assertEqual(captured.get("prompt"), "detailed")


class TestMultiFailureStopsOthers(unittest.TestCase):
    def test_interrupt_while_waiting_stops_the_games(self):
        started = threading.Event()

        def tick(runner):
            runner.steps += 1
            started.set()
            return True

        def interrupt_wait(futures):
            self.assertTrue(started.wait(timeout=2), "the game did not start")
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "rec.jsonl"
            with (
                mock.patch("snake.multi.LayaPolicy", return_value=SimpleNamespace(metadata={})),
                mock.patch("snake.multi.warm_up"),
                mock.patch("snake.multi.GameRunner.tick", tick),
                mock.patch("snake.multi.as_completed", side_effect=interrupt_wait),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_multi(["--games", "1", "--steps", "5", "--fps", "20", "--record", str(record)])
            events = [json.loads(line) for line in record.read_text().splitlines()]
        self.assertLess(sum(e["type"] == "frame" for e in events), 5)
        self.assertFalse(any(e["type"] == "end" for e in events))

    def test_a_failing_game_stops_the_others(self):
        # Offline repro of the previous submit-order wait: one game raising on
        # its first decision used to leave the other game running its full
        # 5-step budget before run_multi returned.
        class MixPolicy:
            def __init__(self, *args, **kwargs):
                self.metadata = {}

            def decide(self, game):
                if game.seed % 2 == 0:
                    raise RuntimeError("backend exploded")
                time.sleep(0.05)
                moves = [m for m in game.moves() if m.legal]
                return SimpleNamespace(
                    executed=(moves or game.moves())[0].direction,
                    to_dict=lambda: {},
                    inference_ms=0.1,
                    intervened=False,
                )

        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "rec.jsonl"
            with mock.patch("snake.multi.LayaPolicy", MixPolicy), mock.patch("snake.multi.warm_up"):
                with self.assertRaises(RuntimeError):
                    run_multi(["--games", "2", "--seed", "2000", "--steps", "5", "--fps", "4", "--record", str(record)])
            events = [json.loads(line) for line in record.read_text().splitlines()]
        frames = [e for e in events if e["type"] == "frame"]
        self.assertLess(len(frames), 5, "the healthy game ran its full budget despite the failure")
        self.assertFalse(any(e["type"] == "end" for e in events), "a failed run must not record a summary")


class TestRecordingTimeline(unittest.TestCase):
    def write_recording(self, path, events):
        lines = [json.dumps({"type": "metadata", "model": {}})]
        lines += [json.dumps(e) for e in events]
        lines.append(json.dumps({"type": "end", "summary": {}}))
        path.write_text("\n".join(lines) + "\n")

    def test_load_multi_sorts_frames_by_timestamp(self):
        # Concurrent writers append under a lock, so file order is not time
        # order; replay walks a cursor and must see timestamps ascending.
        a = a_frame(1000, at=0.0)
        b = a_frame(1000, at=0.2)
        c = a_frame(1000, at=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "rec.jsonl"
            self.write_recording(record, [a, b, c])
            _, frames, _ = load_multi(record)
        self.assertEqual([f["at"] for f in frames], [0.0, 0.1, 0.2])

    def test_load_multi_rejects_frames_without_a_numeric_timestamp(self):
        for missing in ({"type": "frame", "seed": 1, "game": {}}, {**a_frame(1), "at": "0.1"}):
            with tempfile.TemporaryDirectory() as tmp:
                record = Path(tmp) / "rec.jsonl"
                self.write_recording(record, [a_frame(1), missing])
                with self.assertRaises(ValueError):
                    load_multi(record)


class TestBackendCaption(unittest.TestCase):
    def test_caption_follows_the_recorded_engine(self):
        caption = backend_caption({"model": {"engine_label": "torch cpu · float32"}})
        self.assertIn("TORCH CPU", caption)
        self.assertNotIn("NATIVE", caption)
        self.assertNotIn("H800", caption)

    def test_unknown_engine_stays_unknown(self):
        cases = ({}, {"model": {}}, {"model": {"engine_label": "unknown"}}, {"model": {"engine_label": None}})
        for metadata in cases:
            self.assertIn("UNKNOWN ENGINE", backend_caption(metadata))

    def test_native_recording_does_not_claim_hardware_it_did_not_record(self):
        caption = backend_caption({"model": {"engine_label": "native CUDA sm_90a · BF16"}})
        self.assertIn("NATIVE CUDA SM_90A · BF16", caption)
        self.assertNotIn("H800", caption)


if __name__ == "__main__":
    unittest.main()
