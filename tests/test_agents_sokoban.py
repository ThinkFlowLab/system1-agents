"""Sokoban transitions, level selection, the offline agent loop, and the visual mode."""

import json
import struct
import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, skipUnless
from unittest.mock import patch

from s1a.agents import sokoban
from s1a.agents._sokoban import parse_ascii
from s1a.decision_models import RandomModel, RuleModel, ScriptedModel
from s1a.run import started_runner
from s1a.tool import loop, series

try:
    from s1a.agents import _sokoban_render
except ImportError:  # the visual extra is not installed
    _sokoban_render = None


def level(text, max_steps=200, **fields):
    board, state = parse_ascii(text)
    row = {"level_id": "fixture", "board": board.to_dict(), "initial_state": state.to_dict(), "max_steps": max_steps}
    row.update(fields)
    return row


SIMPLE = "#####\n#@$.#\n#####"


class TextOnlyModel(ScriptedModel):
    """The stand-in for every text-only backend — jev, clm, laya, laya-served, Cua Nano, Cua 4B in text modality —
    whose `supports_images` is False; the real classes need their extras, and the guard reads only the attribute."""

    supports_images = False


def png_size(data):
    """The width and height out of a PNG's IHDR chunk."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


class TestSokoban(IsolatedAsyncioTestCase):
    async def test_push_win_reset_and_all_four_candidates(self):
        env = sokoban.SokobanEnv(level(SIMPLE))
        await env.reset()
        self.assertEqual(set(await env.candidates()), {"up", "down", "left", "right"})
        before = await env.observe()
        await env.step("up")  # Wall moves consume the budget.
        self.assertEqual((await env.observe())["board"], before["board"])
        self.assertEqual(env.game.steps, 1)
        await env.step("right")
        self.assertTrue(env.done)
        self.assertEqual(env.score, 1)
        self.assertEqual(await env.candidates(), {})
        with self.assertRaises(RuntimeError):
            await env.step("right")
        await env.reset()
        self.assertEqual(await env.observe(), before)
        self.assertEqual(env.score, 0)

    async def test_cannot_push_two_boxes_or_pull(self):
        env = sokoban.SokobanEnv(level("########\n# @$$..#\n########"))
        await env.reset()
        boxes = env.game.state.boxes
        await env.step("right")
        self.assertEqual(env.game.state.boxes, boxes)
        await env.step("left")
        self.assertEqual(env.game.state.boxes, boxes)

    async def test_budget_termination_and_unknown_action(self):
        env = sokoban.SokobanEnv(level(SIMPLE, max_steps=1))
        await env.reset()
        with self.assertRaises(ValueError):
            await env.step("undo")
        self.assertEqual(env.game.steps, 0)
        await env.step("up")
        self.assertTrue(env.done)
        self.assertEqual(env.score, 0)
        self.assertEqual(await env.candidates(), {})

    async def test_offline_deepagent_completes_and_records_text_board(self):
        env = sokoban.SokobanEnv(level(SIMPLE))
        async with started_runner():
            episode = await loop.run_episode(
                sokoban.SPEC,
                env,
                model_name="rule",
                seed=0,
                chat=None,
                decision_model=RuleModel("test-right", lambda state, candidates: "right"),
                rethink_on=False,
                max_acts=3,
                timeout_s=30,
                prices=None,
                log=False,
            )
        sokoban.annotate(env, episode)
        self.assertIsNone(episode.error)
        self.assertEqual((episode.score, episode.steps), (1, 1))
        self.assertEqual(episode.extra["sokoban"]["observation_mode"], "text")
        self.assertIn("board", episode.views[0]["state"])
        self.assertNotIn("solution", json.dumps(episode.views))


class TestSokobanLevels(TestCase):
    def flags(self, *extra):
        return series.parser(sokoban.SPEC).parse_args(
            ["--model", "random", "--rethink", "off", "--episodes", "1", *extra]
        )

    def test_level_selection_does_not_wrap_and_honors_budget(self):
        run = sokoban.make_series(self.flags("--seed", "99", "--max-steps", "7"))
        env = run.env_for(99)
        self.assertEqual(env.game.max_steps, 7)
        for args in [
            ("--seed", "-1"),
            ("--seed", "100"),
            ("--seed", "99", "--episodes", "2"),
            ("--rethink", "on"),
            ("--model", "rule"),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sokoban.make_series(self.flags(*args))

    def test_all_imported_levels_replay_reference_solutions(self):
        # Evaluator-only references never enter the agent's observations or baseline.
        path = Path(__file__).parent / "fixtures/sokoban_reference.jsonl"
        references = {r["level_id"]: r for r in map(json.loads, path.read_text().splitlines())}
        levels = sokoban.load_levels()
        self.assertEqual(len(levels), 100)
        self.assertEqual(set(references), {r["level_id"] for r in levels})
        for row in levels:
            game = sokoban.SokobanEnv(row).game
            for action in references[row["level_id"]]["solution_actions"]:
                game.step(action)
            self.assertTrue(game.board.solved(game.state), row["level_id"])
            self.assertEqual(game.steps, references[row["level_id"]]["optimal_distance"])


@skipUnless(_sokoban_render, "the visual extra (Pillow)")
class TestSokobanVisual(IsolatedAsyncioTestCase):
    async def test_the_picture_replaces_the_board_and_follows_the_level(self):
        env = sokoban.SokobanEnv(level(SIMPLE, language="zh", render={"theme": "warm", "tile_size": 24}), visual=True)
        await env.reset()
        state = await env.observe()
        self.assertNotIn("board", state)
        self.assertEqual(state["board_size"], [5, 3])
        self.assertEqual(state["render"], {"theme": "warm", "tile_size": 24})
        self.assertEqual(state["language"], "zh")
        self.assertEqual(await env.candidates(), {"up": "向上", "down": "向下", "left": "向左", "right": "向右"})
        self.assertEqual(env.rules(), sokoban.VISUAL_RULES["zh"])
        self.assertTrue(env.requires_images())
        (image,) = await env.images()
        self.assertEqual(image.media_type, "image/png")
        self.assertEqual(png_size(image.data), (5 * 24, 3 * 24))

    async def test_a_blocked_move_renders_the_same_board_and_a_push_changes_it(self):
        env = sokoban.SokobanEnv(level(SIMPLE), visual=True)
        await env.reset()
        still = (await env.images())[0].data
        await env.step("up")  # into the wall: the board does not change
        self.assertEqual((await env.images())[0].data, still)
        await env.step("right")  # the winning push
        self.assertNotEqual((await env.images())[0].data, still)

    async def test_text_envs_keep_the_spec_rules_and_move_labels(self):
        env = sokoban.SokobanEnv(level(SIMPLE, language="zh"))
        await env.reset()
        self.assertEqual(env.rules(), sokoban.RULES)
        self.assertFalse(env.requires_images())
        self.assertEqual(
            await env.candidates(), {action: f"Move {action}" for action in ("up", "down", "left", "right")}
        )
        self.assertEqual(await env.images(), ())

    async def test_offline_deepagent_decides_over_the_picture(self):
        env = sokoban.SokobanEnv(level(SIMPLE), visual=True)
        model = ScriptedModel(choose="right")
        async with started_runner():
            episode = await loop.run_episode(
                sokoban.SPEC,
                env,
                model_name="rule",
                seed=0,
                chat=None,
                decision_model=model,
                rethink_on=False,
                max_acts=3,
                timeout_s=30,
                prices=None,
                log=False,
            )
        sokoban.annotate(env, episode)
        self.assertIsNone(episode.error)
        self.assertEqual((episode.score, episode.steps), (1, 1))
        self.assertEqual(episode.extra["sokoban"]["observation_mode"], "visual")
        observation, questions = model.calls[0]
        self.assertEqual(len(observation.images), 1)
        self.assertEqual(observation.images[0].media_type, "image/png")
        self.assertNotIn("board", observation.state)
        self.assertEqual(questions["pick"].rules, (sokoban.VISUAL_RULES["en"],))

    async def test_a_text_only_backend_is_refused_before_any_decision(self):
        env = sokoban.SokobanEnv(level(SIMPLE), visual=True)
        model = TextOnlyModel(choose="right")  # jev, clm, laya, laya-served, Cua Nano, Cua 4B in text modality
        with self.assertRaisesRegex(ValueError, "reads text only"):
            async with started_runner():
                await loop.run_episode(
                    sokoban.SPEC,
                    env,
                    model_name="jev",
                    seed=0,
                    chat=None,
                    decision_model=model,
                    rethink_on=False,
                    max_acts=3,
                    timeout_s=30,
                    prices=None,
                    log=False,
                )
        self.assertEqual(model.calls, [])  # refused before the backend was ever asked

    async def test_random_stays_the_offline_smoke_exception(self):
        env = sokoban.SokobanEnv(level(SIMPLE, max_steps=3), visual=True)
        async with started_runner():
            episode = await loop.run_episode(
                sokoban.SPEC,
                env,
                model_name="random",
                seed=0,
                chat=None,
                decision_model=RandomModel(seed=0),
                rethink_on=False,
                max_acts=3,
                timeout_s=30,
                prices=None,
                log=False,
            )
        self.assertIsNone(episode.error)
        sokoban.annotate(env, episode)
        self.assertEqual(episode.extra["sokoban"]["observation_mode"], "visual")

    async def test_the_reference_solutions_replay_over_rendered_boards(self):
        path = Path(__file__).parent / "fixtures/sokoban_reference.jsonl"
        references = {r["level_id"]: r["solution_actions"] for r in map(json.loads, path.read_text().splitlines())}
        run = sokoban.make_series(self.flags_for_visual())
        self.assertTrue(run.env_for(0).visual)
        for row in sokoban.load_levels()[:5]:
            env = sokoban.SokobanEnv(row, visual=True)
            await env.reset()
            first = (await env.images())[0].data
            for action in references[env.level_id]:
                await env.step(action)
                self.assertTrue((await env.images())[0].data.startswith(b"\x89PNG"))
            self.assertEqual(env.score, 1.0)
            await env.reset()
            self.assertEqual((await env.images())[0].data, first)

    @staticmethod
    def flags_for_visual():
        return series.parser(sokoban.SPEC).parse_args(
            ["--model", "random", "--rethink", "off", "--episodes", "1", "--visual"]
        )


class TestVisualFlags(TestCase):
    def flags(self, *extra):
        return series.parser(sokoban.SPEC).parse_args(
            ["--model", "random", "--rethink", "off", "--episodes", "1", *extra]
        )

    def test_visual_needs_an_image_model_and_names_the_missing_extra(self):
        with self.assertRaisesRegex(ValueError, "image"):
            sokoban.make_series(self.flags("--visual", "--model", "llm"))
        with patch.dict(sys.modules, {"s1a.agents._sokoban_render": None}):
            with self.assertRaisesRegex(ValueError, "uv sync --extra visual"):
                sokoban.make_series(self.flags("--visual"))
