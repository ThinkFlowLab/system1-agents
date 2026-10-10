"""Sokoban transitions, level selection, and the offline agent loop."""

import json
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from s1a.agents import sokoban
from s1a.agents._sokoban import parse_ascii
from s1a.decision_models import RuleModel
from s1a.run import started_runner
from s1a.tool import loop, series


def level(text, max_steps=200):
    board, state = parse_ascii(text)
    return {"level_id": "fixture", "board": board.to_dict(), "initial_state": state.to_dict(), "max_steps": max_steps}


SIMPLE = "#####\n#@$.#\n#####"


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
