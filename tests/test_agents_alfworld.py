# coding: utf-8
"""The ALFWorld adapter on one shipped game: file order, the oracle's moves, and what counts as a checked-empty place."""

from __future__ import annotations

from argparse import Namespace
from unittest import IsolatedAsyncioTestCase, skipUnless
from unittest.mock import patch

try:
    from s1a.agents.alfworld import AlfworldEnv, make_series, oracle_rule, solvable_game_files

    HAVE_ALFWORLD = bool(solvable_game_files())
except (ModuleNotFoundError, ValueError):  # the alfworld extra or ALFWORLD_DATA is missing
    HAVE_ALFWORLD = False


@skipUnless(HAVE_ALFWORLD, "the alfworld extra and its data are not installed")
class TestAlfworldEnv(IsolatedAsyncioTestCase):
    async def test_seed_order_the_oracle_and_the_checked_places(self) -> None:
        files = solvable_game_files()
        env = AlfworldEnv([files[1], files[0]], 50, every_command=True)
        rule = oracle_rule(env)
        await env.reset()
        self.assertEqual(env.game_file, files[1], "seed i plays files[i], no shuffle")
        self.assertEqual(rule({}, await env.candidates()), "look", "the expert observes before it plans")
        closed = [
            c for c in await env.candidates() if c.startswith("go to ") and any(w in c for w in ("drawer", "cabinet"))
        ]
        await env.step(closed[0])
        place = closed[0][len("go to ") :]
        state = await env.observe()
        self.assertEqual(state["progress"]["at"], place)
        if "is closed" in env.history[-1]["observation"]:
            self.assertNotIn(place, state["places_already_checked_and_empty"], "closed is not checked")
            await env.step(f"open {place}")
            state = await env.observe()
            self.assertEqual(state["progress"]["at"], place, "opening does not move the agent")
            self.assertEqual(
                place in state["places_already_checked_and_empty"], "nothing" in env.history[-1]["observation"]
            )
        with self.assertRaises(RuntimeError):
            rule({}, {"go to nowhere": ""})

    async def test_the_oracle_stops_when_the_expert_falls_back_to_look(self) -> None:
        env = AlfworldEnv([solvable_game_files()[0]], 50, every_command=True)
        rule = oracle_rule(env)
        await env.reset()
        await env.step(rule({}, await env.candidates()))
        self.assertNotEqual(env.expert_next, "look", "after the first look the expert has a plan")
        env._info["extra.expert_plan"] = ["look"]  # the wrapper's plan when the expert's move is not admissible
        with self.assertRaises(RuntimeError):
            rule({}, await env.candidates())

    async def test_more_episodes_than_games_is_an_error(self) -> None:
        flags = Namespace(offset=0, stride=1, episodes=3, max_steps=50, model_name="rule")
        with patch("s1a.agents.alfworld.solvable_game_files", return_value=["a/game.tw-pddl", "b/game.tw-pddl"]):
            with self.assertRaisesRegex(ValueError, "--episodes 3 but only 2 games"):
                make_series(flags)

    async def test_the_decision_models_see_no_look_or_examine(self) -> None:
        env = AlfworldEnv([solvable_game_files()[0]], 50, every_command=False)
        await env.reset()
        self.assertFalse([c for c in await env.candidates() if c.startswith(("look", "examine", "inventory"))])
