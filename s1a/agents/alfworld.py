# coding: utf-8
"""ALFWorld text games (TextWorld): ``s1a run alfworld --model jev --rethink on --episodes 10``.

Needs the ``alfworld`` and ``textworld`` packages and ``ALFWORLD_DATA`` (Python 3.11; see the README).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import random
import re
import sys
from typing import Any

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType
from alfworld.info import ALFWORLD_DATA

from s1a.env import Env
from s1a.jobs import Episode
from s1a.spec import Budget, Series, ToolAgentSpec

RULES = (
    "A household task in one room, played through admissible text commands. Work in phases: first find the "
    "target object by going to the places where such an object usually is (countertops, tables, shelves, "
    "then cabinets, drawers, fridge for food and drink); open a closed receptacle before looking inside; take the "
    "object; then do any processing the task names (clean at a sinkbasin, heat with a microwave, cool with a "
    "fridge, look at it under a desklamp by using the lamp); finally put it in or on the target receptacle. For a look-at or examine-with-lamp task you must be holding the named object when you use the desklamp. "
    "Never repeat a 'go to' that already showed the place empty of the target; do not waste steps on look, "
    "inventory or examine unless nothing else is sensible."
)
_TASK_RE = re.compile(r"Your task is to: (.+)")


def solvable_game_files() -> list[str]:
    """The shipped game files ALFWorld itself evaluates on: one per trial, flagged solvable, no movable or sliced tasks."""
    pattern = os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid_unseen", "**", "game.tw-pddl")
    files = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        if "movable" in path or "Sliced" in path:
            continue
        with open(path, encoding="utf-8") as handle:
            if json.load(handle).get("solvable") is True:
                files.append(path)
    return files


class AlfworldEnv:
    """All selected games are registered once; every ``reset()`` loads the next game (TextWorld's batch env)."""

    def __init__(self, game_files: list[str], max_steps: int, *, every_command: bool) -> None:
        """``every_command`` offers look, examine and inventory too: what the oracle needs to follow the expert."""
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["expert_plan", "gamefile"])
        env_id = textworld.gym.register_games(
            game_files,
            infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=max_steps + 5,
            wrappers=[AlfredDemangler(), AlfredExpert(expert_type=AlfredExpertType.HANDCODED)],
        )
        self._env = textworld.gym.make(env_id)
        self._loaded: list[str] = []
        # ponytail: the gym env shuffles and keeps the drawn game private; play the files in the given order and
        # record each as the iterator yields it, so seed i is files[i].
        self._env._gamefiles_iterator = self._tracking(list(game_files))
        self._every_command = every_command
        self._max_steps = max_steps
        self._info: dict[str, Any] = {}
        self._history: list[dict[str, str]] = []
        self._holding = ""
        self._at = ""
        self._visited_empty: list[str] = []
        self._won = False
        self._task = ""
        self.game_file = ""
        self.game_name = ""

    def _tracking(self, source: Any) -> Any:
        for game_file in source:
            self._loaded.append(str(game_file))
            yield game_file

    async def reset(self) -> None:
        random.seed(len(self._loaded))  # the handcoded expert draws from the global random module
        obs, infos = self._env.reset()
        text = obs[0]
        self._info = {key: value[0] for key, value in infos.items()}
        match = _TASK_RE.search(text)
        self._task = match.group(1).strip() if match else ""
        self.game_file = self._loaded[-1] if self._loaded else ""
        self.game_name = os.path.basename(os.path.dirname(self.game_file))
        self._history = [{"action": "start", "observation": text.split("Your task is to")[0].strip()}]
        self._holding, self._at, self._visited_empty, self._won = "", "", [], False

    @property
    def history(self) -> list[dict[str, str]]:
        """Every action and its full observation, the start entry first."""
        return list(self._history)

    @property
    def expert_next(self) -> str:
        plan = self._info.get("extra.expert_plan") or []
        return str(plan[0]) if plan else ""

    async def observe(self) -> dict[str, Any]:
        return {
            "task": self._task,
            "holding": self._holding or "nothing",
            "places_already_checked_and_empty": self._visited_empty[-8:],
            "recent_steps": [
                {"action": h["action"], "observation": h["observation"][: 600 if h["action"] == "start" else 300]}
                for h in self._history[-6:]
            ],
            "steps_used": len(self._history) - 1,
            "won": self._won,
            # The rethink rail's stall digest: where the agent is, what it holds and what it last saw, without
            # the step history; it repeats on a revisit or a "Nothing happens", not on open, heat or clean.
            "progress": {
                "holding": self._holding,
                "at": self._at,
                "checked": sorted(self._visited_empty),
                "seen": self._history[-1]["observation"][:200],
            },
        }

    async def candidates(self) -> dict[str, str]:
        if self.done:
            return {}
        commands = list(self._info["admissible_commands"])
        if self._every_command:
            return {command: "" for command in commands}
        moves = [command for command in commands if not command.startswith(("examine", "look", "inventory"))]
        return {command: "" for command in (moves or commands)}

    async def step(self, key: str) -> None:
        obs, _scores, _dones, infos = self._env.step([key])
        text = obs[0].strip()
        self._info = {name: value[0] for name, value in infos.items()}
        self._history.append({"action": key, "observation": text})
        if key.startswith("take ") and "You pick up" in text:
            self._holding = key[len("take ") :].split(" from ")[0]
        elif key.startswith(("move ", "put ")) and ("You move" in text or "You put" in text):
            self._holding = ""
        if key.startswith("go to ") and "Nothing happens" not in text:
            self._at = key[len("go to ") :]
        if key.startswith(("go to ", "open ")) and "you see nothing" in text:
            place = key.split(" ", 2)[2] if key.startswith("go to ") else key[len("open ") :]
            if place not in self._visited_empty:
                self._visited_empty.append(place)
        self._won = bool(self._info.get("won"))

    @property
    def done(self) -> bool:
        return self._won or len(self._history) - 1 >= self._max_steps

    @property
    def score(self) -> float:
        return 1.0 if self._won else 0.0


def oracle_rule(env: AlfworldEnv):
    """Follows the expert plan shipped with each game: the upper bound. It stops when the expert has no move."""

    def rule(state: dict[str, Any], candidates: dict[str, str]) -> str:
        key = env.expert_next
        # The expert's first move is always look; afterwards look is the wrapper's plan when it has none.
        if key == "look" and len(env.history) > 1:
            raise RuntimeError("oracle: the expert has no move for this state (its plan fell back to look)")
        if key not in candidates:
            raise RuntimeError(f"oracle: the expert's move {key!r} is not among the admissible commands")
        return key

    return rule


def flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--offset", type=int, default=0, help="first game, in file order")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="every Nth game from the offset; 11 spreads twelve games over the six task types",
    )


def make_series(flags: argparse.Namespace) -> Series:
    """The seeds are game indices; one batch env serves every episode, so the seed picks nothing by itself."""
    files = solvable_game_files()
    indices = list(range(flags.offset, len(files), flags.stride))[: flags.episodes]
    if len(indices) < flags.episodes:
        raise ValueError(
            f"--episodes {flags.episodes} but only {len(indices)} games from --offset {flags.offset} with "
            f"--stride {flags.stride} ({len(files)} solvable games)"
        )
    env = AlfworldEnv([files[index] for index in indices], flags.max_steps, every_command=flags.model == "rule")

    def annotate(_env: Env, episode: Episode) -> None:
        episode.extra["game"] = env.game_name
        episode.extra["game_file"] = env.game_file  # traj_data.json beside it feeds evals.replay.thor_replay
        episode.extra["history"] = env.history
        print(
            f"{'WON ' if episode.score else 'lost'} {episode.steps:>3} steps  {episode.final_state['task']}",
            file=sys.stderr,
        )

    return Series(
        seeds=indices,
        env_for=lambda seed: env,
        session=contextlib.nullcontext(),
        baseline=("oracle", oracle_rule(env)),
        annotate=annotate,
    )


SPEC = ToolAgentSpec(
    name="alfworld",
    description="ALFWorld household tasks in text; success on unseen games is the score.",
    rules=RULES,
    budget=Budget(max_steps=50, timeout_s=300, stall_after=8),
    flags=flags,
    series=make_series,
)
