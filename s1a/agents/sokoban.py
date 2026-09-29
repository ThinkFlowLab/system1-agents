"""Text-board Sokoban using Valen's selected 100 evaluation levels."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

from s1a.agents._sokoban import ACTIONS, Board, State, SokobanEnv as Game
from s1a.env import Env
from s1a.jobs import Episode
from s1a.spec import Budget, Series, ToolAgentSpec

LEVELS = Path(__file__).with_name("_data") / "sokoban_levels.jsonl"
RULES = (
    "Push every box onto a goal. Board legend: # wall, space floor, . goal, @ player, "
    "$ box, * box on goal, + player on goal. Up/down change row; left/right change column. "
    "You can push one box into an empty floor or goal, but cannot pull or push two boxes. "
    "All four directions are offered; blocked moves consume a step. Avoid trapping boxes away from goals."
)


def load_levels() -> list[dict[str, Any]]:
    return [json.loads(line) for line in LEVELS.read_text(encoding="utf-8").splitlines()]


class SokobanEnv:
    def __init__(self, level: dict[str, Any], max_steps: int = 200) -> None:
        self.level_id = level["level_id"]
        self.game = Game(
            Board.from_dict(level["board"]),
            State.from_dict(level["initial_state"]),
            max_steps=min(max_steps, level["max_steps"]),
        )

    async def reset(self) -> None:
        self.game.reset()

    async def observe(self) -> dict[str, Any]:
        return {
            "board": "\n".join(self.game.board.ascii(self.game.state)),
            "steps_remaining": self.game.max_steps - self.game.steps,
        }

    async def candidates(self) -> dict[str, str]:
        return {} if self.done else {action: f"Move {action}" for action in ACTIONS}

    async def step(self, key: str) -> None:
        self.game.step(key)

    @property
    def done(self) -> bool:
        return self.game.board.solved(self.game.state) or self.game.steps >= self.game.max_steps

    @property
    def score(self) -> float:
        return float(self.game.board.solved(self.game.state))


def annotate(env: Env, episode: Episode) -> None:
    assert isinstance(env, SokobanEnv)
    episode.extra["sokoban"] = {
        "level_id": env.level_id,
        "observation_mode": "text",
        "solved": bool(env.score),
        "pushes": env.game.pushes,
        "step_limit_reached": env.game.steps >= env.game.max_steps and not env.score,
    }


def make_series(flags: argparse.Namespace) -> Series:
    if flags.rethink != "off":
        raise ValueError("sokoban uses unassisted decisions; use --rethink off")
    if flags.model == "rule":
        raise ValueError("sokoban has no rule baseline; use --model random, jev, laya, cua or llm")
    levels = load_levels()
    if flags.seed < 0 or flags.seed + flags.episodes > len(levels):
        raise ValueError(f"sokoban requires 0 <= --seed and --seed + --episodes <= {len(levels)}")
    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=lambda seed: SokobanEnv(levels[seed], max_steps=flags.max_steps),
        session=contextlib.nullcontext(),
        baseline=None,
        annotate=annotate,
    )


SPEC = ToolAgentSpec(
    name="sokoban",
    description="Text-board Sokoban on Valen's selected 100 levels; score is 1 when solved, otherwise 0.",
    rules=RULES,
    budget=Budget(max_steps=200, timeout_s=1200, stall_after=0),
    flags=lambda parser: None,
    series=make_series,
)
