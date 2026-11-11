"""Sokoban on Valen's selected 100 evaluation levels: text boards, or the same boards rendered per move."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any, Callable

from s1a.agents._sokoban import ACTIONS, Board, State, SokobanEnv as Game
from s1a.decision_models.types import Image
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
# Valen evaluation/sokoban/dataset.py at c96c4f73, verbatim: the instructions the preview checkpoint read.
VISUAL_RULES = {
    "en": (
        "Legend: dark walls, light floor, green rings are goals, crossed squares are boxes, and the round face "
        "is the player. Goal rings remain visible under boxes and the player. Move one cell up, down, left or "
        "right. You can push one box, but cannot pull, pass through walls, or push two boxes together. Put "
        "every box on a goal using the fewest total moves. An invalid move consumes a step without changing "
        "the board."
    ),
    "zh": (
        "图例：深色墙，浅色地板，绿色圆环为目标，带叉方块为箱子，圆脸为玩家；箱子和玩家下方的目标环仍可见。"
        "每次向上下左右移动一格，可以推动一个箱子，不能拉箱、穿墙或连推两个箱子。"
        "请用最少移动步数将所有箱子推到目标。无效动作消耗一步但不改变棋盘。"
    ),
}
DIRECTION_NAMES = {
    "en": dict(zip(ACTIONS, ("Up", "Down", "Left", "Right"))),
    "zh": dict(zip(ACTIONS, ("向上", "向下", "向左", "向右"))),
}


def load_levels() -> list[dict[str, Any]]:
    return [json.loads(line) for line in LEVELS.read_text(encoding="utf-8").splitlines()]


def require_renderer() -> Callable[..., Any]:
    """Valen's ``render``, imported only on the visual path so text runs need no Pillow."""
    try:
        from s1a.agents._sokoban_render import render
    except ImportError as exc:
        raise ValueError(f"sokoban --visual needs Pillow ({exc}); uv sync --extra visual") from exc
    return render


class SokobanEnv:
    def __init__(self, level: dict[str, Any], max_steps: int = 200, *, visual: bool = False) -> None:
        self.level_id = level["level_id"]
        self.visual = visual
        self.language = level.get("language") or "en"
        render = level.get("render") or {}
        self.theme, self.tile_size = render.get("theme", "classic"), render.get("tile_size", 44)
        if visual and self.language not in VISUAL_RULES:
            raise ValueError(f"unknown level language {self.language!r}")
        self._render = require_renderer() if visual else None
        self.game = Game(
            Board.from_dict(level["board"]),
            State.from_dict(level["initial_state"]),
            max_steps=min(max_steps, level["max_steps"]),
        )

    async def reset(self) -> None:
        self.game.reset()

    async def observe(self) -> dict[str, Any]:
        if not self.visual:  # the ASCII board never rides with the picture: the checkpoint read images only
            return {
                "board": "\n".join(self.game.board.ascii(self.game.state)),
                "steps_remaining": self.game.max_steps - self.game.steps,
            }
        return {
            "steps_remaining": self.game.max_steps - self.game.steps,
            "board_size": [self.game.board.width, self.game.board.height],
            "render": {"theme": self.theme, "tile_size": self.tile_size},
            "language": self.language,
        }

    async def candidates(self) -> dict[str, str]:
        if self.done:
            return {}
        return dict(DIRECTION_NAMES[self.language]) if self.visual else {action: f"Move {action}" for action in ACTIONS}

    def rules(self) -> str:
        return VISUAL_RULES[self.language] if self.visual else RULES

    def requires_images(self) -> bool:
        return self.visual

    async def images(self) -> tuple[Image, ...]:
        if self._render is None:
            return ()
        picture = self._render(self.game.board, self.game.state, self.theme, self.tile_size)
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        return (Image(buffer.getvalue(), "image/png"),)

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
        "observation_mode": "visual" if env.visual else "text",
        "solved": bool(env.score),
        "pushes": env.game.pushes,
        "step_limit_reached": env.game.steps >= env.game.max_steps and not env.score,
    }


def add_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Valen's rendered board instead of ASCII: one PNG per move, rules per level language; "
        "needs an image-reading model and uv sync --extra visual",
    )


def make_series(flags: argparse.Namespace) -> Series:
    if flags.rethink != "off":
        raise ValueError("sokoban uses unassisted decisions; use --rethink off")
    if flags.model == "rule":
        raise ValueError("sokoban has no rule baseline; use --model random, jev, laya, cua or llm")
    if flags.visual and flags.model == "llm":
        raise ValueError("sokoban --visual needs a model that reads images; --model llm reads tool text only")
    if flags.visual:
        require_renderer()
    levels = load_levels()
    if flags.seed < 0 or flags.seed + flags.episodes > len(levels):
        raise ValueError(f"sokoban requires 0 <= --seed and --seed + --episodes <= {len(levels)}")
    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=lambda seed: SokobanEnv(levels[seed], max_steps=flags.max_steps, visual=flags.visual),
        session=contextlib.nullcontext(),
        baseline=None,
        annotate=annotate,
    )


SPEC = ToolAgentSpec(
    name="sokoban",
    description=(
        "Sokoban on Valen's selected 100 levels as a text board, or with --visual as Valen's rendered image; "
        "score is 1 when solved, otherwise 0."
    ),
    rules=RULES,
    budget=Budget(max_steps=200, timeout_s=1200, stall_after=0),
    flags=add_flags,
    series=make_series,
)
