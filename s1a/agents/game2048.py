# coding: utf-8
"""2048 (the original MIT game, self-hosted): ``s1a run game2048 --model jev --rethink on --episodes 10``."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from s1a.config import ROOT
from s1a.env import Env
from s1a.jobs import SHOWCASE_DIR, Episode
from s1a.spec import Budget, Series, ToolAgentSpec
from s1a.tool.hands import BrowserHands, Frames, Hands, serve_static, serving

GAME_DIR = ROOT / "evals" / "2048"
KEYS = {"UP": "ArrowUp", "DOWN": "ArrowDown", "LEFT": "ArrowLeft", "RIGHT": "ArrowRight"}
RULES = (
    "2048: tiles slide as far as they can in the chosen direction and equal neighbours merge into their sum. "
    "Keep the largest tile in one corner and build a monotone chain along that edge; prefer moves that merge tiles "
    "or keep the corner intact; avoid a move that pulls the largest tile out of its corner or leaves the board scattered."
)
READ_STATE = r"""
(args) => {
  const raw = localStorage.getItem('gameState');
  const msg = document.querySelector('.game-message').className;
  const scoreEl = document.querySelector('.score-container');
  const score = parseInt((scoreEl.childNodes[0] || {}).textContent) || 0;
  const rows = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  if (!raw) {
    // The game clears its stored state on game over (a win keeps it); the rendered tiles still hold the board.
    document.querySelectorAll('.tile-container .tile').forEach(el => {
      const at = /tile-position-(\d)-(\d)/.exec(el.className);
      const value = parseInt((el.querySelector('.tile-inner') || {}).textContent) || 0;
      if (at) rows[at[2] - 1][at[1] - 1] = Math.max(rows[at[2] - 1][at[1] - 1], value);
    });
    return JSON.stringify({grid: rows, score, over: msg.includes('game-over'), won: msg.includes('game-won')});
  }
  const s = JSON.parse(raw);
  s.grid.cells.forEach((col, x) => col.forEach((cell, y) => { if (cell) rows[y][x] = cell.value; }));
  return JSON.stringify({grid: rows, score: s.score, over: s.over, won: s.won});
}
"""
# The game draws every new tile from Math.random; a seeded mulberry32 in its place makes an episode's tile
# sequence a function of the seed and the moves, so every model plays the same boards.
SEED_RANDOM = """
(seed) => {
  let s = seed >>> 0;
  Math.random = () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return 'ok';
}
"""


class Game2048Env:
    def __init__(self, hands: Hands, seed: int, *, frames_dir: Path | None) -> None:
        self._hands = hands
        self._seed = seed
        self._state: dict[str, Any] = {}
        self._stuck: set[str] = set()
        self._frames = Frames(hands, frames_dir) if frames_dir is not None else None  # --showcase only

    @property
    def frames_dir(self) -> Path | None:
        return self._frames.directory if self._frames is not None else None

    async def reset(self) -> None:
        await self._hands.wait_ready(".restart-button", 10.0)
        await self._hands.read(SEED_RANDOM, self._seed)
        await self._hands.read("(args) => { document.querySelector('.restart-button').click(); return 'ok'; }", {})
        await asyncio.sleep(0.3)
        self._state = await self._hands.read(READ_STATE, {})
        self._stuck = set()
        if self._frames is not None:
            await self._frames.reset()

    async def observe(self) -> dict[str, Any]:
        grid = self._state["grid"]
        flat = [value for row in grid for value in row]
        largest = max(flat)
        positions = [(y, x) for y, row in enumerate(grid) for x, value in enumerate(row) if value == largest]
        corners = [at for at in positions if at[0] in (0, 3) and at[1] in (0, 3)]
        return {
            "grid_rows_top_to_bottom": grid,
            "score": self._state["score"],
            "largest_tile": largest,
            "largest_tile_row_col": list((corners or positions)[0]),  # a tie prefers the corner the rules ask for
            "empty_cells": flat.count(0),
            "moves_that_did_nothing_last_time": sorted(self._stuck),
            # The rethink rail's stall digest: the score alone, so six merge-free moves count as a stall.
            "progress": {"score": self._state["score"], "largest_tile": largest},
        }

    async def candidates(self) -> dict[str, str]:
        if self.done:
            return {}
        return {key: f"slide every tile {key.lower()}" for key in KEYS if key not in self._stuck}

    async def step(self, key: str) -> None:
        before = self._state["grid"]
        await self._hands.press(KEYS[key])
        await asyncio.sleep(0.15)
        self._state = await self._hands.read(READ_STATE, {})
        if self._state["grid"] == before:
            self._stuck.add(key)
        else:
            self._stuck = set()
        if self._frames is not None:
            await self._frames.take()

    @property
    def done(self) -> bool:
        return bool(self._state.get("over")) or bool(self._state.get("won")) or len(self._stuck) == 4

    @property
    def score(self) -> float:
        return float(self._state.get("score", 0))


def priority_rule(state: dict[str, Any], candidates: dict[str, str]) -> str:
    for key in ("DOWN", "LEFT", "RIGHT", "UP"):
        if key in candidates:
            return key
    return next(iter(candidates))


def make_series(flags: argparse.Namespace) -> Series:
    site = serve_static(GAME_DIR)
    hands = BrowserHands(headless=not flags.headed)
    url = f"http://127.0.0.1:{site.server_address[1]}/index.html"

    def frames_dir_for(seed: int) -> Path | None:
        return SHOWCASE_DIR / "frames" / f"2048-{seed}" if flags.showcase else None

    def annotate(env: Env, episode: Episode) -> None:
        episode.extra["largest"] = episode.final_state["largest_tile"]
        episode.frames_dir = frames_dir_for(episode.seed)

    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=lambda seed: Game2048Env(hands, seed, frames_dir=frames_dir_for(seed)),
        session=serving(site, hands.session(url, ready_selector=".restart-button", timeout_s=15.0)),
        baseline=("priority", priority_rule),
        annotate=annotate,
    )


SPEC = ToolAgentSpec(
    name="game2048",
    description="2048 on the original MIT page: score and largest tile at a move cap.",
    rules=RULES,
    budget=Budget(max_steps=300, timeout_s=600, stall_after=6),
    flags=lambda parser: None,
    series=make_series,
)
