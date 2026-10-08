# coding: utf-8
"""Nim, the template for a tool-front agent: copy this file to ``s1a/agents/<name>.py`` and replace each part.

Two players alternately take 1, 2 or 3 stones from one pile; whoever takes the last stone wins. The agent moves first
and a fixed opponent answers with the winning reply whenever one exists. ``s1a run <name> --model random
--rethink off --episodes 3`` plays it without any key; ``--model rule`` plays the winning strategy.
"""

from __future__ import annotations

import argparse
import contextlib
import random
from typing import Any

from s1a.spec import Budget, Series, ToolAgentSpec

PILE = 10
# The rules text: facts Jev recognises, under about 120 words. It names the winning shape, so no arithmetic is asked of
# Jev beyond reading the pile size; the candidate descriptions below say what each move leaves.
RULES = (
    "Nim: two players alternately take 1, 2 or 3 stones from one pile; whoever takes the last stone wins. "
    "A move that leaves the opponent 4 or 8 stones wins. From 4 or 8 stones no move is safe; take 1 and wait."
)


class NimEnv:
    """The mechanics only: reset, observe, candidates, step, done, score. Rules, budgets and names live in SPEC."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._pile = PILE
        self._moves: list[int] = []
        self._won: bool | None = None

    async def reset(self) -> None:
        self._pile, self._moves, self._won = PILE, [], None

    async def observe(self) -> dict[str, Any]:
        # ``progress`` is what the rethink rail compares between steps; histories and counters stay outside it.
        return {"pile": self._pile, "moves_so_far": list(self._moves), "progress": {"pile": self._pile}}

    async def candidates(self) -> dict[str, str]:
        # Human-readable keys, one line each, saying what the move leaves; empty once the game is over.
        if self.done:
            return {}
        return {f"take_{n}": f"take {n}, leaving {self._pile - n} stones" for n in (1, 2, 3) if n <= self._pile}

    async def step(self, key: str) -> None:
        take = int(key.rsplit("_", 1)[1])
        self._pile -= take
        self._moves.append(take)
        if self._pile == 0:
            self._won = True
            return
        reply = self._pile % 4 or self._rng.choice([n for n in (1, 2, 3) if n <= self._pile])
        self._pile -= reply
        self._moves.append(-reply)
        if self._pile == 0:
            self._won = False

    @property
    def done(self) -> bool:
        return self._won is not None

    @property
    def score(self) -> float:
        return 1.0 if self._won else 0.0


def winning_rule(state: dict[str, Any], candidates: dict[str, str]) -> str:
    """The hand-written baseline, when one is known: leave a multiple of 4."""
    leave = state["pile"] % 4
    key = f"take_{leave}"
    return key if leave and key in candidates else next(iter(candidates))


def make_series(flags: argparse.Namespace) -> Series:
    """Everything one run needs, built from the parsed flags; ``session`` opens a page for browser-backed games."""
    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=lambda seed: NimEnv(seed),
        session=contextlib.nullcontext(),
        baseline=("winning", winning_rule),
        annotate=lambda env, episode: None,
    )


SPEC = ToolAgentSpec(
    name="nim",
    description="Nim against a fixed opponent from a pile of 10; a win scores 1.",
    rules=RULES,
    budget=Budget(max_steps=10, timeout_s=30, stall_after=0),  # every act changes the pile, so a stall cannot happen
    flags=lambda parser: None,  # add the agent's own flags here, after the shared ones
    series=make_series,
)
