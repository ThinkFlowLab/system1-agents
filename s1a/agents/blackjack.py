# coding: utf-8
"""Blackjack (RLCard): ``s1a run blackjack --model jev --rethink off --episodes 100``."""

from __future__ import annotations

import argparse
import contextlib
from typing import Any

import rlcard

from s1a.spec import Budget, Series, ToolAgentSpec

RULES = (
    "Blackjack against a dealer who hits until 17. Aim to beat the dealer's final total without exceeding 21. "
    "Cards are written suit then rank (H7 = seven of hearts, SA = ace of spades, DT = ten of diamonds). "
    "Basic strategy: stand on hard 17 or more; stand on hard 13 to 16 when the dealer shows 2 to 6, otherwise hit; "
    "stand on hard 12 only against a dealer 4 to 6; always hit hard 11 or less; hit soft 17 or less; stand on soft 19 or more; "
    "stand on soft 18 unless the dealer shows 9, ten or ace."
)
_RANK_VALUE = {"A": 11, "T": 10, "J": 10, "Q": 10, "K": 10}


def hand_total(cards: list[str]) -> tuple[int, bool]:
    total, aces = 0, 0
    for card in cards:
        rank = card[-1]
        total += _RANK_VALUE.get(rank, 0) or int(rank)
        aces += rank == "A"
    while total > 21 and aces:
        total, aces = total - 10, aces - 1
    return total, aces > 0


class BlackjackEnv:
    def __init__(self, seed: int) -> None:
        self._env = rlcard.make("blackjack", config={"seed": seed})
        self._state: dict[str, Any] = {}
        self._score = 0.0

    async def reset(self) -> None:
        self._state, _ = self._env.reset()
        self._score = 0.0

    def _hands(self) -> tuple[list[str], list[str]]:
        raw = self._state["raw_obs"]
        player = next(v for k, v in raw.items() if "player" in k and isinstance(v, list))
        dealer = next(v for k, v in raw.items() if "dealer" in k and isinstance(v, list))
        return list(player), list(dealer)

    async def observe(self) -> dict[str, Any]:
        player, dealer = self._hands()
        total, soft = hand_total(player)
        # RLCard lists the dealer's hand without the hole card while the hand runs, and hole card first once it is over.
        showing = dealer[1:2] if self.done else dealer[:1]
        view = {"player_cards": player, "player_total": total, "soft_hand": soft, "dealer_showing": showing}
        if self.done:
            view["dealer_cards"] = dealer
        return view

    async def candidates(self) -> dict[str, str]:
        if self._env.is_over():
            return {}
        return {name: f"{name} (blackjack action)" for name in self._state["raw_legal_actions"]}

    async def step(self, key: str) -> None:
        action_id = list(self._state["legal_actions"].keys())[self._state["raw_legal_actions"].index(key)]
        self._state, _ = self._env.step(action_id)
        if self._env.is_over():
            self._score = float(self._env.get_payoffs()[0])

    @property
    def done(self) -> bool:
        return self._env.is_over()

    @property
    def score(self) -> float:
        return self._score


def basic_strategy(state: dict[str, Any], candidates: dict[str, str]) -> str:
    total, soft = state["player_total"], state["soft_hand"]
    up = state["dealer_showing"][0][-1]
    dealer = _RANK_VALUE.get(up, 0) or int(up)
    if soft:
        if total >= 19 or (total == 18 and dealer <= 8):
            return "stand"
        return "hit"
    if total >= 17 or (13 <= total <= 16 and dealer <= 6) or (total == 12 and 4 <= dealer <= 6):
        return "stand"
    return "hit"


def make_series(flags: argparse.Namespace) -> Series:
    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=lambda seed: BlackjackEnv(seed=seed),
        session=contextlib.nullcontext(),
        baseline=("basic", basic_strategy),
        annotate=lambda env, episode: None,
    )


SPEC = ToolAgentSpec(
    name="blackjack",
    description="One hand of Blackjack against a dealer who hits until 17; the payoff is the score.",
    rules=RULES,
    budget=Budget(max_steps=12, timeout_s=60, stall_after=0),  # one hand never stalls
    flags=lambda parser: None,
    series=make_series,
)
