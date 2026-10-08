"""Route local labelled tickets through the shared tool loop; labels remain evaluator-only."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from s1a.env import Env
from s1a.jobs import Episode
from s1a.spec import Budget, Series, ToolAgentSpec

DEFAULT_DATASET = Path(__file__).with_name("_data") / "ticket_router_eval.jsonl"
QUEUES = {
    "logistics": "Delivery tracking, delivery progress or delivery problems",
    "payment": "Charges, failed payments or duplicate payments",
    "returns": "Requests for returns, exchanges or refunds",
    "account": "Login or account access problems",
    "human": "Insufficient information, multiple independent requests or an explicit request for a human",
}
RULES = "Route the current explicit request to exactly one queue. logistics: delivery tracking or delivery problems. payment: charges, failed payments or duplicate payments. returns: requests for returns, exchanges or refunds. account: login or account access problems. human: insufficient information, multiple independent requests that cannot be uniquely routed, or an explicit request for a human. An explicit human request takes priority. Recognize negation and resolved, quoted or hypothetical background; do not route by keywords alone. Order status is context, not the user's intent. A payment problem with an explicit request for a refund belongs to returns. If no unique queue fits, choose human."
PUBLIC_FIELDS = ("id", "title", "description", "order_status")


def load_tickets(path: Path) -> list[dict[str, Any]]:
    """Read local JSONL; the environment validates records and copies only supported fields."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TicketRouterEnv:
    def __init__(
        self, seed: int, *, tickets: list[dict[str, Any]] | None = None, batch_size: int | None = None
    ) -> None:
        rows = load_tickets(DEFAULT_DATASET) if tickets is None else tickets
        if not rows:
            raise ValueError("ticket_router requires a non-empty labelled dataset")
        self._tickets: list[dict[str, str]] = []
        ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("id", "title", "description", "label")
            ):
                raise ValueError("tickets require non-empty string id, title, description and label")
            if row["id"] in ids or row["label"] not in QUEUES:
                raise ValueError("ticket ids must be unique and labels must name a supported queue")
            if "order_status" in row and not isinstance(row["order_status"], str):
                raise ValueError("order_status, when present, must be a string")
            ids.add(row["id"])
            self._tickets.append({key: row[key] for key in (*PUBLIC_FIELDS, "label") if key in row})
        self._size = len(rows) if batch_size is None else batch_size
        if type(self._size) is not int or not 1 <= self._size <= len(rows):
            raise ValueError("batch_size must be between 1 and the number of tickets")
        self._seed = seed
        self._batch: list[dict[str, str]] = []
        self._results: list[dict[str, Any]] = []
        self._index = 0
        self._correct = 0

    async def reset(self) -> None:
        self._batch = list(self._tickets)
        random.Random(self._seed).shuffle(self._batch)
        self._batch = self._batch[: self._size]
        self._results = []
        self._index = self._correct = 0

    async def observe(self) -> dict[str, Any]:
        if self.done:
            return {"ticket": None, "progress": {"ticket_id": None}}
        row = self._batch[self._index]
        return {
            "ticket": {key: row[key] for key in PUBLIC_FIELDS if key in row},
            "progress": {"ticket_id": row["id"]},
        }

    async def candidates(self) -> dict[str, str]:
        return {} if self.done else dict(QUEUES)

    async def step(self, key: str) -> None:
        if self.done or key not in QUEUES:
            raise ValueError("cannot route a completed batch or choose an unknown queue")
        row = self._batch[self._index]
        correct = key == row["label"]
        self._results.append({"id": row["id"], "expected": row["label"], "predicted": key, "correct": correct})
        self._correct += int(correct)
        self._index += 1

    @property
    def done(self) -> bool:
        return self._index >= len(self._batch)

    @property
    def score(self) -> float:
        return float(self._correct)

    def report(self) -> dict[str, Any]:
        """Evaluator-only metrics. Never include these labels or predictions in observe()."""
        support = Counter(row["label"] for row in self._batch)
        hits = Counter(row["expected"] for row in self._results if row["correct"])
        total = len(self._batch)
        # Fingerprint includes state and labels, permitting checks that all models saw identical data.
        fingerprint = hashlib.sha256(
            json.dumps(self._batch, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "seed": self._seed,
            "total": total,
            "processed": self._index,
            "correct": self._correct,
            "accuracy": self._correct / total if total else 0.0,
            "coverage": self._index / total if total else 0.0,
            "per_class_support": {key: support[key] for key in QUEUES},
            "per_class_recall": {key: hits[key] / support[key] if support[key] else None for key in QUEUES},
            "ticket_ids": [row["id"] for row in self._batch],
            "unprocessed_ids": [row["id"] for row in self._batch[self._index :]],
            "batch_sha256": fingerprint,
            "routes": [dict(row) for row in self._results],
        }


KEYWORDS = {
    "logistics": ("delivery", "shipping", "shipment", "parcel", "package", "courier", "tracking", "delivered"),
    "payment": ("payment", "pay", "paid", "charge", "charges", "charged", "checkout", "transaction"),
    "returns": ("refund", "return", "returns", "exchange", "cancel", "money back", "larger size"),
    "account": ("login", "log in", "sign in", "account", "password", "verification code", "profile"),
}


def keyword_rule(state: dict[str, Any], candidates: dict[str, str]) -> str:
    """A deliberately simple lexical baseline: ambiguous matches go to human."""
    ticket = state["ticket"]
    text = (ticket["title"] + " " + ticket["description"]).casefold()

    def matches_word(word: str) -> bool:
        return re.search(r"\b" + re.escape(word) + r"\b", text) is not None

    if any(matches_word(word) for word in ("human", "real person", "live agent")):
        return "human"
    matches = [key for key, words in KEYWORDS.items() if any(matches_word(word) for word in words)]
    return matches[0] if len(matches) == 1 and matches[0] in candidates else "human"


def add_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="local labelled tickets in JSONL")
    parser.add_argument("--batch-size", type=int, default=None, help="tickets per episode (default: entire dataset)")


def annotate(env: Env, episode: Episode) -> None:
    assert isinstance(env, TicketRouterEnv)
    episode.extra["ticket_router"] = env.report()


def make_series(flags: argparse.Namespace) -> Series:
    if flags.rethink != "off":
        raise ValueError("ticket_router handles independent tickets; use --rethink off")
    if flags.episodes <= 0:
        raise ValueError("ticket_router requires at least one episode")
    tickets = load_tickets(flags.dataset)
    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=lambda seed: TicketRouterEnv(seed, tickets=tickets, batch_size=flags.batch_size),
        session=contextlib.nullcontext(),
        baseline=("keywords", keyword_rule),
        annotate=annotate,
    )


SPEC = ToolAgentSpec(
    name="ticket_router",
    description="Route a seeded batch of local tickets to five queues; score counts correct routes.",
    rules=RULES,
    budget=Budget(max_steps=30, timeout_s=300, stall_after=0),
    flags=add_flags,
    series=make_series,
)
