# coding: utf-8
"""Who Wants to Be a Millionaire (self-hosted page, Open Trivia DB): ``s1a run millionaire --model jev --rethink off --episodes 5``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

from s1a.config import ROOT
from s1a.env import Env
from s1a.jobs import SHOWCASE_DIR, Episode
from s1a.spec import Budget, Series, ToolAgentSpec
from s1a.tool.hands import BrowserHands, Frames, Hands, serve_static, serving

PAGE_DIR = ROOT / "evals" / "millionaire"
QUESTIONS = PAGE_DIR / "questions.json"
OPENTDB = "https://opentdb.com/api.php"
OPENTDB_PAUSE_S = 5.5  # the API allows one request per five seconds per IP
DIFFICULTIES = ("easy", "medium", "hard")
PER_DIFFICULTY = 5
LADDERS = 6
MAX_LADDERS_PER_FETCH = 50 // PER_DIFFICULTY  # opentdb answers response_code 1 above amount 50
LIFELINE_KEY = "5050"
LOADING_POLLS = 20
RULES = (
    "A quiz with a 15-question prize ladder. Pick the answer that is factually correct. "
    "If no offered answer is clearly right and the 50:50 lifeline is offered, use the lifeline first; it removes two "
    "wrong answers and the question is asked again."
)
READ_STATE = "(args) => window.__state"


def fetch_ladders(count: int) -> list[list[dict[str, Any]]]:
    """``count`` ladders of 15 four-choice questions (5 easy, 5 medium, 5 hard) from the Open Trivia Database, CC BY-SA 4.0."""
    if not 1 <= count <= MAX_LADDERS_PER_FETCH:
        raise ValueError(f"1 to {MAX_LADDERS_PER_FETCH} ladders per fetch (the API caps amount at 50), got {count}")
    banks: dict[str, list[dict[str, Any]]] = {}
    for index, difficulty in enumerate(DIFFICULTIES):
        if index:
            time.sleep(OPENTDB_PAUSE_S)
        params = {"amount": PER_DIFFICULTY * count, "difficulty": difficulty, "type": "multiple", "encode": "url3986"}
        response = httpx.get(OPENTDB, params=params, timeout=30.0)
        response.raise_for_status()
        payload = response.json()
        if payload.get("response_code") != 0:
            raise RuntimeError(f"opentdb response_code {payload.get('response_code')} for {difficulty}")
        banks[difficulty] = [
            {
                "difficulty": unquote(row["difficulty"]),
                "category": unquote(row["category"]),
                "question": unquote(row["question"]),
                "correct": unquote(row["correct_answer"]),
                "incorrect": [unquote(answer) for answer in row["incorrect_answers"]],
            }
            for row in payload["results"]
        ]
    return [
        [
            question
            for difficulty in DIFFICULTIES
            for question in banks[difficulty][ladder * PER_DIFFICULTY : (ladder + 1) * PER_DIFFICULTY]
        ]
        for ladder in range(count)
    ]


def ensure_questions() -> None:
    """Fetch the ladders on the first run; the file is gitignored, so every checkout plays its own draw."""
    if QUESTIONS.exists():
        return
    print(
        f"millionaire: fetching {LADDERS} question ladders from opentdb.com into {QUESTIONS}, about 15 s",
        file=sys.stderr,
    )
    QUESTIONS.write_text(json.dumps(fetch_ladders(LADDERS), ensure_ascii=False, indent=1), encoding="utf-8")


class MillionaireEnv:
    """One episode plays ladder ``seed`` of ``questions.json`` (the page's ``?ladder=`` query)."""

    def __init__(self, hands: Hands, url: str, seed: int, *, frames_dir: Path | None) -> None:
        self._hands = hands
        self._url = f"{url}?ladder={seed}"
        self._state: dict[str, Any] = {}
        self._frames = Frames(hands, frames_dir) if frames_dir is not None else None  # --showcase only

    @property
    def frames_dir(self) -> Path | None:
        return self._frames.directory if self._frames is not None else None

    async def reset(self) -> None:
        await self._hands.read("(url) => { location.assign(url); return 'ok'; }", self._url)
        await asyncio.sleep(0.5)
        await self._hands.wait_ready("#answers button", 10.0)
        self._state = await self._hands.read(READ_STATE, {})
        if self._frames is not None:
            await self._frames.reset()

    async def observe(self) -> dict[str, Any]:
        state = self._state
        return {
            "question": state.get("question"),
            "difficulty": state.get("difficulty"),
            "category": state.get("category"),
            "level": state.get("level"),
            "prize": state.get("prize_for_this_question"),
            "winnings": state.get("winnings"),
            "status": state.get("status"),
        }

    async def candidates(self) -> dict[str, str]:
        if self.done:
            return {}
        answers = {key: text for key, text in (self._state.get("answers") or {}).items()}
        if self._state.get("fifty_fifty_available"):
            answers[LIFELINE_KEY] = "use the 50:50 lifeline: two wrong answers are removed"
        return answers

    async def step(self, key: str) -> None:
        await self._hands.press("5" if key == LIFELINE_KEY else key)
        await asyncio.sleep(0.5)
        self._state = await self._hands.read(READ_STATE, {})
        for _ in range(LOADING_POLLS):  # a correct answer shows the next question after a short pause
            if self._state.get("status") != "loading":
                break
            await asyncio.sleep(0.1)
            self._state = await self._hands.read(READ_STATE, {})
        if self._frames is not None:
            await self._frames.take()

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._state.get("history") or [])

    @property
    def done(self) -> bool:
        return self._state.get("status") in {"won", "lost"}

    @property
    def score(self) -> float:
        return float(self._state.get("winnings", 0))


def make_series(flags: argparse.Namespace) -> Series:
    ensure_questions()
    ladders = len(json.loads(QUESTIONS.read_text(encoding="utf-8")))
    if flags.seed + flags.episodes > ladders:
        raise ValueError(
            f"seeds {flags.seed} to {flags.seed + flags.episodes - 1} need {flags.seed + flags.episodes} ladders and "
            f"{QUESTIONS} holds {ladders}: python evals/millionaire/fetch_questions.py --ladders N"
        )
    site = serve_static(PAGE_DIR)
    hands = BrowserHands(headless=not flags.headed)
    url = f"http://127.0.0.1:{site.server_address[1]}/index.html"

    def frames_dir_for(seed: int) -> Path | None:
        return SHOWCASE_DIR / "frames" / f"millionaire-{seed}" if flags.showcase else None

    def annotate(env: Env, episode: Episode) -> None:
        assert isinstance(env, MillionaireEnv)
        episode.extra["history"] = env.history
        episode.frames_dir = env.frames_dir

    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=lambda seed: MillionaireEnv(hands, url, seed, frames_dir=frames_dir_for(seed)),
        session=serving(site, hands.session(url, ready_selector="#answers button", timeout_s=15.0)),
        baseline=None,
        annotate=annotate,
    )


SPEC = ToolAgentSpec(
    name="millionaire",
    description="A 15-question quiz ladder from Open Trivia DB; winnings are the score, the 50:50 lifeline is a candidate.",
    rules=RULES,
    budget=Budget(
        max_steps=20, timeout_s=180, stall_after=0
    ),  # 15 questions plus lifelines; every act changes the question
    flags=lambda parser: None,
    series=make_series,
)
