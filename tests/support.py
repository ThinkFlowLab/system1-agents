# coding: utf-8
"""Shared test doubles: a counter environment, its spec, and the browser front offline.

Scripted models and transports come from ``s1a.decision_models`` (``ScriptedModel``, ``ScriptedTransport``).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import patch


from s1a.browser import browse
from s1a.browser.decision_model import BrowserPolicy
from s1a.decision_models import Observation, Question, Reply, ScriptedModel
from s1a.config import chat_model_from_env
from s1a.spec import BrowserAgentSpec, Budget, Series, ToolAgentSpec

CHAT_ENV = {"LLM_API_KEY": "k", "LLM_BASE_URL": "https://chat.test/v1", "MODEL_NAME": "m", "OPENROUTER_API_KEY": "r"}
POLICY = BrowserPolicy(prefetch_values=True, batch_actions=False, goal_value_cache=False)


class CounterEnv:
    def __init__(self) -> None:
        self.n = 0

    async def reset(self) -> None:
        self.n = 0

    async def observe(self) -> dict[str, Any]:
        return {"n": self.n}

    async def candidates(self) -> dict[str, str]:
        return {} if self.done else {"inc": "add one", "noop": "do nothing"}

    async def step(self, key: str) -> None:
        self.n += key == "inc"

    @property
    def done(self) -> bool:
        return self.n >= 3

    @property
    def score(self) -> float:
        return float(self.n)


def counter_series(flags: Namespace) -> Series:
    return Series(
        seeds=range(flags.seed, flags.seed + flags.episodes),
        env_for=lambda seed: CounterEnv(),
        session=contextlib.nullcontext(),
        baseline=("always-inc", lambda observation, offered: "inc"),
        annotate=lambda env, episode: None,
    )


COUNTER = ToolAgentSpec(
    name="counter",
    description="Count to three.",
    rules="Pick inc until n is three.",
    budget=Budget(max_steps=5, timeout_s=30, stall_after=0),
    flags=lambda parser: None,
    series=counter_series,
)


class NoDecisionModel(ScriptedModel):
    """A browser-front model that must never be asked: the subagent is faked, so no decision turn happens."""

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        raise AssertionError("no decision is made offline")


def browser_result(summary: str, *, status: str) -> str:
    """The subagent's structured completion around ``summary``, as the chat model's final text."""
    return json.dumps(
        {"browser_result": {"status": status, "terminal_reason": "runtime_completion_validated", "summary": summary}}
    )


async def browse_offline(
    spec: BrowserAgentSpec, policy: BrowserPolicy, *, model_name: str, max_steps: int
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """``browse`` with the subagent factory and the browser run faked: the answer, what the factory saw, the files.

    No Runner, no browser, no key: a decision model's final text is a DONE summary, the chat model's a completion.
    """
    seen: dict[str, Any] = {}

    def fake_factory(model: Any, **kwargs: Any) -> Any:
        seen.update(model=model, **kwargs)
        return object()

    async def fake_run(agent: Any, goal: str, *, timeout_s: float) -> dict[str, Any]:
        seen.update(goal=goal, timeout_s=timeout_s)
        done = json.dumps({"status": "DONE", "reason": "", "url": "https://x", "answer": "Three flights."})
        final = browser_result("42", status="completed") if model_name == "llm" else done
        return {"ok": True, "final": final, "screenshot": None, "error": None}

    with (
        tempfile.TemporaryDirectory() as tmp,
        patch.dict(os.environ, CHAT_ENV, clear=True),
        patch.object(browse, "create_browser_agent", fake_factory),
        patch.object(browse, "run_task", fake_run),
    ):
        answer = await browse.browse(
            spec,
            policy,
            model_name=model_name,
            goal="Show one-way flights",
            timeout_s=30,
            max_steps=max_steps,
            logs_dir=Path(tmp),
            headless=True,
            chat=chat_model_from_env(),
            decision_model=None if model_name == "llm" else NoDecisionModel(),
        )
        files = sorted(p.name for p in Path(tmp).iterdir())
    return answer, seen, files
