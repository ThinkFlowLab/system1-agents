# coding: utf-8
"""The real player: one task through the production ``browse`` front with a decision model or the chat model.

``--model jev`` (or another decision model name) fills the model slot through ``build_model``; ``llm`` lets the chat
model decide every step. Both get the same budget (``--max-steps``, ``--timeout``). Decision time is the sum of the
counted decision-model calls (retries and failures included); the chat-model arm has none, so its decision time is
N/A, not zero. Needs a started Runner (``started_runner``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evals.webharbor.runner import Outcome, Player
from evals.webharbor.tasks import TaskSpec


@dataclass(frozen=True)
class LiveBudget:
    max_steps: int = 40
    timeout_s: float = 600.0
    headless: bool = True


def goal(task: TaskSpec) -> str:
    """The task's own ``instruction.md``: it names the local URL and asks for the final answer as text."""
    return task.instruction


def make_player(budget: LiveBudget) -> Player:
    from s1a.browser import browse, prompts
    from s1a.browser.decision_model import BrowserPolicy
    from s1a.config import chat_model_from_env
    from s1a.decision_models import build_model
    from s1a.spec import BrowserAgentSpec, Budget

    from evals.recovery.live import CountingDecisionModel

    chat = chat_model_from_env()

    async def play(task: TaskSpec, arm: str, logs_dir: Path) -> Outcome:
        spec = BrowserAgentSpec(
            name=f"webharbor_{task.site}",
            description=f"WebHarbor {task.id} on the local {task.site} mirror.",
            rules=prompts.OPERATION_RULES["en"],
            language="en",
            budget=Budget(max_steps=budget.max_steps, timeout_s=budget.timeout_s, stall_after=3),
            goal=None,
        )
        proxy = None if arm == "llm" else CountingDecisionModel(build_model(arm))
        try:
            answer = await browse.browse(
                spec,
                BrowserPolicy(prefetch_values=False, batch_actions=False, goal_value_cache=False, rethink_on=False),
                model_name=arm,
                goal=goal(task),
                timeout_s=budget.timeout_s,
                max_steps=budget.max_steps,
                logs_dir=logs_dir,
                headless=budget.headless,
                chat=chat,
                decision_model=proxy,
            )
        finally:
            if proxy is not None:
                await proxy.close()
        error = answer.get("error")
        if error and str(error).startswith("timeout"):
            error = "timeout"
        elif error:
            error = "agent: " + str(error)[:120]
        screenshot = answer.get("screenshot")
        return Outcome(
            answer=str(answer.get("final") or ""),
            error=None if answer.get("ok") else (error or "agent: not ok"),
            total_ms=answer.get("elapsed_ms"),
            decision_ms=None if proxy is None else sum(int(call.get("ms") or 0) for call in proxy.calls),
            screenshot=Path(screenshot) if screenshot else None,
        )

    return play
