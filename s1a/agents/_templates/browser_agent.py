# coding: utf-8
"""The template for a browser-front agent: copy this file to ``s1a/agents/<name>.py``.

Change the name, the description and the goal; keep the shipped operation rules until a site family shows that it
needs its own, then write that string here. ``s1a run <name> --model jev --goal "..."`` runs one task; the
policy switches (``--batch``, ``--prefetch``, ``--goal-values``) are run-time flags, not spec fields.
"""

from __future__ import annotations

from s1a.browser import prompts
from s1a.spec import BrowserAgentSpec, Budget

SPEC = BrowserAgentSpec(
    name="site_task",
    description="A page task on one site family with enumerable controls; the goal comes from the caller.",
    rules=prompts.OPERATION_RULES["en"],  # the operation rules Jev reads with every decision
    language="en",
    budget=Budget(
        max_steps=100, timeout_s=300, stall_after=3
    ),  # stall_after: actions without a page change before BLOCKED
    goal=None,  # a fixed task string makes the agent a demo; None takes --goal
)
