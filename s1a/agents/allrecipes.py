# coding: utf-8
"""Allrecipes, WebVoyager task Allrecipes--0: ``s1a run allrecipes --model jev --batch on --headed``.

The browser-use demo against the chat model in the same model_name (``--model llm``). ``TASK`` is task ``Allrecipes--0`` of
the WebVoyager task set (MinorJerry/WebVoyager, He et al. 2024, Apache License 2.0; see NOTICE), and the goal wraps
it in the WebVoyager runner's instruction shape. The run is headed because Allrecipes answers a headless Chromium
with a bot wall.
"""

from __future__ import annotations

from s1a.browser import prompts
from s1a.spec import BrowserAgentSpec, Budget

SITE = "https://www.allrecipes.com/"
TASK = (
    "Provide a recipe for vegetarian lasagna with more than 100 reviews and a rating of at least 4.5 stars "
    "suitable for 6 people."
)

SPEC = BrowserAgentSpec(
    name="allrecipes",
    description="Allrecipes, WebVoyager task 0: a vegetarian lasagna with over 100 reviews, 4.5 stars or more, for 6.",
    rules=prompts.OPERATION_RULES["en"],
    language="en",
    budget=Budget(max_steps=100, timeout_s=300, stall_after=3),
    goal=f"Go to {SITE} and complete this task:\n\n{TASK}\n\nWhen finished, provide your final answer as text.",
)
