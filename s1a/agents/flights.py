# coding: utf-8
"""Google Flights, one-way Zurich to London: ``s1a run flights --model jev --batch on --profile-out run.json``.

The timing demo against browser-use/jev-ultrafast; the clock runs from the first decision to the final DONE.
``--batch on`` is the measured best arm: 10.4 s, 3 of 3 verified (docs/benchmarks.md).
"""

from __future__ import annotations

from datetime import date, timedelta

from s1a.browser import prompts
from s1a.spec import BrowserAgentSpec, Budget


def _goal_date() -> str:
    """The first Sunday at least 28 days from today, as Google Flights spells it: ``September 20, 2026``."""
    day = date.today() + timedelta(days=28)
    day += timedelta(days=(6 - day.weekday()) % 7)
    return f"{day:%B} {day.day}, {day.year}"


GOAL_DATE = _goal_date()

SPEC = BrowserAgentSpec(
    name="flights",
    description=f"Google Flights: find one-way Zurich to London flights on {GOAL_DATE} and stop at the results.",
    rules=prompts.OPERATION_RULES["en"],
    language="en",
    budget=Budget(max_steps=100, timeout_s=180, stall_after=3),
    goal=(
        "Open https://www.google.com/travel/flights?hl=en and find one-way flights from Zurich to London on "
        f"{GOAL_DATE}, for one adult in economy. Stop when matching flight options are visible."
    ),
)
