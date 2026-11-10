# coding: utf-8
"""Screen and fact-check questions, plus the offline lexical stand-in.

The stand-in is the same kind of baseline as ``ticket_router``'s keyword rule: it makes the
fixture path deterministic without a model. It is not a quality claim for Jev or for a draft
System1-Omni worker. Labels on the shipped sample match this stand-in on purpose.
"""

from __future__ import annotations

import re
from typing import Any

# Whole words only, so "supermodel", "chipotle" and "agency" are not hotspots.
KEEP = re.compile(r"\b(?:models?|agents?|inference|chips?|research|checkpoints?|llms?)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]{5,}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

STATE_LIMIT = 1500
SOURCE_LIMIT = 400

SCREEN_CHOICES = {
    "keep": "Relevant AI, model, agent, inference, chip, checkpoint or research hotspot",
    "drop": "Not a relevant hotspot for this desk",
}
SCREEN_SCORE = ["Not relevant", "Relevant hotspot"]
FACT_CHOICES = {
    "supported": "The sources state the claim",
    "unsupported": "A name or number in the claim is absent from the sources",
    "needs_human": "The sources do not settle the claim",
}
FACT_SCORE = ["Unsupported", "Needs a human", "Supported"]
FACT_LEVEL = {"unsupported": 0, "needs_human": 1, "supported": 2}

SCREEN_INSTRUCTIONS = (
    "Keep a relevant hotspot for an AI-agent news desk: models, agents, inference, chips, "
    "checkpoints or research. Drop sports, weather, recipes, traffic, celebrity and anything without that hook."
)
SCREEN_SCORE_INSTRUCTIONS = "How relevant is this item to an AI-agent news desk?"
FACT_INSTRUCTIONS = (
    "Judge the claims against the sources only. supported: the sources state the claim. "
    "unsupported: a name or number in the claim is absent from the sources. "
    "needs_human: the sources do not settle the claim."
)
FACT_SCORE_INSTRUCTIONS = "How well do the sources support the claims?"


def lexical_screen(item: dict[str, Any]) -> str:
    """Keep when the title or summary names a desk topic; otherwise drop."""
    text = f"{item.get('title', '')}\n{item.get('summary', '')}"
    return "keep" if KEEP.search(text) else "drop"


def lexical_fact(draft: dict[str, Any]) -> str:
    """Supported when each claim's longer words sit in the sources and every number does too.

    A missing number is unsupported even when the wording overlaps. Thin overlap is needs_human.
    An empty claim list or an empty source list is needs_human. This reads no label.
    """
    claims = draft.get("claims") or []
    sources = draft.get("sources") or []
    if not claims or not sources:
        return "needs_human"
    blob = "\n".join(str(source.get("text", "")) for source in sources).casefold()
    unsupported = False
    overlapped = 0
    for claim in claims:
        folded = str(claim).casefold()
        numbers = _NUMBER.findall(folded)
        if any(number not in blob for number in numbers):
            unsupported = True
        words = _WORD.findall(folded)
        if words and sum(word in blob for word in words) / len(words) >= 0.6:
            overlapped += 1
    if unsupported:
        return "unsupported"
    if overlapped == len(claims):
        return "supported"
    return "needs_human"


def screen_level(choice: str) -> int:
    if choice not in SCREEN_CHOICES:
        raise ValueError(f"screen choice must be keep or drop, got {choice!r}")
    return 1 if choice == "keep" else 0


def fact_level(choice: str) -> int:
    if choice not in FACT_LEVEL:
        raise ValueError(f"fact-check choice must be one of {', '.join(FACT_LEVEL)}, got {choice!r}")
    return FACT_LEVEL[choice]


def one_hot(keys: list[str], chosen: str) -> dict[str, float]:
    if chosen not in keys:
        raise ValueError(f"{chosen!r} is not one of {keys}")
    return {key: (1.0 if key == chosen else 0.0) for key in keys}


def choice_question(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score_question(instructions: str, criteria: list[str]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": list(criteria)}


def screen_questions() -> dict[str, dict[str, Any]]:
    """One question per request. Callers send ``screen`` and ``relevance`` as two prefills."""
    return {
        "screen": choice_question(SCREEN_INSTRUCTIONS, SCREEN_CHOICES),
        "relevance": score_question(SCREEN_SCORE_INSTRUCTIONS, SCREEN_SCORE),
    }


def fact_questions() -> dict[str, dict[str, Any]]:
    return {
        "verdict": choice_question(FACT_INSTRUCTIONS, FACT_CHOICES),
        "support": score_question(FACT_SCORE_INSTRUCTIONS, FACT_SCORE),
    }


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= STATE_LIMIT:
        return text
    return text[: STATE_LIMIT - 1].rstrip() + "…"


def screen_state(item: dict[str, Any]) -> str:
    """Short prefill. Labels, drafts and urls stay out of the text the worker reads."""
    return _clip(f"Title: {item['title'].strip()}\nSummary: {item['summary'].strip()}\nSource: {item['source']}")


def fact_state(item: dict[str, Any]) -> str:
    """Short prefill of the human's claims and sources. No label line."""
    draft = item["draft"]
    lines = [f"Title: {item['title'].strip()}", "Claims:"]
    lines.extend(f"- {str(claim).strip()}" for claim in draft["claims"])
    lines.append("Sources:")
    for source in draft["sources"]:
        lines.append(f"- [{source['id']}] {str(source['text']).strip()[:SOURCE_LIMIT]}")
    return _clip("\n".join(lines))


class RuleDecider:
    """Offline keep/drop and fact verdicts. No network and no label reads."""

    name = "rule"

    def screen(self, item: dict[str, Any]) -> dict[str, Any]:
        choice = lexical_screen(item)
        level = screen_level(choice)
        return _ruled(choice, list(SCREEN_CHOICES), level, SCREEN_SCORE)

    def fact_check(self, item: dict[str, Any]) -> dict[str, Any]:
        choice = lexical_fact(item["draft"])
        level = fact_level(choice)
        return _ruled(choice, list(FACT_CHOICES), level, FACT_SCORE)


def _ruled(choice: str, keys: list[str], level: int, rubric: list[str]) -> dict[str, Any]:
    return {
        "choice": choice,
        "probabilities": one_hot(keys, choice),
        "score": float(level),
        "score_level": level,
        "score_name": rubric[level],
        "score_probabilities": one_hot([str(i) for i in range(len(rubric))], str(level)),
        "latency_ms": 0,
        "backend": "rule",
        "model": None,
        "requested_model": None,
        "worker": None,
        "calls": [],
    }
