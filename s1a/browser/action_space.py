# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Modifications Copyright 2026 ThinkFlowLab
# SPDX-License-Identifier: Apache-2.0

"""The browser policy's action space: one numbered element table plus one candidate head per operation.

The policy probe's elements become the table and the heads (CLICK, TYPE_TEXT, PRESS_ENTER, SELECT) plus an
optional value head. ``build_observation`` and ``build_questions`` turn one tick into the ``Observation`` and the typed
``ChoiceQuestion`` set a decision model decides over in one call; ``interpret`` maps the validated ``Decision``
back onto a candidate. The wire body and the answer validation live in ``s1a.decision_models``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from s1a.browser import prompts
from s1a.decision_models import ChoiceQuestion, Decision, Observation, Question, Usage

NONE_VALUE = "none"
_OPERATION_LABELS = {
    "CLICK": (
        "Press a control on the page: a button, a link, a menu entry, an autocomplete suggestion or a calendar day."
    ),
    "TYPE_TEXT": "Type a value into an editable field, replacing what it holds.",
    "PRESS_ENTER": "Submit a filled field by pressing Enter in it, the way a person submits a search.",
    "SELECT": "Pick one of the listed options of a native dropdown.",
    "SCROLL_DOWN": "Move the viewport down the page.",
    "SCROLL_UP": "Move the viewport up the page.",
    "WAIT": "Give the page time to finish updating.",
    "DONE": "The page shows every requirement of the task met.",
    "BLOCKED": "None of the offered operations can move the task forward.",
}
_HISTORY_KEYS = ("action", "kind", "text", "page_changed")
SUBMITTABLE_ROLES = frozenset({"searchbox", "textbox", "combobox"})
DEAD_TARGET_REPEATS = 2  # clicks in a row on one label without a page change before its click is withheld


@dataclass(frozen=True)
class Candidate:
    """One executable choice: an element (by probe item) under one operation."""

    operation: str
    index: str
    item: dict[str, Any]
    option_label: str = ""


@dataclass
class ActionSpace:
    """Numbered element table plus per-operation candidate heads built from one probe."""

    elements: list[dict[str, Any]] = field(default_factory=list)
    heads: dict[str, dict[str, Candidate]] = field(default_factory=dict)
    operations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Move:
    """One tick's validated outcome: the operation, its candidate, and the decision's numbers for the tick record."""

    operation: str
    candidate: Candidate | None
    value_choice: str
    confidence: float
    operation_probabilities: dict[str, float]
    latency_ms: int
    usage: Usage
    model: str
    backend_confidence: float | None = None
    # Every answered head's backend-native confidence (operation, target, text value), by question name.
    backend_confidences: dict[str, float] = field(default_factory=dict)


def submittable(item: dict[str, Any]) -> bool:
    """Enter in this field submits something: an editable text-like field that already holds a value."""
    return (
        bool(item.get("editable"))
        and str(item.get("role") or "") in SUBMITTABLE_ROLES
        and bool(str(item.get("value") or "").strip())
    )


def dead_targets(history: list[dict[str, Any]]) -> set[str]:
    """Labels whose last ``DEAD_TARGET_REPEATS`` clicks in a row left the page unchanged.

    The probe's occlusion check sees what covers a control now; an overlay that returns after each click
    leaves the control looking clickable on every probe. Recent clicks that did nothing are the signal the
    probe cannot give.
    """
    clicks = [entry for entry in history if entry.get("kind") == "click"]
    dead: set[str] = set()
    for label in {str(entry.get("action") or "") for entry in clicks}:
        recent = [entry for entry in clicks if entry.get("action") == label][-DEAD_TARGET_REPEATS:]
        if len(recent) == DEAD_TARGET_REPEATS and all(entry.get("page_changed") is False for entry in recent):
            dead.add(label)
    return dead


def build_action_space(snapshot: dict[str, Any], history: list[dict[str, Any]]) -> ActionSpace:
    """Fold probe elements into jev's table: one row per element, one head per operation.

    An element the probe found occluded (``actionable`` false -- something like a cookie-consent banner
    sits on top of it) gets a table row, so its ``blocked_by`` name is visible state, but no candidate
    head: a real click/type there would not reach it, so it is never offered as a move. Absent the field
    entirely (older/mocked snapshots), the element is treated as actionable, matching the probe's default.
    A control whose last ``DEAD_TARGET_REPEATS`` clicks in ``history`` changed nothing keeps its row,
    marked ``click_did_nothing``, and gets no CLICK candidate until a click on it moves the page.
    """
    retired = dead_targets(history)
    space = ActionSpace()
    for item in snapshot.get("elements") or []:
        if not item.get("target_id"):
            continue
        index = str(len(space.elements) + 1)
        row = {"index": index, "role": item.get("role"), "label": item.get("label"), "value": item.get("value") or ""}
        for key in ("checked", "selected", "expanded", "region", "blocked_by"):
            if item.get(key):
                row[key] = item[key]
        operations: list[str] = []
        actionable = item.get("actionable", True)
        options = item.get("options")
        if actionable and isinstance(options, list) and options:
            head = space.heads.setdefault("SELECT", {})
            row["options"] = []
            for position, option in enumerate(options, start=1):
                key = f"{index}:{position}"
                head[key] = Candidate("SELECT", key, item, option_label=str(option.get("label", "")))
                row["options"].append({"index": key, "label": option.get("label")})
            operations.append("SELECT")
        elif actionable:
            if item.get("editable"):
                space.heads.setdefault("TYPE_TEXT", {})[index] = Candidate("TYPE_TEXT", index, item)
                operations.append("TYPE_TEXT")
            if submittable(item):
                space.heads.setdefault("PRESS_ENTER", {})[index] = Candidate("PRESS_ENTER", index, item)
                operations.append("PRESS_ENTER")
            clickable = not (item.get("editable") and str(item.get("expanded")) == "true")
            if clickable and str(item.get("label") or "") in retired:
                row["click_did_nothing"] = True
                clickable = False
            if clickable:
                space.heads.setdefault("CLICK", {})[index] = Candidate("CLICK", index, item)
                operations.append("CLICK")
        row["operations"] = operations
        space.elements.append(row)
    space.operations = [op for op in ("CLICK", "TYPE_TEXT", "PRESS_ENTER", "SELECT") if op in space.heads]
    if snapshot.get("can_scroll_down"):
        space.operations.append("SCROLL_DOWN")
    if snapshot.get("can_scroll_up"):
        space.operations.append("SCROLL_UP")
    space.operations += ["WAIT", "DONE", "BLOCKED"]
    return space


def build_observation(space: ActionSpace, snapshot: dict[str, Any], history: list[dict[str, Any]]) -> Observation:
    """The state a decision model reads on one tick: the page, the element table and the last ten actions."""
    return Observation(
        {
            "page": {
                "url": snapshot.get("url", ""),
                "title": snapshot.get("title", ""),
                "text": snapshot.get("text", ""),
            },
            "elements": space.elements,
            "recent_actions": [{key: entry.get(key) for key in _HISTORY_KEYS} for entry in history[-10:]],
        }
    )


def build_questions(
    space: ActionSpace, *, goal: str, values: list[str], rules: str, language: str
) -> dict[str, Question]:
    """The heads of one tick: ``operation``, one ``<op>_target`` per head, and ``text_value`` when values are offered.

    ``rules`` is the agent's operation text. Over ``JevModel`` these questions produce the request body the
    policy has always sent; the wire oracle in ``tests/test_decision_models_jev.py`` pins it.
    """
    questions: dict[str, Question] = {
        "operation": ChoiceQuestion({op: _OPERATION_LABELS[op] for op in space.operations}, goal=goal, rules=rules)
    }
    for operation, head in space.heads.items():
        questions[f"{operation.lower()}_target"] = ChoiceQuestion(
            {
                key: {
                    "element": f"[{key}] {candidate.item.get('label', '')}",
                    "current_value": candidate.item.get("value") or "",
                    **({"option": candidate.option_label} if candidate.option_label else {}),
                    **{
                        k: candidate.item[k]
                        for k in ("role", "checked", "selected", "expanded", "region")
                        if candidate.item.get(k)
                    },
                }
                for key, candidate in head.items()
            },
            goal=goal,
            operation=operation,
            rules=(rules, prompts.TARGET_RULES[language]),
        )
    if values and "TYPE_TEXT" in space.heads:
        questions["text_value"] = ChoiceQuestion(
            {**{value: value for value in values}, NONE_VALUE: "No offered value fits the chosen field."},
            goal=goal,
            rules=prompts.VALUE_RULES[language],
        )
    return questions


def top_probabilities(
    decision: Decision, move: Move, space: ActionSpace, limit: int
) -> tuple[dict[str, float], dict[str, str]]:
    """The ``limit`` most probable keys of the head that settled the move (the operation's target head, or the
    operation head when the operation takes no target), with a label per key for the replay."""
    if move.candidate is not None:
        choice = decision.choice(f"{move.operation.lower()}_target")
        labels = {key: str(candidate.item.get("label") or "") for key, candidate in space.heads[move.operation].items()}
    else:
        choice = decision.choice("operation")
        labels = dict(_OPERATION_LABELS)
    top = sorted(choice.probabilities.items(), key=lambda pair: -pair[1])[:limit]
    return {key: round(p, 3) for key, p in top}, {key: labels.get(key, key) for key, _ in top}


def interpret(decision: Decision, space: ActionSpace) -> Move:
    """Map a validated decision onto one candidate (or a control operation).

    ``decide_many`` has already checked every head, so a target key is always one the head offered.
    """
    operation = decision.choice("operation")
    candidate: Candidate | None = None
    if operation.key in space.heads:
        candidate = space.heads[operation.key][decision.choice(f"{operation.key.lower()}_target").key]
    value_choice = ""
    if operation.key == "TYPE_TEXT" and "text_value" in decision.answers:
        value_choice = decision.choice("text_value").key
    return Move(
        operation=operation.key,
        candidate=candidate,
        value_choice=value_choice,
        confidence=operation.confidence,
        operation_probabilities=dict(operation.probabilities),
        latency_ms=decision.latency_ms,
        usage=decision.usage,
        model=decision.model,
        backend_confidence=operation.backend_confidence,
        backend_confidences={
            name: answer.backend_confidence
            for name, answer in decision.answers.items()
            if answer.backend_confidence is not None
        },
    )
