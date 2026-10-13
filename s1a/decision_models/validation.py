# coding: utf-8
"""The invariants every answer must satisfy, checked on the backend's dicts; a breach is ``MODEL_CALL_FAILED``.

``validate_choice`` and ``validate_noul`` are the only constructors of ``Choice`` and ``Noul`` from a backend's answer.
"""

from __future__ import annotations

import math
from typing import Any, TypeGuard

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.types import (
    Answer,
    Choice,
    ChoiceQuestion,
    Decision,
    Json,
    Noul,
    NoulQuestion,
    Question,
    Reply,
)


def unit_interval(value: object) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1 and math.isfinite(value)


def choice_faults(fields: Json, ids: list[str]) -> list[str]:
    """Why ``fields`` is not a choice answer over ``ids``: empty when it is one.

    ``choice`` is one of ``ids``; ``probabilities`` covers exactly ``ids`` with values in [0, 1] summing to 1 (within
    0.02) and peaks at the chosen key (ties allowed); ``confidence`` lies in [0, 1].
    """
    faults: list[str] = []
    key, distribution, confidence = fields.get("choice"), fields.get("probabilities"), fields.get("confidence")
    if not isinstance(key, str) or key not in ids:
        faults.append(f"choice {key!r} is not one of {ids}")
    if not unit_interval(confidence):
        faults.append(f"confidence {confidence!r} is not in [0, 1]")
    if not isinstance(distribution, dict):
        return faults + ["probabilities is not an object"]
    if set(distribution) != set(ids):
        faults.append(f"probabilities cover {sorted(map(str, distribution))}, not {sorted(ids)}")
    if not all(unit_interval(weight) for weight in distribution.values()):
        return faults + ["a probability is not in [0, 1]"]
    if not math.isclose(sum(distribution.values()), 1.0, abs_tol=0.02):
        faults.append(f"probabilities sum to {sum(distribution.values()):.3f}, not 1")
    if isinstance(key, str) and key in distribution and distribution[key] < max(distribution.values()) - 1e-6:
        faults.append(f"the chosen key {key!r} is not the peak of the distribution")
    return faults


def validate_choice(fields: Json, ids: list[str]) -> Choice:
    """The typed choice with its numbers normalised, or ``MODEL_CALL_FAILED`` naming every fault."""
    faults = choice_faults(fields, ids)
    if faults:
        raise build_error(
            StatusCode.MODEL_CALL_FAILED, error_msg="decisions answer is not an offered choice: " + "; ".join(faults)
        )
    return Choice(
        key=str(fields["choice"]),
        probabilities={str(key): float(weight) for key, weight in fields["probabilities"].items()},
        confidence=float(fields["confidence"]),
    )


def validate_noul(fields: Json) -> Noul:
    """The typed noul with ``p`` in the unit interval and a confidence, ``max(p, 1 - p)`` when the backend gave none."""
    p, confidence = fields.get("noul"), fields.get("confidence")
    if not unit_interval(p):
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="decisions answer is not a noul probability")
    p = float(p)
    return Noul(p=p, confidence=float(confidence) if unit_interval(confidence) else max(p, 1.0 - p))


def validate_answer(fields: Any, question: Question) -> Answer:
    """One answer as its question's type; a non-object answer reads as an empty one and fails on its fields."""
    fields = fields if isinstance(fields, dict) else {}
    match question:
        case ChoiceQuestion():
            return validate_choice(fields, question.ids)
        case NoulQuestion():
            return validate_noul(fields)
    raise TypeError(f"not a question: {question!r}")


def validate_answers(reply: Reply, questions: dict[str, Question]) -> Decision:
    """Every asked question answered and valid, as a ``Decision``; answers nobody asked for are dropped."""
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        if name not in reply.answers:
            raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg=f"no answer for question {name!r}")
        answers[name] = validate_answer(reply.answers[name], question)
    return Decision(answers, reply.latency_ms, reply.usage, reply.model, reply.raw)
