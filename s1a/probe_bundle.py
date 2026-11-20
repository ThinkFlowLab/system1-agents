# coding: utf-8
"""Cross-question interference on ``decide_many`` (#127): isolated vs bundled vs reversed-bundle.

``s1a probe --bundle`` asks every head of a multi-question case alone, then all heads in one ``decide_many``
(file / production order, and again with the question-name order reversed), then repeats the bundled request in
the same order as a non-determinism control (the same idea as ``s1a probe --order same:N``). It reports, per head,
the key or binary flip rate versus the isolated answer and the total variation between the probability
distributions. Both answers stay in the record.

This does not change ``decide_many`` or the fit probe when ``--bundle`` is off. There is no mitigation here.

Each JSONL line: ``{"state": {...}, "questions": {"name": {...}, ...}, "note": "..."}``. A question is either
``{"type": "choice", "options": {"key": "text"|object, ...}, "rules": "...", "goal": "...", "operation": "..."}``
or ``{"type": "noul", "question": "...", "criteria": {"true": "...", "false": "..."}}``. A case needs at least two
questions; dict order is the file / production order.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from s1a.decision_models import Choice, ChoiceQuestion, DecisionModel, Noul, NoulQuestion, Observation, Question

CONTROL_REPEATS = 1  # one same-order bundled repeat; a "flip" here is the backend disagreeing with itself


def _well_formed_question(spec: object) -> bool:
    if not isinstance(spec, dict) or spec.get("type") not in {"choice", "noul"}:
        return False
    if spec["type"] == "choice":
        options = spec.get("options")
        return (
            isinstance(options, dict)
            and bool(options)
            and all(isinstance(key, str) for key in options)
            and all(isinstance(value, (str, dict)) for value in options.values())
        )
    question = spec.get("question")
    if not isinstance(question, str) or not question.strip():
        return False
    criteria = spec.get("criteria")
    return criteria is None or (isinstance(criteria, dict) and set(criteria) == {"true", "false"})


def _well_formed(case: object) -> bool:
    return (
        isinstance(case, dict)
        and isinstance(case.get("state"), (dict, str))
        and isinstance(case.get("questions"), dict)
        and len(case["questions"]) >= 2
        and all(isinstance(name, str) and _well_formed_question(spec) for name, spec in case["questions"].items())
    )


def parse_question(spec: dict[str, Any]) -> Question:
    """A ``ChoiceQuestion`` or ``NoulQuestion`` from one case's question object."""
    if spec["type"] == "choice":
        return ChoiceQuestion(
            spec["options"],
            goal=str(spec.get("goal") or ""),
            rules=spec["rules"] if isinstance(spec.get("rules"), (str, list, tuple)) else "",
            operation=str(spec.get("operation") or ""),
        )
    criteria = spec.get("criteria")
    return NoulQuestion(spec["question"], criteria=dict(criteria) if criteria is not None else None)


def questions_of(case: dict[str, Any]) -> dict[str, Question]:
    """The case's questions in file order."""
    return {name: parse_question(spec) for name, spec in case["questions"].items()}


def reversed_questions(questions: dict[str, Question]) -> dict[str, Question]:
    """The same questions with the name order reversed (directional interference)."""
    return {name: questions[name] for name in reversed(list(questions))}


def read_cases(path: Path) -> list[dict[str, Any]]:
    """The JSONL multi-question cases; a malformed line is a ValueError naming the file and line."""
    cases: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: not JSON: {exc}") from exc
        if not _well_formed(case):
            raise ValueError(
                f"{path}:{line_number}: a bundle case needs state (object or text) and at least two questions "
                "(choice with options, or noul with a question)"
            )
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def snapshot(answer: Choice | Noul) -> dict[str, Any]:
    """The comparable record of one head: the key or noul probability, the binary/key, the distribution, confidence."""
    if isinstance(answer, Choice):
        return {
            "type": "choice",
            "key": answer.key,
            "binary": answer.key,
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
        }
    return {
        "type": "noul",
        "noul": answer.p,
        "binary": answer.p >= 0.5,
        "probabilities": {"true": answer.p, "false": 1.0 - answer.p},
        "confidence": answer.confidence,
    }


def total_variation(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Total variation between two recorded heads (½ Σ|p−q| over the same keys; for noul this equals |p−q|)."""
    keys = set(left["probabilities"]) | set(right["probabilities"])
    return 0.5 * sum(abs(left["probabilities"].get(key, 0.0) - right["probabilities"].get(key, 0.0)) for key in keys)


def _answers(decision) -> dict[str, dict[str, Any]]:
    return {name: snapshot(answer) for name, answer in decision.answers.items()}


async def _ask(
    decision_model: DecisionModel, state: dict[str, Any] | str, questions: dict[str, Question]
) -> dict[str, Any]:
    decision = await decision_model.decide_many(Observation(state), questions)
    return {"answers": _answers(decision), "ms": decision.latency_ms, "heads": list(questions)}


async def run(
    cases: list[dict[str, Any]], decision_model: DecisionModel, *, repeats: int = CONTROL_REPEATS
) -> dict[str, Any]:
    """Isolated, bundled, reversed-bundle and same-order bundled repeats; per-head flip rates and mean TV."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rows: list[dict[str, Any]] = []
    for case in cases:
        questions = questions_of(case)
        state = case["state"]
        isolated = {}
        isolated_ms = []
        for name, question in questions.items():
            asked = await _ask(decision_model, state, {name: question})
            isolated[name] = asked["answers"][name]
            isolated_ms.append(asked["ms"])
        bundled = await _ask(decision_model, state, questions)
        reversed_ = await _ask(decision_model, state, reversed_questions(questions))
        control = [await _ask(decision_model, state, questions) for _ in range(repeats)]
        heads = list(questions)
        flips = {"bundled": {}, "reversed": {}, "control": {}}
        tv = {"bundled": {}, "reversed": {}, "control": {}}
        for name in heads:
            base = isolated[name]
            flips["bundled"][name] = bundled["answers"][name]["binary"] != base["binary"]
            flips["reversed"][name] = reversed_["answers"][name]["binary"] != base["binary"]
            flips["control"][name] = any(
                repeat["answers"][name]["binary"] != bundled["answers"][name]["binary"] for repeat in control
            )
            tv["bundled"][name] = round(total_variation(base, bundled["answers"][name]), 6)
            tv["reversed"][name] = round(total_variation(base, reversed_["answers"][name]), 6)
            tv["control"][name] = round(
                statistics.mean(
                    total_variation(bundled["answers"][name], repeat["answers"][name]) for repeat in control
                ),
                6,
            )
        rows.append(
            {
                "note": str(case.get("note") or ""),
                "heads": heads,
                "isolated": isolated,
                "bundled": bundled,
                "reversed": reversed_,
                "control": control,
                "isolated_ms": isolated_ms,
                "flips": flips,
                "tv": tv,
            }
        )
    heads = list(dict.fromkeys(name for row in rows for name in row["heads"]))
    per_head: dict[str, dict[str, Any]] = {}
    for name in heads:
        on = [row for row in rows if name in row["heads"]]
        count = len(on)
        per_head[name] = {
            "cases": count,
            "bundled_flips": sum(row["flips"]["bundled"][name] for row in on),
            "reversed_flips": sum(row["flips"]["reversed"][name] for row in on),
            "control_flips": sum(row["flips"]["control"][name] for row in on),
            "bundled_flip_rate": round(sum(row["flips"]["bundled"][name] for row in on) / count, 3) if count else 0.0,
            "reversed_flip_rate": round(sum(row["flips"]["reversed"][name] for row in on) / count, 3) if count else 0.0,
            "control_flip_rate": round(sum(row["flips"]["control"][name] for row in on) / count, 3) if count else 0.0,
            "mean_tv_bundled": round(statistics.mean(row["tv"]["bundled"][name] for row in on), 3) if count else 0.0,
            "mean_tv_reversed": round(statistics.mean(row["tv"]["reversed"][name] for row in on), 3) if count else 0.0,
            "mean_tv_control": round(statistics.mean(row["tv"]["control"][name] for row in on), 3) if count else 0.0,
        }
    all_ms = [row["bundled"]["ms"] for row in rows] + [row["reversed"]["ms"] for row in rows]
    all_ms.extend(repeat["ms"] for row in rows for repeat in row["control"])
    all_ms.extend(ms for row in rows for ms in row["isolated_ms"])
    return {
        "bundle": True,
        "repeats": repeats,
        "rows": rows,
        "cases": len(rows),
        "heads": heads,
        "per_head": per_head,
        "median_ms": int(statistics.median(all_ms)) if all_ms else 0,
    }


def _rate(hits: int, total: int) -> str:
    return f"{hits}/{total} ({(hits / total if total else 0.0):.0%})"


def render(summary: dict[str, Any]) -> str:
    """The interference check as text: one line per case per head, then per-head totals."""
    lines = [f"{'case':<28}{'head':<16}{'isolated':<12}{'bundled':<12}{'reversed':<12}{'ctrl':<8}"]
    for row in summary["rows"]:
        for name in row["heads"]:
            isolated = row["isolated"][name]
            bundled = row["bundled"]["answers"][name]
            reversed_ = row["reversed"]["answers"][name]
            marks = []
            if row["flips"]["bundled"][name]:
                marks.append("B")
            if row["flips"]["reversed"][name]:
                marks.append("R")
            if row["flips"]["control"][name]:
                marks.append("C")
            lines.append(
                f"{row['note'][:27]:<28}{name[:15]:<16}"
                f"{_label(isolated):<12}{_label(bundled):<12}{_label(reversed_):<12}"
                f"{''.join(marks):<8}"
            )
    lines.append(
        f"bundle: isolated vs decide_many (file order and reversed)   control: {summary['repeats']} same-order "
        f"bundled repeat(s)   cases: {summary['cases']}   median ms: {summary['median_ms']}"
    )
    for name, stats in summary["per_head"].items():
        n = stats["cases"]
        lines.append(
            f"  {name}: bundled flips {_rate(stats['bundled_flips'], n)} mean TV {stats['mean_tv_bundled']:.3f}   "
            f"reversed flips {_rate(stats['reversed_flips'], n)} mean TV {stats['mean_tv_reversed']:.3f}   "
            f"control flips {_rate(stats['control_flips'], n)} mean TV {stats['mean_tv_control']:.3f}"
        )
    return "\n".join(lines)


def _label(record: dict[str, Any]) -> str:
    if record["type"] == "choice":
        return str(record["key"])[:11]
    return f"{record['noul']:.2f}"
