# coding: utf-8
"""The fit probe: hand-written cases, one choice question each; accuracy and latency say whether a task is a Jev task.

Each JSONL line: ``{"state": {...}, "options": {"key": "description", ...}, "rules": "...", "accept": ["key", ...],
"note": "..."}``. ``accept`` lists every key that counts as right. A task fits when at least ``MIN_CASES`` cases were
asked and at least ``FIT_THRESHOLD`` of them are right; a case that needs arithmetic or deduction to answer is a stop
by itself, whatever the score.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from s1a.decision_models import DecisionModel, ChoiceQuestion, Observation

FIT_THRESHOLD = 0.8
MIN_CASES = 8


def verdict(cases: int, accuracy: float) -> str:
    if cases < MIN_CASES:
        return f"too few cases: {cases} of {MIN_CASES}"
    return "fits" if accuracy >= FIT_THRESHOLD else "not a decision-model task"


def _well_formed(case: object) -> bool:
    return (
        isinstance(case, dict)
        and isinstance(case.get("state"), dict)
        and isinstance(case.get("options"), dict)
        and all(isinstance(text, str) for text in case["options"].values())
        and isinstance(case.get("rules"), str)
        and isinstance(case.get("accept"), list)
        and all(isinstance(key, str) for key in case["accept"])
    )


def read_cases(path: Path) -> list[dict[str, Any]]:
    """The JSONL cases; a malformed line is a ValueError naming the file and line."""
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
                f"{path}:{line_number}: a case needs state (object), options (key to text), rules (text) and accept (list of keys)"
            )
        if not case["options"]:
            raise ValueError(f"{path}:{line_number}: options must offer at least one choice")
        if not case["accept"]:
            raise ValueError(f"{path}:{line_number}: accept must name at least one offered option")
        unknown = sorted(set(case["accept"]) - set(case["options"]))
        if unknown:
            raise ValueError(f"{path}:{line_number}: accept names keys not in options: {', '.join(unknown)}")
        cases.append(case)
    return cases


async def pick(
    decision_model: DecisionModel, *, state: dict[str, Any], options: dict[str, str], rules: str
) -> dict[str, Any]:
    """One ``choice`` question over ``options``: the chosen key, a probability per option, the confidence, the latency.

    The dict the CLI prints and the MCP server returns: ``choice``, ``probabilities``, ``confidence``, ``ms``.
    """
    decision = await decision_model.decide_many(Observation(state), {"pick": ChoiceQuestion(options, rules=rules)})
    return {**decision.choice("pick").as_dict(), "ms": decision.latency_ms}


async def run(cases: list[dict[str, Any]], decision_model: DecisionModel) -> dict[str, Any]:
    """Every case through ``pick``; the rows, the accuracy, the median latency and the verdict."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        answer = await pick(decision_model, state=case["state"], options=case["options"], rules=case["rules"])
        rows.append(
            {
                "note": str(case.get("note") or ""),
                "choice": answer["choice"],
                "ok": answer["choice"] in case["accept"],
                "confidence": round(float(answer["confidence"]), 2),
                "ms": answer["ms"],
            }
        )
    right = sum(row["ok"] for row in rows)
    accuracy = right / len(rows) if rows else 0.0
    return {
        "rows": rows,
        "right": right,
        "cases": len(rows),
        "accuracy": round(accuracy, 3),
        "median_ms": int(statistics.median(row["ms"] for row in rows)) if rows else 0,
        "verdict": verdict(len(rows), accuracy),
    }


def render(summary: dict[str, Any]) -> str:
    lines = [f"{'case':<28}{'answer':<16}{'':<6}{'conf':>5}{'ms':>7}"]
    for row in summary["rows"]:
        lines.append(
            f"{row['note'][:27]:<28}{row['choice'][:15]:<16}{'ok' if row['ok'] else 'WRONG':<6}{row['confidence']:>5.2f}{row['ms']:>7}"
        )
    lines.append(
        f"right: {summary['right']}/{summary['cases']} ({summary['accuracy']:.0%})   median ms: {summary['median_ms']}   "
        f"verdict: {summary['verdict']}"
    )
    return "\n".join(lines)
