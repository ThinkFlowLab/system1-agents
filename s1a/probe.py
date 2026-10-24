# coding: utf-8
"""The fit probe: hand-written cases, one choice question each; accuracy and latency say whether a task is a Jev task.

Each JSONL line: ``{"state": {...}, "options": {"key": "description", ...}, "rules": "...", "accept": ["key", ...],
"note": "..."}``. ``accept`` lists every key that counts as right. A task fits when at least ``MIN_CASES`` cases were
asked and at least ``FIT_THRESHOLD`` of them are right; a case that needs arithmetic or deduction to answer is a stop
by itself, whatever the score.

``run_order`` is the option-order check (#90): every case is asked in its file order, then again with the same option
dict reordered (``reverse``, or ``shuffle:N`` seeded permutations). Keys, texts, state and rules stay byte-identical;
only the dict order changes. It reports the flip rate (share of cases whose chosen key changes under any reorder),
accuracy in each order, and one row per case with every order asked and the key chosen in it.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

from s1a.decision_models import DecisionModel, ChoiceQuestion, Observation

FIT_THRESHOLD = 0.8
MIN_CASES = 8
MAX_SHUFFLES = 50


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

    The dict the CLI prints and the MCP server returns: ``choice``, ``probabilities``, ``confidence`` (always ``max_k p_k``), ``ms``, and ``backend_confidence`` when the backend sent its own number.
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


def parse_order(text: str) -> tuple[str, int]:
    """``reverse``, ``shuffle:N`` or ``same:N`` (1 <= N <= ``MAX_SHUFFLES``) as ``(kind, count)``; else a ValueError.

    ``same:N`` is the control: N more asks in the file order, so a "flip" there is the backend disagreeing with
    itself (sampling, a served model behind a load balancer, GPU non-determinism), not position bias.
    """
    if text == "reverse":
        return "reverse", 1
    kind, sep, count = text.partition(":")
    if kind in ("shuffle", "same") and sep and count.isdigit() and 1 <= int(count) <= MAX_SHUFFLES:
        return kind, int(count)
    raise ValueError(f"--order must be reverse, shuffle:N or same:N with 1 <= N <= {MAX_SHUFFLES}, got {text!r}")


def reorderings(options: dict[str, str], kind: str, count: int, rng: random.Random) -> list[dict[str, str]]:
    """The reordered copies of ``options`` to ask: same keys and texts, only the dict order differs.

    ``reverse`` gives one copy, ``same`` gives ``count`` copies in the file order (the control). ``shuffle`` gives ``count`` seeded permutations, each different from the file order
    when the options allow it (a one-option case has no other order, so its copy equals the original).
    """
    keys = list(options)
    if kind == "reverse":
        return [{key: options[key] for key in reversed(keys)}]
    if kind == "same":  # the non-determinism control: the file order again, ``count`` times
        return [dict(options) for _ in range(count)]
    out: list[dict[str, str]] = []
    for _ in range(count):
        order = keys[:]
        for _ in range(20):  # resample the identity; with two or more keys this ends almost surely
            rng.shuffle(order)
            if order != keys or len(keys) < 2:
                break
        out.append({key: options[key] for key in order})
    return out


async def run_order(
    cases: list[dict[str, Any]], decision_model: DecisionModel, order: str, *, seed: int = 0
) -> dict[str, Any]:
    """Each case in file order, then in every reordering ``order`` asks for; flip rate, per-order accuracy, rows."""
    kind, count = parse_order(order)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for case in cases:
        original = await pick(decision_model, state=case["state"], options=case["options"], rules=case["rules"])
        asks = [{"options": list(case["options"]), "choice": original["choice"], "ms": original["ms"]}]
        for options in reorderings(case["options"], kind, count, rng):
            answer = await pick(decision_model, state=case["state"], options=options, rules=case["rules"])
            asks.append({"options": list(options), "choice": answer["choice"], "ms": answer["ms"]})
        reordered = asks[1:]
        rows.append(
            {
                "note": str(case.get("note") or ""),
                "choice": original["choice"],
                "ok": original["choice"] in case["accept"],
                "reordered_choices": [ask["choice"] for ask in reordered],
                "reordered_ok": sum(ask["choice"] in case["accept"] for ask in reordered),
                "flipped": any(ask["choice"] != original["choice"] for ask in reordered),
                "asks": asks,
            }
        )
    cases_asked = len(rows)
    reordered_asks = sum(len(row["reordered_choices"]) for row in rows)
    right = sum(row["ok"] for row in rows)
    accuracy = right / cases_asked if cases_asked else 0.0
    flips = sum(row["flipped"] for row in rows)
    all_ms = [ask["ms"] for row in rows for ask in row["asks"]]
    return {
        "order": order,
        "seed": seed,
        "rows": rows,
        "cases": cases_asked,
        "flips": flips,
        "flip_rate": round(flips / cases_asked, 3) if cases_asked else 0.0,
        "right": right,
        "accuracy": round(accuracy, 3),
        "reordered_asks": reordered_asks,
        "reordered_right": sum(row["reordered_ok"] for row in rows),
        "reordered_accuracy": round(sum(row["reordered_ok"] for row in rows) / reordered_asks, 3)
        if reordered_asks
        else 0.0,
        "median_ms": int(statistics.median(all_ms)) if all_ms else 0,
        "verdict": verdict(cases_asked, accuracy),  # from the file order, as without --order
    }


def render_order(summary: dict[str, Any]) -> str:
    """The order check as text: one line per case (file-order answer, reordered answers, FLIP), then the totals."""
    lines = [f"{'case':<28}{'answer':<16}{'':<6}{'reordered':<24}{'':<5}"]
    for row in summary["rows"]:
        reordered = ",".join(row["reordered_choices"])
        lines.append(
            f"{row['note'][:27]:<28}{row['choice'][:15]:<16}{'ok' if row['ok'] else 'WRONG':<6}"
            f"{reordered[:23]:<24}{'FLIP' if row['flipped'] else '':<5}"
        )
    lines.append(
        f"order: {summary['order']} (seed {summary['seed']})   flips: {summary['flips']}/{summary['cases']} "
        f"({summary['flip_rate']:.0%})   accuracy file order: {summary['accuracy']:.0%}   "
        f"reordered: {summary['reordered_right']}/{summary['reordered_asks']} ({summary['reordered_accuracy']:.0%})   "
        f"median ms: {summary['median_ms']}   verdict: {summary['verdict']}"
    )
    return "\n".join(lines)
