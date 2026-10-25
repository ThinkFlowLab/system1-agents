# coding: utf-8
"""Five-stage desk run. Human pick and human publish are gates. Nothing is posted.

Stages, in order: ingest (already a list of items), screen, human pick, fact-check, human publish.
A missing gate raises ``GateRequired`` unless ``allow_unsafe_skip_gates`` is set. That flag is unsafe:
it invents a pick and holds every item. It still does not publish.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from s1a.content.ingest import normalize_item, public_view, screen_view
from s1a.content.latency import CLOCK, PROCEDURE
from s1a.content.rules import FACT_LEVEL, SCREEN_CHOICES, fact_level, screen_level

STAGES = ("ingest", "screen", "human_pick", "fact_check", "human_publish")
PUBLISH_DECISIONS = ("publish", "hold")


class Decider(Protocol):
    name: str

    def screen(self, item: dict[str, Any]) -> dict[str, Any]: ...

    def fact_check(self, item: dict[str, Any]) -> dict[str, Any]: ...


class GateRequired(ValueError):
    """A human gate has no decision. ``packet`` is the work finished before the stop, if any."""

    def __init__(self, message: str, packet: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.packet = packet


def items_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(items), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_workflow(
    items: Sequence[Mapping[str, Any]],
    *,
    decider: Decider,
    picks: Sequence[str] | None = None,
    publish: Mapping[str, str] | None = None,
    pick_hook: Callable[[list[dict[str, Any]]], list[str]] | None = None,
    publish_hook: Callable[[list[dict[str, Any]]], Mapping[str, str]] | None = None,
    allow_unsafe_skip_gates: bool = False,
    workers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Walk the five stages. ``items`` are already ingested. Labels are read only when scoring.

    Explicit ``picks`` / ``publish`` win over hooks. Hooks win over the unsafe skip. The unsafe skip
    never wins over a decision the caller actually supplied.
    """
    if not items:
        raise ValueError("content workflow needs at least one item")
    rows = [normalize_item(dict(item)) for item in items]
    seen: set[str] = set()
    for item in rows:
        if item["id"] in seen:
            raise ValueError(f"duplicate content id {item['id']!r}")
        seen.add(item["id"])

    screen_records = [_score_screen(item, decider.screen(screen_view(item))) for item in rows]
    screen = _stage_summary(screen_records, kept_key="choice")
    kept = [record for record in screen_records if record["choice"] == "keep"]
    packet_base = _packet_base(rows, decider.name, workers)

    pick_ids, pick_source, pick_unsafe = _resolve_picks(
        kept, picks=picks, pick_hook=pick_hook, allow_unsafe_skip_gates=allow_unsafe_skip_gates
    )
    if pick_ids is None:
        packet = {
            **packet_base,
            "stopped_at": "human_pick",
            "screen": screen,
            "human_pick": None,
            "fact_check": None,
            "human_publish": _unpublished(),
        }
        raise GateRequired(
            "human pick gate: pass picks, answer the prompt, or pass --allow-unsafe-skip-gates "
            "(unsafe: it invents a pick and holds every item; it still does not publish)",
            packet,
        )

    by_id = {item["id"]: item for item in rows}
    fact_records = []
    for ident in pick_ids:
        item = by_id[ident]
        if not item.get("draft"):
            fact_records.append(_no_draft(ident))
            continue
        fact_records.append(_score_fact(item, decider.fact_check(public_view(item))))
    fact = _stage_summary(fact_records, kept_key=None)

    decisions, publish_source, publish_unsafe = _resolve_publish(
        fact_records,
        publish=publish,
        publish_hook=publish_hook,
        allow_unsafe_skip_gates=allow_unsafe_skip_gates,
    )
    if decisions is None:
        preview = ", ".join(f"{record['id']}={record['choice']}" for record in fact_records)
        packet = {
            **packet_base,
            "stopped_at": "human_publish",
            "screen": screen,
            "human_pick": {"ids": pick_ids, "source": pick_source, "unsafe": pick_unsafe},
            "fact_check": fact,
            "human_publish": _unpublished(),
        }
        raise GateRequired(
            "human publish gate: pass --publish for "
            f"{', '.join(pick_ids)} (fact-check {preview}). Nothing was published",
            packet,
        )

    return {
        "schema": "content-workflow-v1",
        "stages": list(STAGES),
        "backend": decider.name,
        "unsafe_skip": pick_unsafe or publish_unsafe,
        "auto_publish": False,
        "ingest": packet_base["ingest"],
        "screen": screen,
        "human_pick": {"ids": pick_ids, "source": pick_source, "unsafe": pick_unsafe},
        "fact_check": fact,
        "human_publish": {
            "decisions": decisions,
            "source": publish_source,
            "unsafe": publish_unsafe,
            "executed": [],
            "auto_publish": False,
            "note": "Recorded the human decision. Nothing was published.",
        },
        "workers": packet_base["workers"],
        "latency_gate": {"attached": False, "clock": CLOCK, "procedure": PROCEDURE},
    }


def _packet_base(rows: list[dict[str, Any]], backend: str, workers: Mapping[str, Any] | None) -> dict[str, Any]:
    counts = Counter(item["source"] for item in rows)
    return {
        "schema": "content-workflow-packet-v1",
        "backend": backend,
        "auto_publish": False,
        "ingest": {
            "count": len(rows),
            "sources": {name: counts.get(name, 0) for name in ("rss", "x", "fixture")},
            "sha256": items_sha256(rows),
        },
        "workers": dict(workers) if workers else {"screen": None, "fact_check": None},
        "latency_gate": {"attached": False, "clock": CLOCK, "procedure": PROCEDURE},
    }


def _unpublished() -> dict[str, Any]:
    return {
        "decisions": {},
        "source": None,
        "unsafe": False,
        "executed": [],
        "auto_publish": False,
        "note": "Recorded no publish decision. Nothing was published.",
    }


def _resolve_picks(
    kept: list[dict[str, Any]],
    *,
    picks: Sequence[str] | None,
    pick_hook: Callable[[list[dict[str, Any]]], list[str]] | None,
    allow_unsafe_skip_gates: bool,
) -> tuple[list[str] | None, str, bool]:
    kept_ids = [record["id"] for record in kept]
    if picks is not None:
        return _validate_picks(list(picks), kept_ids), "explicit", False
    if pick_hook is not None:
        shown = [
            {"id": record["id"], "title": record["title"], "summary": record["summary"], "source": record["source"]}
            for record in kept
        ]
        return _validate_picks(list(pick_hook(shown)), kept_ids), "hook", False
    if allow_unsafe_skip_gates:
        if not kept_ids:
            raise ValueError("screen kept nothing; the unsafe skip has no item to invent")
        invented = kept_ids if len(kept_ids) < 3 else kept_ids[:3]
        return _validate_picks(invented, kept_ids), "unsafe-skip", True
    return None, "", False


def _validate_picks(picks: list[str], kept_ids: list[str]) -> list[str]:
    if not picks or any(not isinstance(ident, str) or not ident.strip() for ident in picks):
        raise ValueError("a human pick is a list of non-empty ids")
    cleaned = [ident.strip() for ident in picks]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("a human pick repeats an id")
    missing = [ident for ident in cleaned if ident not in kept_ids]
    if missing:
        raise ValueError(f"human pick {missing[0]!r} was not kept by the screen")
    if len(kept_ids) >= 3 and not 3 <= len(cleaned) <= 5:
        raise ValueError("the human pick is 3 to 5 kept items")
    if len(kept_ids) < 3 and cleaned != kept_ids:
        raise ValueError("fewer than 3 items were kept; the human pick must be every kept id")
    return cleaned


def _resolve_publish(
    fact_records: list[dict[str, Any]],
    *,
    publish: Mapping[str, str] | None,
    publish_hook: Callable[[list[dict[str, Any]]], Mapping[str, str]] | None,
    allow_unsafe_skip_gates: bool,
) -> tuple[dict[str, str] | None, str, bool]:
    expected = [record["id"] for record in fact_records]
    if publish is not None:
        return _validate_publish(dict(publish), expected), "explicit", False
    if publish_hook is not None:
        shown = [
            {"id": record["id"], "choice": record["choice"], "score_name": record["score_name"]}
            for record in fact_records
        ]
        return _validate_publish(dict(publish_hook(shown)), expected), "hook", False
    if allow_unsafe_skip_gates:
        return {ident: "hold" for ident in expected}, "unsafe-skip", True
    return None, "", False


def _validate_publish(decisions: Mapping[str, str], expected: list[str]) -> dict[str, str]:
    if set(decisions) != set(expected):
        raise ValueError("the publish decision must name each picked id once, and no other id")
    cleaned: dict[str, str] = {}
    for ident in expected:
        value = decisions[ident]
        if value not in PUBLISH_DECISIONS:
            raise ValueError("a publish decision is publish or hold")
        cleaned[ident] = value
    return cleaned


def _score_screen(item: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    expected = item.get("label")
    expected_level = screen_level(expected) if expected in SCREEN_CHOICES else None
    record = _attach(
        item, decision, expected_choice=expected if expected in SCREEN_CHOICES else None, expected_level=expected_level
    )
    record["title"] = item["title"]
    record["summary"] = item["summary"]
    record["source"] = item["source"]
    return record


def _score_fact(item: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    draft = item.get("draft") or {}
    expected = draft.get("label")
    expected_level = fact_level(expected) if expected in FACT_LEVEL else None
    return _attach(
        item,
        decision,
        expected_choice=expected if expected in FACT_LEVEL else None,
        expected_level=expected_level,
    )


def _no_draft(ident: str) -> dict[str, Any]:
    return {
        "id": ident,
        "choice": "needs_human",
        "probabilities": {"supported": 0.0, "unsupported": 0.0, "needs_human": 1.0},
        "score": 1.0,
        "score_level": 1,
        "score_name": "Needs a human",
        "score_probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
        "expected_choice": None,
        "expected_score_level": None,
        "choice_correct": None,
        "score_correct": None,
        "latency_ms": 0,
        "backend": "withheld",
        "model": None,
        "requested_model": None,
        "worker": None,
        "calls": [],
        "note": "no_draft",
    }


def _attach(
    item: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    expected_choice: str | None,
    expected_level: int | None,
) -> dict[str, Any]:
    choice_correct = None if expected_choice is None else decision["choice"] == expected_choice
    score_correct = None if expected_level is None else decision["score_level"] == expected_level
    return {
        "id": item["id"],
        "choice": decision["choice"],
        "probabilities": dict(decision["probabilities"]),
        "score": decision["score"],
        "score_level": decision["score_level"],
        "score_name": decision["score_name"],
        "score_probabilities": dict(decision["score_probabilities"]),
        "expected_choice": expected_choice,
        "expected_score_level": expected_level,
        "choice_correct": choice_correct,
        "score_correct": score_correct,
        "latency_ms": decision["latency_ms"],
        "backend": decision["backend"],
        "model": decision["model"],
        "requested_model": decision["requested_model"],
        "worker": decision["worker"],
        "calls": list(decision["calls"]),
    }


def _stage_summary(records: list[dict[str, Any]], *, kept_key: str | None) -> dict[str, Any]:
    labeled = [record for record in records if record["expected_choice"] is not None]
    choice_correct = sum(1 for record in labeled if record["choice_correct"])
    score_labeled = [record for record in records if record["expected_score_level"] is not None]
    score_correct = sum(1 for record in score_labeled if record["score_correct"])
    summary: dict[str, Any] = {
        "total": len(records),
        "labeled": len(labeled),
        "choice_correct": choice_correct,
        "score_correct": score_correct,
        "choice_accuracy": _ratio(choice_correct, len(labeled)),
        "score_accuracy": _ratio(score_correct, len(score_labeled)),
        "records": records,
    }
    if kept_key is not None:
        kept_ids = [record["id"] for record in records if record["choice"] == "keep"]
        summary["kept"] = len(kept_ids)
        summary["dropped"] = len(records) - len(kept_ids)
        summary["kept_ids"] = kept_ids
    return summary


def _ratio(correct: int, labeled: int) -> float | None:
    if labeled == 0:
        return None
    return correct / labeled
