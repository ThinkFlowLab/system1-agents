# coding: utf-8
"""The rail front: one question to a decision model at one callback hook, acted on above a threshold, and its
labelled-set evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.common.logging import logger
from openjiuwen.core.single_agent.rail.base import AgentCallback, AgentCallbackContext, AgentCallbackEvent
from openjiuwen.harness.rails.base import DeepAgentRail

from s1a.decision_models import (
    DecisionModel,
    ChoiceQuestion,
    NoulQuestion,
    Observation,
    Question,
    build_model,
)
from s1a.jobs import now_iso
from s1a.pricing import cost_usd
from s1a.spec import Json, RailSpec, Verdict

QUESTION = "check"
RAIL_MODEL_NAMES = ("jev", "laya")


def question(spec: RailSpec) -> Question:
    """The one question a rail asks: the spec's rules as a noul statement, or a choice among its criteria."""
    match spec.question:
        case "noul":
            return NoulQuestion(spec.rules, dict(spec.criteria))
        case "choice":
            return ChoiceQuestion(dict(spec.criteria), goal=spec.rules)
    raise ValueError(f"{spec.name}: unknown question type {spec.question!r}")


async def ask(spec: RailSpec, state: Json, decision_model: DecisionModel) -> Verdict:
    """One decision over ``state``; the flagged key's probability, banded by the spec's thresholds."""
    decision = await decision_model.decide_many(Observation(state), {QUESTION: question(spec)})
    match spec.question:
        case "noul":
            p = decision.noul(QUESTION).p
        case _:
            p = decision.choice(QUESTION).probabilities[spec.flagged]
    return Verdict(p=p, band=spec.thresholds.band(p), ms=decision.latency_ms, input_tokens=decision.usage.input_tokens)


class DecisionModelRail(DeepAgentRail):
    """The spec's question on the spec's hook; the spec's action runs in the act band. ``ticks`` records every verdict."""

    def __init__(self, spec: RailSpec, decision_model: DecisionModel) -> None:
        super().__init__()
        self._spec = spec
        self._decision_model = decision_model
        self.ticks: list[dict[str, Any]] = []

    def get_callbacks(self) -> dict[AgentCallbackEvent, AgentCallback]:
        return {self._spec.hook: self.on_event}

    async def on_event(self, ctx: AgentCallbackContext) -> None:
        state = self._spec.state_of(ctx)
        if state is None:
            return
        try:
            verdict = await ask(self._spec, state, self._decision_model)
        except BaseError as exc:
            logger.warning(
                "[DecisionModelRail %s] decision failed, %s: %s", self._spec.name, self._spec.on_failure, exc
            )
            self.ticks.append({"error": str(exc), "failed": self._spec.on_failure})
            if self._spec.on_failure == "closed":
                verdict = Verdict(p=self._spec.thresholds.act, band="act", ms=0, input_tokens=0)
                await self._spec.act(ctx, verdict)
            return
        self.ticks.append(
            {"p": round(verdict.p, 3), "band": verdict.band, "ms": verdict.ms, "input_tokens": verdict.input_tokens}
        )
        if verdict.band == "act":
            await self._spec.act(ctx, verdict)


def read_labelled(path: Path) -> list[dict[str, Any]]:
    """JSONL records, each with a ``state`` for Jev and a boolean ``label``: does the flagged statement hold."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: not JSON: {exc}") from exc
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("state"), (dict, str))
            or not isinstance(record.get("label"), bool)
        ):
            raise ValueError(f"{path}:{line_number}: a record needs a state (object or text) and a boolean label")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


async def evaluate(
    spec: RailSpec, labelled_set: Path, *, decision_model: DecisionModel, results_dir: Path
) -> dict[str, Any]:
    """Every record through the same question; precision and recall of the act band against the labels, one job folder."""
    records = read_labelled(labelled_set)
    verdicts = [await ask(spec, record["state"], decision_model) for record in records]
    acted = [verdict.band == "act" for verdict in verdicts]
    labels = [bool(record["label"]) for record in records]
    tp = sum(a and label for a, label in zip(acted, labels))
    fp = sum(a and not label for a, label in zip(acted, labels))
    fn = sum(label and not a for a, label in zip(acted, labels))
    # Local models have no Jev API charges, as on the tool and browser fronts.
    jev_input_tokens = sum(verdict.input_tokens for verdict in verdicts) if decision_model.name == "jev" else 0
    summary = {
        "rail": spec.name,
        "records": len(records),
        "positives": sum(labels),
        "acted": sum(acted),
        "uncertain": sum(verdict.band == "uncertain" for verdict in verdicts),
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "accuracy": round(sum(a == label for a, label in zip(acted, labels)) / len(records), 3),
        "median_ms": int(statistics.median(verdict.ms for verdict in verdicts)),
        "jev_input_tokens": jev_input_tokens,
        "cost_usd": cost_usd(jev_input_tokens, 0, 0, 0, None),
        "thresholds": {"allow": spec.thresholds.allow, "act": spec.thresholds.act},
        "labelled_set": str(labelled_set),
        "finished_at": now_iso(),
    }
    job_dir = results_dir / spec.name / f"{datetime.now():%Y-%m-%d__%H-%M-%S-%f}__{decision_model.name}"
    job_dir.mkdir(parents=True)
    (job_dir / "verdicts.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "label": label,
                    "p": round(verdict.p, 3),
                    "band": verdict.band,
                    "ms": verdict.ms,
                    "note": record.get("note"),
                },
                ensure_ascii=False,
            )
            + "\n"
            for record, label, verdict in zip(records, labels, verdicts)
        ),
        encoding="utf-8",
    )
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["job_dir"] = str(job_dir)
    return summary


def parser(spec: RailSpec) -> argparse.ArgumentParser:
    build = argparse.ArgumentParser(prog=f"s1a run {spec.name}", description=spec.description)
    build.add_argument(
        "--labelled-set",
        type=Path,
        default=spec.labelled_set,
        required=spec.labelled_set is None,
        help="JSONL records with a state and a boolean label; the spec's set when it names one",
    )
    build.add_argument(
        "--model", choices=RAIL_MODEL_NAMES, default="jev", help="who answers the question: jev, or laya in process"
    )
    return build


async def play(spec: RailSpec, args: argparse.Namespace, *, results_dir: Path) -> dict[str, Any]:
    """Evaluate the rail on its labelled set with a fresh model; the summary names its job folder."""
    decision_model = build_model(args.model)
    try:
        await decision_model.warm()
        return await evaluate(spec, args.labelled_set, decision_model=decision_model, results_dir=results_dir)
    finally:
        await decision_model.close()
