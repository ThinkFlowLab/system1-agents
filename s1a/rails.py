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
RAIL_MODEL_NAMES = ("jev", "clm", "laya", "laya-served")


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


def failure_category(exc: BaseError) -> str:
    """A decision failure as its status name, plus its cause's type when it has one (``MODEL_CALL_FAILED (ReadTimeout)``).

    Never the exception's message: an endpoint's error body can quote the submitted state, and so the guarded tool
    output, back into it.
    """
    category = getattr(exc.status, "name", type(exc).__name__)
    cause = exc.__cause__
    return f"{category} ({type(cause).__name__})" if cause is not None else category


class DecisionModelRail(DeepAgentRail):
    """The spec's question on the spec's hook; the spec's action runs in the act band. ``ticks`` records every verdict.

    Each tick names the ``handling`` the event got, so a pass on an uncertain or partly inspected event never reads as
    a confident allow:

    - ``pass``: allow band, every character inspected (or the spec tracks no coverage).
    - ``pass_uncertain``: uncertain band, every character inspected; the model abstained and the event passes whole.
    - ``pass_partial``: allow or uncertain band over part of the event, and the spec has no ``on_partial``.
    - ``withhold_uninspected``: allow or uncertain band over part of the event; the spec's ``on_partial`` ran.
    - ``act``: act band; the spec's action ran.
    - ``act_on_failure`` / ``pass_on_failure``: the decision failed on a closed / open rail.

    With ``coverage_of`` set, a tick also carries ``coverage`` (``complete`` or ``partial``), ``chars`` and
    ``inspected_chars``. A tick holds numbers, bands and, on a failure, the error's category and code (see
    ``failure_category``); never the event's content and never an exception's message, which can echo it.
    """

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
        coverage = self._spec.coverage_of(ctx) if self._spec.coverage_of is not None else None
        seen: dict[str, Any] = (
            {
                "coverage": "complete" if coverage.complete else "partial",
                "chars": coverage.total,
                "inspected_chars": coverage.inspected,
            }
            if coverage is not None
            else {}
        )
        try:
            verdict = await ask(self._spec, state, self._decision_model)
        except BaseError as exc:
            error = failure_category(exc)
            logger.warning(
                "[DecisionModelRail %s] decision failed, %s: %s", self._spec.name, self._spec.on_failure, error
            )
            closed = self._spec.on_failure == "closed"
            self.ticks.append(
                {
                    "error": error,
                    "error_code": exc.code,
                    "failed": self._spec.on_failure,
                    "handling": "act_on_failure" if closed else "pass_on_failure",
                    **seen,
                }
            )
            if closed:
                verdict = Verdict(p=self._spec.thresholds.act, band="act", ms=0, input_tokens=0)
                await self._spec.act(ctx, verdict)
            return
        if verdict.band == "act":
            handling = "act"
        elif coverage is not None and not coverage.complete:
            handling = "withhold_uninspected" if self._spec.on_partial is not None else "pass_partial"
        else:
            handling = "pass_uncertain" if verdict.band == "uncertain" else "pass"
        self.ticks.append(
            {
                "p": round(verdict.p, 3),
                "band": verdict.band,
                "ms": verdict.ms,
                "input_tokens": verdict.input_tokens,
                "handling": handling,
                **seen,
            }
        )
        if handling == "act":
            await self._spec.act(ctx, verdict)
        elif handling == "withhold_uninspected":
            assert self._spec.on_partial is not None and coverage is not None
            await self._spec.on_partial(ctx, coverage)


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
    jev_input_tokens = sum(verdict.input_tokens for verdict in verdicts) if decision_model.bills_input_tokens else 0
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
        "--model",
        choices=RAIL_MODEL_NAMES,
        default="jev",
        help="who answers the question: jev, clm or laya-served over HTTP, laya in process",
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
