# coding: utf-8
"""The rail front: one question to a decision model at one callback hook, acted on above a threshold, and its
labelled-set evaluation."""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import statistics
from collections.abc import Callable
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
BANDS = ("allow", "uncertain", "act", "failure")  # a labelled-set row's outcome; failure: the decision itself failed
Z95 = 1.959963984540054  # standard normal quantile for a 95% Wilson interval


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes in n trials; None when n is 0."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"wilson expects 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return None
    phat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def auroc(labels: list[bool], scores: list[float]) -> float | None:
    """ROC AUC from scores (higher = more positive) via pairwise ranks; ties count as 0.5. None if one class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    greater = sum(p > n for p in pos for n in neg)
    ties = sum(p == n for p in pos for n in neg)
    return (greater + 0.5 * ties) / (len(pos) * len(neg))


def _ci(k: int, n: int) -> list[float] | None:
    interval = wilson(k, n)
    return [round(interval[0], 3), round(interval[1], 3)] if interval is not None else None


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
    - ``act_on_failure`` / ``pass_on_failure`` / ``ask_on_failure``: the decision failed on a closed / open / ask rail.
    - ``ask``: uncertain band, and the spec's ``on_ask`` ran (the tool-call gate's interrupt path).

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
            asking = self._spec.on_failure == "ask"
            handling = "act_on_failure" if closed else "ask_on_failure" if asking else "pass_on_failure"
            self.ticks.append(
                {
                    "error": error,
                    "error_code": exc.code,
                    "failed": self._spec.on_failure,
                    "handling": handling,
                    **seen,
                }
            )
            if closed:
                verdict = Verdict(p=self._spec.thresholds.act, band="act", ms=0, input_tokens=0)
                await self._spec.act(ctx, verdict)
            elif asking and self._spec.on_ask is not None:
                verdict = Verdict(p=0.5, band="uncertain", ms=0, input_tokens=0)
                await self._spec.on_ask(ctx, verdict)
            return
        if verdict.band == "act":
            handling = "act"
        elif verdict.band == "uncertain" and self._spec.on_ask is not None:
            handling = "ask"
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
        elif handling == "ask":
            assert self._spec.on_ask is not None
            await self._spec.on_ask(ctx, verdict)
        elif handling == "withhold_uninspected":
            assert self._spec.on_partial is not None and coverage is not None
            await self._spec.on_partial(ctx, coverage)


def read_labelled(path: Path, data: bytes | None = None) -> list[dict[str, Any]]:
    """JSONL records, each with a ``state`` for Jev and a boolean ``label``: does the flagged statement hold.

    ``data`` is the file's bytes when the caller has already read them, so what is parsed is what it hashed."""
    records: list[dict[str, Any]] = []
    text = (path.read_bytes() if data is None else data).decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
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
    """Every record through the same question; precision and recall of the act band against the labels, one job folder.

    A record's state goes through ``spec.labelled_state`` first, as the hook would clip it. A decision that fails
    (endpoint, key, deadline, malformed answer) is a ``failure`` row, not an abort: every record stays in the
    denominator, and on a closed rail it counts as acted, as the runtime quarantines it. The summary also reports
    F1, false-allow and ask rates with Wilson 95% intervals, AUROC from the probabilities, and the same figures
    per ``split`` when records carry that field."""
    data = labelled_set.read_bytes()  # one read before the first await: the digest covers the bytes evaluated
    records = read_labelled(labelled_set, data)
    clip: Callable[[Json], Json] = spec.labelled_state or (lambda state: state)
    verdicts: list[Verdict | None] = []
    errors: list[str | None] = []
    for record in records:
        try:
            verdicts.append(await ask(spec, clip(record["state"]), decision_model))
            errors.append(None)
        except BaseError as exc:  # the error's class and code only: its message can echo the page
            verdicts.append(None)
            errors.append(getattr(getattr(exc, "status", None), "name", None) or type(exc).__name__)
    bands = [verdict.band if verdict else "failure" for verdict in verdicts]
    acted = [band == "act" or (band == "failure" and spec.on_failure == "closed") for band in bands]
    labels = [bool(record["label"]) for record in records]
    answered = [verdict for verdict in verdicts if verdict]
    tp = sum(a and label for a, label in zip(acted, labels))
    fp = sum(a and not label for a, label in zip(acted, labels))
    fn = sum(label and not a for a, label in zip(acted, labels))
    jev_input_tokens = sum(verdict.input_tokens for verdict in answered) if decision_model.bills_input_tokens else 0
    summary = {
        "rail": spec.name,
        "model": decision_model.name,
        "records": len(records),
        "positives": sum(labels),
        "acted": sum(acted),
        "uncertain": bands.count("uncertain"),
        "failures": bands.count("failure"),
        "bands": {
            str(flag).lower(): {band: sum(b == band and lab == flag for b, lab in zip(bands, labels)) for band in BANDS}
            for flag in (True, False)
        },
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "accuracy": round(sum(a == label for a, label in zip(acted, labels)) / len(records), 3),
        "median_ms": int(statistics.median(verdict.ms for verdict in answered)) if answered else None,
        "jev_input_tokens": jev_input_tokens,
        "cost_usd": cost_usd(jev_input_tokens, 0, 0, 0, None),
        "thresholds": {"allow": spec.thresholds.allow, "act": spec.thresholds.act},
        "labelled_set": str(labelled_set),
        "labelled_set_sha256": hashlib.sha256(data).hexdigest(),
        "finished_at": now_iso(),
    }
    summary.update(_detection_metrics(labels, bands, verdicts, spec.on_failure, records))
    job_dir = results_dir / spec.name / f"{datetime.now():%Y-%m-%d__%H-%M-%S-%f}__{decision_model.name}"
    job_dir.mkdir(parents=True)
    (job_dir / "verdicts.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "label": label,
                    "p": round(verdict.p, 3) if verdict else None,
                    "band": verdict.band if verdict else "failure",
                    "ms": verdict.ms if verdict else None,
                    "note": record.get("note"),
                }
                | ({"error": error} if error else {}),
                ensure_ascii=False,
            )
            + "\n"
            for record, label, verdict, error in zip(records, labels, verdicts, errors)
        ),
        encoding="utf-8",
    )
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["job_dir"] = str(job_dir)
    return summary


def _detection_metrics(
    labels: list[bool],
    bands: list[str],
    verdicts: list[Verdict | None],
    on_failure: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """F1, false-allow, ask rate, Wilson intervals, AUROC, and per-split copies of the same."""

    def pack(mask: list[bool]) -> dict[str, Any]:
        sub_labels = [lab for lab, keep in zip(labels, mask) if keep]
        sub_bands = [band for band, keep in zip(bands, mask) if keep]
        sub_verdicts = [verdict for verdict, keep in zip(verdicts, mask) if keep]
        n = len(sub_labels)
        asked = [band == "uncertain" or (band == "failure" and on_failure == "ask") for band in sub_bands]
        acted = [band == "act" or (band == "failure" and on_failure == "closed") for band in sub_bands]
        tp = sum(a and lab for a, lab in zip(acted, sub_labels))
        fp = sum(a and not lab for a, lab in zip(acted, sub_labels))
        fn = sum(lab and not a for a, lab in zip(acted, sub_labels))
        positives = sum(sub_labels)
        false_allow = sum(lab and not a and not q for lab, a, q in zip(sub_labels, acted, asked))
        f1 = (2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        scored = [(lab, verdict.p) for lab, verdict in zip(sub_labels, sub_verdicts) if verdict is not None]
        return {
            "records": n,
            "positives": positives,
            "acted": sum(acted),
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "precision_ci": _ci(tp, tp + fp) if tp + fp else None,
            "recall_ci": _ci(tp, tp + fn) if tp + fn else None,
            "false_allow": round(false_allow / positives, 3) if positives else None,
            "false_allow_ci": _ci(false_allow, positives) if positives else None,
            "ask_rate": round(sum(asked) / n, 3) if n else None,
            "ask_rate_ci": _ci(sum(asked), n) if n else None,
            "auroc": (
                round(auc, 3)
                if (auc := auroc([lab for lab, _ in scored], [p for _, p in scored])) is not None
                else None
            ),
        }

    metrics = pack([True] * len(labels))
    splits: dict[str, Any] = {}
    names = sorted({str(record["split"]) for record in records if record.get("split") not in (None, "")})
    for name in names:
        splits[name] = pack([str(record.get("split")) == name for record in records])
    if splits:
        metrics["splits"] = splits
    return metrics


def parser(spec: RailSpec) -> argparse.ArgumentParser:
    build = argparse.ArgumentParser(prog=f"s1a run {spec.name}", description=spec.description)
    build.add_argument(
        "--labelled-set",
        type=Path,
        default=spec.labelled_set,
        required=spec.labelled_set is None,
        help="JSONL records with a state and a boolean label; the spec's set when it names one",
    )
    models = RAIL_MODEL_NAMES + (("rule",) if spec.name == "tool_gate" else ())
    build.add_argument(
        "--model",
        choices=models,
        default="jev",
        help="who answers the question: jev, clm or laya-served over HTTP, laya in process"
        + (", rule for the static tool-gate baseline" if spec.name == "tool_gate" else ""),
    )
    return build


async def play(spec: RailSpec, args: argparse.Namespace, *, results_dir: Path) -> dict[str, Any]:
    """Evaluate the rail on its labelled set with a fresh model; the summary names its job folder."""
    if args.model == "rule":
        from s1a.agents.tool_gate import RuleNoulModel

        decision_model = RuleNoulModel()
    else:
        decision_model = build_model(args.model)
    try:
        await decision_model.warm()
        return await evaluate(spec, args.labelled_set, decision_model=decision_model, results_dir=results_dir)
    finally:
        await decision_model.close()
