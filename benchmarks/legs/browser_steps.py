"""Score browser decision backends step by step on myjev-browser-decisions (issue #109).

Each record of the dataset is one browser step in the request format jev-ultrafast sends to Jev: the page state,
the goal, the last actions and one Choice question per head (``operation`` and ``<op>_target``), with gold keys.
``predict`` sends every step through the repository's ``DecisionModel`` layer, so any text backend can answer;
``score`` re-scores saved evidence; ``thresholds`` turns a calibration run into a deferral threshold table.

The dataset is CC BY-NC-SA 4.0 and includes Mind2Web, whose authors ask that it not be republished: rows are
downloaded at run time from a pinned revision and checked against pinned hashes, never vendored.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from benchmarks.evidence import outside_repo

DATASET = "Mlchemist/myjev-browser-decisions"
REVISION = "8c0c5d532458f7ba25c9684b267694848e815d5a"
# LFS files carry their SHA-256; the small id list is a plain git blob, checked by its git object id.
SPLITS = {
    "benchmark": ("data/benchmark.jsonl", "sha256", "f95b894c4161ad7c059624916e6f7e4db9460bbd16c3b16a485481534436bd37"),
    "calibration": (
        "data/calibration.jsonl",
        "sha256",
        "c470db23180974d317ca4f511a43db408c275062ae622f9f6e6f71a6fd6bc910",
    ),
    "jev_answered_ids": ("data/benchmark_jev_answered_ids.txt", "git-blob", "fce3e76890c50848f6795c17feee2e69653d42ec"),
}
TARGET_OPERATIONS = ("CLICK", "TYPE_TEXT", "SELECT")
SHAPES = ("ours", "dataset")
ECE_BINS = 10
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(20))


# -- dataset files -----------------------------------------------------------------------------------------------


def file_digest(data: bytes, kind: str) -> str:
    if kind == "sha256":
        return hashlib.sha256(data).hexdigest()
    if kind == "git-blob":
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    raise ValueError(f"unknown digest kind {kind!r}")


def verify(path: Path, name: str) -> str:
    """The pinned digest of ``name`` when ``path`` matches it; ``ValueError`` otherwise."""
    _, kind, expected = SPLITS[name]
    actual = file_digest(path.read_bytes(), kind)
    if actual != expected:
        raise ValueError(f"{path} is not {DATASET}@{REVISION[:12]} {name}: {kind} {actual} != {expected}")
    return expected


def url_for(name: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SPLITS[name][0]}"


def fetch(names: Iterable[str], out: Path, *, transport: Any = None) -> dict[str, Path]:
    """Download the pinned files into ``out`` (outside the repository), verifying each before it is kept."""
    import httpx

    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    with httpx.Client(timeout=600, follow_redirects=True, transport=transport) as client:
        for name in names:
            target = out / Path(SPLITS[name][0]).name
            if not target.exists():
                response = client.get(url_for(name))
                response.raise_for_status()
                partial = target.with_suffix(target.suffix + ".part")
                partial.write_bytes(response.content)
                try:
                    verify(partial, name)
                except ValueError:
                    partial.unlink()
                    raise
                partial.rename(target)
            verify(target, name)
            saved[name] = target
    return saved


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


# -- request shapes ------------------------------------------------------------------------------------------------


def to_questions(record: dict[str, Any], shape: str = "ours") -> tuple[Any, dict[str, Any]]:
    """One record's observation and questions for ``DecisionModel.decide_many``.

    ``dataset`` keeps jev-ultrafast's text verbatim, so ``JevModel`` re-sends the dataset's body unchanged.
    ``ours`` keeps the state, the question names and the element rows (they already have the keys our action space
    sends) and replaces the operation descriptions and rules with the ones ``BrowserDecisionModel`` sends, so the
    score measures the model on the questions our agent asks. See docs/results/browser-steps.md for the
    differences that remain.
    """
    from s1a.browser import prompts
    from s1a.browser.action_space import _OPERATION_LABELS
    from s1a.decision_models import ChoiceQuestion, Observation

    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    request = record["request"]
    questions: dict[str, Any] = {}
    for name, question in request["questions"].items():
        if question.get("type") != "choice":
            raise ValueError(f"{record_id(record)}: question {name!r} is not a choice")
        instructions = question.get("instructions") or {}
        options = dict(question["criteria"])
        rules = instructions.get("rules") or ()
        if shape == "ours":
            if name == "operation":
                options = {op: _OPERATION_LABELS.get(op, text) for op, text in options.items()}
                rules = prompts.OPERATION_RULES["en"]
            else:
                rules = (prompts.OPERATION_RULES["en"], prompts.TARGET_RULES["en"])
        questions[name] = ChoiceQuestion(
            options,
            goal=instructions.get("goal", ""),
            rules=tuple(rules) if isinstance(rules, list) else rules,
            operation=instructions.get("operation", ""),
        )
    return Observation(request["state"]), questions


def record_id(record: dict[str, Any]) -> str:
    return str((record.get("meta") or {}).get("id") or "")


def wire_body(model_id: str, observation: Any, questions: dict[str, Any]) -> dict[str, Any]:
    """The Jev request body these questions become; saved with every step so the evidence shows what was asked."""
    from s1a.decision_models.jev import jev_question

    return {
        "model": model_id,
        "state": observation.state,
        "questions": {name: jev_question(question) for name, question in questions.items()},
    }


# -- predict -------------------------------------------------------------------------------------------------------


def agents_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


async def predict_steps(
    model: Any,
    records: Iterable[dict[str, Any]],
    results: Path,
    *,
    shape: str = "ours",
    limit: int | None = None,
    keep_request: bool = True,
) -> dict[str, int]:
    """One request per step, in dataset order, no retries; every outcome is written, failures included."""
    counts = {"attempted": 0, "failed": 0}
    with results.open("x", encoding="utf-8") as stream:
        for record in records:
            if limit is not None and counts["attempted"] >= limit:
                break
            step: dict[str, Any] = {"id": record_id(record), "ts": time.time()}
            observation, questions = to_questions(record, shape)
            if keep_request:
                step["request"] = wire_body(model.model, observation, questions)
            started = time.perf_counter()
            try:
                decision = await model.decide_many(observation, questions, attempts=1)
            except Exception as exc:  # noqa: BLE001 - every failure is evidence, scored as wrong
                step["error"] = f"{type(exc).__name__}: {exc}"
                counts["failed"] += 1
            else:
                step["answers"] = {name: answer.as_dict() for name, answer in decision.answers.items()}
                step["decision_ms"] = decision.latency_ms
                step["model"] = decision.model
                step["usage"] = {
                    "input_tokens": decision.usage.input_tokens,
                    "output_tokens": decision.usage.output_tokens,
                    "known": decision.usage.known,
                }
            step["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)
            counts["attempted"] += 1
            stream.write(json.dumps(step, ensure_ascii=False) + "\n")
            stream.flush()
    return counts


def predict(args: argparse.Namespace) -> None:
    from s1a.decision_models import build_model

    if args.split in SPLITS and not args.unpinned:
        digest = verify(args.data, args.split)
    else:
        digest = file_digest(args.data.read_bytes(), "sha256")
    output = outside_repo(args.out)
    environment = json.loads(args.run_meta.read_text(encoding="utf-8")) if args.run_meta else {}
    model = build_model(args.model)
    output.mkdir(parents=True, exist_ok=False)
    meta = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": args.split,
        "data_sha256_or_pin": digest,
        "pinned": args.split in SPLITS and not args.unpinned,
        "shape": args.shape,
        "backend": args.model,
        "model": model.model,
        "agents_sha": agents_sha(),
        "argv": sys.argv,
        "limit": args.limit,
        "environment": environment,
        "started": time.time(),
    }
    (output / "run.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    async def run() -> dict[str, int]:
        await model.warm()
        try:
            return await predict_steps(
                model, read_records(args.data), output / "results.jsonl", shape=args.shape, limit=args.limit
            )
        finally:
            await model.close()

    counts = asyncio.run(run())
    completion = {**counts, "finished": time.time()}
    (output / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(completion))


# -- scoring -------------------------------------------------------------------------------------------------------


def judge(record: dict[str, Any], step: dict[str, Any] | None) -> dict[str, Any]:
    """One step against its gold: a missing or failed step is wrong on every metric, never skipped."""
    gold = record["gold"]
    gold_op = gold["operation"]
    target_head = f"{gold_op.lower()}_target"
    gold_keys = [str(key) for key in gold.get(target_head) or []]
    out: dict[str, Any] = {
        "id": record_id(record),
        "gold_operation": gold_op,
        "has_target": bool(gold_keys),
        "answered": False,
        "operation_ok": False,
        "element_ok": False,
        "step_ok": False,
        "confidence": None,
        "operation_probabilities": None,
        "latency_ms": None,
    }
    if step is None or "answers" not in step:
        return out
    answers = step["answers"]
    operation = answers.get("operation") or {}
    out["answered"] = True
    out["operation_ok"] = operation.get("choice") == gold_op
    out["confidence"] = operation.get("confidence")
    out["operation_probabilities"] = operation.get("probabilities")
    out["latency_ms"] = step.get("decision_ms")
    if gold_keys:
        out["element_ok"] = str((answers.get(target_head) or {}).get("choice")) in gold_keys
        out["step_ok"] = out["operation_ok"] and out["element_ok"]
    else:
        out["step_ok"] = out["operation_ok"]
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def multiclass_brier(probabilities: dict[str, float], gold: str) -> float:
    """Sum over the offered operations (and the gold one) of the squared gap to the one-hot gold vector."""
    keys = set(probabilities) | {gold}
    return sum((float(probabilities.get(key, 0.0)) - float(key == gold)) ** 2 for key in keys)


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {"value": numerator / denominator if denominator else None, "n": numerator, "of": denominator}


def summarize(judged: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracies with their denominators, calibration of the operation confidence and decision latency."""
    total = len(judged)
    targeted = [row for row in judged if row["has_target"]]
    answered = [row for row in judged if row["answered"]]
    confident = [row for row in answered if isinstance(row["confidence"], (int, float))]
    brier = (
        statistics.fmean((row["confidence"] - float(row["operation_ok"])) ** 2 for row in confident)
        if confident
        else None
    )
    multiclass = [
        multiclass_brier(row["operation_probabilities"], row["gold_operation"])
        for row in confident
        if isinstance(row["operation_probabilities"], dict)
    ]
    latencies = [float(row["latency_ms"]) for row in answered if isinstance(row["latency_ms"], (int, float))]
    return {
        "steps": total,
        "operation_accuracy": ratio(sum(row["operation_ok"] for row in judged), total),
        "element_accuracy": ratio(sum(row["element_ok"] for row in targeted), len(targeted)),
        "step_success": ratio(sum(row["step_ok"] for row in judged), total),
        "invalid_or_failed": ratio(total - len(answered), total),
        "operation_confidence_brier": brier,
        "operation_multiclass_brier": statistics.fmean(multiclass) if multiclass else None,
        "operation_confidence_ece": ece(confident),
        "calibration_steps": len(confident),
        "decision_ms_p50": percentile(latencies, 0.5),
        "decision_ms_p95": percentile(latencies, 0.95),
        "by_gold_operation": {
            op: ratio(sum(row["step_ok"] for row in judged if row["gold_operation"] == op), count)
            for op, count in sorted(_count(row["gold_operation"] for row in judged).items())
        },
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def ece(rows: list[dict[str, Any]], bins: int = ECE_BINS) -> float | None:
    """Expected calibration error of the operation confidence against operation correctness, equal-width bins."""
    if not rows:
        return None
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(bins)]
    for row in rows:
        buckets[min(bins - 1, int(float(row["confidence"]) * bins))].append(row)
    return sum(
        len(bucket)
        / len(rows)
        * abs(statistics.fmean(r["confidence"] for r in bucket) - statistics.fmean(r["operation_ok"] for r in bucket))
        for bucket in buckets
        if bucket
    )


def load_run(run: Path) -> dict[str, dict[str, Any]]:
    steps: dict[str, dict[str, Any]] = {}
    for step in read_records(run / "results.jsonl"):
        if step["id"] in steps:
            raise ValueError(f"{run}: step {step['id']} appears twice")
        steps[step["id"]] = step
    return steps


def judge_run(run: Path, data: Path) -> list[dict[str, Any]]:
    steps = load_run(run)
    records = list(read_records(data))
    known = {record_id(record) for record in records}
    orphans = sorted(set(steps) - known)
    if orphans:
        raise ValueError(f"{run}: {len(orphans)} step(s) are not in {data}, e.g. {orphans[0]}")
    return [judge(record, steps.get(record_id(record))) for record in records]


def score(args: argparse.Namespace) -> dict[str, Any]:
    judged = judge_run(args.run, args.data)
    report: dict[str, Any] = {"run": str(args.run), "full": summarize(judged)}
    if args.limit is not None:
        report["attempted_prefix"] = summarize(judged[: args.limit])
    ids = read_ids(args.ids)
    if ids is not None:
        report["jev_answered"] = summarize([row for row in judged if row["id"] in ids])
    meta_path = args.run / "run.meta.json"
    if meta_path.exists():
        report["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return report


def threshold_table(judged: list[dict[str, Any]], thresholds: Iterable[float] = THRESHOLDS) -> list[dict[str, Any]]:
    """Per threshold: how many steps the backend keeps (operation confidence at or above it) and how they fare.

    A failed step has no confidence, so it is always deferred; its share stays in the denominator.
    """
    total = len(judged)
    table = []
    for threshold in thresholds:
        kept = [row for row in judged if isinstance(row["confidence"], (int, float)) and row["confidence"] >= threshold]
        table.append(
            {
                "threshold": threshold,
                "kept": ratio(len(kept), total),
                "kept_step_success": ratio(sum(row["step_ok"] for row in kept), len(kept)),
                "deferred": ratio(total - len(kept), total),
            }
        )
    return table


def thresholds(args: argparse.Namespace) -> dict[str, Any]:
    judged = judge_run(args.run, args.data)
    return {"run": str(args.run), "steps": len(judged), "table": threshold_table(judged)}


# -- CLI -----------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    from s1a.decision_models import DECISION_MODEL_NAMES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    get = commands.add_parser("fetch", help="download and verify the pinned files")
    get.add_argument("--out", type=Path, required=True)
    get.add_argument("--split", choices=sorted(SPLITS), action="append", required=True)

    run = commands.add_parser("predict", help="send every step through one decision backend")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", required=True, help="benchmark, calibration, or a name for an unpinned file")
    run.add_argument("--model", choices=[n for n in DECISION_MODEL_NAMES if n not in ("rule",)], required=True)
    run.add_argument("--shape", choices=SHAPES, default="ours")
    run.add_argument("--limit", type=int)
    run.add_argument("--run-meta", type=Path, help="JSON describing the backend: revision, hardware, timeout")
    run.add_argument("--unpinned", action="store_true", help="accept a file that is not the pinned split")
    run.add_argument("--out", type=Path, required=True)

    rescore = commands.add_parser("score", help="re-score a saved run")
    rescore.add_argument("--run", type=Path, required=True)
    rescore.add_argument("--data", type=Path, required=True)
    rescore.add_argument("--ids", type=Path, help="benchmark_jev_answered_ids.txt for the Jev-answered subset")
    rescore.add_argument("--limit", type=int, help="also report the first N steps when the run used --limit")

    table = commands.add_parser("thresholds", help="deferral threshold table from a calibration run")
    table.add_argument("--run", type=Path, required=True)
    table.add_argument("--data", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "fetch":
        for name, path in fetch(args.split, outside_repo(args.out)).items():
            print(f"{name}: {path}")
    elif args.command == "predict":
        predict(args)
    elif args.command == "score":
        print(json.dumps(score(args), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(thresholds(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
