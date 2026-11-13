"""Pinned upstream scoring and reproducible public231 evidence."""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmarks.client import decode

PIN = "7ce310c7262ed49cc85853339a8a42459298e3f3"
HASHES = {
    "jevbench/__init__.py": "64c28dc1534502c429a3b67eee79ab7909fab995b7d116c9b089f0f3bb8652f7",
    "jevbench/tasks.py": "b9c2f8ba9a7301519656d79d47c31b8ab0542a5545498ce3b339d5e93b976812",
    "jevbench/scoring.py": "6aa17d212ea20a3eeb6876149367842e8462c11dcb54d73df062fb406368331e",
    "jevbench/summarize.py": "aa5f24da407d004cd80bad1cd957b6987aba152591084c096d1de51dd9775cbe",
    "jevbench/metrics.py": "4038c8423ad1cf28601afb6babe777b71f08e379b9ca125b09b52c0ac7d60515",
    "datasets/public/easy.jsonl": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "datasets/public/original.jsonl": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "datasets/public/hard.jsonl": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(dump(value) + "\n")


def outside_repo(path):
    path = Path(path).resolve()
    if path.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Evidence must be outside the source repository")
    return path


def load_reference():
    root = Path(os.environ["JEVBENCH_ROOT"]).resolve()
    for name, expected in HASHES.items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f"Pinned reference hash mismatch: {name}")
    for name, module in tuple(sys.modules.items()):
        if name == "jevbench" or name.startswith("jevbench."):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise ValueError("Another JevBench root is already imported")
    sys.path.insert(0, str(root))
    upstream = SimpleNamespace(
        **{name: importlib.import_module(f"jevbench.{name}") for name in ("tasks", "scoring", "summarize", "metrics")}
    )
    items = []
    for tier in ("easy", "original", "hard"):
        for line, text in enumerate(
            (root / f"datasets/public/{tier}.jsonl").read_text(encoding="utf-8").split("\n")[:-1], 1
        ):
            task = upstream.tasks.Task.from_dict(decode(text))
            identity = dict(
                suite=f"jevbench-{tier}",
                task_id=task.id,
                source_file=f"{tier}.jsonl",
                source_line=line,
                family=task.family,
                tier=tier,
                split=task.split,
            )
            items.append((identity, task))
    dataset = dict(
        jevbench_pin=PIN,
        root=str(root),
        hashes=HASHES,
        dataset_hash=upstream.tasks.dataset_hash([task for _, task in items]),
    )
    return upstream, items, dataset


def body_for(task, model):
    body = {"state": task.state, "questions": {"decision": task.question}, "thinking": {"enabled": False}}
    if model is not None:
        body["model"] = model
    return body


def score_record(
    upstream,
    identity,
    task,
    raw,
    model,
    round_number,
    kind,
    raw_hash,
    *,
    latency_basis="client_single_request_wall_seconds",
):
    status, error, probs, source = "ok", None, None, "unavailable"
    response = raw["response"]
    code = raw["http_status"]
    if raw["client_error"] is not None:
        status, error = "client_error", raw["client_error"]
    elif code is None or not 200 <= code < 300:
        status, error = ("refused" if code == 422 else "http_error"), raw["response_body"]
    else:
        answers = response.get("answers") if isinstance(response, dict) else None
        answer = answers.get("decision") if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != task.question["type"]:
            status, error = "parse_error", "Missing or mismatched typed decision answer"
        else:
            probs = answer.get("probabilities")
            source = "response_probabilities" if probs is not None else "unavailable"
            if probs is None and task.question["type"] == "noul":
                value = answer.get("noul")
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    probs, source = {"no": 1 - value, "yes": value}, "native_noul_scalar"
    try:
        scored = upstream.scoring.score_task(probs, task)
    except OverflowError:
        scored = upstream.scoring.score_task(None, task)
        scored["error"] = "Probability exceeds the scorer's numeric range"
    return dict(
        identity,
        round=round_number,
        kind=kind,
        ts=raw["ts"],
        attempted=True,
        ok=status == "ok",
        status=status,
        status_code=code,
        error=error or scored.get("error"),
        latency_s=raw["latency_s"],
        latency_basis=latency_basis,
        **{key: scored.get(key) for key in ("valid", "strict_valid", "renormalized", "correct", "predicted", "probs")},
        expected=task.expected,
        candidate_count=len(task.labels),
        probs_as_returned=probs,
        probs_source=source,
        score_source="jevbench_7ce310c7",
        model_configured=model,
        model=response.get("model") if isinstance(response, dict) else None,
        usage=response.get("usage") if isinstance(response, dict) else None,
        raw_sha256=raw_hash,
    )


def summarize(upstream, tasks, records):
    # Upstream accuracy is attempted-only; keep it explicitly diagnostic.
    projections = [dict(row, model=str(row["model"]) if row["model"] is not None else "<missing>") for row in records]
    original = upstream.summarize.summarize(tasks, projections)
    correct = sum(bool(row["correct"]) for row in records)
    latency = {
        name: upstream.metrics.latency_summary([row["latency_s"] for row in rows])
        for name, rows in (
            ("all_attempts", records),
            ("success", [r for r in records if r["ok"]]),
            ("failure", [r for r in records if not r["ok"]]),
        )
    }
    return dict(
        planned=len(tasks),
        attempted=len(records),
        valid=sum(r["valid"] for r in records),
        correct=correct,
        accuracy=correct / len(tasks),
        coverage=len(records) / len(tasks),
        latency=latency,
        upstream_attempted_metrics=original,
    )
