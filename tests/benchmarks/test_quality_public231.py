"""Deterministic saved Omni responses, scored by the pinned public231 oracle."""

import copy
import json
import os
from pathlib import Path

import pytest

from benchmarks.evidence import body_for, digest, dump, load_reference
from benchmarks.quality_public231 import adapt_record, aggregate, load_collector


@pytest.fixture
def reference():
    if not os.environ.get("JEVBENCH_ROOT"):
        pytest.skip("Set JEVBENCH_ROOT to the pinned reference")
    return load_reference()


@pytest.fixture
def collector():
    source = os.environ.get("OMNI_BENCH_SOURCE")
    if not source:
        pytest.skip("Set OMNI_BENCH_SOURCE to the frozen raw collector")
    return load_collector(Path(source))


@pytest.fixture
def input_manifest(tmp_path, reference):
    _, items, dataset = reference
    rows = [
        dict(
            id=task.id,
            request=body_for(task, None),
            expected={"decision": task.expected == "yes" if task.question["type"] == "noul" else task.expected},
            source={**identity, "jevbench_pin": dataset["jevbench_pin"]},
            labels=task.labels,
            provenance=task.provenance,
            group=task.group,
            reference_expected=task.expected,
        )
        for identity, task in items
    ]
    data = "".join(dump(row) + "\n" for row in rows).encode("utf-8")
    assert digest(data) == "00d1f10ca1142499a598b5f1c6602d329cce1cef04c961c46fc0318daa2c67d3"
    (tmp_path / "public231.jsonl").write_bytes(data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        dump(
            dict(
                output={"path": "public231.jsonl", "sha256": digest(data)},
                sources={"jevbench_pin": dataset["jevbench_pin"], "reference_dataset_hash": dataset["dataset_hash"]},
            )
        ),
        encoding="utf-8",
    )
    return manifest


def response_record(answer, *, kind="invalid_response", status=200):
    text = dump({"answers": {"decision": answer}, "model": "configured"})
    return dict(
        id="unused",
        state="completed",
        status=status,
        response=text,
        response_sha256=digest(text.encode()),
        answers={},
        error_kind=kind,
        error="wire schema rejected" if kind else None,
        latency_ms=12.0,
    )


def test_wire_rejection_does_not_discard_pinned_distribution(reference):
    upstream, items, _ = reference
    identity, task = next(row for row in items if row[1].question["type"] == "noul")
    raw = response_record({"type": "noul", "probabilities": {"no": 0.5, "yes": 0.505}})
    row = adapt_record(upstream, identity, task, raw, "configured", 1, "fixture")
    assert row["valid"] and row["renormalized"] and not row["strict_valid"]
    assert row["wire_valid"] is False and row["ok"] is True
    assert row["predicted"] == "yes"
    assert row["latency_s"] is None and row["latency_basis"] is None and row["ts"] is None


def test_scalar_tie_and_label_only(reference):
    upstream, items, _ = reference
    for value, expected in ((0.5, "no"), (0.8, "yes")):
        identity, task = next(row for row in items if row[1].question["type"] == "noul")
        row = adapt_record(
            upstream, identity, task, response_record({"type": "noul", "noul": value}), "configured", 1, "fixture"
        )
        assert row["predicted"] == expected and row["probs_source"] == "native_noul_scalar"
    identity, task = next(row for row in items if row[1].question["type"] == "choice")
    row = adapt_record(
        upstream,
        identity,
        task,
        response_record({"type": "choice", "choice": task.expected}),
        "configured",
        1,
        "fixture",
    )
    assert not row["valid"] and row["probs"] is None


def test_ordinal_accuracy_uses_argmax(reference):
    upstream, items, _ = reference
    identity, original = next(row for row in items if row[1].question["type"] == "score")
    task = copy.copy(original)
    task.labels, task.expected = ["0", "1", "2"], 0
    probabilities = {"0": 0.4, "1": 0.35, "2": 0.25}
    row = adapt_record(
        upstream,
        identity,
        task,
        response_record({"type": "score", "score": 0.85, "probabilities": probabilities}),
        "configured",
        1,
        "fixture",
    )
    assert row["predicted"] == "0" and row["correct"] is (task.expected == 0)


@pytest.mark.parametrize("kind,status", [("http_422", 422), ("timeout", None)])
def test_failures_are_scored_as_invalid(reference, kind, status):
    upstream, items, _ = reference
    identity, task = items[0]
    raw = response_record({"type": "choice", "probabilities": {}}, kind=kind, status=status)
    if status is None:
        raw.update(response=None, response_sha256=None)
    row = adapt_record(upstream, identity, task, raw, "configured", 1, "fixture")
    assert not row["valid"] and not row["correct"] and row["attempted"]


def saved_run(tmp_path, collector, input_path, *, count=231):
    cases = collector.load_cases(input_path)
    run = tmp_path / "round-1"
    run.mkdir()
    (run / "requests.jsonl").write_bytes(input_path.read_bytes())
    config = dict(
        manifest_sha256=digest(input_path.read_bytes()),
        runner_sha256=collector.SOURCE_SHA256,
        model="configured",
        metadata={"model_revision": "fixture"},
        phase="measured",
        warmup_manifest_sha256=None,
        latency_basis="post_to_body_json_and_schema_validation",
    )
    (run / "config.json").write_text(dump(config))
    records = []
    for case in cases[:count]:
        q = case["request"]["questions"]["decision"]
        answer = {"type": q["type"]}
        if q["type"] == "noul":
            answer["noul"] = float(case["expected"]["decision"])
        else:
            labels = list(q["criteria"]) if q["type"] == "choice" else list(map(str, range(len(q["criteria"]))))
            answer["probabilities"] = {key: float(key == str(case["expected"]["decision"])) for key in labels}
            answer[q["type"]] = case["expected"]["decision"]
        raw = response_record(answer, kind=None)
        raw.update(id=case["id"], answers=collector.read_answers(case, json.loads(raw["response"])))
        records.append(raw)
    (run / "responses.jsonl").write_text("".join(dump(row) + "\n" for row in records))
    state = dict(
        manifest_sha256=config["manifest_sha256"],
        config_sha256=digest((run / "config.json").read_bytes()),
        responses_sha256=digest((run / "responses.jsonl").read_bytes()),
        attempted_ids=[r["id"] for r in records],
        completed_ids=[r["id"] for r in records],
        active_ids=[],
        started_at=1.0,
        finished_at=2.0,
        wall_seconds=1.0,
        stop_reason="completed" if count == 231 else "interrupted",
    )
    (run / "completion.json").write_text(dump(state))
    return run


def test_full_plan_missing_round_and_raw_integrity(tmp_path, collector, input_manifest):
    manifest = input_manifest
    inputs = manifest.with_name("public231.jsonl")
    run = saved_run(tmp_path, collector, inputs, count=230)
    result, records = aggregate(
        [run, tmp_path / "missing", tmp_path / "missing-3"], manifest, collector, "fixture-config"
    )
    assert result["distinct_tasks"] == 231 and result["pooled"]["planned"] == 693
    assert result["rounds"][0]["accuracy"] == 230 / 231
    assert result["rounds"][1]["accuracy"] == 0 and result["rounds"][1]["coverage"] == 0
    assert len(records) == 693 and sum(row["attempted"] for row in records) == 230
    (run / "responses.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        aggregate([run], manifest, collector, "fixture-config")


def test_source_hash_is_required(tmp_path):
    source = tmp_path / "bench.py"
    source.write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="collector hash"):
        load_collector(source)


def test_recorded_round_identity_is_separate_from_stable_model_config(tmp_path, collector, input_manifest):
    import shutil

    manifest = input_manifest
    first = saved_run(tmp_path, collector, manifest.with_name("public231.jsonl"))
    runs = [first, tmp_path / "round-2", tmp_path / "round-3"]
    for path in runs[1:]:
        shutil.copytree(first, path)

    def set_metadata(path, **values):
        config = json.loads((path / "config.json").read_text())
        config["metadata"].update(values)
        (path / "config.json").write_text(dump(config))
        completion = json.loads((path / "completion.json").read_text())
        completion["config_sha256"] = digest((path / "config.json").read_bytes())
        (path / "completion.json").write_text(dump(completion))

    for number, run in enumerate(runs, 1):
        set_metadata(run, round=number)
    result, records = aggregate(runs, manifest, collector, "fixture-config")
    assert result["distinct_tasks"] == 231 and result["pooled"]["planned"] == 693
    assert len(records) == 693 and result["evidence_complete"]
    assert [row["source"]["recorded_round"] for row in result["rounds"]] == [1, 2, 3]
    with pytest.raises(ValueError, match="Recorded round"):
        aggregate([runs[1], runs[0], runs[2]], manifest, collector, "fixture-config")
    set_metadata(runs[1], round=True)
    with pytest.raises(ValueError, match="Recorded round"):
        aggregate(runs, manifest, collector, "fixture-config")
    set_metadata(runs[1], round=2, model_revision="other")
    with pytest.raises(ValueError, match="Model configuration"):
        aggregate(runs, manifest, collector, "fixture-config")


def test_started_without_terminal_keeps_attempt_and_full_denominator(tmp_path, collector, reference, input_manifest):
    manifest = input_manifest
    run = saved_run(tmp_path, collector, manifest.with_name("public231.jsonl"), count=0)
    _, items, _ = reference
    state = json.loads((run / "completion.json").read_text())
    state.update(
        attempted_ids=[items[0][1].id],
        active_ids=[items[0][1].id],
        finished_at=None,
        wall_seconds=None,
        stop_reason=None,
    )
    (run / "completion.json").write_text(dump(state))
    result, rows = aggregate([run], manifest, collector, "fixture-config")
    assert rows[0]["attempted"] and rows[0]["status"] == "started_without_saved_terminal"
    assert result["rounds"][0]["attempted"] == 1 and result["rounds"][0]["planned"] == 231
    assert result["rounds"][0]["upstream_attempted_metrics"]["brier_mean"] is None
    assert not result["evidence_complete"]
