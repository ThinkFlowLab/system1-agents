"""Score the same frozen Omni raw offline with the pinned public231 scorer."""

import argparse
import copy
import importlib.util
from pathlib import Path

from benchmarks.client import decode
from benchmarks.evidence import (
    body_for,
    digest,
    dump,
    load_reference,
    outside_repo,
    score_record,
    summarize,
    write_json,
)

COLLECTOR_SHA256 = "5aa20a2c612b7da4cd2f1c1442d37e77feb9d6b9ada4375b59a08f4cada63044"
RAW_CLOCK = "post_to_body_json_and_schema_validation"


def load_collector(path):
    # Only this reviewed interface is executable; never import a file selected by raw metadata.
    if digest(path.read_bytes()) != COLLECTOR_SHA256:
        raise ValueError("Frozen Omni collector hash mismatch")
    spec = importlib.util.spec_from_file_location("public231_omni_raw", path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.SOURCE_SHA256 = COLLECTOR_SHA256
    return collector


def adapt_record(upstream, identity, task, saved, model, number, raw_file):
    response = None
    if saved["response"] is not None:
        try:
            response = decode(saved["response"])
        except ValueError:
            pass
    # invalid_response is the Omni wire verdict, not the pinned scorer's probability verdict.
    client_error = saved["error"] if saved["error_kind"] in ("timeout", "transport", "interrupted") else None
    raw = dict(
        response=response,
        response_body=saved["response"],
        http_status=saved["status"],
        client_error=client_error,
        ts=None,
        latency_s=None,
    )
    row = score_record(
        upstream, identity, task, raw, model, number, "measured", saved["response_sha256"], latency_basis=None
    )
    row.update(
        wire_valid=saved["error_kind"] is None,
        wire_error_kind=saved["error_kind"],
        raw_record_state=saved["state"],
        raw_file=raw_file,
        raw_latency_basis=RAW_CLOCK,
    )
    return row


def verify_inputs(manifest_path, items, dataset):
    manifest = decode(manifest_path.read_text(encoding="utf-8"))
    path = manifest_path.parent / manifest["output"]["path"]
    if path.resolve() != manifest_path.with_name("public231.jsonl").resolve():
        raise ValueError("Frozen input path must be public231.jsonl beside its manifest")
    data = path.read_bytes()
    if digest(data) != manifest["output"]["sha256"]:
        raise ValueError("Frozen input hash mismatch")
    if (
        manifest["sources"]["jevbench_pin"] != dataset["jevbench_pin"]
        or manifest["sources"]["reference_dataset_hash"] != dataset["dataset_hash"]
    ):
        raise ValueError("Frozen input reference mismatch")
    expected = []
    for identity, task in items:
        target = task.expected == "yes" if task.question["type"] == "noul" else task.expected
        expected.append(
            dict(
                id=task.id,
                request=body_for(task, None),
                expected={"decision": target},
                source={**identity, "jevbench_pin": dataset["jevbench_pin"]},
                labels=task.labels,
                provenance=task.provenance,
                group=task.group,
                reference_expected=task.expected,
            )
        )
    if data != "".join(dump(row) + "\n" for row in expected).encode("utf-8"):
        raise ValueError("Input order, requests or evaluator fields differ from the pinned dataset")
    return data


def aggregate(runs, input_manifest, collector, model_config_id):
    if not runs or len({path.resolve() for path in runs}) != len(runs) or not model_config_id.strip():
        raise ValueError("Provide unique planned round paths and a model_config_id")
    upstream, items, dataset = load_reference()
    frozen = verify_inputs(input_manifest, items, dataset)
    tasks = [task for _, task in items]
    rounds, all_rows, pooled_tasks, attempted_pooled = [], [], [], []
    binding = None
    for number, run in enumerate(runs, 1):
        records, source, config, started_ids = {}, None, None, set()
        if run.exists():
            if (run / "requests.jsonl").read_bytes() != frozen:
                raise ValueError("Omni requests differ from the complete frozen 231 plan")
            config = decode((run / "config.json").read_text(encoding="utf-8"))
            if config["runner_sha256"] != COLLECTOR_SHA256 or config["latency_basis"] != RAW_CLOCK:
                raise ValueError("Omni runner or clock differs from the frozen interface")
            identity = {key: value for key, value in config.items() if key not in ("python", "httpx")}
            metadata = dict(config.get("metadata", {}))
            recorded_round = metadata.pop("round", None)
            if "round" in config.get("metadata", {}) and (type(recorded_round) is not int or recorded_round != number):
                raise ValueError("Recorded round differs from the ordered planned round")
            identity["metadata"] = metadata
            if binding is not None and identity != binding:
                raise ValueError("Model configuration differs between planned rounds")
            binding = identity
            collector.summarize_saved(run)  # raw/config/response hashes, typed decode and complete plan associations
            completion = decode((run / "completion.json").read_text(encoding="utf-8"))
            started_ids = set(completion["attempted_ids"])
            source = dict(
                run=str(run.resolve()),
                config_sha256=completion["config_sha256"],
                manifest_sha256=completion["manifest_sha256"],
                responses_sha256=completion["responses_sha256"],
                stop_reason=completion["stop_reason"],
                phase=config["phase"],
                recorded_round=recorded_round,
            )
            if (run / "responses.jsonl").exists():
                records = {
                    row["id"]: row
                    for row in (
                        decode(line) for line in (run / "responses.jsonl").read_text(encoding="utf-8").splitlines()
                    )
                }
        rows, attempted = [], []
        for identity, task in items:
            if task.id in records:
                row = adapt_record(
                    upstream, identity, task, records[task.id], config["model"], number, str(run / "responses.jsonl")
                )
                attempted.append(row)
            else:
                started = task.id in started_ids
                row = dict(
                    identity,
                    round=number,
                    attempted=started,
                    ok=False,
                    valid=False,
                    strict_valid=False,
                    renormalized=False,
                    correct=False,
                    predicted=None,
                    probs=None,
                    status="started_without_saved_terminal" if started else "unexecuted",
                    expected=task.expected,
                    model=None,
                    model_configured=config["model"] if config else None,
                    probs_source="unavailable",
                    raw_file=None,
                    raw_sha256=None,
                    latency_s=None,
                    latency_basis=None,
                    ts=None,
                )
                if started:
                    attempted.append(row)
            rows.append(dict(row, model_config_id=model_config_id, planned_run=str(run.resolve())))
            clone = copy.copy(task)
            clone.id = f"{number}:{task.id}"
            clone.group = f"{number}:{task.group}" if task.group else None
            pooled_tasks.append(clone)
        all_rows.extend(rows)
        attempted_pooled.extend(dict(row, task_id=f"{number}:{row['task_id']}") for row in attempted)
        rounds.append(dict(round=number, source=source, **summarize(upstream, tasks, attempted)))
    result = dict(
        schema="public231-omni-offline-quality-v1",
        model_config_id=model_config_id,
        distinct_tasks=231,
        rounds=rounds,
        pooled=summarize(upstream, pooled_tasks, attempted_pooled),
        dataset=dataset,
        input_manifest_sha256=digest(input_manifest.read_bytes()),
        model_binding=binding,
        identity_basis="saved metadata; build ancestry not independently proven",
        latency_reported=False,
        raw_latency_basis=RAW_CLOCK,
        evidence_complete=all(row["source"] and row["source"]["stop_reason"] == "completed" for row in rounds),
    )
    return result, all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True, help="one path for each planned round")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True, help="frozen Omni bench.py; only imported offline")
    parser.add_argument("--model-config-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result, records = aggregate(args.run, args.input_manifest, load_collector(args.collector), args.model_config_id)
    output = outside_repo(args.out)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result)
    (output / "records.jsonl").write_text("".join(dump(row) + "\n" for row in records), encoding="utf-8")
    print(dump(result))


if __name__ == "__main__":
    main()
