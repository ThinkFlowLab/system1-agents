"""Collect serial public231 rounds or validate and aggregate saved evidence."""

import argparse
import copy
from contextlib import closing
import math
import os
import sys
import time
from pathlib import Path

from benchmarks.client import DecisionClient, decode
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


def validate_environment(meta):
    for group, fields in (("checkpoint", ("repo", "revision")), ("backend", ("name", "version"))):
        if any(not meta.get(group, {}).get(field) for field in fields):
            raise ValueError(f"run-meta requires actual {group}: {fields}")
    value = meta.get("sampling", {}).get("temperature")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("run-meta requires actual positive sampling.temperature")
    if not meta.get("hardware"):
        raise ValueError("run-meta requires actual hardware")


def predict(args):
    upstream, items, dataset = load_reference()
    environment = args.run_meta.read_bytes()
    meta = decode(environment)
    validate_environment(meta)
    if "stop_reason" in meta:
        raise ValueError("stop_reason is collector-owned state; omit it from the supplied environment")
    if args.rounds < 1 or not 0 <= args.warmup <= 231 or not 1 <= args.limit <= 231:
        raise ValueError("Invalid rounds, warmup or limit")
    output = outside_repo(args.out)
    plan = dict(
        warmup=args.warmup,
        rounds=args.rounds,
        tasks=231,
        limit=args.limit,
        order="dataset sequential",
        identities=[identity for identity, _ in items],
    )
    collector = dict(
        run_id=output.name,
        argv=sys.argv,
        endpoint=args.endpoint,
        path=args.path,
        model_configured=args.model,
        response_parse="typed",
        thinking=False,
        dataset=dataset,
        plan=plan,
        env_file=str(args.run_meta.resolve()),
        env_sha256=digest(environment),
        stop_reason=None,
        ts=time.time(),
    )
    if any(key in meta and meta[key] != value for key, value in collector.items()):
        raise ValueError("Supplied environment conflicts with collector metadata")
    meta.update(collector)
    with closing(DecisionClient(args.endpoint, path=args.path, model=args.model, timeout_s=args.timeout)) as client:
        output.mkdir(parents=True, exist_ok=False)
        with (output / "environment.json").open("xb") as stream:
            stream.write(environment)
        write_json(output / "run.meta.json", meta)
        stop, errors = None, 0
        try:
            for number in range(args.rounds + 1):
                kind = "warmup" if number == 0 else "measured"
                directory = output / ("warmup" if number == 0 else f"round-{number}")
                (directory / "raw").mkdir(parents=True)
                selected = items[: args.warmup if number == 0 else args.limit]
                write_json(
                    directory / "predict.meta.json", dict(round=number, planned=args.warmup if number == 0 else 231)
                )
                with (directory / ("records.jsonl" if number == 0 else "results.jsonl")).open(
                    "x", encoding="utf-8"
                ) as stream:
                    for identity, task in selected:
                        raw = client.request(body_for(task, args.model))
                        raw_path = directory / "raw" / (digest(task.id.encode()) + ".json")
                        write_json(raw_path, raw)
                        record = score_record(
                            upstream, identity, task, raw, args.model, number, kind, digest(raw_path.read_bytes())
                        )
                        stream.write(dump(record) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        if raw["client_error"] == "KeyboardInterrupt":
                            stop = "interrupted"
                            break
                        errors = errors + 1 if not record["ok"] and record["status_code"] != 422 else 0
                        if record["status_code"] in (401, 403, 429) or errors >= 3:
                            stop = "access/rate limit or three consecutive infrastructure errors"
                            break
                if stop:
                    break
        except KeyboardInterrupt:
            stop = "interrupted"
        except Exception as exc:
            stop = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_json(output / "completion.json", dict(stop_reason=stop, ts=time.time()))
            meta["stop_reason"] = stop
            write_json(output / "run.meta.final.json", meta)
            (output / "run.meta.final.json").replace(output / "run.meta.json")
    return output


def read_round(directory, expected, upstream, model, number):
    path = directory / ("records.jsonl" if number == 0 else "results.jsonl")
    records = []
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            records = [decode(line) for line in stream]
    if len(records) > len(expected):
        raise ValueError("Too many or duplicate records")
    expected_raw = {digest(task.id.encode()) + ".json" for _, task in expected[: len(records)]}
    actual_raw = {path.name for path in (directory / "raw").glob("*")}
    if expected_raw != actual_raw:
        raise ValueError(
            f"Raw/record mismatch in {directory}: missing={sorted(expected_raw - actual_raw)}, "
            f"orphan={sorted(actual_raw - expected_raw)}"
        )
    for row, (identity, task) in zip(records, expected):
        raw_path = directory / "raw" / (digest(task.id.encode()) + ".json")
        data = raw_path.read_bytes()
        raw = decode(data)
        if dump(raw["request"]) != dump(body_for(task, model)):
            raise ValueError("Raw request differs from frozen task")
        parsed = None
        if raw["response_body"] is not None:
            try:
                parsed = decode(raw["response_body"])
            except ValueError:
                pass
        if raw["response"] != parsed:
            raise ValueError("Decoded response differs from original body")
        if (
            not isinstance(raw["latency_s"], (int, float))
            or not math.isfinite(raw["latency_s"])
            or raw["latency_s"] < 0
        ):
            raise ValueError("Invalid measured latency")
        recomputed = score_record(
            upstream, identity, task, raw, model, number, "warmup" if number == 0 else "measured", digest(data)
        )
        if row != recomputed:
            raise ValueError("Record differs from raw evidence or planned identity/order")
    return records


def aggregate(args):
    upstream, items, dataset = load_reference()
    run = outside_repo(args.run)
    meta = decode((run / "run.meta.json").read_text(encoding="utf-8"))
    validate_environment(meta)
    environment = (run / "environment.json").read_bytes()
    if digest(environment) != meta["env_sha256"]:
        raise ValueError("Environment source hash mismatch")
    env = decode(environment)
    if any(key not in meta or meta[key] != value for key, value in env.items()):
        raise ValueError("Run metadata differs from supplied environment")
    plan = meta["plan"]
    if (
        any(meta["dataset"][key] != dataset[key] for key in ("jevbench_pin", "hashes", "dataset_hash"))
        or plan["identities"] != [identity for identity, _ in items]
        or plan["tasks"] != 231
    ):
        raise ValueError("Frozen plan/reference mismatch")
    if plan["rounds"] < 1 or not 0 <= plan["warmup"] <= 231 or not 1 <= plan["limit"] <= 231:
        raise ValueError("Invalid frozen run plan")
    expected_dirs = {f"round-{i}" for i in range(1, plan["rounds"] + 1)}
    if {p.name for p in run.glob("round-*")} - expected_dirs:
        raise ValueError("Unknown round directory")
    warmup = read_round(run / "warmup", items[: plan["warmup"]], upstream, meta["model_configured"], 0)
    rounds, pooled_tasks, pooled_records = [], [], []
    for number in range(1, plan["rounds"] + 1):
        records = read_round(
            run / f"round-{number}", items[: plan["limit"]], upstream, meta["model_configured"], number
        )
        tasks = [task for _, task in items]
        rounds.append(dict(round=number, **summarize(upstream, tasks, records)))
        # Give repeated tasks distinct IDs only for upstream duplicate checking.
        for task in tasks:
            clone = copy.copy(task)
            clone.id = f"{number}:{task.id}"
            clone.group = f"{number}:{task.group}" if task.group else None
            pooled_tasks.append(clone)
        pooled_records.extend(dict(row, task_id=f"{number}:{row['task_id']}") for row in records)
    sufficient = (
        plan["rounds"] == 3
        and plan["warmup"] == 5
        and plan["limit"] == 231
        and len(warmup) == 5
        and all(r["attempted"] == 231 for r in rounds)
    )
    result = dict(
        sufficient=sufficient,
        status="sufficient" if sufficient else "insufficient",
        warmup_attempted=len(warmup),
        rounds=rounds,
        pooled=summarize(upstream, pooled_tasks, pooled_records),
        distinct_tasks=231,
        cost_usd=None,
        cost_basis="unmetered is not free",
        latency_success_basis="HTTP success with a typed decision structure; schema validity is separate",
    )
    output = outside_repo(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("predict")
    collector.add_argument("--endpoint", required=True)
    collector.add_argument("--path", required=True, choices=("/v1/systemone", "/v1/decisions"))
    collector.add_argument("--model")
    collector.add_argument("--thinking", choices=("off",), default="off")
    collector.add_argument("--warmup", type=int, default=5)
    collector.add_argument("--rounds", type=int, default=3)
    collector.add_argument("--limit", type=int, default=231)
    collector.add_argument("--timeout", type=float, default=120)
    collector.add_argument("--run-meta", type=Path, required=True)
    collector.add_argument("--out", type=Path, required=True)
    merger = commands.add_parser("aggregate")
    merger.add_argument("--run", type=Path, required=True)
    merger.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = predict(args) if args.command == "predict" else aggregate(args)
    print(result if isinstance(result, Path) else dump(result))


if __name__ == "__main__":
    main()
