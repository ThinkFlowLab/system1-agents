"""CPU protocol fixtures only; no model service or network connection."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from benchmarks.client import DecisionClient
from benchmarks.evidence import body_for, digest, dump, load_reference, score_record, summarize
from benchmarks.legs import leg1_public231 as cli


@pytest.fixture
def reference():
    if not os.environ.get("JEVBENCH_ROOT"):
        pytest.skip("Set JEVBENCH_ROOT to the pinned external reference")
    return load_reference()


def test_multifield_single_post_preserves_input():
    body = {
        "state": [{"context": "structured"}],
        "questions": {
            "second": {"type": "choice", "criteria": {"z": "last", "a": "first"}},
            "first": {"type": "noul", "instructions": "accept?"},
        },
        "thinking": {"enabled": False},
    }
    before, calls = copy.deepcopy(body), []

    def handler(request):
        calls.append(json.loads(request.content))
        assert str(request.url) == "http://fixture.invalid/v1/decisions"
        return httpx.Response(200, text=' {"answers": {"second": {}, "first": {}}, "model":"fixture"}\n')

    client = DecisionClient("http://fixture.invalid", path="/v1/decisions", transport=httpx.MockTransport(handler))
    raw = client.request(body)
    client.close()
    assert body == before and calls == [body]
    assert list(calls[0]["questions"]) == ["second", "first"]
    assert list(calls[0]["questions"]["second"]["criteria"]) == ["z", "a"]
    assert raw["response_body"].startswith(" ") and raw["response_body"].endswith("\n")
    assert set(raw["response"]["answers"]) == {"second", "first"}
    assert raw["http_status"] == 200 and raw["latency_s"] > 0


@pytest.mark.parametrize("status,text", [(422, "too long"), (500, "<html>failure</html>"), (200, "broken json")])
def test_raw_bad_responses_survive(status, text):
    client = DecisionClient(
        "http://fixture.invalid",
        path="/v1/systemone",
        transport=httpx.MockTransport(lambda _: httpx.Response(status, text=text)),
    )
    raw = client.request({"questions": {}})
    client.close()
    assert (raw["http_status"], raw["response_body"], raw["response"]) == (status, text, None)
    assert raw["latency_s"] > 0


@pytest.mark.parametrize("error", [httpx.ReadTimeout("fixture timeout"), KeyboardInterrupt()])
def test_network_error_has_real_elapsed(error):
    def handler(_):
        raise error

    client = DecisionClient("http://fixture.invalid", path="/v1/systemone", transport=httpx.MockTransport(handler))
    raw = client.request({"state": "fixture"})
    client.close()
    assert raw["http_status"] is None and raw["response_body"] is None
    assert raw["client_error"] and raw["latency_s"] > 0


@pytest.mark.parametrize("url", ["http://user:secret@fixture.invalid", "http://fixture.invalid?key=x"])
def test_credentials_cannot_enter_endpoint_evidence(url):
    with pytest.raises(ValueError):
        DecisionClient(url, path="/v1/systemone")


def raw_answer(answer, *, status=200, seconds=1):
    response = {"answers": {"decision": answer}, "model": "fixture", "usage": {"decision_count": 1}}
    return dict(
        request={},
        response=response,
        response_body=dump(response),
        http_status=status,
        client_error=None,
        latency_s=seconds,
        ts=123.0,
    )


@pytest.mark.parametrize(
    "probs,valid,renormalized",
    [
        ({"no": 0.4, "yes": 0.595}, True, True),
        ({"no": 0.4, "yes": 0.5}, False, False),
        ({"no": 0.4}, False, False),
        ({"no": 0.4, "yes": 0.6, "extra": 0}, False, False),
        ({"no": -0.1, "yes": 1.1}, False, False),
    ],
)
def test_upstream_alone_decides_probability_band(reference, probs, valid, renormalized):
    upstream, items, _ = reference
    identity, task = next(item for item in items if item[1].question["type"] == "noul")
    row = score_record(
        upstream,
        identity,
        task,
        raw_answer({"type": "noul", "probabilities": probs}),
        "configured",
        1,
        "measured",
        "fixture-hash",
    )
    assert row["ok"] and row["valid"] is valid and row["renormalized"] is renormalized
    assert row["probs_as_returned"] == probs
    if renormalized:
        assert row["strict_valid"] is False and sum(row["probs"].values()) == pytest.approx(1)
    if not valid:
        assert row["correct"] is False and row["probs"] is None


def test_native_scalar_and_missing_probabilities(reference):
    upstream, items, _ = reference
    for kind in ("noul", "choice", "score"):
        identity, task = next(item for item in items if item[1].question["type"] == kind)
        answer = {"type": kind, "noul": 0.6} if kind == "noul" else {"type": kind, kind: task.expected}
        row = score_record(upstream, identity, task, raw_answer(answer), None, 1, "measured", "h")
        assert row["valid"] is (kind == "noul")
        assert row["model_configured"] is None and row["model"] == "fixture"
        if kind == "noul":
            assert row["probs"] == {"no": 0.4, "yes": 0.6}
        else:
            assert row["probs"] is None and row["probs_as_returned"] is None


@pytest.mark.parametrize("answers", [None, [], "bad", {"decision": None}, {"decision": {"type": "wrong"}}])
def test_malformed_typed_structure_is_recorded(reference, answers):
    upstream, items, _ = reference
    identity, task = items[0]
    raw = raw_answer({})
    raw["response"]["answers"] = answers
    row = score_record(upstream, identity, task, raw, None, 1, "measured", "h")
    assert row["status"] == "parse_error" and not row["ok"] and not row["valid"]


def test_nonfinite_response_is_not_json():
    client = DecisionClient(
        "http://fixture.invalid",
        path="/v1/systemone",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, text='{"answers":{"decision":{"type":"noul","probabilities":{"no":NaN,"yes":1}}}}'
            )
        ),
    )
    raw = client.request({})
    client.close()
    assert raw["response"] is None and "NaN" in raw["response_body"]


@pytest.fixture
def collect(reference, monkeypatch, tmp_path):
    upstream, items, _ = reference
    env = tmp_path / "actual-environment.json"
    env.write_text(
        dump(
            dict(
                checkpoint={"repo": "CPU-fixture", "revision": "fixture-only"},
                backend={"name": "httpx.MockTransport", "version": httpx.__version__},
                sampling={"temperature": 1.0},
                hardware="CPU protocol fixture, no inference",
                dtype="fixture-original",
                tokenizer={"revision": "fixture-tokenizer"},
                base={"revision": "fixture-base"},
                adapter={"revision": "fixture-adapter"},
                head={"revision": "fixture-head"},
                context_limit=16384,
                cache={"enabled": False},
                job_id="CPU-fixture",
            )
        )
    )
    lookup = {dump(body_for(task, "configured")): task for _, task in items}

    def run(*, warmup=5, rounds=3, limit=231, statuses=None, response_text=None):
        calls = []

        def handler(request):
            body = json.loads(request.content)
            assert "targets" not in body and "provenance" not in body and "expected" not in body
            assert body["thinking"] == {"enabled": False}
            task = lookup[dump(body)]
            calls.append(body)
            status = statuses[min(len(calls) - 1, len(statuses) - 1)] if statuses else 200
            if response_text is not None:
                return httpx.Response(status, text=response_text)
            answer = {
                "type": task.question["type"],
                "probabilities": {label: float(label == str(task.expected)) for label in task.labels},
            }
            return httpx.Response(status, json={"answers": {"decision": answer}, "model": "fixture-response"})

        class FixtureClient(DecisionClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, transport=httpx.MockTransport(handler))

            def request(self, body):
                raw = super().request(body)
                raw["latency_s"] = 999 if len(calls) <= warmup else (1 if raw["http_status"] == 200 else 120)
                return raw

        monkeypatch.setattr(cli, "DecisionClient", FixtureClient)
        output = tmp_path / f"run-{len(list(tmp_path.glob('run-*')))}"
        args = SimpleNamespace(
            endpoint="http://fixture.invalid",
            path="/v1/decisions",
            model="configured",
            timeout=10,
            run_meta=env,
            out=output,
            warmup=warmup,
            rounds=rounds,
            limit=limit,
        )
        cli.predict(args)
        return output, calls, args

    return run


def aggregate(run, out=None):
    return cli.aggregate(SimpleNamespace(run=run, out=out or run))


@pytest.mark.parametrize(
    "endpoint", ["http://user:DUMMY_SECRET@fixture.invalid", "http://fixture.invalid?key=DUMMY_SECRET"]
)
def test_predict_cli_rejects_credentials_before_output(collect, monkeypatch, endpoint):
    run, calls, args = collect(warmup=0, rounds=1, limit=1)
    args.out = run.parent / "rejected-endpoint"
    monkeypatch.setattr(
        "sys.argv",
        [
            "public231",
            "predict",
            "--endpoint",
            endpoint,
            "--path",
            args.path,
            "--run-meta",
            str(args.run_meta),
            "--out",
            str(args.out),
        ],
    )
    with pytest.raises(ValueError, match="without credentials"):
        cli.main()
    assert not args.out.exists() and len(calls) == 1
    assert all("DUMMY_SECRET" not in path.read_text() for path in run.parent.rglob("*") if path.is_file())


@pytest.mark.parametrize("failure", ["mkdir", "environment.json", "run.meta.json"])
def test_predict_closes_client_when_output_setup_fails(collect, monkeypatch, failure):
    run, calls, args = collect(warmup=0, rounds=1, limit=1)
    args.out = run.parent / "failed-output"
    created, original_client, original_open = [], cli.DecisionClient, Path.open

    def client(*args, **kwargs):
        result = original_client(*args, **kwargs)
        created.append(result)
        return result

    def open_file(path, *options, **kwargs):
        if path == args.out / failure:
            raise OSError("fixture output failure")
        return original_open(path, *options, **kwargs)

    if failure == "mkdir":
        args.out.mkdir()
    monkeypatch.setattr(cli, "DecisionClient", client)
    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(OSError):
        cli.predict(args)
    assert len(created) == 1 and created[0].http.is_closed
    assert len(calls) == 1


@pytest.mark.parametrize("separator", ["\u2028", "\u0085", "\u2029"])
def test_collect_and_aggregate_preserve_unicode_in_error_body(collect, separator):
    run, calls, _ = collect(
        warmup=0, rounds=1, limit=1, statuses=[422], response_text=dump({"detail": f"bad{separator}input"})
    )
    with (run / "round-1/results.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    assert len(calls) == len(records) == 1 and records[0]["status"] == "refused"
    assert separator in records[0]["error"]
    result = aggregate(run)
    assert result["pooled"]["planned"] == 231 and result["pooled"]["attempted"] == 1
    assert result["pooled"]["correct"] == result["pooled"]["accuracy"] == 0
    assert not result["sufficient"]


@pytest.mark.parametrize("line", ["\n", "{broken}\n"])
def test_aggregate_rejects_blank_or_malformed_physical_line(collect, line):
    run, _, _ = collect(warmup=0, rounds=1, limit=1)
    (run / "round-1/results.jsonl").write_text(line)
    with pytest.raises(ValueError):
        aggregate(run)


def test_complete_5_plus_3_rounds_and_no_warmup_in_latency(collect):
    run, calls, _ = collect()
    result = aggregate(run)
    assert len(calls) == 698 and result["sufficient"]
    assert result["pooled"]["planned"] == result["pooled"]["correct"] == 693
    assert result["pooled"]["latency"]["all_attempts"] == {"n": 693, "p50_s": 1, "p95_s": 1}
    assert result["pooled"]["latency"]["failure"] == {"n": 0, "p50_s": None, "p95_s": None}
    assert all(round_["attempted"] == 231 for round_ in result["rounds"])
    assert len(list((run / "warmup/raw").glob("*.json"))) == 5
    second = aggregate(run, run / "offline-recompute")
    assert result == second
    with pytest.raises(FileExistsError):
        aggregate(run)


@pytest.mark.parametrize("options", [{"limit": 10}, {"warmup": 0}, {"rounds": 1}])
def test_debug_run_is_insufficient_with_fixed_denominator(collect, options):
    run, _, _ = collect(**options)
    result = aggregate(run)
    assert not result["sufficient"]
    if "limit" in options:
        assert result["pooled"]["accuracy"] == 30 / 693
        assert result["rounds"][0]["accuracy"] == 10 / 231


@pytest.mark.parametrize(
    "statuses,attempted", [([401], 1), ([403], 1), ([429], 1), ([500], 3), ([500, 500, 422, 500, 500, 500], 6)]
)
def test_stop_policy_leaves_unattempted_in_plan(collect, statuses, attempted):
    run, calls, _ = collect(warmup=0, statuses=statuses)
    result = aggregate(run)
    assert len(calls) == attempted
    assert result["pooled"]["accuracy"] == 0 and result["pooled"]["planned"] == 693
    assert result["pooled"]["attempted"] == attempted and not result["sufficient"]
    assert json.loads((run / "run.meta.json").read_text())["stop_reason"]


def test_failures_and_partial_denominator_with_two_latencies(collect):
    run, _, _ = collect(warmup=0, limit=2, rounds=1, statuses=[200, 500])
    result = aggregate(run)["rounds"][0]
    assert result["accuracy"] == 1 / 231
    assert result["latency"]["success"] == {"n": 1, "p50_s": 1, "p95_s": 1}
    assert result["latency"]["failure"] == {"n": 1, "p50_s": 120, "p95_s": 120}
    assert result["latency"]["all_attempts"]["p50_s"] == 60.5
    assert result["latency"]["all_attempts"]["p95_s"] == pytest.approx(114.05)


@pytest.mark.parametrize("change", ["duplicate", "unknown", "score", "raw", "environment", "order"])
def test_offline_tampering_rejected(collect, change):
    run, _, _ = collect(warmup=0, rounds=1, limit=2)
    path = run / "round-1/results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "unknown":
        rows[0]["task_id"] = "unknown"
    elif change == "score":
        rows[0]["correct"] = False
    elif change == "order":
        rows.reverse()
    elif change == "raw":
        raw = run / "round-1/raw" / (digest(rows[0]["task_id"].encode()) + ".json")
        raw.write_text(raw.read_text() + " ")
    else:
        path_env = run / "environment.json"
        path_env.write_text(path_env.read_text() + " ")
    path.write_text("".join(dump(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        aggregate(run)


def test_output_and_environment_boundaries(collect, tmp_path):
    run, _, args = collect(warmup=0, limit=1, rounds=1)
    with pytest.raises(FileExistsError):
        cli.predict(args)
    args.out = Path(__file__).resolve().parents[2] / "forbidden-evidence"
    with pytest.raises(ValueError, match="outside"):
        cli.predict(args)
    env = json.loads(args.run_meta.read_text())
    env["sampling"]["temperature"] = 0
    args.run_meta.write_text(dump(env))
    args.out = tmp_path / "invalid-temperature"
    with pytest.raises(ValueError, match="temperature"):
        cli.predict(args)
    assert not args.out.exists()


def test_reference_hash_tampering_rejected(reference, monkeypatch, tmp_path):
    _, _, dataset = reference
    from benchmarks.evidence import HASHES

    root = tmp_path / "reference"
    for name in HASHES:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((Path(dataset["root"]) / name).read_bytes())
    (root / "jevbench/scoring.py").write_text("raise RuntimeError('must not execute')\n")
    monkeypatch.setenv("JEVBENCH_ROOT", str(root))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_reference()


def test_empty_summary_has_null_latency(reference):
    upstream, items, _ = reference
    result = summarize(upstream, [task for _, task in items], [])
    assert result["accuracy"] == 0 and result["coverage"] == 0
    assert all(value == {"n": 0, "p50_s": None, "p95_s": None} for value in result["latency"].values())


@pytest.mark.parametrize("literal", ["1e309", "NaN", "Infinity"])
def test_nonfinite_wire_numbers_are_retained_as_parse_failures(literal):
    response = '{"answers":{"decision":{"type":"noul","noul":' + literal + "}}}"
    client = DecisionClient(
        "http://fixture.invalid",
        path="/v1/systemone",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=response)),
    )
    raw = client.request({})
    client.close()
    assert raw["response"] is None and raw["response_body"] == response


def test_scorer_numeric_overflow_counts_wrong(reference):
    upstream, items, _ = reference
    identity, task = next(item for item in items if item[1].question["type"] == "noul")
    probs = {"no": 10**400, "yes": 0}
    row = score_record(
        upstream, identity, task, raw_answer({"type": "noul", "probabilities": probs}), "configured", 1, "measured", "h"
    )
    assert row["ok"] and not row["valid"] and not row["correct"]
    assert row["probs_as_returned"] == probs


@pytest.mark.parametrize("directory,filename", [("warmup", "records.jsonl"), ("round-1", "results.jsonl")])
@pytest.mark.parametrize("change", ["missing_row", "missing_records_file", "missing_raw", "unknown_raw"])
def test_raw_and_records_must_correspond(collect, directory, filename, change):
    run, _, _ = collect(warmup=2, rounds=1, limit=2)
    directory = run / directory
    path = directory / filename
    rows = path.read_text().splitlines()
    row = json.loads(rows[-1])
    raw = directory / "raw" / (digest(row["task_id"].encode()) + ".json")
    changed = raw
    if change == "missing_row":
        path.write_text(rows[0] + "\n")
    elif change == "missing_records_file":
        path.unlink()
    elif change == "missing_raw":
        raw.unlink()
    else:
        changed = directory / "raw/unknown.json"
        changed.write_bytes(raw.read_bytes())
    with pytest.raises(ValueError, match="Raw/record mismatch") as error:
        aggregate(run)
    assert changed.name in str(error.value)
    assert not (run / "summary.json").exists()


@pytest.mark.parametrize("field", ["dtype", "tokenizer", "base", "adapter", "head", "context_limit", "cache", "job_id"])
@pytest.mark.parametrize("change", ["different", "missing"])
def test_all_environment_fields_must_match(collect, field, change):
    run, _, _ = collect(warmup=0, rounds=1, limit=1)
    path = run / "run.meta.json"
    meta = json.loads(path.read_text())
    if change == "different":
        meta[field] = "fixture-contradiction"
    else:
        del meta[field]
    path.write_text(dump(meta))
    with pytest.raises(ValueError, match="supplied environment"):
        aggregate(run)
    assert not (run / "summary.json").exists()


def test_conflicting_environment_controls_rejected_before_collection(collect):
    run, calls, args = collect(warmup=0, rounds=1, limit=1)
    env = json.loads(args.run_meta.read_text())
    env["endpoint"] = "http://conflicting.invalid"
    args.run_meta.write_text(dump(env))
    args.out = run.parent / "conflicting-run"
    with pytest.raises(ValueError, match="collector"):
        cli.predict(args)
    assert len(calls) == 1 and not args.out.exists()


def test_matching_environment_controls_are_compatible(collect):
    run, _, args = collect(warmup=0, rounds=1, limit=1)
    env = json.loads(args.run_meta.read_text())
    env.update(endpoint=args.endpoint, path=args.path, thinking=False)
    args.run_meta.write_text(dump(env))
    args.out = run.parent / "matching-run"
    cli.predict(args)
    assert aggregate(args.out)["pooled"]["attempted"] == 1


@pytest.mark.parametrize("stop_reason", [None, "stale stop reason"])
def test_environment_stop_reason_rejected_before_request(collect, stop_reason):
    run, calls, args = collect(warmup=0, rounds=1, limit=1, statuses=[401])
    assert aggregate(run)["pooled"]["attempted"] == 1
    env = json.loads(args.run_meta.read_text())
    env["stop_reason"] = stop_reason
    args.run_meta.write_text(dump(env))
    args.out = run.parent / "invalid-stop-reason"
    with pytest.raises(ValueError, match="stop_reason"):
        cli.predict(args)
    assert len(calls) == 1 and not args.out.exists()
