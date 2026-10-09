"""Offline replay contract checks; no model inference or external HTTP traffic."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from benchmarks.client import DecisionClient
from benchmarks.evidence import digest, dump
from benchmarks.legs import changed_image_replay as replay


PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6nH0AAAAASUVORK5CYII="
)
JPEG = "data:image/jpeg;base64,/9j/2Q=="


def item(identity="frame-1", kind="choice", image=PNG):
    request = {"kind": kind, "state": ["Inspect this frame.", {"image": image}], "question": "Game over?"}
    if kind == "choice":
        request["options"] = ["No", "Yes"]
    return {"id": identity, "group": "screenshots", "request": request}


def answer(request):
    kind = request["kind"]
    options = request.get("options", ["false", "true"] if kind == "noul" else list("012345"))
    return {
        "kind": kind,
        "effective_kind": kind,
        "options": options,
        "probabilities": [1.0] + [0.0] * (len(options) - 1),
        "choice_index": 0,
        "choice": options[0],
        "model": "CPU fixture, no inference",
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    calls = []
    rows = [item()]
    manifest = tmp_path / "input.jsonl"
    environment = tmp_path / "actual-environment.json"
    environment.write_text(
        dump({"checkpoint": {"repo": "fixture", "revision": "test"}, "backend": {"name": "mock", "version": "1"}})
    )
    args = SimpleNamespace(
        manifest=manifest,
        run_meta=environment,
        endpoint="http://fixture.invalid",
        path="/v1/decide",
        timeout=1,
        cache_state="unknown",
        out=tmp_path / "output",
    )

    def configure(new_rows=None, handler=None):
        if new_rows is not None:
            rows[:] = new_rows
        manifest.write_text("".join(dump(row) + "\n" for row in rows), encoding="utf-8")

        def respond(request):
            calls.append(request)
            return handler(request) if handler else httpx.Response(200, json=answer(json.loads(request.content)))

        class FixtureClient(DecisionClient):
            def __init__(self, *positional, **kwargs):
                assert kwargs.get("model") is None
                super().__init__(*positional, **kwargs, transport=httpx.MockTransport(respond))

        monkeypatch.setattr(replay, "DecisionClient", FixtureClient)
        return args, calls

    return configure


def evidence(output):
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    raw = [json.loads((output / row["raw_file"]).read_text()) for row in rows]
    completion = json.loads((output / "completion.json").read_text())
    return rows, raw, completion


@pytest.mark.parametrize("path", ["/v1/decide", "/v1/systemone"])
def test_serial_three_kinds_preserve_repeated_images_options_and_evidence(setup, path):
    inputs = [item("first"), item("second", "noul", JPEG), item("third", "score"), item("reversed"), item("again")]
    inputs[3]["request"]["options"].reverse()
    inputs[2]["request"]["state"][1] = {"type": "image_url", "image_url": {"url": PNG}}
    before = copy.deepcopy(inputs)
    args, calls = setup(inputs)
    args.path, args.cache_state = path, "cold"
    output = replay.predict(args)
    records, raw, complete = evidence(output)
    assert inputs == before
    assert [json.loads(call.content) for call in calls] == [row["request"] for row in inputs]
    assert all(str(call.url) == args.endpoint + path for call in calls)
    assert [row["sequence"] for row in records] == [1, 2, 3, 4, 5]
    assert [row["id"] for row in records] == [row["id"] for row in inputs]
    assert [row["kind"] for row in records] == ["choice", "noul", "score", "choice", "choice"]
    assert all(row["ok"] and row["status"] == "ok" for row in records)
    assert all(row["latency_s"] > 0 for row in records)
    for row, original, response in zip(records, inputs, raw):
        assert row["request_sha256"] == digest(dump(original["request"]).encode())
        assert row["raw_sha256"] == digest((output / row["raw_file"]).read_bytes())
        assert row["response_body_sha256"] == digest(response["response_body"].encode())
        assert response["request"] == original["request"]
        assert json.loads(response["response_body"]) == response["response"]
    assert records[0]["image_uri_sha256"] == records[2]["image_uri_sha256"] == records[3]["image_uri_sha256"]
    assert (output / "manifest.jsonl").read_bytes() == args.manifest.read_bytes()
    assert (output / "environment.json").read_bytes() == args.run_meta.read_bytes()
    meta = json.loads((output / "run.meta.json").read_text())
    assert meta["cache_state_declared"] == "cold"
    assert meta["manifest_sha256"] == digest(args.manifest.read_bytes())
    assert meta["environment_sha256"] == digest(args.run_meta.read_bytes())
    assert complete["planned"] == complete["attempted"] == complete["succeeded"] == 5
    assert complete["remaining"] == complete["failed"] == 0 and complete["stop_reason"] is None


def test_multiple_images_retain_uri_hash_order(setup):
    row = item()
    row["request"]["state"].extend([{"image": JPEG}, {"image": PNG}])
    args, calls = setup([row])
    records, _, _ = evidence(replay.predict(args))
    assert json.loads(calls[0].content) == row["request"]
    assert records[0]["image_uri_sha256"] == [digest(uri.encode()) for uri in (PNG, JPEG, PNG)]


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"id": "later", "group": "x", "request": []},
        item("frame-1"),
        item("later", "other"),
        item("later", image="https://example.invalid/image.png"),
        item("later", image="data:image/png;base64,not-base64"),
        item("later", image="data:image/png;base64,dGV4dA=="),
        item("later", image="data:image/gif;base64,R0lGODlh"),
    ],
)
def test_bad_later_input_prevents_every_request(setup, broken):
    args, calls = setup([item(), broken])
    with pytest.raises(ValueError):
        replay.predict(args)
    assert not calls and not args.out.exists()


@pytest.mark.parametrize("change", ["options", "state", "question", "noul_options"])
def test_bad_request_fields_preflight(setup, change):
    row = item()
    if change == "options":
        row["request"]["options"] = ["Only one"]
    elif change == "state":
        row["request"]["state"] = ["No screenshot"]
    elif change == "question":
        row["request"]["question"] = 3
    else:
        row["request"]["kind"] = "noul"
    args, calls = setup([row])
    with pytest.raises(ValueError):
        replay.predict(args)
    assert not calls and not args.out.exists()


@pytest.mark.parametrize("source", ["", "not json\n", "{}\n\n"])
def test_empty_or_invalid_manifest_preflight(setup, source):
    args, calls = setup()
    args.manifest.write_text(source)
    with pytest.raises(ValueError):
        replay.predict(args)
    assert not calls and not args.out.exists()


def test_invalid_unicode_in_later_request_is_rejected_before_http(setup):
    args, calls = setup()
    row = item("later")
    row["request"]["question"] = "\ud800"
    args.manifest.write_text(json.dumps(item()) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        replay.predict(args)
    assert not calls and not args.out.exists()


@pytest.mark.parametrize("status,attempted", [(400, 2), (422, 2), (401, 1), (403, 1), (429, 1), (500, 1)])
def test_http_rejection_and_access_errors_keep_raw_text(setup, status, attempted):
    text = ' {"error":{"message":"rejected"}}\n'
    args, calls = setup([item(), item("next")], lambda _: httpx.Response(status, text=text))
    output = replay.predict(args)
    records, raw, complete = evidence(output)
    assert len(calls) == complete["attempted"] == complete["failed"] == attempted
    assert complete["remaining"] == 2 - attempted
    assert bool(complete["stop_reason"]) is (attempted == 1)
    assert all(r["response_body"] == text and r["http_status"] == status for r in raw)
    assert all(not r["ok"] and r["error"] for r in records)


@pytest.mark.parametrize("error", [httpx.ReadTimeout("fixture timeout"), KeyboardInterrupt()])
def test_transport_failure_and_interruption_save_completion(setup, error):
    def handler(_):
        raise error

    args, calls = setup([item(), item("next")], handler)
    records, raw, complete = evidence(replay.predict(args))
    assert len(calls) == complete["attempted"] == 1 and complete["remaining"] == 1
    assert raw[0]["client_error"] and raw[0]["response_body"] is None
    assert records[0]["status"] == "client_error" and records[0]["latency_s"] > 0
    assert complete["stop_reason"]


@pytest.mark.parametrize(
    "patch",
    [
        {"kind": "noul"},
        {"effective_kind": "noul"},
        {"options": ["Yes", "No"]},
        {"probabilities": [1]},
        {"probabilities": [True, False]},
        {"probabilities": [-1, 2]},
        {"choice_index": True},
        {"choice_index": 2},
        {"choice": "missing"},
    ],
)
def test_malformed_response_structure_is_retained(setup, patch):
    body = answer(item()["request"]) | patch
    args, _ = setup(handler=lambda _: httpx.Response(200, json=body))
    records, raw, complete = evidence(replay.predict(args))
    assert records[0]["status"] == "parse_error" and not records[0]["ok"]
    assert raw[0]["response"] == body and complete["failed"] == 1


@pytest.mark.parametrize("text", ["broken json", '{"probabilities":[NaN]}', "[]"])
def test_unparseable_response_keeps_original_text(setup, text):
    args, _ = setup(handler=lambda _: httpx.Response(200, text=text))
    records, raw, _ = evidence(replay.predict(args))
    assert records[0]["status"] == "parse_error" and raw[0]["response_body"] == text


@pytest.mark.parametrize("status", [200, 400, 422])
def test_escaped_surrogate_response_is_retained_and_sequence_continues(setup, status):
    text = '{"error":"\\ud800"}'

    def handler(request):
        if len(calls) == 1:
            return httpx.Response(status, text=text)
        return httpx.Response(200, json=answer(json.loads(request.content)))

    args, calls = setup([item(), item("next")], handler)
    records, raw, complete = evidence(replay.predict(args))
    assert len(calls) == complete["attempted"] == 2 and complete["remaining"] == 0
    assert complete["failed"] == complete["succeeded"] == 1 and complete["stop_reason"] is None
    assert records[0]["status"] == ("parse_error" if status == 200 else "refused")
    assert records[0]["latency_s"] > 0 and not records[0]["ok"] and records[1]["ok"]
    assert raw[0]["http_status"] == status and raw[0]["response_body"] == text
    assert raw[0]["response"] == {"error": "\ud800"}
    assert records[0]["response_body_sha256"] == digest(text.encode())
    assert records[0]["raw_sha256"] == digest((args.out / records[0]["raw_file"]).read_bytes())


def test_structure_check_does_not_compare_reference_values(setup):
    body = answer(item()["request"]) | {"probabilities": [0.2, 0.2]}
    args, _ = setup(handler=lambda _: httpx.Response(200, json=body))
    records, raw, _ = evidence(replay.predict(args))
    assert records[0]["ok"] and raw[0]["response"] == body


def test_output_never_overwrites_existing_evidence(setup):
    args, calls = setup()
    replay.predict(args)
    before = (args.out / "completion.json").read_bytes()
    with pytest.raises(FileExistsError):
        replay.predict(args)
    assert len(calls) == 1 and (args.out / "completion.json").read_bytes() == before


def test_output_must_be_outside_repository(setup):
    args, calls = setup()
    args.out = replay.Path(replay.__file__).resolve().parents[2] / "rejected-evidence"
    with pytest.raises(ValueError, match="outside"):
        replay.predict(args)
    assert not calls and not args.out.exists()
