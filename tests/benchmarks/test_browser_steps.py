"""myjev-browser-decisions step benchmark on a synthetic fixture: no dataset rows, no network."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from benchmarks.legs import browser_steps as bs
from s1a.decision_models import JevModel
from s1a.decision_models.fakes import ScriptedModel, ScriptedTransport

RULES = "Advance the goal with one operation."
TARGET_RULES = [RULES, "Choose only an offered element index."]


def element_row(key, label, role="button", value=""):
    return {"element": f"[{key}] {label}", "current_value": value, "role": role}


def record(step_id, gold, *, ops=("CLICK", "TYPE_TEXT", "SCROLL_DOWN", "WAIT", "DONE", "BLOCKED")):
    """One synthetic step in the dataset's jev-ultrafast shape (invented content, not a dataset row)."""
    goal = "Search for blue widgets"
    questions = {
        "operation": {
            "type": "choice",
            "criteria": {op: f"upstream text for {op}" for op in ops},
            "instructions": {"goal": goal, "rules": RULES},
        },
        "click_target": {
            "type": "choice",
            "criteria": {"1": element_row("1", "Home", "link"), "2": element_row("2", "Search")},
            "instructions": {"goal": goal, "operation": "CLICK", "rules": TARGET_RULES},
        },
        "type_text_target": {
            "type": "choice",
            "criteria": {"3": element_row("3", "Query", "textbox")},
            "instructions": {"goal": goal, "operation": "TYPE_TEXT", "rules": TARGET_RULES},
        },
    }
    state = {
        "page": {"url": "https://shop.invalid/", "title": "Shop", "text": "Home Search"},
        "elements": [
            {"index": "1", "role": "link", "label": "Home", "value": "", "operations": ["CLICK"]},
            {"index": "2", "role": "button", "label": "Search", "value": "", "operations": ["CLICK"]},
            {"index": "3", "role": "textbox", "label": "Query", "value": "", "operations": ["TYPE_TEXT", "CLICK"]},
        ],
        "recent_actions": [],
    }
    return {
        "request": {"model": "jev-latest", "state": state, "questions": questions},
        "gold": gold,
        "meta": {"id": step_id, "source": "synthetic"},
    }


RECORDS = [
    record("s:1", {"operation": "CLICK", "click_target": ["2"]}),
    record("s:2", {"operation": "TYPE_TEXT", "type_text_target": ["3"]}),
    record("s:3", {"operation": "SCROLL_DOWN"}),
    record("s:4", {"operation": "CLICK", "click_target": ["1", "2"]}),
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def answer(choice, keys, p=0.9):
    rest = (1 - p) / (len(keys) - 1) if len(keys) > 1 else 0.0
    return {"choice": choice, "probabilities": {k: (p if k == choice else rest) for k in keys}}


OPS = ["CLICK", "TYPE_TEXT", "SCROLL_DOWN", "WAIT", "DONE", "BLOCKED"]


def reply(op, click="2", text="3", p=0.9):
    return {
        "operation": answer(op, OPS, p),
        "click_target": answer(click, ["1", "2"]),
        "type_text_target": answer(text, ["3"], 1.0),
    }


def test_dataset_shape_resends_the_dataset_body_unchanged():
    transport = ScriptedTransport()
    model = JevModel(transport)
    for item in RECORDS:
        observation, questions = bs.to_questions(item, "dataset")
        asyncio.run(model.decide_many(observation, questions))
        sent = transport.bodies[-1]
        assert sent["state"] == item["request"]["state"]
        assert json.dumps(sent["questions"]) == json.dumps(item["request"]["questions"])


def test_ours_shape_swaps_only_the_text_our_agent_sends():
    from s1a.browser import prompts
    from s1a.browser.action_space import _OPERATION_LABELS

    item = RECORDS[0]
    observation, questions = bs.to_questions(item, "ours")
    assert observation.state == item["request"]["state"]
    assert list(questions) == list(item["request"]["questions"])
    operation = questions["operation"]
    assert list(operation.options) == OPS
    assert operation.options["CLICK"] == _OPERATION_LABELS["CLICK"]
    assert operation.rules == (prompts.OPERATION_RULES["en"],)
    click = questions["click_target"]
    assert dict(click.options) == item["request"]["questions"]["click_target"]["criteria"]
    assert click.rules == (prompts.OPERATION_RULES["en"], prompts.TARGET_RULES["en"])
    assert click.operation == "CLICK" and click.goal == "Search for blue widgets"


def test_gold_is_never_sent(tmp_path):
    transport = ScriptedTransport()
    asyncio.run(bs.predict_steps(JevModel(transport), RECORDS, tmp_path / "r.jsonl"))
    assert len(transport.bodies) == len(RECORDS)
    assert all("gold" not in json.dumps(body) and "meta" not in body for body in transport.bodies)


def run_and_judge(tmp_path, model, records=RECORDS, limit=None):
    data = write_jsonl(tmp_path / "data.jsonl", records)
    run = tmp_path / "run"
    run.mkdir()
    counts = asyncio.run(bs.predict_steps(model, records, run / "results.jsonl", limit=limit))
    return counts, bs.judge_run(run, data), run, data


def test_scores_with_denominators_and_failures_count_as_wrong(tmp_path):
    from openjiuwen.core.common.exception.codes import StatusCode
    from openjiuwen.core.common.exception.errors import build_error

    answers = [
        reply("CLICK", click="2"),  # step right
        reply("CLICK", click="2", p=0.6),  # wrong operation (gold TYPE_TEXT)
        {"operation": answer("SCROLL_DOWN", OPS)},  # incomplete reply: validation fails, counted wrong
        reply("CLICK", click="1"),  # one of two gold keys
    ]
    model = ScriptedModel(answers=answers)
    counts, judged, _, _ = run_and_judge(tmp_path, model)
    assert counts == {"attempted": 4, "failed": 1}
    summary = bs.summarize(judged)
    assert summary["steps"] == 4
    assert summary["step_success"] == {"value": 0.5, "n": 2, "of": 4}
    assert summary["operation_accuracy"] == {"value": 0.5, "n": 2, "of": 4}
    assert summary["element_accuracy"] == {"value": 1.0, "n": 3, "of": 3}  # step 2's type target is still right
    assert summary["invalid_or_failed"] == {"value": 0.25, "n": 1, "of": 4}
    assert summary["calibration_steps"] == 3
    assert summary["operation_confidence_brier"] == pytest.approx(((0.9 - 1) ** 2 * 2 + 0.6**2) / 3)
    assert summary["decision_ms_p50"] == 9
    assert summary["by_gold_operation"]["SCROLL_DOWN"] == {"value": 0.0, "n": 0, "of": 1}
    assert build_error(StatusCode.MODEL_CALL_FAILED, error_msg="x")  # the error type the layer raises exists


def test_transport_error_is_recorded_and_scored_wrong(tmp_path):
    model = JevModel(ScriptedTransport(error=httpx.ConnectError("refused")))
    counts, judged, run, _ = run_and_judge(tmp_path, model)
    assert counts["failed"] == 4
    steps = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    assert all("ConnectError" in step["error"] and "request" in step for step in steps)
    summary = bs.summarize(judged)
    assert summary["step_success"]["value"] == 0.0 and summary["invalid_or_failed"]["value"] == 1.0
    assert summary["operation_confidence_brier"] is None and summary["decision_ms_p50"] is None


def test_unattempted_steps_stay_in_the_denominator(tmp_path):
    model = ScriptedModel(answers=[reply("CLICK")])
    _, judged, _, _ = run_and_judge(tmp_path, model, limit=1)
    assert bs.summarize(judged)["step_success"] == {"value": 0.25, "n": 1, "of": 4}
    assert bs.summarize(judged[:1])["step_success"] == {"value": 1.0, "n": 1, "of": 1}


def test_orphan_and_duplicate_steps_are_errors(tmp_path):
    data = write_jsonl(tmp_path / "data.jsonl", RECORDS)
    run = tmp_path / "run"
    run.mkdir()
    write_jsonl(run / "results.jsonl", [{"id": "elsewhere:1"}])
    with pytest.raises(ValueError, match="not in"):
        bs.judge_run(run, data)
    write_jsonl(run / "results.jsonl", [{"id": "s:1"}, {"id": "s:1"}])
    with pytest.raises(ValueError, match="twice"):
        bs.judge_run(run, data)


def test_ece_and_multiclass_brier():
    rows = [{"confidence": 0.9, "operation_ok": True}, {"confidence": 0.9, "operation_ok": False}]
    assert bs.ece(rows) == pytest.approx(0.4)
    assert bs.ece([]) is None
    assert bs.multiclass_brier({"A": 0.7, "B": 0.3}, "A") == pytest.approx(0.18)
    assert bs.multiclass_brier({"A": 1.0}, "B") == pytest.approx(2.0)


def test_threshold_table_defers_failures(tmp_path):
    answers = [reply("CLICK", p=0.95), reply("CLICK", p=0.5), reply("SCROLL_DOWN", p=0.7), {}]
    _, judged, _, _ = run_and_judge(tmp_path, ScriptedModel(answers=answers))
    table = {row["threshold"]: row for row in bs.threshold_table(judged, (0.0, 0.6, 0.9))}
    assert table[0.0]["kept"] == {"value": 0.75, "n": 3, "of": 4}
    assert table[0.6]["kept"]["n"] == 2 and table[0.6]["kept_step_success"] == {"value": 1.0, "n": 2, "of": 2}
    assert table[0.9]["deferred"] == {"value": 0.75, "n": 3, "of": 4}


def test_verify_checks_pinned_digests(tmp_path, monkeypatch):
    path = tmp_path / "calibration.jsonl"
    path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="is not"):
        bs.verify(path, "calibration")
    monkeypatch.setitem(
        bs.SPLITS, "calibration", ("data/calibration.jsonl", "sha256", bs.file_digest(b"{}\n", "sha256"))
    )
    assert bs.verify(path, "calibration")
    # git blob ids match `git hash-object`
    assert bs.file_digest(b"hello\n", "git-blob") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_fetch_keeps_only_verified_files(tmp_path, monkeypatch):
    body = b'{"x": 1}\n'
    monkeypatch.setitem(bs.SPLITS, "calibration", ("data/calibration.jsonl", "sha256", bs.file_digest(body, "sha256")))
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=body)

    saved = bs.fetch(["calibration"], tmp_path, transport=httpx.MockTransport(handler))
    assert saved["calibration"].read_bytes() == body
    assert seen == [f"https://huggingface.co/datasets/{bs.DATASET}/resolve/{bs.REVISION}/data/calibration.jsonl"]

    bad = tmp_path / "bad"
    with pytest.raises(ValueError):
        bs.fetch(["calibration"], bad, transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"no")))
    assert not any(bad.iterdir())


def test_cli_predict_score_thresholds_end_to_end(tmp_path, monkeypatch, capsys):
    data = write_jsonl(tmp_path / "data.jsonl", RECORDS)
    monkeypatch.setattr("s1a.decision_models.build_model", lambda name: ScriptedModel(answers=[reply("CLICK")] * 4))
    out = tmp_path / "run"
    bs.main(["predict", "--data", str(data), "--split", "synthetic", "--model", "random", "--out", str(out)])
    meta = json.loads((out / "run.meta.json").read_text())
    assert meta["pinned"] is False and meta["shape"] == "ours" and meta["revision"] == bs.REVISION
    assert json.loads((out / "completion.json").read_text())["attempted"] == 4
    capsys.readouterr()
    ids = tmp_path / "ids.txt"
    ids.write_text("s:1\ns:3\n")
    bs.main(["score", "--run", str(out), "--data", str(data), "--ids", str(ids)])
    report = json.loads(capsys.readouterr().out)
    assert report["full"]["step_success"] == {"value": 0.5, "n": 2, "of": 4}
    assert report["jev_answered"]["steps"] == 2
    bs.main(["thresholds", "--run", str(out), "--data", str(data)])
    assert len(json.loads(capsys.readouterr().out)["table"]) == len(bs.THRESHOLDS)


def test_predict_refuses_a_pinned_split_name_with_the_wrong_file(tmp_path):
    data = write_jsonl(tmp_path / "data.jsonl", RECORDS)
    with pytest.raises(ValueError, match="is not"):
        bs.main(
            ["predict", "--data", str(data), "--split", "benchmark", "--model", "random", "--out", str(tmp_path / "o")]
        )


def test_evidence_must_live_outside_the_repository(tmp_path):
    data = write_jsonl(tmp_path / "data.jsonl", RECORDS)
    inside = Path(bs.__file__).resolve().parents[1] / "should-not-exist"
    with pytest.raises(ValueError, match="outside"):
        bs.main(
            ["predict", "--data", str(data), "--split", "x", "--unpinned", "--model", "random", "--out", str(inside)]
        )
