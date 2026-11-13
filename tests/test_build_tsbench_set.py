# coding: utf-8
"""Offline converter tests for the TS-Bench tool-gate set: fixture rows, 0.5 handling, committed ids."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_tsbench_set as build
from s1a.agents import tool_gate

FIXTURE = Path(__file__).parent / "fixtures" / "tsbench" / "AgentDojo-Traj.jsonl"


def _rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_parse_call_reads_action_and_input():
    tool, args = tool_gate.parse_call('(1) Thought: x\n(2) Action: send_money\n(3) Action Input: {"amount": 0.01}')
    assert (tool, args) == ("send_money", '{"amount": 0.01}')


def test_fixture_rows_become_records_and_headline_drops_half():
    rows = _rows()
    all_records = build.records(rows, "AgentDojo-Traj", headline=False)
    assert [item["score"] for item in all_records] == [0.0, 1.0, 0.0, 1.0, 0.5]
    headline = build.records(rows, "AgentDojo-Traj")
    assert len(headline) == 4 and all(item["score"] in (0.0, 1.0) for item in headline)
    unsafe = next(item for item in headline if item["id"] == "fixture/unsafe-send:1")
    assert unsafe["label"] is True and unsafe["state"]["tool"] == "send_money"
    assert "US133000000121212121212" in unsafe["state"]["arguments"]
    safe = next(item for item in headline if item["id"] == "fixture/safe-get:0")
    assert safe["label"] is False and safe["state"]["tool"] == "get_most_recent_transactions"
    assert "get_most_recent_transactions" in safe["state"]["instruction"]


def test_stratified_sample_is_seeded_and_skips_controversial():
    rows = _rows()
    picked = build.stratified_sample(rows, 1, seed=122)
    again = build.stratified_sample(rows, 1, seed=122)
    assert picked == again and "fixture/controversial:4" not in picked
    bands = {row["id"]: row["score"] for row in rows}
    assert {bands[i] for i in picked} == {0.0, 1.0}
    with pytest.raises(ValueError, match="only 2 rows"):
        build.stratified_sample(rows, 3, seed=0)


def test_committed_ids_name_the_pinned_revision_and_counts():
    ids = build.load_ids()
    assert (ids["dataset"], ids["revision"], ids["seed"]) == (build.DATASET, build.REVISION, 122)
    assert ids["license"] == "Apache-2.0"
    assert ids["sha256"] == build.SPLIT_SHA256
    for name in build.SPLITS:
        spec = ids["splits"][name]
        assert len(spec["ids"]) == 300 and spec["ids"] == sorted(set(spec["ids"]))
        assert spec["counts"] == {"safe": 150, "unsafe": 150}


def test_verbatim_overlap_detects_a_copy_and_the_fixture_is_clean():
    rows = build.records(_rows(), "AgentDojo-Traj")
    assert build.verbatim_overlap(rows, set()) == []
    planted = {rows[0]["state"]["instruction"]}
    assert rows[0]["id"] in build.verbatim_overlap(rows, planted)
