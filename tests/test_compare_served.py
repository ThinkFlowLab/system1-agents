# coding: utf-8
"""evals/ticket_router/compare_served.py: the table and the agreement over jobs, including one that stopped early."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.ticket_router import compare_served

TICKETS = [("t1", "payment"), ("t2", "returns"), ("t3", "account")]


def write_job(root: Path, name: str, predicted: list[str]) -> Path:
    """One seed-0 trial that decided the first ``len(predicted)`` of three tickets."""
    trial = root / name / "ticket_router--0__x"
    (trial / "agent").mkdir(parents=True)
    routes = [
        {"id": tid, "expected": expected, "predicted": key, "correct": key == expected}
        for (tid, expected), key in zip(TICKETS, predicted)
    ]
    ticks = [
        {"key": key, "probabilities": {key: 0.9, "human": 0.1}, "ms": 50 + i, "source": name}
        for i, key in enumerate(predicted)
    ]
    router = {
        "seed": 0,
        "total": 3,
        "correct": sum(r["correct"] for r in routes),
        "ticket_ids": [t for t, _ in TICKETS],
        "routes": routes,
    }
    episode = {"decisions": ticks, "extra": {"ticket_router": router}}
    (trial / "agent" / "episode.json").write_text(json.dumps(episode), encoding="utf-8")
    (trial / "result.json").write_text(json.dumps({"agent_result": {"metadata": {"elapsed_s": 1.0}}}), encoding="utf-8")
    return root / name


def test_a_job_that_stopped_early_is_compared_on_what_it_decided(tmp_path: Path, capsys) -> None:
    full = write_job(tmp_path, "full", ["payment", "returns", "human"])
    early = write_job(tmp_path, "early", ["payment"])  # the episode stopped after one decision
    compare_served.main([f"A={full}", f"B={early}"])
    out = capsys.readouterr().out
    assert "| A | 2/3 |" in out and "| B | 1/3 |" in out
    assert "- A and B route 1 of 1 (seed, ticket) pairs the same" in out


def test_a_job_that_decided_nothing_still_gets_its_row(tmp_path: Path, capsys) -> None:
    full = write_job(tmp_path, "full", ["payment", "returns", "human"])
    none = write_job(tmp_path, "none", [])  # the server was down: every episode stopped at its first decision
    compare_served.main([f"A={full}", f"B={none}"])
    out = capsys.readouterr().out
    assert "| B | 0/3 | | | 1.0 | | no decision recorded |" in out
    assert "- A and B route 0 of 0 (seed, ticket) pairs the same" in out


def test_a_tick_that_does_not_match_its_route_is_an_error(tmp_path: Path) -> None:
    job = write_job(tmp_path, "job", ["payment", "returns"])
    episode_path = next(job.glob("*/agent/episode.json"))
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["decisions"][1]["key"] = "account"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    with pytest.raises(ValueError, match="t2 routed returns"):
        compare_served.routes(compare_served.load(job))
