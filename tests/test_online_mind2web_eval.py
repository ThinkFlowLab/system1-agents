# coding: utf-8
"""The seeded Online-Mind2Web live slice (issue 116), offline: no network, no model, no live sites.

A fixture official file and a scripted player stand in for the gated Hugging Face download and ``browse``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from evals.online_mind2web import __main__ as cli
from evals.online_mind2web import runner, score, tasks

REVISION = "6aa56e07c9d247fcc2b72fc3d94ad0f857f1e5d4"
EASY = [
    ("0632e496d37badee0350dad358f047c5", "https://cookpad.com/", "Browse gluten-free cookie recipes.", 4),
    ("2532fd402d3c741b79894e6ff2269f53", "https://www.thumbtack.com/", "Find electricians near 10203.", 3),
    ("34ccd15a8ea8fd3895af83f5ccf62369", "https://us.megabus.com/", "Find what to do when I lose an item.", 3),
    ("3621b099326c7aebd2e2dac6be3b52d1", "https://www.nvidia.com/", "Open the lab leader's profile.", 5),
    (
        "5e1b8254c123c80178cc28e0afdb14f0_110325",
        "https://ticketmaster.com/",
        "Find a help page about how presale works.",
        4,
    ),
    (
        "6db4a0e346976f2729ba9afcd3208941",
        "https://www.fedex.com/en-us/home.html",
        "Look up tracking for 3023858502.",
        2,
    ),
    ("783ce6a3499fa7cf25bc12f8f0ecbbbb", "https://www.mayoclinic.org/", "Find Florida internship programs.", 5),
    ("a11ecdff735b51372d536c866011af6f", "https://www.coursera.org/", "Explore Psychology courses.", 4),
    ("a6f0434ce6aff5f9b03681241b03ad82", "https://finance.yahoo.com/", "Find Tesla's close on March 17, 2023.", 3),
    ("bd1e3770b7181f6fce9c35e18caa9785", "https://craigslist.org/", "Browse solar panel installer listings.", 3),
    ("d5c34bf39eb6096ae5d439325cde4d32", "https://www.dmv.virginia.gov/", "Find a DMV center in Richmond.", 5),
    ("d71be72aa25c3eab8eea47a0e60382e2", "https://www.apple.com/", "Find specs for the latest Macbook Air.", 4),
    ("aa000000000000000000000000000001", "https://www.accuweather.com/", "Show daily weather for New York City.", 4),
    ("aa000000000000000000000000000002", "https://www.drugs.com/", "Browse the natural products database.", 2),
]
MEDIUM = [
    ("207e933d1bba815bcb58664b5d82c085", "https://www.apartments.com/", "Find Ohio City apartments with parking.", 6),
    ("864244b6969e0f8733b0eb1ca06cd51f", "https://www.espn.com/", "Find the last 2023 F1 race winner's time.", 6),
    (
        "c698ff3fc0f6cbce39947c597ab5749b_121425",
        "https://www.eventbrite.com/",
        "Find school activity events in New York.",
        8,
    ),
    ("fa9adb815b85d259f943d81874a052e5", "https://soundcloud.com/", "Browse a user who reposted a Top 50 song.", 6),
    ("bb000000000000000000000000000001", "https://www.nba.com/", "Find the league leader in blocked shots.", 5),
]
SKIPPED = [
    (
        "cc000000000000000000000000000001",
        "easy",
        "https://www.bestbuy.com/",
        "Add a $50 Uber gift card to the cart.",
        4,
    ),
    (
        "cc000000000000000000000000000002",
        "easy",
        "https://www.apartments.com/",
        "Add Elevate at Chicago, IL, to favorites.",
        4,
    ),
    (
        "cc000000000000000000000000000003",
        "medium",
        "https://www.amazon.com/",
        "Find headphones and add them to the cart.",
        8,
    ),
    (
        "cc000000000000000000000000000004",
        "easy",
        "https://www.tumblr.com/",
        "Find the posts liked by the user Taylor Swift.",
        4,
    ),
]


def official_rows() -> list[dict]:
    rows = []
    for task_id, website, text, length in EASY:
        rows.append(
            {
                "task_id": task_id,
                "website": website,
                "confirmed_task": text,
                "reference_length": length,
                "level": "easy",
            }
        )
    for task_id, website, text, length in MEDIUM:
        rows.append(
            {
                "task_id": task_id,
                "website": website,
                "confirmed_task": text,
                "reference_length": length,
                "level": "medium",
            }
        )
    for task_id, level, website, text, length in SKIPPED:
        rows.append(
            {"task_id": task_id, "website": website, "confirmed_task": text, "reference_length": length, "level": level}
        )
    return rows


def fixture_slice() -> dict:
    official = tasks.parse_official(official_rows())
    chosen = tasks.select_ids(official, seed=116, n_easy=12, n_medium=4)
    by_id = {task.id: task for task in official}
    return {
        "name": "online-mind2web-16",
        "dataset": "osunlp/Online-Mind2Web",
        "source_file": "Online_Mind2Web.json",
        "revision": REVISION,
        "expected_tasks": len(official_rows()),
        "seed": 116,
        "n_easy": 12,
        "n_medium": 4,
        "selection": "test fixture",
        "tasks": [
            {
                "id": task_id,
                "level": by_id[task_id].level,
                "website": by_id[task_id].website,
                "reference_length": by_id[task_id].reference_length,
            }
            for task_id in chosen
        ],
    }


def write_pair(tmp_path: Path) -> tuple[Path, Path]:
    official = tmp_path / "Official.json"
    official.write_text(json.dumps(official_rows()), encoding="utf-8")
    slice_path = tmp_path / "slice.json"
    slice_path.write_text(json.dumps(fixture_slice()), encoding="utf-8")
    return slice_path, official


def test_committed_slice_loads_and_pins_the_revision():
    slice_ = tasks.load_slice()
    assert slice_.dataset == "osunlp/Online-Mind2Web"
    assert slice_.revision == REVISION
    assert slice_.seed == 116
    assert (slice_.n_easy, slice_.n_medium, len(slice_.tasks)) == (12, 4, 16)
    assert slice_.expected_tasks == 300
    assert list(slice_.ids) == sorted(slice_.ids)
    assert len(set(slice_.ids)) == 16
    assert {task.level for task in slice_.tasks} == {"easy", "medium"}


def test_fixture_slice_is_reproducible_from_the_seed():
    slice_ = tasks.parse_slice(fixture_slice())
    official = tasks.parse_official(official_rows())
    tasks.check_selection(slice_, official)
    again = tasks.select_ids(official, seed=116, n_easy=12, n_medium=4)
    assert again == list(slice_.ids)
    assert tasks.select_ids(official, seed=116, n_easy=12, n_medium=4) == again


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"revision": "short"}, "40-character"),
        ({"tasks": []}, "names no tasks"),
        ({"n_easy": 99}, "easy tasks"),
        (
            {"tasks": [{"id": "not-a-hash", "level": "easy", "website": "https://x.example/", "reference_length": 1}]},
            "hex",
        ),
    ],
)
def test_bad_slices_are_rejected(patch, message):
    data = fixture_slice()
    data.update(patch)
    with pytest.raises(tasks.SliceError, match=message):
        tasks.parse_slice(data)


def test_public_read_drops_cart_favorites_login_and_payment():
    official = tasks.parse_official(official_rows())
    skipped = {task.id for task in official if not tasks.is_public_read(task)}
    assert skipped == {row[0] for row in SKIPPED}
    assert "cc000000000000000000000000000001" not in tasks.eligible_ids(official, "easy")


def test_check_selection_rejects_a_missing_id_and_a_drifted_website(tmp_path):
    data = fixture_slice()
    rows = [row for row in official_rows() if row["task_id"] != data["tasks"][0]["id"]]
    data["expected_tasks"] = len(rows)
    slice_ = tasks.parse_slice(data)
    with pytest.raises(tasks.SliceError, match="missing"):
        tasks.check_selection(slice_, tasks.parse_official(rows))
    rows = official_rows()
    for row in rows:
        if row["task_id"] == slice_.ids[0]:
            row["website"] = "https://example.invalid/"
    with pytest.raises(tasks.SliceError, match="official file"):
        tasks.check_selection(slice_, tasks.parse_official(rows))


def test_fetch_needs_a_token_and_verifies_the_slice(tmp_path, monkeypatch):
    slice_ = tasks.parse_slice(fixture_slice())
    dest = tmp_path / "Official.json"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(tasks.SliceError, match="HF_TOKEN"):
        tasks.fetch_official(slice_, dest)
    body = json.dumps(official_rows()).encode()

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def opener(request):
        assert "Bearer secret-token" in request.headers["Authorization"]
        return _Resp(body)

    path = tasks.fetch_official(slice_, dest, token="secret-token", opener=opener)
    assert path == dest
    tasks.check_selection(slice_, tasks.load_official(dest))


def test_goal_names_the_site_and_the_task():
    task = tasks.parse_official(official_rows())[0]
    text = tasks.goal(task)
    assert task.website in text and task.confirmed_task in text
    assert text.startswith("Go to ") and "final answer" in text


def _player(answers: dict[str, runner.Outcome | Exception]):
    async def play(task, arm, logs_dir):
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "profile").mkdir(exist_ok=True)  # each attempt gets its own directory
        key = f"{task.id}:{arm}"
        result = answers[key]
        if isinstance(result, Exception):
            raise result
        return result

    return play


@pytest.mark.asyncio
async def test_every_attempt_is_played_and_recorded(tmp_path):
    official = tasks.bind(tasks.parse_slice(fixture_slice()), tasks.parse_official(official_rows()))
    first, second = official[0], official[1]
    scores = {first.id: 4, second.id: 2}

    def judge(task, outcome, logs_dir):
        del outcome, logs_dir
        return scores[task.id]

    attempts = await runner.run_slice(
        [first, second],
        ["jev", "llm"],
        _player(
            {
                f"{first.id}:jev": runner.Outcome("ok", total_ms=1000, decision_ms=200),
                f"{first.id}:llm": runner.Outcome("ok", total_ms=2000, decision_ms=None),
                f"{second.id}:llm": runner.Outcome("ok", total_ms=1500, decision_ms=None),
                f"{second.id}:jev": runner.Outcome("ok", total_ms=800, decision_ms=100),
            }
        ),
        tmp_path / "out",
        judge=judge,
        agents_sha="abc",
    )
    assert [row.task_id for row in attempts] == [first.id, first.id, second.id, second.id]
    assert [row.arm for row in attempts] == ["jev", "llm", "llm", "jev"]  # second task flips arm order
    assert [row.passed for row in attempts] == [True, True, False, False]
    assert attempts[2].failure == "judge:2"
    recorded = runner.load_attempts(tmp_path / "out" / "attempts.jsonl")
    assert len(recorded) == 4
    assert (tmp_path / "out" / f"{first.id}__jev__r0" / "attempt.json").is_file()


@pytest.mark.asyncio
async def test_failures_stay_in_the_denominator_with_their_reason(tmp_path):
    official = tasks.parse_official(official_rows())
    a, b, c, d = official[:4]
    attempts = await runner.run_slice(
        [a, b, c, d],
        ["jev"],
        _player(
            {
                f"{a.id}:jev": runner.Outcome("", error="timeout after 600s", total_ms=600000),
                f"{b.id}:jev": runner.Outcome("", error="BLOCKED by the site"),
                f"{c.id}:jev": RuntimeError("browser crashed"),
                f"{d.id}:jev": runner.Outcome("looks done", total_ms=3000, decision_ms=50),
            }
        ),
        tmp_path / "out",
    )
    assert [row.failure for row in attempts] == ["timeout", "BLOCKED", "agent: browser crashed", "unevaluated"]
    assert all(not row.passed for row in attempts)
    table = score.render(attempts, revision=REVISION, dataset="osunlp/Online-Mind2Web", agents_sha="deadbeef")
    assert "0 / 4" in table and "timeout x1" in table and "unevaluated x1" in table and "BLOCKED x1" in table
    assert "deadbeef" in table and "live-web" in table and "#109" in table and "#104" in table
    assert "N/A" not in table.split("\n")[4] or "0%" in table  # rate is 0%, not N/A


def test_missing_or_failing_judge_is_not_a_pass():
    task = tasks.parse_official(official_rows())[0]
    finished = runner.Outcome("an answer", total_ms=10, decision_ms=1)
    unevaluated = runner.judged_attempt(task, "jev", finished, judge=None, logs_dir=Path("."), agents_sha=None)
    assert unevaluated == runner.Attempt(task.id, "jev", False, "unevaluated", 10, 1)

    def boom(task, outcome, logs_dir):
        raise RuntimeError("no API key")

    failed = runner.judged_attempt(task, "jev", finished, judge=boom, logs_dir=Path("."), agents_sha=None)
    assert failed.passed is False and failed.failure is not None and failed.failure.startswith("judge: no API key")


def test_wilson_interval():
    interval = score.wilson(0, 16)
    assert interval is not None
    lo, hi = interval
    assert 0 <= lo <= hi <= 1
    assert score.wilson(0, 0) is None


def test_table_counts_every_attempt_and_writes_na():
    rows = [
        score.Attempt("a", "jev", True, None, 1000, 200, judge_score=4),
        score.Attempt("b", "jev", False, "timeout", 600000, 10),
        score.Attempt("c", "llm", False, "unevaluated", 2000, None),
    ]
    table = score.render(
        rows,
        revision=REVISION,
        dataset="osunlp/Online-Mind2Web",
        agents_sha="sha",
        model="jev,llm",
        budget="max_steps=40 timeout_s=600",
        date="2026-11-18",
    )
    assert "| jev | 1 / 2 | 50% |" in table
    assert "| llm | 0 / 1 | 0% |" in table
    assert "median decision" in table and "N/A" in table  # llm has no decision time
    assert "Agents SHA `sha`" in table and "2026-11-18" in table


def test_cli_show_select_and_score(tmp_path, capsys):
    slice_path, official = write_pair(tmp_path)
    assert cli.main(["--slice", str(slice_path), "show"]) == 0
    out = capsys.readouterr().out
    assert REVISION in out and "0632e496d37badee0350dad358f047c5" in out
    assert cli.main(["--slice", str(slice_path), "select", "--official", str(official)]) == 0
    selected = capsys.readouterr().out.splitlines()
    assert selected == list(tasks.load_slice(slice_path).ids)
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(
        json.dumps(score.Attempt("x", "jev", False, "unevaluated", 10, 1, agents_sha="cafed00d").to_json()) + "\n",
        encoding="utf-8",
    )
    assert cli.main(["--slice", str(slice_path), "score", str(attempts)]) == 0
    scored = capsys.readouterr().out
    assert "0 / 1" in scored and "unevaluated x1" in scored and "cafed00d" in scored


def test_cli_run_without_downloaded_tasks_explains(tmp_path, capsys):
    slice_path, _ = write_pair(tmp_path)
    assert cli.main(["--slice", str(slice_path), "run", "--official", str(tmp_path / "missing.json")]) == 2
    assert "fetch" in capsys.readouterr().err
