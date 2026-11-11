# coding: utf-8
"""The seeded WebHarbor slice (issue 118), offline: no Docker, no network, no model.

A fake control plane (a loopback HTTP server with the real routes and bearer check), a fake ``docker`` that plays the
container's verifier, and a scripted player stand in for the real ones.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from evals.webharbor import __main__ as cli
from evals.webharbor import control, environment, runner, score, tasks, verdict

TOKEN = "t" * 40
TOML = """
version = "1.0"
[task]
name = "webharbor/allrecipes--1"
[metadata]
webharbor_revision = "381dbd9882680806c81ed626d226f114c464a321"
[environment]
docker_image = "wildawm/webharbor:allrecipes-v1"
[environment.env]
WH_SITE = "allrecipes"
WH_PORT = "40000"
WH_TASK_QUES = "Find a zucchini lasagna."
WH_START_URL = "http://localhost:40000/"
WH_VERIFY_MODULE = "verify_1"
"""


def registry(ids: list[str]) -> list[dict]:
    return [
        {"name": "webharbor-webvoyager", "version": "1.0", "tasks": [{"name": n, "path": f"tasks/{n}"} for n in ids]},
        {"name": "webharbor-test", "version": "1.0", "tasks": []},
    ]


# --- the slice ----------------------------------------------------------------------------------------------------


def test_committed_slice_loads_and_pins_revisions():
    slice_ = tasks.load_slice()
    assert slice_.dataset_ref == "webharbor-webvoyager@1.0"
    assert len(slice_.tasks) == 12 and all(t.id.startswith("allrecipes--") for t in slice_.tasks)
    assert len(slice_.registry_revision) == 40 and slice_.port == 40000


def test_committed_slice_is_reproducible_from_the_seed():
    slice_ = tasks.load_slice()
    pool = [f"allrecipes--{i}" for i in range(45)] + ["amazon--0", "amazon--1"]
    picked = tasks.select(registry(pool), slice_.dataset, slice_.dataset_version, "allrecipes", 12, slice_.seed)
    assert picked == [t.id for t in slice_.tasks]


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"registry_revision": "main"}, "40-character"),
        ({"tasks": []}, "no tasks"),
        ({"tasks": ["amazon--0"]}, "not on the slice's site"),
        ({"tasks": ["allrecipes--1", "allrecipes--1"]}, "repeats"),
        ({"tasks": [{"id": "allrecipes--1", "state_change": "maybe"}]}, "state_change"),
        ({"tasks": ["../etc"]}, "bad task entry"),
        ({"port": 0}, "out of range"),
        ({"seed": "x"}, "seed"),
    ],
)
def test_bad_slices_are_rejected(patch, message):
    data = json.loads(tasks.SLICE_PATH.read_text(encoding="utf-8")) | patch
    with pytest.raises(tasks.SliceError, match=message):
        tasks.parse_slice(data)


def test_slice_against_registry_names_missing_ids():
    slice_ = tasks.load_slice()
    tasks.check_against_registry(slice_, registry([t.id for t in slice_.tasks]))
    with pytest.raises(tasks.SliceError, match="allrecipes--42"):
        tasks.check_against_registry(slice_, registry([t.id for t in slice_.tasks[:-1]]))
    with pytest.raises(tasks.SliceError, match="no dataset"):
        tasks.check_against_registry(slice_, [{"name": "other", "version": "1.0", "tasks": []}])


def _task_dir(root: Path, task_id: str = "allrecipes--1", toml: str = TOML) -> Path:
    path = root / "tasks" / task_id
    (path / "tests").mkdir(parents=True)
    (path / "task.toml").write_text(toml, encoding="utf-8")
    (path / "instruction.md").write_text("Find a zucchini lasagna at http://localhost:40000/.\n", encoding="utf-8")
    (path / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    return path


def test_task_toml_is_parsed_and_checked(tmp_path):
    spec = tasks.load_task(tasks.load_slice(), _task_dir(tmp_path))
    assert (spec.site, spec.port, spec.verify_module) == ("allrecipes", 40000, "verify_1")
    assert spec.instruction.startswith("Find a zucchini") and ("WH_SITE", "allrecipes") in spec.env


def test_task_that_disagrees_with_the_slice_is_rejected(tmp_path):
    other = TOML.replace("allrecipes-v1", "allrecipes-v2")
    with pytest.raises(tasks.SliceError, match="image"):
        tasks.load_task(tasks.load_slice(), _task_dir(tmp_path, toml=other))
    with pytest.raises(tasks.SliceError, match="WH_VERIFY_MODULE"):
        tasks.parse_task("x--1", TOML.replace('WH_VERIFY_MODULE = "verify_1"', ""), "do it")
    with pytest.raises(tasks.SliceError, match="missing"):
        tasks.load_task(tasks.load_slice(), tmp_path / "tasks" / "allrecipes--9")


# --- the control plane --------------------------------------------------------------------------------------------


class FakeControl(BaseHTTPRequestHandler):
    sites = {"allrecipes": "ok", "amazon": "not_ready", "booking": "boom"}
    calls: list[str] = []

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if self._authorized() and self.path == "/health":
            self._send(503, {"ok": False, "sites": {"allrecipes": {"ready": True}, "amazon": {"ready": False}}})

    def do_POST(self):
        if not self._authorized():
            return
        site = self.path.removeprefix("/reset/")
        self.calls.append(site)
        state = self.sites.get(site)
        if state is None:
            self._send(404, {"error": f"unknown site: {site}"})
        elif state == "ok":
            self._send(200, {"site": site, "pid": 7, "ready": True})
        elif state == "not_ready":
            self._send(503, {"site": site, "pid": 8, "ready": False})
        else:
            self._send(503, {"site": site, "ready": False, "error": "RuntimeError: reset failed"})


@pytest.fixture
def control_url():
    server = HTTPServer(("127.0.0.1", 0), FakeControl)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_control_plane_reset_and_health(control_url):
    plane = control.ControlPlane(control_url, TOKEN, timeout_s=5)
    assert plane.reset("allrecipes")["ready"] is True
    assert plane.health()["sites"]["amazon"]["ready"] is False  # a 503 health still returns the table


@pytest.mark.parametrize(
    "site, error, text",
    [
        ("amazon", control.ResetFailed, "not ready"),
        ("booking", control.ResetFailed, "reset failed"),
        ("nowhere", control.UnknownSite, "nowhere"),
    ],
)
def test_control_plane_reset_errors(control_url, site, error, text):
    with pytest.raises(error, match=text):
        control.ControlPlane(control_url, TOKEN, timeout_s=5).reset(site)


def test_control_plane_auth_errors_never_echo_the_token(control_url):
    wrong = "w" * 40
    with pytest.raises(control.ControlPlaneAuthError) as caught:
        control.ControlPlane(control_url, wrong, timeout_s=5).reset("allrecipes")
    assert wrong not in str(caught.value)
    with pytest.raises(control.ControlPlaneAuthError, match="not set"):
        control.ControlPlane(control_url, None).reset("allrecipes")
    with pytest.raises(control.ControlPlaneAuthError, match="shorter"):
        control.ControlPlane(control_url, "short").reset("allrecipes")


def test_control_plane_unreachable_is_a_control_plane_error():
    with pytest.raises(control.ControlPlaneError, match="unreachable"):
        control.ControlPlane("http://127.0.0.1:9", TOKEN, timeout_s=2).reset("allrecipes")


def test_token_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(control.TOKEN_ENV, TOKEN)
    assert control.ControlPlane.from_env().token == TOKEN


# --- the verdict --------------------------------------------------------------------------------------------------


def test_verdict_reads_reward_and_failed_checks(tmp_path):
    (tmp_path / "reward.json").write_text(json.dumps({"reward": 0, "checks_total": 3, "checks_passed": 2}))
    (tmp_path / "verifier_verdict.json").write_text(
        "log line\n{\n" + '"pass": false, "evidence": [{"name": "named_recipe", "pass": false}, {"pass": true}]}'
    )
    got = verdict.read_verdict(tmp_path)
    assert (got.passed, got.checks_total, got.failed_checks, got.error) == (False, 3, ("named_recipe",), None)


@pytest.mark.parametrize("content, error", [(None, "no reward.json"), ("{", "not JSON"), ('{"reward": 0.5}', "0/1")])
def test_missing_or_bad_reward_is_an_error_not_a_pass(tmp_path, content, error):
    if content is not None:
        (tmp_path / "reward.json").write_text(content)
    got = verdict.read_verdict(tmp_path)
    assert not got.passed and error in got.error


# --- the docker site and the runner -------------------------------------------------------------------------------


class FakeDocker:
    """Plays ``docker``: run/rm/exec/cp; ``bash /tests/test.sh`` grades the copied answer like a site verifier."""

    def __init__(self, *, answer_key: str = "Zucchini", run_fails: bool = False, verifier_writes: bool = True):
        self.argv: list[list[str]] = []
        self.answer_key, self.run_fails, self.verifier_writes = answer_key, run_fails, verifier_writes
        self.agent_logs: dict[str, str] = {}

    def __call__(self, argv):
        self.argv.append(list(argv))
        cmd = argv[1]
        ok = subprocess.CompletedProcess(argv, 0, "", "")
        if cmd == "run" and self.run_fails:
            return subprocess.CompletedProcess(argv, 125, "", "Unable to find image\npull access denied")
        if cmd == "cp" and argv[3].endswith(":/logs/agent"):
            src = Path(argv[2].removesuffix("/."))
            self.agent_logs = {p.name: p.read_text(errors="ignore") for p in src.rglob("*") if p.is_file()}
        if cmd == "cp" and argv[2].endswith(":/logs/verifier/."):
            if self.verifier_writes:
                passed = self.answer_key in self.agent_logs.get("final_answer.txt", "")
                Path(argv[3], "reward.json").write_text(
                    json.dumps({"reward": int(passed), "checks_total": 2, "checks_passed": 2 if passed else 1})
                )
        if cmd == "image":
            return subprocess.CompletedProcess(argv, 0, "wildawm/webharbor@sha256:" + "a" * 64 + "\n", "")
        return ok


def _site(fake: FakeDocker) -> environment.DockerSite:
    return environment.DockerSite(run=fake, probe=lambda url: True, poll_s=0)


def _player(answers: dict[str, str | Exception]):
    async def play(task, arm, logs_dir):
        value = answers[arm]
        if isinstance(value, Exception):
            raise value
        if value == "timeout":
            return runner.Outcome("", error="timeout", total_ms=600_000, decision_ms=1_000)
        shot = logs_dir / "final.png"
        shot.write_bytes(b"png")
        return runner.Outcome(value, total_ms=10_000, decision_ms=None if arm == "llm" else 2_000, screenshot=shot)

    return play


def _specs(tmp_path) -> list:
    slice_ = tasks.load_slice()
    task_dir = _task_dir(tmp_path)
    return [(tasks.load_task(slice_, task_dir), task_dir)]


def test_every_attempt_is_reset_played_verified_and_recorded(tmp_path):
    fake = FakeDocker()
    play = _player({"jev": "Vegetarian Zucchini Lasagna", "llm": "I could not find one"})
    attempts = asyncio.run(runner.run_slice(_site(fake), _specs(tmp_path), ["jev", "llm"], play, tmp_path / "out"))
    assert [(a.arm, a.passed) for a in attempts] == [("jev", True), ("llm", False)]
    assert attempts[1].failure == "verifier: fail" and attempts[0].environment_ms == 8_000
    assert attempts[1].decision_ms is None and attempts[1].environment_ms is None
    runs = [argv for argv in fake.argv if argv[1] == "run"]
    assert len(runs) == 2 and "127.0.0.1:40000:40000" in runs[0] and "WH_SITE=allrecipes" in runs[0]
    assert fake.argv[-1][:3] == ["docker", "rm", "-f"]  # closed at the end
    assert ["docker", "exec", "s1a-webharbor", "bash", "/tests/test.sh"] in fake.argv
    assert set(fake.agent_logs) >= {"final_answer.txt", "termination.json", "screenshot_manifest.jsonl", "final.png"}
    lines = (tmp_path / "out" / "attempts.jsonl").read_text().splitlines()
    assert [runner.load_attempts(tmp_path / "out" / "attempts.jsonl")[i] for i in range(2)] == attempts
    assert len(lines) == 2


def test_failures_stay_in_the_denominator_with_their_reason(tmp_path):
    fake = FakeDocker()
    play = _player({"jev": "timeout", "llm": RuntimeError("key sk-secret leaked")})
    attempts = asyncio.run(runner.run_slice(_site(fake), _specs(tmp_path), ["jev", "llm"], play, tmp_path / "o"))
    assert [a.failure for a in attempts] == ["timeout", "agent raised RuntimeError"]
    assert not any(a.passed for a in attempts)
    assert "sk-secret" not in (tmp_path / "o" / "attempts.jsonl").read_text()


def test_reset_failure_and_verifier_error_are_failed_attempts(tmp_path):
    attempts = asyncio.run(
        runner.run_slice(
            _site(FakeDocker(run_fails=True)), _specs(tmp_path / "x"), ["jev"], _player({"jev": "x"}), tmp_path / "a"
        )
    )
    assert attempts[0].failure.startswith("reset: docker run failed: pull access denied")
    attempts = asyncio.run(
        runner.run_slice(
            _site(FakeDocker(verifier_writes=False)),
            _specs(tmp_path / "y"),
            ["jev"],
            _player({"jev": "Zucchini"}),
            tmp_path / "b",
        )
    )
    assert attempts[0].failure == "verifier: verifier wrote no reward.json" and not attempts[0].passed


def test_control_plane_reset_is_used_when_configured(tmp_path, control_url):
    fake = FakeDocker()
    site = environment.DockerSite(control=control.ControlPlane(control_url, TOKEN, timeout_s=5), run=fake)
    attempts = asyncio.run(
        runner.run_slice(site, _specs(tmp_path), ["jev"], _player({"jev": "Zucchini"}), tmp_path / "c")
    )
    assert attempts[0].passed and not [argv for argv in fake.argv if argv[1] in ("run", "rm")]
    bad = environment.DockerSite(control=control.ControlPlane(control_url, "w" * 40, timeout_s=5), run=fake)
    with pytest.raises(environment.SiteError, match="401"):
        bad.reset(_specs(tmp_path / "d")[0][0])


def test_docker_missing_is_a_site_error(tmp_path):
    def missing(argv):
        raise FileNotFoundError("docker")

    with pytest.raises(environment.SiteError, match="not installed"):
        environment.DockerSite(run=missing, probe=lambda url: True).reset(_specs(tmp_path)[0][0])


def test_site_that_never_answers_times_out(tmp_path):
    site = environment.DockerSite(run=FakeDocker(), probe=lambda url: False, ready_timeout_s=0, poll_s=0)
    with pytest.raises(environment.SiteError, match="not ready"):
        site.reset(_specs(tmp_path)[0][0])


def test_image_digest_is_na_until_known():
    fake = FakeDocker()
    site = _site(fake)
    assert site.image_digest() == "N/A"
    site._image = "wildawm/webharbor:allrecipes-v1"
    assert site.image_digest().startswith("wildawm/webharbor@sha256:")


# --- scoring ------------------------------------------------------------------------------------------------------


def test_wilson_interval():
    assert score.wilson(0, 0) is None
    low, high = score.wilson(5, 10)
    assert round(low, 3) == 0.237 and round(high, 3) == 0.763
    assert score.wilson(0, 12)[0] == 0.0 and score.wilson(12, 12)[1] == 1.0
    with pytest.raises(ValueError):
        score.wilson(3, 2)


def test_table_counts_every_attempt_and_writes_na():
    rows = [
        score.Attempt("allrecipes--1", "jev", True, None, total_ms=10_000, decision_ms=2_000),
        score.Attempt("allrecipes--6", "jev", False, "timeout", total_ms=600_000, decision_ms=4_000),
        score.Attempt("allrecipes--1", "llm", False, "reset: docker run failed: x"),
    ]
    summary = score.summarize(rows)
    assert summary["jev"]["attempts"] == 2 and summary["jev"]["rate"] == 0.5
    assert summary["llm"]["median_decision_ms"] is None
    table = score.render(rows, image_digest="N/A", task_revision="abc", dataset="webharbor-webvoyager@1.0")
    assert "| jev | 1 / 2 | 50% |" in table and "timeout x1" in table
    assert "| llm | 0 / 1 | 0% | 0%-79% | N/A | N/A | N/A |" in table
    assert "image `N/A`" in table and "revision `abc`" in table
    empty = score.render([], image_digest="", task_revision="abc", dataset="d")
    assert "0 / 0" in empty


# --- the CLI ------------------------------------------------------------------------------------------------------


def test_cli_show_select_and_score(tmp_path, capsys):
    assert cli.main(["show"]) == 0
    assert "allrecipes--42\tunchecked" in capsys.readouterr().out
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps(registry([f"allrecipes--{i}" for i in range(45)])))
    assert cli.main(["select", "--registry", str(reg)]) == 0
    assert "matches slice.json" in capsys.readouterr().err
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(json.dumps(score.Attempt("allrecipes--1", "jev", True, None).to_json()) + "\n")
    assert cli.main(["score", str(attempts)]) == 0
    assert "| jev | 1 / 1 | 100% |" in capsys.readouterr().out


def test_cli_run_without_downloaded_tasks_explains(tmp_path, capsys):
    assert cli.main(["run", "--tasks-dir", str(tmp_path)]) == 2
    assert "fetch" in capsys.readouterr().err
