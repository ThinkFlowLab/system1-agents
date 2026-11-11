# coding: utf-8
"""Reset a WebHarbor site to its seeded state and grade an episode with the site's own verifier.

``DockerSite`` runs one per-site image (``wildawm/webharbor:<site>-v1``) with the site port published to the host,
where the s1a browser front drives it like any web page. Reset is a fresh container per episode, or, with a
``ControlPlane`` (the all-sites image), ``POST /reset/<site>``. After the episode the agent's final answer, termination
record and final screenshot are copied into the container's ``/logs/agent`` (the layout WebHarbor's harnesses
write), the task's ``tests/`` into ``/tests``, and ``tests/test.sh`` runs inside the container exactly as Harbor's
verifier phase does. The verdict comes from ``/logs/verifier/reward.json``; no LLM judge is involved.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from evals.webharbor.control import ControlPlane
from evals.webharbor.tasks import TaskSpec
from evals.webharbor.verdict import Verdict, read_verdict

Run = Callable[[Sequence[str]], subprocess.CompletedProcess]
Probe = Callable[[str], bool]
NA = "N/A"


class SiteError(RuntimeError):
    """The site could not be reset or graded; the attempt still counts, as a failure with this reason."""


@dataclass(frozen=True)
class Episode:
    """What the agent left for the verifier."""

    answer: str
    status: str  # "completed", "error" or "timeout": the harness termination status
    screenshot: Path | None = None
    finished_at: float = 0.0


class Environment(Protocol):
    def image_digest(self) -> str: ...
    def reset(self, task: TaskSpec) -> None: ...
    def verify(self, task: TaskSpec, task_dir: Path, episode: Episode, out_dir: Path) -> Verdict: ...
    def close(self) -> None: ...


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=600, check=False)


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - the local site URL
            return response.status < 500
    except Exception:  # noqa: BLE001 - not up yet
        return False


def agent_logs(episode: Episode, dest: Path) -> None:
    """``/logs/agent`` as WebHarbor's ``build_trajectory.py`` reads it: answer, termination, screenshot manifest."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "final_answer.txt").write_text(episode.answer, encoding="utf-8")
    (dest / "termination.json").write_text(json.dumps({"status": episode.status}), encoding="utf-8")
    manifest = []
    if episode.screenshot is not None and episode.screenshot.is_file():
        (dest / "screenshots").mkdir(exist_ok=True)
        shutil.copyfile(episode.screenshot, dest / "screenshots" / "final.png")
        manifest.append({"path": "screenshots/final.png", "ts": episode.finished_at or time.time()})
    (dest / "screenshot_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
    )


@dataclass
class DockerSite:
    container: str = "s1a-webharbor"
    control: ControlPlane | None = None
    run: Run = _run
    probe: Probe = _probe
    ready_timeout_s: float = 120.0
    poll_s: float = 1.0
    _image: str | None = None

    def _docker(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        try:
            result = self.run(["docker", *args])
        except FileNotFoundError:
            raise SiteError("docker is not installed") from None
        except subprocess.TimeoutExpired:
            raise SiteError(f"docker {args[0]} timed out") from None
        if check and result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-1:] or ["no stderr"]
            raise SiteError(f"docker {args[0]} failed: {tail[0][:200]}")
        return result

    def image_digest(self) -> str:
        if self._image is None:
            return NA
        result = self._docker("image", "inspect", "--format", "{{index .RepoDigests 0}}", self._image, check=False)
        digest = (result.stdout or "").strip()
        return digest if result.returncode == 0 and "@sha256:" in digest else NA

    def reset(self, task: TaskSpec) -> None:
        self._image = task.image
        if self.control is not None:
            try:
                self.control.reset(task.site)
            except Exception as exc:  # noqa: BLE001 - the control-plane error classes carry no secret
                raise SiteError(f"reset failed: {exc}") from None
            return
        self._docker("rm", "-f", self.container, check=False)
        env_args = [arg for key, value in task.env for arg in ("-e", f"{key}={value}")]
        self._docker(
            "run", "-d", "--name", self.container, "-p", f"127.0.0.1:{task.port}:{task.port}", *env_args, task.image
        )
        deadline = time.monotonic() + self.ready_timeout_s
        while not self.probe(task.start_url):
            if time.monotonic() >= deadline:
                raise SiteError(f"site {task.site} not ready after {self.ready_timeout_s:.0f} s")
            time.sleep(self.poll_s)

    def verify(self, task: TaskSpec, task_dir: Path, episode: Episode, out_dir: Path) -> Verdict:
        tests = task_dir / "tests"
        if not (tests / "test.sh").is_file():
            raise SiteError(f"{task.id}: tests/test.sh missing; download the task first")
        with tempfile.TemporaryDirectory(prefix="s1a-webharbor-") as scratch:
            logs = Path(scratch) / "agent"
            agent_logs(episode, logs)
            self._docker("exec", self.container, "mkdir", "-p", "/logs/agent", "/tests")
            self._docker("cp", f"{tests}/.", f"{self.container}:/tests")
            self._docker("cp", f"{logs}/.", f"{self.container}:/logs/agent")
        self._docker("exec", self.container, "bash", "/tests/test.sh", check=False)  # non-zero is a 0 reward
        verifier = out_dir / "verifier"
        verifier.mkdir(parents=True, exist_ok=True)
        self._docker("cp", f"{self.container}:/logs/verifier/.", str(verifier), check=False)
        return read_verdict(verifier)

    def close(self) -> None:
        if self.control is None:
            self._docker("rm", "-f", self.container, check=False)


def describe(environment: Any) -> dict[str, Any]:
    reset = "control-plane" if getattr(environment, "control", None) is not None else "fresh-container"
    return {"reset": reset, "container": getattr(environment, "container", None)}
