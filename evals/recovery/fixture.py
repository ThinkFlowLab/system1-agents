# coding: utf-8
"""The local form fixtures and the independent submit oracle for the recovery eval.

Three tasks, one page each, served from 127.0.0.1 on a free port from a daemon thread (the same shape as
``s1a.tool.hands.serve_static``):

- ``normal``: the field keeps what it is typed; a direct type-and-submit completes it.
- ``recoverable``: an ``input`` handler restores the initial value until the ``Enable editing`` button is clicked,
  which flips a page flag (and removes the button, changing the page key). Typing into the restored field is a
  no-op at the page level, so it stalls the policy; the unlock needs a different action.
- ``permanently_blocked``: the field always restores its initial value and the page offers no unlock control, so the
  policy should stop clearly after its recovery budget is spent.

The server records every real ``POST /submit/<task>`` with the form fields it received. That record is the only
success criterion the eval uses; a model's DONE or answer is never read. The pages themselves are the environment,
and the recorded POST is the independent oracle.

One fixture server is one trial's oracle: ``start_fixture`` mints a unique ``trial_id`` and starts an empty record
namespace, so a submit from an earlier trial can never verify a later one. The runner starts a fresh fixture inside
``run_trial`` for exactly this reason, and every recorded submit carries the trial id it belongs to.
"""

from __future__ import annotations

import dataclasses
import html
import http.server
import itertools
import json
import socketserver
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import parse_qs, urlencode, urlparse

_TRIAL_IDS = itertools.count(1)


@dataclass(frozen=True)
class FormTask:
    """One fixture task: its page, the field value a real person would submit, and whether a page unlock exists.

    ``validate_submission`` is off by default, so the original default tasks keep their exact original behaviour. An
    opt-in task rejects a submitted value that differs from ``expected_value`` at the server (HTTP 422) and re-serves
    the same retryable form with a visible validation error; a correct value still returns the result page. The oracle
    records every real POST either way.
    """

    name: str
    path: str  # the URL path the page is served at
    title: str
    expected_value: str  # the value the oracle looks for in the recorded submit
    kind: str  # normal | recoverable | blocked: how the page reacts to typing
    validate_submission: bool = False  # opt-in server-side validation; the default tasks leave it off

    def __post_init__(self) -> None:
        if self.kind not in ("normal", "recoverable", "blocked"):
            raise ValueError(f"task {self.name}: kind must be normal, recoverable or blocked, got {self.kind!r}")
        if not self.path.startswith("/") or not self.expected_value.strip():
            raise ValueError(f"task {self.name}: path must start with / and expected_value must not be blank")


NORMAL = FormTask("normal", "/normal", "Recovery normal", "hello world", "normal")
RECOVERABLE = FormTask("recoverable", "/recoverable", "Recovery recoverable", "hello world", "recoverable")
BLOCKED = FormTask("permanently_blocked", "/blocked", "Recovery permanently blocked", "hello world", "blocked")
# The DEFAULT_TASKS are retained unchanged: every field is as before and validate_submission stays False.
DEFAULT_TASKS: tuple[FormTask, ...] = (NORMAL, RECOVERABLE, BLOCKED)


def validated_tasks(tasks: tuple[FormTask, ...]) -> tuple[FormTask, ...]:
    """The same tasks with server-side validation on; ``FormTask`` is frozen, so the originals are never mutated."""
    return tuple(dataclasses.replace(task, validate_submission=True) for task in tasks)


def validation_error(task: FormTask, submitted: str) -> str:
    """The ordinary visible message a rejected form shows; it echoes the submitted value, never the expected one."""
    return f"The value '{submitted}' is not valid for {task.name}. Fix the field and submit again."


_REVERT_SCRIPT = """
const input = document.getElementById('value');
window.__editing = false;
input.addEventListener('input', function () { if (!window.__editing) { input.value = 'original'; } });
"""
_ENABLE_SCRIPT = """
document.getElementById('enable').addEventListener('click', function () {
  window.__editing = true;
  document.getElementById('status').textContent = 'Editing enabled';
  this.remove();
});
"""


def page_html(task: FormTask, *, error: str | None = None) -> str:
    """The fixture page: a labelled field, a Submit button, and the task's own revert/unlock behaviour.

    ``error`` re-renders the retryable form with a visible validation message (HTML escaped). With no error the page
    is byte-for-byte the original one, so the default fixture is unchanged.
    """
    initial = "" if task.kind == "normal" else "original"
    status = '<p id="status">Locked</p>\n' if task.kind != "normal" else ""
    enable = '<button type="button" id="enable">Enable editing</button>\n' if task.kind == "recoverable" else ""
    banner = f'<p id="validation-error" role="alert">{html.escape(error)}</p>\n' if error else ""
    script = ""
    if task.kind == "blocked":
        script = f"<script>{_REVERT_SCRIPT}</script>"
    elif task.kind == "recoverable":
        script = f"<script>{_REVERT_SCRIPT}{_ENABLE_SCRIPT}</script>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{task.title}</title></head><body>\n"
        f"<h1>{task.title}</h1>\n"
        f"{status}"
        f"{banner}"
        f"<form method='post' action='/submit/{task.name}'>\n"
        f"<label for='value'>Value</label>\n"
        f"<input id='value' name='value' type='text' value='{initial}'>\n"
        f"{enable}"
        "<button type='submit'>Submit</button>\n"
        "</form>\n"
        f"{script}\n"
        "</body></html>"
    )


def result_html(task_name: str, value: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Submitted {task_name}</title></head><body>\n"
        f"<h1>Submitted</h1><p id='value'>value={value}</p>\n"
        "</body></html>"
    )


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve the fixture pages and record the real form POSTs; everything else is a 404."""

    server: "_Server"  # set by the server factory; the type checker cannot see BaseHTTPRequestHandler's server

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
        path = urlparse(self.path).path
        task = self.server.tasks.get(path)
        if task is not None:
            self._send(200, page_html(task))
            return
        if path == "/":
            rows = "".join(f"<li><a href='{t.path}'>{t.name}</a></li>" for t in self.server.tasks.values())
            self._send(200, f"<!doctype html><html><body><ul>{rows}</ul></body></html>")
            return
        if path == "/oracle":
            self._send(200, json.dumps(self.server.submission_records(), ensure_ascii=False), "application/json")
            return
        self._send(404, "<!doctype html><html><body><h1>404</h1></body></html>")

    def do_POST(self) -> None:  # noqa: N802 - http.server's spelling
        path = urlparse(self.path).path
        if not path.startswith("/submit/"):
            self._send(404, "<!doctype html><html><body><h1>404</h1></body></html>")
            return
        task_name = path[len("/submit/") :]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        fields = {key: values[-1] for key, values in parse_qs(raw).items()}
        self.server.record_submission(task_name, fields)  # every real POST is recorded, valid or not
        task = self.server.tasks_by_name.get(task_name)
        value = fields.get("value", "")
        if task is not None and task.validate_submission and value != task.expected_value:
            # A rejected submit stays on the retryable form: no success result page and no oracle completion.
            self._send(422, page_html(task, error=validation_error(task, value)))
            return
        self._send(200, result_html(task_name, value))


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: Any, tasks: dict[str, FormTask], trial_id: str) -> None:
        super().__init__(address, handler)
        self.tasks = tasks
        self.tasks_by_name = {task.name: task for task in tasks.values()}
        self.trial_id = trial_id
        self._submissions: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_submission(self, task_name: str, fields: dict[str, str]) -> None:
        with self._lock:
            self._submissions.append({"trial": self.trial_id, "task": task_name, "fields": fields})

    def submission_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._submissions]


class Fixture:
    """A running fixture server for exactly one trial: its base URL, its own empty submit namespace and the oracle."""

    def __init__(self, server: _Server) -> None:
        self._server = server
        self.trial_id = server.trial_id
        self.base_url = f"http://127.0.0.1:{server.server_address[1]}"

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def task_url(self, task: FormTask) -> str:
        return self.url(task.path)

    def submissions(self, task: FormTask | None = None) -> list[dict[str, Any]]:
        """Copies of this trial's recorded real submits, optionally just the named task's."""
        records = self._server.submission_records()
        if task is not None:
            records = [record for record in records if record["task"] == task.name]
        return records

    def verified(self, task: FormTask) -> bool:
        """The independent oracle: the server saw a real submit carrying the task's expected field value."""
        return any(record["fields"].get("value") == task.expected_value for record in self.submissions(task))

    def oracle_state(self, task: FormTask) -> dict[str, Any]:
        """What this trial's oracle holds: its id, the recorded submits, and whether any verifies completion."""
        return {
            "trial": self.trial_id,
            "task": task.name,
            "verified": self.verified(task),
            "submissions": self.submissions(task),
        }


@contextmanager
def start_fixture(tasks: tuple[FormTask, ...] = DEFAULT_TASKS) -> Iterator[Fixture]:
    """Yield a one-trial Fixture: a fresh, empty oracle serving ``tasks`` on a free loopback port.

    The unique ``trial_id`` names this trial and every submit it records, so a later fixture starts with no history
    and a previous trial's success can never verify a later one. The runner opens one of these per trial.
    """
    table = {task.path: task for task in tasks}
    if len(table) != len(tasks):
        raise ValueError("every fixture task needs a distinct path")
    trial_id = f"trial-{next(_TRIAL_IDS):04d}"
    server = _Server(("127.0.0.1", 0), _Handler, table, trial_id)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield Fixture(server)
    finally:
        server.shutdown()
        server.server_close()


def post_submit(fixture: Fixture, task: FormTask, value: str) -> tuple[int, str]:
    """A plain HTTP form POST against a fixture; returns the status and body for tests of the oracle without a browser.

    A 422 (a validated task rejecting a mismatched value) is a normal response here, not an exception.
    """
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    body = urlencode({"value": value}).encode()
    request = Request(
        fixture.url(f"/submit/{task.name}"), data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - a loopback URL built by this module
            return response.status, response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:  # a rejected submit is an expected status, so read its body and return it
        return exc.code, exc.read().decode("utf-8", errors="replace")
