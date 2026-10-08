# coding: utf-8
"""Frames of a browser run, taken by its own Playwright MCP session: a stdio proxy that asks for a screenshot after
every browser tool call and writes it as ``t<elapsed ms>-<n>.png``.

    PLAYWRIGHT_MCP_COMMAND=<python> \\
    PLAYWRIGHT_MCP_ARGS="-m evals.replay.cast --frames runs/cast/flights-jev -- npx -y @playwright/mcp@0.0.78" \\
    s1a run flights --model jev --headed

jiuwen launches the MCP server from those two variables and appends its own flags (``--isolated``, ``--headless``)
to the args, which land after ``--`` and reach the real server. The proxy forwards every JSON-RPC line both ways;
after each ``tools/call`` of a ``browser_*`` tool it sends one ``browser_take_screenshot`` of its own, keeps that
response for itself and saves the PNG as ``t<elapsed ms>-<n>-<tool>.png``; ``calls.jsonl`` beside the frames holds
each call's tool name and its request and response times on the same clock. The run's own client is the only client of the browser, so the policy
behaves as it does without frames, one screenshot per step later. Two such folders, one per model, go side by side
on the wall clock with ``python -m evals.replay --from-frames``.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCREENSHOT_TOOL = "browser_take_screenshot"
OWN_ID_PREFIX = "cast-"


class Proxy:
    def __init__(self, frames_dir: Path, server: list[str]) -> None:
        self.frames_dir = frames_dir
        self.server = subprocess.Popen(server, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.started = time.monotonic()
        self.pending: dict[Any, tuple[str, int]] = {}  # the client's tool calls in flight: tool and request ms, by id
        self.calls: dict[
            str, tuple[str, int, int]
        ] = {}  # our screenshot id -> the call's tool, request ms, response ms
        self.own = 0
        self.saved = 0
        self.outstanding = 0  # screenshot requests of our own still unanswered
        self.client_done = False
        self.lock = threading.Lock()

    def run(self) -> int:
        threading.Thread(target=self.client_to_server, daemon=True).start()
        self.server_to_client()
        return int(self.server.wait())

    def send(self, message: dict[str, Any]) -> None:
        assert self.server.stdin is not None
        with self.lock:
            if self.server.stdin.closed:
                raise RuntimeError(f"server stdin closed before {message.get('id')} could be sent; a frame is lost")
            self.server.stdin.write((json.dumps(message) + "\n").encode())
            self.server.stdin.flush()

    def maybe_close(self) -> None:
        """The server's stdin closes once the client is done and no screenshot of ours is still owed."""
        assert self.server.stdin is not None
        with self.lock:
            if self.client_done and not self.pending and self.outstanding == 0 and not self.server.stdin.closed:
                self.server.stdin.close()

    def client_to_server(self) -> None:
        for raw in sys.stdin.buffer:
            message = _parse(raw)
            if message is not None and message.get("method") == "tools/call":
                name = str((message.get("params") or {}).get("name") or "")
                if name.startswith("browser_") and name != SCREENSHOT_TOOL:
                    self.pending[message.get("id")] = (name, self.elapsed_ms())
            with self.lock:
                assert self.server.stdin is not None
                self.server.stdin.write(raw)
                self.server.stdin.flush()
        self.client_done = True
        self.maybe_close()

    def server_to_client(self) -> None:
        assert self.server.stdout is not None
        for raw in self.server.stdout:
            message = _parse(raw)
            identifier = message.get("id") if message is not None else None
            if isinstance(identifier, str) and identifier.startswith(OWN_ID_PREFIX):
                self.save(message)
                with self.lock:
                    self.outstanding -= 1
                self.maybe_close()
                continue
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            with self.lock:  # the client's EOF must not close the server between the pop and the screenshot request
                call = self.pending.pop(identifier, None) if message is not None else None
                if call is not None:
                    self.own += 1
                    self.outstanding += 1
                    self.calls[f"{OWN_ID_PREFIX}{self.own}"] = (call[0], call[1], self.elapsed_ms())
            if call is not None:
                self.send(
                    {
                        "jsonrpc": "2.0",
                        "id": f"{OWN_ID_PREFIX}{self.own}",
                        "method": "tools/call",
                        "params": {"name": SCREENSHOT_TOOL, "arguments": {"type": "png"}},
                    }
                )
            self.maybe_close()

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)

    def save(self, message: dict[str, Any]) -> None:
        tool, request_ms, response_ms = self.calls.pop(str(message.get("id")), ("", 0, 0))
        for item in (message.get("result") or {}).get("content") or []:
            if item.get("type") == "image" and item.get("data"):
                elapsed_ms = self.elapsed_ms()
                self.frames_dir.mkdir(parents=True, exist_ok=True)
                self.saved += 1
                name = f"t{elapsed_ms:08d}-{self.saved:04d}-{tool}.png"  # the sequence keeps two frames in one ms apart
                (self.frames_dir / name).write_bytes(base64.b64decode(item["data"]))
                record = {
                    "n": self.saved,
                    "tool": tool,
                    "request_ms": request_ms,
                    "response_ms": response_ms,
                    "frame_ms": elapsed_ms,
                }
                with (self.frames_dir / "calls.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record) + "\n")
                return


def _parse(raw: bytes) -> dict[str, Any] | None:
    try:
        message = json.loads(raw)
    except ValueError:
        return None
    return message if isinstance(message, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, required=True, help="where the PNGs go")
    argv = sys.argv[1:]
    if "--" not in argv:
        parser.error("give the MCP server command after --")
    own, server = argv[: argv.index("--")], argv[argv.index("--") + 1 :]
    args = parser.parse_args(own)
    if not server:
        parser.error("give the MCP server command after --")
    sys.exit(Proxy(args.frames, server).run())


if __name__ == "__main__":
    main()
