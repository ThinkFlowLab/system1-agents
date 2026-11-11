# coding: utf-8
"""WebHarbor's control plane (``control_server.py``, port 8101 by default): health and per-site reset.

Every request carries ``Authorization: Bearer <WEBSYN_CONTROL_TOKEN>``. ``POST /reset/<site>`` kills the site, restores
its seeded database and waits for it to answer; 200 with ``ready: true`` is the only success. The token is never
logged or put into an error message.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

TOKEN_ENV = "WEBSYN_CONTROL_TOKEN"
DEFAULT_URL = "http://127.0.0.1:8101"
MIN_TOKEN = 32  # control_server.py refuses to start with a shorter token


class ControlPlaneError(RuntimeError):
    """The control plane could not be reached or did not do what was asked."""


class ControlPlaneAuthError(ControlPlaneError):
    """401: the token is missing or wrong."""


class UnknownSite(ControlPlaneError):
    """404: the control plane does not serve this site."""


class ResetFailed(ControlPlaneError):
    """The reset ran but the site did not come back ready (503, or 200 without ``ready``)."""


Opener = Callable[[urllib.request.Request, float], Any]


def _urlopen(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - a configured local control URL


@dataclass
class ControlPlane:
    base_url: str = DEFAULT_URL
    token: str | None = None
    timeout_s: float = 120.0  # a reset restarts the site and waits up to 60 s for it
    opener: Opener = _urlopen

    @classmethod
    def from_env(cls, base_url: str = DEFAULT_URL, **kwargs: Any) -> "ControlPlane":
        return cls(base_url=base_url, token=os.environ.get(TOKEN_ENV), **kwargs)

    def _call(self, method: str, path: str) -> tuple[int, dict[str, Any]]:
        if not self.token:
            raise ControlPlaneAuthError(f"{TOKEN_ENV} is not set")
        if len(self.token) < MIN_TOKEN:
            raise ControlPlaneAuthError(f"{TOKEN_ENV} is shorter than {MIN_TOKEN} characters")
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            method=method,
            data=b"" if method == "POST" else None,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
        )
        try:
            with self.opener(request, self.timeout_s) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read() if exc.fp is not None else b""
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ControlPlaneError(f"control plane {self.base_url} unreachable: {type(reason).__name__}") from None
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        if status == 401:
            raise ControlPlaneAuthError("control plane rejected the token (401)")
        return status, body

    def health(self) -> dict[str, Any]:
        """``/health``: ``{"ok": bool, "sites": {site: {pid, alive, ready, port}}}``; a 503 still returns the body."""
        status, body = self._call("GET", "/health")
        if status not in (200, 503) or "sites" not in body:
            raise ControlPlaneError(f"/health answered {status} without a site table")
        return body

    def reset(self, site: str) -> dict[str, Any]:
        """Reseed and restart one site; returns the body (``site``, ``pid``, ``ready``, maybe ``cleanup_warning``)."""
        status, body = self._call("POST", f"/reset/{site}")
        if status == 404:
            raise UnknownSite(f"control plane does not serve {site!r}")
        if status == 200 and body.get("ready") is True:
            return body
        detail = body.get("error") or ("site not ready after reset" if status in (200, 503) else f"HTTP {status}")
        raise ResetFailed(f"reset {site}: {detail}")
