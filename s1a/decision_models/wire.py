# coding: utf-8
"""The TypeSafe wire: the async client and the two backends.

Bodies are built and answers validated in ``s1a.decision_models``; ``JevDecisionsClient`` is the transport
behind ``JevModel``. Nothing in here reads an answer.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Any

import httpx

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

DEFAULT_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"
TYPESAFE_DECISIONS_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_DEFAULT_MODEL = "jev-latest"
DECISIONS_BACKENDS = ("typesafe", "openrouter")
DECISIONS_TIMEOUT_S = 5.0  # Jev answers in 0.4 to 1.3 s through the proxy; a dead connection must not stall a step
_RETRY_STATUSES = frozenset({429, 503, 529})
_TRANSPORT_RETRIES = 1
# RemoteProtocolError is usually a stale keep-alive closed before the request was sent; retrying it accepts a rare
# double-billed decision when the server closed the socket after reading the request instead.
_RETRIED_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)
_MAX_RETRY_AFTER_S = 5.0


def decisions_timeout_from_env() -> float:
    """The deadline for one decision, retries included: ``S1A_DECISION_TIMEOUT_S`` when set, else
    ``DECISIONS_TIMEOUT_S``. A local System One server (OneJev, CLM) behind ``TYPESAFE_API_URL`` can take several
    seconds on a page with many options, past the 5 s that fits Jev."""
    raw = (os.getenv("S1A_DECISION_TIMEOUT_S") or "").strip()
    if not raw:
        return DECISIONS_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    # float() also reads nan, inf and 1e9999 (inf): nan passes `<= 0` and poisons the deadline, inf removes it
    if not math.isfinite(value) or value <= 0:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR,
            error_msg=f"S1A_DECISION_TIMEOUT_S must be a finite, positive number of seconds, not {raw!r}",
        )
    return value


def decisions_backend_from_env() -> str:
    """``typesafe`` when a TypeSafe key is set and no proxy URL overrides it; otherwise the OpenRouter proxy."""
    return "typesafe" if os.getenv("TYPESAFE_API_KEY") and not os.getenv("TYPESAFE_API_URL") else "openrouter"


def client_from_env(backend: str, *, timeout_s: float) -> "JevDecisionsClient":
    """Build the decisions client for one backend from the environment.

    ``typesafe`` talks to TypeSafe directly with ``TYPESAFE_API_KEY`` and its own model names;
    ``openrouter`` talks to the proxy with ``OPENROUTER_API_KEY`` (``TYPESAFE_API_URL`` / ``TYPESAFE_MODEL``
    override the proxy URL and model).
    """
    match backend:
        case "typesafe":
            return JevDecisionsClient(
                api_key=os.getenv("TYPESAFE_API_KEY") or "",
                url=TYPESAFE_DECISIONS_URL,
                model=TYPESAFE_DEFAULT_MODEL,
                timeout_s=timeout_s,
            )
        case "openrouter":
            return JevDecisionsClient(
                api_key=os.getenv("OPENROUTER_API_KEY") or "",
                url=os.getenv("TYPESAFE_API_URL") or DEFAULT_DECISIONS_URL,
                model=os.getenv("TYPESAFE_MODEL") or DEFAULT_MODEL,
                timeout_s=timeout_s,
            )
        case _:
            raise build_error(StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg=f"unknown decisions backend {backend!r}")


def _retry_after_s(response: httpx.Response, *, fallback: float) -> float:
    """The server's ``Retry-After`` in seconds, capped at ``_MAX_RETRY_AFTER_S``; ``fallback`` without one."""
    try:
        return min(float(response.headers["Retry-After"]), _MAX_RETRY_AFTER_S)
    except (KeyError, ValueError):
        return fallback


class JevDecisionsClient:
    """Async HTTP client for the decisions endpoint with a warm keep-alive connection."""

    def __init__(self, *, api_key: str, url: str, model: str, timeout_s: float) -> None:
        if not api_key:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="no Jev key: set TYPESAFE_API_KEY or OPENROUTER_API_KEY in .env or the environment",
            )
        # the same guard as S1A_DECISION_TIMEOUT_S, for a deadline passed in code (JevModel.from_env(timeout_s=...))
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"timeout_s must be a finite, positive number of seconds, not {timeout_s!r}",
            )
        self.url = url
        self.model = model
        self._timeout_s = timeout_s  # the deadline for one decision, retries included
        self._client = httpx.AsyncClient(timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"})

    async def warm(self) -> None:
        """Open the TLS connection ahead of the first decision; failures here are ignored."""
        try:
            await self._client.get(self.url.rsplit("/", 1)[0] + "/", timeout=5.0)
        except httpx.HTTPError:
            return None

    async def decide(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """One decision: the payload and the last attempt's latency in ms; every retry fits one overall deadline."""
        deadline = time.monotonic() + self._timeout_s
        attempt = 0
        while True:
            started = time.perf_counter()
            remaining = deadline - time.monotonic()
            try:
                response = await self._client.post(self.url, json=body, timeout=max(remaining, 0.01))
            except _RETRIED_TRANSPORT as exc:
                attempt += 1
                if attempt <= _TRANSPORT_RETRIES and time.monotonic() < deadline:
                    continue
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="decisions connection failed"
                ) from exc
            except httpx.HTTPError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="decisions connection failed"
                ) from exc
            if response.status_code in _RETRY_STATUSES:
                backoff = _retry_after_s(response, fallback=0.5 * 2**attempt)
                attempt += 1
                if attempt < 3 and time.monotonic() + backoff < deadline:
                    await asyncio.sleep(backoff)
                    continue
            if response.is_error:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"decisions endpoint returned HTTP {response.status_code}: {response.text[:200]}",
                )
            try:
                payload = response.json()
            except ValueError as exc:  # JSONDecodeError, or UnicodeDecodeError from a body in a bad encoding
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="decisions endpoint returned a malformed body"
                ) from exc
            if not isinstance(payload, dict):
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, error_msg="decisions endpoint returned a non-object body"
                )
            return payload, round((time.perf_counter() - started) * 1000)

    async def close(self) -> None:
        await self._client.aclose()
