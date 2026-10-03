# coding: utf-8
"""Laya served over HTTP (system1-omni's worker, or plain laya-serve), behind the decision-model interface.

The interface is specified in ``docs/api/laya-systemone.openapi.yaml`` (target) and
``docs/api/laya-systemone.current.openapi.yaml`` (today's servers); ``docs/served-laya.md`` has the design.
``ServedLayaClient`` owns the connection, the deadline, the retries and the error mapping and reads no answer;
``ServedLayaModel`` builds the body the in-process Laya would read and records who answered.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.laya import LAYA_DEFAULT_MAX_LEN, check_window, laya_question
from s1a.decision_models.types import Json, Observation, Question, Reply, Usage

logger = logging.getLogger(__name__)

SERVED_TIMEOUT_S = 5.0  # room for one retry, short enough to fail a step
HEALTH_TIMEOUT_S = 2.0
HEALTH_MAX_AGE_S = 30.0  # the worker reports the live device; laya moves a model to the CPU on a GPU OOM
DEFAULT_SERVED_MODEL = "english"
DEFAULT_RETRY_AFTER_S = 0.5
_RETRIED_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError)
_RETRIED_STATUSES = frozenset({502, 504})  # the frontend could not reach the worker, or it was too slow
_REQUEST_ERRORS = frozenset({400, 413, 422})
_NOT_UP = (
    "the system1-omni worker listens only once it has loaded and warmed up, so it may still be starting; "
    "start it per system1-omni's recipe/laya/apple-silicon.md"
)
_SLOW = (
    "if the worker is loading another checkpoint it holds every request until that is ready: "
    "send again, or raise LAYA_SERVED_TIMEOUT_S"
)


def _reason(response: httpx.Response) -> str:
    """The server's reason: ``code`` and ``detail`` from problem+json, ``detail`` from JSON, else the body's start."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200].strip()
    if isinstance(body, dict):
        code, detail = body.get("code"), body.get("detail")
        if code and detail:
            return f"{code}: {detail}"
        if code or detail:
            return str(code or detail)
    return response.text[:200].strip()


def _retry_after_s(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("retry-after", DEFAULT_RETRY_AFTER_S)))
    except ValueError:
        return DEFAULT_RETRY_AFTER_S


def parse_server_timing(header: str | None) -> dict[str, float]:
    """``queue;dur=0.2, infer;dur=27.4`` -> ``{"queue": 0.2, "infer": 27.4}``; entries without a duration are dropped."""
    timings: dict[str, float] = {}
    for entry in (header or "").split(","):
        name, *params = [part.strip() for part in entry.split(";")]
        for param in params:
            key, _, value = param.partition("=")
            if name and key.strip().lower() == "dur":
                try:
                    timings[name] = float(value.strip().strip('"'))
                except ValueError:
                    pass
    return timings


class ServedLayaClient:
    """HTTP to one ``/v1/systemone`` server: one deadline per decision, one retry, the error mapping.
    httpx's timeout bounds each read, not a request, so every request also runs under ``asyncio.wait_for``: a
    server that trickles its body would otherwise outlast the deadline."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        timeout_s: float = SERVED_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.url = url.rstrip("/")
        self._timeout_s = timeout_s
        self._clock = clock
        self._sleep = sleep
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers, transport=transport)

    def deadline(self) -> float:
        return self._clock() + self._timeout_s

    async def health(self, deadline: float | None = None) -> Json:
        """``GET /health`` once. Refused or unreachable raises; any other failure returns ``{}`` with a warning.
        Within a decision's ``deadline`` it takes at most half of what is left, so the decision keeps the rest."""
        timeout = HEALTH_TIMEOUT_S
        if deadline is not None:
            timeout = min(timeout, (deadline - self._clock()) / 2)
            if timeout <= 0:
                return {}
        try:
            response = await asyncio.wait_for(self._client.get(f"{self.url}/health", timeout=timeout), timeout)
        except TimeoutError:
            logger.warning("[laya-served] /health at %s took longer than %.1f s", self.url, timeout)
            return {}
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"no served Laya at {self.url}: {_NOT_UP}"
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("[laya-served] /health at %s failed: %s", self.url, exc)
            return {}
        if response.status_code != 200:
            logger.warning("[laya-served] /health at %s returned HTTP %s", self.url, response.status_code)
            return {}
        try:
            body = response.json()
        except ValueError:
            logger.warning("[laya-served] /health at %s returned a body that is not JSON", self.url)
            return {}
        return body if isinstance(body, dict) else {}

    async def decide(
        self, body: Json, request_id: str, deadline: float | None = None
    ) -> tuple[Json, dict[str, str], int]:
        """One decision: the payload, the response headers and the last attempt's round trip in ms. Every attempt
        sends the same ``X-Request-Id``, so the server's logs tie a retry to its first try. ``deadline`` (from
        ``deadline()``) is shared with a ``/health`` read made for the same decision; a fresh one starts otherwise."""
        deadline = self.deadline() if deadline is None else deadline
        retried = False
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise self._no_answer(None)
            started = time.perf_counter()
            post = self._client.post(
                f"{self.url}/v1/systemone", json=body, headers={"X-Request-Id": request_id}, timeout=remaining
            )
            try:
                response = await asyncio.wait_for(post, remaining)
            except TimeoutError as exc:
                raise self._no_answer(exc) from exc
            except httpx.TimeoutException as exc:
                if isinstance(exc, httpx.ConnectTimeout) and not retried:
                    retried = True
                    continue
                raise self._no_answer(exc) from exc
            except _RETRIED_TRANSPORT as exc:
                if not retried:
                    retried = True
                    continue
                if isinstance(exc, httpx.ConnectError):
                    raise build_error(
                        StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"no served Laya at {self.url}: {_NOT_UP}"
                    ) from exc
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"served Laya unreachable at {self.url}: {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"served Laya call failed at {self.url}: {exc}"
                ) from exc
            ms = round((time.perf_counter() - started) * 1000)
            status = response.status_code
            if status in _RETRIED_STATUSES and not retried:
                retried = True
                continue
            if status == 503 and not retried:
                wait = _retry_after_s(response)
                if self._clock() + wait < deadline:
                    retried = True
                    await self._sleep(wait)
                    continue
            if status == 401:
                raise build_error(
                    StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                    error_msg=f"served Laya at {self.url} refused the token: set LAYA_SERVED_API_KEY to the worker's LAYA_API_KEY",
                )
            if status in _REQUEST_ERRORS:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"served Laya rejected the request (HTTP {status}): {_reason(response)}",
                )
            if status == 503:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"served Laya at {self.url} is overloaded: {_reason(response)}",
                )
            if status in _RETRIED_STATUSES:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"served Laya unreachable behind {self.url} (HTTP {status}): {_reason(response)}",
                )
            if response.is_error:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"served Laya failed (HTTP {status}): {_reason(response)}",
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served Laya returned a malformed body"
                ) from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
                raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="served Laya returned no answers object")
            return payload, dict(response.headers), ms

    def _no_answer(self, cause: Exception | None) -> Exception:
        return build_error(
            StatusCode.MODEL_CALL_FAILED,
            cause=cause,
            error_msg=f"no answer from served Laya at {self.url} within {self._timeout_s:g} s; {_SLOW}",
        )

    async def close(self) -> None:
        await self._client.aclose()


def served_by_from_health(health: Json, routing: Any, read_at: str | None) -> Json:
    """Who answered, from a ``/health`` reading: the worker's entry for ``routing.model``, else its top-level fields
    when they describe ``routing.repo`` (the top level is one model, the worker's primary), else ``routing.repo``
    alone. The last covers plain laya-serve, whose ``/health`` names no checkpoint and reports the configured device,
    and a checkpoint the worker loaded since this reading, which the next reading lists."""
    routing = routing if isinstance(routing, dict) else {}
    raw_models = health.get("models")
    models: dict[str, Any] = raw_models if isinstance(raw_models, dict) else {}
    entry = models.get(routing.get("model")) if routing.get("model") in models else None
    if entry is None and health.get("checkpoint") and health.get("checkpoint") == routing.get("repo"):
        entry = health
    raw_compile = health.get("compile")
    compile_state: dict[str, Any] = raw_compile if isinstance(raw_compile, dict) else {}
    if not isinstance(entry, dict):
        return {"checkpoint": routing.get("repo"), "revision": None, "device": None, "source": "routing"}
    return {
        "checkpoint": entry.get("checkpoint") or routing.get("repo"),
        "revision": entry.get("revision"),
        "device": entry.get("device"),
        "weights_dtype": entry.get("weights_dtype"),
        "autocast_dtype": entry.get("autocast_dtype"),
        "compiled": compiled(compile_state.get("enabled"), entry.get("device")),
        "source": "health",
        "read_at": read_at,
    }


def compiled(enabled: Any, device: Any) -> bool | None:
    """Whether the answer ran a compiled model. ``--compile`` applies on the GPU only: the worker runs every model on
    the CPU uncompiled, also after a fallback. ``None`` when the server does not say."""
    if not isinstance(enabled, bool) or not isinstance(device, str):
        return None
    return enabled and not device.startswith("cpu")


def identity(served_by: Json, fallback: str) -> str:
    checkpoint, revision = served_by.get("checkpoint"), served_by.get("revision")
    if checkpoint and revision:
        return f"{checkpoint}@{str(revision)[:12]}"
    return str(checkpoint or fallback)


class ServedLayaModel(DecisionModel):
    """Laya behind ``/v1/systemone``: the in-process Laya's questions over HTTP, with who answered in every record."""

    name = "laya-served"
    deterministic = True

    def __init__(
        self,
        client: ServedLayaClient,
        *,
        model: str = DEFAULT_SERVED_MODEL,
        max_len: int = LAYA_DEFAULT_MAX_LEN,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._model = model
        self._max_len = max_len
        self._clock = clock
        self._health: Json = {}
        self._health_read_at: str | None = None
        self._health_tried: float | None = None

    @property
    def model(self) -> str:
        return self._model

    async def _read_health(self, *, strict: bool, deadline: float | None = None) -> None:
        self._health_tried = self._clock()
        try:
            health = await self._client.health(deadline)
        except Exception:
            if strict:
                raise
            if self._health_read_at is not None:  # with no reading yet, the decision itself reports the failure
                logger.warning("[laya-served] kept the previous /health reading from %s", self._health_read_at)
            return
        if health:
            self._health = health
            self._health_read_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    async def warm(self) -> None:
        """Read ``/health`` once: fails early when no server is up, and gives the first identity reading."""
        await self._read_health(strict=True)

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        deadline = self._client.deadline()  # one budget for the identity refresh and the decision
        if self._health_tried is None or self._clock() - self._health_tried >= HEALTH_MAX_AGE_S:
            await self._read_health(strict=False, deadline=deadline)
        body = {
            "model": self._model,
            "state": observation.state,
            "questions": {name: laya_question(question) for name, question in questions.items()},
        }
        request_id = uuid.uuid4().hex
        payload, headers, ms = await self._client.decide(body, request_id, deadline)
        usage = Usage.from_payload(payload.get("usage"))
        check_window(
            usage,
            len(questions),
            self._max_len,
            "shorten the state, or raise the worker's LAYA_MAX_LEN and LAYA_SERVED_MAX_LEN together",
        )
        sent = payload.get("served_by")
        if isinstance(sent, dict):
            served_by = {**sent, "source": "response"}
        else:
            served_by = served_by_from_health(self._health, payload.get("routing"), self._health_read_at)
        answers = payload["answers"]
        lower = {key.lower(): value for key, value in headers.items()}
        return Reply(
            answers={name: answers[name] for name in questions if name in answers},
            latency_ms=ms,
            usage=usage,
            model=identity(served_by, self._model),
            raw={
                **payload,
                "served_by": served_by,
                "url": self._client.url,
                "request_id": request_id,
                "server_timing": parse_server_timing(lower.get("server-timing")),
            },
        )

    async def close(self) -> None:
        await self._client.close()

    @classmethod
    def from_env(cls) -> "ServedLayaModel":
        """``LAYA_SERVED_URL`` (required), ``LAYA_SERVED_MODEL``, ``LAYA_SERVED_API_KEY``, ``LAYA_SERVED_TIMEOUT_S``,
        ``LAYA_SERVED_MAX_LEN``. No cloud key is read."""
        url = os.getenv("LAYA_SERVED_URL")
        if not url:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="--model laya-served needs LAYA_SERVED_URL, e.g. http://127.0.0.1:8000 (the worker) "
                "or http://127.0.0.1:8080 (the frontend)",
            )
        try:
            timeout_s = float(os.getenv("LAYA_SERVED_TIMEOUT_S") or SERVED_TIMEOUT_S)
            max_len = int(os.getenv("LAYA_SERVED_MAX_LEN") or LAYA_DEFAULT_MAX_LEN)
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                cause=exc,
                error_msg="LAYA_SERVED_TIMEOUT_S must be a number and LAYA_SERVED_MAX_LEN an integer",
            ) from exc
        if timeout_s <= 0 or max_len <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="LAYA_SERVED_TIMEOUT_S and LAYA_SERVED_MAX_LEN must be positive",
            )
        client = ServedLayaClient(url=url, api_key=os.getenv("LAYA_SERVED_API_KEY") or None, timeout_s=timeout_s)
        return cls(client, model=os.getenv("LAYA_SERVED_MODEL") or DEFAULT_SERVED_MODEL, max_len=max_len)
