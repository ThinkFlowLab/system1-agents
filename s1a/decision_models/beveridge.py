# coding: utf-8
"""Beveridge (Large-C / Pro-C / Pro-C-Web) as a decision backend.

Beveridge (https://arxiv.org/abs/2611.18655, Apache-2.0) answers typed questions over a document with a probability
distribution per question and no generated text. The HTTP server is ``POST /v1/decide`` (not ``/v1/systemone``), so
this cannot ride ``--model jev``. In-process use is ``bev.load()`` of Beveridge-Large-C when ``BEV_URL`` is unset;
``BEV_URL`` wins and posts to that server. ``BEV_MODEL`` selects Large-C, Pro-C or Pro-C-Web (or a repo id / path).
Large-C is text-only (no images, 4,096-token context, English). ``score`` questions are out of scope: Agents has no
``ScoreQuestion``. Weights are not vendored; CI never downloads them.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError, build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.omnijev import omnijev_option
from s1a.decision_models.served import parse_server_timing
from s1a.decision_models.types import ChoiceQuestion, Json, NoulQuestion, Observation, Question, Reply, Usage

logger = logging.getLogger(__name__)

# Paper / this adapter cap at 15; bev-decisions 0.2.5 allows 16. Fail closed here on more than 15.
MAX_BEV_QUESTIONS = 15
DEFAULT_BEV_MODEL = "sathiskumarjothi0/Beveridge-Large-C"
PRO_BEV_MODEL = "sathiskumarjothi0/Beveridge-Pro-C"
WEB_BEV_MODEL = "sathiskumarjothi0/Beveridge-Pro-C-Web"
BEV_TIMEOUT_S = 30.0
BEV_HEALTH_TIMEOUT_S = 2.0
BEV_MODEL_ALIASES = {
    "large-c": DEFAULT_BEV_MODEL,
    "large": DEFAULT_BEV_MODEL,
    "default": DEFAULT_BEV_MODEL,
    "beveridge-large-c": DEFAULT_BEV_MODEL,
    "pro-c": PRO_BEV_MODEL,
    "pro": PRO_BEV_MODEL,
    "beveridge-pro-c": PRO_BEV_MODEL,
    "pro-c-web": WEB_BEV_MODEL,
    "web": WEB_BEV_MODEL,
    "beveridge-pro-c-web": WEB_BEV_MODEL,
}
_RETRIED_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError)
_RETRIED_STATUSES = frozenset({502, 504})
_NOT_UP = (
    "a Beveridge server listens at /v1/decide once the checkpoint is loaded; start it with "
    "`bev serve --model sathiskumarjothi0/Beveridge-Large-C --port 8000` and point BEV_URL at it"
)


def resolve_bev_model(raw: str | None) -> str:
    """``BEV_MODEL``: a short alias, a Hugging Face id, or a local path. Default is Beveridge-Large-C."""
    value = (raw or "").strip()
    if not value:
        return DEFAULT_BEV_MODEL
    aliased = BEV_MODEL_ALIASES.get(value.lower())
    if aliased:
        return aliased
    return value


def beveridge_question(question: Question) -> Json:
    """Choice / noul in Beveridge's ``/v1/decide`` shape: string instructions, string option labels.

    Beveridge requires a non-empty ``instructions`` string (not Jev's instruction object). Browser element rows
    become their label and value via ``omnijev_option``. We do not send ``score`` questions or an ``images`` field.
    """
    match question:
        case ChoiceQuestion():
            parts = [question.goal, question.operation, *question.rules]
            return {
                "type": "choice",
                "instructions": "\n".join(part for part in parts if part) or "Which option comes next?",
                "criteria": {key: omnijev_option(description) for key, description in question.options.items()},
            }
        case NoulQuestion():
            body: Json = {"type": "noul", "instructions": question.question}
            if question.criteria:
                body["criteria"] = dict(question.criteria)
            return body
    raise TypeError(f"not a question: {question!r}")


def beveridge_payload(body: object) -> Json:
    """The decide response: ``{\"answers\": {...}}``. Missing or malformed answers fail closed."""
    if not isinstance(body, dict):
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="beveridge returned a malformed body")
    if isinstance(body.get("answers"), dict):
        return body
    raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="beveridge returned no answers object")


class BeveridgeClient:
    """HTTP to one Beveridge server: ``POST /v1/decide``, one deadline per decision, one transport / 502 / 504 retry."""

    def __init__(
        self,
        *,
        url: str,
        endpoint: str,
        api_key: str | None = None,
        timeout_s: float = BEV_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        health_url: str | None = None,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"timeout_s must be a finite number above 0, not {timeout_s!r}",
            )
        self.url = url.rstrip("/")
        self.endpoint = endpoint
        self.health_url = health_url
        self._timeout_s = timeout_s
        self._clock = clock
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers, transport=transport)

    async def warm(self) -> Json:
        """``GET /health``; the upstream server returns ``{\"ok\": true}`` and does not name the checkpoint."""
        if not self.health_url:
            return {}
        try:
            response = await asyncio.wait_for(
                self._client.get(self.health_url, timeout=BEV_HEALTH_TIMEOUT_S), BEV_HEALTH_TIMEOUT_S
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"no served Beveridge at {self.url}: {_NOT_UP}"
            ) from exc
        except (TimeoutError, httpx.HTTPError) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                cause=exc,
                error_msg=f"served Beveridge at {self.url} did not answer: {exc}",
            ) from exc
        if response.is_error:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                error_msg=f"served Beveridge at {self.url} returned HTTP {response.status_code}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served Beveridge returned a malformed body"
            ) from exc
        return body if isinstance(body, dict) else {}

    async def decide(self, body: Json, request_id: str) -> tuple[Json, int, dict[str, str]]:
        """One decision: the payload, its round trip in ms and the response headers."""
        headers = {"X-Request-Id": request_id}
        deadline = self._clock() + self._timeout_s
        retried = False
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise self._no_answer(None)
            started = time.perf_counter()
            post = self._client.post(self.endpoint, json=body, headers=headers, timeout=remaining)
            try:
                response = await asyncio.wait_for(post, remaining)
            except TimeoutError as exc:
                raise self._no_answer(exc) from exc
            except _RETRIED_TRANSPORT as exc:
                if not retried:
                    retried = True
                    continue
                if isinstance(exc, httpx.ConnectError):
                    raise build_error(
                        StatusCode.MODEL_CALL_FAILED,
                        cause=exc,
                        error_msg=f"no served Beveridge at {self.url}: {_NOT_UP}",
                    ) from exc
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    cause=exc,
                    error_msg=f"served Beveridge unreachable at {self.url}: {exc}",
                ) from exc
            except httpx.HTTPError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    cause=exc,
                    error_msg=f"served Beveridge call failed at {self.url}: {exc}",
                ) from exc
            ms = round((time.perf_counter() - started) * 1000)
            if response.status_code in _RETRIED_STATUSES and not retried:
                retried = True
                continue
            if response.is_error:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"served Beveridge failed (HTTP {response.status_code}): {response.text[:200].strip()}",
                )
            try:
                raw = response.json()
            except ValueError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served Beveridge returned a malformed body"
                ) from exc
            return beveridge_payload(raw), ms, dict(response.headers)

    def _no_answer(self, cause: Exception | None) -> Exception:
        return build_error(
            StatusCode.MODEL_CALL_FAILED,
            cause=cause,
            error_msg=f"no answer from served Beveridge at {self.url} within {self._timeout_s:g} s; {_NOT_UP}",
        )

    async def close(self) -> None:
        await self._client.aclose()


class BeveridgeModel(DecisionModel):
    """Beveridge in process (``bev.load``) or over ``POST /v1/decide``; choice and noul; text only."""

    name = "beveridge"
    supports_images = False
    deterministic = True
    bills_input_tokens = False

    def __init__(
        self,
        *,
        model: str = DEFAULT_BEV_MODEL,
        client: BeveridgeClient | None = None,
        agent: Any = None,
    ) -> None:
        if (client is None) == (agent is None):
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="BeveridgeModel needs exactly one of a served client or an in-process agent",
            )
        self._client = client
        self._agent = agent
        self._model = model
        self._served: Json = {}
        self._served_at: str | None = None
        self._warmed = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def path(self) -> str:
        return "served" if self._client is not None else "in-process"

    async def warm(self) -> None:
        if self._client is None:
            self._warmed = True
            return
        self._served = await self._client.warm()
        self._served_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._warmed = True

    def _served_by(self) -> Json:
        if self._client is None:
            return {
                "path": "in-process",
                "checkpoint": self._model,
                "source": "load",
                "read_at": self._served_at,
            }
        return {
            "url": self._client.url,
            "path": "served",
            "checkpoint": self._served.get("model") or self._served.get("checkpoint") or self._model,
            "device": self._served.get("device"),
            "source": "health" if self._client.health_url else "served",
            "read_at": self._served_at,
        }

    async def _warm_once(self) -> None:
        if self._warmed or self._client is None or not self._client.health_url:
            return
        try:
            await self.warm()
        except BaseError as exc:
            logger.warning("[beveridge] could not read /health at %s: %s", self._client.url, exc)

    def _check(self, questions: dict[str, Question]) -> None:
        super()._check(questions)
        if len(questions) > MAX_BEV_QUESTIONS:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=(
                    f"beveridge accepts at most {MAX_BEV_QUESTIONS} questions in one decide_many "
                    f"(got {len(questions)}; the paper caps a request at 15)"
                ),
            )

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        asked = {name: beveridge_question(question) for name, question in questions.items()}
        if self._client is not None:
            return await self._decide_served(observation, questions, asked)
        return await self._decide_in_process(observation, questions, asked)

    async def _decide_served(
        self, observation: Observation, questions: dict[str, Question], asked: dict[str, Json]
    ) -> Reply:
        assert self._client is not None
        # Official body is state + questions only; do not send model, images, or a System One path.
        body: Json = {"state": observation.state, "questions": asked}
        await self._warm_once()
        request_id = uuid.uuid4().hex
        payload, ms, headers = await self._client.decide(body, request_id)
        answers = payload["answers"]
        lower = {key.lower(): value for key, value in headers.items()}
        raw: Json = {
            **payload,
            "url": self._client.url,
            "request_id": request_id,
            "requested_model": self._model,
            "served_by": self._served_by(),
            "server_timing": parse_server_timing(lower.get("server-timing")),
        }
        return Reply(
            answers={name: answers[name] for name in questions if name in answers},
            latency_ms=ms,
            usage=Usage.from_payload(payload.get("usage")),
            model=str(payload.get("model") or self._model),
            raw=raw,
        )

    async def _decide_in_process(
        self, observation: Observation, questions: dict[str, Question], asked: dict[str, Json]
    ) -> Reply:
        assert self._agent is not None
        started = time.perf_counter()
        try:
            answers = await asyncio.to_thread(self._agent.decide, observation.state, asked)
        except (ValueError, RuntimeError) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"beveridge forward pass failed: {exc}"
            ) from exc
        ms = round((time.perf_counter() - started) * 1000)
        if not isinstance(answers, dict):
            raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="beveridge returned no answers object")
        # In-process bev.Bev.decide returns the answers dict directly, not an {"answers": ...} envelope.
        raw: Json = {
            "answers": answers,
            "requested_model": self._model,
            "served_by": self._served_by(),
        }
        echoed = getattr(self._agent, "model_name", None) or getattr(self._agent, "name", None)
        return Reply(
            answers={name: answers[name] for name in questions if name in answers},
            latency_ms=ms,
            usage=Usage(),
            model=str(echoed or self._model),
            raw=raw,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    @classmethod
    def from_env(cls) -> "BeveridgeModel":
        """``BEV_URL`` (served ``/v1/decide``) wins when set; otherwise ``bev.load()`` of the selected checkpoint.

        ``BEV_MODEL`` is ``large-c`` (default), ``pro-c``, ``pro-c-web``, a Hugging Face id, or a local path.
        ``BEV_API_KEY`` is only for a gateway in front of ``bev serve`` (the upstream server has no auth).
        """
        try:
            timeout_s = float(os.getenv("BEV_TIMEOUT_S") or BEV_TIMEOUT_S)
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg="BEV_TIMEOUT_S must be a number"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="BEV_TIMEOUT_S must be a finite number above 0",
            )
        model = resolve_bev_model(os.getenv("BEV_MODEL"))
        url = (os.getenv("BEV_URL") or "").strip()
        if url:
            base = url.rstrip("/")
            client = BeveridgeClient(
                url=base,
                endpoint=f"{base}/v1/decide",
                api_key=(os.getenv("BEV_API_KEY") or "").strip() or None,
                timeout_s=timeout_s,
                health_url=f"{base}/health",
            )
            return cls(client=client, model=model)
        try:
            import bev
        except ImportError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=(
                    "--model beveridge needs BEV_URL (a bev serve /v1/decide, e.g. http://127.0.0.1:8000) or "
                    "the beveridge extra: uv sync --extra beveridge"
                ),
            ) from exc
        device = (os.getenv("BEV_DEVICE") or "").strip() or None
        try:
            agent = bev.load(model, device=device)
        except Exception as exc:  # the package raises RuntimeError on a missing repo; torch may raise too
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                cause=exc,
                error_msg=f"beveridge failed to load {model!r}: {exc}",
            ) from exc
        if not callable(getattr(agent, "decide", None)):
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="bev.load() returned an object without a callable decide(state, questions)",
            )
        return cls(agent=agent, model=model)
