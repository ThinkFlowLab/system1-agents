# coding: utf-8
"""CLM behind ``/v1/systemone``: the schema CLM's own client sends, over HTTP.

CLM is a frozen Qwen3-8B encoder plus two projection heads, and the engine that scores with them
does not compute embeddings: an ``/v1/embeddings`` server runs beside it. ``clm-serve`` owns both
halves and speaks the same ``/v1/systemone`` request the other served models do, so this backend
posts the body ``clm.client`` would post and records which server answered.

Correspondence with system1-omni: the engine there is ``omni-clm``, which is a library plus the
``clm-run`` harness and deliberately serves no HTTP of its own (``src/models/clm/README.md``); a
frontend owns that socket. This backend talks to ``clm-serve`` today, and to whatever serves the
same schema later without a change here.
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

import httpx

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError, build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.served import parse_server_timing
from s1a.decision_models.types import (
    ChoiceQuestion,
    Json,
    NoulQuestion,
    Observation,
    Question,
    Reply,
    Usage,
)

logger = logging.getLogger(__name__)

CLM_TIMEOUT_S = 30.0  # one 8B forward pass per cold candidate set, plus the encoder's own latency
CLM_HEALTH_TIMEOUT_S = 2.0
DEFAULT_CLM_MODEL = "clm-latest"
_RETRIED_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError)
# What system1-omni's frontend answers when it cannot reach the engine behind it, or when it was too slow:
# transient by construction, so one retry, as the sibling backend does for the same deployment shape.
_RETRIED_STATUSES = frozenset({502, 504})
_NOT_UP = (
    "clm-serve listens only once it has loaded the checkpoint, so it may still be starting; start it per "
    "system1-omni's recipe/clm/README.md (it needs an /v1/embeddings server beside it)"
)


def clm_question(question: Question) -> Json:
    """One question in CLM's schema, the shape ``clm.client`` builds.

    A choice's ``criteria`` is ``{key: description}`` and the action head embeds each description as the
    candidate's own text with nothing prefixed. A noul takes the statement as its instructions and
    ``criteria`` only when the caller spells out what true and false mean.
    """
    match question:
        case ChoiceQuestion():
            # CLM's ``instructions`` is the question as text — its own examples send a sentence, and the
            # state head embeds ``state + instructions`` — so the caller's goal, operation and rules are
            # joined into one string rather than sent as the object Jev wants.
            parts = [question.goal, question.operation, *question.rules]
            return {
                "type": "choice",
                "instructions": "\n".join(part for part in parts if part) or None,
                "criteria": dict(question.options),
            }
        case NoulQuestion():
            body: Json = {"type": "noul", "instructions": question.question}
            if question.criteria:
                body["criteria"] = dict(question.criteria)
            return body
    raise TypeError(f"not a question: {question!r}")


def server_timing(lower: dict[str, str]) -> dict[str, float]:
    """The response's ``Server-Timing``, plus the engine's own ``X-CLM-Latency-Ms`` under ``clm``.

    That second header is the one worth keeping: it separates the engine's scoring time from the round trip
    ``Reply.latency_ms`` measures, which is what tells a slow engine from a slow connection. It is folded in here
    rather than kept beside it because ``server_timing`` is the provenance key for durations -- a separate key
    would sit in ``raw`` and never reach ``Decision.provenance``.
    """
    timings = parse_server_timing(lower.get("server-timing"))
    engine_ms = lower.get("x-clm-latency-ms")
    if engine_ms is not None:
        try:
            timings["clm"] = float(engine_ms)
        except ValueError:  # a header that is not a number is not worth failing a decision over
            pass
    return timings


class ClmClient:
    """HTTP to one ``/v1/systemone`` server: one deadline per decision, one retry on a refused connection.

    ``httpx``'s timeout bounds each read rather than a request, so every post also runs under
    ``asyncio.wait_for``: a server that trickles its body would otherwise outlast the deadline.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        timeout_s: float = CLM_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"timeout_s must be a finite number above 0, not {timeout_s!r}",
            )
        self.url = url.rstrip("/")
        self._timeout_s = timeout_s
        self._clock = clock
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers, transport=transport)

    async def warm(self) -> Json:
        """``GET /health`` once: fail early when nothing is listening, and name what is served.

        ``/health`` and not ``/v1/models``: clm-serve serves both, but system1-omni's ``omni-jev`` frontend —
        the path this backend is meant to be used through — routes only ``/v1/systemone`` and ``/health``.
        Reading the one the frontend does not expose made every decision fail with its 404.
        """
        try:
            response = await asyncio.wait_for(
                self._client.get(f"{self.url}/health", timeout=CLM_HEALTH_TIMEOUT_S), CLM_HEALTH_TIMEOUT_S
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"no served CLM at {self.url}: {_NOT_UP}"
            ) from exc
        except (TimeoutError, httpx.HTTPError) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"served CLM at {self.url} did not answer: {exc}"
            ) from exc
        if response.is_error:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                error_msg=f"served CLM at {self.url} returned HTTP {response.status_code}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served CLM returned a malformed body"
            ) from exc
        return body if isinstance(body, dict) else {}

    async def decide(self, body: Json, request_id: str) -> tuple[Json, int, dict[str, str]]:
        """One decision: the payload, its round trip in ms and the response headers. Every attempt sends the same
        ``X-Request-Id``, so a server's logs tie a retry to its first try."""
        headers = {"X-Request-Id": request_id}
        retried = False
        while True:
            started = time.perf_counter()
            post = self._client.post(f"{self.url}/v1/systemone", json=body, headers=headers, timeout=self._timeout_s)
            try:
                response = await asyncio.wait_for(post, self._timeout_s)
            except TimeoutError as exc:
                raise self._no_answer(exc) from exc
            except _RETRIED_TRANSPORT as exc:
                if not retried:
                    retried = True
                    continue
                if isinstance(exc, httpx.ConnectError):
                    raise build_error(
                        StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"no served CLM at {self.url}: {_NOT_UP}"
                    ) from exc
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"served CLM unreachable at {self.url}: {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"served CLM call failed at {self.url}: {exc}"
                ) from exc
            ms = round((time.perf_counter() - started) * 1000)
            if response.status_code in _RETRIED_STATUSES and not retried:
                retried = True
                continue
            if response.is_error:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"served CLM failed (HTTP {response.status_code}): {response.text[:200].strip()}",
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served CLM returned a malformed body"
                ) from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
                raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="served CLM returned no answers object")
            return payload, ms, dict(response.headers)

    def _no_answer(self, cause: Exception) -> Exception:
        return build_error(
            StatusCode.MODEL_CALL_FAILED,
            cause=cause,
            error_msg=f"no answer from served CLM at {self.url} within {self._timeout_s:g} s; {_NOT_UP}",
        )

    async def close(self) -> None:
        await self._client.aclose()


class ClmModel(DecisionModel):
    """CLM behind ``/v1/systemone``: choice and noul over HTTP, with who answered in every record."""

    name = "clm"
    deterministic = True  # the engine's decision is stable to six decimals; a re-ask only pays the encoder again
    # CLM's usage.input_tokens counts the encoder cache misses a decision paid for, not the tokens the request
    # carried, so it is not a Jev-priced quantity and must not be billed as one.
    bills_input_tokens = False

    def __init__(self, client: ClmClient, *, model: str = DEFAULT_CLM_MODEL) -> None:
        self._client = client
        self._model = model
        self._served: Json = {}
        self._served_at: str | None = None
        self._warmed = False

    @property
    def model(self) -> str:
        return self._model

    async def warm(self) -> None:
        self._served = await self._client.warm()
        self._served_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._warmed = True

    def _served_by(self) -> Json:
        """Who answered, from the ``/health`` reading: the URL asked, the names served, whether the engine has its
        encoder, and the device its vector cache sits on. What the guidance means by worker/frontend provenance.

        The reading is taken once per model and then kept, so ``read_at`` says how old it is: what is behind the URL
        can be restarted mid-run, and this record would not notice. The sibling backend caches per decision and
        keeps the previous reading when a fresh one fails, which is why it logs that; here one read is the design.
        """
        cache = self._served.get("cache")
        return {
            "url": self._client.url,
            "models": self._served.get("models"),
            "embedder": self._served.get("embedder"),
            "device": cache.get("device") if isinstance(cache, dict) else None,
            "source": "health",
            "read_at": self._served_at,
        }

    async def _warm_once(self) -> None:
        """The agent fronts call ``warm()``; ``decide`` and ``probe`` do not, and their record should still say what
        answered. A failure here is the decision's to report, not this read's, so it is logged and dropped."""
        if self._warmed:
            return
        try:
            await self.warm()
        except BaseError as exc:
            logger.warning("[clm] could not read /health at %s: %s", self._client.url, exc)

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        body = {
            "model": self._model,
            "state": observation.state,
            "questions": {name: clm_question(question) for name, question in questions.items()},
        }
        await self._warm_once()
        request_id = uuid.uuid4().hex
        payload, ms, headers = await self._client.decide(body, request_id)
        answers = payload["answers"]
        lower = {key.lower(): value for key, value in headers.items()}
        return Reply(
            answers={name: answers[name] for name in questions if name in answers},
            latency_ms=ms,
            usage=Usage.from_payload(payload.get("usage")),
            model=str(payload.get("model") or self._model),
            raw={
                **payload,
                "url": self._client.url,
                "request_id": request_id,
                "served_by": self._served_by(),
                "server_timing": server_timing(lower),
            },
        )

    async def close(self) -> None:
        await self._client.close()

    @classmethod
    def from_env(cls) -> "ClmModel":
        """``CLM_URL`` (required), ``CLM_MODEL``, ``CLM_API_KEY``, ``CLM_TIMEOUT_S``. No cloud key is read."""
        url = os.getenv("CLM_URL")
        if not url:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="--model clm needs CLM_URL, e.g. http://127.0.0.1:8091 (clm-serve)",
            )
        try:
            timeout_s = float(os.getenv("CLM_TIMEOUT_S") or CLM_TIMEOUT_S)
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg="CLM_TIMEOUT_S must be a number"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="CLM_TIMEOUT_S must be a finite number above 0",
            )
        client = ClmClient(url=url, api_key=os.getenv("CLM_API_KEY") or None, timeout_s=timeout_s)
        return cls(client, model=os.getenv("CLM_MODEL") or DEFAULT_CLM_MODEL)
