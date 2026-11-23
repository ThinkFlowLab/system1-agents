# coding: utf-8
"""Clef / Clef-flash served over HTTP: Workers AI or a local ``/v1/systemone`` server.

Clef and Clef-flash (https://huggingface.co/Cloudflare/clef, Apache-2.0) are multimodal System One decision
models. Workers AI is ``POST /client/v4/accounts/$ACCOUNT_ID/ai/run/@cf/cloudflare/clef`` (or ``clef-flash``)
with a Cloudflare token; a local SGLang server (``lmsysorg/sglang:dev-clef``) speaks ``POST /v1/systemone``.
Screenshots travel in Clef's ``images`` array (PNG/JPEG/WebP data URLs, max 4). An observation without an image
runs as text. This adapter never loads weights. ``score`` questions and ``videos`` are out of scope: Agents has
no ``ScoreQuestion`` or video observation, and the Workers AI Clef / Clef-flash input schema has no ``videos``
field (that extension is documented on Clef-omni).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
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
from s1a.decision_models.omnijev import omnijev_option
from s1a.decision_models.served import parse_server_timing
from s1a.decision_models.types import (
    ChoiceQuestion,
    Image,
    Json,
    NoulQuestion,
    Observation,
    Question,
    Reply,
    Usage,
)

logger = logging.getLogger(__name__)

CLEF_MODELS = ("clef", "clef-flash")
DEFAULT_CLEF_MODEL = "clef-flash"  # the 9B variant; set CLEF_MODEL=clef for the 27B model
CLEF_TIMEOUT_S = 30.0
CLEF_HEALTH_TIMEOUT_S = 2.0
MAX_CLEF_IMAGES = 4  # Workers AI schema: images.maxItems = 4
WORKERS_AI_RUN = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"
_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
_RETRIED_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError)
_RETRIED_STATUSES = frozenset({502, 504})
_NOT_UP_LOCAL = (
    "a Clef server listens at /v1/systemone once the checkpoint is loaded; start it per "
    "https://huggingface.co/Cloudflare/clef (lmsysorg/sglang:dev-clef) and point CLEF_URL at it"
)
_NOT_UP_WORKERS = (
    "Workers AI needs a token with Workers AI permission; see https://developers.cloudflare.com/workers-ai/models/clef/"
)


def workers_ai_url(account_id: str, model: str) -> str:
    """The official Workers AI run URL for ``clef`` or ``clef-flash``."""
    if model not in CLEF_MODELS:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR,
            error_msg=f"CLEF_MODEL must be clef or clef-flash, not {model!r}",
        )
    return WORKERS_AI_RUN.format(account=account_id, model=model)


def clef_image(image: Image) -> tuple[str, Json]:
    """One Clef ``images`` data URL and the digest the step record keeps instead of the bytes.

    Official Workers AI schema: ``data:image/png;base64,...`` (or jpeg / webp). Remote URLs are not accepted.
    """
    if image.media_type not in _MEDIA_TYPES:
        raise build_error(
            StatusCode.MODEL_CALL_FAILED,
            error_msg=f"clef cannot send media type {image.media_type!r}; use PNG, JPEG or WebP",
        )
    if not image.data:
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="clef cannot send an empty screenshot")
    data_uri = f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}"
    record: Json = {
        "sha256": hashlib.sha256(image.data).hexdigest(),
        "media_type": image.media_type,
        "bytes": len(image.data),
    }
    return data_uri, record


def clef_images(images: tuple[Image, ...]) -> tuple[list[str], list[Json]]:
    """Every screenshot as a data URL, or a call failure past the Workers AI max of four."""
    if len(images) > MAX_CLEF_IMAGES:
        raise build_error(
            StatusCode.MODEL_CALL_FAILED,
            error_msg=f"clef accepts at most {MAX_CLEF_IMAGES} images per the Workers AI schema, not {len(images)}",
        )
    urls: list[str] = []
    records: list[Json] = []
    for image in images:
        url, record = clef_image(image)
        urls.append(url)
        records.append(record)
    return urls, records


def clef_question(question: Question) -> Json:
    """Choice / noul in Clef's System One shape: string instructions, flattened option labels.

    Clef's Workers AI schema requires ``instructions`` and, for a choice, ``criteria``. Browser element rows become
    their label and value (the same flattening ``omnijev_option`` already does); we do not send ``score`` questions
    or a ``videos`` field.
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


def clef_payload(body: object) -> Json:
    """The System One decision object: used as-is, or unwrapped from the Workers AI REST envelope.

    Local ``/v1/systemone`` returns ``{model, answers, usage}``. The Workers AI REST API wraps that as
    ``{success, result, errors, messages}`` (https://developers.cloudflare.com/workers-ai/get-started/rest-api/).
    Missing or malformed answers fail closed; a ``success: false`` envelope surfaces the Cloudflare errors.
    """
    if not isinstance(body, dict):
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="clef returned a malformed body")
    if isinstance(body.get("answers"), dict):
        return body
    result = body.get("result")
    if isinstance(result, dict) and isinstance(result.get("answers"), dict):
        return result
    if body.get("success") is False:
        errors = body.get("errors")
        detail = errors if errors else body
        raise build_error(
            StatusCode.MODEL_CALL_FAILED,
            error_msg=f"Workers AI rejected the Clef request: {detail!r}"[:400],
        )
    raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="clef returned no answers object")


class ClefClient:
    """HTTP to one Clef endpoint: one deadline per decision, one transport / 502 / 504 retry."""

    def __init__(
        self,
        *,
        url: str,
        endpoint: str,
        api_key: str | None = None,
        timeout_s: float = CLEF_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        health_url: str | None = None,
        path: str = "local",
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"timeout_s must be a finite number above 0, not {timeout_s!r}",
            )
        self.url = url.rstrip("/")
        self.endpoint = endpoint
        self.health_url = health_url
        self.path = path
        self._timeout_s = timeout_s
        self._clock = clock
        self._not_up = _NOT_UP_WORKERS if path == "workers-ai" else _NOT_UP_LOCAL
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers, transport=transport)

    async def warm(self) -> Json:
        """``GET /health`` on a local server; Workers AI has no health route, so this is a no-op there."""
        if not self.health_url:
            return {}
        try:
            response = await asyncio.wait_for(
                self._client.get(self.health_url, timeout=CLEF_HEALTH_TIMEOUT_S), CLEF_HEALTH_TIMEOUT_S
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"no served Clef at {self.url}: {self._not_up}"
            ) from exc
        except (TimeoutError, httpx.HTTPError) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"served Clef at {self.url} did not answer: {exc}"
            ) from exc
        if response.is_error:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                error_msg=f"served Clef at {self.url} returned HTTP {response.status_code}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served Clef returned a malformed body"
            ) from exc
        return body if isinstance(body, dict) else {}

    async def decide(self, body: Json, request_id: str) -> tuple[Json, int, dict[str, str]]:
        """One decision: the unwrapped payload, its round trip in ms and the response headers."""
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
                        error_msg=f"no served Clef at {self.url}: {self._not_up}",
                    ) from exc
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    cause=exc,
                    error_msg=f"served Clef unreachable at {self.url}: {exc}",
                ) from exc
            except httpx.HTTPError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    cause=exc,
                    error_msg=f"served Clef call failed at {self.url}: {exc}",
                ) from exc
            ms = round((time.perf_counter() - started) * 1000)
            if response.status_code in _RETRIED_STATUSES and not retried:
                retried = True
                continue
            if response.is_error:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"served Clef failed (HTTP {response.status_code}): {response.text[:200].strip()}",
                )
            try:
                raw = response.json()
            except ValueError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served Clef returned a malformed body"
                ) from exc
            return clef_payload(raw), ms, dict(response.headers)

    def _no_answer(self, cause: Exception | None) -> Exception:
        return build_error(
            StatusCode.MODEL_CALL_FAILED,
            cause=cause,
            error_msg=f"no answer from served Clef at {self.url} within {self._timeout_s:g} s; {self._not_up}",
        )

    async def close(self) -> None:
        await self._client.aclose()


class ClefModel(DecisionModel):
    """Clef behind Workers AI or ``/v1/systemone``: choice and noul; images when the observation has them."""

    name = "clef"
    supports_images = True
    deterministic = True
    bills_input_tokens = False  # do not price Clef at Jev's rate; Workers AI has its own list price

    def __init__(self, client: ClefClient, *, model: str = DEFAULT_CLEF_MODEL) -> None:
        if model not in CLEF_MODELS:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"CLEF_MODEL must be clef or clef-flash, not {model!r}",
            )
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
        return {
            "url": self._client.url,
            "path": self._client.path,
            "checkpoint": self._served.get("model") or self._served.get("checkpoint") or self._model,
            "device": self._served.get("device"),
            "source": "health" if self._client.health_url else self._client.path,
            "read_at": self._served_at,
        }

    async def _warm_once(self) -> None:
        if self._warmed or not self._client.health_url:
            return
        try:
            await self.warm()
        except BaseError as exc:
            logger.warning("[clef] could not read /health at %s: %s", self._client.url, exc)

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        body: Json = {
            "model": self._model,
            "state": observation.state,
            "questions": {name: clef_question(question) for name, question in questions.items()},
        }
        records: list[Json] = []
        if observation.images:
            urls, records = clef_images(observation.images)
            body["images"] = urls
        await self._warm_once()
        request_id = uuid.uuid4().hex
        payload, ms, headers = await self._client.decide(body, request_id)
        answers = payload["answers"]
        lower = {key.lower(): value for key, value in headers.items()}
        raw: Json = {
            **payload,
            "url": self._client.url,
            "request_id": request_id,
            "served_by": self._served_by(),
            "server_timing": parse_server_timing(lower.get("server-timing")),
        }
        if lower.get("cf-ray"):
            raw["cf_ray"] = lower["cf-ray"]
        if records:
            raw["sent_images"] = records
        return Reply(
            answers={name: answers[name] for name in questions if name in answers},
            latency_ms=ms,
            usage=Usage.from_payload(payload.get("usage")),
            model=str(payload.get("model") or self._model),
            raw=raw,
        )

    async def close(self) -> None:
        await self._client.close()

    @classmethod
    def from_env(cls) -> "ClefModel":
        """``CLEF_URL`` (local ``/v1/systemone``) or ``CLOUDFLARE_ACCOUNT_ID`` + ``CLOUDFLARE_AUTH_TOKEN``.

        ``CLEF_MODEL`` is ``clef`` or ``clef-flash`` (default ``clef-flash``). ``CLEF_URL`` wins when both a local
        URL and Cloudflare credentials are set. ``CLOUDFLARE_API_TOKEN`` is accepted as an alias for the official
        Clef-docs name ``CLOUDFLARE_AUTH_TOKEN``. Optional ``CLEF_API_KEY`` / ``CLEF_TIMEOUT_S`` for the local path.
        """
        model = (os.getenv("CLEF_MODEL") or DEFAULT_CLEF_MODEL).strip()
        if model not in CLEF_MODELS:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"CLEF_MODEL must be clef or clef-flash, not {model!r}",
            )
        try:
            timeout_s = float(os.getenv("CLEF_TIMEOUT_S") or CLEF_TIMEOUT_S)
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg="CLEF_TIMEOUT_S must be a number"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="CLEF_TIMEOUT_S must be a finite number above 0",
            )
        url = (os.getenv("CLEF_URL") or "").strip()
        if url:
            base = url.rstrip("/")
            client = ClefClient(
                url=base,
                endpoint=f"{base}/v1/systemone",
                api_key=os.getenv("CLEF_API_KEY") or None,
                timeout_s=timeout_s,
                health_url=f"{base}/health",
                path="local",
            )
            return cls(client, model=model)
        account = (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip()
        token = (os.getenv("CLOUDFLARE_AUTH_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
        if not account or not token:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=(
                    "--model clef needs CLEF_URL (local /v1/systemone, e.g. http://127.0.0.1:30000) or "
                    "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN (Workers AI)"
                ),
            )
        endpoint = workers_ai_url(account, model)
        client = ClefClient(
            url=endpoint,
            endpoint=endpoint,
            api_key=token,
            timeout_s=timeout_s,
            path="workers-ai",
        )
        return cls(client, model=model)
