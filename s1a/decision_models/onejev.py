# coding: utf-8
"""OneJev served over HTTP by ``qev`` / ``flide`` (TypeSafe-compatible ``/v1/systemone`` + ``media``).

OneJev (https://github.com/OmniJev/OneJev, Apache-2.0) answers typed Choice / Noul / Score questions over
screenshots in one forward pass. The server speaks System One plus a ``media`` field; state references images
with ``<image:N>``. This adapter is the served counterpart of in-process ``omnijev``: it posts observation +
questions over HTTP and never loads weights. Start a server with ``qev serve --model OmniJev/OneJev1.6-4B``
(or ``flide serve``) and set ``ONEJEV_URL``.
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
from s1a.decision_models.omnijev import omnijev_answer, omnijev_question
from s1a.decision_models.served import parse_server_timing
from s1a.decision_models.types import Image, Json, Observation, Question, Reply, Usage

logger = logging.getLogger(__name__)

ONEJEV_TIMEOUT_S = 30.0  # one multimodal forward pass; raise for 27B or a cold load
ONEJEV_HEALTH_TIMEOUT_S = 2.0
DEFAULT_ONEJEV_MODEL = "OmniJev/OneJev1.6-4B"
_RETRIED_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError)
_RETRIED_STATUSES = frozenset({502, 504})
_NOT_UP = (
    "qev/flide listens only once the checkpoint is loaded; start it per "
    "https://github.com/OmniJev/OneJev (qev serve --model OmniJev/OneJev1.6-4B) and point ONEJEV_URL at it"
)
_MEDIA_TYPES = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp"}


def onejev_media(image: Image, index: int) -> tuple[Json, Json]:
    """One ``media`` entry (1-based ``<image:N>``) and the digest the step record keeps instead of the bytes."""
    if image.media_type not in _MEDIA_TYPES:
        raise build_error(
            StatusCode.MODEL_CALL_FAILED,
            error_msg=f"onejev cannot send media type {image.media_type!r}; use PNG, JPEG or WebP",
        )
    if not image.data:
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="onejev cannot send an empty screenshot")
    data_uri = f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}"
    record: Json = {
        "index": index,
        "sha256": hashlib.sha256(image.data).hexdigest(),
        "media_type": image.media_type,
        "bytes": len(image.data),
    }
    return {"type": "image", "data": data_uri}, record


def onejev_state(observation: Observation) -> Json:
    """Observation state with ``<image:N>`` markers for every screenshot (OneJev's documented convention)."""
    markers = {f"image_{i}": f"<image:{i}>" for i in range(1, len(observation.images) + 1)}
    if len(observation.images) == 1:
        markers = {"screen": "<image:1>"}
    if isinstance(observation.state, dict):
        return {**observation.state, **markers}
    return {"text": observation.state, **markers}


def onejev_question(question: Question) -> Json:
    """Choice / noul in OneJev's TypeSafe shape: string instructions, flattened option labels.

    The observation's text state travels in ``state`` with the image markers; instructions carry the goal, rules
    and ask (OmniJev's short prompt), so the screenshot is not duplicated as prose.
    """
    return omnijev_question(question, context="", prompt="short")


class OneJevClient:
    """HTTP to one ``qev`` / ``flide`` ``/v1/systemone`` server: one deadline per decision, one transport retry."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        timeout_s: float = ONEJEV_TIMEOUT_S,
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
        """``GET /health`` once: fail early when nothing is listening."""
        try:
            response = await asyncio.wait_for(
                self._client.get(f"{self.url}/health", timeout=ONEJEV_HEALTH_TIMEOUT_S), ONEJEV_HEALTH_TIMEOUT_S
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"no served OneJev at {self.url}: {_NOT_UP}"
            ) from exc
        except (TimeoutError, httpx.HTTPError) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"served OneJev at {self.url} did not answer: {exc}"
            ) from exc
        if response.is_error:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                error_msg=f"served OneJev at {self.url} returned HTTP {response.status_code}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served OneJev returned a malformed body"
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
            post = self._client.post(f"{self.url}/v1/systemone", json=body, headers=headers, timeout=remaining)
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
                        error_msg=f"no served OneJev at {self.url}: {_NOT_UP}",
                    ) from exc
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    cause=exc,
                    error_msg=f"served OneJev unreachable at {self.url}: {exc}",
                ) from exc
            except httpx.HTTPError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    cause=exc,
                    error_msg=f"served OneJev call failed at {self.url}: {exc}",
                ) from exc
            ms = round((time.perf_counter() - started) * 1000)
            if response.status_code in _RETRIED_STATUSES and not retried:
                retried = True
                continue
            if response.is_error:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"served OneJev failed (HTTP {response.status_code}): {response.text[:200].strip()}",
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served OneJev returned a malformed body"
                ) from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
                raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="served OneJev returned no answers object")
            return payload, ms, dict(response.headers)

    def _no_answer(self, cause: Exception | None) -> Exception:
        return build_error(
            StatusCode.MODEL_CALL_FAILED,
            cause=cause,
            error_msg=f"no answer from served OneJev at {self.url} within {self._timeout_s:g} s; {_NOT_UP}",
        )

    async def close(self) -> None:
        await self._client.aclose()


class OneJevModel(DecisionModel):
    """OneJev behind ``/v1/systemone`` + ``media``: choice and noul over the observation's screenshot(s)."""

    name = "onejev"
    supports_images = True
    deterministic = True
    bills_input_tokens = False  # the client's own server, not Jev's pricing

    def __init__(self, client: OneJevClient, *, model: str = DEFAULT_ONEJEV_MODEL) -> None:
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
            "checkpoint": self._served.get("model") or self._served.get("checkpoint") or self._model,
            "device": self._served.get("device"),
            "source": "health",
            "read_at": self._served_at,
        }

    async def _warm_once(self) -> None:
        if self._warmed:
            return
        try:
            await self.warm()
        except BaseError as exc:
            logger.warning("[onejev] could not read /health at %s: %s", self._client.url, exc)

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        if not observation.images:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                error_msg="onejev decides over a screenshot; this observation carries no image",
            )
        media: list[Json] = []
        records: list[Json] = []
        for index, image in enumerate(observation.images, start=1):
            entry, record = onejev_media(image, index)
            media.append(entry)
            records.append(record)
        body: Json = {
            "model": self._model,
            "state": onejev_state(observation),
            "media": media,
            "questions": {name: onejev_question(question) for name, question in questions.items()},
        }
        await self._warm_once()
        request_id = uuid.uuid4().hex
        payload, ms, headers = await self._client.decide(body, request_id)
        answers = payload["answers"]
        mapped = {name: omnijev_answer(answers[name]) for name in questions if name in answers}
        lower = {key.lower(): value for key, value in headers.items()}
        return Reply(
            answers=mapped,
            latency_ms=ms,
            usage=Usage.from_payload(payload.get("usage")),
            model=str(payload.get("model") or self._model),
            raw={
                **payload,
                "url": self._client.url,
                "request_id": request_id,
                "served_by": self._served_by(),
                "server_timing": parse_server_timing(lower.get("server-timing")),
                "sent_images": records,
            },
        )

    async def close(self) -> None:
        await self._client.close()

    @classmethod
    def from_env(cls) -> "OneJevModel":
        """``ONEJEV_URL`` (required), ``ONEJEV_MODEL``, ``ONEJEV_API_KEY``, ``ONEJEV_TIMEOUT_S``. No cloud key."""
        url = os.getenv("ONEJEV_URL")
        if not url:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="--model onejev needs ONEJEV_URL, e.g. http://127.0.0.1:8000 (qev/flide serve)",
            )
        try:
            timeout_s = float(os.getenv("ONEJEV_TIMEOUT_S") or ONEJEV_TIMEOUT_S)
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg="ONEJEV_TIMEOUT_S must be a number"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="ONEJEV_TIMEOUT_S must be a finite number above 0",
            )
        client = OneJevClient(url=url, api_key=os.getenv("ONEJEV_API_KEY") or None, timeout_s=timeout_s)
        return cls(client, model=os.getenv("ONEJEV_MODEL") or DEFAULT_ONEJEV_MODEL)
