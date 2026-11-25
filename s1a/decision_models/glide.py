# coding: utf-8
"""GLiDE served over HTTP: hosted Fastino or a local ``glide.server``.

GLiDE (https://arxiv.org/abs/2611.26509) is a family of decision models that score typed questions in one
forward pass. Hosted inference is ``POST https://api.fastino.ai/v1/systemone`` with model id ``fastino/GLiDE``;
a local server (``python -m glide.server``) speaks the same path. This adapter never loads weights. Weights are
under the Fastino Community License, not Apache-2.0, and are not vendored. ``score`` questions are out of scope:
Agents has no ``ScoreQuestion``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import httpx

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.jev import jev_question
from s1a.decision_models.omnijev import omnijev_option
from s1a.decision_models.served import parse_server_timing
from s1a.decision_models.types import ChoiceQuestion, Image, Json, Observation, Question, Reply, Usage

logger = logging.getLogger(__name__)

HOSTED_GLIDE_URL = "https://api.fastino.ai/v1/systemone"
DEFAULT_GLIDE_MODEL = "fastino/GLiDE"
GLIDE_TIMEOUT_S = 30.0
MAX_GLIDE_IMAGES = 4  # official Fastino limit
MAX_GLIDE_IMAGE_BYTES = 3_932_160  # 3.75 MiB raw; 5 MiB of base64 in a data URL
MAX_TRANSIENT_RETRIES = 5  # 425 / 429 / 503, within the decision deadline
DEFAULT_WARMUP_RETRY_S = 30.0  # Fastino's default when a 425 has no Retry-After
DEFAULT_TRANSIENT_RETRY_S = 1.0
_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_RETRIED_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError)
_RETRIED_STATUSES = frozenset({502, 504})
_TRANSIENT_STATUSES = frozenset({425, 429, 503})
_NOT_UP_HOSTED = "the Fastino GLiDE API needs a valid FASTINO_API_KEY; see https://docs.fastino.ai/inference/systemone"
_NOT_UP_LOCAL = (
    "a GLiDE server listens at /v1/systemone once the checkpoint is loaded; start it per "
    "https://huggingface.co/fastino/GLiDE-27B and point GLIDE_URL at it"
)


def retry_after_s(response: httpx.Response, *, default: float) -> float:
    """Seconds to wait before the next attempt: ``Retry-After`` as a delay, or ``default`` when missing or unusable."""
    raw = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if raw is None or not str(raw).strip():
        return default
    try:
        delay = float(str(raw).strip())
    except ValueError:
        return default
    if not math.isfinite(delay) or delay < 0:
        return default
    return delay


def glide_image(image: Image) -> tuple[Json, Json]:
    """One GLiDE ``state`` image part (inline data URL) and the digest the step record keeps instead of the bytes.

    Official Fastino schema: ``{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}``.
    Remote URLs are not used: the local server does not fetch them, and a data URL works on both paths.
    """
    if image.media_type not in _MEDIA_TYPES:
        raise build_error(
            StatusCode.MODEL_CALL_FAILED,
            error_msg=f"glide cannot send media type {image.media_type!r}; use JPEG, PNG, GIF or WebP",
        )
    if not image.data:
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="glide cannot send an empty screenshot")
    if len(image.data) > MAX_GLIDE_IMAGE_BYTES:
        raise build_error(
            StatusCode.MODEL_CALL_FAILED,
            error_msg=(
                f"glide accepts at most {MAX_GLIDE_IMAGE_BYTES} bytes per image per the Fastino limit, "
                f"not {len(image.data)}"
            ),
        )
    data_uri = f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}"
    part: Json = {"type": "image_url", "image_url": {"url": data_uri}}
    record: Json = {
        "sha256": hashlib.sha256(image.data).hexdigest(),
        "media_type": image.media_type,
        "bytes": len(image.data),
    }
    return part, record


def glide_images(images: tuple[Image, ...]) -> tuple[list[Json], list[Json]]:
    """Every screenshot as a state part, or a call failure past Fastino's max of four."""
    if len(images) > MAX_GLIDE_IMAGES:
        raise build_error(
            StatusCode.MODEL_CALL_FAILED,
            error_msg=f"glide accepts at most {MAX_GLIDE_IMAGES} images per the Fastino limit, not {len(images)}",
        )
    parts: list[Json] = []
    records: list[Json] = []
    for image in images:
        part, record = glide_image(image)
        parts.append(part)
        records.append(record)
    return parts, records


def glide_question(question: Question) -> Json:
    """Choice / noul in Fastino's System One shape, reused from ``jev_question``.

    Fastino accepts ``instructions`` as a non-empty string or a non-empty JSON object, so the existing Jev
    object (``goal`` / ``operation`` / ``rules``, or ``{question: ...}`` for noul) is sent as-is when it is
    non-empty. An empty instructions object is mapped to a short fallback string (Fastino rejects empty
    instructions with 422). Choice ``criteria`` values that are not already strings (browser element rows)
    are flattened with ``omnijev_option``, because Fastino documents criteria as option keys to descriptions.
    """
    body = jev_question(question)
    instructions = body.get("instructions")
    if instructions == {} or instructions is None or instructions == "":
        fallback = "Which option comes next?" if isinstance(question, ChoiceQuestion) else "Does this hold?"
        if isinstance(question, ChoiceQuestion) and question.goal:
            fallback = question.goal
        body["instructions"] = fallback
    criteria = body.get("criteria")
    if isinstance(criteria, dict):
        body["criteria"] = {
            key: value if isinstance(value, str) else omnijev_option(value) for key, value in criteria.items()
        }
    return body


def glide_state(observation: Observation) -> Json | str | list[Json]:
    """Text state as-is, or a part array when the observation has screenshots.

    Fastino's image layout is ``state`` parts, not Clef's ``images`` array and not OneJev's ``media``. A dict
    or other non-string state is sent as a compact-JSON text part so the array only mixes text parts with
    ``image_url`` parts, as the official docs describe.
    """
    if not observation.images:
        return observation.state
    parts, _records = glide_images(observation.images)
    state = observation.state
    if isinstance(state, str):
        text = state
    else:
        text = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    prefix: list[Json] = [{"type": "text", "text": text}] if text else []
    return [*prefix, *parts]


def glide_payload(body: object) -> Json:
    """The System One decision object, or a call failure when ``answers`` is missing or not an object."""
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="glide returned no answers object")
    return body


class GlideClient:
    """HTTP to one GLiDE endpoint: one deadline per decision, transport / 502 / 504 plus Fastino 425 / 429 / 503."""

    def __init__(
        self,
        *,
        url: str,
        endpoint: str,
        api_key: str | None = None,
        timeout_s: float = GLIDE_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        path: str = "local",
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"timeout_s must be a finite number above 0, not {timeout_s!r}",
            )
        self.url = url.rstrip("/")
        self.endpoint = endpoint
        self.path = path
        self._timeout_s = timeout_s
        self._clock = clock
        self._sleeper = sleeper or asyncio.sleep
        self._not_up = _NOT_UP_HOSTED if path == "hosted" else _NOT_UP_LOCAL
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers, transport=transport)

    async def decide(self, body: Json, request_id: str) -> tuple[Json, int, dict[str, str]]:
        """One decision: the payload, its round trip in ms and the response headers."""
        headers = {"X-Request-Id": request_id}
        deadline = self._clock() + self._timeout_s
        retried = False
        transient = 0
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
                        error_msg=f"no served GLiDE at {self.url}: {self._not_up}",
                    ) from exc
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    cause=exc,
                    error_msg=f"served GLiDE unreachable at {self.url}: {exc}",
                ) from exc
            except httpx.HTTPError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    cause=exc,
                    error_msg=f"served GLiDE call failed at {self.url}: {exc}",
                ) from exc
            ms = round((time.perf_counter() - started) * 1000)
            if response.status_code in _RETRIED_STATUSES and not retried:
                retried = True
                continue
            if response.status_code in _TRANSIENT_STATUSES and transient < MAX_TRANSIENT_RETRIES:
                default = DEFAULT_WARMUP_RETRY_S if response.status_code == 425 else DEFAULT_TRANSIENT_RETRY_S
                delay = retry_after_s(response, default=default)
                if deadline - self._clock() - delay <= 0:
                    raise build_error(
                        StatusCode.MODEL_CALL_FAILED,
                        error_msg=(f"served GLiDE failed (HTTP {response.status_code}): {response.text[:200].strip()}"),
                    )
                transient += 1
                await self._sleeper(delay)
                continue
            if response.is_error:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"served GLiDE failed (HTTP {response.status_code}): {response.text[:200].strip()}",
                )
            try:
                raw = response.json()
            except ValueError as exc:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="served GLiDE returned a malformed body"
                ) from exc
            return glide_payload(raw), ms, dict(response.headers)

    def _no_answer(self, cause: Exception | None) -> Exception:
        return build_error(
            StatusCode.MODEL_CALL_FAILED,
            cause=cause,
            error_msg=f"no answer from served GLiDE at {self.url} within {self._timeout_s:g} s; {self._not_up}",
        )

    async def close(self) -> None:
        await self._client.aclose()


class GlideModel(DecisionModel):
    """GLiDE behind Fastino or ``/v1/systemone``: choice and noul; images when the observation has them."""

    name = "glide"
    supports_images = True
    deterministic = True
    bills_input_tokens = False  # do not price GLiDE at Jev's rate; Fastino has its own list price

    def __init__(self, client: GlideClient, *, model: str = DEFAULT_GLIDE_MODEL) -> None:
        if not model.strip():
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="GLIDE_MODEL must be a non-empty model id, e.g. fastino/GLiDE",
            )
        self._client = client
        self._model = model
        self._served_at: str | None = None

    @property
    def model(self) -> str:
        return self._model

    async def warm(self) -> None:
        self._served_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _served_by(self) -> Json:
        return {
            "url": self._client.url,
            "path": self._client.path,
            "checkpoint": self._model,
            "source": self._client.path,
            "read_at": self._served_at,
        }

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        body: Json = {
            "model": self._model,
            "state": glide_state(observation),
            "questions": {name: glide_question(question) for name, question in questions.items()},
        }
        if self._client.path == "hosted":
            body["store"] = False
        records: list[Json] = []
        if observation.images:
            _parts, records = glide_images(observation.images)
        if self._served_at is None:
            await self.warm()
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
        if lower.get("x-fastino-data-retention"):
            raw["data_retention"] = lower["x-fastino-data-retention"]
        if lower.get("x-fastino-request-id"):
            raw["fastino_request_id"] = lower["x-fastino-request-id"]
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
    def from_env(cls) -> "GlideModel":
        """``GLIDE_URL`` (local ``/v1/systemone``) or ``FASTINO_API_KEY`` (hosted Fastino).

        ``GLIDE_URL`` wins when both a local URL and ``FASTINO_API_KEY`` are set. ``GLIDE_MODEL`` defaults to
        ``fastino/GLiDE``. Optional ``GLIDE_API_KEY`` for a local server that expects ``X-API-Key``.
        ``GLIDE_TIMEOUT_S`` is the per-decision deadline (default 30 s). Hosted requests send ``store: false``.
        """
        model = (os.getenv("GLIDE_MODEL") or DEFAULT_GLIDE_MODEL).strip()
        if not model:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="GLIDE_MODEL must be a non-empty model id, e.g. fastino/GLiDE",
            )
        try:
            timeout_s = float(os.getenv("GLIDE_TIMEOUT_S") or GLIDE_TIMEOUT_S)
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg="GLIDE_TIMEOUT_S must be a number"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="GLIDE_TIMEOUT_S must be a finite number above 0",
            )
        url = (os.getenv("GLIDE_URL") or "").strip()
        if url:
            base = url.rstrip("/")
            client = GlideClient(
                url=base,
                endpoint=f"{base}/v1/systemone",
                api_key=(os.getenv("GLIDE_API_KEY") or os.getenv("FASTINO_API_KEY") or "").strip() or None,
                timeout_s=timeout_s,
                path="local",
            )
            return cls(client, model=model)
        key = (os.getenv("FASTINO_API_KEY") or "").strip()
        if not key:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=(
                    "--model glide needs GLIDE_URL (local /v1/systemone, e.g. http://127.0.0.1:8000) or "
                    "FASTINO_API_KEY (hosted https://api.fastino.ai/v1/systemone)"
                ),
            )
        client = GlideClient(
            url=HOSTED_GLIDE_URL,
            endpoint=HOSTED_GLIDE_URL,
            api_key=key,
            timeout_s=timeout_s,
            path="hosted",
        )
        return cls(client, model=model)
