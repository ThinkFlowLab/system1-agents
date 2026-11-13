# coding: utf-8
"""Cua-S1 4B 0.2 multimodal served over HTTP by system1-omni's worker, behind the decision-model interface.

The worker (system1-omni ``src/frontend/cua_s1.py``, CUDA only) answers ``POST /v1/systemone`` with a body of
``model``, ``state: {"image": <PNG/JPEG data URL>}`` and up to eight ``choice`` questions, and ``GET /health`` with its
modality. The desktop client stays on the machine that owns the window: it sends the screenshot it observed, gets a
probability per offered key back, and the environment clicks with the same capture it took that screenshot from.

``ServedLayaClient`` carries the transport (one deadline, one retry, the error mapping); this module builds the body,
checks it against the worker's limits before sending, validates the answers and records who answered. Image bytes
never enter the reply or the step records: only the screenshot's digest and size do.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.cua import cua_context
from s1a.decision_models.served import HEALTH_MAX_AGE_S, ServedLayaClient, parse_server_timing
from s1a.decision_models.types import ChoiceQuestion, Image, Json, Observation, Question, Reply, Usage

CUA_SERVED_MODEL = "cua-s1-4b-0.2"  # the only ``model`` the worker accepts
CUA_SERVED_TIMEOUT_S = 15.0  # one 4B forward pass per question on the worker's GPU, plus one retry
MAX_QUESTIONS = 8
MAX_OPTIONS = 26
MAX_TEXT = 16384  # characters per question: instructions plus the escaped option labels
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 1024 * 1024
MAX_SIDE = 2048
MEDIA_TYPES = {"image/png": "PNG", "image/jpeg": "JPEG"}
# The worker rejects text holding Qwen's vision control tokens; a page or window title may contain them verbatim.
MEDIA_TOKENS = ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
_NOT_UP = (
    "the worker listens only once it has loaded and warmed up, so it may still be starting; "
    "start system1-omni's src/frontend/cua_s1.py on a CUDA host and point CUA_S1_URL at it"
)
_SLOW = "the worker answers one request at a time: send again, or raise CUA_S1_TIMEOUT_S"


def _clean(text: str) -> str:
    for token in MEDIA_TOKENS:
        text = text.replace(token, "")
    return text


def _escaped_len(text: str) -> int:
    """The length the worker counts for an option label: its JSON-escaped form."""
    return len(json.dumps(text, ensure_ascii=False)) - 2


def _bad_image(reason: str) -> Exception:
    return build_error(
        StatusCode.MODEL_CALL_FAILED, error_msg=f"served Cua-S1 4B cannot read this screenshot: {reason}"
    )


def screenshot_payload(image: Image) -> tuple[str, Json]:
    """The data URL the worker reads and what the record keeps instead of the bytes.

    A screenshot above the worker's 1,048,576 pixels or 2,048 per side is scaled down, keeping its aspect ratio, and
    sent as PNG. Pixel targets are fractions of the capture, so the selected point does not move.
    """
    from PIL import Image as PILImage, UnidentifiedImageError

    if image.media_type not in MEDIA_TYPES:
        raise _bad_image(f"media type {image.media_type!r} is not PNG or JPEG")
    record: Json = {"sha256": hashlib.sha256(image.data).hexdigest(), "media_type": image.media_type}
    try:
        with PILImage.open(io.BytesIO(image.data)) as source:
            if source.format != MEDIA_TYPES[image.media_type]:
                raise _bad_image(f"the bytes are {source.format}, not {MEDIA_TYPES[image.media_type]}")
            if getattr(source, "n_frames", 1) != 1:
                raise _bad_image("it has more than one frame")
            width, height = source.size
            record["size"] = [width, height]
            scale = min(1.0, MAX_SIDE / max(width, height), math.sqrt(MAX_PIXELS / (width * height)))
            if scale >= 1.0:
                data, media_type = image.data, image.media_type
            else:
                sent = (max(1, math.floor(width * scale)), max(1, math.floor(height * scale)))
                buffer = io.BytesIO()
                source.convert("RGB").resize(sent, PILImage.Resampling.LANCZOS).save(buffer, format="PNG")
                data, media_type = buffer.getvalue(), "image/png"
                record["sent_size"] = list(sent)
    except (UnidentifiedImageError, OSError, ValueError, PILImage.DecompressionBombError) as exc:
        raise _bad_image(f"not a readable image ({exc})") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise _bad_image(f"{len(data)} bytes, above the worker's {MAX_IMAGE_BYTES}")
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}", record


def cua_served_question(observation: Observation, question: ChoiceQuestion, name: str) -> Json:
    """One worker question: the in-process adapter's goal, state and rules as ``instructions``, a label per key.

    The worker counts at most 16,384 characters per question, labels included; the instructions are cut from the end,
    which drops the rules before the state, as Nano's byte window does.
    """
    if len(question.options) > MAX_OPTIONS:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR,
            error_msg=f"served Cua-S1 4B supports at most {MAX_OPTIONS} options, not {len(question.options)} ({name!r})",
        )
    criteria: dict[str, str | None] = {}
    for key, description in question.options.items():
        text = description if isinstance(description, str) else json.dumps(description, ensure_ascii=False)
        criteria[key] = _clean(text) or None  # None: the worker shows the key itself
    labels = sum(_escaped_len(label if label is not None else key) for key, label in criteria.items())
    room = MAX_TEXT - labels
    if room <= 0 or any(len(key) > 256 for key in criteria):
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR,
            error_msg=f"the options of {name!r} do not fit served Cua-S1 4B's {MAX_TEXT} characters or 256-character keys",
        )
    instructions = _clean(cua_context(observation, question, name))[:room]
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def cua_served_answer(answer: Any, question: ChoiceQuestion, name: str) -> Json:
    """The worker's answer as the dict ``validate_answers`` reads. It must cover exactly the offered keys with finite
    probabilities summing to one; the choice is re-derived as their peak, never taken on trust."""
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict) or set(probabilities) != set(question.options):
        raise build_error(
            StatusCode.MODEL_CALL_FAILED,
            error_msg=f"served Cua-S1 4B answered {name!r} without one probability per offered key",
        )
    values = {key: probabilities[key] for key in question.options}
    if not all(
        isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1
        for p in values.values()
    ) or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-3):
        raise build_error(
            StatusCode.MODEL_CALL_FAILED, error_msg=f"served Cua-S1 4B returned invalid probabilities for {name!r}"
        )
    peak = max(values, key=values.__getitem__)
    if answer.get("choice") != peak:
        raise build_error(
            StatusCode.MODEL_CALL_FAILED,
            error_msg=f"served Cua-S1 4B chose {answer.get('choice')!r} for {name!r} but its peak is {peak!r}",
        )
    confidence = answer.get("confidence")
    confidence = (
        float(confidence) if isinstance(confidence, (int, float)) and math.isfinite(confidence) else values[peak]
    )
    return {
        "choice": peak,
        "probabilities": {k: float(p) for k, p in values.items()},
        "confidence": max(0.0, min(1.0, confidence)),
    }


class ServedCuaFourBModel(DecisionModel):
    """Cua-S1 4B 0.2 multimodal behind ``/v1/systemone``: choice questions over the observation's one screenshot."""

    name = "cua"
    supports_images = True
    question_types = frozenset({"choice"})
    deterministic = True
    bills_input_tokens = False  # the client's own GPU, not Jev's pricing

    def __init__(self, client: ServedLayaClient, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._client = client
        self._clock = clock
        self._health: Json = {}
        self._health_read_at: str | None = None
        self._health_tried: float | None = None

    @property
    def model(self) -> str:
        return CUA_SERVED_MODEL

    async def _read_health(self, *, strict: bool, deadline: float | None = None) -> None:
        self._health_tried = self._clock()
        try:
            health = await self._client.health(deadline)
        except Exception:
            if strict:
                raise
            return
        if health:
            modality = health.get("modality")
            if modality is not None and modality != "multimodal":
                raise build_error(
                    StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                    error_msg=f"the worker at {self._client.url} serves the {modality!r} adapter; "
                    "screenshot decisions need system1-omni's multimodal worker (src/frontend/cua_s1.py)",
                )
            self._health = health
            self._health_read_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    async def warm(self) -> None:
        """Read ``/health`` once: fails early when no worker is up or it serves the text adapter. Model loading stays
        on the worker, before it listens, so it never counts against an episode."""
        await self._read_health(strict=True)

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        if len(questions) > MAX_QUESTIONS:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"served Cua-S1 4B answers at most {MAX_QUESTIONS} questions per request, not {len(questions)}",
            )
        asked: dict[str, Json] = {}
        for name, question in questions.items():
            assert isinstance(question, ChoiceQuestion)
            asked[name] = cua_served_question(observation, question, name)
        if len(observation.images) != 1:
            # a call failure, not a config error: the usual cause is a screenshot that failed to capture
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                error_msg=f"served Cua-S1 4B decides over exactly one screenshot; this observation carries "
                f"{len(observation.images)}",
            )
        url, image = screenshot_payload(observation.images[0])
        deadline = self._client.deadline()  # one budget for the identity refresh and the decision
        if self._health_tried is None or self._clock() - self._health_tried >= HEALTH_MAX_AGE_S:
            await self._read_health(strict=False, deadline=deadline)
        body = {"model": CUA_SERVED_MODEL, "state": {"image": url}, "questions": asked}
        request_id = uuid.uuid4().hex
        payload, headers, ms = await self._client.decide(body, request_id, deadline)
        answers = payload["answers"]
        typed: dict[str, Json] = {}
        for name, question in questions.items():
            if name in answers:
                assert isinstance(question, ChoiceQuestion)
                typed[name] = cua_served_answer(answers[name], question, name)
        sent = payload.get("model")
        named = isinstance(sent, str) and bool(sent)
        identity = sent if isinstance(sent, str) and sent else CUA_SERVED_MODEL
        served_by = {
            "checkpoint": identity,
            "modality": self._health.get("modality"),
            "source": "response" if named else "request",
            "health_read_at": self._health_read_at,
        }
        lower = {key.lower(): value for key, value in headers.items()}
        return Reply(
            answers=typed,
            latency_ms=ms,
            usage=Usage.from_payload(payload.get("usage")),
            model=identity,
            raw={
                **payload,
                "served_by": served_by,
                "url": self._client.url,
                "request_id": request_id,
                "server_timing": parse_server_timing(lower.get("server-timing")),
                "image": image,
            },
        )

    async def close(self) -> None:
        await self._client.close()

    @classmethod
    def from_env(cls) -> "ServedCuaFourBModel":
        """``CUA_S1_URL`` (required), ``CUA_S1_API_KEY``, ``CUA_S1_TIMEOUT_S``. ``CUA_S1_MODALITY``, when set, must be
        ``multimodal``. No torch, no weights and no cloud key on this side."""
        url = os.getenv("CUA_S1_URL")
        if not url:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="served Cua-S1 4B needs CUA_S1_URL, e.g. http://127.0.0.1:8000 (system1-omni's worker)",
            )
        modality = os.getenv("CUA_S1_MODALITY") or "multimodal"
        if modality != "multimodal":
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"CUA_S1_URL serves screenshot decisions only; CUA_S1_MODALITY is {modality!r}, not multimodal",
            )
        try:
            timeout_s = float(os.getenv("CUA_S1_TIMEOUT_S") or CUA_SERVED_TIMEOUT_S)
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg="CUA_S1_TIMEOUT_S must be a number"
            ) from exc
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="CUA_S1_TIMEOUT_S must be a finite number above 0"
            )
        client = ServedLayaClient(
            url=url,
            api_key=os.getenv("CUA_S1_API_KEY") or None,
            timeout_s=timeout_s,
            label="served Cua-S1 4B",
            tag="cua-served",
            not_up=_NOT_UP,
            slow=_SLOW,
            token_hint="set CUA_S1_API_KEY to the token the frontend in front of the worker expects",
        )
        return cls(client)
