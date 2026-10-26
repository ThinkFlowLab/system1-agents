# coding: utf-8
"""TypeSafe Jev behind a ``JevTransport``: the wire body the repo has always sent, the answers as they came."""

from __future__ import annotations

import base64
import io
import json
import uuid
from typing import cast

from PIL import Image as PillowImage

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.base import DecisionModel, JevImageTransport, JevTransport
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
from s1a.decision_models.wire import (
    JEV_IMAGES_HEADER,
    client_from_env,
    decisions_backend_from_env,
    decisions_timeout_from_env,
)

# TypeSafe's documented image limits (docs.typesafe.ai/go/images, "dreamer-ready", may change between releases).
# An image outside them is rejected by the API, never resized, so we resize or reject before sending.
JEV_IMAGE_MIN_SIDE = 280
JEV_IMAGE_MAX_SIDE = 1568
JEV_IMAGE_MAX_BYTES = 16 * 1024 * 1024
JEV_MAX_BODY_BYTES = 32 * 1024 * 1024
SCREENSHOT_KEY = "screenshot"
_FORMATS = {"image/png": ("PNG", "png"), "image/jpeg": ("JPEG", "jpg")}


def jev_question(question: Question) -> Json:
    """One question in the decisions request shape.

    A choice question sends ``goal``, ``operation`` and ``rules`` only when set, one rule as a string and several as
    a list, so the bodies match what every front sent before the layer existed.
    """
    match question:
        case ChoiceQuestion():
            instructions: Json = {}
            if question.goal:
                instructions["goal"] = question.goal
            if question.operation:
                instructions["operation"] = question.operation
            if question.rules:
                instructions["rules"] = question.rules[0] if len(question.rules) == 1 else list(question.rules)
            return {"type": "choice", "criteria": dict(question.options), "instructions": instructions}
        case NoulQuestion():
            criteria = {"criteria": dict(question.criteria)} if question.criteria else {}
            return {"type": "noul", **criteria, "instructions": {"question": question.question}}
    raise TypeError(f"not a question: {question!r}")


def jev_image_asset(image: Image) -> tuple[Json, Json]:
    """One screenshot as a TypeSafe asset, plus what was sent (for step records).

    PNG or JPEG still only. The longest side is scaled down to ``JEV_IMAGE_MAX_SIDE`` with the aspect ratio kept
    (re-encoded as PNG); an image whose shorter side is then under ``JEV_IMAGE_MIN_SIDE``, or whose file is over
    ``JEV_IMAGE_MAX_BYTES``, is a config error, so nothing the API would reject is sent.
    """
    if image.media_type not in _FORMATS or not image.data:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR,
            error_msg=f"hosted Jev reads PNG or JPEG screenshots, not {image.media_type!r}",
        )
    pillow_format, wire_format = _FORMATS[image.media_type]
    data = image.data
    try:
        with PillowImage.open(io.BytesIO(data)) as picture:
            if (
                picture.format != pillow_format
                or getattr(picture, "n_frames", 1) != 1
                or getattr(picture, "custom_mimetype", None) == "image/apng"
            ):
                raise ValueError("not a still image matching its media type")
            picture.load()
            width, height = picture.size
            resized = max(width, height) > JEV_IMAGE_MAX_SIDE
            if resized:
                scale = JEV_IMAGE_MAX_SIDE / max(width, height)
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                converted = picture if picture.mode in {"RGB", "RGBA", "L", "LA"} else picture.convert("RGBA")
                out = io.BytesIO()
                converted.resize(size, PillowImage.Resampling.LANCZOS).save(out, format="PNG")
                data, wire_format = out.getvalue(), "png"
                width, height = size
    except (OSError, ValueError, SyntaxError, PillowImage.DecompressionBombError) as exc:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg=f"invalid screenshot for hosted Jev: {exc}"
        ) from exc
    if min(width, height) < JEV_IMAGE_MIN_SIDE:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR,
            error_msg=(
                f"screenshot is {width}x{height} after resizing; hosted Jev rejects a side under "
                f"{JEV_IMAGE_MIN_SIDE} px"
            ),
        )
    if len(data) > JEV_IMAGE_MAX_BYTES:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR,
            error_msg=f"screenshot is {len(data)} bytes; hosted Jev takes at most {JEV_IMAGE_MAX_BYTES} per image",
        )
    asset = {"type": "image", "format": wire_format, "data_base64": base64.b64encode(data).decode("ascii")}
    sent = {"width": width, "height": height, "bytes": len(data), "format": wire_format, "resized": resized}
    return asset, sent


def _with_screenshot(state: Json | str, reference: str) -> Json:
    """The state with the screenshot's reference under ``screenshot`` (once, so every question reads it)."""
    if isinstance(state, str):
        return {"text": state, SCREENSHOT_KEY: reference}
    if SCREENSHOT_KEY in state:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR,
            error_msg=f"the observation state already has a {SCREENSHOT_KEY!r} field; cannot attach the screenshot",
        )
    return {**state, SCREENSHOT_KEY: reference}


class JevModel(DecisionModel):
    """TypeSafe Jev over HTTP; the transport owns the connection, its retries and the round-trip clock."""

    name = "jev"
    bills_input_tokens = True

    def __init__(self, transport: JevTransport) -> None:
        self._transport = transport
        # experimental: only a direct TypeSafe client built with S1A_JEV_IMAGES on carries screenshots (#97)
        self.supports_images = bool(getattr(transport, "images", False))

    @property
    def model(self) -> str:
        return self._transport.model

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        body = {
            "model": self.model,
            "state": observation.state,
            "questions": {name: jev_question(question) for name, question in questions.items()},
        }
        sent_image: Json | None = None
        if self.supports_images and observation.images:
            if len(observation.images) != 1:
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED,
                    error_msg=f"hosted Jev takes one current screenshot, got {len(observation.images)}",
                )
            asset, sent_image = jev_image_asset(observation.images[0])
            reference = f"screenshot-{uuid.uuid4()}"  # fresh per request, so it cannot match any text by accident
            body["state"] = _with_screenshot(observation.state, reference)
            body["assets"] = {reference: asset}
            size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
            if size > JEV_MAX_BODY_BYTES:
                raise build_error(
                    StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                    error_msg=f"request is {size} bytes with the screenshot; TypeSafe takes at most {JEV_MAX_BODY_BYTES}",
                )
            transport = cast(JevImageTransport, self._transport)  # supports_images implies transport.images
            payload, ms = await transport.decide(body, headers={JEV_IMAGES_HEADER: "true"})
        else:
            payload, ms = await self._transport.decide(body)
        fields = payload if isinstance(payload, dict) else {}
        answers = fields.get("answers")
        answers = answers if isinstance(answers, dict) else {}
        return Reply(
            answers={name: answers[name] for name in questions if name in answers},
            latency_ms=ms,
            usage=Usage.from_payload(fields.get("usage")),
            model=str(fields.get("model") or self.model),
            raw={
                **fields,
                "requested_model": self.model,  # which Jev was asked for, next to the one that answered
                **({"sent_image": sent_image} if sent_image is not None else {}),
            },
        )

    async def warm(self) -> None:
        await self._transport.warm()

    async def close(self) -> None:
        await self._transport.close()

    @classmethod
    def from_env(cls, *, timeout_s: float | None = None) -> "JevModel":
        """TypeSafe directly or the OpenRouter proxy, whichever the environment names; the decision deadline is
        ``timeout_s`` when given, else ``S1A_DECISION_TIMEOUT_S``, else 5 s."""
        deadline = timeout_s if timeout_s is not None else decisions_timeout_from_env()
        return cls(client_from_env(decisions_backend_from_env(), timeout_s=deadline))
