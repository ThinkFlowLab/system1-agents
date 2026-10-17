# coding: utf-8
"""The native JEV-27B-VL flat protocol behind Agents' decision-model interface."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import time
import uuid

import httpx
from PIL import Image as PillowImage

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.jev import jev_question
from s1a.decision_models.served import parse_server_timing
from s1a.decision_models.types import ChoiceQuestion, Json, NoulQuestion, Observation, Question, Reply, Usage
from s1a.decision_models.validation import unit_interval
from s1a.decision_models.wire import decisions_timeout_from_env

MODEL_ID = "autotrust/JEV-27B-VL"
PROTOCOL = "jev27-bare-v1"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_BODY_BYTES = 8 * 1024 * 1024


def _image_url(observation: Observation) -> str:
    if len(observation.images) != 1:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="jev-vl-served requires exactly one current PNG/JPEG image"
        )
    image = observation.images[0]
    formats = {"image/png": "PNG", "image/jpeg": "JPEG"}
    if image.media_type not in formats or not image.data or len(image.data) > MAX_IMAGE_BYTES:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR,
            error_msg="jev-vl-served requires a nonempty PNG/JPEG of at most 4 MiB",
        )
    try:
        with PillowImage.open(io.BytesIO(image.data)) as picture:
            w, h = picture.size
            if (
                picture.format != formats[image.media_type]
                or getattr(picture, "n_frames", 1) != 1
                or getattr(picture, "custom_mimetype", None) == "image/apng"
                or w > 2048
                or h > 2048
                or w * h > 1048576
                or min(w, h) <= 0
                or max(w, h) > 200 * min(w, h)
            ):
                raise ValueError("image format, frame count or dimensions exceed the native worker's limits")
            # verify checks PNG CRCs; load also checks compressed pixels before a forced decision can bypass HTTP.
            picture.verify()
        with PillowImage.open(io.BytesIO(image.data)) as picture:
            picture.load()
    except (OSError, ValueError, SyntaxError, PillowImage.DecompressionBombError) as exc:
        raise build_error(
            StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg=f"invalid JEV-VL image: {exc}"
        ) from exc
    return f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}"


def _question(question: Question) -> tuple[str, list[str]]:
    if isinstance(question, ChoiceQuestion):
        instructions = jev_question(question)["instructions"]
        text = json.dumps(instructions, ensure_ascii=False, sort_keys=True) if instructions else "Choose one option."
        options = [
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            for value in question.options.values()
        ]
        return text, options
    assert isinstance(question, NoulQuestion)
    text = question.question
    if question.criteria is not None:
        text = json.dumps({"question": text, "criteria": question.criteria}, ensure_ascii=False, sort_keys=True)
    return text, ["false", "true"]


def _answer(payload: Json, question: Question, options: list[str]) -> Json:
    for key, expected in (
        ("model", MODEL_ID),
        ("protocol", PROTOCOL),
        ("kind", question.type),
        ("effective_kind", question.type),
        ("options", options),
    ):
        if payload.get(key) != expected:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, error_msg=f"JEV-VL response {key} does not match the request"
            )
    index, probabilities = payload.get("choice_index"), payload.get("probabilities")
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(options)
        or payload.get("choice") != options[index]
    ):
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="JEV-VL returned an invalid choice index or choice")
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != len(options)
        or not all(unit_interval(p) for p in probabilities)
        or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-4)
        or probabilities[index] != max(probabilities)
    ):
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="JEV-VL returned an invalid probability distribution")
    if isinstance(question, ChoiceQuestion):
        return {
            "choice": question.ids[index],
            "probabilities": dict(zip(question.ids, probabilities)),
            "confidence": probabilities[index],
        }
    return {"noul": probabilities[1], "confidence": max(probabilities)}


class ServedJevVLModel(DecisionModel):
    """One current image, sequential flat requests and one deadline for all questions; no automatic retry."""

    name = "jev-vl-served"
    supports_images = True
    deterministic = True
    bills_input_tokens = False

    def __init__(
        self,
        *,
        url: str,
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise build_error(StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="timeout_s must be finite and positive")
        try:
            parsed = httpx.URL(url.strip())
        except httpx.InvalidURL as exc:
            raise build_error(StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg="invalid JEV_VL_URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.query or parsed.fragment:
            raise build_error(StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="JEV_VL_URL must be an HTTP(S) base URL")
        self.url = str(parsed).rstrip("/")
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    @property
    def model(self) -> str:
        return MODEL_ID

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg=f"no JEV-VL answer within {self._timeout_s:g} s")
        return remaining

    async def _post(self, body: bytes, request_id: str, deadline: float) -> tuple[Json, dict[str, str]]:
        remaining = self._remaining(deadline)
        try:
            response = await asyncio.wait_for(
                self._client.post(
                    f"{self.url}/v1/systemone",
                    content=body,
                    headers={"Content-Type": "application/json", "X-Request-Id": request_id},
                    timeout=remaining,
                ),
                remaining,
            )
        except (TimeoutError, httpx.HTTPError) as exc:
            raise build_error(StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"JEV-VL call failed: {exc}") from exc
        if response.is_error:
            code = (
                StatusCode.MODEL_SERVICE_CONFIG_ERROR if response.status_code == 401 else StatusCode.MODEL_CALL_FAILED
            )
            raise build_error(code, error_msg=f"JEV-VL HTTP {response.status_code}: {response.text[:300]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg="JEV-VL returned malformed JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="JEV-VL returned a non-object body")
        return payload, dict(response.headers)

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        started = time.monotonic()
        deadline = started + self._timeout_s
        image = _image_url(observation)
        try:
            text = (
                observation.state
                if isinstance(observation.state, str)
                else json.dumps(observation.state, ensure_ascii=False)
            )
            prepared = []
            for name, question in questions.items():
                question_text, options = _question(question)
                body = json.dumps(
                    {
                        "model": MODEL_ID,
                        "kind": question.type,
                        "state": [text, {"image": image}],
                        "question": question_text,
                        "options": options,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                if len(body) > MAX_BODY_BYTES:
                    raise build_error(
                        StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="JEV-VL request body exceeds 8 MiB"
                    )
                prepared.append((name, question, options, body))
        except (TypeError, ValueError, UnicodeError) as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, cause=exc, error_msg="JEV-VL request is not valid UTF-8 JSON"
            ) from exc
        self._remaining(deadline)
        decision_id = uuid.uuid4().hex
        answers: dict[str, Json] = {}
        responses: list[Json] = []
        timings: Json = {}
        usage = Usage()
        calls = 0
        for name, question, options, body in prepared:
            self._remaining(deadline)
            request_id = uuid.uuid4().hex
            forced = isinstance(question, ChoiceQuestion) and len(options) == 1
            if forced:
                key = question.ids[0]
                answers[name] = {"choice": key, "probabilities": {key: 1.0}, "confidence": 1.0}
                payload, headers = {"forced": True, "num_model_requests": 0}, {}
            else:
                payload, headers = await self._post(body, request_id, deadline)
                answers[name] = _answer(payload, question, options)
                sent_usage = payload.get("usage")
                mapped: Json = {}
                if isinstance(sent_usage, dict):
                    if "prompt_tokens" in sent_usage:
                        mapped["input_tokens"] = sent_usage["prompt_tokens"]
                    if "completion_tokens" in sent_usage:
                        mapped["output_tokens"] = sent_usage["completion_tokens"]
                used = Usage.from_payload(mapped)
                usage = Usage(
                    usage.input_tokens + used.input_tokens,
                    usage.output_tokens + used.output_tokens,
                    known=usage.known and used.known,
                )
                calls += 1
                timings[name] = parse_server_timing(headers.get("server-timing"))
            responses.append(
                {
                    "question_id": name,
                    "request_id": request_id,
                    "payload": payload,
                    "headers": headers,
                    "forced": forced,
                }
            )
        self._remaining(deadline)
        return Reply(
            answers=answers,
            latency_ms=round((time.monotonic() - started) * 1000),
            usage=usage,
            model=MODEL_ID,
            raw={
                "responses": responses,
                "forced": calls == 0,
                "num_model_requests": calls,
                "url": self.url,
                "request_id": decision_id,
                "server_timing": timings,
                "served_by": {"checkpoint": MODEL_ID, "revision": None, "device": None, "source": "response"}
                if calls
                else {},
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    @classmethod
    def from_env(cls) -> "ServedJevVLModel":
        url = (os.getenv("JEV_VL_URL") or "").strip()
        if not url:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="--model jev-vl-served needs JEV_VL_URL, e.g. http://127.0.0.1:8080",
            )
        return cls(url=url, timeout_s=decisions_timeout_from_env())
