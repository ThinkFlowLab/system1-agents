# coding: utf-8
"""Laya in process: every call is one forward pass on a thread, so the event loop stays free.

``laya`` (torch, transformers) is imported inside ``from_env`` only; the module imports without the extra.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import time
from contextlib import AbstractContextManager, nullcontext
from importlib import metadata
from typing import Any

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.jev import jev_question
from s1a.decision_models.types import ChoiceQuestion, Json, Observation, Question, Reply, Usage

LAYA_DEFAULT_MODEL = "convaiinnovations/laya"
LAYA_DEFAULT_MAX_LEN = 512  # the window Laya assumes when a checkpoint config names none


def laya_question(question: Question) -> Json:
    """Laya takes the Jev choice question as it is (the library renders a dict instruction as JSON); a noul
    question needs a string instruction, and its true/false criteria are native."""
    if isinstance(question, ChoiceQuestion):
        return jev_question(question)
    criteria = {"criteria": dict(question.criteria)} if question.criteria else {}
    return {"type": "noul", "instructions": question.question, **criteria}


def without_weight_init() -> AbstractContextManager[Any]:
    """transformers' ``no_init_weights``. ``laya.load`` builds the encoder from its config, which draws every weight
    at random (about 30 s of the load on CPU), then loads the checkpoint over all of them with ``strict=True``, so
    the draw is thrown away. The helper sits in ``transformers.initialization`` from 5.0 and in
    ``transformers.modeling_utils`` before; without either the load runs as it is."""
    for module in ("transformers.initialization", "transformers.modeling_utils"):
        try:
            return importlib.import_module(module).no_init_weights()
        except (ImportError, AttributeError):
            continue
    return nullcontext()


class LayaModel(DecisionModel):
    """Laya's ``Agent`` (or anything with ``system_one(state, questions)`` and a ``cfg``) behind the interface."""

    name = "laya"
    deterministic = True

    def __init__(self, agent: Any, *, model: str) -> None:
        self._agent = agent
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        asked = {name: laya_question(question) for name, question in questions.items()}
        started = time.perf_counter()
        try:
            payload = await asyncio.to_thread(self._agent.system_one, observation.state, asked)
        except (ValueError, RuntimeError) as exc:  # option overflow, and torch (CUDA included) failures
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"laya forward pass failed: {exc}"
            ) from exc
        ms = round((time.perf_counter() - started) * 1000)
        usage = Usage.from_payload(payload.get("usage"))
        self._check_the_window(usage, len(questions))
        answers = payload.get("answers")  # Laya's own dicts, ``type`` and ``action`` included
        return Reply(
            answers=dict(answers) if isinstance(answers, dict) else {},
            latency_ms=ms,
            usage=usage,
            model=str(payload.get("model") or self._model),
            raw=payload,
        )

    def _check_the_window(self, usage: Usage, questions: int) -> None:
        """Laya cuts each option to 48 tokens, shrinks every option when the head overflows, and cuts the state to
        what is left, all silently. A filled window raises: a decision over a cut state is a guess.
        ``input_tokens`` is the attention-mask sum over every question's row and a row is at most ``max_len`` long,
        so the sum reaches ``questions * max_len`` only when every row hit the window. Exact for one question; with
        several, a cut on the widest head alone goes unseen."""
        window = int(self._agent.cfg.get("max_len", LAYA_DEFAULT_MAX_LEN)) * questions
        if usage.input_tokens >= window:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=(
                    f"the laya {window}-token window filled ({usage.input_tokens} tokens over {questions} question(s)): "
                    "the state or the options were cut; raise LAYA_MAX_LEN / LAYA_HEAD_MAX_LEN or shorten the state"
                ),
            )

    # ponytail: warm() with one tiny forward pass so CUDA kernels compile before the first real turn

    @classmethod
    def from_env(cls) -> "LayaModel":
        """``LAYA_MODEL`` (a hub id or a path), ``LAYA_SUBFOLDER``, ``LAYA_DEVICE``; ``LAYA_MAX_LEN`` and
        ``LAYA_HEAD_MAX_LEN`` override the checkpoint's window."""
        try:
            import laya
        except ImportError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="--model laya needs the laya extra: uv sync --extra laya",
            ) from exc
        model = os.getenv("LAYA_MODEL") or LAYA_DEFAULT_MODEL
        subfolder = os.getenv("LAYA_SUBFOLDER") or None
        with without_weight_init():
            agent = laya.load(model, device=os.getenv("LAYA_DEVICE") or None, subfolder=subfolder)
        if not callable(getattr(agent, "system_one", None)):
            try:
                version = metadata.version("laya")
            except metadata.PackageNotFoundError:
                version = "unknown"
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=(
                    f"laya {version} loaded an agent without a callable system_one(state, questions); "
                    "the s1a laya model needs that method"
                ),
            )
        for key, variable in (("max_len", "LAYA_MAX_LEN"), ("head_max_len", "LAYA_HEAD_MAX_LEN")):
            value = os.getenv(variable)
            if value:
                agent.cfg[key] = int(value)
        return cls(agent, model=f"{model}/{subfolder}" if subfolder else model)
