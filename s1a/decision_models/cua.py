# coding: utf-8
"""Cua-S1 Nano in process: an 855K-parameter option scorer, one forward pass per request on a thread.

``cua_s1`` (torch, safetensors) is imported inside ``from_env`` only; the module imports without the extra.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.common.logging import logger

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.types import ChoiceQuestion, Json, Observation, Question, Reply

CUA_DEFAULT_CHECKPOINT = "cua-ai/cua-s1-nano-0.1"
CUA_DEFAULT_SUBFOLDER = "text"
CUA_DEFAULT_CONTEXT_BYTES = 256  # the checkpoint's context window, in UTF-8 bytes
CUA_DEFAULT_OPTION_BYTES = 96  # the window per option


@dataclass(frozen=True)
class Element:
    """What ``NanoScorer.score_elements`` reads per question: the same three fields as ``cua_s1.nano.NanoElement``."""

    element_id: str
    context: str
    options: tuple[str, ...]


class NanoScorer(Protocol):
    """The slice of ``cua_s1.nano.NanoScorer`` the adapter uses: one softmax per element over its own options."""

    def score_elements(
        self,
        elements: Sequence[Element],
        collator: Any,  # cua_s1's NanoByteCollator, opaque here
        device: Any = None,  # cua_s1 accepts a torch.device or None for the scorer's own
    ) -> dict[str, dict[int, float]]: ...


def cua_context(observation: Observation, question: ChoiceQuestion, name: str) -> str:
    """The text Nano reads for one question: a header, the state, then the instructions, so a cut drops the rules first."""
    state = (
        observation.state if isinstance(observation.state, str) else json.dumps(observation.state, ensure_ascii=False)
    )
    lines = [f"# s1a / {name}"]
    if question.goal:
        lines.append(f"goal: {question.goal}")
    lines.append(state)
    lines.extend(question.rules)
    return "\n".join(lines)


def cua_option(key: str, description: str | Json) -> str:
    """One option in the shape the checkpoint was trained on, ``action:role:label``, the key before its description."""
    text = description if isinstance(description, str) else json.dumps(description, ensure_ascii=False)
    return f"click:Button:{key}" + (f" {text}" if text else "")


class CuaS1Model(DecisionModel):
    """Cua-S1 Nano's ``NanoScorer`` behind the interface: choice questions only, text only, deterministic."""

    name = "cua"
    bills_input_tokens = False
    question_types = frozenset({"choice"})
    deterministic = True

    def __init__(
        self,
        scorer: NanoScorer,
        collator: Any,  # cua_s1's NanoByteCollator, handed back to the scorer untouched
        *,
        model: str,
        context_bytes: int,
        option_bytes: int,
    ) -> None:
        self._scorer = scorer
        self._collator = collator
        self._model = model
        self._context_bytes = context_bytes
        self._option_bytes = option_bytes
        self._warned_cut = False

    @property
    def model(self) -> str:
        return self._model

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        elements = []
        for name, question in questions.items():
            assert isinstance(question, ChoiceQuestion)
            context = cua_context(observation, question, name)
            options = tuple(cua_option(key, text) for key, text in question.options.items())
            self._warn_when_cut(name, context, options)
            elements.append(Element(name, context, options))
        started = time.perf_counter()
        try:
            scores = await asyncio.to_thread(self._scorer.score_elements, elements, self._collator)
        except (ValueError, RuntimeError) as exc:  # a malformed batch, and torch failures
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"cua forward pass failed: {exc}"
            ) from exc
        ms = round((time.perf_counter() - started) * 1000)
        answers: dict[str, Json] = {}
        for name, question in questions.items():
            assert isinstance(question, ChoiceQuestion)  # question_types admits choice only
            ids = question.ids
            probabilities = {key: float(scores[name][index]) for index, key in enumerate(ids)}
            key = max(probabilities, key=probabilities.__getitem__)
            answers[name] = {"choice": key, "probabilities": probabilities, "confidence": probabilities[key]}
        return Reply(answers=answers, latency_ms=ms, model=self._model, raw={"scores": scores})

    def _warn_when_cut(self, name: str, context: str, options: tuple[str, ...]) -> None:
        """Nano reads the first ``context_bytes`` of the context and ``option_bytes`` of each option and drops the rest
        silently; one warning per instance names the first request that overflowed."""
        if self._warned_cut:
            return
        context_over = len(context.encode("utf-8")) - self._context_bytes
        options_over = sum(len(option.encode("utf-8")) > self._option_bytes for option in options)
        if context_over > 0 or options_over:
            logger.warning(
                "[cua] question %r overflows the window: context by %d byte(s), %d option(s) over %d bytes; "
                "the tail is cut on this and every later request",
                name,
                max(context_over, 0),
                options_over,
                self._option_bytes,
            )
            self._warned_cut = True

    @classmethod
    def from_env(cls) -> "CuaS1Model":
        """``CUA_S1_CHECKPOINT`` (a hub id or a local directory), ``CUA_S1_SUBFOLDER`` (``text``), ``CUA_S1_DEVICE``."""
        try:
            from cua_s1.nano import load_nano_checkpoint
        except ImportError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="--model cua needs the cua extra: uv sync --extra cua"
            ) from exc
        checkpoint = os.getenv("CUA_S1_CHECKPOINT") or CUA_DEFAULT_CHECKPOINT
        subfolder = os.getenv("CUA_S1_SUBFOLDER") or CUA_DEFAULT_SUBFOLDER
        device = os.getenv("CUA_S1_DEVICE") or "auto"
        directory = checkpoint_directory(checkpoint, subfolder)
        scorer, collator, config = load_nano_checkpoint(directory, device)
        return cls(
            scorer,
            collator,
            model=f"{checkpoint}/{subfolder}",
            context_bytes=int(config.get("context_tokens", CUA_DEFAULT_CONTEXT_BYTES)),
            option_bytes=int(config.get("option_tokens", CUA_DEFAULT_OPTION_BYTES)),
        )


def checkpoint_directory(checkpoint: str, subfolder: str) -> Path:
    """A local checkpoint directory as it is; a hub id fetched once into the Hugging Face cache, ``subfolder`` only."""
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        return local / subfolder
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(checkpoint, allow_patterns=[f"{subfolder}/*"])) / subfolder
