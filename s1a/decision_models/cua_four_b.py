# coding: utf-8
"""Optional Cua-S1 4B adapter: a closed choice over text or one current screenshot."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.cua import checkpoint_directory, cua_context
from s1a.decision_models.types import ChoiceQuestion, Json, Observation, Question, Reply


@dataclass(frozen=True)
class Option:
    """The fields read by the official ``cua_s1.four_b.FourBModel``."""

    element_id: str
    role: str
    label: str
    action: str
    entity_id: str | None = None


def grounded_option(key: str, description: str | Json, observation: Observation) -> Option:
    """Preserve desktop click/fill semantics in the official scorer's input contract."""
    text = description if isinstance(description, str) else json.dumps(description, ensure_ascii=False)
    action, _, label = key.partition(":")
    if action in {"click", "type"} and isinstance(observation.state, dict):
        elements = observation.state.get("elements", [])
        matches = [e for e in elements if isinstance(e, dict) and label in (e.get("label"), e.get("identifier"))]
        if len(matches) == 1:
            element = matches[0]
            return Option(
                key,
                str(element.get("role", "element")).removeprefix("AX"),
                label,
                "fill" if action == "type" else "click",
                text if action == "type" else None,
            )
    return Option(key, "action", text, "select")


class CuaFourBModel(DecisionModel):
    name = "cua"
    bills_input_tokens = False
    question_types = frozenset({"choice"})
    deterministic = True

    def __init__(self, scorer: Any, *, model: str, modality: str) -> None:
        if modality not in {"text", "multimodal"}:
            raise ValueError("CUA_S1_MODALITY must be text or multimodal")
        self._scorer = scorer
        self._model = model
        self._modality = modality
        self.supports_images = modality == "multimodal"

    @property
    def model(self) -> str:
        return self._model

    async def warm(self) -> None:
        await asyncio.to_thread(self._scorer.load)

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        if self.supports_images and len(observation.images) != 1:
            raise build_error(StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="Cua-S1 4B needs exactly one screenshot")
        for question in questions.values():
            assert isinstance(question, ChoiceQuestion)
            if len(question.options) > 26:
                raise build_error(
                    StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="Cua-S1 4B supports at most 26 options"
                )
        started = time.perf_counter()
        try:
            answers = await asyncio.to_thread(self._forward, observation, questions)
        except (ValueError, RuntimeError, OSError) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"Cua-S1 4B forward pass failed: {exc}"
            ) from exc
        return Reply(answers=answers, latency_ms=round((time.perf_counter() - started) * 1000), model=self._model)

    def _forward(self, observation: Observation, questions: dict[str, Question]) -> dict[str, Json]:
        # The official scorer accepts a file path. Keep image bytes out of JSON observations and run logs.
        with tempfile.TemporaryDirectory(prefix="s1a-cua-") as temporary:
            screenshot = None
            if self.supports_images:
                screenshot = Path(temporary) / "screen.png"
                screenshot.write_bytes(observation.images[0].data)
            answers = {}
            for name, question in questions.items():
                assert isinstance(question, ChoiceQuestion)
                options = [grounded_option(key, text, observation) for key, text in question.options.items()]
                context = cua_context(observation, question, name)
                results = self._scorer.forward(
                    options,
                    app=str(observation.state.get("app", "s1a")) if isinstance(observation.state, dict) else "s1a",
                    # Upstream's multimodal prompt ignores ax_tree; carry the goal and rules in its task field.
                    task_family=context if self.supports_images else name,
                    ax_tree=None if self.supports_images else context,
                    screenshot=screenshot,
                    modality=self._modality,
                )
                probabilities = {result.element_id: float(result.probability) for result in results}
                if len(results) != len(options) or set(probabilities) != set(question.options):
                    raise ValueError("scorer returned missing, duplicate or unknown option keys")
                key = max(probabilities, key=probabilities.__getitem__)
                answers[name] = {"choice": key, "probabilities": probabilities, "confidence": probabilities[key]}
            return answers

    @classmethod
    def from_env(cls) -> "CuaFourBModel":
        try:
            from cua_s1.four_b import FourBModel
            import peft  # noqa: F401 -- fail before downloading weights when the extra is absent
            import torchvision  # noqa: F401
        except ImportError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="CUA_S1_VARIANT=4b needs: uv sync --extra cua-four-b",
            ) from exc
        modality = os.getenv("CUA_S1_MODALITY", "multimodal")
        if modality not in {"text", "multimodal"}:
            raise ValueError("CUA_S1_MODALITY must be text or multimodal")
        base = os.getenv("CUA_S1_BASE_MODEL") or "Qwen/Qwen3.5-4B"
        adapter = os.getenv("CUA_S1_CHECKPOINT") or "cua-ai/cua-s1-4b-0.2"
        scorer = FourBModel(
            base_model=base,
            lora_adapter_path=checkpoint_directory(adapter, modality),
            device=os.getenv("CUA_S1_DEVICE") or "cpu",
            dtype=os.getenv("CUA_S1_DTYPE") or "float32",
            modality=modality,
        )
        return cls(scorer, model=f"{base}+{adapter}/{modality}", modality=modality)
