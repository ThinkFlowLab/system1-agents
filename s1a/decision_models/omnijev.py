# coding: utf-8
"""OmniJev in process: a Qwen3.5 vision-language decision model that reads a screenshot, one call per request on a thread.

OmniJev (https://github.com/tinnel123666888/OmniJev, Apache-2.0) answers the TypeSafe question shape over an image:
``MSO1.system_one({"images": [path]}, questions)``. It ships as a repository, not a package, so ``OMNIJEV_REPO``
names a local clone; torch, transformers and peft come from the ``omnijev`` extra. Everything heavy is imported inside
``from_env``; the module imports without the extra.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Protocol

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.types import ChoiceQuestion, Image, Json, Observation, Question, Reply

OMNIJEV_DEFAULT_CHECKPOINT = "tinnel123/OmniJev-0.8B"
OMNIJEV_DEFAULT_REVISION = "v1.1"
OMNIJEV_DEFAULT_BASE = "Qwen/Qwen3.5-0.8B"
OMNIJEV_STATE_CHARS = 6000  # the text state read with the picture; the picture carries the rest
OMNIJEV_PAGE_TEXT_CHARS = 2000  # a browser page's free text, capped on its own so the actions and elements stay in
OMNIJEV_PROMPTS = ("full", "short")  # full: text state + goal + rules + ask; short: goal + ask over the screenshot
OMNIJEV_DEFAULT_PROMPT = "full"
_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


class OmniJevAgent(Protocol):
    """The slice of ``mso.infer.MSO1`` the adapter uses."""

    def system_one(self, state: Json, questions: dict[str, Json]) -> dict[str, Json]: ...


def omnijev_option(description: str | Json) -> str:
    """One option's description as text: a browser element row (``{"element": "[12] Where from?", ...}``) becomes its
    label and value; any other dict becomes compact JSON."""
    if isinstance(description, str):
        return description
    if "element" in description:
        label = str(description["element"]).split("] ", 1)[-1]
        if description.get("option"):
            label += f" / {description['option']}"
        value = str(description.get("current_value") or "")
        return label + (f" = {value}" if value else "")
    return json.dumps(description, ensure_ascii=False)


def omnijev_context(observation: Observation) -> str:
    """The observation's text state, read in front of every question (OmniJev reads text context in the instructions).

    A browser state is reordered so the whole budget cannot go to the page's free text: the recent actions come first
    (the screenshot cannot show them, and the rules depend on them, e.g. PRESS_ENTER after a Search click that did
    nothing), then the page with its text capped at ``OMNIJEV_PAGE_TEXT_CHARS``, then the element table."""
    state = observation.state
    if isinstance(state, dict) and isinstance(state.get("page"), dict):
        page = dict(state["page"])
        page_text = str(page.get("text") or "")
        if len(page_text) > OMNIJEV_PAGE_TEXT_CHARS:
            page["text"] = page_text[:OMNIJEV_PAGE_TEXT_CHARS] + " …"
        rest = {key: value for key, value in state.items() if key not in ("recent_actions", "page")}
        state = {"recent_actions": state.get("recent_actions", []), "page": page, **rest}
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return text[:OMNIJEV_STATE_CHARS]


def omnijev_question(question: Question, context: str, *, prompt: str = "full") -> Json:
    """One question in OmniJev's shape. ``full``: the text state, the goal and the rules in front of the ask.
    ``short``: the goal and the ask only; the screenshot carries the page and the rules stay out."""
    full = prompt == "full"
    lines = [f"State: {context}"] if context and full else []
    if isinstance(question, ChoiceQuestion):
        if question.goal:
            lines.append(f"Task: {question.goal}")
        if full:
            lines.extend(question.rules)
        lines.append(
            f"Which element should {question.operation} act on?" if question.operation else "Which option comes next?"
        )
        criteria = {key: omnijev_option(description) for key, description in question.options.items()}
        return {"type": "choice", "instructions": "\n".join(lines), "criteria": criteria}
    lines.append(question.question)
    for verdict, text in (question.criteria or {}).items():
        lines.append(f"{verdict}: {text}")
    return {"type": "noul", "instructions": "\n".join(lines)}


def omnijev_answer(answer: Json) -> Json:
    """OmniJev's answer in the shape ``decide_many`` validates. A choice's probabilities sum to ``1 - abstain`` (its
    "none of these"); they are renormalised over the keys it returned and the confidence recomputed with Jev's
    definition on them. The key stays OmniJev's own, so a malformed answer still fails validation. ``abstain`` stays
    in the reply's raw payload."""
    if "noul" in answer:
        return {"noul": float(answer["noul"])}
    raw = {key: float(p) for key, p in (answer.get("probabilities") or {}).items()}
    total = sum(raw.values())
    if not raw or total <= 0:
        return {"choice": answer.get("choice"), "probabilities": raw, "confidence": 0.0}
    probabilities = {key: p / total for key, p in raw.items()}
    k = len(probabilities)
    peak = max(probabilities.values())
    confidence = 1.0 if k < 2 else max(0.0, min(1.0, (k * peak - 1) / (k - 1)))
    return {"choice": answer.get("choice"), "probabilities": probabilities, "confidence": confidence}


class OmniJevModel(DecisionModel):
    """OmniJev's ``MSO1`` behind the interface: choice and noul questions over one screenshot, deterministic."""

    name = "omnijev"
    supports_images = True
    deterministic = True
    bills_input_tokens = False  # in process, not Jev's pricing

    def __init__(self, agent: OmniJevAgent, *, model: str, prompt: str = OMNIJEV_DEFAULT_PROMPT) -> None:
        if prompt not in OMNIJEV_PROMPTS:
            raise ValueError(f"omnijev prompt is one of {OMNIJEV_PROMPTS}, not {prompt!r}")
        self._agent = agent
        self._model = model
        self._prompt = prompt

    @property
    def model(self) -> str:
        return self._model

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        if not observation.images:
            # a call failure, not a config error: the usual cause is a screenshot that failed to capture
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                error_msg="omnijev decides over a screenshot; this observation carries no image",
            )
        context = omnijev_context(observation)
        asked = {name: omnijev_question(question, context, prompt=self._prompt) for name, question in questions.items()}
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="s1a-omnijev-") as folder:
            paths = [_write_image(image, Path(folder), index) for index, image in enumerate(observation.images)]
            try:
                payload = await asyncio.to_thread(self._agent.system_one, {"images": paths}, asked)
            except (ValueError, RuntimeError, OSError) as exc:  # a bad image, and torch failures
                raise build_error(
                    StatusCode.MODEL_CALL_FAILED, cause=exc, error_msg=f"omnijev forward pass failed: {exc}"
                ) from exc
        ms = round((time.perf_counter() - started) * 1000)
        answers = {name: omnijev_answer(payload[name]) for name in questions if name in payload}
        return Reply(answers=answers, latency_ms=ms, model=self._model, raw=payload)

    @classmethod
    def from_env(cls) -> "OmniJevModel":
        """``OMNIJEV_REPO`` (a local clone of the OmniJev repository, required), ``OMNIJEV_CHECKPOINT`` (a hub id or a
        local directory), ``OMNIJEV_REVISION`` (for a hub id) and ``OMNIJEV_BASE`` (the Qwen3.5 backbone of the same
        size, a hub id or a local directory), ``OMNIJEV_PROMPT`` (``full`` or ``short``, see ``omnijev_question``)."""
        prompt = os.getenv("OMNIJEV_PROMPT") or OMNIJEV_DEFAULT_PROMPT
        if prompt not in OMNIJEV_PROMPTS:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"OMNIJEV_PROMPT is one of {', '.join(OMNIJEV_PROMPTS)}, not {prompt!r}",
            )
        repo = os.getenv("OMNIJEV_REPO")
        if not repo or not (Path(repo) / "mso" / "infer.py").is_file():
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=(
                    "--model omnijev needs OMNIJEV_REPO, a local clone of "
                    "https://github.com/tinnel123666888/OmniJev, and the omnijev extra: uv sync --extra omnijev"
                ),
            )
        # OmniJev ships as a repository, not a package: its clone stays first on sys.path for the process lifetime,
        # so its top-level packages (mso, ...) shadow any installed ones of the same name
        sys.path.insert(0, str(Path(repo).resolve()))
        try:
            from mso.infer import MSO1
        except ImportError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg=f"--model omnijev could not import OmniJev ({exc}); uv sync --extra omnijev",
            ) from exc
        checkpoint = os.getenv("OMNIJEV_CHECKPOINT") or OMNIJEV_DEFAULT_CHECKPOINT
        revision = os.getenv("OMNIJEV_REVISION") or OMNIJEV_DEFAULT_REVISION
        base = os.getenv("OMNIJEV_BASE") or OMNIJEV_DEFAULT_BASE
        agent = MSO1(_local(checkpoint, revision), _local(base, None))
        return cls(agent, model=checkpoint if Path(checkpoint).is_dir() else f"{checkpoint}@{revision}", prompt=prompt)


def _local(name: str, revision: str | None) -> str:
    """A local directory as it is; a hub id fetched once into the Hugging Face cache."""
    if Path(name).expanduser().is_dir():
        return str(Path(name).expanduser())
    from huggingface_hub import snapshot_download

    return snapshot_download(name, revision=revision)


def _write_image(image: Image, folder: Path, index: int) -> str:
    path = folder / f"{index}{_SUFFIXES.get(image.media_type, '.png')}"
    path.write_bytes(image.data)
    return str(path)
