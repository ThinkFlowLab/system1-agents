# coding: utf-8
"""Laya in process: every call is one forward pass on a thread, so the event loop stays free.

``laya`` (torch, transformers) is imported inside ``from_env`` only; the module imports without the extra.
"""

from __future__ import annotations

import asyncio
import os
import time
from importlib import metadata
from typing import Any

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error

from s1a.decision_models.base import DecisionModel
from s1a.decision_models.jev import jev_question
from s1a.decision_models.types import ChoiceQuestion, Json, Observation, Question, Reply, Usage

LAYA_DEFAULT_MODEL = "convaiinnovations/laya"
LAYA_DEFAULT_MAX_LEN = 512  # the window Laya assumes when a checkpoint config names none
LAYA_BROWSER_LABEL_CHARS = 40  # a row's label/value, kept over its full text (Jev's window is 32K; Laya's is not)
LAYA_BROWSER_TITLE_CHARS = 80
LAYA_BROWSER_HISTORY_KEPT = 3  # of the browser front's last ten actions; the freshest ones carry the signal
_FLAG_LETTERS = (
    ("checked", "C"),
    ("selected", "S"),
    ("expanded", "X"),
    ("blocked_by", "B"),
    ("click_did_nothing", "D"),
)


def laya_question(question: Question) -> Json:
    """Laya takes the Jev choice question as it is (the library renders a dict instruction as JSON); a noul
    question needs a string instruction, and its true/false criteria are native."""
    if isinstance(question, ChoiceQuestion):
        return jev_question(question)
    criteria = {"criteria": dict(question.criteria)} if question.criteria else {}
    return {"type": "noul", "instructions": question.question, **criteria}


def _laya_browser_row(row: Json) -> str:
    """One element row as a short line instead of a JSON object: the repeated key names (``role``, ``label``, ...)
    are what a tiny window can least afford. ``[CX]``-style flags stand in for the sparse boolean/id fields."""
    label = str(row.get("label") or "")[:LAYA_BROWSER_LABEL_CHARS]
    value = str(row.get("value") or "")[:LAYA_BROWSER_LABEL_CHARS]
    flags = "".join(letter for key, letter in _FLAG_LETTERS if row.get(key))
    parts = [str(row.get("index", "")), str(row.get("role") or ""), label]
    if value:
        parts.append(f"={value}")
    if flags:
        parts.append(f"[{flags}]")
    return " ".join(part for part in parts if part)


def laya_state(state: Json | str) -> Json | str:
    """The browser front's per-tick state, folded to fit Laya's window: no ``page.text`` (the choice heads already
    carry each candidate's own text; the free-form page dump is for the chat model's DONE answer, which Laya never
    writes), the element table as one short line per row instead of a JSON object per row, and the last
    ``LAYA_BROWSER_HISTORY_KEPT`` actions instead of ten. Anything that is not this shape (a plain string, the tool
    front's state, a rail's) passes through: it already fits the window Laya was sized for.

    This is what made Laya's real-page window error mean anything other than "raise LAYA_MAX_LEN and hope": on a
    dozen-element page the JSON-shaped state alone ran well past a 512-token window before a single instruction
    token was spent.
    """
    if not (
        isinstance(state, dict) and isinstance(state.get("page"), dict) and isinstance(state.get("elements"), list)
    ):
        return state
    compact: Json = {
        "page": {
            "url": str(state["page"].get("url", "")),
            "title": str(state["page"].get("title", ""))[:LAYA_BROWSER_TITLE_CHARS],
        },
        "elements": [_laya_browser_row(row) for row in state["elements"]],
    }
    recent = state.get("recent_actions")
    if recent:
        compact["recent_actions"] = [
            f"{entry.get('kind', '')}:{entry.get('action', '')}" + ("" if entry.get("page_changed") else " (no change)")
            for entry in recent[-LAYA_BROWSER_HISTORY_KEPT:]
        ]
    return compact


class LayaModel(DecisionModel):
    """Laya's ``Agent`` (or anything with ``system_one(state, questions)`` and a ``cfg``) behind the interface."""

    name = "laya"
    deterministic = True

    def __init__(self, agent: Any, *, model: str, compact_browser_state: bool = True) -> None:
        self._agent = agent
        self._model = model
        self._compact_browser_state = compact_browser_state

    @property
    def model(self) -> str:
        return self._model

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        asked = {name: laya_question(question) for name, question in questions.items()}
        state = laya_state(observation.state) if self._compact_browser_state else observation.state
        started = time.perf_counter()
        try:
            payload = await asyncio.to_thread(self._agent.system_one, state, asked)
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
        ``LAYA_HEAD_MAX_LEN`` override the checkpoint's window. ``LAYA_COMPACT_BROWSER_STATE`` (default on;
        ``0``/``false``/``no`` turns it off) folds a browser-shaped state through ``laya_state`` before every call."""
        try:
            import laya
        except ImportError as exc:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR,
                error_msg="--model laya needs the laya extra: uv sync --extra laya",
            ) from exc
        model = os.getenv("LAYA_MODEL") or LAYA_DEFAULT_MODEL
        subfolder = os.getenv("LAYA_SUBFOLDER") or None
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
        compact = (os.getenv("LAYA_COMPACT_BROWSER_STATE") or "1").strip().lower() not in ("0", "false", "no")
        return cls(agent, model=f"{model}/{subfolder}" if subfolder else model, compact_browser_state=compact)
