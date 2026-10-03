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
from typing import Any, TypeGuard

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
LAYA_BROWSER_OPTION_CHARS = 28  # per target option: the label (and value) Laya reads, all options share one budget
LAYA_BROWSER_BLOCKER_CHARS = 24  # the overlay's name on a blocked row: the link to the button that closes it
_FLAG_LETTERS = (
    ("checked", "C"),
    ("selected", "S"),
    ("expanded", "X"),
    ("click_did_nothing", "D"),
)
_HEAD_ASKS = {
    "operation": "Which operation comes next?",
    "text_value": "Which value should be typed into the field?",
}


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


def _laya_browser_option(option: Any) -> Any:
    """A browser target option (``{"element": "[12] Where from?", "current_value": ...}``) as one short line."""
    if not (isinstance(option, dict) and "element" in option):
        return option
    label = str(option["element"]).split("] ", 1)[-1][:LAYA_BROWSER_OPTION_CHARS]
    if option.get("option"):
        label += f" / {str(option['option'])[:LAYA_BROWSER_OPTION_CHARS]}"
    value = str(option.get("current_value") or "")
    return label + (f" = {value[:LAYA_BROWSER_OPTION_CHARS]}" if value else "")


def laya_browser_question(asked: Json, name: str = "") -> Json:
    """A browser-front choice question, shaped for Laya's one shared option budget (``head_max_len`` holds the
    instruction and every option): the agent's long rules dropped, the instruction reduced to the goal and the ask
    of its head (``name``: ``operation``, ``<op>_target``, ``text_value``), each target option reduced to its
    element's label and value. Unfolded, a 23-element target head left each option about six tokens,
    ``12: {"element": "[``, so Laya never saw an element's name. Two options that shorten to the same text (a result
    list, a calendar) get their key in front, so Laya can still tell them apart.
    Anything that is not a goal-bearing choice question passes through."""
    instructions = asked.get("instructions")
    if asked.get("type") != "choice" or not (isinstance(instructions, dict) and instructions.get("goal")):
        return asked
    operation = instructions.get("operation")
    if name in _HEAD_ASKS:
        ask = _HEAD_ASKS[name]
    elif operation:
        ask = f"Which element should {operation} act on?"
    else:
        ask = "Which option fits?"
    criteria = {key: _laya_browser_option(option) for key, option in asked["criteria"].items()}
    texts = list(criteria.values())
    criteria = {
        key: (f"[{key}] {text}" if isinstance(text, str) and texts.count(text) > 1 else text)
        for key, text in criteria.items()
    }
    return {"type": "choice", "instructions": f"Task: {instructions['goal']} {ask}", "criteria": criteria}


def _is_on(flag: Any) -> bool:
    """A row flag that is set. The probe sends aria-* and checkbox state as strings (``"true"``, ``"false"``),
    and ``"false"`` is a non-empty, truthy string; the policy's own flags (``click_did_nothing``) are booleans."""
    if isinstance(flag, str):
        return flag.strip().lower() == "true"
    return flag is True


def _laya_browser_row(row: Json) -> str:
    """One element row as a short line instead of a JSON object: the repeated key names (``role``, ``label``, ...)
    are what a tiny window can least afford. ``[CX]``-style flags stand in for the sparse boolean fields; a blocked
    row keeps its overlay's name, the only link to the button that closes it."""
    label = str(row.get("label") or "")[:LAYA_BROWSER_LABEL_CHARS]
    value = str(row.get("value") or "")[:LAYA_BROWSER_LABEL_CHARS]
    flags = "".join(letter for key, letter in _FLAG_LETTERS if _is_on(row.get(key)))
    parts = [str(row.get("index", "")), str(row.get("role") or ""), label]
    if value:
        parts.append(f"={value}")
    if flags:
        parts.append(f"[{flags}]")
    if row.get("blocked_by"):
        parts.append(f"(blocked by {str(row['blocked_by'])[:LAYA_BROWSER_BLOCKER_CHARS]})")
    return " ".join(part for part in parts if part)


def is_browser_state(state: Json | str) -> TypeGuard[Json]:
    """The browser front's per-tick state: a ``page`` object and an ``elements`` list."""
    return isinstance(state, dict) and isinstance(state.get("page"), dict) and isinstance(state.get("elements"), list)


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
    if not is_browser_state(state):
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
            f"{entry.get('kind', '')}:{entry.get('action', '')}"
            + (" (no change)" if entry.get("page_changed") is False else "")  # None: not measured, not "no change"
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
        state = observation.state
        if self._compact_browser_state and is_browser_state(state):  # its questions are the browser heads
            state = laya_state(state)
            asked = {name: laya_browser_question(question, name) for name, question in asked.items()}
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
        compact = (os.getenv("LAYA_COMPACT_BROWSER_STATE") or "1").strip().lower() not in ("0", "false", "no")
        return cls(agent, model=f"{model}/{subfolder}" if subfolder else model, compact_browser_state=compact)
