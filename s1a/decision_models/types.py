# coding: utf-8
"""What a decision model reads and returns: an observation, typed questions, typed answers."""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Union

Json = dict[str, Any]
QuestionType = Literal["choice", "noul"]
PROVENANCE_KEYS = ("served_by", "url", "request_id", "server_timing")


@dataclass(frozen=True)
class Image:
    """One picture in an observation. Text-only models drop them and warn once."""

    data: bytes
    media_type: str = "image/png"

    @classmethod
    def from_base64(cls, encoded: str, media_type: str = "image/png") -> "Image":
        return cls(base64.b64decode(encoded), media_type)


@dataclass(frozen=True)
class Observation:
    """The state a decision model decides on: a JSON object (or plain text) plus zero or more images."""

    state: Json | str
    images: tuple[Image, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, (dict, str)):
            raise TypeError(f"an observation's state is a dict or a str, not {type(self.state).__name__}")
        object.__setattr__(self, "images", tuple(self.images))


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick one key among ``options``; each value describes its key (a string, or a dict for the browser's element rows).

    ``goal``, ``rules`` and ``operation`` are the instructions every backend renders its own way; one rule may be
    given as a plain string.
    """

    options: Mapping[str, str | Json]
    goal: str = ""
    rules: str | tuple[str, ...] = ()
    operation: str = ""

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("a choice question needs at least one option")
        if not all(isinstance(key, str) for key in self.options):
            raise ValueError("option keys must be strings")
        rules = (self.rules,) if isinstance(self.rules, str) else tuple(self.rules)
        object.__setattr__(self, "rules", rules)

    @property
    def type(self) -> QuestionType:
        return "choice"

    @property
    def ids(self) -> list[str]:
        return list(self.options)


@dataclass(frozen=True)
class NoulQuestion:
    """Does ``question`` hold for the observation? ``criteria`` may spell out what true and false mean."""

    question: str
    criteria: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("a noul question needs a question")
        if self.criteria is not None and set(self.criteria) != {"true", "false"}:
            raise ValueError("noul criteria are exactly the keys true and false")

    @property
    def type(self) -> QuestionType:
        return "noul"


Question = Union[ChoiceQuestion, NoulQuestion]


@dataclass(frozen=True)
class Choice:
    """A validated choice answer: the key, a probability per key (sum 1 within 0.02, peak at the key), a confidence.

    Only ``validation.validate_choice`` builds one from a backend's answer.
    """

    key: str
    probabilities: dict[str, float]
    confidence: float

    def as_dict(self) -> Json:
        """The CLI and MCP output shape."""
        return {"choice": self.key, "probabilities": dict(self.probabilities), "confidence": self.confidence}


@dataclass(frozen=True)
class Noul:
    """A validated noul answer: the probability the statement holds and a confidence.

    Only ``validation.validate_noul`` builds one from a backend's answer; a missing confidence becomes max(p, 1-p).
    """

    p: float
    confidence: float

    def as_dict(self) -> Json:
        return {"noul": self.p, "confidence": self.confidence}


Answer = Union[Choice, Noul]


@dataclass(frozen=True)
class Usage:
    """Tokens a backend reported, and whether it reported them at all.

    ``known`` distinguishes an explicit zero from missing or malformed usage: a direct ``Usage()`` is a known zero
    for a local model or a test double, while ``from_payload`` marks a payload without a valid input count unknown,
    so a consumer can tell "no tokens" from "we do not know how many tokens".
    """

    input_tokens: int = 0
    output_tokens: int = 0
    known: bool = True

    @classmethod
    def from_payload(cls, usage: Any) -> "Usage":
        """Tolerant read of a backend's usage: counts are kept, but ``known`` is False when they are not usable.

        A present, valid input count is required; an absent output count is an implicit zero (the decision
        backends report input only). A bool, negative, non-integral or otherwise invalid count leaves ``known``
        False while the tolerant numeric value is still kept.
        """
        if not isinstance(usage, dict):
            return cls(0, 0, known=False)
        input_tokens, output_tokens = _count(usage.get("input_tokens")), _count(usage.get("output_tokens"))
        known = "input_tokens" in usage and _valid_count(usage.get("input_tokens"))
        if "output_tokens" in usage:
            known = known and _valid_count(usage.get("output_tokens"))
        return cls(input_tokens, output_tokens, known=known)


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _valid_count(value: Any) -> bool:
    """A usable token count: a non-negative integer, never a bool; a float only when it is whole."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and float(value).is_integer() and value >= 0


@dataclass(frozen=True)
class Reply:
    """What a backend said to one request, before validation: one dict per question name, plus the request's facts."""

    answers: dict[str, Json]
    latency_ms: int
    usage: Usage = Usage()
    model: str = ""
    raw: Json = field(default_factory=dict, repr=False, compare=False)  # the whole payload, for artifacts


@dataclass(frozen=True)
class Decision:
    """One request, validated: an answer per question, plus latency, usage and the model that answered."""

    answers: dict[str, Answer]
    latency_ms: int
    usage: Usage = Usage()
    model: str = ""
    raw: Json = field(default_factory=dict, repr=False, compare=False)  # the whole payload, for artifacts

    def choice(self, name: str) -> Choice:
        """KeyError when the question was not asked, TypeError when it was a noul question."""
        answer = self.answers[name]
        if not isinstance(answer, Choice):
            raise TypeError(f"{name!r} is a noul answer, not a choice")
        return answer

    def noul(self, name: str) -> Noul:
        answer = self.answers[name]
        if not isinstance(answer, Noul):
            raise TypeError(f"{name!r} is a choice answer, not a noul")
        return answer

    @property
    def provenance(self) -> Json:
        """Where a served answer came from, for step records: the server's identity, URL and timings; empty in process."""
        return {key: self.raw[key] for key in PROVENANCE_KEYS if key in self.raw}
