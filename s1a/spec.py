# coding: utf-8
"""What an S1A agent is: one frozen spec per front, stated in full by every agent module as ``SPEC``."""

from __future__ import annotations

import argparse
import re
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentCallbackEvent

from s1a.browser import prompts
from s1a.env import Env
from s1a.jobs import Episode
from s1a.decision_models.baselines import Rule

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")  # the module name under s1a.agents, so the two never differ
LANGUAGES = tuple(prompts.OPERATION_RULES)
RAIL_HOOKS = (
    AgentCallbackEvent.BEFORE_TOOL_CALL,
    AgentCallbackEvent.AFTER_TOOL_CALL,
)  # the hooks whose ctx.inputs is ToolCallInputs
Json = dict[str, object]
Band = Literal["allow", "uncertain", "act"]


def _check_name_and_text(name: str, description: str, rules: str) -> None:
    if not _NAME.match(name):
        raise ValueError(
            f"agent name must be its module name, lowercase letters, digits and _ after a letter: {name!r}"
        )
    if not rules.strip() or not description.strip():
        raise ValueError(f"{name}: rules and description must not be empty")


def positive_int(text: str) -> int:
    """An argparse type for a count that must be 1 or more."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {text}")
    return value


def positive_float(text: str) -> float:
    """An argparse type for seconds that must be above 0."""
    value = float(text)
    if not value > 0:  # rejects nan too
        raise argparse.ArgumentTypeError(f"must be above 0, got {text}")
    return value


@dataclass(frozen=True)
class Budget:
    """Per-episode limits; ``--max-steps`` and ``--timeout`` override the first two at run time on both fronts."""

    max_steps: int
    timeout_s: float
    stall_after: int  # tool front: acts without a score change before a rethink; browser: actions without a page change before BLOCKED; 0 never

    def __post_init__(self) -> None:
        if not isinstance(self.max_steps, int) or self.max_steps <= 0 or self.timeout_s <= 0 or self.stall_after < 0:
            raise ValueError(f"budget out of range: {self}")


@dataclass(frozen=True)
class Series:
    """One run's resources, built from the parsed flags: the seeds, the envs, the page session, the baseline."""

    seeds: Sequence[int]
    env_for: Callable[[int], Env]
    session: AbstractAsyncContextManager[None]
    baseline: tuple[str, Rule] | None
    annotate: Callable[[Env, Episode], None]


@dataclass(frozen=True)
class ToolAgentSpec:
    """A tool-front agent: a DeepAgent plays through ``observe`` and ``act`` with a decision model in the model slot."""

    name: str
    description: str
    rules: str  # what Jev reads on every decision: facts to recognise, under about 120 words
    budget: Budget
    flags: Callable[[argparse.ArgumentParser], None]  # the agent's own switches, added after the shared ones
    series: Callable[[argparse.Namespace], Series]

    def __post_init__(self) -> None:
        _check_name_and_text(self.name, self.description, self.rules)


@dataclass(frozen=True)
class BrowserAgentSpec:
    """A browser-front agent: openJiuwen's browser subagent with the Jev policy in its model slot."""

    name: str
    description: str
    rules: str  # the operation rules the policy sends with every decision; one string per site family
    language: str  # picks the target, value and answer rule tables: en or cn
    budget: Budget  # max_steps: the subagent's iteration cap
    goal: str | None  # a fixed task, or None when every call brings its own

    def __post_init__(self) -> None:
        _check_name_and_text(self.name, self.description, self.rules)
        if self.language not in LANGUAGES:
            raise ValueError(f"{self.name}: language must be one of {LANGUAGES}, got {self.language!r}")
        if self.goal is not None and not self.goal.strip():
            raise ValueError(f"{self.name}: goal must be None or a non-blank task")


@dataclass(frozen=True)
class Thresholds:
    """Bands over the probability a rail acts on: at or below ``allow`` nothing happens, at or above ``act`` the rail acts."""

    allow: float
    act: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.allow < self.act <= 1.0:
            raise ValueError(f"thresholds must satisfy 0 <= allow < act <= 1: {self}")

    def band(self, p: float) -> Band:
        if p <= self.allow:
            return "allow"
        if p >= self.act:
            return "act"
        return "uncertain"


@dataclass(frozen=True)
class Verdict:
    """One rail decision: the probability that the flagged statement holds, its band, and the request's latency."""

    p: float
    band: Band
    ms: int
    input_tokens: int  # the request's usage, 0 when the endpoint reports none


@dataclass(frozen=True)
class RailSpec:
    """A rail-front agent: one Jev question at one callback hook of a DeepAgent, acted on above a threshold."""

    name: str
    description: str
    hook: AgentCallbackEvent
    question: Literal["noul", "choice"]
    rules: str  # noul: the yes-or-no question Jev answers; choice: the goal
    criteria: dict[str, str]  # noul: what "true" and "false" mean; choice: key to description
    flagged: str  # the key whose probability the thresholds band: "true" for noul
    state_of: Callable[[AgentCallbackContext], Json | None]  # what Jev sees at the hook; None skips the event
    thresholds: Thresholds
    act: Callable[[AgentCallbackContext, Verdict], Awaitable[None]]
    on_failure: Literal["open", "closed"]  # closed: a decision error runs act with a verdict at the act threshold
    labelled_set: Path | None  # JSONL records {"state", "label"}; the label says whether the flagged statement holds
    # What the hook would send for a labelled record's state (the injection guard clips the text as state_of does), so
    # a set can keep pages in full and the evaluation measures the runtime clip. None sends the state as is.
    labelled_state: Callable[[Json], Json] | None = None

    def __post_init__(self) -> None:
        _check_name_and_text(self.name, self.description, self.rules)
        if self.hook not in RAIL_HOOKS:
            raise ValueError(f"{self.name}: hook must be one of {[hook.name for hook in RAIL_HOOKS]}, got {self.hook}")
        if self.on_failure not in ("open", "closed"):
            raise ValueError(f"{self.name}: on_failure must be open or closed, got {self.on_failure!r}")
        match self.question:
            case "noul":
                if set(self.criteria) != {"true", "false"} or self.flagged != "true":
                    raise ValueError(f"{self.name}: a noul rail needs criteria true and false, flagged true")
            case "choice":
                if len(self.criteria) < 2 or self.flagged not in self.criteria:
                    raise ValueError(
                        f"{self.name}: a choice rail needs two or more criteria including {self.flagged!r}"
                    )
            case _:
                raise ValueError(f"{self.name}: question must be noul or choice, got {self.question!r}")
