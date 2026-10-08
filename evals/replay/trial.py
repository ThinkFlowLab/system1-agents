# coding: utf-8
"""One finished trial folder as data: its result, its decisions, the views per step, its frames, and a time axis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.table import model_label, window_s


@dataclass(frozen=True)
class Trial:
    path: Path
    eval_name: str
    model: str
    seed: int
    score: float
    elapsed_s: float
    steps: int
    cost_usd: float | None
    decisions: list[dict[str, Any]]  # the ticks whose act was taken: decisions[k] chose from views[k]
    rejected: int  # ticks with no act: refused by the act tool, or cut off before the act ran (timeout, env error)
    views: list[dict[str, Any]]
    final_state: dict[str, Any]
    extra: dict[str, Any]
    frames: list[Path]


BROWSER_LABEL_CHARS = 60  # a browser step's key on the replay page: the operation and the target's label


def read_trial(path: Path) -> Trial:
    """A trial from its ``result.json`` and ``agent/episode.json`` (frames: the PNGs under ``agent/frames``), or a
    browser run from its logs folder when ``answer.json`` is inside."""
    if (path / "answer.json").is_file():
        return read_browser_run(path)
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    episode = json.loads((path / "agent" / "episode.json").read_text(encoding="utf-8"))
    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    task_name = str(result.get("task_name") or "")
    seed_text = task_name.rsplit("/", 1)[-1]
    ticks = list(episode.get("decisions") or [])
    steps = metadata.get("steps")
    decisions = taken(ticks) if steps is None else taken(ticks)[: int(steps)]
    frames_dir = path / "agent" / "frames"
    return Trial(
        path=path,
        eval_name=str(result.get("source") or path.parent.parent.name),
        model=model_label(result),
        seed=int(seed_text) if seed_text.isdigit() else 0,
        score=float(((result.get("verifier_result") or {}).get("rewards") or {}).get("reward") or 0.0),
        elapsed_s=float(metadata.get("elapsed_s") or window_s(result)),
        steps=len(decisions) if steps is None else int(steps),
        cost_usd=agent_result.get("cost_usd"),
        decisions=decisions,
        rejected=len(ticks) - len(decisions),
        views=list(episode.get("views") or []),
        final_state=dict(episode.get("final_state") or {}),
        extra=dict(episode.get("extra") or {}),
        frames=sorted(frames_dir.glob("*.png")) if frames_dir.is_dir() else [],
    )


def read_browser_run(path: Path) -> Trial:
    """A browser run's logs folder as a trial: ``answer.json`` (the run's result with its agent and model),
    ``decision_ticks.json`` for a decision model or ``chat_calls.json`` for the chat model, and the frames
    ``evals.replay.cast`` saved under ``frames/``.

    A decision-model run's steps sit on the policy's own clock (each tick's ``elapsed_ms``); a chat-model run's
    steps get the call latencies plus the spread overhead, like a tool trial. Both clocks start within half a
    second of the frames' clock, the MCP session's.
    """
    answer = json.loads((path / "answer.json").read_text(encoding="utf-8"))
    model = str(answer.get("model") or answer["slot"])  # runs written by 0.1.0 name the model under "slot"
    elapsed_ms = int(answer["elapsed_ms"])
    extra: dict[str, Any] = {"front": "browser"}
    if model == "llm":
        calls = json.loads((path / "chat_calls.json").read_text(encoding="utf-8"))
        acted = [call for call in calls if call["tool_calls"]]
        decisions = [_chat_decision(step, call) for step, call in enumerate(acted, start=1)]
        candidates: list[dict[str, str]] = [{} for _ in decisions]
    else:
        record = json.loads((path / "decision_ticks.json").read_text(encoding="utf-8"))
        ticks = list(record["ticks"])
        decisions = [_tick_decision(tick, model) for tick in ticks]
        candidates = [dict(tick.get("candidates") or {}) for tick in ticks]
        extra["times"] = [0] + [int(tick["elapsed_ms"]) for tick in ticks[:-1]] + [elapsed_ms] if ticks else [0]
        extra["history"] = list((record.get("report") or {}).get("history") or [])
    terminal = answer.get("terminal") or {}
    final_state = {
        "url": str(terminal.get("url") or ""),
        "title": str(terminal.get("title") or ""),
        "answer": str(answer.get("final") or ""),
    }
    score = 1.0 if answer["ok"] else 0.0
    views = [{"state": {}, "candidates": keys, "done": False, "score": 0.0} for keys in candidates]
    views.append({"state": final_state, "candidates": {}, "done": True, "score": score})
    frames_dir = path / "frames"
    return Trial(
        path=path,
        eval_name=str(answer["agent"]),
        model=model,
        seed=0,
        score=score,
        elapsed_s=round(elapsed_ms / 1000, 1),
        steps=len(decisions),
        cost_usd=(answer.get("usage") or {}).get("cost_usd"),
        decisions=decisions,
        rejected=0,
        views=views,
        final_state=final_state,
        extra=extra,
        frames=sorted(frames_dir.glob("*.png")) if frames_dir.is_dir() else [],
    )


def _tick_decision(tick: dict[str, Any], model: str) -> dict[str, Any]:
    target = str(tick.get("target") or "")
    key = f"{tick['operation']} · {target}" if target else str(tick["operation"])
    return {
        "step": int(tick["tick"]),
        "key": key[:BROWSER_LABEL_CHARS],
        "ms": int(tick["decision_ms"]),
        "confidence": float(tick["confidence"]),
        "probabilities": dict(tick.get("probabilities") or {}),
        "source": model,
    }


def _chat_decision(step: int, call: dict[str, Any]) -> dict[str, Any]:
    """One chat call that issued tool calls: its first tool, without the MCP server prefix, and what it named."""
    tool = str(call["tool_calls"][0]).rsplit("browser_", 1)[-1]
    arguments = list(call.get("tool_args") or [])
    named = _argument(str(arguments[0])) if arguments else ""
    key = f"{tool} · {named}" if named else tool
    return {
        "step": step,
        "key": key[:BROWSER_LABEL_CHARS],
        "ms": int(call["ms"]),
        "confidence": 0.0,
        "probabilities": {},
        "source": "llm",
    }


def _argument(text: str) -> str:
    """What a tool call named, from its JSON arguments: the element, the typed text, the URL, the query or the key."""
    try:
        arguments = json.loads(text)
    except ValueError:
        return ""
    if not isinstance(arguments, dict):
        return ""
    for name in ("element", "text", "url", "query", "key"):
        if arguments.get(name):
            return str(arguments[name])
    return ""


def taken(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ticks the act tool accepted; a slot model's tick has no flag and counts as accepted."""
    return [tick for tick in ticks if tick.get("accepted", True)]


def time_axis(trial: Trial) -> list[int]:
    """Milliseconds at which each view holds: the times the run recorded (``extra["times"]``, a browser run on the
    policy's clock), else the decision latencies plus the loop's overhead spread evenly."""
    if "times" in trial.extra:
        return [int(t) for t in trial.extra["times"]]
    if trial.steps == 0:
        return [0]
    latencies = [int(decision.get("ms") or 0) for decision in trial.decisions][: trial.steps]
    latencies += [0] * (trial.steps - len(latencies))
    overhead = max(0.0, trial.elapsed_s * 1000 - sum(latencies)) / trial.steps
    times, clock = [0], 0.0
    for latency in latencies:
        clock += latency + overhead
        times.append(round(clock))
    return times


def views_of(trial: Trial) -> list[dict[str, Any]]:
    """The recorded views, or what can be rebuilt for a trial recorded before views were kept."""
    if trial.views:
        return trial.views
    match trial.eval_name:
        case "blackjack":
            return _blackjack_views(trial)
        case _:
            empty = {"state": {}, "candidates": {}, "done": False, "score": 0.0}
            return [dict(empty) for _ in range(trial.steps)] + [_final_view(trial)]


def _final_view(trial: Trial) -> dict[str, Any]:
    return {"state": trial.final_state, "candidates": {}, "done": True, "score": trial.score}


def _blackjack_views(trial: Trial) -> list[dict[str, Any]]:
    """The player's hand before each decision: the first two cards, plus one per ``hit`` played so far."""
    cards = list(trial.final_state.get("player_cards") or [])
    showing = list(trial.final_state.get("dealer_showing") or [])
    views, hits = [], 0
    for decision in trial.decisions[: trial.steps]:
        state = {"player_cards": cards[: 2 + hits], "dealer_showing": showing}
        views.append({"state": state, "candidates": {"hit": "", "stand": ""}, "done": False, "score": 0.0})
        hits += decision.get("key") == "hit"
    return views + [_final_view(trial)]
