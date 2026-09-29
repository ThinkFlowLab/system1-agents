# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Modifications Copyright 2026 ThinkFlowLab
# SPDX-License-Identifier: Apache-2.0

"""``BrowserDecisionModel``: a decision model in the browser subagent's model slot.

On a browser turn (the tool list contains ``browser_click``) it settles and probes the page through
the bound ``BrowserAgentRuntime``, asks the model (every head of the action space in one ``decide_many``,
re-asked once on an unusable answer), and returns exactly one ``browser_*`` tool call. Every other call
(summaries, value generation) goes to the wrapped chat model. Jev over HTTP and Laya in process fill the
model_name alike.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from urllib.parse import urlparse
from uuid import uuid4

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError, build_error
from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk, ToolCall
from openjiuwen.core.foundation.llm.model import Model
from openjiuwen.harness.tools.browser_move.utils.parsing import extract_json_object
from s1a.browser import prompts
from s1a.browser.action_space import (
    NONE_VALUE,
    Move,
    build_action_space,
    build_observation,
    build_questions,
    interpret,
    top_probabilities,
)
from s1a.browser.probe_js import POLICY_PROBE_JS, STAMP_ATTRIBUTE
from s1a.decision_models import DecisionModel, Image, Observation
from s1a.spec import BrowserAgentSpec

BROWSER_TURN_TOOL = "browser_click"
# The runtime registers MCP tools under a server prefix (mcp_<server>_browser_click); match by suffix.
_BROWSER_TOOL_SUFFIXES = (
    BROWSER_TURN_TOOL,
    "browser_type",
    "browser_select_option",
    "browser_press_key",
    "browser_navigate",
    "browser_run_code_unsafe",
)
BATCH_TOOL = "browser_run_code_unsafe"
DECISION_ATTEMPTS = 2  # the same request once more when an answer fails validation; a transport error is final
TOP_CANDIDATES = 6  # keys kept per tick for the replay's probability bars
_OPERATION_TOOLS = {
    "TYPE_TEXT": "browser_type",
    "PRESS_ENTER": "browser_press_key",
    "SELECT": "browser_select_option",
    "SCROLL_DOWN": "browser_press_key",
    "SCROLL_UP": "browser_press_key",
}
BATCH_ACTION_TIMEOUT_MS = (
    2000  # a stamp lost to a re-render fails fast; the returned probe re-stamps and Jev re-decides
)
MAX_CONSECUTIVE_WAITS = 5
FINAL_TEXT_CHARS = 20000  # the whole-page text the answer step reads on DONE
ANSWER_MAX_TOKENS = 400
SUMMARY_TEXT_CHARS = (
    2000  # viewport text kept in the terminal summary for diagnosis; the harness caps the summary at 8000
)
MAX_URL_VISITS = 3  # the fourth arrival at one URL ends the run: the policy is circling
MAX_PREFETCHED_FIELDS = 8
PROBE_SETTLE_MS = 500  # must equal probe_js.py's JS default so an un-escalated probe's timing is unchanged
PROBE_QUIET_MS = 60  # must equal probe_js.py's JS default DOM-quiet window
ACTION_SETTLE_START_MS = 250  # first in-page wait for an action whose effect has not shown yet
ACTION_SETTLE_BUDGET_MS = 1000  # total in-page wait one action gets before the model sees an unchanged page
MAX_PROBE_SETTLE_MS = 1500  # keeps load(3s)+settle+1s JS lastResort >=1s under the 30s transport request timeout
WAIT_SETTLE_BUDGET_MS = 3000  # total in-page settle time one WAIT streak may spend before the step gives up
URL_RE = re.compile(r"https?://[^\s'\"<>]+")
# The viewport as PNG, for a decision model that reads images; one run-code call through the probe's own executor.
# A background tab renders no frames, so the page is brought to the front first; the capture waits 15 s at most.
SCREENSHOT_JS = (
    "async (page) => { await page.bringToFront(); "
    'const png = await page.screenshot({type: "png", timeout: 15000}); '
    "return JSON.stringify({png: png.toString('base64')}); }"
)
_SCREENSHOT_RE = re.compile(r'png\\*"\s*:\s*\\*"([A-Za-z0-9+/=]+)')


@dataclass(frozen=True)
class BrowserPolicy:
    """The run-time switches of the Jev browser policy, set per run by flags or the environment, never by the spec."""

    prefetch_values: bool  # generate a typed value for every editable field as soon as a probe shows it
    batch_actions: bool  # each action and the next probe in one browser_run_code_unsafe call; needs unsafe_dev
    goal_value_cache: bool  # offer values extracted from the goal to Jev as a choice head


@dataclass
class _Run:
    """One task's worth of Jev state, from the goal turn to a terminal ``DONE``/``BLOCKED``.

    A fresh ``_Run`` is the unit of isolation between tasks served by the same model instance: its
    goal, history, pending action and prefetched values never leak into the run that follows it.
    """

    goal: str
    started_at: float
    history: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    prefetched: dict[str, asyncio.Task[str | None]] = field(default_factory=dict)
    values: list[str] = field(default_factory=list)
    values_task: asyncio.Task[list[str]] | None = None
    tick: int = 0
    consecutive_waits: int = 0
    settle_spent_ms: int = 0  # in-page wait spent since the page last progressed; resets on progress and on an action
    ticks: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    answer: str = ""
    url_visits: dict[str, int] = field(default_factory=dict)
    last_url: str = ""  # the URL of the latest good probe; a probe landing elsewhere counts a visit to its URL
    token: str = field(default_factory=lambda: uuid4().hex[:6])  # keeps tool-call ids unique across runs


class BrowserDecisionModel(Model):
    """A ``Model`` whose browser-turn answers come from a decision model instead of a chat model."""

    def __init__(
        self,
        spec: BrowserAgentSpec,
        policy: BrowserPolicy,
        fallback: Model,
        *,
        decision_model: DecisionModel,
        value_model: Model | None,
    ) -> None:
        """``policy.goal_value_cache`` offers goal-extracted values to Jev as a choice head; off means every typed
        value comes from the chat model given the field and page context. ``policy.prefetch_values`` makes that
        chat-model call for every editable field as soon as a probe shows it, one per field per document,
        whether or not Jev ever types there; off makes it only when TYPE_TEXT is chosen. ``policy.batch_actions``
        sends each action and the next probe as one ``browser_run_code_unsafe`` call when that tool is
        offered, skipping the runtime's target validation and one transport round trip per step.
        ``value_model`` answers the typed-value calls when given; the fallback chat model otherwise.
        ``decision_model`` decides every browser step; its ``name`` (``jev``, ``laya``) is the ``--model`` value."""
        super().__init__(fallback.model_client_config, fallback.model_config)
        self._spec = spec
        self._fallback = fallback
        self._value_model = value_model or fallback  # typed values want a fast small model; chat turns do not
        self._goal_value_cache = policy.goal_value_cache
        self._prefetch_enabled = policy.prefetch_values
        self._batch_actions = policy.batch_actions
        self._language = spec.language
        self._decision_model = decision_model
        self._runtime: Any = None
        self._tool_names: dict[str, str] = {}
        self._run: _Run | None = None

    # -- wiring -------------------------------------------------------------

    def bind_runtime(self, runtime: Any) -> None:
        self._runtime = runtime

    # -- public accessors (delegate to the current/most recent run) ---------

    @property
    def ticks(self) -> list[dict[str, Any]]:
        return self._run.ticks if self._run is not None else []

    @property
    def started_at(self) -> float:
        return self._run.started_at if self._run is not None else 0.0

    # -- Model surface ------------------------------------------------------

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        names = self._browser_tool_names(tools)
        if BROWSER_TURN_TOOL not in names:
            return await self._fallback.invoke(messages, tools=tools, **kwargs)
        self._tool_names = names
        return await self._decide_message(messages)

    async def stream(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AsyncIterator[AssistantMessageChunk]:
        names = self._browser_tool_names(tools)
        if BROWSER_TURN_TOOL not in names:
            async for chunk in self._fallback.stream(messages, tools=tools, **kwargs):
                yield chunk
            return
        self._tool_names = names
        message = await self._decide_message(messages)
        yield AssistantMessageChunk(
            content=message.content, tool_calls=message.tool_calls, finish_reason=message.finish_reason
        )

    # -- one tick -----------------------------------------------------------

    @staticmethod
    def _browser_tool_names(tools: Any) -> dict[str, str]:
        """Map each browser tool suffix to the name the runtime registered it under."""
        names: dict[str, str] = {}
        for tool in tools or []:
            name = str(getattr(tool, "name", "") or "")
            for suffix in _BROWSER_TOOL_SUFFIXES:
                if name == suffix or name.endswith("_" + suffix):
                    names.setdefault(suffix, name)
        return names

    @staticmethod
    def _cancel_run_tasks(run: _Run) -> None:
        """Cancel a superseded run's background work so it cannot leak into the run that follows it."""
        if run.values_task is not None:
            run.values_task.cancel()
        for task in run.prefetched.values():
            task.cancel()
        run.prefetched.clear()

    async def _decide_message(self, messages: Any) -> AssistantMessage:
        if self._runtime is None:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="BrowserDecisionModel.bind_runtime() was not called"
            )
        goal = self._goal_from(messages)
        run = self._run
        if run is None or run.finished or goal != run.goal:
            if run is not None:
                self._cancel_run_tasks(run)
            run = _Run(goal=goal, started_at=time.perf_counter())
            self._run = run
            if self._goal_value_cache:
                run.values_task = asyncio.create_task(self._extract_values(goal))
            await self._decision_model.warm()
            url = URL_RE.search(goal)
            if url:
                nav_args = {"url": url.group(0).rstrip(".,)")}
                return self._tool_message(run, "browser_navigate", nav_args, label="navigate")

        batched = self._batched_snapshot(run, messages)
        snapshot, probe_ms = await (self._batched_probe(run, batched) if batched else self._probe(run))
        settle_probes, settle_ms = 0, 0
        while True:  # a settle probe can set the guards too, so they are read on every pass
            if snapshot.get("stalled"):
                reason = f"{self._spec.budget.stall_after} actions without page change"
                return await self._final(run, "BLOCKED", snapshot, reason)
            if snapshot.get("oscillating"):
                return await self._final(run, "BLOCKED", snapshot, "oscillating between two pages")
            if snapshot.get("revisiting"):
                return await self._final(run, "BLOCKED", snapshot, "reached the same page four times")
            space = build_action_space(snapshot, run.history)
            for operation, tool in _OPERATION_TOOLS.items():
                if tool not in self._tool_names:
                    space.operations = [op for op in space.operations if op != operation]
                    space.heads.pop(operation, None)
            acted = next((h for h in reversed(run.history) if h["kind"] != "wait"), None)
            if acted is not None and acted["kind"] == "scroll" and acted["page_changed"] is False:
                space.operations = [op for op in space.operations if op != acted["action"]]  # the page end was reached
            values = run.values if run.values_task is None or run.values_task.done() else []
            if run.values_task is not None and run.values_task.done() and not run.values:
                try:
                    run.values = run.values_task.result()
                except Exception:  # noqa: BLE001 - a failed goal-value extraction degrades to an empty value list
                    logger.warning("[BrowserDecisionModel] goal-value extraction failed", exc_info=True)
                    run.values = []
                values = run.values
            observation = build_observation(space, snapshot, run.history)
            screenshot_ms = None
            if self._decision_model.supports_images:
                screenshot, screenshot_ms = await self._screenshot()
                if screenshot is not None:
                    observation = Observation(observation.state, images=(screenshot,))
            questions = build_questions(
                space, goal=run.goal, values=values, rules=self._spec.rules, language=self._language
            )
            try:
                decision = await self._decision_model.decide_many(observation, questions, attempts=DECISION_ATTEMPTS)
            except BaseError as exc:
                logger.warning("[BrowserDecisionModel] decision failed: %s", exc, exc_info=True)
                return await self._final(run, "BLOCKED", snapshot, f"decision failed: {exc}")
            move = interpret(decision, space)
            probabilities, candidates = top_probabilities(decision, move, space, TOP_CANDIDATES)
            run.tick += 1
            record = {
                "tick": run.tick,
                "probe_ms": probe_ms,
                "settle_probes": settle_probes,
                "settle_ms": settle_ms,
                "action_settle_probes": snapshot.get("action_settle_probes", 0),
                "action_settle_ms": snapshot.get("action_settle_ms", 0),
                "decision_ms": move.latency_ms,  # the key is the artifact contract: replay and the table read it
                "input_tokens": move.usage.input_tokens,
                "output_tokens": move.usage.output_tokens,
                "operation": move.operation,
                "target": move.candidate.item.get("label") if move.candidate else None,
                "confidence": round(move.confidence, 3),
                "elements": len(space.elements),
                "elapsed_ms": round((time.perf_counter() - run.started_at) * 1000),
                "probabilities": probabilities,  # the replay's bars: the settled head's top keys
                "candidates": candidates,
            }
            if screenshot_ms is not None:  # an image-reading model: the capture is part of the step's cost
                record["screenshot_ms"] = screenshot_ms
            run.ticks.append(record)
            if move.operation == "WAIT":
                run.consecutive_waits += 1
                run.history.append({"action": "wait", "kind": "wait", "text": None, "page_changed": None})
                if run.consecutive_waits > MAX_CONSECUTIVE_WAITS:
                    return await self._final(run, "BLOCKED", snapshot, "waited without progress")
                snapshot, settle_probes, settle_ms, progressed = await self._settle_wait(run, snapshot)
                if not progressed:
                    record["settle_probes"] += settle_probes
                    record["settle_ms"] += settle_ms
                    return await self._final(run, "BLOCKED", snapshot, "waited without progress")
                probe_ms = settle_ms
                continue
            run.consecutive_waits = 0
            run.settle_spent_ms = 0
            if move.operation == "DONE":
                run.answer = await self._answer(run, snapshot)
                return await self._final(run, "DONE", snapshot, "")
            if move.operation == "BLOCKED":
                return await self._final(run, "BLOCKED", snapshot, "")
            return await self._act(run, move, snapshot, record)

    async def _settle_wait(self, run: _Run, asked_snapshot: dict[str, Any]) -> tuple[dict[str, Any], int, int, bool]:
        """Spend a WAIT verdict in-page instead of paying another decisions request.

        the model's answer on an unchanged page cannot change until the page does, so a decision
        request spent re-asking it is pure latency; re-probing costs nothing by comparison. Escalates
        ``settle_ms`` by doubling from ``PROBE_SETTLE_MS`` (clamped to ``MAX_PROBE_SETTLE_MS``) on every
        retry, and stops once either the probed ``page_key`` differs from the one the model just saw
        or ``WAIT_SETTLE_BUDGET_MS`` is spent -- a page that never goes DOM-quiet (a CSS animation, a
        ticking clock) would otherwise burn its full settle cap forever.

        The budget (``run.settle_spent_ms``) counts the wait since the page last progressed, across WAIT
        verdicts: a per-verdict budget would be re-granted on every WAIT and let ``MAX_CONSECUTIVE_WAITS``
        verdicts stretch one stuck step to minutes, while a page rendering in stages earns a fresh budget
        with each stage. The returned flag is ``False`` once the budget is gone, and the caller must treat
        it as terminal -- re-asking the model about a snapshot it already answered WAIT on buys the
        same answer at full request cost. ``MAX_CONSECUTIVE_WAITS`` bounds the other shape of stuck, where
        ``page_key`` keeps flipping but the page stays unactionable. A probe that failed shows nothing and
        counts as unchanged.
        """
        asked_page_key = asked_snapshot.get("page_key")
        snapshot = asked_snapshot
        settle_ms = PROBE_SETTLE_MS
        probes = 0
        measured_ms = 0
        while run.settle_spent_ms < WAIT_SETTLE_BUDGET_MS:
            requested_ms = min(settle_ms, WAIT_SETTLE_BUDGET_MS - run.settle_spent_ms)
            snapshot, probe_ms = await self._probe(run, settle_ms=requested_ms)
            probes += 1
            measured_ms += probe_ms
            run.settle_spent_ms += requested_ms
            if _progressed(snapshot, asked_page_key):
                run.settle_spent_ms = 0
                return snapshot, probes, measured_ms, True
            settle_ms = min(settle_ms * 2, MAX_PROBE_SETTLE_MS)
        return snapshot, probes, measured_ms, False

    async def _probe(self, run: _Run, *, settle_ms: int = PROBE_SETTLE_MS) -> tuple[dict[str, Any], int]:
        started = time.perf_counter()
        pending, run.pending = run.pending, None
        after = {"kind": pending["kind"], "node": pending["node"]} if pending is not None else None
        snapshot = await self._raw_probe(settle_ms=settle_ms, quiet_ms=PROBE_QUIET_MS, after=after)
        return await self._finish_probe(run, snapshot, pending, started)

    async def _batched_probe(self, run: _Run, snapshot: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Use the probe a batched action call already returned instead of probing again."""
        started = time.perf_counter()
        pending, run.pending = run.pending, None
        if snapshot.get("error"):
            logger.warning("[BrowserDecisionModel] batched action reported %s", snapshot["error"])
        for item in snapshot.get("elements") or []:
            if isinstance(item, dict) and item.get("node") is not None:
                item.setdefault("target_id", f"batch:{item['node']}")  # unregistered: batched steps act by selector
        return await self._finish_probe(run, snapshot, pending, started)

    async def _finish_probe(
        self, run: _Run, snapshot: dict[str, Any], pending: dict[str, Any] | None, started: float
    ) -> tuple[dict[str, Any], int]:
        if pending is not None:
            changed = _progressed(snapshot, pending["page_key"])
            if not changed:
                snapshot, changed = await self._settle_action(run, snapshot, pending["page_key"])
            pending["entry"]["page_changed"] = changed
            limit = self._spec.budget.stall_after
            recent = run.history[-limit:] if limit else []
            if (
                limit
                and len(recent) == limit
                and all(h["page_changed"] is False and h["kind"] != "wait" for h in recent)
            ):
                snapshot["stalled"] = True
            acted = [h for h in run.history if h["kind"] != "wait"][-4:]
            labels = [h["action"] for h in acted]
            if (
                len(labels) == 4
                and labels[0] != labels[1]
                and labels[:2] == labels[2:]
                and all(h["page_changed"] is False for h in acted[2:])
            ):
                snapshot["oscillating"] = True
        url = str(snapshot.get("url") or "")
        if not snapshot.get("error") and url != run.last_url:
            run.last_url = url
            run.url_visits[url] = run.url_visits.get(url, 0) + 1
            if run.url_visits[url] > MAX_URL_VISITS:
                snapshot["revisiting"] = True
        self._prefetch_values(run, snapshot)
        return snapshot, round((time.perf_counter() - started) * 1000)

    async def _screenshot(self) -> tuple[Image | None, int]:
        """The viewport as PNG and the milliseconds the capture took. ``None`` when it fails: the step goes on
        without the picture and the decision model says whether it can decide without one."""
        started = time.perf_counter()
        executor = getattr(self._runtime, "code_executor", None)
        raw: Any = None
        if callable(executor):
            try:
                raw = await executor(SCREENSHOT_JS)
            except Exception:  # noqa: BLE001 - a failed capture degrades to a text-only observation
                logger.warning("[BrowserDecisionModel] screenshot failed", exc_info=True)
        ms = round((time.perf_counter() - started) * 1000)
        found = _SCREENSHOT_RE.search(json.dumps(raw, default=str)) if raw is not None else None
        if found is None:
            logger.warning("[BrowserDecisionModel] no screenshot in the run-code result")
            return None, ms
        return Image(base64.b64decode(found.group(1))), ms

    async def _raw_probe(self, *, settle_ms: int, quiet_ms: int, after: dict[str, Any] | None) -> dict[str, Any]:
        params = {
            "stamp_attribute": STAMP_ATTRIBUTE,
            "after": after,
            "max_items": 250,
            "settle_ms": settle_ms,
            "quiet_ms": quiet_ms,
        }
        snapshot = await self._runtime.probe_for_policy(POLICY_PROBE_JS, params)
        if snapshot.get("error"):
            logger.warning("[BrowserDecisionModel] policy probe reported %s", snapshot["error"])
        if snapshot.get("visibility") == "hidden" and await self._runtime.activate_page(str(snapshot.get("url") or "")):
            logger.info("[BrowserDecisionModel] activated hidden tab %s", snapshot.get("url"))
            snapshot = await self._runtime.probe_for_policy(POLICY_PROBE_JS, params)
        return snapshot

    async def _answer(self, run: _Run, snapshot: dict[str, Any]) -> str:
        """One chat call on DONE: the goal and the whole page's text in, a short answer out; "" when it fails."""
        try:
            page = await self._answer_probe(snapshot)
            reply = await self._fallback.invoke(
                [
                    {"role": "system", "content": prompts.ANSWER_RULES[self._language]},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"task": run.goal, "page_text": str(page.get("text") or "")}, ensure_ascii=False
                        ),
                    },
                ],
                max_tokens=ANSWER_MAX_TOKENS,
                temperature=0,
            )
        except Exception:  # noqa: BLE001 - the summary must still reach the harness; DONE without an answer fails there
            logger.warning("[BrowserDecisionModel] answer call failed", exc_info=True)
            return ""
        return str(reply.content or "").strip()

    async def _answer_probe(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """One more probe with the whole document's text; the decision snapshot when the probe fails."""
        params = {
            "stamp_attribute": STAMP_ATTRIBUTE,
            "after": None,
            "max_items": 250,
            "settle_ms": PROBE_SETTLE_MS,
            "quiet_ms": PROBE_QUIET_MS,
            "all_text": True,
            "text_chars": FINAL_TEXT_CHARS,
        }
        full = await self._runtime.probe_for_policy(POLICY_PROBE_JS, params)
        if full.get("error"):
            logger.warning("[BrowserDecisionModel] answer probe reported %s", full["error"])
            return snapshot
        return full

    async def _settle_action(self, run: _Run, snapshot: dict[str, Any], before_key: Any) -> tuple[dict[str, Any], bool]:
        """Wait in-page for an action's delayed effect (a closing dialog, a late re-render) before deciding.

        A probe right after the action can return before the effect shows; the model would then be
        asked about the page it already acted on and answer WAIT at full request cost. The waits double
        from ``ACTION_SETTLE_START_MS`` and are wall-clock waits (``quiet_ms`` equals the window, so a quiet
        DOM does not cut them short). ``ACTION_SETTLE_BUDGET_MS`` bounds the penalty for an action that
        truly did nothing, and the time counts against the WAIT streak budget so one step never settles
        longer than ``WAIT_SETTLE_BUDGET_MS`` in total.
        """
        wait_ms, spent, probes = ACTION_SETTLE_START_MS, 0, 0
        while spent < ACTION_SETTLE_BUDGET_MS:
            requested = min(wait_ms, ACTION_SETTLE_BUDGET_MS - spent)
            snapshot = await self._raw_probe(settle_ms=requested, quiet_ms=requested, after=None)
            probes += 1
            spent += requested
            run.settle_spent_ms += requested
            if _progressed(snapshot, before_key):
                break
            wait_ms = min(wait_ms * 2, MAX_PROBE_SETTLE_MS)
        snapshot["action_settle_probes"] = probes
        snapshot["action_settle_ms"] = spent
        return snapshot, _progressed(snapshot, before_key)

    async def _act(self, run: _Run, move: Move, snapshot: dict[str, Any], record: dict[str, Any]) -> AssistantMessage:
        candidate = move.candidate
        generation_id = str(snapshot.get("generation_id") or "")
        text: str | None = None
        match move.operation:
            case "CLICK":
                assert candidate is not None, "CLICK without a candidate"
                name, args, kind = (
                    "browser_click",
                    {"generation_id": generation_id, "target_id": candidate.item["target_id"]},
                    "click",
                )
            case "TYPE_TEXT":
                assert candidate is not None, "TYPE_TEXT without a candidate"
                text, record["value_source"], record["value_ms"], record["value_error"] = await self._value_for(
                    run, move, snapshot
                )
                if text is None:
                    reason = f"no value for field {candidate.item.get('label')!r}: {record['value_error']}"
                    return await self._final(run, "BLOCKED", snapshot, reason)
                name, kind = "browser_type", "fill"
                args = {
                    "generation_id": generation_id,
                    "target_id": candidate.item["target_id"],
                    "text": text,
                }
            case "PRESS_ENTER":
                assert candidate is not None, "PRESS_ENTER without a candidate"
                # browser_press_key takes a key alone and sends it to the focused element, the field just typed into;
                # the batched path presses on the field's own locator and does not depend on focus
                name, args, kind = "browser_press_key", {"key": "Enter"}, "submit"
            case "SELECT":
                assert candidate is not None, "SELECT without a candidate"
                name, kind = "browser_select_option", "select"
                args = {
                    "generation_id": generation_id,
                    "target_id": candidate.item["target_id"],
                    "values": [candidate.option_label],
                }
            case "SCROLL_DOWN":
                name, args, kind = "browser_press_key", {"key": "PageDown"}, "scroll"
            case "SCROLL_UP":
                name, args, kind = "browser_press_key", {"key": "PageUp"}, "scroll"
            case _:
                return await self._final(run, "BLOCKED", snapshot, f"unsupported operation {move.operation}")
        label = candidate.item.get("label", "") if candidate else move.operation
        if candidate and candidate.option_label:
            label = f"{label} → {candidate.option_label}"
        entry = {"action": label, "kind": kind, "text": text, "page_changed": None}
        run.history.append(entry)
        run.pending = {
            "kind": kind,
            "node": candidate.item.get("node") if candidate else None,
            "page_key": snapshot.get("page_key"),
            "entry": entry,
        }
        record["text"] = text
        if self._batch_actions and BATCH_TOOL in self._tool_names:
            message = self._tool_message(
                run, BATCH_TOOL, {"code": self._batched_code(name, args, candidate, kind)}, label=label
            )
            assert message.tool_calls, "a tool message without its call"
            run.pending["batched_call_id"] = message.tool_calls[0].id
            return message
        return self._tool_message(run, name, args, label=label)

    @staticmethod
    def _batched_code(name: str, args: dict[str, Any], candidate: Any, kind: str) -> str:
        """Playwright code for one action followed by the policy probe, returned as the call's result.

        The target is the stamped selector from the last probe. A re-render between probe and action
        drops the stamp, so a missing target re-probes (which re-stamps) and re-finds the element by role
        and label. An action that still fails returns the probe with ``error`` set instead of throwing; the
        policy settles it in-page like an unchanged page, then re-decides on fresh stamps. A scroll has no
        target and only presses its key.
        """
        item = candidate.item if candidate else {}
        selector = item.get("selector_hint") or f'[{STAMP_ATTRIBUTE}="{item.get("node")}"]'
        timeout = f"{{timeout: {BATCH_ACTION_TIMEOUT_MS}}}"
        match name:
            case "browser_click":
                action = f"await target.click({timeout});"
            case "browser_type":
                action = f"await target.fill({json.dumps(args['text'])}, {timeout});"
            case "browser_select_option":
                action = f"await target.selectOption({{label: {json.dumps(args['values'][0])}}}, {timeout});"
            case "browser_press_key" if kind == "submit":
                action = f"await target.press({json.dumps(args['key'])}, {timeout});"  # focuses the field, then submits
            case _:
                action = f"await page.keyboard.press({json.dumps(args['key'])});"
        params = {
            "stamp_attribute": STAMP_ATTRIBUTE,
            "after": {"kind": kind, "node": item.get("node")},
            "max_items": 250,
            "settle_ms": PROBE_SETTLE_MS,
            "quiet_ms": PROBE_QUIET_MS,
        }
        recover = {**params, "after": None}
        locate = (
            f"  let target = page.locator({json.dumps(selector)});\n"
            "  if (await target.count() === 0) {\n"
            f"    const fresh = await page.evaluate(probe, {json.dumps(recover)});\n"
            f"    const hit = (fresh.elements || []).find((e) => e.label === {json.dumps(item.get('label'))} && e.role === {json.dumps(item.get('role'))});\n"
            f"    if (!hit) return {{...fresh, error: 'target not found after re-stamp'}};\n"
            f"    target = page.locator('[{STAMP_ATTRIBUTE}=\"' + hit.node + '\"]');\n"
            "  }\n"
        )
        return (
            "async (page) => {\n"
            f"  const probe = {POLICY_PROBE_JS};\n"
            f"{locate if candidate else ''}"
            "  try {\n"
            f"    {action}\n"
            "  } catch (err) {\n"
            f"    const fresh = await page.evaluate(probe, {json.dumps(recover)});\n"
            "    return {...fresh, error: String(err && err.message || err).slice(0, 300)};\n"
            "  }\n"
            f"  return await page.evaluate(probe, {json.dumps(params)});\n"
            "}"
        )

    @staticmethod
    def _batched_snapshot(run: _Run, messages: Any) -> dict[str, Any] | None:
        """The probe carried by the tool result of the last batched call, if the conversation has it."""
        call_id = (run.pending or {}).get("batched_call_id")
        if not call_id:
            return None
        for message in reversed(list(messages or [])):
            is_dict = isinstance(message, dict)
            if (message.get("tool_call_id") if is_dict else getattr(message, "tool_call_id", None)) != call_id:
                continue
            content = message.get("content") if is_dict else getattr(message, "content", None)
            parsed = extract_json_object(content if isinstance(content, str) else json.dumps(content))
            if isinstance(parsed, dict) and isinstance(parsed.get("elements"), list):
                return parsed
            logger.warning("[BrowserDecisionModel] batched call %s returned no probe; probing again", call_id)
            return None
        logger.warning("[BrowserDecisionModel] batched call %s has no tool result in the conversation", call_id)
        return None

    @staticmethod
    def _document_key(snapshot: dict[str, Any]) -> str:
        url = urlparse(str(snapshot.get("url") or ""))
        return f"{url.netloc}{url.path}"

    @classmethod
    def _field_key(cls, item: dict[str, Any], snapshot: dict[str, Any]) -> str:
        return f"{cls._document_key(snapshot)}|{item.get('label', '')}"

    def _prefetch_values(self, run: _Run, snapshot: dict[str, Any]) -> None:
        """Start value generation for every editable field now, so a later TYPE_TEXT does not wait for it.

        One call per field per document: the key is the label under the URL path, so a value generated on the
        first probe stays usable while the page fills in and re-stamps, and a navigation cancels the previous
        document's outstanding calls. Prefilled fields are included: a wrong default (the site's guessed origin)
        is replaced as often as an empty field is filled.
        """
        if not self._prefetch_enabled:
            return
        scope = self._document_key(snapshot) + "|"
        for key in [key for key in run.prefetched if not key.startswith(scope)]:
            run.prefetched.pop(key).cancel()
        for item in snapshot.get("elements") or []:
            if not item.get("editable") or not item.get("target_id"):
                continue
            key = self._field_key(item, snapshot)
            if key not in run.prefetched and len(run.prefetched) < MAX_PREFETCHED_FIELDS:
                run.prefetched[key] = asyncio.create_task(self._generate_value(run, item, snapshot))

    async def _value_for(
        self, run: _Run, move: Move, snapshot: dict[str, Any]
    ) -> tuple[str | None, str, int, str | None]:
        """The value to type, its source, the wait in ms, and why there is none: the value model's exception or its
        empty answer. A missing value ends the run as BLOCKED with that reason in the summary and the tick."""
        started = time.perf_counter()
        if move.value_choice and move.value_choice != NONE_VALUE:
            return move.value_choice, "cache", round((time.perf_counter() - started) * 1000), None
        field_item = move.candidate.item if move.candidate else {}
        task = run.prefetched.pop(self._field_key(field_item, snapshot), None)
        source = "prefetch" if task is not None else "llm"
        error: str | None = None
        try:
            value = await (task if task is not None else self._generate_value(run, field_item, snapshot))
        except Exception as exc:  # noqa: BLE001 - a failed value call degrades to _act's "no value" BLOCKED path
            logger.warning("[BrowserDecisionModel] value generation failed", exc_info=True)
            value, error = None, f"{type(exc).__name__}: {exc}"
        if value is None and error is None:
            error = "value model returned no usable text"
        return value, source, round((time.perf_counter() - started) * 1000), error

    async def _generate_value(self, run: _Run, field_item: dict[str, Any], snapshot: dict[str, Any]) -> str | None:
        context = {
            "goal": run.goal,
            "field": {k: field_item.get(k) for k in ("label", "role", "value", "region")},
            "page": {"title": snapshot.get("title"), "text": str(snapshot.get("text", ""))[:6000]},
            "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in run.history[-6:]],
        }
        reply = await self._value_model.invoke(
            [
                {"role": "system", "content": prompts.VALUE_GENERATION[self._language]},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            max_tokens=256,
            temperature=0,
            response_format={"type": "json_object"},
        )
        value = _json_field(reply.content, "text")
        return value.strip() if isinstance(value, str) and value.strip() else None

    async def _extract_values(self, goal: str) -> list[str]:
        try:
            reply = await self._fallback.invoke(
                [
                    {"role": "system", "content": prompts.VALUE_EXTRACTION[self._language]},
                    {"role": "user", "content": goal},
                ],
                max_tokens=256,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception:  # noqa: BLE001 - value cache is an optimisation; the per-field LLM path remains
            logger.warning(
                "[BrowserDecisionModel] value extraction failed; falling back to per-field generation", exc_info=True
            )
            return []
        values = _json_field(reply.content, "values")
        return [str(v) for v in values if str(v).strip()][:24] if isinstance(values, list) else []

    # -- messages -----------------------------------------------------------

    def _tool_message(self, run: _Run, name: str, args: dict[str, Any], *, label: str) -> AssistantMessage:
        call = ToolCall(
            id=f"{self._decision_model.name}-{run.token}-{run.tick}-{len(run.history)}",
            type="function",
            name=self._tool_names.get(name, name),
            arguments=json.dumps(args),
        )
        return AssistantMessage(content="", tool_calls=[call], finish_reason="tool_calls")

    async def _final(self, run: _Run, status: str, snapshot: dict[str, Any], reason: str) -> AssistantMessage:
        self._cancel_run_tasks(run)  # no value call outlives its run
        if status == "BLOCKED" and not run.answer and any(h.get("page_changed") for h in run.history):
            run.answer = await self._answer(run, snapshot)  # the page the run reached may already hold the answer
        run.finished = True
        summary = {
            "status": status,
            "reason": reason,
            "url": snapshot.get("url"),
            "title": snapshot.get("title"),
            "steps": len([h for h in run.history if h["kind"] != "wait"]),
            "elapsed_ms": round((time.perf_counter() - run.started_at) * 1000),
            "page_text": str(snapshot.get("text", ""))[:SUMMARY_TEXT_CHARS],
            "answer": run.answer,
        }
        return AssistantMessage(content=json.dumps(summary, ensure_ascii=False), finish_reason="stop")

    @staticmethod
    def _goal_from(messages: Any) -> str:
        """The first user message is the task; later user messages (image notes, rail inserts) are data."""
        for message in messages or []:
            role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
            if role == "user":
                content = getattr(message, "content", None) or (
                    message.get("content") if isinstance(message, dict) else ""
                )
                goal = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                return goal.strip()
        return ""

    def report(self) -> dict[str, Any]:
        """Timing summary in the shape of jev-ultrafast's performance notes, for the current/most recent run.

        ``decisions``, ``median_decision_ms`` and ``jev_input_tokens`` are the same for every decision model: the
        artifact readers (``evals/replay``, ``evals/table.py``) know them by those keys.
        """
        run = self._run
        if run is None:
            return {
                "elapsed_ms": 0,
                "decisions": 0,
                "interactions": 0,
                "waits": 0,
                "median_decision_ms": 0,
                "jev_input_tokens": 0,
                "median_probe_ms": 0,
                "settle_probes": 0,
                "settle_ms": 0,
                "values": {"cache": 0, "prefetch": 0, "llm": 0},
                "history": [],
            }
        jev = [t["decision_ms"] for t in run.ticks]
        return {
            "elapsed_ms": round((time.perf_counter() - run.started_at) * 1000) if run.started_at else 0,
            "decisions": len(jev),
            "interactions": len([h for h in run.history if h["kind"] != "wait"]),
            "waits": len([h for h in run.history if h["kind"] == "wait"]),
            "median_decision_ms": int(statistics.median(jev)) if jev else 0,
            # Laya and Cua run in process: their tokens are free and unpriced (s1a/tool/loop.py does the same).
            "jev_input_tokens": (
                sum(int(t.get("input_tokens") or 0) for t in run.ticks) if self._decision_model.name == "jev" else 0
            ),
            "median_probe_ms": int(statistics.median(t["probe_ms"] for t in run.ticks)) if run.ticks else 0,
            "settle_probes": sum(t.get("settle_probes", 0) for t in run.ticks),
            "settle_ms": sum(t.get("settle_ms", 0) for t in run.ticks),
            "values": {
                source: len([t for t in run.ticks if t.get("value_source") == source])
                for source in ("cache", "prefetch", "llm")
            },
            "history": run.history,
        }


def _progressed(snapshot: dict[str, Any], before_key: Any) -> bool:
    """A page moved on only when a probe that succeeded shows a different ``page_key``."""
    return not snapshot.get("error") and snapshot.get("page_key") != before_key


def _json_field(content: Any, key: str) -> Any:
    try:
        return json.loads(content if isinstance(content, str) else "{}").get(key)
    except (ValueError, AttributeError):
        return None
