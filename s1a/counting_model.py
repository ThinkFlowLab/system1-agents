# coding: utf-8
"""``CountingModel``: a chat-model wrapper that records every call's latency, tokens and tool calls.

The record is appended before the call runs and updated when it ends, so a call that failed, timed out or was
cancelled is counted too, not lost. The record's ``error`` summary keeps only the error's type -- never its message,
the prompt or a secret -- and the compact ``call_digests`` keep no raw tool arguments. The full in-memory record does
retain ``tool_args`` for internal routing, kept for compatibility with earlier consumers. A call only marks its usage
known when it ends normally with ``usage_metadata``; a reply without it, or a stream that errored, was cancelled or
interrupted after a partial usage, leaves the cost unknown (``usage_known`` False), never a confirmed zero, so a
consumer can tell "no tokens" from "we do not know how many tokens". The caller's exception is always re-raised.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk, Model

RECORD_KEYS = ("ms", "status", "error", "usage_known", "input_tokens", "output_tokens", "cache_tokens", "tool_calls")


def start_record(started: float) -> dict[str, Any]:
    """The record appended before the call runs: a failed or cancelled call is still visible in ``calls``."""
    return {
        "ms": 0,
        "status": "running",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_tokens": 0,
        "usage_known": False,  # no reply yet: the tokens are unknown, not zero
        "tool_calls": [],
        "tool_args": [],
        "content_chars": 0,
    }


def finish_record(
    record: dict[str, Any], reply: Any, started: float, status: str, *, error: BaseException | None = None
) -> None:
    """Update the record in place when the call ends: elapsed ms, status, and the reply's tokens when it has usage.

    ``reply`` is the ``AssistantMessage`` (or the merged ``AssistantMessageChunk``) on success and None otherwise.
    A partial reply keeps whatever tokens it reported, but ``usage_known`` is only True when the call ended normally
    (``status == "ok"``) with usage metadata: a stream that errored, was cancelled or interrupted after a partial
    usage keeps its token counts while still marking the total unknown, never a confirmed zero. The ``error`` summary
    keeps only the error's type, never its message, the prompt or a secret.
    """
    record["ms"] = round((time.perf_counter() - started) * 1000)
    record["status"] = status
    if error is not None:
        record["error"] = type(error).__name__
    if reply is None:
        return
    usage = getattr(reply, "usage_metadata", None)
    calls = list(getattr(reply, "tool_calls", None) or [])
    content = getattr(reply, "content", None)
    record["input_tokens"] = usage.input_tokens if usage is not None else 0
    record["output_tokens"] = usage.output_tokens if usage is not None else 0
    record["cache_tokens"] = int(usage.cache_tokens or 0) if usage is not None else 0
    record["usage_known"] = usage is not None and status == "ok"
    record["tool_calls"] = [str(call.name) for call in calls]
    record["tool_args"] = [str(call.arguments) for call in calls]
    record["content_chars"] = len(content) if isinstance(content, str) else 0


def usage_known(calls: list[dict[str, Any]]) -> bool:
    """Whether every recorded call reported its usage: one unknown call makes the whole total unknown, not zero.

    A record without the field predates the flag and was only ever appended on success, so it counts as known.
    """
    return all(bool(call.get("usage_known", True)) for call in calls)


def call_digests(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The compact records worth keeping for a failed or unpriced call: no prompt, no raw tool arguments.

    Only the calls whose usage is unknown or whose status is not ``ok`` are kept, so an episode's ``extra`` stays
    small while a failed or cancelled planner call remains traceable.
    """
    return [
        {key: call.get(key) for key in RECORD_KEYS}
        for call in calls
        if call.get("status") != "ok" or not call.get("usage_known", True)
    ]


class CountingModel(Model):
    """Forwards ``invoke`` and ``stream`` to ``inner``; appends one record per call to ``calls``.

    The record is appended before the call runs and updated in a ``finally``-safe way when it ends, so a failed,
    timed-out or cancelled call is counted with its status and elapsed time. A cancellation or exception is never
    swallowed: it is recorded and re-raised unchanged.
    """

    def __init__(self, inner: Model, calls: list[dict[str, Any]]) -> None:
        super().__init__(inner.model_client_config, inner.model_config)
        self._inner = inner
        self._calls = calls

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        started = time.perf_counter()
        record = start_record(started)
        self._calls.append(record)
        try:
            reply = await self._inner.invoke(messages, tools=tools, **kwargs)
        except asyncio.CancelledError:
            finish_record(record, None, started, "cancelled")
            raise
        except BaseException as exc:  # noqa: BLE001 - the caller's exception is recorded and re-raised unchanged
            finish_record(record, None, started, "error", error=exc)
            raise
        finish_record(record, reply, started, "ok")
        return reply

    async def stream(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AsyncIterator[AssistantMessageChunk]:
        started = time.perf_counter()
        record = start_record(started)
        self._calls.append(record)
        merged: AssistantMessageChunk | None = None
        try:
            async for chunk in self._inner.stream(messages, tools=tools, **kwargs):
                merged = chunk if merged is None else merged + chunk
                yield chunk
        except asyncio.CancelledError:
            finish_record(record, merged, started, "cancelled")
            raise
        except GeneratorExit:
            # The consumer stopped early or closed the stream: keep the partial usage and let the close finish.
            finish_record(record, merged, started, "interrupted")
            raise
        except BaseException as exc:  # noqa: BLE001 - a failed stream is recorded, then re-raised unchanged
            finish_record(record, merged, started, "error", error=exc)
            raise
        else:
            finish_record(record, merged, started, "ok")
