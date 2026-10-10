# coding: utf-8
"""``CountingModel`` records latency, tokens and tool calls for invoke and for stream, and forwards the reply."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase

from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk, ToolCall
from openjiuwen.core.foundation.llm.schema.message import UsageMetadata

from s1a.counting_model import CountingModel, call_digests, usage_known
from s1a.tool.models import placeholder_model

ACT = ToolCall(id="c1", type="function", name="act", arguments='{"key": "inc"}')


def _inner(reply: AssistantMessage, chunks: list[AssistantMessageChunk]) -> Any:
    inner = placeholder_model()

    async def invoke(messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        return reply

    async def stream(messages: Any, *, tools: Any = None, **kwargs: Any):
        for chunk in chunks:
            yield chunk

    inner.invoke = invoke
    inner.stream = stream
    return inner


class TestCountingModel(IsolatedAsyncioTestCase):
    async def test_invoke_records_tokens_and_tool_calls_and_returns_the_reply(self) -> None:
        reply = AssistantMessage(
            content="",
            tool_calls=[ACT],
            usage_metadata=UsageMetadata(input_tokens=120, output_tokens=9, cache_tokens=100),
        )
        calls: list[dict[str, Any]] = []
        model = CountingModel(_inner(reply, []), calls)

        returned = await model.invoke([{"role": "user", "content": "play"}], tools=[])

        self.assertIs(returned, reply)
        (record,) = calls
        self.assertEqual((record["input_tokens"], record["output_tokens"], record["cache_tokens"]), (120, 9, 100))
        self.assertEqual((record["tool_calls"], record["tool_args"]), (["act"], ['{"key": "inc"}']))
        self.assertGreaterEqual(record["ms"], 0)
        self.assertEqual((record["status"], record["usage_known"]), ("ok", True))

    async def test_a_reply_without_usage_counts_zero_tokens_but_marks_the_usage_unknown(self) -> None:
        calls: list[dict[str, Any]] = []
        await CountingModel(_inner(AssistantMessage(content="four words of text"), []), calls).invoke("hi")
        self.assertEqual(
            (calls[0]["input_tokens"], calls[0]["output_tokens"], calls[0]["cache_tokens"], calls[0]["content_chars"]),
            (0, 0, 0, 18),
        )
        self.assertFalse(calls[0]["usage_known"], "no usage is an unknown cost, never a confirmed zero")
        self.assertFalse(usage_known(calls))

    async def test_a_failed_invoke_is_recorded_with_its_elapsed_time_and_re_raised(self) -> None:
        calls: list[dict[str, Any]] = []
        inner = placeholder_model()

        async def invoke(messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
            await asyncio.sleep(0.01)
            raise RuntimeError("planner down")

        inner.invoke = invoke
        with self.assertRaises(RuntimeError):
            await CountingModel(inner, calls).invoke("hi")

        (record,) = calls
        self.assertEqual((record["status"], record["error"]), ("error", "RuntimeError"))
        self.assertGreater(record["ms"], 0, "a failed call still records its elapsed time")
        self.assertFalse(record["usage_known"])
        self.assertEqual((record["input_tokens"], record["output_tokens"]), (0, 0))
        self.assertNotIn("planner down", str(record), "only the error type is recorded, never its message")

    async def test_a_cancelled_invoke_is_recorded_and_re_raised(self) -> None:
        calls: list[dict[str, Any]] = []
        inner = placeholder_model()

        async def invoke(messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError

        inner.invoke = invoke
        with self.assertRaises(asyncio.CancelledError):
            await CountingModel(inner, calls).invoke("hi")

        (record,) = calls
        self.assertEqual(record["status"], "cancelled")
        self.assertGreater(record["ms"], 0)
        self.assertFalse(record["usage_known"])

    async def test_a_failed_stream_keeps_its_partial_call_and_re_raises(self) -> None:
        calls: list[dict[str, Any]] = []
        inner = placeholder_model()
        chunks = [
            AssistantMessageChunk(content="", tool_calls=[ACT]),
            AssistantMessageChunk(content="", usage_metadata=UsageMetadata(input_tokens=40, output_tokens=3)),
        ]

        async def stream(messages: Any, *, tools: Any = None, **kwargs: Any):
            yield chunks[0]
            await asyncio.sleep(0.01)
            raise RuntimeError("stream dropped")

        inner.stream = stream
        seen = []
        with self.assertRaises(RuntimeError):
            async for chunk in CountingModel(inner, calls).stream("hi"):
                seen.append(chunk)

        self.assertEqual(seen, [chunks[0]])
        (record,) = calls
        self.assertEqual(record["status"], "error")
        self.assertGreater(record["ms"], 0)
        self.assertEqual(record["tool_calls"], ["act"], "the partial call is kept even when the stream fails")

    async def test_a_failed_stream_after_partial_usage_keeps_the_tokens_but_marks_usage_unknown(self) -> None:
        calls: list[dict[str, Any]] = []
        inner = placeholder_model()
        partial = AssistantMessageChunk(content="", usage_metadata=UsageMetadata(input_tokens=40, output_tokens=3))

        async def stream(messages: Any, *, tools: Any = None, **kwargs: Any):
            yield partial
            await asyncio.sleep(0.01)
            raise RuntimeError("stream dropped")

        inner.stream = stream
        with self.assertRaises(RuntimeError):
            async for _ in CountingModel(inner, calls).stream("hi"):
                pass

        (record,) = calls
        self.assertEqual((record["status"], record["usage_known"]), ("error", False))
        self.assertEqual(
            (record["input_tokens"], record["output_tokens"]),
            (40, 3),
            "the partial token counts survive even though the total is unknown",
        )

    async def test_a_cancelled_stream_after_partial_usage_keeps_the_tokens_but_marks_usage_unknown(self) -> None:
        calls: list[dict[str, Any]] = []
        inner = placeholder_model()
        partial = AssistantMessageChunk(content="", usage_metadata=UsageMetadata(input_tokens=40, output_tokens=3))

        async def stream(messages: Any, *, tools: Any = None, **kwargs: Any):
            yield partial
            await asyncio.sleep(1.0)

        inner.stream = stream
        model = CountingModel(inner, calls)

        async def consume() -> None:
            async for _ in model.stream("hi"):
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        (record,) = calls
        self.assertEqual((record["status"], record["usage_known"]), ("cancelled", False))
        self.assertEqual((record["input_tokens"], record["output_tokens"]), (40, 3))

    async def test_an_interrupted_stream_after_partial_usage_keeps_the_tokens_but_marks_usage_unknown(self) -> None:
        calls: list[dict[str, Any]] = []
        inner = placeholder_model()
        partial = AssistantMessageChunk(content="", usage_metadata=UsageMetadata(input_tokens=40, output_tokens=3))

        async def stream(messages: Any, *, tools: Any = None, **kwargs: Any):
            yield partial
            yield AssistantMessageChunk(content="never reached")

        inner.stream = stream
        agen = CountingModel(inner, calls).stream("hi")
        self.assertIs(await agen.__anext__(), partial)
        await agen.aclose()

        (record,) = calls
        self.assertEqual((record["status"], record["usage_known"]), ("interrupted", False))
        self.assertEqual((record["input_tokens"], record["output_tokens"]), (40, 3))

    async def test_an_early_consumer_stop_is_recorded_as_interrupted(self) -> None:
        calls: list[dict[str, Any]] = []
        inner = placeholder_model()
        chunks = [AssistantMessageChunk(content="one"), AssistantMessageChunk(content="two")]

        async def stream(messages: Any, *, tools: Any = None, **kwargs: Any):
            for chunk in chunks:
                yield chunk

        inner.stream = stream
        agen = CountingModel(inner, calls).stream("hi")
        self.assertEqual((await agen.__anext__()).content, "one")
        await agen.aclose()

        (record,) = calls
        self.assertEqual(record["status"], "interrupted")
        self.assertFalse(record["usage_known"])

    async def test_call_digests_keep_only_the_failed_or_unpriced_records(self) -> None:
        calls: list[dict[str, Any]] = [
            {"ms": 5, "status": "ok", "usage_known": True},
            {"ms": 9, "status": "error", "usage_known": False, "error": "TimeoutError"},
            {"ms": 7, "status": "ok", "usage_known": False},
        ]
        digests = call_digests(calls)
        self.assertEqual([digest["status"] for digest in digests], ["error", "ok"])
        self.assertEqual(digests[0]["error"], "TimeoutError")
        self.assertNotIn("tool_args", digests[0], "no raw arguments in the compact trace")

    async def test_stream_yields_every_chunk_and_records_the_merged_call(self) -> None:
        chunks = [
            AssistantMessageChunk(content="", tool_calls=[ACT]),
            AssistantMessageChunk(content="", usage_metadata=UsageMetadata(input_tokens=40, output_tokens=3)),
        ]
        calls: list[dict[str, Any]] = []
        model = CountingModel(_inner(AssistantMessage(content=""), chunks), calls)

        seen = [chunk async for chunk in model.stream("play", tools=[])]

        self.assertEqual(len(seen), 2)
        (record,) = calls
        self.assertEqual((record["input_tokens"], record["output_tokens"]), (40, 3))
        self.assertEqual(record["tool_calls"], ["act"])
        self.assertTrue(record["usage_known"], "a stream that ends normally with usage still knows its tokens")
