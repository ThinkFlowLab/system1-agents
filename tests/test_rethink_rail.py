# coding: utf-8
"""RethinkRail on scripted act results: repeats block a key, stalls ask for a plan, too many stalls give up."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.foundation.llm import AssistantMessage

from s1a.recovery import RecoveryLimits
from s1a.tool.rethink import RethinkRail, draft_plan, parse_tool_args, parse_tool_result, progress_digest
from s1a.tool.models import EvalState

BOARD = {"grid": [[2, 0], [0, 0]]}


def act(
    key: str, score: float, *, state: dict[str, Any] = BOARD, tool: str = "act", done: bool = False
) -> SimpleNamespace:
    result = {"state": state, "candidates": {"LEFT": "", "RIGHT": "", "UP": ""}, "done": done, "score": score}
    return SimpleNamespace(
        inputs=SimpleNamespace(tool_name=tool, tool_args={"key": key}, tool_result=json.dumps(result))
    )


class FakePlanner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, messages: Any, **kwargs: Any) -> AssistantMessage:
        self.calls.append({"messages": messages, **kwargs})
        return AssistantMessage(content="  Slide RIGHT, then UP to open the corner.  ")


class FailingPlanner:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages: Any, **kwargs: Any) -> AssistantMessage:
        self.calls += 1
        raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="chat endpoint returned HTTP 401")


class _LeakyPlanner:
    """A planner whose failure message echoes a credential: the rail must persist only the stable type."""

    def __init__(self, secret: str) -> None:
        self._secret = secret
        self.calls = 0

    async def invoke(self, messages: Any, **kwargs: Any) -> AssistantMessage:
        self.calls += 1
        raise RuntimeError(f"provider exploded near token={self._secret}")


class BlankPlanner:
    """A planner whose reply normalizes to nothing: the bounded branch must read it as a failure, not a plan.

    ``SimpleNamespace`` stands in for the reply so a ``None`` content reaches ``draft_plan`` unnormalized by
    ``AssistantMessage``'s own validation; ``draft_plan`` only reads ``reply.content``.
    """

    def __init__(self, content: Any = "") -> None:
        self.content = content
        self.calls = 0

    async def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(content=self.content)


class SlowPlanner:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages: Any, **kwargs: Any) -> AssistantMessage:
        self.calls += 1
        await asyncio.sleep(5)
        return AssistantMessage(content="late")


def bounded_rail(
    state: EvalState,
    refresh: Any,
    *,
    planner: Any = None,
    limits: RecoveryLimits | None = None,
    stall_after: int = 2,
    repeat_after: int = 0,
    give_up_after: int = 3,
) -> RethinkRail:
    return RethinkRail(
        state,
        rules="desktop rules",
        initial_score=0.0,
        planner=planner,
        stall_after=stall_after,
        repeat_after=repeat_after,
        give_up_after=give_up_after,
        refresh=refresh,
        limits=limits or RecoveryLimits(max_attempts=3, timeout_s=5.0),
    )


def rail(
    state: EvalState, *, planner: Any = None, stall_after: int = 0, repeat_after: int = 0, give_up_after: int = 3
) -> RethinkRail:
    return RethinkRail(
        state,
        rules="2048 rules",
        initial_score=8.0,
        planner=planner,
        stall_after=stall_after,
        repeat_after=repeat_after,
        give_up_after=give_up_after,
    )


class TestRethinkRail(IsolatedAsyncioTestCase):
    async def test_progress_records_nothing(self) -> None:
        state = EvalState()
        guard = rail(state, stall_after=2, repeat_after=2)
        for score in (12, 16, 20, 24):
            await guard.after_tool_call(act("LEFT", score))
        self.assertEqual((state.rethinks, state.blocked, state.plan), ([], set(), ""))

    async def test_exact_repeat_blocks_the_key_for_one_turn(self) -> None:
        state = EvalState()
        guard = rail(state, repeat_after=3)
        for _ in range(3):
            await guard.after_tool_call(act("LEFT", 8))
        self.assertEqual(state.blocked, {"LEFT"})
        self.assertEqual(state.rethinks, [{"kind": "repeat", "step": 3, "key": "LEFT"}])
        self.assertIn("LEFT", state.notices[0])

    async def test_a_repeat_after_productive_moves_is_no_repeat(self) -> None:
        state = EvalState()
        guard = rail(state, repeat_after=3)
        for score in (12, 16, 16):  # two merges, then a slide: a legal 2048 line
            await guard.after_tool_call(act("LEFT", score))
        self.assertEqual((state.rethinks, state.blocked), ([], set()))
        for _ in range(2):
            await guard.after_tool_call(act("LEFT", 16))
        self.assertEqual(state.rethinks, [{"kind": "repeat", "step": 5, "key": "LEFT"}])

    async def test_a_stall_on_another_key_keeps_the_repeat_block(self) -> None:
        state = EvalState()
        guard = rail(state, stall_after=6, repeat_after=3)
        for key in ("RIGHT", "RIGHT", "RIGHT", "LEFT", "LEFT", "LEFT"):
            await guard.after_tool_call(act(key, 8))
        self.assertEqual(state.blocked, {"LEFT", "RIGHT"})

    async def test_a_semantic_stall_asks_for_a_plan_and_blocks_the_most_repeated_key(self) -> None:
        state, planner = EvalState(), FakePlanner()
        guard = rail(state, planner=planner, stall_after=3)
        for key in ("LEFT", "RIGHT", "LEFT"):
            await guard.after_tool_call(act(key, 8))
        self.assertEqual(state.plan, "Slide RIGHT, then UP to open the corner.")
        self.assertEqual(state.blocked, {"LEFT"})
        self.assertEqual(state.rethinks[0]["kind"], "stall")
        self.assertEqual(state.rethinks[0]["blocked"], "LEFT")
        payload = json.loads(planner.calls[0]["messages"][1]["content"])
        self.assertEqual(payload["rules"], "2048 rules")
        self.assertEqual([step["key"] for step in payload["recent_steps"]], ["LEFT", "RIGHT", "LEFT"])
        self.assertEqual(payload["candidates"], {"LEFT": "", "RIGHT": "", "UP": ""})
        await guard.after_tool_call(act("UP", 16))
        self.assertEqual(len(state.rethinks), 1)

    async def test_a_failed_plan_is_the_episodes_error_and_no_give_up(self) -> None:
        state, planner = EvalState(), FailingPlanner()
        guard = rail(state, planner=planner, stall_after=2, give_up_after=1)
        for _ in range(6):
            await guard.after_tool_call(act("LEFT", 8))
        self.assertTrue(state.error.startswith("plan failed: "), state.error)
        self.assertIn("HTTP 401", state.error)
        self.assertFalse(state.give_up)
        self.assertEqual(planner.calls, 1)
        self.assertEqual(
            [event for event in state.rethinks if event["kind"] != "repeat"],
            [{"kind": "plan_failed", "step": 2, "error": state.error.removeprefix("plan failed: ")}],
        )

    async def test_a_stall_is_seen_through_the_progress_part_when_the_state_has_one(self) -> None:
        state = EvalState()
        guard = rail(state, stall_after=2)
        first = {"progress": {"holding": "mug"}, "recent_steps": ["go to shelf 1"], "steps_used": 1}
        second = {"progress": {"holding": "mug"}, "recent_steps": ["go to shelf 2"], "steps_used": 2}
        await guard.after_tool_call(act("go to shelf 1", 8, state=first))
        await guard.after_tool_call(act("go to shelf 2", 8, state=second))
        self.assertEqual(state.blocked, {"go to shelf 1"})
        self.assertEqual(progress_digest(first), progress_digest(second))
        self.assertNotEqual(progress_digest({"a": 1}), progress_digest({"a": 2}))

    async def test_a_stall_without_a_planner_still_blocks(self) -> None:
        state = EvalState()
        guard = rail(state, stall_after=2)
        for key in ("LEFT", "LEFT"):
            await guard.after_tool_call(act(key, 8))
        self.assertEqual((state.plan, state.blocked), ("", {"LEFT"}))

    async def test_too_many_stalls_without_progress_give_up(self) -> None:
        state = EvalState()
        guard = rail(state, planner=FakePlanner(), stall_after=2, give_up_after=1)
        for _ in range(2):
            await guard.after_tool_call(act("LEFT", 8))
        self.assertFalse(state.give_up)
        for _ in range(2):
            await guard.after_tool_call(act("LEFT", 8))
        self.assertTrue(state.give_up)
        self.assertEqual(state.rethinks[-1]["kind"], "give_up")

    async def test_a_stall_in_a_new_state_restarts_the_give_up_count(self) -> None:
        state = EvalState()
        guard = rail(state, stall_after=2, give_up_after=1)
        shelf = {"progress": {"at": "shelf 1"}}
        drawer = {"progress": {"at": "drawer 1"}}
        for place in (shelf, shelf, drawer, drawer):  # a binary score: two stalls in two different places
            await guard.after_tool_call(act(f"go to {place['progress']['at']}", 8, state=place))
        self.assertEqual([event["kind"] for event in state.rethinks], ["stall", "stall"])
        self.assertFalse(state.give_up)

    async def test_other_tools_and_unparseable_results_are_ignored(self) -> None:
        state = EvalState()
        guard = rail(state, stall_after=1, repeat_after=1)
        await guard.after_tool_call(act("LEFT", 8, tool="observe"))
        await guard.after_tool_call(
            SimpleNamespace(inputs=SimpleNamespace(tool_name="act", tool_args={}, tool_result="not json"))
        )
        self.assertEqual(state.rethinks, [])


class TestParsers(TestCase):
    def test_args_and_results_from_strings_and_objects(self) -> None:
        self.assertEqual(parse_tool_args('{"key": "a"}'), {"key": "a"})
        self.assertEqual(parse_tool_args({"key": "b"}), {"key": "b"})
        self.assertEqual(parse_tool_args("nope"), {})
        self.assertEqual(parse_tool_result('{"score": 1}'), {"score": 1})
        self.assertEqual(parse_tool_result(SimpleNamespace(content='{"score": 2}')), {"score": 2})
        self.assertIsNone(parse_tool_result("[1, 2]"))


class FakeRefresh:
    """A refresh callback over a view queue, with the active state updatable by the test."""

    def __init__(self, state: dict[str, Any] | None = None, *, done: bool = False) -> None:
        self.state = state if state is not None else BOARD
        self.done = done
        self.calls = 0

    async def __call__(self) -> dict[str, Any]:
        self.calls += 1
        return {"state": self.state, "candidates": {"UP": ""}, "done": self.done, "score": 0.0}


class TestDraftPlan(IsolatedAsyncioTestCase):
    async def test_the_reusable_planner_reads_the_rule_the_steps_the_state_and_the_candidates(self) -> None:
        planner = FakePlanner()
        recent = [{"key": "LEFT", "score": 0.0}, {"key": "RIGHT", "score": 0.0}]
        plan = await draft_plan(
            planner, rules="desktop rules", recent_steps=recent, state={"n": 1}, candidates={"UP": "go up"}
        )
        self.assertEqual(plan, "Slide RIGHT, then UP to open the corner.")
        payload = json.loads(planner.calls[0]["messages"][1]["content"])
        self.assertEqual(payload["rules"], "desktop rules")
        self.assertEqual(payload["recent_steps"], recent)
        self.assertEqual(payload["state"], {"n": 1})
        self.assertEqual(payload["candidates"], {"UP": "go up"})


class TestBoundedRethink(IsolatedAsyncioTestCase):
    """The bounded branch: a stall refreshes, then either plans for the next observation or stops on the budget."""

    async def test_a_stall_refreshes_then_plans_for_the_next_turn(self) -> None:
        state, refresh, planner = EvalState(), FakeRefresh(), FakePlanner()
        guard = bounded_rail(state, refresh, planner=planner)
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        self.assertEqual(state.plan, "Slide RIGHT, then UP to open the corner.")
        self.assertEqual(refresh.calls, 1)
        event = state.rethinks[-1]
        self.assertEqual(event["kind"], "stall")
        self.assertEqual(event["trigger"], "RIGHT")
        self.assertEqual(event["attempt"], 1)
        self.assertEqual(event["termination"], "planned")
        self.assertEqual(event["fresh_obs"]["candidates"], {"UP": ""})
        self.assertEqual([a["key"] for a in event["recent_actions"]], ["LEFT", "RIGHT"])
        self.assertGreaterEqual(event["spent_s"], 0.0)

    async def test_a_stall_with_no_planner_records_the_attempt_without_a_plan(self) -> None:
        state, refresh = EvalState(), FakeRefresh()
        guard = bounded_rail(state, refresh, planner=None)
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        self.assertEqual((state.plan, refresh.calls), ("", 1))
        self.assertEqual(state.rethinks[-1]["termination"], "no_planner")

    async def test_delayed_progress_after_the_refresh_skips_the_planner(self) -> None:
        state = EvalState()
        refresh, planner = FakeRefresh({"grid": [[0, 2], [0, 0]]}), FakePlanner()
        guard = bounded_rail(state, refresh, planner=planner)
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        self.assertEqual((state.plan, planner.calls), ("", []))
        self.assertEqual(state.rethinks[-1]["termination"], "delayed_progress")
        self.assertFalse(state.give_up)

    async def test_a_fresh_done_skips_the_planner(self) -> None:
        state = EvalState()
        refresh, planner = FakeRefresh(BOARD, done=True), FakePlanner()
        guard = bounded_rail(state, refresh, planner=planner)
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        self.assertEqual((state.plan, planner.calls), ("", []))
        self.assertEqual(state.rethinks[-1]["termination"], "done")

    async def test_a_done_act_or_a_spent_budget_is_not_replanned(self) -> None:
        for marker in ("done", "budget"):
            with self.subTest(marker=marker):
                state, refresh = EvalState(max_acts=1), FakeRefresh()
                if marker == "budget":
                    state.acts.append({"key": "LEFT", "score": 0.0, "done": False})
                guard = bounded_rail(state, refresh, planner=FakePlanner())
                await guard.after_tool_call(act("LEFT", 0, done=marker == "done"))
                await guard.after_tool_call(act("RIGHT", 0, done=marker == "done"))
                self.assertEqual((state.rethinks, refresh.calls), ([], 0))

    async def test_attempts_never_reset_when_the_episode_makes_progress(self) -> None:
        state = EvalState()
        holder = {"state": {"progress": {"at": "A"}}}
        planner = FakePlanner()

        async def refresh() -> dict[str, Any]:
            return {"state": holder["state"], "candidates": {}, "done": False, "score": 0.0}

        guard = bounded_rail(state, refresh, planner=planner, limits=RecoveryLimits(max_attempts=2, timeout_s=5.0))
        for key in ("LEFT", "LEFT"):  # two same-state acts: the first stall, attempt 1
            await guard.after_tool_call(act(key, 0, state=holder["state"]))
        holder["state"] = {"progress": {"at": "B"}}
        # Progress to B clears the window but keeps both attempts; the plan at attempt 1 gives stall_after fresh
        # actions before the next stall, so two more stalls are needed to spend the second attempt and give up.
        for key in ("UP", "UP", "UP", "UP", "UP"):
            await guard.after_tool_call(act(key, 1, state=holder["state"]))
        self.assertEqual(state.give_up, True)
        self.assertEqual([event["attempt"] for event in state.rethinks], [1, 2, 2])
        self.assertEqual(state.rethinks[-1]["termination"], "give_up")

    async def test_a_repeat_still_blocks_the_key_without_asking_for_a_plan(self) -> None:
        state, refresh = EvalState(), FakeRefresh()
        guard = bounded_rail(state, refresh, planner=FakePlanner(), stall_after=0, repeat_after=2)
        for _ in range(2):
            await guard.after_tool_call(act("LEFT", 0))
        self.assertEqual(state.blocked, {"LEFT"})
        self.assertEqual(state.rethinks, [{"kind": "repeat", "step": 2, "key": "LEFT"}])
        self.assertEqual(refresh.calls, 0)

    async def test_a_bounded_repeat_needs_an_unchanged_window_not_just_the_same_key(self) -> None:
        changing = EvalState()
        guard = bounded_rail(changing, FakeRefresh(), planner=FakePlanner(), stall_after=0, repeat_after=2)
        for display in ("1", "12", "123"):  # the same key repeated, but the window moves on every click
            await guard.after_tool_call(act("LEFT", 0, state={"progress": {"display": display}}))
        self.assertEqual((changing.blocked, changing.rethinks), (set(), []))

        frozen = EvalState()
        guard = bounded_rail(frozen, FakeRefresh(), planner=FakePlanner(), stall_after=0, repeat_after=2)
        for _ in range(2):  # the same key and the same window: a real no-op, blocked
            await guard.after_tool_call(act("LEFT", 0, state={"progress": {"display": "1"}}))
        self.assertEqual(frozen.blocked, {"LEFT"})
        self.assertEqual(frozen.rethinks, [{"kind": "repeat", "step": 2, "key": "LEFT"}])

    async def test_a_changed_display_needs_three_subsequent_noops_to_stall(self) -> None:
        state, refresh = EvalState(), FakeRefresh()
        guard = bounded_rail(state, refresh, planner=FakePlanner(), stall_after=3)
        for value in ("1", "12", "12", "12"):
            refresh.state = {"progress": {"display": value}}
            await guard.after_tool_call(act("type", 0, state=refresh.state))
        self.assertEqual(refresh.calls, 0, "only two actions were ineffective after the display changed")
        await guard.after_tool_call(act("type", 0, state=refresh.state))
        self.assertEqual(refresh.calls, 1, "the third unchanged action triggers recovery")

    async def test_a_score_increase_clears_both_noop_and_cycle_detection(self) -> None:
        for values in (("1", "1", "1"), ("1", "2", "1", "2")):
            with self.subTest(values=values):
                state, refresh = EvalState(), FakeRefresh()
                guard = bounded_rail(state, refresh, planner=FakePlanner(), stall_after=3)
                for index, value in enumerate(values):
                    refresh.state = {"progress": {"display": value}}
                    score = float(index == len(values) - 1)
                    await guard.after_tool_call(act("type", score, state=refresh.state))
                self.assertEqual(refresh.calls, 0, "score progress must suppress a stale no-op or cycle window")

    async def test_a_moving_desktop_display_is_progress_even_with_a_frozen_score(self) -> None:
        # The reviewer's exact shape: score stays 0 until the task finishes, so only the display digest moves.
        state, refresh, planner = EvalState(), FakeRefresh(), FakePlanner()
        guard = bounded_rail(
            state,
            refresh,
            planner=planner,
            stall_after=3,
            limits=RecoveryLimits(max_attempts=2, timeout_s=5.0),
        )
        for display in ("1", "12", "12", "123", "1234", "1234", "12345", "123456", "123456"):
            await guard.after_tool_call(act("LEFT", 0, state={"progress": {"display": display}}))
        self.assertEqual(state.rethinks, [], "a moved display is progress, not a stall")
        self.assertFalse(state.give_up)
        self.assertEqual((refresh.calls, state.plan), (0, ""))

    async def test_a_real_noop_desktop_stall_still_recovers(self) -> None:
        state, refresh, planner = EvalState(), FakeRefresh({"progress": {"display": "1"}}), FakePlanner()
        guard = bounded_rail(state, refresh, planner=planner, stall_after=3)
        for key in ("LEFT", "RIGHT", "UP"):
            await guard.after_tool_call(act(key, 0, state={"progress": {"display": "1"}}))
        self.assertEqual(refresh.calls, 1)
        self.assertEqual(state.rethinks[-1]["termination"], "planned")
        self.assertEqual(state.rethinks[-1]["attempt"], 1)

    async def test_a_two_state_cycle_still_recovers(self) -> None:
        # Every action changes the digest, so the no-op counter never fires; the A-B-A-B cycle must still recover.
        state, refresh, planner = EvalState(), FakeRefresh(), FakePlanner()
        guard = bounded_rail(state, refresh, planner=planner, stall_after=3)
        for display in ("A", "B", "A", "B"):
            await guard.after_tool_call(act("LEFT", 0, state={"progress": {"display": display}}))
        self.assertEqual(refresh.calls, 1, "a two-state cycle is a stall even though each step moved")
        self.assertEqual(state.rethinks[-1]["attempt"], 1)

    async def test_progress_and_a_delayed_refresh_keep_the_global_budget(self) -> None:
        state, planner = EvalState(), FakePlanner()
        refresh = FakeRefresh({"progress": {"at": "B"}})
        guard = bounded_rail(
            state,
            refresh,
            planner=planner,
            stall_after=2,
            limits=RecoveryLimits(max_attempts=2, timeout_s=5.0),
        )
        for _ in range(2):  # two same-state acts: attempt 1, whose refresh shows delayed progress (no plan)
            await guard.after_tool_call(act("LEFT", 0, state={"progress": {"at": "A"}}))
        self.assertEqual(state.rethinks[-1]["termination"], "delayed_progress")
        self.assertEqual(state.rethinks[-1]["attempt"], 1)
        for _ in range(2):  # the window restarted: attempt 2 plans, still on the same global budget
            await guard.after_tool_call(act("UP", 0, state={"progress": {"at": "B"}}))
        self.assertEqual((state.rethinks[-1]["termination"], state.rethinks[-1]["attempt"]), ("planned", 2))
        for _ in range(2):  # attempt 3 is refused before any refresh: the budget is global, not per window
            await guard.after_tool_call(act("UP", 0, state={"progress": {"at": "B"}}))
        self.assertTrue(state.give_up)
        self.assertEqual([event["attempt"] for event in state.rethinks], [1, 2, 2])
        self.assertEqual(refresh.calls, 2, "a spent budget never runs another refresh")

    async def test_a_provider_error_never_persists_a_key(self) -> None:
        fake_key = "sk-FAKE-SECRET-abcdef123456"
        for leaky in ("refresh", "planner"):
            with self.subTest(leaky=leaky):
                state = EvalState()

                async def boom() -> dict[str, Any]:
                    raise RuntimeError(f"provider exploded near token={fake_key}")

                async def fake_refresh() -> dict[str, Any]:
                    return {"state": BOARD, "candidates": {"UP": ""}, "done": False, "score": 0.0}

                planner = _LeakyPlanner(fake_key) if leaky == "planner" else FakePlanner()
                guard = bounded_rail(state, boom if leaky == "refresh" else fake_refresh, planner=planner)
                for key in ("LEFT", "RIGHT"):
                    await guard.after_tool_call(act(key, 0))
                persisted = json.dumps(state.rethinks) + str(state.error)
                self.assertNotIn(fake_key, persisted, "the provider text can echo a credential and is never stored")
                self.assertEqual(state.rethinks[-1]["error"], "RuntimeError")

    async def test_a_refresh_failure_carries_a_reason_and_an_actionable_next_step(self) -> None:
        state = EvalState()

        async def boom() -> dict[str, Any]:
            raise RuntimeError("window gone")

        guard = bounded_rail(state, boom, planner=FakePlanner())
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        event = state.rethinks[-1]
        self.assertEqual(event["termination"], "error")
        self.assertEqual(event["error"], "RuntimeError", "only the stable type is kept, never the provider text")
        self.assertNotIn("window gone", str(event))
        self.assertIn("start a new task", event["next_action"])
        self.assertNotIn("unsafe_dev", event["next_action"])

    async def test_a_permission_failure_points_at_the_permission_flow(self) -> None:
        state = EvalState()

        async def denied() -> dict[str, Any]:
            raise RuntimeError("permission denied: Accessibility")

        guard = bounded_rail(state, denied, planner=FakePlanner())
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        event = state.rethinks[-1]
        self.assertIn("permission flow", event["next_action"])
        self.assertIn("leave the task blocked", event["next_action"])

    async def test_a_spent_budget_escalation_adds_no_model_call(self) -> None:
        state, refresh = EvalState(), FakeRefresh()
        planner = FakePlanner()
        guard = bounded_rail(state, refresh, planner=planner, limits=RecoveryLimits(max_attempts=1, timeout_s=5.0))
        for key in ("LEFT", "RIGHT", "UP", "DOWN"):
            await guard.after_tool_call(act(key, 0))
        give_up = [event for event in state.rethinks if event["termination"] == "give_up"]
        self.assertTrue(give_up)
        self.assertIn("recovery attempts spent", give_up[-1]["error"])
        self.assertIn("start a new task", give_up[-1]["next_action"])
        self.assertEqual(len(planner.calls), 1, "the escalation is a message, not another planner call")

    async def test_an_empty_or_whitespace_plan_is_a_planner_failure(self) -> None:
        for content in ("", "   ", "\n\t ", None):
            with self.subTest(content=content):
                state, refresh = EvalState(), FakeRefresh()
                planner = BlankPlanner(content)
                guard = bounded_rail(state, refresh, planner=planner)
                for key in ("LEFT", "RIGHT"):
                    await guard.after_tool_call(act(key, 0))

                event = state.rethinks[-1]
                self.assertTrue(state.error.startswith("rethink planner failed: "), state.error)
                self.assertIn("empty plan", state.error)
                self.assertEqual((event["termination"], event["phase"]), ("error", "planner"))
                self.assertEqual(event["attempt"], 1, "the charged attempt is kept")
                self.assertGreaterEqual(event["spent_s"], 0.0, "the consumed active time is kept")
                self.assertIsInstance(event["fresh_obs"], dict, "the successful refresh is kept")
                self.assertIn("start a new task", event["next_action"])
                self.assertEqual(planner.calls, 1, "one plan attempt, never a silent retry")
                self.assertEqual(refresh.calls, 1, "a failed plan stops the episode: no further refresh")
                self.assertEqual(state.plan, "")
                self.assertFalse(state.give_up, "a failed plan is not a spent budget")

    async def test_a_refresh_timeout_is_recorded_and_stops_the_episode(self) -> None:
        state = EvalState()

        async def hang() -> dict[str, Any]:
            await asyncio.sleep(5)
            return {}

        guard = bounded_rail(state, hang, planner=FakePlanner(), limits=RecoveryLimits(max_attempts=3, timeout_s=0.05))
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        self.assertEqual(state.error, "rethink refresh timed out")
        self.assertFalse(state.give_up)
        event = state.rethinks[-1]
        self.assertEqual((event["termination"], event["phase"]), ("error", "refresh"))
        self.assertGreater(event["spent_s"], 0.0)  # the finally refreshes spent_s even on a timeout
        self.assertEqual(event["trigger_reason"], "stalled")

    async def test_a_planner_timeout_is_recorded_and_stops_the_episode(self) -> None:
        state, refresh = EvalState(), FakeRefresh()
        guard = bounded_rail(
            state, refresh, planner=SlowPlanner(), limits=RecoveryLimits(max_attempts=3, timeout_s=0.05)
        )
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        self.assertEqual(state.error, "rethink planner timed out")
        self.assertFalse(state.give_up)
        event = state.rethinks[-1]
        self.assertEqual((event["termination"], event["phase"]), ("error", "planner"))
        self.assertGreater(event["spent_s"], 0.0)
        self.assertIsInstance(event["fresh_obs"], dict)

    async def test_an_external_cancel_during_the_refresh_records_the_event_and_re_raises(self) -> None:
        state = EvalState()

        async def hang() -> dict[str, Any]:
            await asyncio.sleep(5)

        guard = bounded_rail(state, hang, planner=FakePlanner(), limits=RecoveryLimits(max_attempts=3, timeout_s=5.0))
        await guard.after_tool_call(act("LEFT", 0))
        with self.assertRaises(asyncio.TimeoutError):  # the outer wait_for cancels the hanging task
            await asyncio.wait_for(guard.after_tool_call(act("RIGHT", 0)), timeout=0.02)
        event = state.rethinks[-1]
        self.assertEqual((event["termination"], event["phase"]), ("cancelled", "refresh"))
        self.assertEqual(event["trigger"], "RIGHT")
        self.assertEqual(event["trigger_reason"], "stalled")
        self.assertIsNone(event["fresh_obs"])
        self.assertEqual([a["key"] for a in event["recent_actions"]], ["LEFT", "RIGHT"])
        self.assertIsNone(state.error)  # a cancel is not the episode's error and is never swallowed
        self.assertFalse(state.give_up)

    async def test_an_external_cancel_during_the_planner_keeps_the_fresh_observation(self) -> None:
        state, refresh = EvalState(), FakeRefresh()
        entered = asyncio.Event()

        class WaitingPlanner:
            async def invoke(self, messages: Any, **kwargs: Any) -> AssistantMessage:
                entered.set()
                return await asyncio.Future[AssistantMessage]()

        guard = bounded_rail(
            state, refresh, planner=WaitingPlanner(), limits=RecoveryLimits(max_attempts=3, timeout_s=30.0)
        )
        await guard.after_tool_call(act("LEFT", 0))
        task = asyncio.create_task(guard.after_tool_call(act("RIGHT", 0)))
        try:
            # Entry, rather than a short sleep, determines where cancellation lands.
            await asyncio.wait_for(entered.wait(), timeout=5.0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        event = state.rethinks[-1]
        self.assertEqual((event["termination"], event["phase"]), ("cancelled", "planner"))
        self.assertIsInstance(event["fresh_obs"], dict)
        self.assertIsNone(state.error)

    async def test_a_refresh_failure_is_recorded_and_stops_the_episode(self) -> None:
        state = EvalState()

        async def boom() -> dict[str, Any]:
            raise RuntimeError("window gone")

        guard = bounded_rail(state, boom, planner=FakePlanner())
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        self.assertEqual(state.error, "rethink refresh failed: RuntimeError")
        self.assertNotIn("window gone", state.error)
        self.assertEqual(state.rethinks[-1]["termination"], "error")

    async def test_a_failed_plan_is_recorded_and_stops_the_episode(self) -> None:
        state, refresh = EvalState(), FakeRefresh()
        planner = FailingPlanner()
        guard = bounded_rail(state, refresh, planner=planner)
        for key in ("LEFT", "RIGHT"):
            await guard.after_tool_call(act(key, 0))
        self.assertTrue(state.error.startswith("plan failed: "), state.error)
        self.assertNotIn("HTTP 401", state.error, "the provider's 401 text is never persisted")
        self.assertEqual(planner.calls, 1)
        self.assertEqual(state.rethinks[-1]["termination"], "error")
        self.assertEqual(state.rethinks[-1]["error"], state.error.removeprefix("plan failed: "))


class TestBoundedContract(TestCase):
    """refresh and limits are one switch: one without the other is a contract error, never a silent game branch."""

    def test_refresh_without_limits_or_limits_without_refresh_is_rejected(self) -> None:
        async def refresh() -> dict[str, Any]:
            return {}

        with self.assertRaises(ValueError):
            RethinkRail(
                EvalState(),
                rules="r",
                initial_score=0.0,
                planner=None,
                stall_after=2,
                repeat_after=0,
                give_up_after=3,
                refresh=refresh,
            )
        with self.assertRaises(ValueError):
            RethinkRail(
                EvalState(),
                rules="r",
                initial_score=0.0,
                planner=None,
                stall_after=2,
                repeat_after=0,
                give_up_after=3,
                limits=RecoveryLimits(),
            )

    def test_both_together_turn_on_bounded_and_neither_keeps_the_game_branch(self) -> None:
        async def refresh() -> dict[str, Any]:
            return {}

        bounded_state = EvalState()
        bounded = RethinkRail(
            bounded_state,
            rules="r",
            initial_score=0.0,
            planner=None,
            stall_after=2,
            repeat_after=0,
            give_up_after=3,
            refresh=refresh,
            limits=RecoveryLimits(),
        )
        self.assertEqual((bounded_state.bounded_recovery, bounded._recovery is not None), (True, True))
        self.assertIsNotNone(bounded._refresh)
        legacy = RethinkRail(
            EvalState(), rules="r", initial_score=0.0, planner=None, stall_after=2, repeat_after=0, give_up_after=3
        )
        self.assertIsNone(legacy._recovery)
        self.assertIsNone(legacy._refresh)
