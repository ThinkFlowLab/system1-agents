# coding: utf-8
"""The bounded recovery budget: finite positive limits, per-episode attempts and active seconds that never reset."""

from __future__ import annotations

import asyncio
import gc
import warnings
from unittest import IsolatedAsyncioTestCase, TestCase

from s1a.recovery import RecoveryBudget, RecoveryExhausted, RecoveryLimits, recovery_next_action


class TestRecoveryLimits(TestCase):
    def test_the_defaults_are_three_attempts_and_fifteen_seconds(self) -> None:
        limits = RecoveryLimits()
        self.assertEqual((limits.max_attempts, limits.timeout_s), (3, 15.0))

    def test_a_non_positive_or_unbounded_limit_is_rejected(self) -> None:
        for kwargs in (
            {"max_attempts": 0},
            {"max_attempts": -1},
            {"max_attempts": True},
            {"max_attempts": 1.5},
            {"max_attempts": 3.0},
            {"timeout_s": 0},
            {"timeout_s": -1.0},
            {"timeout_s": float("nan")},
            {"timeout_s": float("inf")},
            {"timeout_s": float("-inf")},
            {"timeout_s": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RecoveryLimits(**kwargs)  # type: ignore[arg-type]


class TestRecoveryNextAction(TestCase):
    def test_a_stopped_recovery_gets_a_short_actionable_next_step(self) -> None:
        action = recovery_next_action(termination="give_up", error="recovery attempts spent: 3/3")
        self.assertIn("fresh observation", action)
        self.assertIn("action that triggered recovery", action)
        self.assertIn("clear the blocker", action)
        self.assertIn("start a new task", action)

    def test_a_permission_failure_is_routed_through_the_permission_flow(self) -> None:
        for error in ("permission denied", "accessibility not granted", "403 forbidden"):
            with self.subTest(error=error):
                action = recovery_next_action(termination="error", stage="refresh", error=error)
                self.assertIn("permission flow", action)
                self.assertIn("leave the task blocked", action)

    def test_the_next_step_never_advises_an_unsafe_or_automatic_action(self) -> None:
        for termination in ("give_up", "timeout", "error", "cancelled"):
            with self.subTest(termination=termination):
                action = recovery_next_action(termination=termination, stage="planner", error="boom")
                self.assertNotIn("unsafe_dev", action)
                self.assertNotIn("retry", action)
                self.assertIn("start a new task", action)
        for error in ("permission denied", "403 forbidden"):
            with self.subTest(error=error):
                action = recovery_next_action(termination="error", stage="refresh", error=error)
                self.assertNotIn("unsafe_dev", action)
                self.assertNotIn("retry", action)

    def test_a_stage_is_named_when_known(self) -> None:
        self.assertIn("planner step", recovery_next_action(termination="timeout", stage="planner", error=None))
        self.assertIn("timeout", recovery_next_action(termination="timeout", stage=None, error=None))


class TestRecoveryBudget(IsolatedAsyncioTestCase):
    async def test_attempts_only_grow_and_a_spent_budget_refuses_another(self) -> None:
        budget = RecoveryBudget(RecoveryLimits(max_attempts=2, timeout_s=10))
        budget.begin_attempt()
        self.assertEqual(budget.attempts, 1)
        budget.begin_attempt()
        self.assertEqual(budget.attempts, 2)
        self.assertTrue(budget.exhausted)
        with self.assertRaises(RecoveryExhausted):
            budget.begin_attempt()
        self.assertEqual(budget.attempts, 2)  # a refused attempt is not counted

    async def test_a_call_is_cancelled_by_the_remaining_timeout_and_charged(self) -> None:
        budget = RecoveryBudget(RecoveryLimits(max_attempts=3, timeout_s=0.05))
        with self.assertRaises(asyncio.TimeoutError):
            await budget.call(asyncio.sleep(1.0))
        self.assertLessEqual(budget.remaining_s, 0.05)
        self.assertGreater(budget.spent_s, 0.0)

    async def test_the_remaining_timeout_shrinks_with_the_active_time_only(self) -> None:
        budget = RecoveryBudget(RecoveryLimits(max_attempts=3, timeout_s=1.0))
        self.assertEqual(budget.remaining_s, 1.0)

        async def quick() -> str:
            return "ok"

        self.assertEqual(await budget.call(quick()), "ok")
        self.assertLess(budget.remaining_s, 1.0)
        self.assertFalse(budget.exhausted)

    async def test_a_rejected_call_closes_its_awaitable_without_running_it_or_warning(self) -> None:
        budget = RecoveryBudget(RecoveryLimits(max_attempts=1, timeout_s=0.05))
        with self.assertRaises(asyncio.TimeoutError):
            await budget.call(asyncio.sleep(1.0))  # spend the only active second
        self.assertLessEqual(budget.remaining_s, 0.0)
        ran = False

        async def work() -> str:
            nonlocal ran
            ran = True
            return "should never run"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(RecoveryExhausted):
                await budget.call(work())
            gc.collect()  # the leak would warn here, when a never-awaited coroutine is collected
        self.assertFalse(ran)
        self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])
