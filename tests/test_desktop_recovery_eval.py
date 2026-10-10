# coding: utf-8
"""The desktop recovery eval's offline pieces: the oracle, the wasted/no-op count, the terminal next_action.

The real driver path lives in ``tests/system/test_recovery_desktop.py`` and is opt-in. These tests never launch the
Windows fixture: they exercise the pure helpers and the loop's bounded act-budget terminal through the repository's
own offline machinery.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from openjiuwen.core.foundation.llm import AssistantMessage, AssistantMessageChunk, Model, ToolCall
from openjiuwen.core.runner import Runner

from evals.desktop_recovery import (
    FIXTURE_SOURCE,
    DesktopTrialRecord,
    _recovery_fields,
    count_wasted_actions,
    paired_summary,
    read_events,
    read_oracle,
)
from s1a.decision_models import RuleModel
from s1a.jobs import Episode, now_iso
from s1a.recovery import RecoveryLimits
from s1a.spec import Budget, ToolAgentSpec
from s1a.tool import loop as agent
from s1a.tool.loop import run_episode


def _record(
    task: str,
    arm: str,
    *,
    verified: bool,
    errored: bool = False,
    repeat: int = 0,
    recovery_failed: bool = False,
    next_action: str | None = None,
    elapsed_s: float = 1.0,
    wasted_actions: int = 0,
    model_calls: int = 2,
) -> DesktopTrialRecord:
    return DesktopTrialRecord(
        task=task,
        arm=arm,
        repeat=repeat,
        seed=repeat,
        verified=verified,
        terminal="verified" if verified else ("timeout" if errored else "BLOCKED"),
        errored=errored,
        scripted_model="scripted-rule",
        driver={"bin": "/x/cua-driver.exe", "version": "0.30.1", "channel": "test"},
        platform="test",
        recovery_attempts=1 if arm == "on" else 0,
        recovery_spent_s=0.1 if arm == "on" else 0.0,
        recovery_failed=recovery_failed,
        recovery_termination="planned" if arm == "on" else None,
        recovery_next_action=next_action,
        next_action_source="terminal" if next_action else None,
        wasted_actions=wasted_actions,
        noop_clicks=wasted_actions,
        steps=12 if not verified else 2,
        model_calls=model_calls,
        decision_calls=model_calls,
        planner_calls=1 if arm == "on" else 0,
        chat_calls=0,
        elapsed_s=elapsed_s,
        cost_usd=0.0,
        oracle={"verified": verified},
    )


class TestOracleIsolation(TestCase):
    """Success is this trial's own result.json; another trial's oracle can never verify it."""

    def test_each_trial_directory_is_its_own_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            (first / "result.json").write_text('{"finished": true, "pid": 1}', encoding="utf-8")
            (second / "result.json").write_text('{"finished": false, "pid": 2}', encoding="utf-8")
            self.assertTrue(read_oracle(first)["finished"])
            self.assertFalse(read_oracle(second)["finished"], "the solved first trial must not verify the second")
            (second / "app_events.jsonl").write_text(
                '{"event": "noop", "pid": 2}\nnot json\n{"event": "noop", "pid": 2}\n', encoding="utf-8"
            )
            self.assertEqual([event["event"] for event in read_events(second)], ["noop", "noop"])

    def test_a_missing_or_broken_oracle_is_never_a_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_oracle(Path(tmp) / "absent"), {})
            broken = Path(tmp) / "broken"
            broken.mkdir()
            (broken / "result.json").write_text("{", encoding="utf-8")
            self.assertEqual(read_oracle(broken), {})


class TestWastedActions(TestCase):
    def test_only_acts_whose_window_digest_did_not_change_count(self) -> None:
        def view(title: str) -> dict[str, Any]:
            return {"state": {"progress": {"title": title, "elements": [], "values": []}}}

        # act 1: locked->locked (no-op), act 2: locked->unlocked (progress), act 3: unlocked->unlocked (no-op)
        views = [view("locked"), view("locked"), view("unlocked"), view("unlocked")]
        self.assertEqual(count_wasted_actions(views, 3), 2)
        self.assertEqual(count_wasted_actions(views, 0), 0)


class TestTerminalFields(TestCase):
    def _episode(
        self,
        *,
        policy: str,
        error: str | None,
        rethinks: list[dict[str, Any]],
        terminal: dict[str, Any] | None = None,
        output: str = "",
    ) -> Episode:
        stamp = now_iso()
        extra: dict[str, Any] = {"rethinks": rethinks, "output": output}
        if terminal is not None:
            extra["terminal"] = terminal
        return Episode(
            env="desktop_recovery",
            policy=policy,
            seed=0,
            score=0.0,
            steps=6,
            elapsed_s=1.0,
            started_at=stamp,
            finished_at=stamp,
            final_state={},
            chat_calls=1,
            chat_input_tokens=0,
            chat_output_tokens=0,
            chat_cache_tokens=0,
            jev_input_tokens=0,
            invalid_keys=0,
            cost_usd=0.0,
            error=error,
            extra=extra,
        )

    def test_the_structured_terminal_is_consumed_not_regex_parsed(self) -> None:
        # The full next action survives even when the legacy output is truncated mid-JSON: the structured terminal is
        # the source, so no regex over the truncated text is needed.
        terminal = {"status": "BLOCKED", "reason": "act budget spent", "next_action": "Review: start a new task."}
        episode = self._episode(
            policy="scripted-on",
            error=None,
            rethinks=[{"kind": "stall", "attempt": 1, "termination": "planned", "spent_s": 0.1}],
            terminal=terminal,
            output='{"status": "BLOCKED", "next_action": "Review: start a new ta',
        )
        fields = _recovery_fields(episode, bounded=True)
        self.assertEqual(fields["next_action_source"], "terminal")
        self.assertEqual(fields["recovery_next_action"], "Review: start a new task.")
        self.assertTrue(fields["recovery_failed"])

    def test_a_failed_event_supplies_the_next_action_before_the_terminal(self) -> None:
        rethinks = [
            {"kind": "stall", "attempt": 2, "termination": "give_up", "next_action": "event action", "spent_s": 0.5}
        ]
        terminal = {"status": "BLOCKED", "reason": "give_up", "next_action": "terminal action"}
        episode = self._episode(
            policy="scripted-on", error="recovery attempts spent", rethinks=rethinks, terminal=terminal
        )
        fields = _recovery_fields(episode, bounded=True)
        self.assertEqual(fields["next_action_source"], "event")
        self.assertEqual(fields["recovery_next_action"], "event action")
        self.assertTrue(fields["recovery_failed"])
        self.assertEqual((fields["recovery_attempts"], fields["recovery_spent_s"]), (2, 0.5))

    def test_a_plan_only_stall_falls_back_to_the_terminal_next_action(self) -> None:
        rethinks = [{"kind": "stall", "attempt": 3, "termination": "planned", "spent_s": 0.2}]
        terminal = {"status": "BLOCKED", "reason": "act budget spent", "next_action": "Review it; start a new task."}
        episode = self._episode(policy="scripted-on", error=None, rethinks=rethinks, terminal=terminal)
        fields = _recovery_fields(episode, bounded=True)
        self.assertEqual(fields["next_action_source"], "terminal")
        self.assertEqual(fields["recovery_next_action"], "Review it; start a new task.")
        self.assertTrue(fields["recovery_failed"])

    def test_an_error_does_not_fabricate_an_unrecorded_operator_action(self) -> None:
        episode = self._episode(policy="scripted-on", error="rethink planner timed out", rethinks=[], output="")
        fields = _recovery_fields(episode, bounded=True)
        self.assertIsNone(fields["next_action_source"])
        self.assertIsNone(fields["recovery_next_action"])
        self.assertTrue(fields["recovery_failed"])

    def test_a_bounded_success_is_not_a_recovery_failure(self) -> None:
        rethinks = [{"kind": "stall", "attempt": 1, "termination": "planned", "spent_s": 0.1}]
        terminal = {"status": "DONE", "reason": "environment done"}
        episode = self._episode(policy="scripted-on", error=None, rethinks=rethinks, terminal=terminal)
        fields = _recovery_fields(episode, bounded=True)
        self.assertFalse(fields["recovery_failed"], "a completed task is not a failed recovery")
        self.assertIsNone(fields["recovery_next_action"])
        self.assertIsNone(fields["next_action_source"])

    def test_an_off_arm_without_recovery_is_not_a_recovery_failure(self) -> None:
        episode = self._episode(policy="scripted-off", error=None, rethinks=[], terminal={"status": "DONE"})
        fields = _recovery_fields(episode, bounded=False)
        self.assertFalse(fields["recovery_failed"])
        self.assertIsNone(fields["recovery_next_action"])


class TestPairedSummaryDenominator(TestCase):
    def test_every_planned_trial_counts_even_when_errored(self) -> None:
        records = [
            _record("normal", "off", verified=True),
            _record("recoverable", "off", verified=False),
            _record("permanent", "off", verified=False),
            _record("normal", "on", verified=True),
            _record("recoverable", "on", verified=True, recovery_failed=False),
            _record("permanent", "on", verified=False, recovery_failed=True, next_action="start a new task"),
            _record("permanent", "on", verified=False, errored=True, repeat=1),
        ]
        summary = paired_summary(records)
        self.assertEqual(summary["planned_trials"], 7)
        self.assertEqual(summary["arms"]["off"]["planned"], 3)
        self.assertEqual(summary["arms"]["on"]["planned"], 4, "the errored trial still counts in the denominator")
        self.assertEqual(summary["arms"]["off"]["completion_rate"], 0.333)
        self.assertEqual(summary["arms"]["on"]["completion_rate"], 0.5)
        self.assertEqual(summary["arms"]["on"]["errored"], 1)
        self.assertEqual(summary["arms"]["on"]["recovery_failed"], 1)
        self.assertEqual(summary["on_minus_off_completion_rate"], 0.167)
        self.assertIn("not a trained model", summary["note"])
        self.assertIn("cost is 0 by construction", summary["note"])

    def test_no_records_is_an_empty_but_valid_summary(self) -> None:
        summary = paired_summary([])
        self.assertEqual(summary["planned_trials"], 0)
        self.assertIsNone(summary["arms"]["on"]["completion_rate"])


class TestFixtureSource(TestCase):
    def test_the_committed_source_is_small_and_uses_its_own_oracle(self) -> None:
        source = FIXTURE_SOURCE.read_text(encoding="utf-8")
        self.assertLessEqual(FIXTURE_SOURCE.stat().st_size, 8192, "the fixture source must stay under 8 KB")
        self.assertIn("result.json", source)
        self.assertIn("WS_EX_NOACTIVATE", source)


class RefreshingEnv:
    """A desktop-shaped env: two keys that never move the window, and refresh re-reads it."""

    def __init__(self) -> None:
        self.refreshes = 0

    async def reset(self) -> None:
        return None

    async def observe(self) -> dict[str, Any]:
        return {"progress": {"title": "locked"}}

    async def candidates(self) -> dict[str, str]:
        return {"left": "no-op", "right": "no-op"}

    async def step(self, key: str) -> None:
        return None

    @property
    def done(self) -> bool:
        return False

    @property
    def score(self) -> float:
        return 0.0

    async def refresh(self) -> None:
        self.refreshes += 1


class ScriptedChatModel(Model):
    """The planner offline: one plan line, no key: the tool-loop fallback is only used by the rail."""

    def __init__(self) -> None:
        from s1a.tool.models import placeholder_model

        source = placeholder_model()
        super().__init__(source.model_client_config, source.model_config)

    def _answer(self, tools: Any) -> AssistantMessage:
        act = next(
            (str(getattr(tool, "name", "")) for tool in tools or [] if "act" in str(getattr(tool, "name", ""))), None
        )
        if act is None:
            return AssistantMessage(content="act now", finish_reason="stop")
        return AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c", type="function", name=act, arguments=json.dumps({"key": "left"}))],
            finish_reason="tool_calls",
        )

    async def invoke(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AssistantMessage:
        return self._answer(tools)

    async def stream(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> AsyncIterator[AssistantMessageChunk]:
        message = self._answer(tools)
        yield AssistantMessageChunk(
            content=message.content, tool_calls=message.tool_calls, finish_reason=message.finish_reason
        )


SPEC_STALL = ToolAgentSpec(
    name="desktop_shaped",
    description="Stall forever.",
    rules="r",
    budget=Budget(max_steps=12, timeout_s=60, stall_after=2),
    flags=lambda parser: None,
    series=lambda flags: None,
)


class TestBoundedActBudgetTerminal(IsolatedAsyncioTestCase):
    """The regression: an act cap after a plan must still carry a reason and a next action, not a bare BLOCKED."""

    async def test_the_terminal_after_a_spent_act_budget_carries_a_next_action(self) -> None:
        env = RefreshingEnv()
        picks = {"n": 0}

        def ping_pong(observation: dict[str, Any], offered: dict[str, str]) -> str:
            picks["n"] += 1
            return "left" if picks["n"] % 2 else "right"

        with tempfile.TemporaryDirectory() as tmp, patch.object(agent, "WORKSPACE", Path(tmp)):
            await Runner.start()
            try:
                episode = await run_episode(
                    SPEC_STALL,
                    env,
                    model_name="rule",
                    seed=0,
                    chat=ScriptedChatModel(),
                    decision_model=RuleModel("ping-pong", ping_pong),
                    rethink_on=True,
                    max_acts=6,
                    timeout_s=60.0,
                    prices=None,
                    log=False,
                    limits=RecoveryLimits(max_attempts=3, timeout_s=15.0),
                )
            finally:
                await Runner.stop()
        output = episode.extra["output"]
        events = episode.extra["rethinks"]
        self.assertEqual([event["termination"] for event in events], ["planned", "planned"])
        self.assertEqual(episode.steps, 6)
        self.assertIn("BLOCKED", output)
        self.assertIn('"next_action"', output)
        self.assertIn("start a new task", output, "the full next action must survive the terminal output budget")
        self.assertNotIn("recovery timeout", output, "an act cap is not dressed up as a recovery timeout")
        # The evaluator consumes the runtime's structured terminal; the full escalation is read from it, not the
        # truncated output, so a long next action survives the terminal output's short budget.
        terminal = episode.extra["terminal"]
        self.assertEqual(terminal["status"], "BLOCKED")
        self.assertIn("start a new task", terminal["next_action"])
        fields = _recovery_fields(episode, bounded=True)
        self.assertEqual(fields["next_action_source"], "terminal")
        self.assertIn("start a new task", fields["recovery_next_action"])
