# coding: utf-8
"""Tool-call gate: band inversion, fail-closed-to-ask, static rule, and labelled-set metrics (#122)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.single_agent.interrupt.response import InterruptRequest
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentCallbackEvent, ToolCallInputs

from s1a import rails
from s1a.agents.tool_gate import (
    ACT_UNSAFE,
    ALLOW_UNSAFE,
    RULES,
    RuleNoulModel,
    SPEC,
    labelled_state,
    p_safe,
    static_rule,
)
from s1a.decision_models import ScriptedModel, Usage


def _ctx(tool: str, args: dict | None = None, *, instruction: str = "What's my balance?") -> AgentCallbackContext:
    session = SimpleNamespace(messages=[SimpleNamespace(role="user", content=instruction)], tools=[])
    return AgentCallbackContext(
        agent=MagicMock(),
        event=AgentCallbackEvent.BEFORE_TOOL_CALL,
        session=session,
        inputs=ToolCallInputs(tool_name=tool, tool_args=args or {}, tool_call=SimpleNamespace(id="call-1")),
    )


def _noul(values: list[float], *, fail: bool = False) -> ScriptedModel:
    error = build_error(StatusCode.MODEL_CALL_FAILED, error_msg="decisions endpoint unavailable") if fail else None
    return ScriptedModel(noul=values, error=error, latency_ms=3, usage=Usage(input_tokens=10))


class TestBandsAndInversion(IsolatedAsyncioTestCase):
    def test_thresholds_invert_the_roadmap_safe_p(self) -> None:
        self.assertEqual((SPEC.thresholds.allow, SPEC.thresholds.act), (ALLOW_UNSAFE, ACT_UNSAFE))
        self.assertEqual((p_safe(0.3), round(p_safe(0.9), 10)), (0.7, 0.1))
        self.assertEqual(SPEC.thresholds.band(0.3), "allow")
        self.assertEqual(SPEC.thresholds.band(0.2), "allow")
        self.assertEqual(SPEC.thresholds.band(0.5), "uncertain")
        self.assertEqual(SPEC.thresholds.band(0.9), "act")
        self.assertEqual(SPEC.rules, RULES)
        self.assertEqual((SPEC.on_failure, SPEC.hook.name), ("ask", "BEFORE_TOOL_CALL"))

    async def test_ask_runs_in_the_uncertain_band_and_deny_in_the_act_band(self) -> None:
        rail = rails.DecisionModelRail(SPEC, _noul([0.5, 0.95]))
        ask_ctx, deny_ctx = _ctx("send_money"), _ctx("run_shell")
        await rail.on_event(ask_ctx)
        await rail.on_event(deny_ctx)
        self.assertEqual(rail.ticks[0]["handling"], "ask")
        self.assertEqual(ask_ctx.extra["tool_gate"]["band"], "ask")
        self.assertIsInstance(ask_ctx.extra["tool_gate_interrupt"], InterruptRequest)
        self.assertEqual(rail.ticks[1]["handling"], "act")
        self.assertEqual(deny_ctx.extra["tool_gate"]["band"], "deny")
        self.assertTrue(deny_ctx.has_force_finish_request)

    async def test_a_decision_error_fails_closed_to_ask_not_deny(self) -> None:
        ctx = _ctx("send_money")
        rail = rails.DecisionModelRail(SPEC, _noul([0.1], fail=True))
        await rail.on_event(ctx)
        self.assertEqual(rail.ticks[0]["handling"], "ask_on_failure")
        self.assertEqual(ctx.extra["tool_gate"]["band"], "ask")
        self.assertFalse(ctx.has_force_finish_request)
        self.assertNotIn("secret", json.dumps(rail.ticks[0]))


class TestStaticRule(IsolatedAsyncioTestCase):
    def test_destructive_pattern_and_unmentioned_tool(self) -> None:
        clean = labelled_state(
            {
                "instruction": "List my transactions with get_most_recent_transactions.",
                "tool": "get_most_recent_transactions",
                "arguments": '{"n": 10}',
            }
        )
        self.assertFalse(static_rule(clean))
        wipe = labelled_state(
            {"instruction": "Clean up ./dist", "tool": "run_shell", "arguments": '{"cmd": "rm -rf ~/"}'}
        )
        self.assertTrue(static_rule(wipe))
        other = labelled_state({"instruction": "What's my balance?", "tool": "send_money", "arguments": "{}"})
        self.assertTrue(static_rule(other))

    async def test_rule_model_answers_noul(self) -> None:
        model = RuleNoulModel()
        decision = await rails.ask(
            SPEC, labelled_state({"instruction": "What's my balance?", "tool": "send_money"}), model
        )
        self.assertEqual((decision.p, decision.band), (1.0, "act"))
        decision = await rails.ask(
            SPEC,
            labelled_state(
                {"instruction": "Use get_balance to check the account.", "tool": "get_balance", "arguments": "{}"}
            ),
            model,
        )
        self.assertEqual((decision.p, decision.band), (0.0, "allow"))


class TestEvaluate(IsolatedAsyncioTestCase):
    async def test_summary_reports_f1_false_allow_ask_rate_auroc_and_splits(self) -> None:
        records = [
            {
                "state": labelled_state({"instruction": "a", "tool": "t"}),
                "label": False,
                "split": "AgentDojo-Traj",
                "id": "a",
            },
            {
                "state": labelled_state({"instruction": "b", "tool": "t"}),
                "label": True,
                "split": "AgentDojo-Traj",
                "id": "b",
            },
            {"state": labelled_state({"instruction": "c", "tool": "t"}), "label": True, "split": "ASB-Traj", "id": "c"},
            {"state": labelled_state({"instruction": "d", "tool": "t"}), "label": True, "split": "ASB-Traj", "id": "d"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            labelled = Path(tmp) / "set.jsonl"
            labelled.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            summary = await rails.evaluate(
                SPEC, labelled, decision_model=_noul([0.1, 0.95, 0.5, 0.2]), results_dir=Path(tmp) / "results"
            )
        self.assertEqual((summary["records"], summary["positives"]), (4, 3))
        self.assertEqual((summary["acted"], summary["uncertain"]), (1, 1))
        self.assertEqual((summary["precision"], summary["recall"], summary["f1"]), (1.0, 0.333, 0.5))
        self.assertEqual(summary["false_allow"], 0.333)
        self.assertEqual(summary["ask_rate"], 0.25)
        self.assertIsNotNone(summary["auroc"])
        self.assertIn("AgentDojo-Traj", summary["splits"])
        self.assertIn("ASB-Traj", summary["splits"])
        self.assertEqual(summary["precision_ci"][1], 1.0)

    async def test_a_failed_decision_counts_as_ask_not_acted(self) -> None:
        class FailsFirst(ScriptedModel):
            def __init__(self) -> None:
                super().__init__(noul=[0.95], latency_ms=3, usage=Usage(input_tokens=10))
                self._attempts = 0

            async def decide_many(self, observation, questions):
                self._attempts += 1
                if self._attempts == 1:
                    raise build_error(StatusCode.MODEL_CALL_FAILED, error_msg="endpoint down")
                return await super().decide_many(observation, questions)

        records = [
            {"state": labelled_state({"tool": "a"}), "label": True, "split": "AgentDojo-Traj"},
            {"state": labelled_state({"tool": "b"}), "label": True, "split": "AgentDojo-Traj"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            labelled = Path(tmp) / "set.jsonl"
            labelled.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            summary = await rails.evaluate(SPEC, labelled, decision_model=FailsFirst(), results_dir=Path(tmp))
        self.assertEqual((summary["failures"], summary["acted"], summary["ask_rate"]), (1, 1, 0.5))
        self.assertEqual((summary["precision"], summary["recall"]), (1.0, 0.5))
