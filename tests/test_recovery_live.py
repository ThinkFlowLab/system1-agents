# coding: utf-8
"""The live recovery eval's non-browser pieces: the counting proxy, the plan order, the denominator and the cost.

These tests never build a model, open a browser or call an API. They exercise the counting proxy's honesty (retries
and failures are real calls, not ticks), the arm alternation, the fact that a raised trial stays in the denominator,
and the cost rule (null unless ``CHAT_USD_PER_M_*`` was explicitly set).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from s1a.decision_models import ChoiceQuestion, DecisionModel, Observation, Question, Reply, Usage
from s1a.pricing import ChatPrices

import evals.recovery.live as live
from evals.recovery.fixture import BLOCKED, DEFAULT_TASKS, NORMAL, RECOVERABLE, FormTask, post_submit, start_fixture
from evals.recovery.live import (
    ARMS,
    CountingDecisionModel,
    LiveConfig,
    LiveTrial,
    _cost,
    _failed_trial,
    _planned_trials,
    play_trial,
    render_markdown,
    run_live,
    summarize,
)

_QUESTION = ChoiceQuestion({"a": "A", "b": "B"})


def _choice_reply(key: str) -> Reply:
    return Reply(
        answers={"operation": {"choice": key, "probabilities": {"a": 1.0, "b": 0.0}, "confidence": 1.0}},
        latency_ms=7,
        usage=Usage(input_tokens=11, output_tokens=3),
        model="fake-v1",
    )


class _FakeDecisionModel(DecisionModel):
    """A stand-in backend: it can fail, answer unusably for a while, then answer correctly."""

    name = "fake"
    bills_input_tokens = False
    deterministic = False

    def __init__(self, *, fail_times: int = 0, invalid_times: int = 0, model: str = "fake-v1") -> None:
        self._fail_times = fail_times
        self._invalid_times = invalid_times
        self._model = model
        self.raw_calls = 0

    @property
    def model(self) -> str:
        return self._model

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        self.raw_calls += 1
        if self.raw_calls <= self._fail_times:
            raise RuntimeError("boom")
        if self.raw_calls <= self._fail_times + self._invalid_times:
            return _choice_reply("not-offered")
        return _choice_reply("a")


def test_proxy_counts_every_decide_call_including_the_validation_retry() -> None:
    proxy = CountingDecisionModel(_FakeDecisionModel(invalid_times=1))
    decision = asyncio.run(proxy.decide_many(Observation({"x": 1}), {"operation": _QUESTION}, attempts=2))

    assert decision.choice("operation").key == "a"
    assert proxy.decide_many_calls == 1, "one tick"
    assert len(proxy.calls) == 2, "the unusable answer cost a second real backend call"
    assert max(0, len(proxy.calls) - proxy.decide_many_calls) == 1, "the retry is not hidden"
    assert all(call["status"] == "ok" for call in proxy.calls)
    assert proxy.calls[0]["input_tokens"] == 11


def test_proxy_counts_a_raising_decide_as_a_failed_call_and_reraises() -> None:
    proxy = CountingDecisionModel(_FakeDecisionModel(fail_times=5))
    with pytest.raises(RuntimeError):
        asyncio.run(proxy.decide_many(Observation({"x": 1}), {"operation": _QUESTION}, attempts=2))

    assert len(proxy.calls) == 1, "the failure is one real call, never lost and never retried on a transport error"
    assert proxy.calls[0]["status"] == "error"
    assert proxy.calls[0]["error"] == "RuntimeError"
    assert proxy.calls[0]["usage_known"] is False, "a failed call's tokens are unknown, never a confirmed zero"
    assert proxy.usage_known is False


def test_proxy_usage_known_and_tokens_include_the_validation_retry() -> None:
    proxy = CountingDecisionModel(_FakeDecisionModel(invalid_times=1))
    asyncio.run(proxy.decide_many(Observation({"x": 1}), {"operation": _QUESTION}, attempts=2))

    assert proxy.usage_known is True
    assert sum(call["input_tokens"] for call in proxy.calls) == 22, "the retry's own tokens are kept too"


class _JevTransport:
    """A Jev transport double: returns one canned payload and records the request bodies; no network call."""

    model = "jev-fake"

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.bodies: list[dict] = []

    async def decide(self, body: dict) -> tuple[dict, int]:
        self.bodies.append(body)
        return self._payload, 4

    async def warm(self) -> None:
        return None

    async def close(self) -> None:
        return None


def _jev_reply_payload(*, usage: dict | None = None) -> dict:
    """A valid choice answer for ``_QUESTION``, optionally carrying a ``usage`` field."""
    payload = {
        "answers": {"operation": {"choice": "a", "probabilities": {"a": 1.0, "b": 0.0}, "confidence": 1.0}},
        "model": "jev-fake",
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _priced() -> ChatPrices:
    return ChatPrices(usd_per_input_token=1e-6, usd_per_output_token=2e-6, usd_per_cached_input_token=1e-6)


def _jev_proxy(payload: dict) -> CountingDecisionModel:
    from s1a.decision_models.jev import JevModel

    return CountingDecisionModel(JevModel(_JevTransport(payload)))


def test_jev_reply_without_usage_marks_the_call_and_the_cost_unknown() -> None:
    proxy = _jev_proxy(_jev_reply_payload())  # no usage key: the backend reported none
    decision = asyncio.run(proxy.decide_many(Observation({"x": 1}), {"operation": _QUESTION}, attempts=2))

    assert decision.choice("operation").key == "a"
    assert len(proxy.calls) == 1, "a valid answer is one call"
    assert proxy.calls[0]["usage_known"] is False
    assert proxy.usage_known is False
    cost, basis = _cost(
        usage_known=True,
        decision_usage_known=proxy.usage_known,
        jev_input_tokens=proxy.calls[0]["input_tokens"],
        chat_input_tokens=0,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        prices=_priced(),
    )
    assert cost is None and "decision call did not report usage" in basis


def test_jev_reply_with_explicit_zero_usage_is_known() -> None:
    proxy = _jev_proxy(_jev_reply_payload(usage={"input_tokens": 0, "output_tokens": 0}))
    asyncio.run(proxy.decide_many(Observation({"x": 1}), {"operation": _QUESTION}, attempts=2))

    assert proxy.calls[0]["usage_known"] is True, "an explicit zero is reported, not missing"
    assert proxy.usage_known is True
    cost, _ = _cost(
        usage_known=True,
        decision_usage_known=proxy.usage_known,
        jev_input_tokens=0,
        chat_input_tokens=0,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        prices=_priced(),
    )
    assert cost == 0.0, "known zero usage with configured prices is a zero estimate, never unknown"


@pytest.mark.parametrize("bad", [{"input_tokens": -1}, {"input_tokens": True}, {"input_tokens": 1.5}, {}])
def test_jev_reply_with_malformed_usage_is_unknown(bad: dict) -> None:
    proxy = _jev_proxy(_jev_reply_payload(usage=bad))
    asyncio.run(proxy.decide_many(Observation({"x": 1}), {"operation": _QUESTION}, attempts=2))

    assert proxy.calls[0]["usage_known"] is False
    assert proxy.usage_known is False


def test_jev_reply_with_a_valid_input_only_usage_is_known() -> None:
    proxy = _jev_proxy(_jev_reply_payload(usage={"input_tokens": 9}))
    asyncio.run(proxy.decide_many(Observation({"x": 1}), {"operation": _QUESTION}, attempts=2))

    assert proxy.calls[0]["input_tokens"] == 9
    assert proxy.calls[0]["usage_known"] is True, "an input-only usage is valid: output may stay zero"


@pytest.mark.parametrize("bills_input_tokens", [False, True])
def test_proxy_preserves_the_inner_interface(bills_input_tokens: bool) -> None:
    inner = _FakeDecisionModel(model="fake-v9")
    inner.deterministic = True
    inner.question_types = frozenset({"choice"})
    inner.supports_images = True
    inner.bills_input_tokens = bills_input_tokens
    proxy = CountingDecisionModel(inner)

    assert (proxy.name, proxy.model) == ("fake", "fake-v9")
    assert proxy.deterministic is True
    assert proxy.question_types == frozenset({"choice"})
    assert proxy.supports_images is True
    assert proxy.bills_input_tokens is bills_input_tokens


def test_planned_trials_alternate_the_arm_order_per_repeat() -> None:
    planned = _planned_trials(2, (NORMAL, RECOVERABLE), ARMS)
    first = [(task.name, arm, index) for task, arm, index in planned if task.name == "normal"]
    assert first == [("normal", "off", 0), ("normal", "on", 0), ("normal", "on", 1), ("normal", "off", 1)]
    assert len(planned) == 8, "2 tasks x 2 repeats x 2 arms, every plan explicit"


def _trial(
    task: str,
    arm: str,
    repeat: int,
    *,
    verified: bool = False,
    errored: bool = False,
    decision_calls: int = 1,
    chat_calls: int = 1,
    elapsed_s: float = 1.0,
    wasted_actions: int = 0,
    cost: float | None = None,
) -> LiveTrial:
    return LiveTrial(
        task=task,
        arm=arm,
        repeat=repeat,
        verified=verified,
        terminal="verified" if verified else ("timeout" if errored else "BLOCKED"),
        errored=errored,
        error="boom" if errored else None,
        model="fake",
        platform="test",
        elapsed_s=elapsed_s,
        decision_calls=decision_calls,
        chat_calls=chat_calls,
        wasted_actions=wasted_actions,
        cost_usd=cost,
    )


def test_summary_keeps_an_errored_trial_in_the_denominator() -> None:
    trials = [
        _trial("normal", "off", 0, verified=True, cost=None),
        _trial("normal", "on", 0, verified=True, cost=None),
        _trial("permanently_blocked", "off", 0, errored=True, cost=None),
    ]
    summary = summarize(trials, ARMS, model_id="fake-v1", model_name="fake")

    assert summary["planned_trials"] == 3
    assert summary["arms"]["off"]["planned"] == 2, "the raised trial still counts"
    assert summary["arms"]["off"]["errored"] == 1
    assert summary["arms"]["off"]["completion_rate"] == 0.5
    assert summary["arms"]["off"]["cost_usd"] is None, "one unknown cost makes the arm total unknown, not zero"
    assert summary["paired"]["incomplete_pairs"] == 1
    assert "planner attempts" in render_markdown(summary)


def test_cost_is_null_unless_prices_are_explicitly_configured() -> None:
    unknown, basis = _cost(
        usage_known=True,
        decision_usage_known=True,
        jev_input_tokens=0,
        chat_input_tokens=1000,
        chat_output_tokens=10,
        chat_cache_tokens=0,
        prices=None,
    )
    assert unknown is None
    assert "not verified" in basis

    prices = ChatPrices(usd_per_input_token=1e-6, usd_per_output_token=2e-6, usd_per_cached_input_token=1e-6)
    valued, basis = _cost(
        usage_known=True,
        decision_usage_known=True,
        jev_input_tokens=0,
        chat_input_tokens=1_000_000,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        prices=prices,
    )
    assert valued == 1.0
    assert "estimate" in basis and "not a provider bill" in basis

    unvalued, basis = _cost(
        usage_known=False,
        decision_usage_known=True,
        jev_input_tokens=0,
        chat_input_tokens=1_000_000,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        prices=prices,
    )
    assert unvalued is None and "did not report usage" in basis

    incomplete, basis = _cost(
        usage_known=True,
        decision_usage_known=False,
        jev_input_tokens=1_000,
        chat_input_tokens=1_000_000,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        prices=prices,
    )
    assert incomplete is None and "decision call did not report usage" in basis


def test_a_raised_trial_is_recorded_not_dropped() -> None:
    trial = _failed_trial(BLOCKED, "on", 2, TimeoutError("took too long"), model_id="fake-v1", elapsed_s=3.5)

    assert trial.errored and trial.terminal == "timeout"
    assert trial.task == "permanently_blocked" and trial.arm == "on" and trial.repeat == 2
    assert trial.model == "fake-v1" and trial.cost_usd is None
    assert trial.oracle == {"task": "permanently_blocked", "verified": False, "submissions": []}


def test_fixture_tasks_are_the_three_expected_kinds() -> None:
    assert [task.kind for task in (NORMAL, RECOVERABLE, BLOCKED)] == ["normal", "recoverable", "blocked"]
    assert isinstance(NORMAL, FormTask)


def _fake_chat():
    """A chat model good enough for ``CountingModel``: a real client config, no network call ever made here."""
    from openjiuwen.core.foundation.llm import init_model

    return init_model(provider="openai", model_name="fake-chat", api_key="x", api_base="http://127.0.0.1:1/v1")


def test_play_trial_keeps_a_verified_post_and_marks_errored_when_browse_raises(monkeypatch, tmp_path) -> None:
    task, tasks = NORMAL, (NORMAL,)
    with start_fixture(tasks) as fixture:

        async def fake_browse(spec, policy, **kwargs):  # noqa: ANN001, ANN003 - mirrors browse's signature loosely
            post_submit(fixture, task, task.expected_value)  # the real POST this trial's own fixture server records
            raise RuntimeError("provider headers: {Authorization: Bearer sk-secret}")

        monkeypatch.setattr(live.browse, "browse", fake_browse)
        trial = asyncio.run(
            play_trial(
                fixture,
                task,
                "on",
                0,
                config=LiveConfig(),
                logs_dir=tmp_path / "logs",
                tasks=tasks,
                model_name="jev",
                chat=_fake_chat(),
                decision_model=_FakeDecisionModel(),
                prices=None,
            )
        )

    assert trial.verified is True, "the fixture recorded the real POST"
    assert trial.errored is True, "the run still raised after the POST: verified and errored are independent"
    assert trial.error == "RuntimeError", "the summary is the exception type, never the provider's message"
    assert "sk-secret" not in json.dumps(trial.as_json())
    assert trial.oracle["verified"] is True and trial.oracle["submissions"]


def test_play_trial_keeps_a_verified_post_and_a_returned_finalization_timeout(monkeypatch, tmp_path) -> None:
    task, tasks = NORMAL, (NORMAL,)
    with start_fixture(tasks) as fixture:

        async def fake_browse(spec, policy, **kwargs):  # noqa: ANN001, ANN003 - mirrors browse's signature loosely
            post_submit(fixture, task, task.expected_value)  # the real POST this trial's own fixture records
            return {"status": None, "error": "timeout after 180 s", "report": {}, "ticks": []}

        monkeypatch.setattr(live.browse, "browse", fake_browse)
        trial = asyncio.run(
            play_trial(
                fixture,
                task,
                "on",
                0,
                config=LiveConfig(),
                logs_dir=tmp_path / "logs",
                tasks=tasks,
                model_name="jev",
                chat=_fake_chat(),
                decision_model=_FakeDecisionModel(),
                prices=None,
            )
        )

    assert trial.verified is True, "the fixture recorded the real POST"
    assert trial.errored is True, "browse returned a timeout instead of raising: it must still fail the trial"
    assert trial.error == "timeout after 180 s"
    assert trial.terminal == "verified"
    assert trial.oracle["verified"] is True and trial.oracle["submissions"]


def test_chat_setup_failure_closes_the_decision_model(monkeypatch, tmp_path) -> None:
    closed = {"n": 0}

    class _FakeDecision(DecisionModel):
        name = "fake"

        @property
        def model(self) -> str:
            return "fake-v1"

        async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
            raise AssertionError("never called: chat setup fails first")

        async def close(self) -> None:
            closed["n"] += 1

    monkeypatch.setattr(live, "build_model", lambda model_name: _FakeDecision())

    def _no_chat():
        raise RuntimeError("chat model: set OPENAI_API_KEY (or LLM_API_KEY) and MODEL_NAME")

    monkeypatch.setattr(live, "chat_model_from_env", _no_chat)

    with pytest.raises(RuntimeError):
        asyncio.run(run_live(model_name="jev", repeat=1, tasks=(NORMAL,), arms=("off",), results_dir=tmp_path))

    assert closed["n"] == 1, "a chat setup failure must not leak the already-built decision model"
    manifest = next((tmp_path / "recovery_live").rglob("manifest.json"))
    planned = json.loads(manifest.read_text(encoding="utf-8"))
    assert planned["status"] in {"planned", "running"}, "the planned manifest is on disk even when setup fails"
    assert planned["planned_trials"] == 1 and planned["plan"][0]["task"] == "normal"


class _NullDecision(DecisionModel):
    """A stand-in backend that is never asked to decide: the trial function is replaced in these tests."""

    name = "fake"
    bills_input_tokens = False

    @property
    def model(self) -> str:
        return "fake-v1"

    async def _decide(self, observation: Observation, questions: dict[str, Question]) -> Reply:
        raise AssertionError("no model call: run_trial is replaced")

    async def close(self) -> None:
        return None


def _run_live_capturing_tasks(monkeypatch, tmp_path, config: LiveConfig):
    """Run ``run_live`` with the model, browser and Runner replaced; capture the tasks it passes to a trial."""
    from contextlib import asynccontextmanager

    captured: dict = {}

    async def fake_run_trial(task, arm, repeat, **kwargs):
        captured["tasks"] = kwargs["tasks"]
        return _trial(task.name, arm, repeat)

    @asynccontextmanager
    async def noop_runner():
        yield

    monkeypatch.setattr(live, "build_model", lambda model_name: _NullDecision())
    monkeypatch.setattr(live, "run_trial", fake_run_trial)
    monkeypatch.setattr(live, "started_runner", noop_runner)
    run = asyncio.run(
        run_live(
            model_name="jev",
            repeat=1,
            tasks=(NORMAL,),
            arms=("off",),
            config=config,
            results_dir=tmp_path,
            chat=_fake_chat(),
        )
    )
    return captured, run


def test_live_config_and_default_tasks_default_to_no_validation() -> None:
    assert LiveConfig().validate_submission is False
    assert all(task.validate_submission is False for task in DEFAULT_TASKS)


def test_run_live_uses_validated_fixtures_only_when_the_flag_is_on(monkeypatch, tmp_path) -> None:
    captured, run = _run_live_capturing_tasks(monkeypatch, tmp_path, LiveConfig(validate_submission=True))

    assert captured["tasks"] and all(task.validate_submission for task in captured["tasks"])
    assert NORMAL.validate_submission is False, "the supplied immutable task is never mutated"
    assert run.manifest["config"]["validate_submission"] is True, "the flag is recorded in the manifest"


def test_run_live_defaults_keep_the_supplied_tasks_unvalidated(monkeypatch, tmp_path) -> None:
    captured, run = _run_live_capturing_tasks(monkeypatch, tmp_path, LiveConfig())

    assert captured["tasks"] == (NORMAL,)
    assert not any(task.validate_submission for task in captured["tasks"])
    assert run.manifest["config"]["validate_submission"] is False
