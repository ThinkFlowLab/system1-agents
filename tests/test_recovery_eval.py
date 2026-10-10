# coding: utf-8
"""The recovery eval's non-browser pieces: the fixture oracle, the plan-gated scripted switch, and the denominator.

The real-browser path lives in ``tests/system/test_recovery_browser.py`` and is opt-in. These tests never launch a
browser: they exercise the loopback fixture server with plain HTTP and the paired summary with hand-built records.
"""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from evals.recovery import runner
from evals.recovery.fixture import (
    BLOCKED,
    DEFAULT_TASKS,
    NORMAL,
    RECOVERABLE,
    FormTask,
    page_html,
    post_submit,
    result_html,
    start_fixture,
    validated_tasks,
)
from evals.recovery.runner import EvalConfig, run_eval
from evals.recovery.scripted import SCRIPTED_MODEL_ID, ScriptedChatModel, ScriptedRecoveryModel
from evals.recovery.summary import TrialRecord, paired_summary, render_markdown
from s1a.browser import action_space
from s1a.tool.rethink import REPLAN_PROMPT


def _snapshot(*, value: str, with_enable: bool, with_input: bool = True) -> dict:
    elements = []
    if with_input:
        elements.append(
            {"node": 1, "target_id": "t1", "role": "textbox", "label": "Value", "value": value, "editable": True}
        )
    if with_enable:
        elements.append(
            {"node": 2, "target_id": "t2", "role": "button", "label": "Enable editing", "value": "", "editable": False}
        )
    elements.append({"node": 3, "target_id": "t3", "role": "button", "label": "Submit", "value": "", "editable": False})
    return {
        "url": "http://127.0.0.1:1/recoverable",
        "title": "Recovery",
        "text": "Value Submit",
        "page_key": "k1",
        "generation_id": "g1",
        "can_scroll_down": False,
        "can_scroll_up": False,
        "elements": elements,
    }


def _turn(model: ScriptedRecoveryModel, snapshot: dict, plan: str = "") -> str:
    """One scripted turn over the observation production builds, with the plan attached as draft_plan leaves it."""
    space = action_space.build_action_space(snapshot, [])
    questions = action_space.build_questions(space, goal="complete the form", values=[], rules="r", language="en")
    observation = action_space.build_observation(space, snapshot, [])
    if plan and isinstance(observation.state, dict):
        observation.state["plan"] = plan
    reply = asyncio.run(model._decide(observation, questions))
    operation = reply.answers["operation"]
    return operation["choice"]


class TestFixtureOracle(TestCase):
    def test_only_a_real_submit_verifies_and_the_pages_carry_their_controls(self) -> None:
        assert "Enable editing" in page_html(RECOVERABLE) and "__editing" in page_html(RECOVERABLE)
        assert "Enable editing" not in page_html(BLOCKED) and "__editing" in page_html(BLOCKED)
        assert "Enable editing" not in page_html(NORMAL) and "value=''" in page_html(NORMAL)
        with start_fixture() as fixture:
            self.assertFalse(fixture.verified(NORMAL))
            post_submit(fixture, NORMAL, "hello world")
            self.assertTrue(fixture.verified(NORMAL))
            self.assertFalse(fixture.verified(RECOVERABLE), "one task's submit never verifies another")
            post_submit(fixture, RECOVERABLE, "wrong")
            self.assertFalse(fixture.verified(RECOVERABLE), "a real submit with the wrong value is not completion")
            self.assertEqual(len(fixture.submissions(NORMAL)), 1)
            self.assertEqual(fixture.oracle_state(NORMAL)["verified"], True)

    def test_the_success_page_escapes_the_submitted_value_as_data(self) -> None:
        hostile = "<script>alert('x')</script>"
        with start_fixture() as fixture:
            status, body = post_submit(fixture, NORMAL, hostile)
            self.assertEqual(status, 200)
            self.assertNotIn(hostile, body, "model text must never run in the fixture origin")
            self.assertIn("&lt;script&gt;", body)
            self.assertFalse(fixture.verified(NORMAL), "the hostile value does not match the expected one")
            self.assertEqual(
                fixture.submissions(NORMAL)[0]["fields"]["value"],
                hostile,
                "the oracle records the raw submitted value, independent of the rendered page",
            )

    def test_result_html_escapes_both_the_task_name_and_the_value(self) -> None:
        body = result_html("<b>task</b>", "<script>x</script>")
        self.assertIn("&lt;b&gt;", body)
        self.assertIn("&lt;script&gt;", body)
        self.assertNotIn("<b>", body)
        self.assertNotIn("<script>", body)


class TestValidatedFixtureSubmission(TestCase):
    """The opt-in server-side validation: a mismatched POST is rejected with the retryable form, never a success page.

    This is a variant for the exploratory follow-up, not a change to the default fixtures: DEFAULT_TASKS keep
    validate_submission off and their exact original behaviour.
    """

    def test_default_tasks_stay_unvalidated_and_the_fixture_is_unchanged(self) -> None:
        self.assertTrue(all(task.validate_submission is False for task in DEFAULT_TASKS))
        with start_fixture() as fixture:
            status, body = post_submit(fixture, NORMAL, "wrong")
            self.assertEqual(status, 200, "the default fixture always serves the result page")
            self.assertIn("Submitted", body)
            self.assertNotIn("validation-error", body, "the default page gains no new markup")
            self.assertFalse(fixture.verified(NORMAL), "the oracle still requires the expected value")

    def test_a_mismatched_value_is_rejected_with_the_same_retryable_form(self) -> None:
        tasks = validated_tasks((NORMAL,))
        with start_fixture(tasks) as fixture:
            task = tasks[0]
            status, body = post_submit(fixture, task, "wrong")
            self.assertEqual(status, 422)
            self.assertIn("<form", body, "the rejected page still shows the form to retry")
            self.assertIn('id="validation-error"', body)
            self.assertNotIn("Submitted", body, "a rejected submit never shows the success result page")
            self.assertFalse(fixture.verified(task))
            self.assertEqual(len(fixture.submissions(task)), 1, "the invalid attempt is recorded, not dropped")

    def test_a_correct_value_later_succeeds_and_the_invalid_attempt_survives(self) -> None:
        tasks = validated_tasks((NORMAL,))
        with start_fixture(tasks) as fixture:
            task = tasks[0]
            post_submit(fixture, task, "wrong")
            status, body = post_submit(fixture, task, task.expected_value)
            self.assertEqual(status, 200)
            self.assertIn("Submitted", body)
            self.assertTrue(fixture.verified(task))
            values = [record["fields"]["value"] for record in fixture.submissions(task)]
            self.assertEqual(values, ["wrong", task.expected_value], "every real POST, rejected ones included")

    def test_the_validation_error_html_escapes_submitted_data(self) -> None:
        tasks = validated_tasks((NORMAL,))
        with start_fixture(tasks) as fixture:
            hostile = "<script>alert('x')</script>"
            status, body = post_submit(fixture, tasks[0], hostile)
            self.assertEqual(status, 422)
            self.assertNotIn(hostile, body)
            self.assertIn("&lt;script&gt;", body)

    def test_building_validated_tasks_never_mutates_the_originals(self) -> None:
        built = validated_tasks(DEFAULT_TASKS)
        self.assertTrue(all(task.validate_submission for task in built))
        self.assertTrue(all(task.validate_submission is False for task in DEFAULT_TASKS))
        self.assertEqual([task.name for task in built], [task.name for task in DEFAULT_TASKS])


class TestTrialOracleIsIsolated(TestCase):
    """A success in one trial must never carry into the next, whichever arm order the runner picked."""

    def test_each_trial_gets_a_fresh_namespace_and_a_later_trial_cannot_reuse_a_submit(self) -> None:
        with start_fixture() as first:
            first_id = first.trial_id
            post_submit(first, RECOVERABLE, RECOVERABLE.expected_value)
            self.assertTrue(first.verified(RECOVERABLE))
            self.assertEqual(first.submissions(RECOVERABLE)[0]["trial"], first_id)
        with start_fixture() as second:  # the very next trial starts empty, so that success cannot verify it
            self.assertNotEqual(second.trial_id, first_id)
            self.assertFalse(second.verified(RECOVERABLE))
            self.assertEqual(second.submissions(RECOVERABLE), [])

    def test_on_then_off_order_cannot_let_the_on_submit_verify_the_off_trial(self) -> None:
        with start_fixture() as on_trial:  # arm "on" succeeds and records a real submit
            post_submit(on_trial, NORMAL, NORMAL.expected_value)
            self.assertTrue(on_trial.verified(NORMAL))
        with start_fixture() as off_trial:  # arm "off" never submits; its own empty oracle must say so
            self.assertFalse(off_trial.verified(NORMAL))
            self.assertEqual(off_trial.oracle_state(NORMAL)["verified"], False)

    def test_a_failed_trial_leaves_nothing_for_the_next(self) -> None:
        with start_fixture() as failed:  # a wrong-value submit is not completion and must not survive it
            post_submit(failed, RECOVERABLE, "wrong")
            self.assertFalse(failed.verified(RECOVERABLE))
        with start_fixture() as next_trial:
            self.assertFalse(next_trial.verified(RECOVERABLE))
            self.assertEqual(next_trial.submissions(RECOVERABLE), [])


class TestScriptedSwitchIsPlanGated(TestCase):
    def test_recoverable_faults_until_a_plan_then_unlocks(self) -> None:
        model = ScriptedRecoveryModel(RECOVERABLE)
        snapshot = _snapshot(value="original", with_enable=True)
        self.assertEqual(_turn(model, snapshot), "TYPE_TEXT", "no plan yet: the injected fault keeps filling")
        self.assertEqual(_turn(model, snapshot), "TYPE_TEXT")
        self.assertEqual(_turn(model, snapshot, plan="click Enable editing"), "CLICK", "a plan unlocks the control")
        self.assertTrue(model._unlocked)
        enabled = _snapshot(value="original", with_enable=False)
        self.assertEqual(_turn(model, enabled), "TYPE_TEXT")
        filled = _snapshot(value=RECOVERABLE.expected_value, with_enable=False)
        self.assertEqual(_turn(model, filled), "CLICK", "once the field holds the value, submit")
        self.assertEqual(_turn(model, {"elements": []}), "DONE")

    def test_blocked_never_leaves_the_fault(self) -> None:
        model = ScriptedRecoveryModel(BLOCKED)
        snapshot = _snapshot(value="original", with_enable=False)
        for plan in ("", "click Enable editing", ""):
            self.assertEqual(_turn(model, snapshot, plan=plan), "TYPE_TEXT", "no offered unlock exists on this page")

    def test_normal_types_then_submits_then_stops(self) -> None:
        model = ScriptedRecoveryModel(NORMAL)
        empty = _snapshot(value="", with_enable=False)
        self.assertEqual(_turn(model, empty), "TYPE_TEXT")
        filled = _snapshot(value=NORMAL.expected_value, with_enable=False)
        self.assertEqual(_turn(model, filled), "CLICK")
        self.assertEqual(_turn(model, {"elements": []}), "DONE")

    def test_scripted_model_is_labelled_and_makes_no_network_call(self) -> None:
        model = ScriptedRecoveryModel(NORMAL)
        self.assertEqual(model.model, SCRIPTED_MODEL_ID)
        self.assertEqual(model.name, "scripted", "a scripted double is not the jev backend")

    def test_scripted_decision_latency_is_measured_not_a_fabricated_constant(self) -> None:
        model = ScriptedRecoveryModel(NORMAL)
        snapshot = _snapshot(value="", with_enable=False)
        space = action_space.build_action_space(snapshot, [])
        questions = action_space.build_questions(space, goal="x", values=[], rules="r", language="en")
        observation = action_space.build_observation(space, snapshot, [])
        reply = asyncio.run(model._decide(observation, questions))
        self.assertGreaterEqual(reply.latency_ms, 0)
        self.assertLess(reply.latency_ms, 1000, "a real perf_counter value, not a claimed benchmark latency")

    def test_chat_model_counts_the_planner_calls_separately_from_chat(self) -> None:
        chat = ScriptedChatModel(value="v", plan="click Enable editing")
        asyncio.run(chat.invoke([{"role": "system", "content": REPLAN_PROMPT}]))
        asyncio.run(chat.invoke([{"role": "system", "content": "something else"}]))
        self.assertEqual((chat.invoke_calls, chat.planner_calls), (2, 1))


def _record(
    task: str,
    arm: str,
    *,
    verified: bool,
    repeat: int = 0,
    errored: bool = False,
    model_calls: int = 3,
    elapsed_s: float = 1.0,
    wasted_actions: int = 0,
) -> TrialRecord:
    return TrialRecord(
        task=task,
        arm=arm,
        repeat=repeat,
        verified=verified,
        terminal="verified" if verified else ("timeout" if errored else "BLOCKED"),
        errored=errored,
        scripted_model=SCRIPTED_MODEL_ID,
        platform="test",
        model_calls=model_calls,
        elapsed_s=elapsed_s,
        wasted_actions=wasted_actions,
        cost_usd=0.0,
    )


class TestPairedSummaryDenominator(TestCase):
    def test_every_planned_trial_counts_even_when_errored(self) -> None:
        records = [
            _record("normal", "off", verified=True),
            _record("recoverable", "off", verified=False),
            _record("permanently_blocked", "off", verified=False),
            _record("normal", "on", verified=True),
            _record("recoverable", "on", verified=True),
            _record("permanently_blocked", "on", verified=False),
            _record("permanently_blocked", "on", verified=False, repeat=1, errored=True),
        ]
        summary = paired_summary(records)

        self.assertEqual(summary["planned_trials"], 7)
        self.assertEqual(summary["arms"]["off"]["planned"], 3)
        self.assertEqual(summary["arms"]["on"]["planned"], 4, "the errored trial still counts in the denominator")
        self.assertEqual(summary["arms"]["off"]["completion_rate"], 0.333)
        self.assertEqual(summary["arms"]["on"]["completion_rate"], 0.5)
        self.assertEqual(summary["arms"]["on"]["errored"], 1)
        self.assertEqual(summary["on_minus_off_completion_rate"], 0.167)
        paired = summary["paired"]
        self.assertEqual(
            (paired["pairs"], paired["incomplete_pairs"], paired["on_only_verified"], paired["neither_verified"]),
            (3, 1, 1, 1),
        )
        text = render_markdown(summary)
        self.assertIn("| on | 4 | 2 | 0.5 |", text)
        self.assertIn("scripted doubles", text)

    def test_no_records_is_an_empty_but_valid_summary(self) -> None:
        summary = paired_summary([])
        self.assertEqual(summary["planned_trials"], 0)
        self.assertIsNone(summary["arms"]["on"]["completion_rate"])

    def test_paired_mean_deltas_are_on_minus_off_over_complete_pairs(self) -> None:
        records = [
            _record("normal", "off", verified=True, repeat=0, model_calls=4, elapsed_s=10.0, wasted_actions=1),
            _record("normal", "on", verified=True, repeat=0, model_calls=6, elapsed_s=13.0, wasted_actions=3),
            _record("recoverable", "off", verified=False, repeat=0, model_calls=8, elapsed_s=20.0, wasted_actions=5),
            _record("recoverable", "on", verified=True, repeat=0, model_calls=5, elapsed_s=17.0, wasted_actions=2),
            _record("normal", "on", verified=True, repeat=1, model_calls=9, elapsed_s=99.0, wasted_actions=0),
        ]
        paired = paired_summary(records)["paired"]
        self.assertEqual(paired["pairs"], 2)
        self.assertEqual(paired["incomplete_pairs"], 1, "the unpaired on-only repeat is not silently dropped")
        self.assertEqual(paired["mean_delta_model_calls"], -0.5)
        self.assertEqual(paired["mean_delta_elapsed_s"], 0.0)
        self.assertEqual(paired["mean_delta_wasted_actions"], -0.5)
        self.assertIn("Paired mean on-minus-off deltas", render_markdown(paired_summary(records)))


class TestRunArgumentValidation(TestCase):
    """Bad arguments must fail loudly, not quietly produce a plausible but meaningless report."""

    def test_repeat_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(run_eval(repeat=0, config=EvalConfig()))

    def test_arms_must_be_known_and_distinct(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(run_eval(arms=(), config=EvalConfig()))
        with self.assertRaises(ValueError):
            asyncio.run(run_eval(arms=("off", "off"), config=EvalConfig()))
        with self.assertRaises(ValueError):
            asyncio.run(run_eval(arms=("sideways",), config=EvalConfig()))

    def test_tasks_must_be_present_and_distinct(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(run_eval(tasks=(), config=EvalConfig()))
        with self.assertRaises(ValueError):
            asyncio.run(run_eval(tasks=(NORMAL, NORMAL), config=EvalConfig()))


def _get(url: str) -> tuple[int, str]:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - a loopback URL built by the fixture under test
        return response.status, response.read().decode("utf-8", errors="replace")


def _post(url: str, value: str) -> tuple[int, str]:
    body = urlencode({"value": value}).encode()
    request = Request(url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - a loopback URL built by the fixture under test
            return response.status, response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:  # a rejected submit is an expected status, so read its body and return it
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _goal_url(goal: str) -> str:
    prefix = "Open "
    start = goal.index(prefix) + len(prefix)
    return goal[start : goal.index(" and ", start)]


@asynccontextmanager
async def _noop_runner():
    yield


def _recording_browse(tasks: tuple[FormTask, ...], observations: list[dict]) -> object:
    """Read the trial fixture over HTTP and submit values without launching a browser."""
    by_path = {task.path: task for task in tasks}

    async def fake_browse(spec, policy, *, model_name, goal, **kwargs):
        url = _goal_url(goal)
        parsed = urlparse(url)
        task = by_path.get(parsed.path)
        if task is None:
            raise AssertionError(f"trial opened {url!r}, which is no requested task's path")
        status, body = _get(url)
        observations.append({"task": task.name, "on": policy.rethink_on, "status": status, "body": body})
        if status != 200:
            raise AssertionError(f"{task.name}: fixture served {status} at {url}, not the requested page")
        if body != page_html(task):
            raise AssertionError(f"{task.name}: fixture served a page that is not the requested task's page")
        base = f"{parsed.scheme}://{parsed.netloc}"
        index_status, index_body = _get(f"{base}/")
        if index_status != 200 or f"href='{task.path}'" not in index_body:
            raise AssertionError(f"{task.name}: fixture index does not offer the requested task")
        wrong_status, _ = _post(f"{base}/submit/{task.name}", "tampered")
        if task.validate_submission and wrong_status != 422:
            raise AssertionError(f"{task.name}: a wrong value was not rejected with 422 (got {wrong_status})")
        expected_status, _ = _post(f"{base}/submit/{task.name}", task.expected_value)
        if expected_status != 200:
            raise AssertionError(f"{task.name}: the expected value did not submit (got {expected_status})")
        return {"status": "BLOCKED", "report": {"history": [], "recovery": {}}, "ticks": []}

    return fake_browse


class TestRunEvalServesItsOwnTasks(TestCase):
    """Custom routes, page behavior and submission validation must reach both recovery arms."""

    def _run(self, tasks: tuple[FormTask, ...], observations: list[dict]) -> object:
        with (
            mock.patch.object(runner, "started_runner", _noop_runner),
            mock.patch("evals.recovery.runner.browse.browse", _recording_browse(tasks, observations)),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                with mock.patch.object(runner, "HOME", tmp_path):
                    return asyncio.run(
                        run_eval(
                            tasks=tasks,
                            repeat=1,
                            config=EvalConfig(timeout_s=5.0),
                            results_dir=tmp_path / "results",
                        )
                    )

    def test_custom_path_and_reused_default_path_are_served_to_both_arms(self) -> None:
        custom = FormTask("custom_widget", "/widget", "Custom widget", "widget value", "normal")
        reused = FormTask(
            "relocked_normal", "/normal", "Relocked normal", "hello world", "recoverable", validate_submission=True
        )
        observations: list[dict] = []
        run = self._run((custom, reused), observations)

        self.assertEqual(len(run.records), 4, "two tasks times the off and on arms")
        for record in run.records:
            self.assertFalse(record.errored, f"{record.task}/{record.arm}: a served mismatch must surface, not play")
            self.assertTrue(record.verified, f"{record.task}/{record.arm}: the requested task's oracle saw its submit")
            self.assertEqual(record.oracle["task"], record.task)
            self.assertTrue(record.oracle["verified"])
            self.assertTrue(all(row["task"] == record.task for row in record.oracle["submissions"]))

        by_task: dict[str, list[dict]] = {}
        for observed in observations:
            by_task.setdefault(observed["task"], []).append(observed)
        self.assertEqual(set(by_task), {custom.name, reused.name}, "only the requested tasks are ever served")
        for task in (custom, reused):
            seen = by_task[task.name]
            self.assertEqual(len(seen), 2, "the off and on arms both hit the requested task")
            self.assertEqual({row["on"] for row in seen}, {False, True})
            for row in seen:
                self.assertEqual(row["status"], 200)
                self.assertEqual(row["body"], page_html(task), "the served page is exactly the requested task's page")


class TestScriptedPlayTrialKeepsOracleAndErrorIndependent(TestCase):
    """A successful POST and a run failure are different facts: keep both, never let one clear the other."""

    def _play(self, fixture, answer: dict, *, post: bool = True) -> tuple:
        async def fake_browse(spec, policy, **kwargs):  # noqa: ANN001, ANN003 - mirrors browse's signature loosely
            if post:
                post_submit(fixture, NORMAL, NORMAL.expected_value)
            return answer

        with mock.patch("evals.recovery.runner.browse.browse", fake_browse):
            with tempfile.TemporaryDirectory() as tmp:
                return asyncio.run(
                    runner.play_trial(fixture, NORMAL, "on", 0, config=EvalConfig(timeout_s=5.0), logs_dir=Path(tmp))
                )

    def test_a_verified_post_then_a_returned_timeout_is_both_verified_and_errored(self) -> None:
        with start_fixture() as fixture:
            episode, record = self._play(fixture, {"status": None, "error": "timeout after 120 s"})
        self.assertTrue(record.verified, "the fixture recorded the real POST")
        self.assertTrue(record.errored, "the run still reported a timeout: verified and errored are independent")
        self.assertEqual(record.terminal, "verified")
        self.assertEqual(episode.error, "timeout after 120 s")

    def test_a_played_blocked_status_is_score_zero_not_a_harness_error(self) -> None:
        with start_fixture() as fixture:
            episode, record = self._play(
                fixture, {"status": "BLOCKED", "error": "jev recovery give_up: spent"}, post=False
            )
        self.assertFalse(record.verified)
        self.assertEqual(record.terminal, "BLOCKED")
        self.assertFalse(record.errored, "a played BLOCKED keeps score 0; a policy message is not a harness error")
        self.assertIsNone(episode.error)

    def test_a_played_blocked_episode_without_a_run_error_is_not_errored(self) -> None:
        with start_fixture() as fixture:
            _, record = self._play(fixture, {"status": "BLOCKED"}, post=False)
        self.assertFalse(record.verified)
        self.assertFalse(record.errored, "a played BLOCKED episode is a score 0, not a raised trial")


class TestRunEvalRunIdIsUnique(TestCase):
    """The clock alone has one-second precision; two runs in the same second must not share logs or summaries."""

    def _run(self, results_dir: Path, suffix: str) -> object:
        fixed = datetime(2026, 10, 7, 12, 0, 0)

        class _Clock:
            @staticmethod
            def now() -> datetime:
                return fixed

        with (
            mock.patch.object(runner, "started_runner", _noop_runner),
            mock.patch("evals.recovery.runner.browse.browse", _recording_browse((NORMAL,), [])),
            mock.patch.object(runner, "datetime", _Clock),
            mock.patch.object(runner, "uuid4", return_value=SimpleNamespace(hex=suffix)),
            mock.patch.object(runner, "HOME", results_dir),
        ):
            return asyncio.run(
                run_eval(
                    tasks=(NORMAL,),
                    arms=("off",),
                    repeat=1,
                    config=EvalConfig(timeout_s=5.0),
                    results_dir=results_dir / "results",
                )
            )

    def test_two_runs_in_the_same_second_get_distinct_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = self._run(Path(tmp), "abc123")
            second = self._run(Path(tmp), "def456")
        self.assertNotEqual(first.paired_dir, second.paired_dir)
        self.assertTrue(first.paired_dir.name.endswith("abc123"))
        self.assertTrue(second.paired_dir.name.endswith("def456"))
