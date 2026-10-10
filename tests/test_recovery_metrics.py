# coding: utf-8
"""Reporting regressions: overlapping probe timings and rounded completion rates."""

from evals.recovery.runner import _trial_metrics
from evals.recovery.summary import TrialRecord, paired_summary


def test_recorded_probe_time_does_not_double_count_settle_components() -> None:
    # The measured probe already contains action settling. WAIT probes can also
    # appear in both probe_ms and settle_ms; these are not disjoint durations.
    answer = {"ticks": [{"probe_ms": 100, "settle_ms": 100, "action_settle_ms": 80}]}
    metrics = _trial_metrics(answer, object(), object())
    assert metrics["recorded_probe_ms"] == 100


def test_completion_delta_is_calculated_before_display_rounding() -> None:
    records = [
        TrialRecord(
            task=str(i),
            arm=arm,
            repeat=0,
            verified=i < successes,
            terminal="verified" if i < successes else "BLOCKED",
            errored=False,
            scripted_model="scripted",
            platform="test",
        )
        for arm, successes in (("off", 1), ("on", 2))
        for i in range(3)
    ]
    summary = paired_summary(records)
    assert summary["on_minus_off_completion_rate"] == 0.333


def test_successful_desktop_arm_is_not_a_recovery_failure() -> None:
    from evals.desktop_recovery import _recovery_fields
    from s1a.jobs import Episode

    episode = Episode(
        env="desktop",
        policy="scripted-on",
        seed=0,
        score=1.0,
        steps=2,
        elapsed_s=0.1,
        started_at="",
        finished_at="",
        final_state={},
        chat_calls=0,
        chat_input_tokens=0,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        jev_input_tokens=0,
        invalid_keys=0,
        cost_usd=0.0,
    )
    episode.extra["terminal"] = {"status": "DONE", "reason": "environment done"}
    fields = _recovery_fields(episode, bounded=True)
    assert not fields["recovery_failed"]
    assert fields["recovery_next_action"] is None
    assert fields["next_action_source"] is None


def test_desktop_cli_fails_when_a_planned_trial_errors(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    from evals import desktop_recovery as desktop

    run = SimpleNamespace(
        summary={},
        records=[SimpleNamespace(errored=True)],
        run_dir=tmp_path,
        job_dirs=[],
    )
    monkeypatch.setattr(desktop, "default_config", lambda **kwargs: object())
    monkeypatch.setattr(desktop, "run_sync", lambda **kwargs: run)
    monkeypatch.setattr(desktop, "render_markdown", lambda summary: "")
    assert desktop.main(["--driver", "unused"]) == 1
