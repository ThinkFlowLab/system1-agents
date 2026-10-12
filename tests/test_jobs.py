# coding: utf-8
"""Summaries and the Harbor-shaped job folders."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import TestCase

from s1a.jobs import Episode, summarize, write_job


def _episode(seed: int, score: float, steps: int) -> Episode:
    stamp = datetime.now().isoformat()
    return Episode(
        env="counter",
        policy="fixed",
        seed=seed,
        score=score,
        steps=steps,
        elapsed_s=0.1,
        started_at=stamp,
        finished_at=stamp,
        final_state={"n": steps},
        chat_calls=2,
        chat_input_tokens=500,
        chat_output_tokens=20,
        chat_cache_tokens=50,
        jev_input_tokens=300,
        invalid_keys=1,
        cost_usd=0.001,
        decisions=[{"step": 1, "key": "inc", "confidence": 1.0, "ms": 10, "source": "fixed"}],
        extra={"rethinks": [{"kind": "repeat", "step": 1, "key": "inc"}]},
        views=[{"state": {"n": 0}, "candidates": {"inc": "add one"}, "done": False, "score": 0.0}],
    )


class TestSummaryAndJob(TestCase):
    def test_summary_mean_and_bootstrap_bounds(self) -> None:
        summary = summarize([_episode(0, 1.0, 3), _episode(1, 0.0, 3), _episode(2, 1.0, 3), _episode(3, 1.0, 3)])
        self.assertEqual(summary["mean_score"], 0.75)
        low, high = summary["mean_score_ci95"]
        self.assertLessEqual(low, 0.75)
        self.assertGreaterEqual(high, 0.75)
        self.assertEqual((summary["wins"], summary["losses"], summary["episodes"]), (3, 0, 4))
        self.assertEqual((summary["median_decision_ms"], summary["rethinks"]), (10, 4))
        self.assertEqual((summary["decisions"], summary["chat_calls"], summary["invalid_keys"]), (4, 8, 4))
        self.assertEqual((summary["jev_input_tokens"], summary["chat_input_tokens"]), (1200, 2000))
        self.assertEqual((summary["chat_cache_tokens"], summary["scored"]), (200, 4))
        self.assertEqual((summary["cost_usd"], summary["mean_elapsed_s"]), (0.004, 0.1))

    def test_the_median_latency_keeps_zero_millisecond_decisions(self) -> None:
        fast = _episode(0, 1.0, 3)
        fast.decisions = [
            {"step": 1, "key": "inc", "ms": 0, "source": "rule"},
            {"step": 2, "key": "inc", "ms": 0, "source": "rule"},
        ]
        slow = _episode(1, 1.0, 3)
        slow.decisions = [{"step": 1, "key": "inc", "ms": 9, "source": "rule"}]
        self.assertEqual(summarize([fast, slow])["median_decision_ms"], 0)

    def test_errored_episodes_are_counted_but_not_scored(self) -> None:
        failed = _episode(1, 0.0, 0)
        failed.error = "decision failed: decisions endpoint returned HTTP 401"
        failed.decisions = []
        episodes = [_episode(0, 1.0, 3), failed]
        summary = summarize(episodes)
        self.assertEqual((summary["errors"], summary["scored"], summary["episodes"]), (1, 1, 2))
        self.assertEqual(
            (summary["mean_score"], summary["wins"], summary["losses"], summary["mean_steps"]), (1.0, 1, 0, 3.0)
        )
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = write_job("counter", episodes, results_dir=Path(tmp))
            stats = json.loads((job_dir / "result.json").read_text())["stats"]
            self.assertEqual((stats["n_completed_trials"], stats["n_errored_trials"]), (1, 1))
            trials = sorted(p for p in job_dir.iterdir() if p.is_dir())
            first, second = (json.loads((trial / "result.json").read_text()) for trial in trials)
        self.assertIsNone(first["exception_info"])
        self.assertEqual(second["exception_info"], {"error": failed.error})

    def test_a_series_whose_every_episode_errored_summarizes_without_a_score(self) -> None:
        failed = _episode(0, 0.0, 0)
        failed.error = "decision failed"
        summary = summarize([failed])
        self.assertEqual((summary["scored"], summary["mean_score"], summary["mean_score_ci95"]), (0, None, None))

    def test_an_unknown_cost_in_one_episode_makes_the_total_unknown(self) -> None:
        priced, unpriced = _episode(0, 1.0, 3), _episode(1, 1.0, 3)
        unpriced.cost_usd = None
        self.assertIsNone(summarize([priced, unpriced])["cost_usd"])

    def test_job_folder_has_the_harbor_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = write_job("counter", [_episode(0, 1.0, 3), _episode(1, 0.0, 2)], results_dir=Path(tmp))
            self.assertTrue(job_dir.name.endswith("__fixed"))
            config = json.loads((job_dir / "config.json").read_text())
            self.assertEqual(config["datasets"], [{"name": "counter", "n_tasks": 2}])
            self.assertEqual(config["agents"], [{"name": "s1a-evals/fixed", "model_name": "fixed"}])
            self.assertEqual(json.loads((job_dir / "result.json").read_text())["n_total_trials"], 2)
            self.assertEqual(json.loads((job_dir / "summary.json").read_text())["mean_score"], 0.5)
            trials = sorted(p for p in job_dir.iterdir() if p.is_dir())
            self.assertEqual(len(trials), 2)
            first = json.loads((trials[0] / "result.json").read_text())
            self.assertEqual(first["task_name"], "counter/0")
            self.assertEqual(first["source"], "counter")
            self.assertEqual(first["verifier_result"]["rewards"]["reward"], 1.0)
            self.assertEqual(first["agent_info"]["name"], "s1a-evals/fixed")
            self.assertEqual(first["agent_result"]["n_input_tokens"], 800)
            self.assertEqual(first["agent_result"]["n_cache_tokens"], 50)
            self.assertEqual(first["agent_result"]["cost_usd"], 0.001)
            self.assertEqual(first["agent_result"]["metadata"]["chat_calls"], 2)
            self.assertTrue(first["agent_result"]["metadata"]["usage_known"])
            episode = json.loads((trials[0] / "agent" / "episode.json").read_text())
            self.assertEqual(episode["final_state"], {"n": 3})
            self.assertEqual(episode["extra"]["rethinks"][0]["kind"], "repeat")
            self.assertEqual(episode["views"][0]["state"], {"n": 0})
            self.assertFalse((trials[0] / "agent" / "frames").exists())

    def test_an_unknown_usage_episode_is_flagged_in_the_harbor_metadata(self) -> None:
        episode = _episode(0, 1.0, 3)
        episode.usage_known = False
        episode.cost_usd = None
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = write_job("counter", [episode], results_dir=Path(tmp))
            trial = next(p for p in job_dir.iterdir() if p.is_dir())
            result = json.loads((trial / "result.json").read_text())
        self.assertFalse(result["agent_result"]["metadata"]["usage_known"])
        self.assertIsNone(result["agent_result"]["cost_usd"])

    def test_job_folder_takes_the_episode_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames-2048-0"
            frames.mkdir()
            for name in ("000.png", "001.png"):
                (frames / name).write_bytes(b"png")
            episode = _episode(0, 1.0, 1)
            episode.frames_dir = frames
            job_dir = write_job("counter", [episode], results_dir=Path(tmp) / "showcase")
            trial = next(p for p in job_dir.iterdir() if p.is_dir())
            self.assertEqual(sorted(p.name for p in (trial / "agent" / "frames").iterdir()), ["000.png", "001.png"])
            self.assertFalse(frames.exists())
