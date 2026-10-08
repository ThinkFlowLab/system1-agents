# coding: utf-8
"""The results table: job folders from write_job, one row per eval and model."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest import TestCase

from s1a.jobs import Episode, write_job
from evals.table import markdown, read_results, rows, model_label


def _episode(seed: int, score: float, cost: float | None) -> Episode:
    stamp = datetime.now().isoformat()
    return Episode(
        env="blackjack",
        policy="jev",
        seed=seed,
        score=score,
        steps=2,
        elapsed_s=1.5,
        started_at=stamp,
        finished_at=stamp,
        final_state={},
        chat_calls=0,
        chat_input_tokens=0,
        chat_output_tokens=0,
        chat_cache_tokens=0,
        jev_input_tokens=600,
        invalid_keys=0,
        cost_usd=cost,
        decisions=[
            {"step": 1, "key": "hit", "ms": 400, "source": "jev"},
            {"step": 2, "key": "stand", "ms": 380, "source": "jev"},
        ],
    )


class TestTable(TestCase):
    def test_model_label_drops_any_suite_prefix(self) -> None:
        self.assertEqual(model_label({"agent_info": {"name": "s1a-evals/jev"}}), "jev")
        self.assertEqual(model_label({"agent_info": {"name": "jiuwen-jev-evals/llm"}}), "llm")
        self.assertEqual(model_label({"agent_info": {"name": "basic"}}), "basic")

    def test_rows_merge_job_folders_per_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_job("blackjack", [_episode(0, 1.0, 0.0001), _episode(1, -1.0, 0.0001)], results_dir=root)
            write_job("blackjack", [_episode(2, 1.0, 0.0001)], results_dir=root)

            table = rows(read_results(root))

        by_key = {(row["eval"], row["model"]): row for row in table}
        blackjack = by_key[("blackjack", "jev")]
        self.assertEqual((blackjack["N"], blackjack["mean_score"], blackjack["median_s"]), (3, 0.333, 1.5))
        self.assertEqual(
            (blackjack["mean_steps"], blackjack["mean_decisions"], blackjack["mean_cost_usd"]), (2, 2, 0.0001)
        )
        text = markdown(table)
        self.assertIn("| blackjack | jev | 3 | 0 |", text)

    def test_errored_trials_are_counted_and_not_scored(self) -> None:
        failed = _episode(1, 0.0, 0.0)
        failed.error = "decision failed: HTTP 401"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_job("blackjack", [_episode(0, 1.0, 0.0001), failed], results_dir=root)
            write_job("blackjack", [failed], results_dir=root)
            (row,) = rows(read_results(root))
        self.assertEqual((row["N"], row["errors"], row["mean_score"], row["mean_cost_usd"]), (1, 2, 1.0, 0.0001))
        self.assertIn("| blackjack | jev | 1 | 2 | 1.0 [1.0, 1.0] |", markdown([row]))

    def test_a_model_whose_every_trial_errored_has_no_score(self) -> None:
        failed = _episode(0, 0.0, None)
        failed.error = "decision failed: HTTP 401"
        with tempfile.TemporaryDirectory() as tmp:
            write_job("blackjack", [failed], results_dir=Path(tmp))
            (row,) = rows(read_results(Path(tmp)))
        self.assertEqual((row["N"], row["errors"], row["mean_score"]), (0, 1, None))
        self.assertIn("| blackjack | jev | 0 | 1 | n/a | n/a | n/a | n/a | n/a | n/a |", markdown([row]))

    def test_rows_are_keyed_by_the_results_folder_not_the_recorded_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_job("blackjack", [_episode(0, 1.0, 0.0001)], results_dir=root)
            write_job("blackjack_before_fixes", [_episode(0, 0.0, 0.0001)], results_dir=root)
            table = rows(read_results(root))
        self.assertEqual(
            [(row["eval"], row["N"], row["mean_score"]) for row in table],
            [("blackjack", 1, 1.0), ("blackjack_before_fixes", 1, 0.0)],
        )

    def test_an_unknown_cost_makes_the_model_cost_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_job("blackjack", [_episode(0, 1.0, 0.0001), _episode(1, 1.0, None)], results_dir=root)
            (row,) = rows(read_results(root))
        self.assertIsNone(row["mean_cost_usd"])
        self.assertIn("n/a", markdown([row]))
