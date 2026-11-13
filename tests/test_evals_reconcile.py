# coding: utf-8
"""Attempt-manifest coverage is additive to the existing score table."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from evals.reconcile import PlannedAttempt, load_plan, markdown, read_artifacts, reconcile
from evals.table import read_results, rows as score_rows


def _result(
    root: Path, eval_name: str, model: str, task_name: str, *, errored: bool = False, job_name: str = "job-1"
) -> Path:
    folder = root / eval_name / job_name / task_name.rsplit("/", 1)[-1]
    folder.mkdir(parents=True)
    result = {
        "task_name": task_name,
        "agent_info": {"name": f"s1a-evals/{model}"},
        "exception_info": {"error": "fixture error"} if errored else None,
        "agent_result": {"metadata": {"steps": 2}, "cost_usd": 0.0},
        "verifier_result": {"rewards": {"reward": 0}},
    }
    path = folder / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


class TestReconcile(TestCase):
    def _plan(self, path: Path) -> list:
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "attempts": [
                        {
                            "id": "done",
                            "eval": "game2048",
                            "model": "jev",
                            "task_name": "game2048/0",
                            "status": "planned",
                        },
                        {
                            "id": "failed",
                            "eval": "game2048",
                            "model": "jev",
                            "task_name": "game2048/1",
                            "status": "planned",
                        },
                        {
                            "id": "missing",
                            "eval": "game2048",
                            "model": "jev",
                            "task_name": "game2048/2",
                            "status": "planned",
                        },
                        {
                            "id": "stopped",
                            "eval": "game2048",
                            "model": "jev",
                            "task_name": "game2048/3",
                            "status": "interrupted",
                            "reason": "worker stopped",
                        },
                        {
                            "id": "unsupported",
                            "eval": "desktop",
                            "model": "cua",
                            "task_name": "desktop/0",
                            "status": "unsupported",
                            "reason": "no display",
                        },
                        {
                            "id": "contradiction",
                            "eval": "desktop",
                            "model": "cua",
                            "task_name": "desktop/1",
                            "status": "unsupported",
                            "reason": "no display",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return load_plan(path)

    def test_reports_every_attempt_and_unplanned_or_contradictory_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            root.mkdir()
            _result(root, "game2048", "jev", "game2048/0")
            _result(root, "game2048", "jev", "game2048/1", errored=True)
            _result(root, "desktop", "cua", "desktop/1")
            _result(root, "blackjack", "jev", "blackjack/99")
            attempts = self._plan(Path(tmp) / "plan.json")
            result = reconcile(attempts, read_artifacts(root))

        self.assertEqual(
            [(row.attempt_id, row.status) for row in result],
            [
                ("done", "recorded"),
                ("failed", "error"),
                ("missing", "missing"),
                ("stopped", "interrupted"),
                ("unsupported", "unsupported"),
                ("contradiction", "unexpected-result"),
                ("", "unplanned-result"),
            ],
        )
        report = markdown(result)
        self.assertIn(
            "Planned attempts: 6; recorded outcomes: 3; missing: 1; interrupted: 1; unsupported: 1; ambiguous: 0; errors: 1; unexpected results: 2.",
            report,
        )
        self.assertIn("game2048/job-1/1/result.json", report)

    def test_reconciliation_does_not_mutate_existing_score_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            root.mkdir()
            _result(root, "game2048", "jev", "game2048/0")
            before = score_rows(read_results(root))
            attempts = self._plan(Path(tmp) / "plan.json")
            reconcile(attempts, read_artifacts(root))
            after = score_rows(read_results(root))
        self.assertEqual(before, after)
        self.assertEqual((before[0]["attempts"], before[0]["score_n"], before[0]["mean_score"]), (1, 1, 0.0))

    def test_multiple_artifacts_for_one_key_are_order_invariant_and_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            root.mkdir()
            _result(root, "game2048", "jev", "game2048/0", job_name="job-a")
            _result(root, "game2048", "jev", "game2048/0", errored=True, job_name="job-b")
            artifacts = read_artifacts(root)
            attempt = PlannedAttempt("repeat", "game2048", "jev", "game2048/0", "planned")
            forward = reconcile([attempt], artifacts)
            reverse = reconcile([attempt], list(reversed(artifacts)))

        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward), 1)
        self.assertEqual(forward[0].status, "ambiguous")
        self.assertEqual(forward[0].reason, "2 result artifacts match this key; no result was assigned.")
        self.assertEqual(
            forward[0].artifact,
            "game2048/job-a/0/result.json; game2048/job-b/0/result.json",
        )
        self.assertIn("ambiguous: 1", markdown(forward))
        self.assertIn("recorded outcomes: 0", markdown(forward))

    def test_manifest_rejects_duplicate_match_keys_and_terminal_state_without_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "attempts": [
                            {"id": "a", "eval": "e", "model": "m", "task_name": "t", "status": "planned"},
                            {"id": "b", "eval": "e", "model": "m", "task_name": "t", "status": "planned"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate result match key"):
                load_plan(path)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "attempts": [
                            {"id": "a", "eval": "e", "model": "m", "task_name": "t", "status": "interrupted"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires a non-empty reason"):
                load_plan(path)
