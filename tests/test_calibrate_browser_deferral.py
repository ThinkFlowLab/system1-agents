from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.calibrate_browser_deferral import calibrate, read_confidences


class TestCalibration(unittest.TestCase):
    def test_ties_are_not_split(self) -> None:
        row = calibrate([0.1, 0.2, 0.2, 0.9], [0.5])[0]
        self.assertEqual(row["defer_below"], 0.2)
        self.assertEqual((row["deferred_steps"], row["actual_rate"]), (1, 0.25))

    def test_rejects_invalid_rate_and_empty_data(self) -> None:
        with self.assertRaises(ValueError):
            calibrate([], [0.1])
        with self.assertRaises(ValueError):
            calibrate([0.5], [1.1])

    def test_reads_complete_artifacts_and_rejects_artifacts_without_finite_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "decision_ticks.json"
            artifact.write_text(
                json.dumps({"ticks": [{"confidence": 0.2}, {"confidence": 0.8}, {"confidence": True}]}),
                encoding="utf-8",
            )
            self.assertEqual(read_confidences([artifact]), [0.2, 0.8])

            artifact.write_text(json.dumps({"ticks": [{"confidence": None}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no finite confidence"):
                read_confidences([artifact])
