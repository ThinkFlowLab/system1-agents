from __future__ import annotations

import unittest

from scripts.calibrate_browser_deferral import calibrate


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
