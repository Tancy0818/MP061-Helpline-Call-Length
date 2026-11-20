"""Synthetic boundary tests; no assignment CSVs are required."""

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from helpline.data_checks import normalize_language, stuck_periods, target_outliers


class DataCheckTests(unittest.TestCase):
    def test_mixed_boundary_day_is_not_part_of_stuck_period(self):
        frame = pd.DataFrame({
            "date": ["2026-03-01", "2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"],
            "value": [3, 2, 2, 2, 4],
        })
        runs = stuck_periods(frame, "value", minimum_days=2)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["start_date"], "2026-03-02")
        self.assertEqual(runs[0]["end_date"], "2026-03-03")

    def test_missing_calendar_day_breaks_stuck_period(self):
        frame = pd.DataFrame({"date": ["2026-03-01", "2026-03-03"], "value": [2, 2]})
        self.assertEqual(stuck_periods(frame, "value", minimum_days=2), [])

    def test_upper_gap_finds_cluster_without_low_value_false_positive(self):
        values = pd.Series([0.1, 1, 10, 11, 12, 13, 14, 15, 100, 120])
        mask, gap = target_outliers(values)
        self.assertEqual(values[mask].tolist(), [100, 120])
        self.assertEqual(gap["lower"], 15)
        mask, gap = target_outliers(pd.Series([10, 11, 12, 13, 14]))
        self.assertFalse(mask.any())
        self.assertIsNone(gap)

    def test_language_normalization_does_not_mutate_input(self):
        values = pd.Series([" Hindi ", "hin.", "ENGLISH"])
        original = values.copy()
        actual = normalize_language(values, {"hin.": "hindi"})
        self.assertEqual(actual.tolist(), ["hindi", "hindi", "english"])
        pd.testing.assert_series_equal(values, original)


if __name__ == "__main__":
    unittest.main()
