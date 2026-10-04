"""Diagnosis cut-offs are robust, not tuned to a sliver of validation F1.

The 914 v6 run chose cut-offs at the exact validation-F1 maximum: valve leakage 0.87
and air filter 0.20, where 0.74 and 0.11 lose under 0.01 F1. Probabilities a little
lower on other flights (test: valve recall 0.49; a preset air-filter flight named on
34% of samples at median probability 0.15) then miss the fault. The rule: the lowest
cut-off within CUT_F1_TOL of the best F1 among those that keep the recall floor.

    validation/venv/bin/python validation_v5/tests/test_cutoffs_v6.py
"""
import os
import sys
import unittest

import numpy as np

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)
sys.path.insert(0, os.path.join(os.path.dirname(V5), "backend"))

from train_v5 import CUT_F1_TOL, choose_cutoff  # noqa: E402

GRID = np.linspace(0.01, 0.99, 99)


class TestChooseCutoff(unittest.TestCase):
    def test_plateau_takes_the_lowest_cut_within_tolerance(self):
        # positives score 0.95, negatives 0.10, plus one negative at 0.80: every cut in
        # (0.10, 0.80] has F1 0.995, cuts above 0.80 reach 1.0 - a sliver, high up
        y = np.r_[np.ones(100), np.zeros(1000)]
        p = np.r_[np.full(100, 0.95), np.full(999, 0.10), [0.80]]
        c = choose_cutoff(y, p, GRID, min_recall=0.55)
        self.assertLess(c, 0.80)
        self.assertGreater(c, 0.10)

    def test_recall_floor_is_kept(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.ones(200), np.zeros(2000)]
        p = np.r_[rng.uniform(0.2, 1.0, 200), rng.uniform(0.0, 0.6, 2000)]
        c = choose_cutoff(y, p, GRID, min_recall=0.9)
        self.assertGreaterEqual(((p >= c) & (y == 1)).sum() / 200, 0.9)

    def test_tolerance_is_one_hundredth_of_f1(self):
        self.assertEqual(CUT_F1_TOL, 0.01)


if __name__ == "__main__":
    unittest.main()
