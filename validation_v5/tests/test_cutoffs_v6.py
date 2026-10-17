"""Diagnosis cut-offs are robust, not tuned to a sliver of validation F1.

The 914 v6 run chose cut-offs at the exact validation-F1 maximum: valve leakage 0.87
and air filter 0.20, where 0.74 and 0.11 lose under 0.01 F1. Probabilities a little
lower on other flights (test: valve recall 0.49; a preset air-filter flight named on
34% of samples at median probability 0.15) then miss the fault. An ABSOLUTE 0.01 alone
was too loose for near-perfect faults (combustion / bearing cut-offs fell to 0.03 / 0.06,
-0.05 / -0.035 test F1); a RELATIVE 20%-more-error alone was too loose for weak ones
(oil degradation 0.39 -> 0.15, ignition 0.40 -> 0.20). The rule takes the stricter of
the two: the lowest cut-off within CUT_F1_TOL of the best F1 AND within CUT_ERR_TOL more
error, among those that keep the recall floor.

    validation/venv/bin/python validation_v5/tests/test_cutoffs_v6.py
"""
import os
import sys
import unittest

import numpy as np

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)
sys.path.insert(0, os.path.join(os.path.dirname(V5), "backend"))

from train_v5 import CUT_ERR_TOL, CUT_F1_TOL, choose_cutoff  # noqa: E402

GRID = np.linspace(0.01, 0.99, 99)


class TestChooseCutoff(unittest.TestCase):
    def test_imperfect_fault_moves_lower_on_a_plateau(self):
        # valve-like: 10% of positives never caught; 2 negatives at 0.80. The best F1
        # needs a cut above 0.80 (0.947); every cut in (0.10, 0.80] is 0.9375 - 18%
        # more errors, inside the tolerance - so the lowest of those is taken.
        y = np.r_[np.ones(100), np.zeros(1000)]
        p = np.r_[np.full(90, 0.95), np.full(10, 0.05), np.full(998, 0.10), [0.80, 0.80]]
        c = choose_cutoff(y, p, GRID, min_recall=0.55)
        self.assertLess(c, 0.80)
        self.assertGreater(c, 0.10)

    def test_near_perfect_fault_stays_above_the_noise(self):
        # combustion-like: perfect above 0.05, one stray negative AT 0.05. Every cut in
        # (0.05, 0.95] is perfect; below it F1 is 0.995 - within an absolute 0.01, but
        # infinitely more errors than the best, so the cut-off stays above the stray.
        y = np.r_[np.ones(100), np.zeros(1000)]
        p = np.r_[np.full(100, 0.95), np.full(999, 0.01), [0.05]]
        c = choose_cutoff(y, p, GRID, min_recall=0.55)
        self.assertGreater(c, 0.05)

    def test_weak_fault_does_not_slide_far(self):
        # oil-degradation-like: best F1 ~0.64. 20% more error would accept F1 ~0.57 -
        # a far lower cut-off (914 v6: 0.39 -> 0.15, test F1 -0.033). The absolute
        # 0.01-F1 bound holds it: the chosen cut's F1 is within 0.01 of the best.
        import evaluate as E
        rng = np.random.default_rng(1)
        y = np.r_[np.ones(300), np.zeros(3000)]
        p = np.r_[rng.beta(4, 3, 300), rng.beta(1.2, 6, 3000)]
        c = choose_cutoff(y, p, GRID, min_recall=0.55)
        f1 = lambda cut: E.prf(y, p >= cut)[2]  # noqa: E731
        best = max(f1(g) for g in GRID if E.prf(y, p >= g)[1] >= 0.55)
        self.assertGreaterEqual(f1(c), best - 0.01 - 1e-12)

    def test_recall_floor_is_kept(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.ones(200), np.zeros(2000)]
        p = np.r_[rng.uniform(0.2, 1.0, 200), rng.uniform(0.0, 0.6, 2000)]
        c = choose_cutoff(y, p, GRID, min_recall=0.9)
        self.assertGreaterEqual(((p >= c) & (y == 1)).sum() / 200, 0.9)

    def test_tolerances(self):
        self.assertEqual((CUT_F1_TOL, CUT_ERR_TOL), (0.01, 0.20))


if __name__ == "__main__":
    unittest.main()
