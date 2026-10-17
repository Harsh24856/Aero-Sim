"""The gap report's arithmetic: where the severity bias comes from, and how often one
oil fault is named as the other. Pure numpy - runs on CI's CPU runner.

    validation/venv/bin/python validation_v5/tests/test_gaps_v5.py
"""
import os
import sys
import unittest

import numpy as np

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)
sys.path.insert(0, os.path.join(os.path.dirname(V5), "backend"))

import gaps_v5  # noqa: E402


class TestBiasSplit(unittest.TestCase):
    def test_bias_split_attributes_bias_to_gated_out_cells(self):
        t = np.array([[0.4], [0.4], [0.0]])
        ung = np.array([[0.38], [0.30], [0.05]])
        gat = np.array([[0.38], [0.00], [0.00]])
        r = gaps_v5.bias_split(t, ung, gat)
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["gated_out_n"], 1)
        self.assertAlmostEqual(r["bias"], (-0.02 - 0.4) / 2)
        self.assertAlmostEqual(r["gated_out_bias"], -0.4)
        self.assertAlmostEqual(r["share_from_gated_out"], 0.4 / 0.42, places=6)

    def test_bias_split_zero_bias_has_zero_share(self):
        t = np.array([[0.4]])
        p = np.array([[0.4]])
        self.assertEqual(gaps_v5.bias_split(t, p, p)["share_from_gated_out"], 0.0)


class TestConfusionShare(unittest.TestCase):
    def test_confusion_share_counts_only_misnamed_windows(self):
        y = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])
        pred = np.array([[0, 1], [1, 0], [0, 1], [0, 1]])
        self.assertAlmostEqual(gaps_v5.confusion_share(y, pred, 0, 1), 2 / 3)


if __name__ == "__main__":
    unittest.main()
