"""Checks for specialists_v5 (GPU needed: train_v5 refuses to import without it).

    validation/venv/bin/python validation_v5/tests/test_specialists_v5.py [cache_dir]
"""
import os
import sys
import unittest

import numpy as np

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)
CACHE = sys.argv.pop(1) if len(sys.argv) > 1 else os.path.join(V5, "cache", "914")

import specialists_v5 as S  # noqa: E402
from train_v5 import WarmUp  # noqa: E402


class TestSpecialists(unittest.TestCase):
    def test_cosine_schedule(self):
        s = WarmUp(1e-3, 100, 1000)
        self.assertAlmostEqual(float(s(0)), 1e-5, places=7)           # warm-up starts low
        self.assertAlmostEqual(float(s(99)), 1e-3, places=7)          # peak after warm-up
        self.assertAlmostEqual(float(s(999)), 5e-5, places=7)         # decays to 5%
        self.assertAlmostEqual(float(WarmUp(1e-3, 100)(5000)), 1e-3)  # no decay: flat, as before

    def test_sensor_bias_cuts_false_alarms(self):
        rng = np.random.default_rng(0)
        y = (rng.random((4000, 12)) < 0.1) * rng.integers(1, 7, (4000, 12))
        logits = rng.normal(0, 1, (4000, 12, 7))
        logits[np.arange(4000)[:, None], np.arange(12)[None, :], y] += 1.5    # weakly informative
        b, rep = S.tune_sensor_bias(logits, y)
        z = logits.copy()
        fa0 = (z.argmax(-1)[y == 0] != 0).mean()
        self.assertGreater(b, 0.0)
        self.assertLess(rep["false_alarm_rate"], fa0)

    def test_severity_gate(self):
        sev = np.array([[0.5, 0.4], [0.3, 0.2]])
        diag = np.array([[0.9, 0.1], [0.2, 0.8]])
        out = S.gate_severity(sev, diag, {"temperature": [1.0, 1.0], "cut": [0.5, 0.5]})
        np.testing.assert_allclose(out, [[0.5, 0.0], [0.0, 0.2]])

    def test_severity_calibration_removes_bias(self):
        from sklearn.isotonic import IsotonicRegression
        p = np.linspace(0.1, 0.6, 200)
        t = p + 0.1                                     # the specialist under-predicts by 0.1
        iso = {1: IsotonicRegression(out_of_bounds="clip").fit(p, t)}
        sev = np.stack([p, p], 1)
        out = S.calibrate_severity(sev, iso)
        np.testing.assert_allclose(out[:, 0], p)        # a column without a model passes through
        self.assertLess(abs((out[:, 1] - t).mean()), 1e-6)

    def test_gated_rul_returns_calendar_marker_below_threshold(self):
        import rul_v5 as R

        class Reg:
            def predict(self, X):
                return np.full(len(X), 0.3)
        g = R.GatedRUL(Reg(), None, 0.5)
        np.testing.assert_allclose(g.predict(np.zeros((3, 2)), np.array([0.9, 0.4, 0.5])), [0.3, 1e3, 0.3])

    @unittest.skipUnless(os.path.exists(os.path.join(CACHE, "contract_v5.json")), "no cache")
    def test_cal_halves_disjoint_by_flight(self):
        from pipeline_v5 import Cache
        c = Cache(CACHE)
        a, b = S.split_cal(c)
        fa, fb = set(c.ends[np.sort(a), 0].astype(int)), set(c.ends[np.sort(b), 0].astype(int))
        self.assertTrue(fa and fb and not fa & fb)
        self.assertEqual(len(a) + len(b), len(c.end_ids(["cal"], jitter=False)))


if __name__ == "__main__":
    unittest.main()
