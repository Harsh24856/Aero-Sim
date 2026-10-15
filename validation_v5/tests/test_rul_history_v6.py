"""v6: the cache carries the engine's logbook history and RUL reads it; data and
exports from before v6 behave exactly as they did.

    validation/venv/bin/python validation_v5/tests/test_rul_history_v6.py

The cache tests build small caches from two 120-flight samples (set AERO_V6_SAMPLE /
AERO_V5_SAMPLE to their engine directories, e.g. .../gen_v6/914); they skip without them.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)
sys.path.insert(0, os.path.join(os.path.dirname(V5), "backend"))

import features_v5 as F  # noqa: E402
import rul_v5  # noqa: E402
from pipeline_v5 import Cache, build_cache  # noqa: E402

V6_SAMPLE = os.environ.get("AERO_V6_SAMPLE", "")
V5_SAMPLE = os.environ.get("AERO_V5_SAMPLE", "")


def fake_outputs(n: int) -> dict:
    rng = np.random.default_rng(0)
    return {"health": rng.uniform(0, 1, (n, 1)), "severity": rng.uniform(0, 0.4, (n, 14)),
            "diagnosis": rng.uniform(0, 1, (n, 14)), "family": rng.uniform(0, 1, (n, 6))}


@unittest.skipUnless(os.path.isdir(V6_SAMPLE) and os.path.isdir(V5_SAMPLE), "set AERO_V6_SAMPLE / AERO_V5_SAMPLE")
class TestHistoryInCacheAndRul(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v6 = Cache(cls._build(V6_SAMPLE))
        cls.v5 = Cache(cls._build(V5_SAMPLE))

    @staticmethod
    def _build(src: str) -> str:
        dst = tempfile.mkdtemp()
        build_cache(src, dst)
        return dst

    def test_cache_contract_records_hist_cols(self):
        self.assertEqual(self.v6.contract["hist_cols"], F.HIST_COLS)
        E = np.asarray(self.v6.ends[:2000])
        self.assertTrue(((self.v6.labels(E, "hist_wear_ratio") >= 0.5) & (self.v6.labels(E, "hist_wear_ratio") <= 2.0)).all())

    def test_old_data_cache_has_no_hist_cols_and_neutral_values(self):
        self.assertEqual(self.v5.contract["hist_cols"], [])
        E = np.asarray(self.v5.ends[:2000])
        self.assertTrue((self.v5.labels(E, "hist_wear_ratio") == 1.0).all())

    def test_rul_inputs_unchanged_without_hist(self):
        seq, ctx, _ = self.v5.batch(np.arange(64))
        o = fake_outputs(64)
        legacy = rul_v5.rul_inputs(o, seq, ctx, self.v5.contract)
        self.assertEqual(legacy.shape[1], 1 + 14 + 1 + 6 + 1 + 1 + ctx.shape[1])
        np.testing.assert_array_equal(legacy, rul_v5.rul_inputs(o, seq, ctx, self.v5.contract, hist=np.ones((64, 3))))

    def test_rul_inputs_adds_three_hist_columns(self):
        seq, ctx, E = self.v6.batch(np.arange(64))
        o = fake_outputs(64)
        hist = np.stack([self.v6.labels(E, c) for c in F.HIST_COLS], 1)
        X = rul_v5.rul_inputs(o, seq, ctx, self.v6.contract, hist=hist)
        base = rul_v5.rul_inputs(o, seq, ctx, self.v5.contract)
        self.assertEqual(X.shape[1], base.shape[1] + 4)
        lag, ratio, sev = hist[:, 0], hist[:, 1], hist[:, 2]
        np.testing.assert_allclose(X[:, -4], ratio)
        np.testing.assert_allclose(X[:, -3], sev)
        np.testing.assert_allclose(X[:, -2], (o["severity"].max(1) - sev) / np.maximum(lag, 1.0))
        then = np.where(sev <= 0.0, 0.04, sev)
        np.testing.assert_allclose(X[:, -1], rul_v5.two_point_remaining(o["severity"].max(1) / 0.775, then / 0.775,
                                                                         lag, self.v6.contract["tbo_hours"]))


class TestTwoPointRemaining(unittest.TestCase):
    """A fault's remaining life from two points on its growth curve (the logbook and
    now). The v6 RUL regressor under-learnt this from data where the logbook showed no
    fault: test windows read +155 h long there, -51 h elsewhere."""
    TBO = 2000.0

    @staticmethod
    def _sev(age_h, a, b, tbo=2000.0):
        from degradation_v4 import DegradationState
        return DegradationState._progress(age_h / tbo * 100.0, a, b)

    def _true_remaining(self, age_now, a, b):
        lo, hi = age_now, 50 * self.TBO
        for _ in range(80):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if self._sev(m, a, b) < 0.8 else (lo, m)
        return hi - age_now

    def test_recovers_remaining_life_of_typical_faults(self):
        from degradation_v4 import A_RANGE, B_RANGE
        rng = np.random.default_rng(1)
        errs = []
        for _ in range(300):
            a = rng.uniform(*A_RANGE) * rng.uniform(2.0, 9.0)
            b = rng.uniform(*B_RANGE)
            age = rng.uniform(150.0, 900.0); lag = rng.uniform(20.0, 200.0)
            s1, s0 = self._sev(age, a, b), self._sev(age - lag, a, b)
            if not (0.15 <= s1 <= 0.7 and s0 >= 0.08):          # a visible past point
                continue
            est = rul_v5.two_point_remaining(np.array([s1]), np.array([s0]), np.array([lag]), self.TBO)[0] * self.TBO
            errs.append(abs(est - self._true_remaining(age, a, b)) / self._true_remaining(age, a, b))
        self.assertGreater(len(errs), 50)
        self.assertLess(np.median(errs), 0.15)                 # b varies 1.3-1.7; the estimate assumes 1.5

    def test_a_fault_absent_from_the_logbook_reads_as_fast(self):
        # Invisible (progress ~0.05) 134 h ago, 0.44 now (0.34 effective): test-set labels average 170 h.
        fast = rul_v5.two_point_remaining(np.array([0.44]), np.array([0.05]), np.array([134.0]), self.TBO)[0] * self.TBO
        slow = rul_v5.two_point_remaining(np.array([0.44]), np.array([0.32]), np.array([134.0]), self.TBO)[0] * self.TBO
        self.assertLess(fast, slow)
        self.assertLess(fast, 400.0)

    def test_no_growth_or_no_logbook_is_capped_not_infinite(self):
        out = rul_v5.two_point_remaining(np.array([0.3, 0.3, 0.02]), np.array([0.3, 0.05, 0.0]),
                                         np.array([100.0, 0.0, 100.0]), self.TBO)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, rul_v5.TWO_POINT_CAP)


class TestRulScoredAsServed(unittest.TestCase):
    """RUL candidates are chosen on what is served: predictions smoothed within each
    flight (914 v6: raw scoring picked a model whose served TBO-limited error was
    4.31% on validation, over the 4% gate)."""

    def _flights(self):
        rng = np.random.default_rng(0)
        fl = np.repeat([3, 1, 2], 40)
        hrs = np.concatenate([np.sort(rng.uniform(100, 120, 40)) for _ in range(3)])
        order = rng.permutation(len(fl))                      # windows arrive in any order
        raw = rng.uniform(200, 900, len(fl))
        return raw[order], fl[order], hrs[order]

    def test_smooth_by_flight_matches_per_flight_loop(self):
        raw, fl, hrs = self._flights()
        want = raw.copy()
        for f in np.unique(fl):
            g = np.where(fl == f)[0]
            g = g[np.argsort(hrs[g], kind="stable")]
            want[g] = rul_v5.smooth_within_flight(raw[g], hrs[g])
        np.testing.assert_array_equal(rul_v5.smooth_by_flight(raw, fl, hrs), want)

    def test_served_report_scores_smoothed_predictions(self):
        import evaluate as E
        raw, fl, hrs = self._flights()
        tbo, cal = 2000.0, np.full(len(raw), 900.0)
        true_h, wl = np.full(len(raw), 600.0), np.arange(len(raw)) % 2 == 0
        got = rul_v5.served_report(raw / tbo, true_h / tbo, cal, tbo, wl, fl, hrs)
        want = E.rul_report(true_h, rul_v5.smooth_by_flight(np.minimum(raw, cal), fl, hrs), cal, tbo, wl)
        self.assertEqual(got["tbo_limited_mae_pct"], want["tbo_limited_mae_pct"])
        self.assertNotEqual(got["tbo_limited_mae_pct"],
                            E.rul_report(true_h, np.minimum(raw, cal), cal, tbo, wl)["tbo_limited_mae_pct"])


if __name__ == "__main__":
    unittest.main()
