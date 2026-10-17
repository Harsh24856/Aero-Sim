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
        self.assertEqual(X.shape[1], base.shape[1] + 3)
        lag, ratio, sev = hist[:, 0], hist[:, 1], hist[:, 2]
        np.testing.assert_allclose(X[:, -3], ratio)
        np.testing.assert_allclose(X[:, -2], sev)
        np.testing.assert_allclose(X[:, -1], (o["severity"].max(1) - sev) / np.maximum(lag, 1.0))


class TestLogbookTrend(unittest.TestCase):
    """W2: remaining life from the fault's MEASURED growth across the inspection logbook
    (25 / 50 / 100 / 200 h) and now - least squares on the growth curve's linear form."""
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

    def _logbook(self, age, a, b, depth, noise=None):
        """Effective severities at the inspections, floored (and noisy) as the logbook is."""
        out = []
        for lag in F.LOG_LAGS_H:
            s = self._sev(age - lag, a, b) * depth if age > lag else 0.0
            s = 0.0 if s < 0.08 else (s if noise is None else min(max(s * (1.0 + noise.normal(0, 0.10)), 0.0), 1.0))
            out.append(s)
        return np.array([out])

    def test_unbiased_on_training_distribution_faults_with_a_noisy_logbook(self):
        # A model INPUT: it must be informative and unbiased. Each fault's depth (0.55-1)
        # and exponent (1.3-1.7) are unknown, and the logbook has 10% noise: median
        # |error| ~32%, almost all from depth (the exact-curve test below pins the maths).
        from degradation_v4 import A_RANGE, B_RANGE
        rng = np.random.default_rng(1)
        signed = []
        for _ in range(600):
            a = rng.uniform(*A_RANGE) * rng.uniform(2.0, 9.0); b = rng.uniform(*B_RANGE)
            depth = rng.uniform(0.55, 1.0); age = rng.uniform(100.0, 1200.0)
            now = self._sev(age, a, b) * depth
            log = self._logbook(age, a, b, depth, noise=rng)
            if not (0.15 <= now / depth <= 0.75) or (log > 0).sum() < 1:
                continue
            est = rul_v5.logbook_trend(np.array([now]), log, self.TBO)[0] * self.TBO
            true = self._true_remaining(age, a, b)
            signed.append((est - true) / true)
        self.assertGreater(len(signed), 100)
        self.assertLess(abs(np.median(signed)), 0.10)
        self.assertLess(np.median(np.abs(signed)), 0.50)     # measured 32-40%; the units bug gave 365%

    def test_the_failed_regime_now_reads_fast(self):
        # The v6 failure: invisible 134 h ago, 0.34 now (test labels average 170 h). The
        # 25 h and 50 h inspections see it, so its growth is measured.
        age, a, b, depth = 200.0, 0.0022 * 5.5, 1.5, 0.775
        now = self._sev(age, a, b) * depth
        est = rul_v5.logbook_trend(np.array([now]), self._logbook(age, a, b, depth), self.TBO)[0] * self.TBO
        self.assertLess(abs(est - self._true_remaining(age, a, b)), 0.25 * self._true_remaining(age, a, b))

    def test_inspections_are_dated_from_the_flight_start(self):
        # One flight ages the engine up to ~225 h (x180 life clock). The logbook is taken
        # at the start, so 60 h into the flight its "25 h" inspection is 85 h old.
        age0, a, b, depth, elapsed = 150.0, 0.0022 * 5.5, 1.5, 0.775, 60.0
        log = self._logbook(age0, a, b, depth)                      # inspections before the flight
        now = self._sev(age0 + elapsed, a, b) * depth
        est = rul_v5.logbook_trend(np.array([now]), log, self.TBO, elapsed_h=np.array([elapsed]))[0] * self.TBO
        true = self._true_remaining(age0 + elapsed, a, b)
        self.assertLess(abs(est - true), 0.05 * true)
        stale = rul_v5.logbook_trend(np.array([now]), log, self.TBO)[0] * self.TBO
        self.assertGreater(abs(stale - true), 3 * abs(est - true))    # ignoring it is what went wrong

    def test_a_fault_newer_than_the_logbook_is_bounded(self):
        # Invisible at every inspection, 0.36 now, 40 h into the flight: it grew from under
        # the floor within 65 h. A bound, not a measurement - a separate input.
        b = rul_v5.logbook_bound(np.array([0.36, 0.36, 0.02]), np.zeros((3, 4)), self.TBO,
                                 elapsed_h=np.array([40.0, 400.0, 40.0]))
        self.assertLess(b[0], b[1])                      # less time to grow -> faster -> less life
        self.assertLess(b[0] * self.TBO, 400.0)
        self.assertEqual(b[2], rul_v5.TREND_CAP)         # no fault now: nothing to bound
        seen = rul_v5.logbook_bound(np.array([0.36]), np.array([[0.2, 0, 0, 0]]), self.TBO)
        self.assertEqual(seen[0], rul_v5.TREND_CAP)      # the logbook saw it: logbook_trend measures it

    def test_nothing_measured_is_capped(self):
        out = rul_v5.logbook_trend(np.array([0.3, 0.02, 0.3]),
                                   np.array([[0, 0, 0, 0], [0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3]], float), 2000.0)
        np.testing.assert_allclose(out, rul_v5.TREND_CAP)    # no visible past / no fault now / no growth


class TestTrendGatedRUL(unittest.TestCase):
    """Where the trained gate judges the engine fault-limited AND the logbook measured
    the fault's growth, the measured time to removal answers (weight 1) or is blended."""

    class Gated:
        threshold = 0.5
        class clf:
            @staticmethod
            def predict_proba(X):
                return np.stack([1 - X[:, 0], X[:, 0]], 1)
        def predict(self, X, p_wear_limited=None):
            return np.full(len(X), 0.5)

    def _X(self, p_wl, trend):
        X = np.zeros((1, 40)); X[0, 0], X[0, -3] = p_wl, trend
        return X

    def test_uses_the_trend_only_when_fault_limited_and_measured(self):
        m = rul_v5.TrendGatedRUL(self.Gated(), weight=1.0)
        self.assertAlmostEqual(m.predict(self._X(0.9, 0.14))[0], 0.14)
        self.assertAlmostEqual(m.predict(self._X(0.1, 0.14))[0], 0.5)                  # gate: not fault-limited
        self.assertAlmostEqual(m.predict(self._X(0.9, rul_v5.TREND_CAP))[0], 0.5)      # not measured

    def test_blend(self):
        self.assertAlmostEqual(rul_v5.TrendGatedRUL(self.Gated(), weight=0.5).predict(self._X(0.9, 0.1))[0], 0.3)


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
