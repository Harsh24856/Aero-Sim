"""generate_dataset_v5: the P2 throttle-step missions (plan 2026-10-18-v5-real-world-fixes).

Live, a pilot moves the throttle in one step; the training missions only ever eased it,
and the detector false-alarmed for up to 125 s after every large move. The extra
flights fly two new missions with instant steps, numbered and seeded apart from the
existing 6,000 so those stay byte-identical.

    backend/.venv/bin/python -m unittest tests.test_generate_v5_steps
"""
import hashlib
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_dataset_v5 as G  # noqa: E402

THR = G.ALL_COLS.index("throttle")
ALT = G.ALL_COLS.index("altitude")


def flight(scen_id, seed, mission=None):
    rows, meta = G.run_scenario("Rotax_916_iS", scen_id, np.random.default_rng(seed), mission=mission)
    return np.asarray(rows, np.float64), meta


class TestOldFlightsUnchanged(unittest.TestCase):
    def test_existing_missions_are_byte_identical(self):
        # Pins the existing flights so adding missions cannot move them. Rounded to 3
        # decimals so a different libm (CI runs Linux) cannot flip the last bit of a long
        # integration. Re-pinned for the P3 physics (mixture, governor, thermostat), which
        # changes every flight on purpose.
        for sid, want in ((0, "c638da6316ea8c23"), (1, "8f41fc655fc9c8ab")):
            rows, _ = G.run_scenario("Rotax_916_iS", sid, np.random.default_rng(123))
            got = hashlib.sha256(np.round(np.asarray(rows, np.float64), 3).tobytes()).hexdigest()[:16]
            self.assertEqual(got, want, sid)


class TestStepMissions(unittest.TestCase):
    def test_throttle_steps_moves_the_throttle_at_once(self):
        rows, meta = flight(100000, 5, "throttle_steps")
        self.assertEqual(meta["mission"], "throttle_steps")
        steps = np.abs(np.diff(rows[:, THR])) >= 0.38      # one second, less the 0.007 noise
        self.assertGreaterEqual(int(steps.sum()), 3)

    def test_idle_descent_starts_high_and_idles(self):
        rows, meta = flight(100001, 7, "idle_descent")
        self.assertEqual(meta["mission"], "idle_descent")
        n = len(rows)
        self.assertGreaterEqual(rows[: n // 4, ALT].max(), 4000.0)
        self.assertLessEqual(rows[n // 3: 2 * n // 3, THR].min(), 0.2)
        self.assertGreaterEqual(rows[-30:, THR].mean(), 0.9)


class TestExtraFlights(unittest.TestCase):
    def test_start_id_and_missions_restrict_the_extra_set(self):
        with tempfile.TemporaryDirectory() as d:
            m = G.generate_engine("Rotax_916_iS", G.engine_seed("Rotax_916_iS") + 100000, 2, out_dir=d,
                                  start_id=100000, missions=["throttle_steps", "idle_descent"])
            idx = pd.read_parquet(os.path.join(d, "916", "index.parquet"))
        self.assertEqual(m["flights"], 2)
        self.assertTrue((idx.scenario_id >= 100000).all())
        self.assertTrue(idx.mission.isin(["throttle_steps", "idle_descent"]).all())


if __name__ == "__main__":
    unittest.main()
