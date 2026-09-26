"""The v4 time model (timescale_v4.py): the life clock advances LIFE_SCALE x the
duty-weighted flight usage, the flight clock is unscaled, and a long flight keeps
every AI input inside the range the models were trained on.

Run from backend/:
    .venv/bin/python -m unittest tests.test_timescale_v4 -v
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scenarios_v4                              # noqa: E402
import timescale_v4 as TS                        # noqa: E402
from twin_v4 import UAVEngineTwinV4              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TimeScale(unittest.TestCase):
    def test_constant_is_the_derived_one(self):
        # Spec T: 0.05 severity per window / (0.0077 per h x 128 s) -> 0.05 h per flight s.
        self.assertEqual(TS.LIFE_SCALE, 180.0)
        self.assertEqual(TS.engine_hours(100.0, 1.0), 280.0)
        self.assertEqual(TS.describe()["life_scale"], 180.0)

    def test_life_clock_is_scaled_flight_clock_is_not(self):
        tw = UAVEngineTwinV4(engine_model="Rotax_914_ULF",
                             engine=scenarios_v4.make_engine("Rotax_914_ULF", "healthy", seed=1), seed=1)
        tw.altitude, tw.airspeed, tw.throttle = 1500.0, 48.0, 0.75
        for _ in range(60 * 100):
            out = tw.step()
        self.assertAlmostEqual(out["time"], 60.0, places=6)                 # flight clock: 1:1
        gained = out["engine_hours"] - out["start_engine_hours"]
        self.assertAlmostEqual(gained, 180.0 * out["life_used_hours"], places=3)
        # Duty is clamped to [0.25, 1.5] in physics_v4, so 60 flight seconds age the
        # engine between 0.75 h and 4.5 h - and never by the unscaled 60 s.
        self.assertTrue(0.75 <= gained <= 4.5, gained)
        self.assertLess(out["life_used_hours"], 60.0 / 3600.0 * 1.5 + 1e-9)

    def test_long_flight_stays_inside_the_training_range(self):
        """30 flight minutes (90 engine hours): every one of the 29 model inputs within
        6 standard deviations of the training scaler, and the two clocks consistent."""
        scaler_path = os.path.join(ROOT, "validation_v4", "artifacts", "914", "scaler.pkl")
        if not os.path.exists(scaler_path):
            self.skipTest("914 scaler not trained yet")
        import joblib
        sys.path.insert(0, os.path.join(ROOT, "validation_v4"))
        from tf_data_pipeline import FEATURE_COLS
        sc = joblib.load(scaler_path)
        tw = UAVEngineTwinV4(engine_model="Rotax_914_ULF",
                             engine=scenarios_v4.make_engine("Rotax_914_ULF", "bearing_wear", seed=2), seed=2)
        worst = 0.0
        for k in range(30 * 60 * 100):
            s = k / 100.0
            tw.altitude = min(300.0 + 3.0 * s, 4000.0)        # climb, then cruise with power changes
            tw.airspeed = 45.0 if s < 300 else 52.0
            tw.throttle = 0.95 if s < 300 else (0.65 if (s // 240) % 2 else 0.85)
            out = tw.step()
            if out["sample_new"]:
                z = [abs((out[c] - m) / sd) for c, m, sd in zip(FEATURE_COLS, sc.mean_, sc.scale_)]
                worst = max(worst, max(z))
                json.dumps(out)                            # the broadcast must serialise
        self.assertLess(worst, 6.0)
        self.assertAlmostEqual(out["time"], 1800.0, places=4)
        self.assertGreater(out["engine_hours"] - out["start_engine_hours"], 30.0)


if __name__ == "__main__":
    unittest.main()
