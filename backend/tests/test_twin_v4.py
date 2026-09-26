"""The live v4 twin (twin_v4.py) and its demo scenarios (scenarios_v4.py).

Each fault preset is compared with THE SAME ENGINE WITH THE FAULT REMOVED, so the
test measures the fault's own effect and not the baseline wear the engine also
carries at that age (a healthy 914 at 600 h already reads ~8 sigma low on oil
pressure from bearing clearance growth alone).

Run from backend/:
    .venv/bin/python -m unittest tests.test_twin_v4 -v
"""
import copy
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import safety                                    # noqa: E402
import scenarios_v4 as S                         # noqa: E402
from sensors_v4 import SENSOR_SPEC               # noqa: E402
from twin_v4 import RESIDUAL_CHANNELS, UAVEngineTwinV4   # noqa: E402

ENGINE = "Rotax_914_ULF"
FLIGHT_S = 180


def fly(record, seconds=FLIGHT_S, altitude=1500.0, throttle=0.75):
    """Mean residual per channel, in sensor-noise units, after the first minute."""
    tw = UAVEngineTwinV4(engine_model=record["engine_model"], engine=record, seed=5)
    tw.altitude, tw.airspeed, tw.throttle = altitude, 50.0, throttle
    res, outs = [], []
    for _ in range(seconds * 100):
        o = tw.step()
        assert safety.telemetry_problem(o) is None, safety.telemetry_problem(o)
        if o["sample_new"]:
            outs.append(o)
            if o["time"] > 60:
                res.append([o[f"res_{c}"] / SENSOR_SPEC[c]["noise_sd"] for c in RESIDUAL_CHANNELS])
    return dict(zip(RESIDUAL_CHANNELS, np.mean(res, axis=0))), outs


def without_faults(record):
    r = copy.deepcopy(record)
    r["fault_plan"]["faults"], r["sensor_plan"] = [], []
    return r


class Presets(unittest.TestCase):
    # preset -> (flight altitude, [(channel, sign, min |effect| in noise sigmas)])
    EXPECT = {
        "bearing_wear":    (1500.0, [("oil_pressure", -1, 10.0)]),
        "oil_pump":        (1500.0, [("oil_pressure", -1, 5.0)]),
        "cooling_hot_day": (1500.0, [("cht", +1, 2.0), ("oil_temp", +1, 2.0)]),
        "turbo_high":      (6000.0, [("engine_rpm", -1, 5.0)]),     # above the critical altitude
        "prop_erosion":    (1500.0, [("engine_rpm", +1, 2.0)]),     # the prop absorbs less torque
    }

    def test_every_fault_preset_moves_its_residuals_the_expected_way(self):
        for name, (alt, checks) in self.EXPECT.items():
            rec = S.make_engine(ENGINE, name, seed=11)
            faulty, outs = fly(rec, altitude=alt)
            clean, _ = fly(without_faults(rec), altitude=alt)
            self.assertTrue(outs[-1]["truth"]["faults_present"], name)
            for ch, sign, size in checks:
                effect = faulty[ch] - clean[ch]
                with self.subTest(preset=name, channel=ch):
                    self.assertGreater(sign * effect, size, f"{name} {ch} effect {effect:+.1f} sigma")

    def test_healthy_engine_residuals_stay_within_noise(self):
        res, outs = fly(S.make_engine(ENGINE, "healthy", seed=3))
        for ch, v in res.items():
            self.assertLess(abs(v), 5.0, f"healthy {ch} residual {v:+.1f} sigma")
        self.assertFalse(outs[-1]["truth"]["faults_present"])
        self.assertFalse(outs[-1]["truth"]["sensor_faults"])

    def test_sensor_presets_break_the_instrument_not_the_engine(self):
        _, outs = fly(S.make_engine(ENGINE, "sensor_cht_dropout", seed=4), seconds=320)
        after = [o for o in outs if o["time"] > S.SENSOR_ONSET_S]
        # A dropout reads the bottom of the sender's range (-40 C for CHT), not zero.
        floor = SENSOR_SPEC["cht"]["sat"][0]
        self.assertTrue(any(o["cht"] == floor for o in after), "CHT never dropped out")
        self.assertEqual(after[-1]["truth"]["sensor_faults"], {"cht": "dropout"})
        self.assertFalse(after[-1]["truth"]["faults_present"])      # the engine itself is fine

        _, outs = fly(S.make_engine(ENGINE, "sensor_egt_stuck", seed=4), seconds=320)
        after = [o["egt"] for o in outs if o["time"] > S.SENSOR_ONSET_S + 1]
        self.assertEqual(len(set(after)), 1, "EGT kept moving after it stuck")

    def test_presets_respect_hardware(self):
        self.assertNotIn("turbo_high", S.available("Rotax_912_ULS"))
        self.assertIn("turbo_high", S.available("Rotax_914_ULF"))
        with self.assertRaises(ValueError):
            S.make_engine("Rotax_912_ULS", "turbo_high")
        # The 915's TBO is 1,200 h: presets are kept inside its life.
        self.assertLessEqual(S.make_engine("Rotax_915_iS", "bearing_wear")["engine_hours"], 0.8 * 1200)

    def test_no_preset_starts_worn_out(self):
        # A random baseline once put the 1,450 h presets past the end of life (true
        # RUL 0 before takeoff). Only the preset's own fault may limit the life, and
        # the healthy preset must reach its scheduled overhaul.
        from twin_v4 import degradation_from_record
        for eng in ("Rotax_912_ULS", "Rotax_914_ULF", "Rotax_915_iS", "Rotax_916_iS"):
            for name in S.available(eng):
                for seed in range(5):
                    rec = S.make_engine(eng, name, seed=seed)
                    d = degradation_from_record(rec)
                    with self.subTest(engine=eng, preset=name, seed=seed):
                        self.assertGreater(d.condition_at(rec["engine_hours"]), 0.0)
                        if name == "healthy":
                            self.assertGreaterEqual(d.wear_out_hours(), rec["tbo_hours"])
                            self.assertGreaterEqual(d.condition_at(rec["engine_hours"]), 0.9)

    def test_every_engine_runs_every_preset_without_nan(self):
        for eng in ("Rotax_912_ULS", "Rotax_914_ULF", "Rotax_915_iS", "Rotax_916_iS"):
            for name in S.available(eng):
                tw = UAVEngineTwinV4(engine_model=eng, engine=S.make_engine(eng, name, seed=1), seed=1)
                for _ in range(1000):
                    o = tw.step()
                with self.subTest(engine=eng, preset=name):
                    self.assertIsNone(safety.telemetry_problem(o))


class ResumeAndInject(unittest.TestCase):
    def test_restore_continues_the_same_engine(self):
        tw = UAVEngineTwinV4(engine_model=ENGINE, engine=S.make_engine(ENGINE, "oil_pump", seed=6), seed=6)
        for _ in range(3000):
            snap = tw.step()
        fresh = UAVEngineTwinV4(engine_model=ENGINE)          # a default engine, then restored
        fresh.restore_state(snap)
        self.assertAlmostEqual(fresh.engine_hours, snap["engine_hours"], places=3)
        self.assertEqual(fresh.record["scenario"], "oil_pump")
        self.assertAlmostEqual(fresh.eng.cht_k, snap["physics_state"]["eng"]["cht_k"])
        nxt = fresh.step()
        self.assertAlmostEqual(nxt["time"], snap["time"] + 0.01, places=6)
        self.assertLess(abs(nxt["cht"] - snap["cht"]), 5.0)       # no thermal snap back to cold

    def test_injection_changes_the_engine_and_the_truth(self):
        tw = UAVEngineTwinV4(engine_model=ENGINE, engine=S.make_engine(ENGINE, "healthy", seed=7), seed=7)
        tw.inject_fault("cooling_degradation", 0.5)
        tw.inject_sensor("oil_temp", "bias", 1.0)
        for _ in range(300):
            o = tw.step()
        self.assertIn("cooling_degradation", o["truth"]["faults_present"])
        self.assertEqual(o["truth"]["sensor_faults"], {"oil_temp": "bias"})
        self.assertEqual(len(o["engine_record"]["fault_plan"]["faults"]), 1)
        with self.assertRaises(ValueError):
            UAVEngineTwinV4(engine_model="Rotax_912_ULS").inject_fault("wastegate_fault")

    def test_telemetry_problem_catches_a_bad_v4_step(self):
        o = UAVEngineTwinV4(engine_model=ENGINE).step()
        self.assertIsNone(safety.telemetry_problem(o))
        o["margin_min"] = math.nan
        self.assertIn("margin_min", safety.telemetry_problem(o))


if __name__ == "__main__":
    unittest.main()
