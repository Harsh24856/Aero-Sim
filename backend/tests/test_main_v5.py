"""main.py under physics v5: the take-off hold, repairs and the CAN overlay.

Run alone (main reads AERO_PHYSICS_VERSION once, at import), from backend/:
    .venv/bin/python -m unittest tests.test_main_v5
"""
import os
import sys
import unittest

os.environ["AERO_PHYSICS_VERSION"] = "v5"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
import twin_v5  # noqa: E402
from twin_v5 import UAVEngineTwinV5  # noqa: E402

if main.PHYSICS_VERSION != os.environ["AERO_PHYSICS_VERSION"]:   # main reads it once, at import
    raise ImportError(f"main is already loaded as physics {main.PHYSICS_VERSION}, this module needs "
                      f"{os.environ['AERO_PHYSICS_VERSION']} - run it in its own process (scripts/test_all.sh)")

AI = {"status": "ok", "model_version": "v5", "fault_detected": True, "detection_confidence": 0.9,
      "fault_modes": {"oil_pump_degradation": {"probability": 0.9, "present": True, "threshold": 0.3,
                                               "severity": 0.4, "family": "oil"},
                      "bearing_wear": {"probability": 0.1, "present": False, "threshold": 0.4,
                                       "severity": 0.0, "family": "oil"}},
      "faults_present": ["oil_pump_degradation"], "families": {"oil": 0.9}, "families_present": ["oil"],
      "sensors": {}, "faulty_sensors": [], "health_percent": 80.0, "wear_condition": 0.8}


class TestMainV5(unittest.TestCase):
    def test_ai_payload_is_the_twins_list(self):
        self.assertEqual(main.AI_FEATURE_COLS_V5, twin_v5.AI_PAYLOAD)

    def test_hold_clears_families(self):
        main.state["ai_last_sample_time"] = None
        main.state["ai_settle_until"] = 0.0
        held = main.hold_alerts_outside_envelope({"time": 10, "airspeed": 5.0}, AI)
        self.assertTrue(held["settling"])
        self.assertEqual(held["families_present"], [])

    def test_repair_takes_the_family_with_it(self):
        held = main.without_removed(AI, fault="oil_pump_degradation")
        self.assertEqual(held["families_present"], [])

    def _fly(self, legs, engine="Rotax_912_ULS"):
        """legs: (seconds, throttle); returns the hold result for each second."""
        main.CURRENT_ENGINE_MODEL = engine
        main.state["ai_last_sample_time"] = None
        main.state["ai_settle_until"] = 0.0
        main.state["ai_throttle_hist"] = []
        out, t = [], 0
        for secs, thr in legs:
            for _ in range(secs):
                t += 1
                out.append(main.hold_alerts_outside_envelope(
                    {"time": t, "airspeed": 45.0, "throttle": thr}, AI))
        return out

    def test_throttle_chop_holds_alerts(self):
        # Live 916: a chop from full power to 0.15 gave "FAULT 100%" on a healthy engine.
        r = self._fly([(200, 0.75), (int(main.THROTTLE_HOLD_S) + 5, 0.15)])
        self.assertFalse(r[199].get("settling", False))
        self.assertTrue(r[200]["settling"])
        self.assertEqual(r[200]["settle_reason"], "throttle")
        self.assertTrue(r[200 + int(main.THROTTLE_HOLD_S) - 2]["settling"])
        self.assertFalse(r[-1].get("settling", False))

    def test_throttle_advance_holds_alerts(self):
        # 916: 0.15 -> full gave "FAULT" until the step left the 128 s window.
        r = self._fly([(200, 0.15), (5, 1.0)])
        self.assertEqual(r[200]["settle_reason"], "throttle")
        self.assertEqual(r[200]["settle_window_s"], main.AI_WINDOW_S)

    def test_engines_trained_on_throttle_steps_are_not_held(self):
        # P2: the 914 and 916 run detection/sensor nets fine-tuned on step flights.
        for engine in ("Rotax_914_ULF", "Rotax_916_iS"):
            r = self._fly([(200, 0.75), (60, 0.15), (60, 1.0)], engine=engine)
            self.assertFalse(any(x.get("settling") for x in r), engine)

    def test_915_is_still_held(self):
        r = self._fly([(200, 0.75), (5, 0.15)], engine="Rotax_915_iS")
        self.assertEqual(r[200]["settle_reason"], "throttle")

    def test_small_throttle_change_does_not_hold(self):
        r = self._fly([(200, 0.75), (60, 0.50)])
        self.assertFalse(any(x.get("settling") for x in r))

    def test_slow_throttle_reduction_does_not_hold(self):
        # The training missions ease the throttle down; a 0.6 drop over a minute is in-envelope.
        r = self._fly([(200, 0.75)] + [(1, 0.75 - 0.01 * i) for i in range(1, 61)])
        self.assertFalse(any(x.get("settling") for x in r))

    def test_takeoff_hold_names_its_reason(self):
        main.state["ai_last_sample_time"] = None
        main.state["ai_settle_until"] = 0.0
        held = main.hold_alerts_outside_envelope({"time": 10, "airspeed": 5.0, "throttle": 1.0}, AI)
        self.assertEqual(held["settle_reason"], "takeoff")

    def test_throttle_hold_freezes_rul_and_keeps_its_floor(self):
        main.reset_rul_filter()
        ok = {**AI, "tbo_hours": 2000.0}
        for t in range(1, 61):
            shown = main.smooth_rul({"time": t, "wear": 0.5}, {**ok, "rul_hours": 900.0})["rul_hours"]
        held = main.smooth_rul({"time": 61, "wear": 0.5},
                               {**ok, "rul_hours": 1400.0, "settling": True, "settle_reason": "throttle"})
        self.assertTrue(held["rul_ready"])
        self.assertEqual(held["rul_hours"], shown)
        after = main.smooth_rul({"time": 62, "wear": 0.5}, {**ok, "rul_hours": 1400.0})
        self.assertLessEqual(after["rul_hours"], shown)

    def test_throttle_hold_rul_still_counts_down_hours_flown(self):
        # Live 916: the frozen gauge read 1,678 h beside a 1,677 h calendar countdown.
        main.reset_rul_filter()
        ok = {**AI, "tbo_hours": 2000.0}
        shown = main.smooth_rul({"time": 1, "wear": 0.5}, {**ok, "rul_hours": 900.0})["rul_hours"]
        held = main.smooth_rul({"time": 61, "wear": 0.5005},
                               {**ok, "rul_hours": 1400.0, "settling": True, "settle_reason": "throttle"})
        self.assertAlmostEqual(held["rul_hours"], shown - 1.0, places=3)

    def test_display_smoothing_damps_cruise_flicker(self):
        # Live 916 cruise: the detector jumped between 0.7% and 50% second to second.
        main.reset_display_filter()
        shown = []
        for t in range(1, 121):
            raw = 0.5 if t % 2 else 0.007
            r = main.smooth_display({"time": t}, {**AI, "detection_confidence": raw,
                                                  "fault_detected": raw >= 0.5})
            shown.append(r)
        self.assertTrue(all(0.15 < r["detection_confidence"] < 0.35 for r in shown[30:]))
        self.assertFalse(any(r["fault_detected"] for r in shown))
        self.assertEqual(shown[-1]["detection_confidence_raw"], 0.007)

    def test_display_smoothing_still_raises_a_real_alarm(self):
        main.reset_display_filter()
        for t in range(1, 61):
            main.smooth_display({"time": t}, {**AI, "detection_confidence": 0.01, "fault_detected": False})
        hits = [t for t in range(61, 91)
                if main.smooth_display({"time": t}, {**AI, "detection_confidence": 1.0,
                                                     "fault_detected": True})["fault_detected"]]
        self.assertLessEqual(hits[0] - 61, 8)
        last = main.smooth_display({"time": 91}, {**AI, "detection_confidence": 1.0, "fault_detected": True})
        self.assertGreater(last["detection_confidence"], 0.9)

    def test_display_smoothing_health(self):
        main.reset_display_filter()
        for t, h in zip(range(1, 61), [89.0, 93.0] * 30):
            r = main.smooth_display({"time": t}, {**AI, "health_percent": h})
        self.assertAlmostEqual(r["health_percent"], 91.0, delta=0.5)
        self.assertEqual(r["health_percent_raw"], 93.0)

    def test_display_smoothing_restarts_on_new_flight_and_after_a_hold(self):
        main.reset_display_filter()
        for t in range(1, 61):
            main.smooth_display({"time": t}, {**AI, "detection_confidence": 1.0})
        fresh = main.smooth_display({"time": 2}, {**AI, "detection_confidence": 0.02})
        self.assertLess(fresh["detection_confidence"], 0.02)
        held = {**AI, "settling": True, "detection_confidence": 0.0, "health_percent": None}
        self.assertEqual(main.smooth_display({"time": 3}, held), held)
        after = main.smooth_display({"time": 4}, {**AI, "detection_confidence": 0.9, "health_percent": 70.0})
        self.assertLess(after["detection_confidence"], 0.1)
        self.assertFalse(after["fault_detected"])
        self.assertEqual(after["health_percent"], 70.0)

    def test_can_overlay_912_turbo_residuals_stay_zero(self):
        main.twin = UAVEngineTwinV5(engine_model="Rotax_912_ULS", seed=1)
        out = None
        for _ in range(100):
            out = main.twin.step()
        merged = main.apply_measured_v4(out, {"egt": out["egt"] + 10.0})
        self.assertEqual(merged["res_airbox_temp_c"], out["res_airbox_temp_c"])
        self.assertEqual(merged["res_wastegate_position"], out["res_wastegate_position"])
        self.assertAlmostEqual(merged["res_egt"], out["res_egt"] + 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
