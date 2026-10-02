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
from twin_v5 import UAVEngineTwinV5  # noqa: E402

AI = {"status": "ok", "model_version": "v5", "fault_detected": True, "detection_confidence": 0.9,
      "fault_modes": {"oil_pump_degradation": {"probability": 0.9, "present": True, "threshold": 0.3,
                                               "severity": 0.4, "family": "oil"},
                      "bearing_wear": {"probability": 0.1, "present": False, "threshold": 0.4,
                                       "severity": 0.0, "family": "oil"}},
      "faults_present": ["oil_pump_degradation"], "families": {"oil": 0.9}, "families_present": ["oil"],
      "sensors": {}, "faulty_sensors": [], "health_percent": 80.0, "wear_condition": 0.8}


class TestMainV5(unittest.TestCase):
    def test_hold_clears_families(self):
        main.state["ai_last_sample_time"] = None
        main.state["ai_settle_until"] = 0.0
        held = main.hold_alerts_outside_envelope({"time": 10, "airspeed": 5.0}, AI)
        self.assertTrue(held["settling"])
        self.assertEqual(held["families_present"], [])

    def test_repair_takes_the_family_with_it(self):
        held = main.without_removed(AI, fault="oil_pump_degradation")
        self.assertEqual(held["families_present"], [])

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
