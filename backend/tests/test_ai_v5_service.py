"""aiv5 fed by twin_v5: the aiv4 response contract, and live == offline assembly.

Needs TensorFlow (the models), so run from backend/ with the AI environment:
    ../validation/venv/bin/python -m unittest tests.test_ai_v5_service -v
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import aiv5  # noqa: E402  (loads backend/models_v5)
from twin_v5 import UAVEngineTwinV5  # noqa: E402

AIV4_KEYS = {"status", "model_version", "engine_model", "placeholder_models", "models_engine",
             "fault_detected", "detection_confidence", "fault_modes", "faults_present", "sensors",
             "faulty_sensors", "wear_condition", "health_percent", "margin_min", "rul_hours",
             "rul_mae_hours", "tbo_hours", "rul_calendar_hours", "rul_percent_remaining",
             "wear_limited", "rul_out_of_range", "steps_collected", "steps_needed"}
PAYLOAD = aiv5.F.FEATURE_COLS + ["engine_hours", "life_used_hours", "ai_context", "time"]


def samples(tw, n):
    out = []
    while len(out) < n:
        s = tw.step()
        if s["sample_new"]:
            out.append({k: s[k] for k in PAYLOAD})
    return out


class TestAiV5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        aiv5.select_engine({"engine_model": "Rotax_914_ULF"})
        cls.flight = samples(UAVEngineTwinV5(engine_model="Rotax_914_ULF", seed=11), 140)
        cls.results = [aiv5.step(p) for p in cls.flight]

    def test_warms_up_then_answers(self):
        self.assertEqual(self.results[0]["status"], "warming_up")
        self.assertEqual(self.results[126]["status"], "warming_up")
        self.assertEqual(self.results[127]["status"], "ok")

    def test_aiv4_shape_plus_v5_fields(self):
        r = self.results[-1]
        self.assertLessEqual(AIV4_KEYS, set(r))
        self.assertEqual(r["model_version"], "v5")
        self.assertEqual(r["severity_kind"], "effective")
        self.assertEqual(len(r["sensors"]), 12)
        self.assertLessEqual(r["rul_hours"], r["rul_calendar_hours"])
        for f in r["fault_modes"].values():
            self.assertIn("family", f)

    def test_rul_never_rises_within_flight(self):
        ruls = [r["rul_hours"] for r in self.results if r["status"] == "ok"]
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(ruls, ruls[1:])))

    def test_live_equals_offline_assembly(self):
        """The service's window through Deployed directly gives the same heads."""
        dep = aiv5.loaded_engines["Rotax_914_ULF"]["dep"]
        rows = np.asarray([[p[c] for c in aiv5.F.FEATURE_COLS] for p in self.flight[-128:]])
        o = dep({"seq": dep.scale_seq(rows)[None], "ctx": dep.scale_ctx(np.asarray(self.flight[-1]["ai_context"]))[None]})
        self.assertAlmostEqual(float(o["detection"][0, 0]), self.results[-1]["detection_confidence"], places=4)

    def test_rejects_bad_payload(self):
        self.assertEqual(aiv5.step({"altitude": 1.0})["status"], "error")


if __name__ == "__main__":
    unittest.main()
