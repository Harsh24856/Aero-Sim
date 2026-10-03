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

    def test_old_export_unchanged(self):
        """The live 914 export predates hist_cols: a logbook in the payload changes nothing."""
        self.assertFalse(aiv5.loaded_engines["Rotax_914_ULF"]["dep"].contract.get("hist_cols"))
        aiv5.select_engine({"engine_model": "Rotax_914_ULF"})
        plain = [aiv5.step(p) for p in self.flight[:128]][-1]
        aiv5.select_engine({"engine_model": "Rotax_914_ULF"})
        hist = {"lag_h": 80.0, "wear_ratio": 1.7, "sev_max": 0.4}
        with_hist = [aiv5.step({**p, "engine_history": hist}) for p in self.flight[:128]][-1]
        for k in ("rul_hours", "rul_hours_raw", "detection_confidence", "wear_condition"):
            self.assertEqual(plain[k], with_hist[k], k)

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

    def test_rul_is_the_models_own_value(self):
        """In-flight smoothing is main.smooth_rul's job (it knows about the take-off
        hold); a second smoother here locked ground-roll predictions in."""
        from unittest import mock
        aiv5.select_engine({"engine_model": "Rotax_914_ULF"})
        for p in self.flight[:128]:                      # its own full window, whatever ran before
            aiv5.flight.update(p)
        e = aiv5.loaded_engines["Rotax_914_ULF"]
        real = e["dep"].predict_window
        outs = iter([300.0, 900.0])                      # a low ground-roll guess, then the real one

        def fake(seq, ctx, hours, tbo=None, hist=None):
            o = real(seq, ctx, hours, tbo, hist) if tbo is not None else real(seq, ctx, hours, hist=hist)
            return {**o, "rul_hours": next(outs), "rul_calendar_hours": 1500.0}
        with mock.patch.object(e["dep"], "predict_window", side_effect=fake):
            aiv5.run_inference()
            r = aiv5.run_inference()
        self.assertEqual(r["rul_hours"], 900.0)

    def test_tree_models_single_threaded_in_request_threads(self):
        """FastAPI runs /step in worker threads; an OpenMP limit set on the import
        thread does not reach them (78 ms vs 14 ms per inference)."""
        import threading
        from threadpoolctl import threadpool_info
        seen = []
        th = threading.Thread(target=lambda: seen.extend(
            p["num_threads"] for p in threadpool_info() if p.get("user_api") == "openmp"))
        th.start(); th.join()
        self.assertTrue(seen and all(n == 1 for n in seen), seen)

    def test_placeholder_engine_uses_its_own_tbo(self):
        """A 915 served by the 914 export: calendar and RUL on the 915's TBO."""
        aiv5.select_engine({"engine_model": "Rotax_915_iS"})
        try:
            r = None
            for p in self.flight[:128]:
                r = aiv5.step(p)
            tbo = aiv5.loaded_engines["Rotax_915_iS"]["tbo_hours"]
            self.assertEqual(r["tbo_hours"], tbo)
            self.assertAlmostEqual(r["rul_calendar_hours"], max(0.0, tbo - self.flight[127]["engine_hours"]), places=2)
            self.assertLessEqual(r["rul_hours"], r["rul_calendar_hours"] + 1e-6)
        finally:
            aiv5.select_engine({"engine_model": "Rotax_914_ULF"})

    def test_live_equals_offline_assembly(self):
        """The service's window through Deployed directly gives the same heads."""
        dep = aiv5.loaded_engines["Rotax_914_ULF"]["dep"]
        rows = np.asarray([[p[c] for c in aiv5.F.FEATURE_COLS] for p in self.flight[-128:]])
        o = dep({"seq": dep.scale_seq(rows)[None], "ctx": dep.scale_ctx(np.asarray(self.flight[-1]["ai_context"]))[None]})
        self.assertAlmostEqual(float(o["detection"][0, 0]), self.results[-1]["detection_confidence"], places=4)

    def test_null_input_is_an_error_not_a_crash(self):
        """A NaN in the twin reaches aiv5 as null (safety.json_safe)."""
        p = dict(self.flight[-1])
        p["egt"] = None
        self.assertEqual(aiv5.step(p)["status"], "error")
        p = dict(self.flight[-1])
        p["ai_context"] = [None] * len(p["ai_context"])
        self.assertEqual(aiv5.step(p)["status"], "error")

    def test_rejects_bad_payload(self):
        self.assertEqual(aiv5.step({"altitude": 1.0})["status"], "error")


if __name__ == "__main__":
    unittest.main()
