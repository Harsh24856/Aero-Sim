"""The live twin feeds aiv4 exactly the inputs the models were trained on.

Chain: validation_v4/tf_data_pipeline.py -> export_deployable_v4.py writes it into each
manifest -> aiv4.py refuses to start if the pipeline disagrees with a manifest -> this
test checks the backend twin, sensors and fault taxonomy against the same manifest.
No TensorFlow needed, so it runs in the backend's own environment.

Run from backend/:
    .venv/bin/python -m unittest tests.test_ai_v4_contract -v
"""
import glob
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main                                      # noqa: E402  (AI_FEATURE_COLS_V4)
import twin_v4                                   # noqa: E402
from degradation_v4 import FAULT_NAMES           # noqa: E402
from sensors_v4 import SENSOR_CHANNELS, SENSOR_FAULT_TYPES   # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_v4")


class Contract(unittest.TestCase):
    def manifests(self):
        found = sorted(glob.glob(os.path.join(MODELS, "*", "manifest.json")))
        if not found:
            self.skipTest("no models_v4 export yet")
        return [json.load(open(p)) for p in found]

    def test_twin_matches_every_manifest(self):
        for m in self.manifests():
            c = m["contract"]
            with self.subTest(engine=m["engine"]):
                self.assertEqual(twin_v4.FEATURE_COLS, c["feature_cols"])
                self.assertEqual(twin_v4.RESIDUAL_CHANNELS, c["residual_channels"])
                self.assertEqual(list(SENSOR_CHANNELS), c["sensor_channels"])
                self.assertEqual(list(SENSOR_FAULT_TYPES), c["sensor_fault_types"])
                self.assertEqual(list(FAULT_NAMES), c["fault_modes"])
                self.assertEqual(c["window_size"], 128)
                self.assertEqual(c["sample_hz"], 1)
                self.assertEqual(twin_v4.FAULT_PRESENT_SEV, c["fault_present_severity"])

    def test_rul_inputs_are_live_not_labels(self):
        for m in self.manifests():
            live = m["contract"]["rul_live_inputs"]
            self.assertEqual(live["health_index"], "health head")
            self.assertEqual(live["margin_min"], "measured sensors")
            self.assertIn("rul_live_inputs", m)            # the RUL error reported is the live one

    def test_main_sends_every_input_the_twin_produces(self):
        out = twin_v4.UAVEngineTwinV4(engine_model="Rotax_914_ULF").step()
        missing = [c for c in main.AI_FEATURE_COLS_V4 if c not in out]
        self.assertEqual(missing, [])
        self.assertEqual(main.AI_FEATURE_COLS_V4[:29], twin_v4.FEATURE_COLS)


if __name__ == "__main__":
    unittest.main()
