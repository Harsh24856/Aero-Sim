"""twin_v5: the v5 twin emits the training contract every flight second.

    cd backend && .venv/bin/python -m unittest tests.test_twin_v5
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import features_v5 as F  # noqa: E402
import scenarios_v4  # noqa: E402
from twin_v5 import UAVEngineTwinV5  # noqa: E402


def fly(tw, seconds):
    samples = []
    for _ in range(int(seconds / tw.dt)):
        out = tw.step()
        if out["sample_new"]:
            samples.append(out)
    return samples


class TestTwinV5(unittest.TestCase):
    def test_contract_every_second(self):
        tw = UAVEngineTwinV5(engine_model="Rotax_914_ULF", seed=1)
        s = fly(tw, 5)
        self.assertEqual(len(s), 5)
        for out in s:
            for c in F.FEATURE_COLS:
                self.assertIn(c, out, c)
                self.assertTrue(np.isfinite(out[c]), c)
            self.assertEqual(len(out["ai_context"]), F.N_LONG + 3)
            self.assertTrue(np.isfinite(out["ai_context"]).all())
        self.assertEqual(out["ai_context"][-1], 1.0)                  # x180 life clock flag

    def test_912_turbo_channels_blank(self):
        tw = UAVEngineTwinV5(engine_model="Rotax_912_ULS", seed=1)
        out = fly(tw, 2)[-1]
        self.assertEqual(out["res_airbox_temp_c"], 0.0)
        self.assertEqual(out["res_wastegate_position"], 0.0)

    def test_healthy_residuals_small_and_fault_moves_them(self):
        rec = scenarios_v4.make_engine("Rotax_914_ULF", seed=3)
        tw = UAVEngineTwinV5(engine_model="Rotax_914_ULF", engine=rec, seed=3)
        base = fly(tw, 20)[-1]
        tw.inject_fault("cooling_degradation", 0.9)
        after = fly(tw, 60)[-1]
        self.assertIn("cooling_degradation", after["truth"]["faults_present"])
        self.assertEqual(after["truth"]["severity_kind"], "effective")
        self.assertGreater(abs(after["res_cht"]), abs(base["res_cht"]))

    def test_restore_keeps_long_horizon(self):
        tw = UAVEngineTwinV5(engine_model="Rotax_914_ULF", seed=4)
        last = fly(tw, 30)[-1]
        tw2 = UAVEngineTwinV5(engine_model="Rotax_914_ULF", seed=4)
        tw2.restore_state(last)
        np.testing.assert_allclose(tw2.lh.state, tw.lh.state)

    def test_sensor_drift_injection(self):
        tw = UAVEngineTwinV5(engine_model="Rotax_914_ULF", seed=5)
        fly(tw, 2)
        tw.inject_sensor("egt", "drift", 1.0)
        self.assertEqual(tw.sensors.faults[-1].drift_rate, 4.0)


if __name__ == "__main__":
    unittest.main()
