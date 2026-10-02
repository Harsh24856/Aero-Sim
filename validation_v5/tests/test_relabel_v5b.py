"""relabel_v5b on a synthetic flight.

    validation/venv/bin/python validation_v5/tests/test_relabel_v5b.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)

import relabel_v5b as R  # noqa: E402
from degradation_v4 import FAULT_MODES  # noqa: E402
from degradation_v5 import FAULT_NAMES  # noqa: E402
from sensors_v5 import FAULTABLE_CHANNELS, SENSOR_FAULT_TYPES  # noqa: E402


class TestRelabel(unittest.TestCase):
    def setUp(self):
        n = 4
        self.g = pd.DataFrame({"t": np.arange(n) * 600.0, "effect_z": [0.5, 2.0, 3.5, 9.0]})
        self.old = {f"fm_{f}": np.zeros(n) for f in FAULT_NAMES}
        self.old.update({f"fmv_{f}": np.zeros(n) for f in FAULT_NAMES})
        self.old.update({f"sf_{c}_flag": np.zeros(n) for c in FAULTABLE_CHANNELS})
        self.old["fm_valve_leakage"] = np.array([0.2, 0.2, 0.2, 0.2])
        self.faults = [{"name": "valve_leakage", "depth": 0.5 * FAULT_MODES["valve_leakage"]["depth"]}]
        ch = FAULTABLE_CHANNELS[0]
        self.old[f"sf_{ch}_flag"] = np.full(n, SENSOR_FAULT_TYPES.index("drift"), float)
        # drift at severity 0 -> 1 sigma per 600 s: offsets 0,1,2,3 sigma at t = 0,600,1200,1800
        self.sf = [{"channel": ch, "kind": "drift", "onset_s": 0.0, "severity": 0.0}]

    def run_(self):
        return R.relabel_flight(self.g, np.arange(4), self.faults, self.sf, self.old, z_fault=3.0, z_sensor=3.0)

    def test_effective_severity(self):
        new = self.run_()
        np.testing.assert_allclose(new["fm_valve_leakage"], 0.1)          # 0.2 damage x half depth

    def test_fault_visible_only_above_threshold(self):
        new = self.run_()
        np.testing.assert_array_equal(new["fmv_valve_leakage"], [0, 0, 1, 1])
        np.testing.assert_array_equal(new["fault_present"], [0, 0, 1, 1])

    def test_per_channel_effect_thresholds(self):
        """With eff_ columns, a 5-sigma effect is visible on a quiet channel (bar 3) but
        not on a channel whose healthy spread is 31 sigma."""
        g = self.g.copy()
        g["eff_cht"], g["eff_oil_pressure"] = [0.0, 5.0, 0.0, 0.0], [5.0, 0.0, 40.0, 0.0]
        new = R.relabel_flight(g, np.arange(4), self.faults, [], self.old, z_fault=1.0, z_sensor=3.0,
                               z_effect={"cht": 3.0, "oil_pressure": 31.0})
        np.testing.assert_array_equal(new["fmv_valve_leakage"], [0, 1, 1, 0])

    def test_small_drift_relabelled_none(self):
        new = self.run_()
        k = SENSOR_FAULT_TYPES.index("drift")
        np.testing.assert_array_equal(new[f"sf_{FAULTABLE_CHANNELS[0]}_flag"], [0, 0, 0, k])
        np.testing.assert_array_equal(new["sensor_fault_any"], [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
