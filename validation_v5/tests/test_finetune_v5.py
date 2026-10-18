"""Fine-tuning a trained specialist: per-fault loss weights for the diagnosis head.
Pure numpy - no GPU (the Trainer itself needs one; this checks the weights it uses).

    validation/venv/bin/python validation_v5/tests/test_finetune_v5.py
"""
import os
import sys
import unittest

V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V5)
sys.path.insert(0, os.path.join(os.path.dirname(V5), "backend"))

from degradation_v5 import FAULT_NAMES  # noqa: E402
from train_v5 import fault_weights  # noqa: E402

APP = [n for n in FAULT_NAMES if n != "turbo_degradation"]


class TestFaultWeights(unittest.TestCase):
    def test_fault_weights_scales_named_fault(self):
        w = fault_weights(FAULT_NAMES, {"air_filter_fouling": 2.0}, APP)
        self.assertEqual(w[FAULT_NAMES.index("air_filter_fouling")], 2.0)
        self.assertEqual(w[FAULT_NAMES.index("bearing_wear")], 1.0)
        self.assertEqual(w[FAULT_NAMES.index("turbo_degradation")], 0.0)

    def test_fault_weights_rejects_a_fault_the_engine_lacks(self):
        with self.assertRaises(ValueError):
            fault_weights(FAULT_NAMES, {"turbo_degradation": 2.0}, APP)

    def test_empty_weights_equal_todays_mask(self):
        w = fault_weights(FAULT_NAMES, {}, APP)
        self.assertEqual(list(w), [1.0 if n in APP else 0.0 for n in FAULT_NAMES])


if __name__ == "__main__":
    unittest.main()
