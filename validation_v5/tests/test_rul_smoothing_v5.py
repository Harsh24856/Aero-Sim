"""The in-flight RUL rule never rises and does not add error on noisy predictions.

    validation/venv/bin/python validation_v5/tests/test_rul_smoothing_v5.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rul_v5 import RulSmoother, smooth_within_flight  # noqa: E402

rng = np.random.default_rng(0)
raw_err, sm_err = [], []
for _ in range(300):
    h = np.linspace(500, 500 + rng.choice([1, 125]), 40)      # x1 and x180 life clocks
    true = 1500 - h
    p = true + rng.normal(0, 80, 40)
    s = smooth_within_flight(p, h)
    assert (np.diff(s) <= 1e-9).all() and (s >= 0).all()
    live = RulSmoother()
    np.testing.assert_allclose([live.update(a, b) for a, b in zip(p, h)], s, atol=1e-9)   # live == offline
    raw_err.append(np.abs(p - true).mean())
    sm_err.append(np.abs(s - true).mean())
assert np.mean(sm_err) < np.mean(raw_err), (np.mean(sm_err), np.mean(raw_err))
print(f"ok: raw error {np.mean(raw_err):.1f} h, smoothed {np.mean(sm_err):.1f} h")
