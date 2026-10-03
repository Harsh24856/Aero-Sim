"""The v5 input contract: one definition, imported by the dataset generator, the
training pipeline and the live twin, so the three can never disagree.

Per-second features (the 128 s window the encoder reads)
    8  flight / atmosphere   altitude, airspeed, aoa, throttle, ambient_temp_c,
                             air_density, isa_dev_c, humidity_frac
   14  instruments           the twelve v4 channels + airbox_temp_c and
                             wastegate_position (turbo engines; 0 on the 912)
    3  derived               prop_rpm, thrust_margin, lift_weight_margin
   14  residuals             every instrument minus the healthy on-board twin
   --
   39

v4 had residuals on six channels. v5 gives every instrument one, for two reasons:
the manifold-pressure, airbox-temperature and wastegate residuals are what make
induction and turbo faults observable (Phase 1), and a residual per channel lets
the sensor-fault head treat all channels alike (reading + residual) with one
shared classifier (Phase 4), and makes drift on coolant, vibration and battery
channels visible at all.

Per-window context (long horizon, Phase 3)
    For each residual channel: an EMA over 10 min and over 60 min (level), and a
    TREND = (EMA_3min - EMA_15min) / (15 min - 3 min). For a residual growing as a
    ramp a*t, an EMA with time constant tau lags it by exactly tau, so the trend
    recovers a (units per second) - which is what makes a slow sensor drift, and
    a slowly developing fault, visible beyond the 128 s window. All filters are
    causal and start from zero at the start of the recording, identically in the
    generator's arrays and in the live twin's incremental version.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from sensors_v5 import FAULTABLE_CHANNELS, SENSOR_CHANNELS, TURBO_CHANNELS  # noqa: F401

CONTRACT_VERSION = "v5.0"
# The on-board twin residuals are taken against: degradation_v5.fleet_wear_health
# at the engine's hours. Exports without this key were trained against a
# brand-new engine ("new_engine"); twin_v5 serves each model the twin it learnt.
TWIN = "calibrated_wear"
# The engine's logbook baseline (degradation_v5.engine_history), one value per flight:
# the twin is calibrated with hist_wear_ratio, and RUL reads all three.
HIST_COLS = ["hist_lag_h", "hist_wear_ratio", "hist_sev_max"]
HIST_NEUTRAL = [0.0, 1.0, 0.0]           # "no logbook": data and exports from before v6

FLIGHT_COLS = ["altitude", "airspeed", "aoa", "throttle",
               "ambient_temp_c", "air_density", "isa_dev_c", "humidity_frac"]
MEASURED_COLS = list(SENSOR_CHANNELS)                          # 14
DERIVED_COLS = ["prop_rpm", "thrust_margin", "lift_weight_margin"]
RESIDUAL_CHANNELS = list(SENSOR_CHANNELS)                      # 14
RESIDUAL_COLS = [f"res_{c}" for c in RESIDUAL_CHANNELS]
FEATURE_COLS = FLIGHT_COLS + MEASURED_COLS + DERIVED_COLS + RESIDUAL_COLS
N_FEATURES = len(FEATURE_COLS)                                 # 39

# Long-horizon context, per residual channel.
TAU_LEVEL_S = (600.0, 3600.0)          # 10 min, 60 min
TAU_TREND_S = (180.0, 900.0)           # 3 min, 15 min -> trend
LONG_KINDS = ("ema10m", "ema60m", "trend")
LONG_COLS = [f"lh_{c}_{k}" for c in RESIDUAL_CHANNELS for k in LONG_KINDS]
N_LONG = len(LONG_COLS)                                        # 42

WINDOW = 128


def _alpha(tau_s: float, dt: float = 1.0) -> float:
    return 1.0 - math.exp(-dt / tau_s)


def long_horizon_array(res: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Causal long-horizon context for one recording.

    res: [T, 14] residuals in RESIDUAL_CHANNELS order. Returns [T, 42] in
    LONG_COLS order (per channel: ema10m, ema60m, trend)."""
    res = np.asarray(res, dtype=np.float64)

    def ema(tau):
        a = _alpha(tau, dt)
        return lfilter([a], [1.0, -(1.0 - a)], res, axis=0)

    e10, e60 = ema(TAU_LEVEL_S[0]), ema(TAU_LEVEL_S[1])
    trend = (ema(TAU_TREND_S[0]) - ema(TAU_TREND_S[1])) / (TAU_TREND_S[1] - TAU_TREND_S[0])
    out = np.stack([e10, e60, trend], axis=2)                  # [T, C, 3]
    return out.reshape(res.shape[0], -1).astype(np.float32)


class LongHorizon:
    """Incremental version of long_horizon_array for the live twin (one 1 Hz
    sample at a time); identical to the array form to float precision."""

    def __init__(self, dt: float = 1.0):
        self.a = [_alpha(t, dt) for t in (*TAU_LEVEL_S, *TAU_TREND_S)]
        self.state = np.zeros((4, len(RESIDUAL_CHANNELS)))

    def update(self, res_row) -> np.ndarray:
        x = np.asarray(res_row, dtype=np.float64)
        for i, a in enumerate(self.a):
            self.state[i] += a * (x - self.state[i])
        return self.current()

    def current(self) -> np.ndarray:
        """The 42 values for the samples seen so far, without taking a new one."""
        e10, e60, e3, e15 = self.state
        trend = (e3 - e15) / (TAU_TREND_S[1] - TAU_TREND_S[0])
        return np.stack([e10, e60, trend], axis=1).reshape(-1).astype(np.float32)


def residuals(measured: dict, twin_truth: dict, turbocharged: bool) -> list:
    """Instrument reading minus the healthy twin's TRUE value, per channel. The
    turbo instruments do not exist on a naturally aspirated engine: 0."""
    out = []
    for c in RESIDUAL_CHANNELS:
        if c in TURBO_CHANNELS and not turbocharged:
            out.append(0.0)
        else:
            out.append(float(measured[c]) - float(twin_truth.get(c, 0.0)))
    return out


def contract() -> dict:
    """The contract as data (written to contract_v5.json by the pipeline)."""
    return {
        "version": CONTRACT_VERSION, "window": WINDOW, "dt_s": 1.0, "twin": TWIN,
        "feature_cols": FEATURE_COLS, "n_features": N_FEATURES,
        "flight_cols": FLIGHT_COLS, "measured_cols": MEASURED_COLS,
        "derived_cols": DERIVED_COLS, "residual_channels": RESIDUAL_CHANNELS,
        "residual_cols": RESIDUAL_COLS, "faultable_channels": list(FAULTABLE_CHANNELS),
        "turbo_channels": list(TURBO_CHANNELS),
        "long_cols": LONG_COLS, "n_long": N_LONG,
        "long_def": {"tau_level_s": list(TAU_LEVEL_S), "tau_trend_s": list(TAU_TREND_S),
                     "trend": "(ema(tau3)-ema(tau15))/(tau15-tau3)", "init": "zero at recording start"},
    }
