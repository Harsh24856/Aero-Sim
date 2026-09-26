"""Demo scenarios for the live v4 twin, as engine records.

An ENGINE RECORD is everything needed to rebuild one physical engine exactly - the
same dict the `engines` table stores (dbv4.py) and twin_v4.UAVEngineTwinV4 flies:

    {"engine_model", "tbo_hours", "engine_hours", "degradation_seed",
     "fault_plan":  {"baseline": {"a", "b", "scale"},
                     "faults": [{"name", "onset_h", "a", "b", "depth", "sense"}]},
     "sensor_plan": [{"channel", "kind", "onset_flight_s", "severity"}],
     "scenario", "environment": {Inputs field: value}}

WHY PRESETS START MID-FAULT. Faults take a median 476 engine hours to develop and a
10-minute flight ages the engine 30 h at LIFE_SCALE 180 (timescale_v4.py). So a preset
places the engine at the point in its life where the fault is already physically
visible (severity ~0.35) and the flight shows it growing from there - the honest way
to demonstrate a slow fault on a live clock. Sensor faults start 200 flight seconds
in: after the AI's 128 s window has filled on clean data, so the audience sees the
instrument fail and the AI notice.
"""
from __future__ import annotations

import math

import numpy as np

import physics_v4 as V
from degradation_v4 import FAULT_MODES, A_RANGE, B_RANGE, DegradationState, applicable_faults

# Growth parameters of a typical fault: the medians of what the generator draws
# (a = U(A_RANGE) x U(2, 9), b = U(B_RANGE), depth = spec depth x U(0.55, 1)).
TYPICAL_A = 0.5 * (A_RANGE[0] + A_RANGE[1]) * 5.5
TYPICAL_B = 0.5 * (B_RANGE[0] + B_RANGE[1])
TYPICAL_DEPTH_FRAC = 0.775
SENSOR_ONSET_S = 200.0

# name -> (label, start engine hours, [(fault, severity at start)], [(channel, kind, severity)], environment)
PRESETS = {
    "healthy":            ("Healthy engine, 300 h", 300.0, [], [], {}),
    "bearing_wear":       ("Bearing wear developing, 1,450 h", 1450.0, [("bearing_wear", 0.35)], [], {}),
    "oil_pump":           ("Oil pump degradation, 900 h", 900.0, [("oil_pump_degradation", 0.35)], [], {}),
    "cooling_hot_day":    ("Cooling degradation on a hot day (ISA +20)", 1100.0,
                           [("cooling_degradation", 0.35)], [], {"isa_dev_c": 20.0, "humidity_frac": 0.6}),
    "turbo_high":         ("Turbo degradation (climb above 15,000 ft to see it)", 1200.0,
                           [("turbo_degradation", 0.40)], [], {}),
    "prop_erosion":       ("Prop erosion, 1,300 h", 1300.0, [("prop_erosion", 0.35)], [], {}),
    "sensor_cht_dropout": ("Sensor: CHT dropout", 600.0, [], [("cht", "dropout", 0.8)], {}),
    "sensor_egt_stuck":   ("Sensor: EGT stuck", 600.0, [], [("egt", "stuck", 1.0)], {}),
}
DEFAULT_PRESET = "healthy"


def age_for_severity(severity: float, a: float, b: float, tbo: float) -> float:
    """Engine hours since onset at which a fault with growth (a, b) reaches `severity`.
    Inverse of DegradationState._progress: s = 1 - exp(-a (age / tbo * 100)^b)."""
    x = -math.log(max(1e-9, 1.0 - severity))
    return tbo / 100.0 * (x / a) ** (1.0 / b)


def fault_entry(name: str, severity: float, engine_hours: float, tbo: float) -> dict:
    """One fault_plan entry that has reached `severity` at `engine_hours`."""
    spec = FAULT_MODES[name]
    return {"name": name,
            "onset_h": engine_hours - age_for_severity(severity, TYPICAL_A, TYPICAL_B, tbo),
            "a": TYPICAL_A, "b": TYPICAL_B,
            "depth": spec["depth"] * TYPICAL_DEPTH_FRAC, "sense": spec["sense"]}


def sensor_entry(channel: str, kind: str, severity: float, onset_flight_s: float = SENSOR_ONSET_S) -> dict:
    return {"channel": channel, "kind": kind, "onset_flight_s": float(onset_flight_s),
            "severity": float(severity)}


def available(engine_model: str) -> list:
    """Presets this engine can fly (no turbo fault on a 912)."""
    s = V.ENGINE_SPECS_V4[engine_model]
    ok = set(applicable_faults(s.turbocharged, s.intercooled))
    return [k for k, (_, _, faults, _, _) in PRESETS.items() if all(f in ok for f, _ in faults)]


def listing(engine_model: str) -> list:
    return [{"name": k, "label": PRESETS[k][0], "start_engine_hours": PRESETS[k][1]}
            for k in available(engine_model)]


def make_engine(engine_model: str, preset: str = DEFAULT_PRESET, seed: int | None = None) -> dict:
    """A fresh engine record for a preset. Baseline wear is drawn from the same
    distribution as the training data; the preset's faults are placed exactly."""
    if preset not in PRESETS:
        raise ValueError(f"unknown scenario {preset!r}; options {list(PRESETS)}")
    if preset not in available(engine_model):
        raise ValueError(f"scenario {preset!r} needs hardware {engine_model} does not have")
    label, hours, faults, sensors, env = PRESETS[preset]
    tbo = V.ENGINE_SPECS_V4[engine_model].tbo_hours
    # The 915's TBO is 1,200 h: keep every preset inside its life.
    hours = min(hours, 0.8 * tbo)
    # Baseline wear is drawn as the dataset draws it, but only from engines whose
    # baseline alone reaches the scheduled overhaul. Some draws wear the engine out
    # far earlier (one seed at ~1,073 h), which put the 1,450 h presets past the end
    # of life before the flight began - the dataset generator rejects those engines
    # too. In a preset the injected fault, not a random baseline, limits the life.
    # "Healthy" must also look healthy: at least 90% of new at its start hours (one
    # draw was 17% worn at 300 h, which reads 0.36 bar low on oil pressure against the
    # twin - real wear, but not what the preset promises).
    pick = np.random.default_rng(seed)
    for _ in range(500):
        s = int(pick.integers(1, 2**31 - 1))
        base = DegradationState(np.random.default_rng(s), hours, tbo, n_faults=0)
        if base.condition_at(tbo) > 0.0 and (preset != "healthy" or base.condition_at(hours) >= 0.9):
            break
    seed = s
    return {
        "engine_model": engine_model, "tbo_hours": tbo, "engine_hours": hours,
        "degradation_seed": seed, "scenario": preset,
        "fault_plan": {"baseline": {"a": base.base_a, "b": base.base_b, "scale": base.base_scale},
                       "faults": [fault_entry(f, s, hours, tbo) for f, s in faults]},
        "sensor_plan": [sensor_entry(c, k, s) for c, k, s in sensors],
        "environment": dict(env),
    }
