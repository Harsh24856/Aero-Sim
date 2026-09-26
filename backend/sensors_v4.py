"""Sensor model: what the instrument reports, as opposed to what the engine does.

Kept strictly separate from physics_v4, because the C-MAPSS work showed how much
depends on that boundary being clean. The noise STRUCTURE here is not invented -
it is the structure measured from the released NASA C-MAPSS data in
simulation/cmapss/noise.py, where three successive discoveries were each forced
by a classifier that could otherwise tell real data from generated:

  1. Real sensor noise is SMOOTH plus WHITE in proportions that differ wildly by
     channel. Matching only variance and autocorrelation still pins the
     cycle-to-cycle step size, and it came out 2.4x too large.
  2. The smooth component's TIMESCALE matters independently of its size. An
     AR(1) fast enough to match lag-1 correlation spends its variance inside a
     short window; real slow channels spread it across the whole flight.
  3. Channels are CROSS-CORRELATED, because they share the same physical
     disturbances. Generating each independently leaves a single sample
     trivially recognisable even when every marginal matches.

A model trained on data without these will look excellent in validation and
degrade on real telemetry, because it will have learned to rely on a smoothness
that real instruments do not have.

SENSOR FAULTS ARE NOT ENGINE FAULTS. A drifting thermocouple and a genuinely
overheating engine produce different joint patterns - the residual against the
physics twin moves for one and not the others - and the two are labelled
separately so the model can be asked to tell them apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Channels the aircraft actually instruments.
SENSOR_CHANNELS = ["egt", "cht", "coolant_temp", "oil_temp", "oil_pressure",
                   "engine_rpm", "fuel_flow", "manifold_pressure_kpa",
                   "vibx", "viby", "vibz", "battery_voltage"]

# Per-channel instrument characteristics.
#   noise_sd        total measurement noise, in the channel's own units
#   white_frac      fraction of that variance that is white; the rest is slow
#                   drift. Values follow the pattern measured on C-MAPSS:
#                   thermocouples are nearly white, speeds and pressures are
#                   dominated by slow drift.
#   quantum         display/telemetry resolution
#   sat             (min, max) the instrument can physically report
SENSOR_SPEC = {
    "egt":                  {"noise_sd": 6.0,    "white_frac": 0.88, "quantum": 1.0,   "sat": (0.0, 1200.0)},
    "cht":                  {"noise_sd": 1.4,    "white_frac": 0.82, "quantum": 1.0,   "sat": (-40.0, 300.0)},
    "coolant_temp":         {"noise_sd": 1.1,    "white_frac": 0.80, "quantum": 1.0,   "sat": (-40.0, 200.0)},
    "oil_temp":             {"noise_sd": 1.0,    "white_frac": 0.70, "quantum": 1.0,   "sat": (-40.0, 200.0)},
    "oil_pressure":         {"noise_sd": 0.055,  "white_frac": 0.55, "quantum": 0.01,  "sat": (0.0, 10.0)},
    "engine_rpm":           {"noise_sd": 9.0,    "white_frac": 0.25, "quantum": 10.0,  "sat": (0.0, 8000.0)},
    "fuel_flow":            {"noise_sd": 0.28,   "white_frac": 0.60, "quantum": 0.1,   "sat": (0.0, 80.0)},
    "manifold_pressure_kpa": {"noise_sd": 0.45,  "white_frac": 0.50, "quantum": 0.1,   "sat": (0.0, 250.0)},
    # Vibration channels are RMS amplitudes (physics_v4._vibration), so they
    # cannot go negative: a dead accelerometer reads zero. The old (-5, 5)
    # range suited the signed sine they used to be, and pinned a dropout at -5.
    "vibx":                 {"noise_sd": 0.010,  "white_frac": 0.95, "quantum": 0.001, "sat": (0.0, 5.0)},
    "viby":                 {"noise_sd": 0.010,  "white_frac": 0.95, "quantum": 0.001, "sat": (0.0, 5.0)},
    "vibz":                 {"noise_sd": 0.012,  "white_frac": 0.95, "quantum": 0.001, "sat": (0.0, 5.0)},
    "battery_voltage":      {"noise_sd": 0.035,  "white_frac": 0.65, "quantum": 0.01,  "sat": (0.0, 20.0)},
}

# Slow-drift time constant, in samples at 1 Hz. CALIBRATED to the C-MAPSS
# finding that the smooth component must spread its variance over the whole
# record rather than a short window.
DRIFT_TAU_S = 420.0

# Groups of channels that share a physical disturbance and therefore correlate.
# Thermocouples see the same cooling air; the vibration axes share the same
# structure; the fluid channels share the same pump and plumbing.
CORRELATION_GROUPS = [
    (["egt", "cht", "coolant_temp", "oil_temp"], 0.45),
    (["vibx", "viby", "vibz"], 0.60),
    (["oil_pressure", "fuel_flow", "manifold_pressure_kpa"], 0.30),
]

SENSOR_FAULT_TYPES = ["none", "bias", "drift", "stuck", "spike", "noise", "dropout"]


@dataclass
class SensorFault:
    channel: str
    kind: str
    onset_s: float
    severity: float          # 0..1
    _stuck_value: float | None = None


@dataclass
class SensorBank:
    """Stateful instrument bank. One per simulated aircraft."""
    rng: np.random.Generator
    dt: float = 1.0
    faults: list = field(default_factory=list)
    _drift: dict = field(default_factory=dict)
    _bias: dict = field(default_factory=dict)
    _chol: dict = field(default_factory=dict)

    def __post_init__(self):
        for c in SENSOR_CHANNELS:
            # A fixed calibration offset per installation, drawn once.
            self._drift[c] = 0.0
            self._bias[c] = float(self.rng.normal(0.0, 0.25 * SENSOR_SPEC[c]["noise_sd"]))
        # Cholesky factors for each correlated group.
        for chans, rho in CORRELATION_GROUPS:
            n = len(chans)
            m = np.full((n, n), rho)
            np.fill_diagonal(m, 1.0)
            self._chol[tuple(chans)] = np.linalg.cholesky(m)

    # ------------------------------------------------------------------
    def add_random_faults(self, max_faults: int = 2, p_any: float = 0.30,
                          duration_s: float = 20000.0) -> None:
        """Optionally afflict a few channels. Sensor faults are independent of
        engine health by construction - that is the whole point of modelling
        them separately."""
        if self.rng.random() > p_any:
            return
        n = int(self.rng.integers(1, max_faults + 1))
        for c in self.rng.choice(SENSOR_CHANNELS, size=n, replace=False):
            kind = str(self.rng.choice(SENSOR_FAULT_TYPES[1:]))
            self.faults.append(SensorFault(
                channel=str(c), kind=kind,
                onset_s=float(self.rng.uniform(0.05, 0.9) * duration_s),
                severity=float(self.rng.uniform(0.2, 1.0))))

    # ------------------------------------------------------------------
    def read(self, truth: dict, t_s: float) -> tuple[dict, dict, dict]:
        """Convert true physics values into instrument readings.

        Returns (measured, fault_flag, fault_severity), the last two being the
        supervision targets for the sensor-fault head.
        """
        # -- correlated noise, smooth + white --------------------------------
        noise = {}
        for chans, _ in CORRELATION_GROUPS:
            L = self._chol[tuple(chans)]
            z_w = L @ self.rng.normal(0.0, 1.0, len(chans))
            z_s = L @ self.rng.normal(0.0, 1.0, len(chans))
            for i, c in enumerate(chans):
                spec = SENSOR_SPEC[c]
                wf = spec["white_frac"]
                sd = spec["noise_sd"]
                # Slow component is an OU process, so its variance lives on the
                # flight timescale rather than the sample timescale.
                alpha = self.dt / DRIFT_TAU_S
                self._drift[c] += -alpha * self._drift[c] + np.sqrt(
                    max(2.0 * alpha, 1e-9)) * sd * np.sqrt(1.0 - wf) * z_s[i]
                noise[c] = self._drift[c] + sd * np.sqrt(wf) * z_w[i]
        for c in SENSOR_CHANNELS:
            if c not in noise:
                spec = SENSOR_SPEC[c]
                wf, sd = spec["white_frac"], spec["noise_sd"]
                alpha = self.dt / DRIFT_TAU_S
                self._drift[c] += -alpha * self._drift[c] + np.sqrt(
                    max(2.0 * alpha, 1e-9)) * sd * np.sqrt(1.0 - wf) * self.rng.normal()
                noise[c] = self._drift[c] + sd * np.sqrt(wf) * self.rng.normal()

        measured, flag, sev = {}, {}, {}
        for c in SENSOR_CHANNELS:
            spec = SENSOR_SPEC[c]
            v = float(truth.get(c, 0.0)) + self._bias[c] + noise[c]
            flag[c], sev[c] = 0, 0.0

            for f in self.faults:
                if f.channel != c or t_s < f.onset_s:
                    continue
                age = t_s - f.onset_s
                s = f.severity
                if f.kind == "bias":
                    v += s * 8.0 * spec["noise_sd"]
                elif f.kind == "drift":
                    v += s * 0.9 * spec["noise_sd"] * (age / 600.0)
                elif f.kind == "stuck":
                    if f._stuck_value is None:
                        f._stuck_value = v
                    v = f._stuck_value
                elif f.kind == "spike":
                    if self.rng.random() < 0.05 * s:
                        v += self.rng.choice([-1.0, 1.0]) * s * 25.0 * spec["noise_sd"]
                elif f.kind == "noise":
                    v += self.rng.normal(0.0, s * 6.0 * spec["noise_sd"])
                elif f.kind == "dropout":
                    if self.rng.random() < 0.08 * s:
                        v = spec["sat"][0]
                flag[c] = SENSOR_FAULT_TYPES.index(f.kind)
                sev[c] = max(sev[c], s)

            lo, hi = spec["sat"]
            v = min(max(v, lo), hi)
            q = spec["quantum"]
            if q > 0:
                v = round(v / q) * q
            measured[c] = float(v)

        return measured, flag, sev
