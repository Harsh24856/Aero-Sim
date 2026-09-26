"""Instrument model for physics v5: sensors_v4 plus two turbo instruments, and
sensor-fault labels that follow what the data can actually show.

Changes from sensors_v4 (docs/v5_model_improvement_plan.md, Phases 1-2)
----------------------------------------------------------------------
1. TWO NEW INSTRUMENTS on turbocharged engines, which the real engines carry:
   `airbox_temp_c` (the 914 TCU's airbox temperature sensor; the 915/916 iS
   intake-air temperature) and `wastegate_position` (the 914 servo; the iS
   electric wastegate actuator). The 912 has neither; the generator and twin
   write 0.0 for them there (absent instrument, constant input).
   Sensor faults stay on the original twelve channels (FAULTABLE_CHANNELS), so
   the sensor-fault head's classes are unchanged.
2. BALANCED FAULT PLANS. v4 gave 30% of flights one or two faults of random
   channel and kind - about 10-15 training flights per (channel, kind) cell.
   `plan_faults` takes a `cell` index the generator cycles through, so every
   (channel, kind) pair gets the same share, and the fault rate is a parameter
   (v5: 60% of flights).
3. VISIBILITY LABELS. v4 labelled a fault from its onset, when e.g. a drift has
   zero offset and a stuck sensor at steady cruise reads exactly what a healthy
   one would. Those windows are unlearnable and taught the head to guess. The
   label (`flag`) now switches on when the fault is VISIBLE:
     bias, noise    immediately (their size is set well above the noise)
     drift          once |offset| >= 1 sigma of that sensor's noise
     stuck          once the true value has moved >= max(1 sigma, 1 quantum)
                    from the frozen reading, or after 30 s on a channel whose
                    noise is at least two quanta (so a healthy reading would
                    visibly move)
     spike, dropout once the first spike / dropout sample has occurred
   The onset-based label is still returned (`active`) for reference.
4. DRIFT RATE. v4 drifted 0.9 sigma per 600 s at full severity, about 0.2 sigma
   over a 128 s window - invisible in the window the models see. v5 drifts
   (1.0 + 3.0 s) sigma per 600 s: a failing thermocouple junction or pressure
   transducer that walks several sigma over a flight hour. ASSUMPTION (no
   sensor-datasheet figure for an in-flight failure rate); it is the rate at
   which a drift becomes a maintenance finding within one sortie.
5. EARLY ONSETS. Onsets may fall anywhere from 2% to 85% of the flight, so
   windows containing the transition itself are in the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sensors_v4 import (  # noqa: F401  re-exported
    CORRELATION_GROUPS, DRIFT_TAU_S, SENSOR_CHANNELS as FAULTABLE_CHANNELS,
    SENSOR_FAULT_TYPES, SENSOR_SPEC as SENSOR_SPEC_V4,
)

TURBO_CHANNELS = ["airbox_temp_c", "wastegate_position"]
SENSOR_CHANNELS = list(FAULTABLE_CHANNELS) + TURBO_CHANNELS          # 14 instruments

SENSOR_SPEC = dict(SENSOR_SPEC_V4)
SENSOR_SPEC.update({
    # ASSUMED: automotive-grade NTC intake-air sensor, ~1 C noise, 0.5 C display step.
    "airbox_temp_c":      {"noise_sd": 0.8,  "white_frac": 0.80, "quantum": 0.5,   "sat": (-40.0, 200.0)},
    # ASSUMED: servo / actuator position feedback as a 0-1 fraction open, 0.5% step.
    "wastegate_position": {"noise_sd": 0.01, "white_frac": 0.70, "quantum": 0.005, "sat": (0.0, 1.0)},
})

FAULT_KINDS = SENSOR_FAULT_TYPES[1:]                      # bias drift stuck spike noise dropout
N_CELLS = len(FAULTABLE_CHANNELS) * len(FAULT_KINDS)      # 72 (channel, kind) cells
STUCK_OBVIOUS_S = 30.0


def cell_of(index: int) -> tuple[str, str]:
    """The (channel, kind) cell for a cycling index - kind varies fastest, so
    consecutive flights cover different kinds."""
    i = index % N_CELLS
    return FAULTABLE_CHANNELS[i // len(FAULT_KINDS)], FAULT_KINDS[i % len(FAULT_KINDS)]


@dataclass
class SensorFault:
    channel: str
    kind: str
    onset_s: float
    severity: float          # 0..1
    drift_rate: float = 0.0  # sigma per 600 s, drift only
    _stuck_value: float | None = None
    _stuck_true: float = 0.0  # the TRUE value when it froze (visibility reference)
    _events: int = 0         # spikes / dropouts so far
    visible: bool = False


@dataclass
class SensorBank:
    """Stateful instrument bank. One per simulated aircraft.

    `turbocharged` False blanks the two turbo instruments (the 912 has neither).
    """
    rng: np.random.Generator
    dt: float = 1.0
    turbocharged: bool = True
    faults: list = field(default_factory=list)
    _drift: dict = field(default_factory=dict)
    _bias: dict = field(default_factory=dict)
    _chol: dict = field(default_factory=dict)

    def __post_init__(self):
        for c in SENSOR_CHANNELS:
            self._drift[c] = 0.0
            self._bias[c] = float(self.rng.normal(0.0, 0.25 * SENSOR_SPEC[c]["noise_sd"]))
        for chans, rho in CORRELATION_GROUPS:
            n = len(chans)
            m = np.full((n, n), rho)
            np.fill_diagonal(m, 1.0)
            self._chol[tuple(chans)] = np.linalg.cholesky(m)

    # ------------------------------------------------------------------
    def plan_faults(self, duration_s: float, p_any: float = 0.60,
                    cell: int | None = None, p_second: float = 0.25) -> None:
        """Afflict this flight's instruments. With `cell`, the first fault is that
        (channel, kind) cell, so cycling `cell` over flights balances every cell;
        a second fault on another channel follows with probability `p_second`."""
        if self.rng.random() > p_any:
            return
        if cell is not None:
            first = cell_of(cell)
        else:
            first = (str(self.rng.choice(FAULTABLE_CHANNELS)), str(self.rng.choice(FAULT_KINDS)))
        plan = [first]
        if self.rng.random() < p_second:
            others = [c for c in FAULTABLE_CHANNELS if c != first[0]]
            plan.append((str(self.rng.choice(others)), str(self.rng.choice(FAULT_KINDS))))
        for ch, kind in plan:
            s = float(self.rng.uniform(0.2, 1.0))
            self.faults.append(SensorFault(
                channel=ch, kind=kind,
                onset_s=float(self.rng.uniform(0.02, 0.85) * duration_s),
                severity=s,
                drift_rate=(1.0 + 3.0 * s) if kind == "drift" else 0.0))

    # ------------------------------------------------------------------
    def _noise(self) -> dict:
        noise = {}
        alpha = self.dt / DRIFT_TAU_S
        for chans, _ in CORRELATION_GROUPS:
            L = self._chol[tuple(chans)]
            z_w = L @ self.rng.normal(0.0, 1.0, len(chans))
            z_s = L @ self.rng.normal(0.0, 1.0, len(chans))
            for i, c in enumerate(chans):
                spec = SENSOR_SPEC[c]
                wf, sd = spec["white_frac"], spec["noise_sd"]
                self._drift[c] += -alpha * self._drift[c] + np.sqrt(
                    max(2.0 * alpha, 1e-9)) * sd * np.sqrt(1.0 - wf) * z_s[i]
                noise[c] = self._drift[c] + sd * np.sqrt(wf) * z_w[i]
        for c in SENSOR_CHANNELS:
            if c not in noise:
                spec = SENSOR_SPEC[c]
                wf, sd = spec["white_frac"], spec["noise_sd"]
                self._drift[c] += -alpha * self._drift[c] + np.sqrt(
                    max(2.0 * alpha, 1e-9)) * sd * np.sqrt(1.0 - wf) * self.rng.normal()
                noise[c] = self._drift[c] + sd * np.sqrt(wf) * self.rng.normal()
        return noise

    def read(self, truth: dict, t_s: float) -> tuple[dict, dict, dict, dict]:
        """True physics values -> instrument readings.

        Returns (measured, flag, severity, active): `flag` is the VISIBLE fault
        kind index per faultable channel (the sensor-fault head's target),
        `active` the kind index from onset (reference only).
        """
        noise = self._noise()
        measured, flag, sev, active = {}, {}, {}, {}
        for c in SENSOR_CHANNELS:
            spec = SENSOR_SPEC[c]
            sd, q = spec["noise_sd"], spec["quantum"]
            if c in TURBO_CHANNELS and not self.turbocharged:
                measured[c] = 0.0
                continue
            true_v = float(truth.get(c, 0.0))
            v = true_v + self._bias[c] + noise[c]
            if c in FAULTABLE_CHANNELS:
                flag[c], sev[c], active[c] = 0, 0.0, 0

            for f in self.faults:
                if f.channel != c or t_s < f.onset_s:
                    continue
                age = t_s - f.onset_s
                s = f.severity
                k = SENSOR_FAULT_TYPES.index(f.kind)
                if f.kind == "bias":
                    v += s * 8.0 * sd
                    f.visible = True
                elif f.kind == "drift":
                    off = f.drift_rate * sd * (age / 600.0)
                    v += off
                    f.visible = f.visible or abs(off) >= sd
                elif f.kind == "stuck":
                    if f._stuck_value is None:
                        f._stuck_value, f._stuck_true = v, true_v
                    v = f._stuck_value
                    moved = abs(true_v - f._stuck_true) >= max(sd, q)
                    flat_is_odd = sd >= 2.0 * q and age >= STUCK_OBVIOUS_S
                    f.visible = f.visible or moved or flat_is_odd
                elif f.kind == "spike":
                    if self.rng.random() < 0.05 * s:
                        v += self.rng.choice([-1.0, 1.0]) * s * 25.0 * sd
                        f._events += 1
                    f.visible = f._events > 0
                elif f.kind == "noise":
                    v += self.rng.normal(0.0, s * 6.0 * sd)
                    f.visible = True
                elif f.kind == "dropout":
                    if self.rng.random() < 0.08 * s:
                        v = spec["sat"][0]
                        f._events += 1
                    f.visible = f._events > 0
                active[c] = k
                if f.visible:
                    flag[c] = k
                    sev[c] = max(sev[c], s)

            lo, hi = spec["sat"]
            v = min(max(v, lo), hi)
            if q > 0:
                v = round(v / q) * q
            measured[c] = float(v)
        return measured, flag, sev, active
