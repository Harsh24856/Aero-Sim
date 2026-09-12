"""Engine failure modes for physics v3 (PS 26054 section C).

WHY THIS IS SEPARATE FROM THE FAULTS ALREADY IN physics.py
----------------------------------------------------------
physics.py's `_inject_faults` corrupts SENSOR READINGS - Bias, Drift, Spike,
Stuck-At, Noise. Those are instrumentation failures: the engine is fine, the
wire or the thermocouple is not.

The problem statement asks for something different - ENGINE failure modes:
misfire, injector abnormalities, cooling degradation, combustion instability.
None of those were simulated anywhere. Every one of those words appeared only in
advisory.py, as a label mapped onto a sensor channel, which is presentation
rather than modelling. This module makes them real physical behaviours so the
models can actually learn them.

DESIGN
------
Each mode is a latent condition with its own onset and its own growth rate. Once
started it worsens monotonically - these are degradation processes, not the
reversible stress accumulators in `_auto_fault_step`. Onset probability rises
with accumulated wear and with how hard the engine is being worked, so failures
cluster where they should rather than firing uniformly at random.

The class exposes the severities (which become the `fm_*` training labels) and a
set of small accessors that physics.py multiplies into the relevant channel. No
channel is touched here directly - physics.py stays the single place where
telemetry is produced.
"""

import numpy as np

MODES = ["misfire", "injector_fouling", "cooling_degradation", "combustion_instability"]


class FailureModes:
    """Latent engine-fault state for one twin.

    `enabled=False` makes every accessor a no-op identity, so physics v2 keeps
    exactly its old behaviour and the v2 bit-identity gate still holds.
    """

    # Per-second onset probability at reference conditions (wear 0.5, severity 0.5).
    # Deliberately small: a mode should be present in a minority of scenarios, or
    # the "healthy" class disappears from the training set.
    BASE_ONSET = {
        "misfire":                2.0e-5,
        "injector_fouling":       3.0e-5,
        "cooling_degradation":    2.5e-5,
        "combustion_instability": 1.5e-5,
    }

    # How fast severity climbs once started, per second. Injector fouling and
    # cooling degradation are slow, creeping processes; misfire and combustion
    # instability escalate faster once they take hold.
    GROWTH = {
        "misfire":                1.2e-4,
        "injector_fouling":       4.0e-5,
        "cooling_degradation":    5.0e-5,
        "combustion_instability": 9.0e-5,
    }

    def __init__(self, enabled: bool = True, rng: np.random.RandomState | None = None):
        self.enabled = enabled
        self._rng = rng if rng is not None else np.random
        self.severity = {m: 0.0 for m in MODES}
        self.active = {m: False for m in MODES}
        self.onset_time = {m: None for m in MODES}
        # Refreshed each step so the accessors are cheap and deterministic within
        # a step - important because several are read more than once.
        self._misfire_now = False
        self._rpm_noise = 0.0
        self._egt_noise = 0.0

    # ---------------------------------------------------------------- update --
    def step(self, dt, t, rpm_frac, power_frac, wear):
        """Advance onsets and severities by one timestep."""
        if not self.enabled:
            return

        # Hard operation and an already-worn engine both make onset likelier.
        duty = 0.5*rpm_frac**2 + 0.5*power_frac**2
        hazard_scale = (0.3 + 1.4*duty) * (0.4 + 1.6*wear)

        for m in MODES:
            if not self.active[m]:
                if self._rng.rand() < self.BASE_ONSET[m] * hazard_scale * dt:
                    self.active[m] = True
                    self.onset_time[m] = t
            else:
                # Monotonic: degradation does not heal. Growth is faster under load.
                self.severity[m] = min(1.0, self.severity[m]
                                       + self.GROWTH[m] * (0.5 + duty) * dt)

        # Misfire is intermittent by nature: it fires on individual cycles rather
        # than as a steady offset, so it is resampled every step.
        s_mis = self.severity["misfire"]
        self._misfire_now = bool(s_mis > 0.0 and self._rng.rand() < 0.25*s_mis)

        # Combustion instability raises cycle-to-cycle VARIANCE while leaving the
        # mean where it was - that is exactly what makes it hard to threshold and
        # worth a learned detector.
        s_ci = self.severity["combustion_instability"]
        if s_ci > 0.0:
            self._rpm_noise = float(self._rng.randn() * 60.0 * s_ci)
            self._egt_noise = float(self._rng.randn() * 25.0 * s_ci)
        else:
            self._rpm_noise = 0.0
            self._egt_noise = 0.0

    # ------------------------------------------------------------- accessors --
    def torque_multiplier(self):
        """Misfire drops individual firing cycles -> lost torque, RPM ripple."""
        if not self.enabled:
            return 1.0
        return 1.0 - (0.35*self.severity["misfire"] if self._misfire_now else 0.0)

    def fuel_multiplier(self):
        """A fouled injector delivers poorly: more fuel for the same power."""
        if not self.enabled:
            return 1.0
        return 1.0 + 0.25*self.severity["injector_fouling"]

    def egt_delta(self):
        """Fouling raises EGT; a misfiring cylinder dumps unburnt charge and cools it."""
        if not self.enabled:
            return 0.0
        d = 40.0*self.severity["injector_fouling"] + self._egt_noise
        if self._misfire_now:
            d -= 60.0*self.severity["misfire"]
        return d

    def cooling_multiplier(self):
        """Degraded cooling: fouled fins, weak pump, blocked ducts."""
        if not self.enabled:
            return 1.0
        return 1.0 - 0.60*self.severity["cooling_degradation"]

    def cht_tau_multiplier(self):
        """Degraded cooling also makes the head SLOWER to shed heat."""
        if not self.enabled:
            return 1.0
        return 1.0 + 2.0*self.severity["cooling_degradation"]

    def vib_multiplier(self):
        """Misfire and rough combustion both show up as vibration."""
        if not self.enabled:
            return 1.0
        return (1.0 + 0.8*self.severity["misfire"]
                + 0.5*self.severity["combustion_instability"])

    def rpm_noise(self):
        if not self.enabled:
            return 0.0
        return self._rpm_noise

    def injection_delta(self):
        """Fouling shifts effective injection phasing."""
        if not self.enabled:
            return 0.0
        return -3.0*self.severity["injector_fouling"]

    # ----------------------------------------------------------------- labels --
    def as_labels(self):
        """The `fm_*` columns written to the dataset and used as training targets."""
        return {f"fm_{m}": float(self.severity[m]) for m in MODES}

    def any_active(self):
        return any(self.severity[m] > 0.0 for m in MODES)

    def snapshot(self):
        return {"severity": dict(self.severity), "active": dict(self.active)}

    def restore(self, snap):
        """Resuming a flight must not reset latent degradation."""
        if not snap:
            return
        for m in MODES:
            self.severity[m] = float(snap.get("severity", {}).get(m, self.severity[m]))
            self.active[m] = bool(snap.get("active", {}).get(m, self.active[m]))
