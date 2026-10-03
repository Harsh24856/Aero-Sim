"""Component-level degradation driving the physics_v4 health modifiers.

This replaces v3's single `wear` scalar plus four hand-written failure modes.
The structure is taken from the C-MAPSS damage model, which was validated
against the released NASA data in simulation/cmapss/:

  - each affected component degrades on its own EXPONENTIAL trajectory,
    d(t) = 1 - exp{a . t^b}, with (a, b) drawn per unit,
  - components degrade at different rates and start at different initial wear,
  - failure is reaching an OPERABILITY LIMIT, not a counter expiring.

WHY THIS MATTERS FOR REAL-WORLD TRANSFER
    v3 defined a fault by the sensor pattern it should produce, so the AI could
    only ever learn the rule that was written. Here a fault is defined by the
    COMPONENT it damages; the sensor pattern is whatever the physics and the
    closed-loop ECU produce in response. A model trained on this learns the
    consequence of a physical failure, which is the thing that also happens on a
    real engine.

TWO KINDS OF FAULT, KEPT SEPARATE
    engine degradation  the engine really is worse; physics changes
    sensor fault        the engine is fine, the instrument is lying
    Conflating them is the classic way to build a model that cannot tell a real
    overheat from a failed thermocouple.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from physics_v4 import Health

# ---------------------------------------------------------------------------
# Fault taxonomy. Each entry names the component, the modifiers it drives, and
# how far it can go before that component is effectively destroyed.
#
# `sense` is +1 when degradation DECREASES the modifier (efficiency, flow) and
# -1 when it INCREASES it (friction, which rises with wear).
# ---------------------------------------------------------------------------
FAULT_MODES = {
    "air_filter_fouling":   {"mods": ["air_filter_mod"], "depth": 0.45, "sense": +1},
    "compression_loss":     {"mods": ["compression_mod"], "depth": 0.18, "sense": +1},
    "valve_leakage":        {"mods": ["valve_leak_mod"], "depth": 0.30, "sense": +1},
    "turbo_degradation":    {"mods": ["turbo_eff_mod", "turbo_flow_mod"], "depth": 0.30, "sense": +1},
    "wastegate_fault":      {"mods": ["wastegate_mod"], "depth": 0.55, "sense": +1},
    "intercooler_fouling":  {"mods": ["intercooler_mod"], "depth": 0.50, "sense": +1},
    "injector_fouling":     {"mods": ["injector_flow_mod"], "depth": 0.28, "sense": +1},
    "ignition_degradation": {"mods": ["ignition_mod"], "depth": 0.25, "sense": +1},
    "combustion_instability": {"mods": ["combustion_eff_mod"], "depth": 0.22, "sense": +1},
    "bearing_wear":         {"mods": ["friction_mod"], "depth": 0.60, "sense": -1},
    "oil_pump_degradation": {"mods": ["oil_pump_mod"], "depth": 0.40, "sense": +1},
    "oil_degradation":      {"mods": ["oil_quality_mod"], "depth": 0.45, "sense": +1},
    "cooling_degradation":  {"mods": ["cooling_mod"], "depth": 0.45, "sense": +1},
    "prop_erosion":         {"mods": ["prop_eff_mod"], "depth": 0.20, "sense": +1},
}
FAULT_NAMES = list(FAULT_MODES)
N_FAULTS = len(FAULT_NAMES)


def applicable_faults(turbocharged: bool, intercooled: bool) -> list:
    """Fault modes an engine can physically have, given its hardware."""
    skip = set()
    if not turbocharged:
        skip |= {"turbo_degradation", "wastegate_fault"}
    if not intercooled:
        skip.add("intercooler_fouling")
    return [n for n in FAULT_NAMES if n not in skip]

# Baseline wear-out affects every engine regardless of any specific fault: rings
# bed in and then wear, bearings lose clearance, the radiator silts up. This is
# what makes a healthy engine at 1800 h measurably different from one at 100 h,
# which is what the RUL head has to learn.
BASELINE_MODS = {
    "compression_mod": 0.07,
    "friction_mod": 0.22,
    "cooling_mod": 0.14,
    "oil_quality_mod": 0.16,
    "volumetric_eff_mod": 0.05,
    "prop_eff_mod": 0.05,
}

# Published C-MAPSS ranges, reused because they were validated there and the
# shape of a wear-out curve is not engine-specific.
A_RANGE = (0.0010, 0.0035)
B_RANGE = (1.30, 1.70)
BASE_SCALE_RANGE = (0.75, 1.30)


@dataclass
class FaultEvent:
    name: str
    onset_h: float          # engine hours at which it begins
    a: float
    b: float
    depth: float
    sense: int


class DegradationState:
    """One engine's degradation history: baseline wear plus any active faults."""

    def __init__(self, rng: np.random.Generator, start_hours: float,
                 tbo_hours: float, n_faults: int | None = None,
                 forced: int | None = None, allowed: list | None = None):
        self.rng = rng
        self.tbo = float(tbo_hours)
        self.start_hours = float(start_hours)

        # Baseline wear-out rate, per engine. Spread so two engines of the same
        # age are not identical - manufacturing and usage variation.
        self.base_a = float(rng.uniform(*A_RANGE))
        self.base_b = float(rng.uniform(*B_RANGE))
        self.base_scale = float(rng.uniform(*BASE_SCALE_RANGE))

        # Faults. Most flights carry none; some carry one; a few carry two, which
        # is what makes diagnosis non-trivial and is common in service.
        if n_faults is None:
            n_faults = int(rng.choice([0, 1, 1, 1, 2], p=[0.34, 0.22, 0.22, 0.14, 0.08]))
        self.faults: list[FaultEvent] = []

        # STRATIFIED, not uniform-random. Drawing faults at random left 10 of
        # the 14 modes with zero coverage in a 25-scenario pilot, and coverage
        # by luck does not become coverage at scale - it becomes a long tail of
        # rare classes the diagnosis head cannot learn. `forced` lets the
        # generator cycle deterministically through the taxonomy so every mode
        # receives a guaranteed share of scenarios.
        #
        # ONLY FAULTS THIS ENGINE CAN HAVE. `allowed` is the subset matching its
        # hardware (see applicable_faults). Cycling through all 14 gave the 912
        # turbocharger, wastegate and intercooler faults, and the 914 intercooler
        # faults - labels with no physical effect on engines without those parts,
        # so the diagnosis head was asked to learn classes that are pure noise.
        pool = list(allowed) if allowed else list(FAULT_NAMES)
        n_faults = min(n_faults, len(pool))
        if forced is not None and n_faults > 0:
            first = pool[forced % len(pool)]
            pool.remove(first)
            chosen = [first] + list(rng.choice(pool, size=n_faults - 1, replace=False))
        else:
            chosen = list(rng.choice(pool, size=n_faults, replace=False)) if n_faults else []

        for name in chosen:
            spec = FAULT_MODES[name]
            self.faults.append(FaultEvent(
                name=str(name),
                onset_h=float(rng.uniform(0.0, max(self.tbo * 0.9, 1.0))),
                a=float(rng.uniform(*A_RANGE)) * float(rng.uniform(2.0, 9.0)),
                b=float(rng.uniform(*B_RANGE)),
                depth=float(spec["depth"]) * float(rng.uniform(0.55, 1.0)),
                sense=int(spec["sense"])))

    # ------------------------------------------------------------------
    @staticmethod
    def _progress(hours: float, a: float, b: float) -> float:
        """Exponential damage fraction in [0, 1]. C-MAPSS eq. (5) form."""
        if hours <= 0.0:
            return 0.0
        x = min(a * (hours ** b), 50.0)
        return float(min(1.0, 1.0 - math.exp(-x)))

    def condition_at(self, hours: float) -> float:
        """Scalar CONDITION index in [0, 1]: 1 = as-new, 0 = worn out.

        WHY THIS IS SEPARATE FROM THE OPERABILITY MARGIN.
        physics_v4.margins() answers "how close is this engine to a certified
        limit right now", which depends on the operating point: a worn engine
        loafing at low power sits far from every limit and its margin reads 1.0.
        That is correct for airworthiness and useless as a degradation target -
        in the first pilot it left the health label saturated at 1.0 for the
        median row and produced zero run-to-failure trajectories.

        This index instead measures how worn the engine IS, independent of what
        it is doing, so it decreases monotonically over life and is learnable.
        Both are kept: this one supervises the health head, the margin stays a
        label and a feature because it is what actually grounds an aircraft.
        """
        frac = self._progress(hours / max(self.tbo, 1.0) * 100.0,
                              self.base_a, self.base_b) * self.base_scale
        worst = min(frac, 1.0)
        for f in self.faults:
            age = hours - f.onset_h
            if age > 0.0:
                worst = max(worst, self._progress(
                    age / max(self.tbo, 1.0) * 100.0, f.a, f.b))
        return float(min(max(1.0 - worst, 0.0), 1.0))

    def wear_out_hours(self, cap_mult: float = 20.0) -> float:
        """Engine age at which this engine's condition reaches zero.

        WHY THIS EXISTS. rul_hours_true was `tbo - engine_hours`: the calendar
        interval to scheduled overhaul, and nothing else. It carried no
        dependence on faults, wear or health, so two engines at the same age -
        one healthy, one with 60% turbo degradation - received the same label.
        Measured on the 914 test split, corr(engine_hours_norm, rul) was exactly
        -1.0000, which made the RUL head a subtraction of one of its own inputs
        rather than a prognostic.

        condition_at() is the max of the baseline wear curve and each fault's
        damage progression, all of them monotone increasing in age, so condition
        is monotone DECREASING and a bisection finds the crossing exactly.

        Returns inf when nothing reaches condition zero within cap_mult x TBO -
        which is the common case for an engine carrying no fault, since the
        baseline curve is scaled by base_scale and need not reach 1.0 at all.
        The caller is expected to take min(tbo, wear_out) so the label is
        "hours until this engine comes off wing", by whichever route arrives
        first.
        """
        lo = float(self.start_hours)
        if self.condition_at(lo) <= 0.0:
            return lo
        hi = max(lo, 1.0)
        cap = cap_mult * self.tbo
        while hi < cap:
            hi *= 2.0
            if self.condition_at(hi) <= 0.0:
                break
        else:
            return float("inf")
        if self.condition_at(hi) > 0.0:
            return float("inf")
        for _ in range(80):                      # ~1e-22 relative, far past need
            mid = 0.5 * (lo + hi)
            if self.condition_at(mid) <= 0.0:
                hi = mid
            else:
                lo = mid
        return float(hi)

    def health_at(self, hours: float) -> tuple[Health, dict]:
        """Health modifiers and per-fault severities at a given engine age."""
        h = Health()
        # -- baseline wear-out ------------------------------------------------
        frac = self._progress(hours / max(self.tbo, 1.0) * 100.0,
                              self.base_a, self.base_b) * self.base_scale
        frac = min(frac, 1.0)
        for mod, depth in BASELINE_MODS.items():
            cur = getattr(h, mod)
            if mod == "friction_mod":
                setattr(h, mod, cur + depth * frac)
            else:
                setattr(h, mod, cur - depth * frac)

        # -- specific faults ---------------------------------------------------
        sev = {n: 0.0 for n in FAULT_NAMES}
        for f in self.faults:
            age = hours - f.onset_h
            if age <= 0.0:
                continue
            s = self._progress(age / max(self.tbo, 1.0) * 100.0, f.a, f.b)
            sev[f.name] = max(sev[f.name], s)
            for mod in FAULT_MODES[f.name]["mods"]:
                cur = getattr(h, mod)
                delta = f.depth * s
                setattr(h, mod, cur + delta if f.sense < 0 else cur - delta)

        # Clamp to physically possible territory.
        for k, v in h.as_dict().items():
            setattr(h, k, float(min(max(v, 0.05), 2.5)))
        return h, sev
