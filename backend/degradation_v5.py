"""Degradation for physics v5: degradation_v4 with labels a model can learn.

Changes (docs/v5_model_improvement_plan.md, Phase 2)
----------------------------------------------------
1. RUL WITHOUT FUTURE FAULTS. v4's RUL label came from `wear_out_hours()`, which
   includes faults whose onset is still in the future. An engine that is healthy
   now - every sensor reading normal - was labelled with a short remaining life
   caused by a fault that had not started and could not be seen. No model can
   learn that; it set an error floor (v4 model cards: 18.6-21% of TBO on live
   inputs). v5 labels RUL from what exists at that hour: baseline wear plus the
   faults already under way, projected along their known progression law:
       rul_hours = min(TBO, wear_out_known(hours)) - hours
   The v4 label is kept as `rul_hours_oracle` for reference.
2. ONSET HAZARD. v4 drew every fault's onset uniformly over 0-0.9 TBO. Wear-out
   faults (bearings, rings and valves, oil pump, turbocharger, radiator silting,
   propeller erosion) are more likely late in life; v5 draws their onset with a
   linearly rising hazard (density proportional to age, Beta(2,1) over 0-0.9 TBO).
   Random faults (ignition, injector, combustion, filter, oil condition, wastegate
   and intercooler) keep a uniform onset: they depend on maintenance and chance,
   not on age.
3. REMOVAL, NOT DESTRUCTION. v4 wore an engine out only when its condition reached
   exactly 0 - a component at 100% damage. Faults saturate towards 1.0 and almost
   never got there before TBO, so in 600 pilot flights no fault ever shortened an
   engine's remaining life: RUL depended on baseline wear alone. An engine comes off
   the wing long before total destruction; v5 removes it when any FAULTED component
   reaches REMOVAL_SEVERITY (0.8 damage - an ASSUMED maintenance threshold), when
   baseline wear is used up (as v4), or at TBO - whichever is first. Faults now
   shorten life, which is what makes the no-future-faults label (1) matter. (The
   threshold applies to faults only: applied to baseline wear too, 57% of pilot
   rows became wear-limited, while most real Rotax engines reach their TBO.)
4. NORMAL AGEING WITHIN LIMITS. v4's baseline wear gave an engine at TBO up to 29%
   more friction, 18% less cooling and 21% worse oil (depths x the 1.3 worst-case
   scale): a normally aged engine then broke the 2 bar oil-pressure minimum at an
   ordinary standard-day cruise and ran its oil past 130 C in the certified hot-day
   climb. Rotax sets TBO with margin - a normally worn engine at TBO is inside its
   limits - and oil is changed every ~100 h, so it cannot degrade over the engine's
   life. v5 baseline depths: friction 0.15, cooling 0.10, oil condition 0.06
   (compression, breathing and propeller unchanged).
5. WEAR-ONLY HEALTH. `health_wear_only(hours)` is the same engine with its baseline
   wear but none of its faults. The generator flies it alongside, so a fault is
   labelled PRESENT only when its physical effect on the instruments is visible
   against normal wear - "wear alone is never a fault".
"""
from __future__ import annotations

import math

import numpy as np

from degradation_v4 import (  # noqa: F401  re-exported
    A_RANGE, B_RANGE, BASE_SCALE_RANGE, BASELINE_MODS, FAULT_MODES, FAULT_NAMES, N_FAULTS,
    DegradationState, FaultEvent, applicable_faults,
)
from physics_v4 import Health

REMOVAL_SEVERITY = 0.8
REMOVAL_CONDITION = 1.0 - REMOVAL_SEVERITY      # engines already past it do not fly
BASELINE_MODS_V5 = {
    "compression_mod": 0.07,
    "friction_mod": 0.15,
    "cooling_mod": 0.10,
    "oil_quality_mod": 0.06,
    "volumetric_eff_mod": 0.05,
    "prop_eff_mod": 0.05,
}



def wear_fraction(hours: float, tbo_hours: float, a: float, b: float, scale: float) -> float:
    """How far baseline wear has gone (0 new .. 1 worn out) on the curve (a, b, scale)."""
    return min(DegradationState._progress(hours / max(tbo_hours, 1.0) * 100.0, a, b) * scale, 1.0)


def health_from_wear(frac: float) -> Health:
    """Health with BASELINE_MODS_V5 applied at wear fraction `frac`."""
    h = Health()
    for mod, depth in BASELINE_MODS_V5.items():
        cur = getattr(h, mod)
        setattr(h, mod, cur + depth * frac if mod == "friction_mod" else cur - depth * frac)
    for k, v in h.as_dict().items():
        setattr(h, k, float(min(max(v, 0.05), 2.5)))
    return h


def baseline_health(hours: float, tbo_hours: float, a: float, b: float, scale: float) -> Health:
    """Health after baseline wear alone, for a wear curve (a, b, scale)."""
    return health_from_wear(wear_fraction(hours, tbo_hours, a, b, scale))


# The middle of every per-engine wear parameter range.
# ponytail: the curve at the mean parameters, not the mean of the curves; close
# enough for a twin reference (tests: healthy p95 under 8 sigma, 2.5x below a new-engine twin).
FLEET_WEAR = (sum(A_RANGE) / 2, sum(B_RANGE) / 2, sum(BASE_SCALE_RANGE) / 2)


def fleet_wear_health(hours: float, tbo_hours: float) -> Health:
    """Fleet-average baseline wear at these engine hours: all an on-board twin can
    know about normal ageing (the hour meter and a fleet wear curve), never this
    engine's own wear rate. Residuals against it keep how far this engine has aged
    away from the fleet and drop the wear clock everyone shares."""
    return baseline_health(hours, tbo_hours, *FLEET_WEAR)


def calibrated_wear_health(hours: float, tbo_hours: float, ratio: float) -> Health:
    """The fleet wear curve scaled to THIS engine: `ratio` (from engine_history) is how
    far it had worn compared with the fleet when its logbook baseline was taken."""
    return health_from_wear(min(max(wear_fraction(hours, tbo_hours, *FLEET_WEAR) * ratio, 0.0), 1.0))


HIST_LAG_H = (20.0, 200.0)        # the logbook baseline is from a flight this many hours ago
HIST_NOISE = 0.10                 # relative error of that wear estimate
HIST_CONTAMINATION = 0.5          # a fault active then reads as this much extra wear per unit severity
HIST_RATIO = (0.5, 2.0)
HIST_YOUNG = 0.02                 # below this fleet wear the ratio carries no information
HIST_SEV_FLOOR = 0.08             # a past fault weaker than this was not visible (FAULT_PRESENT_SEV)


def engine_history(deg: "DegradationStateV5", hours: float, rng: np.random.Generator,
                   lag_h: float | None = None) -> dict:
    """A logbook-style baseline for an engine at `hours`: its wear compared with the
    fleet, and its worst fault severity, as measured on a flight `lag_h` hours ago
    (default: drawn from HIST_LAG_H; never before the engine's hour zero). The wear
    estimate is noisy and a fault already active then inflates it - what calibration
    from real past flights would give. The past severity is what that flight could see:
    EFFECTIVE units (progress x depth / spec depth - the units the severity head
    reports), zero below HIST_SEV_FLOOR, with the same relative noise. Never reads the
    current flight.
    ponytail: HIST_NOISE and HIST_CONTAMINATION are modelling knobs; tune them if live
    calibration data ever exists."""
    lag = float(rng.uniform(*HIST_LAG_H)) if lag_h is None else float(lag_h)
    lag = min(lag, float(hours))
    h0 = float(hours) - lag
    seen = [DegradationState._progress((h0 - f.onset_h) / max(deg.tbo, 1.0) * 100.0, f.a, f.b)
            * f.depth / FAULT_MODES[f.name]["depth"] for f in deg.faults if f.onset_h < h0]
    true_sev = max(seen, default=0.0)
    fleet = wear_fraction(h0, deg.tbo, *FLEET_WEAR)
    noise = float(rng.normal(0.0, HIST_NOISE))
    sev_noise = float(rng.normal(0.0, HIST_NOISE))      # drawn last: earlier draws unchanged
    sev_max = 0.0 if true_sev < HIST_SEV_FLOOR else min(max(true_sev * (1.0 + sev_noise), 0.0), 1.0)
    if fleet < HIST_YOUNG:
        ratio = 1.0
    else:
        observed = wear_fraction(h0, deg.tbo, deg.base_a, deg.base_b, deg.base_scale) * (1.0 + noise) \
            + HIST_CONTAMINATION * sev_max
        ratio = min(max(observed / fleet, HIST_RATIO[0]), HIST_RATIO[1])
    return {"lag_h": lag, "wear_ratio": float(ratio), "sev_max": float(sev_max)}


WEAR_OUT_FAULTS = {"bearing_wear", "compression_loss", "valve_leakage",
                   "oil_pump_degradation", "turbo_degradation",
                   "cooling_degradation", "prop_erosion"}
ONSET_SPAN = 0.9                 # onset falls in 0 .. 0.9 x TBO, as in v4


class DegradationStateV5(DegradationState):

    def __init__(self, rng: np.random.Generator, start_hours: float, tbo_hours: float,
                 n_faults: int | None = None, forced: int | None = None,
                 allowed: list | None = None):
        super().__init__(rng, start_hours, tbo_hours, n_faults=n_faults,
                         forced=forced, allowed=allowed)
        span = max(self.tbo * ONSET_SPAN, 1.0)
        for f in self.faults:
            if f.name in WEAR_OUT_FAULTS:
                # Beta(2,1): density 2x on [0,1], inverse CDF sqrt(u).
                f.onset_h = float(span * math.sqrt(rng.random()))

    # ------------------------------------------------------------------
    def _known(self, hours: float) -> list:
        return [f for f in self.faults if f.onset_h <= hours]

    def condition_known(self, at_hours: float, known: list) -> float:
        """Condition at `at_hours` counting only the faults in `known`."""
        frac = self._progress(at_hours / max(self.tbo, 1.0) * 100.0,
                              self.base_a, self.base_b) * self.base_scale
        worst = min(frac, 1.0)
        for f in known:
            age = at_hours - f.onset_h
            if age > 0.0:
                worst = max(worst, self._progress(age / max(self.tbo, 1.0) * 100.0, f.a, f.b))
        return float(min(max(1.0 - worst, 0.0), 1.0))

    def _removed(self, age: float, known: list) -> bool:
        """Due for removal at this age: baseline wear used up, or a faulted
        component at REMOVAL_SEVERITY. Monotone in age (both only grow)."""
        base = self._progress(age / max(self.tbo, 1.0) * 100.0, self.base_a, self.base_b) * self.base_scale
        if base >= 1.0:
            return True
        for f in known:
            a = age - f.onset_h
            if a > 0.0 and self._progress(a / max(self.tbo, 1.0) * 100.0, f.a, f.b) >= REMOVAL_SEVERITY:
                return True
        return False

    def _removal_age(self, start: float, known: list, cap_mult: float = 20.0) -> float:
        """First age >= start at which the engine is due for removal (bisection)."""
        cond = lambda a: self._removed(a, known)  # noqa: E731
        lo = float(start)
        if cond(lo):
            return lo
        hi = max(lo, 1.0)
        cap = cap_mult * self.tbo
        while hi < cap:
            hi *= 2.0
            if cond(hi):
                break
        else:
            return float("inf")
        if not cond(hi):
            return float("inf")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if cond(mid):
                hi = mid
            else:
                lo = mid
        return float(hi)

    def wear_out_known(self, hours: float) -> float:
        """Age at which the engine is due for removal if nothing new goes wrong
        after `hours`: baseline wear plus the faults already under way."""
        return self._removal_age(hours, self._known(hours))

    def wear_out_hours(self, cap_mult: float = 20.0) -> float:
        """Oracle: removal age counting every fault, including ones not yet started."""
        return self._removal_age(self.start_hours, list(self.faults), cap_mult)

    def health_at(self, hours: float) -> tuple[Health, dict]:
        """degradation_v4.health_at with the v5 baseline-wear depths."""
        h = self.health_wear_only(hours)
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
        for k, v in h.as_dict().items():
            setattr(h, k, float(min(max(v, 0.05), 2.5)))
        return h, sev

    def n_known(self, hours: float) -> int:
        return sum(1 for f in self.faults if f.onset_h <= hours)

    def health_wear_only(self, hours: float) -> Health:
        """Baseline wear only - the same engine without any of its faults."""
        return baseline_health(hours, self.tbo, self.base_a, self.base_b, self.base_scale)
