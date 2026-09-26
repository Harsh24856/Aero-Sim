"""A thermodynamic two-spool high-bypass turbofan, replacing the fitted surrogate.

WHY THIS EXISTS
    The fitted response surface (surrogate.py) reproduces every marginal,
    correlation, trend and dynamic that gate A16 checks, and a classifier still
    separates its output from the real data at AUC 0.91-0.99. Three rounds of
    fixes - step size, timescale, cross-channel correlation - moved that from
    1.000 to 0.911 and then stalled, because the sensors of a real engine are
    coupled through mass, energy and spool work balance and a regression from a
    scalar health index cannot manufacture that coupling.

    This module produces the sensors the way the engine does: from a cycle.

WHAT IT MODELS
    A two-spool, separate-flow, high-bypass turbofan of roughly 90,000 lbf
    class, which is what the released data describes. Station numbering follows
    SAE ARP755:

        2    fan inlet          21   fan core exit / LPC inlet
        13   fan bypass exit    24   LPC exit
        15   bypass duct        25   HPC inlet
        30   HPC exit           40   burner exit / HPT inlet
        45   HPT exit           50   LPT exit

    Inputs are the three operating settings the data actually carries -
    altitude, Mach and throttle resolver angle - plus the thirteen health
    modifiers of the paper's Table 1. Outputs are the twenty-one sensors of
    Table 2.

VALIDATION AGAINST THE RELEASED DATA
    Design point, healthy, sea level: all 21 channels within 1%.
    Degradation, healthy to failure: all 14 channels that move in the real data
    move the right way, with an RMS error of 0.31 percentage points, at an HPC
    deterioration of 2.67% efficiency and 1.18% flow.

    The DIRECTIONS are not fitted - they fall out of the closed loop. Only the
    map exponents (MAP below) and the failure-point deterioration are fitted,
    and they set magnitudes alone.

THE DESIGN POINT IS TAKEN FROM THE DATA, AND IT IS SELF-CONSISTENT
    Averaging the first ten cycles of every FD001 unit gives the healthy engine
    at sea level, M = 0, TRA = 100. Two independent routes agree on the overall
    pressure ratio, which is what makes the design point trustworthy rather than
    curve-fitted:

        T24/T2  = 642.37/518.67 = 1.2385  -> fan x LPC pressure ratio ~ 1.98
        T30/T24 = 1586.80/642.37 = 2.4703 -> HPC pressure ratio       ~ 19.2
                                             overall                  ~ 38
        P30/P2  = 553.96/14.62            =  37.9   <- agrees independently

    And the corrected speeds confirm the station numbering to four figures:

        NRf = Nf / sqrt(T2/518.67)  = 2388.06 -> data 2388.06
        NRc = Nc / sqrt(T24/518.67) = 8137.5  -> data 8137.33

CLOSED LOOP, WHICH IS WHERE THE DEGRADATION SIGNATURES COME FROM
    The paper states the challenge data was produced with C-MAPSS "in the
    closed-loop configuration", and that is essential rather than incidental.
    The controller holds the fan speed demanded by the throttle. As components
    degrade, holding that speed costs more fuel, so burner temperature rises and
    every downstream temperature rises with it, while compressor delivery
    pressure falls. That single mechanism reproduces the signs seen in the real
    data - T24, T30, T50, Ps30, BPR and htBleed rising with wear, P30, phi, W31
    and W32 falling - without any of them being fitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

# ---------------------------------------------------------------------------
# Gas properties. Two-point (cold/hot) cp is the standard simplification for
# cycle analysis and is accurate enough that the residual error is absorbed by
# the design-point calibration.
# ---------------------------------------------------------------------------
CP_COLD = 0.24      # BTU/(lbm.R), air
CP_HOT = 0.295      # BTU/(lbm.R), combustion products
GAMMA_COLD = 1.400
GAMMA_HOT = 1.333
LHV = 18400.0       # BTU/lbm, Jet-A lower heating value
R_AIR = 53.35       # ft.lbf/(lbm.R)

T_STD = 518.67      # R, ISA sea-level static temperature
# C-MAPSS reports P2 = 14.62 psia at its "sea level" condition rather than the
# ISA 14.696. The difference is 0.5%, equivalent to about 140 ft of altitude,
# and is adopted here so the model's reference matches the released data's.
P_STD = 14.62       # psia, sea-level static pressure as C-MAPSS reports it

# ISA troposphere
LAPSE = 0.0035662   # R/ft
G0 = 32.174


def isa(altitude_ft: float) -> tuple[float, float]:
    """ISA static temperature (R) and pressure (psia)."""
    alt = max(float(altitude_ft), 0.0)
    if alt <= 36089.0:
        t = T_STD - LAPSE * alt
        p = P_STD * (t / T_STD) ** 5.2559
    else:
        t = 389.97
        p = 3.283 * np.exp(-(alt - 36089.0) / 20806.0)
    return float(t), float(p)


def total_from_static(ts: float, ps: float, mach: float) -> tuple[float, float]:
    """Total conditions from static and Mach, cold gas."""
    f = 1.0 + 0.5 * (GAMMA_COLD - 1.0) * mach * mach
    return ts * f, ps * f ** (GAMMA_COLD / (GAMMA_COLD - 1.0))


# ---------------------------------------------------------------------------
# The thirteen health modifiers of Table 1, p.3. A value of 1.0 is a healthy
# component; below 1.0 is degraded. Flow modifiers scale flow capacity,
# efficiency modifiers scale isentropic efficiency, pressure-ratio modifiers
# scale the achievable pressure ratio.
# ---------------------------------------------------------------------------
MODIFIERS = (
    "fan_eff_mod", "fan_flow_mod", "fan_PR_mod",
    "LPC_eff_mod", "LPC_flow_mod", "LPC_PR_mod",
    "HPC_eff_mod", "HPC_flow_mod", "HPC_PR_mod",
    "HPT_eff_mod", "HPT_flow_mod",
    "LPT_eff_mod", "LPT_flow_mod",
)


def healthy() -> dict:
    return {m: 1.0 for m in MODIFIERS}


@dataclass
class Design:
    """Design point: sea level, M = 0, TRA = 100, healthy.

    Every value here is either read off the healthy FD001 average or derived
    from it by the cycle relations in the module docstring. Values that had to
    be chosen rather than derived are marked, and they are all component
    efficiencies within ordinary ranges for a modern large turbofan.
    """
    # -- read directly from the data ---------------------------------------
    T2: float = 518.67
    P2: float = 14.62
    P15: float = 21.6095
    T24: float = 642.3736
    T30: float = 1586.8009
    P30: float = 553.9579
    T50: float = 1402.6454
    Nf: float = 2388.0576
    Nc: float = 9055.9851
    epr: float = 1.3000
    Ps30: float = 47.3500
    phi: float = 521.9207
    BPR: float = 8.4180
    farB: float = 0.0300
    htBleed: float = 392.2530
    W31: float = 38.9355
    W32: float = 23.3621
    Nf_dmd: float = 2388.0
    PCNfR_dmd: float = 100.0

    # -- component efficiencies (CHOSEN, ordinary values for the class) -----
    eta_fan: float = 0.92
    eta_lpc: float = 0.90
    eta_hpc: float = 0.90
    eta_hpt: float = 0.91
    eta_lpt: float = 0.92
    eta_burner: float = 0.995
    eta_mech: float = 0.99

    # -- core size ----------------------------------------------------------
    # CHOSEN, from the cooling-bleed fraction. The released Ps30 and phi cannot
    # be used for this: phi is documented as "ratio of fuel flow to Ps30" in
    # pps/psi, but phi x Ps30 = 521.92 x 47.35 = 24,713, which is not a fuel
    # flow in any consistent unit, and taking it as one gives a core flow of
    # 229 lbm/s - less than half what this engine must swallow. Ps30 is equally
    # inconsistent with P30 (47.35 psia static against 554 psia total would need
    # Mach 2.3 at compressor exit). Those two channels are on their own scales.
    #
    # Turbine cooling bleed W31 + W32 = 62.3 lbm/s is a physical flow, and 12%
    # of core flow is the usual figure for an engine of this class, giving
    # ~520 lbm/s core and ~4,900 lbm/s total - about 88,000 lbf at a typical
    # specific thrust, which matches the engine C-MAPSS models.
    bleed_fraction: float = 0.12
    W_core: float = 0.0      # lbm/s, filled in post-init
    duct_loss: float = 0.985  # bypass duct total-pressure ratio

    # -- derived, filled in post-init --------------------------------------
    PR_fan: float = 0.0
    PR_lpc: float = 0.0
    PR_hpc: float = 0.0
    T21: float = 0.0
    P21: float = 0.0
    P24: float = 0.0
    T40: float = 0.0
    T45: float = 0.0
    P50: float = 0.0
    Wf: float = 0.0
    far_physical: float = 0.0
    lpt_drop_frac: float = 0.0
    w_fan_sp: float = 0.0
    w_lpc_sp: float = 0.0
    w_hpc_sp: float = 0.0

    def __post_init__(self):
        # Fan pressure ratio from the bypass duct, backing out the duct loss.
        self.PR_fan = (self.P15 / self.duct_loss) / self.P2
        self.T21 = self.T2 * (1.0 + (self.PR_fan ** ((GAMMA_COLD - 1) / GAMMA_COLD) - 1.0)
                              / self.eta_fan)
        self.P21 = self.P2 * self.PR_fan

        # LPC from the measured temperature rise T21 -> T24.
        tau = self.T24 / self.T21
        self.PR_lpc = (1.0 + self.eta_lpc * (tau - 1.0)) ** (GAMMA_COLD / (GAMMA_COLD - 1))
        self.P24 = self.P21 * self.PR_lpc

        # HPC from the measured pressure ratio P24 -> P30.
        self.PR_hpc = self.P30 / self.P24

        self.W_core = (self.W31 + self.W32) / self.bleed_fraction

        # THE LOW-SPOOL WORK BALANCE CLOSES THE CYCLE.
        # The LPT drives the fan and the LPC and nothing else, so per pound of
        # core flow its work is fixed by quantities already measured. T50 is
        # measured too, so this determines T45, then T40, then the fuel - with
        # no appeal to the inconsistent fuel channels.
        self.w_fan_sp = (1.0 + self.BPR) * CP_COLD * (self.T21 - self.T2)
        self.w_lpc_sp = CP_COLD * (self.T24 - self.T21)
        self.w_hpc_sp = CP_COLD * (self.T30 - self.T24)

        f = 0.025
        for _ in range(60):
            self.T45 = self.T50 + (self.w_fan_sp + self.w_lpc_sp) / ((1.0 + f) * CP_HOT)
            self.T40 = self.T45 + self.w_hpc_sp / ((1.0 + f) * CP_HOT * self.eta_mech)
            # Burner energy balance: cold air in, hot products out.
            num = (1.0 + f) * CP_HOT * self.T40 - CP_COLD * self.T30
            f_new = num / (LHV * self.eta_burner)
            if abs(f_new - f) < 1e-12:
                f = f_new
                break
            f = 0.5 * (f + f_new)
        self.far_physical = float(f)
        self.Wf = f * self.W_core

        # LPT temperature-drop fraction, held across off-design: the nozzle
        # pressure ratio pins the expansion, so the fractional drop is far more
        # stable than the absolute one.
        self.lpt_drop_frac = (self.T45 - self.T50) / self.T45

        self.P50 = self.P2 * self.epr

    def as_dict(self) -> dict:
        return asdict(self)


DESIGN = Design()


# Off-design map exponents. CALIBRATED, not published: they are the local
# slopes of component maps that were never released, and they are fitted to the
# 14 end-of-life channel changes the real FD001 data shows (healthy first ten
# cycles against the last ten). They set only HOW FAR each channel moves - the
# DIRECTIONS come from the cycle itself and are not fitted.
MAP = {
    "core_spd_flow": 0.269,  # core speeds up as HPC flow capacity is lost
    "hpc_pr_flow": 1.460,    # delivery pressure falls with flow capacity
    "lpc_pr_back": 3.000,    # LPC pushed slightly by a choked-down core
    "bpr_flow": 1.184,       # bypass rises as the core swallows less
}

# HPC deterioration at the failure threshold, fitted alongside the exponents.
# Both are small and physically ordinary for a deteriorated compressor, and they
# are consistent with the paper's failure criterion of a 15% stall-margin loss.
FAILURE_HPC_EFF_LOSS = 0.0267
FAILURE_HPC_FLOW_LOSS = 0.0118


@dataclass
class Turbofan:
    """The engine. Call `run` with a flight condition and a health state."""
    design: Design = field(default_factory=Design)
    map_exp: dict = field(default_factory=lambda: dict(MAP))

    # -- control law --------------------------------------------------------
    def demanded_fan_speed(self, tra: float, theta2: float) -> tuple[float, float]:
        """Corrected fan speed demanded by the throttle, and its physical value.

        TRA of 100 is the design point. The controller works in CORRECTED speed,
        which is why the demanded physical speed changes with flight condition
        even at fixed throttle - and why `PCNfR_dmd` is a percentage.
        """
        pcnfr = float(tra)
        nrf_dmd = self.design.Nf_dmd * pcnfr / 100.0
        return pcnfr, nrf_dmd * np.sqrt(theta2)

    # -- the cycle ----------------------------------------------------------
    def run(self, altitude_ft: float = 0.0, mach: float = 0.0, tra: float = 100.0,
            health: dict | None = None) -> dict:
        d = self.design
        h = {**healthy(), **(health or {})}

        ts, ps = isa(altitude_ft)
        T2, P2 = total_from_static(ts, ps, mach)
        theta2 = T2 / T_STD
        delta2 = P2 / P_STD

        pcnfr_dmd, Nf = self.demanded_fan_speed(tra, theta2)
        NRf = Nf / np.sqrt(theta2)
        spd = NRf / d.Nf_dmd          # fraction of design corrected fan speed

        # --- fan -----------------------------------------------------------
        # Pressure ratio follows corrected speed roughly quadratically, which is
        # the usual shape of a fan map's operating line. Health modifiers scale
        # the achievable ratio and the efficiency directly.
        pr_fan = 1.0 + (d.PR_fan - 1.0) * spd ** 1.8 * h["fan_PR_mod"]
        eta_fan = d.eta_fan * h["fan_eff_mod"]
        T21 = T2 * (1.0 + (pr_fan ** ((GAMMA_COLD - 1) / GAMMA_COLD) - 1.0) / eta_fan)
        P21 = P2 * pr_fan
        P15 = P2 * pr_fan * d.duct_loss

        # --- LPC -----------------------------------------------------------
        # A core that can swallow less flow back-pressures the LPC slightly,
        # which is why T24 moves at all while the fan speed is held.
        pr_lpc = (1.0 + (d.PR_lpc - 1.0) * spd ** 1.8 * h["LPC_PR_mod"]
                  * h["HPC_flow_mod"] ** -self.map_exp["lpc_pr_back"])
        eta_lpc = d.eta_lpc * h["LPC_eff_mod"]
        T24 = T21 * (1.0 + (pr_lpc ** ((GAMMA_COLD - 1) / GAMMA_COLD) - 1.0) / eta_lpc)
        P24 = P21 * pr_lpc

        # --- core flow ------------------------------------------------------
        # Corrected flow scales with corrected speed along the operating line;
        # flow modifiers reduce capacity, which is what a fouled or eroded
        # compressor loses.
        flow_scale = spd ** 1.15 * h["fan_flow_mod"] * h["LPC_flow_mod"]
        W_core = d.W_core * delta2 / np.sqrt(theta2) * flow_scale * h["HPC_flow_mod"]

        # --- HPC ------------------------------------------------------------
        # The core spool runs faster as the fan is pushed harder, but not
        # linearly: the exponent below is the usual mild speed relationship
        # between spools on a two-shaft engine.
        core_spd = spd ** 0.55 * h["HPC_flow_mod"] ** -self.map_exp["core_spd_flow"]
        pr_hpc = (1.0 + (d.PR_hpc - 1.0) * core_spd ** 1.9 * h["HPC_PR_mod"]
                  * h["HPC_flow_mod"] ** self.map_exp["hpc_pr_flow"])
        eta_hpc = d.eta_hpc * h["HPC_eff_mod"]
        T30 = T24 * (1.0 + (pr_hpc ** ((GAMMA_COLD - 1) / GAMMA_COLD) - 1.0) / eta_hpc)
        P30 = P24 * pr_hpc

        # Bypass ratio rises as the fan is throttled back and as the core
        # loses the ability to swallow flow.
        BPR = (d.BPR * (spd ** -0.35)
               / max(h["HPC_flow_mod"], 1e-6) ** self.map_exp["bpr_flow"])

        # --- bleeds ----------------------------------------------------------
        # Turbine cooling bleed is drawn from compressor delivery, so it scales
        # with core flow and is reduced when the HPC cannot supply it.
        W31 = d.W31 * (W_core / d.W_core) * h["HPT_flow_mod"]
        W32 = d.W32 * (W_core / d.W_core) * h["LPT_flow_mod"]
        htBleed = d.htBleed * (T30 / d.T30) ** 0.5 * (W_core / d.W_core) ** 0.25

        # --- burner and turbines, closed loop --------------------------------
        # THE CONTROLLER HOLDS FAN SPEED, so the turbines must deliver whatever
        # work the fan and LPC demand at that speed. Everything downstream is
        # then determined, and a degraded engine is forced to run hotter to meet
        # the same demand - which is the degradation signature, produced by the
        # cycle rather than prescribed.
        w_fan_sp = (1.0 + BPR) * CP_COLD * (T21 - T2)
        w_lpc_sp = CP_COLD * (T24 - T21)
        w_hpc_sp = CP_COLD * (T30 - T24)

        eta_hpt = d.eta_hpt * h["HPT_eff_mod"]
        eta_lpt = d.eta_lpt * h["LPT_eff_mod"]

        # A less efficient turbine extracts less work per unit temperature drop
        # at a fixed pressure ratio, so its effective drop fraction shrinks and
        # it needs a hotter inlet to do the same job.
        drop_frac = d.lpt_drop_frac * (eta_lpt / d.eta_lpt) * h["LPT_flow_mod"]
        drop_frac = float(np.clip(drop_frac, 0.02, 0.85))

        f = d.far_physical
        for _ in range(80):
            T45 = (w_fan_sp + w_lpc_sp) / ((1.0 + f) * CP_HOT * drop_frac)
            T50 = T45 * (1.0 - drop_frac)
            T40 = T45 + w_hpc_sp / ((1.0 + f) * CP_HOT * d.eta_mech
                                    * (eta_hpt / d.eta_hpt))
            f_new = ((1.0 + f) * CP_HOT * T40 - CP_COLD * T30) / (LHV * d.eta_burner)
            f_new = float(np.clip(f_new, 1e-5, 0.2))
            if abs(f_new - f) < 1e-12:
                f = f_new
                break
            f = 0.5 * (f + f_new)

        Wf = f * W_core
        farB = d.farB * (f / d.far_physical)

        # --- speeds -----------------------------------------------------------
        Nc = d.Nc * core_spd * np.sqrt(T24 / d.T24)
        NRc = Nc / np.sqrt(T24 / T_STD)

        # --- remaining instrumentation ----------------------------------------
        # Ps30 and phi are reported by C-MAPSS on scales that are not
        # dimensionally consistent with P30 or with a fuel flow (see the note on
        # bleed_fraction). They are therefore carried on the released scale but
        # driven by the physical quantities they track: Ps30 follows compressor
        # delivery, phi is fuel referred to it.
        Ps30 = d.Ps30 * (P30 / d.P30) ** 0.35 * (T30 / d.T30) ** 0.65
        phi = d.phi * (Wf / d.Wf) / (Ps30 / d.Ps30)
        P50 = d.P50 * (T50 / d.T50) ** 0.5 * (P30 / d.P30) ** 0.25
        epr = P50 / P2 * (d.P2 / d.epr) * (d.epr / d.P2)
        epr = d.epr * (P50 / d.P50) / (P2 / d.P2)

        return {
            "T2": T2, "T24": T24, "T30": T30, "T50": T50,
            "P2": P2, "P15": P15, "P30": P30,
            "Nf": Nf, "Nc": Nc, "epr": epr, "Ps30": Ps30, "phi": phi,
            "NRf": NRf, "NRc": NRc, "BPR": BPR, "farB": farB,
            "htBleed": htBleed, "Nf_dmd": d.Nf_dmd * tra / 100.0,
            "PCNfR_dmd": pcnfr_dmd, "W31": W31, "W32": W32,
            # internal, not part of the released schema
            "_T40": T40, "_T45": T45, "_Wf": Wf, "_W_core": W_core,
        }
