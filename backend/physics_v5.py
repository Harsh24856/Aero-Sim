"""Physics v5: physics_v4 with the fault signatures a real Rotax actually has.

v4 built several faults on the same equation, so no model could tell them apart
from any sensor - not a model weakness, a physics one (docs/v5_model_improvement_plan.md,
Phase 1). v5 changes only those equations. A HEALTHY engine is identical to v4 by
construction: every change below is a function of a health modifier and vanishes
at the as-new value. The published Rotax figures v4 was calibrated to (rated power,
idle speed, cruise temperatures, oil pressure band, critical altitude) therefore
still hold, and tests/test_physics_v5.py checks both.

What changed, and why
---------------------
1. IGNITION vs COMBUSTION (v4: both multiplied `eta_comb`, so identical).
   - A degraded ignition circuit (coil, lead, plug) is a weak or late spark on one
     of the two circuits: the burn starts late. Modelled as an effective spark
     retard, which the Rotax ignition check measures directly - running on one
     circuit at 4,000 rpm drops 50-100 rpm on a healthy engine and at most 300 rpm
     (max 120 rpm difference between circuits; Rotax 912 operating procedure, as
     reported by operators, rotax-owner.com). Late burning moves heat from the
     cylinder walls into the exhaust: EGT up, CHT down.
   - Combustion inefficiency (incomplete burn) keeps v4's form: less heat released
     from the same fuel, so power falls, fuel per unit power rises, EGT falls.
2. OIL FAMILY (v4: pump wear and bearing wear entered one `clearance` term).
   - Pump wear: a worn gear pump leaks internally, a loss that is fixed per
     revolution, so the delivered-flow deficit is largest at low rpm and small near
     rated speed.
   - Bearing clearance: flow escapes through opened clearances, worse through hot
     thin oil and at high speed; the extra friction also raises oil temperature and
     fuel per unit power (already in v4's friction path).
   - Oil degradation: loss of viscosity index (shear, fuel dilution) - pressure
     falls more the hotter the oil, plus v4's bounded friction penalty (small oil
     temperature rise).
   All three stay inside the Rotax band: 0.8 bar floor below 3,500 rpm, 2-5 bar
   normal, 7 bar cold maximum (the relief-valve clamp is unchanged).
3. TURBO vs WASTEGATE. No equation change is needed: v4 already makes compressor
   wear heat the charge (`turbo_eff_mod` enters the compressor temperature rise),
   while a leaking wastegate only costs capacity. What was missing were the
   sensors that see it. The 914's TCU reads an airbox temperature sensor and drives
   the wastegate through a servo (Rotax 914 maintenance documentation; "Understanding
   the 914 Rotax"); the 915/916 iS engine management reads intake air temperature
   and positions its wastegate electrically. So `airbox_temp_c` and
   `wastegate_position` are added as instruments on turbocharged engines
   (sensors_v5.py) and as twin residuals (twin_v5.py). Below critical altitude a
   worn turbo then shows as "more closure AND a hotter charge", a leaking
   wastegate as "more closure, normal charge temperature". On the naturally
   aspirated 912 they are absent (no such sensor), and read as constants.
4. AIR FILTER: unchanged (<= 4.5% pressure loss at rated flow, a realistic
   clean-to-clogged range). It becomes visible through the new MAP residual on the
   912 and through wastegate closure and charge temperature on turbo engines,
   which the boost controller must spend to make up the loss.

Every equation changed here is listed in `CHANGED_EQUATIONS` for the Simulink
builder.
"""
from __future__ import annotations

import math

import physics_v4 as V4
from physics_v4 import (  # noqa: F401  re-exported so v5 callers need one import
    ENGINE_SPECS_V4 as ENGINE_SPECS, EngineSpec, Health, Inputs, atmosphere,
    AFR_STOICH, AFR_STOICH_ETHANOL, CARB_DENSITY_EXP, CYCLE_QUALITY, DESIGN_OCTANE_MON,
    EXHAUST_BACKPRESSURE, GAMMA, LHV_AVGAS, LHV_ETHANOL, R_AIR, VE_CURV_HI, VE_CURV_LO,
    VE_PEAK, VE_PEAK_N,
)

PHYSICS_VERSION = "v5"

# -- ignition ----------------------------------------------------------------
# Effective spark retard (in the knock model's units) per unit of ignition loss.
# CALIBRATED in tests/test_physics_v5.py so a fully degraded circuit (the fault's
# full depth, ignition_mod 0.75) costs about 150-250 rpm at a ~4,000 rpm
# operating point: more than a healthy single-circuit check (50-100 rpm) and
# inside the 300 rpm limit, which is where a technician would reject the circuit.
K_IGN_RETARD = 0.22
# Share of the wall heat that late burning moves into the exhaust instead.
K_IGN_WALL = 0.9

# -- oil ---------------------------------------------------------------------
# Pump internal leakage: fraction of rated-speed delivery lost per unit of pump
# wear, scaled up at low speed because the leak is per revolution.
K_PUMP_LEAK = 0.55
# Bearing clearance: v4's 1.9 gain on the as-new-relative clearance, now shaped
# by speed and oil temperature. At rated speed on 90 C oil it equals v4.
K_BEARING = 1.9
# Viscosity-index loss of degraded oil: pressure loss per unit degradation per
# degree above 50 C, normalised so a fully degraded charge (0.55) loses about a
# fifth of its pressure at a 95 C cruise, like v4, but little when cool.
K_OIL_VI = 0.83

# -- knock reference -----------------------------------------------------------
# v4 referenced knock to the design point on a STANDARD day only, so the certified
# hot-day full-power climb (FAR 23 cooling case, ISA+23) already "retarded" the
# 912/915/916 by 0.08 - 15% power lost and EGT pushed over the 912's 880 C limit,
# on the engine's own design fuel. A certified engine must make its power there
# without retard. v5 references knock to the whole certified envelope: sea level
# to critical altitude, standard day AND ISA+23, design fuel. Retard (or, on the
# fixed-timing 912/914, the detonation-margin loss it stands for) now appears only
# outside certification: low-octane fuel, hotter days, boost above the limit.
KNOCK_DESIGN_ISA_C = (0.0, 23.0)

# -- exhaust gas temperature --------------------------------------------------
# v4 read EGT as a fixed 0.47 of the exhaust enthalpy rise at every load, so EGT
# was the same at idle, cruise and full power and sat 120-170 C below the Rotax
# normal cruise EGT (912 ~780 C; 914/915/916 ~850 C, the engines' own `egt_nominal_c`).
# A real probe sits downstream of the port behind a pipe that loses a roughly
# fixed amount of heat, so the fraction still present at the probe falls as the
# exhaust flow falls: EGT rises with power. Modelled as
#     fraction = EGT_FRAC_75 * (m_exh / (0.75 m_exh_rated)) ** EGT_FLOW_EXP
# with m_exh_rated the engine's own full-throttle rated-speed sea-level exhaust
# flow. CALIBRATED so a 75% sea-level cruise reads close to egt_nominal_c, idle
# reads ~400-500 C, and full power stays under egt_limit_c (tests/test_physics_v5.py).
EGT_FRAC_75 = 0.557
EGT_FLOW_EXP = 0.265
# Rich-mixture EGT depression. Rich of stoichiometric the excess fuel leaves
# unburnt and cools the charge, so EGT peaks near stoichiometric and sits well
# below the peak at the rich power mixture: roughly 100-150 F (55-85 K) at
# lambda 0.88 on a typical aero piston EGT-mixture curve. v4 had no such term,
# so leaning a rich engine left EGT flat. Normalised to the design mixture
# (lambda 0.88), where the EGT calibration above was made, so a healthy engine at
# its design mixture is unchanged; leaning to stoichiometric now raises EGT ~50 K.
K_RICH_EGT = 0.5
LAMBDA_DESIGN = 0.88

CHANGED_EQUATIONS = {
    "_cycle.eta_comb": "eta_comb = 0.98 * combustion_eff_mod * min(1, lam) * (1 - 0.25 max(0, lam-1)^2); ignition_mod REMOVED",
    "_cycle.ign_retard": "ign_retard = K_IGN_RETARD * max(0, 1 - ignition_mod)",
    "_cycle.spark_eff": "spark_eff = 1 / (1 + 2.2 * (knock_retard + ign_retard))",
    "_cycle.wall_frac": "wall_frac = 0.32 * (1 - K_IGN_WALL * ign_retard / (1 + ign_retard))",
    "_thermal.q_wall": "q_wall = wall_frac * exhaust_power + 0.55 * friction_power + pumping_power",
    "_oil_pressure_pa.pump": "pump = max(0.15, 1 - K_PUMP_LEAK * (1 - oil_pump_mod) / max(speed_frac, 0.2)^0.8)",
    "_oil_pressure_pa.clearance": "clearance = 1 + K_BEARING * max(0, friction_mod-1) * (0.55 + 0.45 speed_frac)",
    "_oil_pressure_pa.visc": "visc_eff = visc(T_oil) * max(0.35, 1 - K_OIL_VI (1-oil_quality_mod) max(0, T_oil-50)/45)",
    "_oil_pressure_pa.p": "p = 4.2e5 * speed_frac^0.85 * visc_eff^0.45 * pump / clearance, clamped [0.8, 7] bar",
    "_design_knock_demand": "max knock demand over alt in [0, crit_alt] x isa in {0, +23} at rated rpm, full throttle, design fuel",
    "_cycle.lambda_actual": "lam = clamp(lam_commanded / injector_flow_mod, 0.65, 1.40), used by eta_comb and reported as lambda",
    "_cycle.egt_target": "egt_target = t_in + egt_rise * EGT_FRAC_75 * (max(m_exh / m_exh_rated, 0.02) / 0.75) ^ EGT_FLOW_EXP * (1 - K_RICH_EGT max(0, 1-lam)) / (1 - K_RICH_EGT (1-0.88))",
    "step.outputs": "adds airbox_temp_c (= charge_temp_c on turbo engines, ambient on the 912)",
}


class PistonEngineV5(V4.PistonEngineV4):
    """physics_v4 with separable ignition, oil and turbo fault signatures."""

    PHYSICS_VERSION = "v5"

    def __init__(self, engine_model: str = "Rotax_914_ULF", dt: float = 0.01):
        super().__init__(engine_model, dt)
        # Rated exhaust flow, the EGT probe model's reference (full throttle, rated
        # speed, sea level, standard day, healthy).
        rated = self._cycle(Inputs(altitude_m=0.0, throttle=1.0), Health(),
                            self.spec.rated_rpm * 2 * math.pi / 60.0, warm=True)
        self.m_exh_rated = rated["m_dot_air"] + rated["m_dot_fuel"]

    def _design_knock_demand(self) -> float:
        omega_rated = self.spec.rated_rpm * 2 * math.pi / 60.0
        top = self.spec.crit_alt_m if self.spec.turbocharged else 0.0
        return max(max(self._cycle(Inputs(altitude_m=top * k / 6.0, throttle=1.0, isa_dev_c=isa,
                                          fuel_octane_mon=DESIGN_OCTANE_MON),
                                   Health(), omega_rated, warm=True)["knock_demand_raw"]
                       for k in range(7) for isa in KNOCK_DESIGN_ISA_C), 1e-6)

    # ------------------------------------------------------------------
    def _cycle(self, u: Inputs, h: Health, omega: float, warm: bool = False) -> dict:
        s = self.spec
        rpm = omega * 60.0 / (2 * math.pi)
        rpm = max(rpm, 1.0)

        p_amb, t_amb, rho_amb, _ = atmosphere(
            u.altitude_m, u.isa_dev_c, u.qnh_offset_pa, u.humidity_frac)

        # -- L2 induction (unchanged from v4) ------------------------------
        filt_loss = (1.0 - h.air_filter_mod) * 0.10
        p_in = p_amb * (1.0 - filt_loss * (rpm / s.rated_rpm) ** 2)
        t_in = t_amb

        boost_pr = 1.0
        wg_frac = 0.0
        limit = u.boost_limit_pa if u.boost_limit_pa else s.max_map_pa
        if s.turbocharged:
            spool = min(1.0, max(0.0, (rpm - 2000.0) / 2000.0))
            cap_pr = 1.0 + (self.pr_max - 1.0) * h.turbo_flow_mod * h.wastegate_mod * spool
            need_pr = limit / max(p_in, 1.0)
            boost_pr = max(1.0, min(cap_pr, need_pr))
            wg_frac = (min(max(1.0 - (boost_pr - 1.0) / (cap_pr - 1.0), 0.0), 1.0)
                       if cap_pr > 1.0 + 1e-9 else 0.0)

            eta_c = 0.72 * h.turbo_eff_mod
            t_rise_ideal = t_in * (boost_pr ** ((GAMMA - 1) / GAMMA) - 1.0)
            t_in = t_in + t_rise_ideal / max(eta_c, 0.2)

            if s.intercooled:
                eff_ic = 0.65 * h.intercooler_mod
                t_in = t_in - eff_ic * (t_in - t_amb)

        p_boost = p_in * boost_pr

        thr = min(max(u.throttle, 0.0), 1.0)
        map_pa = p_boost * (s.idle_map_frac + (1.0 - s.idle_map_frac) * thr ** 1.35)
        map_pa = min(map_pa, limit if s.turbocharged else p_boost)

        # -- L3 charge and combustion --------------------------------------
        n_frac = rpm / s.rated_rpm
        curv = VE_CURV_LO if n_frac <= VE_PEAK_N else VE_CURV_HI
        ve = (VE_PEAK - curv * (n_frac - VE_PEAK_N) ** 2) \
            * self.ve_scale * h.volumetric_eff_mod
        ve *= (1.0 - 0.35 * (1.0 - h.compression_mod))
        ve *= (1.0 - 0.25 * (1.0 - h.valve_leak_mod))
        ve = max(ve, 0.05)

        rho_charge = map_pa / (R_AIR * t_in)
        m_dot_air = ve * rho_charge * s.displacement_m3 * (rpm / 60.0) / 2.0

        stoich = AFR_STOICH * (1 - u.fuel_ethanol_frac) + AFR_STOICH_ETHANOL * u.fuel_ethanol_frac
        lhv = LHV_AVGAS * (1 - u.fuel_ethanol_frac) + LHV_ETHANOL * u.fuel_ethanol_frac
        if s.fuel_injected:
            lam = u.target_lambda
        else:
            lam = u.target_lambda * (rho_amb / 1.225) ** CARB_DENSITY_EXP
        lam = min(max(lam, 0.65), 1.25)
        m_dot_fuel = m_dot_air / (stoich * lam) * h.injector_flow_mod
        # v5: the mixture the cylinder actually burns. v4 fed the COMMANDED lambda to
        # the combustion model, so a fouled injector (less fuel, same air) lost heat
        # release in proportion to the missing fuel even while the engine was still
        # rich and oxygen-limited - EGT fell 77 C. Running rich, the air limits the
        # heat, so leaning toward stoichiometric costs little power and RAISES EGT;
        # lean of stoichiometric, power and EGT both fall. That is the real shape.
        lam = min(max(lam / max(h.injector_flow_mod, 1e-3), 0.65), 1.40)

        # v5: ignition no longer scales heat release (see module docstring, 1).
        eta_comb = 0.98 * h.combustion_eff_mod
        eta_comb *= min(1.0, lam)
        eta_comb *= 1.0 - 0.25 * max(0.0, lam - 1.0) ** 2
        eta_comb = max(eta_comb, 0.30)

        q_in = m_dot_fuel * lhv * eta_comb

        cr_eff = 1.0 + (s.compression_ratio - 1.0) * h.compression_mod
        eta_th = (1.0 - cr_eff ** -(GAMMA - 1)) * CYCLE_QUALITY

        knock_demand = (map_pa / 101325.0) * (t_in / 288.15) * (cr_eff / 10.0)
        octane_factor = DESIGN_OCTANE_MON / max(u.fuel_octane_mon, 60.0)
        humid_factor = 1.0 / (1.0 + 0.35 * u.humidity_frac)
        ratio = knock_demand / max(self.knock_design, 1e-6) * octane_factor * humid_factor
        retard = max(0.0, ratio - 1.0)
        # v5: a weak ignition circuit acts as a late spark on top of any knock retard.
        ign_retard = K_IGN_RETARD * max(0.0, 1.0 - h.ignition_mod)
        spark_eff = 1.0 / (1.0 + 2.2 * (retard + ign_retard))
        eta_th *= spark_eff
        # Late burning: heat that would have gone into the walls leaves in the exhaust.
        wall_frac = 0.32 * (1.0 - K_IGN_WALL * ign_retard / (1.0 + ign_retard))

        indicated_power = q_in * eta_th

        # -- L4 mechanical (unchanged from v4) -------------------------------
        mps = 2.0 * s.stroke_m * (rpm / 60.0)
        p_max = map_pa * cr_eff * 3.0
        fmep = (0.60e5 + 0.006 * p_max
                + 3.0e3 * mps + 0.25e3 * mps ** 2) * h.friction_mod
        oil_visc_factor = self._oil_viscosity_factor(
            self.oil_k if not warm else 363.15)
        fmep *= (0.88 + 0.12 * oil_visc_factor)
        fmep *= 1.0 + 0.35 * max(0.0, 1.0 - h.oil_quality_mod)

        friction_power = fmep * s.displacement_m3 * (rpm / 60.0) / 2.0

        pmep = max(0.0, EXHAUST_BACKPRESSURE * p_amb - map_pa)
        pumping_power = pmep * s.displacement_m3 * (rpm / 60.0) / 2.0

        brake_power = max(0.0, indicated_power - friction_power - pumping_power)
        brake_torque = brake_power / max(omega, 1.0)

        exhaust_power = q_in - indicated_power
        m_dot_exh = m_dot_air + m_dot_fuel
        egt_rise = exhaust_power / max(m_dot_exh * 1150.0, 1e-6)
        m_ref = getattr(self, "m_exh_rated", None)
        if m_ref:
            probe = EGT_FRAC_75 * (max(m_dot_exh / m_ref, 0.02) / 0.75) ** EGT_FLOW_EXP
        else:
            probe = 0.47        # during the parent's calibration, which reads no EGT
        rich = (1.0 - K_RICH_EGT * max(0.0, 1.0 - lam)) / (1.0 - K_RICH_EGT * (1.0 - LAMBDA_DESIGN))
        egt_target_k = t_in + egt_rise * probe * rich

        return {
            "rpm": rpm, "p_amb": p_amb, "t_amb": t_amb, "rho_amb": rho_amb,
            "map_pa": map_pa, "boost_pr": boost_pr, "charge_temp_k": t_in,
            "m_dot_air": m_dot_air, "m_dot_fuel": m_dot_fuel, "lambda": lam,
            "ve": ve, "eta_comb": eta_comb, "eta_th": eta_th, "retard": retard,
            "ign_retard": ign_retard, "wall_frac": wall_frac,
            "knock_demand_raw": knock_demand,
            "indicated_power_w": indicated_power, "friction_power_w": friction_power,
            "pumping_power_w": pumping_power,
            "brake_power_w": brake_power, "brake_torque_nm": brake_torque,
            "exhaust_power_w": exhaust_power, "egt_target_k": egt_target_k,
            "fmep_pa": fmep, "wastegate": wg_frac,
        }

    # ------------------------------------------------------------------
    def _thermal(self, cyc: dict, u: Inputs, h: Health) -> None:
        """v4's thermal network with the wall-heat share taken from the cycle, so
        late (ignition-degraded) burning heats the exhaust instead of the head."""
        # v4 computes q_wall = 0.32 * exhaust + ...; feed it an exhaust power
        # scaled so that 0.32 * scaled == wall_frac * exhaust, leaving every other
        # term of the v4 network (which does not read exhaust power) untouched.
        scaled = dict(cyc)
        scaled["exhaust_power_w"] = cyc["exhaust_power_w"] * cyc.get("wall_frac", 0.32) / 0.32
        super()._thermal(scaled, u, h)

    # ------------------------------------------------------------------
    def _oil_pressure_pa(self, cyc: dict, h: Health) -> float:
        """Gear pump against bearing clearance through hot oil, with the three oil
        faults acting through the mechanisms that really differ (docstring, 2)."""
        s = self.spec
        rpm = cyc["rpm"]
        speed_frac = rpm / s.rated_rpm
        tc = self.oil_k - 273.15
        visc = self._oil_viscosity_factor(self.oil_k)

        # Degraded oil loses viscosity index: it thins more the hotter it runs.
        q_loss = max(0.0, 1.0 - max(h.oil_quality_mod, 0.25))
        visc_eff = visc * max(0.35, 1.0 - K_OIL_VI * q_loss * max(0.0, tc - 50.0) / 45.0)

        # Worn pump: internal leakage per revolution, so worst at low speed.
        pump = max(0.15, 1.0 - K_PUMP_LEAK * max(0.0, 1.0 - h.oil_pump_mod)
                   / max(speed_frac, 0.2) ** 0.8)

        # Opened bearing clearances: the leak grows with speed. (Temperature already
        # acts through viscosity; an extra hot-oil amplification here pushed normally
        # aged engines below the 2 bar minimum on hot days, which a Rotax at TBO does
        # not do, and temperature dependence is oil degradation's own signature.)
        bearing = max(0.0, h.friction_mod - 1.0)
        clearance = 1.0 + K_BEARING * bearing * (0.55 + 0.45 * speed_frac)

        p = 4.2e5 * max(speed_frac, 1e-3) ** 0.85 * visc_eff ** 0.45 * pump / clearance
        return min(max(p, 0.8e5), 7.0e5)

    # ------------------------------------------------------------------
    def step(self, u: Inputs | None = None, h: Health | None = None) -> dict:
        out = super().step(u, h)
        out["physics_version"] = PHYSICS_VERSION
        # The airbox / intake-air temperature sensor the turbo engines carry; the
        # carburetted 912 has none, so it reads ambient (a constant residual).
        out["airbox_temp_c"] = (out["charge_temp_c"] if self.spec.turbocharged
                                else out["ambient_temp_c"])
        return out
