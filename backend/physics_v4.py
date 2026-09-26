"""Physics v4 - a real, layered, closed-loop aero-piston engine model.

COMPLETELY NEW. v2 and v3 are untouched and still selectable; nothing here
imports from or modifies them. What carries over is not code but four lessons
from building and validating the C-MAPSS turbofan in simulation/cmapss/:

  1. STATION-BY-STATION PHYSICS, NOT A TORQUE LOOKUP.
     v3's torque came from an interpolated RPM->torque table with altitude and
     wear applied as multipliers. That cannot respond correctly to anything the
     table did not anticipate - a hot day, a fouled air filter, a failing
     wastegate - because those act on air mass, not on torque directly. v4
     computes air through the induction path, fuel to match it, and torque out
     of the resulting cylinder pressure, so every effect enters where it
     physically acts.

  2. COMPONENT-LEVEL HEALTH, NOT A SINGLE `wear` SCALAR.
     The C-MAPSS work showed that a scalar health index driving a fitted
     response surface cannot reproduce the joint structure real sensors have
     (a discriminator separated it at AUC 0.91). Health here is FOURTEEN
     component modifiers, the piston-engine analogue of C-MAPSS's Table 1, each
     entering the physics at the component it belongs to.

  3. CLOSED LOOP IS WHERE DEGRADATION SIGNATURES COME FROM.
     This was the single most important finding. In C-MAPSS the controller holds
     demanded fan speed, so a degraded engine must burn more to hold it and runs
     hotter - which reproduced the real data's degradation signs with nothing
     fitted to them. The same is true here: the ECU holds a boost target and a
     lambda target, so a worn engine opens the wastegate further, injects longer
     and advances or retards timing to compensate. EGT rising with injector
     fouling is then a CONSEQUENCE of the control law, not a rule someone wrote.

  4. DERIVED PHYSICS AND FITTED INSTRUMENTATION MUST STAY SEPARATE.
     Everything in this module is derived from thermodynamics and published
     engine geometry. Sensor noise, bias, drift and quantisation are NOT here -
     they belong to a sensor model layered on top, so the boundary between
     "what the engine does" and "what the instrument reports" stays auditable.
     Values that had to be chosen rather than derived are marked ASSUMED or
     CALIBRATED at the point of use.

LAYERS
    L0  environment + pilot/mission demand   (inputs)
    L1  atmosphere                            -> ambient p, T, rho
    L2  induction                             -> MAP, charge temp, air mass flow
    L3  combustion                            -> fuel flow, IMEP, exhaust temp
    L4  mechanical                            -> friction, brake torque, shaft dynamics
    L5  drivetrain + propeller                -> prop speed, absorbed torque, thrust
    L6  thermal network                       -> EGT, CHT, coolant, oil temperature
    L7  lubrication                           -> oil pressure
    L8  electrical                            -> alternator, bus voltage
    L9  health modifiers                      (inputs, from the degradation model)
    L10 ECU / closed-loop control             -> wastegate, injection, timing

UNITS
    SI throughout inside the model: Pa, K, kg/s, N.m, W, rad/s. The public
    step() result converts to the units the existing telemetry schema uses
    (degC, RPM, kW, L/h, bar) so it stays a drop-in for the dashboard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_AIR = 287.05          # J/(kg.K)
GAMMA = 1.40
CP_AIR = 1005.0         # J/(kg.K)
T0_ISA = 288.15         # K
P0_ISA = 101325.0       # Pa
LAPSE = 0.0065          # K/m
G0 = 9.80665
LHV_AVGAS = 43.5e6      # J/kg, AVGAS 100LL / MOGAS
AFR_STOICH = 14.7       # gasoline
LHV_ETHANOL = 26.8e6
AFR_STOICH_ETHANOL = 9.0

# Ratio of real indicated efficiency to the ideal Otto value: the ideal
# air-standard cycle ignores finite burn duration, heat loss to the walls and
# exhaust blowdown. Applying it explicitly keeps the absolute efficiency
# physical instead of hiding the error inside a per-engine scale factor.
#
# CALIBRATED to one published number: Rotax 912 ULS takeoff fuel flow, 27 L/h at
# 73.5 kW (BSFC 264 g/kWh). It was 0.62, the textbook figure, which was only
# consistent while rich mixtures released more heat than their oxygen could
# support; with combustion correctly oxygen-limited it becomes 0.677.
CYCLE_QUALITY = 0.6769

# Fuel the engines are certified on, against which the knock margin is defined.
DESIGN_OCTANE_MON = 95.0

# Volumetric efficiency shape: peak value, the speed (fraction of rated) where
# it peaks, and a curvature either side of it. The previous curve peaked at 40%
# of rated speed and put the 912's torque peak at 2000 rpm - a low-revving
# tractor engine, not a tuned aero engine whose intake is sized for high speed.
#
# ASYMMETRIC because one quadratic cannot be both: steep enough above the peak
# to put the torque maximum where it belongs collapses VE to ~0.15 at idle,
# where a real engine still breathes ~0.8 and the model then could not idle at
# all. Below the peak VE falls gently; above it, flow losses rise fast.
#   VE_PEAK_N, VE_CURV_HI,  CALIBRATED jointly to the one published torque-curve
#   VE_CURV_LO              point: 912 ULS maximum torque 128 N.m at 5100 rpm
#                           against 121 N.m at the 5800 rpm rating. The fit
#                           gives 127.7 N.m at 5050 rpm, ratio 1.055 - all
#                           three within 1% - and VE 0.70 at idle, in the range
#                           of a small four-stroke at low speed.
VE_PEAK = 0.92
VE_PEAK_N = 0.94
VE_CURV_LO = 0.45
VE_CURV_HI = 10.0

# Exhaust back-pressure as a multiple of ambient for a free exhaust. ASSUMED at
# the usual 5%. Sets the pumping loss: an engine throttled below this pressure
# must pull its charge in against it, which is the main reason part-throttle
# fuel economy is worse than full throttle and why idle needs any MAP at all.
EXHAUST_BACKPRESSURE = 1.05

# Installed heat-transfer coefficients of the radiator, head-to-coolant jacket
# and oil cooler, in W/K for the 914 (other engines scale with rated power).
# These are airframe properties Rotax does not publish, so they are CALIBRATED -
# to the case a real installation is certified against, not to cruise: a
# full-power climb at 38 m/s on a 100 F (ISA+23) sea-level day, the FAR 23
# cooling condition. There the 914 must sit just inside its limits: coolant 110,
# head 130 (limit 135), oil 125 (limit 130). Cruise temperatures then fall
# where they fall, with the oil thermostat holding oil near 90.
#
# The earlier values were fitted to cruise at an unstated power. In mission
# data they left HEALTHY engines over a certified limit on 12-34% of rows - in
# climb, at 0.92 throttle - which a certified installation never does.
UA_COOL = 522.1
UA_HEAD = 600.9
UA_OIL = 78.7

# Oil thermostat: closed (cooler bypassed) below the first temperature, fully
# open at the second. ASSUMED at typical aero-engine thermostat settings.
OIL_TSTAT_OPEN_C = 85.0
OIL_TSTAT_FULL_C = 100.0

# Carburettor mixture drift with air density. A fixed jet meters fuel by
# pressure drop, so air/fuel ratio scales with sqrt(air density): the mixture
# goes RICHER as density falls (exponent +0.5 in pure theory). ASSUMED at 0.3 for
# the constant-depression carburettor, which compensates partly. The previous
# exponent had the wrong sign and ran the 912/914 LEANER with altitude - the
# opposite of its own comment, and of every carburetted aero engine.
CARB_DENSITY_EXP = 0.3


# ---------------------------------------------------------------------------
# L9 - HEALTH MODIFIERS.
#
# The piston-engine analogue of C-MAPSS Table 1. 1.0 = new component, below 1.0
# = degraded (above 1.0 where the quantity is a loss, e.g. friction). These are
# INPUTS: the degradation model drives them, and this module only consumes them.
# Separating them this way is what lets a fault be defined by the component it
# damages rather than by the sensor pattern it is supposed to produce.
# ---------------------------------------------------------------------------
@dataclass
class Health:
    # -- breathing ---------------------------------------------------------
    air_filter_mod: float = 1.0        # filter fouling -> induction pressure loss
    volumetric_eff_mod: float = 1.0    # valve/port condition -> trapped mass
    compression_mod: float = 1.0       # ring + bore wear -> lost compression
    valve_leak_mod: float = 1.0        # seat recession / burnt valve

    # -- boost (914 / 915 / 916 only) --------------------------------------
    turbo_eff_mod: float = 1.0         # compressor isentropic efficiency
    turbo_flow_mod: float = 1.0        # compressor flow capacity
    wastegate_mod: float = 1.0         # actuator authority
    intercooler_mod: float = 1.0       # charge-cooler effectiveness

    # -- fuel and ignition -------------------------------------------------
    injector_flow_mod: float = 1.0     # fouled injector / blocked jet
    ignition_mod: float = 1.0          # coil, plug, lead condition
    combustion_eff_mod: float = 1.0    # overall burn completeness

    # -- mechanical and fluids ---------------------------------------------
    friction_mod: float = 1.0          # bearing / piston wear (>1 = more friction)
    oil_pump_mod: float = 1.0          # pump delivery
    oil_quality_mod: float = 1.0       # viscosity / additive depletion
    cooling_mod: float = 1.0           # radiator + oil cooler effectiveness
    prop_eff_mod: float = 1.0          # blade erosion / pitch mechanism

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# L0 - INPUTS.
#
# Everything the caller supplies each step. The NEW fields relative to v3
# (which took only altitude, throttle, airspeed and AOA) are marked, because
# each one is an effect v3 physically could not represent.
# ---------------------------------------------------------------------------
@dataclass
class Inputs:
    # -- flight condition ---------------------------------------------------
    altitude_m: float = 0.0
    airspeed_ms: float = 45.0
    aoa_deg: float = 2.0
    throttle: float = 0.75             # 0..1 lever position

    # -- atmosphere (NEW) ---------------------------------------------------
    # v3 assumed a standard day at every altitude, so a hot-and-high takeoff and
    # a cold-day sea-level climb were indistinguishable to it. Density altitude
    # is what the engine actually feels.
    isa_dev_c: float = 0.0             # NEW: ISA temperature deviation
    qnh_offset_pa: float = 0.0         # NEW: pressure deviation from standard
    humidity_frac: float = 0.0         # NEW: 0..1, displaces oxygen and raises knock margin

    # -- fuel (NEW) ---------------------------------------------------------
    # The 912/914 are certified on different fuels from the 915/916, and fuel
    # quality drives the knock limit, which drives timing, which drives EGT.
    fuel_octane_mon: float = 95.0      # NEW: motor octane number
    fuel_ethanol_frac: float = 0.0     # NEW: E-fraction, changes stoich AFR and LHV

    # -- installation (NEW) -------------------------------------------------
    cooling_airflow_factor: float = 1.0  # NEW: cowling/duct effectiveness multiplier
    electrical_load_a: float = 12.0      # NEW: alternator load
    oil_thermostat_open: bool = True     # NEW: oil cooler bypass state

    # -- ECU setpoints (NEW) -------------------------------------------------
    # v3 had a fixed injection-timing output with no controller behind it. These
    # make the control loop explicit and let a fault be expressed as the
    # controller running out of authority.
    target_lambda: float = 0.88        # NEW: rich-of-stoich at power, 1.0 at cruise
    boost_limit_pa: float | None = None  # NEW: manifold pressure limit the ECU holds

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Engine geometry and calibration.
#
# Bore, stroke, cylinder count, displacement and compression ratio are published
# Rotax figures and are the only numbers here taken as fact. Everything else is
# marked. Rated power is used to calibrate one scalar (see `bmep_scale`), so the
# model reproduces the certified rating by construction rather than by luck.
# ---------------------------------------------------------------------------
@dataclass
class EngineSpec:
    name: str
    bore_m: float
    stroke_m: float
    cylinders: int
    compression_ratio: float
    rated_power_w: float
    rated_rpm: float
    max_rpm: float
    idle_rpm: float
    gear_ratio: float
    turbocharged: bool
    intercooled: bool
    fuel_injected: bool
    tbo_hours: float
    # boost schedule: ECU-held absolute manifold pressure at full throttle (Pa).
    # ASSUMED where not published; the 914's 40 inHg airbox limit is well known.
    max_map_pa: float = 101325.0
    # Critical altitude: the highest altitude at which the turbocharger can
    # still hold max_map_pa. It sizes the compressor's maximum pressure ratio,
    # so the engine holds rated power up to here and loses it above. Without it
    # the compressor made whatever ratio the target needed at any altitude, and
    # the 915/916 GAINED power all the way to 7500 m.
    crit_alt_m: float = 0.0
    # Manifold pressure at the idle stop, as a fraction of the pressure upstream
    # of the throttle. CALIBRATED per engine, as the idle stop screw is adjusted
    # on a real installation: each settles at its published 1400 rpm on the
    # ground against its own propeller, warm, which lands idle MAP at 29-37 kPa.
    # Per engine because the turbo engines drive larger propellers, and absorbed
    # torque scales with diameter to the fifth power. The previous throttle map
    # left 2.2% of ambient at idle (2.3 kPa), below what the engine needs to beat
    # its own friction, so every engine fell to the solver's floor at 700 rpm.
    idle_map_frac: float = 0.3
    # thermal masses (ASSUMED, sized to give the published warm-up behaviour)
    cht_mass_j_per_k: float = 9.0e3
    oil_mass_j_per_k: float = 2.6e4
    coolant_mass_j_per_k: float = 1.4e4
    oil_capacity_l: float = 3.0

    # -- airframe, carried over from v3 so the aero outputs survive ----------
    wing_area_m2: float = 25.0
    mass_kg: float = 450.0
    cd0: float = 0.030
    cd_k: float = 0.045

    # -- OPERABILITY LIMITS, the C-MAPSS idea applied to a piston engine.
    # Each is a certified limit from the Rotax operating manual. Health is the
    # minimum of the normalised margins against these, so "how close is this
    # engine to being unairworthy" is computed from the physics rather than
    # declared by a wear counter.
    cht_limit_c: float = 135.0
    egt_limit_c: float = 950.0
    oil_temp_limit_c: float = 130.0
    oil_press_min_bar: float = 2.0
    # Rotax specifies oil pressure piecewise: minimum 0.8 bar below 3500 rpm,
    # 2-5 bar normal above it. A single 2.0 bar floor at every speed flagged
    # healthy low-speed readings of 1.5-2 bar - descent, loiter - as limit
    # violations, on ~10% of a healthy 915's flight time.
    oil_press_min_low_bar: float = 0.8
    oil_press_nominal_low_bar: float = 2.0
    oil_press_split_rpm: float = 3500.0
    cht_nominal_c: float = 105.0
    egt_nominal_c: float = 850.0
    oil_temp_nominal_c: float = 95.0
    oil_press_nominal_bar: float = 3.5

    @property
    def displacement_m3(self) -> float:
        return (math.pi / 4.0) * self.bore_m ** 2 * self.stroke_m * self.cylinders

    @property
    def piston_area_m2(self) -> float:
        return (math.pi / 4.0) * self.bore_m ** 2 * self.cylinders


# All four engines share the 61 mm stroke. The 914 alone uses the 79.5 mm bore
# (1211 cc); the 912 ULS, 915 iS and 916 iS use 84 mm (1352 cc). These are
# published Rotax geometry.
_STROKE = 0.061

ENGINE_SPECS_V4: dict[str, EngineSpec] = {
    "Rotax_912_ULS": EngineSpec(
        name="Rotax_912_ULS", idle_map_frac=0.291, bore_m=0.084, stroke_m=_STROKE, cylinders=4,
        compression_ratio=10.5, rated_power_w=73_500.0, rated_rpm=5800.0,
        max_rpm=5800.0, idle_rpm=1400.0, gear_ratio=2.4286,
        turbocharged=False, intercooled=False, fuel_injected=False,
        tbo_hours=2000.0, max_map_pa=101325.0,
        egt_limit_c=880.0, egt_nominal_c=780.0),   # carburetted, lower EGT limit
    "Rotax_914_ULF": EngineSpec(
        name="Rotax_914_ULF", idle_map_frac=0.314, bore_m=0.0795, stroke_m=_STROKE, cylinders=4,
        compression_ratio=9.0, rated_power_w=84_500.0, rated_rpm=5800.0,
        max_rpm=5800.0, idle_rpm=1400.0, gear_ratio=2.4286,
        turbocharged=True, intercooled=False, fuel_injected=False,
        tbo_hours=2000.0,
        # 40 inHg absolute is the published airbox limit the 914's wastegate
        # controller holds at takeoff power.
        max_map_pa=135_500.0,
        crit_alt_m=4572.0),                        # ASSUMED, as the 915
    "Rotax_915_iS": EngineSpec(
        name="Rotax_915_iS", idle_map_frac=0.364, bore_m=0.084, stroke_m=_STROKE, cylinders=4,
        compression_ratio=10.5, rated_power_w=104_000.0, rated_rpm=5800.0,
        max_rpm=5800.0, idle_rpm=1400.0, gear_ratio=2.4286,
        turbocharged=True, intercooled=True, fuel_injected=True,
        tbo_hours=1200.0, max_map_pa=152_000.0,    # ASSUMED boost limit
        crit_alt_m=4572.0),                        # published: 15,000 ft
    "Rotax_916_iS": EngineSpec(
        name="Rotax_916_iS", idle_map_frac=0.367, bore_m=0.084, stroke_m=_STROKE, cylinders=4,
        compression_ratio=10.5, rated_power_w=118_000.0, rated_rpm=5800.0,
        max_rpm=5800.0, idle_rpm=1400.0, gear_ratio=2.4286,
        turbocharged=True, intercooled=True, fuel_injected=True,
        tbo_hours=2000.0, max_map_pa=162_000.0,    # ASSUMED boost limit
        crit_alt_m=4572.0),                        # ASSUMED, as the 915
}


# ---------------------------------------------------------------------------
# L1 - ATMOSPHERE
# ---------------------------------------------------------------------------
def atmosphere(altitude_m: float, isa_dev_c: float = 0.0,
               qnh_offset_pa: float = 0.0, humidity_frac: float = 0.0):
    """Ambient static pressure, temperature and density.

    Humidity displaces air: water vapour is lighter than dry air, so a humid day
    is a low-density day at the same pressure and temperature. v3 ignored this
    entirely; it matters most on exactly the hot, humid, low-level days where a
    MALE-UAV takeoff is already marginal.
    """
    alt = max(0.0, float(altitude_m))
    t_std = T0_ISA - LAPSE * alt
    p = P0_ISA * (t_std / T0_ISA) ** (G0 / (LAPSE * R_AIR)) + qnh_offset_pa
    t = t_std + isa_dev_c
    p = max(p, 1000.0)

    # Saturation vapour pressure (Tetens), then partial pressures.
    tc = t - 273.15
    p_sat = 610.78 * math.exp(17.27 * tc / (tc + 237.3)) if tc > -40 else 10.0
    p_v = max(0.0, min(humidity_frac, 1.0)) * p_sat
    p_d = max(p - p_v, 1000.0)
    rho = p_d / (R_AIR * t) + p_v / (461.495 * t)
    return p, t, rho, p_v


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class PistonEngineV4:
    """Layered, closed-loop aero-piston engine.

    Call `step(inputs, health)` at a fixed dt. State (shaft speed, temperatures,
    controller integrators) persists across calls.
    """

    PHYSICS_VERSION = "v4"

    def __init__(self, engine_model: str = "Rotax_914_ULF", dt: float = 0.01):
        if engine_model not in ENGINE_SPECS_V4:
            raise ValueError(f"unknown engine {engine_model!r}; "
                             f"options {list(ENGINE_SPECS_V4)}")
        self.spec = ENGINE_SPECS_V4[engine_model]
        self.engine_model = engine_model
        self.dt = float(dt)
        self.t = 0.0

        s = self.spec
        # Rotational inertia of crank + gearbox + propeller referred to the
        # crankshaft. ASSUMED; sized so a throttle slam takes ~1.5 s to settle,
        # which is the observed behaviour of this engine class.
        self.inertia = 0.42

        # Compressor pressure-ratio ceiling, sized so the turbocharger can just
        # hold the boost limit at the critical altitude and no higher.
        self.pr_max = (s.max_map_pa / atmosphere(s.crit_alt_m)[0]
                       if s.turbocharged else 1.0)

        # Design-point knock demand, established before anything else so the
        # knock model has a reference. Computed with the reference fuel on a
        # standard day at rated speed and full throttle.
        self.ve_scale = 1.0
        self.knock_design = 1.0
        self.knock_design = self._design_knock_demand()

        # ONE CALIBRATED SCALAR, applied to the TRAPPED AIR MASS, set so the
        # model produces the certified rated power at rated RPM on a standard
        # day at full throttle. The direct analogue of the C-MAPSS margin gain -
        # a single scalar fitted to a single published number.
        #
        # It previously multiplied brake power on the way out. That left the
        # fuel, friction and exhaust heat unscaled, so the 914 reported 34.9% of
        # its fuel energy as shaft work while the fuel it burned supported only
        # 31.6%: energy was not conserved, and BSFC read 10% better than the
        # engine could achieve. Scaling the air instead moves fuel, power and
        # exhaust heat together.
        self.ve_scale = self._calibrate_to_rating()

        # -- state -----------------------------------------------------------
        self.omega = s.idle_rpm * 2 * math.pi / 60.0
        self.cht_k = 288.15
        self.oil_k = 288.15
        self.coolant_k = 288.15
        self.egt_k = 288.15
        self.wastegate = 0.0        # 0 = closed (max boost), 1 = fully open
        self.life_used_h = 0.0      # duty-weighted engine hours consumed
        self.battery_v = 13.8

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _design_knock_demand(self) -> float:
        """Knock demand at the hardest point of the certified envelope.

        The engine is certified to make rated power on its specified fuel
        anywhere from sea level to its critical altitude, so none of that
        envelope may need retard. For the 914, which has no intercooler, the
        hardest point is the critical altitude: the compressor works hardest
        there and the charge is hottest. Referencing sea level alone made the
        914 retard timing - and lose 14% of its power - at 3000 m, well inside
        its certified envelope, then REGAIN power above the critical altitude as
        falling boost cooled the charge.
        """
        omega_rated = self.spec.rated_rpm * 2 * math.pi / 60.0
        top = self.spec.crit_alt_m if self.spec.turbocharged else 0.0
        return max(max(self._cycle(Inputs(altitude_m=top * k / 6.0, throttle=1.0,
                                          fuel_octane_mon=DESIGN_OCTANE_MON),
                                   Health(), omega_rated, warm=True)["knock_demand_raw"]
                       for k in range(7)), 1e-6)

    def _calibrate_to_rating(self) -> float:
        """Air-mass scale so full throttle at rated RPM, sea level, ISA gives
        rated power.

        Indicated power is linear in trapped air mass while friction and
        pumping are not, so brake = scale x indicated - losses solves exactly:
        scale = (rated + losses) / indicated.
        """
        probe = Inputs(altitude_m=0.0, throttle=1.0, isa_dev_c=0.0)
        omega_rated = self.spec.rated_rpm * 2 * math.pi / 60.0
        res = self._cycle(probe, Health(), omega_rated, warm=True)
        losses = res["indicated_power_w"] - res["brake_power_w"]
        if res["indicated_power_w"] <= 1.0:
            return 1.0
        return (self.spec.rated_power_w + losses) / res["indicated_power_w"]

    # ------------------------------------------------------------------
    # L2 + L3 + L4 - the gas path and the cylinder
    # ------------------------------------------------------------------
    def _cycle(self, u: "Inputs", h: Health, omega: float,
               warm: bool = False) -> dict:
        """One engine-cycle evaluation at a given shaft speed. No integration."""
        s = self.spec
        rpm = omega * 60.0 / (2 * math.pi)
        rpm = max(rpm, 1.0)

        p_amb, t_amb, rho_amb, _ = atmosphere(
            u.altitude_m, u.isa_dev_c, u.qnh_offset_pa, u.humidity_frac)

        # -- L2 induction --------------------------------------------------
        # Air filter: a fouled filter is a pressure loss that grows with flow.
        filt_loss = (1.0 - h.air_filter_mod) * 0.10
        p_in = p_amb * (1.0 - filt_loss * (rpm / s.rated_rpm) ** 2)
        t_in = t_amb

        boost_pr = 1.0
        wg_frac = 0.0
        limit = u.boost_limit_pa if u.boost_limit_pa else s.max_map_pa
        if s.turbocharged:
            # THE COMPRESSOR HAS A FIXED CEILING, NOT THE RATIO THE TARGET ASKS
            # FOR. It is sized to hold the boost limit at the critical altitude,
            # so near sea level it has headroom and the wastegate bypasses the
            # excess; above the critical altitude it runs out and MAP falls.
            #
            # That headroom is also what makes turbo wear realistic: a degraded
            # compressor at low altitude is simply compensated by the wastegate
            # closing further, and shows up first as a LOWER CRITICAL ALTITUDE -
            # which is how it presents in service. Previously the compressor had
            # no headroom anywhere, so any wear cost boost immediately at sea
            # level, while a healthy one held rated power to any altitude.
            # Turbine power comes from exhaust flow, which is negligible at idle.
            # ASSUMED: no boost below 2000 rpm, full compressor capacity by 4000.
            # The previous ramp started from zero rpm and gave ~1.5 pressure
            # ratio at idle, which held every turbo engine 170-230 rpm above its
            # published 1400 rpm idle.
            spool = min(1.0, max(0.0, (rpm - 2000.0) / 2000.0))
            cap_pr = 1.0 + (self.pr_max - 1.0) * h.turbo_flow_mod * h.wastegate_mod * spool
            need_pr = limit / max(p_in, 1.0)
            boost_pr = max(1.0, min(cap_pr, need_pr))
            # Fraction of compressor capacity the wastegate is bypassing. The
            # clamp to `limit` below IS the ideal wastegate; this reports it.
            wg_frac = (min(max(1.0 - (boost_pr - 1.0) / (cap_pr - 1.0), 0.0), 1.0)
                       if cap_pr > 1.0 + 1e-9 else 0.0)

            eta_c = 0.72 * h.turbo_eff_mod       # ASSUMED design efficiency
            t_rise_ideal = t_in * (boost_pr ** ((GAMMA - 1) / GAMMA) - 1.0)
            t_in = t_in + t_rise_ideal / max(eta_c, 0.2)

            if s.intercooled:
                # Effectiveness: fraction of the boost heat given back to ambient.
                eff_ic = 0.65 * h.intercooler_mod   # ASSUMED
                t_in = t_in - eff_ic * (t_in - t_amb)

        p_boost = p_in * boost_pr

        # Throttle plate: from the idle stop to wide open. The floor is the idle
        # air the throttle stop and bypass admit.
        thr = min(max(u.throttle, 0.0), 1.0)
        map_pa = p_boost * (s.idle_map_frac + (1.0 - s.idle_map_frac) * thr ** 1.35)
        map_pa = min(map_pa, limit if s.turbocharged else p_boost)

        # -- L3 charge and combustion ---------------------------------------
        # Volumetric efficiency: peaks near rated speed, where an aero engine's
        # intake and valve timing are tuned. Health acts here, where breathing
        # losses physically act.
        n_frac = rpm / s.rated_rpm
        curv = VE_CURV_LO if n_frac <= VE_PEAK_N else VE_CURV_HI
        ve = (VE_PEAK - curv * (n_frac - VE_PEAK_N) ** 2) \
            * self.ve_scale * h.volumetric_eff_mod
        # Lost compression lets charge escape past the rings during compression.
        ve *= (1.0 - 0.35 * (1.0 - h.compression_mod))
        ve *= (1.0 - 0.25 * (1.0 - h.valve_leak_mod))
        ve = max(ve, 0.05)

        rho_charge = map_pa / (R_AIR * t_in)
        # Four-stroke: one intake per two revolutions.
        m_dot_air = ve * rho_charge * s.displacement_m3 * (rpm / 60.0) / 2.0

        # Fuel. Injected engines hold the commanded lambda; carburetted engines
        # drift richer as density falls because a fixed jet meters fuel by
        # pressure drop, not by mass - which is exactly why the 912/914 need
        # mixture management and the 915/916 do not.
        stoich = AFR_STOICH * (1 - u.fuel_ethanol_frac) + AFR_STOICH_ETHANOL * u.fuel_ethanol_frac
        lhv = LHV_AVGAS * (1 - u.fuel_ethanol_frac) + LHV_ETHANOL * u.fuel_ethanol_frac
        if s.fuel_injected:
            lam = u.target_lambda
        else:
            lam = u.target_lambda * (rho_amb / 1.225) ** CARB_DENSITY_EXP
        lam = min(max(lam, 0.65), 1.25)
        m_dot_fuel = m_dot_air / (stoich * lam) * h.injector_flow_mod

        # Combustion efficiency: falls when the mixture is far from stoichiometric,
        # when ignition is weak, and with explicit burn-quality degradation.
        #
        # RICH OF STOICHIOMETRIC THE OXYGEN IS THE LIMIT, NOT THE FUEL. Only a
        # fraction lambda of the fuel finds oxygen to burn; the rest leaves as
        # unburnt fuel and CO, carrying its energy out with it. So heat release
        # is capped at the stoichiometric value however much fuel is added -
        # which is why a rich mixture runs a COOLER exhaust, and why engines are
        # run rich at full power at all. This previously applied only a small
        # quadratic penalty, so at the default lambda of 0.88 every engine
        # released ~13% more heat than its air could burn, and a carburetted
        # engine going richer with altitude got HOTTER: the 914 ran a median EGT
        # of 905 C with a 1011 C p99 against its 950 C limit.
        eta_comb = 0.98 * h.combustion_eff_mod * h.ignition_mod
        eta_comb *= min(1.0, lam)                            # rich: oxygen-limited
        eta_comb *= 1.0 - 0.25 * max(0.0, lam - 1.0) ** 2    # lean: slow burn
        eta_comb = max(eta_comb, 0.30)

        q_in = m_dot_fuel * lhv * eta_comb                    # W of heat released

        # Otto-cycle thermal efficiency on the EFFECTIVE compression ratio, so
        # ring and bore wear reduce efficiency through the mechanism they
        # actually act on rather than through a fudge multiplier.
        cr_eff = 1.0 + (s.compression_ratio - 1.0) * h.compression_mod
        eta_th = (1.0 - cr_eff ** -(GAMMA - 1)) * CYCLE_QUALITY

        # Knock margin: the ECU retards timing when octane, charge temperature
        # or boost push it toward detonation, and retarded timing costs
        # efficiency and dumps heat into the exhaust.
        # KNOCK IS REFERENCED TO THE ENGINE'S OWN DESIGN POINT, not to an
        # absolute threshold. A turbocharged engine is certified to run at its
        # rated boost on its specified fuel, so at design conditions it must
        # retard nothing. Referencing an absolute limit instead penalised the
        # 915/916 for operating exactly as designed and cut their thermal
        # efficiency by 64%. Retard therefore measures DEVIATION from design -
        # hotter charge, poorer fuel, boost above the certified limit - which is
        # what actually drives a knock-limited ECU to pull timing.
        knock_demand = (map_pa / 101325.0) * (t_in / 288.15) * (cr_eff / 10.0)
        octane_factor = DESIGN_OCTANE_MON / max(u.fuel_octane_mon, 60.0)
        # Water vapour is inert and raises the knock margin.
        humid_factor = 1.0 / (1.0 + 0.35 * u.humidity_frac)
        ratio = knock_demand / max(self.knock_design, 1e-6) * octane_factor * humid_factor
        retard = max(0.0, ratio - 1.0)
        spark_eff = 1.0 / (1.0 + 2.2 * retard)
        eta_th *= spark_eff

        indicated_power = q_in * eta_th

        # -- L4 mechanical ---------------------------------------------------
        # Friction mean effective pressure: the standard Chen-Flynn form, rising
        # with speed and peak pressure. Bearing and piston wear multiply it.
        # Chen-Flynn friction correlation: a constant term, a peak-pressure term
        # for the rings and bearings, and linear plus quadratic piston-speed
        # terms for shear and windage. Coefficients are the usual published
        # values for a small four-stroke and give ~1.5 bar FMEP at rated speed,
        # which is the right magnitude for this engine class.
        mps = 2.0 * s.stroke_m * (rpm / 60.0)          # mean piston speed, m/s
        p_max = map_pa * cr_eff * 3.0                  # peak cylinder pressure, Pa
        fmep = (0.60e5 + 0.006 * p_max
                + 3.0e3 * mps + 0.25e3 * mps ** 2) * h.friction_mod
        # Cold, thick oil costs more friction. Real and worth having: it is why a
        # cold engine makes less power and why oil degradation shows up as loss.
        oil_visc_factor = self._oil_viscosity_factor(
            self.oil_k if not warm else 363.15)
        # Cold oil costs friction, but modestly: a cold engine makes perhaps 20%
        # less power, not 90% less.
        fmep *= (0.88 + 0.12 * oil_visc_factor)
        # Degraded oil loses its additive package and its film strength, which
        # is a bounded friction penalty of up to about a third.
        fmep *= 1.0 + 0.35 * max(0.0, 1.0 - h.oil_quality_mod)

        friction_power = fmep * s.displacement_m3 * (rpm / 60.0) / 2.0

        # Pumping loss: work spent drawing the charge in below exhaust
        # back-pressure. Zero at wide open throttle, largest at idle. Kept
        # apart from friction because it heats the gas, not the oil and walls.
        pmep = max(0.0, EXHAUST_BACKPRESSURE * p_amb - map_pa)
        pumping_power = pmep * s.displacement_m3 * (rpm / 60.0) / 2.0

        brake_power = max(0.0, indicated_power - friction_power - pumping_power)
        brake_torque = brake_power / max(omega, 1.0)

        # Exhaust: whatever heat did not become work. Pumping work is routed to
        # the walls in `_thermal`, not here: added to the exhaust stream it is
        # divided by exhaust mass flow, which is smallest exactly when pumping
        # is largest, and it pushed low-power EGT up by ~100 C - over the 912's
        # 880 C limit at 38 kPa MAP. The EGT model's fixed probe fraction has no
        # pipe heat loss to take that back out, which a real probe does.
        exhaust_power = q_in - indicated_power
        m_dot_exh = m_dot_air + m_dot_fuel
        egt_rise = exhaust_power / max(m_dot_exh * 1150.0, 1e-6)   # cp of products
        # Fraction of the exhaust enthalpy rise still present at the EGT probe,
        # which sits downstream of the port after real heat loss to the head and
        # pipe. CALIBRATED so a healthy cruise reads about 85% of the certified
        # EGT limit: at 0.55 the model put a healthy 914 at 955 degC against a
        # 950 degC limit, which drove its health index to zero at normal cruise.
        egt_target_k = t_in + egt_rise * 0.47

        return {
            "rpm": rpm, "p_amb": p_amb, "t_amb": t_amb, "rho_amb": rho_amb,
            "map_pa": map_pa, "boost_pr": boost_pr, "charge_temp_k": t_in,
            "m_dot_air": m_dot_air, "m_dot_fuel": m_dot_fuel, "lambda": lam,
            "ve": ve, "eta_comb": eta_comb, "eta_th": eta_th, "retard": retard,
            "knock_demand_raw": knock_demand,
            "indicated_power_w": indicated_power, "friction_power_w": friction_power,
            "pumping_power_w": pumping_power,
            "brake_power_w": brake_power, "brake_torque_nm": brake_torque,
            "exhaust_power_w": exhaust_power, "egt_target_k": egt_target_k,
            "fmep_pa": fmep, "wastegate": wg_frac,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _oil_viscosity_factor(oil_k: float, quality_mod: float = 1.0) -> float:
        """Relative viscosity, 1.0 at 90 degC, bounded.

        TEMPERATURE ONLY. An earlier version also divided by the oil-quality
        modifier, which compounded a cold engine with degraded oil into a
        viscosity factor of 23 and a friction multiplier near 9x - enough that
        the engine could not overcome its own friction and the equilibrium solve
        settled at idle with zero power. Oil condition is a separate, bounded
        friction penalty (see `_cycle`), not a viscosity multiplier.

        The bound matters on its own: real multigrade oil at 20 degC is roughly
        three times its 90 degC viscosity, not twenty.
        """
        tc = oil_k - 273.15
        v = math.exp(-0.045 * (tc - 90.0))
        return float(min(max(v, 0.45), 3.0))

    # ------------------------------------------------------------------
    # L10 - ECU. The boost limit is held in `_cycle`: the wastegate bypasses
    # whatever compressor capacity exceeds the limit, and a worn compressor is
    # compensated until the wastegate runs out of travel. That is modelled as an
    # ideal wastegate rather than a PI loop because a PI loop never acted here -
    # MAP was clamped to the limit before the controller saw it, so its error was
    # never positive and it output zero on every step. At the 1 Hz rate the data
    # is recorded, a real loop settles well inside one sample anyway.
    # ------------------------------------------------------------------
    # L6 / L7 / L8 - thermal, lubrication, electrical
    # ------------------------------------------------------------------
    def _thermal(self, cyc: dict, u: Inputs, h: Health) -> None:
        s, dt = self.spec, self.dt

        # Heat split. Roughly a third of released heat goes to the walls in a
        # spark-ignition engine; the coolant and oil share it.
        q_wall = (0.32 * cyc["exhaust_power_w"] + 0.55 * cyc["friction_power_w"]
                  + cyc["pumping_power_w"])

        # Cooling capacity rises with airspeed (ram) and with coolant-to-ambient
        # difference, and falls with radiator fouling.
        ram = (0.25 + 0.75 * min(u.airspeed_ms / 50.0, 1.6)) * u.cooling_airflow_factor
        # CALIBRATED so a warm cruise sits where the Rotax operating manual puts
        # it - coolant about 90 degC, oil about 100 degC, head about 110 degC -
        # rather than at the unrealistically cool 60 degC the first sizing gave.
        # These are heat-transfer coefficients of the installed radiator and oil
        # cooler, which are airframe properties and not published by the engine
        # manufacturer, so they have to be set this way.
        # The cooling system is sized for the engine it is installed behind: a
        # 160 hp engine carries a bigger radiator and oil cooler than a 100 hp
        # one. Without this scaling the 916 ran to 168 degC CHT against a 135
        # degC limit, purely because it inherited the 914's radiator.
        size = s.rated_power_w / 84_500.0          # 914 is the calibration point
        ua_cool = UA_COOL * size * ram * h.cooling_mod
        # OIL THERMOSTAT. A cooling system certified for a hot-day full-power
        # climb has far more capacity than a standard-day cruise needs, and
        # without regulation it ran cruise oil at 73-82 C, below the 90-110 C
        # Rotax wants. The thermostat bypasses the cooler until the oil is warm
        # and opens progressively above that. `oil_thermostat_open=False` now
        # means a thermostat STUCK in bypass, which is a real failure.
        bypass = 25.0 / 70.0
        if u.oil_thermostat_open:
            x = (self.oil_k - 273.15 - OIL_TSTAT_OPEN_C) / (OIL_TSTAT_FULL_C - OIL_TSTAT_OPEN_C)
            opening = bypass + (1.0 - bypass) * min(1.0, max(0.0, x))
        else:
            opening = bypass
        ua_oil = UA_OIL * opening * size * ram * h.cooling_mod

        q_cool_out = ua_cool * (self.coolant_k - cyc["t_amb"])
        self.coolant_k += dt * (0.72 * q_wall - q_cool_out) / s.coolant_mass_j_per_k

        # Head runs hotter than coolant, coupled through the water jacket.
        ua_head = UA_HEAD * size
        q_head_out = ua_head * (self.cht_k - self.coolant_k)
        self.cht_k += dt * (0.28 * q_wall - q_head_out) / s.cht_mass_j_per_k

        # Oil picks up friction heat directly and is cooled by its own cooler.
        q_oil_in = 0.45 * cyc["friction_power_w"] + 0.10 * q_wall
        q_oil_out = ua_oil * (self.oil_k - cyc["t_amb"]) + 120.0 * (self.oil_k - self.coolant_k)
        self.oil_k += dt * (q_oil_in - q_oil_out) / s.oil_mass_j_per_k

        # EGT is a fast first-order lag on the computed exhaust temperature.
        tau_egt = 4.0
        self.egt_k += dt * (cyc["egt_target_k"] - self.egt_k) / tau_egt

        for name in ("coolant_k", "cht_k", "oil_k", "egt_k"):
            setattr(self, name, min(max(getattr(self, name), 220.0), 1400.0))

    def _oil_pressure_pa(self, cyc: dict, h: Health) -> float:
        """Gear-pump delivery against bearing clearance, through hot oil.

        Pressure is pump flow divided by leakage area, so it rises with speed,
        rises with viscosity (cold oil) and falls as bearing clearances open up
        with wear - three effects v3 represented with a single multiplier.
        """
        s = self.spec
        rpm = cyc["rpm"]
        visc = self._oil_viscosity_factor(self.oil_k)

        # A gear pump delivers flow proportional to speed; that flow leaks
        # through the bearing clearances, and pressure is what it takes to push
        # it through. So pressure rises with speed, rises with viscosity (cold
        # oil) and FALLS as clearances open with wear - three separate effects
        # that v3 collapsed into one multiplier.
        speed_frac = rpm / s.rated_rpm
        # Wear opens clearances. friction_mod above 1 means more wear.
        wear = max(0.0, h.friction_mod - 1.0) + max(0.0, 1.0 - h.oil_pump_mod)
        clearance = 1.0 + 1.9 * wear

        # CALIBRATED: 4.2 bar at rated speed on 90 degC oil, which is mid-range
        # for the Rotax green arc (2-5 bar), with the relief valve at 7 bar and
        # a 0.8 bar floor at idle.
        # Thin, degraded oil leaks past the same clearance more readily, so it
        # shows as lost pressure - the classic signature of oil past its life.
        quality = max(h.oil_quality_mod, 0.25)
        p = 4.2e5 * speed_frac ** 0.85 * visc ** 0.45 * quality ** 0.35 / clearance
        return min(max(p, 0.8e5), 7.0e5)

    def _electrical(self, cyc: dict, u: Inputs) -> tuple[float, float, float]:
        rpm = cyc["rpm"]
        alt_capacity = 40.0 * min(1.0, max(0.0, (rpm - 1200.0) / 2200.0))  # A
        load = max(0.0, u.electrical_load_a)
        net = alt_capacity - load
        target = 14.2 if net >= 0 else 12.2 + 0.05 * net
        self.battery_v += self.dt * (target - self.battery_v) / 2.0
        return self.battery_v, min(alt_capacity, load), alt_capacity

    # ------------------------------------------------------------------
    # L5 - propeller
    # ------------------------------------------------------------------
    def _propeller(self, cyc: dict, u: Inputs, h: Health, rho: float) -> dict:
        s = self.spec
        prop_rpm = cyc["rpm"] / s.gear_ratio
        n = prop_rpm / 60.0
        d = self.prop_diameter
        j = u.airspeed_ms / max(n * d, 1e-3)          # advance ratio

        # Erosion removes blade area and twist, which costs torque absorbed as
        # well as thrust produced. It previously cut thrust alone, so the engine
        # never felt it and the fault left no residual on any engine channel;
        # with the prop unloading, the engine now speeds up at fixed throttle,
        # which is how a damaged propeller is actually noticed.
        ct = max(0.0, 0.115 - 0.085 * j) * h.prop_eff_mod
        cq = max(1e-4, 0.0165 - 0.0090 * j) * h.prop_eff_mod
        thrust = ct * rho * n ** 2 * d ** 4
        q_prop = cq * rho * n ** 2 * d ** 5           # at the prop shaft
        return {"prop_rpm": prop_rpm, "advance_ratio": j,
                "thrust_n": thrust, "prop_torque_nm": q_prop}

    # Propeller diameter per engine, sized so each absorbs its own rated torque
    # at rated RPM. CALIBRATED - this is the same failure v3 had to fix, where
    # one 2.0 m propeller was shared by engines making 90 to 193 N.m.
    PROP_D = {"Rotax_912_ULS": 1.78, "Rotax_914_ULF": 1.83,
              "Rotax_915_iS": 1.93, "Rotax_916_iS": 1.98}

    @property
    def prop_diameter(self) -> float:
        return self.PROP_D[self.engine_model]

    # ------------------------------------------------------------------
    # L5b - AERODYNAMICS. Carried over from v3, which had a working airframe
    # model that v4 would otherwise have thrown away. Kept because the thrust
    # and lift margins are what turn an engine fault into a MISSION
    # consequence, which is the whole point of a UAV digital twin.
    # ------------------------------------------------------------------
    def _aero(self, u: Inputs, rho: float, thrust_n: float) -> dict:
        s = self.spec
        v = max(u.airspeed_ms, 1.0)
        q = 0.5 * rho * v * v
        aoa = u.aoa_deg
        # Lift curve with a stall break, as v3 had it.
        cl = 0.0875 * aoa if aoa <= 12.0 else max(0.25, 1.15 - 0.055 * (aoa - 12.0))
        cd = s.cd0 + s.cd_k * cl * cl
        lift = cl * q * s.wing_area_m2
        drag = cd * q * s.wing_area_m2
        weight = s.mass_kg * G0
        return {"lift": lift, "drag": drag,
                "thrust_margin": (thrust_n - drag) / max(drag, 1e-6),
                "lift_weight_margin": (lift - weight) / weight}

    # ------------------------------------------------------------------
    # L12 - VIBRATION. Carried over from v3's three-axis channels, but driven
    # by physical causes rather than injected directly: rotating imbalance from
    # wear, combustion roughness from misfire or poor burn, and the engine's
    # own firing order. A four-cylinder four-stroke fires twice per revolution,
    # so second order dominates and shows up on the vertical axis.
    # ------------------------------------------------------------------
    def _vibration(self, cyc: dict, h: Health) -> tuple[float, float, float]:
        rpm = cyc["rpm"]
        base = 0.035 + 0.055 * (rpm / self.spec.rated_rpm) ** 2

        imbalance = max(0.0, h.friction_mod - 1.0) + max(0.0, 1.0 - h.prop_eff_mod)
        roughness = (max(0.0, 1.0 - h.combustion_eff_mod)
                     + max(0.0, 1.0 - h.ignition_mod)
                     + 0.5 * max(0.0, 1.0 - h.compression_mod))

        # RMS AMPLITUDE PER AXIS, which is what an engine vibration monitor
        # reports. The previous channels were the instantaneous value of a
        # ~90 Hz sine sampled at 1 Hz - aliased, so each row carried a
        # pseudo-random point in [-A, A] rather than the amplitude A itself.
        # Components at different orders add in quadrature; the two vertical
        # components share the firing frequency, so their 0.7 rad phase offset
        # is kept.
        a1 = base * (1.0 + 6.0 * imbalance)
        vx = math.sqrt(a1 ** 2 + (0.4 * roughness) ** 2) / math.sqrt(2.0)
        vy = vx
        a2, b2 = base * (1.2 + 4.0 * imbalance), 0.9 * roughness
        vz = math.sqrt(a2 ** 2 + b2 ** 2 + 2.0 * a2 * b2 * math.cos(0.7)) / math.sqrt(2.0)
        return vx, vy, vz

    # ------------------------------------------------------------------
    # L11 - OPERABILITY MARGINS AND HEALTH INDEX.
    #
    # This is the C-MAPSS health-index construction transplanted onto a piston
    # engine, and it is the most valuable single idea carried over. Each margin
    # is normalised to [0,1]: 1 when the engine is at its nominal operating
    # point, 0 when it has reached a certified limit. Health is their MINIMUM,
    # so whichever limit the engine is closest to violating governs - and a
    # failure is defined as reaching a real limit, not as a wear counter hitting
    # a number. v3 had no equivalent: its health was the wear scalar itself,
    # which meant health could not respond to a hot day or a fouled radiator.
    # ------------------------------------------------------------------
    def margins(self, out: dict) -> dict:
        s = self.spec

        def up(val, nom, lim):      # margin that closes as the value RISES
            return float(min(max((lim - val) / max(lim - nom, 1e-6), 0.0), 1.0))

        def down(val, nom, lim):    # margin that closes as the value FALLS
            return float(min(max((val - lim) / max(nom - lim, 1e-6), 0.0), 1.0))

        low = out["engine_rpm"] < s.oil_press_split_rpm
        m = {
            "cht": up(out["cht"], s.cht_nominal_c, s.cht_limit_c),
            "egt": up(out["egt"], s.egt_nominal_c, s.egt_limit_c),
            "oil_temp": up(out["oil_temp"], s.oil_temp_nominal_c, s.oil_temp_limit_c),
            "oil_press": down(out["oil_pressure"],
                              s.oil_press_nominal_low_bar if low else s.oil_press_nominal_bar,
                              s.oil_press_min_low_bar if low else s.oil_press_min_bar),
        }
        m["health_index"] = min(m.values())
        return m

    def warm_start(self, u: "Inputs | None" = None, h: "Health | None" = None) -> None:
        """Initialise to a typical warmed-through state.

        Dataset generation ran a 900-step warm-up per scenario, twice over
        (engine plus twin), purely to get the thermal states off ambient. That
        was the single largest cost in generation and produced nothing that is
        recorded. Seeding the states at their warm values and letting a short
        settle finish the job gives the same starting condition for a fraction
        of the work.
        """
        self.cht_k = 273.15 + 108.0
        self.coolant_k = 273.15 + 88.0
        self.oil_k = 273.15 + 96.0
        self.egt_k = 273.15 + 790.0
        self.battery_v = 14.1
        if u is not None:
            p_amb, t_amb, rho, _ = atmosphere(
                u.altitude_m, u.isa_dev_c, u.qnh_offset_pa, u.humidity_frac)
            self.omega = self._equilibrium_omega(u, h or Health(), rho)

    # ------------------------------------------------------------------
    # Shaft equilibrium, for coarse time steps.
    #
    # Explicitly integrating the shaft ODE needs dt of order 0.01 s to stay
    # stable, because the inertia is small and the torques are large. Dataset
    # generation records at 1 Hz, and substepping 100x to reach it would cost
    # roughly ten hours per engine for no benefit: a 1 Hz sample cannot resolve
    # a 1.5 s shaft transient anyway.
    #
    # So at coarse dt the shaft is solved for the speed where engine torque
    # balances propeller load - which is where it would settle - and then
    # relaxed toward that speed with the real first-order time constant. The
    # transient is preserved at exactly the resolution the data records it, and
    # the solve is unconditionally stable.
    # ------------------------------------------------------------------
    SHAFT_TAU = 1.5             # s, measured from the explicit model

    def _torque_imbalance(self, u: Inputs, h: Health, omega: float,
                          rho: float) -> float:
        cyc = self._cycle(u, h, omega)
        prop = self._propeller(cyc, u, h, rho)
        return cyc["brake_torque_nm"] - prop["prop_torque_nm"] / self.spec.gear_ratio

    def _equilibrium_omega(self, u: Inputs, h: Health,
                           rho: float, iters: int = 24) -> float:
        """Shaft speed where engine torque equals propeller load.

        Imbalance falls monotonically with speed - the propeller absorbs torque
        as the square of speed while engine torque is flat or falling - so
        bisection is well posed and cannot diverge.
        """
        s = self.spec
        lo = s.idle_rpm * 2 * math.pi / 60.0 * 0.5
        hi = s.max_rpm * 1.15 * 2 * math.pi / 60.0
        f_lo = self._torque_imbalance(u, h, lo, rho)
        if f_lo <= 0.0:
            return lo
        f_hi = self._torque_imbalance(u, h, hi, rho)
        if f_hi >= 0.0:
            return hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self._torque_imbalance(u, h, mid, rho) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.5:
                break
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------
    # Public step
    # ------------------------------------------------------------------
    def step(self, u: Inputs | None = None, h: Health | None = None) -> dict:
        u = u or Inputs()
        h = h or Health()
        s, dt = self.spec, self.dt

        cyc = self._cycle(u, h, self.omega)
        prop = self._propeller(cyc, u, h, cyc["rho_amb"])

        # -- shaft dynamics ---------------------------------------------------
        if dt >= 0.25:
            # Coarse step: solve for equilibrium, relax toward it. Stable at any
            # dt and preserves the transient at the recorded resolution.
            omega_eq = self._equilibrium_omega(u, h, cyc["rho_amb"])
            alpha = 1.0 - math.exp(-dt / self.SHAFT_TAU)
            self.omega += alpha * (omega_eq - self.omega)
        else:
            load_at_crank = prop["prop_torque_nm"] / s.gear_ratio
            net_torque = cyc["brake_torque_nm"] - load_at_crank
            self.omega += dt * net_torque / self.inertia
        self.omega = min(max(self.omega, s.idle_rpm * 2 * math.pi / 60.0 * 0.5),
                         s.max_rpm * 1.15 * 2 * math.pi / 60.0)

        # Re-evaluate at the settled speed so the reported row is self-consistent:
        # torque, fuel and temperatures all belong to the same shaft speed.
        if dt >= 0.25:
            cyc = self._cycle(u, h, self.omega)
            prop = self._propeller(cyc, u, h, cyc["rho_amb"])
        self.wastegate = cyc["wastegate"]

        self._thermal(cyc, u, h)
        oil_p = self._oil_pressure_pa(cyc, h)
        v_bus, i_alt, alt_cap = self._electrical(cyc, u)
        aero = self._aero(u, cyc["rho_amb"], prop["thrust_n"])
        vx, vy, vz = self._vibration(cyc, h)

        # -- L13 life accounting -------------------------------------------
        # Consumption is duty-weighted rather than wall-clock: an engine held at
        # high power and high temperature uses its life faster than one loafing,
        # which is the behaviour the C-MAPSS damage model assumes and v3 only
        # approximated.
        duty = (0.45 * (cyc["rpm"] / s.rated_rpm) ** 2
                + 0.35 * min(cyc["brake_power_w"] / s.rated_power_w, 1.5)
                + 0.20 * max(0.0, (self.cht_k - 273.15) / s.cht_limit_c))
        self.life_used_h += dt / 3600.0 * max(0.25, duty)

        self.t += dt
        rho_fuel = 720.0    # kg/m^3

        return {
            # -- bookkeeping --------------------------------------------------
            "time": round(self.t, 4),
            "elapsed_hours": round(self.t / 3600.0, 6),
            "physics_version": self.PHYSICS_VERSION,
            "engine_model": self.engine_model,
            "tbo_hours": s.tbo_hours,

            # -- L0/L1 environment --------------------------------------------
            "altitude": u.altitude_m, "airspeed": u.airspeed_ms, "aoa": u.aoa_deg,
            "throttle": u.throttle, "air_density": round(cyc["rho_amb"], 5),
            "ambient_temp_c": round(cyc["t_amb"] - 273.15, 3),
            "ambient_pressure_pa": round(cyc["p_amb"], 1),
            "isa_dev_c": u.isa_dev_c, "humidity_frac": u.humidity_frac,

            # -- L2 induction --------------------------------------------------
            "manifold_pressure_kpa": round(cyc["map_pa"] / 1000.0, 3),
            "boost_pressure_ratio": round(cyc["boost_pr"], 4),
            "charge_temp_c": round(cyc["charge_temp_k"] - 273.15, 2),
            "air_mass_flow_kgs": round(cyc["m_dot_air"], 6),
            "volumetric_efficiency": round(cyc["ve"], 4),
            "wastegate_position": round(self.wastegate, 4),

            # -- L3 combustion -------------------------------------------------
            "fuel_flow": round(cyc["m_dot_fuel"] / rho_fuel * 3.6e6, 4),   # L/h
            "fuel_flow_kgs": round(cyc["m_dot_fuel"], 6),
            "lambda": round(cyc["lambda"], 4),
            "combustion_efficiency": round(cyc["eta_comb"], 4),
            "thermal_efficiency": round(cyc["eta_th"], 4),
            "knock_retard": round(cyc["retard"], 4),
            "injection_timing": round(12.0 - 8.0 * cyc["retard"], 3),

            # -- L4 mechanical -------------------------------------------------
            "engine_rpm": round(cyc["rpm"], 2),
            "torque_nm": round(cyc["brake_torque_nm"], 3),
            "power_kw": round(cyc["brake_power_w"] / 1000.0, 4),
            "indicated_power_kw": round(cyc["indicated_power_w"] / 1000.0, 4),
            "friction_power_kw": round(cyc["friction_power_w"] / 1000.0, 4),
            "fmep_bar": round(cyc["fmep_pa"] / 1e5, 4),

            # -- L5 drivetrain / propeller -------------------------------------
            "prop_rpm": round(prop["prop_rpm"], 2),
            "prop_torque": round(prop["prop_torque_nm"], 3),
            "advance_ratio": round(prop["advance_ratio"], 4),
            "thrust": round(prop["thrust_n"], 3),

            # -- L6 thermal -----------------------------------------------------
            "egt": round(self.egt_k - 273.15, 2),
            "cht": round(self.cht_k - 273.15, 2),
            "coolant_temp": round(self.coolant_k - 273.15, 2),
            "oil_temp": round(self.oil_k - 273.15, 2),

            # -- L7 lubrication --------------------------------------------------
            "oil_pressure": round(oil_p / 1e5, 4),          # bar

            # -- L8 electrical ---------------------------------------------------
            "battery_voltage": round(v_bus, 3),
            "alternator_current": round(i_alt, 2),
            "alternator_capacity": round(alt_cap, 2),

            # -- L5b aerodynamics (from v3) --------------------------------------
            "lift": round(aero["lift"], 3), "drag": round(aero["drag"], 3),
            "thrust_margin": round(aero["thrust_margin"], 5),
            "lift_weight_margin": round(aero["lift_weight_margin"], 5),

            # -- L12 vibration (from v3, now physically driven) ------------------
            "vibx": round(vx, 5), "viby": round(vy, 5), "vibz": round(vz, 5),

            # -- L13 life ---------------------------------------------------------
            "life_used_hours": round(self.life_used_h, 6),
            "rul_hours_true": round(max(0.0, s.tbo_hours - self.life_used_h), 4),

            # -- L9 health (echoed so every row carries its own ground truth) ----
            **{f"h_{k}": round(v, 5) for k, v in h.as_dict().items()},
        }
