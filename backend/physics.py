"""
UAV Piston Engine Digital Twin -- Python/NumPy port of the validated Simulink model
(UAV_Piston_Engine.slx). Every formula below was extracted directly from the Simulink
block diagram (not re-derived), so this is a faithful 1:1 port of the golden model.

Multi-engine support: ENGINE_CONFIGS holds the full spec set for each selectable
engine. Each selectable engine has its own trained Detection/Diagnosis/Severity/RUL
model set and scaler under backend/models, so the AI service can switch models with
the physics engine instead of silently reusing 914 weights.
"""
import numpy as np

from failure_modes import FailureModes


# Which physics generation the twin runs by default.
#
# v3 deliberately CHANGES the observable distribution (real thermal calibration,
# plus wear that is actually visible in the sensors), which invalidates every
# model trained on v2 data (backend/models/). The retrained v3 models are deployed
# in backend/models_v3/, so v3 is the default; pass physics_version="v2" for legacy.
DEFAULT_PHYSICS_VERSION = "v3"

# Time Between Overhauls, real engine hours. This is the scale the v3 RUL label
# is expressed against: rul_hours_true = TBO_HOURS - accumulated_hours.
#
# TODO CONFIRM against Rotax service documentation before generating the final
# dataset. These are plausible published figures but are NOT yet sourced, and
# they set the scale everything downstream trains on.
TBO_HOURS = {
    "Rotax_912_ULS": 2000.0,
    "Rotax_914_ULF": 2000.0,
    "Rotax_915_iS":  1200.0,
    "Rotax_916_iS":  2000.0,   # launched with a 2,000 h TBO; the 915 iS carries ~1,200 h
}

ENGINE_CONFIGS = {
    "Rotax_914_ULF": {
        "RPM_POINTS": [4300, 4800, 5000, 5500, 5800],
        "TORQUE_POINTS": [90, 95, 105, 128, 139],
        "GEAR_RATIO": 2.43, "BSFC": 0.300, "INV_INERTIA": 2.0,
        "RPM_MAX": 5800.0, "RPM_MIN": 1400.0,
        "D_PROP": 2.0, "CT": 0.05, "CQ": 0.005,
        "THRUST_SAT": (0.0, 2000.0), "PROPTORQUE_SAT": (0.0, 400.0),
        "S_WING": 25.0, "AIRCRAFT_MASS_KG": 450.0, "G": 9.81,
        "AOA_POINTS": [-15,-10,-5,0,5,10,12,15,20,25],
        "CL_POINTS": [-0.8,-0.5,-0.25,0.25,0.686,1.123,1.15,1.0,0.7,0.4],
        "CD0": 0.03, "CD_K": 0.045,
    },
    "Rotax_912_ULS": {
        "RPM_POINTS": [4300, 4800, 5000, 5500, 5800],
        "TORQUE_POINTS": [84.3, 88.7, 97.4, 119.8, 121.0],
        "GEAR_RATIO": 2.43, "BSFC": 0.300, "INV_INERTIA": 2.0,
        "RPM_MAX": 5800.0, "RPM_MIN": 1400.0,
        "D_PROP": 2.0, "CT": 0.05, "CQ": 0.005,
        "THRUST_SAT": (0.0, 2000.0), "PROPTORQUE_SAT": (0.0, 400.0),
        "S_WING": 25.0, "AIRCRAFT_MASS_KG": 450.0, "G": 9.81,
        "AOA_POINTS": [-15,-10,-5,0,5,10,12,15,20,25],
        "CL_POINTS": [-0.8,-0.5,-0.25,0.25,0.686,1.123,1.15,1.0,0.7,0.4],
        "CD0": 0.03, "CD_K": 0.045,
    },
    "Rotax_915_iS": {
        "RPM_POINTS": [4300, 4800, 5000, 5500, 5800],
        "TORQUE_POINTS": [170.0, 180.0, 185.0, 172.0, 171.0],
        "GEAR_RATIO": 2.54, "BSFC": 0.300, "INV_INERTIA": 2.0,
        "RPM_MAX": 5800.0, "RPM_MIN": 1400.0,
        "D_PROP": 2.0, "CT": 0.05, "CQ": 0.005,
        "THRUST_SAT": (0.0, 2000.0), "PROPTORQUE_SAT": (0.0, 400.0),
        "S_WING": 25.0, "AIRCRAFT_MASS_KG": 450.0, "G": 9.81,
        "AOA_POINTS": [-15,-10,-5,0,5,10,12,15,20,25],
        "CL_POINTS": [-0.8,-0.5,-0.25,0.25,0.686,1.123,1.15,1.0,0.7,0.4],
        "CD0": 0.03, "CD_K": 0.045,
    },
    "Rotax_916_iS": {
        "RPM_POINTS": [4300, 4800, 5000, 5500, 5800],
        "TORQUE_POINTS": [150.0, 165.0, 180.0, 175.0, 193.0],
        "GEAR_RATIO": 2.54, "BSFC": 0.300, "INV_INERTIA": 2.0,
        "RPM_MAX": 5800.0, "RPM_MIN": 1400.0,
        "D_PROP": 2.0, "CT": 0.05, "CQ": 0.005,
        "THRUST_SAT": (0.0, 2000.0), "PROPTORQUE_SAT": (0.0, 400.0),
        "S_WING": 25.0, "AIRCRAFT_MASS_KG": 450.0, "G": 9.81,
        "AOA_POINTS": [-15,-10,-5,0,5,10,12,15,20,25],
        "CL_POINTS": [-0.8,-0.5,-0.25,0.25,0.686,1.123,1.15,1.0,0.7,0.4],
        "CD0": 0.03, "CD_K": 0.045,
    },
}


# Physics v3 propeller per engine. v2 keeps the shared ENGINE_CONFIGS propeller
# bit-for-bit (the live demo and backend/models/ were built on it).
#
# All four engines shared one 2.0 m prop sized for the 914. The 915/916 make 170-193 Nm
# against 90-139 Nm, so that prop could not absorb their torque: they pinned at the
# 5,800 rpm redline from 57%/51% throttle at 1,500 m, which drove oil-pressure and
# vibration stress to 100% and health to 0% at ordinary cruise.
#
# Sized by steady-state torque balance to match the 914 on the threshold that matters -
# the throttle at which rpm reaches 85% of redline and _conditions() starts vibration
# stress. 914: 90/78/39% at 0/1,500/8,000 m; 915 at 2.30 m: 89/77/38%; 916 at 2.28 m:
# 89/77/39%. (Matching redline throttle instead left both faulting from ~55-65%,
# because their torque peaks mid-range.) Full throttle then absorbs ~450 Nm and makes
# ~2,000 N, so the torque and thrust caps are raised for these two to avoid clipping.
V3_PROP_OVERRIDES = {
    # 912: makes less torque than the 914 on the same 2.0 m prop, so it reached the
    # vibration-stress rpm only at 97/84/42% throttle (0/1,500/8,000 m) and got fewer
    # faults. 1.97 m matches the 914 exactly: 90/78/39%.
    "Rotax_912_ULS": {"D_PROP": 1.97},
    "Rotax_915_iS": {"D_PROP": 2.30, "PROPTORQUE_SAT": (0.0, 600.0), "THRUST_SAT": (0.0, 2500.0)},
    "Rotax_916_iS": {"D_PROP": 2.28, "PROPTORQUE_SAT": (0.0, 600.0), "THRUST_SAT": (0.0, 2500.0)},
}


class UAVEngineTwin:
    FAULT_CHANNELS = ["egt","cht","oil_pressure","oil_temp","vibx","viby","vibz","rpm"]

    PHYSICS_VERSIONS = ("v2", "v3")

    def __init__(self, dt=0.01, engine_model="Rotax_914_ULF", physics_version=None):
        if engine_model not in ENGINE_CONFIGS:
            raise ValueError(f"Unknown engine_model {engine_model!r}. Options: {list(ENGINE_CONFIGS)}")
        self.engine_model = engine_model
        cfg = ENGINE_CONFIGS[engine_model]

        self.physics_version = physics_version or DEFAULT_PHYSICS_VERSION
        if self.physics_version not in self.PHYSICS_VERSIONS:
            raise ValueError(f"Unknown physics_version {self.physics_version!r}. "
                             f"Options: {list(self.PHYSICS_VERSIONS)}")
        _v3 = self.physics_version == "v3"
        _g = 1.0 if _v3 else 0.0      # gates every v3-only term to exactly zero
        self.TBO_HOURS = TBO_HOURS[engine_model]

        # Former class-level constants, now per-instance so each twin can run a
        # DIFFERENT engine model simultaneously if needed (e.g. comparing two at once).
        self.RPM_POINTS = np.array(cfg["RPM_POINTS"], dtype=float)
        self.TORQUE_POINTS = np.array(cfg["TORQUE_POINTS"], dtype=float)
        self.GEAR_RATIO = cfg["GEAR_RATIO"]
        self.BSFC = cfg["BSFC"]
        self.INV_INERTIA = cfg["INV_INERTIA"]
        self.RPM_MAX = cfg["RPM_MAX"]
        self.RPM_MIN = cfg["RPM_MIN"]
        self.D_PROP = cfg["D_PROP"]
        self.CT = cfg["CT"]
        self.CQ = cfg["CQ"]
        self.THRUST_SAT = cfg["THRUST_SAT"]
        self.PROPTORQUE_SAT = cfg["PROPTORQUE_SAT"]
        self.S_WING = cfg["S_WING"]
        self.AIRCRAFT_MASS_KG = cfg["AIRCRAFT_MASS_KG"]
        self.G = cfg["G"]
        self.AOA_POINTS = np.array(cfg["AOA_POINTS"], dtype=float)
        self.CL_POINTS = np.array(cfg["CL_POINTS"], dtype=float)
        self.CD0 = cfg["CD0"]
        self.CD_K = cfg["CD_K"]
        if _v3:
            for key, value in V3_PROP_OVERRIDES.get(engine_model, {}).items():
                setattr(self, key, value)

        # Real max power (kW), computed properly across the whole torque curve rather
        # than assumed at redline - power = torque*rpm/9549 can peak at a MID-range
        # RPM point even if torque itself peaks elsewhere. Used to normalize fault-
        # trigger thresholds (_conditions/_update_wear) so they scale correctly per
        # engine instead of the old hardcoded "85.0" (914-specific) constant.
        power_at_points = self.TORQUE_POINTS * self.RPM_POINTS / 9549.0
        self.MAX_POWER_KW = float(np.max(power_at_points))

        self.dt = dt
        self.t = 0.0
        self.omega = 3000.0 * 2*np.pi/60.0  # init condition, matches Simulink Integrator IC

        # Thermal time constants (seconds) - real engines take time to heat up, they don't
        # snap to a steady-state temperature instantly. Larger thermal mass = slower response.
        self.EGT_TAU = 6.0       # exhaust gas: fast response
        self.CHT_TAU = 50.0      # cylinder head: moderate thermal mass
        self.OILTEMP_TAU = 100.0 # oil reservoir: largest thermal mass, slowest
        # WARM-START initial temperatures, deliberately NOT ambient.
        #
        # These feed the AI directly, and the training data (generated from realistic
        # flight profiles) contains essentially no cold-soaked engine: its throttle
        # minimum is 0.118 and CHT sits at its 260 ceiling for ~97% of rows. Starting
        # at 20C put EGT/CHT/oil_temp 4-7 standard deviations BELOW anything the model
        # ever saw, and with EGT_TAU/CHT_TAU/OILTEMP_TAU of 30-100s it stayed there for
        # minutes - which is exactly the window the AI runs in. Confirmed by dumping a
        # real live scaled window: cht z-score -7.3, engine_rpm -5.2, vs ~0 for training
        # windows. The RUL head then extrapolated wildly (~531h instead of ~1.2h).
        #
        # Physically this is also the more honest default: a UAV beginning a monitored
        # flight has completed run-up and taxi, so its engine IS at operating
        # temperature. A true cold-start is a different scenario the model was never
        # trained to assess, so it should not be the default one it is fed.
        self.egt_state = 620.0 if _v3 else 650.0   # typical warmed EGT for the calibration in use
        self.cht_state = 95.0 if _v3 else 260.0    # v2 warm-started AT the sensor ceiling; v3 at a real head temp
        self.oiltemp_state = 85.0   # normal operating oil temperature

        # live-adjustable exogenous inputs (settable at any time from the API)
        self.altitude = 2000.0
        self.throttle = 0.5
        self.airspeed = 40.0
        self.aoa = 5.0
        # ISA temperature deviation, degrees C. 0.0 is a standard day and
        # reproduces this model's behaviour exactly as it was before hot-weather
        # support existed - every default run is bit-identical to before.
        self.isa_dev_c = 0.0

        # fault config, 8 channels: [EGT,CHT,OilPressure,OilTemp,VibX,VibY,VibZ,RPM]
        # These are AUTO-COMPUTED every step from operating conditions (see _auto_fault_step).
        # They are not meant to be set manually/externally anymore.
        self.fault_type = np.zeros(8)
        self.severity = np.zeros(8)
        self.fault_start = np.full(8, 9999.0)
        self.fault_duration = np.zeros(8)
        self._stuck_vals = np.zeros(8)

        # Auto-fault stress accumulators (0..1 per channel) and hysteresis state
        self.stress = np.zeros(8)
        self._auto_active = np.zeros(8, dtype=bool)
        self.RISE_RATE = 0.04   # stress gained per second while condition is true
        self.DECAY_RATE = 0.08  # stress lost per second while condition is false
        self.TRIGGER_THRESHOLD = 0.6
        self.CLEAR_THRESHOLD = 0.3

        # --- Irreversible wear / RUL model ---
        # Unlike stress (reversible, decays when conditions improve), wear only ever
        # accumulates. Driven by the SAME stress conditions (sustained high power, low
        # cooling, high RPM) but never recovers - this is what makes RUL meaningful:
        # a "distance to failure" that actually gets closer over time.
        self.wear = 0.0            # 0 = new engine, 1 = failed
        self.WEAR_K = 0.000179      # calibrated empirically against actual scenario severity

        # --- DEGRADATION SIGNATURES (v3) -------------------------------------
        # Until v3, self.wear was written and never read back into ANY sensor
        # channel: a 1,900-hour engine read identically to a new one. Measured
        # correlation of wear with the observables on v2 data was
        # +0.13 / +0.11 / -0.04 - statistically nothing - so the RUL head was
        # asked to infer remaining life from a window carrying no degradation
        # signal, and could only lean on elapsed time.
        #
        # These are the signatures a maintainer actually reads on a worn piston
        # engine. Values are the FULL-LIFE deltas, applied proportionally to wear.
        # All are gated by _g, so under v2 every term is exactly 0.0.
        self.WEAR_OIL_PRESS_FRAC = 0.30 * _g   # bearing clearance -> pressure falls
        self.WEAR_CHT_RISE_C     = 25.0 * _g   # blow-by / deposits -> hotter heads
        self.WEAR_EGT_RISE_C     = 45.0 * _g   # combustion phasing -> hotter exhaust
        self.WEAR_OILTEMP_RISE_C = 10.0 * _g   # friction heat into the oil (15 ran too hot, see OILT_COEF)
        self.WEAR_VIB_FRAC       = 1.50 * _g   # imbalance / bearing wear
        self.WEAR_BSFC_FRAC      = 0.18 * _g   # more fuel for the same power
        self.WEAR_TORQUE_FRAC    = 0.12 * _g   # lost peak torque

        # Thermal calibration. v2's coefficients produced impossible temperatures
        # that the sensor clip then hid: on v2 data `cht` sat at its 260 C clip in
        # 99.8% of rows while the unclipped truth averaged 537 C and peaked at
        # 866 C - three to four times reality for a Rotax, whose heads run about
        # 100-150 C. CHT was therefore a CONSTANT in training and its diagnosis
        # and severity heads learned from a flat line. EGT was pinned 31.8% of
        # the time. v3 puts both in real operating bands.
        if _v3:
            self.EGT_COEF = (350.0, 250.0, 0.030, 1.20)
            self.CHT_COEF = ( 75.0,  30.0, 0.004, 0.25, 0.35)
            self.SENSOR_EGT_MAX = 1000.0   # sensor range; ~950 is the OPERATING limit
            self.SENSOR_CHT_MAX =  200.0   # sensor range; ~135-150 is the operating limit
            # 180, not 150: a hot-day, high-power, worn engine legitimately runs 130-150 C,
            # and a 150 stop then hid Bias faults on top of it (3.4% of rows pinned).
            self.SENSOR_OILT_MAX = 180.0
            # Oil temperature target (base, throttle, rpm, airspeed cooling, ISA).
            # The v2-inherited (60, 30, 0.01, 0.3, 0.8) put 22-24% of FAULT-FREE v3 rows
            # above the 130 C Rotax limit (median 113 C). These give median ~104 C,
            # standard day / low wear ~91-95 C, and ~3.5% above 130 C on hot high-power
            # legs - a Rotax runs 90-110 C normally.
            self.OILT_COEF = (70.0, 30.0, 0.006, 0.3, 0.5)
        else:
            self.EGT_COEF = (300.0, 400.0, 0.050, 5.00)
            self.CHT_COEF = (200.0, 250.0, 0.030, 5.00, 1.80)
            self.SENSOR_EGT_MAX = 950.0
            self.SENSOR_CHT_MAX = 260.0
            self.SENSOR_OILT_MAX = 150.0
            self.OILT_COEF = (60.0, 30.0, 0.01, 0.3, 0.8)

        # --- Electrical + injection (PS section B). Monitor-only channels, except
        # injection_timing which IS a model input (see tf_data_pipeline.py).
        self.BATT_NOMINAL_V = 12.6
        self.ALT_CUTIN_RPM = 2200.0        # alternator starts charging above this
        self.INJ_BASE_DEG = 12.0           # base injection advance, deg BTDC
        self.battery_voltage = self.BATT_NOMINAL_V
        self.battery_current = 0.0
        self.alternator_output = 0.0
        self.injection_timing = self.INJ_BASE_DEG

        # Engine failure modes (PS section C). Disabled under v2, so every
        # accessor is an identity and v2 stays bit-identical.
        self.failure_modes = FailureModes(enabled=_v3)

        # Drift faults (type 2) displace a channel by sev*(t - onset), which is
        # UNBOUNDED in time. On oil pressure, whose auto-fault severity is
        # negative, that ran the channel to zero and pinned it there: measured on
        # v3 smoke data, oil_pressure was fault-injected in 22% of rows and sat at
        # exactly 0.0 in 16.2% - an engine reading zero oil pressure would have
        # seized. It also destroyed the best available wear indicator, dropping
        # oil_press_ratio's correlation with wear from -0.9999 on clean rows to
        # -0.14 overall.
        #
        # v3 caps drift displacement at a fraction of the healthy value, so a
        # drifting sensor still degrades badly but stays physically possible.
        # None under v2 keeps the old unbounded behaviour bit-identical.
        self.DRIFT_CAP_FRAC = 0.6 if _v3 else None
        self.failed = False
        self.failure_time = None

        # Stuck-At is a rare independent connector/wiring glitch, not condition-driven
        # (unlike Bias/Drift/Noise which ARE condition-driven). Applied to OilPressure,
        # matching a real failure mode (stuck sender unit).
        self._stuck_active = False
        self._stuck_start = 0.0

    def restore_state(self, snapshot: dict):
        """Restores this twin to the exact operating point captured in a
        final_telemetry snapshot (see db.py/main.py) - used by /resume so
        "continue simulation" genuinely picks up where a past run left off,
        not just the surface params (altitude/throttle/airspeed/aoa) but the
        actual internal physics state: engine angular momentum, the thermal-lag
        integrator states (so CHT/EGT/oil temp do not snap back to cold-start),
        accumulated wear, and per-channel stress accumulators.

        Fault TIMERS (fault_start/fault_duration/_auto_active) are deliberately
        NOT restored - they are momentary activation state, not the underlying
        condition. Stress + operating conditions are restored, so if the engine
        was genuinely stressed when it stopped, faults will naturally re-arm
        within moments of resuming - functionally equivalent without needing to
        perfectly replay exact timer phase, which is not something a real pilot
        continuing a flight would perceive as different anyway."""
        self.t = float(snapshot.get("time", 0.0))
        self.altitude = float(snapshot.get("altitude", self.altitude))
        self.throttle = float(snapshot.get("throttle", self.throttle))
        self.airspeed = float(snapshot.get("airspeed", self.airspeed))
        self.aoa = float(snapshot.get("aoa", self.aoa))
        # Without this a resumed hot-day flight silently reverts to a standard day.
        self.isa_dev_c = float(snapshot.get("isa_dev_c", self.isa_dev_c))
        # Without these a resumed flight silently reverts to a fresh electrical
        # state - the same class of bug as the ISA deviation not being restored.
        self.battery_voltage = float(snapshot.get("battery_voltage", self.battery_voltage))
        self.battery_current = float(snapshot.get("battery_current", self.battery_current))
        self.alternator_output = float(snapshot.get("alternator_output", self.alternator_output))
        self.injection_timing = float(snapshot.get("injection_timing", self.injection_timing))
        self.failure_modes.restore(snapshot.get("failure_modes"))

        engine_rpm = snapshot.get("engine_rpm")
        if engine_rpm is not None:
            self.omega = float(engine_rpm) * 2 * np.pi / 60.0

        healthy = snapshot.get("healthy") or {}
        if "egt" in healthy: self.egt_state = float(healthy["egt"])
        if "cht" in healthy: self.cht_state = float(healthy["cht"])
        if "oil_temp" in healthy: self.oiltemp_state = float(healthy["oil_temp"])

        self.wear = float(snapshot.get("wear", self.wear))
        self.failed = bool(snapshot.get("failed", self.failed))
        self.failure_time = snapshot.get("failure_time", self.failure_time)

        fault_stress = snapshot.get("fault_stress") or {}
        for i, ch in enumerate(self.FAULT_CHANNELS):
            if ch in fault_stress:
                self.stress[i] = float(fault_stress[ch])

    # Air-density envelope actually present in the training data, measured across
    # all four engines' train parquets (12.9M rows for the 914, ~0.8M each for the
    # others): 0.5206 .. 1.2250 kg/m3, spanning 0 .. ~8075 m on a standard day.
    # air_density IS an AI feature, so outside this range the models extrapolate.
    # Clamping isa_dev_c alone does NOT keep density inside it - ISA-30 at sea
    # level reaches 1.368 and ISA+50 at 8000 m reaches 0.44 - so the twin reports
    # the condition instead of pretending it cannot happen.
    TRAINED_DENSITY_MIN = 0.5206
    TRAINED_DENSITY_MAX = 1.2250

    # ---------------- Atmosphere (ISA troposphere model) ----------------
    def atmosphere(self, altitude):
        """ISA troposphere with an optional temperature deviation (hot/cold day).

        The deviation is applied to TEMPERATURE ONLY. Pressure still follows the
        STANDARD temperature ratio, because on a real hot day the pressure at a
        given altitude is unchanged - it is the density that falls, via the ideal
        gas law at the higher temperature. Folding the deviation into the
        pressure ratio as well would double-count it and overstate the effect.

        This is the physically dominant hot-weather mechanism for a piston aero
        engine and the reason 'hot and high' is the classic problem: lower rho
        means less mass flow, so less power, thrust and climb performance.
        """
        T_std = 288.15 - 0.0065*altitude          # standard-day temperature
        T = T_std + self.isa_dev_c                # actual temperature
        P = 101325.0 * (T_std/288.15)**5.256      # pressure follows the STANDARD ratio
        rho = P / (287.05*T)                      # density uses the ACTUAL temperature
        return T, P, rho

    # ---------------- Torque map (Rotax 914 UL/F) ----------------
    def torque_available(self, throttle, engine_rpm):
        thr = np.clip(throttle, 0.0, 1.0)
        rpm_c = np.clip(engine_rpm, self.RPM_POINTS[0], self.RPM_POINTS[-1])
        base_torque = np.interp(rpm_c, self.RPM_POINTS, self.TORQUE_POINTS)
        # Applied HERE rather than to power_kw so the loss propagates physically -
        # through the RK4 RPM dynamics and therefore into prop speed and thrust -
        # the way a genuinely tired engine behaves.
        return (thr * base_torque * (1.0 - self.WEAR_TORQUE_FRAC*self.wear)
                * self.failure_modes.torque_multiplier())

    # ---------------- Propeller ----------------
    def propeller(self, rho, prop_rpm):
        n = prop_rpm/60.0
        thrust = self.CT*rho*n**2*self.D_PROP**4
        prop_torque = self.CQ*rho*n**2*self.D_PROP**5
        thrust = float(np.clip(thrust, *self.THRUST_SAT))
        prop_torque = float(np.clip(prop_torque, *self.PROPTORQUE_SAT))
        return thrust, prop_torque

    # ---------------- Aerodynamics ----------------
    def aerodynamics(self, rho, V, aoa, prop_thrust):
        CL = float(np.interp(aoa, self.AOA_POINTS, self.CL_POINTS))
        CD = self.CD0 + self.CD_K*CL**2
        q = 0.5*rho*V**2
        lift = q*self.S_WING*CL
        drag = q*self.S_WING*CD
        thrust_margin = prop_thrust - drag
        lift_margin = lift - self.AIRCRAFT_MASS_KG*self.G
        return lift, drag, thrust_margin, lift_margin

    # ---------------- Engine shaft dynamics (derivative) ----------------
    def _domega_dt(self, omega, throttle, rho):
        engine_rpm = omega*60.0/(2*np.pi)
        prop_rpm = engine_rpm/self.GEAR_RATIO
        thrust, prop_torque = self.propeller(rho, prop_rpm)
        torque_avail = self.torque_available(throttle, engine_rpm)
        engine_prop_torque = prop_torque/self.GEAR_RATIO  # gear ratio applied ONCE
        net_torque = torque_avail - engine_prop_torque
        domega = self.INV_INERTIA*net_torque
        return domega, torque_avail, engine_rpm, prop_rpm, thrust, prop_torque

    def step(self):
        """Advance by one dt using RK4 (matches Simulink ode4 solver)."""
        throttle = self.throttle
        T, P, rho = self.atmosphere(self.altitude)
        dt = self.dt

        # Advance latent failure modes BEFORE the RK4 so torque_multiplier() is
        # constant across k1..k4 - otherwise a misfire could fire on some stages
        # of the integrator and not others, which is not a physical behaviour.
        # Uses the previous step's operating point: a one-step lag, negligible.
        _prev_rpm = self.omega*60.0/(2*np.pi)
        self.failure_modes.step(dt, self.t,
                                _prev_rpm/self.RPM_MAX,
                                min(1.0, (self._last_power_kw/self.MAX_POWER_KW) if hasattr(self, "_last_power_kw") else 0.0),
                                self.wear)

        k1,_,_,_,_,_ = self._domega_dt(self.omega, throttle, rho)
        k2,_,_,_,_,_ = self._domega_dt(self.omega+0.5*dt*k1, throttle, rho)
        k3,_,_,_,_,_ = self._domega_dt(self.omega+0.5*dt*k2, throttle, rho)
        k4,_,_,_,_,_ = self._domega_dt(self.omega+dt*k3, throttle, rho)
        omega_new = self.omega + (dt/6.0)*(k1+2*k2+2*k3+k4)

        omega_min = self.RPM_MIN*2*np.pi/60.0
        omega_max = self.RPM_MAX*2*np.pi/60.0
        self.omega = float(np.clip(omega_new, omega_min, omega_max))

        _, torque_avail, engine_rpm, prop_rpm, thrust, prop_torque = self._domega_dt(self.omega, throttle, rho)

        power_kw = torque_avail*engine_rpm/9549.0
        # A worn engine burns more fuel for the same shaft power. Note the ABSOLUTE
        # fuel flow can still fall, because the torque derate reduces power; the
        # diagnostic signal is the RATIO (BSFC), which tf_data_pipeline turns into
        # the bsfc_ratio aux feature.
        fuel_flow = (power_kw*self.BSFC*(1.0 + self.WEAR_BSFC_FRAC*self.wear)
                     * self.failure_modes.fuel_multiplier())
        self._last_power_kw = power_kw

        lift, drag, thrust_margin, lift_margin = self.aerodynamics(rho, self.airspeed, self.aoa, thrust)

        # Algebraic steady-state TARGETS (what each thermal signal is heading toward,
        # given current operating conditions -- same formulas as before)
        # Real CHT/OilTemp depend strongly on cooling airflow (airspeed) - confirmed by
        # aviation sources: "reduced cooling airflow increases CHT", shock-cooling/shock-
        # heating during low-airspeed high-power ground ops. Capped since a fixed-size
        # radiator/fin area has diminishing returns at very high speed.
        cool = min(max(self.airspeed, 0.0), 60.0) * self.failure_modes.cooling_multiplier()
        # Ambient offset is applied to EGT and oil temperature only. NOT to CHT:
        # SENSOR_LIMITS clips CHT at 260 and cht_state initializes at exactly that
        # ceiling (it is the value present in ~97% of training rows), so an ambient
        # term there would either be clipped into invisibility or, if the ceiling
        # were raised, push AI feature #17 outside its trained envelope. The
        # dominant hot-day effect is carried by air density in atmosphere(), which
        # is physically correct and costs the model nothing.
        e0, e1, e2, e3 = self.EGT_COEF
        c0, c1, c2, c3, c4 = self.CHT_COEF
        egt_target = (e0 + e1*throttle + e2*engine_rpm + e3*power_kw
                      + 0.6*self.isa_dev_c + self.WEAR_EGT_RISE_C*self.wear
                      + self.failure_modes.egt_delta())
        cool_cht_rise, cool_oil_rise = self.failure_modes.cooling_heat_delta(power_kw)
        cht_target = (c0 + c1*throttle + c2*engine_rpm + c3*power_kw - c4*cool + cool_cht_rise
                      + (0.7 if self.physics_version == "v3" else 0.0)*self.isa_dev_c
                      + self.WEAR_CHT_RISE_C*self.wear)
        o0, o1, o2, o3, o4 = self.OILT_COEF
        oiltemp_target = (o0 + o1*throttle + o2*engine_rpm - o3*cool + cool_oil_rise
                          + o4*self.isa_dev_c + self.WEAR_OILTEMP_RISE_C*self.wear)

        # First-order thermal lag: signal chases the target at a rate set by tau.
        # This is what makes CHT/OilTemp/EGT actually warm up over real time instead
        # of snapping to a value the instant throttle changes.
        self.egt_state     += dt * (egt_target - self.egt_state) / self.EGT_TAU
        self.cht_state      += dt * (cht_target - self.cht_state) / (
            self.CHT_TAU * self.failure_modes.cht_tau_multiplier())
        self.oiltemp_state += dt * (oiltemp_target - self.oiltemp_state) / self.OILTEMP_TAU

        egt, cht, oil_t = self.egt_state, self.cht_state, self.oiltemp_state

        # OilPressure: real engines use a pump + pressure relief valve, so pressure RISES
        # then SATURATES with RPM (not indefinitely linear), and hot oil has lower
        # viscosity -> lower pressure at the same RPM (well-documented: e.g. real GA
        # engines commonly show ~60psi at normal oil temp dropping to ~30-35psi when oil
        # temp is abnormally high). Cold oil (high viscosity) gives a pressure boost.
        oil_p = 90.0*(1.0 - np.exp(-engine_rpm/1500.0)) + 3.0*throttle - 0.15*(self.oiltemp_state - 80.0)
        # Falling oil pressure is the classic wear indicator: increasing bearing
        # clearance lets the pump bypass more flow at the same speed. This exact
        # expression at wear=0 is mirrored by tf_data_pipeline.expected_oil_pressure.
        oil_p *= (1.0 - self.WEAR_OIL_PRESS_FRAC*self.wear)

        # Vibration: rotating-unbalance force scales with the SQUARE of RPM (F=m*r*omega^2),
        # not linearly - this is fundamental rotating-machinery vibration physics.
        rpm_k = (engine_rpm/1000.0)**2
        wear_vib = (1.0 + self.WEAR_VIB_FRAC*self.wear) * self.failure_modes.vib_multiplier()
        vibx = (0.02  + 0.00345*rpm_k) * wear_vib
        viby = (0.025 + 0.00310*rpm_k) * wear_vib
        vibz = (0.03  + 0.00379*rpm_k) * wear_vib

        # --- Electrical + injection (PS section B) ----------------------------
        # Alternator output rises with RPM above cut-in and saturates; the battery
        # charges on the surplus and discharges below cut-in. A worn engine drives
        # its accessories through worn bearings, so output droops slightly.
        # v3: an aero alternator reaches rated output a few hundred rpm above cut-in, not at
        # redline. Ramping to RPM_MAX left cruise (~3000 rpm) at 4.4 A against an 8 A load, so
        # every flight ran the battery down and the bus sat at the discharge voltage. v2 keeps
        # the old ramp so its recorded channels stay bit-identical.
        alt_full_rpm = (self.ALT_CUTIN_RPM + 800.0 if self.physics_version == "v3"
                        else self.RPM_MAX)
        alt_frac = float(np.clip((engine_rpm - self.ALT_CUTIN_RPM) /
                                 max(alt_full_rpm - self.ALT_CUTIN_RPM, 1.0), 0.0, 1.0))
        self.alternator_output = 20.0*alt_frac*(1.0 - 0.15*self.WEAR_VIB_FRAC*self.wear)
        electrical_load = 8.0                      # avionics + payload, amps
        surplus = self.alternator_output - electrical_load
        if self.physics_version == "v3" and surplus > 0.0:
            # A regulator tapers the charge as the battery fills, so the ammeter settles near a
            # small float current instead of sitting at the full surplus for the whole flight.
            surplus *= float(np.clip((14.4 - self.battery_voltage) / 0.8, 0.05, 1.0))
        self.battery_current = surplus
        # Charging pulls the bus up toward regulator voltage; discharging sags it.
        target_v = 14.2 if self.battery_current > 0 else 11.9
        self.battery_voltage += dt*(target_v - self.battery_voltage)/5.0

        # Injection timing: base advance, retarded as load rises (knock margin),
        # and advanced slightly on a hot engine. IS a model input - it carries
        # misfire/injector signatures.
        self.injection_timing = float(np.clip(
            self.INJ_BASE_DEG - 6.0*throttle + 0.02*(cht - 100.0)
            + self.failure_modes.injection_delta(), 0.0, 30.0))

        # Combustion instability shows as cycle-to-cycle RPM scatter.
        rpm_reported = engine_rpm + self.failure_modes.rpm_noise()
        healthy = np.array([egt, cht, oil_p, oil_t, vibx, viby, vibz, rpm_reported])
        self._auto_fault_step(dt, engine_rpm, power_kw, self.airspeed)
        self._update_wear(dt, engine_rpm, power_kw)
        faulty, flags = self._inject_faults(healthy, self.t)

        # Real sensors/gauges have physical saturation limits no matter what is driving
        # the reading (a stuck needle still has a stop; a drifting thermocouple readout
        # still maxes out its display range) - clip so long-duration Drift faults cannot
        # produce physically impossible values (e.g. negative oil pressure, 3000C EGT).
        SENSOR_LIMITS = np.array([
            [0, self.SENSOR_EGT_MAX],   # EGT C - sensor range, not the operating limit
            [0, self.SENSOR_CHT_MAX],   # CHT C - v2 used 260, which the mis-calibrated
                                        # model sat against in 99.8% of rows
            [0, 150],    # OilPressure
            [0, self.SENSOR_OILT_MAX],    # OilTemp C
            [0, 1.0],    # VibX g
            [0, 1.0],    # VibY g
            [0, 1.0],    # VibZ g
            [0, 8000],   # RPM sensor reading (wider than physical max - faulty sensors
                         # commonly DO report implausible values, that's the point of
                         # the fault, but the electronics still saturate somewhere)
        ])
        faulty = np.clip(faulty, SENSOR_LIMITS[:,0], SENSOR_LIMITS[:,1])

        self.t += dt

        return {
            "time": round(self.t,4),
            # Engine hours. self.t stays in SECONDS internally - the RK4 integrator
            # and every tau are defined against it - so hours are DERIVED here
            # rather than changing the clock's units and destabilising the solver.
            # This is the column tf_data_pipeline and telemetry_logs consume.
            "elapsed_hours": round(self.t/3600.0, 6),
            "tbo_hours": self.TBO_HOURS,
            "rul_hours_true": max(0.0, self.TBO_HOURS - self.wear*self.TBO_HOURS),
            "physics_version": self.physics_version,
            **self.failure_modes.as_labels(),
            "battery_voltage": round(self.battery_voltage, 4),
            "battery_current": round(self.battery_current, 4),
            "alternator_output": round(self.alternator_output, 4),
            "injection_timing": round(self.injection_timing, 4),
            "altitude": self.altitude, "air_density": rho,
            "throttle": throttle, "airspeed": self.airspeed, "aoa": self.aoa,
            # Display/telemetry only - deliberately NOT AI features.
            "ambient_temp_c": round(T - 273.15, 3), "isa_dev_c": self.isa_dev_c,
            "density_in_envelope": bool(
                self.TRAINED_DENSITY_MIN <= rho <= self.TRAINED_DENSITY_MAX),
            "torque_available_nm": torque_avail, "engine_rpm": engine_rpm,
            "prop_rpm": prop_rpm, "prop_torque": prop_torque, "power_kw": power_kw,
            "fuel_flow": fuel_flow, "thrust": thrust, "lift": lift, "drag": drag,
            "thrust_margin": thrust_margin, "lift_weight_margin": lift_margin,
            # post-fault (what a real sensor would actually report) - use these for the live API
            "egt": faulty[0], "cht": faulty[1], "oil_pressure": faulty[2], "oil_temp": faulty[3],
            "vibx": faulty[4], "viby": faulty[5], "vibz": faulty[6], "rpm_fault": faulty[7],
            # clean healthy values (no fault applied) - use these for golden-model validation
            "healthy": {
                "egt": healthy[0], "cht": healthy[1], "oil_pressure": healthy[2], "oil_temp": healthy[3],
                "vibx": healthy[4], "viby": healthy[5], "vibz": healthy[6], "rpm": healthy[7],
            },
            "fault_flags": dict(zip(self.FAULT_CHANNELS, flags.tolist())),
            "fault_stress": dict(zip(self.FAULT_CHANNELS, self.stress.tolist())),
            "wear": self.wear, "failed": self.failed, "failure_time": self.failure_time,
        }

    def _conditions(self, engine_rpm, power_kw, airspeed):
        """Boolean stress condition per channel, derived purely from operating state.
        These thresholds encode realistic risk factors:
          - EGT: sustained near-max power (thermal/mixture stress)
          - CHT: high power combined with low airspeed (poor cooling airflow)
          - OilPressure: sustained high RPM (pump wear)
          - OilTemp: sustained high power
          - Vib X/Y/Z: sustained high RPM (mechanical/bearing stress near redline)
          - RPM sensor: no physical driver -> handled as a rare independent glitch
        """
        rpm_frac = engine_rpm / self.RPM_MAX
        power_frac = power_kw / self.MAX_POWER_KW  # per-engine rated power, not hardcoded to 914
        cooling = min(1.0, airspeed / 50.0)
        if self.physics_version == "v3":
            # v3 thresholds. The dataset floors airspeed at ~30 m/s (no sub-stall flight),
            # which made the v2 CHT condition (cooling < 0.6, i.e. < 30 m/s) unreachable,
            # and EGT's > 85% power became too rare - both channels got ZERO faults in a
            # 300k-row sample, so their diagnosis could not be learned.
            return np.array([
                power_frac > 0.75,                          # EGT
                (power_frac > 0.60) and (airspeed < 42.0),  # CHT: working hard with poor airflow
                rpm_frac > 0.90,                            # OilPressure
                power_frac > 0.75,                          # OilTemp
                rpm_frac > 0.85,                            # VibX
                rpm_frac > 0.85,                            # VibY
                rpm_frac > 0.85,                            # VibZ
                False,                                      # RPM (no physical condition)
            ], dtype=bool), rpm_frac, power_frac, cooling
        return np.array([
            power_frac > 0.85,                          # EGT
            (power_frac > 0.70) and (cooling < 0.6),    # CHT
            rpm_frac > 0.90,                             # OilPressure
            power_frac > 0.75,                           # OilTemp
            rpm_frac > 0.85,                             # VibX
            rpm_frac > 0.85,                             # VibY
            rpm_frac > 0.85,                             # VibZ
            False,                                       # RPM (no physical condition)
        ], dtype=bool), rpm_frac, power_frac, cooling

    def _auto_fault_step(self, dt, engine_rpm, power_kw, airspeed):
        """Update stress accumulators from real operating conditions and arm/clear
        faults automatically with hysteresis. This fully replaces manual fault
        triggering - fault_type/severity/fault_start/fault_duration are all derived
        here, every step, from the actual inputs.
        """
        conditions, rpm_frac, power_frac, cooling = self._conditions(engine_rpm, power_kw, airspeed)

        for i in range(8):
            if conditions[i]:
                self.stress[i] = min(1.0, self.stress[i] + self.RISE_RATE*dt)
            else:
                self.stress[i] = max(0.0, self.stress[i] - self.DECAY_RATE*dt)

        # Rare independent RPM sensor glitch - electrical noise (Spike) or a stuck
        # connector/sender unit (Stuck-At), since both are real failure modes for this
        # channel and neither is condition-driven.
        if self.physics_version == "v3":
            self._rpm_glitch_step_v3(dt)
        elif not self._auto_active[7] and np.random.rand() < 0.001*dt*100:
            self._auto_active[7] = True
            glitch_is_stuck = np.random.rand() < 0.5
            self.fault_type[7] = 4 if glitch_is_stuck else 3   # Stuck-At or Spike
            self.severity[7] = 30.0
            self.fault_start[7] = self.t
            self.fault_duration[7] = 5.0 if glitch_is_stuck else 2.0
        elif self._auto_active[7] and self.t > self.fault_start[7] + self.fault_duration[7]:
            self._auto_active[7] = False
            self.fault_type[7] = 0

        if self.physics_version == "v3":
            self._stress_faults_step_v3()
            return

        # Channel-specific type + severity, scaled continuously by current stress
        auto_type = [2, 1, 2, 1, 5, 5, 5]       # Drift,Bias,Drift,Bias,Noise,Noise,Noise
        def sev(i):
            s = self.stress[i]
            if i == 0:  return 0.5 + 3.0*s        # EGT drift, deg/s
            if i == 1:  return 20.0 + 80.0*s       # CHT bias, deg
            if i == 2:  return -(0.05 + 0.30*s)    # OilPressure drift, dropping
            if i == 3:  return 10.0 + 30.0*s       # OilTemp bias, deg
            return 0.01 + 0.05*s                    # Vib noise std dev

        for i in range(7):
            if not self._auto_active[i] and self.stress[i] > self.TRIGGER_THRESHOLD:
                self._auto_active[i] = True
                self.fault_type[i] = auto_type[i]
                self.fault_start[i] = self.t
                self.fault_duration[i] = 1e9  # stays on while condition persists
            elif self._auto_active[i] and self.stress[i] < self.CLEAR_THRESHOLD:
                self._auto_active[i] = False
                self.fault_type[i] = 0
            if self._auto_active[i]:
                self.severity[i] = sev(i)

    # ---- physics v3 sensor faults -------------------------------------------------
    # Measured on a v3 sample before this change:
    #  - every channel only ever got ONE fault type (EGT always Drift, vibration always
    #    Noise, ...), so fault-type diagnosis was structurally uninformative;
    #  - the RPM glitch was active in 24% of rows, a Spike showed on a single 1 s sample,
    #    and glitched readings deviated by a median 26 rpm against 33 rpm of normal
    #    noise, so the RPM diagnosis learned nothing on any engine.
    # Stuck-At is kept off the stress-driven channels: temperatures and pressure sit at a
    # steady value for long stretches, so a frozen reading would be invisible - label noise.
    V3_FAULT_POOL = {
        0: (1, 2, 5),      # EGT: Bias, Drift, Noise
        1: (1, 2, 5),      # CHT
        2: (1, 2),         # oil pressure: Bias, Drift
        3: (1, 2),         # oil temp
        4: (5, 1, 3),      # vib x: Noise, Bias, Spike
        5: (5, 1, 3),      # vib y
        6: (5, 1, 3),      # vib z
    }
    RPM_GLITCH_RATE_V3 = 0.0015        # onsets per second -> active ~3% of the time

    def _v3_severity(self, i, ft, s):
        """Visible, stress-scaled magnitude for fault type ft on channel i (stress s 0..1).
        Bias: offset. Drift: rate per second (capped by DRIFT_CAP_FRAC). Noise: std dev.
        Spike: jump size (applied intermittently in _inject_faults)."""
        if i == 0:   # EGT, C
            return {1: 40.0 + 100.0*s, 2: 0.5 + 3.0*s, 5: 10.0 + 30.0*s}[ft]
        if i == 1:   # CHT, C
            return {1: 20.0 + 80.0*s, 2: 0.2 + 1.0*s, 5: 5.0 + 15.0*s}[ft]
        if i == 2:   # oil pressure, dropping
            return {1: -(10.0 + 30.0*s), 2: -(0.05 + 0.30*s)}[ft]
        if i == 3:   # oil temp, C
            return {1: 10.0 + 30.0*s, 2: 0.1 + 0.5*s}[ft]
        return {5: 0.01 + 0.05*s, 1: 0.03 + 0.10*s, 3: 0.20 + 0.30*s}[ft]   # vibration, g

    def _stress_faults_step_v3(self):
        for i in range(7):
            if not self._auto_active[i] and self.stress[i] > self.TRIGGER_THRESHOLD:
                self._auto_active[i] = True
                self.fault_type[i] = float(np.random.choice(self.V3_FAULT_POOL[i]))
                self.fault_start[i] = self.t
                self.fault_duration[i] = 1e9  # stays on while condition persists
            elif self._auto_active[i] and self.stress[i] < self.CLEAR_THRESHOLD:
                self._auto_active[i] = False
                self.fault_type[i] = 0
            if self._auto_active[i]:
                self.severity[i] = self._v3_severity(i, int(self.fault_type[i]), self.stress[i])

    def _rpm_glitch_step_v3(self, dt):
        if not self._auto_active[7] and np.random.rand() < self.RPM_GLITCH_RATE_V3 * dt:
            self._auto_active[7] = True
            stuck = np.random.rand() < 0.3
            self.fault_type[7] = 4 if stuck else 3
            self.severity[7] = 40.0                        # spike size: severity*10 rpm
            self.fault_start[7] = self.t
            self.fault_duration[7] = float(np.random.uniform(20.0, 60.0) if stuck else np.random.uniform(3.0, 8.0))
        elif self._auto_active[7] and self.t > self.fault_start[7] + self.fault_duration[7]:
            self._auto_active[7] = False
            self.fault_type[7] = 0

    def _update_wear(self, dt, engine_rpm, power_kw):
        """Irreversible wear accumulation. Uses CONTINUOUS operating severity (not the
        threshold-gated stress array, which is zero below trigger conditions - that
        would mean gentle flying never wears the engine at all, which is wrong). Every
        second of operation costs a little life; harder operation costs much more,
        scaling roughly quadratically with RPM and power (mirrors real fatigue physics
        where cyclic stress damage scales with stress amplitude squared or worse).
        """
        if self.failed:
            return
        rpm_frac = engine_rpm / self.RPM_MAX
        power_frac = power_kw / self.MAX_POWER_KW  # per-engine rated power
        severity = 0.5*rpm_frac**2 + 0.5*power_frac**2   # 0 (idle) to ~1 (redline+max power)
        floor = 0.05  # engine wears even at idle, just very slowly
        wear_drive = floor + (1-floor)*severity
        self.wear = min(1.0, self.wear + self.WEAR_K*wear_drive*dt)
        if self.wear >= 1.0 and not self.failed:
            self.failed = True
            self.failure_time = self.t

    def _inject_faults(self, healthy, t):
        faulty = healthy.copy()
        flags = np.zeros(8)
        for i in range(8):
            active = (t >= self.fault_start[i]) and (t < self.fault_start[i]+self.fault_duration[i])
            if not active:
                self._stuck_vals[i] = healthy[i]
                continue
            ft, sev = self.fault_type[i], self.severity[i]
            if ft == 1:
                faulty[i] = healthy[i] + sev
            elif ft == 2:
                disp = sev*(t-self.fault_start[i])
                if self.DRIFT_CAP_FRAC is not None:
                    cap = self.DRIFT_CAP_FRAC*abs(healthy[i])
                    disp = float(np.clip(disp, -cap, cap))
                faulty[i] = healthy[i] + disp
            elif ft == 3:
                if self.physics_version == "v3":
                    # v3: repeated jumps on about half the samples for the whole fault,
                    # both directions, so a spike is visible at any sampling rate.
                    if np.random.rand() < 0.5:
                        faulty[i] = healthy[i] + sev*10*(1.0 if np.random.rand() < 0.5 else -1.0)
                elif (t - self.fault_start[i]) % 5 < 0.1:
                    faulty[i] = healthy[i] + sev*10
            elif ft == 4:
                faulty[i] = self._stuck_vals[i]
            elif ft == 5:
                faulty[i] = healthy[i] + sev*np.random.randn()
            flags[i] = ft
        return faulty, flags
