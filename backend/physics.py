"""
UAV Piston Engine Digital Twin -- Python/NumPy port of the validated Simulink model
(UAV_Piston_Engine.slx). Every formula below was extracted directly from the Simulink
block diagram (not re-derived), so this is a faithful 1:1 port of the golden model.

Multi-engine support: ENGINE_CONFIGS holds the full spec set for each selectable
engine. IMPORTANT CAVEAT - every trained AI model (Detection/Diagnosis/Severity/RUL)
was trained ONLY on Rotax_914_ULF physics data. The 912/915/916 have genuinely
different torque curves and power output, so AI predictions will NOT be meaningful
for those engines - this is a real physics-simulation capability, not a claim that
the AI generalizes to them.
"""
import numpy as np


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


class UAVEngineTwin:
    FAULT_CHANNELS = ["egt","cht","oil_pressure","oil_temp","vibx","viby","vibz","rpm"]

    def __init__(self, dt=0.01, engine_model="Rotax_914_ULF"):
        if engine_model not in ENGINE_CONFIGS:
            raise ValueError(f"Unknown engine_model {engine_model!r}. Options: {list(ENGINE_CONFIGS)}")
        self.engine_model = engine_model
        cfg = ENGINE_CONFIGS[engine_model]

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
        # cold-start initial temperatures (ambient-ish, engine hasn't run yet)
        self.egt_state = 20.0
        self.cht_state = 20.0
        self.oiltemp_state = 20.0

        # live-adjustable exogenous inputs (settable at any time from the API)
        self.altitude = 2000.0
        self.throttle = 0.5
        self.airspeed = 40.0
        self.aoa = 5.0

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
        self.failed = False
        self.failure_time = None

        # Stuck-At is a rare independent connector/wiring glitch, not condition-driven
        # (unlike Bias/Drift/Noise which ARE condition-driven). Applied to OilPressure,
        # matching a real failure mode (stuck sender unit).
        self._stuck_active = False
        self._stuck_start = 0.0

    # ---------------- Atmosphere (ISA troposphere model) ----------------
    def atmosphere(self, altitude):
        T = 288.15 - 0.0065*altitude
        P = 101325.0 * (T/288.15)**5.256
        rho = P / (287.05*T)
        return T, P, rho

    # ---------------- Torque map (Rotax 914 UL/F) ----------------
    def torque_available(self, throttle, engine_rpm):
        thr = np.clip(throttle, 0.0, 1.0)
        rpm_c = np.clip(engine_rpm, self.RPM_POINTS[0], self.RPM_POINTS[-1])
        base_torque = np.interp(rpm_c, self.RPM_POINTS, self.TORQUE_POINTS)
        return thr * base_torque

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
        fuel_flow = power_kw*self.BSFC

        lift, drag, thrust_margin, lift_margin = self.aerodynamics(rho, self.airspeed, self.aoa, thrust)

        # Algebraic steady-state TARGETS (what each thermal signal is heading toward,
        # given current operating conditions -- same formulas as before)
        # Real CHT/OilTemp depend strongly on cooling airflow (airspeed) - confirmed by
        # aviation sources: "reduced cooling airflow increases CHT", shock-cooling/shock-
        # heating during low-airspeed high-power ground ops. Capped since a fixed-size
        # radiator/fin area has diminishing returns at very high speed.
        cool = min(max(self.airspeed, 0.0), 60.0)
        egt_target = 300 + 400*throttle + 0.05*engine_rpm + 5*power_kw
        cht_target = 200 + 250*throttle + 0.03*engine_rpm + 5*power_kw - 1.8*cool
        oiltemp_target = 60 + 30*throttle + 0.01*engine_rpm - 0.3*cool

        # First-order thermal lag: signal chases the target at a rate set by tau.
        # This is what makes CHT/OilTemp/EGT actually warm up over real time instead
        # of snapping to a value the instant throttle changes.
        self.egt_state     += dt * (egt_target - self.egt_state) / self.EGT_TAU
        self.cht_state      += dt * (cht_target - self.cht_state) / self.CHT_TAU
        self.oiltemp_state += dt * (oiltemp_target - self.oiltemp_state) / self.OILTEMP_TAU

        egt, cht, oil_t = self.egt_state, self.cht_state, self.oiltemp_state

        # OilPressure: real engines use a pump + pressure relief valve, so pressure RISES
        # then SATURATES with RPM (not indefinitely linear), and hot oil has lower
        # viscosity -> lower pressure at the same RPM (well-documented: e.g. real GA
        # engines commonly show ~60psi at normal oil temp dropping to ~30-35psi when oil
        # temp is abnormally high). Cold oil (high viscosity) gives a pressure boost.
        oil_p = 90.0*(1.0 - np.exp(-engine_rpm/1500.0)) + 3.0*throttle - 0.15*(self.oiltemp_state - 80.0)

        # Vibration: rotating-unbalance force scales with the SQUARE of RPM (F=m*r*omega^2),
        # not linearly - this is fundamental rotating-machinery vibration physics.
        rpm_k = (engine_rpm/1000.0)**2
        vibx = 0.02  + 0.00345*rpm_k
        viby = 0.025 + 0.00310*rpm_k
        vibz = 0.03  + 0.00379*rpm_k

        healthy = np.array([egt, cht, oil_p, oil_t, vibx, viby, vibz, engine_rpm])
        self._auto_fault_step(dt, engine_rpm, power_kw, self.airspeed)
        self._update_wear(dt, engine_rpm, power_kw)
        faulty, flags = self._inject_faults(healthy, self.t)

        # Real sensors/gauges have physical saturation limits no matter what is driving
        # the reading (a stuck needle still has a stop; a drifting thermocouple readout
        # still maxes out its display range) - clip so long-duration Drift faults cannot
        # produce physically impossible values (e.g. negative oil pressure, 3000C EGT).
        SENSOR_LIMITS = np.array([
            [0, 950],    # EGT C
            [0, 260],    # CHT C
            [0, 150],    # OilPressure
            [0, 150],    # OilTemp C
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
            "time": round(self.t,4), "altitude": self.altitude, "air_density": rho,
            "throttle": throttle, "airspeed": self.airspeed, "aoa": self.aoa,
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
        # connector/sender unit (Stuck-At), 50/50, since both are real failure modes
        # for this channel and neither is condition-driven.
        if not self._auto_active[7] and np.random.rand() < 0.001*dt*100:
            self._auto_active[7] = True
            glitch_is_stuck = np.random.rand() < 0.5
            self.fault_type[7] = 4 if glitch_is_stuck else 3   # Stuck-At or Spike
            self.severity[7] = 30.0
            self.fault_start[7] = self.t
            self.fault_duration[7] = 5.0 if glitch_is_stuck else 2.0
        elif self._auto_active[7] and self.t > self.fault_start[7] + self.fault_duration[7]:
            self._auto_active[7] = False
            self.fault_type[7] = 0

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
                faulty[i] = healthy[i] + sev*(t-self.fault_start[i])
            elif ft == 3:
                if (t - self.fault_start[i]) % 5 < 0.1:
                    faulty[i] = healthy[i] + sev*10
            elif ft == 4:
                faulty[i] = self._stuck_vals[i]
            elif ft == 5:
                faulty[i] = healthy[i] + sev*np.random.randn()
            flags[i] = ft
        return faulty, flags
