"""Live physics-v4 twin for main.py (AERO_PHYSICS_VERSION=v4).

Three things run together, exactly as backend/generate_dataset_v4.py builds every
training row:
  the ENGINE being flown   PistonEngineV4 with the Health its degradation implies
  the ON-BOARD TWIN        PistonEngineV4 with a perfect Health, same inputs
  the SENSORS              sensors_v4.SensorBank: noise, quantisation, instrument faults

Physics steps at 100 Hz for smooth display. Sensors are read, and residuals formed,
once per flight second - the training row rate. Measured on a 30-minute 914 flight,
sampling the 100 Hz pair once a second moves no model feature's mean by more than
0.002 scaler standard deviations against the 1 Hz generator, so one engine pair
serves both the display and the AI (docs/v4_integration_plan.md, B1).

Two clocks (timescale_v4.py): `time` is flight seconds; `engine_hours` is the life
clock, start + LIFE_SCALE x duty-weighted usage, and it drives the degradation.

Interface is the one main.py already uses for v2/v3 twins: settable altitude /
throttle / airspeed / aoa / isa_dev_c, dt, step(), wear, MAX_POWER_KW,
restore_state(). The injected truth rides along in `truth` for the ground-truth
panel and is never an AI input.
"""
from __future__ import annotations

import numpy as np

import physics_v4 as V
import scenarios_v4
import timescale_v4 as TS
from degradation_v4 import DegradationState, FaultEvent, applicable_faults
from sensors_v4 import SENSOR_CHANNELS, SENSOR_FAULT_TYPES, SensorBank, SensorFault

# Contract with validation_v4/tf_data_pipeline.py (checked by tests/test_ai_v4_contract.py).
RESIDUAL_CHANNELS = ["egt", "cht", "oil_temp", "oil_pressure", "engine_rpm", "fuel_flow"]
FEATURE_COLS = (["altitude", "airspeed", "aoa", "throttle", "ambient_temp_c", "air_density",
                 "isa_dev_c", "humidity_frac"]
                + list(SENSOR_CHANNELS)
                + ["prop_rpm", "thrust_margin", "lift_weight_margin"]
                + [f"res_{c}" for c in RESIDUAL_CHANNELS])          # the 29 model inputs, in order
FAULT_PRESENT_SEV = 0.08

# Installation and fuel state the dataset samples per flight; these are its central values.
DEFAULT_ENV = {"qnh_offset_pa": 0.0, "humidity_frac": 0.4, "fuel_octane_mon": 95.0,
               "fuel_ethanol_frac": 0.0, "cooling_airflow_factor": 1.0, "electrical_load_a": 14.0,
               "oil_thermostat_open": True, "target_lambda": 0.89}
# generate_dataset_v4.py's warm-up point and settle length: every training flight
# starts from this thermal state, so every live flight does too.
WARM_INPUTS = {"altitude_m": 1500.0, "airspeed_ms": 45.0, "throttle": 0.8}
WARM_SETTLE_S = 300
STATE_FIELDS = ("omega", "cht_k", "oil_k", "coolant_k", "egt_k", "wastegate", "battery_v")
HEALTHY = V.Health()


def degradation_from_record(rec: dict) -> DegradationState:
    """The exact DegradationState an engine record describes."""
    plan = rec.get("fault_plan") or {}
    d = DegradationState(np.random.default_rng(int(rec["degradation_seed"])),
                         float(rec["engine_hours"]), float(rec["tbo_hours"]), n_faults=0)
    base = plan.get("baseline")
    if base:
        d.base_a, d.base_b, d.base_scale = float(base["a"]), float(base["b"]), float(base["scale"])
    d.faults = [FaultEvent(**f) for f in plan.get("faults", [])]
    return d


class UAVEngineTwinV4:
    PHYSICS_VERSION = "v4"

    def __init__(self, dt: float = 0.01, engine_model: str = "Rotax_914_ULF",
                 engine: dict | None = None, seed: int | None = None):
        self.dt = float(dt)
        self.engine_model = engine_model
        self.spec = V.ENGINE_SPECS_V4[engine_model]
        self.MAX_POWER_KW = self.spec.rated_power_w / 1000.0
        self.TBO_HOURS = self.spec.tbo_hours
        self.steps_per_sample = int(round(1.0 / (self.dt * TS.FLIGHT_HZ)))
        self.rng = np.random.default_rng(seed)
        self.altitude, self.airspeed, self.aoa, self.throttle = 1500.0, 45.0, 2.0, 0.75
        self._load(engine or scenarios_v4.make_engine(engine_model, seed=seed))
        self._warm()

    # ------------------------------------------------------------------ setup
    def _load(self, rec: dict) -> None:
        if rec["engine_model"] != self.engine_model:
            raise ValueError(f"engine record is a {rec['engine_model']}, twin is a {self.engine_model}")
        self.record = {**rec, "fault_plan": {**(rec.get("fault_plan") or {})},
                       "sensor_plan": list(rec.get("sensor_plan") or [])}
        self.record["fault_plan"]["faults"] = list(self.record["fault_plan"].get("faults") or [])
        self.start_engine_hours = float(rec["engine_hours"])
        self.deg = degradation_from_record(self.record)
        self.wear_out_h = self.deg.wear_out_hours()
        env = {**DEFAULT_ENV, **(rec.get("environment") or {})}
        self.isa_dev_c = float(env.pop("isa_dev_c", 0.0))
        self.env = env
        self.sensors = SensorBank(rng=self.rng, dt=1.0 / TS.FLIGHT_HZ)
        self.sensors.faults = [SensorFault(channel=s["channel"], kind=s["kind"],
                                           onset_s=float(s["onset_flight_s"]), severity=float(s["severity"]))
                               for s in self.record["sensor_plan"]]
        self.health, self.severity = self.deg.health_at(self.start_engine_hours)
        self.eng = V.PistonEngineV4(self.engine_model, dt=self.dt)
        self.ref = V.PistonEngineV4(self.engine_model, dt=self.dt)
        self.t, self._n, self._sample = 0.0, 0, None

    def _warm(self) -> None:
        """generate_dataset_v4.py's warm start, run at its own 1 Hz (the settle costs
        0.1 s there against 3 s at 100 Hz), then handed to the 100 Hz pair."""
        w = V.Inputs(isa_dev_c=self.isa_dev_c, **WARM_INPUTS, **self.env)
        for fine, h in ((self.eng, self.health), (self.ref, HEALTHY)):
            coarse = V.PistonEngineV4(self.engine_model, dt=1.0)
            coarse.warm_start(w, h)
            for _ in range(WARM_SETTLE_S):
                coarse.step(w, h)
            for f in STATE_FIELDS:
                setattr(fine, f, getattr(coarse, f))
            fine.life_used_h = 0.0          # usage belongs to the flight, which starts now

    # ------------------------------------------------------------------ clocks
    @property
    def engine_hours(self) -> float:
        return TS.engine_hours(self.start_engine_hours, self.eng.life_used_h)

    @property
    def wear(self) -> float:
        """Engine hours as a fraction of TBO. main.py derives the hour meter and the RUL
        filter's consumed hours from wear x TBO, as it does for v3."""
        return self.engine_hours / self.TBO_HOURS

    # ------------------------------------------------------------------ step
    def _inputs(self) -> V.Inputs:
        return V.Inputs(altitude_m=self.altitude, airspeed_ms=self.airspeed, aoa_deg=self.aoa,
                        throttle=self.throttle, isa_dev_c=self.isa_dev_c, **self.env)

    def _read(self, o: dict, r: dict, hours: float) -> dict:
        """One 1 Hz sample: what the instruments read, residuals against the twin,
        the operating margin from MEASURED values, and the injected truth."""
        meas, flag, _ = self.sensors.read(o, self.t)
        m_meas = self.eng.margins({**o, **meas})
        m_true = self.eng.margins(o)
        return {
            "measured": meas,
            "residuals": {f"res_{c}": meas[c] - r[c] for c in RESIDUAL_CHANNELS},
            "twin": {c: r[c] for c in SENSOR_CHANNELS},
            "margins": m_meas,
            "truth": {
                "fault_severity": {n: round(v, 4) for n, v in self.severity.items() if v > 0.0},
                "faults_present": [n for n, v in self.severity.items() if v >= FAULT_PRESENT_SEV],
                "sensor_faults": {c: SENSOR_FAULT_TYPES[k] for c, k in flag.items() if k},
                "wear_condition": round(self.deg.condition_at(hours), 5),
                "margin_min": round(m_true["health_index"], 5),
                "true_values": {c: o[c] for c in SENSOR_CHANNELS},
                "rul_hours": round(max(0.0, min(self.TBO_HOURS, self.wear_out_h) - hours), 3),
                "rul_calendar_hours": round(max(0.0, self.TBO_HOURS - hours), 3),
                "wear_limited": bool(self.wear_out_h < self.TBO_HOURS),
            },
        }

    def step(self) -> dict:
        u = self._inputs()
        o = self.eng.step(u, self.health)
        r = self.ref.step(u, HEALTHY)
        self._n += 1
        self.t = self._n * self.dt
        hours = self.engine_hours
        new = self._n % self.steps_per_sample == 0
        if new:
            # Degradation moves on the life clock, once per flight second.
            self.health, self.severity = self.deg.health_at(hours)
        if new or self._sample is None:
            self._sample = self._read(o, r, hours)
        s = self._sample

        out = dict(o)
        out.update(self.env)                               # the inputs as the engine received them
        # The instruments, not the physics: gauges and the AI see what the sensors
        # read, held between 1 Hz samples. The physics' own values are in truth.
        out.update(s["measured"])
        out.update(s["residuals"])
        out.update({
            "time": round(self.t, 4),
            "sample_new": new,
            "engine_hours": round(hours, 4),
            "start_engine_hours": self.start_engine_hours,
            "life_used_hours": self.eng.life_used_h,       # flight clock (an AI input, never scaled)
            "wear": hours / self.TBO_HOURS,
            "tbo_hours": self.TBO_HOURS,
            "life_scale": TS.LIFE_SCALE,
            "margin_min": round(s["margins"]["health_index"], 5),
            "margins": s["margins"],
            "twin": s["twin"],
            "truth": s["truth"],
            "scenario": self.record.get("scenario"),
            "applicable_faults": applicable_faults(self.spec.turbocharged, self.spec.intercooled),
            "engine_record": {**self.record, "engine_hours": round(hours, 4)},
            "physics_state": {"eng": {f: getattr(self.eng, f) for f in STATE_FIELDS},
                              "ref": {f: getattr(self.ref, f) for f in STATE_FIELDS}},
        })
        return out

    # ------------------------------------------------------------------ resume / recovery
    def restore_state(self, snap: dict) -> None:
        """Pick a flight up from a saved step (final_telemetry, or the last good step
        after a physics fault): same engine record at its current hours, same thermal
        and shaft state, same flight clock."""
        rec = snap.get("engine_record")
        if rec:
            self._load(rec)
        for k in ("altitude", "throttle", "airspeed", "aoa", "isa_dev_c"):
            if isinstance(snap.get(k), (int, float)):
                setattr(self, k, float(snap[k]))
        for k in DEFAULT_ENV:
            if k in snap:
                self.env[k] = snap[k]
        ps = snap.get("physics_state") or {}
        for engine, key in ((self.eng, "eng"), (self.ref, "ref")):
            for f, v in (ps.get(key) or {}).items():
                if f in STATE_FIELDS and isinstance(v, (int, float)):
                    setattr(engine, f, float(v))
        self.t = float(snap.get("time", 0.0))
        self._n = int(round(self.t / self.dt))
        self._sample = None

    # ------------------------------------------------------------------ injection (POST /inject)
    def inject_fault(self, name: str, severity: float = 0.35) -> dict:
        if name not in applicable_faults(self.spec.turbocharged, self.spec.intercooled):
            raise ValueError(f"{name!r} cannot occur on a {self.engine_model}")
        entry = scenarios_v4.fault_entry(name, float(severity), self.engine_hours, self.TBO_HOURS)
        self.record["fault_plan"]["faults"].append(entry)
        self.deg.faults.append(FaultEvent(**entry))
        self.deg.start_hours = self.engine_hours
        self.wear_out_h = self.deg.wear_out_hours()
        self.health, self.severity = self.deg.health_at(self.engine_hours)
        return entry

    def inject_sensor(self, channel: str, kind: str, severity: float = 0.8) -> dict:
        if channel not in SENSOR_CHANNELS or kind not in SENSOR_FAULT_TYPES[1:]:
            raise ValueError(f"channel must be one of {SENSOR_CHANNELS}, kind one of {SENSOR_FAULT_TYPES[1:]}")
        entry = scenarios_v4.sensor_entry(channel, kind, float(severity), onset_flight_s=self.t)
        self.record["sensor_plan"].append(entry)
        self.sensors.faults.append(SensorFault(channel=channel, kind=kind, onset_s=self.t,
                                               severity=float(severity)))
        return entry

    def clear_fault(self, name: str) -> int:
        """Remove a component fault (injected or from the preset) as if it never
        happened: the damage goes with it. Returns how many entries were removed."""
        plan = self.record["fault_plan"]
        n = len(plan["faults"])
        plan["faults"] = [f for f in plan["faults"] if f["name"] != name]
        self.deg.faults = [f for f in self.deg.faults if f.name != name]
        self.wear_out_h = self.deg.wear_out_hours()
        self.health, self.severity = self.deg.health_at(self.engine_hours)
        return n - len(plan["faults"])

    def clear_sensor(self, channel: str) -> int:
        """Repair a sensor: every fault on that channel stops from the next reading."""
        n = len(self.record["sensor_plan"])
        self.record["sensor_plan"] = [e for e in self.record["sensor_plan"] if e["channel"] != channel]
        self.sensors.faults = [f for f in self.sensors.faults if f.channel != channel]
        return n - len(self.record["sensor_plan"])
