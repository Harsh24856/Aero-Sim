"""Live physics-v5 twin for main.py (AERO_PHYSICS_VERSION=v5).

twin_v4.UAVEngineTwinV4 with the v5 engine, instruments and inputs - the same
three things backend/generate_dataset_v5.py runs for every training row:
  the ENGINE being flown   physics_v5.PistonEngineV5 with its DegradationStateV5 health
  the ON-BOARD TWIN        PistonEngineV5 with a perfect Health, same inputs
  the SENSORS              sensors_v5.SensorBank: 14 instruments (12 can fault)

Once per flight second it emits the 39 model inputs (features_v5.FEATURE_COLS) and
the 45 context values the v5 models read beside the window:
  42 long-horizon residual statistics (features_v5.LongHorizon over residuals in
     noise-sigma units, from the flight's first second - as the training cache did)
   3 hours as a fraction of TBO, this flight's usage, the x180 life-clock flag.
The context belongs to the flight, not to the AI window, so it lives here and
survives an AI reset; restore_state carries it across a pause.

Ground truth uses the v5b label meanings (validation_v5/relabel_v5b.py): severity
is EFFECTIVE (damage x this engine's depth / the fault's full depth), RUL counts only
faults already under way.

Interface: identical to UAVEngineTwinV4, so main.py drives either.
"""
from __future__ import annotations

import numpy as np

import features_v5 as F
import physics_v5 as P
import scenarios_v4
import timescale_v4 as TS
from degradation_v4 import FAULT_MODES
from degradation_v5 import DegradationStateV5, FaultEvent, applicable_faults
from sensors_v5 import FAULTABLE_CHANNELS, SENSOR_CHANNELS, SENSOR_FAULT_TYPES, SENSOR_SPEC, SensorBank, SensorFault
from twin_v4 import DEFAULT_ENV, STATE_FIELDS, WARM_INPUTS, WARM_SETTLE_S

RESIDUAL_CHANNELS = list(F.RESIDUAL_CHANNELS)          # 14
FEATURE_COLS = list(F.FEATURE_COLS)                    # the 39 model inputs, in order
RES_SIGMA = np.array([SENSOR_SPEC[c]["noise_sd"] for c in RESIDUAL_CHANNELS])
FAULT_PRESENT_SEV = 0.08
HEALTHY = P.Health()


def degradation_from_record(rec: dict) -> DegradationStateV5:
    """The exact v5 DegradationState an engine record describes (no random faults)."""
    plan = rec.get("fault_plan") or {}
    d = DegradationStateV5(np.random.default_rng(int(rec["degradation_seed"])),
                           float(rec["engine_hours"]), float(rec["tbo_hours"]), n_faults=0)
    base = plan.get("baseline")
    if base:
        d.base_a, d.base_b, d.base_scale = float(base["a"]), float(base["b"]), float(base["scale"])
    d.faults = [FaultEvent(**f) for f in plan.get("faults", [])]
    return d


class UAVEngineTwinV5:
    PHYSICS_VERSION = "v5"
    RESIDUAL_CHANNELS = RESIDUAL_CHANNELS

    def __init__(self, dt: float = 0.01, engine_model: str = "Rotax_914_ULF",
                 engine: dict | None = None, seed: int | None = None):
        self.dt = float(dt)
        self.engine_model = engine_model
        self.spec = P.ENGINE_SPECS[engine_model]
        self.turbo = bool(self.spec.turbocharged)
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
        env = {**DEFAULT_ENV, **(rec.get("environment") or {})}
        self.isa_dev_c = float(env.pop("isa_dev_c", 0.0))
        self.env = env
        self.sensors = SensorBank(rng=self.rng, dt=1.0 / TS.FLIGHT_HZ, turbocharged=self.turbo)
        self.sensors.faults = [self._sensor_fault(s["channel"], s["kind"], float(s["onset_flight_s"]),
                                                  float(s["severity"])) for s in self.record["sensor_plan"]]
        self.health, self.severity = self.deg.health_at(self.start_engine_hours)
        self.eng = P.PistonEngineV5(self.engine_model, dt=self.dt)
        self.ref = P.PistonEngineV5(self.engine_model, dt=self.dt)
        self.lh = F.LongHorizon()
        self.t, self._n, self._sample = 0.0, 0, None

    @staticmethod
    def _sensor_fault(channel: str, kind: str, onset_s: float, severity: float) -> SensorFault:
        return SensorFault(channel=channel, kind=kind, onset_s=onset_s, severity=severity,
                           drift_rate=(1.0 + 3.0 * severity) if kind == "drift" else 0.0)

    def _warm(self) -> None:
        """generate_dataset_v5.py's warm start (1 Hz, 300 s), handed to the 100 Hz pair."""
        w = P.Inputs(isa_dev_c=self.isa_dev_c, **WARM_INPUTS, **self.env)
        for fine, h in ((self.eng, self.health), (self.ref, HEALTHY)):
            coarse = P.PistonEngineV5(self.engine_model, dt=1.0)
            coarse.warm_start(w, h)
            for _ in range(WARM_SETTLE_S):
                coarse.step(w, h)
            for f in STATE_FIELDS:
                setattr(fine, f, getattr(coarse, f))
            fine.life_used_h = 0.0

    # ------------------------------------------------------------------ clocks
    @property
    def engine_hours(self) -> float:
        return TS.engine_hours(self.start_engine_hours, self.eng.life_used_h)

    @property
    def wear(self) -> float:
        return self.engine_hours / self.TBO_HOURS

    # ------------------------------------------------------------------ step
    def _inputs(self) -> P.Inputs:
        return P.Inputs(altitude_m=self.altitude, airspeed_ms=self.airspeed, aoa_deg=self.aoa,
                        throttle=self.throttle, isa_dev_c=self.isa_dev_c, **self.env)

    def effective_severity(self) -> dict:
        """v5b severity: damage x (this engine's depth / the fault's full depth)."""
        mult = {f.name: f.depth / FAULT_MODES[f.name]["depth"] for f in self.deg.faults}
        return {n: v * mult.get(n, 1.0) for n, v in self.severity.items()}

    def _read(self, o: dict, r: dict, hours: float, new: bool = True) -> dict:
        meas, flag, _, _ = self.sensors.read(o, self.t)
        res = F.residuals(meas, r, self.turbo)
        # One context step per 1 Hz sample, as training's long_horizon_array: the
        # display-only read at a flight's first physics step does not take one.
        lh = self.lh.update(np.asarray(res) / RES_SIGMA) if new else self.lh.current()
        ctx = [float(v) for v in lh] + [hours / self.TBO_HOURS, float(self.eng.life_used_h),
                                         1.0 if TS.LIFE_SCALE > 1.0 else 0.0]
        m_meas = self.eng.margins({**o, **meas})
        m_true = self.eng.margins(o)
        eff = self.effective_severity()
        wear_out_known = self.deg.wear_out_known(hours)
        return {
            "measured": meas,
            "residuals": {f"res_{c}": v for c, v in zip(RESIDUAL_CHANNELS, res)},
            "context": ctx,
            "twin": {c: r.get(c, 0.0) for c in SENSOR_CHANNELS},
            "margins": m_meas,
            "truth": {
                "fault_severity": {n: round(v, 4) for n, v in eff.items() if v > 0.0},
                "severity_kind": "effective",
                "faults_present": [n for n, v in eff.items() if v >= FAULT_PRESENT_SEV],
                "sensor_faults": {c: SENSOR_FAULT_TYPES[k] for c, k in flag.items() if k},
                "wear_condition": round(self.deg.condition_at(hours), 5),
                "margin_min": round(m_true["health_index"], 5),
                "true_values": {c: o.get(c, 0.0) for c in SENSOR_CHANNELS},
                "rul_hours": round(max(0.0, min(self.TBO_HOURS, wear_out_known) - hours), 3),
                "rul_calendar_hours": round(max(0.0, self.TBO_HOURS - hours), 3),
                "wear_limited": bool(wear_out_known < self.TBO_HOURS),
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
            self.health, self.severity = self.deg.health_at(hours)
        if new or self._sample is None:
            self._sample = self._read(o, r, hours, new)
        s = self._sample

        out = dict(o)
        out.update(self.env)
        out["isa_dev_c"] = self.isa_dev_c
        out.update(s["measured"])
        out.update(s["residuals"])
        out.update({
            "time": round(self.t, 4),
            "sample_new": new,
            "engine_hours": round(hours, 4),
            "start_engine_hours": self.start_engine_hours,
            "life_used_hours": self.eng.life_used_h,
            "wear": hours / self.TBO_HOURS,
            "tbo_hours": self.TBO_HOURS,
            "life_scale": TS.LIFE_SCALE,
            "ai_context": s["context"],
            "margin_min": round(s["margins"]["health_index"], 5),
            "margins": s["margins"],
            "twin": s["twin"],
            "truth": s["truth"],
            "scenario": self.record.get("scenario"),
            "applicable_faults": applicable_faults(self.spec.turbocharged, self.spec.intercooled),
            "engine_record": {**self.record, "engine_hours": round(hours, 4)},
            "physics_state": {"eng": {f: getattr(self.eng, f) for f in STATE_FIELDS},
                              "ref": {f: getattr(self.ref, f) for f in STATE_FIELDS},
                              "long_horizon": self.lh.state.tolist()},
        })
        return out

    # ------------------------------------------------------------------ resume / recovery
    def restore_state(self, snap: dict) -> None:
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
        if ps.get("long_horizon") is not None:
            self.lh.state = np.asarray(ps["long_horizon"], dtype=np.float64)
        # This flight's usage is a model input (the context's second-last value) and
        # restarting it at 0 beside settled 60-minute averages never happens in training.
        if isinstance(snap.get("life_used_hours"), (int, float)):
            self.eng.life_used_h = float(snap["life_used_hours"])
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
        self.health, self.severity = self.deg.health_at(self.engine_hours)
        return entry

    def inject_sensor(self, channel: str, kind: str, severity: float = 0.8) -> dict:
        if channel not in FAULTABLE_CHANNELS or kind not in SENSOR_FAULT_TYPES[1:]:
            raise ValueError(f"channel must be one of {FAULTABLE_CHANNELS}, kind one of {SENSOR_FAULT_TYPES[1:]}")
        entry = scenarios_v4.sensor_entry(channel, kind, float(severity), onset_flight_s=self.t)
        self.record["sensor_plan"].append(entry)
        self.sensors.faults.append(self._sensor_fault(channel, kind, self.t, float(severity)))
        return entry

    def clear_fault(self, name: str) -> int:
        plan = self.record["fault_plan"]
        n = len(plan["faults"])
        plan["faults"] = [f for f in plan["faults"] if f["name"] != name]
        self.deg.faults = [f for f in self.deg.faults if f.name != name]
        self.health, self.severity = self.deg.health_at(self.engine_hours)
        return n - len(plan["faults"])

    def clear_sensor(self, channel: str) -> int:
        n = len(self.record["sensor_plan"])
        self.record["sensor_plan"] = [e for e in self.record["sensor_plan"] if e["channel"] != channel]
        self.sensors.faults = [f for f in self.sensors.faults if f.channel != channel]
        return n - len(self.record["sensor_plan"])
