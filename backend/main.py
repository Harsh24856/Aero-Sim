"""
UAV Digital Twin - real-time FastAPI backend.
Runs the validated Python physics twin continuously in the background at real-time
pace. Parameters (altitude/throttle/airspeed/aoa) can be changed at any time via
POST /params and take effect on the very next simulation step - the loop itself never
stops until POST /stop is called. Live telemetry streams over WebSocket at ~20Hz.
"""
import asyncio
import copy
import importlib
import json
import math
import os
import time
import traceback
from collections import deque
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from physics import UAVEngineTwin, ENGINE_CONFIGS

# Physics version for every twin this process creates. v4 (default) pairs with
# backend/aiv4.py (backend/models_v4/); AERO_PHYSICS_VERSION=v3 runs physics v3 with aiv3.py
# (backend/models_v3/), and v2 the legacy physics with ai.py (backend/models/). /state
# reports both so a mismatch is visible.
PHYSICS_VERSION = os.environ.get("AERO_PHYSICS_VERSION", "v4").strip().lower()
if PHYSICS_VERSION not in ("v2", "v3", "v4"):
    raise ValueError(f"AERO_PHYSICS_VERSION must be v2, v3 or v4, got {PHYSICS_VERSION!r}")

# v3 runs persist through dbv3 (real engine-hour RUL), v4 through dbv4 (engines with
# an hour meter, both clocks). Same function names as db, so nothing below changes.
db = importlib.import_module({"v2": "db", "v3": "dbv3", "v4": "dbv4"}[PHYSICS_VERSION])
import advisory
import residual
import safety
import summary
import scenarios_v4
import timescale_v4
import twin_v4
from twin_v4 import UAVEngineTwinV4

# Sensor offsets from zeroing (POST /residuals/zero) describe the installed senders, so they
# survive a backend restart. Kept next to main.py and out of git.
RESIDUAL_OFFSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".residual_offsets.json")


def load_residual_offsets() -> dict:
    try:
        with open(RESIDUAL_OFFSETS_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_residual_offsets(offsets: dict) -> None:
    try:
        tmp = RESIDUAL_OFFSETS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(offsets, f, indent=2)
        os.replace(tmp, RESIDUAL_OFFSETS_PATH)
    except OSError as e:
        print(f"[residual] could not save sensor offsets: {e}")

# ai.py runs as its OWN process under a different Python environment (see ai.py's
# module docstring - the models segfault under this backend's TF version). Calls are
# async/non-blocking so a slow or down AI service never freezes telemetry to other
# clients, and every call is wrapped in try/except for the same reason.
AI_SERVICE_URL = os.environ.get("AERO_AI_URL", "http://127.0.0.1:8100")
# 5 s read timeout: five Metal predictions per step can take over a second on a busy
# machine, and the worker is off the physics loop, so waiting costs nothing but that
# one sample. 1.5 s was measured to time out the first full inference after every
# warm-up, which reset the AI's window every ~130 samples - it never stayed live.
# A connect that takes longer than 0.5 s still means the service is down.
ai_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=0.5))

# ---- safety net settings -----------------------------------------------------------
AI_QUEUE_MAX = 8              # samples waiting for the AI worker before it is declared stalled
AI_WARMUP_BACKLOG = 2         # warm-up fast-forward waits for the AI once this many are queued
AI_WARMUP_MAX_WAIT_S = 2.0    # ...but never longer than this per sample; then real-time pacing
RECOVERY_WINDOW_S = 30.0      # physics recoveries counted over this window
MAX_RECOVERIES = 3            # more than this inside the window halts the session safely
CLIENT_QUEUE_MAX = 3           # frames buffered per browser; older ones are dropped
CLIENT_STALL_S = 5.0           # a browser that accepts nothing for this long is disconnected
DB_TIMEOUT_S = 8.0            # endpoint-side wait for Supabase; the write thread may finish later
DB_MAX_PENDING = 4            # telemetry writes allowed in flight before new ones are skipped
ai_breaker = safety.CircuitBreaker(failure_threshold=3, cooldown_s=5.0)

# The AI models were trained on data sampled at exactly 1 real second per timestep -
# a 128-step window means 128 SECONDS of history to the model. The physics loop runs
# at dt=0.01s (100Hz) for smooth real-time simulation, so the AI must only be called
# once per 100 physics steps (once per simulated second), never on every tick - feeding
# it raw 0.01s ticks would make 128 steps span 1.28s instead of 128s, a timescale
# mismatch that would make every prediction meaningless.
AI_STEPS_PER_CALL = int(round(1.0 / 0.01))  # = 100, i.e. once per simulated second

app = FastAPI(title="UAV Digital Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP only - lock this down before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

def new_twin(engine_model: str, engine: Optional[dict] = None):
    """The ONLY place twins are built, so the physics version cannot be missed.
    v4 flies an engine record (scenarios_v4 preset or an engines row); None = the
    default preset for this engine model."""
    if PHYSICS_VERSION == "v4":
        return UAVEngineTwinV4(dt=0.01, engine_model=engine_model, engine=engine)
    return UAVEngineTwin(dt=0.01, engine_model=engine_model, physics_version=PHYSICS_VERSION)


CURRENT_ENGINE_MODEL = "Rotax_914_ULF"
twin = new_twin(CURRENT_ENGINE_MODEL)

# Physics residuals (backend/residual.py): measured sensors minus what physics v3
# expects. Model-free, so it keeps producing sensor-integrity and wear evidence while
# the AI is warming up or down. Fed at the AI cadence (one sample per simulated
# second) and cleared wherever the AI buffer is. Physics v2 returns enabled=False.
residual_monitor = residual.ResidualMonitor(lag="exp")


def reset_residuals():
    # Sensor offsets from zeroing belong to one engine installation (POST /residuals/zero).
    residual_monitor.offsets = dict(state["residual_offsets"].get(CURRENT_ENGINE_MODEL, {}))
    residual_monitor._zeroing = None
    residual_monitor.reset()
    state["residuals"] = None
    state["measured"] = {}
    state["measured_at"] = None
    reset_rul_filter()


# Measured sensors from the aircraft (can_ingest.py -> POST /measured). While fresh they
# REPLACE the twin's own sensor channels, so the AI and the residuals judge the real engine
# against the twin's physics; the twin keeps its own values alongside as twin_sensors.
MEASURED_FRESH_S = 1.0
MEASURED_KEYS = {"rpm": "rpm_fault", "egt": "egt", "cht": "cht", "oil_pressure": "oil_pressure",
                 "oil_temp": "oil_temp", "vibx": "vibx", "viby": "viby", "vibz": "vibz",
                 "fuel_flow": "fuel_flow"}
MEASURED_LIMITS = {**residual.SENSOR_LIMITS, "fuel_flow": (0.0, 200.0),
                   "coolant_temp": (0.0, 200.0), "manifold_pressure_kpa": (0.0, 250.0),
                   "battery_voltage": (0.0, 20.0)}
# v4 measures twelve channels, named as twin_v4 names them. The bus carries oil pressure
# in psi (can_bus.py, the physical spec); v4 works in bar.
MEASURED_KEYS_V4 = {"rpm": "engine_rpm", "egt": "egt", "cht": "cht", "oil_pressure": "oil_pressure",
                    "oil_temp": "oil_temp", "vibx": "vibx", "viby": "viby", "vibz": "vibz",
                    "fuel_flow": "fuel_flow", "coolant_temp": "coolant_temp",
                    "manifold_pressure_kpa": "manifold_pressure_kpa", "battery_voltage": "battery_voltage"}
PSI_PER_BAR = 14.5038


def measured_age_s() -> Optional[float]:
    at = state["measured_at"]
    return None if at is None else time.monotonic() - at


def data_source() -> str:
    age = measured_age_s()
    return "can" if age is not None and age <= MEASURED_FRESH_S and state["measured"] else "sim"


def apply_measured(out: dict) -> dict:
    """The twin's step with fresh measured sensor values overlaid (a new dict)."""
    if data_source() != "can":
        return {**out, "data_source": "sim"}
    measured = state["measured"]
    if PHYSICS_VERSION == "v4":
        return apply_measured_v4(out, measured)
    merged = {**out, "data_source": "can",
              "twin_sensors": {ch: out[key] for ch, key in MEASURED_KEYS.items() if ch in measured}}
    for ch, value in measured.items():
        merged[MEASURED_KEYS[ch]] = value
    return merged


def apply_measured_v4(out: dict, measured: dict) -> dict:
    """The aircraft's instruments replace the twin's simulated ones; residuals are
    re-taken against the on-board twin and the operating margin re-read from them."""
    merged = {**out, "data_source": "can", "twin_sensors": {}}
    for ch, value in measured.items():
        key = MEASURED_KEYS_V4.get(ch)
        if key is None:
            continue
        merged["twin_sensors"][key] = out.get(key)
        merged[key] = value / PSI_PER_BAR if ch == "oil_pressure" else value
    ref = out.get("twin") or {}
    for c in twin_v4.RESIDUAL_CHANNELS:
        if isinstance(ref.get(c), (int, float)):
            merged[f"res_{c}"] = merged[c] - ref[c]
    merged["margins"] = twin.eng.margins(merged)
    merged["margin_min"] = merged["margins"]["health_index"]
    return merged


PENDING_LOGS_MAX = 2000        # 10 s rows: over 5 hours of flight held while the DB is unreachable
SIM_RETRY_S = 15.0


def pending_simulation_args(user_id: Optional[str], mission: Optional[str] = None) -> Optional[tuple]:
    """What start_simulation needs, captured at /start so a retry records the flight
    as it began (a copy: the live record's hours move on)."""
    if not user_id or not importlib.import_module("db")._enabled:
        return None
    if PHYSICS_VERSION == "v4":
        return (user_id, CURRENT_ENGINE_MODEL, copy.deepcopy(twin.record), model_manifest_snapshot(), mission)
    return (user_id, CURRENT_ENGINE_MODEL)


async def retry_simulation_row() -> None:
    """Create the flight's simulations row now, if /start could not, then write the
    telemetry rows held since. Never raises; the next attempt waits SIM_RETRY_S."""
    args = state["pending_simulation"]
    if args is None or state["simulation_id"] is not None or state["sim_retrying"]:
        return
    state["sim_retrying"] = True
    try:
        sim_id = await db_call(db.start_simulation, *args)
        if sim_id is None:
            state["sim_retry_at"] = time.monotonic() + SIM_RETRY_S
            return
        if PHYSICS_VERSION == "v4" and getattr(twin, "record", None) is not None:
            twin.record["engine_id"] = args[2].get("engine_id")
        state["simulation_id"], state["pending_simulation"] = sim_id, None
        held, state["pending_logs"] = state["pending_logs"], []
        print(f"[db] simulation row created late (id {sim_id}); writing {len(held)} held telemetry rows")
        for t, raw, ai, res in held:
            await db_call(write_telemetry_row, sim_id, t, raw, ai, res)
    finally:
        state["sim_retrying"] = False


def write_telemetry_row(simulation_id, time_offset_s, raw, ai, residuals):
    if PHYSICS_VERSION in ("v3", "v4"):
        return db.log_telemetry(simulation_id, time_offset_s, raw, ai, residuals=residuals)
    return db.log_telemetry(simulation_id, time_offset_s, raw, ai)


def log_telemetry(simulation_id, time_offset_s, raw, ai):
    """Thread target for telemetry_logs writes. v3 (dbv3) also persists the physics
    residuals; db.py (v2) keeps its original signature."""
    if PHYSICS_VERSION in ("v3", "v4"):
        return db.log_telemetry(simulation_id, time_offset_s, raw, ai, residuals=state["residuals"])
    return db.log_telemetry(simulation_id, time_offset_s, raw, ai)
STEPS_PER_BROADCAST = 5  # 0.01s * 5 = 20Hz telemetry rate

# How often a telemetry_logs row is written, in SIMULATED seconds.
#
# This is deliberately decoupled from the AI cadence. ai.py's rolling window is
# defined in one-simulated-second samples, so the AI must keep being fed every
# second - throttling that would corrupt the very sequence the model was trained
# on. Only the DB write is thinned.
#
# 10s rather than 20s, chosen from the actual run distribution: the mean run here
# is ~237 simulated seconds, and the AI produces nothing for the first 128 of them
# (window fill). That leaves ~109 diagnostic seconds on an average run - about 11
# plottable points at 10s, but only ~5 at 20s. Five points cannot show a trend,
# which is the whole purpose of the chart.
DB_LOG_INTERVAL_S = 10

# Each selectable engine has its OWN genuinely-trained model set now - AI predictions
# are meaningful for 912/914/915/916, each using its own weights and scaler (see
# ai.py's ENGINE_REGISTRY).
# Fallback only: refresh_ai_version() replaces this with the engines the running AI
# service actually has models for (v3 has no 912 export yet).
AI_VALID_ENGINES = {"Rotax_914_ULF", "Rotax_912_ULS", "Rotax_915_iS", "Rotax_916_iS"}


state = {
    "running": False,
    "task": None,
    "last_telemetry": None,
    "last_ai_result": None,
    "ai_step_counter": 0,
    "ai_warmed_up": False,   # True once the AI has returned a real (non-warmup) result
    # v4: after a fault is removed, the last AI result with that fault taken out. Shown
    # while the AI's window refills in real time, so the cockpit never says "buffering".
    "ai_hold": None,
    # False while the AI service cannot be reached. Warmup fast-forward exists only to
    # fill the AI's 128 s window quickly; with no AI to fill it, fast-forwarding just
    # ran physics unthrottled (1,386 simulated seconds in ~40 s) and wrote a telemetry
    # row every ~0.3 s of wall time. Unreachable -> real-time pacing.
    "ai_reachable": True,
    "simulation_id": None,   # current Supabase simulations.id, or None if not persisted
    # A flight whose simulations row could not be created at /start (e.g. the network
    # was down): the start_simulation arguments, retried until it succeeds, and the
    # telemetry rows held meanwhile so the history is complete once it does.
    "pending_simulation": None,
    "pending_logs": [],
    "sim_retry_at": 0.0,
    "sim_retrying": False,
    "db_log_counter": 0,     # simulated seconds since the last telemetry_logs write
    # Tracks whether the PHYSICS SESSION is logically ongoing - deliberately
    # separate from simulation_id, which only tracks DB persistence and stays None
    # whenever no user_id is provided. Using simulation_id itself to decide
    # "is this a fresh start" was a real bug: with no user_id, simulation_id never
    # becomes non-None, so EVERY /start (including a genuine resume-from-pause)
    # was wrongly treated as fresh, resetting the twin's time/altitude/wear every
    # time. session_active survives a pause (frontend sends final=false to /stop)
    # and is only cleared by a genuine stop/reset/engine-switch.
    "session_active": False,
    "sim_time_offset": 0.0,  # seconds since this simulation started, for telemetry_logs
    "ai_model_version": None,  # reported by the AI service's /health on each fresh /start
    "residuals": None,         # latest residual.ResidualMonitor.update() output
    # ---- safety net (see simulation_loop) ----
    "sim_health": "idle",      # idle | running | recovered | halted
    "last_error": None,        # most recent loop/physics failure, for /health and the UI
    "recoveries": 0,           # physics recoveries this session
    "recovery_times": [],      # monotonic stamps inside RECOVERY_WINDOW_S
    "last_good_telemetry": None,
    "ai_resync": False,        # the AI's window has a gap: /reset it before the next sample
    "ai_settle_until": 0.0,    # sim time until which AI alerts are held (see hold_alerts_outside_envelope)
    "ai_last_sample_time": None,
    "ai_last_seen_time": None,   # last sample time sent to the AI; a drop means a new flight
    # Wall-clock ms from a physics sample entering the AI queue to its diagnosis being
    # stored for the next broadcast - the "real-time" figure reported in /health.
    "ai_latency_ms": deque(maxlen=300),
    "measured": {},            # latest bounded sensor values from POST /measured
    "measured_at": None,       # monotonic time of the last accepted /measured
    "measured_posts": 0,
    "last_data_source": "sim", # a switch sim <-> can restarts the AI window and residuals
    "rul_filter": {"ema": None, "shown": None, "t": None, "wear": None},   # see smooth_rul
    # ---- engine-failure shutdown (see health_failure_seconds) ----
    "health_zero_since": None,   # sim time health first hit the floor, None while it is above it
    "health_zero_engine_hours": None,  # hour meter at that instant, for the failure message
    "shutdown_engine_hours": None,     # engine hours the failure window consumed
    "shutdown_reason": None,     # "engine_failure" when the flight ended itself; None otherwise
    "closed_simulation_id": None,  # the row that shutdown closed, so the UI can poll its summary
    "residual_offsets": load_residual_offsets(),   # engine -> sensor offsets from zeroing (begin_zeroing)
    "ai_samples_dropped": 0,
    "db_pending": 0,
    "db_skipped": 0,
    "loop_lag_ms": 0.0,        # how far the loop is behind real time (EWMA)
    "steps": 0,
    # ---- physics v4 ----
    "scenario": None,          # preset chosen with POST /scenario, flown at the next fresh /start
    "ai_health": None,         # aiv4 /health: which engines have their own models, and their metrics
    "advisories_logged": {},   # advisory code -> level already written to maintenance_events
}

# Must be a SUPERSET of the AI service's feature columns, same names. ai.py (v2)
# uses the first 24; aiv3.py adds injection_timing. Both select their own columns
# by name, so sending the superset to either is harmless.
AI_FEATURE_COLS = [
    "altitude", "throttle", "airspeed", "aoa", "air_density",
    "torque_available_nm", "engine_rpm", "prop_rpm", "prop_torque", "power_kw", "fuel_flow",
    "thrust", "lift", "drag", "thrust_margin", "lift_weight_margin",
    "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm_fault",
    "injection_timing",
]
# v4: the 29 inputs the models were trained on (twin_v4.FEATURE_COLS), plus the hour
# meter and this flight's usage for the RUL head's aux inputs.
AI_FEATURE_COLS_V4 = twin_v4.FEATURE_COLS + ["engine_hours", "life_used_hours"]


def final_rul_hours(ai: dict):
    """RUL for simulations.final_rul_hours. ai.py (v2) reports simulated-timescale
    hours in rul_hours_internal; aiv3.py reports real engine hours in rul_hours."""
    if ai.get("model_version") in ("v3", "v4"):
        return ai.get("rul_hours")
    return ai.get("rul_hours_internal")


async def refresh_ai_version():
    """Records which model version the AI service is serving and warns on mismatch."""
    global AI_VALID_ENGINES
    try:
        resp = await ai_client.get(f"{AI_SERVICE_URL}/health")
        health = resp.json()
        state["ai_model_version"] = health.get("model_version", "v2")
        state["ai_health"] = health
        if isinstance(health.get("available_engines"), list) and health["available_engines"]:
            AI_VALID_ENGINES = set(health["available_engines"])
    except Exception:
        state["ai_model_version"] = None
        return
    if state["ai_model_version"] != PHYSICS_VERSION:
        print(f"WARNING: physics {PHYSICS_VERSION} but the AI service serves "
              f"{state['ai_model_version']} models - predictions are out of distribution. "
              "Run ai.py with physics v2, aiv3.py with AERO_PHYSICS_VERSION=v3.")


async def call_ai_service(telemetry: dict):
    """Sends one timestep to the AI service. Never raises: failures come back as an
    ai_service_unavailable result, and feed the circuit breaker."""
    try:
        cols = AI_FEATURE_COLS_V4 if PHYSICS_VERSION == "v4" else AI_FEATURE_COLS
        payload = {c: telemetry[c] for c in cols}
        payload["time"] = telemetry["time"]
        resp = await ai_client.post(f"{AI_SERVICE_URL}/step", json=safety.json_safe(payload))
        resp.raise_for_status()
        result = resp.json()
        if not isinstance(result, dict) or "status" not in result:
            raise ValueError(f"malformed AI response: {str(result)[:120]}")
        return result
    except Exception as e:
        return {"status": "ai_service_unavailable", "error": f"{type(e).__name__}: {e}"[:300]}


async def ai_reset_buffer():
    try:
        await ai_client.post(f"{AI_SERVICE_URL}/reset")
    except Exception:
        pass


# The v3 dataset never flies below 32 m/s (generate_dataset_v3.py airspeed floor), but
# every in-app flight starts with a ground roll. Until the AI's 128 s window holds only
# in-envelope samples its fault and failure-mode calls are extrapolation - the 916 raised
# a ~60 s false misfire/combustion alarm after every takeoff - so those alerts are held.
AI_ENVELOPE_MIN_AIRSPEED_MS = 32.0
AI_ENVELOPE_MIN_AIRSPEED_MS_V4 = 28.0   # generate_dataset_v4.py's airspeed floor
AI_WINDOW_S = 128.0


def hold_alerts_outside_envelope(sample: dict, result: dict) -> dict:
    """Return result with fault alerts cleared while the AI window still contains
    below-envelope (ground roll) samples. Health and RUL pass through unchanged."""
    if PHYSICS_VERSION not in ("v3", "v4"):
        return result
    t = sample.get("time")
    if not isinstance(t, (int, float)):
        return result
    last = state["ai_last_sample_time"]
    if last is not None and t < last:           # clock restarted: a new flight
        state["ai_settle_until"] = 0.0
    state["ai_last_sample_time"] = t
    airspeed = sample.get("airspeed")
    floor = AI_ENVELOPE_MIN_AIRSPEED_MS_V4 if PHYSICS_VERSION == "v4" else AI_ENVELOPE_MIN_AIRSPEED_MS
    if isinstance(airspeed, (int, float)) and airspeed < floor:
        state["ai_settle_until"] = t + AI_WINDOW_S
    if result.get("status") != "ok" or t >= state["ai_settle_until"]:
        return result
    if PHYSICS_VERSION == "v4":
        # Same rule for v4: while the window still holds ground roll every call is
        # extrapolation. Wear condition goes too - it is a model output on the same
        # window - and the model's own numbers stay available as *_raw.
        return {**result, "settling": True, "fault_detected": False, "detection_confidence": 0.0,
                "detection_confidence_raw": result.get("detection_confidence"),
                "settle_seconds_left": max(0.0, round(state["ai_settle_until"] - t, 1)),
                "settle_window_s": AI_WINDOW_S,
                "fault_modes": {k: {**v, "present": False} for k, v in (result.get("fault_modes") or {}).items()},
                "fault_modes_raw": result.get("fault_modes"), "faults_present": [],
                "sensors": {c: {**v, "condition": "none"} for c, v in (result.get("sensors") or {}).items()},
                "sensors_raw": result.get("sensors"), "faulty_sensors": [],
                "wear_condition": None, "health_percent": None,
                "health_percent_raw": result.get("health_percent")}
    # Severities go too: leaving them at 100% while the status reads nominal put a red number
    # next to a green "OK" in the cockpit.
    # Health goes with them: it is computed from the same detection and severity outputs, so
    # leaving it at 0 while the panel says nominal is the same contradiction in the other
    # direction. The model's own numbers stay available as *_raw.
    # Per-channel fault TYPES go too, or the cockpit prints "Bias" next to a 0% severity while
    # the status line says nominal.
    # The detector's confidence goes with fault_detected, or the cockpit prints
    # "100% conf" beside a nominal status - the same contradiction, one field over.
    diagnosis = result.get("diagnosis") or {}
    # How long the hold still has to run, so the cockpit can show progress instead of a
    # bare "stabilising". It is a countdown to a moving target: any sample below the
    # envelope pushes ai_settle_until out again, and the bar honestly goes back with it.
    held = {**result, "settling": True, "fault_detected": False, "faulty_channels": [],
            "settle_seconds_left": max(0.0, round(state["ai_settle_until"] - t, 1)),
            "settle_window_s": AI_WINDOW_S,
            "detection_confidence": 0.0, "detection_confidence_raw": result.get("detection_confidence"),
            "severity_percent": {ch: 0.0 for ch in (result.get("severity_percent") or {})},
            "severity_percent_raw": result.get("severity_percent") or {},
            "health_percent": 100.0, "health_percent_raw": result.get("health_percent"),
            "diagnosis": {ch: {**v, "fault_type": "none"} if isinstance(v, dict) else v
                          for ch, v in diagnosis.items()},
            "diagnosis_raw": diagnosis}
    modes = result.get("failure_modes")
    if isinstance(modes, dict):
        held["failure_modes"] = {k: ({**v, "present": False} if isinstance(v, dict) else v)
                                 for k, v in modes.items()}
    return held


# ---- engine-failure shutdown -------------------------------------------------
# A real UAV does not keep flying on an engine the health monitor has written off.
# Health pinned at the floor for this many CONSECUTIVE simulated seconds ends the
# flight the way a genuine failure would: throttle to idle, session closed with
# outcome "engine_failure". A transient dip (a hard throttle slam gives one or two
# seconds of it) resets the clock and changes nothing.
# AERO_HEALTH_FAILURE_HOLD_S overrides the hold; 0 disables the shutdown entirely,
# which is the switch to reach for if a demo must never be ended by the monitor.
HEALTH_FAILURE_PCT = 1.0
HEALTH_FAILURE_HOLD_S = float(os.environ.get("AERO_HEALTH_FAILURE_HOLD_S", 20.0))


def engine_hours_now():
    """The cockpit's engine hour meter: wear x TBO. None until telemetry carries both."""
    t = state["last_telemetry"] or {}
    w, tbo = t.get("wear"), t.get("tbo_hours")
    return w * tbo if isinstance(w, (int, float)) and isinstance(tbo, (int, float)) else None


def health_failure_seconds(sim_time: float, result) -> float:
    """Seconds of unbroken zero health. Zero whenever health recovers, the AI is not
    reporting, or alerts are held for the takeoff window - a held result reads
    nominal by construction and must never end a flight."""
    ai = result if isinstance(result, dict) else {}
    health = ai.get("health_percent")
    if (ai.get("status") != "ok" or ai.get("settling")
            or not isinstance(health, (int, float)) or health > HEALTH_FAILURE_PCT):
        state["health_zero_since"] = None
        return 0.0
    if state["health_zero_since"] is None or sim_time < state["health_zero_since"]:
        state["health_zero_since"] = sim_time      # first zero, or a new flight's clock
        # The hour meter at that instant, so the shutdown can report the window in the
        # unit the cockpit actually shows wear in - engine hours, not wall seconds.
        state["health_zero_engine_hours"] = engine_hours_now()
        return 0.0
    return sim_time - state["health_zero_since"]


# Remaining life can only fall - hours accumulate and nothing repairs the engine in flight -
# but the RUL head answers each 128 s window afresh, and its sample-to-sample jitter (a few
# hours) is larger than the real decrease over a minute, so the raw value wanders upward.
# What the cockpit, advisory and database see is smoothed and never rises within a flight:
#     shown = min(previous shown - engine hours consumed since, EMA of the model's RUL)
# The model's own value stays in rul_hours_raw. Reset on a new flight, engine or data source.
RUL_SMOOTH_S = 60.0


def reset_rul_filter():
    state["rul_filter"] = {"ema": None, "shown": None, "t": None, "wear": None}


def smooth_rul(sample: dict, result: dict) -> dict:
    if result.get("status") != "ok" or not isinstance(result.get("rul_hours"), (int, float)):
        return result
    # NOT WHILE ALERTS ARE HELD. The shown value never rises within a flight, so
    # whatever seeds the filter becomes a floor for the rest of it - and during the
    # takeoff hold the AI's 128-sample window still contains ground roll, which is
    # outside the envelope its RUL head was trained on. Measured on a 916 flight: the
    # head's raw prediction was 9.5 h low (inside its own 12.5 h band) while the
    # display sat 19.6 h low, the difference being an early held-window estimate the
    # filter could never climb back from. Passing raw through until the hold releases
    # lets the first in-envelope prediction set the floor instead.
    t, wear, tbo = sample.get("time"), sample.get("wear"), result.get("tbo_hours")
    raw = float(result["rul_hours"])
    # No engine has more life left than its overhaul interval. The head's output is
    # TBO * relu(z), which is unbounded above - measured live at 2101 h on a 2000 h
    # TBO during the takeoff window, i.e. "105% remaining" on the gauge.
    ceiling = float(tbo) if isinstance(tbo, (int, float)) and tbo else None

    def presented(value):
        value = max(0.0, value)
        return min(value, ceiling) if ceiling else value

    if result.get("settling"):
        # NO NUMBER AT ALL while the window is not clean. Passing the raw prediction
        # through was worse than useless: on a 914 the gauge read 99.1%, then 99.0%,
        # then 100% - RUL rising, which is the one thing it must never do. The window
        # still holds ground roll here, so the head is extrapolating and its output
        # wanders. The cockpit shows "stabilising" until the first in-envelope
        # prediction, which then seeds the floor (see reset above).
        reset_rul_filter()
        return {**result, "rul_ready": False, "rul_hours_raw": raw,
                "rul_hours": None, "rul_percent_remaining": None}
    f = state["rul_filter"]
    if f["t"] is not None and isinstance(t, (int, float)) and t < f["t"]:
        reset_rul_filter()                       # clock restarted: a new flight
        f = state["rul_filter"]
    if f["ema"] is None:
        f["ema"] = f["shown"] = presented(raw)
    else:
        dt = max(0.0, float(t) - f["t"]) if isinstance(t, (int, float)) and f["t"] is not None else 1.0
        f["ema"] += (1.0 - math.exp(-dt / RUL_SMOOTH_S)) * (raw - f["ema"])
        consumed = (max(0.0, float(wear) - f["wear"]) * float(tbo)
                    if isinstance(wear, (int, float)) and f["wear"] is not None and tbo else 0.0)
        f["shown"] = min(f["shown"] - consumed, f["ema"])
    f["shown"] = presented(f["shown"])
    f["t"] = t if isinstance(t, (int, float)) else f["t"]
    f["wear"] = float(wear) if isinstance(wear, (int, float)) else f["wear"]
    out = {**result, "rul_ready": True, "rul_hours_raw": raw,
           "rul_hours": round(f["shown"], 3)}
    if tbo:
        out["rul_percent_remaining"] = round(max(0.0, min(100.0, 100.0 * f["shown"] / float(tbo))), 4)
    return out


async def ai_worker(queue: asyncio.Queue):
    """The ONLY sender of /step. One worker, one queue: samples reach the AI strictly in
    simulated-time order however slow it is, and the loop never awaits the network.

    Circuit breaker: after 3 consecutive failures the AI is skipped for 5 s, then probed.
    Any gap in the AI's 128-sample window (dropped samples, an outage) makes the next
    sample start with /reset, so the model never sees a window with a hole in it.
    """
    while True:
        enqueued_at, out = await queue.get()
        try:
            if not ai_breaker.allow():
                state["ai_resync"] = True          # this sample is lost to the AI
                state["ai_reachable"] = False
                state["last_ai_result"] = {"status": "ai_service_unavailable",
                                           "error": ai_breaker.last_error or "AI service unreachable",
                                           "retrying": True}
                continue
            # A RESTARTED CLOCK IS A NEW FLIGHT, whatever the session bookkeeping says.
            # /start clears the AI's rolling buffer only when it treats the session as
            # fresh (session_active False); reopening the page after leaving without
            # Stop, or any resume path, leaves session_active True - and the AI then
            # predicts from a window still holding the previous flight's telemetry.
            # The simulated clock going backwards is the unambiguous signal, so resync
            # on it regardless of how the flight was started.
            sample_t = out.get("time")
            prev_t = state["ai_last_seen_time"]
            if (isinstance(sample_t, (int, float)) and isinstance(prev_t, (int, float))
                    and sample_t < prev_t):
                state["ai_resync"] = True
            if isinstance(sample_t, (int, float)):
                state["ai_last_seen_time"] = sample_t
            if state["ai_resync"]:
                state["ai_resync"] = False
                # A silent refill (fault removed) runs in real time: fast-forwarding would
                # visibly jump the flight ahead two minutes.
                state["ai_warmed_up"] = state["ai_hold"] is not None
                reset_rul_filter()          # the RUL floor belongs to the old flight
                await ai_reset_buffer()
            result = await call_ai_service(out)
            status = result.get("status")
            if status == "ok":   # warm-up samples are fast-forwarded, so they would skew it
                state["ai_latency_ms"].append((time.monotonic() - enqueued_at) * 1000.0)
            if status == "ai_unsupported_engine":
                # The service is healthy; it just has no model for this engine. Not a
                # failure, so no breaker trip and no resync churn.
                ai_breaker.record_success()
                state["ai_reachable"] = True
            elif status in ("ok", "warming_up"):
                if ai_breaker.record_success():
                    # Back from an outage: the AI's buffer spans the gap. Start clean.
                    state["ai_warmed_up"] = False
                    await ai_reset_buffer()
                state["ai_reachable"] = True
                if status == "ok":
                    state["ai_warmed_up"] = True
            else:
                # One failed or slow sample loses only that sample: the AI's window just
                # skips a second. Throwing the whole 128 s window away (the old rule) turned
                # a single slow reply into two minutes of "Buffering". The window is reset
                # only after a real outage - the breaker opening - or dropped backlog.
                ai_breaker.record_failure(result.get("error", ""))
                if ai_breaker.state != "closed":
                    state["ai_resync"] = True
                    state["ai_reachable"] = False
            if status == "warming_up" and state["ai_hold"] is not None:
                state["last_ai_result"] = state["ai_hold"]
            else:
                if status == "ok":
                    state["ai_hold"] = None
                state["last_ai_result"] = smooth_rul(out, hold_alerts_outside_envelope(out, result))
        except asyncio.CancelledError:
            raise
        except Exception as e:                      # never let the worker die
            print(f"[ai_worker] unexpected error: {e}")
            state["ai_resync"] = True
        finally:
            queue.task_done()


def spawn_db_write(fn, *args):
    """Fire-and-forget Supabase write, bounded: a slow database never piles up threads."""
    if state["db_pending"] >= DB_MAX_PENDING:
        state["db_skipped"] += 1
        return

    async def run():
        state["db_pending"] += 1
        try:
            await asyncio.to_thread(fn, *args)
        except Exception as e:
            print(f"[db] background write failed: {e}")
        finally:
            state["db_pending"] -= 1
    asyncio.create_task(run())


async def db_call(fn, *args, timeout: float = DB_TIMEOUT_S):
    """Awaited Supabase call with a deadline. Returns None on timeout or error, so an
    endpoint (Stop, Reset, Resume) can never hang on the database."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout)
    except Exception as e:
        print(f"[db] {getattr(fn, '__name__', fn)} failed or timed out: {e}")
        return None

# WebSocket -> per-client frame queue. Each browser has its own sender task, so the
# simulation loop never awaits a socket.
clients: dict = {}


class ParamUpdate(BaseModel):
    altitude: Optional[float] = None
    throttle: Optional[float] = None
    airspeed: Optional[float] = None
    aoa: Optional[float] = None
    # ISA temperature deviation in degrees C (hot/cold day). Clamped in the
    # handler. Deliberately NOT an AI feature - see AI_FEATURE_COLS.
    isa_dev_c: Optional[float] = None
    # physics v4 installation and fuel state (twin_v4.DEFAULT_ENV); ignored by v2/v3.
    qnh_offset_pa: Optional[float] = None
    humidity_frac: Optional[float] = None
    fuel_octane_mon: Optional[float] = None
    fuel_ethanol_frac: Optional[float] = None
    cooling_airflow_factor: Optional[float] = None
    electrical_load_a: Optional[float] = None
    target_lambda: Optional[float] = None
    oil_thermostat_open: Optional[bool] = None
    # "can" from can_ingest.py. While the aircraft is on the bus it owns the set-points,
    # and untagged updates (the cockpit sliders/autopilot) are ignored.
    source: Optional[str] = None


class MeasuredUpdate(BaseModel):
    """Sensor readings from the aircraft's CAN bus (can_ingest.py). Units as physics v3."""
    rpm: Optional[float] = None
    egt: Optional[float] = None
    cht: Optional[float] = None
    oil_pressure: Optional[float] = None
    oil_temp: Optional[float] = None
    vibx: Optional[float] = None
    viby: Optional[float] = None
    vibz: Optional[float] = None
    fuel_flow: Optional[float] = None
    # physics v4 only
    coolant_temp: Optional[float] = None
    manifold_pressure_kpa: Optional[float] = None
    battery_voltage: Optional[float] = None


async def client_sender(ws: WebSocket, queue: asyncio.Queue):
    """Writes queued frames to one browser. A tab that is busy (heavy 3D rendering, a
    background tab) simply receives fewer, newer frames. One that accepts nothing for
    CLIENT_STALL_S is closed - closed, not just forgotten, so the page's reconnect logic
    sees onclose and opens a fresh connection instead of waiting on a dead socket."""
    try:
        while True:
            payload = await queue.get()
            await asyncio.wait_for(ws.send_text(payload), timeout=CLIENT_STALL_S)
    except asyncio.CancelledError:
        raise
    except Exception:
        pass
    finally:
        clients.pop(ws, None)
        try:
            await asyncio.wait_for(ws.close(code=1011), timeout=1.0)
        except Exception:
            pass


async def broadcast(msg: dict):
    """Hands the frame to every client's queue without awaiting any socket. When a
    client's queue is full its oldest frame is dropped - live telemetry only ever needs
    the newest state."""
    if not clients:
        return
    try:
        payload = json.dumps(safety.json_safe(msg), allow_nan=False)
    except (TypeError, ValueError) as e:
        print(f"[broadcast] unserialisable payload skipped: {e}")
        return
    for ws, (queue, _task) in list(clients.items()):
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(payload)


def ai_status_label() -> str:
    r = state["last_ai_result"] or {}
    if r.get("status") == "ai_unsupported_engine":
        return "unsupported"
    if r.get("status") == "ok":
        return "ok"
    if r.get("status") == "warming_up":
        return "warming_up"
    if r.get("status") == "ai_service_unavailable" or not state["ai_reachable"]:
        return "unavailable"
    return "pending"


def ai_latency_stats() -> Optional[dict]:
    """p50/p95/max of the recent sample-to-diagnosis latency, or None before any sample."""
    samples = sorted(state["ai_latency_ms"])
    if not samples:
        return None
    pick = lambda q: samples[min(len(samples) - 1, int(q * len(samples)))]
    return {"p50": round(pick(0.50), 1), "p95": round(pick(0.95), 1),
            "max": round(samples[-1], 1), "n": len(samples)}


def sim_status() -> dict:
    """Compact health block attached to every broadcast and to /health."""
    return {
        "health": state["sim_health"],
        "recoveries": state["recoveries"],
        "last_error": state["last_error"],
        "ai": ai_status_label(),
        "ai_breaker": ai_breaker.state,
        "loop_lag_ms": round(state["loop_lag_ms"], 1),
        "ai_latency_ms": ai_latency_stats(),
        # Set once, on the broadcast that ends a flight the engine could not finish.
        "shutdown_reason": state["shutdown_reason"],
        "closed_simulation_id": state["closed_simulation_id"],
        "shutdown_hold_s": HEALTH_FAILURE_HOLD_S,
        # Engine hours the failure window consumed - the cockpit shows wear in hours,
        # and 20 s of full-load flight is several of them.
        "shutdown_engine_hours": state["shutdown_engine_hours"],
    }


def record_recovery(err: BaseException) -> bool:
    """Rebuild the twin from the last good step. False when the session must halt."""
    global twin
    now = time.monotonic()
    state["recovery_times"] = [t for t in state["recovery_times"] if now - t < RECOVERY_WINDOW_S] + [now]
    state["last_error"] = f"{type(err).__name__}: {err}"[:300]
    print(f"[simulation] physics fault, attempt {len(state['recovery_times'])}: {state['last_error']}")
    traceback.print_exception(err)
    if len(state["recovery_times"]) > MAX_RECOVERIES:
        state["sim_health"] = "halted"
        return False
    fresh = new_twin(CURRENT_ENGINE_MODEL)
    snap = state["last_good_telemetry"]
    if snap:
        try:
            fresh.restore_state(snap)
        except Exception as restore_err:
            print(f"[simulation] restore from last good step failed, using clean twin: {restore_err}")
            fresh = new_twin(CURRENT_ENGINE_MODEL)
            for k in ("altitude", "throttle", "airspeed", "aoa", "isa_dev_c"):
                v = snap.get(k)
                if isinstance(v, (int, float)) and safety.clamp_param(k, v) is not None:
                    setattr(fresh, k, safety.clamp_param(k, v))
    twin = fresh
    state["recoveries"] += 1
    state["sim_health"] = "recovered"
    state["ai_resync"] = True          # the AI window must not bridge the fault
    return True


async def simulation_loop():
    """Runs until state['running'] is set False. Every stage is guarded, so a single
    fault degrades one feature for one step instead of killing the flight.

    SAFETY NET
      physics   each step is validated (safety.telemetry_problem). An exception or a
                non-finite value rebuilds the twin from the last good step; more than
                MAX_RECOVERIES in RECOVERY_WINDOW_S halts the session cleanly (the UI is
                told, Start works again) instead of looping on a broken state.
      AI        never awaited here. Samples go to ai_worker through a bounded queue;
                a stalled AI drops samples and resyncs rather than blocking physics.
      residual  / advisory / broadcast / database: each isolated, failures logged.
      crash     anything unexpected ends the loop with sim_health "halted" and
                running False - never a silently dead task behind running=True.

    WARMUP FAST-FORWARD: the AI needs 128 simulated seconds before its first
    prediction, so until it is warmed up the loop runs faster than real time - but only
    as fast as the AI worker keeps up (AI_WARMUP_BACKLOG), and only while the AI is
    reachable. Samples still arrive strictly in order."""
    global twin
    queue: asyncio.Queue = asyncio.Queue(maxsize=AI_QUEUE_MAX)
    worker = asyncio.create_task(ai_worker(queue))
    step_count = 0
    next_tick = time.monotonic()
    if state["sim_health"] in ("idle", "halted"):
        state["sim_health"] = "running"
    try:
        while state["running"]:
            # ---- physics (guarded) ----
            try:
                out = twin.step()
                problem = safety.telemetry_problem(out)
                if problem:
                    raise FloatingPointError(f"invalid physics step: {problem}")
            except Exception as err:
                if not record_recovery(err):
                    state["running"] = False
                    await broadcast({**(state["last_good_telemetry"] or {}), "ai": state["last_ai_result"],
                                     "sim_status": sim_status()})
                    break
                await asyncio.sleep(0)
                continue
            # Recovery restores the twin from its OWN step; everything downstream sees
            # the measured sensors when the aircraft is on the bus.
            state["last_good_telemetry"] = out
            out = apply_measured(out)
            state["last_telemetry"] = out
            state["steps"] += 1
            step_count += 1

            # ---- once per simulated second: residuals, AI, database ----
            state["ai_step_counter"] += 1
            # v4: the twin reads its sensors once per flight second and flags that step.
            new_sample = (bool(out.get("sample_new")) if PHYSICS_VERSION == "v4"
                          else state["ai_step_counter"] >= AI_STEPS_PER_CALL)
            if new_sample:
                state["ai_step_counter"] = 0
                state["sim_time_offset"] += 1.0
                # A new data stream (the aircraft joined or left the bus): the AI window and
                # the residual lag states describe the old one, so both start again.
                if out.get("data_source") != state["last_data_source"]:
                    state["last_data_source"] = out.get("data_source")
                    residual_monitor.reset()
                    state["residuals"] = None
                    state["ai_resync"] = True
                    state["ai_warmed_up"] = False
                    reset_rul_filter()
                try:
                    state["residuals"] = (residual.twin_residuals_v4(out) if PHYSICS_VERSION == "v4"
                                          else residual_monitor.update(out, dt=1.0))
                    if isinstance(state["residuals"], dict) and state["residuals"].get("zeroed") is not None:
                        state["residual_offsets"][CURRENT_ENGINE_MODEL] = dict(state["residuals"]["zeroed"])
                        save_residual_offsets(state["residual_offsets"])
                    if isinstance(state["residuals"], dict):
                        state["residuals"]["offsets"] = dict(residual_monitor.offsets)
                except Exception as e:
                    state["residuals"] = {"enabled": False, "reason": f"residual monitor error: {e}"}

                try:
                    queue.put_nowait((time.monotonic(), out))
                except asyncio.QueueFull:
                    # The AI is stalled. Drop the backlog rather than block physics;
                    # the worker resets the AI's window before its next sample.
                    dropped = 0
                    while not queue.empty():
                        queue.get_nowait(); queue.task_done(); dropped += 1
                    state["ai_samples_dropped"] += dropped + 1
                    state["ai_resync"] = True
                    queue.put_nowait((time.monotonic(), out))

                # ---- engine failure ----
                # Health written off for HEALTH_FAILURE_HOLD_S straight: the flight ends
                # itself rather than cruising on with a 0% engine and a red panel.
                if should_end_flight(state["sim_time_offset"], state["last_ai_result"]):
                    await engine_failure_shutdown()
                    break

                state["db_log_counter"] += 1
                if state["db_log_counter"] >= DB_LOG_INTERVAL_S:
                    state["db_log_counter"] = 0
                    if state["simulation_id"] is None and state["pending_simulation"] is not None:
                        if len(state["pending_logs"]) < PENDING_LOGS_MAX:
                            state["pending_logs"].append((state["sim_time_offset"], out,
                                                          state["last_ai_result"], state["residuals"]))
                        if time.monotonic() >= state["sim_retry_at"]:
                            state["sim_retry_at"] = time.monotonic() + SIM_RETRY_S
                            asyncio.create_task(retry_simulation_row())
                    else:
                        spawn_db_write(log_telemetry, state["simulation_id"], state["sim_time_offset"],
                                       out, state["last_ai_result"])
                if PHYSICS_VERSION == "v4" and state["simulation_id"] is not None:
                    log_new_advisories(out)

                # Warm-up fast-forward waits (briefly) for the AI to keep its order.
                if not state["ai_warmed_up"] and ai_breaker.state == "closed":
                    waited = 0.0
                    while (queue.qsize() >= AI_WARMUP_BACKLOG and state["running"]
                           and waited < AI_WARMUP_MAX_WAIT_S and ai_breaker.state == "closed"):
                        await asyncio.sleep(0.005)
                        waited += 0.005

            # ---- broadcast at 20 Hz (guarded) ----
            if step_count % STEPS_PER_BROADCAST == 0:
                try:
                    payload = client_frame(out)
                    payload["ai"] = state["last_ai_result"]
                    payload["residuals"] = state["residuals"]
                    try:
                        payload["advisory"] = advisory.build_advisory(state["last_ai_result"], out, state["residuals"])
                    except Exception as e:
                        payload["advisory"] = {"severity": "nominal", "headline": "Advisory unavailable",
                                               "insufficient_data": True, "items": [], "error": str(e)[:200]}
                    payload["sim_status"] = sim_status()
                    await broadcast(payload)
                except Exception as e:
                    print(f"[simulation] broadcast failed: {e}")
                if state["sim_health"] == "recovered" and not state["recovery_times"]:
                    state["sim_health"] = "running"

            # ---- pacing ----
            # Only while the AI is genuinely keeping up: any recent failure (even before
            # the breaker opens) drops to real time, so an outage never speeds physics up.
            # Never while the aircraft is on the CAN bus: it streams in real time, and a
            # fast-forwarded twin would fill the AI window with seconds of stale readings.
            fast_forward = (not state["ai_warmed_up"] and state["ai_reachable"]
                            and ai_breaker.state == "closed" and ai_breaker.failures == 0
                            and ai_status_label() != "unsupported"
                            and out.get("data_source") != "can")
            if fast_forward:
                next_tick = time.monotonic()
                await asyncio.sleep(0)
            else:
                next_tick += twin.dt
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    state["loop_lag_ms"] *= 0.98
                    await asyncio.sleep(sleep_for)
                else:
                    state["loop_lag_ms"] = 0.98 * state["loop_lag_ms"] + 0.02 * (-sleep_for * 1000.0)
                    next_tick = time.monotonic()   # fell behind, resync rather than spiral

            # Recovery stamps age out of the window on their own.
            if state["recovery_times"] and time.monotonic() - state["recovery_times"][-1] >= RECOVERY_WINDOW_S:
                state["recovery_times"] = []
    except asyncio.CancelledError:
        raise
    except Exception as err:
        state["last_error"] = f"{type(err).__name__}: {err}"[:300]
        state["sim_health"] = "halted"
        state["running"] = False
        print("[simulation] loop crashed, session halted safely:")
        traceback.print_exception(err)
        try:
            await broadcast({**(state["last_good_telemetry"] or {}), "ai": state["last_ai_result"],
                             "sim_status": sim_status()})
        except Exception:
            pass
    finally:
        worker.cancel()
        try:
            await worker
        except (asyncio.CancelledError, Exception):
            pass


async def stop_loop(timeout: float = 1.5):
    """Stops the loop and WAITS for it to exit, so a fast Stop->Start can never leave two
    loops stepping the same twin."""
    state["running"] = False
    task = state["task"]
    state["task"] = None
    if task is None or task.done():
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


async def start_loop():
    await stop_loop()
    state["running"] = True
    # Last flight's failure card must not reappear over this one.
    state["shutdown_reason"] = None
    state["closed_simulation_id"] = None
    state["health_zero_since"] = None
    state["health_zero_engine_hours"] = None
    state["shutdown_engine_hours"] = None
    if state["sim_health"] == "halted":
        state["recovery_times"] = []
        state["sim_health"] = "running"
    state["task"] = asyncio.create_task(simulation_loop())


class StartRequest(BaseModel):
    user_id: Optional[str] = None   # Supabase auth.users.id, if the frontend user is logged in
    # v4 only. engine_id continues one of the user's saved engines from its hour meter;
    # scenario starts a fresh engine from a preset (GET /scenarios). Neither: the preset
    # chosen with POST /scenario, else the default.
    engine_id: Optional[int] = None
    scenario: Optional[str] = None
    # v4: the engine's atmosphere, fuel and installation inputs for this flight (a
    # mission's hot day, or what the pilot set before Start), applied to the fresh
    # engine - a separate /params could land before /start has built it.
    inputs: Optional[dict] = None
    # v4: the mission profile being flown (frontend lib/missionPresets.ts id), recorded
    # on the simulations row so reports and replays can name the sortie.
    mission: Optional[str] = None


async def choose_engine(req: "StartRequest") -> tuple[Optional[dict], Optional[str]]:
    """(engine record, None) or (None, reason it cannot fly)."""
    if req.engine_id is not None:
        if not req.user_id:
            return None, "sign in to continue a saved engine"
        rec = await db_call(db.get_engine, req.engine_id)
        if not rec or rec.get("user_id") != req.user_id:
            return None, "engine not found"                 # same answer for another user's engine
        if rec["engine_model"] != CURRENT_ENGINE_MODEL:
            return None, f"that engine is a {rec['engine_model']} - select it before starting"
        if rec.get("status") == "worn_out":
            return None, "that engine is worn out - it needs an overhaul before it flies again"
        return rec, None
    try:
        return scenarios_v4.make_engine(CURRENT_ENGINE_MODEL, req.scenario or state["scenario"]
                                        or scenarios_v4.DEFAULT_PRESET), None
    except ValueError as e:
        return None, str(e)


def client_frame(out: dict) -> dict:
    """A telemetry step as browsers receive it. v4: the restore data (physics state,
    engine record) stays in last_telemetry for /stop and recovery and is left out of
    the frame; the engine being flown is summarised instead."""
    frame = dict(out)
    if PHYSICS_VERSION == "v4":
        frame.pop("physics_state", None)
        frame.pop("engine_record", None)
        frame["engine"] = engine_summary()
    return frame


def engine_summary() -> dict:
    """The engine being flown, for the cockpit (v4)."""
    rec = getattr(twin, "record", None) or {}
    placeholders = (state["ai_health"] or {}).get("placeholder_engines") or []
    return {"engine_id": rec.get("engine_id"), "scenario": rec.get("scenario"),
            "start_engine_hours": getattr(twin, "start_engine_hours", None),
            "tbo_hours": getattr(twin, "TBO_HOURS", None), **timescale_v4.describe(),
            "placeholder_models": CURRENT_ENGINE_MODEL in placeholders}


def model_manifest_snapshot() -> Optional[dict]:
    """Test metrics of the models about to fly, stored on the run (v4)."""
    return ((state["ai_health"] or {}).get("metrics") or {}).get(CURRENT_ENGINE_MODEL)


def log_new_advisories(out: dict) -> None:
    """Maintenance history on the life clock: an advisory at caution or above is
    written when it first appears or escalates, never once a second."""
    try:
        adv = advisory.build_advisory(state["last_ai_result"], out, state["residuals"])
    except Exception:
        return
    rec = getattr(twin, "record", None) or {}
    for it in adv.get("items") or []:
        level = it.get("severity")
        if level not in ("caution", "warning"):
            continue
        prev = state["advisories_logged"].get(it["code"])
        if prev == level or prev == "warning":
            continue
        state["advisories_logged"][it["code"]] = level
        spawn_db_write(db.log_maintenance_event, rec.get("engine_id"), state["simulation_id"],
                       out.get("engine_hours"), level, it.get("message", "")[:500], it.get("action", "")[:500])


@app.post("/start")
async def start_sim(req: StartRequest = StartRequest()):
    global twin
    if state["running"] and state["task"] is not None and not state["task"].done():
        return {"status": "already_running"}
    # /start is also called to RESUME from a pause (frontend's onTogglePause -
    # pausing calls /stop with final=false, resuming calls /start again, since the
    # backend loop must actually halt while paused). The freshness check MUST be
    # session_active, NOT simulation_id - simulation_id stays None whenever no
    # user_id is provided (anonymous use), which would otherwise make EVERY start
    # look "fresh" including genuine resumes, silently resetting the twin's
    # time/altitude/wear on every pause/resume cycle. This was a real, confirmed
    # bug - session_active is independent of whether persistence succeeds.
    if not state["session_active"]:
        engine = None
        if PHYSICS_VERSION == "v4":
            engine, problem = await choose_engine(req)
            if problem:
                return {"status": "error", "message": problem}
        state["session_active"] = True
        state["advisories_logged"] = {}
        twin = new_twin(CURRENT_ENGINE_MODEL, engine)
        # A fresh session takes off from the ground, matching the frontend's own
        # auto-climb takeoff sequence (Simulator.tsx starts its visual at altitude 0
        # and climbs automatically) - the twin's own __init__ default (2000m) would
        # otherwise briefly mismatch that visual until the first /params call caught up.
        twin.altitude = 0.0
        if req.inputs:
            apply_env_inputs(req.inputs)
        state["sim_time_offset"] = 0.0
        state["last_telemetry"] = None
        state["last_ai_result"] = None
        state["ai_hold"] = None
        state["ai_step_counter"] = 0
        state["ai_warmed_up"] = False
        state["ai_resync"] = False
        state["recoveries"] = 0
        state["recovery_times"] = []
        state["last_error"] = None
        state["last_good_telemetry"] = None
        state["sim_health"] = "idle"
        ai_breaker.reset()
        reset_residuals()
        # REAL BUG FIXED HERE: this branch reset the PHYSICS twin for a fresh
        # session, but never told ai.py to clear its own rolling 128-step buffer -
        # unlike /reset and /select_engine, which both already do this. Without it,
        # a fresh start's first ~128 seconds of predictions were computed from a
        # window mixing the PREVIOUS session's stale telemetry with the new
        # session's genuinely fresh data, producing exactly the "starts low then
        # climbs" artifact as the old data got pushed out of the sliding window.
        try:
            await ai_client.post(f"{AI_SERVICE_URL}/reset")
        except Exception:
            pass
        await refresh_ai_version()
        # simulation_id tracks DB persistence ONLY - independent of session_active,
        # since a session can be genuinely fresh but still have no user_id to
        # persist under (db.start_simulation returns None in that case, by design).
        mission = (req.mission or "")[:64] or None
        retry_args = pending_simulation_args(req.user_id, mission)
        state["pending_logs"], state["sim_retry_at"] = [], 0.0
        if PHYSICS_VERSION == "v4":
            state["simulation_id"] = await db_call(db.start_simulation, req.user_id, CURRENT_ENGINE_MODEL,
                                                   twin.record, model_manifest_snapshot(), mission)
        else:
            state["simulation_id"] = await db_call(db.start_simulation, req.user_id, CURRENT_ENGINE_MODEL)
        # Not persisted although it should be: keep retrying from the simulation loop.
        state["pending_simulation"] = retry_args if state["simulation_id"] is None else None
    await start_loop()
    return {"status": "started", "simulation_id": state["simulation_id"]}


class SummarizeRequest(BaseModel):
    user_id: str


class StopRequest(BaseModel):
    # False = this is a PAUSE, not a genuine stop - the frontend's onTogglePause
    # sends final=false, since the physics session should still be resumable
    # (session_active stays True, twin state is preserved). True (the default,
    # matching onStopClick's genuine Stop button and any caller that omits this)
    # ends the session for real - the next /start will create a fresh twin.
    final: bool = True


async def close_session(outcome: str):
    """Persist the end of a flight: one last telemetry row at the exact moment of
    stopping, the simulations row closed with `outcome`, and the narrative summary
    kicked off. Returns the closed row id (None when nothing was being persisted).
    Shared by /stop and the engine-failure shutdown, so a flight that ends itself is
    recorded exactly like one the pilot ended."""
    if state["simulation_id"] is None and state["pending_simulation"] is not None:
        await retry_simulation_row()               # one last try before the flight is lost
    state["pending_simulation"], state["pending_logs"] = None, []
    if state["simulation_id"] is None:
        return None
    ai = state["last_ai_result"] or {}
    # One last row at the exact moment of stopping, regardless of where the
    # 10s interval happened to fall. Without this the logged series can end
    # up to 10 simulated seconds before the state stored in final_telemetry,
    # so a chart would disagree with the run's own summary numbers.
    if state["last_telemetry"] is not None:
        await db_call(log_telemetry, state["simulation_id"],
                      state["sim_time_offset"], state["last_telemetry"], ai)
    # state["last_telemetry"] is the full raw physics dict as it stood at the
    # exact moment of stopping - already proven JSON-serializable, since this
    # same dict passes through json.dumps() in every WebSocket broadcast.
    await db_call(db.end_simulation, state["simulation_id"], outcome,
                  ai.get("health_percent"), final_rul_hours(ai), safety.json_safe(state["last_telemetry"]))
    # Post-flight narrative summary. Fire-and-forget ON PURPOSE: /stop is also
    # reached via navigator.sendBeacon on pagehide, which cannot consume a
    # response at all, and the Stop button awaits res.ok - so a synchronous
    # LLM call here would add seconds of latency to a working path and make
    # it depend on a third-party API. The frontend polls Supabase for the
    # result instead. Captured into a local first because the next line
    # clears state["simulation_id"] before the task ever runs.
    _sim_id = state["simulation_id"]
    spawn_background(summary.generate_summary, _sim_id)
    state["simulation_id"] = None
    return _sim_id


def should_end_flight(sim_time: float, result) -> bool:
    """True when this flight must end: health at the floor for HEALTH_FAILURE_HOLD_S
    consecutive simulated seconds.

    The zero case is the whole reason this is a function. The loop used to test
    `health_failure_seconds(...) >= HEALTH_FAILURE_HOLD_S` inline, so setting the hold
    to 0 to DISABLE the shutdown instead made it fire on the first simulated second of
    every flight - healthy, warming up, or not reporting at all - because 0 >= 0. A
    bench run with AERO_HEALTH_FAILURE_HOLD_S=0 halted every flight at t=1 s.
    """
    if HEALTH_FAILURE_HOLD_S <= 0:
        return False
    return health_failure_seconds(sim_time, result) >= HEALTH_FAILURE_HOLD_S


async def engine_failure_shutdown():
    """End the flight the way the engine just did: throttle to idle, loop stopped,
    session closed as "engine_failure", and one final frame broadcast carrying the
    reason - which is what the cockpit turns into its failure card. Called from
    inside the loop, so it sets running False and returns rather than awaiting
    stop_loop() (which would wait for the very task calling it)."""
    ai = state["last_ai_result"] or {}
    now_h, then_h = engine_hours_now(), state["health_zero_engine_hours"]
    state["shutdown_engine_hours"] = (round(now_h - then_h, 2)
                                      if now_h is not None and then_h is not None else None)
    state["running"] = False
    state["session_active"] = False
    state["shutdown_reason"] = "engine_failure"
    state["health_zero_since"] = None
    state["sim_health"] = "halted"
    try:
        twin.throttle = 0.0
    except Exception:
        pass
    print(f"[simulation] engine failure: health at {ai.get('health_percent')}% for "
          f"{HEALTH_FAILURE_HOLD_S:.0f}s ({state['shutdown_engine_hours']} engine hours) - "
          f"flight ended at t={state['sim_time_offset']:.0f}s")
    state["closed_simulation_id"] = await close_session("engine_failure")
    try:
        await broadcast({**(state["last_telemetry"] or {}), "ai": ai,
                         "residuals": state["residuals"], "sim_status": sim_status()})
    except Exception as e:
        print(f"[simulation] final broadcast failed: {e}")


@app.post("/stop")
async def stop_sim(req: StopRequest = StopRequest()):
    await stop_loop()
    if req.final:
        state["session_active"] = False
    if req.final:
        # Returned so the caller can poll for the summary close_session started -
        # /stop previously returned no id at all, which left the frontend with no
        # way to find the row it had just closed.
        return {"status": "stopped", "final": True, "simulation_id": await close_session("stopped")}
    return {"status": "stopped", "final": req.final, "simulation_id": None}


@app.post("/reset")
async def reset_sim():
    global twin
    was_running = state["running"]
    await stop_loop()
    # Previously built without engine_model, so every reset silently rebuilt a 914
    # twin while the AI service stayed on whichever engine was selected.
    twin = new_twin(CURRENT_ENGINE_MODEL)
    # A reset genuinely ends whatever flight was in progress - close out its
    # Supabase record properly rather than silently abandoning it. This endpoint
    # relaunches the loop directly (not via /start), so it does not have a user_id
    # to open a fresh row - the NEXT /start call will create one normally.
    if state["simulation_id"] is not None:
        ai = state["last_ai_result"] or {}
        await db_call(db.end_simulation, state["simulation_id"], "reset",
                      ai.get("health_percent"), final_rul_hours(ai), safety.json_safe(state["last_telemetry"]))
        state["simulation_id"] = None
    state["pending_simulation"], state["pending_logs"] = None, []
    state["last_telemetry"] = None
    state["last_ai_result"] = None
    state["ai_hold"] = None
    state["ai_step_counter"] = 0
    state["ai_warmed_up"] = False
    state["ai_resync"] = False
    state["recoveries"] = 0
    state["recovery_times"] = []
    state["last_error"] = None
    state["last_good_telemetry"] = None
    state["sim_health"] = "idle"
    ai_breaker.reset()
    reset_residuals()
    # This endpoint already creates its own fresh twin directly (above), bypassing
    # /start's session_active check entirely - keep session_active consistent with
    # whatever this reset actually does: True if the loop is relaunched (an
    # ongoing session continuing with fresh physics), False if not (idle, ready
    # for a genuinely fresh /start next time).
    state["session_active"] = was_running
    try:
        await ai_client.post(f"{AI_SERVICE_URL}/reset")   # clear the AI's rolling 128-step buffer too
    except Exception:
        pass   # AI service down is not a reason to fail the reset - degrade gracefully
    if was_running:
        await start_loop()
    return {"status": "reset"}


@app.get("/engines")
async def list_engines():
    """Single source of truth is physics.py's ENGINE_CONFIGS - the frontend never
    hardcodes a duplicate list, it just asks this endpoint."""
    return {
        "engines": list(ENGINE_CONFIGS.keys()),
        "current": CURRENT_ENGINE_MODEL,
        "ai_valid_engines": list(AI_VALID_ENGINES),
    }


class EngineSelect(BaseModel):
    engine_model: str


@app.post("/select_engine")
async def select_engine(sel: EngineSelect):
    """Switches the physics twin to a different engine model. ALWAYS leaves the
    simulation stopped afterward - selecting an engine is the start of a genuinely
    new session, and must never auto-continue a previous run's physics state.

    This matters even (especially) when the previous session was simply abandoned -
    e.g. the user closed the /simulate tab without clicking Stop, so state["running"]
    was still True on the backend. An earlier version of this endpoint restarted the
    loop automatically in that case, which silently continued the OLD session's
    physics under the NEW engine without ever calling /start or creating a fresh
    Supabase record - the exact bug where "selecting an engine sometimes restarts
    the old one" came from. Now it always requires an explicit /start."""
    global twin, CURRENT_ENGINE_MODEL
    if sel.engine_model not in ENGINE_CONFIGS:
        return {"status": "error", "message": f"Unknown engine_model. Options: {list(ENGINE_CONFIGS)}"}

    await stop_loop()
    state["session_active"] = False   # always ends the session - see docstring
    # A different engine is genuinely a different flight/aircraft - close out
    # whatever simulation record was open under the OLD engine before switching.
    if state["simulation_id"] is not None:
        ai = state["last_ai_result"] or {}
        await db_call(db.end_simulation, state["simulation_id"], "engine_switched",
                      ai.get("health_percent"), final_rul_hours(ai), safety.json_safe(state["last_telemetry"]))
        state["simulation_id"] = None
    state["pending_simulation"], state["pending_logs"] = None, []
    CURRENT_ENGINE_MODEL = sel.engine_model
    state["scenario"] = None          # presets depend on the hardware; choose again
    twin = new_twin(CURRENT_ENGINE_MODEL)
    state["last_telemetry"] = None
    state["last_ai_result"] = None
    state["ai_hold"] = None
    state["ai_step_counter"] = 0
    state["ai_warmed_up"] = False
    state["ai_resync"] = False
    state["recoveries"] = 0
    state["recovery_times"] = []
    state["last_error"] = None
    state["last_good_telemetry"] = None
    state["sim_health"] = "idle"
    ai_breaker.reset()
    reset_residuals()
    try:
        # Tell ai.py WHICH engine's models to switch to as well (not just reset) -
        # it maintains its own active_engine independently, this keeps the two
        # services in agreement. ai.py's /select_engine also resets its buffer,
        # so a separate /reset call here would be redundant.
        await ai_client.post(f"{AI_SERVICE_URL}/select_engine", json={"engine_model": CURRENT_ENGINE_MODEL})
    except Exception:
        pass
    await refresh_ai_version()
    # No auto-restart here, intentionally - see docstring.

    return {
        "status": "ok",
        "engine_model": CURRENT_ENGINE_MODEL,
        "ai_valid": CURRENT_ENGINE_MODEL in AI_VALID_ENGINES,
        "max_power_kw": twin.MAX_POWER_KW,
    }


class ResumeRequest(BaseModel):
    simulation_id: int
    user_id: str


@app.post("/resume")
async def resume_sim(req: ResumeRequest):
    """Restores the physics twin to the EXACT state a past simulation stopped at
    (see physics.py's restore_state) and CONTINUES THE SAME Supabase run.

    This previously opened a brand-new simulations row per resume, which
    fragmented a single flight across several rows: each one held a slice of the
    telemetry, every chart restarted its time axis at zero, and no page could
    show the flight as the continuous thing it physically is. Resuming now
    reuses the original id and carries the telemetry time axis on from the last
    logged offset, so one flight stays one run however many times it is paused,
    stopped and continued.

    SECURITY: get_simulation() reads via the service_role client, which bypasses
    RLS - the ownership check below (sim["user_id"] == req.user_id) is therefore
    the ONLY thing standing between this endpoint and any user resuming anyone
    else's simulation by guessing an id. This check is not optional."""
    global twin, CURRENT_ENGINE_MODEL
    sim = await db_call(db.get_simulation, req.simulation_id)
    if sim is None:
        return {"status": "error", "message": "Simulation not found"}
    if sim.get("user_id") != req.user_id:
        return {"status": "error", "message": "Not authorized to resume this simulation"}
    if not sim.get("final_telemetry"):
        return {"status": "error", "message": "This simulation has no saved final state to resume from"}
    run_version = sim.get("model_version") or "v2"
    if run_version != PHYSICS_VERSION:
        # One run must stay on one timescale: a v2 run's RUL is simulated hours, a
        # v3 run's is engine hours, and continuing across them would mix the two.
        return {"status": "error",
                "message": f"This run was recorded on physics {run_version}, but the backend is running "
                           f"{PHYSICS_VERSION}. Resume it with AERO_PHYSICS_VERSION={run_version}."}

    await stop_loop()
    state["session_active"] = True   # otherwise a later pause->resume via /start
                                      # would see session_active still False, wrongly
                                      # treat itself as "fresh", and discard the
                                      # state just restored here
    try:
        restored = new_twin(sim["engine_model"])
        restored.restore_state(sim["final_telemetry"])
        probe = restored.step()                      # a snapshot that cannot fly is refused
        problem = safety.telemetry_problem(probe)
        if problem:
            raise ValueError(problem)
        restored = new_twin(sim["engine_model"])
        restored.restore_state(sim["final_telemetry"])
    except Exception as e:
        state["session_active"] = False
        return {"status": "error", "message": f"The saved state of this run could not be restored safely ({e})."}
    CURRENT_ENGINE_MODEL = sim["engine_model"]
    twin = restored
    state["last_telemetry"] = None
    state["last_ai_result"] = None
    state["ai_hold"] = None
    state["ai_step_counter"] = 0
    state["ai_warmed_up"] = False
    state["ai_resync"] = False
    state["recoveries"] = 0
    state["recovery_times"] = []
    state["last_error"] = None
    state["last_good_telemetry"] = None
    state["sim_health"] = "idle"
    ai_breaker.reset()
    reset_residuals()
    state["db_log_counter"] = 0
    # Continue the SAME run: keep its id and pick the telemetry time axis back up
    # where the previous session left off, so the series is continuous rather than
    # folding back over itself at zero.
    state["simulation_id"] = req.simulation_id
    state["pending_simulation"], state["pending_logs"] = None, []
    state["sim_time_offset"] = (await db_call(db.get_max_time_offset, req.simulation_id)) or 0.0
    # Clear ended_at/outcome - the run is flying again and should not read as
    # finished. end_simulation() sets them again at the next genuine stop.
    await db_call(db.reopen_simulation, req.simulation_id)
    try:
        await ai_client.post(f"{AI_SERVICE_URL}/select_engine", json={"engine_model": CURRENT_ENGINE_MODEL})
    except Exception:
        pass

    # BUG FIX: this endpoint restored the twin's state and opened a fresh Supabase
    # record, but never actually started the loop - confirmed by a real test where
    # /state kept returning telemetry: null after calling /resume. Without these two
    # lines, "Continue Simulation" would report success but nothing would ever run.
    await start_loop()

    return {
        "status": "ok",
        "engine_model": CURRENT_ENGINE_MODEL,
        "simulation_id": state["simulation_id"],
        "restored_altitude": twin.altitude,
        "restored_throttle": twin.throttle,
        "restored_airspeed": twin.airspeed,
        "restored_aoa": twin.aoa,
        "restored_isa_dev_c": twin.isa_dev_c,
        "restored_wear": twin.wear,
    }


ENV_PARAMS = ("qnh_offset_pa", "humidity_frac", "fuel_octane_mon", "fuel_ethanol_frac",
              "cooling_airflow_factor", "electrical_load_a", "target_lambda")


def apply_env_inputs(values: dict) -> list:
    """Set the v4 twin's environment inputs from `values` (None = leave alone), each
    bounded to safety.PARAM_LIMITS. Returns the names refused as non-finite; a v2/v3
    twin has no such inputs and ignores them."""
    env = getattr(twin, "env", None)
    if env is None:
        return []
    rejected = []
    for name in ENV_PARAMS:
        raw = values.get(name)
        if raw is None:
            continue
        v = safety.clamp_param(name, raw)
        if v is None:
            rejected.append(name)
            continue
        env[name] = v
    if isinstance(values.get("oil_thermostat_open"), bool):
        env["oil_thermostat_open"] = values["oil_thermostat_open"]
    isa = values.get("isa_dev_c")
    if isa is not None:
        v = safety.clamp_param("isa_dev_c", isa)
        if v is None:
            rejected.append("isa_dev_c")
        else:
            twin.isa_dev_c = v
    return rejected


@app.post("/params")
async def update_params(p: ParamUpdate):
    """Every value is bounded to safety.PARAM_LIMITS before it reaches the twin, and a
    non-finite value is refused (and reported) instead of poisoning the physics."""
    if data_source() == "can" and p.source != "can":
        return {"status": "ignored", "reason": "the aircraft is on the CAN bus and owns the set-points",
                "altitude": twin.altitude, "throttle": twin.throttle,
                "airspeed": twin.airspeed, "aoa": twin.aoa, "rejected": []}
    rejected = []
    for name in ("altitude", "throttle", "airspeed", "aoa", "isa_dev_c"):
        raw = getattr(p, name)
        if raw is None:
            continue
        v = safety.clamp_param(name, raw)
        if v is None:
            rejected.append(name)
            continue
        setattr(twin, name, v)
    rejected += apply_env_inputs({n: getattr(p, n) for n in (*ENV_PARAMS, "oil_thermostat_open")})
    return {"status": "ok", "altitude": twin.altitude, "throttle": twin.throttle,
            "airspeed": twin.airspeed, "aoa": twin.aoa, "rejected": rejected}


ZERO_SAMPLES = 60   # seconds of steady, known-healthy running averaged into the offsets


@app.post("/residuals/zero")
async def zero_residuals():
    """Calibrate sender offsets on a known-healthy ground run (residual.py begin_zeroing).
    Uses the twin's accumulated wear as the known engine condition; the offsets apply to
    this engine from the next sample and are kept until the backend restarts."""
    if not state["running"]:
        return {"status": "error", "detail": "start a flight and hold a steady operating point first"}
    if PHYSICS_VERSION == "v4":
        return {"status": "error", "detail": "v4 residuals are taken against the on-board twin and need no zeroing"}
    if PHYSICS_VERSION != "v3":
        return {"status": "error", "detail": "physics residuals need physics v3"}
    residual_monitor.begin_zeroing(known_wear=float(twin.wear), samples=ZERO_SAMPLES)
    state["residuals"] = {"enabled": True, "zeroing": True, "samples_left": ZERO_SAMPLES,
                          "deviations": [], "saturated": [], "channels": {}}
    return {"status": "ok", "samples": ZERO_SAMPLES, "known_wear": round(float(twin.wear), 4)}


@app.post("/measured")
async def update_measured(m: MeasuredUpdate):
    """Measured sensor values from the aircraft. A non-finite or out-of-range value is
    refused and reported, never overlaid - a corrupted reading must not reach the AI."""
    accepted, rejected = {}, []
    for ch in (MEASURED_KEYS_V4 if PHYSICS_VERSION == "v4" else MEASURED_KEYS):
        raw = getattr(m, ch)
        if raw is None:
            continue
        lo, hi = MEASURED_LIMITS[ch]
        if not isinstance(raw, (int, float)) or not (lo <= float(raw) <= hi):
            rejected.append(ch)
            continue
        accepted[ch] = float(raw)
    if accepted:
        state["measured"] = {**state["measured"], **accepted}
        state["measured_at"] = time.monotonic()
        state["measured_posts"] += 1
    return {"status": "ok", "accepted": sorted(accepted), "rejected": rejected, "data_source": data_source()}


# NOTE: manual fault triggering was removed. Faults are now fully auto-derived from
# operating conditions (throttle/altitude/airspeed -> RPM/power -> stress -> fault) in
# physics.py:_auto_fault_step, evaluated every simulation step.


@app.post("/summarize/{sim_id}")
async def summarize(sim_id: int, req: SummarizeRequest):
    """Regenerate the post-flight summary for a run the caller owns."""
    row = await db_call(db.get_simulation, sim_id)
    if not row:
        return {"status": "error", "detail": "simulation not found"}
    if row.get("user_id") != req.user_id:
        # Same shape as a missing row on purpose - do not confirm existence of
        # another user's run.
        return {"status": "error", "detail": "simulation not found"}
    result = await db_call(summary.generate_summary, sim_id, timeout=90.0)
    if not result:
        return {"status": "error", "detail": "summary unavailable"}
    return {"status": "ok", "groq_result": result}


@app.get("/state")
async def get_state():
    return {
        "running": state["running"],
        "engine_model": CURRENT_ENGINE_MODEL,
        "physics_version": PHYSICS_VERSION,
        "ai_model_version": state["ai_model_version"],
        "ai_valid": CURRENT_ENGINE_MODEL in AI_VALID_ENGINES,
        "data_source": data_source(),
        "params": {"altitude": twin.altitude, "throttle": twin.throttle,
                    "airspeed": twin.airspeed, "aoa": twin.aoa,
                    "isa_dev_c": twin.isa_dev_c},
        "telemetry": safety.json_safe(state["last_telemetry"]),
        "ai": state["last_ai_result"],
        "residuals": safety.json_safe(state["residuals"]),
        "advisory": _safe_advisory(),
        "sim_status": sim_status(),
        **(v4_state() if PHYSICS_VERSION == "v4" else {}),
    }


def v4_state() -> dict:
    return {"time_model": timescale_v4.describe(), "engine": engine_summary(),
            "scenario_pending": state["scenario"],
            "engine_hours": round(twin.engine_hours, 4)}


class ScenarioSelect(BaseModel):
    name: str


class InjectRequest(BaseModel):
    kind: str                       # "fault" or "sensor"
    name: Optional[str] = None      # fault name (kind=fault)
    channel: Optional[str] = None   # sensor channel (kind=sensor)
    type: Optional[str] = None      # sensor fault type (kind=sensor)
    severity: Optional[float] = None


def engine_inputs() -> Optional[dict]:
    """The engine's atmosphere, fuel and installation inputs, for the cockpit before a flight."""
    env = getattr(twin, "env", None)
    return None if env is None else {**env, "isa_dev_c": twin.isa_dev_c}


def applicable_faults_now() -> list:
    spec = getattr(twin, "spec", None)
    return [] if spec is None else twin_v4.applicable_faults(spec.turbocharged, spec.intercooled)


@app.get("/scenarios")
async def list_scenarios():
    """Demo presets this engine can fly (v4)."""
    if PHYSICS_VERSION != "v4":
        return {"status": "error", "detail": "scenarios need physics v4"}
    return {"status": "ok", "engine_model": CURRENT_ENGINE_MODEL, "selected": state["scenario"],
            "scenarios": scenarios_v4.listing(CURRENT_ENGINE_MODEL), "inputs": engine_inputs(),
            "applicable_faults": applicable_faults_now(), **timescale_v4.describe()}


@app.post("/scenario")
async def select_scenario(sel: ScenarioSelect):
    """Choose the preset the next fresh flight starts from. A flight in progress is not
    changed: stop it first (a preset is a different engine, not a different sky)."""
    if PHYSICS_VERSION != "v4":
        return {"status": "error", "detail": "scenarios need physics v4"}
    if sel.name not in scenarios_v4.available(CURRENT_ENGINE_MODEL):
        return {"status": "error", "detail": f"{sel.name!r} is not available for {CURRENT_ENGINE_MODEL}"}
    if state["session_active"]:
        return {"status": "error", "detail": "stop the current flight before choosing another engine"}
    global twin
    state["scenario"] = sel.name
    twin = new_twin(CURRENT_ENGINE_MODEL, scenarios_v4.make_engine(CURRENT_ENGINE_MODEL, sel.name))
    return {"status": "ok", "scenario": sel.name, "start_engine_hours": twin.start_engine_hours,
            "inputs": engine_inputs()}


@app.post("/inject")
async def inject(req: InjectRequest):
    """Inject a component or sensor fault into the engine being flown (v4, for Q&A)."""
    if PHYSICS_VERSION != "v4":
        return {"status": "error", "detail": "injection needs physics v4"}
    try:
        if req.kind == "fault":
            entry = twin.inject_fault(req.name or "", req.severity if req.severity is not None else 0.35)
        elif req.kind == "sensor":
            entry = twin.inject_sensor(req.channel or "", req.type or "", req.severity if req.severity is not None else 0.8)
        else:
            return {"status": "error", "detail": "kind must be 'fault' or 'sensor'"}
    except ValueError as e:
        return {"status": "error", "detail": str(e)}
    return {"status": "ok", "injected": entry}


def without_removed(ai: dict, fault: Optional[str] = None, channel: Optional[str] = None) -> dict:
    """An AI result with one removed fault or sensor fault taken out; the detection
    call goes with it when nothing else is left (the settle hold's rule)."""
    modes = {k: ({**v, "present": False, "severity": 0.0} if k == fault else v)
             for k, v in (ai.get("fault_modes") or {}).items()}
    sensors = {c: ({**v, "condition": "none"} if c == channel else v)
               for c, v in (ai.get("sensors") or {}).items()}
    present = [k for k, v in modes.items() if isinstance(v, dict) and v.get("present")]
    faulty = [c for c, v in sensors.items() if isinstance(v, dict) and v.get("condition") != "none"]
    held = {**ai, "fault_modes": modes, "faults_present": present, "sensors": sensors,
            "faulty_sensors": faulty, "held_after_repair": True}
    if not present and not faulty:
        held.update(fault_detected=False, detection_confidence=0.0,
                    detection_confidence_raw=ai.get("detection_confidence"))
    return held


@app.post("/inject/clear")
async def clear_injection(req: InjectRequest):
    """Take a component fault (by name) or a sensor fault (by channel) back out (v4)."""
    if PHYSICS_VERSION != "v4":
        return {"status": "error", "detail": "injection needs physics v4"}
    if req.kind == "fault":
        removed = twin.clear_fault(req.name or "")
    elif req.kind == "sensor":
        removed = twin.clear_sensor(req.channel or "")
    else:
        return {"status": "error", "detail": "kind must be 'fault' or 'sensor'"}
    if not removed:
        return {"status": "error", "detail": "nothing to remove"}
    # The AI's window still holds the faulty samples: start it clean. While it refills
    # (in real time) the cockpit keeps the last result, minus what was just removed.
    last = state["last_ai_result"]
    if isinstance(last, dict) and last.get("status") == "ok":
        state["ai_hold"] = state["last_ai_result"] = without_removed(
            last, fault=req.name if req.kind == "fault" else None,
            channel=req.channel if req.kind == "sensor" else None)
    state["ai_resync"] = True
    return {"status": "ok", "removed": removed}


def _safe_advisory():
    try:
        return advisory.build_advisory(state["last_ai_result"], state["last_telemetry"], state["residuals"])
    except Exception as e:
        return {"severity": "nominal", "headline": "Advisory unavailable", "insufficient_data": True,
                "items": [], "error": str(e)[:200]}


def spawn_background(fn, *args):
    """Fire-and-forget work in a thread whose failure is logged, never raised."""
    async def run():
        try:
            await asyncio.to_thread(fn, *args)
        except Exception as e:
            print(f"[background] {getattr(fn, '__name__', fn)} failed: {e}")
    asyncio.create_task(run())


@app.get("/health")
async def health():
    """Liveness of the simulation itself - what a supervisor or the UI should poll."""
    task = state["task"]
    return {
        "status": "ok" if state["sim_health"] != "halted" else "degraded",
        "running": state["running"],
        "loop_alive": bool(task and not task.done()),
        "physics_version": PHYSICS_VERSION,
        "engine_model": CURRENT_ENGINE_MODEL,
        "data_source": data_source(),
        "measured_age_s": None if measured_age_s() is None else round(measured_age_s(), 2),
        "measured_posts": state["measured_posts"],
        "steps": state["steps"],
        "sim_status": sim_status(),
        "ai_samples_dropped": state["ai_samples_dropped"],
        "db_pending": state["db_pending"],
        "db_skipped": state["db_skipped"],
        "clients": len(clients),
        **({"time_model": timescale_v4.describe()} if PHYSICS_VERSION == "v4" else {}),
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Close an open flight record cleanly instead of leaving it 'in progress' forever."""
    was_session = state["session_active"]
    await stop_loop()
    if was_session and state["simulation_id"] is not None:
        ai = state["last_ai_result"] or {}
        await db_call(db.end_simulation, state["simulation_id"], "server_shutdown",
                      ai.get("health_percent"), final_rul_hours(ai),
                      safety.json_safe(state["last_telemetry"]), timeout=4.0)
    await ai_client.aclose()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE_MAX)
    sender = asyncio.create_task(client_sender(websocket, queue))
    clients[websocket] = (queue, sender)
    try:
        if state["last_telemetry"]:
            first = {**client_frame(state["last_telemetry"]), "ai": state["last_ai_result"],
                     "residuals": state["residuals"], "sim_status": sim_status()}
            queue.put_nowait(json.dumps(safety.json_safe(first), allow_nan=False))
        while True:
            await websocket.receive_text()  # keep connection alive / detect disconnect
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] connection closed on error: {e}")
    finally:
        clients.pop(websocket, None)
        sender.cancel()
