"""
UAV Digital Twin - real-time FastAPI backend.
Runs the validated Python physics twin continuously in the background at real-time
pace. Parameters (altitude/throttle/airspeed/aoa) can be changed at any time via
POST /params and take effect on the very next simulation step - the loop itself never
stops until POST /stop is called. Live telemetry streams over WebSocket at ~20Hz.
"""
import asyncio
import importlib
import json
import os
import time
import traceback
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from physics import UAVEngineTwin, ENGINE_CONFIGS

# Physics version for every twin this process creates. v3 (default) pairs with the
# v3 AI service (backend/aiv3.py, backend/models_v3/); AERO_PHYSICS_VERSION=v2 runs the
# legacy physics with ai.py and backend/models/. /state reports both so a mismatch is visible.
PHYSICS_VERSION = os.environ.get("AERO_PHYSICS_VERSION", "v3").strip().lower()
if PHYSICS_VERSION not in ("v2", "v3"):
    raise ValueError(f"AERO_PHYSICS_VERSION must be v2 or v3, got {PHYSICS_VERSION!r}")

# v3 runs persist through dbv3 (real engine-hour RUL). Same function names as db,
# so nothing below changes.
db = importlib.import_module("dbv3" if PHYSICS_VERSION == "v3" else "db")
import advisory
import residual
import safety
import summary

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

def new_twin(engine_model: str) -> UAVEngineTwin:
    """The ONLY place twins are built, so the physics version cannot be missed."""
    return UAVEngineTwin(dt=0.01, engine_model=engine_model, physics_version=PHYSICS_VERSION)


CURRENT_ENGINE_MODEL = "Rotax_914_ULF"
twin = new_twin(CURRENT_ENGINE_MODEL)

# Physics residuals (backend/residual.py): measured sensors minus what physics v3
# expects. Model-free, so it keeps producing sensor-integrity and wear evidence while
# the AI is warming up or down. Fed at the AI cadence (one sample per simulated
# second) and cleared wherever the AI buffer is. Physics v2 returns enabled=False.
residual_monitor = residual.ResidualMonitor(lag="exp")


def reset_residuals():
    residual_monitor.reset()
    state["residuals"] = None


def log_telemetry(simulation_id, time_offset_s, raw, ai):
    """Thread target for telemetry_logs writes. v3 (dbv3) also persists the physics
    residuals; db.py (v2) keeps its original signature."""
    if PHYSICS_VERSION == "v3":
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
    # False while the AI service cannot be reached. Warmup fast-forward exists only to
    # fill the AI's 128 s window quickly; with no AI to fill it, fast-forwarding just
    # ran physics unthrottled (1,386 simulated seconds in ~40 s) and wrote a telemetry
    # row every ~0.3 s of wall time. Unreachable -> real-time pacing.
    "ai_reachable": True,
    "simulation_id": None,   # current Supabase simulations.id, or None if not persisted
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
    "ai_samples_dropped": 0,
    "db_pending": 0,
    "db_skipped": 0,
    "loop_lag_ms": 0.0,        # how far the loop is behind real time (EWMA)
    "steps": 0,
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


def final_rul_hours(ai: dict):
    """RUL for simulations.final_rul_hours. ai.py (v2) reports simulated-timescale
    hours in rul_hours_internal; aiv3.py reports real engine hours in rul_hours."""
    if ai.get("model_version") == "v3":
        return ai.get("rul_hours")
    return ai.get("rul_hours_internal")


async def refresh_ai_version():
    """Records which model version the AI service is serving and warns on mismatch."""
    global AI_VALID_ENGINES
    try:
        resp = await ai_client.get(f"{AI_SERVICE_URL}/health")
        health = resp.json()
        state["ai_model_version"] = health.get("model_version", "v2")
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
        payload = {c: telemetry[c] for c in AI_FEATURE_COLS}
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
AI_WINDOW_S = 128.0


def hold_alerts_outside_envelope(sample: dict, result: dict) -> dict:
    """Return result with fault alerts cleared while the AI window still contains
    below-envelope (ground roll) samples. Health and RUL pass through unchanged."""
    if PHYSICS_VERSION != "v3":
        return result
    t = sample.get("time")
    if not isinstance(t, (int, float)):
        return result
    last = state["ai_last_sample_time"]
    if last is not None and t < last:           # clock restarted: a new flight
        state["ai_settle_until"] = 0.0
    state["ai_last_sample_time"] = t
    airspeed = sample.get("airspeed")
    if isinstance(airspeed, (int, float)) and airspeed < AI_ENVELOPE_MIN_AIRSPEED_MS:
        state["ai_settle_until"] = t + AI_WINDOW_S
    if result.get("status") != "ok" or t >= state["ai_settle_until"]:
        return result
    held = {**result, "settling": True, "fault_detected": False, "faulty_channels": []}
    modes = result.get("failure_modes")
    if isinstance(modes, dict):
        held["failure_modes"] = {k: ({**v, "present": False} if isinstance(v, dict) else v)
                                 for k, v in modes.items()}
    return held


async def ai_worker(queue: asyncio.Queue):
    """The ONLY sender of /step. One worker, one queue: samples reach the AI strictly in
    simulated-time order however slow it is, and the loop never awaits the network.

    Circuit breaker: after 3 consecutive failures the AI is skipped for 5 s, then probed.
    Any gap in the AI's 128-sample window (dropped samples, an outage) makes the next
    sample start with /reset, so the model never sees a window with a hole in it.
    """
    while True:
        out = await queue.get()
        try:
            if not ai_breaker.allow():
                state["ai_resync"] = True          # this sample is lost to the AI
                state["ai_reachable"] = False
                state["last_ai_result"] = {"status": "ai_service_unavailable",
                                           "error": ai_breaker.last_error or "AI service unreachable",
                                           "retrying": True}
                continue
            if state["ai_resync"]:
                state["ai_resync"] = False
                state["ai_warmed_up"] = False
                await ai_reset_buffer()
            result = await call_ai_service(out)
            status = result.get("status")
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
            state["last_ai_result"] = hold_alerts_outside_envelope(out, result)
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


def sim_status() -> dict:
    """Compact health block attached to every broadcast and to /health."""
    return {
        "health": state["sim_health"],
        "recoveries": state["recoveries"],
        "last_error": state["last_error"],
        "ai": ai_status_label(),
        "ai_breaker": ai_breaker.state,
        "loop_lag_ms": round(state["loop_lag_ms"], 1),
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
            state["last_telemetry"] = out
            state["last_good_telemetry"] = out
            state["steps"] += 1
            step_count += 1

            # ---- once per simulated second: residuals, AI, database ----
            state["ai_step_counter"] += 1
            if state["ai_step_counter"] >= AI_STEPS_PER_CALL:
                state["ai_step_counter"] = 0
                state["sim_time_offset"] += 1.0
                try:
                    state["residuals"] = residual_monitor.update(out, dt=1.0)
                except Exception as e:
                    state["residuals"] = {"enabled": False, "reason": f"residual monitor error: {e}"}

                try:
                    queue.put_nowait(out)
                except asyncio.QueueFull:
                    # The AI is stalled. Drop the backlog rather than block physics;
                    # the worker resets the AI's window before its next sample.
                    dropped = 0
                    while not queue.empty():
                        queue.get_nowait(); queue.task_done(); dropped += 1
                    state["ai_samples_dropped"] += dropped + 1
                    state["ai_resync"] = True
                    queue.put_nowait(out)

                state["db_log_counter"] += 1
                if state["db_log_counter"] >= DB_LOG_INTERVAL_S:
                    state["db_log_counter"] = 0
                    spawn_db_write(log_telemetry, state["simulation_id"], state["sim_time_offset"],
                                   out, state["last_ai_result"])

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
                    payload = dict(out)
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
            fast_forward = (not state["ai_warmed_up"] and state["ai_reachable"]
                            and ai_breaker.state == "closed" and ai_breaker.failures == 0
                            and ai_status_label() != "unsupported")
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
    if state["sim_health"] == "halted":
        state["recovery_times"] = []
        state["sim_health"] = "running"
    state["task"] = asyncio.create_task(simulation_loop())


class StartRequest(BaseModel):
    user_id: Optional[str] = None   # Supabase auth.users.id, if the frontend user is logged in


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
        state["session_active"] = True
        twin = new_twin(CURRENT_ENGINE_MODEL)
        # A fresh session takes off from the ground, matching the frontend's own
        # auto-climb takeoff sequence (Simulator.tsx starts its visual at altitude 0
        # and climbs automatically) - the twin's own __init__ default (2000m) would
        # otherwise briefly mismatch that visual until the first /params call caught up.
        twin.altitude = 0.0
        state["sim_time_offset"] = 0.0
        state["last_telemetry"] = None
        state["last_ai_result"] = None
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
        state["simulation_id"] = await db_call(db.start_simulation, req.user_id, CURRENT_ENGINE_MODEL)
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


@app.post("/stop")
async def stop_sim(req: StopRequest = StopRequest()):
    await stop_loop()
    if req.final:
        state["session_active"] = False
    if req.final and state["simulation_id"] is not None:
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
        await db_call(db.end_simulation, state["simulation_id"], "stopped",
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
        # Returned so the caller can poll for the summary that the task above is
        # generating. /stop previously returned no id at all, which left the
        # frontend with no way to find the row it had just closed.
        return {"status": "stopped", "final": req.final, "simulation_id": _sim_id}
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
    state["last_telemetry"] = None
    state["last_ai_result"] = None
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
    CURRENT_ENGINE_MODEL = sel.engine_model
    twin = new_twin(CURRENT_ENGINE_MODEL)
    state["last_telemetry"] = None
    state["last_ai_result"] = None
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


@app.post("/params")
async def update_params(p: ParamUpdate):
    """Every value is bounded to safety.PARAM_LIMITS before it reaches the twin, and a
    non-finite value is refused (and reported) instead of poisoning the physics."""
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
    return {"status": "ok", "altitude": twin.altitude, "throttle": twin.throttle,
            "airspeed": twin.airspeed, "aoa": twin.aoa, "rejected": rejected}


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
        "params": {"altitude": twin.altitude, "throttle": twin.throttle,
                    "airspeed": twin.airspeed, "aoa": twin.aoa,
                    "isa_dev_c": twin.isa_dev_c},
        "telemetry": safety.json_safe(state["last_telemetry"]),
        "ai": state["last_ai_result"],
        "residuals": safety.json_safe(state["residuals"]),
        "advisory": _safe_advisory(),
        "sim_status": sim_status(),
    }


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
        "steps": state["steps"],
        "sim_status": sim_status(),
        "ai_samples_dropped": state["ai_samples_dropped"],
        "db_pending": state["db_pending"],
        "db_skipped": state["db_skipped"],
        "clients": len(clients),
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
            first = {**state["last_telemetry"], "ai": state["last_ai_result"],
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
