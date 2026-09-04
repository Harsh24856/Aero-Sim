"""
UAV Digital Twin - real-time FastAPI backend.
Runs the validated Python physics twin continuously in the background at real-time
pace. Parameters (altitude/throttle/airspeed/aoa) can be changed at any time via
POST /params and take effect on the very next simulation step - the loop itself never
stops until POST /stop is called. Live telemetry streams over WebSocket at ~20Hz.
"""
import asyncio
import json
import time
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from physics import UAVEngineTwin, ENGINE_CONFIGS
import db

# ai.py runs as its OWN process under a different Python environment (see ai.py's
# module docstring - the models segfault under this backend's TF version). Calls are
# async/non-blocking so a slow or down AI service never freezes telemetry to other
# clients, and every call is wrapped in try/except for the same reason.
AI_SERVICE_URL = "http://127.0.0.1:8100"
ai_client = httpx.AsyncClient(timeout=2.0)

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

CURRENT_ENGINE_MODEL = "Rotax_914_ULF"
twin = UAVEngineTwin(dt=0.01, engine_model=CURRENT_ENGINE_MODEL)
STEPS_PER_BROADCAST = 5  # 0.01s * 5 = 20Hz telemetry rate

# Rotax_912_ULS has real physics simulation but NO trained models yet (912' s split
# checkpoints do not exist). 914/915/916 each have their OWN genuinely-trained models
# now - AI predictions ARE meaningful for all three, each using its own weights,
# scaler, and max-power normalization (see ai.py's ENGINE_MODEL_SETS).
AI_VALID_ENGINES = {"Rotax_914_ULF", "Rotax_915_iS", "Rotax_916_iS"}


state = {
    "running": False,
    "task": None,
    "last_telemetry": None,
    "last_ai_result": None,
    "ai_step_counter": 0,
    "ai_warmed_up": False,   # True once the AI has returned a real (non-warmup) result
    "simulation_id": None,   # current Supabase simulations.id, or None if not persisted
    "sim_time_offset": 0.0,  # seconds since this simulation started, for telemetry_logs
}

# Must match ai.py's FEATURE_COLS exactly - the subset of the physics telemetry dict
# that the AI models were trained on.
AI_FEATURE_COLS = [
    "altitude", "throttle", "airspeed", "aoa", "air_density",
    "torque_available_nm", "engine_rpm", "prop_rpm", "prop_torque", "power_kw", "fuel_flow",
    "thrust", "lift", "drag", "thrust_margin", "lift_weight_margin",
    "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm_fault",
]


async def call_ai_service(telemetry: dict):
    """Sends one timestep to ai.py, updates state['last_ai_result']. Never raises -
    if the AI service is down or slow, the physics/telemetry loop keeps running
    unaffected; the frontend just stops receiving fresh AI fields until it recovers.
    Returns the parsed result so the caller can detect the warmup-to-ready transition."""
    payload = {c: telemetry[c] for c in AI_FEATURE_COLS}
    payload["time"] = telemetry["time"]
    try:
        resp = await ai_client.post(f"{AI_SERVICE_URL}/step", json=payload)
        result = resp.json()
        state["last_ai_result"] = result
        return result
    except Exception as e:
        result = {"status": "ai_service_unavailable", "error": str(e)}
        state["last_ai_result"] = result
        return result

clients: set = set()


class ParamUpdate(BaseModel):
    altitude: Optional[float] = None
    throttle: Optional[float] = None
    airspeed: Optional[float] = None
    aoa: Optional[float] = None


async def broadcast(msg: dict):
    if not clients:
        return
    dead = []
    payload = json.dumps(msg)
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def simulation_loop():
    """Runs forever until state['running'] is set False by /stop.

    WARMUP FAST-FORWARD: the AI needs a genuine 128 SIMULATED seconds of history before
    it can predict at all (the model's window size) - but physics itself is cheap to
    compute, so there is no reason to wait 128 REAL seconds in real-time pace just to
    fill that buffer. While state['ai_warmed_up'] is False, this loop skips the
    real-time sleep AND awaits each AI call sequentially (not fire-and-forget) - the
    sequential await is required for correctness, not just style: without it, physics
    steps racing ahead of the AI could deliver timesteps out of order into ai.py's
    rolling buffer, corrupting the very temporal sequence the model depends on. Warmup
    wall-clock time becomes bounded by the AI's own inference speed (~15-20s for 128
    sequential calls) instead of artificially waiting out 128 real seconds. Once warmed
    up, this switches to normal real-time pacing with fire-and-forget AI calls, exactly
    as before - full model accuracy is unaffected either way, since the AI still
    receives the complete, correctly-ordered 128 simulated seconds it was trained on."""
    step_count = 0
    next_tick = time.monotonic()
    while state["running"]:
        out = twin.step()
        state["last_telemetry"] = out
        step_count += 1

        state["ai_step_counter"] += 1
        if state["ai_step_counter"] >= AI_STEPS_PER_CALL:
            state["ai_step_counter"] = 0
            state["sim_time_offset"] += 1.0   # one simulated second has elapsed
            if state["ai_warmed_up"]:
                asyncio.create_task(call_ai_service(out))   # fire-and-forget, never blocks
                # Logs the PREVIOUS ai result (last_ai_result), not this call's -
                # same "eventually consistent" tradeoff broadcast() already makes,
                # since the fire-and-forget call has not resolved yet at this point.
                # asyncio.to_thread is REQUIRED here, not optional: supabase-py's
                # .execute() is a synchronous/blocking HTTP call - calling it directly
                # in this async loop would stall the whole event loop (100Hz physics +
                # WebSocket broadcasts to every client) for the duration of each
                # Supabase round-trip. Running it in a thread keeps persistence fully
                # fire-and-forget, matching call_ai_service's own non-blocking pattern.
                asyncio.create_task(asyncio.to_thread(
                    db.log_telemetry, state["simulation_id"], state["sim_time_offset"], out, state["last_ai_result"]))
            else:
                result = await call_ai_service(out)   # sequential during warmup - see docstring
                if result.get("status") == "ok":
                    state["ai_warmed_up"] = True
                asyncio.create_task(asyncio.to_thread(
                    db.log_telemetry, state["simulation_id"], state["sim_time_offset"], out, result))

        if step_count % STEPS_PER_BROADCAST == 0:
            payload = dict(out)
            payload["ai"] = state["last_ai_result"]
            await broadcast(payload)

        if state["ai_warmed_up"]:
            next_tick += twin.dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                next_tick = time.monotonic()  # fell behind, resync rather than spiral
        else:
            next_tick = time.monotonic()  # keep the real-time clock ready for the moment warmup ends
            await asyncio.sleep(0)  # yield control so broadcasts/other requests are not starved


class StartRequest(BaseModel):
    user_id: Optional[str] = None   # Supabase auth.users.id, if the frontend user is logged in


@app.post("/start")
async def start_sim(req: StartRequest = StartRequest()):
    if state["running"]:
        return {"status": "already_running"}
    state["running"] = True
    # IMPORTANT: /start is also called to RESUME from a pause (see the frontend's
    # onTogglePause - pausing genuinely calls /stop, resuming genuinely calls /start
    # again, since the backend loop must actually halt while paused). Without this
    # check, every pause/resume cycle would fragment one continuous flight into
    # multiple separate Supabase simulation rows. Only create a new row (and reset
    # the time offset) if there genuinely isn't one already in progress.
    if state["simulation_id"] is None:
        state["sim_time_offset"] = 0.0
        state["simulation_id"] = db.start_simulation(req.user_id, CURRENT_ENGINE_MODEL)
    state["task"] = asyncio.create_task(simulation_loop())
    return {"status": "started", "simulation_id": state["simulation_id"]}


@app.post("/stop")
async def stop_sim():
    state["running"] = False
    if state["task"]:
        state["task"] = None
    if state["simulation_id"] is not None:
        ai = state["last_ai_result"] or {}
        # state["last_telemetry"] is the full raw physics dict as it stood at the
        # exact moment of stopping - already proven JSON-serializable, since this
        # same dict passes through json.dumps() in every WebSocket broadcast.
        await asyncio.to_thread(
            db.end_simulation, state["simulation_id"], "stopped",
            ai.get("health_percent"), ai.get("rul_hours_internal"), state["last_telemetry"])
        state["simulation_id"] = None
    return {"status": "stopped"}


@app.post("/reset")
async def reset_sim():
    global twin
    was_running = state["running"]
    state["running"] = False
    await asyncio.sleep(0.05)
    twin = UAVEngineTwin(dt=0.01)
    # A reset genuinely ends whatever flight was in progress - close out its
    # Supabase record properly rather than silently abandoning it. This endpoint
    # relaunches the loop directly (not via /start), so it does not have a user_id
    # to open a fresh row - the NEXT /start call will create one normally.
    if state["simulation_id"] is not None:
        ai = state["last_ai_result"] or {}
        await asyncio.to_thread(
            db.end_simulation, state["simulation_id"], "reset",
            ai.get("health_percent"), ai.get("rul_hours_internal"), state["last_telemetry"])
        state["simulation_id"] = None
    state["last_telemetry"] = None
    state["last_ai_result"] = None
    state["ai_step_counter"] = 0
    state["ai_warmed_up"] = False
    try:
        await ai_client.post(f"{AI_SERVICE_URL}/reset")   # clear the AI's rolling 128-step buffer too
    except Exception:
        pass   # AI service down is not a reason to fail the reset - degrade gracefully
    if was_running:
        state["running"] = True
        state["task"] = asyncio.create_task(simulation_loop())
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

    state["running"] = False
    await asyncio.sleep(0.05)
    # A different engine is genuinely a different flight/aircraft - close out
    # whatever simulation record was open under the OLD engine before switching.
    if state["simulation_id"] is not None:
        ai = state["last_ai_result"] or {}
        await asyncio.to_thread(
            db.end_simulation, state["simulation_id"], "engine_switched",
            ai.get("health_percent"), ai.get("rul_hours_internal"), state["last_telemetry"])
        state["simulation_id"] = None
    CURRENT_ENGINE_MODEL = sel.engine_model
    twin = UAVEngineTwin(dt=0.01, engine_model=CURRENT_ENGINE_MODEL)
    state["last_telemetry"] = None
    state["last_ai_result"] = None
    state["ai_step_counter"] = 0
    state["ai_warmed_up"] = False
    try:
        # Tell ai.py WHICH engine's models to switch to as well (not just reset) -
        # it maintains its own active_engine independently, this keeps the two
        # services in agreement. ai.py's /select_engine also resets its buffer,
        # so a separate /reset call here would be redundant.
        await ai_client.post(f"{AI_SERVICE_URL}/select_engine", json={"engine_model": CURRENT_ENGINE_MODEL})
    except Exception:
        pass
    # No auto-restart here, intentionally - see docstring.

    return {
        "status": "ok",
        "engine_model": CURRENT_ENGINE_MODEL,
        "ai_valid": CURRENT_ENGINE_MODEL in AI_VALID_ENGINES,
        "max_power_kw": twin.MAX_POWER_KW,
    }


class ContinueFromRequest(BaseModel):
    user_id: Optional[str] = None
    engine_model: str
    final_telemetry: dict   # the exact JSONB snapshot stored in simulations.final_telemetry


@app.post("/continue_from")
async def continue_from(req: ContinueFromRequest):
    """Genuinely resumes physics from a past simulation's final state - not just
    the pilot-input params /params already handles (altitude/throttle/airspeed/aoa),
    but the actual internal dynamics: engine rotational speed (omega, derived back
    from the stored engine_rpm), the three thermal-lag states (egt/cht/oil_temp -
    these chase a target over real time, so restoring only the CURRENT value and
    not the lag state would make them jump discontinuously on the next step), and
    accumulated wear (irreversible - this is what RUL is actually measuring, so
    resetting it to 0 would silently give a "continued" flight a fresh engine).

    Known limitation: the AI's 128-step rolling buffer is NOT pre-populated from
    telemetry_logs history - it starts empty and goes through the same warmup as a
    fresh /start. The PHYSICS genuinely continues from the exact point; the AI's
    context window does not (yet) carry over. This is disclosed here, not hidden."""
    global twin, CURRENT_ENGINE_MODEL
    if req.engine_model not in ENGINE_CONFIGS:
        return {"status": "error", "message": f"Unknown engine_model. Options: {list(ENGINE_CONFIGS)}"}

    state["running"] = False
    await asyncio.sleep(0.05)
    if state["simulation_id"] is not None:
        ai = state["last_ai_result"] or {}
        await asyncio.to_thread(
            db.end_simulation, state["simulation_id"], "engine_switched",
            ai.get("health_percent"), ai.get("rul_hours_internal"), state["last_telemetry"])
        state["simulation_id"] = None

    CURRENT_ENGINE_MODEL = req.engine_model
    twin = UAVEngineTwin(dt=0.01, engine_model=CURRENT_ENGINE_MODEL)

    snap = req.final_telemetry
    healthy = snap.get("healthy", {})
    twin.altitude = snap.get("altitude", twin.altitude)
    twin.throttle = snap.get("throttle", twin.throttle)
    twin.airspeed = snap.get("airspeed", twin.airspeed)
    twin.aoa = snap.get("aoa", twin.aoa)
    twin.wear = snap.get("wear", twin.wear)
    twin.t = snap.get("time", twin.t)
    # engine_rpm -> omega (rad/s): the twin's actual integrated state variable.
    engine_rpm_snapshot = healthy.get("rpm", snap.get("engine_rpm"))
    if engine_rpm_snapshot is not None:
        twin.omega = float(engine_rpm_snapshot) * 2 * np.pi / 60.0
    # Thermal lag states - restoring only the instantaneous value (not the lag
    # itself, since the twin has no separate "target" to store) means the very
    # next step continues chasing correctly, since state == target-so-far here.
    if "egt" in healthy: twin.egt_state = float(healthy["egt"])
    if "cht" in healthy: twin.cht_state = float(healthy["cht"])
    if "oil_temp" in healthy: twin.oiltemp_state = float(healthy["oil_temp"])

    state["last_telemetry"] = None
    state["last_ai_result"] = None
    state["ai_step_counter"] = 0
    state["ai_warmed_up"] = False
    state["sim_time_offset"] = 0.0
    state["simulation_id"] = db.start_simulation(req.user_id, CURRENT_ENGINE_MODEL)
    try:
        await ai_client.post(f"{AI_SERVICE_URL}/select_engine", json={"engine_model": CURRENT_ENGINE_MODEL})
    except Exception:
        pass

    state["running"] = True
    state["task"] = asyncio.create_task(simulation_loop())

    return {
        "status": "ok",
        "simulation_id": state["simulation_id"],
        "restored": {
            "altitude": twin.altitude, "throttle": twin.throttle,
            "airspeed": twin.airspeed, "aoa": twin.aoa,
            "engine_rpm": twin.omega * 60.0 / (2 * np.pi),
            "wear": twin.wear, "t": twin.t,
        },
    }


@app.post("/params")
async def update_params(p: ParamUpdate):
    if p.altitude is not None:
        twin.altitude = p.altitude
    if p.throttle is not None:
        twin.throttle = max(0.0, min(1.0, p.throttle))
    if p.airspeed is not None:
        twin.airspeed = p.airspeed
    if p.aoa is not None:
        twin.aoa = p.aoa
    return {"status": "ok", "altitude": twin.altitude, "throttle": twin.throttle,
            "airspeed": twin.airspeed, "aoa": twin.aoa}


# NOTE: manual fault triggering was removed. Faults are now fully auto-derived from
# operating conditions (throttle/altitude/airspeed -> RPM/power -> stress -> fault) in
# physics.py:_auto_fault_step, evaluated every simulation step.


@app.get("/state")
async def get_state():
    return {
        "running": state["running"],
        "engine_model": CURRENT_ENGINE_MODEL,
        "ai_valid": CURRENT_ENGINE_MODEL in AI_VALID_ENGINES,
        "params": {"altitude": twin.altitude, "throttle": twin.throttle,
                    "airspeed": twin.airspeed, "aoa": twin.aoa},
        "telemetry": state["last_telemetry"],
        "ai": state["last_ai_result"],
    }


@app.on_event("shutdown")
async def shutdown_event():
    await ai_client.aclose()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        if state["last_telemetry"]:
            await websocket.send_text(json.dumps(state["last_telemetry"]))
        while True:
            await websocket.receive_text()  # keep connection alive / detect disconnect
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
