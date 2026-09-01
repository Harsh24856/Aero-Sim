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

# Every trained AI model was fit ONLY on Rotax_914_ULF physics data - the other 3
# engines are real, correctly-simulated physics options, but predictions on them
# would be meaningless (different torque/power regime the model never saw).
AI_VALID_ENGINE = "Rotax_914_ULF"


state = {
    "running": False,
    "task": None,
    "last_telemetry": None,
    "last_ai_result": None,
    "ai_step_counter": 0,
    "ai_warmed_up": False,   # True once the AI has returned a real (non-warmup) result
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
            if state["ai_warmed_up"]:
                asyncio.create_task(call_ai_service(out))   # fire-and-forget, never blocks
            else:
                result = await call_ai_service(out)   # sequential during warmup - see docstring
                if result.get("status") == "ok":
                    state["ai_warmed_up"] = True

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


@app.post("/start")
async def start_sim():
    if state["running"]:
        return {"status": "already_running"}
    state["running"] = True
    state["task"] = asyncio.create_task(simulation_loop())
    return {"status": "started"}


@app.post("/stop")
async def stop_sim():
    state["running"] = False
    if state["task"]:
        state["task"] = None
    return {"status": "stopped"}


@app.post("/reset")
async def reset_sim():
    global twin
    was_running = state["running"]
    state["running"] = False
    await asyncio.sleep(0.05)
    twin = UAVEngineTwin(dt=0.01)
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
        "ai_valid_engine": AI_VALID_ENGINE,
    }


class EngineSelect(BaseModel):
    engine_model: str


@app.post("/select_engine")
async def select_engine(sel: EngineSelect):
    """Switches the physics twin to a different engine model. Always does a full
    reset (not a live swap) - a different engine has a genuinely different torque
    curve/power output, so continuing mid-flight with old telemetry would not make
    physical sense. Also resets the AI service's rolling buffer, since straddling
    a discontinuous engine-switch across its 128-step window would corrupt it -
    and predictions are only meaningful for AI_VALID_ENGINE regardless."""
    global twin, CURRENT_ENGINE_MODEL
    if sel.engine_model not in ENGINE_CONFIGS:
        return {"status": "error", "message": f"Unknown engine_model. Options: {list(ENGINE_CONFIGS)}"}

    was_running = state["running"]
    state["running"] = False
    await asyncio.sleep(0.05)
    CURRENT_ENGINE_MODEL = sel.engine_model
    twin = UAVEngineTwin(dt=0.01, engine_model=CURRENT_ENGINE_MODEL)
    state["last_telemetry"] = None
    state["last_ai_result"] = None
    state["ai_step_counter"] = 0
    state["ai_warmed_up"] = False
    try:
        await ai_client.post(f"{AI_SERVICE_URL}/reset")
    except Exception:
        pass
    if was_running:
        state["running"] = True
        state["task"] = asyncio.create_task(simulation_loop())

    return {
        "status": "ok",
        "engine_model": CURRENT_ENGINE_MODEL,
        "ai_valid": CURRENT_ENGINE_MODEL == AI_VALID_ENGINE,
        "max_power_kw": twin.MAX_POWER_KW,
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
        "ai_valid": CURRENT_ENGINE_MODEL == AI_VALID_ENGINE,
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
