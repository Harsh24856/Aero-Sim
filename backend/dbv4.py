"""
Supabase persistence for physics-v4 runs served by aiv4.py (migration
add_physics_v4_timescale_and_engines). Same function names as db.py / dbv3.py, so
main.py swaps modules on AERO_PHYSICS_VERSION=v4; unchanged functions are re-exported.

What v4 adds:
  - ENGINES carry the life clock between flights. A run is a flight OF an engine:
    it starts at engines.engine_hours and writes end_engine_hours back when it ends,
    so the next flight of the same engine continues from there with the same faults
    (the engine record - fault plan, seed, sensor plan - is the engines row).
  - Both clocks on every row: telemetry_logs.time_offset_s is flight seconds (never
    scaled) and telemetry_logs.engine_hours the life clock; simulations.life_scale
    records the scale the run used (the table rejects a v4 run without it).
  - v4 sensors (coolant, manifold pressure, battery), twin residuals, the v4 AI
    output (fault modes, sensor conditions, wear condition, RUL against the
    calendar) and the injected ground truth.
"""
from datetime import datetime, timezone

import db
from db import (                                   # noqa: F401  re-exported unchanged
    get_simulation,
    get_telemetry_rows,
    save_groq_result,
    get_max_time_offset,
    reopen_simulation,
)
import timescale_v4 as TS

MODEL_VERSION = "v4"
RESIDUAL_CHANNELS = ["egt", "cht", "oil_temp", "oil_pressure", "engine_rpm", "fuel_flow"]


def _r(v, n=4):
    return round(float(v), n) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


# ---------------------------------------------------------------- engines
def get_or_create_engine(user_id: str | None, record: dict) -> int | None:
    """The engines row for this record: its own (when it carries an engine_id this
    user owns), otherwise a new one named after the engine model, e.g. '914 #3'."""
    if not db._enabled or not user_id:
        return None
    try:
        eid = record.get("engine_id")
        if eid is not None:
            got = db._client.table("engines").select("id,user_id").eq("id", eid).execute().data
            if got and got[0]["user_id"] == user_id:
                return int(eid)
        model = record["engine_model"]
        n = len(db._client.table("engines").select("id").eq("user_id", user_id)
                .eq("engine_model", model).execute().data or [])
        row = db._client.table("engines").insert({
            "user_id": user_id, "name": f"{model.split('_')[1]} #{n + 1}", "engine_model": model,
            "tbo_hours": record["tbo_hours"], "engine_hours": record["engine_hours"],
            "degradation_seed": record["degradation_seed"], "fault_plan": record.get("fault_plan") or {},
            "sensor_plan": record.get("sensor_plan") or [], "scenario": record.get("scenario"),
        }).execute()
        return int(row.data[0]["id"])
    except Exception as e:
        print(f"[dbv4] get_or_create_engine failed: {e}")
        return None


def get_engine(engine_id: int) -> dict | None:
    """One engines row, as an engine record twin_v4 can fly (service role: callers
    check ownership)."""
    if not db._enabled:
        return None
    try:
        rows = db._client.table("engines").select("*").eq("id", engine_id).execute().data
        if not rows:
            return None
        r = rows[0]
        return {"engine_id": r["id"], "user_id": r["user_id"], "engine_model": r["engine_model"],
                "tbo_hours": r["tbo_hours"], "engine_hours": r["engine_hours"],
                "degradation_seed": r["degradation_seed"], "fault_plan": r["fault_plan"],
                "sensor_plan": r.get("sensor_plan") or [], "scenario": r.get("scenario"), "status": r.get("status"),
                "name": r.get("name")}
    except Exception as e:
        print(f"[dbv4] get_engine failed: {e}")
        return None


def update_engine_hours(engine_id: int | None, record: dict, engine_hours: float,
                        worn_out: bool = False) -> None:
    """Advance the hour meter and store the fault plan as it now stands (an injected
    fault is part of the engine from then on)."""
    if not db._enabled or engine_id is None:
        return
    try:
        upd = {"engine_hours": float(engine_hours), "fault_plan": record.get("fault_plan") or {},
               "updated_at": datetime.now(timezone.utc).isoformat()}
        if worn_out:
            upd["status"] = "worn_out"
        db._client.table("engines").update(upd).eq("id", engine_id).execute()
    except Exception as e:
        print(f"[dbv4] update_engine_hours failed: {e}")


# ---------------------------------------------------------------- simulations
def start_simulation(user_id: str | None, engine_model: str, record: dict | None = None,
                     model_manifest: dict | None = None, mission: str | None = None) -> int | None:
    """A v4 simulations row for one flight of one engine. Sets record['engine_id']
    so the engine record carried in telemetry names its row. `mission` is the mission
    profile flown (frontend lib/missionPresets.ts id), None for a free flight."""
    if not db._enabled or not user_id or record is None:
        return None
    try:
        engine_id = get_or_create_engine(user_id, record)
        record["engine_id"] = engine_id
        result = db._client.table("simulations").insert({
            "user_id": user_id, "engine_model": engine_model, "model_version": MODEL_VERSION,
            "tbo_hours": record["tbo_hours"], "engine_id": engine_id,
            "life_scale": TS.LIFE_SCALE, "start_engine_hours": float(record["engine_hours"]),
            "scenario": record.get("scenario"), "model_manifest": model_manifest,
            "mission": mission,
        }).execute()
        return result.data[0]["id"]
    except Exception as e:
        print(f"[dbv4] start_simulation failed: {e}")
        return None


def end_simulation(simulation_id: int | None, outcome: str, final_health_percent: float | None,
                   final_rul_hours: float | None, final_telemetry: dict | None = None):
    """Close the run and advance its engine's hour meter to where the flight ended."""
    if not db._enabled or simulation_id is None:
        return
    t = final_telemetry or {}
    try:
        db._client.table("simulations").update({
            "ended_at": datetime.now(timezone.utc).isoformat(), "outcome": outcome,
            "final_health_percent": final_health_percent, "final_rul_hours": final_rul_hours,
            "final_telemetry": final_telemetry, "end_engine_hours": _r(t.get("engine_hours")),
        }).eq("id", simulation_id).execute()
    except Exception as e:
        print(f"[dbv4] end_simulation failed: {e}")
    rec = t.get("engine_record") or {}
    if rec.get("engine_id") is not None and isinstance(t.get("engine_hours"), (int, float)):
        update_engine_hours(rec["engine_id"], rec, t["engine_hours"], worn_out=outcome == "engine_failure")


# ---------------------------------------------------------------- telemetry
def _fault_modes(ai: dict) -> dict | None:
    return {n: {"probability": _r(f.get("probability")), "present": bool(f.get("present")),
                "severity": _r(f.get("severity"))}
            for n, f in (ai.get("fault_modes") or {}).items()} or None


def _truth(raw: dict) -> dict | None:
    t = raw.get("truth")
    if not isinstance(t, dict):
        return None
    return {"faults_present": t.get("faults_present"), "fault_severity": t.get("fault_severity"),
            "sensor_faults": t.get("sensor_faults"), "wear_condition": t.get("wear_condition"),
            "rul_hours": t.get("rul_hours"), "margin_min": t.get("margin_min")}


def log_telemetry(simulation_id: int | None, time_offset_s: float, raw: dict, ai: dict | None,
                  residuals: dict | None = None) -> int | None:
    """One telemetry_logs row: what the instruments read, both clocks, the twin
    residuals, the v4 AI output and the ground truth."""
    if not db._enabled or simulation_id is None:
        return None
    try:
        row = {"simulation_id": simulation_id, "time_offset_s": time_offset_s,
               "engine_hours": _r(raw.get("engine_hours")), "margin_min": _r(raw.get("margin_min"), 5),
               "residuals": {c: _r(raw.get(f"res_{c}")) for c in RESIDUAL_CHANNELS},
               "truth": _truth(raw)}
        for k in ("altitude", "throttle", "airspeed", "aoa", "engine_rpm", "prop_rpm", "power_kw",
                  "fuel_flow", "thrust", "lift", "drag", "egt", "cht", "oil_pressure", "oil_temp",
                  "vibx", "viby", "vibz", "coolant_temp", "manifold_pressure_kpa", "battery_voltage"):
            row[k] = _r(raw.get(k))
        ai_v4 = bool(ai and ai.get("status") == "ok" and ai.get("model_version") == MODEL_VERSION)
        if ai_v4:
            row.update({
                "fault_detected": bool(ai.get("fault_detected")),
                "detection_confidence": _r(ai.get("detection_confidence"), 5),
                "health_percent": _r(ai.get("health_percent")),
                "rul_percent_remaining": _r(ai.get("rul_percent_remaining")),
                "rul_hours": _r(ai.get("rul_hours")),
                "rul_calendar_hours": _r(ai.get("rul_calendar_hours")),
                "rul_band_hours": _r(ai.get("rul_mae_hours"), 1),
                "wear_limited": ai.get("wear_limited"),
                "fault_modes": _fault_modes(ai),
            })
        result = db._client.table("telemetry_logs").insert(row).execute()
        log_id = result.data[0]["id"]
        if ai_v4 and ai.get("sensors"):
            log_channel_diagnostics(log_id, ai["sensors"])
        return log_id
    except Exception as e:
        print(f"[dbv4] log_telemetry failed: {e}")
        return None


def log_channel_diagnostics(log_id: int, sensors: dict, _severity: dict | None = None):
    """12 rows per telemetry row, v4 vocabulary (none / bias / drift / stuck / spike /
    noise / dropout). severity_percent stays NULL: the sensor-severity head was never
    trained, so there is nothing honest to write."""
    if not db._enabled:
        return
    try:
        rows = [{"log_id": log_id, "channel_name": ch, "fault_type": s.get("condition"),
                 "confidence": _r(s.get("confidence")), "severity_percent": None}
                for ch, s in sensors.items()]
        if rows:
            db._client.table("channel_diagnostics").insert(rows).execute()
    except Exception as e:
        print(f"[dbv4] log_channel_diagnostics failed: {e}")


def log_maintenance_event(engine_id: int | None, simulation_id: int | None, engine_hours: float,
                          level: str, subject: str, action: str) -> None:
    """One advisory in the engine's maintenance history, stamped on the life clock."""
    if not db._enabled or engine_id is None:
        return
    try:
        db._client.table("maintenance_events").insert({
            "engine_id": engine_id, "simulation_id": simulation_id, "engine_hours": float(engine_hours),
            "level": level, "subject": subject, "action": action}).execute()
    except Exception as e:
        print(f"[dbv4] log_maintenance_event failed: {e}")
