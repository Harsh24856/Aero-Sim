"""
Supabase persistence for physics-v3 runs served by aiv3.py.

Same function names as db.py, so main.py swaps modules on AERO_PHYSICS_VERSION=v3.
It reuses db.py's client and every function whose behaviour does not depend on the
model version.

What differs from db.py (schema: migration add_physics_v3_model_versioning):
  - simulations.model_version = 'v3' and simulations.tbo_hours are written at start,
    so the frontend never reads a v3 run on the v2 simulated timescale (all rows
    that predate the migration default to 'v2' and render as legacy).
  - telemetry_logs.rul_hours / simulations.final_rul_hours hold REAL ENGINE HOURS
    (main.final_rul_hours picks the right field before calling end_simulation).
  - telemetry_logs.failure_modes, degradation_index and residual_deviations carry
    the aiv3.py failure modes and the backend/residual.py physics residuals. The
    residual fields are model-free and are written even while the AI warms up.
"""
import db
from db import (                                   # noqa: F401  re-exported unchanged
    log_channel_diagnostics,
    end_simulation,
    get_simulation,
    get_telemetry_rows,
    save_groq_result,
    get_max_time_offset,
    reopen_simulation,
)
from physics import TBO_HOURS

MODEL_VERSION = "v3"


def start_simulation(user_id: str | None, engine_model: str) -> int | None:
    """Creates a v3 simulations row. None if persistence is off or the user is anonymous."""
    if not db._enabled or not user_id:
        return None
    try:
        result = db._client.table("simulations").insert({
            "user_id": user_id,
            "engine_model": engine_model,
            "model_version": MODEL_VERSION,
            "tbo_hours": TBO_HOURS.get(engine_model),
        }).execute()
        return result.data[0]["id"]
    except Exception as e:
        print(f"[dbv3] start_simulation failed: {e}")
        return None


def log_telemetry(simulation_id: int | None, time_offset_s: float, raw: dict, ai: dict | None,
                  residuals: dict | None = None) -> int | None:
    """Inserts one telemetry_logs row (raw physics + v3 AI outputs + physics residuals)."""
    if not db._enabled or simulation_id is None:
        return None
    try:
        row = {
            "simulation_id": simulation_id,
            "time_offset_s": time_offset_s,
            "altitude": raw.get("altitude"), "throttle": raw.get("throttle"),
            "airspeed": raw.get("airspeed"), "aoa": raw.get("aoa"),
            "engine_rpm": raw.get("engine_rpm"), "prop_rpm": raw.get("prop_rpm"),
            "power_kw": raw.get("power_kw"), "fuel_flow": raw.get("fuel_flow"),
            "thrust": raw.get("thrust"), "lift": raw.get("lift"), "drag": raw.get("drag"),
            "egt": raw.get("egt"), "cht": raw.get("cht"),
            "oil_pressure": raw.get("oil_pressure"), "oil_temp": raw.get("oil_temp"),
            "vibx": raw.get("vibx"), "viby": raw.get("viby"), "vibz": raw.get("vibz"),
        }
        ai_v3 = bool(ai and ai.get("status") == "ok" and ai.get("model_version") == MODEL_VERSION)
        if ai and ai.get("status") == "ok" and not ai_v3:
            print(f"[dbv3] AI result is {ai.get('model_version') or 'v2'}, not v3 - "
                  "rul_hours units would be wrong, skipping AI fields for this row")
        if ai_v3:
            row.update({
                "fault_detected": ai.get("fault_detected"),
                "detection_confidence": ai.get("detection_confidence"),
                "health_percent": ai.get("health_percent"),
                "rul_percent_remaining": ai.get("rul_percent_remaining"),
                "rul_hours": ai.get("rul_hours"),          # engine hours
                "failure_modes": {
                    m: {"severity_percent": round(float(v.get("severity_percent", 0.0)), 2),
                        "present": bool(v.get("present"))}
                    for m, v in (ai.get("failure_modes") or {}).items()
                } or None,
            })
        if residuals and residuals.get("enabled"):
            row["degradation_index"] = residuals.get("degradation_index")
            row["residual_deviations"] = list(residuals.get("deviations") or [])
        result = db._client.table("telemetry_logs").insert(row).execute()
        log_id = result.data[0]["id"]
        if ai_v3 and ai.get("diagnosis"):
            log_channel_diagnostics(log_id, ai["diagnosis"], ai.get("severity_percent") or {})
        return log_id
    except Exception as e:
        print(f"[dbv3] log_telemetry failed: {e}")
        return None
