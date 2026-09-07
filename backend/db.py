"""
Supabase integration for persisting simulation runs, telemetry, and AI diagnostics.
Uses the service_role key (full trust, bypasses RLS) since the backend writes on
behalf of whichever user started the simulation - RLS still protects reads made
directly by the frontend using the anon key, which is the actual security boundary.

Schema: users login via Supabase Auth directly (not reinvented here). simulations
(one row per run) -> telemetry_logs (one row per timestep) -> channel_diagnostics
(one row per sensor channel per timestep) - matches the ER diagram exactly.
"""
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

_client: Client | None = None
_enabled = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)

if _enabled:
    _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
else:
    print("WARNING: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set - "
          "simulation persistence is disabled, the simulator will still run locally.")

def start_simulation(user_id: str | None, engine_model: str) -> int | None:
    """Creates a new simulations row. Returns its id, or None if persistence is
    disabled or no user_id was provided (an anonymous/not-logged-in session)."""
    if not _enabled or not user_id:
        return None
    try:
        result = _client.table("simulations").insert({
            "user_id": user_id,
            "engine_model": engine_model,
        }).execute()
        return result.data[0]["id"]
    except Exception as e:
        print(f"[db] start_simulation failed: {e}")
        return None


def log_telemetry(simulation_id: int | None, time_offset_s: float, raw: dict, ai: dict | None) -> int | None:
    """Inserts one telemetry_logs row (raw physics + AI outputs for this instant)."""
    if not _enabled or simulation_id is None:
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
        if ai and ai.get("status") == "ok":
            row.update({
                "fault_detected": ai.get("fault_detected"),
                "detection_confidence": ai.get("detection_confidence"),
                "health_percent": ai.get("health_percent"),
                "rul_percent_remaining": ai.get("rul_percent_remaining"),
                "rul_hours": ai.get("rul_hours_internal"),
            })
        result = _client.table("telemetry_logs").insert(row).execute()
        log_id = result.data[0]["id"]
        if ai and ai.get("status") == "ok" and ai.get("diagnosis"):
            log_channel_diagnostics(log_id, ai["diagnosis"], ai.get("severity_percent") or {})
        return log_id
    except Exception as e:
        print(f"[db] log_telemetry failed: {e}")
        return None

def log_channel_diagnostics(log_id: int, diagnosis: dict, severity_percent: dict):
    """Inserts one channel_diagnostics row per sensor channel (8 channels: egt/cht/
    oil_pressure/oil_temp/vibx/viby/vibz/rpm), matching ai.py's per-channel output."""
    if not _enabled:
        return
    try:
        rows = []
        for channel, d in diagnosis.items():
            rows.append({
                "log_id": log_id,
                "channel_name": channel,
                "fault_type": d.get("fault_type"),
                "confidence": d.get("confidence"),
                "severity_percent": severity_percent.get(channel),
            })
        if rows:
            _client.table("channel_diagnostics").insert(rows).execute()
    except Exception as e:
        print(f"[db] log_channel_diagnostics failed: {e}")


def end_simulation(simulation_id: int | None, outcome: str, final_health_percent: float | None,
                    final_rul_hours: float | None, final_telemetry: dict | None = None):
    """Updates the simulations row once a run stops (landed/crashed/manually stopped).
    final_telemetry is the COMPLETE raw physics dict as it stood at the exact moment
    of stopping - the full sensor state, not just the two aggregated AI numbers
    already captured above. Stored as-is (JSONB) rather than picked apart into
    columns, since this is a point-in-time snapshot for quick reference, not
    something queried/filtered on individually (telemetry_logs already covers that)."""
    if not _enabled or simulation_id is None:
        return
    try:
        _client.table("simulations").update({
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome,
            "final_health_percent": final_health_percent,
            "final_rul_hours": final_rul_hours,
            "final_telemetry": final_telemetry,
        }).eq("id", simulation_id).execute()
    except Exception as e:
        print(f"[db] end_simulation failed: {e}")

def get_simulation(simulation_id: int) -> dict | None:
    """Fetches one simulation row (used by /resume to look up its engine_model +
    final_telemetry snapshot server-side, via the trusted service_role client -
    RLS is bypassed here deliberately, since this is a backend-internal lookup,
    not a client request that should be scoped to "only their own rows" (the
    /resume endpoint itself still requires the correct user_id to proceed)."""
    if not _enabled:
        return None
    try:
        result = _client.table("simulations").select("*").eq("id", simulation_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"[db] get_simulation failed: {e}")
        return None


def get_telemetry_rows(simulation_id: int) -> list[dict]:
    """Every per-second row for one run, oldest first.

    service_role, so this bypasses RLS - callers must have already established
    that the requester owns the run (main.py does this for /summarize, and the
    /stop hook only ever passes an id it just created itself).
    """
    if not _enabled:
        return []
    try:
        result = (_client.table("telemetry_logs")
                  .select("*")
                  .eq("simulation_id", simulation_id)
                  .order("time_offset_s", desc=False)
                  .limit(5000)
                  .execute())
        return result.data or []
    except Exception as e:
        print(f"[db] get_telemetry_rows failed: {e}")
        return []


def save_groq_result(simulation_id: int, payload: dict) -> None:
    """Persist the post-flight narrative summary onto the run row."""
    if not _enabled:
        return
    try:
        _client.table("simulations").update({"groq_result": payload}).eq("id", simulation_id).execute()
    except Exception as e:
        print(f"[db] save_groq_result failed: {e}")
