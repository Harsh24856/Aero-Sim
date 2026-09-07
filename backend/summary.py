"""Post-flight mission summary via Groq (PS 26054 section F, mission reports).

SCOPE - read this before extending it
-------------------------------------
This is a NARRATIVE layer over a run that has already finished. It is NOT the
maintenance advisory: that lives in advisory.py, is deterministic, runs onboard
in real time, and must keep working with no network. Nothing safety-relevant
should ever depend on this module - if Groq is down, the product loses a prose
paragraph on a report page and nothing else.

DESIGN NOTES
------------
* Follows db.py's env pattern exactly: module-level load_dotenv() + an
  `_enabled` flag that degrades to a printed no-op. Missing GROQ_API_KEY is a
  normal, supported state, not an error.
* The model is fed AGGREGATES, never raw rows. A long run is ~800 telemetry_logs
  rows; sending those would be slow, expensive, and would bury the signal. We
  compute per-channel min/mean/max, trajectory endpoints and fault counts, and
  send roughly a page of numbers.
* Everything is wrapped so that a failure writes an "error" summary rather than
  raising: this is invoked from a fire-and-forget task inside the physics
  process, where an unhandled exception is a silent dead task.
"""

import os
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv

import db

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

_enabled = bool(GROQ_API_KEY)
_client = None

if _enabled:
    try:
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
    except Exception as exc:                      # pragma: no cover - import/credential shape
        print(f"[summary] Groq client unavailable, summaries disabled: {exc}")
        _enabled = False
else:
    print("[summary] GROQ_API_KEY not set - post-flight summaries disabled (this is fine).")


# ai.py's own ceiling: the training generator's 20,000s censoring cutoff, above
# which its RUL head is extrapolating rather than interpolating.
#
# An earlier version of this file scaled RUL by 360 to present "real-world
# hours" against a 2,000h TBO. That was wrong to do here. The mapping is an
# assumption rather than a measured property of the model, and it fails exactly
# where it matters: 29 of 143 recorded runs have final_rul_hours ABOVE this
# ceiling (max 546.6), so scaling would have reported ~197,000 hours of
# remaining life. RUL is handed over on the model's own scale, with the ceiling
# and an explicit out-of-range flag so the model can caveat instead of guess.
MAX_SIM_LIFE_HOURS = 20000.0 / 3600.0   # 5.556

# Channels whose min/mean/max are worth putting in front of the model.
_NUMERIC_COLS = [
    "altitude", "throttle", "airspeed", "engine_rpm", "power_kw", "fuel_flow",
    "thrust", "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz",
    "health_percent", "rul_percent_remaining",
]

SYSTEM_PROMPT = (
    "You are a propulsion health analyst writing a post-flight summary for the "
    "operators and maintenance engineers of a MALE UAV powered by a Rotax aero "
    "piston engine. You are given aggregated telemetry and AI diagnostic output "
    "from one completed simulated mission.\n\n"
    "Write for someone deciding whether this airframe flies again tomorrow.\n"
    "Rules:\n"
    "- Ground every claim in the numbers provided. Never invent a value.\n"
    "- If the diagnostic data is sparse or absent, say so plainly.\n"
    "- Note that the first 128 simulated seconds of any mission have no AI "
    "output by design (model window fill), so early blanks are not a fault.\n"
    "- Be concise and specific. No preamble, no marketing language.\n"
    "- RUL: `final_rul_hours_model_scale` is on the MODEL's own compressed "
    "scale, not real flight hours - never present it as real hours or convert "
    "it. Judge it against `rul_model_ceiling_hours`, and prefer discussing the "
    "health/RUL percentages. If `rul_is_extrapolated_beyond_ceiling` is true, "
    "say plainly that the estimate lies beyond the model's trained range and "
    "should be treated as indicative only.\n"
    "- The `units` object gives the unit of every channel. Use exactly those "
    "units and never convert or relabel them - temperatures are Celsius, not "
    "Fahrenheit. Airspeed may additionally be given in knots (1 m/s = 1.94 kt).\n"
    "- At most 5 findings and 5 recommendations.\n\n"
    "Return ONE JSON object, exactly this shape. `findings` and `recommendations` are "
    "each a SINGLE flat array of strings - not an array per item:\n"
    "{\n"
    '  "headline": "one line, under 90 characters",\n'
    '  "summary": "one paragraph",\n'
    '  "findings": ["first finding", "second finding"],\n'
    '  "recommendations": ["first action", "second action"],\n'
    '  "risk": "low"\n'
    "}"
)


def _salvage(raw: str) -> Optional[dict]:
    """Recover a usable object from not-quite-valid JSON.

    Observed failure with gpt-oss-120b: it closes and reopens the array for every
    element, emitting `"findings":["a"],["b"],["c"]]` instead of one flat array.
    The prose in those responses is good, so discarding the whole answer over a
    bracket is the wrong trade. Repairs that specific pattern, then re-parses.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r"\]\s*,\s*\[", ", ", raw)      # ],[  ->  ,
    repaired = re.sub(r"\]\s*\]", "]", repaired)       # ]]   ->  ]
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _stats(rows: list[dict], col: str) -> Optional[dict]:
    vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
    if not vals:
        return None
    return {
        "min": round(min(vals), 3),
        "mean": round(sum(vals) / len(vals), 3),
        "max": round(max(vals), 3),
    }


def build_digest(sim_row: dict, log_rows: list[dict]) -> dict[str, Any]:
    """Condense a finished run into the compact object handed to the model."""
    ai_rows = [r for r in log_rows if isinstance(r.get("health_percent"), (int, float))]
    faults = [r for r in log_rows if r.get("fault_detected")]

    digest: dict[str, Any] = {
        # Stated explicitly because the model otherwise guesses: it rendered a
        # 752 C EGT as "752 F" when left to infer units from the key name alone.
        "units": {
            "altitude": "m", "airspeed": "m/s", "throttle": "fraction 0-1",
            "engine_rpm": "rpm", "power_kw": "kW", "fuel_flow": "L/h",
            "thrust": "N", "egt": "deg C", "cht": "deg C",
            "oil_pressure": "psi", "oil_temp": "deg C",
            "vibx": "g", "viby": "g", "vibz": "g",
            "health_percent": "percent", "rul_percent_remaining": "percent",
        },
        "engine_model": sim_row.get("engine_model"),
        "outcome": sim_row.get("outcome"),
        "duration_simulated_seconds": max((r.get("time_offset_s") or 0) for r in log_rows) if log_rows else 0,
        "telemetry_rows": len(log_rows),
        "rows_with_ai_output": len(ai_rows),
        "ai_warmup_rows_without_output": len(log_rows) - len(ai_rows),
        "fault_detected_rows": len(faults),
        "final_health_percent": sim_row.get("final_health_percent"),
        "final_rul_hours_model_scale": sim_row.get("final_rul_hours"),
        "rul_model_ceiling_hours": round(MAX_SIM_LIFE_HOURS, 3),
        "rul_is_extrapolated_beyond_ceiling": (
            sim_row["final_rul_hours"] > MAX_SIM_LIFE_HOURS
            if isinstance(sim_row.get("final_rul_hours"), (int, float)) else None),
        "channels": {c: _stats(log_rows, c) for c in _NUMERIC_COLS},
    }

    if ai_rows:
        digest["health_trajectory"] = {
            "first": round(ai_rows[0]["health_percent"], 3),
            "last": round(ai_rows[-1]["health_percent"], 3),
            "min": round(min(r["health_percent"] for r in ai_rows), 3),
        }

    final_tel = sim_row.get("final_telemetry") or {}
    if isinstance(final_tel, dict):
        digest["final_wear"] = final_tel.get("wear")
        digest["engine_failed"] = final_tel.get("failed")

    return digest


def _fallback(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_summary(simulation_id: int) -> Optional[dict[str, Any]]:
    """Build and persist a summary for a finished run. Never raises.

    Returns the stored object, or None when persistence itself is unavailable.
    """
    if not db._enabled:
        print("[summary] Supabase disabled; nothing to summarize.")
        return None

    try:
        sim_row = db.get_simulation(simulation_id)
        if not sim_row:
            print(f"[summary] simulation {simulation_id} not found")
            return None

        log_rows = db.get_telemetry_rows(simulation_id)
        digest = build_digest(sim_row, log_rows)

        if not _enabled or _client is None:
            payload = _fallback("GROQ_API_KEY not configured")
            payload["digest"] = digest
            db.save_groq_result(simulation_id, payload)
            return payload

        completion = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(digest, default=str)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        parsed = _salvage(raw) or {
            "headline": "Mission summary", "summary": raw,
            "findings": [], "recommendations": [], "risk": "low",
        }

        payload = {
            "status": "ok",
            "model": GROQ_MODEL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "headline": parsed.get("headline"),
            "summary": parsed.get("summary"),
            "findings": parsed.get("findings") or [],
            "recommendations": parsed.get("recommendations") or [],
            "risk": parsed.get("risk"),
            "digest": digest,
        }
        db.save_groq_result(simulation_id, payload)
        print(f"[summary] simulation {simulation_id} summarized ({GROQ_MODEL})")
        return payload

    except Exception as exc:
        salvaged = None
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            failed = (body.get("error") or {}).get("failed_generation")
            salvaged = _salvage(failed) if failed else None
        if salvaged:
            print(f"[summary] simulation {simulation_id}: recovered from json_validate_failed")
            payload = {
                "status": "ok", "model": GROQ_MODEL,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "headline": salvaged.get("headline"), "summary": salvaged.get("summary"),
                "findings": salvaged.get("findings") or [],
                "recommendations": salvaged.get("recommendations") or [],
                "risk": salvaged.get("risk"), "recovered": True,
            }
            try:
                payload["digest"] = digest
            except NameError:
                pass
            db.save_groq_result(simulation_id, payload)
            return payload
        print(f"[summary] generation failed for {simulation_id}: {exc}")
        try:
            payload = _fallback(str(exc))
            db.save_groq_result(simulation_id, payload)
            return payload
        except Exception:
            return None
