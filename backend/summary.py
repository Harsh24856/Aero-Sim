"""Post-flight mission summary via Groq (PS 26054 section F, mission reports).

SCOPE - read this before extending it
---------------------------------------
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
    print("[summary] GROQ_API_KEY not set - post-flight summaries use the local template (this is fine).")


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
MAX_SIM_LIFE_HOURS = 20000.0 / 3600.0   # 5.556 - v2 runs only

# Operating limits handed to the model as reference. Without them it judged a
# 585 C EGT on a Rotax 916 iS "far above typical limits" - the limit is ~950 C.
# Values are the ones this codebase already uses: physics.py's sensor comments
# (EGT operating limit ~950 C, CHT ~135-150 C) and advisory.py's oil-temp limit.
REFERENCE_LIMITS = {
    "egt_operating_max_c": 950.0,
    "cht_operating_max_c": 150.0,
    "oil_temp_max_c": 130.0,
}

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
    "- Check `model_version` first.\n"
    "  - v2: `final_rul_hours_model_scale` is on the MODEL's own compressed "
    "scale, not real flight hours - never present it as real hours or convert "
    "it. Judge it against `rul_model_ceiling_hours`, and prefer discussing the "
    "health/RUL percentages. If `rul_is_extrapolated_beyond_ceiling` is true, "
    "say plainly that the estimate lies beyond the model's trained range and "
    "should be treated as indicative only.\n"
    "  - v3: `final_rul_engine_hours` IS real engine hours remaining against "
    "`tbo_hours`; `rul_percent_of_tbo` is that as a percentage. Report them "
    "directly.\n"
    "  - v4: the same RUL fields as v3. `engine_hours_start` / `engine_hours_end` are "
    "the engine's hour meter; engine hours advance `life_scale` times faster than "
    "flight time by design, so do not compare them with the flight duration. "
    "`health_percent` is the engine's WEAR CONDITION (100 = as new), not an "
    "operating margin. `component_faults` counts the rows on which the AI named each "
    "failing component; `rul_wear_limited_rows` counts rows where wear, not the "
    "overhaul date, limits the remaining life.\n"
    "- Judge temperatures ONLY against `reference_limits`. Do not call a value "
    "high unless it exceeds its limit there.\n"
    "- If `ai_diagnostics_available` is false, the AI service produced no output "
    "for this run: say diagnostics were unavailable. That is a gap in monitoring, "
    "not evidence of an engine problem, and must not raise the risk level on its own.\n"
    "- v3 runs may include `failure_modes` (engine faults flagged by the AI) and "
    "`physics_residuals` (sensors disagreeing with the physics model, and a wear "
    "index from 0 to 1). Use them when present.\n"
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
        "model_version": sim_row.get("model_version") or "v2",
        "ai_diagnostics_available": len(ai_rows) > 0,
        "reference_limits": REFERENCE_LIMITS,
        "channels": {c: _stats(log_rows, c) for c in _NUMERIC_COLS},
    }
    rul = sim_row.get("final_rul_hours")
    if digest["model_version"] == "v4":
        _v4_digest(digest, sim_row, log_rows)
    if digest["model_version"] in ("v3", "v4"):
        tbo = sim_row.get("tbo_hours")
        digest["tbo_hours"] = tbo
        digest["final_rul_engine_hours"] = rul
        digest["rul_percent_of_tbo"] = (round(100.0 * rul / tbo, 1)
                                        if isinstance(rul, (int, float)) and tbo else None)
    else:
        digest["final_rul_hours_model_scale"] = rul
        digest["rul_model_ceiling_hours"] = round(MAX_SIM_LIFE_HOURS, 3)
        digest["rul_is_extrapolated_beyond_ceiling"] = (
            rul > MAX_SIM_LIFE_HOURS if isinstance(rul, (int, float)) else None)

    # v3 telemetry rows carry failure modes and physics residuals (dbv3.py).
    fm_counts: dict[str, int] = {}
    for r in log_rows:
        for mode, v in (r.get("failure_modes") or {}).items():
            if isinstance(v, dict) and v.get("present"):
                fm_counts[mode] = fm_counts.get(mode, 0) + 1
    if fm_counts:
        digest["failure_modes"] = {"rows_flagged_per_mode": fm_counts}
    idx_rows = [r for r in log_rows if isinstance(r.get("degradation_index"), (int, float))]
    if idx_rows:
        dev_counts: dict[str, int] = {}
        for r in idx_rows:
            for ch in r.get("residual_deviations") or []:
                dev_counts[ch] = dev_counts.get(ch, 0) + 1
        digest["physics_residuals"] = {
            "wear_index_first": round(idx_rows[0]["degradation_index"], 4),
            "wear_index_last": round(idx_rows[-1]["degradation_index"], 4),
            "rows": len(idx_rows),
            "rows_deviating_per_channel": dev_counts,
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


def _v4_digest(digest: dict[str, Any], sim_row: dict, log_rows: list[dict]) -> None:
    """Physics v4: bar not psi, this engine's own certified limits, both clocks, and
    the component faults the AI named (dbv4.py rows)."""
    import physics_v4
    spec = physics_v4.ENGINE_SPECS_V4.get(sim_row.get("engine_model") or "")
    digest["units"].update({"oil_pressure": "bar", "coolant_temp": "deg C",
                            "manifold_pressure_kpa": "kPa", "battery_voltage": "V",
                            "health_percent": "percent wear condition (100 = new)"})
    if spec is not None:
        digest["reference_limits"] = {"cht_max_c": spec.cht_limit_c, "egt_max_c": spec.egt_limit_c,
                                      "oil_temp_max_c": spec.oil_temp_limit_c,
                                      "oil_pressure_min_bar": spec.oil_press_min_bar,
                                      "oil_pressure_min_bar_below_3500_rpm": spec.oil_press_min_low_bar}
    digest["life_scale"] = sim_row.get("life_scale")
    digest["engine_hours_start"] = sim_row.get("start_engine_hours")
    digest["engine_hours_end"] = sim_row.get("end_engine_hours")
    counts: dict[str, int] = {}
    for r in log_rows:
        for name, v in (r.get("fault_modes") or {}).items():
            if isinstance(v, dict) and v.get("present"):
                counts[name] = counts.get(name, 0) + 1
    if counts:
        digest["component_faults"] = {"rows_flagged_per_fault": counts}
    digest["rul_wear_limited_rows"] = sum(1 for r in log_rows if r.get("wear_limited"))


LOCAL_MODEL_NAME = "local template (offline)"


def local_summary(digest: dict[str, Any], reason: str) -> dict[str, Any]:
    """Deterministic post-flight summary built from the digest alone - no network.

    Used when Groq is not configured or unreachable, so every report still has a
    headline, findings and recommendations. It states only what the digest holds.
    """
    import advisory                                  # local: summary is imported by main before advisory

    ch = digest.get("channels") or {}
    limits = digest.get("reference_limits") or REFERENCE_LIMITS
    engine = (digest.get("engine_model") or "Engine").replace("_", " ")
    health = digest.get("final_health_percent")
    findings: list[str] = []
    recs: list[str] = []
    rank = {"low": 0, "moderate": 1, "high": 2}
    risk = "low"

    def raise_risk(level: str) -> None:
        nonlocal risk
        if rank[level] > rank[risk]:
            risk = level

    if not digest.get("ai_diagnostics_available"):
        findings.append("AI diagnostics produced no output for this run; judge it on the sensor limits only.")
    if isinstance(health, (int, float)):
        findings.append(f"Final engine health {health:.1f}%.")
        raise_risk("high" if health < 50 else "moderate" if health < 80 else "low")

    if digest.get("model_version") in ("v3", "v4") and isinstance(digest.get("final_rul_engine_hours"), (int, float)):
        pct = digest.get("rul_percent_of_tbo")
        findings.append(f"Remaining useful life {digest['final_rul_engine_hours']:.0f} engine hours"
                        + (f" ({pct:.1f}% of the {digest.get('tbo_hours'):.0f} h TBO)." if pct is not None else "."))

    exceeded = []
    for col, key, name in (("egt", "egt_operating_max_c", "EGT"), ("cht", "cht_operating_max_c", "CHT"),
                           ("oil_temp", "oil_temp_max_c", "oil temperature")):
        peak = (ch.get(col) or {}).get("max")
        limit = limits.get(key)
        if isinstance(peak, (int, float)) and limit and peak > limit:
            exceeded.append(f"{name} {peak:.0f} C (limit {limit:.0f} C)")
    if exceeded:
        findings.append("Temperature limits exceeded: " + ", ".join(exceeded) + ".")
        recs.append("Inspect cooling, mixture and temperature sensors before the next sortie.")
        raise_risk("moderate")
    oil_min = (ch.get("oil_pressure") or {}).get("min")
    if isinstance(oil_min, (int, float)) and oil_min < advisory.OIL_PRESSURE_MIN_PSI:
        findings.append(f"Oil pressure fell to {oil_min:.1f} psi, below the {advisory.OIL_PRESSURE_MIN_PSI:.0f} psi minimum.")
        recs.append("Check oil level, pump and pressure sender before further flight.")
        raise_risk("high")

    modes = (digest.get("failure_modes") or {}).get("rows_flagged_per_mode") or {}
    if modes:
        findings.append("AI flagged engine failure modes: "
                        + ", ".join(f"{m.replace('_', ' ')} ({n} rows)" for m, n in modes.items()) + ".")
        recs.append("Review the flagged failure modes on the replay page and inspect the matching subsystems.")
        raise_risk("moderate")
    deviations = (digest.get("physics_residuals") or {}).get("rows_deviating_per_channel") or {}
    if deviations:
        findings.append("Sensors disagreed with the physics model on: "
                        + ", ".join(f"{c.replace('_', ' ')} ({n} rows)" for c, n in deviations.items()) + ".")
        recs.append("Verify calibration of the sensors that disagreed with physics.")

    if not recs:
        recs.append("No limit exceedances or flagged faults - continue routine monitoring.")
    if risk == "high":
        recs.insert(0, "Ground the aircraft pending inspection.")

    duration = digest.get("duration_simulated_seconds") or 0
    headline = f"{engine}: {'health ' + format(health, '.0f') + '%, ' if isinstance(health, (int, float)) else ''}{risk} risk"
    summary_text = (f"{engine} flew {duration:.0f} simulated seconds with {digest.get('telemetry_rows', 0)} telemetry rows "
                    f"and {digest.get('fault_detected_rows', 0)} rows with a detected fault. "
                    + " ".join(findings[:3]))
    return {
        "status": "ok",
        "model": LOCAL_MODEL_NAME,
        "offline": True,
        "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "headline": headline[:90],
        "summary": summary_text,
        "findings": findings[:5],
        "recommendations": recs[:5],
        "risk": risk,
        "digest": digest,
    }


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

    digest = None
    try:
        sim_row = db.get_simulation(simulation_id)
        if not sim_row:
            print(f"[summary] simulation {simulation_id} not found")
            return None

        log_rows = db.get_telemetry_rows(simulation_id)
        digest = build_digest(sim_row, log_rows)

        if not _enabled or _client is None:
            payload = local_summary(digest, "GROQ_API_KEY not configured")
            db.save_groq_result(simulation_id, payload)
            print(f"[summary] simulation {simulation_id} summarized ({LOCAL_MODEL_NAME})")
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
            # Groq unreachable (no network at the venue, rate limit, outage): fall back
            # to the local template rather than leaving the report without a summary.
            payload = (local_summary(digest, f"Groq unavailable: {exc}") if digest is not None
                       else _fallback(str(exc)))
            db.save_groq_result(simulation_id, payload)
            return payload
        except Exception:
            return None
