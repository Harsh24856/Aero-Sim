"""Does the twin hold up against an engine it was not built from? (plan P3, item 13)

For each mismatch profile in backend/aircraft_sim.py a PLANT engine and the v3 TWIN fly the
same set-points. The plant's sensors (with the mismatch applied) replace the twin's sensor
channels exactly as main.py's apply_measured() does, and the result is scored the way the
live system sees it: the AI service (/step) and the physics residual monitor.

Ground truth comes from the plant alone: its injected sensor-fault flags, its engine
failure-mode labels and its true remaining hours. "sensor_drift" is a genuine sensor fault
(CHT drifting 1 C/min) that SHOULD be caught; every other profile is a healthy engine that
should NOT raise alarms.

Runs against the live AI service on :8100 (no second TensorFlow process - the laptop has
8 GB). The AI service keeps one rolling window, so this refuses to run while a flight is
live on :8000 unless --force.

    cd backend && ./.venv/bin/python ../validation/mismatch_eval.py
    ./.venv/bin/python ../validation/mismatch_eval.py --engines 914 916 --seconds 420
"""
import argparse
import json
import os
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from aircraft_sim import MISMATCH_PROFILES, SENSOR_KEYS, apply_mismatch   # noqa: E402
from physics import UAVEngineTwin                                         # noqa: E402
import residual                                                           # noqa: E402

AI_URL = "http://127.0.0.1:8100"
MAIN_URL = "http://127.0.0.1:8000"
ENGINES = {"912": "Rotax_912_ULS", "914": "Rotax_914_ULF", "915": "Rotax_915_iS", "916": "Rotax_916_iS"}
AI_FEATURE_COLS = [
    "altitude", "throttle", "airspeed", "aoa", "air_density",
    "torque_available_nm", "engine_rpm", "prop_rpm", "prop_torque", "power_kw", "fuel_flow",
    "thrust", "lift", "drag", "thrust_margin", "lift_weight_margin",
    "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm_fault",
    "injection_timing",
]
# Cruise below every stress threshold, so a healthy plant should give a quiet twin.
CRUISE = dict(throttle=0.5, altitude=1500.0, airspeed=45.0, aoa=2.0, isa_dev_c=0.0)


def to_float(v):
    return float(v) if isinstance(v, (int, float, np.floating)) else v


def run_profile(client: httpx.Client, engine: str, name: str, seconds: int, seed: int) -> dict:
    mismatch = MISMATCH_PROFILES[name]
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    plant = UAVEngineTwin(dt=1.0, engine_model=engine, physics_version=mismatch.get("physics_version", "v3"))
    twin = UAVEngineTwin(dt=1.0, engine_model=engine, physics_version="v3")
    for e in (plant, twin):
        for k, v in CRUISE.items():
            setattr(e, k, v)
    monitor = residual.ResidualMonitor(lag="euler")
    client.post(f"{AI_URL}/select_engine", json={"engine_model": engine}).raise_for_status()
    client.post(f"{AI_URL}/reset").raise_for_status()

    rows = []
    for _ in range(seconds):
        p = plant.step()
        t = twin.step()
        measured = apply_mismatch({k: float(p[src]) for k, src in SENSOR_KEYS.items()}, mismatch, p["time"], rng)
        lo_hi = {**residual.SENSOR_LIMITS, "fuel_flow": (0.0, 200.0)}
        merged = dict(t)
        for ch, v in measured.items():
            lo, hi = lo_hi[ch]
            merged[{"rpm": "rpm_fault"}.get(ch, ch)] = min(hi, max(lo, v))
        ai = client.post(f"{AI_URL}/step", json={**{c: to_float(merged[c]) for c in AI_FEATURE_COLS},
                                                  "time": merged["time"]}).json()
        res = monitor.update(merged, dt=1.0)
        rows.append({
            "plant_flags": [c for c, f in p["fault_flags"].items() if f],
            "plant_modes": [k for k, v in p.items() if k.startswith("fm_") and isinstance(v, (int, float)) and v > 0],
            "rul_true": p["rul_hours_true"], "tbo": p["tbo_hours"],
            "ai": ai, "deviations": (res or {}).get("deviations") or [],
        })

    scored = [r for r in rows if r["ai"].get("status") == "ok"]
    clean = [r for r in scored if not r["plant_flags"] and not r["plant_modes"]]
    out = {"engine": engine, "profile": name, "seconds": seconds, "ai_samples": len(scored), "clean_samples": len(clean)}
    if not scored:
        out["error"] = f"no AI output (last status {rows[-1]['ai'].get('status') if rows else None})"
        return out
    rate = lambda rs, f: round(sum(1 for r in rs if f(r)) / len(rs), 4) if rs else None
    out.update({
        "ai_false_alarm_rate": rate(clean, lambda r: r["ai"].get("fault_detected")),
        "ai_channel_false_call_rate": rate(clean, lambda r: bool(r["ai"].get("faulty_channels"))),
        "failure_mode_false_rate": rate(clean, lambda r: any(
            isinstance(v, dict) and v.get("present") for v in (r["ai"].get("failure_modes") or {}).values())),
        "residual_deviation_rate": rate(clean, lambda r: bool(r["deviations"])),
        "health_mean": round(float(np.mean([r["ai"].get("health_percent", 0) for r in scored])), 1),
        "health_min": round(float(min(r["ai"].get("health_percent", 0) for r in scored)), 1),
        "rul_error_pct_tbo": round(float(np.mean([abs(r["ai"].get("rul_hours", 0) - r["rul_true"]) / r["tbo"] * 100
                                                  for r in scored])), 2),
    })
    channels_dev = {}
    for r in clean:
        for c in r["deviations"]:
            channels_dev[c] = channels_dev.get(c, 0) + 1
    out["residual_deviating_channels"] = channels_dev
    if "drift_per_min" in mismatch:
        ch = next(iter(mismatch["drift_per_min"]))
        first_res = next((i for i, r in enumerate(rows) if ch in r["deviations"]), None)
        first_ai = next((i for i, r in enumerate(rows) if ch in (r["ai"].get("faulty_channels") or [])), None)
        rate_c = mismatch["drift_per_min"][ch]
        out["drift_channel"] = ch
        out["residual_first_flag_s"] = first_res
        out["residual_flag_at_offset_c"] = None if first_res is None else round(rate_c * first_res / 60.0, 1)
        out["ai_first_flag_s"] = first_ai
        out["ai_flag_at_offset_c"] = None if first_ai is None else round(rate_c * first_ai / 60.0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", nargs="+", default=["914"], choices=sorted(ENGINES))
    ap.add_argument("--profiles", nargs="+", default=sorted(MISMATCH_PROFILES), choices=sorted(MISMATCH_PROFILES))
    ap.add_argument("--seconds", type=int, default=420, help="simulated seconds per run (128 are AI warm-up)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--force", action="store_true", help="run even if a flight is live on :8000")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mismatch_eval.json"))
    a = ap.parse_args()

    client = httpx.Client(timeout=30.0)
    try:
        if client.get(f"{MAIN_URL}/health").json().get("running") and not a.force:
            sys.exit("a flight is running on :8000 and shares the AI service's window - stop it or pass --force")
    except httpx.HTTPError:
        pass                                          # physics backend not running: fine
    client.get(f"{AI_URL}/health").raise_for_status()

    results = []
    for key in a.engines:
        for name in a.profiles:
            t0 = time.monotonic()
            r = run_profile(client, ENGINES[key], name, a.seconds, a.seed)
            r["wall_s"] = round(time.monotonic() - t0, 1)
            results.append(r)
            print(json.dumps(r), flush=True)
    client.post(f"{AI_URL}/reset")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print("\n| Engine | Profile | AI false alarm | Failure-mode false | Residual false dev | Health mean / min | RUL err % TBO | Drift caught (residual / AI) |")
    print("|---|---|---|---|---|---|---|---|")
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    for r in results:
        drift = "-"
        if "drift_channel" in r:
            fmt = lambda s, c: "not flagged" if s is None else f"{s} s (+{c} C)"
            drift = f"{fmt(r['residual_first_flag_s'], r['residual_flag_at_offset_c'])} / {fmt(r['ai_first_flag_s'], r['ai_flag_at_offset_c'])}"
        print(f"| {r['engine'].split('_')[1]} | {r['profile']} | {pct(r.get('ai_false_alarm_rate'))} | "
              f"{pct(r.get('failure_mode_false_rate'))} | {pct(r.get('residual_deviation_rate'))} | "
              f"{r.get('health_mean')} / {r.get('health_min')} | {r.get('rul_error_pct_tbo')} | {drift} |")
    print(f"\nresults -> {a.out}")


if __name__ == "__main__":
    main()
