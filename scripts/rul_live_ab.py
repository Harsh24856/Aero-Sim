#!/usr/bin/env python3
"""A/B two RUL heads on live simulated flights, against the twin's own true RUL.

The offline test split says one head is better. This says whether that survives
contact with the running stack: the same flight profile, the same engine, the
prediction compared against `wear * tbo_hours` straight off the physics twin - which
is the exact quantity the label was built from, so there is no proxy in between.

    scripts/rul_live_ab.py --engine Rotax_914_ULF \
        --model new=/path/914_rul.keras --model old=/tmp/914_rul_OLD.keras

For each named model it installs the file into backend/models_v3/<key>/, restarts the
AI service so the weights are actually reloaded, flies one profile, and reports MAE
and bias against truth - overall and in the near-new band, which is what a judge sees
on a fresh start.

The engine-failure shutdown is disabled for the run (AERO_HEALTH_FAILURE_HOLD_S=0):
a head that reads a hard-working engine as unhealthy would otherwise end the flight
and shorten its own comparison.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "http://127.0.0.1:8000"
AI = "http://127.0.0.1:8100"
KEY = {"Rotax_912_ULS": "912", "Rotax_914_ULF": "914",
       "Rotax_915_iS": "915", "Rotax_916_iS": "916"}


def post(url, path, body=None, timeout=30):
    req = urllib.request.Request(url + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url, path="/state", timeout=30):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return json.loads(r.read())


def wait_for(url, path, timeout_s):
    for _ in range(timeout_s):
        try:
            get(url, path, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def start_services(env_extra):
    """AI service then physics backend, each waited for. Returns the two Popens."""
    env = {**os.environ, **env_extra, "AERO_PHYSICS_VERSION": "v3"}
    ai = subprocess.Popen(["../validation/venv/bin/uvicorn", "aiv3:app",
                           "--host", "127.0.0.1", "--port", "8100"],
                          cwd=os.path.join(ROOT, "backend"), env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_for(AI, "/health", 240):
        raise RuntimeError("AI service did not come up")
    be = subprocess.Popen([".venv/bin/python", "-m", "uvicorn", "main:app",
                           "--host", "127.0.0.1", "--port", "8000"],
                          cwd=os.path.join(ROOT, "backend"), env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_for(API, "/health", 120):
        raise RuntimeError("physics backend did not come up")
    return ai, be


def stop_services(procs):
    for p in procs:
        if p and p.poll() is None:
            p.send_signal(signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()


def fly(engine, seconds, throttle, airspeed, altitude, settle_grace=40):
    """One flight. Returns rows of (sim_time, true_rul, predicted_rul, wear, settling)."""
    post(API, "/select_engine", {"engine_model": engine})
    post(API, "/start")
    for _ in range(300):                      # wait for the AI to fill its window
        if (get(API).get("ai") or {}).get("status") == "ok":
            break
        time.sleep(2)
    rows, t0 = [], time.time()
    while time.time() - t0 < seconds:
        post(API, "/params", {"throttle": throttle, "airspeed": airspeed, "altitude": altitude})
        s = get(API)
        ai, tel = s.get("ai") or {}, s.get("telemetry") or {}
        wear, tbo = tel.get("wear"), tel.get("tbo_hours")
        pred = ai.get("rul_hours")
        if (ai.get("status") == "ok" and isinstance(wear, (int, float))
                and isinstance(pred, (int, float))):
            rows.append({"t": tel.get("time"), "true": tbo*(1.0 - wear), "pred": pred,
                         "raw": ai.get("rul_hours_raw"), "wear": wear, "tbo": tbo,
                         "settling": bool(ai.get("settling"))})
        time.sleep(2)
    post(API, "/stop", {"final": True})
    # The takeoff hold blanks alerts for the first 128 s of in-envelope flight; RUL is
    # not held, but the engine is still settling, so skip that opening stretch.
    return [r for r in rows if not r["settling"]][settle_grace//2:]


def score(rows):
    if not rows:
        return None
    tbo = rows[0]["tbo"]
    err = [r["pred"] - r["true"] for r in rows]
    nn = [e for e, r in zip(err, rows) if r["true"] >= 0.90*tbo]
    n = len(err)
    return {
        "n": n, "tbo": tbo,
        "mae_h": sum(abs(e) for e in err)/n,
        "mae_pct": 100*sum(abs(e) for e in err)/n/tbo,
        "bias_h": sum(err)/n,
        "bias_pct": 100*sum(err)/n/tbo,
        "near_new_n": len(nn),
        "near_new_bias_h": (sum(nn)/len(nn)) if nn else float("nan"),
        "near_new_bias_pct": (100*sum(nn)/len(nn)/tbo) if nn else float("nan"),
        "true_range_h": [min(r["true"] for r in rows), max(r["true"] for r in rows)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="Rotax_914_ULF")
    ap.add_argument("--model", action="append", required=True,
                    help="name=/path/to/<key>_rul.keras, repeatable")
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--throttle", type=float, default=0.65)
    ap.add_argument("--airspeed", type=float, default=45.0)
    ap.add_argument("--altitude", type=float, default=1500.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    key = KEY[a.engine]
    live = os.path.join(ROOT, "backend", "models_v3", key, f"{key}_rul.keras")
    keep = live + ".ab_backup"
    shutil.copy2(live, keep)
    results = {}
    try:
        for spec in a.model:
            name, path = spec.split("=", 1)
            print(f"\n=== {name}: {path}", flush=True)
            shutil.copy2(path, live)
            procs = start_services({"AERO_HEALTH_FAILURE_HOLD_S": "0"})
            try:
                rows = fly(a.engine, a.seconds, a.throttle, a.airspeed, a.altitude)
                s = score(rows)
                results[name] = {"score": s, "rows": rows}
                if s:
                    print(f"  {s['n']} samples, true RUL {s['true_range_h'][1]:.0f} -> "
                          f"{s['true_range_h'][0]:.0f} h", flush=True)
                    print(f"  MAE {s['mae_h']:.1f} h ({s['mae_pct']:.2f}% TBO) | "
                          f"bias {s['bias_h']:+.1f} h ({s['bias_pct']:+.2f}%) | "
                          f"near-new bias {s['near_new_bias_h']:+.1f} h "
                          f"({s['near_new_bias_pct']:+.2f}%, n={s['near_new_n']})", flush=True)
            finally:
                stop_services(procs)
    finally:
        shutil.copy2(keep, live)
        os.remove(keep)

    print("\nmodel      MAE h   % TBO    bias h   near-new bias")
    for name, r in results.items():
        s = r["score"]
        if s:
            print(f"{name:9s} {s['mae_h']:7.1f} {s['mae_pct']:7.2f}  {s['bias_h']:+8.1f}  "
                  f"{s['near_new_bias_h']:+8.1f} h ({s['near_new_bias_pct']:+.2f}%)")
    if a.out:
        json.dump({k: v["score"] for k, v in results.items()}, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
