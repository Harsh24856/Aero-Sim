#!/usr/bin/env python3
"""End-to-end check of the v4 chain on every engine and every preset (plan Phase 7, D1).

The same code the live stack runs - twin_v4 (engine + on-board twin + sensors),
aiv4's window, models, per-fault cut-offs and live RUL inputs, and advisory.py - flown
faster than real time, so all engines x presets finish in minutes instead of hours.
main.py's HTTP wiring is exercised separately against the running stack.

Each flight: 300 s climb, then cruise with a power change every 2 minutes (the turbo
preset climbs above the critical altitude, where that fault shows). The AI is scored
every 5 flight seconds once its 128 s window is full.

Criteria (docs/v4_integration_plan.md, Phase 7):
  healthy        no component fault called on more than 5% of scored samples
  fault preset   its fault named within 5 minutes of the window filling
  sensor preset  that sensor named with the right condition, no engine fault >5%
  wear-limited   when the truth is wear-limited, RUL below the calendar countdown

    cd backend && ../validation/venv/bin/python3 ../scripts/e2e_v4.py [--engines 914]
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import advisory                       # noqa: E402
import aiv4                           # noqa: E402  (loads the exported models)
import scenarios_v4 as S              # noqa: E402
from twin_v4 import UAVEngineTwinV4   # noqa: E402

FLIGHT_S = 128 + 300 + 60
SCORE_EVERY = 5


def fly(engine_model: str, preset: str) -> dict:
    rec = S.make_engine(engine_model, preset, seed=7)
    tw = UAVEngineTwinV4(engine_model=engine_model, engine=rec, seed=7)
    aiv4.active_engine = engine_model
    aiv4.window = aiv4.RollingWindowV4(aiv4.loaded_engines[engine_model]["scaler"])
    cruise = 6000.0 if preset == "turbo_high" else 1500.0
    samples, advisories = [], set()
    t_sample = 0
    while t_sample < FLIGHT_S:
        s = tw.t
        tw.altitude = min(300.0 + s * (cruise - 300.0) / 300.0, cruise)
        tw.airspeed = 48.0
        tw.throttle = 0.92 if s < 300 else (0.70 if (s // 120) % 2 else 0.85)
        out = tw.step()
        if not out["sample_new"]:
            continue
        t_sample += 1
        aiv4.window.update(out)
        if aiv4.window.is_ready() and t_sample % SCORE_EVERY == 0:
            ai = aiv4.run_inference()
            adv = advisory.build_advisory(ai, out)
            advisories.update(it["code"] for it in adv["items"])
            samples.append((t_sample, ai, out["truth"]))
    return {"samples": samples, "advisories": advisories, "record": rec}


def judge(preset: str, r: dict) -> tuple[bool, str]:
    samples = r["samples"]
    t0 = samples[0][0]
    within = [x for x in samples if x[0] <= t0 + 300]
    engine_calls = sum(1 for _, ai, _ in samples if ai["faults_present"]) / len(samples)
    truth = samples[-1][2]
    notes = [f"engine-fault calls {engine_calls:.0%}"]
    ok = True
    if preset == "healthy":
        ok &= engine_calls <= 0.05
    faults = [f for f, _ in S.PRESETS[preset][2]]
    for f in faults:
        hits = [t for t, ai, _ in within if f in ai["faults_present"]]
        notes.append(f"{f} named " + (f"after {hits[0] - t0} s" if hits else "NOT within 5 min"))
        ok &= bool(hits)
    for ch, kind, _ in S.PRESETS[preset][3]:
        after = [(t, ai) for t, ai, _ in samples if t > S.SENSOR_ONSET_S]
        hits = [t for t, ai in after if ai["sensors"][ch]["condition"] == kind]
        any_flag = [t for t, ai in after if ai["sensors"][ch]["condition"] != "none"]
        notes.append(f"{ch} {kind}: right kind {len(hits)}/{len(after)}, flagged at all {len(any_flag)}/{len(after)}")
        ok &= bool(hits) and engine_calls <= 0.05
    if truth["wear_limited"]:
        below = sum(1 for _, ai, _ in samples if ai["rul_hours"] < ai["rul_calendar_hours"] - 1) / len(samples)
        notes.append(f"wear-limited: RUL below calendar on {below:.0%}")
        ok &= below >= 0.5
    last = samples[-1][1]
    notes.append(f"RUL {last['rul_hours']:.0f} h (true {truth['rul_hours']:.0f}, calendar {last['rul_calendar_hours']:.0f})")
    return ok, "; ".join(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="912,914,915,916")
    args = ap.parse_args()
    names = {k: n for n, k in aiv4.ENGINE_REGISTRY.items()}
    failures = 0
    for key in args.engines.split(","):
        em = names[key.strip()]
        tag = " (PLACEHOLDER 914 models)" if aiv4.loaded_engines[em]["placeholder"] else ""
        print(f"=== {em}{tag}", flush=True)
        for preset in S.available(em):
            t = time.time()
            ok, notes = judge(preset, fly(em, preset))
            failures += 0 if ok or tag else 1
            print(f"  {'PASS' if ok else 'FAIL'} {preset:20s} {time.time() - t:5.1f}s  {notes}", flush=True)
    print("ALL PASS" if failures == 0 else f"{failures} failure(s) on engines with their own models")


if __name__ == "__main__":
    main()
