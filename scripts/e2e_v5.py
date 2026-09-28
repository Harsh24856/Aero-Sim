#!/usr/bin/env python3
"""End-to-end check of the v5 chain on every engine and every preset (plan.md Phase 3).

scripts/e2e_v4.py for v5: twin_v5 (engine + on-board twin + 14 instruments +
long-horizon context), aiv5's window, the exported specialists, the assembly, the
RUL smoother and advisory.py - flown faster than real time. Same flights and the
same criteria as v4, so the two reports compare line for line (plan Phase 6):

  healthy        no component fault called on more than 5% of scored samples
  fault preset   its fault - or its FAMILY (v5 answers at family level when unsure) -
                 named within 5 minutes of the window filling
  sensor preset  that sensor named with the right condition, no engine fault >5%
  wear-limited   when the truth is wear-limited, RUL below the calendar countdown
Also reports the mean inference time per second of flight.

    cd backend && ../validation/venv/bin/python3 ../scripts/e2e_v5.py [--engines 914]
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import advisory                       # noqa: E402
import aiv5                           # noqa: E402  (loads the exported models)
import scenarios_v4 as S              # noqa: E402
from observability_v5 import FAMILY_OF  # noqa: E402
from twin_v5 import UAVEngineTwinV5   # noqa: E402

FLIGHT_S = 128 + 300 + 60
SCORE_EVERY = 5
PAYLOAD = aiv5.F.FEATURE_COLS + ["engine_hours", "life_used_hours", "ai_context", "time"]


def fly(engine_model: str, preset: str) -> dict:
    rec = S.make_engine(engine_model, preset, seed=7)
    tw = UAVEngineTwinV5(engine_model=engine_model, engine=rec, seed=7)
    aiv5.select_engine({"engine_model": engine_model})
    cruise = 6000.0 if preset == "turbo_high" else 1500.0
    samples, advisories, infer_s = [], set(), []
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
        payload = {k: out[k] for k in PAYLOAD}
        if t_sample % SCORE_EVERY == 0 or not aiv5.flight.ready():
            t0 = time.perf_counter()
            ai = aiv5.step(payload)
            if ai["status"] == "ok":
                infer_s.append(time.perf_counter() - t0)
                adv = advisory.build_advisory(ai, out)
                advisories.update(it["code"] for it in adv["items"])
                samples.append((t_sample, ai, out["truth"]))
        else:
            aiv5.flight.update(payload)          # keep the window and context moving, unscored
    return {"samples": samples, "advisories": advisories, "record": rec, "infer_s": infer_s}


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
    for f, _ in S.PRESETS[preset][2]:
        hits = [t for t, ai, _ in within if f in ai["faults_present"]]
        fam = [t for t, ai, _ in within if FAMILY_OF[f] in ai["families_present"]]
        notes.append(f"{f} named " + (f"after {hits[0] - t0} s" if hits else "NOT within 5 min")
                     + ("" if hits else (f", family '{FAMILY_OF[f]}' after {fam[0] - t0} s" if fam else ", family not either")))
        ok &= bool(hits or fam)
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
    notes.append(f"{1000 * sum(r['infer_s']) / max(len(r['infer_s']), 1):.0f} ms/inference")
    return ok, "; ".join(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="912,914,915,916")
    args = ap.parse_args()
    names = {k: n for n, k in aiv5.ENGINE_REGISTRY.items()}
    failures = 0
    for key in args.engines.split(","):
        em = names[key.strip()]
        tag = " (PLACEHOLDER 914 models)" if aiv5.loaded_engines[em]["placeholder"] else ""
        print(f"=== {em}{tag}", flush=True)
        for preset in S.available(em):
            t = time.time()
            ok, notes = judge(preset, fly(em, preset))
            failures += 0 if ok or tag else 1
            print(f"  {'PASS' if ok else 'FAIL'} {preset:20s} {time.time() - t:5.1f}s  {notes}", flush=True)
    print("ALL PASS" if failures == 0 else f"{failures} failure(s) on engines with their own models")


if __name__ == "__main__":
    main()
