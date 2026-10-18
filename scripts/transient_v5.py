#!/usr/bin/env python3
"""False alarms on a healthy engine after a throttle step (plan 2026-10-18 P1b).

The training missions ease the throttle; a pilot steps it. Each case flies a healthy
engine for 300 s, then steps throttle (and sometimes altitude) at once and flies 150 s
more, scoring every second through aiv5 - the raw models, before main.py's hold. It
reports the largest detection after the step and how many seconds it sat at or above
0.5. The P2 gate is 5 s or less per case; --out writes the numbers for comparison.

    cd backend && ../validation/venv/bin/python3 ../scripts/transient_v5.py [--engines 916] [--out f.json]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import aiv5                           # noqa: E402  (loads the exported models)
import scenarios_v4 as S              # noqa: E402
import twin_v5                        # noqa: E402
from twin_v5 import UAVEngineTwinV5   # noqa: E402

PAYLOAD = twin_v5.AI_PAYLOAD + ["time"]
BEFORE_S, AFTER_S = 300, 150
# name: (altitude m, throttle, airspeed m/s) before -> after the step
CASES = {
    "chop 0.75>0.15 @1500":        ((1500, 0.75, 45), (1500, 0.15, 35)),
    "chop full>0.15 @6000":        ((6000, 1.00, 45), (6000, 0.15, 35)),
    "chop full>0.15 6000>1000 m":  ((6000, 1.00, 45), (1000, 0.15, 35)),
    "advance 0.15>full @1500":     ((1500, 0.15, 35), (1500, 1.00, 45)),
    "small 0.75>0.50 @1500":       ((1500, 0.75, 45), (1500, 0.50, 45)),
}
GATE_S = 5


def fly(engine_model: str, before: tuple, after: tuple) -> dict:
    rec = S.make_engine(engine_model, "healthy", seed=7)
    tw = UAVEngineTwinV5(engine_model=engine_model, engine=rec, seed=7)
    aiv5.select_engine({"engine_model": engine_model})
    n, post = 0, []
    while n < BEFORE_S + AFTER_S:
        tw.altitude, tw.throttle, tw.airspeed = before if n < BEFORE_S else after
        out = tw.step()
        if not out["sample_new"]:
            continue
        n += 1
        ai = aiv5.step({k: out[k] for k in PAYLOAD})
        if ai["status"] == "ok" and n > BEFORE_S:
            post.append(ai["detection_confidence"])
    high = [i + 1 for i, d in enumerate(post) if d >= 0.5]
    return {"max_det": round(max(post), 3), "alarm_s": len(high), "last_alarm_s": high[-1] if high else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="916,915,914,912")
    ap.add_argument("--out", default=None, help="write the results as JSON")
    args = ap.parse_args()
    names = {k: n for n, k in aiv5.ENGINE_REGISTRY.items()}
    results = {}
    for key in args.engines.split(","):
        key = key.strip()
        results[key] = {}
        for case, (before, after) in CASES.items():
            r = fly(names[key], before, after)
            results[key][case] = r
            print(f"{key} {case:28s} {'PASS' if r['alarm_s'] <= GATE_S else 'FAIL'}  max det {r['max_det']:.2f}, "
                  f">=0.5 for {r['alarm_s']:3d} s (last +{r['last_alarm_s']} s)", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
