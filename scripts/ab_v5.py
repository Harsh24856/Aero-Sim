#!/usr/bin/env python3
"""Old-vs-new check for one engine's v5/v6 models: the same flights, flown by two
checkouts, side by side (plan: docs/superpowers/plans/2026-10-03-v6-weak-spots.md, Task 7).

    validation/venv/bin/python scripts/ab_v5.py --engine 914 --old origin/main --new .

--old / --new: a checkout directory, or a git ref (a temporary worktree is made for it).
Each side runs in its own process with ITS code and ITS backend/models_v5, through
e2e_v5's flight loop (twin_v5 + aiv5 + advisory). Flights: every applicable fault,
healthy engines at 300 / 1,000 / 1,550 h, and the sensor presets - same seed both sides.
Sensor presets are scored from onset + 128 s, once the window holds the fault.

Prints one table and the A/B part of the merge rule; exits non-zero when it fails:
all new flights pass; exact fault naming not lower; healthy false alarms not higher;
mean RUL error not higher.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"912": "Rotax_912_ULS", "914": "Rotax_914_ULF", "915": "Rotax_915_iS", "916": "Rotax_916_iS"}
WINDOW_S = 128


# --------------------------------------------------------------- one side (worker)
def fly_all(engine_key: str) -> list[dict]:
    """Runs inside a checkout's backend/ dir, with that checkout's modules."""
    import e2e_v5 as E
    from degradation_v5 import applicable_faults
    from observability_v5 import FAMILY_OF
    S, em = E.S, MODELS[engine_key]
    spec = E.aiv5.V.ENGINE_SPECS[em]
    for f in applicable_faults(spec.turbocharged, spec.intercooled):
        S.PRESETS.setdefault(f"x_{f}", (f, 1100.0, [(f, 0.35)], [], {}))
    S.PRESETS["healthy_1000h"] = ("healthy", 1000.0, [], [], {})
    S.PRESETS["healthy_1550h"] = ("healthy", 1550.0, [], [], {})
    rows = []
    for preset in [p for p in S.available(em) if p != "x_turbo_degradation"]:   # turbo_high flies it high
        r = E.fly(em, preset)
        ok, notes = E.judge(preset, r)
        smp = r["samples"]
        faults = [f for f, _ in S.PRESETS[preset][2]]
        sensors = S.PRESETS[preset][3]
        f0 = faults[0] if faults else None
        share = lambda pred: sum(1 for x in smp if pred(x)) / len(smp)  # noqa: E731
        row = {"preset": preset, "pass": bool(ok), "healthy": not faults and not sensors,
               "exact": share(lambda x: f0 in x[1]["faults_present"]) if f0 else None,
               "family": share(lambda x: FAMILY_OF[f0] in x[1]["families_present"]) if f0 else None,
               "any_call": share(lambda x: bool(x[1]["faults_present"])),
               "wrong_call": share(lambda x: bool(x[1]["faults_present"]) and f0 not in x[1]["faults_present"]),
               "rul_err": abs(smp[-1][1]["rul_hours"] - smp[-1][2]["rul_hours"]), "notes": notes}
        if sensors:
            ch, kind, _ = sensors[0]
            seen = [x for x in smp if x[0] > S.SENSOR_ONSET_S + WINDOW_S]
            row["sensor_kind"] = sum(1 for x in seen if x[1]["sensors"][ch]["condition"] == kind) / max(len(seen), 1)
        rows.append(row)
    return rows


# --------------------------------------------------------------- driver
def checkout(spec: str, keep: list) -> str:
    if os.path.isdir(spec):
        return os.path.abspath(spec)
    d = tempfile.mkdtemp(prefix="ab_v5_")
    subprocess.run(["git", "-C", ROOT, "worktree", "add", "-q", "--detach", d, spec], check=True)
    keep.append(d)
    return d


def run_side(root: str, engine_key: str) -> list[dict]:
    if not os.path.isdir(os.path.join(root, "backend", "models_v5")):
        raise SystemExit(f"{root}: no backend/models_v5")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", "--engine", engine_key,
                        "--root", root], cwd=os.path.join(root, "backend"), capture_output=True, text=True)
    rows = [json.loads(line[4:]) for line in p.stdout.splitlines() if line.startswith("ROW ")]
    if p.returncode or not rows:
        raise SystemExit(f"{root}: worker failed\n{p.stderr[-3000:]}")
    return rows


def compare(old: list[dict], new: list[dict]) -> tuple[bool, list[str]]:
    o, n = {r["preset"]: r for r in old}, {r["preset"]: r for r in new}
    common = [k for k in n if k in o]
    mean = lambda rows, k: sum(r[k] for r in rows if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in rows))  # noqa: E731
    fmt = lambda v: "  -  " if v is None else f"{v:4.0%}"  # noqa: E731
    lines = [f"{'flight':26s} {'pass o/n':9s} {'exact o->n':12s} {'wrong o->n':12s} {'calls o->n':12s} {'RUL err o->n':>14s}"]
    for k in common:
        a, b = o[k], n[k]
        lines.append(f"{k:26s} {('P' if a['pass'] else 'F') + '/' + ('P' if b['pass'] else 'F'):9s} "
                     f"{fmt(a['exact'])}->{fmt(b['exact'])}  {fmt(a['wrong_call'])}->{fmt(b['wrong_call'])}  "
                     f"{fmt(a['any_call'])}->{fmt(b['any_call'])} {a['rul_err']:6.0f}->{b['rul_err']:5.0f} h"
                     + (f"   sensor kind {a['sensor_kind']:.0%}->{b['sensor_kind']:.0%}" if "sensor_kind" in b else ""))
    O, N = [o[k] for k in common], [n[k] for k in common]
    hO, hN = [r for r in O if r["healthy"]], [r for r in N if r["healthy"]]
    checks = [
        ("all new flights pass", all(r["pass"] for r in N), f"{sum(r['pass'] for r in N)}/{len(N)}"),
        ("exact fault naming not lower", mean(N, "exact") >= mean(O, "exact") - 1e-9,
         f"{mean(O, 'exact'):.1%} -> {mean(N, 'exact'):.1%}"),
        ("healthy false alarms not higher", mean(hN, "any_call") <= mean(hO, "any_call") + 1e-9,
         f"{mean(hO, 'any_call'):.1%} -> {mean(hN, 'any_call'):.1%}"),
        ("mean RUL error not higher", mean(N, "rul_err") <= mean(O, "rul_err") + 1e-9,
         f"{mean(O, 'rul_err'):.0f} -> {mean(N, 'rul_err'):.0f} h"),
    ]
    lines.append("")
    lines += [f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}" for name, ok, detail in checks]
    ok = all(c[1] for c in checks)
    lines.append(f"A/B RULE: {'PASS' if ok else 'FAIL'}")
    return ok, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=sorted(MODELS))
    ap.add_argument("--old", help="checkout dir or git ref (the models to beat)")
    ap.add_argument("--new", default=".", help="checkout dir or git ref (default: this checkout)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--root", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:                                   # one side: this checkout's code and models
        sys.path[:0] = [os.path.join(a.root, "scripts"), os.path.join(a.root, "backend")]
        for r in fly_all(a.engine):
            print("ROW " + json.dumps(r), flush=True)
        return
    if not a.old:
        ap.error("--old is required")
    temp: list = []
    try:
        old_root, new_root = checkout(a.old, temp), checkout(a.new, temp)
        print(f"engine {a.engine}: old = {old_root}\n            new = {new_root}", flush=True)
        ok, lines = compare(run_side(old_root, a.engine), run_side(new_root, a.engine))
        print("\n".join(lines))
    finally:
        for d in temp:
            subprocess.run(["git", "-C", ROOT, "worktree", "remove", "--force", d], check=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
