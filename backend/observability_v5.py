"""Which faults can the v5 instruments see, where, and can they tell each fault
from its family siblings? (docs/v5_model_improvement_plan.md, Phase 1 task 7)

For every engine, every fault its hardware can have, at severity 0.5, over a grid
of regimes (altitude x power x ISA), the faulty engine and a healthy one are run
to equilibrium on the same inputs. Each instrument's shift is expressed in units
of that instrument's noise (sensors_v5.SENSOR_SPEC noise_sd): the residual the
healthy on-board twin would show, in sigma.

  observable   some instrument shifts >= 2 sigma
  separable    its sigma-signature differs from every family sibling's by
               >= 2 sigma on at least one instrument (same regime)

The table drives the per-fault recall floors (a fault is only held to a floor in
the regimes where it is observable), the family-level answers in the UI, and it
is a realism check: shifts are printed so an implausible magnitude stands out.

    backend/.venv/bin/python backend/observability_v5.py        # ~2 min
"""
from __future__ import annotations

import json
import os

import numpy as np

import physics_v5 as P
from degradation_v4 import FAULT_MODES, applicable_faults
from sensors_v5 import SENSOR_CHANNELS, SENSOR_SPEC, TURBO_CHANNELS

SEVERITY = 0.5
DEPTH_FRAC = 0.775          # mid of the generator's per-fault depth draw (0.55-1.0)
ALTITUDES = (500.0, 3000.0, 6000.0)
THROTTLES = (0.35, 0.70, 1.00)
ISA = (0.0, 20.0)
AIRSPEED = 45.0
SETTLE = 900
Z_OBS = 2.0

FAMILIES = {
    "induction": ["air_filter_fouling", "turbo_degradation", "wastegate_fault", "intercooler_fouling"],
    "cylinder": ["compression_loss", "valve_leakage"],
    "fuel_ignition": ["injector_fouling", "ignition_degradation", "combustion_instability"],
    "oil": ["bearing_wear", "oil_pump_degradation", "oil_degradation"],
    "cooling": ["cooling_degradation"],
    "propeller": ["prop_erosion"],
}
FAMILY_OF = {f: fam for fam, fs in FAMILIES.items() for f in fs}


def health_for(fault: str, severity: float = SEVERITY) -> P.Health:
    h = P.Health()
    spec = FAULT_MODES[fault]
    delta = spec["depth"] * DEPTH_FRAC * severity
    for mod in spec["mods"]:
        cur = getattr(h, mod)
        setattr(h, mod, cur + delta if spec["sense"] < 0 else cur - delta)
    return h


def settle(eng: str, u: P.Inputs, h: P.Health) -> dict:
    e = P.PistonEngineV5(eng, dt=1.0)
    e.warm_start(u, h)
    for _ in range(SETTLE):
        o = e.step(u, h)
    return o


def signature(eng: str, u: P.Inputs, base: dict, h: P.Health, turbo: bool) -> tuple[np.ndarray, dict]:
    o = settle(eng, u, h)
    z, raw = [], {}
    for c in SENSOR_CHANNELS:
        if c in TURBO_CHANNELS and not turbo:
            z.append(0.0)
            continue
        d = float(o[c]) - float(base[c])
        raw[c] = round(d, 4)
        z.append(d / SENSOR_SPEC[c]["noise_sd"])
    return np.array(z), raw


def run() -> dict:
    out = {"severity": SEVERITY, "z_observable": Z_OBS, "families": FAMILIES, "engines": {}}
    for eng, spec in P.ENGINE_SPECS.items():
        faults = applicable_faults(spec.turbocharged, spec.intercooled)
        rows = {f: [] for f in faults}
        for alt in ALTITUDES:
            for thr in THROTTLES:
                for isa in ISA:
                    u = P.Inputs(altitude_m=alt, airspeed_ms=AIRSPEED, throttle=thr, isa_dev_c=isa)
                    base = settle(eng, u, P.Health())
                    sigs = {f: signature(eng, u, base, health_for(f), spec.turbocharged) for f in faults}
                    for f in faults:
                        z, raw = sigs[f]
                        top = np.argsort(-np.abs(z))[:3]
                        sibs = [g for g in FAMILIES[FAMILY_OF[f]] if g in sigs and g != f]
                        sep = all(np.max(np.abs(z - sigs[g][0])) >= Z_OBS for g in sibs)
                        rows[f].append({
                            "alt_m": alt, "throttle": thr, "isa_dev_c": isa,
                            "rpm": round(base["engine_rpm"]),
                            "observable": bool(np.max(np.abs(z)) >= Z_OBS),
                            "separable_in_family": bool(sep) if sibs else None,
                            "max_z": round(float(np.max(np.abs(z))), 1),
                            "top_channels": [f"{SENSOR_CHANNELS[i]} {z[i]:+.1f}s" for i in top if abs(z[i]) >= 1.0],
                            "shift": raw,
                        })
        summary = {}
        for f, rs in rows.items():
            n = len(rs)
            summary[f] = {
                "family": FAMILY_OF[f],
                "observable_regimes": sum(r["observable"] for r in rs), "regimes": n,
                "separable_regimes": (sum(bool(r["separable_in_family"]) for r in rs)
                                      if rs and rs[0]["separable_in_family"] is not None else None),
                "unobservable_in": [f"{r['alt_m']:.0f}m thr{r['throttle']} isa{r['isa_dev_c']:+.0f}"
                                    for r in rs if not r["observable"]],
            }
        out["engines"][eng] = {"summary": summary, "regimes": rows}
    return out


def to_markdown(res: dict) -> str:
    lines = ["# Fault observability, physics v5", "",
             f"Severity {res['severity']}, observable = some instrument shifts >= {res['z_observable']} sigma of its noise; "
             "separable = differs from every family sibling by >= 2 sigma on some instrument. "
             f"{len(ALTITUDES) * len(THROTTLES) * len(ISA)} regimes per engine "
             f"(altitude {ALTITUDES} m x throttle {THROTTLES} x ISA {ISA}).", ""]
    for eng, e in res["engines"].items():
        lines += [f"## {eng}", "", "| Fault | Family | Observable | Separable in family | Strongest signs (cruise, 3000 m ISA) | Unobservable in |",
                  "|---|---|---|---|---|---|"]
        for f, s in e["summary"].items():
            cruise = next(r for r in e["regimes"][f] if r["alt_m"] == 3000.0 and r["throttle"] == 0.70 and r["isa_dev_c"] == 0.0)
            unobs = ", ".join(s["unobservable_in"]) if s["unobservable_in"] else "-"
            sep = "- (no sibling on this engine)" if s["separable_regimes"] is None else f"{s['separable_regimes']}/{s['regimes']}"
            lines.append(f"| {f} | {s['family']} | {s['observable_regimes']}/{s['regimes']} | {sep} | "
                         f"{', '.join(cruise['top_channels']) or '-'} | {unobs} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    res = run()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "validation_v5")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "observability_v5.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    md = to_markdown(res)
    with open(os.path.join(out_dir, "observability_v5.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
