#!/usr/bin/env python3
"""Gap report for one engine's trained specialists - measure before fixing.

    validation/venv/bin/python validation_v5/gaps_v5.py 916b

Per applicable fault: validation positives, validation recall / precision at the
stored cut-off, and test recall at that cut-off (test is read for the report only,
never to choose anything). Severity: the on-fault bias split into cells the
severity gate zeroed and cells it kept. Oil: how often oil-pump is named as oil
degradation and the reverse. Writes artifacts/<engine>/gaps.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

from evaluate import PRESENT_SEV  # noqa: E402  (numpy + sklearn only: no TensorFlow on import)


def bias_split(true: np.ndarray, pred_ungated: np.ndarray, pred_gated: np.ndarray) -> dict:
    """On-fault severity bias of `pred_gated`, and how much of it comes from fault
    cells the gate zeroed (gated 0, ungated > 0)."""
    true, ung, gat = (np.asarray(a, float) for a in (true, pred_ungated, pred_gated))
    fault = true >= PRESENT_SEV
    d = (gat - true)[fault]
    out = (gat[fault] == 0) & (ung[fault] > 0)
    n, n_out = int(fault.sum()), int(out.sum())
    bias = float(d.mean()) if n else 0.0
    out_bias = float(d[out].mean()) if n_out else 0.0
    kept_bias = float(d[~out].mean()) if n - n_out else 0.0
    share = n_out * out_bias / (n * bias) if n and bias != 0 else 0.0
    return {"bias": bias, "n": n, "gated_out_n": n_out, "gated_out_bias": out_bias,
            "kept_bias": kept_bias, "share_from_gated_out": float(share)}


def confusion_share(y: np.ndarray, pred: np.ndarray, a: int, b: int) -> float:
    """Of windows where fault `a` is true, the share predicted `b` and not `a`."""
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    on = y[:, a]
    return float((pred[on, b] & ~pred[on, a]).mean()) if on.any() else 0.0


def miss_visibility(y_old: np.ndarray, pred: np.ndarray, y_new: np.ndarray) -> dict:
    """Of the windows the model misses under the current labels (y_old 1, pred 0), how
    many a stricter labelling (y_new) calls not visible at all."""
    y_old, pred, y_new = (np.asarray(a).astype(bool) for a in (y_old, pred, y_new))
    miss = y_old & ~pred
    n, inv = int(miss.sum()), int((miss & ~y_new).sum())
    return {"misses": n, "invisible": inv, "share_invisible": inv / n if n else 0.0}


def check_same_windows(a: np.ndarray, b: np.ndarray) -> None:
    """Two caches compared window by window must hold the same window ends."""
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise SystemExit("the two caches hold different windows - relabel from the same source cache")


def fault_report(engine: str, fault: str, vs_key: str) -> dict:
    """miss_visibility for one fault on val and test: labels of cache/<engine> (what the
    model was trained and scored on) against cache/<vs_key> (e.g. relabel_v5b
    --per-channel-faults --out <vs_key>)."""
    from degradation_v5 import FAULT_NAMES
    from pipeline_v5 import Cache
    from specialists_v5 import Specialists, paths
    p = paths(engine)
    sp = Specialists(Cache(p["cache"]), p["art"])
    new = Cache(os.path.join(HERE, "cache", vs_key))
    cal = json.load(open(os.path.join(p["art"], "calibration.json")))
    j = FAULT_NAMES.index(fault)
    out = {"engine": engine, "fault": fault, "vs": vs_key}
    for split in ("val", "test"):
        ids = sp.cache.end_ids([split], jitter=False)
        check_same_windows(np.asarray(sp.cache.ends[ids])[:, :2], np.asarray(new.ends[ids])[:, :2])
        s = _split_outputs(sp, ids, cal["temperature"], cal["cutoffs"])
        y_new = new.labels(np.asarray(new.ends[ids]), f"fmv_{fault}")
        out[split] = miss_visibility(s["y_diag"][:, j], s["pred"][:, j], y_new)
    return out


def _split_outputs(sp, ids: np.ndarray, temps, cuts, bs: int = 512) -> dict:
    from assembly_v5 import apply_temperature, calibrate_severity, gate_severity
    from train_v5 import targets
    x = sp.extra
    acc = {"y_diag": [], "pred": [], "y_sev": [], "sev_ung": [], "sev_gat": []}
    for i in range(0, len(ids), bs):
        seq, ctx, Er = sp.cache.batch(ids[i:i + bs])
        o = sp.raw(seq, ctx)
        y = targets(sp.cache, Er)
        sev = calibrate_severity(o["severity"], x["sev_iso"]) if x.get("sev_iso") else o["severity"]
        gated = gate_severity(sev, o["diagnosis"], x["sev_gate"]) if x.get("sev_gate") is not None else sev
        acc["y_diag"].append(y["diagnosis"])
        acc["pred"].append(apply_temperature(o["diagnosis"], temps) >= np.asarray(cuts)[None, :])
        acc["y_sev"].append(y["severity"])
        acc["sev_ung"].append(sev)
        acc["sev_gat"].append(gated)
    return {k: np.concatenate(v) for k, v in acc.items()}


def report(engine: str) -> dict:
    import evaluate as E
    from degradation_v5 import FAULT_NAMES
    from pipeline_v5 import Cache
    from specialists_v5 import Specialists, paths
    p = paths(engine)                          # cache/<engine>, artifacts/<engine>_specialists
    art = p["art"]
    sp = Specialists(Cache(p["cache"]), art)
    cal = json.load(open(os.path.join(art, "calibration.json")))
    temps, cuts = cal["temperature"], cal["cutoffs"]
    app = [FAULT_NAMES.index(n) for n in sp.applicable]
    out = {"engine": engine, "faults": {}, "severity": {}, "oil": {}}
    s = {sp_: _split_outputs(sp, sp.cache.end_ids([sp_], jitter=False), temps, cuts) for sp_ in ("val", "test")}
    for j in app:
        row = {"cutoff": cuts[j]}
        for sp_ in ("val", "test"):
            y, p = s[sp_]["y_diag"][:, j], s[sp_]["pred"][:, j]
            if y.sum() == 0:
                row[sp_] = None                    # no positives: n/a, not recall 0
                continue
            prec, rec, _ = E.prf(y, p)
            row[sp_] = {"positives": int(y.sum()), "recall": float(rec), "precision": float(prec)}
        out["faults"][FAULT_NAMES[j]] = row
    for sp_ in ("val", "test"):
        out["severity"][sp_] = bias_split(s[sp_]["y_sev"][:, app], s[sp_]["sev_ung"][:, app], s[sp_]["sev_gat"][:, app])
    if {"oil_pump_degradation", "oil_degradation"} <= set(sp.applicable):
        a, b = FAULT_NAMES.index("oil_pump_degradation"), FAULT_NAMES.index("oil_degradation")
        for sp_ in ("val", "test"):
            out["oil"][sp_] = {"pump_named_deg": confusion_share(s[sp_]["y_diag"], s[sp_]["pred"], a, b),
                               "deg_named_pump": confusion_share(s[sp_]["y_diag"], s[sp_]["pred"], b, a)}
    with open(os.path.join(art, "gaps.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def _fmt(r):
    return "n/a" if r is None else f"{r['positives']:6d} pos  recall {r['recall']:.2f}  prec {r['precision']:.2f}"


if __name__ == "__main__":
    if "--fault" in sys.argv:
        a = sys.argv
        r = fault_report(a[1], a[a.index("--fault") + 1], a[a.index("--vs") + 1])
        for split in ("val", "test"):
            m = r[split]
            print(f"== {r['engine']} {r['fault']} {split}: {m['misses']} missed windows, "
                  f"{m['invisible']} invisible by {r['vs']} labels ({m['share_invisible']:.0%})")
        sys.exit(0)
    r = report(sys.argv[1])
    print(f"== {r['engine']}")
    for f, row in r["faults"].items():
        print(f"  {f:24s} cut {row['cutoff']:.2f} | val {_fmt(row['val'])} | test {_fmt(row['test'])}")
    for sp_, b in r["severity"].items():
        print(f"  severity {sp_:4s} bias {b['bias']:+.3f} (n {b['n']}); gated-out {b['gated_out_n']} cells "
              f"bias {b['gated_out_bias']:+.3f}; kept bias {b['kept_bias']:+.3f}; share from gate {b['share_from_gated_out']:.2f}")
    for sp_, o in r["oil"].items():
        print(f"  oil {sp_:4s} pump->deg {o['pump_named_deg']:.2f}  deg->pump {o['deg_named_pump']:.2f}")
