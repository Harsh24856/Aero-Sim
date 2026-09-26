#!/usr/bin/env python3
"""Pilot learnability check (docs/v5_model_improvement_plan.md, Phase 2 task 9).

Before generating the full v5 set: can simple models learn the v5 labels, and
are they MORE learnable than the v4-style labels computed on the same windows?
Gradient-boosted trees (sklearn HistGradientBoosting) on window summaries plus
the long-horizon context, trained on the pilot's train+cal flights and scored on
its val+test flights. Each head is scored twice, v5 label vs v4-style label:

  detection  fault_present (effect visible)        vs fault_present_sev (severity >= 0.08)
  diagnosis  fmv_<fault> (present and visible)      vs fm_<fault> >= 0.08
  sensor     sf_<ch>_flag (visible)                 vs sf_<ch>_active (from onset)
  rul        rul_hours (faults already under way)   vs rul_hours_oracle (includes future faults)

    validation/venv/bin/python validation_v5/pilot_learnability.py data/rotax_v5_pilot/914
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

import evaluate as E  # noqa: E402
import features_v5 as F  # noqa: E402
from data_checks_v5 import load  # noqa: E402
from degradation_v5 import FAULT_NAMES, applicable_faults  # noqa: E402
from observability_v5 import FAMILY_OF  # noqa: E402
from sensors_v5 import FAULTABLE_CHANNELS  # noqa: E402

WINDOW, STRIDE = F.WINDOW, 64


def windows(df: pd.DataFrame, idx: pd.DataFrame):
    """Summary features and end-of-window labels for every window of every flight."""
    feats = F.FEATURE_COLS
    X, Xs, meta, lab = [], [], [], []
    res_cols = F.RESIDUAL_COLS
    for sid, g in df.groupby("scenario_id", sort=False):
        g = g.sort_values("t")
        A = g[feats].to_numpy(np.float32)
        lh = F.long_horizon_array(g[res_cols].to_numpy())
        n = len(g)
        for end in range(WINDOW, n + 1, STRIDE):
            w = A[end - WINDOW:end]
            summ = np.concatenate([w.mean(0), w.std(0), w.min(0), w.max(0), w[-32:].mean(0) - w[:32].mean(0)])
            X.append(np.concatenate([summ, lh[end - 1], [g.engine_hours.iloc[end - 1]]]))
            r = g.iloc[end - 1]
            meta.append((sid, r.t))
            lab.append(r)
            # per-channel sensor features: reading + residual stats + that channel's long horizon
            chan = []
            for ci, c in enumerate(FAULTABLE_CHANNELS):
                m = w[:, feats.index(c)]
                rr = w[:, feats.index(f"res_{c}")]
                d = np.abs(np.diff(m))
                run = np.max(np.diff(np.flatnonzero(np.r_[True, d > 0, True]))) if len(d) else 0
                chan.append(np.concatenate([
                    [ci], [m.mean(), m.std(), m.min(), m.max(), d.max(), d.mean(), run,
                           (m == m.min()).mean(), m[-32:].mean() - m[:32].mean(),
                           rr.mean(), rr.std(), rr.min(), rr.max(), rr[-32:].mean() - rr[:32].mean()],
                    lh[end - 1][ci * 3:(ci + 1) * 3]]))
            Xs.append(np.stack(chan))
    return np.array(X), np.array(Xs), pd.DataFrame(lab).reset_index(drop=True), np.array(meta)


def clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          class_weight="balanced", random_state=0)


def main(dirpath: str) -> None:
    t0 = time.time()
    idx, df, man = load(dirpath)
    tbo = man["tbo_hours"]
    split = idx.set_index("scenario_id").split
    X, Xs, L, meta = windows(df, idx)
    fl = meta[:, 0].astype(int)
    tr = np.isin(split.reindex(fl).values, ["train", "cal"])
    te = ~tr
    print(f"{len(X):,} windows ({tr.sum():,} train / {te.sum():,} eval) from {len(idx)} flights "
          f"in {time.time() - t0:.0f}s")
    report = {"windows": int(len(X)), "flights": int(len(idx))}

    # -- detection ------------------------------------------------------------
    det = {}
    for name, col in (("v5 visible", "fault_present"), ("v4 severity", "fault_present_sev")):
        y = L[col].to_numpy().astype(int)
        p = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        det[name] = {"auc": E.auc(y[te], p), "recall_at_p95": E.recall_at_precision(y[te], p),
                     "positive_rate": float(y[te].mean())}
    report["detection"] = det

    # -- diagnosis ------------------------------------------------------------
    spec_turbo = "turbo" in man.get("engine_model", "").lower()
    names = [n for n in FAULT_NAMES if L[f"fm_{n}"].max() > 0]
    diag = {}
    for label, make in (("v5 visible", lambda n: L[f"fmv_{n}"]), ("v4 severity", lambda n: L[f"fm_{n}"] >= 0.08)):
        Y = np.stack([make(n).to_numpy().astype(int) for n in names], 1)
        P = np.zeros((te.sum(), len(names)))
        for j in range(len(names)):
            if Y[tr, j].sum() < 5:
                continue
            P[:, j] = clf().fit(X[tr], Y[tr, j]).predict_proba(X[te])[:, 1]
        rep = E.diagnosis_report(Y[te], P, names)
        diag[label] = {"macro_f1": rep["macro_f1"],
                       "family_f1": E.family_f1(Y[te], P, names, FAMILY_OF),
                       "per_fault_f1": {k: round(v["f1"], 3) for k, v in rep["per_fault"].items()}}
    report["diagnosis"] = diag

    # -- sensor kind (one classifier shared by all channels) -------------------
    sens = {}
    Xc = Xs.reshape(-1, Xs.shape[-1])
    trc = np.repeat(tr, len(FAULTABLE_CHANNELS))
    tec = ~trc
    for label, suffix in (("v5 visible", "flag"), ("v4 onset", "active")):
        Yc = np.stack([L[f"sf_{c}_{suffix}"].to_numpy().astype(int) for c in FAULTABLE_CHANNELS], 1).reshape(-1)
        pred = clf().fit(Xc[trc], Yc[trc]).predict(Xc[tec])
        sens[label] = E.sensor_report(Yc[tec], pred)
        sens[label]["per_kind"] = {k: round(v["recall"], 3) for k, v in sens[label]["per_kind"].items()}
    report["sensor"] = sens

    # -- RUL --------------------------------------------------------------------
    rul = {}
    cal = L.rul_calendar_hours.to_numpy()
    for label, col in (("v5 known faults", "rul_hours"), ("v4 oracle", "rul_hours_oracle")):
        y = L[col].to_numpy()
        reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=0, loss="absolute_error")
        p = np.clip(reg.fit(X[tr], y[tr]).predict(X[te]), 0, tbo)
        wl = y[te] < cal[te] - 1e-3
        rul[label] = E.rul_report(y[te], p, cal[te], tbo, wl)
    report["rul"] = rul

    report["elapsed_s"] = round(time.time() - t0, 1)
    out = os.path.join(HERE, "pilot_learnability.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps(report, indent=2, default=lambda v: round(float(v), 4)))


if __name__ == "__main__":
    main(sys.argv[1])
