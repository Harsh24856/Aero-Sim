#!/usr/bin/env python3
"""Remaining useful life as its own model (docs/v5_model_improvement_plan.md, Phase 4 task 5).

v4's RUL head learned from simulator labels (true wear, true severities, margin
from true values) and was then fed the other heads' outputs in service - 13% of
TBO on labels, 19-21% live. v5 trains RUL ONLY on inputs an aircraft has:

  - the joint model's own outputs (health, severities, detection, families),
    predicted on the CALIBRATION flights, which the joint model never trained on,
    so the errors RUL learns to live with are the errors it will meet in service;
  - the operating margin computed from MEASURED instruments;
  - hours, usage, life-clock flag and the long-horizon residual context.

Target: RUL as a fraction of TBO (rul_hours has no future faults, Phase 2).
Candidates: gradient-boosted trees (absolute-error loss, monotone non-increasing
in engine hours) and a small MLP; the better on the VALIDATION flights ships.
Test flights are not touched here.

In service RUL may only fall within a flight (smooth_within_flight).
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

import evaluate as E  # noqa: E402
import features_v5 as F  # noqa: E402
import physics_v5 as P  # noqa: E402
from pipeline_v5 import CTX_COLS, Cache  # noqa: E402

MARGIN_CH = ["cht", "egt", "oil_temp", "oil_pressure", "engine_rpm"]


def unscale_last(cache: Cache, seq: np.ndarray) -> dict:
    """Instrument readings (engineering units) at the window's last second."""
    sc = cache.contract["scaler"]
    mean, std = np.array(sc["mean"]), np.array(sc["std"])
    last = seq[:, -1, :].astype(np.float64) * std + mean
    return {c: last[:, F.FEATURE_COLS.index(c)] for c in MARGIN_CH}


def measured_margin(cache: Cache, seq: np.ndarray) -> np.ndarray:
    eng = P.PistonEngineV5(cache.contract["engine_model"], dt=1.0)
    v = unscale_last(cache, seq)
    return np.array([eng.margins({c: float(v[c][i]) for c in MARGIN_CH})["health_index"]
                     for i in range(len(seq))])


def features(trainer, ids: np.ndarray, bs: int = 512):
    """Live-style RUL inputs and targets for the given window ends."""
    cache = trainer.cache
    Xs, ys, wl, cal, fl, tt = [], [], [], [], [], []
    tbo = cache.contract["tbo_hours"]
    ctx_hours = CTX_COLS.index("hours_frac")
    for i in range(0, len(ids), bs):
        seq, ctx, Er = cache.batch(ids[i:i + bs])
        o = trainer.model({"seq": seq, "ctx": ctx}, training=False)
        marg = measured_margin(cache, seq)
        thr_mean = seq[:, :, F.FEATURE_COLS.index("throttle")].mean(1)
        X = np.concatenate([np.asarray(o["health"]), np.asarray(o["severity"]),
                            np.asarray(o["detection"]), np.asarray(o["family"]),
                            marg[:, None], thr_mean[:, None], ctx], 1)
        Xs.append(X)
        rul = cache.labels(Er, "rul_hours")
        calh = cache.labels(Er, "rul_calendar_hours")
        ys.append(rul / tbo)
        wl.append(rul < calh - 1e-3)
        cal.append(calh)
        fl.append(cache.flight_of(Er))
        tt.append(cache.labels(Er, "engine_hours"))
    hours_col = 1 + 14 + 1 + 6 + 1 + 1 + ctx_hours
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(wl), np.concatenate(cal),
            np.concatenate(fl), np.concatenate(tt), hours_col)


def fit_rul(trainer, out_dir: str | None = None) -> dict:
    cache = trainer.cache
    tbo = cache.contract["tbo_hours"]
    Xc, yc, _, _, _, _, hc = features(trainer, cache.end_ids(["cal"], jitter=False))
    Xv, yv, wlv, calv, flv, tv, _ = features(trainer, cache.end_ids(["val"], jitter=False))
    mono = np.zeros(Xc.shape[1], int)
    mono[hc] = -1                                   # more hours -> never more life
    cands = {
        "gbt": HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.05,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             monotonic_cst=mono, random_state=0),
        "mlp": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                            max_iter=400, early_stopping=True, random_state=0)),
    }
    res = {}
    for name, m in cands.items():
        m.fit(Xc, yc)
        pv = np.clip(m.predict(Xv), 0, None) * tbo
        pv = np.minimum(pv, calv)                   # never beyond the calendar
        res[name] = E.rul_report(yv * tbo, pv, calv, tbo, wlv, flv, tv)
    best = min(res, key=lambda k: res[k]["wear_limited_mae_pct"])
    out = {"candidates": res, "chosen": best}
    if out_dir:
        with open(os.path.join(out_dir, "rul_v5.pkl"), "wb") as fh:
            pickle.dump(cands[best], fh)
        with open(os.path.join(out_dir, "rul_v5.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    return out


def smooth_within_flight(pred_h: np.ndarray, hours: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Serving rule, one call per scored window (every 64 s): remaining life never
    rises within a flight and falls at least as fast as the engine ages.

    The predicted REMOVAL TIME (hours + RUL) is smoothed first (EMA), then clamped
    non-increasing. Clamping the raw predictions instead locks in every low outlier
    for the rest of the flight: on the 914 smoke run it took TBO-limited error from
    8% to 18% of TBO; on synthetic noise it doubles the error, this halves it."""
    hours = np.asarray(hours, float)
    removal = hours + np.asarray(pred_h, float)
    sm = np.empty_like(removal)
    s = removal[0] if len(removal) else 0.0
    for i, r in enumerate(removal):
        s = s + alpha * (r - s) if i else s
        sm[i] = s
    out = sm - hours
    for i in range(1, len(out)):
        out[i] = min(out[i], out[i - 1] - max(0.0, hours[i] - hours[i - 1]))
    return np.maximum(out, 0.0)
