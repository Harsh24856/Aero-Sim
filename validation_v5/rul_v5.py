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
in engine hours), a small MLP, and the GBT behind a wear-limited classifier (below
its threshold the calendar, TBO - hours, is returned: exactly right for an engine
that reaches TBO, where 914b's GBT still under-predicted by 5.1% of TBO against a
4% gate). The one furthest inside BOTH error gates on the VALIDATION flights ships:
max(wear-limited error / 12%, TBO-limited error / 4%).
Test flights are not touched here.

In service RUL may only fall within a flight (smooth_within_flight).
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

import evaluate as E  # noqa: E402
import features_v5 as F  # noqa: E402
import physics_v5 as P  # noqa: E402
from pipeline_v5 import CTX_COLS  # noqa: E402

MARGIN_CH = ["cht", "egt", "oil_temp", "oil_pressure", "engine_rpm"]


def measured_margin(contract: dict, seq: np.ndarray) -> np.ndarray:
    """Operating margin from the MEASURED instruments at each window's last second
    (seq scaled as the cache stores it)."""
    sc = contract["scaler"]
    last = seq[:, -1, :].astype(np.float64) * np.array(sc["std"]) + np.array(sc["mean"])
    eng = P.PistonEngineV5(contract["engine_model"], dt=1.0)
    idx = {c: F.FEATURE_COLS.index(c) for c in MARGIN_CH}
    return np.array([eng.margins({c: float(last[i, idx[c]]) for c in MARGIN_CH})["health_index"]
                     for i in range(len(seq))])


# Two-point fault extrapolation (v6). A fault grows as s = 1 - exp(-a x^b), x its age in
# % of TBO (degradation_v4._progress). With G(s) = (-ln(1-s))^(1/b) linear in age, two
# points - the logbook's worst severity `lag` hours ago and the severity now - give the
# time left until removal (s = REMOVAL_SEVERITY) without knowing a or the onset:
#     remaining = lag * (G(removal) - G(now)) / (G(now) - G(then)).
# Inputs are fault PROGRESS. rul_inputs converts the model's effective severities
# (progress x depth/spec depth) with the typical depth fraction, and reads a fault
# absent from the logbook (below HIST_SEV_FLOOR then) as the floor's midpoint.
# ponytail: b fixed at the fleet mean (1.5); per-fault b would need a third point.
TWO_POINT_B = 1.5
TWO_POINT_CAP = 2.0                  # fraction of TBO: no growth seen, no logbook, or no fault
_REMOVAL_SEV, _SEV_FLOOR, _DEPTH_FRAC = 0.8, 0.08, 0.775


def two_point_remaining(prog_now, prog_then, lag_h, tbo: float) -> np.ndarray:
    """Hours to removal / TBO, from the fault's progress now and `lag_h` hours ago."""
    def g(s):
        s = np.clip(np.asarray(s, np.float64), 0.0, 0.999)
        return (-np.log1p(-s)) ** (1.0 / TWO_POINT_B)
    now, then = np.asarray(prog_now, np.float64), np.asarray(prog_then, np.float64)
    lag = np.asarray(lag_h, np.float64)
    grow = g(now) - g(then)
    ok = (lag > 0.0) & (now >= _SEV_FLOOR) & (grow > 1e-6)
    rem = np.where(ok, lag * (g(_REMOVAL_SEV) - g(now)) / np.where(ok, grow, 1.0), 0.0) / float(tbo)
    return np.where(ok, np.clip(rem, 0.0, TWO_POINT_CAP), TWO_POINT_CAP)


def rul_inputs(o: dict, seq: np.ndarray, ctx: np.ndarray, contract: dict,
               hist: np.ndarray | None = None) -> np.ndarray:
    """RUL model inputs from the assembled outputs `o` and the scaled window/context
    - one definition for fitting, scoring and serving. When the contract has
    `hist_cols` (v6), the engine's logbook history [n, 3] in F.HIST_COLS order adds
    its wear ratio, past worst severity, severity progression per hour and the
    two-point estimate of the time left (two_point_remaining); missing history counts
    as none. Without hist_cols the inputs are exactly as before."""
    marg = measured_margin(contract, seq)
    thr_mean = seq[:, :, F.FEATURE_COLS.index("throttle")].mean(1)
    # Engine-fault evidence only: `detection` also fires on SENSOR faults, and fed to
    # RUL it took 7-14% off a healthy engine's life when one instrument failed.
    engine_fault = np.asarray(o["diagnosis"]).max(1, keepdims=True)
    X = np.concatenate([np.asarray(o["health"]), np.asarray(o["severity"]),
                        engine_fault, np.asarray(o["family"]),
                        marg[:, None], thr_mean[:, None], ctx], 1)
    if not contract.get("hist_cols"):
        return X
    h = np.tile(F.HIST_NEUTRAL, (len(X), 1)) if hist is None else np.asarray(hist, np.float64)
    lag, ratio, sev = h[:, 0], h[:, 1], h[:, 2]
    sev_now = np.asarray(o["severity"]).max(1)
    progression = (sev_now - sev) / np.maximum(lag, 1.0)
    then = np.where(sev <= 0.0, _SEV_FLOOR / 2, sev)              # absent from the logbook: below the floor
    two_point = two_point_remaining(sev_now / _DEPTH_FRAC, then / _DEPTH_FRAC, lag, contract["tbo_hours"])
    return np.concatenate([X, ratio[:, None], sev[:, None], progression[:, None], two_point[:, None]], 1)


def features(trainer, ids: np.ndarray, bs: int = 512):
    """Live-style RUL inputs and targets for the given window ends."""
    cache = trainer.cache
    Xs, ys, wl, cal, fl, tt = [], [], [], [], [], []
    tbo = cache.contract["tbo_hours"]
    ctx_hours = CTX_COLS.index("hours_frac")
    for i in range(0, len(ids), bs):
        seq, ctx, Er = cache.batch(ids[i:i + bs])
        o = trainer.model({"seq": seq, "ctx": ctx}, training=False)
        hist = (np.stack([cache.labels(Er, c) for c in F.HIST_COLS], 1)
                if cache.contract.get("hist_cols") else None)
        Xs.append(rul_inputs(o, seq, ctx, cache.contract, hist))
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


def fit_rul(trainer, out_dir: str | None = None, cal_ids: np.ndarray | None = None) -> dict:
    """cal_ids: the calibration windows to fit on (default: all of them)."""
    cache = trainer.cache
    tbo = cache.contract["tbo_hours"]
    if cal_ids is None:
        cal_ids = cache.end_ids(["cal"], jitter=False)
    Xc, yc, wlc, _, _, _, hc = features(trainer, cal_ids)
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
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0).fit(Xc, wlc)
    pwl = clf.predict_proba(Xv)[:, 1]
    # Candidates are judged on what is served: predictions smoothed within each flight.
    report = lambda pred: served_report(pred, yv, calv, tbo, wlv, flv, tv)  # noqa: E731
    score = lambda r: max(r["wear_limited_mae_pct"] / 12.0, r["tbo_limited_mae_pct"] / 4.0)  # noqa: E731
    thr = min(np.linspace(0.05, 0.95, 19),
              key=lambda t: score(report(GatedRUL(cands["gbt"], clf, t).predict(Xv, pwl))))
    cands["gated_gbt"] = GatedRUL(cands["gbt"], clf, float(thr))
    if cache.contract.get("hist_cols"):              # v6: a fault the logbook did not show
        for k in ("gbt", "gated_gbt"):
            cands[f"{k}+new_fault"] = NewFaultRUL(cands[k])
    for name, m in cands.items():
        res[name] = report(m.predict(Xv))
    res["gated_gbt"]["threshold"] = float(thr)
    best = min(res, key=lambda k: score(res[k]))
    out = {"candidates": res, "chosen": best}
    if out_dir:
        with open(os.path.join(out_dir, "rul_v5.pkl"), "wb") as fh:
            pickle.dump(cands[best], fh)
        with open(os.path.join(out_dir, "rul_v5.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    return out


class GatedRUL:
    """RUL from `reg` only where `clf` thinks the engine is wear-limited; elsewhere
    a value above any calendar, which the caller's min(pred, calendar) turns into
    TBO - hours."""

    def __init__(self, reg, clf, threshold: float):
        self.reg, self.clf, self.threshold = reg, clf, threshold

    def predict(self, X, p_wear_limited=None):
        p = self.clf.predict_proba(X)[:, 1] if p_wear_limited is None else p_wear_limited
        return np.where(p >= self.threshold, self.reg.predict(X), 1e3)


# rul_inputs column of the engine-fault evidence: after health (1) and severity (14).
ENGINE_FAULT_COL = 1 + 14
NEW_FAULT_EVIDENCE = 0.5


class NewFaultRUL:
    """`base`'s RUL, except for an engine fault that the logbook did not show: there
    the two-point extrapolation (rul_inputs' last column) answers. Held-out windows
    with a mid-severity fault absent from the logbook: model +158 h (MAE 161 h),
    two-point -34 h (MAE 96 h); where the logbook showed the fault the model is better,
    so it keeps those. Needs v6 inputs (hist_cols): columns -3 = logbook severity,
    -1 = two-point estimate. A diagnosed fault only, so a stray severity reading on a
    healthy engine never shortens its life."""

    def __init__(self, base):
        self.base = base

    def predict(self, X):
        X = np.asarray(X)
        new = ((X[:, -3] <= 0.0) & (X[:, -1] < TWO_POINT_CAP)
               & (X[:, ENGINE_FAULT_COL] >= NEW_FAULT_EVIDENCE))
        return np.where(new, X[:, -1], self.base.predict(X))


class RulSmoother:
    """smooth_within_flight one prediction at a time, for the live service. Same
    EMA on the removal time and the same non-increasing clamp; `update` with the
    same sequence returns exactly the array version's values."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.s = None          # smoothed removal time (engine hours)
        self.out = None        # last output (hours left)
        self.hours = None

    def update(self, pred_h: float, hours: float) -> float:
        removal = float(hours) + float(pred_h)
        self.s = removal if self.s is None else self.s + self.alpha * (removal - self.s)
        out = self.s - float(hours)
        if self.out is not None:
            out = min(out, self.out - max(0.0, float(hours) - self.hours))
        self.out, self.hours = out, float(hours)
        return max(out, 0.0)


def smooth_by_flight(pred_h: np.ndarray, flights: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """smooth_within_flight applied to each flight's windows in time order, whatever
    order they come in - the one definition for choosing, scoring and serving."""
    out = np.asarray(pred_h, float).copy()
    flights, hours = np.asarray(flights), np.asarray(hours, float)
    for f in np.unique(flights):
        g = np.where(flights == f)[0]
        g = g[np.argsort(hours[g], kind="stable")]
        out[g] = smooth_within_flight(out[g], hours[g])
    return out


def served_report(pred_frac: np.ndarray, true_frac: np.ndarray, calendar_h: np.ndarray, tbo: float,
                  wear_limited: np.ndarray, flights: np.ndarray, hours: np.ndarray) -> dict:
    """evaluate.rul_report on SERVED predictions: capped at the calendar, then smoothed
    within each flight, as score_v5 and the live service do."""
    raw = np.minimum(np.clip(np.asarray(pred_frac, float), 0, None) * tbo, calendar_h)
    return E.rul_report(np.asarray(true_frac) * tbo, smooth_by_flight(raw, flights, hours),
                        calendar_h, tbo, wear_limited, flights, hours)


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
