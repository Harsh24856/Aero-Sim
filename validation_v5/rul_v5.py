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


# W2: a fault's growth MEASURED across the inspection logbook. A fault grows as
# s = 1 - exp(-a x^b) (degradation_v4._progress, x = age in % of TBO), so
# G(s) = (-ln(1-s))^(1/b) is linear in age: least squares through the visible
# inspections and now gives the rate, and the time left to removal (s = 0.8) needs
# neither a nor the onset. Severities are effective (progress x depth/spec depth): the
# typical depth fraction converts them. One logbook entry could only GUESS the rate when
# the fault was invisible then (v6: +155 h on such test windows).
# ponytail: b and depth fixed at the fleet means. Depth is nearly all of the error (median
# 32% -> 4% if it were known), but fitting it from the line's straightness overfits the
# logbook's 10% noise (57%, biased +56%); fixed it stays unbiased (-3%).
TREND_B, TREND_DEPTH, TREND_CAP = 1.5, 0.775, 2.0     # TREND_CAP: fraction of TBO, "not measured"
_REMOVAL_SEV, _SEV_FLOOR = 0.8, 0.08


def logbook_trend(sev_now: np.ndarray, logbook: np.ndarray, tbo: float) -> np.ndarray:
    """Time to removal / TBO from the severity now and the logbook [n, len(LOG_LAGS_H)];
    TREND_CAP where growth is not measured (no fault now, no visible past entry, no growth)."""
    def g(progress):
        p = np.clip(np.asarray(progress, np.float64), 0.0, 0.999)
        return (-np.log1p(-p)) ** (1.0 / TREND_B)
    def g_eff(s):                                    # effective severity -> progress -> G
        return g(np.asarray(s, np.float64) / TREND_DEPTH)
    now = np.asarray(sev_now, np.float64)
    log = np.asarray(logbook, np.float64)
    t = -np.asarray(F.LOG_LAGS_H, np.float64)[None, :].repeat(len(now), 0)
    seen = log > 0.0
    # points: the visible inspections and now (t = 0)
    T = np.concatenate([np.where(seen, t, np.nan), np.zeros((len(now), 1))], 1)
    Y = np.concatenate([np.where(seen, g_eff(log), np.nan), g_eff(now)[:, None]], 1)
    n = np.sum(~np.isnan(T), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm, ym = np.nanmean(T, 1), np.nanmean(Y, 1)
        slope = np.nansum((T - tm[:, None]) * (Y - ym[:, None]), 1) / np.nansum((T - tm[:, None]) ** 2, 1)
    ok = (now >= _SEV_FLOOR) & (n >= 2) & (slope > 1e-6)
    rem = np.where(ok, (g(_REMOVAL_SEV) - g_eff(now)) / np.where(ok, slope, 1.0), 0.0) / float(tbo)
    return np.where(ok, np.clip(rem, 0.0, TREND_CAP), TREND_CAP)


def rul_inputs(o: dict, seq: np.ndarray, ctx: np.ndarray, contract: dict,
               hist: np.ndarray | None = None, logbook: np.ndarray | None = None) -> np.ndarray:
    """RUL model inputs from the assembled outputs `o` and the scaled window/context
    - one definition for fitting, scoring and serving. When the contract has
    `hist_cols` (v6), the engine's logbook history [n, 3] in F.HIST_COLS order adds
    its wear ratio, past worst severity and severity progression per hour; missing
    history counts as none. With `log_cols` (W2) the inspection logbook [n, 4] adds its
    four severities and logbook_trend. Without them the inputs are exactly as before."""
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
    progression = (np.asarray(o["severity"]).max(1) - sev) / np.maximum(lag, 1.0)
    X = np.concatenate([X, ratio[:, None], sev[:, None], progression[:, None]], 1)
    if not contract.get("log_cols"):
        return X
    log = (np.tile(F.LOG_NEUTRAL, (len(X), 1)) if logbook is None else np.asarray(logbook, np.float64))
    trend = logbook_trend(np.asarray(o["severity"]).max(1), log, contract["tbo_hours"])
    return np.concatenate([X, log, trend[:, None]], 1)


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
        log = (np.stack([cache.labels(Er, c) for c in F.LOG_COLS], 1)
               if cache.contract.get("log_cols") else None)
        Xs.append(rul_inputs(o, seq, ctx, cache.contract, hist, log))
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
