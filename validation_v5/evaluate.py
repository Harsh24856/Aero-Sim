"""v5 evaluation harness (docs/v5_model_improvement_plan.md, section 2 and Phase 0).

Framework-free: every function takes NumPy arrays of predictions and labels, one
entry per scored window, plus the FLIGHT id of each window. Confidence intervals
resample FLIGHTS, not windows - windows of one flight share an engine, a fault
and a weather, so resampling windows would make every interval far too narrow.

Heads and their honest metrics
  detection   AUC; recall at precision >= 0.95
  diagnosis   per-fault precision / recall / F1 at given cut-offs, macro F1 over
              faults present in the set, family-level F1, expected calibration error
  severity    MAE on fault cells (true >= 0.08), MAE on clean cells (true == 0),
              signed bias on fault cells
  sensor      macro F1 over the 7 conditions, recall per kind, false-alarm rate
              per healthy channel-window
  health      MAE
  rul         MAE as % of TBO on wear-limited and TBO-limited windows, gain over
              the calendar countdown, within-flight rises

`gate(...)` applies gates_v5.TARGETS: a gate passes only when the point estimate
meets the target AND the flight-bootstrap CI of (model - baseline) excludes zero
in the model's favour.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

SENSOR_KINDS = ["none", "bias", "drift", "stuck", "spike", "noise", "dropout"]
PRESENT_SEV = 0.08


# ---------------------------------------------------------------------------
# Bootstrap over flights
# ---------------------------------------------------------------------------
def flight_groups(flights: np.ndarray) -> list[np.ndarray]:
    flights = np.asarray(flights)
    order = np.argsort(flights, kind="stable")
    uniq, starts = np.unique(flights[order], return_index=True)
    return np.split(order, starts[1:])


def bootstrap(metric, flights: np.ndarray, *arrays, n: int = 1000, seed: int = 0,
              ci: float = 0.95) -> dict:
    """Point estimate and flight-bootstrap CI of metric(*arrays).

    metric: f(*arrays_subset) -> float. Draws that make the metric undefined
    (e.g. no positives) are skipped."""
    groups = flight_groups(flights)
    point = float(metric(*arrays))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        try:
            v = float(metric(*(a[idx] for a in arrays)))
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    lo, hi = (np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2]) if vals else (np.nan, np.nan))
    return {"value": point, "lo": float(lo), "hi": float(hi), "draws": len(vals)}


def bootstrap_diff(metric, flights, model_arrays: tuple, base_arrays: tuple,
                   n: int = 1000, seed: int = 0) -> dict:
    """CI of metric(model) - metric(baseline), resampling the same flights for both."""
    groups = flight_groups(flights)
    rng = np.random.default_rng(seed)
    point = float(metric(*model_arrays)) - float(metric(*base_arrays))
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        try:
            v = float(metric(*(a[idx] for a in model_arrays))) - float(metric(*(a[idx] for a in base_arrays)))
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    lo, hi = (np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan))
    return {"value": point, "lo": float(lo), "hi": float(hi), "draws": len(vals)}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def auc(y, p) -> float:
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        raise ValueError("one class")
    return float(roc_auc_score(y, p))


def recall_at_precision(y, p, precision: float = 0.95) -> float:
    """Highest recall achievable with precision >= `precision`."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0:
        raise ValueError("no positives")
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / ys.sum()
    ok = prec >= precision
    return float(rec[ok].max()) if ok.any() else 0.0


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
def prf(y, pred) -> tuple[float, float, float]:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp = float((y & pred).sum())
    fp = float((~y & pred).sum())
    fn = float((y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def diagnosis_report(y: np.ndarray, prob: np.ndarray, names: list, cuts=None) -> dict:
    """y, prob: [N, F]. cuts: per-fault cut-offs (default 0.5)."""
    cuts = np.full(len(names), 0.5) if cuts is None else np.asarray(cuts)
    per = {}
    for j, n in enumerate(names):
        if y[:, j].sum() == 0:
            continue
        p, r, f = prf(y[:, j], prob[:, j] >= cuts[j])
        per[n] = {"precision": p, "recall": r, "f1": f, "positives": int(y[:, j].sum()), "cut": float(cuts[j])}
    return {"macro_f1": float(np.mean([v["f1"] for v in per.values()])) if per else float("nan"),
            "per_fault": per}


def macro_f1(y: np.ndarray, prob: np.ndarray, cuts=None) -> float:
    cuts = np.full(y.shape[1], 0.5) if cuts is None else np.asarray(cuts)
    fs = [prf(y[:, j], prob[:, j] >= cuts[j])[2] for j in range(y.shape[1]) if y[:, j].sum()]
    if not fs:
        raise ValueError("no positives")
    return float(np.mean(fs))


def macro_average_precision(y: np.ndarray, prob: np.ndarray) -> float:
    """Threshold-free: mean over faults (with positives) of average precision."""
    from sklearn.metrics import average_precision_score
    aps = [average_precision_score(y[:, j], prob[:, j]) for j in range(y.shape[1]) if y[:, j].sum()]
    if not aps:
        raise ValueError("no positives")
    return float(np.mean(aps))


def family_f1(y: np.ndarray, prob: np.ndarray, names: list, family_of: dict, cuts=None) -> float:
    """Macro F1 at family level: a family is called when any member is called."""
    cuts = np.full(len(names), 0.5) if cuts is None else np.asarray(cuts)
    fams = sorted({family_of[n] for n in names})
    fs = []
    for fam in fams:
        cols = [j for j, n in enumerate(names) if family_of[n] == fam]
        yt = y[:, cols].max(1)
        if yt.sum() == 0:
            continue
        pred = (prob[:, cols] >= cuts[cols]).any(1)
        fs.append(prf(yt, pred)[2])
    return float(np.mean(fs))


def ece(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, averaged over faults (columns)."""
    y, prob = np.atleast_2d(y.T).T, np.atleast_2d(prob.T).T
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for j in range(y.shape[1]):
        e, n = 0.0, len(y)
        b = np.clip(np.digitize(prob[:, j], edges) - 1, 0, bins - 1)
        for k in range(bins):
            m = b == k
            if m.any():
                e += m.sum() / n * abs(prob[m, j].mean() - y[m, j].mean())
        out.append(e)
    return float(np.mean(out))


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------
def severity_report(true: np.ndarray, pred: np.ndarray) -> dict:
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    fault = true >= PRESENT_SEV
    clean = true <= 1e-6
    d = pred - true
    return {"on_fault_mae": float(np.abs(d[fault]).mean()) if fault.any() else float("nan"),
            "clean_mae": float(np.abs(d[clean]).mean()) if clean.any() else float("nan"),
            "on_fault_bias": float(d[fault].mean()) if fault.any() else float("nan"),
            "all_cells_mae": float(np.abs(d).mean())}


def on_fault_mae(true, pred) -> float:
    fault = np.asarray(true) >= PRESENT_SEV
    if not fault.any():
        raise ValueError("no fault cells")
    return float(np.abs(np.asarray(pred)[fault] - np.asarray(true)[fault]).mean())


# ---------------------------------------------------------------------------
# Sensor fault
# ---------------------------------------------------------------------------
def sensor_report(true: np.ndarray, pred: np.ndarray) -> dict:
    """true, pred: int condition index per channel-window (any shape)."""
    t, p = np.asarray(true).ravel(), np.asarray(pred).ravel()
    per, fs = {}, []
    for k, name in enumerate(SENSOR_KINDS):
        if (t == k).sum() == 0:
            continue
        pr, rc, f = prf(t == k, p == k)
        per[name] = {"precision": pr, "recall": rc, "f1": f, "support": int((t == k).sum())}
        fs.append(f)
    healthy = t == 0
    return {"macro_f1": float(np.mean(fs)), "per_kind": per,
            "false_alarm_rate": float((p[healthy] != 0).mean()) if healthy.any() else float("nan"),
            "detected_faulty": float((p[t != 0] != 0).mean()) if (t != 0).any() else float("nan")}


def sensor_macro_f1(true, pred) -> float:
    t, p = np.asarray(true).ravel(), np.asarray(pred).ravel()
    fs = [prf(t == k, p == k)[2] for k in range(len(SENSOR_KINDS)) if (t == k).any()]
    return float(np.mean(fs))


# ---------------------------------------------------------------------------
# Health and RUL
# ---------------------------------------------------------------------------
def mae(true, pred) -> float:
    return float(np.abs(np.asarray(pred, float) - np.asarray(true, float)).mean())


def rul_report(true_h, pred_h, calendar_h, tbo: float, wear_limited: np.ndarray,
               flights: np.ndarray | None = None, t: np.ndarray | None = None) -> dict:
    true_h, pred_h = np.asarray(true_h, float), np.asarray(pred_h, float)
    cal = np.asarray(calendar_h, float)
    wl = np.asarray(wear_limited, bool)
    rep = {
        "wear_limited_mae_pct": 100 * mae(true_h[wl], pred_h[wl]) / tbo if wl.any() else float("nan"),
        "tbo_limited_mae_pct": 100 * mae(true_h[~wl], pred_h[~wl]) / tbo if (~wl).any() else float("nan"),
        "calendar_wear_limited_mae_pct": 100 * mae(true_h[wl], cal[wl]) / tbo if wl.any() else float("nan"),
        "all_mae_pct": 100 * mae(true_h, pred_h) / tbo,
    }
    rep["gain_vs_calendar"] = (1 - rep["wear_limited_mae_pct"] / rep["calendar_wear_limited_mae_pct"]
                               if wl.any() else float("nan"))
    if flights is not None and t is not None:
        rises = 0
        for g in flight_groups(flights):
            seq = pred_h[g[np.argsort(np.asarray(t)[g])]]
            rises += int((np.diff(seq) > 0.02 * tbo).sum())
        rep["within_flight_rises_gt_2pct"] = rises
    return rep


def rul_wl_mae_pct(true_h, pred_h, wl, tbo) -> float:
    wl = np.asarray(wl, bool)
    if not wl.any():
        raise ValueError("no wear-limited windows")
    return 100.0 * mae(np.asarray(true_h)[wl], np.asarray(pred_h)[wl]) / tbo


# ---------------------------------------------------------------------------
# Gate-ready entries (the shape gates_v5.check reads)
# ---------------------------------------------------------------------------
def entry(metric, flights, model_arrays: tuple, base_arrays: tuple | None = None,
          n: int = 1000, seed: int = 0) -> dict:
    """{'point', 'ci', 'baseline': {'point'}, 'diff': {'ci'}} for one metric."""
    b = bootstrap(metric, np.asarray(flights), *model_arrays, n=n, seed=seed)
    out = {"point": b["value"], "ci": [b["lo"], b["hi"]]}
    if base_arrays is not None:
        out["baseline"] = {"point": float(metric(*base_arrays))}
        d = bootstrap_diff(metric, np.asarray(flights), model_arrays, base_arrays, n=n, seed=seed)
        out["diff"] = {"point": d["value"], "ci": [d["lo"], d["hi"]]}
    return out


def point(value: float) -> dict:
    """A point-only entry, for secondary targets."""
    return {"point": float(value), "ci": [None, None]}
