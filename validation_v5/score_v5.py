"""The v5 test scorecard (docs/v5_model_improvement_plan.md, section 2).

Scores a trained engine on its TEST flights - once, after every choice (epoch,
cut-offs, temperatures, RUL model) was made on train / calibration / validation.
Every primary metric carries a 95% flight-bootstrap CI and the CI of its
difference from a do-nothing baseline; gates_v5.check turns that into pass/fail.

Baselines (section 2)
  detection   the largest residual in the window, in sigma          (a threshold rule)
  diagnosis   class prior: every fault at its training prevalence   (calls nothing)
  severity    per-fault training mean damage where a fault is present, 0 elsewhere
  sensor      hand rules: flat line -> stuck, reading at the floor -> dropout,
              a jump far above the channel's typical step -> spike,
              step noise far above the channel's spread -> noise, else none
  health      training-set mean wear condition at the same engine age (decile)
  rul         the calendar countdown (TBO - hours)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

import evaluate as E  # noqa: E402
import gates_v5 as G  # noqa: E402
import model_architectures_v5 as M  # noqa: E402
import rul_v5 as R  # noqa: E402
from degradation_v5 import FAULT_NAMES  # noqa: E402
from observability_v5 import FAMILY_OF  # noqa: E402
from pipeline_v5 import RES_SLICE  # noqa: E402
from train_v5 import apply_temperature, targets  # noqa: E402

N_BOOT = 1000


def _baseline_features(cache, ids: np.ndarray, bs: int = 1024) -> dict:
    """Window statistics the do-nothing baselines use (scaled units)."""
    res_max, stuck, drop, spike, noise = [], [], [], [], []
    for i in range(0, len(ids), bs):
        seq, _, _ = cache.batch(ids[i:i + bs])
        res_max.append(np.abs(seq[:, :, RES_SLICE]).max(axis=(1, 2)))
        m = seq[:, :, M.MEAS_IDX]                                  # [B, T, 12]
        d = np.abs(np.diff(m, axis=1))
        typical = np.median(d, axis=1) + 1e-3
        stuck.append((d < 1e-4).mean(axis=1) > 0.95)
        drop.append(m.min(axis=1) < np.median(m, axis=1) - 4.0)
        spike.append(d.max(axis=1) / typical > 25.0)
        noise.append(d.std(axis=1) / (m.std(axis=1) + 1e-3) > 1.2)
    cat = np.concatenate
    return {"res_max": cat(res_max), "stuck": cat(stuck), "drop": cat(drop), "spike": cat(spike),
            "noise": cat(noise)}


def sensor_rules(b: dict) -> np.ndarray:
    """Hand rules, later rules win: noise < spike < dropout < stuck."""
    k = E.SENSOR_KINDS.index
    pred = np.zeros(b["stuck"].shape, int)
    pred[b["noise"]] = k("noise")
    pred[b["spike"]] = k("spike")
    pred[b["drop"]] = k("dropout")
    pred[b["stuck"]] = k("stuck")
    return pred


def _train_labels(cache, n: int = 40000) -> tuple[dict, np.ndarray]:
    """Targets and engine hours of training windows, read from the end rows only
    (no sequences), for the baselines that need training statistics."""
    ids = np.sort(cache.end_ids(["train"], jitter=False)[:n])
    Er = np.asarray(cache.ends[ids])
    return targets(cache, Er), cache.labels(Er, "engine_hours")


def score(trainer, calibration: dict, rul_model=None, n_boot: int = N_BOOT) -> dict:
    """Test scorecard: every head with CI, baseline, and the gate verdict."""
    cache = trainer.cache
    tbo = cache.contract["tbo_hours"]
    test = cache.end_ids(["test"], jitter=False)
    o, y, Er = trainer.predict(test)
    fl = cache.flight_of(Er)
    base = _baseline_features(cache, test)
    ytr, hours_tr = _train_labels(cache)
    hours = cache.labels(Er, "engine_hours")
    app = [FAULT_NAMES.index(n) for n in trainer.applicable]
    names = [FAULT_NAMES[j] for j in app]
    cuts = np.asarray(calibration["cutoffs"])[app]
    pdiag = apply_temperature(o["diagnosis"], calibration["temperature"])[:, app]
    ydiag = y["diagnosis"][:, app]
    ent = lambda metric, model, baseline: E.entry(metric, fl, model, baseline, n=n_boot)  # noqa: E731
    heads = {}

    # -- detection -------------------------------------------------------------
    yd, pd_ = y["detection"][:, 0], o["detection"][:, 0]
    heads["detection"] = {
        "recall_at_p95": ent(E.recall_at_precision, (yd, pd_), (yd, base["res_max"])),
        "auc": E.point(E.auc(yd, pd_)),
        "baseline_auc": E.point(E.auc(yd, base["res_max"])),
    }

    # -- diagnosis -------------------------------------------------------------
    prior = np.tile(ytr["diagnosis"][:, app].mean(0), (len(ydiag), 1))
    rep = E.diagnosis_report(ydiag, pdiag, names, cuts)
    obs = _observable_faults(cache.contract["engine_model"])
    recalls = [v["recall"] for k, v in rep["per_fault"].items() if k in obs]
    heads["diagnosis"] = {
        "macro_f1": ent(lambda a, b: E.macro_f1(a, b, cuts), (ydiag, pdiag), (ydiag, prior)),
        "min_fault_recall": E.point(min(recalls) if recalls else float("nan")),
        "family_macro_f1": E.point(E.family_f1(ydiag, pdiag, names, FAMILY_OF, cuts)),
        "ece": E.point(E.ece(ydiag, pdiag)),
        "macro_ap": E.point(E.macro_average_precision(ydiag, pdiag)),
        "per_fault": rep["per_fault"],
    }

    # -- severity --------------------------------------------------------------
    ts, ps = y["severity"][:, app], o["severity"][:, app]
    tr_sev = ytr["severity"][:, app]
    on = tr_sev >= E.PRESENT_SEV
    mean_present = np.array([tr_sev[on[:, j], j].mean() if on[:, j].any() else 0.0 for j in range(len(app))])
    bs_sev = np.where(ts >= E.PRESENT_SEV, mean_present[None, :], 0.0)
    sr = E.severity_report(ts, ps)
    heads["severity"] = {
        "mae_on_fault": ent(E.on_fault_mae, (ts, ps), (ts, bs_sev)),
        "mae_clean": E.point(sr["clean_mae"]),
        "bias_on_fault": E.point(sr["on_fault_bias"]),
    }

    # -- sensor fault ----------------------------------------------------------
    ysf, psf = y["sensor"], o["sensor"].argmax(-1)
    srep = E.sensor_report(ysf, psf)
    heads["sensor_fault"] = {
        "macro_f1": ent(E.sensor_macro_f1, (ysf, psf), (ysf, sensor_rules(base))),
        "per_kind": {k: {"recall": E.point(v["recall"])} for k, v in srep["per_kind"].items()},
        "false_alarm_rate": E.point(srep["false_alarm_rate"]),
        "detected_faulty": E.point(srep["detected_faulty"]),
    }

    # -- health: baseline = training mean condition at the same engine age -------
    yh, ph = y["health"][:, 0], o["health"][:, 0]
    edges = np.quantile(hours_tr, np.linspace(0, 1, 11))[1:-1]
    b_tr, b_te = np.digitize(hours_tr, edges), np.digitize(hours, edges)
    h_tr = ytr["health"][:, 0]
    by_bin = np.array([h_tr[b_tr == k].mean() if (b_tr == k).any() else h_tr.mean() for k in range(10)])
    heads["health"] = {"mae": ent(E.mae, (yh, ph), (yh, by_bin[b_te]))}

    # -- RUL -------------------------------------------------------------------
    if rul_model is not None:
        X, yr, wl, cal, flr, hrs, _ = R.features(trainer, test)
        raw = np.minimum(np.clip(rul_model.predict(X), 0, None) * tbo, cal)
        sm = R.smooth_by_flight(raw, flr, hrs)
        rises = 0
        for g in E.flight_groups(flr):
            g = g[np.argsort(hrs[g], kind="stable")]
            rises += bool((np.diff(sm[g]) > 0.02 * tbo).any())
        true_h = yr * tbo
        rr = E.rul_report(true_h, sm, cal, tbo, wl)
        wl_pct = lambda a, b, w: E.rul_wl_mae_pct(a, b, w, tbo)  # noqa: E731
        heads["rul"] = {
            "mae_pct_tbo_wear_limited": E.entry(wl_pct, flr, (true_h, sm, wl), (true_h, cal, wl), n=n_boot),
            "gain_vs_calendar": E.point(rr["gain_vs_calendar"]),
            "mae_pct_tbo_tbo_limited": E.point(rr["tbo_limited_mae_pct"]),
            "frac_flights_nonmonotone": E.point(rises / max(len(np.unique(flr)), 1)),
            "before_smoothing": E.rul_report(true_h, raw, cal, tbo, wl, flr, hrs),
        }
    return {"engine_model": cache.contract["engine_model"], "test_windows": int(len(test)),
            "test_flights": int(len(np.unique(fl))), "heads": heads, "gates": G.check(heads)}


def _observable_faults(engine_model: str) -> set:
    """Faults Phase 1 found observable in at least one regime (the recall floor
    applies to these only)."""
    with open(os.path.join(HERE, "observability_v5.json")) as fh:
        summ = json.load(fh)["engines"][engine_model]["summary"]
    return {f for f, s in summ.items() if s["observable_regimes"] > 0}


def model_card(res: dict) -> str:
    """The scorecard as a short Markdown model card."""
    h, g = res["heads"], res["gates"]

    def fmt(e, pct=False):
        if e is None or e.get("point") is None:
            return "-"
        v = e["point"]
        s = f"{v:.3f}" if not pct else f"{v:.1f}%"
        if e.get("ci") and e["ci"][0] is not None:
            lo, hi = e["ci"]
            s += f" ({lo:.3f}-{hi:.3f})" if not pct else f" ({lo:.1f}-{hi:.1f}%)"
        if "baseline" in e:
            b = e["baseline"]["point"]
            s += f"; baseline {b:.3f}" if not pct else f"; baseline {b:.1f}%"
        return s

    rows = [
        ("Detection: recall at precision 0.95", fmt(h["detection"]["recall_at_p95"]), g.get("detection", {})),
        ("Diagnosis: macro F1", fmt(h["diagnosis"]["macro_f1"]), g.get("diagnosis", {})),
        ("Severity: error on real faults", fmt(h["severity"]["mae_on_fault"]), g.get("severity", {})),
        ("Sensor fault: macro F1", fmt(h["sensor_fault"]["macro_f1"]), g.get("sensor_fault", {})),
        ("Health: error", fmt(h["health"]["mae"]), g.get("health", {})),
    ]
    if "rul" in h:
        rows.append(("RUL: error on wear-limited engines, % of TBO",
                     fmt(h["rul"]["mae_pct_tbo_wear_limited"], pct=True), g.get("rul", {})))
    lines = [f"# {res['engine_model'].replace('_', ' ')} - v5 test scorecard", "",
             f"{res['test_flights']} test flights, {res['test_windows']:,} windows. "
             "95% flight-bootstrap intervals in brackets.", "",
             "| Head | Result | Gate | Why not |", "|---|---|---|---|"]
    for name, val, gv in rows:
        lines.append(f"| {name} | {val} | {'PASS' if gv.get('pass') else 'FAIL'} | "
                     f"{'; '.join(gv.get('reasons', [])) or '-'} |")
    lines += ["", "## Diagnosis by fault", "", "| Fault | Recall | Precision | F1 | Cut-off |", "|---|---|---|---|---|"]
    for f, v in h["diagnosis"]["per_fault"].items():
        lines.append(f"| {f} | {v['recall']:.2f} | {v['precision']:.2f} | {v['f1']:.2f} | {v['cut']:.2f} |")
    lines += ["", "## Sensor faults by kind (recall)", "",
              " | ".join(f"{k} {v['recall']['point']:.2f}" for k, v in h["sensor_fault"]["per_kind"].items()),
              f"; false alarms {100 * h['sensor_fault']['false_alarm_rate']['point']:.2f}% of healthy channel-windows."]
    return "\n".join(lines)
