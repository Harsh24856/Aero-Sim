#!/usr/bin/env python3
"""Finalise one engine's v4 models and export them for backend/aiv4.py.

Three steps (docs/v4_integration_plan.md, Phase 2):

  A2  PER-FAULT CUT-OFFS. The diagnosis head is trained with positives weighted 12x,
      so a flat 0.5 over-calls some faults (ignition, valve, oil on the 914). Each
      fault type gets the cut-off with the best F1 on the VALIDATION flights; test
      is used only to report before/after.

  A3  RUL ON INPUTS AN AIRCRAFT HAS. The RUL head was trained with three of its ten
      aux inputs taken from simulator labels: the true wear condition, the true fault
      severities, and the operating margin from true (not measured) values. Live, the
      first two can only come from the health and severity heads and the third from
      the sensors. This re-scores the trained head that way, on the test flights,
      and that is the RUL number the manifest and the model cards carry.

  A5  EXPORT backend/models_v4/<engine>/: the six phase checkpoints, both scalers and
      manifest.json (input contract, cut-offs, test metrics, RUL error band by life
      stage). aiv4.py refuses to start if its own contract disagrees with it.

Runs on the GPU. These weights were trained on Apple Metal and give very different
answers on the CPU (diagnosis probabilities differ by up to 1.0), so the cut-offs and
RUL numbers must be measured where the model will be served. --cpu is for debugging.

    ../validation/venv/bin/python3 export_deployable_v4.py --engine 914
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np                                  # noqa: E402

PHASES = ["detection", "diagnosis", "severity", "sensor_fault", "health", "rul"]
HEAD_OF = {"detection": "y_detection", "diagnosis": "y_fault_mode", "severity": "y_fault_mode",
           "sensor_fault": "y_sensor_fault", "health": "y_health", "rul": "y_rul_hours"}
MARGIN_CHANNELS = ["egt", "cht", "oil_temp", "oil_pressure", "engine_rpm"]
LIFE_BANDS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 10.0)]   # engine_hours / TBO


def engine_model_of(engine: str) -> str:
    import physics_v4 as V
    return next(k for k in V.ENGINE_SPECS_V4 if k.split("_")[1] == engine)


def live_aux(aux_raw: np.ndarray, health_pred: np.ndarray, sev_pred: np.ndarray,
             margin_meas: np.ndarray, P) -> np.ndarray:
    """The RUL aux vector as aiv4.py builds it: the three label-derived entries
    replaced by model outputs and measured values. Everything else is unchanged."""
    a = aux_raw.copy()
    k = {n: i for i, n in enumerate(P.RUL_AUX_ORDER)}
    a[:, k["health_index"]] = health_pred
    a[:, k["margin_min"]] = margin_meas
    a[:, k["mean_fault_severity"]] = sev_pred.mean(axis=1)
    a[:, k["max_fault_severity"]] = sev_pred.max(axis=1)
    return a


def collect(engine: str, split: str, models: dict, C, P, applicable_mask, margins_fn):
    """Every head's prediction on one split, with the truths and the RUL aux inputs
    (raw) for the same windows."""
    import joblib
    ax = joblib.load(C.aux_scaler_path(engine))
    sc = joblib.load(C.scaler_path(engine))
    idx = [P.FEATURE_COLS.index(c) for c in MARGIN_CHANNELS]
    out = {k: [] for k in ["det", "diag", "sev", "sensor", "health", "rul_label_inputs",
                           "t_det", "t_fm", "t_sensor", "t_health", "t_rul", "aux_raw", "margin_meas"]}
    for xb, yb in C.datasets(engine)[split]:
        x = xb["x"]
        out["det"].append(models["detection"].predict_on_batch(xb)["y_detection"].ravel())
        out["diag"].append(models["diagnosis"].predict_on_batch(xb)["y_fault_mode"])
        out["sev"].append(models["severity"].predict_on_batch(xb)["y_fault_mode"] * applicable_mask)
        out["sensor"].append(models["sensor_fault"].predict_on_batch(xb)["y_sensor_fault"])
        out["health"].append(models["health"].predict_on_batch(xb)["y_health"].ravel())
        out["rul_label_inputs"].append(models["rul"].predict_on_batch(xb)["y_rul_hours"].ravel())
        out["t_det"].append(yb["y_detection"].numpy().ravel())
        out["t_fm"].append(yb["y_fault_mode"].numpy())
        out["t_sensor"].append(yb["y_sensor_fault"].numpy())
        out["t_health"].append(yb["y_health"].numpy().ravel())
        out["t_rul"].append(yb["y_rul_hours"].numpy().ravel())
        out["aux_raw"].append(ax.inverse_transform(xb["x_rul_aux"].numpy().astype(np.float64)))
        last = sc.inverse_transform(x.numpy()[:, -1, :].astype(np.float64))[:, idx]
        out["margin_meas"].append(np.array([margins_fn(dict(zip(MARGIN_CHANNELS, r))) for r in last]))
    return {k: np.concatenate(v) for k, v in out.items()}


def best_cutoffs(prob, truth, applicable, fault_modes, thr=0.08):
    """Per-fault cut-off with the best F1, grid 0.05..0.95."""
    cuts = {}
    t = truth >= thr
    for i, name in enumerate(fault_modes):
        if name not in applicable or not t[:, i].any():
            cuts[name] = 0.5
            continue
        best = (-1.0, 0.5)
        for c in np.round(np.arange(0.05, 0.951, 0.05), 2):
            p = prob[:, i] >= c
            tp = (p & t[:, i]).sum(); fp = (p & ~t[:, i]).sum(); fn = (~p & t[:, i]).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best[0]:
                best = (f1, float(c))
        cuts[name] = best[1]
    return cuts


def scores(prob, truth, cuts, applicable, fault_modes, thr=0.08):
    t = truth >= thr
    rows = {}
    for i, name in enumerate(fault_modes):
        if name not in applicable or not t[:, i].any():
            continue
        p = prob[:, i] >= cuts[name]
        tp = (p & t[:, i]).sum(); fp = (p & ~t[:, i]).sum(); fn = (~p & t[:, i]).sum()
        rows[name] = {"f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                      "recall": float(tp / max(tp + fn, 1)),
                      "precision": float(tp / max(tp + fp, 1))}
    return {"macro_f1": float(np.mean([r["f1"] for r in rows.values()])), "per_fault": rows}


def rul_scores(pred, truth, hours_norm, tbo):
    cal = tbo * np.clip(1.0 - hours_norm, 0.0, None)
    worn = truth < cal - 1.0
    err = np.abs(pred - truth)
    bands = []
    for lo, hi in LIFE_BANDS:
        m = (hours_norm >= lo) & (hours_norm < hi)
        if m.any():
            bands.append({"life_from": lo, "life_to": min(hi, 1.0), "mae_hours": float(err[m].mean()),
                          "n": int(m.sum())})
    return {"mae_hours": float(err.mean()),
            "mae_pct_tbo": float(err.mean() * 100.0 / tbo),
            "mae_pct_tbo_wear_limited": float(err[worn].mean() * 100.0 / tbo) if worn.any() else None,
            "wear_limited_mae_hours": float(err[worn].mean()) if worn.any() else None,
            "calendar_wear_limited_mae_hours": float(np.abs(cal - truth)[worn].mean()) if worn.any() else None,
            "frac_wear_limited": float(worn.mean()),
            "band_by_life": bands}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["912", "914", "915", "916"])
    ap.add_argument("--out", default=os.path.join(ROOT, "backend", "models_v4"))
    ap.add_argument("--gpu", action="store_true", default=True, help="use the GPU (the default)")
    ap.add_argument("--cpu", action="store_true", help="debug only: the CPU gives different answers for these weights")
    args = ap.parse_args()

    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
    from tensorflow import keras
    import train_common as C
    import tf_data_pipeline as P
    import model_architectures as A        # noqa: F401  registers the custom layers
    import physics_v4 as V
    from degradation_v4 import applicable_faults

    os.chdir(HERE)
    e = args.engine
    em = engine_model_of(e)
    spec = V.ENGINE_SPECS_V4[em]
    tbo = C.tbo_of(e)
    applicable = applicable_faults(spec.turbocharged, spec.intercooled)
    mask = np.array([1.0 if f in applicable else 0.0 for f in P.FAULT_MODES], dtype=np.float32)
    margin_engine = V.PistonEngineV4(em, dt=1.0)
    margins_fn = lambda d: margin_engine.margins(d)["health_index"]   # noqa: E731

    missing = [p for p in PHASES if not os.path.exists(C.ckpt_path(e, p))]
    if missing:
        sys.exit(f"{e}: not trained yet - missing {missing}")
    models = {p: keras.models.load_model(C.ckpt_path(e, p), compile=False, safe_mode=False) for p in PHASES}

    print(f"[{e}] predicting validation flights ...", flush=True)
    val = collect(e, "val", models, C, P, mask, margins_fn)
    print(f"[{e}] predicting test flights ...", flush=True)
    test = collect(e, "test", models, C, P, mask, margins_fn)

    # ---- A2: per-fault cut-offs, chosen on validation, reported on test ----
    # Two decision rules, each with its own cut-offs tuned on validation:
    #   diagnosis alone     present = p >= cut-off
    #   with severity       present = p >= cut-off AND the severity head puts it at
    #                       >= 0.08 - the training labels' own definition of present.
    # Live on the 914 the first rule called oil degradation beside bearing wear at
    # p 0.72 (cut-off 0.70) while the severity head gave it 0.069. The rule with the
    # better VALIDATION macro F1 is deployed; test only reports.
    agree = lambda d: d["diag"] * (d["sev"] >= P.DETECTION_THRESHOLD)      # noqa: E731
    cuts_alone = best_cutoffs(val["diag"], val["t_fm"], applicable, P.FAULT_MODES)
    cuts_agree = best_cutoffs(agree(val), val["t_fm"], applicable, P.FAULT_MODES)
    val_alone = scores(val["diag"], val["t_fm"], cuts_alone, applicable, P.FAULT_MODES)["macro_f1"]
    val_agree = scores(agree(val), val["t_fm"], cuts_agree, applicable, P.FAULT_MODES)["macro_f1"]
    require_severity = val_agree > val_alone
    cuts = cuts_agree if require_severity else cuts_alone
    test_prob = agree(test) if require_severity else test["diag"]
    print(f"[{e}] A2 validation macro F1: diagnosis alone {val_alone:.4f}, with severity agreement {val_agree:.4f} "
          f"-> deploying {'with' if require_severity else 'without'} severity agreement")
    flat = {n: 0.5 for n in P.FAULT_MODES}
    before = scores(test["diag"], test["t_fm"], flat, applicable, P.FAULT_MODES)
    after = scores(test_prob, test["t_fm"], cuts, applicable, P.FAULT_MODES)
    # Only faults with recall to lose count: the 914's wastegate goes 0.01 -> 0.00, and
    # it is unobservable below the critical altitude either way.
    recall_kept = all(after["per_fault"][n]["recall"] >= 0.5 * before["per_fault"][n]["recall"]
                      for n in after["per_fault"] if before["per_fault"][n]["recall"] >= 0.10)
    print(f"[{e}] A2 diagnosis macro F1 on test: flat 0.5 -> {before['macro_f1']:.4f}, "
          f"per-fault cut-offs -> {after['macro_f1']:.4f}; recall kept >= 50%: {recall_kept}")

    # ---- A3: RUL with live inputs ----
    k_h = P.RUL_AUX_ORDER.index("engine_hours_norm")
    import joblib
    ax = joblib.load(C.aux_scaler_path(e))
    aux_live = live_aux(test["aux_raw"], test["health"], test["sev"], test["margin_meas"], P)
    zeros = np.zeros((len(aux_live), P.WINDOW_SIZE, P.N_FEATURES), np.float32)
    rul_live = []
    for i in range(0, len(aux_live), 512):
        batch = {"x": zeros[i:i + 512], "x_rul_aux": ax.transform(aux_live[i:i + 512]).astype(np.float32)}
        rul_live.append(models["rul"].predict_on_batch(batch)["y_rul_hours"].ravel())
    rul_live = np.concatenate(rul_live)
    hn = test["aux_raw"][:, k_h]
    rul_label = rul_scores(test["rul_label_inputs"], test["t_rul"], hn, tbo)
    rul_pred = rul_scores(rul_live, test["t_rul"], hn, tbo)
    print(f"[{e}] A3 RUL wear-limited MAE: label inputs {rul_label['mae_pct_tbo_wear_limited']:.2f}% of TBO -> "
          f"live inputs {rul_pred['mae_pct_tbo_wear_limited']:.2f}%  (all windows "
          f"{rul_label['mae_pct_tbo']:.2f}% -> {rul_pred['mae_pct_tbo']:.2f}%)")

    # ---- A5: export ----
    dest = os.path.join(args.out, e)
    os.makedirs(dest, exist_ok=True)
    for p in PHASES:
        shutil.copy2(C.ckpt_path(e, p), os.path.join(dest, f"{p}.keras"))
    shutil.copy2(C.scaler_path(e), os.path.join(dest, "scaler.pkl"))
    shutil.copy2(C.aux_scaler_path(e), os.path.join(dest, "aux_scaler.pkl"))
    results = {}
    for p in PHASES:
        with open(C.result_path(e, p)) as fh:
            results[p] = json.load(fh)["test"]
    sensor_acc = float((test["sensor"].argmax(-1) == test["t_sensor"]).mean())
    manifest = {
        "engine": e, "engine_model": em, "model_version": "v4", "tbo_hours": tbo,
        "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        # The device the cut-offs and RUL numbers below were measured on. aiv4.py
        # reports backend_validated only when it serves on the same one.
        "metrics_device": "CPU" if args.cpu else "GPU",
        "contract": {"window_size": P.WINDOW_SIZE, "sample_hz": 1, "feature_cols": P.FEATURE_COLS,
                     "rul_aux_order": P.RUL_AUX_ORDER, "fault_modes": P.FAULT_MODES,
                     "sensor_channels": P.SENSOR_CHANNELS, "sensor_fault_types": P.SENSOR_FAULT_TYPES,
                     "residual_channels": P.RESIDUAL_CHANNELS,
                     "fault_present_severity": P.DETECTION_THRESHOLD,
                     "rul_live_inputs": {"health_index": "health head", "margin_min": "measured sensors",
                                         "mean_fault_severity": "severity head (applicable faults)",
                                         "max_fault_severity": "severity head (applicable faults)"}},
        "applicable_faults": applicable,
        "heads": {p: {"file": f"{p}.keras", "output": HEAD_OF[p]} for p in PHASES},
        "scalers": {"features": "scaler.pkl", "rul_aux": "aux_scaler.pkl"},
        "fault_thresholds": cuts,
        # present = p >= cut-off, and (when true) the severity head >= fault_present_severity
        "require_severity": bool(require_severity),
        "test": results,
        "diagnosis_cutoffs": {"flat_macro_f1": before["macro_f1"], "tuned_macro_f1": after["macro_f1"],
                              "val_macro_f1_alone": val_alone, "val_macro_f1_with_severity": val_agree,
                              "recall_kept": recall_kept, "per_fault": after["per_fault"]},
        "rul_label_inputs": rul_label,
        "rul_live_inputs": rul_pred,
        "rul_band_hours": [{"life_from": b["life_from"], "life_to": b["life_to"], "mae_hours": b["mae_hours"]}
                           for b in rul_pred["band_by_life"]],
        "sensor_accuracy_check": sensor_acc,
    }
    with open(os.path.join(dest, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=float)
    print(f"[{e}] exported to {dest}")
    ok = after["macro_f1"] >= before["macro_f1"] and recall_kept and \
        (rul_pred["mae_pct_tbo_wear_limited"] or 0.0) <= 15.0
    print(f"[{e}] Phase 2 checks: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
