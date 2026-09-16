#!/usr/bin/env python3
"""Headless runner for phases 0-5 across all four engines, one at a time.

Mirrors the notebooks exactly - same architectures, losses, metrics, callbacks
and checkpoint paths - but runs unattended so you are not babysitting 24 cells.
The notebooks stay useful for interactive work; this is for the ~14 hour grind.

USAGE
    validation/venv/bin/python3 validation/run.py                  # phases 2-5, all engines
    validation/venv/bin/python3 validation/run.py --phases 0,1,2,3,4,5 --engines 915,916 --order engine
                                                                    # full chain after regenerating data
                                                                    # (0 = scaler, 1 = detection)
    validation/venv/bin/python3 validation/run.py --phases 2,3     # a subset
    validation/venv/bin/python3 validation/run.py --engines 914    # one engine
    validation/venv/bin/python3 validation/run.py --order engine   # finish each engine fully
    validation/venv/bin/python3 validation/run.py --force          # redo completed work

DEFAULT ORDER is phase-major: phase2 for every engine, then phase3 for every
engine, and so on. Use --order engine to instead finish one engine end to end,
which surfaces a broken pipeline after ~4 hours rather than ~14.

RESUMABLE. A (phase, engine) pair is skipped when both its .keras checkpoint and
its _test.json exist. Kill it and rerun; it picks up where it stopped.

GATES. After each phase the result is checked against the threshold that matters
for that head - not accuracy, which is meaningless for three of the four. A
failure is reported loudly and, by default, stops that engine's chain, because
phases 2->4 warm-start each other and a broken encoder propagates downward.
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tf_data_pipeline as P
import model_architectures as A
import train_common as C

ALL_PHASES = [2]

# Epoch caps - the same values the notebooks use.
EPOCHS = {1: 30, 2: 20, 3: 20, 4: 20, 5: 80}   # hybrid RUL head was still improving at epoch 49 of 50
# Phase 2 macro-F1 swings +/-0.05 epoch to epoch (912: 0.41 -> 0.35 -> 0.36),
# so patience 3 stopped it at epoch 6 on noise.
PATIENCE = {1: 5, 2: 5, 3: 3, 4: 3, 5: 10}
LR_PATIENCE = {1: 2, 2: 2, 3: 2, 4: 2, 5: 5}   # RUL: see train_common.callbacks
MIN_LR = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1e-5}

PHASE_NAME = {
    1: "phase1_detection",
    2: "phase2_diagnosis",
    3: "phase3_severity",
    4: "phase4_failure_modes",
    5: "phase5_rul",
}
WARM_FROM = {1: None, 2: "phase1_detection", 3: "phase2_diagnosis", 4: "phase3_severity", 5: None}

# RUL near-new weighting (train_common.split_labels): windows with true RUL above
# RUL_NEAR_NEW_FROM of TBO ramp up to RUL_NEAR_NEW_WEIGHT x in the loss.
RUL_NEAR_NEW_WEIGHT = 10.0
RUL_NEAR_NEW_FROM = 0.90
RUL_NEAR_NEW_BIAS_GATE_PCT = 2.0     # |mean error| on true RUL >= 97% TBO, % of TBO
# Training-time feature dropout on x_rul_aux bsfc_ratio. Under physics v3 it is nearly a direct
# read of wear; with normalised aux the hybrid head leaned on it (MAE 1% -> 9.9% when it was
# mean-imputed). Replacing it with its train mean on this share of windows forces a fallback path.
RUL_BSFC_DROPOUT = 0.30
# --rul-finetune: continue from the existing phase-5 checkpoint instead of from scratch.
RUL_FINETUNE = {"lr": 3e-5, "epochs": 15, "patience": 4, "lr_patience": 2, "min_lr": 1e-6}


# ---------------------------------------------------------------------------
# Gates. Each head has a trivial predictor that scores well on the obvious
# metric, so these check the metric that actually moves when it learns:
#   diagnosis  - always-'none' scores 87.4% accuracy, macro_f1 0.0
#   severity   - all-zeros scores 0.095 plain MSE, but 0.954 nonzero MSE
#   modes      - all-zeros scores 0.012-0.032 MAE
# ---------------------------------------------------------------------------
def check_gate(phase, res):
    if phase == 1:
        # Measured on the prop-2.00 data: 0.958 (914), 0.993 (915), 0.988 (916).
        auc = res.get("y_detection_auc", res.get("auc", 0.0))
        ok = auc >= 0.90
        return ok, f"detection auc {auc:.4f} (>=0.90)"
    if phase == 2:
        f1 = res.get("y_diagnosis_macro_f1", res.get("macro_f1", 0.0))
        rc = res.get("y_diagnosis_fault_recall", res.get("fault_recall", 0.0))
        # 0.45, not 0.55: each channel only ever carries ONE fault type, and Spike /
        # Stuck-At exist only as rare RPM glitches, so macro-F1 over five classes is
        # capped near 0.6 even with the common classes perfect.
        ok = f1 >= 0.45 and rc >= 0.30
        return ok, f"macro_f1 {f1:.4f} (>=0.45), fault_recall {rc:.4f} (>=0.30)"
    if phase == 3:
        mse = res.get("y_severity_nonzero_severity_mse", res.get("nonzero_severity_mse", 9.9))
        ok = mse <= 0.10
        return ok, f"nonzero_severity_mse {mse:.4f} (<=0.10; all-zeros baseline 0.954)"
    if phase == 4:
        # Gate on the detectable-fault AUC (severity >= DETECTION_THRESHOLD, the same
        # cut phase 1 uses); the strict >0 AUC is printed alongside, never hidden.
        auc = res.get("auc_all_det", 0.0)
        strict = res.get("auc_all", float("nan"))
        ok = auc >= 0.85
        return ok, f"auc_all_det {auc:.4f} (>=0.85), strict auc_all {strict:.4f}"
    if phase == 5:
        pct = res.get("mae_pct_tbo", 99.0)
        corr = res.get("corr", 0.0)
        mono = bool(res.get("monotonic", False))
        imp = res.get("mae_pct_tbo_bsfc_imputed", 99.0)
        base = res.get("baseline_bsfc_only_pct_tbo", 0.0)
        nn = res.get("near_new_bias_pct_tbo")
        nn_ok = nn is None or abs(nn) <= RUL_NEAR_NEW_BIAS_GATE_PCT
        ok = pct <= 5.0 and corr >= 0.85 and mono and imp <= 5.0 and pct < base and nn_ok
        return ok, (f"mae {pct:.2f}% of TBO (<=5), corr {corr:.4f} (>=0.85), monotonic {mono}, "
                    f"without bsfc_ratio {imp:.2f}% (<=5), bsfc-only baseline {base:.2f}% (model must beat), "
                    f"near-new bias {'n/a' if nn is None else f'{nn:+.2f}%'} (|x|<={RUL_NEAR_NEW_BIAS_GATE_PCT})")
    return True, ""


# ---------------------------------------------------------------------------
def build(phase, engine, cw, lr=None):
    """Returns (model, keep, needs_aux, monitor, mode). `lr` overrides phase 5's rate."""
    if phase == 1:
        # Same as phase1_detection_<key>.ipynb.
        xi = A.build_encoder_input(); enc = A.build_encoder(xi)
        m = keras.Model(xi, {"y_detection": A.build_detection_head(enc)}, name="detection")
        m.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  loss={"y_detection": keras.losses.BinaryCrossentropy()},
                  metrics={"y_detection": [keras.metrics.BinaryAccuracy(name="acc"),
                                           keras.metrics.AUC(name="auc")]})
        return m, ["y_detection"], False, "val_loss", "min"

    if phase == 2:
        xi = A.build_encoder_input(); enc = A.build_encoder(xi)
        m = keras.Model(xi, {"y_detection": A.build_detection_head(enc),
                             "y_diagnosis": A.build_diagnosis_head(enc)})
        m.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  loss={"y_detection": keras.losses.BinaryCrossentropy(),
                        "y_diagnosis": C.weighted_diagnosis_loss(cw)},
                  loss_weights={"y_detection": 1.0, "y_diagnosis": 50.0},
                  metrics={"y_diagnosis": [C.DiagnosisMacroF1(), C.DiagnosisFaultRecall()]})
        return m, ["y_detection", "y_diagnosis"], False, "val_y_diagnosis_macro_f1", "max"

    if phase == 3:
        xi = A.build_encoder_input(); enc = A.build_encoder(xi)
        dia = A.build_diagnosis_head(enc)
        m = keras.Model(xi, {"y_detection": A.build_detection_head(enc),
                             "y_diagnosis": dia,
                             "y_severity": A.build_severity_head(enc, dia)})
        m.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  loss={"y_detection": keras.losses.BinaryCrossentropy(),
                        "y_diagnosis": C.weighted_diagnosis_loss(cw),
                        "y_severity": keras.losses.MeanSquaredError()},
                  loss_weights={"y_detection": 1.0, "y_diagnosis": 50.0, "y_severity": 20.0},
                  metrics={"y_severity": [C.nonzero_severity_mse, C.nonzero_severity_mae]})
        return (m, ["y_detection", "y_diagnosis", "y_severity"], False,
                "val_y_severity_nonzero_severity_mse", "min")

    if phase == 4:
        xi = A.build_encoder_input(); enc = A.build_encoder(xi)
        m = keras.Model(xi, {"y_failure_mode": A.build_failure_mode_head(enc)})
        # BCE on the soft 0..1 severities rather than MSE: MSE's gradient carries a
        # sigmoid' factor that vanishes as soon as an output saturates.
        m.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  loss={"y_failure_mode": keras.losses.BinaryCrossentropy()},
                  metrics={"y_failure_mode": [keras.metrics.MeanAbsoluteError(name="mae")]
                                             + C.failure_mode_metrics()})
        return m, ["y_failure_mode"], False, "val_auc_all_det", "max"   # single output: no y_ prefix

    if phase == 5:
        xi = A.build_encoder_input(); aux = A.build_aux_input()
        h = A.build_rul_lstm_encoder(xi, units=32, num_layers=2, dropout=0.1)
        # Hybrid RUL branch. The LSTM alone was best overall (MAE 22.9 h on 914) but under-predicted
        # nearly-new engines by 80 h: x_rul_aux arrives raw (bsfc_ratio moves ~0.5% near a new
        # engine while cht_excess sits near 19) and early wear sits in small per-sensor level
        # shifts. A standardised ridge on window mean/std + aux cut that bias to -22 h. So the
        # head also gets each sensor's window mean and std, concatenated with the aux features and
        # normalised together (adapted on the train split in run_one). Inputs are unchanged, so
        # aiv3.py still sends the raw window and raw aux.
        # Window variance from standard layers, E[x^2] - E[x]^2 (ReLU clips float round-off below
        # zero). A Lambda would not survive reloading: its closure loses run.py's globals, and
        # aiv3.py / export_edge.py load these checkpoints outside this module.
        win_mean = keras.layers.GlobalAveragePooling1D(name="rul_win_mean")(xi)
        win_sq_mean = keras.layers.GlobalAveragePooling1D(name="rul_win_sq_mean")(
            keras.layers.Multiply(name="rul_win_sq")([xi, xi]))
        win_var = keras.layers.ReLU(name="rul_win_var")(keras.layers.Subtract(name="rul_win_var_raw")(
            [win_sq_mean, keras.layers.Multiply(name="rul_win_mean_sq")([win_mean, win_mean])]))
        feats = keras.layers.Concatenate(name="rul_feats")([win_mean, win_var, aux])
        feats_n = keras.layers.Normalization(name="rul_feat_norm")(feats)
        m = keras.Model({"x": xi, "x_rul_aux": aux},
                        {"y_rul_hours": A.build_rul_head(h, feats_n, init_hours=0.3 * C.tbo_of(engine))})
        m.compile(optimizer=keras.optimizers.Adam(lr or 1e-4, clipnorm=1.0),
                  loss={"y_rul_hours": C.rul_loss(C.tbo_of(engine))},
                  metrics={"y_rul_hours": [keras.metrics.MeanAbsoluteError(name="mae"),
                                           C.mae_pct_tbo(C.tbo_of(engine))]})
        # val_loss, not val_mae: the loss carries the near-new sample weights, the unweighted
        # MAE metric would stop training on exactly the windows the weighting improves.
        return m, ["y_rul_hours"], True, "val_loss", "min"

    raise ValueError(phase)


def rul_checks(model, engine, test_ds, steps):
    """The three RUL checks plus the bsfc_ratio ablation.

    v2's RUL looked fine on MAE while tracking flight duration rather than engine
    condition, so MAE alone is not evidence. Monotonicity and the ablation are.
    """
    tbo = C.tbo_of(engine)
    yt, yp = [], []
    for i, (x, y) in enumerate(test_ds):
        yp.append(model.predict(x, verbose=0)["y_rul_hours"].ravel())
        yt.append(y["y_rul_hours"].numpy().ravel())
        if i >= steps:
            break
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    mae = float(np.mean(np.abs(yt - yp)))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else 0.0
    bins = np.array_split(np.argsort(yt), 10)
    means = [float(yp[b].mean()) for b in bins]
    mono = all(means[i] <= means[i + 1] for i in range(len(means) - 1))

    # Does the head actually use the window, or just read bsfc_ratio (r=+0.95 vs
    # wear)? Two checks:
    #  1. MEAN-impute bsfc_ratio and re-score. (Zeroing it, as first written, puts
    #     it far outside its real ~1.00-1.18 range, so MAE explodes whether or not
    #     the head relies on it - that test was uninformative.)
    #  2. Fit RUL ~ bsfc_ratio alone. The model must beat that single-feature line.
    idx = P.RUL_AUX_ORDER.index("bsfc_ratio")
    bs, yt2, yp2 = [], [], []
    for i, (x, y) in enumerate(test_ds):
        bs.append(x["x_rul_aux"].numpy()[:, idx])
        yt2.append(y["y_rul_hours"].numpy().ravel())
        if i >= steps:
            break
    bs = np.concatenate(bs); yt2 = np.concatenate(yt2)
    bmean = float(bs.mean())
    for i, (x, y) in enumerate(test_ds):
        xa = x["x_rul_aux"].numpy().copy(); xa[:, idx] = bmean
        yp2.append(model.predict({"x": x["x"], "x_rul_aux": xa}, verbose=0)["y_rul_hours"].ravel())
        if i >= steps:
            break
    yp2 = np.concatenate(yp2)[: len(yt2)]
    mae_imp = float(np.mean(np.abs(yp2 - yt2)))
    A_ = np.c_[bs, np.ones_like(bs)]
    coef = np.linalg.lstsq(A_, yt2, rcond=None)[0]
    mae_base = float(np.mean(np.abs(A_ @ coef - yt2)))

    new = yt >= 0.97 * tbo           # nearly-new engines: where every live flight starts
    nn_bias = float(np.mean(yp[new] - yt[new])) if new.any() else None
    nn_mae = float(np.mean(np.abs(yp[new] - yt[new]))) if new.any() else None

    return {"mae_hours": mae, "mae_pct_tbo": 100.0 * mae / tbo, "corr": corr,
            "monotonic": mono, "bin_means": means,
            "near_new_windows": int(new.sum()), "near_new_bias_hours": nn_bias, "near_new_mae_hours": nn_mae,
            "near_new_bias_pct_tbo": None if nn_bias is None else 100.0 * nn_bias / tbo,
            "mae_hours_bsfc_imputed": mae_imp,
            "mae_pct_tbo_bsfc_imputed": 100.0 * mae_imp / tbo,
            "baseline_bsfc_only_pct_tbo": 100.0 * mae_base / tbo}


def fit_scaler(engine, force=False):
    """Phase 0 - same as phase0_scalers_<key>.ipynb: StandardScaler over FEATURE_COLS,
    fit incrementally on the TRAIN split only, then a one-batch pipeline sanity load."""
    import joblib
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    out = C.scaler_path(engine)
    if not force and os.path.exists(out):
        print(f"  SKIP  phase0_scaler {C.ENGINE_KEY[engine]}  (already exists)")
        return True
    print(f"\n{'='*78}\n  phase0_scaler  |  {engine}\n{'='*78}")
    dd = C.data_dir(engine)
    idx = json.load(open(C.index_path(engine)))
    sc = StandardScaler()
    for f in sorted({e["file"] for e in idx["train"]}):
        df = pd.read_parquet(os.path.join(dd, f), columns=P.FEATURE_COLS)
        sc.partial_fit(df)
        print(f"  {f}: {len(df):,} rows")
        del df
    joblib.dump(sc, out)
    x, _ = next(iter(P.make_dataset(C.index_path(engine), dd, "val", out, batch_size=4)))
    ok = tuple(x["x"].shape) == (4, P.WINDOW_SIZE, P.N_FEATURES) and bool(np.all(np.isfinite(x["x"].numpy())))
    print(f"  wrote {out} | sanity batch {tuple(x['x'].shape)} finite={ok}")
    return ok


def run_one(phase, engine, force=False, rul_finetune=False):
    if phase == 0:
        return fit_scaler(engine, force=force)
    key = C.ENGINE_KEY[engine]
    name = PHASE_NAME[phase]
    ckpt = C.ckpt_path(engine, name)
    resj = os.path.join(C.MODELS_DIR, f"{name}_{key}_test.json")

    if not force and os.path.exists(ckpt) and os.path.exists(resj):
        res = json.load(open(resj))
        ok, msg = check_gate(phase, res)
        print(f"  SKIP  {name} {key}  (already done)   {'PASS' if ok else 'FAIL'}  {msg}")
        return ok

    print(f"\n{'='*78}\n  {name}  |  {engine}  ({key})\n{'='*78}")
    t0 = time.time()

    steps = C.step_counts(engine)
    train_ds, val_ds, test_ds = C.datasets(engine)

    cw = None
    if phase in (2, 3):
        cwp = os.path.join(C.MODELS_DIR, f"diag_class_weights_{key}.npy")
        if os.path.exists(cwp):
            cw = np.load(cwp)
        else:
            print("  computing diagnosis class weights (inverse-sqrt, train only)...")
            cw = C.diagnosis_class_weights(engine)
            np.save(cwp, cw)
        print(f"  class weights: {np.round(cw, 3)}")

    finetune = phase == 5 and rul_finetune
    model, keep, aux, monitor, mode = build(phase, engine, cw, lr=RUL_FINETUNE["lr"] if finetune else None)

    if phase == 5:
        if finetune:
            prev_model = keras.models.load_model(ckpt, safe_mode=False, compile=False)
            if "rul_feat_norm" not in [l.name for l in prev_model.layers]:
                raise SystemExit(f"{ckpt} predates the hybrid RUL branch (window stats + normalised aux); "
                                 "retrain phase 5 without --rul-finetune")
        else:
            print("  adapting rul_feat_norm on train-split window stats + x_rul_aux (400 batches)...")
            feat_model = keras.Model(model.inputs, model.get_layer("rul_feats").output)
            batches = []
            for i, (x, _) in enumerate(train_ds):
                batches.append(np.asarray(feat_model.predict({"x": x["x"], "x_rul_aux": x["x_rul_aux"]}, verbose=0)))
                if i + 1 >= 400:
                    break
            feats_all = np.concatenate(batches)
            model.get_layer("rul_feat_norm").adapt(feats_all)
            # rul_feats = [window mean (N_FEATURES), window variance (N_FEATURES), x_rul_aux]
            bsfc_idx = P.RUL_AUX_ORDER.index("bsfc_ratio")
            bsfc_mean = float(feats_all[:, 2 * P.N_FEATURES + bsfc_idx].mean())
            col = tf.one_hot(bsfc_idx, P.N_RUL_AUX)

            def drop_bsfc(x, y):
                a = x["x_rul_aux"]
                drop = tf.cast(tf.random.uniform([tf.shape(a)[0], 1]) < RUL_BSFC_DROPOUT, tf.float32) * col
                return {**x, "x_rul_aux": a * (1.0 - drop) + drop * bsfc_mean}, y

            print(f"  bsfc_ratio dropout {RUL_BSFC_DROPOUT:.0%} of train windows (mean {bsfc_mean:.4f})")
            train_ds = train_ds.map(drop_bsfc)

    prev = name if finetune else WARM_FROM[phase]
    if prev:
        C.warm_start(model, C.ckpt_path(engine, prev))

    epochs = RUL_FINETUNE["epochs"] if finetune else EPOCHS[phase]
    patience = RUL_FINETUNE["patience"] if finetune else PATIENCE[phase]
    lr_patience = RUL_FINETUNE["lr_patience"] if finetune else LR_PATIENCE[phase]
    min_lr = RUL_FINETUNE["min_lr"] if finetune else MIN_LR[phase]
    labels = (C.split_labels(keep, needs_aux=aux, tbo_hours=C.tbo_of(engine),
                             near_new_weight=RUL_NEAR_NEW_WEIGHT, near_new_from=RUL_NEAR_NEW_FROM)
              if phase == 5 else C.split_labels(keep, needs_aux=aux))

    print(f"  steps/epoch {steps['train']:,}  val {steps['val']:,}  "
          f"epochs<={epochs}  monitor {monitor} ({mode})" + ("  [fine-tune from existing checkpoint]" if finetune else ""))

    model.fit(
        train_ds.map(labels),
        validation_data=val_ds.map(labels),
        steps_per_epoch=steps["train"], validation_steps=steps["val"],
        epochs=epochs,
        callbacks=C.callbacks(ckpt, monitor=monitor, mode=mode, patience=patience,
                            lr_patience=lr_patience, min_lr=min_lr),
        verbose=2,
    )

    res = model.evaluate(test_ds.map(labels),
                         steps=steps["test"], verbose=0, return_dict=True)
    res = {k: float(v) for k, v in res.items()}
    if phase == 5:
        res.update(rul_checks(model, engine, test_ds, steps["test"]))

    res["minutes"] = round((time.time() - t0) / 60.0, 1)
    json.dump(res, open(resj, "w"), indent=2)

    ok, msg = check_gate(phase, res)
    print(f"\n  {'GATE PASS' if ok else 'GATE FAIL'}: {msg}")
    print(f"  {res['minutes']:.1f} min -> {ckpt}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="2,3,4,5")
    ap.add_argument("--engines", default=",".join(C.ENGINE_KEY.values()))
    ap.add_argument("--order", choices=["phase", "engine"], default="phase")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rul-finetune", action="store_true",
                    help="phase 5: continue from the existing checkpoint with near-new weighting "
                         "(lr 3e-5, <=15 epochs) instead of training from scratch")
    ap.add_argument("--continue-on-fail", action="store_true",
                    help="keep going past a failed gate (phases 2-4 warm-start, so a "
                         "broken encoder propagates - off by default for a reason)")
    a = ap.parse_args()

    phases = [int(x) for x in a.phases.split(",") if x.strip()]
    key2eng = {v: k for k, v in C.ENGINE_KEY.items()}
    engines = [key2eng[k.strip()] if k.strip() in key2eng else k.strip()
               for k in a.engines.split(",") if k.strip()]

    C.report_gpu()
    for e in engines:
        if not os.path.exists(C.index_path(e)):
            sys.exit(f"ERROR: no dataset for {e} at {C.data_dir(e)} - run the generator first")
        if 0 not in phases and not os.path.exists(C.scaler_path(e)):
            sys.exit(f"ERROR: no scaler for {e} - add phase 0 (--phases 0,...) or run phase0_scalers_{C.ENGINE_KEY[e]}.ipynb")

    jobs = ([(p, e) for p in phases for e in engines] if a.order == "phase"
            else [(p, e) for e in engines for p in phases])

    print(f"\n{len(jobs)} jobs, {a.order}-major order, sequential")
    for p, e in jobs:
        print(f"    phase{p}  {e}")

    t0 = time.time()
    failed, skipped_engines = [], set()
    for p, e in jobs:
        if e in skipped_engines and p not in (0, 5):
            # 2->4 warm-start each other; 5 is an independent branch so it still runs.
            print(f"  SKIP  phase{p} {C.ENGINE_KEY[e]} - earlier phase failed for this engine")
            continue
        try:
            if not run_one(p, e, force=a.force, rul_finetune=a.rul_finetune):
                failed.append((p, e))
                if not a.continue_on_fail and p != 5:
                    skipped_engines.add(e)
        except Exception:
            traceback.print_exc()
            failed.append((p, e))
            if not a.continue_on_fail and p != 5:
                skipped_engines.add(e)

    print(f"\n{'='*78}")
    print(f"  finished in {(time.time()-t0)/3600:.2f} h")
    if failed:
        print("  FAILED / GATE-FAILED:")
        for p, e in failed:
            print(f"    phase{p}  {e}")
    else:
        print("  all gates passed")
    print(f"{'='*78}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
