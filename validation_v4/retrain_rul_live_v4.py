#!/usr/bin/env python3
"""Retrain one engine's RUL head on the inputs a live aircraft has (Phase 2, A3 step 2).

WHY. run.py trains the RUL head with three of its ten aux inputs taken from
simulator labels - true wear condition, true fault severities, and the operating
margin from true values. Live, those come from the health head, the severity head
and the sensors. Scored that way the 914's head fell from 13.07% to 20.58% of TBO on
the wear-limited windows (the calendar alone scores 23.12%): it had learned to lean
on inputs it will never be given.

HOW. The aux inputs are rebuilt exactly as aiv4.py builds them (export_deployable_v4.
live_aux) for every train / val / test window, and only the head is fine-tuned on
them - it reads the aux vector alone, so no window needs to pass through it.

WHY TRAIN-SPLIT PREDICTIONS ARE ACCEPTABLE HERE. Predictions a model makes on its own
training data are usually too good, and a head trained on them learns to trust them.
Measured on the 914 they are not: health MAE 0.055 on train against 0.063 on val,
max-severity MAE 0.205 against 0.213. So the train-split inputs carry the live noise
level and out-of-fold retraining of five models is not needed.

Selection is on VALIDATION (the gate ratio run.py uses: worst of wear-limited/15 and
all/8); test is reported only. The label-trained checkpoint is kept beside it.
Runs on the GPU: these weights were trained and are scored on Metal, and the CPU
gives different answers for them.

    ../validation/venv/bin/python3 retrain_rul_live_v4.py --engine 914
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np                                            # noqa: E402


def gather(e, split, models, C, P, mask, margins_fn):
    """Live aux inputs (raw), true RUL and life stage for every window of a split."""
    import joblib
    from export_deployable_v4 import MARGIN_CHANNELS, live_aux
    sc, ax = joblib.load(C.scaler_path(e)), joblib.load(C.aux_scaler_path(e))
    idx = [P.FEATURE_COLS.index(c) for c in MARGIN_CHANNELS]
    ds = P.make_dataset(C.data_dir(e), split, C.scaler_path(e), aux_scaler_path=C.aux_scaler_path(e),
                        batch_size=256, shuffle_buffer=0, repeat=False)
    A, T = [], []
    for xb, yb in ds:
        health = models["health"].predict_on_batch(xb)["y_health"].ravel()
        sev = models["severity"].predict_on_batch(xb)["y_fault_mode"] * mask
        last = sc.inverse_transform(xb["x"].numpy()[:, -1, :].astype(np.float64))[:, idx]
        margin = np.array([margins_fn(dict(zip(MARGIN_CHANNELS, r))) for r in last])
        raw = ax.inverse_transform(xb["x_rul_aux"].numpy().astype(np.float64))
        A.append(live_aux(raw, health, np.clip(sev, 0.0, 1.0), margin, P))
        T.append(yb["y_rul_hours"].numpy().ravel())
    return np.concatenate(A), np.concatenate(T)


def score(pred, truth, hours_norm, tbo):
    cal = tbo * np.clip(1.0 - hours_norm, 0.0, None)
    worn = truth < cal - 1.0
    err = np.abs(pred - truth)
    wear = float(err[worn].mean() * 100.0 / tbo) if worn.any() else float("nan")
    allm = float(err.mean() * 100.0 / tbo)
    return {"mae_pct_tbo": allm, "mae_pct_tbo_wear_limited": wear,
            "wear_limited_mae_hours": float(err[worn].mean()) if worn.any() else None,
            "calendar_wear_limited_mae_hours": float(np.abs(cal - truth)[worn].mean()) if worn.any() else None,
            "gate_ratio": max(wear / 15.0, allm / 8.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    import joblib
    import tensorflow as tf
    from tensorflow import keras
    import train_common as C
    import tf_data_pipeline as P
    import model_architectures as A    # noqa: F401
    import physics_v4 as V
    from degradation_v4 import applicable_faults
    from export_deployable_v4 import engine_model_of

    os.chdir(HERE)
    e = args.engine
    em = engine_model_of(e)
    spec = V.ENGINE_SPECS_V4[em]
    tbo = C.tbo_of(e)
    mask = np.array([f in applicable_faults(spec.turbocharged, spec.intercooled) for f in P.FAULT_MODES],
                    dtype=np.float32)
    meng = V.PistonEngineV4(em, dt=1.0)
    margins_fn = lambda d: meng.margins(d)["health_index"]     # noqa: E731
    load = lambda p: keras.models.load_model(C.ckpt_path(e, p), compile=False, safe_mode=False)  # noqa: E731
    models = {"health": load("health"), "severity": load("severity")}
    rul = load("rul")
    ax = joblib.load(C.aux_scaler_path(e))
    k_h = P.RUL_AUX_ORDER.index("engine_hours_norm")

    data = {}
    for split in ("train", "val", "test"):
        a, t = gather(e, split, models, C, P, mask, margins_fn)
        data[split] = (ax.transform(a).astype(np.float32), t.astype(np.float32), a[:, k_h])
        print(f"[{e}] {split}: {len(t):,} windows", flush=True)

    # The head alone: aux in, hours out. Its layers are the checkpoint's own, so
    # training it trains the full model.
    head = keras.Model(rul.get_layer("x_rul_aux").output, rul.get_layer("y_rul_hours").output)
    pred = lambda s: head.predict(data[s][0], batch_size=2048, verbose=0).ravel()   # noqa: E731
    before = {s: score(pred(s), data[s][1], data[s][2], tbo) for s in ("val", "test")}
    print(f"[{e}] before: val gate {before['val']['gate_ratio']:.3f}  test wear-limited "
          f"{before['test']['mae_pct_tbo_wear_limited']:.2f}% all {before['test']['mae_pct_tbo']:.2f}%", flush=True)

    head.compile(optimizer=keras.optimizers.Adam(args.lr), loss=C.rul_loss(tbo))
    best, best_w, wait = before["val"]["gate_ratio"], head.get_weights(), 0
    xt, yt, _ = data["train"]
    for ep in range(args.epochs):
        head.fit(xt, yt, batch_size=256, epochs=1, shuffle=True, verbose=0)
        v = score(pred("val"), data["val"][1], data["val"][2], tbo)
        mark = ""
        if v["gate_ratio"] < best - 1e-4:
            best, best_w, wait, mark = v["gate_ratio"], head.get_weights(), 0, "  *"
        else:
            wait += 1
        print(f"  epoch {ep + 1:3d}  val wear-limited {v['mae_pct_tbo_wear_limited']:6.2f}%  "
              f"all {v['mae_pct_tbo']:5.2f}%  gate {v['gate_ratio']:.3f}{mark}", flush=True)
        if wait >= 10:
            break
    head.set_weights(best_w)
    after = {s: score(pred(s), data[s][1], data[s][2], tbo) for s in ("val", "test")}
    t = after["test"]
    print(f"[{e}] after:  test wear-limited {t['mae_pct_tbo_wear_limited']:.2f}% of TBO "
          f"({t['wear_limited_mae_hours']:.0f} h; calendar {t['calendar_wear_limited_mae_hours']:.0f} h), "
          f"all {t['mae_pct_tbo']:.2f}%")

    if after["val"]["gate_ratio"] >= before["val"]["gate_ratio"]:
        print(f"[{e}] no improvement on validation - checkpoint left as it was")
        return
    keep = os.path.join(C.ART_ROOT, e, "rul_label_inputs")
    os.makedirs(keep, exist_ok=True)
    for f in (C.ckpt_path(e, "rul"), C.result_path(e, "rul")):
        if not os.path.exists(os.path.join(keep, os.path.basename(f))):
            shutil.copy2(f, keep)
    rul.save(C.ckpt_path(e, "rul"))
    res = json.load(open(C.result_path(e, "rul")))
    res["test_label_inputs"] = res.get("test_label_inputs", res["test"])
    res["test"] = {**res["test"], "mae_pct_tbo": t["mae_pct_tbo"],
                   "mae_pct_tbo_wear_limited": t["mae_pct_tbo_wear_limited"],
                   "mae_pct_tbo_all": t["mae_pct_tbo"]}
    res["rul_inputs"] = "live: predicted health and severity, margin from measured sensors"
    C.save_result(e, "rul", res)
    print(f"[{e}] saved; the label-trained checkpoint is in {keep}")


if __name__ == "__main__":
    main()
