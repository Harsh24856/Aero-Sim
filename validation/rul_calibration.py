"""Post-hoc RUL calibration fitted on the VALIDATION split (plan item 7).

The RUL head is accurate on average (test MAE 0.6-1.2% of TBO) but compresses the ends
of the range: the healthiest engines are predicted ~100-200 h short. Live flights start
on nearly-new engines, so they sit exactly where the bias is largest.

A monotone piecewise-linear map from predicted to true hours is fitted on validation
windows (binned by prediction, means of truth, forced non-decreasing), then scored on the
test split before and after. Nothing is refitted on test data. The map is stored in each
manifest as `rul_calibration`, which aiv3.py applies after the RUL head; `rul_test` is
updated to the calibrated test MAE so the cockpit's band matches what is served.

    validation/venv/bin/python3 validation/rul_calibration.py --engines 912,914,915,916
Stop the AI service first on an 8 GB machine (two TensorFlow processes do not fit).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tensorflow as tf                     # noqa: E402,F401
from tensorflow import keras                # noqa: E402
import train_common as C                    # noqa: E402

ROOT = os.path.join(os.path.dirname(HERE), "backend", "models_v3")


def collect(model, ds, max_batches):
    yt, yp = [], []
    for i, (x, y) in enumerate(ds):
        yp.append(np.asarray(model.predict({"x": x["x"], "x_rul_aux": x["x_rul_aux"]}, verbose=0)).ravel())
        yt.append(y["y_rul_hours"].numpy().ravel())
        if i + 1 >= max_batches:
            break
    return np.concatenate(yt), np.concatenate(yp)


def fit_knots(yt, yp, n_bins):
    order = np.argsort(yp)
    bins = [b for b in np.array_split(order, n_bins) if len(b)]
    pred = np.array([yp[b].mean() for b in bins])
    true = np.maximum.accumulate(np.array([yt[b].mean() for b in bins]))
    keep = np.r_[True, np.diff(pred) > 1e-6]              # np.interp needs increasing x
    return pred[keep].tolist(), true[keep].tolist()


def apply(yp, knots, tbo):
    return np.clip(np.interp(yp, knots[0], knots[1]), 0.0, tbo)


def score(yt, yp, tbo):
    dec = np.array_split(np.argsort(yt), 10)
    new = yt >= 0.97 * tbo            # nearly-new engines, where every live flight starts
    return {"near_new_windows": int(new.sum()),
            "near_new_bias_hours": round(float(np.mean(yp[new] - yt[new])), 1) if new.any() else None,
            "near_new_mae_hours": round(float(np.mean(np.abs(yp[new] - yt[new]))), 1) if new.any() else None,
            "mae_hours": round(float(np.mean(np.abs(yp - yt))), 2),
            "mae_pct_tbo": round(float(100.0 * np.mean(np.abs(yp - yt)) / tbo), 3),
            "healthiest_decile_bias_hours": round(float(np.mean(yp[dec[-1]] - yt[dec[-1]])), 1),
            "most_worn_decile_bias_hours": round(float(np.mean(yp[dec[0]] - yt[dec[0]])), 1),
            "corr": round(float(np.corrcoef(yt, yp)[0, 1]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="912,914,915,916")
    ap.add_argument("--batches", type=int, default=300, help="batches per split")
    ap.add_argument("--bins", type=int, default=40)
    a = ap.parse_args()
    by_key = {v: k for k, v in C.ENGINE_KEY.items()}
    report = {}
    for key in [k.strip() for k in a.engines.split(",") if k.strip()]:
        engine = by_key[key]
        man_path = os.path.join(ROOT, key, "manifest.json")
        man = json.load(open(man_path))
        tbo = float(man["tbo_hours"])
        model = keras.models.load_model(os.path.join(ROOT, key, man["heads"]["rul"]["file"]),
                                        safe_mode=False, compile=False)
        _, val_ds, test_ds = C.datasets(engine)
        vt, vp = collect(model, val_ds, a.batches)
        tt, tp = collect(model, test_ds, a.batches)
        knots = fit_knots(vt, vp, a.bins)
        before, after = score(tt, tp, tbo), score(tt, apply(tp, knots, tbo), tbo)
        man["rul_calibration"] = {
            "method": f"monotone binned means ({a.bins} bins) fitted on {len(vt)} validation windows",
            "pred_knots": [round(v, 3) for v in knots[0]], "true_knots": [round(v, 3) for v in knots[1]],
            "test_before": before, "test_after": after,
        }
        rul_test = man.get("rul_test") or {}
        rul_test.setdefault("mae_hours_uncalibrated", rul_test.get("mae_hours", before["mae_hours"]))
        rul_test.update({"mae_hours": after["mae_hours"], "mae_pct_tbo": after["mae_pct_tbo"],
                         "corr": after["corr"], "calibrated": True})
        man["rul_test"] = rul_test
        json.dump(man, open(man_path, "w"), indent=2)
        report[key] = {"before": before, "after": after, "val_windows": len(vt), "test_windows": len(tt)}
        print(f"{key}: test before {before} | after {after}", flush=True)
    json.dump(report, open(os.path.join(HERE, "models", "rul_calibration_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
