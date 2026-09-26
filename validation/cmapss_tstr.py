#!/usr/bin/env python3
"""A17 - TSTR gate: train on synthetic, test on real.

THE QUESTION THIS ANSWERS, THAT A16 CANNOT
    A16 shows the generated data has the same distributions, correlations and
    autocorrelation as NASA's. None of that proves the degradation SIGNAL
    survived generation. A generator can match every marginal while flattening
    the temporal trend into noise, and the result would be statistically
    indistinguishable but useless for prognostics - which is the one thing the
    data is for.

    So: train the same model twice, once on real data and once on ours, and
    evaluate BOTH on NASA's real test set. If the synthetic-trained model
    predicts real engines nearly as well as the real-trained one, the signal is
    genuinely there.

        control    real train      -> real test
        treatment  synthetic train -> real test      (never sees a real train row)

PASS CRITERION, DECLARED IN ADVANCE
    synthetic-trained MAE within 15% of real-trained MAE on the real test set.

NO LEAKAGE
    The scaler is fitted on each model's OWN training split. The real test set
    is touched only for evaluation, and the generator's response surface was
    fitted on the real TRAIN split only, so no test information reaches the
    synthetic data by any path.

RUN
    validation/venv/bin/python3 validation/cmapss_tstr.py --subset FD001
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.cmapss import config as C
from simulation.cmapss import parser as P

PROC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cmapss", "processed",
)

WINDOW = 30            # cycles per input window; standard for C-MAPSS
MAE_TOLERANCE = 0.15   # pass if synthetic-trained MAE <= 1.15 x real-trained MAE


def _load_generated(subset: str, split: str, gen_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(gen_dir, f"{split}_{subset}.txt"),
                     sep=r"\s+", header=None, engine="python")
    df.columns = C.ALL_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def make_windows(df: pd.DataFrame, cols: list[str], window: int,
                 rul: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows within each unit. Units shorter than `window` are
    left-padded with their first row rather than dropped, so short test units
    still produce a prediction."""
    X, y = [], []
    vals = df[cols].to_numpy(np.float32)
    starts = df.groupby("unit").indices
    for _, idx in starts.items():
        idx = np.sort(idx)
        v, r = vals[idx], rul[idx]
        if len(idx) < window:
            pad = np.repeat(v[:1], window - len(idx), axis=0)
            v = np.concatenate([pad, v], 0)
            r = np.concatenate([np.repeat(r[:1], window - len(idx)), r])
        for i in range(window, len(v) + 1):
            X.append(v[i - window:i])
            y.append(r[i - 1])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def last_windows(df: pd.DataFrame, cols: list[str], window: int) -> np.ndarray:
    """One window per unit, ending at its final cycle - the C-MAPSS test task."""
    out = []
    vals = df[cols].to_numpy(np.float32)
    for _, idx in sorted(df.groupby("unit").indices.items()):
        idx = np.sort(idx)
        v = vals[idx]
        if len(v) < window:
            v = np.concatenate([np.repeat(v[:1], window - len(v), axis=0), v], 0)
        out.append(v[-window:])
    return np.asarray(out, np.float32)


def build_model(n_features: int, window: int, seed: int):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    inp = keras.Input((window, n_features))
    x = inp
    for d in (1, 2, 4):
        x = keras.layers.Conv1D(64, 3, padding="causal", dilation_rate=d,
                                activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def phm08_score(err: np.ndarray) -> float:
    """Official PHM'08 asymmetric score: late predictions penalised harder."""
    return float(np.sum(np.where(err < 0, np.exp(-err / 13.0) - 1,
                                 np.exp(err / 10.0) - 1)))


def train_and_eval(train_df, train_rul, test_df, test_last, true_rul,
                   cols, seed, epochs, tag):
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = make_windows(train_df, cols, WINDOW, train_rul)

    # Scaler fitted on THIS model's own training data only.
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
    Xtr = sc.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape).astype(np.float32)
    Xte = sc.transform(test_last.reshape(-1, test_last.shape[-1])).reshape(
        test_last.shape).astype(np.float32)

    model = build_model(len(cols), WINDOW, seed)
    model.fit(Xtr, ytr, epochs=epochs, batch_size=256, verbose=0, validation_split=0.1)

    pred = model.predict(Xte, verbose=0).ravel()
    pred = np.clip(pred, 0, C.RUL_CAP)
    truth = np.clip(true_rul, 0, C.RUL_CAP).astype(float)
    err = pred - truth

    return {
        "tag": tag,
        "n_train_windows": int(len(Xtr)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "phm08_score": phm08_score(err),
        "bias": float(err.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="C-MAPSS TSTR gate (A17).")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--gen-dir", default=os.path.join(PROC_DIR, "generated"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--repeats", type=int, default=3,
                    help="training seeds per arm; the gate uses the MEAN, and the "
                         "spread is reported so a single lucky or unlucky run "
                         "cannot decide a pass")
    args = ap.parse_args()

    real_train = P.load_split(args.subset, "train")
    real_test = P.load_split(args.subset, "test")
    true_rul = P.load_test_rul(args.subset)

    syn_train = _load_generated(args.subset, "train", args.gen_dir)

    const = P.constant_sensors(real_train)
    cols = [c for c in C.SENSOR_COLS if c not in const]

    real_rul = P.add_train_rul(real_train)["RUL"].to_numpy(np.float32)
    syn_rul = P.add_train_rul(syn_train)["RUL"].to_numpy(np.float32)
    test_last = last_windows(real_test, cols, WINDOW)

    print(f"=== A17 TSTR GATE - {args.subset} ===")
    print(f"real train {len(real_train):>6} rows | synthetic train {len(syn_train):>6} rows "
          f"| real test {len(real_test):>6} rows / {len(true_rul)} units")
    print(f"features {len(cols)}, window {WINDOW}, epochs {args.epochs}\n")

    def repeat(df, rul, tag):
        runs = [train_and_eval(df, rul, real_test, test_last, true_rul, cols,
                               args.seed + 17 * i, args.epochs, tag)
                for i in range(args.repeats)]
        agg = {"tag": tag, "runs": runs}
        for k in ("mae", "rmse", "phm08_score", "bias"):
            v = np.array([r[k] for r in runs], float)
            agg[k] = float(v.mean())
            agg[k + "_sd"] = float(v.std(ddof=0))
        return agg

    control = repeat(real_train, real_rul, "real->real")
    treat = repeat(syn_train, syn_rul, "synthetic->real")

    ratio = treat["mae"] / control["mae"]
    passed = ratio <= 1.0 + MAE_TOLERANCE

    print(f"{'model':>18} {'MAE':>16} {'RMSE':>16} {'PHM08':>18} {'bias':>16}")
    for r in (control, treat):
        print(f"{r['tag']:>18} {r['mae']:7.2f}+/-{r['mae_sd']:<6.2f} "
              f"{r['rmse']:7.2f}+/-{r['rmse_sd']:<6.2f} "
              f"{r['phm08_score']:8.0f}+/-{r['phm08_score_sd']:<7.0f} "
              f"{r['bias']:+7.2f}+/-{r['bias_sd']:<6.2f}")
    print(f"\nMAE ratio synthetic/real = {ratio:.3f}  (pass <= {1 + MAE_TOLERANCE:.2f})")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")

    rep = {"subset": args.subset, "window": WINDOW, "epochs": args.epochs,
           "seed": args.seed, "repeats": args.repeats, "features": cols,
           "control": control, "treatment": treat,
           "mae_ratio": float(ratio), "tolerance": MAE_TOLERANCE,
           "verdict": "PASS" if passed else "FAIL"}
    path = os.path.join(PROC_DIR, f"tstr_{args.subset}.json")
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"-> {path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
