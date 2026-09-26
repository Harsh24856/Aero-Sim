#!/usr/bin/env python3
"""Classifier two-sample test - the strongest practical proof of "identical".

WHY THE EXISTING GATES ARE NOT ENOUGH
    A16 checks marginals, correlations, autocorrelation, trend and a PCA
    overlay. Each is a projection: data can match every one of them and still
    differ in some direction nobody thought to look at. A16 can only ever say
    "no difference was found by these six tests".

    A classifier two-sample test inverts the burden. Train a discriminator to
    tell real rows from synthetic ones. If a flexible model cannot beat chance,
    then no difference exists that THAT model could find - which covers every
    interaction, every higher moment and every joint structure it can represent,
    not just the handful a human chose to check.

        AUC ~ 0.50   indistinguishable
        AUC ~ 1.00   trivially separable

    This is the standard test in the generative-modelling literature for exactly
    this claim, and it is the closest thing to a proof available. It is run at
    two levels, because they can disagree:

        row      single cycles - are the joint sensor distributions the same?
        window   30-cycle sequences - is the temporal structure the same?

    A generator can pass at row level and fail at window level by matching every
    instantaneous distribution while getting the dynamics wrong.

HONEST READING OF THE RESULT
    Failing to reject is not proof of identity in the mathematical sense; no
    finite test gives that. What a passing C2ST establishes is that a gradient-
    boosted model with access to the full joint distribution, trained
    specifically to find a difference, could not do better than chance on held-
    out data. That is a far stronger statement than any marginal test, and it is
    falsifiable - a real difference shows up as AUC above 0.5.

    It is also possible to pass this test with data that is useless, by matching
    distributions while destroying the degradation signal. That is what the TSTR
    gate (A17) exists to rule out. The two together are the evidence.

NO LEAKAGE
    The discriminator is trained and evaluated on disjoint UNITS, not disjoint
    rows. Splitting rows would let it memorise a unit from its training rows and
    recognise the same unit's test rows, scoring above chance for a reason that
    has nothing to do with real-vs-synthetic.

RUN
    validation/venv/bin/python3 validation/cmapss_c2st.py --subset FD001
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.cmapss import config as C
from simulation.cmapss import parser as P

PROC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cmapss", "processed",
)

WINDOW = 30

# Declared before running. 0.5 is chance; anything a discriminator can exploit
# pushes it up. 0.55 allows for finite-sample slop while still catching any
# difference of practical size - at these sample counts an AUC of 0.55 is far
# outside what chance produces.
AUC_MAX = 0.55


def _load_generated(subset: str, split: str, gen_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(gen_dir, f"{split}_{subset}.txt"),
                     sep=r"\s+", header=None, engine="python")
    df.columns = C.ALL_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def _unit_split(units: np.ndarray, rng: np.random.Generator, frac: float = 0.5):
    u = units.copy()
    rng.shuffle(u)
    cut = int(len(u) * frac)
    return set(u[:cut].tolist()), set(u[cut:].tolist())


def row_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """One cycle = one sample. Cycle index is included because it is part of
    what the released data looks like; excluding it would hide a difference in
    how far through life the rows sit."""
    return np.column_stack([df[cols].to_numpy(float),
                            df["cycle"].to_numpy(float)])


def window_features(df: pd.DataFrame, cols: list[str], window: int) -> np.ndarray:
    """Summary statistics over a sliding window - mean, sd, first-to-last delta
    and lag-1 autocorrelation per channel. These expose temporal structure that
    row-level features cannot."""
    out = []
    vals = df[cols].to_numpy(float)
    for _, idx in df.groupby("unit").indices.items():
        idx = np.sort(idx)
        v = vals[idx]
        if len(v) < window:
            continue
        for i in range(window, len(v) + 1, max(1, window // 3)):
            w = v[i - window:i]
            d = w[1:] - w[:-1]
            sd = w.std(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ac = np.where(sd > 0,
                              ((w[:-1] - w[:-1].mean(0)) * (w[1:] - w[1:].mean(0))).mean(0)
                              / np.where(sd > 0, sd ** 2, 1.0), 0.0)
            out.append(np.concatenate([w.mean(0), sd, w[-1] - w[0],
                                       np.abs(d).mean(0), np.nan_to_num(ac)]))
    return np.asarray(out, float)


def c2st(real: np.ndarray, synth: np.ndarray, real_te: np.ndarray,
         synth_te: np.ndarray, seed: int) -> dict:
    Xtr = np.vstack([real, synth])
    ytr = np.r_[np.zeros(len(real)), np.ones(len(synth))]
    Xte = np.vstack([real_te, synth_te])
    yte = np.r_[np.zeros(len(real_te)), np.ones(len(synth_te))]

    clf = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    acc = float(((p > 0.5).astype(int) == yte).mean())
    # An AUC below 0.5 is still a failure to distinguish, so fold it up.
    return {"auc": auc, "auc_folded": float(max(auc, 1 - auc)), "accuracy": acc,
            "n_train": int(len(Xtr)), "n_test": int(len(Xte))}


def run(subset: str, gen_dir: str, seed: int, repeats: int) -> dict:
    real = P.load_split(subset, "train")
    gen = _load_generated(subset, "train", gen_dir)
    cols = [c for c in C.SENSOR_COLS if c not in P.constant_sensors(real)]

    results = {"row": [], "window": []}
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        r_tr, r_te = _unit_split(real["unit"].unique(), rng)
        g_tr, g_te = _unit_split(gen["unit"].unique(), rng)

        rtr = real[real["unit"].isin(r_tr)]
        rte = real[real["unit"].isin(r_te)]
        gtr = gen[gen["unit"].isin(g_tr)]
        gte = gen[gen["unit"].isin(g_te)]

        results["row"].append(c2st(
            row_features(rtr, cols), row_features(gtr, cols),
            row_features(rte, cols), row_features(gte, cols), seed + rep))
        results["window"].append(c2st(
            window_features(rtr, cols, WINDOW), window_features(gtr, cols, WINDOW),
            window_features(rte, cols, WINDOW), window_features(gte, cols, WINDOW),
            seed + rep))

    summary = {}
    for level, runs in results.items():
        a = np.array([r["auc_folded"] for r in runs])
        summary[level] = {
            "auc_mean": float(a.mean()), "auc_sd": float(a.std(ddof=0)),
            "auc_max": float(a.max()), "runs": runs,
            "pass": bool(a.mean() <= AUC_MAX),
        }

    return {"subset": subset, "window": WINDOW, "repeats": repeats,
            "auc_threshold": AUC_MAX, "levels": summary,
            "verdict": "PASS" if all(v["pass"] for v in summary.values()) else "FAIL"}


def main() -> None:
    ap = argparse.ArgumentParser(description="C-MAPSS classifier two-sample test.")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--gen-dir", default=os.path.join(PROC_DIR, "generated"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    rep = run(args.subset, args.gen_dir, args.seed, args.repeats)
    path = os.path.join(PROC_DIR, f"c2st_{args.subset}.json")
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=2)

    print(f"=== CLASSIFIER TWO-SAMPLE TEST - {args.subset} ===")
    print("can a gradient-boosted discriminator tell real from synthetic?")
    print(f"chance = 0.500, pass if mean AUC <= {AUC_MAX}, "
          f"{args.repeats} unit-disjoint splits\n")
    print(f"{'level':>8} {'AUC mean':>10} {'sd':>7} {'max':>7} {'n_test':>8}  verdict")
    for level, v in rep["levels"].items():
        print(f"{level:>8} {v['auc_mean']:10.4f} {v['auc_sd']:7.4f} {v['auc_max']:7.4f} "
              f"{v['runs'][0]['n_test']:8d}  {'PASS' if v['pass'] else 'FAIL'}")
    print(f"\nVERDICT: {rep['verdict']}")
    print(f"-> {path}")
    sys.exit(0 if rep["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
