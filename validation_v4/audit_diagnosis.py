#!/usr/bin/env python3
"""Settle the diagnosis metrics with sklearn, independent of Keras.

The saved test result reported macro F1 0.5050 and AUC 0.4129 on the same
predictions. Those cannot both be right: AUC is threshold-free and strictly
easier than F1 at a fixed threshold, so a head scoring 0.50 macro F1 must have
AUC well above 0.50. One of the two is miscomputed, and a number nobody can
reproduce is worth nothing in a report.

This recomputes everything from the checkpoint with sklearn - per mode, no
Keras accumulator, no threshold ambiguity - and prints the Keras figure beside
it so the disagreement is visible rather than argued about.

RUN
    validation_v4/audit_diagnosis.py --engine 914
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import tf_data_pipeline as P
import train_common as C


def collect(engine: str, max_batches: int | None) -> tuple[np.ndarray, np.ndarray]:
    model = keras.models.load_model(C.ckpt_path(engine, "diagnosis"),
                                    compile=False, safe_mode=False)
    ds = C.datasets(engine)["test"]
    T, Y = [], []
    for i, (xb, yb) in enumerate(ds):
        if max_batches and i >= max_batches:
            break
        Y.append(model.predict(xb, verbose=0)["y_fault_mode"])
        T.append(yb["y_fault_mode"].numpy())
    return np.concatenate(T), np.concatenate(Y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="914")
    ap.add_argument("--max-batches", type=int, default=400)
    args = ap.parse_args()

    print(f"loading {C.ckpt_path(args.engine, 'diagnosis')}")
    sev, pred = collect(args.engine, args.max_batches)
    truth = (sev >= P.DETECTION_THRESHOLD).astype(int)
    n, k = truth.shape
    print(f"{n:,} test windows, {k} modes, "
          f"{100 * truth.mean():.2f}% of cells positive\n")

    print(f"{'mode':24s} {'n_pos':>7} {'AUC':>7} {'prec':>7} {'rec':>7} {'F1':>7}")
    aucs, f1s, scored = [], [], 0
    for i, name in enumerate(P.FAULT_MODES):
        t, p = truth[:, i], pred[:, i]
        npos = int(t.sum())
        if npos == 0:
            print(f"{name:24s} {npos:>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7}"
                  "   no positives, excluded")
            continue
        auc = roc_auc_score(t, p)
        pr, rc, f1, _ = precision_recall_fscore_support(
            t, (p >= 0.5).astype(int), average="binary", zero_division=0)
        aucs.append(auc)
        f1s.append(f1)
        scored += 1
        print(f"{name:24s} {npos:>7,} {auc:>7.4f} {pr:>7.4f} {rc:>7.4f} {f1:>7.4f}")

    print()
    print(f"macro over {scored} modes with positives:")
    print(f"  sklearn macro AUC : {np.mean(aucs):.4f}")
    print(f"  sklearn macro F1  : {np.mean(f1s):.4f}")

    saved = C.result_path(args.engine, "diagnosis")
    if os.path.exists(saved):
        with open(saved) as fh:
            rep = json.load(fh)["test"]
        print()
        print("as saved by run.py (Keras metrics):")
        print(f"  keras AUC         : {rep.get('auc', float('nan')):.4f}")
        print(f"  keras macro_f1    : {rep.get('macro_f1', float('nan')):.4f}")
        print()
        da = abs(np.mean(aucs) - rep.get("auc", np.nan))
        df = abs(np.mean(f1s) - rep.get("macro_f1", np.nan))
        print(f"  |AUC difference|  : {da:.4f}  "
              f"{'<-- Keras AUC is wrong' if da > 0.05 else 'agrees'}")
        print(f"  |F1 difference|   : {df:.4f}  "
              f"{'<-- MacroF1 is wrong' if df > 0.05 else 'agrees'}")


if __name__ == "__main__":
    main()
