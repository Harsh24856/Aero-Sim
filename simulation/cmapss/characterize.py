"""A5 - characterise the released data as the replication target.

Everything the generator must later reproduce is measured here ONCE, from the
real data, and written to disk as a frozen target. The fidelity gate (A16) then
compares generated output against this file rather than recomputing from the
reference each time - which keeps the target fixed and makes it obvious if it is
ever moved.

Run:
    validation/venv/bin/python3 -m simulation.cmapss.characterize
    validation/venv/bin/python3 -m simulation.cmapss.characterize --subset FD001
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import config as C
from . import parser as P

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cmapss", "processed",
)


def recover_operating_regimes(df: pd.DataFrame, n_conditions: int, seed: int = 0):
    """Cluster the 3 operating-setting columns into the published regime count.

    The paper states six operating conditions for FD002/FD004 but never lists
    them; they have to be recovered from the settings columns. They separate
    cleanly, so k-means with the known k is sufficient - and we verify the
    separation rather than trusting it (see `inertia` and `min_pair_dist`).

    Returns (labels, centres, diagnostics).
    """
    X = df[C.SETTING_COLS].to_numpy(float)
    if n_conditions == 1:
        # Single-condition subsets still carry small jitter in settings; forcing
        # k=1 keeps the interface uniform and records that jitter as spread.
        centre = X.mean(axis=0, keepdims=True)
        labels = np.zeros(len(X), dtype=int)
        diag = {"inertia": float(((X - centre) ** 2).sum()), "min_pair_dist": None}
        return labels, centre, diag

    km = KMeans(n_clusters=n_conditions, n_init=10, random_state=seed).fit(X)
    centres = km.cluster_centers_
    # Smallest distance between any two regime centres. If regimes are genuinely
    # distinct this is large relative to within-cluster spread.
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    diag = {"inertia": float(km.inertia_), "min_pair_dist": float(d.min())}
    return km.labels_, centres, diag


def characterize(subset: str, data_dir: str | None = None) -> dict:
    """Measure every statistic the generator will be held to."""
    meta = C.SUBSETS[subset]
    train = P.load_split(subset, "train", data_dir)
    test = P.load_split(subset, "test", data_dir)
    true_rul = P.load_test_rul(subset, data_dir)

    lengths = P.trajectory_lengths(train)
    test_lengths = P.trajectory_lengths(test)
    const = P.constant_sensors(train)
    informative = [c for c in C.SENSOR_COLS if c not in const]

    labels, centres, regime_diag = recover_operating_regimes(train, meta["n_conditions"])

    # Per-sensor marginals over the whole train split.
    sensor_stats = {}
    for c in C.SENSOR_COLS:
        v = train[c].to_numpy(float)
        sensor_stats[c] = {
            "mean": float(v.mean()), "std": float(v.std(ddof=0)),
            "min": float(v.min()), "max": float(v.max()),
            "p01": float(np.percentile(v, 1)), "p50": float(np.percentile(v, 50)),
            "p99": float(np.percentile(v, 99)),
            "constant": c in const,
        }

    # Correlation over informative sensors only - including flat channels makes
    # the matrix singular and the Frobenius comparison meaningless.
    corr = train[informative].corr().to_numpy()

    # Degradation trend: mean sensor value as a function of cycles-to-failure.
    # This is the signal that must survive generation; matching marginals while
    # flattening this is exactly the failure mode the TSTR gate (A17) exists to
    # catch, and this curve is how we see it directly.
    tr = P.add_train_rul(train, cap=None)
    ctf_bins = [0, 10, 25, 50, 75, 100, 150, 200, 10**9]
    tr["_ctf_bin"] = pd.cut(tr["RUL"], bins=ctf_bins, right=False)
    trend = (
        tr.groupby("_ctf_bin", observed=True)[informative]
        .mean()
        .rename_axis("cycles_to_failure")
    )

    return {
        "subset": subset,
        "published": {
            "train_units": meta["train_units"], "test_units": meta["test_units"],
            "n_conditions": meta["n_conditions"], "fault_modes": list(meta["fault_modes"]),
        },
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "trajectory_lengths": {
            "values": lengths.tolist(),
            "mean": float(lengths.mean()), "std": float(lengths.std(ddof=0)),
            "min": int(lengths.min()), "max": int(lengths.max()),
            "median": float(np.median(lengths)),
        },
        "test_trajectory_lengths": {
            "mean": float(test_lengths.mean()), "std": float(test_lengths.std(ddof=0)),
            "min": int(test_lengths.min()), "max": int(test_lengths.max()),
        },
        "test_true_rul": {
            "mean": float(true_rul.mean()), "std": float(true_rul.std(ddof=0)),
            "min": int(true_rul.min()), "max": int(true_rul.max()),
        },
        "constant_sensors": const,
        "informative_sensors": informative,
        "sensor_stats": sensor_stats,
        "correlation": {"columns": informative, "matrix": corr.tolist()},
        "operating_regimes": {
            "n": meta["n_conditions"],
            "centres": centres.tolist(),
            "counts": np.bincount(labels, minlength=meta["n_conditions"]).tolist(),
            **regime_diag,
        },
        "degradation_trend": {
            "bins": [str(b) for b in trend.index],
            "columns": informative,
            "values": trend.to_numpy().tolist(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Characterise released C-MAPSS data (A5).")
    ap.add_argument("--subset", default=None, help="one of FD001..FD004; default all")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    subsets = [args.subset] if args.subset else list(C.SUBSETS)
    os.makedirs(args.out_dir, exist_ok=True)

    for s in subsets:
        rep = characterize(s, args.data_dir)
        path = os.path.join(args.out_dir, f"target_{s}.json")
        with open(path, "w") as fh:
            json.dump(rep, fh, indent=2)

        L = rep["trajectory_lengths"]
        r = rep["operating_regimes"]
        print(f"{s}: {rep['train_rows']:>6} train rows, {rep['published']['train_units']:>3} units")
        print(f"   traj len  mean {L['mean']:6.1f}  sd {L['std']:5.1f}  range {L['min']}-{L['max']}")
        print(f"   sensors   {len(rep['informative_sensors'])} informative, "
              f"{len(rep['constant_sensors'])} constant {rep['constant_sensors']}")
        sep = "n/a" if r["min_pair_dist"] is None else f"{r['min_pair_dist']:.3f}"
        print(f"   regimes   k={r['n']}  counts={r['counts']}  min centre sep={sep}")
        print(f"   -> {path}")


if __name__ == "__main__":
    main()
