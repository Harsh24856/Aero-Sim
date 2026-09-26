"""Infer each real unit's fault mode, which the released data does not label.

WHY THIS IS NEEDED
    FD003 and FD004 contain two fault modes (HPC degradation and Fan
    degradation); FD001 and FD002 contain one. The released files do not say
    which unit has which. Fitting a single response surface across units of both
    modes averages two distinct sensor responses into one, and the generator
    then cannot reproduce either.

    The symptom is visible in the lifetime distribution: FD003's real spread is
    cv 0.348 against FD001's 0.224, because two modes produce two lifetime
    populations. Generating from a single averaged surface gave cv 0.169.

HOW THE MODE IS INFERRED
    Each unit is summarised by its degradation SIGNATURE: for every informative
    sensor, how far that sensor moved from early life to late life, in units of
    its own spread. Two different physical faults drive different sensors in
    different directions and by different amounts, so units separate in this
    space. k-means with k = the published number of fault modes then labels
    them.

    Sensors are standardised within operating regime first, so in the
    six-condition subsets the signature reflects degradation rather than which
    regimes that unit happened to fly.

WHAT THIS IS NOT
    It is not a claim to have recovered NASA's true fault labels - those were
    never released and cannot be checked. It is a data-driven split into the
    published number of groups, justified by whether it lets the generator
    reproduce the real data (gates A16/A17). If the split were meaningless the
    gates would not improve, so the gates are the test of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import config as C


def unit_signatures(train: pd.DataFrame, cols: list[str],
                    regimes: np.ndarray | None = None,
                    head_frac: float = 0.25,
                    tail_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit degradation signature: late-life minus early-life, per sensor.

    Returns (unit_ids, signatures) with signatures shaped (n_units, n_sensors).
    """
    vals = train[cols].to_numpy(float).copy()

    # Standardise within regime so the signature is about degradation, not about
    # which operating points this unit happened to see.
    if regimes is not None:
        for r in np.unique(regimes):
            m = regimes == r
            mu, sd = vals[m].mean(0), vals[m].std(0)
            vals[m] = (vals[m] - mu) / np.where(sd > 0, sd, 1.0)
    else:
        mu, sd = vals.mean(0), vals.std(0)
        vals = (vals - mu) / np.where(sd > 0, sd, 1.0)

    units = np.sort(train["unit"].unique())
    sig = np.zeros((units.size, len(cols)))
    idx = train.groupby("unit").indices
    for i, u in enumerate(units):
        rows = np.sort(idx[u])
        n = rows.size
        h = max(1, int(round(head_frac * n)))
        t = max(1, int(round(tail_frac * n)))
        sig[i] = vals[rows[-t:]].mean(0) - vals[rows[:h]].mean(0)
    return units, sig


def infer_modes(train: pd.DataFrame, subset: str, cols: list[str],
                regimes: np.ndarray | None = None,
                seed: int = 0) -> tuple[dict[int, int], dict]:
    """Label every train unit with a fault mode in [0, n_modes).

    Returns (unit -> mode, diagnostics). For single-fault subsets every unit is
    mode 0 and no clustering is performed.
    """
    n_modes = len(C.SUBSETS[subset]["fault_modes"])
    units, sig = unit_signatures(train, cols, regimes)

    if n_modes == 1:
        return ({int(u): 0 for u in units},
                {"n_modes": 1, "method": "single published fault mode; no clustering"})

    km = KMeans(n_clusters=n_modes, n_init=20, random_state=seed).fit(sig)
    labels = km.labels_

    # Order modes by population so labels are stable across runs.
    order = np.argsort(-np.bincount(labels, minlength=n_modes))
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(l)] for l in labels])

    lengths = train.groupby("unit")["cycle"].max().loc[units].to_numpy()
    diag = {
        "n_modes": n_modes,
        "method": "k-means on per-unit degradation signatures",
        "counts": np.bincount(labels, minlength=n_modes).tolist(),
        "mean_length_by_mode": [float(lengths[labels == m].mean())
                                if (labels == m).any() else 0.0
                                for m in range(n_modes)],
        "sd_length_by_mode": [float(lengths[labels == m].std(ddof=0))
                              if (labels == m).any() else 0.0
                              for m in range(n_modes)],
        "silhouette": _silhouette(sig, labels),
    }
    return {int(u): int(l) for u, l in zip(units, labels)}, diag


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Cluster separation, for the record. Reported, not gated on."""
    try:
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")
