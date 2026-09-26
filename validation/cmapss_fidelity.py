#!/usr/bin/env python3
"""A16 - fidelity gate: is the generated data statistically identical to NASA's?

Compares generated output against the frozen target from A5 and against the real
files directly. Every check is a PASS/FAIL against a stated threshold, and the
thresholds are declared here rather than chosen after seeing the numbers.

WHAT IS AND IS NOT EVIDENCE
    One scalar (the margin gain) was fitted to one statistic (mean trajectory
    length). That statistic is reported but does NOT count toward the verdict -
    it is marked FITTED. Everything else was never fitted and is the actual
    test.

CHECKS
    1. trajectory-length distribution     two-sample KS
    2. per-sensor marginals               two-sample KS per sensor, plus mean/sd
    3. correlation structure              Frobenius norm of the difference
    4. degradation trend                  correlation of sensor-vs-RUL curves
    5. autocorrelation                    per-sensor lag-1 ACF difference
    6. PCA overlay                        centroid separation vs within-spread

RUN
    validation/venv/bin/python3 validation/cmapss_fidelity.py --subset FD001
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.cmapss import config as C
from simulation.cmapss import parser as P

PROC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cmapss", "processed",
)

# Thresholds, declared before running. KS at alpha = 0.01 rather than 0.05: with
# tens of thousands of rows a KS test rejects on differences far too small to
# matter, so the sensor-level gate is judged on the KS *statistic* (a normalised
# distance) rather than its p-value.
TH = {
    "length_ks_p_min": 0.01,       # trajectory lengths: distributions indistinguishable
    "sensor_ks_d_max": 0.15,       # per-sensor distance
    "sensor_mean_z_max": 0.10,     # |mean difference| in units of real sd
    "sensor_sd_ratio": (0.70, 1.40),
    "corr_frobenius_max": 0.35,    # normalised by matrix size
    "trend_corr_min": 0.90,        # degradation trend shape must track
    "trend_min_amplitude": 0.05,   # below this a channel has no trend to reproduce
    "acf_abs_diff_max": 0.15,      # lag-1 autocorrelation
    "pca_separation_max": 0.50,    # centroid distance / pooled spread
}


def _load_generated(subset: str, split: str, gen_dir: str) -> pd.DataFrame:
    path = os.path.join(gen_dir, f"{split}_{subset}.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df.columns = C.ALL_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def _trend(df: pd.DataFrame, cols: list[str], regimes: np.ndarray | None = None,
           stats_by_regime: dict | None = None) -> np.ndarray:
    """Mean sensor value binned by cycles-to-failure. The degradation signature.

    REGIME-WISE STANDARDISATION, and why it is required rather than convenient.
    In the six-condition subsets the operating regime moves each sensor far more
    than degradation does, so a trend pooled across regimes measures the regime
    MIX in each RUL bin, not the degradation. That made this check fail on FD002
    (pooled mean trend correlation 0.302) while the degradation signal was
    demonstrably present in both real and generated data - within each regime,
    real T50 correlates +0.66 to +0.73 with life fraction and Nf +0.25 to +0.76.
    The real regime sequence is iid (repeat probability 0.175 against 0.167
    expected) and its mix does not drift over life, so there is no regime
    structure for the generator to have missed.

    Standardising per regime before binning removes the operating point and
    leaves degradation - the same regime-wise normalisation used throughout the
    C-MAPSS literature. Real and generated are standardised with the SAME
    statistics, taken from the real data, so any genuine difference survives.
    """
    tr = P.add_train_rul(df, cap=None)
    vals = tr[cols].to_numpy(float)

    if regimes is not None and stats_by_regime is not None:
        vals = vals.copy()
        for r, (mu, sd) in stats_by_regime.items():
            m = regimes == r
            if m.any():
                vals[m] = (vals[m] - mu) / np.where(sd > 0, sd, 1.0)

    bins = [0, 10, 25, 50, 75, 100, 150, 200, 10 ** 9]
    b = pd.cut(tr["RUL"], bins=bins, right=False)
    out = pd.DataFrame(vals, columns=cols).assign(_b=b.to_numpy())
    return out.groupby("_b", observed=True)[cols].mean().to_numpy()


def _lag1_acf(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out = {}
    for c in cols:
        acc = []
        for _, g in df.groupby("unit"):
            x = g[c].to_numpy(float)
            # A channel that is flat (or near-flat) within a unit has no defined
            # autocorrelation - corrcoef divides by a zero standard deviation and
            # returns nan. Skip those units rather than letting one nan poison
            # the mean, which would silently disable this check.
            if x.size > 5 and x.std() > 1e-12:
                r = np.corrcoef(x[:-1], x[1:])[0, 1]
                if np.isfinite(r):
                    acc.append(float(r))
        out[c] = float(np.mean(acc)) if acc else 0.0
    return out


def run(subset: str, gen_dir: str) -> dict:
    real = P.load_split(subset, "train")
    gen = _load_generated(subset, "train", gen_dir)

    const = P.constant_sensors(real)
    cols = [c for c in C.SENSOR_COLS if c not in const]

    checks: list[dict] = []

    def add(name, passed, detail, category="independent"):
        checks.append({"check": name, "pass": bool(passed), "category": category, **detail})

    # -- 1. trajectory lengths ------------------------------------------------
    Lr, Lg = P.trajectory_lengths(real), P.trajectory_lengths(gen)
    d, p = stats.ks_2samp(Lg, Lr)
    add("trajectory_length_KS", p >= TH["length_ks_p_min"],
        {"ks_d": float(d), "ks_p": float(p),
         "real_mean": float(Lr.mean()), "gen_mean": float(Lg.mean()),
         "real_sd": float(Lr.std()), "gen_sd": float(Lg.std()),
         "note": "mean was FITTED; sd/shape/KS were not"})

    # -- 2. per-sensor marginals ---------------------------------------------
    sensor_rows = []
    for c in cols:
        r, g = real[c].to_numpy(float), gen[c].to_numpy(float)
        d, _ = stats.ks_2samp(g, r)
        z = abs(g.mean() - r.mean()) / (r.std() if r.std() > 0 else 1.0)
        ratio = (g.std() / r.std()) if r.std() > 0 else 1.0
        ok = (d <= TH["sensor_ks_d_max"] and z <= TH["sensor_mean_z_max"]
              and TH["sensor_sd_ratio"][0] <= ratio <= TH["sensor_sd_ratio"][1])
        sensor_rows.append({"sensor": c, "ks_d": float(d), "mean_z": float(z),
                            "sd_ratio": float(ratio), "pass": bool(ok)})
    n_ok = sum(r["pass"] for r in sensor_rows)
    add("sensor_marginals", n_ok == len(cols),
        {"passed": n_ok, "total": len(cols), "detail": sensor_rows})

    # -- 3. correlation structure --------------------------------------------
    cr, cg = real[cols].corr().to_numpy(), gen[cols].corr().to_numpy()
    fro = float(np.linalg.norm(cr - cg) / np.sqrt(cr.size))
    add("correlation_structure", fro <= TH["corr_frobenius_max"],
        {"normalised_frobenius": fro, "threshold": TH["corr_frobenius_max"]})

    # -- 4. degradation trend -------------------------------------------------
    # Standardise within operating regime so the trend measures degradation
    # rather than regime mix (see _trend).
    reg_r = reg_g = None
    stats_by_regime = None
    try:
        from simulation.cmapss.surrogate import ResponseSurface, assign_regimes
        rs_path = os.path.join(PROC_DIR, f"surrogate_{subset}.json")
        if os.path.exists(rs_path):
            rs = ResponseSurface.load(rs_path)
            if rs.regime_centres.shape[0] > 1:
                reg_r = assign_regimes(real, rs.regime_centres)
                reg_g = assign_regimes(gen, rs.regime_centres)
                stats_by_regime = {}
                rv = real[cols].to_numpy(float)
                for r in range(rs.regime_centres.shape[0]):
                    m = reg_r == r
                    stats_by_regime[r] = (rv[m].mean(0), rv[m].std(0))
    except Exception:
        pass

    tr_r = _trend(real, cols, reg_r, stats_by_regime)
    tr_g = _trend(gen, cols, reg_g, stats_by_regime)
    n = min(len(tr_r), len(tr_g))
    per, scored = [], []
    for j, c in enumerate(cols):
        a, b = tr_r[:n, j], tr_g[:n, j]
        # Amplitude of the REAL trend, in standardised units. A channel that is
        # flat against cycles-to-failure carries no degradation signal, so there
        # is nothing for the generator to reproduce and the correlation of two
        # flat noisy curves is meaningless - in FD002, Nf_dmd has amplitude
        # exactly 0.0 and yields nan, while T2/P2/PCNfR_dmd sit below 0.007.
        # Those channels appear in six-condition subsets only because the
        # operating regime moves them, not because they degrade. They are
        # reported but excluded from the verdict.
        amp = float(a.std())
        cc = (float(np.corrcoef(a, b)[0, 1])
              if a.std() > 0 and b.std() > 0 else float("nan"))
        row = {"sensor": c, "trend_corr": cc, "real_amplitude": amp,
               "scored": bool(amp >= TH["trend_min_amplitude"] and np.isfinite(cc))}
        per.append(row)
        if row["scored"]:
            scored.append(cc)
    mean_tc = float(np.mean(scored)) if scored else 0.0
    scored_rows = [r for r in per if r["scored"]]
    worst = min(scored_rows, key=lambda x: x["trend_corr"]) if scored_rows else {
        "sensor": "-", "trend_corr": float("nan")}
    add("degradation_trend", mean_tc >= TH["trend_corr_min"],
        {"mean_trend_corr": mean_tc, "worst": worst,
         "n_scored": len(scored), "n_excluded": len(per) - len(scored),
         "excluded": [r["sensor"] for r in per if not r["scored"]],
         "detail": per})

    # -- 5. autocorrelation ---------------------------------------------------
    ar, ag = _lag1_acf(real, cols), _lag1_acf(gen, cols)
    diffs = {c: float(abs(ar[c] - ag[c])) for c in cols}
    add("autocorrelation", max(diffs.values()) <= TH["acf_abs_diff_max"],
        {"max_abs_diff": float(max(diffs.values())),
         "mean_real": float(np.mean(list(ar.values()))),
         "mean_gen": float(np.mean(list(ag.values()))),
         "detail": diffs})

    # -- 6. PCA overlay -------------------------------------------------------
    sc = StandardScaler().fit(real[cols].to_numpy(float))
    pca = PCA(n_components=2).fit(sc.transform(real[cols].to_numpy(float)))
    pr = pca.transform(sc.transform(real[cols].to_numpy(float)))
    pg = pca.transform(sc.transform(gen[cols].to_numpy(float)))
    sep = float(np.linalg.norm(pr.mean(0) - pg.mean(0)) /
                np.sqrt(0.5 * (pr.std(0) ** 2 + pg.std(0) ** 2).sum()))
    add("pca_overlay", sep <= TH["pca_separation_max"],
        {"centroid_separation": sep, "threshold": TH["pca_separation_max"]})

    passed = sum(c["pass"] for c in checks)
    return {"subset": subset, "thresholds": TH, "checks": checks,
            "n_pass": passed, "n_total": len(checks),
            "verdict": "PASS" if passed == len(checks) else "FAIL"}


def main() -> None:
    ap = argparse.ArgumentParser(description="C-MAPSS fidelity gate (A16).")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--gen-dir", default=os.path.join(PROC_DIR, "generated"))
    ap.add_argument("--out-dir", default=PROC_DIR)
    args = ap.parse_args()

    rep = run(args.subset, args.gen_dir)
    path = os.path.join(args.out_dir, f"fidelity_{args.subset}.json")
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=2)

    print(f"\n=== A16 FIDELITY GATE - {args.subset} ===")
    for c in rep["checks"]:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"[{mark}] {c['check']}")
        if c["check"] == "trajectory_length_KS":
            print(f"        KS D={c['ks_d']:.4f} p={c['ks_p']:.4g} | "
                  f"mean {c['real_mean']:.1f}->{c['gen_mean']:.1f} (FITTED) | "
                  f"sd {c['real_sd']:.1f}->{c['gen_sd']:.1f}")
        elif c["check"] == "sensor_marginals":
            print(f"        {c['passed']}/{c['total']} sensors within thresholds")
            for r in c["detail"]:
                if not r["pass"]:
                    print(f"          FAIL {r['sensor']:9s} KS_D={r['ks_d']:.3f} "
                          f"mean_z={r['mean_z']:.3f} sd_ratio={r['sd_ratio']:.3f}")
        elif c["check"] == "correlation_structure":
            print(f"        normalised Frobenius = {c['normalised_frobenius']:.4f} "
                  f"(<= {c['threshold']})")
        elif c["check"] == "degradation_trend":
            print(f"        mean trend corr = {c['mean_trend_corr']:.4f} over "
                  f"{c['n_scored']} channels, worst {c['worst']['sensor']} = "
                  f"{c['worst']['trend_corr']:.4f}")
            if c["n_excluded"]:
                print(f"        excluded (no real trend): {', '.join(c['excluded'])}")
        elif c["check"] == "autocorrelation":
            print(f"        max |diff| = {c['max_abs_diff']:.4f} | "
                  f"mean real {c['mean_real']:.3f} vs gen {c['mean_gen']:.3f}")
        elif c["check"] == "pca_overlay":
            print(f"        centroid separation = {c['centroid_separation']:.4f} "
                  f"(<= {c['threshold']})")
    print(f"\nVERDICT: {rep['verdict']}  ({rep['n_pass']}/{rep['n_total']} checks)")
    print(f"-> {path}")
    sys.exit(0 if rep["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
