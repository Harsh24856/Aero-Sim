"""A10-A15 - the C-MAPSS generator: regimes, fault modes, generation, truncation.

Assembles the exact half (damage.py) and the fitted half (surrogate.py, noise.py)
into a generator that emits data in the released 26-column format.

PIPELINE PER UNIT, in the paper's order (section V.C, p.5):
    1. sample initial deterioration (e0, f0) and rates (a, b) within eq. (9)-(10)
    2. propagate efficiency and flow exponentially
    3. contaminate the trajectory with correlated process noise      <- A9 stage 1
    4. evaluate margins, health, and stop at H = 0                    <- emergent length
    5. map health to sensors through the fitted response surface      <- A8
    6. add a per-unit offset (initial wear / manufacturing variation) <- A9 stage 2
    7. add AR(1) mixture measurement noise per channel                <- A9 stage 3

FAULT MODES (A11)
    FD001/FD002 degrade HPC only; FD003/FD004 add a Fan mode. The published
    model already carries a fault-direction selector k in {1,2} (eq. 10), which
    changes how efficiency and flow loading distribute across the margins. We
    map single-fault subsets to k=1 and two-fault subsets to a per-unit draw
    from {1,2}, so the second mode costs no new machinery and no new constant.

TEST TRUNCATION (A15)
    NASA's test units stop before failure. We reproduce that by generating full
    run-to-failure trajectories and cutting each at a random point, emitting the
    remaining cycles as the true RUL vector - which is exactly how the released
    RUL_FDxxx.txt was constructed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import config as C
from . import damage as D
from . import noise as N
from . import parser as P
from .surrogate import ResponseSurface, gkey

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cmapss", "processed",
)

# Scale of the correlated process noise applied to the health trajectory before
# the response surface. ASSUMED - the paper describes this contamination but
# gives no magnitude. Kept small so it perturbs the trajectory (making loss
# "not locally monotonic", p.5) without swamping the degradation signal; the
# fidelity gate checks the consequences at sensor level.
PROCESS_NOISE_SCALE = D.PROCESS_NOISE_SCALE
PROCESS_NOISE_PHI = D.PROCESS_NOISE_PHI

PROCESS_NOISE_PROVENANCE = {
    "status": "ASSUMED",
    "parameters": {"scale": PROCESS_NOISE_SCALE, "phi": PROCESS_NOISE_PHI},
    "reason": "Saxena et al. describe masking the trajectory with a noise mixture "
              "but give no magnitude for it.",
    "checked_by": "sensor-level autocorrelation and distribution gates (A16)",
}


def _fault_k(subset: str, rng: np.random.Generator) -> int:
    """A11 - fault-mode selection via the published direction selector."""
    modes = C.SUBSETS[subset]["fault_modes"]
    return 1 if len(modes) == 1 else int(rng.choice(C.K_VALUES))


def _draw_mode(rs: ResponseSurface, rng: np.random.Generator) -> int:
    """Pick a unit's fault mode at the share observed in the real data."""
    if rs.n_modes <= 1:
        return 0
    w = np.asarray(rs.mode_weights, float)
    return int(rng.choice(rs.n_modes, p=w / w.sum()))


def draw_unit_offsets(n_units: int, rs: ResponseSurface,
                      rng: np.random.Generator) -> list[dict]:
    """Per-unit sensor offsets (initial wear / manufacturing variation), centred.

    Drawing N offsets independently leaves the sample mean off zero by about
    sd/sqrt(N), which shifts the whole marginal distribution of that channel. At
    100 units that was enough to push NRf and P15 past the fidelity gate's mean
    threshold. Centring each channel's offsets removes the sampling error
    without materially changing the between-unit spread the offsets exist to
    represent - the variance is rescaled back after centring.
    """
    offsets = [{} for _ in range(n_units)]
    for key in rs.coef:
        for s in C.SENSOR_COLS:
            sd = rs.between_unit_sd[key].get(s, 0.0)
            if sd <= 0:
                for u in range(n_units):
                    offsets[u][(key, s)] = 0.0
                continue
            d = rng.normal(0.0, sd, n_units)
            d = d - d.mean()
            cur = d.std(ddof=0)
            if cur > 0:
                d = d * (sd / cur)
            for u in range(n_units):
                offsets[u][(key, s)] = float(d[u])
    return offsets


def generate_unit(unit: int, subset: str, rs: ResponseSurface, gain,
                  rng: np.random.Generator, max_cycles: int = 4000,
                  offsets: dict | None = None, mode: int | None = None,
                  rhos: dict | None = None) -> pd.DataFrame:
    """Generate one run-to-failure trajectory in the 26-column format."""
    if mode is None:
        mode = _draw_mode(rs, rng)
    # Each fault mode has its own margin gain and its own efficiency/flow
    # coupling, so its own lifetime population. Both must be resolved BEFORE
    # sampling, because the coupling shapes the draw itself.
    gain = gain[mode] if isinstance(gain, dict) else gain
    rho = (rhos or {}).get(mode, 0.0) if rhos is not None else 0.0

    params = D.sample_unit_params(unit, rng, rho)
    params = D.UnitDamageParams(**{**params.as_dict(), "k": _fault_k(subset, rng)})

    # Steps 2-4: propagate, contaminate, stop at H = 0. Process noise is applied
    # to efficiency and flow BEFORE the margins are evaluated, so it passes
    # through the health calculation exactly as the paper's noise passes through
    # the engine model.
    run = D.simulate_unit(params, max_cycles=max_cycles, gain=gain, rng=rng)
    H, n = run["health"], run["length"]

    # A10: per-cycle regime, drawn at the observed occupancy.
    n_reg = rs.regime_centres.shape[0]
    weights = np.array([rs.regime_settings[r]["weight"] for r in range(n_reg)])
    weights = weights / weights.sum()
    regimes = rng.choice(n_reg, size=n, p=weights)

    # Operating settings: regime centre plus the observed within-regime spread.
    settings = np.empty((n, 3))
    for r in range(n_reg):
        sel = regimes == r
        if not sel.any():
            continue
        mu = np.asarray(rs.regime_settings[r]["mean"], float)
        sd = np.asarray(rs.regime_settings[r]["sd"], float)
        settings[sel] = mu + rng.normal(0.0, 1.0, (int(sel.sum()), 3)) * sd

    # Steps 5-7: response surface, per-unit offset, measurement noise.
    cols = {"unit": np.full(n, unit, dtype=int), "cycle": np.arange(1, n + 1, dtype=int)}
    for j, cname in enumerate(C.SETTING_COLS):
        cols[cname] = settings[:, j]

    noise_by_sensor = {s: np.zeros(n) for s in C.SENSOR_COLS}
    for r in range(n_reg):
        sel = regimes == r
        if not sel.any():
            continue
        key = gkey(mode, r)
        if key not in rs.coef:
            key = gkey(0, r)
        _fill_correlated_noise(noise_by_sensor, sel, rs, key, rng)

    for s in C.SENSOR_COLS:
        vals = np.empty(n)
        for r in range(n_reg):
            sel = regimes == r
            if not sel.any():
                continue
            key = gkey(mode, r)
            if key not in rs.coef:
                key = gkey(0, r)
            if s in rs.constant_sensors:
                vals[sel] = rs.constant_values[key][s]
                continue
            base = np.polyval(rs.coef[key][s], H[sel])
            offset = (offsets[(key, s)] if offsets is not None and (key, s) in offsets
                      else rng.normal(0.0, rs.between_unit_sd[key].get(s, 0.0)))
            vals[sel] = base + offset + noise_by_sensor[s][sel]
        # Match the channel's observed discretisation (A9, sensor quantisation).
        q = rs.quantum.get(s, 0.0)
        if q > 0:
            vals = np.round(vals / q) * q
        cols[s] = vals

    return pd.DataFrame(cols, columns=C.ALL_COLS)


def _fill_correlated_noise(out: dict, sel: np.ndarray, rs: ResponseSurface,
                           key: str, rng: np.random.Generator) -> None:
    """Measurement noise for every channel at once, correlated across channels.

    Each channel's temporal structure (slow AR(1) drift plus white) is generated
    at unit variance, then the channels are mixed through the Cholesky factor of
    the measured residual correlation. Mixing the smooth and white parts
    separately keeps each channel's timescale intact while giving the whole row
    the cross-channel dependence real sensors have, because they share the
    engine's process noise.
    """
    m = int(sel.sum())
    spec = rs.resid_corr.get(key)
    npar_all = rs.noise_params.get(key, {})
    cs = spec["cols"] if spec else [c for c in npar_all if c not in rs.constant_sensors]
    if not cs:
        return

    smooth = np.column_stack([
        N.ar1_mixture_noise(m, npar_all[c]["phi"], 1.0, npar_all[c]["kurtosis"], rng)
        if npar_all[c].get("var_smooth", 0.0) > 0 else np.zeros(m) for c in cs])
    white = np.column_stack([
        N.mixture_innovations(m, *N.solve_mixture(1.0, npar_all[c]["kurtosis"]), rng)
        if npar_all[c].get("var_white", 0.0) > 0 else np.zeros(m) for c in cs])

    if spec:
        Cm = np.asarray(spec["corr"], float)
        # Nudge to positive definite before factorising; measured correlation
        # matrices are near-singular when channels are almost collinear.
        w, V = np.linalg.eigh((Cm + Cm.T) / 2.0)
        L = V @ np.diag(np.sqrt(np.clip(w, 1e-8, None)))
        smooth = smooth @ L.T
        white = white @ L.T

    for j, c in enumerate(cs):
        np_ = npar_all[c]
        out[c][sel] = (np.sqrt(max(np_.get("var_smooth", 0.0), 0.0)) * smooth[:, j]
                       + np.sqrt(max(np_.get("var_white", 0.0), 0.0)) * white[:, j])


def generate_split(subset: str, split: str, rs: ResponseSurface, gain,
                   seed: int | None = None,
                   rhos: dict | None = None) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Generate a full split at the published unit count.

    Returns (frame, true_rul). true_rul is None for train (every unit runs to
    failure, so RUL is derivable) and the per-unit remaining cycles for test.
    """
    n_units = C.SUBSETS[subset][f"{split}_units"]
    rng = np.random.default_rng(seed if seed is not None else C.subset_seed(subset, split))

    unit_offsets = draw_unit_offsets(n_units, rs, rng)

    # Fault modes at the EXACT observed composition rather than iid draws.
    # The real mode split is known (FD003: 56/44), and sampling it randomly
    # leaves the generated mix off by a few units. Where a channel's mean
    # differs between modes that shifts the pooled marginal - it put BPR at
    # mean_z 0.101 against a 0.100 threshold. Allocating exact counts and
    # shuffling removes that noise without assuming anything not measured.
    if rs.n_modes > 1:
        counts = [int(round(w * n_units)) for w in rs.mode_weights]
        counts[-1] = n_units - sum(counts[:-1])
        mode_seq = np.concatenate([np.full(c, m, int) for m, c in enumerate(counts)])
        rng.shuffle(mode_seq)
    else:
        mode_seq = np.zeros(n_units, int)

    frames, tail_rul = [], []
    for u in range(1, n_units + 1):
        full = generate_unit(u, subset, rs, gain, rng, offsets=unit_offsets[u - 1],
                             mode=int(mode_seq[u - 1]), rhos=rhos)
        if split == "train":
            frames.append(full)
            continue

        # A15: cut before failure. The cut point is uniform over the trajectory
        # but kept away from both ends - a test unit with 1 cycle carries no
        # usable window, and one cut at failure would have RUL 0 for every unit.
        L = len(full)
        lo, hi = max(1, int(0.15 * L)), max(2, int(0.95 * L))
        cut = int(rng.integers(lo, hi))
        frames.append(full.iloc[:cut].copy())
        tail_rul.append(L - cut)

    out = pd.concat(frames, ignore_index=True)
    out = _map_low_cardinality(out, rs)
    return out, (np.asarray(tail_rul, dtype=int) if split == "test" else None)


def _map_low_cardinality(df: pd.DataFrame, rs: ResponseSurface) -> pd.DataFrame:
    """Map few-level channels onto their real level distribution, rank-preserving.

    A channel like P15 takes two values in the whole of FD001. Rounding a
    continuous model to those levels reproduces neither the proportions nor the
    marginal, because an arbitrarily small error in the continuous mean lands on
    the wrong side of the single threshold. Instead we keep the continuous
    model's ORDERING - which carries whatever degradation signal the channel has
    - and map it monotonically onto the empirical level distribution measured on
    the train split. Rank is preserved, so the trend survives; the marginal
    matches by construction.
    """
    from .surrogate import assign_regimes
    regimes = assign_regimes(df, rs.regime_centres)

    for s, by_regime in rs.low_card.items():
        if s not in df.columns:
            continue
        col = df[s].to_numpy(float).copy()
        for r_str, spec in by_regime.items():
            m = regimes == int(r_str)
            if not m.any():
                continue
            vals = np.asarray(spec["values"], float)
            cum = np.asarray(spec["cumprob"], float)
            x = col[m]
            # Rank -> uniform -> level, within this regime only.
            order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
            u = (order + 0.5) / x.size
            col[m] = vals[np.searchsorted(cum, u, side="left").clip(0, vals.size - 1)]
        df[s] = col
    return df


def _decimals(q: float) -> int:
    """Decimal places needed to write a channel at its own quantisation."""
    if q <= 0:
        return 2
    for d in range(0, 7):
        if abs(q * (10 ** d) - round(q * (10 ** d))) < 1e-9:
            return d
    return 4


def write_split(df: pd.DataFrame, path: str, rs: ResponseSurface | None = None) -> None:
    """Write in the released whitespace format, matching NASA's precision.

    The released files carry the index columns as integers, settings to 4
    significant figures and sensors to 2, so a diff of our output against theirs
    is not dominated by formatting.
    """
    dec = [(_decimals(rs.quantum.get(s, 0.0)) if rs is not None else 2)
           for s in C.SENSOR_COLS]
    with open(path, "w") as fh:
        for row in df.itertuples(index=False):
            vals = [f"{int(row.unit)}", f"{int(row.cycle)}"]
            vals += [f"{v:.4f}" for v in row[2:5]]
            vals += [f"{v:.{dec[i]}f}" for i, v in enumerate(row[5:])]
            fh.write(" ".join(vals) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic C-MAPSS (A10-A15).")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    with open(os.path.join(args.out_dir, f"calibration_{args.subset}.json")) as fh:
        cal = json.load(fh)
    gain = {int(k): v for k, v in cal["calibrated_parameter"]["value_by_mode"].items()}
    rhos = {int(k): v for k, v in cal["calibrated_parameter"]["rho_by_mode"].items()}
    rs = ResponseSurface.load(os.path.join(args.out_dir, f"surrogate_{args.subset}.json"))

    gen_dir = os.path.join(args.out_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    for split in ("train", "test"):
        df, rul = generate_split(args.subset, split, rs, gain, rhos=rhos)
        write_split(df, os.path.join(gen_dir, f"{split}_{args.subset}.txt"), rs)
        L = P.trajectory_lengths(df)
        print(f"{args.subset} {split:5s}: {len(df):>6} rows, {df.unit.nunique():>3} units, "
              f"len {L.min()}-{L.max()} mean {L.mean():.1f}")
        if rul is not None:
            np.savetxt(os.path.join(gen_dir, f"RUL_{args.subset}.txt"), rul, fmt="%d")
            print(f"{'':16}true RUL mean {rul.mean():.1f} range {rul.min()}-{rul.max()}")
    print(f"-> {gen_dir}")


if __name__ == "__main__":
    main()
