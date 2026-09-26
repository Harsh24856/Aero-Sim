"""Physics-driven generation: damage model -> health modifiers -> turbofan cycle.

This is the architecture the paper describes, and it replaces the fitted
response surface. The damage model (damage.py) is unchanged and still produces
efficiency, flow and health exactly as published; what changes is what those
drive. Previously they fed a polynomial regression fitted to the released
sensors. Now they set the thirteen health modifiers of Table 1, and the sensors
come out of a thermodynamic cycle.

WHAT THE FAULT MODES BECOME
    Under the surrogate, the two fault modes of FD003/FD004 were clusters
    inferred from degradation signatures - defensible but abstract. Here they
    are what the paper says they are: HPC degradation and Fan degradation,
    driving different components of the same engine. The mode selector stops
    being a label and becomes a physical choice of which modifiers move.

WHAT IS STILL FITTED, AND WHY IT HAS TO BE
    The cycle gives the MEAN sensor reading for a given health and operating
    point. It says nothing about measurement noise, which is a property of the
    instrumentation rather than the engine. So the noise model is still measured
    from the real data - but now as residuals about the PHYSICS prediction
    rather than about a regression, which is the honest place for it. All of the
    noise machinery carries over unchanged: smooth/white decomposition, the
    within-window timescale, the two-component mixture and the cross-channel
    correlation.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import damage as D
from . import noise as N
from . import parser as P
from . import turbofan as TF
from .surrogate import assign_health, assign_regimes, gkey, _raw_lag1_acf

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cmapss", "processed",
)


# ---------------------------------------------------------------------------
# Health -> the thirteen modifiers.
# ---------------------------------------------------------------------------

def modifiers_from_health(health: float, fault_mode: int = 0) -> dict:
    """Map the damage model's health index onto Table 1's health modifiers.

    Health runs 1 (new) to 0 (failed), so `1 - health` is the fraction of the
    engine's life consumed in margin terms. At failure the deterioration equals
    the level fitted against the real end-of-life data in turbofan.py - 2.67%
    efficiency and 1.18% flow - which is a physically ordinary amount for a worn
    compressor and consistent with the paper's 15% stall-margin criterion.

    The fault mode chooses WHICH component degrades, which is what the published
    fault modes actually are:
        mode 0  HPC degradation   (FD001, FD002, and the larger half of FD003/4)
        mode 1  Fan degradation   (the second mode of FD003 and FD004)
    """
    frac = float(np.clip(1.0 - health, 0.0, 1.0))
    mods = TF.healthy()
    eff = TF.FAILURE_HPC_EFF_LOSS * frac
    flow = TF.FAILURE_HPC_FLOW_LOSS * frac

    if fault_mode == 0:
        mods["HPC_eff_mod"] = 1.0 - eff
        mods["HPC_flow_mod"] = 1.0 - flow
    else:
        # Fan degradation. The same deterioration scale is applied, which is an
        # ASSUMPTION - the released data does not label which units carry which
        # mode, so there is no per-mode end-of-life measurement to fit against.
        # It is the fault-mode split of A19 that decides how many units take
        # this branch.
        mods["fan_eff_mod"] = 1.0 - eff
        mods["fan_flow_mod"] = 1.0 - flow
        mods["LPC_eff_mod"] = 1.0 - 0.5 * eff
    return mods


class CycleCache:
    """Memoised turbofan evaluations.

    One cycle solve is cheap but there are tens of thousands of rows, and health
    repeats heavily once rounded. Keyed on (regime, rounded health, mode).
    """

    def __init__(self, engine: TF.Turbofan, settings_by_regime: dict, digits: int = 4):
        self.engine = engine
        self.settings = settings_by_regime
        self.digits = digits
        self._cache: dict = {}

    def get(self, regime: int, health: float, mode: int) -> dict:
        key = (int(regime), round(float(health), self.digits), int(mode))
        hit = self._cache.get(key)
        if hit is None:
            alt_kft, mach, tra = self.settings[int(regime)]
            hit = self.engine.run(alt_kft * 1000.0, mach, tra,
                                  modifiers_from_health(key[1], mode))
            self._cache[key] = hit
        return hit


# ---------------------------------------------------------------------------
# Fitted instrumentation model - noise only, about the physics prediction.
# ---------------------------------------------------------------------------

@dataclass
class PhysicsSurface:
    subset: str
    regime_settings: dict = field(default_factory=dict)   # regime -> (alt_kft, mach, tra)
    regime_weights: list = field(default_factory=list)
    regime_jitter: dict = field(default_factory=dict)     # regime -> sd of the 3 settings
    n_modes: int = 1
    mode_weights: list = field(default_factory=lambda: [1.0])
    gains: dict = field(default_factory=dict)
    rhos: dict = field(default_factory=dict)
    sensors: list = field(default_factory=list)
    constant_sensors: list = field(default_factory=list)
    quantum: dict = field(default_factory=dict)
    between_unit_sd: dict = field(default_factory=dict)   # group -> sensor -> sd
    noise_params: dict = field(default_factory=dict)      # group -> sensor -> params
    resid_corr: dict = field(default_factory=dict)
    low_card: dict = field(default_factory=dict)
    bias: dict = field(default_factory=dict)              # group -> sensor -> mean offset

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["regime_settings"] = {str(k): list(v) for k, v in self.regime_settings.items()}
        d["regime_jitter"] = {str(k): list(v) for k, v in self.regime_jitter.items()}
        d["gains"] = {str(k): v for k, v in self.gains.items()}
        d["rhos"] = {str(k): v for k, v in self.rhos.items()}
        d["provenance"] = {
            "mean_model": "thermodynamic turbofan cycle (simulation/cmapss/turbofan.py)",
            "fitted_here": "measurement noise and per-unit offsets only, measured as "
                           "residuals about the physics prediction on the TRAIN split",
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PhysicsSurface":
        o = cls(subset=d["subset"])
        for k, v in d.items():
            if k in ("provenance",):
                continue
            setattr(o, k, v)
        o.regime_settings = {int(k): tuple(v) for k, v in d["regime_settings"].items()}
        o.regime_jitter = {int(k): np.asarray(v, float) for k, v in d["regime_jitter"].items()}
        o.gains = {int(k): v for k, v in d["gains"].items()}
        o.rhos = {int(k): v for k, v in d["rhos"].items()}
        return o

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "PhysicsSurface":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def fit(subset: str, gains: dict, rhos: dict, unit_mode: dict,
        mode_diag: dict, data_dir: str | None = None) -> PhysicsSurface:
    """Measure the instrumentation model as residuals about the physics."""
    train = P.load_split(subset, "train", data_dir)
    meta = C.SUBSETS[subset]

    from .characterize import recover_operating_regimes
    _, centres, _ = recover_operating_regimes(train, meta["n_conditions"])
    centres = np.asarray(centres, float)
    regimes = assign_regimes(train, centres)

    const = P.constant_sensors(train)
    informative = [c for c in C.SENSOR_COLS if c not in const]

    ps = PhysicsSurface(subset=subset)
    ps.sensors = informative
    ps.constant_sensors = const
    ps.n_modes = mode_diag.get("n_modes", 1)
    ps.gains, ps.rhos = dict(gains), dict(rhos)
    ps.mode_weights = [float(sum(1 for v in unit_mode.values() if v == m) / len(unit_mode))
                       for m in range(ps.n_modes)]

    for r in range(centres.shape[0]):
        m = regimes == r
        X = train.loc[m, C.SETTING_COLS].to_numpy(float)
        ps.regime_settings[r] = tuple(X.mean(0))
        ps.regime_jitter[r] = X.std(0, ddof=0)
        ps.regime_weights.append(float(m.sum() / len(train)))

    for s_ in C.SENSOR_COLS:
        u = np.unique(train[s_].to_numpy(float))
        g = np.diff(u)
        g = g[g > 1e-12]
        ps.quantum[s_] = float(g.min()) if g.size else 0.0

    health = assign_health(train, gains, unit_mode)
    row_mode = train["unit"].map(unit_mode).to_numpy()

    cache = CycleCache(TF.Turbofan(), ps.regime_settings)
    resid_store: dict = {}

    for mode in range(ps.n_modes):
        for r in range(centres.shape[0]):
            m = (regimes == r) & (row_mode == mode)
            if not m.any():
                continue
            sub = train.loc[m]
            key = gkey(mode, r)
            ps.between_unit_sd[key], ps.noise_params[key], ps.bias[key] = {}, {}, {}
            hs = health[m]

            pred = {s_: np.empty(int(m.sum())) for s_ in C.SENSOR_COLS}
            for i, hv in enumerate(hs):
                out = cache.get(r, hv, mode)
                for s_ in C.SENSOR_COLS:
                    pred[s_][i] = out[s_]

            for s_ in C.SENSOR_COLS:
                v = sub[s_].to_numpy(float)
                resid = v - pred[s_]
                # A constant offset between model and data is a calibration
                # difference, not noise; it is recorded and applied rather than
                # being left to inflate the noise variance.
                ps.bias[key][s_] = float(resid.mean())
                resid = resid - ps.bias[key][s_]
                if s_ in const:
                    ps.between_unit_sd[key][s_] = 0.0
                    ps.noise_params[key][s_] = {"phi": 0.0, "var": 0.0, "kurtosis": 3.0,
                                                "var_white": 0.0, "var_smooth": 0.0}
                    continue

                per_unit = pd.Series(resid).groupby(sub["unit"].to_numpy()).mean()
                ps.between_unit_sd[key][s_] = float(per_unit.std(ddof=0))
                within = resid - sub["unit"].map(per_unit).to_numpy()

                meas = N.measure(resid, sub["unit"].to_numpy())
                target = meas["window_var_ratio"]
                if meas["var"] > 0 and meas["var_smooth"] > 0:
                    need = (target * meas["var"] - meas["var_white"]) / meas["var_smooth"]
                    meas["phi"] = N.window_ratio_phi(float(np.clip(need, 1e-6, 1.0)),
                                                     meas.get("window", 30))
                else:
                    meas["phi"] = 0.0
                ps.noise_params[key][s_] = meas
                resid_store.setdefault(key, {})[s_] = within

            cs = [c for c in informative if c in resid_store.get(key, {})]
            if len(cs) >= 2:
                M = np.column_stack([resid_store[key][c] for c in cs])
                with np.errstate(invalid="ignore", divide="ignore"):
                    Cm = np.nan_to_num(np.corrcoef(M, rowvar=False), nan=0.0)
                ps.resid_corr[key] = {"cols": cs, "corr": Cm.tolist()}

    # Low-cardinality channels, per regime, as before.
    for r in range(centres.shape[0]):
        sub_r = train.loc[regimes == r]
        for s_ in C.SENSOR_COLS:
            if s_ in const or sub_r.empty:
                continue
            lv, cnt = np.unique(sub_r[s_].to_numpy(float), return_counts=True)
            if lv.size <= 20:
                ps.low_card.setdefault(s_, {})[str(r)] = {
                    "values": lv.tolist(),
                    "cumprob": np.cumsum(cnt / cnt.sum()).tolist()}

    return ps


# ---------------------------------------------------------------------------
# Generation.
# ---------------------------------------------------------------------------

def _fill_noise(out: dict, sel: np.ndarray, ps: PhysicsSurface, key: str,
                rng: np.random.Generator) -> None:
    """Correlated measurement noise for all channels of one group."""
    m = int(sel.sum())
    spec = ps.resid_corr.get(key)
    npar = ps.noise_params.get(key, {})
    cs = spec["cols"] if spec else [c for c in npar if c not in ps.constant_sensors]
    if not cs:
        return
    smooth = np.column_stack([
        N.ar1_mixture_noise(m, npar[c]["phi"], 1.0, npar[c]["kurtosis"], rng)
        if npar[c].get("var_smooth", 0.0) > 0 else np.zeros(m) for c in cs])
    white = np.column_stack([
        N.mixture_innovations(m, *N.solve_mixture(1.0, npar[c]["kurtosis"]), rng)
        if npar[c].get("var_white", 0.0) > 0 else np.zeros(m) for c in cs])
    if spec:
        Cm = np.asarray(spec["corr"], float)
        w, V = np.linalg.eigh((Cm + Cm.T) / 2.0)
        L = V @ np.diag(np.sqrt(np.clip(w, 1e-8, None)))
        smooth, white = smooth @ L.T, white @ L.T
    for j, c in enumerate(cs):
        out[c][sel] = (np.sqrt(max(npar[c].get("var_smooth", 0.0), 0.0)) * smooth[:, j]
                       + np.sqrt(max(npar[c].get("var_white", 0.0), 0.0)) * white[:, j])


def generate_unit(unit: int, ps: PhysicsSurface, cache: CycleCache,
                  rng: np.random.Generator, mode: int, offsets: dict,
                  max_cycles: int = 4000) -> pd.DataFrame:
    """One run-to-failure trajectory, sensors from the cycle."""
    gain = ps.gains[mode]
    rho = ps.rhos.get(mode, 0.0)
    params = D.sample_unit_params(unit, rng, rho)
    run = D.simulate_unit(params, max_cycles=max_cycles, gain=gain, rng=rng)
    H, n = run["health"], run["length"]

    n_reg = len(ps.regime_settings)
    wts = np.asarray(ps.regime_weights, float)
    regimes = rng.choice(n_reg, size=n, p=wts / wts.sum())

    settings = np.empty((n, 3))
    for r in range(n_reg):
        sel = regimes == r
        if sel.any():
            mu = np.asarray(ps.regime_settings[r], float)
            sd = np.asarray(ps.regime_jitter[r], float)
            settings[sel] = mu + rng.normal(0.0, 1.0, (int(sel.sum()), 3)) * sd

    noise = {s: np.zeros(n) for s in C.SENSOR_COLS}
    for r in range(n_reg):
        sel = regimes == r
        if sel.any():
            key = gkey(mode, r) if gkey(mode, r) in ps.noise_params else gkey(0, r)
            _fill_noise(noise, sel, ps, key, rng)

    cols = {"unit": np.full(n, unit, dtype=int),
            "cycle": np.arange(1, n + 1, dtype=int)}
    for j, cname in enumerate(C.SETTING_COLS):
        cols[cname] = settings[:, j]

    mean = {s: np.empty(n) for s in C.SENSOR_COLS}
    for i in range(n):
        out = cache.get(int(regimes[i]), float(H[i]), mode)
        for s in C.SENSOR_COLS:
            mean[s][i] = out[s]

    for s in C.SENSOR_COLS:
        vals = mean[s].copy()
        for r in range(n_reg):
            sel = regimes == r
            if not sel.any():
                continue
            key = gkey(mode, r) if gkey(mode, r) in ps.noise_params else gkey(0, r)
            vals[sel] += ps.bias[key].get(s, 0.0) + offsets.get((key, s), 0.0)
        vals += noise[s]
        q = ps.quantum.get(s, 0.0)
        if q > 0:
            vals = np.round(vals / q) * q
        cols[s] = vals

    return pd.DataFrame(cols, columns=C.ALL_COLS)


def generate_split(subset: str, split: str, ps: PhysicsSurface,
                   seed: int | None = None) -> tuple[pd.DataFrame, np.ndarray | None]:
    n_units = C.SUBSETS[subset][f"{split}_units"]
    rng = np.random.default_rng(seed if seed is not None else C.subset_seed(subset, split))
    cache = CycleCache(TF.Turbofan(), ps.regime_settings)

    if ps.n_modes > 1:
        counts = [int(round(w * n_units)) for w in ps.mode_weights]
        counts[-1] = n_units - sum(counts[:-1])
        mode_seq = np.concatenate([np.full(c, m, int) for m, c in enumerate(counts)])
        rng.shuffle(mode_seq)
    else:
        mode_seq = np.zeros(n_units, int)

    # Per-unit offsets, centred so a finite fleet does not shift the marginals.
    offsets = [{} for _ in range(n_units)]
    for key, sds in ps.between_unit_sd.items():
        for s, sd in sds.items():
            if sd <= 0:
                for u in range(n_units):
                    offsets[u][(key, s)] = 0.0
                continue
            d = rng.normal(0.0, sd, n_units)
            d = d - d.mean()
            cur = d.std(ddof=0)
            if cur > 0:
                d *= sd / cur
            for u in range(n_units):
                offsets[u][(key, s)] = float(d[u])

    frames, tail = [], []
    for u in range(1, n_units + 1):
        full = generate_unit(u, ps, cache, rng, int(mode_seq[u - 1]), offsets[u - 1])
        if split == "train":
            frames.append(full)
            continue
        L = len(full)
        lo, hi = max(1, int(0.15 * L)), max(2, int(0.95 * L))
        cut = int(rng.integers(lo, hi))
        frames.append(full.iloc[:cut].copy())
        tail.append(L - cut)

    out = pd.concat(frames, ignore_index=True)

    # Low-cardinality channels mapped per regime onto their real level set.
    centres = np.array([ps.regime_settings[r] for r in sorted(ps.regime_settings)], float)
    reg = assign_regimes(out, centres)
    for s, by_r in ps.low_card.items():
        if s not in out.columns:
            continue
        col = out[s].to_numpy(float).copy()
        for r_str, spec in by_r.items():
            m = reg == int(r_str)
            if not m.any():
                continue
            vals = np.asarray(spec["values"], float)
            cum = np.asarray(spec["cumprob"], float)
            x = col[m]
            order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
            uq = (order + 0.5) / x.size
            col[m] = vals[np.searchsorted(cum, uq, side="left").clip(0, vals.size - 1)]
        out[s] = col

    return out, (np.asarray(tail, int) if split == "test" else None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the physics instrumentation model.")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--generate", action="store_true",
                    help="also generate train/test from the fitted model")
    args = ap.parse_args()

    with open(os.path.join(args.out_dir, f"calibration_{args.subset}.json")) as fh:
        cal = json.load(fh)
    gains = {int(k): v for k, v in cal["calibrated_parameter"]["value_by_mode"].items()}
    rhos = {int(k): v for k, v in cal["calibrated_parameter"]["rho_by_mode"].items()}
    unit_mode = {int(k): v for k, v in cal["unit_mode"].items()}

    ps = fit(args.subset, gains, rhos, unit_mode, cal["fault_modes"])
    path = os.path.join(args.out_dir, f"physics_{args.subset}.json")
    ps.save(path)

    print(f"{args.subset}  modes={ps.n_modes} regimes={len(ps.regime_settings)} "
          f"groups={len(ps.noise_params)} informative={len(ps.sensors)}")
    print(f"{'sensor':10s} {'|bias|':>10} {'resid sd':>10} {'data sd':>10} {'bias/sd':>9}")
    train = P.load_split(args.subset, "train")
    for s in ps.sensors:
        b = np.mean([abs(d[s]) for d in ps.bias.values() if s in d])
        rsd = np.sqrt(np.mean([d[s]["var"] for d in ps.noise_params.values() if s in d]))
        dsd = train[s].std()
        print(f"{s:10s} {b:10.4f} {rsd:10.4f} {dsd:10.4f} {b/dsd if dsd else 0:9.3f}")
    print(f"-> {path}")

    if args.generate:
        from .generator import write_split
        gen_dir = os.path.join(args.out_dir, "generated")
        os.makedirs(gen_dir, exist_ok=True)

        class _Shim:
            quantum = ps.quantum
        for split in ("train", "test"):
            df, rul = generate_split(args.subset, split, ps)
            write_split(df, os.path.join(gen_dir, f"{split}_{args.subset}.txt"), _Shim())
            L = df.groupby("unit")["cycle"].max().to_numpy()
            print(f"{args.subset} {split:5s}: {len(df):>6} rows, {df.unit.nunique():>3} units, "
                  f"len {L.min()}-{L.max()} mean {L.mean():.1f}")
            if rul is not None:
                np.savetxt(os.path.join(gen_dir, f"RUL_{args.subset}.txt"), rul, fmt="%d")


if __name__ == "__main__":
    main()
