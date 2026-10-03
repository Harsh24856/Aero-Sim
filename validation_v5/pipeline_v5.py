"""v5 data pipeline (docs/v5_model_improvement_plan.md, Phase 3).

CACHE (built once per engine from data/rotax_v5/<key>/)
    X.f16        [rows, 39]   scaled per-second features, flights contiguous
    ends.f32     [n_ends, E]  one row per candidate window END (every 16 s once a
                              flight has 128 s of history): long-horizon context
                              (42), run-length context, and every label
    flights.parquet           flight -> row offset, rows, split, metadata
    contract_v5.json          feature order, scaler, window, context definition
  Storing context and labels only at candidate ends (1/16 of the rows) keeps the
  cache small; the long-horizon filters are still run over every second.

WINDOWS
    A window is X[end-127 : end+1] plus the context and labels at `end`. Each
    epoch takes one window per 64 s of every flight, at a random multiple-of-16
    offset, in a freshly shuffled order - deterministic for a given (seed, epoch).
    v4 read the same fixed windows in the same order every epoch.

SCALING
    Per-second features: mean / std on TRAIN flights. Residuals first divided by
    their instrument's noise sigma, so every residual is in sigma units.
    Context: same, fitted on train ends.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

import features_v5 as F  # noqa: E402
from degradation_v5 import FAULT_NAMES  # noqa: E402
from sensors_v5 import FAULTABLE_CHANNELS, SENSOR_SPEC  # noqa: E402

WINDOW = F.WINDOW
END_STEP = 16
STRIDE = 64

AUX_COLS = ["hours_frac", "life_used_hours", "life_scale_180"]     # context besides long horizon
LABEL_COLS = (["fault_present", "sensor_fault_any", "health_index", "margin_min",
               "rul_hours", "rul_hours_oracle", "rul_calendar_hours", "engine_hours",
               "limit_exceeded", "altitude", "throttle"]
              + [f"fm_{n}" for n in FAULT_NAMES] + [f"fmv_{n}" for n in FAULT_NAMES]
              + [f"sf_{c}_flag" for c in FAULTABLE_CHANNELS] + [f"sf_{c}_sev" for c in FAULTABLE_CHANNELS])
CTX_COLS = F.LONG_COLS + AUX_COLS
END_COLS = ["flight", "row"] + CTX_COLS + LABEL_COLS
RES_SIGMA = np.array([SENSOR_SPEC[c]["noise_sd"] for c in F.RESIDUAL_CHANNELS], np.float32)
RES_SLICE = slice(F.FEATURE_COLS.index(F.RESIDUAL_COLS[0]), F.FEATURE_COLS.index(F.RESIDUAL_COLS[-1]) + 1)


def _read_flights(src: str):
    idx = pd.read_parquet(os.path.join(src, "index.parquet"))
    for fname, grp in idx.groupby("file", sort=False):
        pf = pq.ParquetFile(os.path.join(src, fname))
        for _, r in grp.iterrows():
            yield r, pf.read_row_group(int(r.row_group)).to_pandas()


def build_cache(src: str, dst: str) -> dict:
    """Build the window cache for one engine. Two passes: statistics on train
    flights, then the scaled arrays."""
    t0 = time.time()
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "manifest.json")) as fh:
        man = json.load(fh)
    tbo = man["tbo_hours"]
    idx = pd.read_parquet(os.path.join(src, "index.parquet"))
    n_rows = int(idx.n_rows.sum())

    # -- pass 1: scaler (train flights), counts ------------------------------
    s1 = np.zeros(F.N_FEATURES)
    s2 = np.zeros(F.N_FEATURES)
    n = 0
    c1 = np.zeros(len(CTX_COLS))
    c2 = np.zeros(len(CTX_COLS))
    cn = 0
    n_ends = 0
    for r, g in _read_flights(src):
        A = g[F.FEATURE_COLS].to_numpy(np.float64, copy=True)
        A[:, RES_SLICE] /= RES_SIGMA
        ends = np.arange(WINDOW - 1, len(g), END_STEP)
        n_ends += len(ends)
        if r.split != "train":
            continue
        s1 += A.sum(0)
        s2 += (A * A).sum(0)
        n += len(A)
        C = _context(g, tbo)[ends]
        c1 += C.sum(0)
        c2 += (C * C).sum(0)
        cn += len(C)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    std[std < 1e-6] = 1.0                       # constant features (e.g. 912 turbo channels)
    cmean = c1 / cn
    cstd = np.sqrt(np.maximum(c2 / cn - cmean ** 2, 1e-12))
    cstd[cstd < 1e-9] = 1.0

    # -- pass 2: arrays ---------------------------------------------------------
    X = np.lib.format.open_memmap(os.path.join(dst, "X.npy"), mode="w+", dtype=np.float16,
                                  shape=(n_rows, F.N_FEATURES))
    ENDS = np.lib.format.open_memmap(os.path.join(dst, "ends.npy"), mode="w+", dtype=np.float32,
                                     shape=(n_ends, len(END_COLS)))
    flights = []
    row = 0
    e = 0
    for fi, (r, g) in enumerate(_read_flights(src)):
        g = g.sort_values("t")
        A = g[F.FEATURE_COLS].to_numpy(np.float64, copy=True)
        A[:, RES_SLICE] /= RES_SIGMA
        X[row:row + len(g)] = ((A - mean) / std).astype(np.float16)
        ends = np.arange(WINDOW - 1, len(g), END_STEP)
        C = (_context(g, tbo)[ends] - cmean) / cstd
        Lb = g[LABEL_COLS].to_numpy(np.float64)[ends]
        k = len(ends)
        ENDS[e:e + k, 0] = fi
        ENDS[e:e + k, 1] = row + ends
        ENDS[e:e + k, 2:2 + len(CTX_COLS)] = C
        ENDS[e:e + k, 2 + len(CTX_COLS):] = Lb
        flights.append({"flight": fi, "scenario_id": int(r.scenario_id), "row0": row, "n_rows": len(g),
                        "end0": e, "n_ends": k, "split": r.split, "mission": r.mission,
                        "life_scale": float(r.life_scale), "primary_family": r.primary_family,
                        "wear_limited": bool(r.wear_limited), "faults": r.faults,
                        "sensor_faults": r.sensor_faults})
        row += len(g)
        e += k
    X.flush()
    ENDS.flush()
    pd.DataFrame(flights).to_parquet(os.path.join(dst, "flights.parquet"), index=False)
    # The twin is a property of the DATA: older datasets were generated against a new engine.
    contract = dict(F.contract(), twin=man["contract"].get("twin", "new_engine"),
                    engine_model=man["engine_model"], tbo_hours=tbo,
                    scaler={"mean": mean.tolist(), "std": std.tolist(),
                            "residual_sigma": RES_SIGMA.tolist(), "residuals_divided_by_sigma_first": True},
                    ctx_cols=CTX_COLS, ctx_scaler={"mean": cmean.tolist(), "std": cstd.tolist()},
                    label_cols=LABEL_COLS, end_cols=END_COLS, end_step=END_STEP, stride=STRIDE,
                    rows=n_rows, ends=n_ends, source=os.path.abspath(src))
    with open(os.path.join(dst, "contract_v5.json"), "w") as fh:
        json.dump(contract, fh, indent=1)
    return {"rows": n_rows, "ends": n_ends, "flights": len(flights), "seconds": round(time.time() - t0, 1)}


def _context(g: pd.DataFrame, tbo: float) -> np.ndarray:
    """Per-row context: long-horizon residual features (in sigma units) + aux."""
    res = g[F.RESIDUAL_COLS].to_numpy(np.float64) / RES_SIGMA
    lh = F.long_horizon_array(res)
    aux = np.stack([g.engine_hours.to_numpy() / tbo, g.life_used_hours.to_numpy(),
                    (g.life_scale.to_numpy() > 1).astype(float)], 1)
    return np.concatenate([lh, aux], 1)


# ---------------------------------------------------------------------------
class Cache:
    """Read side: windows for training and evaluation."""

    def __init__(self, path: str):
        self.path = path
        self.X = np.load(os.path.join(path, "X.npy"), mmap_mode="r")
        self.ends = np.load(os.path.join(path, "ends.npy"), mmap_mode="r")
        self.flights = pd.read_parquet(os.path.join(path, "flights.parquet"))
        with open(os.path.join(path, "contract_v5.json")) as fh:
            self.contract = json.load(fh)
        self.col = {c: i for i, c in enumerate(END_COLS)}
        self.ctx_slice = slice(2, 2 + len(CTX_COLS))

    def end_ids(self, splits, seed: int = 0, epoch: int = 0, jitter: bool = True) -> np.ndarray:
        """One window end per STRIDE seconds of every flight in `splits`, at a
        random (seeded) multiple-of-16 offset, shuffled. jitter=False gives the
        fixed evaluation grid."""
        rng = np.random.default_rng((seed, epoch))
        per = STRIDE // END_STEP
        out = []
        for _, f in self.flights[self.flights.split.isin(list(splits))].iterrows():
            off = int(rng.integers(0, per)) if jitter else per - 1
            out.append(np.arange(f.end0 + off, f.end0 + f.n_ends, per))
        ids = np.concatenate(out) if out else np.zeros(0, int)
        if jitter:
            rng.shuffle(ids)
        return ids

    def batch(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(sequences [B,128,39] float32, context [B,C] float32, end rows [B,E])."""
        E = np.asarray(self.ends[np.sort(ids)])
        order = np.argsort(np.argsort(ids))
        E = E[order]
        rows = E[:, 1].astype(np.int64)
        seq = np.stack([self.X[r - WINDOW + 1:r + 1] for r in rows]).astype(np.float32)
        return seq, E[:, self.ctx_slice].astype(np.float32), E

    def labels(self, E: np.ndarray, name: str) -> np.ndarray:
        return E[:, self.col[name]]

    def flight_of(self, E: np.ndarray) -> np.ndarray:
        return E[:, 0].astype(int)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the v5 window cache for one engine.")
    ap.add_argument("src", help="data/rotax_v5/<key>")
    ap.add_argument("dst", help="cache directory")
    a = ap.parse_args()
    print(build_cache(a.src, a.dst))
