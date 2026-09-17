"""RUL-only dataset (v4 probes) - one engine, one frozen life stage, one window.

WHY A SEPARATE DATASET
----------------------
The v3 scenario dataset trains four heads well and the RUL head badly. Two
properties of it, both measured on the 914 test split, put a floor under RUL
error that no architecture can get under:

1. THE LABEL MOVES INSIDE THE WINDOW. physics v3 ages an engine fast enough to
   be watchable: WEAR_K*dt at cruise burns roughly 0.008 of life per 128 s, so
   `rul_hours_true` falls by a median of 15.8 h (p95 34.3 h) ACROSS A SINGLE
   128-SAMPLE WINDOW. The deployed 914 head scores MAE 22.9 h - the same size as
   the label's own ambiguity. Training longer cannot fix an ambiguous target.

2. LIFE STAGES ARE SKEWED OLD. Wear climbs steeply inside every scenario, so
   even with life-stage seeding the median row sits at 36% of TBO and only 1.6%
   of rows are above 95%. Every live flight starts at ~99%, which is precisely
   where the head had almost no data - hence the 3.5-6.2% under-prediction on a
   near-new engine that the cockpit shows on every demo.

WHAT A PROBE IS
---------------
One probe = one engine held at ONE life stage while it flies a short segment:

    fresh twin -> wear := w0 (drawn uniformly over life, extra density near-new)
    -> random operating point, random ISA day, random session age
    -> warm up 150-450 s (thermal lags settle, running statistics accumulate)
    -> record the next 128 s as the window
    -> after every step, wear is reset to w0

Freezing wear is not a trick, it is the physically honest thing: a real engine
ages ~0.00005% of TBO in five minutes. physics v3's accelerated ageing exists so
a demo flight is watchable; for supervision it only blurs the target. Every
sensor channel is computed from self.wear BEFORE _update_wear runs (physics.py
step), so the recorded window is exactly the engine at w0 and the label
TBO*(1 - w0) is exact.

Session age (the `elapsed_hours` aux feature, and the running severity
statistics that go with it) is drawn INDEPENDENTLY of w0. In the v3 dataset those
six features correlate with wear through the scenario's own progression, which
lets the head read the clock instead of the sensors - and live, that clock always
says "this flight just started". Here they carry no wear information by
construction, so the only way to score is to read the window.

OUTPUT (per engine, under validation/rul_v4_<engine>/)
    {split}_x.npy      float32 (N, 128, 25)  RAW feature columns, unscaled
    {split}_aux.npy    float32 (N, 10)       x_rul_aux, pipeline definitions
    {split}_y.npy      float32 (N,)          engine hours remaining (exact)
    {split}_meta.parquet                     wear, faults, operating point, session age
    meta.json                                contract + generator settings

x is stored UNSCALED on purpose: the engine's StandardScaler stays the single
source of truth and is applied at load time (validation/rul_data.py), exactly as
tf_data_pipeline does, so a rescale never needs a regeneration.

RUN
    validation/venv/bin/python3 backend/generate_rul_dataset.py --all --parallel
    validation/venv/bin/python3 backend/generate_rul_dataset.py --engine Rotax_914_ULF
    validation/venv/bin/python3 backend/generate_rul_dataset.py --smoke   # 200 probes
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "validation"))

from physics import UAVEngineTwin, ENGINE_CONFIGS, TBO_HOURS      # noqa: E402
from failure_modes import MODES as FAILURE_MODES                  # noqa: E402
import tf_data_pipeline as P                                      # noqa: E402

WINDOW_S = P.WINDOW_SIZE          # 128 - the serving contract
DT = 1.0

# Warm-up before the window. Randomised rather than fixed: the slowest thermal lag
# is OILTEMP_TAU = 100 s, so 150 s is ~1.5 taus (still visibly settling) and 450 s
# is ~4.5 (settled). Live, the first prediction lands 128 s after a warm start, so
# a spread of settle states is what the model will actually meet.
WARMUP_MIN_S, WARMUP_MAX_S = 150, 450

# Life stage. Uniform in wear IS uniform in RUL (rul = TBO*(1-wear)), so this
# gives flat coverage across the whole range - the v3 dataset's median sits at
# 36% of TBO. NEAR_NEW_SHARE of probes are redrawn into the first NEAR_NEW_MAX of
# life: that band is 1.6% of v3 rows and 100% of what a judge sees on a fresh
# start, and it is where the deployed head is 3.5-6.2% low.
WEAR_MAX = 0.98
NEAR_NEW_SHARE = 0.25
NEAR_NEW_MAX = 0.10

# Session age for the `elapsed_hours` aux feature, drawn independently of wear.
SESSION_AGE_MAX_H = 2.0

# Flight-leg model, as the v3 generator's but with shorter legs so a set-point
# change can land inside the 128 s window rather than only between probes.
LEG_MIN, LEG_MAX = 120, 400

# FUEL-FLOW MEASUREMENT ERROR - without it this dataset hands the model the answer.
#
# physics v3 computes fuel_flow = power*BSFC*(1 + 0.18*wear) with no sensor error, so
# with wear frozen the bsfc_ratio aux feature correlates -1.000 with the label: an
# exact, invertible read of wear. Measured on a 200-probe smoke run. A head trained
# on that reads one number and ignores the window - the failure the notebook's bsfc
# ablation exists to catch.
#
# Real fuel-flow measurement does not behave that way: a turbine transducer is
# specified around +-2%, and installation, fuel temperature and mixture add more. The
# per-probe term is a calibration offset (constant for that installation), the
# per-sample term is transducer noise. Wear moves BSFC by 18% over a full life, so a
# 2.5% calibration error leaves bsfc alone able to place an engine only within roughly
# +-14% of life - informative, not decisive. It is applied to the recorded fuel_flow
# itself, so the window and the aux feature stay consistent with each other.
FUEL_CAL_SIGMA = 0.025
FUEL_NOISE_SIGMA = 0.010

PROBES_DEFAULT = {"train": 48000, "val": 6000, "test": 6000}
SPLITS = ("train", "val", "test")

SENSOR_CHANNELS = ["egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz"]
META_COLS = (
    ["wear", "rul_hours", "tbo_hours", "session_age_h", "warmup_s", "isa_dev_c",
     "mean_throttle", "mean_rpm", "mean_power_kw", "mean_airspeed", "mean_altitude",
     "max_fault_stress", "fault_channels", "max_fm_severity", "fuel_cal_pct"]
    + [f"fm_{m}" for m in FAILURE_MODES]
)


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def sample_wear(rng):
    """Uniform over life, with a deliberate near-new over-sample."""
    if rng.random() < NEAR_NEW_SHARE:
        return float(rng.uniform(0.0, NEAR_NEW_MAX))
    return float(rng.uniform(0.0, WEAR_MAX))


def sample_leg_target(rng, severity_bias):
    throttle = clip(rng.uniform(0.15, 0.35) + severity_bias*rng.uniform(0.4, 0.65), 0.1, 1.0)
    airspeed = clip(rng.uniform(45, 70) - severity_bias*rng.uniform(20, 55), 32.0, 80.0)
    return rng.uniform(0, 8000), throttle, airspeed, rng.uniform(-10, 20)


def run_probe(engine_model, rng, fuel_noise=True):
    """One probe. Returns (x_raw (128,25), aux (10,), label_hours, meta dict)."""
    twin = UAVEngineTwin(dt=DT, engine_model=engine_model, physics_version="v3")
    w0 = sample_wear(rng)
    twin.wear = w0
    twin.isa_dev_c = float(rng.choice([0.0, 0.0, 0.0, 15.0, 30.0, -15.0]))

    severity_bias = rng.uniform(0.0, 1.0)
    alt, thr, spd, aoa = sample_leg_target(rng, severity_bias)
    twin.altitude, twin.throttle, twin.airspeed, twin.aoa = alt, thr, spd, aoa
    next_leg = rng.uniform(LEG_MIN, LEG_MAX)

    warmup = int(rng.integers(WARMUP_MIN_S, WARMUP_MAX_S + 1))
    session_age_h = float(rng.uniform(0.0, SESSION_AGE_MAX_H))

    n = warmup + WINDOW_S
    rows = []
    for _ in range(n):
        if twin.t >= next_leg:
            alt, thr, spd, aoa = sample_leg_target(rng, severity_bias)
            next_leg = twin.t + rng.uniform(LEG_MIN, LEG_MAX)
        twin.throttle = clip(twin.throttle + rng.normal(0, 0.008) + 0.15*(thr-twin.throttle), 0.1, 1.0)
        twin.altitude = clip(twin.altitude + rng.normal(0, 8.0) + 0.05*(alt-twin.altitude), 0.0, 9000.0)
        twin.airspeed = clip(twin.airspeed + rng.normal(0, 0.4) + 0.10*(spd-twin.airspeed), 32.0, 80.0)
        twin.aoa = clip(twin.aoa + rng.normal(0, 0.2) + 0.10*(aoa-twin.aoa), -15.0, 25.0)
        o = twin.step()
        # THE line this dataset exists for: this engine stays at one life stage.
        # Every channel above was computed from self.wear before _update_wear ran,
        # so the window and the label describe the same engine exactly.
        twin.wear = w0
        rows.append(o)

    df = pd.DataFrame({c: np.array([r[c] for r in rows], dtype=np.float32)
                       for c in P.FEATURE_COLS})
    if fuel_noise:
        cal = float(rng.normal(0.0, FUEL_CAL_SIGMA))          # this installation's offset
        noise = rng.normal(0.0, FUEL_NOISE_SIGMA, size=n)     # transducer, per sample
        df["fuel_flow"] = (df["fuel_flow"].to_numpy(dtype=np.float32)
                           * np.float32(1.0 + cal) * (1.0 + noise).astype(np.float32))
    # Session age is independent of wear: the aux clock carries no life information.
    df["elapsed_hours"] = (session_age_h + np.arange(1, n + 1, dtype=np.float32)*DT/3600.0)
    df["isa_dev_c"] = np.float32(twin.isa_dev_c)

    aux = P.compute_rul_aux(df)
    aux_vec = np.array([aux[k][-1] for k in P.RUL_AUX_ORDER], dtype=np.float32)
    x = df[P.FEATURE_COLS].to_numpy(dtype=np.float32)[-WINDOW_S:]

    win = rows[-WINDOW_S:]
    stress = np.array([[r["fault_stress"][c] for c in SENSOR_CHANNELS] + [r["fault_stress"]["rpm"]]
                       for r in win], dtype=np.float32)
    flags = np.array([[r["fault_flags"][c] for c in SENSOR_CHANNELS] + [r["fault_flags"]["rpm"]]
                      for r in win], dtype=np.float32)
    fm = {m: float(np.max([r[f"fm_{m}"] for r in win])) for m in FAILURE_MODES}
    tbo = float(twin.TBO_HOURS)
    meta = {
        "fuel_cal_pct": round(100.0*cal, 3) if fuel_noise else 0.0,
        "wear": w0, "rul_hours": tbo*(1.0 - w0), "tbo_hours": tbo,
        "session_age_h": session_age_h, "warmup_s": float(warmup),
        "isa_dev_c": float(twin.isa_dev_c),
        "mean_throttle": float(x[:, P.FEATURE_COLS.index("throttle")].mean()),
        "mean_rpm": float(x[:, P.FEATURE_COLS.index("engine_rpm")].mean()),
        "mean_power_kw": float(x[:, P.FEATURE_COLS.index("power_kw")].mean()),
        "mean_airspeed": float(x[:, P.FEATURE_COLS.index("airspeed")].mean()),
        "mean_altitude": float(x[:, P.FEATURE_COLS.index("altitude")].mean()),
        "max_fault_stress": float(stress.max()),
        "fault_channels": float((flags > 0).any(axis=0).sum()),
        "max_fm_severity": float(max(fm.values())),
        **{f"fm_{m}": fm[m] for m in FAILURE_MODES},
    }
    # The twin's own label must agree with the frozen one to within the ONE step of
    # ageing that step() applies after computing the channels (we reset wear
    # immediately after, so the dict's label is one increment ahead). A bigger gap
    # means wear is being written somewhere this loop does not control - which
    # would silently blur the label again, the exact defect this dataset exists to
    # remove.
    one_step_h = tbo*twin.WEAR_K*DT*1.01           # wear_drive <= 1 by construction
    assert abs(rows[-1]["rul_hours_true"] - meta["rul_hours"]) <= one_step_h, "wear did not stay frozen"
    return x, aux_vec, np.float32(meta["rul_hours"]), meta


def generate_engine(engine_model, seed, probes=None, out_dir=None, fuel_noise=True,
                    prefix="rul_v4"):
    probes = probes or PROBES_DEFAULT
    out_dir = out_dir or os.path.join(ROOT, "validation", f"{prefix}_{engine_model.lower()}")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for split in SPLITS:
        n = probes[split]
        # Memmap: x is the only large array here (N x 128 x 25 x 4 B), and writing
        # straight through keeps peak memory at one probe.
        x_path = os.path.join(out_dir, f"{split}_x.npy")
        xm = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
                                       shape=(n, WINDOW_S, P.N_FEATURES))
        aux_arr = np.empty((n, P.N_RUL_AUX), dtype=np.float32)
        y_arr = np.empty((n,), dtype=np.float32)
        metas = []
        for i in range(n):
            x, aux, y, meta = run_probe(engine_model, rng, fuel_noise=fuel_noise)
            xm[i] = x
            aux_arr[i] = aux
            y_arr[i] = y
            metas.append(meta)
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"  [{engine_model}] {split} {i+1:,}/{n:,}  "
                      f"{(i+1)/max(el,1e-9):.1f} probes/s", flush=True)
        xm.flush()
        del xm
        np.save(os.path.join(out_dir, f"{split}_aux.npy"), aux_arr)
        np.save(os.path.join(out_dir, f"{split}_y.npy"), y_arr)
        pd.DataFrame(metas, columns=META_COLS).to_parquet(
            os.path.join(out_dir, f"{split}_meta.parquet"), engine="pyarrow", index=False)
        print(f"  [{engine_model}] {split}: {n:,} probes written", flush=True)

    meta = {
        "engine_model": engine_model,
        "physics_version": "v3",
        "dataset": f"{prefix}_probes",
        "tbo_hours": TBO_HOURS[engine_model],
        "window_size": WINDOW_S,
        "feature_cols": P.FEATURE_COLS,
        "rul_aux_order": P.RUL_AUX_ORDER,
        "probes": {s: probes[s] for s in SPLITS},
        "wear_frozen_per_probe": True,
        "wear_max": WEAR_MAX,
        "near_new_share": NEAR_NEW_SHARE,
        "near_new_max": NEAR_NEW_MAX,
        "warmup_s": [WARMUP_MIN_S, WARMUP_MAX_S],
        "session_age_max_h": SESSION_AGE_MAX_H,
        "fuel_cal_sigma": FUEL_CAL_SIGMA if fuel_noise else 0.0,
        "fuel_noise_sigma": FUEL_NOISE_SIGMA if fuel_noise else 0.0,
        "seed": seed,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{engine_model}] DONE in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)
    return out_dir


def _worker(args):
    return generate_engine(*args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None, choices=list(ENGINE_CONFIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--parallel", action="store_true", help="one process per engine")
    ap.add_argument("--probes", type=int, default=None, help="train probes; val/test scale with it")
    ap.add_argument("--smoke", action="store_true", help="200/40/40 probes, for a quick check")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="rul_v4", help="output directory prefix under validation/")
    ap.add_argument("--no-fuel-noise", action="store_true",
                    help="record fuel flow exactly as physics computes it - bsfc_ratio then "
                         "correlates -1.000 with the label and the head learns nothing else")
    a = ap.parse_args()

    if a.smoke:
        probes = {"train": 200, "val": 40, "test": 40}
    elif a.probes:
        probes = {"train": a.probes, "val": max(1, a.probes//8), "test": max(1, a.probes//8)}
    else:
        probes = PROBES_DEFAULT

    engines = list(ENGINE_CONFIGS) if (a.all or not a.engine) else [a.engine]
    jobs = [(e, 2026 + i, probes, (a.out if len(engines) == 1 else None),
             not a.no_fuel_noise, a.prefix)
            for i, e in enumerate(engines)]

    if a.parallel and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(len(jobs)) as pool:
            pool.map(_worker, jobs)
    else:
        for j in jobs:
            _worker(j)
    print("ALL DONE.")
