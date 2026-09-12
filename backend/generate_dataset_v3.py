"""Dataset v3 generator - 15M rows per engine, life-stage sampled, real-hour RUL.

WHAT IS DIFFERENT FROM v2 (backend/generate_multi_engine_data_v2.py)
--------------------------------------------------------------------
1. LIFE-STAGE SAMPLING - the single change that makes RUL learnable.
   v2 started every scenario at wear = 0.0 and ran to failure or a 20,000 s
   cutoff, so the model only ever saw the START of engine life and 17% of
   scenarios were censored (no RUL label at all). v3 seeds each scenario at a
   random accumulated life, so a window from an engine at 1,500 h is labelled
   "500 h remaining" - and, because physics v3 finally feeds wear back into the
   sensors, it also LOOKS different from one at 100 h. Without both halves of
   that the RUL head can only learn a near-constant.

2. REAL-HOUR, UNCENSORED RUL. The label is `rul_hours_true` straight off the
   twin: TBO_HOURS - wear*TBO_HOURS. It spans the full 0..TBO range instead of
   being capped at MAX_DURATION/3600 = 5.556 h.

3. STREAMING, CHUNKED WRITES. v2 accumulated every row in a Python list and
   built one DataFrame at the end. At 15M rows x ~60 columns that is roughly
   25 GB of RAM and will not complete. v3 flushes each split to numbered parquet
   chunks as it goes, so peak memory is one chunk.

4. THE INDEX IS WRITTEN IN THE FORMAT THE PIPELINE ACTUALLY READS.
   v2's save_chunks wrote {"train": [3, 7, 11, ...]} - plain ints - but
   tf_data_pipeline.scenario_window_generator does entry["file"] and
   entry["scenario_id"], i.e. it needs dicts. The on-disk v2 indexes were in
   dict form because of an ad-hoc conversion step that was never committed, so
   regenerating with v2 produced indexes the pipeline could not read. Fixed here.

5. FAILURE-MODE LABELS. The fm_* severities from backend/failure_modes.py are
   written as training targets for the new head.

RUN
---
    # one engine
    validation/venv/bin/python3 backend/generate_dataset_v3.py --engine Rotax_914_ULF
    # all four, in parallel (recommended - about 16 minutes wall)
    validation/venv/bin/python3 backend/generate_dataset_v3.py --all --parallel
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from physics import UAVEngineTwin, ENGINE_CONFIGS, TBO_HOURS
from failure_modes import MODES as FAILURE_MODES

# ---------------------------------------------------------------------------
TARGET_ROWS = 15_000_000       # per engine
MAX_DURATION = 20000.0         # seconds per scenario
DT = 1.0                       # 1 Hz, matches WINDOW_SIZE semantics
LEG_MIN, LEG_MAX = 300, 800    # seconds between flight-leg target resets
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10
CHUNK_ROWS = 2_500_000         # rows per parquet chunk - keeps peak memory bounded

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "validation")

# Life-stage sampling. Stratified rather than uniform so every decile of engine
# life is represented even at modest scenario counts - a plain uniform draw
# leaves visible gaps, and the RUL head then interpolates across them.
LIFE_STAGE_BINS = 10
LIFE_STAGE_MAX = 0.95          # never start a scenario essentially dead

SENSOR_CHANNELS = ["egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz"]

COLUMNS = (
    ["scenario_id", "time", "elapsed_hours",
     "altitude", "throttle", "airspeed", "aoa", "air_density",
     "torque_available_nm", "engine_rpm", "prop_rpm", "prop_torque", "power_kw",
     "fuel_flow", "thrust", "lift", "drag", "thrust_margin", "lift_weight_margin",
     "injection_timing", "battery_voltage", "battery_current", "alternator_output",
     "ambient_temp_c", "isa_dev_c"]
    + [f"{c}_healthy" for c in SENSOR_CHANNELS]
    + SENSOR_CHANNELS + ["rpm_fault"]
    + [f"{c}_flag" for c in SENSOR_CHANNELS] + ["rpm_flag"]
    + [f"{c}_stress" for c in SENSOR_CHANNELS] + ["rpm_stress"]
    + [f"fm_{m}" for m in FAILURE_MODES]
    + ["wear", "is_failed_now", "start_wear", "tbo_hours", "rul_hours_true"]
)


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def sample_leg_target(severity_bias):
    """Same flight-leg model as v2 - a 'pilot personality' per scenario."""
    throttle = clip(np.random.uniform(0.15, 0.35)
                    + severity_bias*np.random.uniform(0.4, 0.65), 0.1, 1.0)
    airspeed = clip(np.random.uniform(45, 70)
                    - severity_bias*np.random.uniform(20, 55), 0.0, 80.0)
    altitude = np.random.uniform(0, 8000)
    aoa = np.random.uniform(-10, 20)
    return altitude, throttle, airspeed, aoa


def sample_start_wear(scen_idx):
    """Stratified life stage: cycle through deciles, jitter inside each."""
    lo = (scen_idx % LIFE_STAGE_BINS) / LIFE_STAGE_BINS
    hi = lo + 1.0/LIFE_STAGE_BINS
    return float(np.clip(np.random.uniform(lo, hi), 0.0, LIFE_STAGE_MAX))


def run_scenario(engine_model, scen_idx):
    """One flight. Returns a list of rows (without scenario_id prefixed)."""
    twin = UAVEngineTwin(dt=DT, engine_model=engine_model, physics_version="v3")

    # THE key line: start this engine partway through its life.
    start_wear = sample_start_wear(scen_idx)
    twin.wear = start_wear

    # Occasional hot-day / cold-day scenarios so the ambient model is exercised.
    twin.isa_dev_c = float(np.random.choice([0.0, 0.0, 0.0, 15.0, 30.0, -15.0]))

    severity_bias = np.random.uniform(0.0, 1.0)
    altitude, throttle, airspeed, aoa = sample_leg_target(severity_bias)
    twin.altitude, twin.throttle, twin.airspeed, twin.aoa = altitude, throttle, airspeed, aoa
    next_leg_change = np.random.uniform(LEG_MIN, LEG_MAX)

    buf = []
    t = 0.0
    while t < MAX_DURATION:
        if t >= next_leg_change:
            altitude, throttle, airspeed, aoa = sample_leg_target(severity_bias)
            next_leg_change = t + np.random.uniform(LEG_MIN, LEG_MAX)
        twin.throttle = clip(twin.throttle + np.random.normal(0, 0.008) + 0.15*(throttle-twin.throttle), 0.1, 1.0)
        twin.altitude = clip(twin.altitude + np.random.normal(0, 8.0) + 0.05*(altitude-twin.altitude), 0.0, 9000.0)
        twin.airspeed = clip(twin.airspeed + np.random.normal(0, 0.4) + 0.10*(airspeed-twin.airspeed), 0.0, 80.0)
        twin.aoa = clip(twin.aoa + np.random.normal(0, 0.2) + 0.10*(aoa-twin.aoa), -15.0, 25.0)

        o = twin.step()
        h, ff, fs = o["healthy"], o["fault_flags"], o["fault_stress"]
        buf.append(
            [o["time"], o["elapsed_hours"],
             o["altitude"], o["throttle"], o["airspeed"], o["aoa"], o["air_density"],
             o["torque_available_nm"], o["engine_rpm"], o["prop_rpm"], o["prop_torque"],
             o["power_kw"], o["fuel_flow"], o["thrust"], o["lift"], o["drag"],
             o["thrust_margin"], o["lift_weight_margin"],
             o["injection_timing"], o["battery_voltage"], o["battery_current"],
             o["alternator_output"], o["ambient_temp_c"], o["isa_dev_c"]]
            + [h[c] for c in SENSOR_CHANNELS]
            + [o[c] for c in SENSOR_CHANNELS] + [o["rpm_fault"]]
            + [ff[c] for c in SENSOR_CHANNELS] + [ff["rpm"]]
            + [fs[c] for c in SENSOR_CHANNELS] + [fs["rpm"]]
            + [o[f"fm_{m}"] for m in FAILURE_MODES]
            + [o["wear"], int(o["failed"]), start_wear, o["tbo_hours"], o["rul_hours_true"]]
        )
        t = o["time"]
        if o["failed"]:
            break
    return buf


class SplitWriter:
    """Buffers rows for one split and flushes numbered parquet chunks.

    Peak memory is CHUNK_ROWS rows, not the whole dataset - the reason v2's
    accumulate-everything approach cannot be reused at 15M rows.
    """

    def __init__(self, out_dir, split):
        self.out_dir, self.split = out_dir, split
        self.rows, self.chunk_idx, self.total = [], 0, 0
        self.index_entries = []      # {"file", "scenario_id"} - what the pipeline reads
        self._pending_sids = set()

    def add(self, scenario_id, rows):
        self.rows.extend([scenario_id] + r for r in rows)
        self._pending_sids.add(int(scenario_id))
        self.total += len(rows)
        if len(self.rows) >= CHUNK_ROWS:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        self.chunk_idx += 1
        fname = f"{self.split}_chunk_{self.chunk_idx:02d}.parquet"
        df = pd.DataFrame(self.rows, columns=COLUMNS)
        df.to_parquet(os.path.join(self.out_dir, fname), engine="pyarrow", index=False)
        for sid in sorted(self._pending_sids):
            self.index_entries.append({"file": fname, "scenario_id": sid})
        print(f"    wrote {fname}: {len(df):,} rows, {len(self._pending_sids)} scenarios")
        sys.stdout.flush()
        self.rows, self._pending_sids = [], set()


def generate_engine(engine_model, seed, target_rows=TARGET_ROWS, out_dir=None):
    np.random.seed(seed)
    out_dir = out_dir or os.path.join(BASE_DIR, f"chunks_v3_{engine_model.lower()}")
    os.makedirs(out_dir, exist_ok=True)

    writers = {s: SplitWriter(out_dir, s) for s in ("train", "val", "test")}
    split_rng = np.random.RandomState(123)   # split assignment reproducible, independent of content

    total, scen = 0, 0
    n_failed = 0
    t0 = time.time()
    while total < target_rows:
        rows = run_scenario(engine_model, scen)
        if rows and rows[-1][COLUMNS.index("is_failed_now") - 1]:
            n_failed += 1
        r = split_rng.rand()
        split = "train" if r < TRAIN_FRAC else ("val" if r < TRAIN_FRAC + VAL_FRAC else "test")
        writers[split].add(scen, rows)
        total += len(rows)
        scen += 1
        if scen % 25 == 0:
            el = time.time() - t0
            rate = total/max(el, 1e-9)
            eta = (target_rows-total)/max(rate, 1e-9)
            print(f"  [{engine_model}] scen {scen}, rows {total:,}/{target_rows:,}, "
                  f"{rate:,.0f} rows/s, ETA {eta/60:.1f} min")
            sys.stdout.flush()

    for w in writers.values():
        w.flush()

    index = {
        "engine_model": engine_model,
        "physics_version": "v3",
        "tbo_hours": TBO_HOURS[engine_model],
        "rows_total": int(total),
        "n_scenarios_total": int(scen),
        "n_failed_scenarios": int(n_failed),
        # Dict entries - the format tf_data_pipeline.scenario_window_generator
        # actually reads. v2 wrote bare ints here and the pipeline could not load it.
        "train": writers["train"].index_entries,
        "val": writers["val"].index_entries,
        "test": writers["test"].index_entries,
    }
    with open(os.path.join(out_dir, "scenario_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"[{engine_model}] DONE: {total:,} rows, {scen} scenarios "
          f"({writers['train'].total:,}/{writers['val'].total:,}/{writers['test'].total:,}) "
          f"in {(time.time()-t0)/60:.1f} min -> {out_dir}")
    return out_dir


def _worker(args):
    engine, seed, rows, out_dir = args
    return generate_engine(engine, seed, rows, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None, choices=list(ENGINE_CONFIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--parallel", action="store_true", help="one process per engine")
    ap.add_argument("--rows", type=int, default=TARGET_ROWS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    engines = list(ENGINE_CONFIGS) if (a.all or not a.engine) else [a.engine]
    # --out names a single engine's directory, so it only applies to a single job.
    jobs = [(e, 42 + i, a.rows, (a.out if len(engines) == 1 else None))
            for i, e in enumerate(engines)]

    if a.parallel and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(len(jobs)) as pool:
            pool.map(_worker, jobs)
    else:
        for j in jobs:
            _worker(j)
    print("ALL DONE.")
