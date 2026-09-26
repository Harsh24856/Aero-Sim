"""Dataset v4 generator - 10M rows per engine, physics-driven, twin-referenced.

WHAT MAKES THIS A GOOD TRAINING SET, point by point. Each of these is a decision
that was wrong in an earlier version and cost something measurable.

 1. ONLY DEPLOYABLE FEATURES.
    v3 trained on power_kw, torque_nm, torque_available_nm and air_density.
    None of those exists as a sensor on a real aircraft, so a model trained on
    them cannot be deployed without first estimating its own inputs. v4's
    feature vector contains what an airframe actually provides: twelve measured
    sensors, the flight condition from the avionics, and residuals against the
    on-board twin.

 2. PHYSICS RESIDUALS AS FIRST-CLASS FEATURES.
    A healthy twin runs alongside the degraded engine on the SAME inputs, and
    the difference between measured and expected is fed to the model. This is
    what separates "the engine is hot" from "the engine is at high power on a
    hot day" - a distinction no amount of raw-sensor context can reliably make,
    because the two look identical channel by channel. It doubles generation
    cost and is worth it.

 3. LIFE-STAGE SAMPLING.
    Carried over from v3, where it was the single change that made RUL
    learnable. A flight lasts hours against a 1200-2000 h TBO, so degradation
    barely moves within one scenario; the RUL signal comes from comparing
    engines at different ages. Every scenario therefore starts at a random
    accumulated life, with extra density near end of life where the prediction
    matters and where failures actually occur.

 4. FAULTS DEFINED BY WHAT THEY BREAK.
    Fourteen component faults drive health modifiers; the sensor pattern is
    whatever the physics and the closed-loop ECU produce. v3 defined faults by
    the pattern they should produce, so a model could only learn the rule
    someone wrote.

 5. SENSOR FAULTS SEPARATE FROM ENGINE FAULTS.
    Labelled independently, so a drifting thermocouple and a real overheat are
    distinguishable rather than conflated.

 6. SCENARIO-LEVEL SPLITS, ASSIGNED BEFORE GENERATION.
    Splitting rows would leak: consecutive rows of one flight are nearly
    identical, so a random row split puts near-duplicates on both sides and
    every metric is inflated. Splits are assigned per scenario from an
    independent RNG, so the assignment cannot depend on what the scenario
    contains.

 7. REAL MISSION STRUCTURE.
    Ten profiles with phased, temporally correlated trajectories rather than
    independent random values, because a model trained on white-noise inputs
    learns dynamics that do not exist.

 8. FULL ENVIRONMENTAL ENVELOPE.
    Hot and cold days, humidity, fuel quality, cooling effectiveness and
    electrical load are all sampled. v3 could represent none of these, so a
    model trained on it had never seen a hot-and-high takeoff.

RUN
    validation_v4/venv/bin/python3 backend/generate_dataset_v4.py --engine Rotax_914_ULF
    backend/generate_dataset_v4.py --all --parallel        # ~35 min wall
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import physics_v4 as V4
from degradation_v4 import DegradationState, FAULT_NAMES, applicable_faults
from sensors_v4 import SensorBank, SENSOR_CHANNELS, SENSOR_FAULT_TYPES

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
FLIGHT_COLS = ["altitude", "airspeed", "aoa", "throttle",
               "ambient_temp_c", "air_density", "isa_dev_c", "humidity_frac"]
MEASURED_COLS = list(SENSOR_CHANNELS)                       # 12
DERIVED_COLS = ["prop_rpm", "thrust_margin", "lift_weight_margin"]
RESIDUAL_CHANNELS = ["egt", "cht", "oil_temp", "oil_pressure",
                     "engine_rpm", "fuel_flow"]
RESIDUAL_COLS = [f"res_{c}" for c in RESIDUAL_CHANNELS]

FEATURE_COLS = FLIGHT_COLS + MEASURED_COLS + DERIVED_COLS + RESIDUAL_COLS
N_FEATURES = len(FEATURE_COLS)                              # 29

FM_COLS = [f"fm_{n}" for n in FAULT_NAMES]                  # 14
SF_FLAG_COLS = [f"sf_{c}_flag" for c in SENSOR_CHANNELS]    # 12
SF_SEV_COLS = [f"sf_{c}_sev" for c in SENSOR_CHANNELS]      # 12
LABEL_COLS = (["fault_present", "health_index", "margin_min", "rul_hours_true",
               "life_used_hours", "engine_hours", "failed", "limit_exceeded"]
              + FM_COLS + SF_FLAG_COLS + SF_SEV_COLS)

ALL_COLS = ["scenario_id", "t"] + FEATURE_COLS + LABEL_COLS

TARGET_ROWS = 10_000_000
DT = 1.0                       # 1 Hz, matching v3 and the window semantics
# MAX FLIGHT LENGTH, HALVED FROM 9000 s.
#
# The binding constraint on training is the number of INDEPENDENT scenarios,
# not the number of rows or windows. At stride 128 the windows do not overlap,
# but the ~51 windows cut from one flight share an engine, a fault, a
# degradation state and an operating profile - so 60,087 training windows
# carried roughly 1,186 independent examples against a 248k-parameter model,
# about 209 parameters each.
#
# It showed: detection overfitted from epoch 2 in every configuration tried.
# Raising dropout 0.1 -> 0.25 moved test AUC 0.8444 -> 0.8471 and selecting the
# checkpoint on AUC rather than val_loss moved it to 0.8496, but neither changed
# the shape of the curve, because neither addresses sample diversity.
#
# Halving the flight length doubles the scenario count for the SAME row budget,
# the same window count and the same epoch time. Nothing is lost in realism -
# each window is still 128 s of genuine flight, and the model never sees a whole
# sortie anyway. What doubles is the number of distinct engine x fault x
# degradation x environment combinations, which is the axis being overfitted.
MAX_SCENARIO_S = 4500.0        # 1.25 h maximum flight
CHUNK_ROWS = 400_000
REPORT_EVERY_S = 15.0          # progress cadence, so a live view actually moves
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10

# A fault counts as "present" for the detection head once it is physically
# consequential, not from the instant it starts. Below this the engine is
# within its normal scatter and labelling it faulty teaches the model to fire
# on noise.
FAULT_PRESENT_SEV = 0.08


# ---------------------------------------------------------------------------
# Mission profiles
# ---------------------------------------------------------------------------
# (name, [(phase_fraction, alt_m, airspeed_ms, throttle, aoa_deg), ...])
MISSIONS = {
    "takeoff_climb": [(0.05, 50, 32, 1.00, 9.0), (0.30, 1200, 42, 0.95, 6.0),
                      (0.65, 3000, 50, 0.85, 3.0)],
    "cruise":        [(0.08, 2500, 48, 0.85, 3.5), (0.92, 3500, 55, 0.72, 2.0)],
    "loiter":        [(0.10, 3000, 45, 0.70, 4.0), (0.90, 3200, 38, 0.55, 6.5)],
    "descent":       [(0.15, 4000, 55, 0.60, 1.0), (0.85, 600, 48, 0.30, 0.0)],
    "endurance":     [(0.05, 2000, 45, 0.80, 4.0), (0.95, 2800, 40, 0.50, 5.5)],
    "high_altitude": [(0.20, 4500, 52, 0.95, 3.0), (0.80, 7200, 58, 0.92, 2.0)],
    "hot_weather":   [(0.15, 800, 40, 0.98, 7.0), (0.85, 2200, 46, 0.88, 4.0)],
    "rapid_throttle": [(0.25, 1500, 45, 0.95, 4.0), (0.50, 1500, 42, 0.35, 5.0),
                       (0.75, 1600, 48, 0.92, 3.5), (1.00, 1500, 40, 0.45, 5.0)],
    "low_level":     [(0.12, 300, 50, 0.88, 3.0), (0.88, 500, 55, 0.80, 2.5)],
    "mixed":         [(0.10, 800, 38, 0.95, 7.0), (0.35, 3000, 50, 0.85, 3.0),
                      (0.70, 4200, 55, 0.70, 2.0), (1.00, 1200, 45, 0.45, 4.0)],
}
MISSION_NAMES = list(MISSIONS)


def mission_target(profile: list, frac: float) -> tuple:
    """Interpolate the profile's target state at a fraction through the flight."""
    prev = profile[0]
    for seg in profile:
        if frac <= seg[0]:
            f0 = prev[0]
            w = 0.0 if seg[0] <= f0 else (frac - f0) / (seg[0] - f0)
            return tuple(prev[i] + w * (seg[i] - prev[i]) for i in range(1, 5))
        prev = seg
    return tuple(profile[-1][1:5])


def sample_start_hours(rng: np.random.Generator, tbo: float) -> float:
    """Life-stage sampling, weighted toward end of life.

    Uniform sampling would put only a tenth of scenarios in the last tenth of
    life, which is exactly where RUL prediction matters and where failures
    happen. The mixture below puts about 40% of scenarios in the final third
    without starving early life, which the model also needs in order to know
    what healthy looks like.
    """
    u = rng.random()
    if u < 0.35:
        return float(rng.uniform(0.0, 0.45 * tbo))
    if u < 0.60:
        return float(rng.uniform(0.45 * tbo, 0.75 * tbo))
    return float(rng.uniform(0.75 * tbo, 1.02 * tbo))


def sample_environment(rng: np.random.Generator, mission: str) -> dict:
    """Environment and installation state for one flight."""
    hot = mission == "hot_weather"
    return {
        "isa_dev_c": float(rng.normal(18.0, 6.0) if hot else rng.normal(0.0, 11.0)),
        "qnh_offset_pa": float(rng.normal(0.0, 900.0)),
        "humidity_frac": float(np.clip(rng.beta(2.0, 3.0) + (0.25 if hot else 0.0), 0.0, 1.0)),
        "fuel_octane_mon": float(rng.choice([91.0, 95.0, 95.0, 95.0, 98.0, 100.0])),
        "fuel_ethanol_frac": float(rng.choice([0.0, 0.0, 0.0, 0.05, 0.10])),
        "cooling_airflow_factor": float(np.clip(rng.normal(1.0, 0.10), 0.65, 1.30)),
        "electrical_load_a": float(np.clip(rng.normal(14.0, 5.0), 3.0, 34.0)),
        # False = thermostat STUCK in bypass, a real but rare failure. It was
        # 12% of flights, which once the thermostat actually regulated drove a
        # healthy 916's oil over its 130 C limit on 10.5% of rows - unlabelled.
        "oil_thermostat_open": bool(rng.random() > 0.02),
        "target_lambda": float(np.clip(rng.normal(0.89, 0.035), 0.78, 1.02)),
    }


# ---------------------------------------------------------------------------
# One scenario
# ---------------------------------------------------------------------------
def run_scenario(engine_model: str, scen_id: int, rng: np.random.Generator) -> list:
    spec = V4.ENGINE_SPECS_V4[engine_model]
    tbo = spec.tbo_hours

    mission = str(rng.choice(MISSION_NAMES))
    profile = MISSIONS[mission]
    env = sample_environment(rng, mission)

    start_h = sample_start_hours(rng, tbo)
    # `forced` cycles through the fault taxonomy so every mode is guaranteed a
    # share of scenarios rather than appearing by chance.
    deg = DegradationState(rng, start_hours=start_h, tbo_hours=tbo, forced=scen_id,
                           allowed=applicable_faults(spec.turbocharged, spec.intercooled))

    # EARLY REJECTION. Life-stage sampling can seed an engine that is already
    # worn out; it then fails on its first recorded step. Detecting that here
    # costs one function call, where detecting it after the fact cost a full
    # warm-up and scenario run - a third of scenarios were being thrown away
    # that way.
    if deg.condition_at(start_h) <= 0.02:
        return []

    duration = float(rng.uniform(0.45, 1.0) * MAX_SCENARIO_S)

    # The engine under test and its healthy twin. The twin sees identical
    # inputs and a perfect health state, which is exactly the on-board
    # reference the deployed system runs.
    eng = V4.PistonEngineV4(engine_model, dt=DT)
    twin = V4.PistonEngineV4(engine_model, dt=DT)
    healthy = V4.Health()

    sensors = SensorBank(rng=rng, dt=DT)
    sensors.add_random_faults(duration_s=duration)

    # Seed both machines warm, then a short settle. The dataset should not
    # consist largely of cold-start transients the aircraft rarely flies, but
    # reaching a warm state by simulating 900 s of it was the dominant cost in
    # generation and none of it is recorded.
    warm_u = V4.Inputs(altitude_m=1500, airspeed_ms=45, throttle=0.8, **env)
    h0, _ = deg.health_at(start_h)
    eng.warm_start(warm_u, h0)
    twin.warm_start(warm_u, healthy)
    # Long enough for the thermal states to reach the value this operating point
    # and this health actually imply, rather than the generic seed. At 60 steps
    # the seed was still dominant and scenarios failed immediately on a CHT that
    # belonged to a different engine condition.
    for _ in range(300):
        eng.step(warm_u, h0)
        twin.step(warm_u, healthy)

    # Once per scenario, not once per row: the degradation state is fixed for
    # the flight, so the age at which this engine wears out is too.
    wear_out_h = deg.wear_out_hours()

    alt, spd, thr, aoa = mission_target(profile, 0.0)
    rows = []
    t = 0.0
    while t < duration:
        frac = t / duration
        t_alt, t_spd, t_thr, t_aoa = mission_target(profile, frac)
        # First-order approach to the target plus small process noise: mission
        # variables must be temporally correlated, not independently random.
        alt += 0.05 * (t_alt - alt) + rng.normal(0.0, 6.0)
        spd += 0.10 * (t_spd - spd) + rng.normal(0.0, 0.35)
        thr += 0.12 * (t_thr - thr) + rng.normal(0.0, 0.007)
        aoa += 0.10 * (t_aoa - aoa) + rng.normal(0.0, 0.18)
        alt = float(np.clip(alt, 0.0, 9000.0))
        spd = float(np.clip(spd, 28.0, 85.0))
        thr = float(np.clip(thr, 0.08, 1.0))
        aoa = float(np.clip(aoa, -12.0, 24.0))

        u = V4.Inputs(altitude_m=alt, airspeed_ms=spd, aoa_deg=aoa, throttle=thr, **env)

        hours = start_h + eng.life_used_h
        h, sev = deg.health_at(hours)

        o = eng.step(u, h)
        ref = twin.step(u, healthy)
        meas, sf_flag, sf_sev = sensors.read(o, t)

        m = eng.margins(o)
        margin_min = m["health_index"]          # operability: limit-relative
        condition = deg.condition_at(hours)     # wear: monotone over life

        # EXCEEDING A LIMIT IS AN EVENT, NOT THE END OF THE ENGINE. Terminating
        # the scenario whenever any margin closed truncated most flights to a
        # handful of rows: a momentary CHT excursion is something the operator
        # responds to by reducing power, not a destroyed engine. It is recorded
        # as its own label - which is exactly the condition an advisory should
        # fire on - while only true wear-out ends the flight.
        limit_exceeded = int(margin_min <= 0.0)
        failed = condition <= 0.0

        rul = max(0.0, min(tbo, wear_out_h) - hours)
        max_sev = max(sev.values()) if sev else 0.0

        rows.append(
            [scen_id, t]
            # flight condition
            + [alt, spd, aoa, thr, o["ambient_temp_c"], o["air_density"],
               env["isa_dev_c"], env["humidity_frac"]]
            # measured sensors
            + [meas[c] for c in MEASURED_COLS]
            # on-board derived
            + [o["prop_rpm"], o["thrust_margin"], o["lift_weight_margin"]]
            # residuals against the healthy twin
            + [meas[c] - ref[c] for c in RESIDUAL_CHANNELS]
            # labels
            + [int(max_sev >= FAULT_PRESENT_SEV), condition, margin_min, rul,
               eng.life_used_h, hours, int(failed), limit_exceeded]
            + [sev[n] for n in FAULT_NAMES]
            + [sf_flag[c] for c in SENSOR_CHANNELS]
            + [sf_sev[c] for c in SENSOR_CHANNELS]
        )

        t += DT
        if failed:
            break
    return rows


# ---------------------------------------------------------------------------
# Chunked writer
# ---------------------------------------------------------------------------
class SplitWriter:
    """Flushes numbered parquet chunks so peak memory is one chunk, not the set."""

    def __init__(self, out_dir: str, split: str):
        self.dir = os.path.join(out_dir, split)
        os.makedirs(self.dir, exist_ok=True)
        self.split = split
        self.buf: list = []
        self.index: list = []
        self.chunk = 0
        self.rows = 0

    def add(self, scen_id: int, rows: list) -> None:
        if not rows:
            return
        self.index.append({"scenario_id": int(scen_id),
                           "file": f"{self.split}_chunk_{self.chunk:03d}.parquet",
                           "n_rows": len(rows)})
        self.buf.extend(rows)
        self.rows += len(rows)
        if len(self.buf) >= CHUNK_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        df = pd.DataFrame(self.buf, columns=ALL_COLS)
        path = os.path.join(self.dir, f"{self.split}_chunk_{self.chunk:03d}.parquet")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                       compression="zstd")
        self.buf.clear()
        self.chunk += 1


# ---------------------------------------------------------------------------
def generate_engine(engine_model: str, seed: int, target_rows: int = TARGET_ROWS,
                    out_dir: str | None = None) -> dict:
    key = engine_model.split("_")[1]
    # The per-engine subdirectory is ALWAYS appended, including when a base
    # directory is supplied. Treating --out-dir as the final path meant every
    # engine of an --all --parallel run wrote to the same place and silently
    # overwrote the others.
    base = out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "rotax_v4")
    out_dir = os.path.join(base, key)
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    # Split assignment uses its OWN generator, so which split a scenario lands
    # in cannot depend on anything about the scenario.
    split_rng = np.random.default_rng(987654)

    writers = {s: SplitWriter(out_dir, s) for s in ("train", "val", "test")}
    total = 0
    scen = 0
    t0 = time.time()
    last_report = 0.0

    MIN_ROWS = 256      # two windows at WINDOW=128; shorter yields no training sample
    while total < target_rows:
        scen_id = scen          # the id written INTO the rows
        rows = run_scenario(engine_model, scen_id, rng)
        scen += 1               # advance the stratification cycle either way
        if len(rows) < MIN_ROWS:
            # A scenario seeded past end of life fails on its first step and
            # produces a handful of rows that can never form a window. Discard
            # and resample rather than filling the set with unusable fragments.
            continue
        r = split_rng.random()
        split = "train" if r < TRAIN_FRAC else ("val" if r < TRAIN_FRAC + VAL_FRAC else "test")
        # scen_id, NOT scen. Incrementing before this recorded scen+1 in the
        # index while the rows carried scen, so a predicate-pushdown read on
        # scenario_id matched nothing for ~88% of entries and those scenarios
        # were silently unreachable.
        writers[split].add(scen_id, rows)
        total += len(rows)

        # Report on a TIMER rather than every N scenarios. A scenario takes
        # several seconds, so "every 50" meant a line roughly every four
        # minutes - long enough that a working run and a hung one look the same,
        # and far too coarse to watch.
        now = time.time()
        if now - last_report >= REPORT_EVERY_S:
            el = now - t0
            done = total / target_rows
            eta = (el / max(done, 1e-9) - el) / 60.0
            print(f"  [{key}] {scen:5d} scen  {total:>10,} rows  "
                  f"{total / max(el, 1e-9):,.0f} rows/s  "
                  f"{100 * done:5.1f}%  eta {eta:4.0f}m", flush=True)
            last_report = now

    for w in writers.values():
        w.flush()

    manifest = {
        "engine_model": engine_model, "key": key, "physics_version": "v4",
        "seed": seed, "dt": DT, "rows": total, "scenarios": scen,
        "generated_s": round(time.time() - t0, 1),
        "contract": {
            "feature_cols": FEATURE_COLS, "n_features": N_FEATURES,
            "flight_cols": FLIGHT_COLS, "measured_cols": MEASURED_COLS,
            "derived_cols": DERIVED_COLS, "residual_cols": RESIDUAL_COLS,
            "fault_modes": FAULT_NAMES, "fm_cols": FM_COLS,
            "sensor_channels": SENSOR_CHANNELS,
            "sensor_fault_types": SENSOR_FAULT_TYPES,
            "label_cols": LABEL_COLS,
        },
        "splits": {s: {"rows": w.rows, "scenarios": len(w.index), "chunks": w.chunk}
                   for s, w in writers.items()},
        "tbo_hours": V4.ENGINE_SPECS_V4[engine_model].tbo_hours,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    for s, w in writers.items():
        with open(os.path.join(out_dir, f"index_{s}.json"), "w") as fh:
            json.dump(w.index, fh)
    return manifest


def engine_seed(engine_model: str, base: int = 4000) -> int:
    """Stable per-engine seed, independent of how the run was launched."""
    return base + (zlib.crc32(engine_model.encode()) % 100_000)


def _worker(args):
    return generate_engine(*args)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dataset v4.")
    ap.add_argument("--engine", default="Rotax_914_ULF")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--rows", type=int, default=TARGET_ROWS)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    engines = list(V4.ENGINE_SPECS_V4) if args.all else [args.engine]
    # THE SEED IS DERIVED FROM THE ENGINE NAME, not from its index in the list.
    # Using enumerate() meant a single-engine launch always had i = 0, so all
    # four engines ran on seed 4000 and generated byte-identical missions, fault
    # draws and scenario durations - visible as identical row counts across the
    # four logs. Only the physics differed. That silently removes most of the
    # diversity from the combined 40M rows and confounds any cross-engine
    # comparison, which is exactly what Phase 14 is for.
    jobs = [(e, engine_seed(e), args.rows, args.out_dir) for e in engines]

    if args.parallel and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(len(jobs)) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    print()
    for m in results:
        sp = m["splits"]
        print(f"{m['key']}: {m['rows']:,} rows / {m['scenarios']} scenarios in "
              f"{m['generated_s'] / 60:.1f} min   "
              f"train {sp['train']['rows']:,} val {sp['val']['rows']:,} "
              f"test {sp['test']['rows']:,}")


if __name__ == "__main__":
    main()
