#!/usr/bin/env python3
"""Generate the physics-v5 training set (docs/v5_model_improvement_plan.md, Phase 2).

What changed from generate_dataset_v4.py, and why
-------------------------------------------------
PHYSICS    physics_v5 (separable ignition / oil / turbo signatures, load-dependent
           EGT, certified-envelope knock reference) and sensors_v5 (airbox
           temperature and wastegate position on turbo engines).
FEATURES   features_v5: 39 per-second inputs, a residual for every instrument.
THREE ENGINES PER ROW. The engine under test, its healthy twin (residuals, as the
           aircraft computes them) and a WEAR-ONLY engine - the same engine with
           its baseline wear but none of its faults. A fault is labelled PRESENT
           only when its noise-free effect (faulty minus wear-only) on some
           instrument reaches 1 sigma of that instrument's noise, so a fault the
           physics hides in this regime (a dirty filter the boost controller makes
           up) is not labelled as something the model should see.
RUL        degradation_v5: remaining life from baseline wear plus the faults
           ALREADY under way (`rul_hours`); v4's future-fault label kept as
           `rul_hours_oracle`. Wear-out faults get a rising onset hazard.
SENSORS    60% of flights carry sensor faults; the first fault cycles through all
           72 (channel, kind) cells; labels switch on when the fault is visible.
ENVIRONMENT ISA deviation and humidity follow a slow random walk within a flight
           (+-2-3 C, +-0.1 over an hour), so they no longer identify the flight.
LIFE CLOCK 40% of flights age the engine at the live x180 life scale (timescale_v4),
           the rest in real time, so in-window wear trends seen live are in training.
MISSIONS   Turbo engines fly the high-altitude mission ~2.5x as often (about 22% of
           flights), and the mixed mission climbs through critical altitude, so
           turbo and wastegate faults are seen where they are observable.
SPLITS     By flight, STRATIFIED on (life-stage third, mission, primary fault
           family): 60 / 15 / 10 / 15 train / calibration / val / test. The
           calibration split is reserved for Phase 4 calibration and the RUL model.
STORAGE    One parquet row group per flight in data/rotax_v5/<key>/flights_NNN.parquet,
           plus index.parquet (flight -> file, row group, rows, split, metadata),
           so the Phase 3 cache reads flights directly.

    backend/.venv/bin/python backend/generate_dataset_v5.py --engine Rotax_914_ULF --flights 600 --out-dir data/rotax_v5_pilot
    backend/.venv/bin/python backend/generate_dataset_v5.py --all --flights 6000 --jobs 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features_v5 as F  # noqa: E402
import physics_v5 as P  # noqa: E402
import timescale_v4 as TS  # noqa: E402
from degradation_v5 import (  # noqa: E402
    DegradationStateV5, FAULT_NAMES, REMOVAL_CONDITION, applicable_faults,
)
from generate_dataset_v4 import (  # noqa: E402  unchanged v4 helpers
    MISSIONS, MISSION_NAMES, mission_target, sample_environment, sample_start_hours,
)
from observability_v5 import FAMILY_OF  # noqa: E402
from sensors_v5 import (  # noqa: E402
    FAULTABLE_CHANNELS, SENSOR_CHANNELS, SENSOR_SPEC, SensorBank, TURBO_CHANNELS,
)

# ---------------------------------------------------------------------------
# Contract (columns)
# ---------------------------------------------------------------------------
FM_COLS = [f"fm_{n}" for n in FAULT_NAMES]               # severity per fault (health/severity heads)
FMV_COLS = [f"fmv_{n}" for n in FAULT_NAMES]             # present AND visible (diagnosis head)
SF_FLAG_COLS = [f"sf_{c}_flag" for c in FAULTABLE_CHANNELS]      # visible kind index
SF_ACTIVE_COLS = [f"sf_{c}_active" for c in FAULTABLE_CHANNELS]  # kind index from onset
SF_SEV_COLS = [f"sf_{c}_sev" for c in FAULTABLE_CHANNELS]
# Noise-free fault effect per instrument, in that instrument's noise sigma (faulty engine
# minus the same engine with wear only). effect_z is their max; per channel they let a
# label judge visibility against each channel's own healthy spread (relabel_v5b point 2).
EFF_COLS = [f"eff_{c}" for c in SENSOR_CHANNELS]
LABEL_COLS = (["fault_present", "fault_present_sev", "effect_z", "sensor_fault_any",
               "health_index", "margin_min", "rul_hours", "rul_hours_oracle",
               "rul_calendar_hours", "engine_hours", "life_used_hours", "life_scale",
               "failed", "limit_exceeded"]
              + FM_COLS + FMV_COLS + SF_FLAG_COLS + SF_ACTIVE_COLS + SF_SEV_COLS + EFF_COLS)
ALL_COLS = ["scenario_id", "t", "mission_id"] + F.FEATURE_COLS + LABEL_COLS

MAX_SCENARIO_S = 4500.0
DT = 1.0
MIN_ROWS = 256
FAULT_PRESENT_SEV = 0.08          # v4 severity definition, kept as fault_present_sev
EFFECT_Z = 1.0                    # visible = noise-free effect >= 1 sigma on some instrument
SENSOR_FAULT_P = 0.60
LIFE_SCALE_P = 0.40               # share of flights on the live x180 life clock
FLIGHTS_PER_FILE = 400

# Stratified split: a fixed slot pattern per stratum (20 slots).
SPLIT_SLOTS = ["train"] * 12 + ["cal"] * 3 + ["val"] * 2 + ["test"] * 3


def mission_weights(turbo: bool) -> np.ndarray:
    w = np.ones(len(MISSION_NAMES))
    if turbo:
        w[MISSION_NAMES.index("high_altitude")] = 2.5
    return w / w.sum()


def mission_profile(name: str, turbo: bool) -> list:
    """v4 profiles; on turbo engines the mixed mission climbs through the
    critical altitude (4,572 m) instead of stopping at 4,200 m."""
    prof = MISSIONS[name]
    if turbo and name == "mixed":
        prof = [(0.10, 800, 38, 0.95, 7.0), (0.35, 3000, 50, 0.85, 3.0),
                (0.70, 5600, 56, 0.90, 2.0), (1.00, 1200, 45, 0.45, 4.0)]
    return prof


class EnvWalk:
    """Slow within-flight drift of ISA deviation and humidity (OU processes)."""
    TAU_S = 1800.0
    ISA_SD, HUM_SD = 1.5, 0.05

    def __init__(self, env: dict, rng: np.random.Generator):
        self.rng = rng
        self.isa0, self.hum0 = env["isa_dev_c"], env["humidity_frac"]
        self.isa, self.hum = self.isa0, self.hum0

    def step(self) -> tuple[float, float]:
        a = DT / self.TAU_S
        k = math.sqrt(2.0 * a)
        self.isa += -a * (self.isa - self.isa0) + k * self.ISA_SD * self.rng.normal()
        self.hum += -a * (self.hum - self.hum0) + k * self.HUM_SD * self.rng.normal()
        self.hum = float(min(max(self.hum, 0.0), 1.0))
        return self.isa, self.hum


def effects(o: dict, w: dict, turbo: bool) -> list:
    """Signed noise-free fault effect per instrument, in sigma (0 for the turbo
    instruments a naturally aspirated engine does not have)."""
    return [0.0 if (c in TURBO_CHANNELS and not turbo)
            else (float(o[c]) - float(w[c])) / SENSOR_SPEC[c]["noise_sd"] for c in SENSOR_CHANNELS]


def effect_z(o: dict, w: dict, turbo: bool) -> float:
    """Largest noise-free fault effect over the instruments, in sigma."""
    return max(abs(e) for e in effects(o, w, turbo))


def life_stage(start_h: float, tbo: float) -> int:
    return min(int(3.0 * start_h / tbo), 2)


# ---------------------------------------------------------------------------
# One flight
# ---------------------------------------------------------------------------
def run_scenario(engine_model: str, scen_id: int, rng: np.random.Generator):
    spec = P.ENGINE_SPECS[engine_model]
    tbo = spec.tbo_hours
    turbo = spec.turbocharged

    mission = str(rng.choice(MISSION_NAMES, p=mission_weights(turbo)))
    profile = mission_profile(mission, turbo)
    env = sample_environment(rng, mission)
    start_h = sample_start_hours(rng, tbo)
    life_scale = TS.LIFE_SCALE if rng.random() < LIFE_SCALE_P else 1.0

    deg = DegradationStateV5(rng, start_hours=start_h, tbo_hours=tbo, forced=scen_id,
                             allowed=applicable_faults(spec.turbocharged, spec.intercooled))
    # An engine already due for removal (or nearly worn out) does not fly.
    if deg._removed(start_h, deg._known(start_h)) or deg.condition_at(start_h) <= 0.02:
        return [], None

    duration = float(rng.uniform(0.45, 1.0) * MAX_SCENARIO_S)
    eng = P.PistonEngineV5(engine_model, dt=DT)
    twin = P.PistonEngineV5(engine_model, dt=DT)
    wear = P.PistonEngineV5(engine_model, dt=DT)
    healthy = P.Health()

    sensors = SensorBank(rng=rng, dt=DT, turbocharged=turbo)
    # Deterministic schedule: 3 flights in every 5 carry sensor faults, and those
    # flights step through the 72 (channel, kind) cells in order - balanced by
    # construction rather than by chance.
    faulty = (scen_id % 5) < 3
    sensors.plan_faults(duration_s=duration, p_any=1.0 if faulty else 0.0,
                        cell=(scen_id // 5) * 3 + (scen_id % 5))
    walk = EnvWalk(env, rng)

    warm_u = P.Inputs(altitude_m=1500, airspeed_ms=45, throttle=0.8, **env)
    h0, _ = deg.health_at(start_h)
    hw0 = deg.health_wear_only(start_h)
    eng.warm_start(warm_u, h0)
    twin.warm_start(warm_u, healthy)
    wear.warm_start(warm_u, hw0)
    for _ in range(300):
        eng.step(warm_u, h0)
        twin.step(warm_u, healthy)
        wear.step(warm_u, hw0)

    wear_out_oracle = deg.wear_out_hours()
    n_known = deg.n_known(start_h)
    wear_out_known = deg.wear_out_known(start_h)
    wear_limited_start = bool(min(tbo, wear_out_known) < tbo)
    mission_id = MISSION_NAMES.index(mission)

    alt, spd, thr, aoa = mission_target(profile, 0.0)
    rows = []
    t = 0.0
    while t < duration:
        frac = t / duration
        t_alt, t_spd, t_thr, t_aoa = mission_target(profile, frac)
        alt += 0.05 * (t_alt - alt) + rng.normal(0.0, 6.0)
        spd += 0.10 * (t_spd - spd) + rng.normal(0.0, 0.35)
        thr += 0.12 * (t_thr - thr) + rng.normal(0.0, 0.007)
        aoa += 0.10 * (t_aoa - aoa) + rng.normal(0.0, 0.18)
        alt = float(np.clip(alt, 0.0, 9000.0))
        spd = float(np.clip(spd, 28.0, 85.0))
        thr = float(np.clip(thr, 0.08, 1.0))
        aoa = float(np.clip(aoa, -12.0, 24.0))
        isa, hum = walk.step()
        env_now = dict(env, isa_dev_c=isa, humidity_frac=hum)

        u = P.Inputs(altitude_m=alt, airspeed_ms=spd, aoa_deg=aoa, throttle=thr, **env_now)
        hours = TS.engine_hours(start_h, eng.life_used_h, life_scale)
        h, sev = deg.health_at(hours)
        hw = deg.health_wear_only(hours)

        o = eng.step(u, h)
        ref = twin.step(u, healthy)
        w = wear.step(u, hw)
        meas, sf_flag, sf_sev, sf_active = sensors.read(o, t)
        res = F.residuals(meas, ref, turbo)

        k = deg.n_known(hours)
        if k != n_known:                      # a fault started: re-project wear-out
            n_known, wear_out_known = k, deg.wear_out_known(hours)
        condition = deg.condition_at(hours)
        m = eng.margins(o)
        eff = effects(o, w, turbo)
        z = max(abs(e) for e in eff)
        visible = z >= EFFECT_Z
        max_sev = max(sev.values()) if sev else 0.0
        failed = condition <= 0.0

        rows.append(
            [scen_id, t, mission_id]
            + [alt, spd, aoa, thr, o["ambient_temp_c"], o["air_density"], isa, hum]
            + [meas[c] for c in SENSOR_CHANNELS]
            + [o["prop_rpm"], o["thrust_margin"], o["lift_weight_margin"]]
            + res
            + [int(visible and max_sev >= FAULT_PRESENT_SEV), int(max_sev >= FAULT_PRESENT_SEV),
               round(z, 3), int(any(sf_flag.values())),
               condition, m["health_index"],
               max(0.0, min(tbo, wear_out_known) - hours),
               max(0.0, min(tbo, wear_out_oracle) - hours),
               max(0.0, tbo - hours), hours, eng.life_used_h, life_scale,
               int(failed), int(m["health_index"] <= 0.0)]
            + [sev[n] for n in FAULT_NAMES]
            + [int(visible and sev[n] >= FAULT_PRESENT_SEV) for n in FAULT_NAMES]
            + [sf_flag[c] for c in FAULTABLE_CHANNELS]
            + [sf_active[c] for c in FAULTABLE_CHANNELS]
            + [sf_sev[c] for c in FAULTABLE_CHANNELS]
            + [round(e, 3) for e in eff])
        t += DT
        if failed:
            break

    primary = deg.faults[0].name if deg.faults else "none"
    meta = {
        "scenario_id": scen_id, "mission": mission, "life_scale": life_scale,
        "start_hours": round(start_h, 3), "tbo_hours": tbo, "duration_s": round(duration, 1),
        "life_stage": life_stage(start_h, tbo),
        "primary_family": FAMILY_OF.get(primary, "none"),
        "faults": json.dumps([{"name": f.name, "onset_h": round(f.onset_h, 2),
                               "depth": round(f.depth, 4)} for f in deg.faults]),
        "sensor_faults": json.dumps([{"channel": f.channel, "kind": f.kind,
                                      "onset_s": round(f.onset_s, 1), "severity": round(f.severity, 3)}
                                     for f in sensors.faults]),
        "wear_limited": wear_limited_start,
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Writer: one row group per flight
# ---------------------------------------------------------------------------
_SCHEMA = pa.schema([("scenario_id", pa.int32()), ("t", pa.float32()), ("mission_id", pa.int8())]
                    + [(c, pa.float32()) for c in F.FEATURE_COLS + LABEL_COLS])


class FlightWriter:
    def __init__(self, out_dir: str):
        self.dir = out_dir
        self.file_no = 0
        self.in_file = 0
        self.writer = None
        self.index: list = []

    def _open(self):
        path = os.path.join(self.dir, f"flights_{self.file_no:03d}.parquet")
        self.writer = pq.ParquetWriter(path, _SCHEMA, compression="zstd")
        self.in_file = 0

    def add(self, rows: list, meta: dict, split: str) -> None:
        if self.writer is None:
            self._open()
        df = pd.DataFrame(np.asarray(rows, dtype=np.float64), columns=ALL_COLS)
        table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
        self.writer.write_table(table, row_group_size=len(rows) + 1)
        self.index.append(dict(meta, split=split, file=f"flights_{self.file_no:03d}.parquet",
                               row_group=self.in_file, n_rows=len(rows)))
        self.in_file += 1
        if self.in_file >= FLIGHTS_PER_FILE:
            self.close()
            self.file_no += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def engine_seed(engine_model: str, base: int = 5000) -> int:
    return base + (zlib.crc32(engine_model.encode()) % 100_000)


def generate_engine(engine_model: str, seed: int, n_flights: int, out_dir: str | None = None) -> dict:
    key = engine_model.split("_")[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = out_dir or os.path.join(root, "data", "rotax_v5")
    if not os.path.isabs(base):
        base = os.path.join(root, base)
    out = os.path.join(base, key)
    os.makedirs(out, exist_ok=True)

    rng = np.random.default_rng(seed)
    slot_rng = np.random.default_rng(seed + 1)          # split slots: independent of flights
    patterns: dict = {}
    counts: dict = {}
    writer = FlightWriter(out)
    t0 = last = time.time()
    scen = 0
    rows_total = 0
    while len(writer.index) < n_flights:
        scen_id = scen
        rows, meta = run_scenario(engine_model, scen_id, rng)
        scen += 1
        if len(rows) < MIN_ROWS:
            continue
        stratum = (meta["life_stage"], meta["mission"], meta["primary_family"])
        if stratum not in patterns:
            patterns[stratum] = list(slot_rng.permutation(SPLIT_SLOTS))
            counts[stratum] = 0
        split = patterns[stratum][counts[stratum] % len(SPLIT_SLOTS)]
        counts[stratum] += 1
        writer.add(rows, meta, split)
        rows_total += len(rows)
        now = time.time()
        if now - last >= 15.0:
            done = len(writer.index) / n_flights
            el = now - t0
            print(f"  [{key}] {len(writer.index):5d}/{n_flights} flights  {rows_total:>11,} rows  "
                  f"{rows_total / max(el, 1e-9):,.0f} rows/s  {100 * done:5.1f}%  "
                  f"eta {(el / max(done, 1e-9) - el) / 60:5.1f} min", flush=True)
            last = now
    writer.close()

    idx = pd.DataFrame(writer.index)
    idx.to_parquet(os.path.join(out, "index.parquet"), index=False)
    manifest = {
        "engine_model": engine_model, "key": key, "physics_version": "v5",
        "contract": F.contract(), "label_cols": LABEL_COLS,
        "fault_modes": FAULT_NAMES, "faultable_channels": list(FAULTABLE_CHANNELS),
        "seed": seed, "flights": len(idx), "rows": int(idx.n_rows.sum()), "attempted": scen,
        "generated_s": round(time.time() - t0, 1), "tbo_hours": P.ENGINE_SPECS[engine_model].tbo_hours,
        "splits": {s: {"flights": int((idx.split == s).sum()), "rows": int(idx.n_rows[idx.split == s].sum())}
                   for s in ("train", "cal", "val", "test")},
        "settings": {"effect_z": EFFECT_Z, "sensor_fault_p": SENSOR_FAULT_P,
                     "life_scale_p": LIFE_SCALE_P, "split_slots": "12/3/2/3 train/cal/val/test"},
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _worker(args):
    return generate_engine(*args)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dataset v5.")
    ap.add_argument("--engine", default="Rotax_914_ULF")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--flights", type=int, default=6000)
    ap.add_argument("--jobs", type=int, default=1, help="engines generated in parallel")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    engines = list(P.ENGINE_SPECS) if args.all else [args.engine]
    jobs = [(e, engine_seed(e), args.flights, args.out_dir) for e in engines]
    if args.jobs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.jobs, len(jobs))) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]
    for m in results:
        print(f"{m['key']}: {m['flights']} flights, {m['rows']:,} rows in "
              f"{m['generated_s'] / 60:.1f} min  " +
              "  ".join(f"{s} {v['flights']}" for s, v in m["splits"].items()))


if __name__ == "__main__":
    main()
