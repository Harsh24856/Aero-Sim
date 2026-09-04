"""
CORRECTED multi-engine data generation - replaces generate_multi_engine_data.py,
which used the wrong template (generate_training_data.py: fixed 500s scenarios, no
RUL labels at all). This reuses generate_unified_database.py's ACTUAL approach: real
"flight leg" behavior, scenarios run until wear-based failure or a 20,000s censoring
cutoff, RUL computed in hindsight once each scenario's outcome is known.

Targets ~1M rows per engine by generating scenarios until that row count is reached
(scenario length is now variable - could be ~7,000 to 20,000 rows each - not fixed),
rather than a fixed scenario count.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from physics import UAVEngineTwin

TARGET_ROWS = 1_000_000
MAX_DURATION = 20000.0
DT = 1.0
LEG_MIN, LEG_MAX = 300, 800
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10

COLUMNS = [
    "scenario_id","time","altitude","throttle","airspeed","aoa","air_density",
    "torque_available_nm","engine_rpm","prop_rpm","prop_torque","power_kw","fuel_flow",
    "thrust","lift","drag","thrust_margin","lift_weight_margin",
    "egt_healthy","cht_healthy","oil_pressure_healthy","oil_temp_healthy",
    "vibx_healthy","viby_healthy","vibz_healthy",
    "egt","cht","oil_pressure","oil_temp","vibx","viby","vibz","rpm_fault",
    "egt_flag","cht_flag","oil_pressure_flag","oil_temp_flag",
    "vibx_flag","viby_flag","vibz_flag","rpm_flag",
    "egt_stress","cht_stress","oil_pressure_stress","oil_temp_stress",
    "vibx_stress","viby_stress","vibz_stress","rpm_stress",
    "wear","is_failed_now",
    "scenario_failed","scenario_failure_time","scenario_censored","rul_true",
]

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def sample_leg_target(severity_bias):
    throttle = clip(np.random.uniform(0.15, 0.35) + severity_bias*np.random.uniform(0.4, 0.65), 0.1, 1.0)
    airspeed = clip(np.random.uniform(45, 70) - severity_bias*np.random.uniform(20, 55), 0.0, 80.0)
    altitude = np.random.uniform(0, 8000)
    aoa = np.random.uniform(-10, 20)
    return altitude, throttle, airspeed, aoa

def generate_engine_dataset(engine_model: str, seed: int, target_rows: int) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    t_start = time.time()
    scen = 0
    n_failed, n_censored = 0, 0

    while len(rows) < target_rows:
        twin = UAVEngineTwin(dt=DT, engine_model=engine_model)
        severity_bias = np.random.uniform(0.0, 1.0)
        altitude, throttle, airspeed, aoa = sample_leg_target(severity_bias)
        twin.altitude, twin.throttle, twin.airspeed, twin.aoa = altitude, throttle, airspeed, aoa
        next_leg_change = np.random.uniform(LEG_MIN, LEG_MAX)

        buffer = []
        t = 0.0
        while t < MAX_DURATION:
            if t >= next_leg_change:
                altitude, throttle, airspeed, aoa = sample_leg_target(severity_bias)
                next_leg_change = t + np.random.uniform(LEG_MIN, LEG_MAX)
            twin.throttle = clip(twin.throttle + np.random.normal(0, 0.008) + 0.15*(throttle-twin.throttle), 0.1, 1.0)
            twin.altitude = clip(twin.altitude + np.random.normal(0, 8.0) + 0.05*(altitude-twin.altitude), 0.0, 9000.0)
            twin.airspeed = clip(twin.airspeed + np.random.normal(0, 0.4) + 0.10*(airspeed-twin.airspeed), 0.0, 80.0)
            twin.aoa = clip(twin.aoa + np.random.normal(0, 0.2) + 0.10*(aoa-twin.aoa), -15.0, 25.0)

            out = twin.step()
            h, ff, fs = out["healthy"], out["fault_flags"], out["fault_stress"]
            buffer.append([
                out["time"], out["altitude"], out["throttle"], out["airspeed"], out["aoa"],
                out["air_density"], out["torque_available_nm"], out["engine_rpm"], out["prop_rpm"],
                out["prop_torque"], out["power_kw"], out["fuel_flow"], out["thrust"], out["lift"],
                out["drag"], out["thrust_margin"], out["lift_weight_margin"],
                h["egt"], h["cht"], h["oil_pressure"], h["oil_temp"], h["vibx"], h["viby"], h["vibz"],
                out["egt"], out["cht"], out["oil_pressure"], out["oil_temp"],
                out["vibx"], out["viby"], out["vibz"], out["rpm_fault"],
                ff["egt"], ff["cht"], ff["oil_pressure"], ff["oil_temp"],
                ff["vibx"], ff["viby"], ff["vibz"], ff["rpm"],
                fs["egt"], fs["cht"], fs["oil_pressure"], fs["oil_temp"],
                fs["vibx"], fs["viby"], fs["vibz"], fs["rpm"],
                out["wear"], int(out["failed"]),
            ])
            t = out["time"]
            if out["failed"]:
                break

        failed = twin.failed
        failure_time = twin.failure_time if failed else None
        censored = not failed
        n_failed += int(failed)
        n_censored += int(censored)

        for row in buffer:
            rt = row[0]
            rul = (failure_time - rt) if failed else None
            rows.append([scen] + row + [int(failed), failure_time, int(censored), rul])

        scen += 1
        if scen % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  [{engine_model}] scenario {scen}, rows={len(rows):,}/{target_rows:,}, "
                  f"failed={n_failed}, censored={n_censored}, elapsed={elapsed:.1f}s")
            sys.stdout.flush()

    df = pd.DataFrame(rows, columns=COLUMNS)
    print(f"[{engine_model}] generation complete: {len(df):,} rows, {scen} scenarios, "
          f"{n_failed} failed, {n_censored} censored, {time.time()-t_start:.1f}s")
    return df

def save_chunks(df: pd.DataFrame, output_dir: str, engine_model: str):
    os.makedirs(output_dir, exist_ok=True)
    scenario_ids = sorted(df["scenario_id"].unique())
    n = len(scenario_ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    rng = np.random.RandomState(123)
    shuffled = list(scenario_ids)
    rng.shuffle(shuffled)
    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train:n_train + n_val])
    test_ids = set(shuffled[n_train + n_val:])

    def write_split(ids, filename):
        sub = df[df["scenario_id"].isin(ids)].reset_index(drop=True)
        path = os.path.join(output_dir, filename)
        sub.to_parquet(path, engine="pyarrow", index=False)
        print(f"  wrote {filename}: {len(sub):,} rows, {len(ids)} scenarios")

    write_split(train_ids, "train.parquet")
    write_split(val_ids, "val.parquet")
    write_split(test_ids, "test.parquet")

    index = {
        "engine_model": engine_model,
        "train": [int(x) for x in sorted(train_ids)],
        "val": [int(x) for x in sorted(val_ids)],
        "test": [int(x) for x in sorted(test_ids)],
        "n_scenarios_total": int(n), "rows_total": int(len(df)),
    }
    with open(os.path.join(output_dir, "scenario_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"  wrote scenario_index.json ({n} scenarios: {len(train_ids)}/{len(val_ids)}/{len(test_ids)} split)")


if __name__ == "__main__":
    NEW_ENGINES = ["Rotax_912_ULS", "Rotax_915_iS", "Rotax_916_iS"]
    BASE_DIR = "/Users/harsh/Documents/UAV_Engine/validation"

    for i, engine in enumerate(NEW_ENGINES):
        print(f"=== Generating {engine} ({i+1}/{len(NEW_ENGINES)}) - CORRECTED with RUL labels ===")
        df = generate_engine_dataset(engine, seed=42 + i, target_rows=TARGET_ROWS)
        out_dir = os.path.join(BASE_DIR, f"chunks_{engine.lower()}")
        save_chunks(df, out_dir, engine)
        print()

    print("ALL 3 ENGINES DONE (v2, with correct RUL labels).")
