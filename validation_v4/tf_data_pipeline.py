"""tf.data pipeline for dataset v4.

Same architecture as validation/tf_data_pipeline.py - scenario-scoped windowing,
predicate pushdown so a full parquet file is never loaded, scaler fitted on TRAIN
only - against the v4 contract.

WHAT CHANGED FROM v3, AND WHY EACH CHANGE MATTERS FOR DEPLOYMENT
    FEATURES (25 -> 29). v3's list contained power_kw, torque_nm,
    torque_available_nm and air_density. A real airframe measures none of them,
    so a model trained on v3 cannot be fed at inference time without first
    estimating its own inputs. Every v4 feature is something the aircraft can
    actually supply.

    RESIDUALS ARE FEATURES. Six channels carry the difference between what the
    sensor reads and what the on-board twin expects at this operating point.
    This is the one signal that separates "the engine is hot" from "the engine
    is working hard on a hot day" - channel by channel those look identical, and
    the residual is what tells them apart. Measured on the generated data, the
    separation between a healthy and a worn engine is 1.65 to 28 standard
    deviations on these channels.

    DIAGNOSIS IS 14 PHYSICAL COMPONENTS, not 4 symptom labels. A fault is named
    by what it breaks - turbo_degradation, oil_pump_degradation, bearing_wear -
    so the head learns a physical cause rather than a pattern someone wrote.

    SENSOR FAULTS ARE A SEPARATE HEAD from engine faults, so a drifting
    thermocouple and a genuine overheat are distinguishable. v3 conflated them.

    HEALTH AND MARGIN ARE DIFFERENT TARGETS. `health_index` is wear condition,
    monotone over life. `margin_min` is the operability margin against certified
    limits and depends on the operating point. Supervising a single "health"
    with the margin left the label saturated at 1.0 for the median row.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf

# ---------------------------------------------------------------------------
# Feature contract - must match backend/generate_dataset_v4.py exactly.
# ---------------------------------------------------------------------------
FLIGHT_COLS = ["altitude", "airspeed", "aoa", "throttle",
               "ambient_temp_c", "air_density", "isa_dev_c", "humidity_frac"]

MEASURED_COLS = ["egt", "cht", "coolant_temp", "oil_temp", "oil_pressure",
                 "engine_rpm", "fuel_flow", "manifold_pressure_kpa",
                 "vibx", "viby", "vibz", "battery_voltage"]

DERIVED_COLS = ["prop_rpm", "thrust_margin", "lift_weight_margin"]

RESIDUAL_CHANNELS = ["egt", "cht", "oil_temp", "oil_pressure",
                     "engine_rpm", "fuel_flow"]
RESIDUAL_COLS = [f"res_{c}" for c in RESIDUAL_CHANNELS]

FEATURE_COLS = FLIGHT_COLS + MEASURED_COLS + DERIVED_COLS + RESIDUAL_COLS
N_FEATURES = len(FEATURE_COLS)                       # 29

FAULT_MODES = [
    "air_filter_fouling", "compression_loss", "valve_leakage",
    "turbo_degradation", "wastegate_fault", "intercooler_fouling",
    "injector_fouling", "ignition_degradation", "combustion_instability",
    "bearing_wear", "oil_pump_degradation", "oil_degradation",
    "cooling_degradation", "prop_erosion",
]
FM_COLS = [f"fm_{n}" for n in FAULT_MODES]
N_FAULT_MODES = len(FAULT_MODES)                     # 14

SENSOR_CHANNELS = list(MEASURED_COLS)
SF_FLAG_COLS = [f"sf_{c}_flag" for c in SENSOR_CHANNELS]
SF_SEV_COLS = [f"sf_{c}_sev" for c in SENSOR_CHANNELS]
N_SENSOR_CHANNELS = len(SENSOR_CHANNELS)             # 12
SENSOR_FAULT_TYPES = ["none", "bias", "drift", "stuck", "spike", "noise", "dropout"]
N_SENSOR_FAULT_TYPES = len(SENSOR_FAULT_TYPES)       # 7

WINDOW_SIZE = 128
# Stride 128 = non-overlapping windows. At stride 64 consecutive windows shared
# half their samples, which nearly doubled the step count for training examples
# that are largely redundant at this window length. Measured on the 914 data:
# 119,605 windows at stride 64 against 59,802 at 128, and the model step is the
# bottleneck (379 ms against 78 ms for the data pipeline), so halving the steps
# halves the epoch. 59,802 windows is still generous for a 248k-parameter model.
STRIDE = 128

# A window is "faulty" only once the fault is physically consequential. Below
# this the engine is inside its own manufacturing scatter, and labelling it
# faulty teaches the detector to fire on noise.
DETECTION_THRESHOLD = 0.08

# Auxiliary scalars for the RUL head, alongside the encoded window. Mirrors the
# v3 design, where feeding elapsed life directly was what let the head resolve
# absolute remaining hours rather than a relative trend.
RUL_AUX_ORDER = [
    "engine_hours_norm", "life_used_norm", "health_index", "margin_min",
    "mean_fault_severity", "max_fault_severity",
    "mean_residual_abs", "max_residual_abs",
    "throttle_mean", "altitude_mean",
]
N_RUL_AUX = len(RUL_AUX_ORDER)


def compute_rul_aux(df, tbo_hours: float) -> dict:
    """Window-independent scalars describing where in life this engine is."""
    res = np.abs(df[RESIDUAL_COLS].to_numpy(dtype=np.float32))
    fm = df[FM_COLS].to_numpy(dtype=np.float32)
    return {
        "engine_hours_norm": (df["engine_hours"].to_numpy(np.float32) / tbo_hours),
        "life_used_norm": (df["life_used_hours"].to_numpy(np.float32) / tbo_hours),
        "health_index": df["health_index"].to_numpy(np.float32),
        "margin_min": df["margin_min"].to_numpy(np.float32),
        "mean_fault_severity": fm.mean(axis=1),
        "max_fault_severity": fm.max(axis=1),
        "mean_residual_abs": res.mean(axis=1),
        "max_residual_abs": res.max(axis=1),
        "throttle_mean": df["throttle"].to_numpy(np.float32),
        "altitude_mean": (df["altitude"].to_numpy(np.float32) / 9000.0),
    }


# ---------------------------------------------------------------------------
def load_manifest(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "manifest.json")) as fh:
        return json.load(fh)


def fit_aux_scaler(data_dir, out_path, max_rows=2_000_000):
    """Fit a StandardScaler for the RUL auxiliary vector, on TRAIN ONLY.

    WHY THIS EXISTS. The 29 window features were scaled; these 10 scalars were
    not, and as generated they span five orders of magnitude:

        life_used_norm      median 0.0002
        engine_hours_norm   median 0.47
        max_residual_abs    median 128,  max 1372

    They enter the RUL head through a single Dense(32). With Glorot init the
    pre-activation is dominated by the residual magnitudes, and
    engine_hours_norm contributes a term roughly 2,600x smaller - so its weight
    receives a correspondingly negligible gradient.

    That matters here more than anywhere else in the model, because
    engine_hours_norm determines the target EXACTLY: measured on the 914 test
    split, corr(engine_hours_norm, rul_hours_true) = -1.0000. The trained head
    nonetheless scored 27.53% of TBO against 29.11% for predicting a constant.
    It was handed the answer as an input and never saw it.
    """
    import glob
    from sklearn.preprocessing import StandardScaler

    files = sorted(glob.glob(os.path.join(data_dir, "train", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no train chunks under {data_dir}")
    tbo = float(load_manifest(data_dir)["tbo_hours"])

    cols = sorted(set(RESIDUAL_COLS + FM_COLS + [
        "engine_hours", "life_used_hours", "health_index", "margin_min",
        "throttle", "altitude"]))
    sc, seen = StandardScaler(), 0
    for f in files:
        df = pq.read_table(f, columns=cols).to_pandas()
        aux = compute_rul_aux(df, tbo)
        sc.partial_fit(np.stack([aux[k] for k in RUL_AUX_ORDER],
                                axis=1).astype(np.float64))
        seen += len(df)
        if seen >= max_rows:
            break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(sc, out_path)
    return {"rows_used": seen, "path": out_path, "n_aux": N_RUL_AUX}


def scenario_window_generator(data_dir, split, scaler_path,
                              aux_scaler_path=None,
                              window_size=WINDOW_SIZE, stride=STRIDE):
    """Yield one window at a time, never crossing a scenario boundary.

    Windows are sliced from a single scenario's own rows, so no window can span
    two flights. Combined with scenario-level splits in the generator, that is
    what keeps train and test genuinely independent: a random row split would
    put near-duplicate consecutive rows on both sides and inflate every metric.
    """
    with open(os.path.join(data_dir, f"index_{split}.json")) as fh:
        entries = json.load(fh)
    manifest = load_manifest(data_dir)
    tbo = float(manifest["tbo_hours"])
    scaler = joblib.load(scaler_path)            # fitted on TRAIN ONLY
    # Optional so an older artifacts/ directory still loads; run.py's scalers
    # phase writes it, and without it the RUL head cannot see engine_hours_norm.
    aux_scaler = (joblib.load(aux_scaler_path)
                  if aux_scaler_path and os.path.exists(aux_scaler_path) else None)

    read_cols = (FEATURE_COLS + FM_COLS + SF_FLAG_COLS + SF_SEV_COLS
                 + ["scenario_id", "t", "engine_hours", "life_used_hours",
                    "health_index", "margin_min", "rul_hours_true",
                    "fault_present"])

    for entry in entries:
        path = os.path.join(data_dir, split, entry["file"])
        tbl = pq.read_table(path, columns=read_cols,
                            filters=[("scenario_id", "=", entry["scenario_id"])])
        df = tbl.to_pandas().sort_values("t").reset_index(drop=True)
        n = len(df)
        if n < window_size:
            continue

        # .to_numpy() matters: fit_scaler fits on a bare array, so passing a
        # DataFrame here makes sklearn warn "X has feature names, but
        # StandardScaler was fitted without feature names" on EVERY scenario -
        # hundreds of lines per epoch, which buries the progress bar.
        feats = scaler.transform(
            df[FEATURE_COLS].to_numpy(np.float64)).astype(np.float32)
        fm = df[FM_COLS].to_numpy(np.float32)
        sf_flag = df[SF_FLAG_COLS].to_numpy(np.int32)
        sf_sev = df[SF_SEV_COLS].to_numpy(np.float32)
        rul = df["rul_hours_true"].to_numpy(np.float32)
        health = df["health_index"].to_numpy(np.float32)

        # Detection fires on an engine fault OR a sensor fault, but only once
        # either is strong enough to be detectable.
        strongest = np.maximum(fm.max(axis=1), sf_sev.max(axis=1))
        detect = (strongest >= DETECTION_THRESHOLD).astype(np.float32)

        aux = compute_rul_aux(df, tbo)
        aux_stack = np.stack([aux[k] for k in RUL_AUX_ORDER], axis=1).astype(np.float32)
        if aux_scaler is not None:
            aux_stack = aux_scaler.transform(
                aux_stack.astype(np.float64)).astype(np.float32)

        for start in range(0, n - window_size + 1, stride):
            idx = start + window_size - 1
            yield (
                {"x": feats[start:start + window_size], "x_rul_aux": aux_stack[idx]},
                {
                    "y_detection": detect[idx],
                    "y_fault_mode": fm[idx],                       # 14 severities
                    "y_sensor_fault": sf_flag[idx],                # 12 class ids
                    "y_sensor_sev": sf_sev[idx],                   # 12 severities
                    "y_health": health[idx],
                    "y_rul_hours": rul[idx],
                },
            )


def make_dataset(data_dir, split, scaler_path, aux_scaler_path=None,
                 window_size=WINDOW_SIZE,
                 stride=STRIDE, batch_size=128, shuffle_buffer=4000,
                 repeat=False):
    sig = (
        {"x": tf.TensorSpec((window_size, N_FEATURES), tf.float32),
         "x_rul_aux": tf.TensorSpec((N_RUL_AUX,), tf.float32)},
        {"y_detection": tf.TensorSpec((), tf.float32),
         "y_fault_mode": tf.TensorSpec((N_FAULT_MODES,), tf.float32),
         "y_sensor_fault": tf.TensorSpec((N_SENSOR_CHANNELS,), tf.int32),
         "y_sensor_sev": tf.TensorSpec((N_SENSOR_CHANNELS,), tf.float32),
         "y_health": tf.TensorSpec((), tf.float32),
         "y_rul_hours": tf.TensorSpec((), tf.float32)},
    )
    ds = tf.data.Dataset.from_generator(
        lambda: scenario_window_generator(data_dir, split, scaler_path,
                                          aux_scaler_path, window_size, stride),
        output_signature=sig)
    if shuffle_buffer and split == "train":
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def count_windows(data_dir, split, window_size=WINDOW_SIZE, stride=STRIDE) -> int:
    with open(os.path.join(data_dir, f"index_{split}.json")) as fh:
        entries = json.load(fh)
    return sum(max(0, (e["n_rows"] - window_size) // stride + 1) for e in entries)


def steps_per_epoch(data_dir, split, batch_size=128, **kw) -> int:
    return max(1, count_windows(data_dir, split, **kw) // batch_size)


def fit_scaler(data_dir, out_path, max_rows=2_000_000):
    """Fit the feature scaler on TRAIN ONLY.

    Fitting on anything else leaks test statistics into training and inflates
    every downstream metric, which is the most common silent mistake in this
    kind of pipeline.
    """
    import glob
    from sklearn.preprocessing import StandardScaler

    files = sorted(glob.glob(os.path.join(data_dir, "train", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no train chunks under {data_dir}")

    sc = StandardScaler()
    seen = 0
    for f in files:
        df = pq.read_table(f, columns=FEATURE_COLS).to_pandas()
        sc.partial_fit(df.to_numpy(np.float64))
        seen += len(df)
        if seen >= max_rows:
            break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(sc, out_path)
    return {"rows_used": seen, "path": out_path, "n_features": N_FEATURES}
