"""
Scenario-aware, disk-backed sliding-window data pipeline for the UAV digital twin.

Wraps a Python generator (pyarrow predicate-pushdown reads, one scenario at a time)
in tf.data.Dataset - never loads a full parquet file into memory, windows never
cross scenario boundaries.

================================== v3 ========================================
THIS FILE IS THE SPECIFICATION. backend/ai.py and backend/main.py hold copies of
FEATURE_COLS and MUST be updated to match, in the same commit, whenever it changes
here. ai.py asserts its own copy is exactly N_FEATURES long.

What changed from v2, and why:

1. RUL LABEL IS NOW REAL ENGINE HOURS AGAINST TBO, AND IS NOT CENSORED.
   v2 labelled `rul_true` as seconds-to-failure inside a single flight, bounded by
   the generator's MAX_DURATION of 20,000 s - so the label could never exceed
   5.556 h, and 17% of scenarios were censored (NaN) because they never failed.
   v3 seeds each scenario at a random accumulated life, so the label is
   `TBO_hours - accumulated_hours` and spans the full 0..TBO range.
   => v2 MAE figures (914 0.4991 h, 912 0.7628 h, 915 1.0756 h) are NOT comparable
      to v3 figures. Judge v3 as a percentage of TBO, not in absolute hours.

2. x_rul_aux GAINS FOUR DEGRADATION RESIDUALS (6 -> 10).
   The original six are severity/time statistics. They had to be: in v2 `wear` was
   never fed back into any sensor channel, and the measured correlation of wear
   with the observables was +0.13 / +0.11 / -0.04 - statistically nothing. physics
   v3 fixes that, so the aux vector can now carry features that read degradation
   directly. Every one is computed from OBSERVABLE channels via closed-form
   baselines that ai.py must reproduce exactly (see expected_* below), so there is
   no access to `wear` and no label leakage.

3. injection_timing IS A MODEL INPUT (24 -> 25 features).
   Genuinely diagnostic for misfire and injector faults. Battery, alternator and
   ambient temperature stay MONITOR-ONLY - emitted and displayed, but not model
   inputs, because widening the vector further costs scaler/assert churn for
   little diagnostic value.

4. STRIDE 32 -> 64.
   Stride 32 on a 128-sample window is 75% overlap: highly redundant. With 10M
   rows per engine (10x the v2 volume for 912/915/916) stride 64 still yields
   ~156k windows per engine and roughly halves training wall-clock across four
   engines.

5. FAILURE-MODE HEAD.
   v2's FAULT_TYPES (Bias/Drift/Spike/Stuck-At/Noise) are SENSOR faults. The
   problem statement asks for ENGINE failure modes, which physics v3 simulates in
   backend/failure_modes.py. Those get their own label and their own head.
==============================================================================
"""
import tensorflow as tf
import pyarrow.parquet as pq
import numpy as np
import json, joblib, os

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "altitude", "throttle", "airspeed", "aoa", "air_density",
    "torque_available_nm", "engine_rpm", "prop_rpm", "prop_torque", "power_kw", "fuel_flow",
    "thrust", "lift", "drag", "thrust_margin", "lift_weight_margin",
    "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm_fault",
    "injection_timing",
]

# Emitted in telemetry and displayed on the dashboard, deliberately NOT model inputs.
MONITOR_ONLY_COLS = [
    "battery_voltage", "battery_current", "alternator_output",
    "ambient_temp_c", "isa_dev_c", "density_in_envelope",
]

# Sensor channels - the diagnosis/severity heads emit one value per channel.
CHANNELS = ["egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm"]
FLAG_COLS = [f"{c}_flag" for c in CHANNELS]
STRESS_COLS = [f"{c}_stress" for c in CHANNELS]

# Sensor-fault taxonomy (unchanged from v2): what a SENSOR does wrong.
FAULT_TYPES = ["none", "Bias", "Drift", "Spike", "Stuck-At", "Noise"]

# Engine failure modes: what the ENGINE does wrong. One severity per mode, 0..1.
FAILURE_MODES = [
    "misfire",
    "injector_fouling",
    "cooling_degradation",
    "combustion_instability",
]
FAILURE_MODE_COLS = [f"fm_{m}" for m in FAILURE_MODES]

N_FEATURES = len(FEATURE_COLS)          # 25
N_CHANNELS = len(CHANNELS)              # 8
N_FAILURE_MODES = len(FAILURE_MODES)    # 4
N_RUL_AUX = 10

WINDOW_SIZE = 128
STRIDE = 64

# A window counts as "fault present" only once the fault is actually DETECTABLE.
#
# Measured on v3 914 data: 59.9% of rows carry a non-zero fault flag or failure
# mode, but 23% of those positives have a strongest signal below 0.01 and 30%
# below 0.05 - faults that have only just begun. A 128-sample window at 0.3%
# severity is physically indistinguishable from a healthy one, so labelling it
# positive is label noise, not supervision. It forced an accuracy ceiling near
# 86% and taught the head to guess on ambiguous windows.
#
# This does NOT hide those rows - the severity head still regresses the true
# value, and the failure-mode head still sees the real severities. It only stops
# the BINARY detector being graded on cases no detector could call.
DETECTION_THRESHOLD = 0.05

# RUL ground truth written by the generator: engine hours remaining against that
# engine's TBO. v2's column was `rul_true`, in seconds, censored.
RUL_COL = "rul_hours_true"

# Fixed for EVERY engine. Do NOT make this per-engine - ai.py's docstring records
# that change as a confirmed prior regression.
SEVERITY_POWER_DIVISOR = 85.0
SEVERITY_RPM_DIVISOR = 5800.0
RECENT_WINDOW = 60  # seconds

# Nominal brake-specific fuel consumption. Identical (0.300) for all four engines
# in ENGINE_CONFIGS, so a constant here rather than a per-engine lookup.
NOMINAL_BSFC = 0.300


# ---------------------------------------------------------------------------
# Closed-form nominal baselines for the degradation residuals.
#
# These MUST be reproduced exactly in backend/ai.py, because x_rul_aux is
# recomputed there at inference time from the live telemetry stream. They are
# deliberately simple and depend only on observable channels - no physics twin,
# no access to `wear`.
#
# Each mirrors the healthy (wear = 0) form of the corresponding physics v3
# channel. A worn engine departs from these, and that departure IS the signal.
# ---------------------------------------------------------------------------
def expected_oil_pressure(engine_rpm, throttle, oil_temp):
    """physics v3 oil_p evaluated at wear = 0."""
    return 90.0*(1.0 - np.exp(-engine_rpm/1500.0)) + 3.0*throttle - 0.15*(oil_temp - 80.0)


def expected_vibz(engine_rpm):
    """physics v3 vibz evaluated at wear = 0."""
    return 0.03 + 0.00379*(engine_rpm/1000.0)**2


def expected_cht(throttle, engine_rpm, power_kw, airspeed, isa_dev_c=0.0):
    """physics v3 cht_target evaluated at wear = 0 and with healthy cooling.

    When cooling degradation is active the real CHT climbs above this, which is
    precisely the detection signal - so this must NOT model the failure-mode
    cooling multiplier.
    """
    cool = np.clip(airspeed, 0.0, 60.0)
    return 75.0 + 30.0*throttle + 0.004*engine_rpm + 0.25*power_kw - 0.35*cool + 0.7*isa_dev_c


def compute_rul_aux(df):
    """The 10 x_rul_aux features as arrays over one whole scenario.

    Factored out of the generator so ai.py's incremental implementation can be
    parity-tested against it directly - the pattern that caught the v2 feature
    bugs. Returns a dict of 1-D float32 arrays, each len(df) long.
    """
    rpm = df["engine_rpm"].to_numpy(dtype=np.float32)
    power = df["power_kw"].to_numpy(dtype=np.float32)
    throttle = df["throttle"].to_numpy(dtype=np.float32)

    rpm_frac = rpm / SEVERITY_RPM_DIVISOR
    power_frac = power / SEVERITY_POWER_DIVISOR
    severity_proxy = 0.5*rpm_frac**2 + 0.5*power_frac**2
    n = len(severity_proxy)
    ramp = np.arange(1, n + 1, dtype=np.float32)

    running_mean_severity = np.cumsum(severity_proxy) / ramp
    running_max_severity = np.maximum.accumulate(severity_proxy)
    elapsed_hours = df["elapsed_hours"].to_numpy(dtype=np.float32)

    cumsum_padded = np.concatenate([[0.0], np.cumsum(severity_proxy)])
    idx_arr = np.arange(n)
    lo = np.maximum(0, idx_arr - RECENT_WINDOW + 1)
    recent_severity_mean = (cumsum_padded[idx_arr + 1] - cumsum_padded[lo]) / (idx_arr - lo + 1)
    severity_trend = recent_severity_mean - running_mean_severity

    cum_high_throttle_frac = np.cumsum((throttle > 0.7).astype(np.float32)) / ramp

    # --- degradation residuals (new in v3) ---------------------------------
    eps = 1e-6
    oil_p = df["oil_pressure"].to_numpy(dtype=np.float32)
    oil_t = df["oil_temp"].to_numpy(dtype=np.float32)
    cht = df["cht"].to_numpy(dtype=np.float32)
    vibz = df["vibz"].to_numpy(dtype=np.float32)
    fuel = df["fuel_flow"].to_numpy(dtype=np.float32)

    # Oil pressure as a FRACTION of what this RPM should produce - the classic
    # wear tell. Deliberately a ratio, not a difference: physics v3 scales oil
    # pressure by (1 - 0.30*wear), so `expected - measured` evaluates to
    # f(rpm)*0.30*wear and the rpm term swamps the wear term. Measured on smoke
    # data, the difference form scored r(wear) = +0.13 against the ratio's -0.95.
    oil_press_ratio = oil_p / np.maximum(expected_oil_pressure(rpm, throttle, oil_t), eps)

    # Vibration ABOVE the rotating-unbalance baseline for this RPM.
    vib_ratio = vibz / np.maximum(expected_vibz(rpm), eps)

    # Fuel burned per kW versus nominal BSFC. Under physics v3 this rises as
    # 1 + WEAR_BSFC_FRAC*wear, so it is close to a direct read of wear.
    # ABLATE THIS in the RUL notebook: if the head leans on it alone, it has not
    # learned anything from the 128-sample window.
    bsfc_ratio = (fuel / np.maximum(power, eps)) / NOMINAL_BSFC

    # CHT above what this operating point should produce. Captures both wear
    # (blow-by) and cooling degradation, neither of which moves the operating
    # point. A running-minimum proxy was tried first and scored only +0.26.
    airspeed = df["airspeed"].to_numpy(dtype=np.float32)
    isa = (df["isa_dev_c"].to_numpy(dtype=np.float32)
           if "isa_dev_c" in df.columns else np.zeros_like(cht))
    cht_excess = cht - expected_cht(throttle, rpm, power, airspeed, isa)

    return {
        "running_mean_severity": running_mean_severity.astype(np.float32),
        "elapsed_hours": elapsed_hours.astype(np.float32),
        "running_max_severity": running_max_severity.astype(np.float32),
        "recent_severity_mean": recent_severity_mean.astype(np.float32),
        "severity_trend": severity_trend.astype(np.float32),
        "cum_high_throttle_frac": cum_high_throttle_frac.astype(np.float32),
        "oil_press_ratio": oil_press_ratio.astype(np.float32),
        "vib_ratio": vib_ratio.astype(np.float32),
        "bsfc_ratio": bsfc_ratio.astype(np.float32),
        "cht_excess": cht_excess.astype(np.float32),
    }


# Order is part of the contract. ai.py must build its vector in exactly this order.
RUL_AUX_ORDER = [
    "running_mean_severity", "elapsed_hours", "running_max_severity",
    "recent_severity_mean", "severity_trend", "cum_high_throttle_frac",
    "oil_press_ratio", "vib_ratio", "bsfc_ratio", "cht_excess",
]
assert len(RUL_AUX_ORDER) == N_RUL_AUX


def scenario_window_generator(index_path, data_dir, split, scaler_path,
                              window_size=WINDOW_SIZE, stride=STRIDE):
    """Yields one window at a time.

    Reads ONE scenario's rows from disk via pyarrow predicate pushdown (never a
    full parquet file), builds all windows for that scenario, then moves on -
    windows never cross scenario boundaries because each is sliced from that
    scenario's own array only.
    """
    with open(index_path) as f:
        full_index = json.load(f)
    entries = full_index[split]
    scaler = joblib.load(scaler_path)   # fit on TRAIN ONLY
    read_cols = (FEATURE_COLS + FLAG_COLS + STRESS_COLS + FAILURE_MODE_COLS
                 + [RUL_COL, "scenario_id", "time", "elapsed_hours"])

    for entry in entries:
        path = os.path.join(data_dir, entry["file"])
        tbl = pq.read_table(path, columns=read_cols,
                            filters=[("scenario_id", "=", entry["scenario_id"])])
        df = tbl.to_pandas().sort_values("time").reset_index(drop=True)

        feats = scaler.transform(df[FEATURE_COLS]).astype(np.float32)
        flags = df[FLAG_COLS].to_numpy(dtype=np.float32)
        stress = df[STRESS_COLS].to_numpy(dtype=np.float32)
        fmodes = df[FAILURE_MODE_COLS].to_numpy(dtype=np.float32)

        # Detection fires on EITHER a sensor fault or an engine failure mode,
        # but only once it is strong enough to be detectable - see
        # DETECTION_THRESHOLD. Severity/failure-mode heads are unaffected: they
        # still regress the true values including the sub-threshold ones.
        strongest = np.maximum(stress.max(axis=1), fmodes.max(axis=1))
        label_detection = (strongest >= DETECTION_THRESHOLD).astype(np.float32)

        # v3 labels are uncensored by construction. rul_valid is kept so a future
        # partially-labelled dataset can be masked without a schema change.
        rul_hours = df[RUL_COL].to_numpy(dtype=np.float32)
        rul_valid_arr = np.isfinite(rul_hours) & (rul_hours >= 0)
        rul_hours = np.nan_to_num(rul_hours, nan=-1.0)

        aux = compute_rul_aux(df)
        aux_stack = np.stack([aux[k] for k in RUL_AUX_ORDER], axis=1).astype(np.float32)

        n = len(df)
        for start in range(0, max(1, n - window_size + 1), stride):
            end = start + window_size
            if end > n:
                break
            idx = end - 1
            yield (
                {"x": feats[start:end], "x_rul_aux": aux_stack[idx]},
                {
                    "y_detection": label_detection[idx],
                    "y_diagnosis": flags[idx],
                    "y_severity": stress[idx],
                    "y_failure_mode": fmodes[idx],
                    "y_rul_hours": rul_hours[idx],
                    "rul_valid": np.float32(rul_valid_arr[idx]),
                },
            )


def make_dataset(index_path, data_dir, split, scaler_path,
                 window_size=WINDOW_SIZE, stride=STRIDE, batch_size=128,
                 shuffle_buffer=2000):
    """Wraps scenario_window_generator in a tf.data.Dataset.

    output_signature declares the exact shape/dtype of every tensor up front -
    required by from_generator, and it doubles as a contract check against silent
    shape drift.
    """
    output_signature = (
        {
            "x": tf.TensorSpec(shape=(window_size, N_FEATURES), dtype=tf.float32),
            "x_rul_aux": tf.TensorSpec(shape=(N_RUL_AUX,), dtype=tf.float32),
        },
        {
            "y_detection": tf.TensorSpec(shape=(), dtype=tf.float32),
            "y_diagnosis": tf.TensorSpec(shape=(N_CHANNELS,), dtype=tf.float32),
            "y_severity": tf.TensorSpec(shape=(N_CHANNELS,), dtype=tf.float32),
            "y_failure_mode": tf.TensorSpec(shape=(N_FAILURE_MODES,), dtype=tf.float32),
            "y_rul_hours": tf.TensorSpec(shape=(), dtype=tf.float32),
            "rul_valid": tf.TensorSpec(shape=(), dtype=tf.float32),
        },
    )

    ds = tf.data.Dataset.from_generator(
        lambda: scenario_window_generator(index_path, data_dir, split, scaler_path,
                                          window_size, stride),
        output_signature=output_signature,
    )

    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer)

    # The underlying generator is FINITE (one pass over all scenarios in this
    # split). .repeat() makes the Dataset cycle indefinitely, which model.fit()
    # with an explicit steps_per_epoch REQUIRES - without it, steps_per_epoch *
    # epochs can exceed the available windows and training hangs partway through
    # (this exact failure happened during v2 Phase 2 training).
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def steps_per_epoch(index_path, data_dir, split, batch_size=128,
                    window_size=WINDOW_SIZE, stride=STRIDE):
    """Steps for ONE true pass over a split.

    The v2 notebooks used `rows / batch_size`, which is wrong: the number of
    training examples is the number of WINDOWS, roughly `rows / stride`. At v2
    volumes that merely meant an "epoch" was not a full pass. At 10M rows per
    engine the old formula asks for 78,125 steps - about 65 hours per epoch.

    Counts windows exactly, per scenario, so partial trailing windows are not
    over-counted.
    """
    with open(index_path) as f:
        full_index = json.load(f)

    by_file = {}
    for entry in full_index[split]:
        by_file.setdefault(entry["file"], []).append(entry["scenario_id"])

    total_windows = 0
    for fname, sids in by_file.items():
        tbl = pq.read_table(os.path.join(data_dir, fname), columns=["scenario_id"])
        arr = tbl.column("scenario_id").to_numpy()
        wanted = set(sids)
        uniq, counts = np.unique(arr, return_counts=True)
        for sid, n in zip(uniq.tolist(), counts.tolist()):
            if sid in wanted and n >= window_size:
                total_windows += 1 + (n - window_size) // stride

    return max(1, total_windows // batch_size)
