"""
AI inference service for the physics-v3 digital twin - Rotax 914 / 915 / 916.

The v2 service (ai.py) is untouched and still serves backend/models/ for the live
demo. This file serves backend/models_v3/, produced by
validation/split_deployable_heads.py, and must be paired with main.py running
AERO_PHYSICS_VERSION=v3. Run ONE of ai.py / aiv3.py at a time - both bind 8100.

Launched exactly like ai.py, from backend/, with the validated Metal environment:

    cd backend && ../validation/venv/bin/uvicorn aiv3:app --host 127.0.0.1 --port 8100

What v3 changes against ai.py:
  - 25 input features (+ injection_timing) and 10 x_rul_aux inputs (+ four
    degradation residuals: oil_press_ratio, vib_ratio, bsfc_ratio, cht_excess).
  - Five heads per engine. Each comes from the training checkpoint where it
    measured best on the test split: detection phase1, diagnosis phase2,
    severity phase3, failure modes phase4, RUL phase5.
  - RUL is REAL ENGINE HOURS remaining against the engine's TBO. There is no
    rul_hours_internal field: the frontend multiplies that by the v2 RUL_SCALE.
  - Engine failure modes (misfire, injector fouling, cooling degradation,
    combustion instability) with per-mode thresholds measured on the test split.

Contract: copied verbatim from validation/tf_data_pipeline.py (v3) and checked at
startup against every engine's manifest.json, which the export script wrote from
the pipeline module itself. validation/parity_ai_v3.py feeds real test scenarios
through RollingWindowV3 one row at a time and requires x, x_rul_aux and all five
predictions to match the pipeline (measured 2026-09-13: x identical, aux <= 1.6e-5,
predictions identical).

No MC-dropout uncertainty: measured on these exports, model(x, training=...) differs
from .predict() by up to 459 h on RUL, so any direct-call estimate is invalid here.
"""
import json
import os
from collections import deque

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras

MODEL_VERSION = "v3"

# ============================================================================
# CONTRACT - verbatim from validation/tf_data_pipeline.py (v3).
# ============================================================================
WINDOW_SIZE = 128
FEATURE_COLS = [
    "altitude", "throttle", "airspeed", "aoa", "air_density",
    "torque_available_nm", "engine_rpm", "prop_rpm", "prop_torque", "power_kw", "fuel_flow",
    "thrust", "lift", "drag", "thrust_margin", "lift_weight_margin",
    "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm_fault",
    "injection_timing",
]
N_FEATURES = 25
assert len(FEATURE_COLS) == N_FEATURES

RUL_AUX_ORDER = [
    "running_mean_severity", "elapsed_hours", "running_max_severity",
    "recent_severity_mean", "severity_trend", "cum_high_throttle_frac",
    "oil_press_ratio", "vib_ratio", "bsfc_ratio", "cht_excess",
]
CHANNELS = ["egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm"]
FAULT_TYPES = ["none", "Bias", "Drift", "Spike", "Stuck-At", "Noise"]
FAILURE_MODES = ["misfire", "injector_fouling", "cooling_degradation", "combustion_instability"]

RECENT_WINDOW_SECONDS = 60
# Fixed for EVERY engine - making this per-engine is a confirmed prior regression.
SEVERITY_RPM_DIVISOR = 5800.0
SEVERITY_POWER_DIVISOR = 85.0
HIGH_THROTTLE_THRESHOLD = 0.7
NOMINAL_BSFC = 0.300
DETECTION_THRESHOLD = 0.05          # fallback "present" threshold for a failure mode
RUL_OUT_OF_RANGE_FRAC = 1.05        # RUL above 105% of TBO is extrapolation
NEAR_NEW_BAND_FRAC = 0.90           # predictions at/above this share of TBO use the nearly-new error band
RPM_FAULT_CONFIDENCE_FLOOR = 0.70   # same rule as ai.py

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(BACKEND_DIR, "models_v3")

ENGINE_REGISTRY = {
    "Rotax_914_ULF": "914",
    "Rotax_912_ULS": "912",   # no v3 export yet - skipped at load, not an error
    "Rotax_915_iS":  "915",
    "Rotax_916_iS":  "916",
}


def expected_oil_pressure(engine_rpm, throttle, oil_temp):
    """physics v3 oil_p at wear = 0 - verbatim from tf_data_pipeline.py."""
    return 90.0*(1.0 - np.exp(-engine_rpm/1500.0)) + 3.0*throttle - 0.15*(oil_temp - 80.0)


def expected_vibz(engine_rpm):
    """physics v3 vibz at wear = 0 - verbatim from tf_data_pipeline.py."""
    return 0.03 + 0.00379*(engine_rpm/1000.0)**2


def expected_cht(throttle, engine_rpm, power_kw, airspeed, isa_dev_c=0.0):
    """physics v3 cht_target at wear = 0, healthy cooling - verbatim."""
    cool = np.clip(airspeed, 0.0, 60.0)
    return 75.0 + 30.0*throttle + 0.004*engine_rpm + 0.25*power_kw - 0.35*cool + 0.7*isa_dev_c


class RollingWindowV3:
    """Builds the (1,128,25) input and (1,10) x_rul_aux one telemetry row at a time.

    tf_data_pipeline computes the same quantities over a whole scenario with
    cumulative numpy ops; live serving only ever has "so far", so the running
    statistics are accumulated incrementally from the first row after a reset.
    The four residuals are read from the NEWEST row, which is where the pipeline
    takes x_rul_aux (the window's last index).

    cht_excess uses isa_dev_c = 0 ON PURPOSE. The training generator reads only the
    feature columns, labels and time/elapsed_hours from the parquet, so isa_dev_c was
    absent and compute_rul_aux used zeros for every training window. The live ISA
    deviation would be better physics and a different input distribution.
    """

    def __init__(self, scaler):
        self.scaler = scaler
        self.buffer: deque = deque(maxlen=WINDOW_SIZE)
        self.n_steps = 0
        self._severity_sum = 0.0
        self._severity_max = 0.0
        self._recent_severity: deque = deque(maxlen=RECENT_WINDOW_SECONDS)
        self._high_throttle_count = 0

    def update(self, raw: dict):
        """raw must contain every key in FEATURE_COLS plus "time" (simulated seconds,
        the twin's own clock). Extra keys are ignored."""
        self.buffer.append(raw)
        self.n_steps += 1
        rpm_frac = raw["engine_rpm"] / SEVERITY_RPM_DIVISOR
        power_frac = raw["power_kw"] / SEVERITY_POWER_DIVISOR
        severity_proxy = 0.5 * rpm_frac**2 + 0.5 * power_frac**2
        self._severity_sum += severity_proxy
        self._severity_max = max(self._severity_max, severity_proxy)
        self._recent_severity.append(severity_proxy)
        if raw["throttle"] > HIGH_THROTTLE_THRESHOLD:
            self._high_throttle_count += 1

    def is_ready(self) -> bool:
        return len(self.buffer) == WINDOW_SIZE

    def get_window(self) -> np.ndarray:
        """Scaled (1, 128, 25). Named DataFrame because the scaler was fit on named
        columns - a bare array is order-dependent and silently wrong if it drifts."""
        raw_matrix = np.array([[step[c] for c in FEATURE_COLS] for step in self.buffer],
                              dtype=np.float32)
        raw_df = pd.DataFrame(raw_matrix, columns=FEATURE_COLS)
        return self.scaler.transform(raw_df).astype(np.float32)[np.newaxis, ...]

    def get_rul_aux(self) -> np.ndarray:
        """(1, 10) in RUL_AUX_ORDER."""
        n = self.n_steps
        running_mean_severity = self._severity_sum / n
        elapsed_hours = self.buffer[-1]["time"] / 3600.0
        running_max_severity = self._severity_max
        recent_severity_mean = sum(self._recent_severity) / len(self._recent_severity)
        severity_trend = recent_severity_mean - running_mean_severity
        cum_high_throttle_frac = self._high_throttle_count / n

        r = self.buffer[-1]
        f32 = np.float32
        rpm, thr = f32(r["engine_rpm"]), f32(r["throttle"])
        power, oil_t = f32(r["power_kw"]), f32(r["oil_temp"])
        eps = f32(1e-6)
        oil_press_ratio = f32(r["oil_pressure"]) / max(f32(expected_oil_pressure(rpm, thr, oil_t)), eps)
        vib_ratio = f32(r["vibz"]) / max(f32(expected_vibz(rpm)), eps)
        bsfc_ratio = (f32(r["fuel_flow"]) / max(power, eps)) / f32(NOMINAL_BSFC)
        cht_excess = f32(r["cht"]) - f32(expected_cht(thr, rpm, power, f32(r["airspeed"]), 0.0))
        return np.array([[
            running_mean_severity, elapsed_hours, running_max_severity,
            recent_severity_mean, severity_trend, cum_high_throttle_frac,
            oil_press_ratio, vib_ratio, bsfc_ratio, cht_excess,
        ]], dtype=np.float32)


# ============================================================================
# BACKEND GUARD - same reason as ai.py: these weights were trained and validated
# on the Metal GPU backend, and a CPU-only TensorFlow gives different answers.
# ============================================================================
GPU_DEVICES = tf.config.list_physical_devices("GPU")
BACKEND_VALIDATED = bool(GPU_DEVICES)
if not BACKEND_VALIDATED:
    print("=" * 78)
    print("WARNING: TensorFlow sees no GPU backend. These v3 weights were trained and")
    print("         validated on Metal; output here is NOT trustworthy. Launch from backend/ with")
    print("           ../validation/venv/bin/uvicorn aiv3:app --host 127.0.0.1 --port 8100")
    print("=" * 78)


# ============================================================================
# MODEL LOADING - every exported engine, eagerly, contract-checked.
# ============================================================================
def _load(path):
    return keras.models.load_model(path, safe_mode=False, compile=False)


def load_engine(key):
    """Returns None when this engine has no v3 export yet (912 today)."""
    model_dir = os.path.join(MODELS_ROOT, key)
    manifest_path = os.path.join(model_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    manifest = json.load(open(manifest_path))
    c = manifest["contract"]
    mismatches = [name for name, got, want in [
        ("feature_cols", c["feature_cols"], FEATURE_COLS),
        ("rul_aux_order", c["rul_aux_order"], RUL_AUX_ORDER),
        ("failure_modes", c["failure_modes"], FAILURE_MODES),
        ("channels", c["channels"], CHANNELS),
        ("fault_types", c["fault_types"], FAULT_TYPES),
        ("window_size", c["window_size"], WINDOW_SIZE),
        ("severity_rpm_divisor", c["severity_rpm_divisor"], SEVERITY_RPM_DIVISOR),
        ("severity_power_divisor", c["severity_power_divisor"], SEVERITY_POWER_DIVISOR),
    ] if got != want]
    if mismatches:
        raise RuntimeError(f"{key}: models_v3 manifest disagrees with aiv3.py on {mismatches} - "
                           "re-export, or change the contract here deliberately")
    heads = manifest["heads"]
    return {
        "manifest": manifest,
        "tbo_hours": float(manifest["tbo_hours"]),
        "scaler": joblib.load(os.path.join(model_dir, manifest["scaler"]["file"])),
        "detection_model":     _load(os.path.join(model_dir, heads["detection"]["file"])),
        "diagnosis_model":     _load(os.path.join(model_dir, heads["diagnosis"]["file"])),
        "severity_model":      _load(os.path.join(model_dir, heads["severity"]["file"])),
        "failure_modes_model": _load(os.path.join(model_dir, heads["failure_modes"]["file"])),
        "rul_model":           _load(os.path.join(model_dir, heads["rul"]["file"])),
        "failure_mode_thresholds": manifest.get("failure_mode_thresholds") or {},
        # Test-split RUL mean absolute error, shown as the RUL uncertainty band.
        "rul_mae_hours": (manifest.get("rul_test") or {}).get("mae_hours"),
        # The head under-predicts nearly-new engines (true RUL >= 97% TBO) by 37-109 h on the
        # test split - larger than its overall MAE - so that error is the honest band there.
        "rul_near_new_mae_hours": (manifest.get("rul_test") or {}).get("near_new_mae_hours"),
        # Channels whose confident fault calls were no better than chance on the test
        # split (validation/parity_ai_v3.py). Missing section -> every channel trusted.
        "unreliable_channels": {
            ch for ch, r in ((manifest.get("diagnosis_channel_reliability") or {}).get("channels") or {}).items()
            if not r.get("reliable", True)
        },
    }


print("Loading v3 models for all registered engines...")
loaded_engines: dict = {}
for engine_name, key in ENGINE_REGISTRY.items():
    engine = load_engine(key)
    if engine is None:
        print(f"  {engine_name}: no v3 export in {MODELS_ROOT}/{key} - skipped")
        continue
    loaded_engines[engine_name] = engine
    print(f"  {engine_name}: loaded (TBO {engine['tbo_hours']:.0f} h)")
if not loaded_engines:
    raise RuntimeError(f"No v3 engines found under {MODELS_ROOT}")
print(f"All {len(loaded_engines)} engines loaded.")

# Warm every model once. The first .predict() on a model builds its tf.function graph,
# which takes seconds; left to the first real request it arrived as a timeout right
# after the 128 s warm-up and forced a fresh warm-up. Pay it at startup instead.
for engine_name, engine in loaded_engines.items():
    _x = np.zeros((1, WINDOW_SIZE, N_FEATURES), dtype=np.float32)
    _aux = np.zeros((1, len(RUL_AUX_ORDER)), dtype=np.float32)
    for head in ("detection_model", "diagnosis_model", "severity_model", "failure_modes_model"):
        engine[head].predict(_x, verbose=0)
    engine["rul_model"].predict({"x": _x, "x_rul_aux": _aux}, verbose=0)
print("Models warmed (first-prediction graph build done at startup).")

DEFAULT_ENGINE = "Rotax_914_ULF" if "Rotax_914_ULF" in loaded_engines else next(iter(loaded_engines))
active_engine = DEFAULT_ENGINE
unsupported_engine = None      # set when the selected engine has no v3 export


def make_window(engine_name):
    return RollingWindowV3(loaded_engines[engine_name]["scaler"])


window = make_window(active_engine)


# ============================================================================
# INFERENCE - always .predict(), never model(x): on this architecture they differ.
# ============================================================================
def run_inference(engine=None):
    """All five heads of the active engine on the current window. Only call when
    window.is_ready()."""
    engine = engine or loaded_engines[active_engine]
    x = window.get_window()
    aux = window.get_rul_aux()

    det_prob = float(np.squeeze(engine["detection_model"].predict(x, verbose=0)))
    diag_probs = np.asarray(engine["diagnosis_model"].predict(x, verbose=0))[0]      # (8, 6)
    sev = np.asarray(engine["severity_model"].predict(x, verbose=0))[0]              # (8,)
    fm = np.asarray(engine["failure_modes_model"].predict(x, verbose=0))[0]          # (4,)
    rul_hours = float(np.squeeze(
        engine["rul_model"].predict({"x": x, "x_rul_aux": aux}, verbose=0)))
    # Uncertainty band: the overall test MAE, or the (larger) nearly-new error when the
    # prediction itself is in the nearly-new range.
    rul_band = engine["rul_mae_hours"]
    if engine["rul_near_new_mae_hours"] and rul_hours >= NEAR_NEW_BAND_FRAC * engine["tbo_hours"]:
        rul_band = max(rul_band or 0.0, engine["rul_near_new_mae_hours"])

    unreliable = engine["unreliable_channels"]
    diagnosis, faulty_channels = {}, []
    for i, ch in enumerate(CHANNELS):
        cls = int(np.argmax(diag_probs[i]))
        reliable = ch not in unreliable
        diagnosis[ch] = {"fault_type": FAULT_TYPES[cls], "confidence": float(diag_probs[i, cls]),
                         "reliable": reliable}
        # Same confidence floor the health cap and advisory.py use. Without it the rpm head's
        # constant low-confidence (~0.45) Stuck-At call listed rpm as faulty on every sample.
        if cls != 0 and reliable and diag_probs[i, cls] >= RPM_FAULT_CONFIDENCE_FLOOR:
            faulty_channels.append(ch)
    severity_percent = {ch: float(sev[i] * 100.0) for i, ch in enumerate(CHANNELS)}

    thresholds = engine["failure_mode_thresholds"]
    failure_modes = {}
    for i, mode in enumerate(FAILURE_MODES):
        thr = float(thresholds.get(mode, DETECTION_THRESHOLD))
        failure_modes[mode] = {"severity_percent": float(fm[i] * 100.0),
                               "present": bool(fm[i] >= thr), "threshold": thr}

    # Health reflects both sensor-channel faults and engine failure modes. A failure mode
    # counts only once the head flags it present (probability >= its tuned threshold):
    # sub-threshold probabilities of 3-8% used to be subtracted as damage, so healthy
    # engines drifted to ~92% at cruise with nothing detected.
    graded = [severity_percent[c] / 100.0 for c in CHANNELS if c != "rpm"]
    fm_damage = max((float(fm[i]) for i, mode in enumerate(FAILURE_MODES)
                     if failure_modes[mode]["present"]), default=0.0)
    health_percent = 100.0 * (1.0 - max(max(graded), fm_damage))
    # The 70% cap on a confident rpm fault applies only where that call is measured to be
    # informative. On an unreliable channel it capped a healthy 915 at 70% on every sample.
    if (diagnosis["rpm"]["reliable"] and diagnosis["rpm"]["fault_type"] != "none"
            and diagnosis["rpm"]["confidence"] >= RPM_FAULT_CONFIDENCE_FLOOR):
        health_percent = min(health_percent, 70.0)

    tbo = engine["tbo_hours"]
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "engine_model": active_engine,
        "fault_detected": det_prob >= 0.5,
        "detection_confidence": det_prob,
        "faulty_channels": faulty_channels,
        "diagnosis": diagnosis,
        "severity_percent": severity_percent,
        "failure_modes": failure_modes,
        "health_percent": round(health_percent, 4),
        "rul_hours": round(rul_hours, 3),
        "rul_units": "engine_hours",
        "rul_mae_hours": rul_band,
        "tbo_hours": tbo,
        "rul_percent_remaining": round(max(0.0, min(100.0, 100.0 * rul_hours / tbo)), 4),
        "rul_out_of_range": rul_hours > RUL_OUT_OF_RANGE_FRAC * tbo,
        "steps_collected": len(window.buffer),
        "steps_needed": WINDOW_SIZE,
    }


# ============================================================================
# API - same endpoints and request shapes as ai.py.
# ============================================================================
app = FastAPI(title="AeroSim AI Service (v3)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/step")
def step(payload: dict):
    """One raw telemetry timestep in, one AI result out."""
    if active_engine is None:
        # The selected engine has no v3 export. Serving another engine's model here
        # would label its predictions as this engine's - say so instead.
        return {"status": "ai_unsupported_engine", "model_version": MODEL_VERSION,
                "engine_model": unsupported_engine,
                "message": f"No v3 AI model for {unsupported_engine}. Available: {list(loaded_engines)}"}
    window.update(payload)
    if not window.is_ready():
        return {"status": "warming_up", "model_version": MODEL_VERSION,
                "steps_collected": len(window.buffer), "steps_needed": WINDOW_SIZE}
    return run_inference()


@app.post("/reset")
def reset():
    """Clears the rolling buffer - call when a new flight/scenario starts."""
    global window
    if active_engine is not None:
        window = make_window(active_engine)
    return {"status": "reset"}


@app.get("/health")
def health():
    return {
        "status": "alive",
        "model_version": MODEL_VERSION,
        "n_features": N_FEATURES,
        "active_engine": active_engine,
        "available_engines": list(loaded_engines),
        "tbo_hours": {name: e["tbo_hours"] for name, e in loaded_engines.items()},
        "buffer_fill": len(window.buffer),
        "tf_devices": [d.device_type for d in tf.config.list_physical_devices()],
        "backend_validated": BACKEND_VALIDATED,
    }


@app.post("/select_engine")
def select_engine(payload: dict):
    """Switches engines and resets the buffer - a window must never mix two engines."""
    global active_engine, window, unsupported_engine
    requested = payload.get("engine_model")
    if requested not in loaded_engines:
        # Previously the old engine stayed active, so a 912 flight was scored by the
        # 914 model. Deactivate instead; /step reports ai_unsupported_engine.
        active_engine = None
        unsupported_engine = requested
        return {"status": "error",
                "message": f"No v3 models loaded for {requested!r}. Available: {list(loaded_engines)}"}
    unsupported_engine = None
    active_engine = requested
    window = make_window(active_engine)
    return {"status": "ok", "engine_model": active_engine}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("AERO_AI_PORT", "8100")))
