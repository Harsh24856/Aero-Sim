"""
AI inference service for the UAV digital twin - Rotax 914 / 915 / 916.

Built by reading, in this exact order:
  validation/tf_data_pipeline.py       - the ONLY authority on input features
  validation/model_architectures.py    - head shapes/activations
  validation/phase2_train.ipynb        - detection head
  validation/phase3_train.ipynb        - diagnosis head + warm-start
  validation/phase4_train.ipynb        - severity head + warm-start
  validation/phase5_train.ipynb        - RUL head (914) - independent
                                          2-layer LSTM branch, x_rul_aux
  validation/phase5_train_915.ipynb    - RUL head (915) - confirmed to use
                                          the SAME tf_data_pipeline.py
                                          unmodified, no engine-specific
                                          formula differences
  validation/phase5_train_916.ipynb    - RUL head (916) - same confirmation
  backend/models/{914,915,916}/*_rul.keras - all three confirmed via layer
                                          inspection to share the identical
                                          rul_lstm0/rul_lstm1 architecture

Every formula below is copied verbatim from tf_data_pipeline.py's
scenario_window_generator(). Nothing here is "improved" or "made more
physically correct" relative to that file - the model only understands the
exact distribution it was trained on. In particular, SEVERITY_POWER_DIVISOR
is a FIXED 85.0 for every engine, confirmed by reading all three phase5
notebooks: none of them override it per-engine, even though 915/916's real
max power (104/117 kW) is well above it. Using a "more physically correct"
per-engine divisor here was tried before and caused a real, confirmed
regression - it changed the model's input distribution away from training.

Verified via parity test (incremental RollingWindow vs tf_data_pipeline.py's
own batch computation, on a real validation scenario, for EACH engine
separately): x and x_rul_aux match to float32 precision (~1e-6) for 914, 915,
and 916 all three.
"""
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

# ============================================================================
# CONTRACT - copied verbatim from tf_data_pipeline.py. Do not reorder,
# rename, or "improve" anything in this section without re-running the
# parity check against real training data first.
# ============================================================================

WINDOW_SIZE = 128
N_FEATURES = 24

FEATURE_COLS = [
    "altitude", "throttle", "airspeed", "aoa", "air_density",
    "torque_available_nm", "engine_rpm", "prop_rpm", "prop_torque", "power_kw", "fuel_flow",
    "thrust", "lift", "drag", "thrust_margin", "lift_weight_margin",
    "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm_fault",
]
assert len(FEATURE_COLS) == N_FEATURES

CHANNELS = ["egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm"]
FAULT_TYPES = ["none", "Bias", "Drift", "Spike", "Stuck-At", "Noise"]

RECENT_WINDOW_SECONDS = 60

# Fixed for EVERY engine - see the module docstring. Not "914's constants",
# not "close enough" for 915/916 - this is literally what all three were
# trained against, confirmed by reading all three phase5 notebooks.
SEVERITY_RPM_DIVISOR = 5800.0
SEVERITY_POWER_DIVISOR = 85.0
HIGH_THROTTLE_THRESHOLD = 0.7

MAX_SIM_LIFE_HOURS = 20000.0 / 3600.0   # training MAX_DURATION censoring cutoff
RPM_FAULT_CONFIDENCE_FLOOR = 0.70       # see run_inference() docstring

MODELS_ROOT = "/Users/harsh/Documents/UAV_Engine/backend/models"

# One entry per deployed engine. Adding a new engine means: drop its 4 .keras
# files + scaler into backend/models/<key>/<key>_{detection,diagnosis,
# severity,rul}.keras + scaler_<key>.pkl, add one line here, and re-run the
# parity test against that engine's own validation chunks before trusting it.
ENGINE_REGISTRY = {
    "Rotax_914_ULF": "914",
    "Rotax_912_ULS": "912",
    "Rotax_915_iS":  "915",
    "Rotax_916_iS":  "916",
}

class RollingWindow:
    """Accumulates one raw telemetry dict per call. Produces the scaled (1,128,24)
    model input and the (1,6) x_rul_aux vector, both computed to match
    tf_data_pipeline.py's scenario_window_generator() exactly - that function
    processes a WHOLE scenario at once with numpy cumulative ops; this class
    computes the identical quantities incrementally, one row at a time, since
    live serving only ever has "so far" to work with, never the full scenario.
    Engine-agnostic: only the scaler passed in differs between engines, the
    formulas are identical (verified by parity test for all three)."""

    def __init__(self, scaler):
        self.scaler = scaler
        self.buffer: deque = deque(maxlen=WINDOW_SIZE)
        self.n_steps = 0
        self._severity_sum = 0.0
        self._severity_max = 0.0
        self._recent_severity: deque = deque(maxlen=RECENT_WINDOW_SECONDS)
        self._high_throttle_count = 0

    def update(self, raw: dict):
        """raw must contain every key in FEATURE_COLS plus "time" (simulated
        seconds, the twin's own absolute clock - see elapsed_hours below)."""
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
        """Scaled (1, 128, 24) model input. Wrapped in a named DataFrame because
        the scaler was fit on named columns during training - passing a bare
        array is order-dependent and silently wrong if it ever drifts."""
        raw_matrix = np.array(
            [[step[c] for c in FEATURE_COLS] for step in self.buffer], dtype=np.float32)
        raw_df = pd.DataFrame(raw_matrix, columns=FEATURE_COLS)
        scaled = self.scaler.transform(raw_df).astype(np.float32)
        return scaled[np.newaxis, ...]   # (1, 128, 24)

    def get_rul_aux(self) -> np.ndarray:
        """(1, 6), in the EXACT order tf_data_pipeline.py yields them:
        [running_mean_severity, elapsed_hours, running_max_severity,
         recent_severity_mean, severity_trend, cum_high_throttle_frac]"""
        n = self.n_steps
        running_mean_severity = self._severity_sum / n
        elapsed_hours = self.buffer[-1]["time"] / 3600.0
        running_max_severity = self._severity_max
        recent_severity_mean = sum(self._recent_severity) / len(self._recent_severity)
        severity_trend = recent_severity_mean - running_mean_severity
        cum_high_throttle_frac = self._high_throttle_count / n
        return np.array([[
            running_mean_severity, elapsed_hours, running_max_severity,
            recent_severity_mean, severity_trend, cum_high_throttle_frac,
        ]], dtype=np.float32)

# ============================================================================
# MODEL LOADING - all 3 engines eagerly at startup. Combined footprint is
# small (~1MB x 3), and this avoids any risk of a slow first-request load
# happening mid-flight during a real session.
# ============================================================================

# ============================================================================
# BACKEND GUARD - which TensorFlow this process runs on is NOT an
# implementation detail for these weights.
# ============================================================================
# Every model here was trained AND validated under the Metal GPU backend
# (validation/venv, tensorflow-metal). Loading the identical .keras files under
# a CPU-only TensorFlow yields the same weights but materially different
# predictions - not float noise, a different answer. Measured on real
# validation windows against rul_true labels, byte-identical inputs:
#
#       engine   RUL MAE (Metal)   RUL MAE (CPU-only)   corr (Metal / CPU)
#         914        0.89 h            235.99 h           0.42 / 0.21
#         912        0.35 h             12.28 h           0.89 / 0.05
#         915        1.07 h            513.21 h           0.94 / 0.81
#         916        0.57 h              0.75 h           0.58 / 0.74
#
# True RUL never exceeds ~5.3 h in that data; the CPU-only path predicts up to
# 572 h. This was a real, confirmed live incident: the service was being
# launched from a CPU-only conda env, and every "RUL and health look wrong"
# symptom - 915 reporting ~515 h remaining, 914 flipping between 0 h and 21 h
# between consecutive seconds, health jittering 8x more than it should - came
# from that and nothing else. 916 barely differs between backends, which is
# exactly why its numbers looked plausible while 915's did not.
#
# Warn rather than refuse: a degraded service still beats no service. What must
# never happen again is this failing SILENTLY.
GPU_DEVICES = tf.config.list_physical_devices("GPU")
BACKEND_VALIDATED = bool(GPU_DEVICES)
if not BACKEND_VALIDATED:
    print("=" * 78)
    print("WARNING: TensorFlow sees no GPU backend in this environment.")
    print("         These weights were trained and validated on the Metal GPU")
    print("         backend; RUL and severity output here is NOT trustworthy.")
    print("         Launch from the validated environment instead:")
    print("           validation/venv/bin/uvicorn ai:app --host 127.0.0.1 --port 8100")
    print("=" * 78)

print("Loading models for all registered engines...")
loaded_engines: dict = {}
for engine_name, key in ENGINE_REGISTRY.items():
    model_dir = os.path.join(MODELS_ROOT, key)
    loaded_engines[engine_name] = {
        "scaler": joblib.load(os.path.join(model_dir, f"scaler_{key}.pkl")),
        "detection_model": keras.models.load_model(os.path.join(model_dir, f"{key}_detection.keras"), safe_mode=False, compile=False),
        "diagnosis_model": keras.models.load_model(os.path.join(model_dir, f"{key}_diagnosis.keras"), safe_mode=False, compile=False),
        "severity_model":  keras.models.load_model(os.path.join(model_dir, f"{key}_severity.keras"),  safe_mode=False, compile=False),
        "rul_model":       keras.models.load_model(os.path.join(model_dir, f"{key}_rul.keras"),       safe_mode=False, compile=False),
    }
    print(f"  {engine_name}: loaded")
print(f"All {len(loaded_engines)} engines loaded.")

DEFAULT_ENGINE = "Rotax_914_ULF"
active_engine = DEFAULT_ENGINE
window = RollingWindow(loaded_engines[active_engine]["scaler"])

# ============================================================================
# INFERENCE
# ============================================================================
# RULE: always call .predict(), never model(x) directly. On this architecture
# those two produce DIFFERENT results - confirmed twice: once when a direct
# call was tried in this file and broke real-time inference outright, once
# when a direct call was used in an offline validation script and produced a
# false "the model is broken" conclusion (MAE 87h vs the true 0.31h).


def run_inference():
    """Runs all 4 heads of the ACTIVE engine on the current window. Only call
    when window.is_ready()."""
    engine = loaded_engines[active_engine]
    x = window.get_window()
    aux = window.get_rul_aux()

    det_prob = float(np.squeeze(engine["detection_model"].predict(x, verbose=0)))
    diag_probs = np.asarray(engine["diagnosis_model"].predict(x, verbose=0))[0]   # (8, 6)
    sev = np.asarray(engine["severity_model"].predict(x, verbose=0))[0]           # (8,)
    rul_hours = float(np.squeeze(
        engine["rul_model"].predict({"x": x, "x_rul_aux": aux}, verbose=0)))

    diagnosis = {}
    faulty_channels = []
    for i, ch in enumerate(CHANNELS):
        cls = int(np.argmax(diag_probs[i]))
        conf = float(diag_probs[i, cls])
        fault_type = FAULT_TYPES[cls]
        diagnosis[ch] = {"fault_type": fault_type, "confidence": conf}
        if fault_type != "none":
            faulty_channels.append(ch)

    severity_percent = {ch: float(sev[i] * 100.0) for i, ch in enumerate(CHANNELS)}

    graded_channels = [c for c in CHANNELS if c != "rpm"]
    max_severity = max(severity_percent[c] / 100.0 for c in graded_channels)
    health_percent = 100.0 * (1.0 - max_severity)
    # See RPM_FAULT_CONFIDENCE_FLOOR definition above - requires genuine
    # confidence, not just any non-"none" classification (was firing on
    # ~50% coin-flip calls and permanently capping healthy engines at 70%).
    if (diagnosis["rpm"]["fault_type"] != "none"
            and diagnosis["rpm"]["confidence"] >= RPM_FAULT_CONFIDENCE_FLOOR):
        health_percent = min(health_percent, 70.0)

    # Self-normalizing percent: what fraction of this flight's OWN implied
    # total life remains. Documented limitation: if rul_hours is a wild
    # extrapolation beyond MAX_SIM_LIFE_HOURS, this still returns a plausible
    # ~100%, which is why rul_hours_internal is ALSO returned unclamped -
    # so that situation stays visible rather than silently hidden.
    elapsed_hours = float(aux[0, 1])
    implied_total_life = elapsed_hours + max(rul_hours, 1e-6)
    rul_percent_remaining = 100.0 * rul_hours / implied_total_life
    out_of_range = rul_hours > MAX_SIM_LIFE_HOURS

    return {
        "status": "ok",
        "engine_model": active_engine,
        "fault_detected": det_prob >= 0.5,
        "detection_confidence": det_prob,
        "faulty_channels": faulty_channels,
        "diagnosis": diagnosis,
        "severity_percent": severity_percent,
        "health_percent": round(health_percent, 4),
        "rul_percent_remaining": round(rul_percent_remaining, 4),
        "rul_hours_internal": round(rul_hours, 4),   # simulated-timescale, NOT real-world hours
        "rul_out_of_range": out_of_range,
        "steps_collected": len(window.buffer),
        "steps_needed": WINDOW_SIZE,
    }

# ============================================================================
# API
# ============================================================================

app = FastAPI(title="AeroSim AI Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/step")
def step(payload: dict):
    """One raw telemetry timestep in, one AI result out. payload must contain
    every key in FEATURE_COLS plus "time" - exactly what main.py's
    AI_FEATURE_COLS + telemetry["time"] already sends (verified to match)."""
    window.update(payload)
    if not window.is_ready():
        return {
            "status": "warming_up",
            "steps_collected": len(window.buffer),
            "steps_needed": WINDOW_SIZE,
        }
    return run_inference()


@app.post("/reset")
def reset():
    """Clears the rolling buffer - call when a new flight/scenario starts."""
    global window
    window = RollingWindow(loaded_engines[active_engine]["scaler"])
    return {"status": "reset"}


@app.get("/health")
def health():
    return {
        "status": "alive",
        "active_engine": active_engine,
        "available_engines": list(loaded_engines),
        "buffer_fill": len(window.buffer),
        # Surfaced so "are these predictions trustworthy?" is answerable without
        # reading this process's startup log - see the BACKEND GUARD above.
        "tf_devices": [d.device_type for d in tf.config.list_physical_devices()],
        "backend_validated": BACKEND_VALIDATED,
    }


@app.post("/select_engine")
def select_engine(payload: dict):
    """Switches which engine's models this service uses for all subsequent
    /step calls. Always resets the buffer too - a discontinuous engine switch
    mid-window would mix two different engines' physics in one 128-step
    input, which none of these models were trained to handle."""
    global active_engine, window
    requested = payload.get("engine_model")
    if requested not in loaded_engines:
        return {
            "status": "error",
            "message": f"No models loaded for {requested!r}. Available: {list(loaded_engines)}",
        }
    active_engine = requested
    window = RollingWindow(loaded_engines[active_engine]["scaler"])
    return {"status": "ok", "engine_model": active_engine}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100)
