"""
AI inference service for the physics-v4 digital twin - Rotax 912 / 914 / 915 / 916.

Serves backend/models_v4/<engine>/ (validation_v4/export_deployable_v4.py) and pairs
with main.py running AERO_PHYSICS_VERSION=v4. Binds 8100 like ai.py / aiv3.py - run
one of them at a time:

    cd backend && ../validation/venv/bin/uvicorn aiv4:app --host 127.0.0.1 --port 8100

WHAT IT DOES EACH FLIGHT SECOND
  1. appends the twin's 29 features to a 128-sample window (the training window),
  2. runs detection, diagnosis, severity, sensor-fault and health on it,
  3. builds the RUL head's 10 aux inputs from MEASUREMENTS AND MODEL OUTPUTS ONLY -
     predicted wear condition, predicted severities, margin from measured sensors -
     never from simulator labels (the RUL head was trained on labels there; the
     manifest carries its error measured this live way, Phase 2 / A3),
  4. runs RUL last, because it needs step 3.

CONTRACT. Feature, aux, fault and sensor lists come from validation_v4/tf_data_pipeline
(the module training used - imported, not copied) and are checked at start-up against
every engine's manifest and against twin_v4.py's residual channels. A mismatch refuses
to start rather than serve silently shifted inputs. model_architectures is imported for
the same reason: it registers the custom layers the checkpoints need.

PLACEHOLDERS. An engine with no export of its own is served by the 914's models and
every response says so (`placeholder_models: true`). That keeps the whole stack
runnable while 912/915/916 train; exporting them replaces the placeholder with no
code change.
"""
import json
import os
import sys
from collections import deque

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, os.path.join(ROOT, "validation_v4"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf                                        # noqa: E402
if os.environ.get("AERO_AI_DEVICE", "").lower() == "cpu":
    tf.config.set_visible_devices([], "GPU")
from tensorflow import keras                                   # noqa: E402

import model_architectures  # noqa: E402,F401  registers custom layers for load_model
import tf_data_pipeline as P                                   # noqa: E402
import physics_v4 as V                                         # noqa: E402
from degradation_v4 import applicable_faults                   # noqa: E402
from twin_v4 import FEATURE_COLS as TWIN_FEATURES               # noqa: E402
from twin_v4 import RESIDUAL_CHANNELS as TWIN_RESIDUALS         # noqa: E402

MODEL_VERSION = "v4"
MODELS_ROOT = os.path.join(BACKEND_DIR, "models_v4")
PLACEHOLDER_KEY = "914"
MARGIN_CHANNELS = ["egt", "cht", "oil_temp", "oil_pressure", "engine_rpm"]
ENGINE_REGISTRY = {"Rotax_912_ULS": "912", "Rotax_914_ULF": "914",
                   "Rotax_915_iS": "915", "Rotax_916_iS": "916"}
DEVICE = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

# The pipeline and the twin must agree before anything is served.
if list(P.RESIDUAL_CHANNELS) != list(TWIN_RESIDUALS):
    raise RuntimeError(f"twin_v4 residual channels {TWIN_RESIDUALS} != training {P.RESIDUAL_CHANNELS}")
if list(P.FEATURE_COLS) != list(TWIN_FEATURES):
    raise RuntimeError("twin_v4.FEATURE_COLS differs from the training pipeline's FEATURE_COLS")


# ============================================================================
# MODEL LOADING
# ============================================================================
def _load(path):
    return keras.models.load_model(path, safe_mode=False, compile=False)


def load_models(key: str):
    d = os.path.join(MODELS_ROOT, key)
    mpath = os.path.join(d, "manifest.json")
    if not os.path.exists(mpath):
        return None
    m = json.load(open(mpath))
    c = m["contract"]
    mismatches = [n for n, got, want in [
        ("feature_cols", c["feature_cols"], P.FEATURE_COLS),
        ("rul_aux_order", c["rul_aux_order"], P.RUL_AUX_ORDER),
        ("fault_modes", c["fault_modes"], P.FAULT_MODES),
        ("sensor_channels", c["sensor_channels"], P.SENSOR_CHANNELS),
        ("sensor_fault_types", c["sensor_fault_types"], P.SENSOR_FAULT_TYPES),
        ("residual_channels", c["residual_channels"], P.RESIDUAL_CHANNELS),
        ("window_size", c["window_size"], P.WINDOW_SIZE),
    ] if got != want]
    if mismatches:
        raise RuntimeError(f"models_v4/{key}: manifest disagrees with the training pipeline on "
                           f"{mismatches} - re-export, or change the contract deliberately")
    return {"manifest": m, "key": key,
            "scaler": joblib.load(os.path.join(d, m["scalers"]["features"])),
            "aux_scaler": joblib.load(os.path.join(d, m["scalers"]["rul_aux"])),
            "models": {p: _load(os.path.join(d, h["file"])) for p, h in m["heads"].items()},
            "outputs": {p: h["output"] for p, h in m["heads"].items()}}


def engine_entry(engine_model: str, models: dict, placeholder: bool) -> dict:
    """What serving one engine needs: its models (own or placeholder), and its own
    hardware, TBO and limits - those never come from the placeholder."""
    spec = V.ENGINE_SPECS_V4[engine_model]
    applicable = applicable_faults(spec.turbocharged, spec.intercooled)
    m = models["manifest"]
    return {**models, "engine_model": engine_model, "placeholder": placeholder,
            "tbo_hours": float(spec.tbo_hours), "applicable": applicable,
            "mask": np.array([f in applicable for f in P.FAULT_MODES], dtype=np.float32),
            "thresholds": m.get("fault_thresholds") or {},
            "require_severity": bool(m.get("require_severity", False)),
            "bands": m.get("rul_band_hours") or [],
            "margin_engine": V.PistonEngineV4(engine_model, dt=1.0),
            "backend_validated": m.get("metrics_device", "CPU") == DEVICE}


print(f"Loading v4 models from {MODELS_ROOT} ({DEVICE})...")
_own = {k: load_models(k) for k in set(ENGINE_REGISTRY.values())}
if _own.get(PLACEHOLDER_KEY) is None:
    raise RuntimeError(f"No v4 export for the {PLACEHOLDER_KEY} under {MODELS_ROOT} - run "
                       "validation_v4/export_deployable_v4.py --engine 914")
loaded_engines = {}
for name, key in ENGINE_REGISTRY.items():
    own = _own.get(key)
    loaded_engines[name] = engine_entry(name, own or _own[PLACEHOLDER_KEY], placeholder=own is None)
    print(f"  {name}: {'own models' if own else 'PLACEHOLDER (914 models)'}, TBO {loaded_engines[name]['tbo_hours']:.0f} h")

# First predict builds each graph (seconds); pay it at start-up, not on the first sample.
_x = np.zeros((1, P.WINDOW_SIZE, P.N_FEATURES), np.float32)
_a = np.zeros((1, P.N_RUL_AUX), np.float32)
for _m in {id(e["models"]["detection"]): e for e in loaded_engines.values()}.values():
    for _model in _m["models"].values():
        _model.predict_on_batch({"x": _x, "x_rul_aux": _a})
print("Models warmed.")


# ============================================================================
# ROLLING WINDOW
# ============================================================================
class RollingWindowV4:
    """128 one-second samples of the 29 features, scaled as training scaled them
    (float64 through the scaler, then float32), plus the latest sample's
    measurement-side RUL inputs."""

    def __init__(self, scaler):
        self.scaler = scaler
        self.buffer: deque = deque(maxlen=P.WINDOW_SIZE)
        self.last: dict = {}

    def update(self, raw: dict):
        self.buffer.append([float(raw[c]) for c in P.FEATURE_COLS])
        self.last = raw

    def is_ready(self) -> bool:
        return len(self.buffer) == P.WINDOW_SIZE

    def get_window(self) -> np.ndarray:
        return self.scaler.transform(np.asarray(self.buffer, np.float64)).astype(np.float32)[np.newaxis]


def rul_aux(last: dict, tbo: float, health: float, severity: np.ndarray, margin: float) -> np.ndarray:
    """tf_data_pipeline.compute_rul_aux for one row, with the three label-derived
    entries replaced by model outputs and the measured margin (A3)."""
    f32 = np.float32
    res = np.abs(np.array([last[c] for c in P.RESIDUAL_COLS], f32))
    vals = {"engine_hours_norm": f32(last["engine_hours"]) / tbo,
            "life_used_norm": f32(last["life_used_hours"]) / tbo,
            "health_index": f32(health), "margin_min": f32(margin),
            "mean_fault_severity": severity.mean(), "max_fault_severity": severity.max(),
            "mean_residual_abs": res.mean(), "max_residual_abs": res.max(),
            "throttle_mean": f32(last["throttle"]), "altitude_mean": f32(last["altitude"]) / 9000.0}
    return np.array([[vals[k] for k in P.RUL_AUX_ORDER]], np.float32)


# ============================================================================
# INFERENCE
# ============================================================================
active_engine = "Rotax_914_ULF"
window = RollingWindowV4(loaded_engines[active_engine]["scaler"])


def _head(e, phase, inputs):
    return np.asarray(e["models"][phase].predict_on_batch(inputs)[e["outputs"][phase]])


def run_inference(e=None) -> dict:
    e = e or loaded_engines[active_engine]
    last = window.last
    inputs = {"x": window.get_window(), "x_rul_aux": _a}
    det = float(_head(e, "detection", inputs).ravel()[0])
    prob = _head(e, "diagnosis", inputs)[0] * e["mask"]
    sev = np.clip(_head(e, "severity", inputs)[0], 0.0, 1.0) * e["mask"]
    sensor = _head(e, "sensor_fault", inputs)[0]
    health = float(np.clip(_head(e, "health", inputs).ravel()[0], 0.0, 1.0))
    margin = e["margin_engine"].margins({c: last[c] for c in MARGIN_CHANNELS})["health_index"]
    aux = e["aux_scaler"].transform(rul_aux(last, e["tbo_hours"], health, sev.astype(np.float32), margin)
                                    .astype(np.float64)).astype(np.float32)
    rul = float(_head(e, "rul", {"x": inputs["x"], "x_rul_aux": aux}).ravel()[0])

    tbo = e["tbo_hours"]
    hours = float(last["engine_hours"])
    calendar = max(0.0, tbo - hours)
    rul = min(max(rul, 0.0), calendar)                  # never more life than the calendar allows
    life = hours / tbo
    band = next((b["mae_hours"] for b in e["bands"] if b["life_from"] <= life < b["life_to"]),
                e["bands"][-1]["mae_hours"] if e["bands"] else None)

    faults = {}
    for i, name in enumerate(P.FAULT_MODES):
        if name not in e["applicable"]:
            continue
        thr = float(e["thresholds"].get(name, 0.5))
        # The manifest says whether a call also needs the severity head to agree
        # (>= the labels' own 0.08 definition of present); chosen on validation.
        present = prob[i] >= thr and (not e["require_severity"] or sev[i] >= P.DETECTION_THRESHOLD)
        faults[name] = {"probability": round(float(prob[i]), 4), "present": bool(present),
                        "threshold": thr, "severity": round(float(sev[i]), 4)}
    sensors = {}
    for i, ch in enumerate(P.SENSOR_CHANNELS):
        k = int(sensor[i].argmax())
        sensors[ch] = {"condition": P.SENSOR_FAULT_TYPES[k], "confidence": round(float(sensor[i, k]), 4)}
    return {
        "status": "ok", "model_version": MODEL_VERSION, "engine_model": e["engine_model"],
        "placeholder_models": e["placeholder"], "models_engine": e["key"],
        "fault_detected": det >= 0.5, "detection_confidence": round(det, 5),
        "fault_modes": faults,
        "faults_present": [n for n, f in faults.items() if f["present"]],
        "sensors": sensors,
        "faulty_sensors": [c for c, s in sensors.items() if s["condition"] != "none"],
        "wear_condition": round(health, 5),
        "health_percent": round(100.0 * health, 4),      # v3 name, v4 meaning: wear condition x 100
        "margin_min": round(float(margin), 5),
        "rul_hours": round(rul, 3), "rul_mae_hours": None if band is None else round(float(band), 1),
        "tbo_hours": tbo, "rul_calendar_hours": round(calendar, 3),
        "rul_percent_remaining": round(max(0.0, min(100.0, 100.0 * rul / tbo)), 4),
        # Wear-limited: the model puts end of life clearly before the overhaul date -
        # further below the calendar than its own measured error at this life stage.
        "wear_limited": bool(calendar - rul > (band or 0.0)),
        "rul_out_of_range": False,
        "steps_collected": len(window.buffer), "steps_needed": P.WINDOW_SIZE,
    }


# ============================================================================
# API - same endpoints and shapes as aiv3.py.
# ============================================================================
app = FastAPI(title="AeroSim AI Service (v4)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/step")
def step(payload: dict):
    """One 1 Hz twin sample in (the 29 features plus engine_hours and
    life_used_hours), one AI result out."""
    missing = [c for c in P.FEATURE_COLS + ["engine_hours", "life_used_hours"] if c not in payload]
    if missing:
        return {"status": "error", "model_version": MODEL_VERSION, "error": f"missing inputs {missing}"}
    window.update(payload)
    if not window.is_ready():
        return {"status": "warming_up", "model_version": MODEL_VERSION,
                "steps_collected": len(window.buffer), "steps_needed": P.WINDOW_SIZE}
    return run_inference()


@app.post("/reset")
def reset():
    global window
    window = RollingWindowV4(loaded_engines[active_engine]["scaler"])
    return {"status": "reset"}


def _metrics(e: dict) -> dict:
    m = e["manifest"]
    t = m.get("test") or {}
    live = m.get("rul_live_inputs") or {}
    return {"models_engine": e["key"], "placeholder": e["placeholder"],
            "detection_auc": (t.get("detection") or {}).get("auc"),
            "diagnosis_macro_f1": (m.get("diagnosis_cutoffs") or {}).get("tuned_macro_f1"),
            "severity_mae_on_fault": (t.get("severity") or {}).get("mae_on_fault"),
            "sensor_macro_f1": (t.get("sensor_fault") or {}).get("macro_f1"),
            "health_mae": (t.get("health") or {}).get("mae"),
            "rul_mae_pct_tbo": live.get("mae_pct_tbo"),
            "rul_wear_limited_mae_pct_tbo": live.get("mae_pct_tbo_wear_limited"),
            "exported_at": m.get("exported_at")}


@app.get("/health")
def health():
    e = loaded_engines[active_engine]
    return {
        "status": "alive", "model_version": MODEL_VERSION, "n_features": P.N_FEATURES,
        "active_engine": active_engine, "available_engines": list(loaded_engines),
        "placeholder_engines": [n for n, x in loaded_engines.items() if x["placeholder"]],
        "tbo_hours": {n: x["tbo_hours"] for n, x in loaded_engines.items()},
        "metrics": {n: _metrics(x) for n, x in loaded_engines.items()},
        "buffer_fill": len(window.buffer), "device": DEVICE,
        "backend_validated": e["backend_validated"],
    }


@app.post("/select_engine")
def select_engine(payload: dict):
    global active_engine, window
    requested = payload.get("engine_model")
    if requested not in loaded_engines:
        return {"status": "error", "message": f"Unknown engine {requested!r}. Options: {list(loaded_engines)}"}
    active_engine = requested
    window = RollingWindowV4(loaded_engines[active_engine]["scaler"])
    return {"status": "ok", "engine_model": active_engine,
            "placeholder_models": loaded_engines[active_engine]["placeholder"]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("AERO_AI_PORT", "8100")))
