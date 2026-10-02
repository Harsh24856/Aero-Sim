"""
AI inference service for the physics-v5 twin - Rotax 912 / 914 / 915 / 916.

Serves backend/models_v5/<engine>/ (validation_v5/export_v5.py) through
validation_v5/assembly_v5.Deployed - the SAME assembly code training scored with.
Pairs with main.py running AERO_PHYSICS_VERSION=v5. Binds 8100 like aiv4 - run one:

    cd backend && ../validation/venv/bin/uvicorn aiv5:app --host 127.0.0.1 --port 8100

EACH FLIGHT SECOND (POST /step)
  in:  the 39 twin features (twin_v5.FEATURE_COLS), `ai_context` (45 values: the
       flight's long-horizon residual statistics + hours fraction, usage, life-clock
       flag), engine_hours, life_used_hours, time
  1. append the features to a 128 s window; scale window and context exactly as
     the training cache did (float16 rounding included);
  2. five specialists + the assembly (detection / health stackers, sensor bias,
     severity gate), calibrated diagnosis probabilities and per-fault cut-offs;
  3. RUL from the assembled outputs (main.smooth_rul smooths it in flight);
  out: aiv4's response shape plus v5 fields (families, severity_kind, labels).

DEVICE. v5 builds ReLU as max(z, 0) (model_architectures_v5.relu): the Metal graph
bug that tied v3/v4 to the GPU does not apply, and tests/test_export_parity_v5.py
shows the CPU matching the GPU. Serves on the CPU by default (faster at batch 1);
AERO_AI_DEVICE=gpu for the GPU.

PLACEHOLDERS. As aiv4: an engine without its own export is served by the 914's, and
every response says so.
"""
import json
import os
import sys
from collections import deque

# Before numpy / sklearn load OpenMP: request threads (FastAPI runs /step in a worker
# pool) do not inherit a limit set later on the import thread - measured 78 ms per
# inference there against 14 ms single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, os.path.join(ROOT, "validation_v5"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf                                        # noqa: E402
from threadpoolctl import threadpool_limits                    # noqa: E402
# CPU by default: one window at a time is latency-bound, and measured on the 914 export
# the CPU answers in ~42 ms against ~154 ms on the Metal GPU (kernel launches dominate at
# batch 1). The CPU is validated for v5 (tests/test_export_parity_v5.py). AERO_AI_DEVICE=gpu
# selects the GPU.
if os.environ.get("AERO_AI_DEVICE", "cpu").lower() != "gpu":
    tf.config.set_visible_devices([], "GPU")
# The tree models (stackers, RUL) start an OpenMP pool per call: 116 ms of a single-row
# prediction was thread start-up. One thread: ~2 ms.
threadpool_limits(1)

import features_v5 as F                                        # noqa: E402
import physics_v5 as V                                         # noqa: E402
import rul_v5                                                  # noqa: E402
import twin_v5                                                 # noqa: E402
from assembly_v5 import Deployed, apply_temperature            # noqa: E402
from degradation_v5 import FAULT_NAMES, applicable_faults      # noqa: E402
from model_architectures_v5 import FAMILY_NAMES                # noqa: E402
from observability_v5 import FAMILY_OF                         # noqa: E402
from pipeline_v5 import CTX_COLS                               # noqa: E402
from sensors_v5 import FAULTABLE_CHANNELS, SENSOR_FAULT_TYPES  # noqa: E402

MODEL_VERSION = "v5"
MODELS_ROOT = os.path.join(BACKEND_DIR, "models_v5")
PLACEHOLDER_KEY = "914"
ENGINE_REGISTRY = {"Rotax_912_ULS": "912", "Rotax_914_ULF": "914",
                   "Rotax_915_iS": "915", "Rotax_916_iS": "916"}
DEVICE = "GPU" if tf.config.get_visible_devices("GPU") else "CPU"     # AERO_AI_DEVICE=cpu hides the GPU
MARGIN_CHANNELS = ["egt", "cht", "oil_temp", "oil_pressure", "engine_rpm"]
CONTEXT_SETTLED_S = 3600.0          # the 60-minute EMA needs about this long

if list(twin_v5.FEATURE_COLS) != list(F.FEATURE_COLS):
    raise RuntimeError("twin_v5.FEATURE_COLS differs from features_v5.FEATURE_COLS")


# ============================================================================
# MODELS
# ============================================================================
def load(key: str):
    d = os.path.join(MODELS_ROOT, key)
    if not os.path.exists(os.path.join(d, "manifest.json")):
        return None
    dep = Deployed(d)
    c = dep.contract
    bad = [n for n, got, want in [("feature_cols", c["feature_cols"], list(F.FEATURE_COLS)),
                                  ("ctx_cols", c["ctx_cols"], list(CTX_COLS)),
                                  ("window", c["window"], F.WINDOW)] if got != want]
    if bad:
        raise RuntimeError(f"models_v5/{key}: contract disagrees with this code on {bad} - re-export")
    return dep


def engine_entry(engine_model: str, dep: Deployed, placeholder: bool) -> dict:
    spec = V.ENGINE_SPECS[engine_model]
    applicable = applicable_faults(spec.turbocharged, spec.intercooled)
    t = dep.manifest["test"]["primary"]
    return {"dep": dep, "key": dep.manifest["key"], "engine_model": engine_model, "placeholder": placeholder,
            "tbo_hours": float(spec.tbo_hours), "applicable": applicable,
            "temps": dep.calibration["temperature"], "cuts": dep.calibration["cutoffs"],
            "margin_engine": V.PistonEngineV5(engine_model, dt=1.0),
            # the RUL error measured on test flights, in this engine's hours
            "rul_mae_hours": float(t["rul"]) / 100.0 * float(spec.tbo_hours) if "rul" in t else None}


print(f"Loading v5 models from {MODELS_ROOT} ({DEVICE})...")
_own = {k: load(k) for k in set(ENGINE_REGISTRY.values())}
if _own.get(PLACEHOLDER_KEY) is None:
    raise RuntimeError(f"No v5 export for the {PLACEHOLDER_KEY} under {MODELS_ROOT} - run "
                       "validation_v5/export_v5.py 914b")
loaded_engines = {}
for name, key in ENGINE_REGISTRY.items():
    own = _own.get(key)
    loaded_engines[name] = engine_entry(name, own or _own[PLACEHOLDER_KEY], placeholder=own is None)
    print(f"  {name}: {'own models' if own else 'PLACEHOLDER (914 models)'}, TBO {loaded_engines[name]['tbo_hours']:.0f} h")

# First call builds each network's graph; pay it at start-up, not on the first sample.
_z = loaded_engines["Rotax_914_ULF"]["dep"]
_z.predict_window(np.zeros((F.WINDOW, F.N_FEATURES), np.float32), np.zeros(len(CTX_COLS), np.float32), 0.0)
print("Models warmed.")


# ============================================================================
# FLIGHT STATE
# ============================================================================
class Flight:
    """The rolling 128 s window of raw features and the latest context. RUL is
    returned as the model gives it: main.smooth_rul smooths it in flight, and knows
    to discard take-off windows (a smoother here locked those low guesses in)."""

    def __init__(self):
        self.rows: deque = deque(maxlen=F.WINDOW)
        self.last: dict = {}

    def update(self, raw: dict) -> None:
        self.rows.append([float(raw[c]) for c in F.FEATURE_COLS])
        self.last = raw

    def ready(self) -> bool:
        return len(self.rows) == F.WINDOW


active_engine = "Rotax_914_ULF"
flight = Flight()


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def run_inference(e=None) -> dict:
    e = e or loaded_engines[active_engine]
    dep, last = e["dep"], flight.last
    seq = dep.scale_seq(np.asarray(flight.rows))
    ctx = dep.scale_ctx(np.asarray(last["ai_context"], np.float64))
    hours = float(last["engine_hours"])
    o = dep.predict_window(seq, ctx, hours, tbo=e["tbo_hours"])     # placeholders: this engine's TBO

    prob = apply_temperature(o["diagnosis"][None], e["temps"])[0]
    faults = {}
    for i, name in enumerate(FAULT_NAMES):
        if name not in e["applicable"]:
            continue
        cut = float(e["cuts"][i])
        faults[name] = {"probability": round(float(prob[i]), 4), "present": bool(prob[i] >= cut),
                        "threshold": round(cut, 4), "severity": round(float(o["severity"][i]), 4),
                        "family": FAMILY_OF[name]}
    fam_prob = {f: round(float(p), 4) for f, p in zip(FAMILY_NAMES, o["family"])}
    ps = _softmax(np.asarray(o["sensor"], np.float64))
    sensors = {}
    for i, ch in enumerate(FAULTABLE_CHANNELS):
        k = int(ps[i].argmax())
        sensors[ch] = {"condition": SENSOR_FAULT_TYPES[k], "confidence": round(float(ps[i, k]), 4)}

    det = float(o["detection"][0])
    health = float(np.clip(o["health"][0], 0.0, 1.0))
    margin = e["margin_engine"].margins({c: last[c] for c in MARGIN_CHANNELS})["health_index"]
    tbo, calendar = e["tbo_hours"], float(o["rul_calendar_hours"])
    raw_rul = float(o["rul_hours"])
    rul = min(raw_rul, calendar)
    present = [n for n, f in faults.items() if f["present"]]
    t_flight = float(last.get("time", 0.0))
    return {
        "status": "ok", "model_version": MODEL_VERSION, "labels": dep.manifest.get("labels"),
        "engine_model": e["engine_model"], "placeholder_models": e["placeholder"], "models_engine": e["key"],
        "fault_detected": det >= 0.5, "detection_confidence": round(det, 5),
        "fault_modes": faults, "faults_present": present,
        # Family level: what the UI shows when no single fault clears its cut-off.
        "families": fam_prob,
        "families_present": sorted({FAMILY_OF[n] for n in present} | {f for f, p in fam_prob.items() if p >= 0.5}),
        "severity_kind": "effective",
        "sensors": sensors,
        "faulty_sensors": [c for c, s in sensors.items() if s["condition"] != "none"],
        "wear_condition": round(health, 5),
        "health_percent": round(100.0 * health, 4),
        "margin_min": round(float(margin), 5),
        "rul_hours": round(rul, 3), "rul_hours_raw": round(raw_rul, 3),
        "rul_mae_hours": None if e["rul_mae_hours"] is None else round(e["rul_mae_hours"], 1),
        "tbo_hours": tbo, "rul_calendar_hours": round(calendar, 3),
        "rul_percent_remaining": round(max(0.0, min(100.0, 100.0 * rul / tbo)), 4),
        # The RUL model's own call: it returns the calendar unless it judges the engine
        # wear-limited (validation_v5/rul_v5.GatedRUL).
        "wear_limited": bool(raw_rul < calendar - 1e-3),
        "rul_out_of_range": False,
        "context_settling": t_flight < CONTEXT_SETTLED_S,
        "steps_collected": len(flight.rows), "steps_needed": F.WINDOW,
    }


# ============================================================================
# API - same endpoints and shapes as aiv4.py
# ============================================================================
app = FastAPI(title="AeroSim AI Service (v5)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
REQUIRED = F.FEATURE_COLS + ["ai_context", "engine_hours", "life_used_hours"]


@app.post("/step")
def step(payload: dict):
    missing = [c for c in REQUIRED if c not in payload]
    if missing:
        return {"status": "error", "model_version": MODEL_VERSION, "error": f"missing inputs {missing}"}
    if len(payload["ai_context"]) != len(CTX_COLS):
        return {"status": "error", "model_version": MODEL_VERSION,
                "error": f"ai_context has {len(payload['ai_context'])} values, expected {len(CTX_COLS)}"}
    flight.update(payload)
    if not flight.ready():
        return {"status": "warming_up", "model_version": MODEL_VERSION,
                "steps_collected": len(flight.rows), "steps_needed": F.WINDOW}
    return run_inference()


@app.post("/reset")
def reset():
    global flight
    flight = Flight()
    return {"status": "reset"}


def _metrics(e: dict) -> dict:
    m = e["dep"].manifest
    t = m["test"]["primary"]
    return {"models_engine": e["key"], "placeholder": e["placeholder"], "labels": m.get("labels"),
            "detection_recall_p95": t.get("detection"), "diagnosis_macro_f1": t.get("diagnosis"),
            "severity_mae_on_fault": t.get("severity"), "sensor_macro_f1": t.get("sensor_fault"),
            "health_mae": t.get("health"), "rul_wear_limited_mae_pct_tbo": t.get("rul"),
            "gates": m["test"].get("gates"), "exported_at": m.get("exported")}


@app.get("/health")
def health():
    return {
        "status": "alive", "model_version": MODEL_VERSION, "n_features": F.N_FEATURES,
        "active_engine": active_engine, "available_engines": list(loaded_engines),
        "placeholder_engines": [n for n, x in loaded_engines.items() if x["placeholder"]],
        "tbo_hours": {n: x["tbo_hours"] for n, x in loaded_engines.items()},
        "metrics": {n: _metrics(x) for n, x in loaded_engines.items()},
        "buffer_fill": len(flight.rows), "device": DEVICE,
        "backend_validated": True,
    }


@app.post("/select_engine")
def select_engine(payload: dict):
    global active_engine, flight
    requested = payload.get("engine_model")
    if requested not in loaded_engines:
        return {"status": "error", "message": f"Unknown engine {requested!r}. Options: {list(loaded_engines)}"}
    active_engine = requested
    flight = Flight()
    return {"status": "ok", "engine_model": active_engine,
            "placeholder_models": loaded_engines[active_engine]["placeholder"]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("AERO_AI_PORT", "8100")))
