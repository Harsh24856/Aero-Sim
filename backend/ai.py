"""
ai.py - AI inference microservice for the UAV Digital Twin.

Runs under the validation venv (Python 3.11 + TensorFlow 2.16.2) because that is the
exact environment the Phase 8 models were saved under. The backend's own venv
(Python 3.13.9) has no compatible TensorFlow wheel and SEGFAULTS on load_model() for
these specific checkpoints (confirmed: a Keras 3.15.1/TF 2.21.0 vs the save environment's
Keras/TF 2.16.2 mismatch crashes on the custom Lambda layer during deserialization).
Rather than fight that, this runs as its own small local HTTP service - main.py (under
its own venv) calls it over plain HTTP, a standard microservice split for exactly this
kind of cross-environment situation.

Flow: frontend -> main.py (physics engine, produces one raw feature dict per timestep)
      -> POST http://localhost:8100/step with that dict
      -> ai.py maintains a rolling 128-step buffer + running aux-feature state,
         runs all 4 models via a LangGraph graph once the buffer is full,
         returns the combined AI verdict
      -> main.py forwards that verdict to the frontend alongside the telemetry

Run with:
    /Users/harsh/Documents/UAV_Engine/validation/venv/bin/python3 ai.py
"""
import os
import time
from collections import deque
from typing import TypedDict, Optional, Any

import numpy as np
import joblib
from tensorflow import keras
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# Config - paths, feature ordering. Must match tf_data_pipeline.py exactly,
# since that is the contract every model was trained against.
# ---------------------------------------------------------------------------
MODEL_DIR = "/Users/harsh/Documents/UAV_Engine/backend/models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURE_COLS = [
    "altitude", "throttle", "airspeed", "aoa", "air_density",
    "torque_available_nm", "engine_rpm", "prop_rpm", "prop_torque", "power_kw", "fuel_flow",
    "thrust", "lift", "drag", "thrust_margin", "lift_weight_margin",
    "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm_fault",
]
CHANNELS = ["egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm"]
FAULT_TYPES = ["none", "Bias", "Drift", "Spike", "Stuck-At", "Noise"]
WINDOW_SIZE = 128
RECENT_WINDOW_SECONDS = 60   # must match tf_data_pipeline.py's RECENT_WINDOW
# ---------------------------------------------------------------------------
# Load models + scaler ONCE at startup, not per-request.
# ---------------------------------------------------------------------------
print("Loading scaler and models...")
scaler = joblib.load(SCALER_PATH)
detection_model = keras.models.load_model(os.path.join(MODEL_DIR, "phase8_detection.keras"), safe_mode=False)
diagnosis_model = keras.models.load_model(os.path.join(MODEL_DIR, "phase8_diagnosis.keras"), safe_mode=False)
severity_model = keras.models.load_model(os.path.join(MODEL_DIR, "phase8_severity.keras"), safe_mode=False)
rul_model = keras.models.load_model(os.path.join(MODEL_DIR, "phase8_rul.keras"), safe_mode=False)
print("All 4 models + scaler loaded.")


# ---------------------------------------------------------------------------
# Rolling engine state - maintains the 128-step window AND the running
# aux-feature statistics EXACTLY as tf_data_pipeline.py computed them during
# training (cumulative mean/max severity, recent-60s mean, trend, elapsed
# hours, cumulative high-throttle fraction). All derived from raw observable
# signals only (engine_rpm, power_kw, throttle, time) - zero leakage of any
# ground-truth label, exactly matching the training-time definition.
# ---------------------------------------------------------------------------
class EngineState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = deque(maxlen=WINDOW_SIZE)          # raw feature dicts, oldest first
        self.severity_proxy_history = deque(maxlen=RECENT_WINDOW_SECONDS)
        self.cum_severity_sum = 0.0
        self.cum_severity_max = 0.0
        self.cum_high_throttle_count = 0
        self.n_steps = 0
        self.start_time = None

    def update(self, raw: dict):
        """Feed one new raw timestep (dict with at least FEATURE_COLS + time)."""
        self.buffer.append(raw)
        self.n_steps += 1
        if self.start_time is None:
            self.start_time = raw["time"]

        rpm_frac = raw["engine_rpm"] / 5800.0
        power_frac = raw["power_kw"] / 85.0
        severity_proxy = 0.5 * rpm_frac**2 + 0.5 * power_frac**2

        self.cum_severity_sum += severity_proxy
        self.cum_severity_max = max(self.cum_severity_max, severity_proxy)
        self.severity_proxy_history.append(severity_proxy)
        if raw["throttle"] > 0.7:
            self.cum_high_throttle_count += 1

    def is_ready(self) -> bool:
        return len(self.buffer) == WINDOW_SIZE

    def get_window(self) -> np.ndarray:
        """Returns the SCALED (128, 24) window ready for model input."""
        raw_matrix = np.array([[step[c] for c in FEATURE_COLS] for step in self.buffer], dtype=np.float32)
        return scaler.transform(raw_matrix).astype(np.float32)

    def get_rul_aux(self) -> np.ndarray:
        """The 6 auxiliary features, computed identically to training."""
        n = self.n_steps
        running_mean_severity = self.cum_severity_sum / n
        elapsed_hours = (self.buffer[-1]["time"] - self.start_time) / 3600.0
        running_max_severity = self.cum_severity_max
        recent_severity_mean = sum(self.severity_proxy_history) / len(self.severity_proxy_history)
        severity_trend = recent_severity_mean - running_mean_severity
        cum_high_throttle_frac = self.cum_high_throttle_count / n
        return np.array([[running_mean_severity, elapsed_hours, running_max_severity,
                           recent_severity_mean, severity_trend, cum_high_throttle_frac]], dtype=np.float32)

engine_state = EngineState()
# ---------------------------------------------------------------------------
# LangGraph orchestration - one node per model. Detection/Diagnosis/Severity/
# RUL have no dependency on each other (each model is independent, per Phase 5's
# design), so all 4 fan out from START and fan into a single combine node.
# ---------------------------------------------------------------------------
class InferenceState(TypedDict):
    window: Any        # (1, 128, 24) scaled input, shared read-only
    rul_aux: Any        # (1, 6) input, shared read-only
    detection_prob: Optional[float]
    diagnosis_result: Optional[dict]
    severity_result: Optional[dict]
    rul_hours: Optional[float]
    final_output: Optional[dict]


def detect_node(state: InferenceState) -> dict:
    prob = float(np.squeeze(detection_model.predict(state["window"], verbose=0)))
    return {"detection_prob": prob}


def diagnose_node(state: InferenceState) -> dict:
    pred = np.asarray(diagnosis_model.predict(state["window"], verbose=0))[0]   # (8, 6)
    classes = np.argmax(pred, axis=-1)
    confidences = np.max(pred, axis=-1)
    result = {
        CHANNELS[i]: {"fault_type": FAULT_TYPES[int(classes[i])], "confidence": float(confidences[i])}
        for i in range(8)
    }
    return {"diagnosis_result": result}


def severity_node(state: InferenceState) -> dict:
    pred = np.asarray(severity_model.predict(state["window"], verbose=0))[0]   # (8,)
    result = {CHANNELS[i]: float(pred[i]) for i in range(8)}
    return {"severity_result": result}


def rul_node(state: InferenceState) -> dict:
    pred = rul_model.predict({"x": state["window"], "x_rul_aux": state["rul_aux"]}, verbose=0)
    return {"rul_hours": float(np.squeeze(pred))}


def combine_node(state: InferenceState) -> dict:
    severity_result = state["severity_result"]
    diagnosis_result = state["diagnosis_result"]
    detection_prob = state["detection_prob"]
    rul_hours = state["rul_hours"]

    fault_detected = detection_prob > 0.5
    faulty_channels = [c for c in CHANNELS if diagnosis_result[c]["fault_type"] != "none"]

    # Health%: based on the WORST current symptom severity, not lifetime. RPM has no
    # graded severity (its fault is a discrete glitch - documented physics-generator
    # limitation), so it cannot contribute to the severity max; instead a confirmed RPM
    # fault caps health at 70% as a coarse penalty rather than being silently ignored.
    graded_channels = [c for c in CHANNELS if c != "rpm"]
    max_severity = max(severity_result[c] for c in graded_channels)
    health_percent = 100.0 * (1.0 - max_severity)
    if diagnosis_result["rpm"]["fault_type"] != "none":
        health_percent = min(health_percent, 70.0)

    # RUL as %-of-implied-lifetime-remaining, NOT raw hours - sidesteps ever having to
    # claim the simulated RUL scale (~0-5.5h) represents real-world engine TBO (~1000-2000h).
    # elapsed_hours already tracked in EngineState via get_rul_aux(); recomputed here from
    # the same source for clarity.
    elapsed_hours = (engine_state.buffer[-1]["time"] - engine_state.start_time) / 3600.0
    implied_total_life = elapsed_hours + max(rul_hours, 1e-6)
    rul_percent_remaining = 100.0 * rul_hours / implied_total_life

    output = {
        "fault_detected": bool(fault_detected),
        "detection_confidence": round(float(detection_prob if fault_detected else 1 - detection_prob), 4),
        "faulty_channels": faulty_channels,
        "diagnosis": diagnosis_result,
        "severity_percent": {c: round(v * 100, 1) for c, v in severity_result.items()},
        "health_percent": round(health_percent, 1),
        "rul_hours_internal": round(rul_hours, 4),   # simulated-timescale value - NOT real-world hours, internal use only
        "rul_percent_remaining": round(rul_percent_remaining, 1),
        "timestamp": time.time(),
    }
    return {"final_output": output}


graph = StateGraph(InferenceState)
graph.add_node("detect", detect_node)
graph.add_node("diagnose", diagnose_node)
graph.add_node("severity", severity_node)
graph.add_node("rul", rul_node)
graph.add_node("combine", combine_node)

# True 4-way parallel fan-out from START (not chained through "detect") - all 4 models
# are independent (per Phase 5's design: RUL trained on a separate frozen-encoder branch,
# Detection/Diagnosis/Severity share a frozen encoder but do not depend on each other's
# output), so there is no reason for one to block another.
graph.add_edge(START, "detect")
graph.add_edge(START, "diagnose")
graph.add_edge(START, "severity")
graph.add_edge(START, "rul")
graph.add_edge("detect", "combine")
graph.add_edge("diagnose", "combine")
graph.add_edge("severity", "combine")
graph.add_edge("rul", "combine")
graph.add_edge("combine", END)

inference_graph = graph.compile()
print("LangGraph inference graph compiled.")
# ---------------------------------------------------------------------------
# FastAPI service - main.py calls this over local HTTP for every physics
# timestep. Kept as its own process specifically to avoid the TF version
# segfault described at the top of this file.
# ---------------------------------------------------------------------------
app = FastAPI(title="UAV Digital Twin - AI Inference Service")


class TimestepInput(BaseModel):
    time: float
    altitude: float
    throttle: float
    airspeed: float
    aoa: float
    air_density: float
    torque_available_nm: float
    engine_rpm: float
    prop_rpm: float
    prop_torque: float
    power_kw: float
    fuel_flow: float
    thrust: float
    lift: float
    drag: float
    thrust_margin: float
    lift_weight_margin: float
    egt: float
    cht: float
    oil_pressure: float
    oil_temp: float
    vibx: float
    viby: float
    vibz: float
    rpm_fault: float


@app.post("/step")
def step(data: TimestepInput):
    """Call once per physics timestep. Returns None (still warming up) until the
    128-step buffer fills, then returns the full AI verdict on every call after that."""
    engine_state.update(data.dict())

    if not engine_state.is_ready():
        return {"status": "warming_up", "steps_collected": engine_state.n_steps, "steps_needed": WINDOW_SIZE}

    window = engine_state.get_window()[np.newaxis, ...]      # (1, 128, 24)
    rul_aux = engine_state.get_rul_aux()                      # (1, 6)

    result = inference_graph.invoke({"window": window, "rul_aux": rul_aux})
    output = result["final_output"]
    output["status"] = "ok"
    return output


@app.post("/reset")
def reset():
    """Call when a new flight/scenario starts - clears the rolling buffer."""
    engine_state.reset()
    return {"status": "reset"}


@app.get("/health")
def health():
    return {"status": "alive", "models_loaded": True, "buffer_fill": len(engine_state.buffer)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100)
