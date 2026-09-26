"""Shared plumbing for the v4 notebooks.

The notebooks are written to be READ: each one answers one question about the
engine and shows the answer. Path handling, model loading and array bookkeeping
live here so that every notebook cell is one understandable step.

Nothing in this file trains or scores differently from run.py - it calls the
same functions, so a number in a notebook and the same number in the run.py log
always agree.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)                     # artifact and data paths are relative to here
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

import tf_data_pipeline as P
import model_architectures as A    # noqa: F401 - registers custom layers so checkpoints load
import train_common as C
import run as R

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 9.5})
pd.set_option("display.max_colwidth", 120)

ENGINE_NAMES = {"912": "Rotax 912 ULS", "914": "Rotax 914 ULF",
                "915": "Rotax 915 iS", "916": "Rotax 916 iS"}

HEALTHY, FAULTY = "#2c7bb6", "#d7191c"

# (phase, metric key) -> (plain name, what it means)
METRICS = {
    ("detection", "auc"): (
        "Detection score (AUC)",
        "How well faulty windows are ranked above healthy ones. 0.5 is a coin "
        "toss, 1.0 is perfect."),
    ("detection", "auc_high_residual"): (
        "Detection where it matters",
        "The same score, only on windows where the engine has clearly drifted "
        "from its healthy twin - the cases an operator must not miss."),
    ("diagnosis", "macro_f1"): (
        "Fault-naming score (macro F1)",
        "How well the failing part is named, averaged over every fault type so a "
        "rare fault counts as much as a common one. 1.0 is perfect."),
    ("severity", "mae"): (
        "Severity error",
        "Average error in how far a fault has progressed, on a 0 (just started) "
        "to 1 (fully developed) scale. Lower is better."),
    ("severity", "mae_on_fault"): (
        "Severity error where a fault exists",
        "The same error, only on faults that are really there - the plain error "
        "above is flattered by the 97% of cells with no fault. Lower is better."),
    ("sensor_fault", "macro_f1"): (
        "Sensor-condition score (macro F1)",
        "How well each of the 7 conditions is recognised, every kind counting "
        "equally with 'fine' - so always answering 'fine' scores low. 1.0 is perfect."),
    ("sensor_fault", "acc"): (
        "Sensor-fault accuracy",
        "Share of sensor channels whose condition (fine, drifting, stuck, noisy "
        "...) is identified correctly."),
    ("health", "mae"): (
        "Wear-condition error",
        "Average error in the engine's wear condition, on a 1 (new) to 0 (worn "
        "out) scale. Lower is better."),
    ("rul", "mae_pct_tbo"): (
        "Remaining-life error, all engines",
        "Average error in hours left, as a percentage of the overhaul interval "
        "(TBO). Lower is better."),
    ("rul", "mae_pct_tbo_wear_limited"): (
        "Remaining-life error, worn engines",
        "The same, only for engines whose life is cut short by wear - the hard "
        "cases where age alone gives the wrong answer."),
}

PHASE_TIME_MIN = {"detection": 55, "diagnosis": 70, "severity": 35,
                  "sensor_fault": 40, "health": 30, "rul": 25}


# ---------------------------------------------------------------------------
def setup(engine: str, phase: str) -> None:
    """Print a short header so it is obvious what this notebook is looking at."""
    print(f"Engine : {ENGINE_NAMES[engine]}")
    print(f"Step   : {phase}")
    print(f"Device : {C.report_device()}")
    if os.path.exists(os.path.join(C.data_dir(engine), "manifest.json")):
        m = C.manifest(engine)
        print(f"Data   : {m['rows']:,} rows from {m['scenarios']:,} simulated flights")
    else:
        print("Data   : NOT GENERATED YET - run scripts/generate_v4.sh first")


def is_trained(engine: str, phase: str) -> bool:
    return (os.path.exists(C.ckpt_path(engine, phase))
            and os.path.exists(C.result_path(engine, phase)))


def train_or_load(engine: str, phase: str) -> dict:
    """Train this step, or load it instantly if run.py already trained it."""
    if is_trained(engine, phase):
        print("Already trained - loading the saved result (instant).")
    else:
        print(f"Training now. This takes roughly {PHASE_TIME_MIN.get(phase, 30)} "
              f"minutes; progress is printed once per pass over the data.")
    return R.phase_train(engine, phase, force=False, verbose=2)


def load_rows(engine: str, columns: list, split: str = "test",
              max_rows: int = 400_000) -> pd.DataFrame:
    """A slice of the raw generated data, for the 'look at the data' plots."""
    f = sorted(glob.glob(os.path.join(C.data_dir(engine), split, "*.parquet")))[0]
    return pq.read_table(f, columns=columns).to_pandas().head(max_rows)


def example_flights(engine: str):
    """One clean healthy flight and one flight with an engine fault, from the
    test data, plus the twin residual that moves most on the faulty one.

    Both are chosen free of sensor faults, so the picture shows an ENGINE fault
    and nothing else.
    """
    cols = ["scenario_id", "t", "fault_present"] + P.RESIDUAL_COLS + P.SF_SEV_COLS
    df = load_rows(engine, cols)
    df["sensor_fault"] = df[P.SF_SEV_COLS].max(axis=1)
    g = df.groupby("scenario_id").agg(fault=("fault_present", "mean"),
                                      sensor=("sensor_fault", "max"), n=("t", "size"))
    clean = g[(g.fault == 0) & (g.sensor == 0)]
    sick = g[(g.fault > 0.3) & (g.sensor == 0)]
    healthy = df[df.scenario_id == (clean if len(clean) else g).n.idxmax()]
    faulty = df[df.scenario_id == (sick if len(sick) else g).fault.idxmax()]
    spread = healthy[P.RESIDUAL_COLS].std().replace(0, 1)
    channel = (faulty[P.RESIDUAL_COLS].abs().quantile(0.9) / spread).idxmax()
    return healthy, faulty, channel


# ---------------------------------------------------------------------------
def scorecard(engine: str, phase: str) -> pd.DataFrame:
    """Pass/fail against the quality bar, in plain words."""
    res = json.load(open(C.result_path(engine, phase)))["test"]
    rows = []
    for key, op, th in R.GATES.get(phase, []):
        v = res.get(key)
        ok = v is not None and (v >= th if op == ">=" else v <= th)
        name, meaning = METRICS[(phase, key)]
        unit = "%" if key.startswith("mae_pct") else ""
        rows.append({"Measure": name,
                     "Result": "missing" if v is None else f"{v:.3f}{unit}",
                     "Needs": f"{'at least' if op == '>=' else 'at most'} {th}{unit}",
                     "Status": "PASS" if ok else "FAIL",
                     "What it means": meaning})
    if phase in R.SELECT:                      # the number training optimised
        key = R.SELECT[phase][0]
        v = res.get(key)
        name, meaning = METRICS[(phase, key)]
        rows.append({"Measure": name,
                     "Result": "missing" if v is None else f"{v:.3f}",
                     "Needs": "no bar yet - tracked",
                     "Status": "TRACKED",
                     "What it means": meaning})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
_PRED_CACHE: dict = {}


def predict_test(engine: str, phase: str) -> dict:
    """Run this step's model over the whole held-out test split.

    Returns the model's predictions and the true answers for this step's
    output, plus the largest twin residual in each window (for detection) and
    the life-stage inputs (for remaining life). Cached, so plotting twice does
    not predict twice.
    """
    if (engine, phase) in _PRED_CACHE:
        return _PRED_CACHE[(engine, phase)]
    from tensorflow import keras
    head = C.PHASE_HEADS[phase][0]
    model = keras.models.load_model(C.ckpt_path(engine, phase),
                                    compile=False, safe_mode=False)
    res_idx = [P.FEATURE_COLS.index(c) for c in P.RESIDUAL_COLS]
    out = {"pred": [], "true": [], "resid": [], "aux": []}
    for xb, yb in C.datasets(engine)["test"]:
        out["pred"].append(model.predict(xb, verbose=0)[head])
        out["true"].append(yb[head].numpy())
        out["resid"].append(np.abs(xb["x"].numpy()[:, :, res_idx]).max(axis=(1, 2)))
        out["aux"].append(xb["x_rul_aux"].numpy())
    r = {k: np.concatenate(v) for k, v in out.items()}
    _PRED_CACHE[(engine, phase)] = r
    print(f"Predicted {len(r['true']):,} test windows "
          f"({len(r['true']) * P.WINDOW_SIZE / 3600:.0f} hours of flight the model never trained on).")
    return r


def calendar_rul(engine: str, aux: np.ndarray) -> np.ndarray:
    """Hours to scheduled overhaul from engine age alone - what RUL would be if
    nothing wore out early."""
    import joblib
    ax = joblib.load(C.aux_scaler_path(engine))
    k = P.RUL_AUX_ORDER.index("engine_hours_norm")
    hours_norm = ax.inverse_transform(aux.astype(np.float64))[:, k]
    return C.tbo_of(engine) * np.clip(1.0 - hours_norm, 0.0, None)


def applicable_faults(engine: str) -> list:
    """Fault types this engine can physically have (a 912 has no turbo)."""
    backend = os.path.join(os.path.dirname(HERE), "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    import physics_v4 as V
    from degradation_v4 import applicable_faults as af
    s = next(v for k, v in V.ENGINE_SPECS_V4.items() if k.split("_")[1] == engine)
    return af(s.turbocharged, s.intercooled)


def pretty(name: str) -> str:
    return name.replace("_", " ")
