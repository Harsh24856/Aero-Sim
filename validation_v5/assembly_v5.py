"""The v5 specialists as ONE model, shared by training (specialists_v5.py) and
serving (backend/aiv5.py) so the two can never disagree.

Five specialist networks each answer their own heads (SOURCE). The assembly then
applies, in this order, what specialists_v5.assemble fitted on calibration flights:
    detection   GBT over the specialists' outputs + the largest residual
    health      GBT over the network's health + the 45 context values
    sensor      a bias on the "none" logit
    severity    per-fault isotonic calibration (the specialist under-predicted real
                faults by 0.064), then zeroed where calibrated diagnosis says absent
The stackers read the RAW outputs, so they run before anything is replaced.

`Deployed` loads an export (backend/models_v5/<key>/, written by export_v5.py) and
also does what the training cache did offline: scale raw 1 Hz features and context
exactly as pipeline_v5 did, and compute RUL.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

import features_v5 as F  # noqa: E402

RES_SLICE = slice(F.FEATURE_COLS.index(F.RESIDUAL_COLS[0]), F.FEATURE_COLS.index(F.RESIDUAL_COLS[-1]) + 1)
HEADS_OF = {"detection": ["detection"], "diagnosis": ["diagnosis", "family"], "severity": ["severity"],
            "sensor": ["sensor"], "health": ["health"]}
SOURCE = {k: spec for spec, ks in HEADS_OF.items() for k in ks}      # output -> specialist


def apply_temperature(p: np.ndarray, temps) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / np.asarray(temps)[None, :]
    return 1 / (1 + np.exp(-z))


def health_features(o: dict, ctx: np.ndarray) -> np.ndarray:
    return np.concatenate([o["health"], ctx], 1)


def detection_features(o: dict, seq: np.ndarray) -> np.ndarray:
    z = o["sensor"] - o["sensor"].max(-1, keepdims=True)
    p_none = np.exp(z[..., 0]) / np.exp(z).sum(-1)                   # [B, 12]
    res_max = np.abs(seq[:, :, RES_SLICE]).max(axis=(1, 2))
    return np.stack([o["detection"][:, 0], (1 - p_none).max(1), (1 - p_none).mean(1),
                     o["diagnosis"].max(1), o["family"].max(1), res_max], 1)


def calibrate_severity(sev: np.ndarray, iso: dict) -> np.ndarray:
    """iso: fault column -> fitted sklearn IsotonicRegression (true vs predicted
    severity on fault cells). Columns without one pass through."""
    out = sev.copy()
    for j, m in iso.items():
        out[:, int(j)] = m.predict(sev[:, int(j)])
    return out


def gate_severity(sev: np.ndarray, diag: np.ndarray, gate: dict) -> np.ndarray:
    """Severity kept only where calibrated diagnosis probability >= the gate cut."""
    p = apply_temperature(diag, gate["temperature"])
    return np.where(p >= np.asarray(gate["cut"])[None, :], sev, 0.0)


class Assembly:
    """nets: specialist name -> callable({"seq", "ctx"}, training=False) -> dict
    holding at least that specialist's heads. extra: assembly.pkl contents."""

    def __init__(self, nets: dict, extra: dict | None = None):
        self.nets = nets
        self.extra = extra or {}

    def raw(self, seq, ctx) -> dict:
        """Each output from its own specialist, before any assembly step."""
        outs = {h: self.nets[h]({"seq": seq, "ctx": ctx}, training=False) for h in HEADS_OF}
        o = {k: np.asarray(outs[src][k]) for k, src in SOURCE.items()}
        o["severity"] = np.clip(o["severity"], 0.0, 1.0)
        o["health"] = np.clip(o["health"], 0.0, 1.0)
        return o

    def __call__(self, inputs, training=False) -> dict:
        seq, ctx = np.asarray(inputs["seq"]), np.asarray(inputs["ctx"])
        o = self.raw(seq, ctx)
        x = self.extra
        det = x["detection_gbt"].predict_proba(detection_features(o, seq))[:, 1:2] \
            if x.get("detection_gbt") is not None else None
        if x.get("health_gbt") is not None:
            o["health"] = np.clip(x["health_gbt"].predict(health_features(o, ctx)), 0, 1)[:, None]
        if det is not None:
            o["detection"] = det
        if "sensor_bias" in x:
            o["sensor"] = o["sensor"].copy()
            o["sensor"][..., 0] += x["sensor_bias"]
        if x.get("sev_iso"):
            o["severity"] = calibrate_severity(o["severity"], x["sev_iso"])
        if x.get("sev_gate") is not None:
            o["severity"] = gate_severity(o["severity"], o["diagnosis"], x["sev_gate"])
        return o


# ---------------------------------------------------------------------------
# Serving: an export on disk
# ---------------------------------------------------------------------------
def _slim(model, heads: list):
    """A view of `model` that computes only `heads` (shares its layers and weights),
    compiled once: eager Keras at batch 1 cost ~225 ms per second of flight for the
    five networks, compiled ~100 ms. Since the ReLU fix compiled and eager agree to
    float rounding (sensor logits ~1e-5 on values ~200, 7e-8 relative)."""
    import tensorflow as tf
    view = tf.keras.Model(model.inputs, {h: model.output[h] for h in heads}, name=f"{model.name}_{'_'.join(heads)}")
    run = tf.function(lambda x: view(x, training=False), reduce_retracing=True)

    def call(inputs, training=False):
        return run({k: tf.convert_to_tensor(v, tf.float32) for k, v in inputs.items()})
    return call


class Deployed(Assembly):
    """An export directory: five slim specialists, the assembly, RUL and the
    scaling contract. Everything a live service needs besides the 1 Hz twin."""

    def __init__(self, path: str, slim: bool = True):
        import model_architectures_v5 as M
        self.path = path
        with open(os.path.join(path, "manifest.json")) as fh:
            self.manifest = json.load(fh)
        with open(os.path.join(path, "contract_v5.json")) as fh:
            self.contract = json.load(fh)
        with open(os.path.join(path, "calibration.json")) as fh:
            self.calibration = json.load(fh)
        nets = {}
        for h, heads in HEADS_OF.items():
            with open(os.path.join(path, h, "config_v5.json")) as fh:
                cfg = json.load(fh)
            m = M.build_model(channels=cfg["channels"], num_layers=cfg["layers"], dropout=cfg["dropout"],
                              input_dropout=cfg["input_dropout"], head_hidden=cfg["head_hidden"],
                              linear_regression=cfg.get("linear", False), enc_norm=cfg.get("enc_norm", False))
            m.load_weights(os.path.join(path, h, "model_v5.weights.h5"))
            nets[h] = _slim(m, heads) if slim else m
        with open(os.path.join(path, "assembly.pkl"), "rb") as fh:
            extra = pickle.load(fh)
        with open(os.path.join(path, "rul_v5.pkl"), "rb") as fh:
            self.rul = pickle.load(fh)
        super().__init__(nets, extra)
        sc, cs = self.contract["scaler"], self.contract["ctx_scaler"]
        self.mean, self.std = np.array(sc["mean"]), np.array(sc["std"])
        self.res_sigma = np.array(sc["residual_sigma"])
        self.cmean, self.cstd = np.array(cs["mean"]), np.array(cs["std"])
        self.tbo = float(self.contract["tbo_hours"])

    # -- scaling, exactly as pipeline_v5.build_cache ------------------------------
    def scale_seq(self, rows: np.ndarray) -> np.ndarray:
        """rows [..., 39] raw 1 Hz features in FEATURE_COLS order -> model input.
        The cache stored float16; round the same way so live == offline."""
        a = np.array(rows, dtype=np.float64)
        a[..., RES_SLICE] /= self.res_sigma
        return ((a - self.mean) / self.std).astype(np.float16).astype(np.float32)

    def scale_ctx(self, raw_ctx: np.ndarray) -> np.ndarray:
        """raw [..., 45]: 42 long-horizon values (residuals in sigma units, from
        features_v5.LongHorizon) + hours_frac, life_used_hours, life-clock flag."""
        return ((np.asarray(raw_ctx, np.float64) - self.cmean) / self.cstd).astype(np.float32)

    def predict_window(self, seq: np.ndarray, ctx: np.ndarray, engine_hours: float,
                       tbo: float | None = None) -> dict:
        """seq [128, 39] and ctx [45], both already scaled. Returns every head plus
        raw (unsmoothed) RUL hours and the calendar RUL. tbo: the flown engine's
        (default: the export's own) - RUL is a fraction of TBO, so a placeholder
        export answers on the engine it is standing in for."""
        tbo = self.tbo if tbo is None else float(tbo)
        import rul_v5
        s, c = seq[None], ctx[None]
        o = self({"seq": s, "ctx": c})
        X = rul_v5.rul_inputs(o, s, c, self.contract)
        calendar = max(0.0, tbo - engine_hours)
        rul_h = float(min(max(float(self.rul.predict(X)[0]), 0.0) * tbo, calendar))
        return {**{k: v[0] for k, v in o.items()}, "rul_hours": rul_h, "rul_calendar_hours": calendar}
