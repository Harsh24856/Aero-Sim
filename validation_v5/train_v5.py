#!/usr/bin/env python3
"""Train the joint v5 model for one engine (docs/v5_model_improvement_plan.md, Phases 4-5).

    validation/venv/bin/python validation_v5/train_v5.py --cache validation_v5/cache/914 --out validation_v5/artifacts/914

LOSSES, ONE MODEL
    detection   BCE                          y = engine fault visible OR sensor fault visible
    diagnosis   BCE, masked to faults this engine can have (no x12 positive weight:
                v4's inflated every probability and pushed 5-7 cut-offs to 0.95)
    family      BCE
    severity    mean |pred - true| on fault cells (true >= 0.08) + 0.5 x mean on clean
                cells (true == 0) + 0.2 x mean in between, each group averaged on its
                own - SYMMETRIC, so over-prediction costs
                (v4 masked to true >= 0.08 and never saw it)
    sensor      cross-entropy per channel, class-balanced by effective number of samples
    health      |pred - true|
    Balanced by learned uncertainty weights (Kendall, Gal & Cipolla 2018):
    total = sum_i exp(-s_i) L_i + s_i.
SPECIALISTS (specialists_v5.py)
    cfg "heads" restricts the loss to some heads, "select" picks the epoch on that
    head's own validation metric, "linear" gives severity/health linear outputs with
    a Huber loss, and "lr_schedule": "cosine" decays stage 1's learning rate. All
    default to the joint model's behaviour.
SELECTION
    Each epoch is scored on the validation flights with the honest metrics
    (evaluate.py); the best epoch by a composite of them is kept, with early
    stopping. v4 selected on losses that a do-nothing model could win.
DEVICE
    Metal GPU only - the device the live stack serves v4/v5 models on. Refuses to run without it.
CALIBRATION (after training)
    Per-fault temperature on the calibration split; per-fault cut-offs chosen on
    the validation split (grid 0.01-0.99). Test is never touched here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import queue
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tensorflow as tf  # noqa: E402

# GPU ONLY: models are trained on the Metal GPU and served on it. Weights trained on
# one device gave different answers on the other in v4, so a silent CPU fallback is
# refused rather than allowed.
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("No GPU visible to TensorFlow (tensorflow-metal) - v5 trains on the GPU only.")

import evaluate as E  # noqa: E402
from assembly_v5 import apply_temperature  # noqa: E402,F401  (re-exported: score_v5 imports it from here)
import model_architectures_v5 as M  # noqa: E402
from pipeline_v5 import Cache  # noqa: E402
from seed import set_random_seed  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))
import physics_v5 as P  # noqa: E402
from degradation_v5 import FAULT_NAMES, applicable_faults  # noqa: E402
from observability_v5 import FAMILY_OF  # noqa: E402
from sensors_v5 import FAULTABLE_CHANNELS  # noqa: E402

HEADS = ["detection", "diagnosis", "family", "severity", "sensor", "health"]
FAM_M = np.array(M.FAMILY_MATRIX, np.float32)


# ---------------------------------------------------------------------------
# Targets from cache end rows
# ---------------------------------------------------------------------------
def targets(cache: Cache, E_rows: np.ndarray) -> dict:
    L = lambda n: cache.labels(E_rows, n)  # noqa: E731
    fmv = np.stack([L(f"fmv_{n}") for n in FAULT_NAMES], 1).astype(np.float32)
    sf = np.stack([L(f"sf_{c}_flag") for c in FAULTABLE_CHANNELS], 1).astype(np.int32)
    det = np.maximum(L("fault_present"), (sf > 0).any(1)).astype(np.float32)[:, None]
    return {"detection": det, "diagnosis": fmv, "family": np.minimum(fmv @ FAM_M, 1.0),
            "severity": np.stack([L(f"fm_{n}") for n in FAULT_NAMES], 1).astype(np.float32),
            "sensor": sf, "health": L("health_index").astype(np.float32)[:, None]}


def class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    eff = (1.0 - np.power(beta, np.maximum(counts, 1))) / (1.0 - beta)
    w = 1.0 / eff
    return (w / w.sum() * len(counts)).astype(np.float32)


# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, cache: Cache, cfg: dict):
        self.cache, self.cfg = cache, cfg
        spec = P.ENGINE_SPECS[cache.contract["engine_model"]]
        app = set(applicable_faults(spec.turbocharged, spec.intercooled))
        self.fault_mask = tf.constant([[1.0 if n in app else 0.0 for n in FAULT_NAMES]])
        self.applicable = [n for n in FAULT_NAMES if n in app]
        self.model = M.build_model(channels=cfg["channels"], num_layers=cfg["layers"],
                                   dropout=cfg["dropout"], input_dropout=cfg["input_dropout"],
                                   head_hidden=cfg["head_hidden"],
                                   linear_regression=cfg.get("linear", False),
                                   enc_norm=cfg.get("enc_norm", False))
        self.head_w = tf.constant([1.0 if h in cfg.get("heads", HEADS) else 0.0 for h in HEADS])
        self.log_vars = tf.Variable(tf.zeros(len(HEADS)), name="log_vars")
        self.set_stage(cfg["lr"], freeze_encoder=False)
        # sensor class weights from a sample of training windows
        ids = cache.end_ids(["train"], seed=1, epoch=0)[:20000]
        _, _, Er = cache.batch(ids)
        sf = targets(cache, Er)["sensor"].ravel()
        self.sf_w = tf.constant(class_balanced_weights(np.bincount(sf, minlength=M.N_KINDS)))

    # ------------------------------------------------------------------
    def losses(self, out, y):
        bce = tf.keras.losses.binary_crossentropy
        l_det = tf.reduce_mean(bce(y["detection"], out["detection"]))
        per = tf.keras.backend.binary_crossentropy(y["diagnosis"], out["diagnosis"]) * self.fault_mask
        l_diag = tf.reduce_sum(per) / (tf.reduce_sum(self.fault_mask) * tf.cast(tf.shape(per)[0], tf.float32))
        l_fam = tf.reduce_mean(bce(y["family"], out["family"]))
        t, p = y["severity"], out["severity"]
        err = self._err(p - t)

        def mean_over(mask):
            mask = mask * self.fault_mask
            return tf.reduce_sum(mask * err) / tf.maximum(tf.reduce_sum(mask), 1.0)
        # Each group averaged on its own: fault cells are ~2% of cells, so a single
        # mean over all cells is won by predicting zero everywhere.
        if self.cfg.get("sev_fault_only", False):
            # Conditional severity: "if this fault is present, how bad". The clean-cell
            # term pulls every uncertain severity to 0 - the 914 specialist then
            # under-predicted real faults by 0.28. Absent faults are zeroed at serving
            # time by the diagnosis head (specialists_v5.assemble).
            l_sev = mean_over(tf.cast(t >= 0.08, tf.float32))
        else:
            l_sev = (mean_over(tf.cast(t >= 0.08, tf.float32)) + 0.5 * mean_over(tf.cast(t <= 1e-6, tf.float32))
                     + 0.2 * mean_over(tf.cast((t > 1e-6) & (t < 0.08), tf.float32)))
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y["sensor"], logits=out["sensor"])
        cw = tf.gather(self.sf_w, y["sensor"])
        l_sf = tf.reduce_sum(ce * cw) / tf.reduce_sum(cw)
        l_h = tf.reduce_mean(self._err(out["health"] - y["health"]))
        return [l_det, l_diag, l_fam, l_sev, l_sf, l_h]

    def _err(self, d):
        """|d| for the joint model; Huber (delta 0.05) for linear outputs, whose
        gradient then shrinks near the target instead of flipping sign at it."""
        a = tf.abs(d)
        if not self.cfg.get("linear", False):
            return a
        return tf.where(a < 0.05, 0.5 * a * a / 0.05, a - 0.025)

    def set_stage(self, lr: float, freeze_encoder: bool, warmup_steps: int = 0, decay_steps: int = 0) -> None:
        """New optimizer and a freshly traced step: the trainable set and the
        optimizer's variables must match, so both are rebuilt per stage."""
        for layer in self.model.layers:
            if layer.name.startswith(ENCODER_PREFIXES):
                layer.trainable = not freeze_encoder
        sched = lr if not warmup_steps else WarmUp(lr, warmup_steps, decay_steps)
        self.opt = tf.keras.optimizers.AdamW(learning_rate=sched, weight_decay=self.cfg["weight_decay"],
                                             clipnorm=1.0)
        self.train_step = tf.function(self._train_step, reduce_retracing=True)

    def _train_step(self, seq, ctx, y):
        with tf.GradientTape() as tape:
            out = self.model({"seq": seq, "ctx": ctx}, training=True)
            ls = tf.stack(self.losses(out, y))
            if self.cfg["balance"] == "uncertainty":
                total = tf.reduce_sum(self.head_w * (tf.exp(-self.log_vars) * ls + self.log_vars))
            else:
                total = tf.reduce_sum(self.head_w * ls)
        vars_ = self.model.trainable_variables + ([self.log_vars] if self.cfg["balance"] == "uncertainty" else [])
        grads = tape.gradient(total, vars_)
        self.opt.apply_gradients(zip(grads, vars_))
        return ls

    def predict(self, ids: np.ndarray, bs: int = 512) -> tuple[dict, dict, np.ndarray]:
        outs = {h: [] for h in HEADS}
        ys = {h: [] for h in HEADS}
        ends = []
        for i in range(0, len(ids), bs):
            seq, ctx, Er = self.cache.batch(ids[i:i + bs])
            o = self.model({"seq": seq, "ctx": ctx}, training=False)
            y = targets(self.cache, Er)
            for h in HEADS:
                v = np.asarray(o[h])
                outs[h].append(np.clip(v, 0.0, 1.0) if h in ("severity", "health") else v)
                ys[h].append(y[h])
            ends.append(Er)
        return ({h: np.concatenate(v) for h, v in outs.items()},
                {h: np.concatenate(v) for h, v in ys.items()}, np.concatenate(ends))

    def score(self, o: dict, y: dict, cuts=None) -> dict:
        app = [FAULT_NAMES.index(n) for n in self.applicable]
        yd, pd_ = y["diagnosis"][:, app], o["diagnosis"][:, app]
        names = [FAULT_NAMES[j] for j in app]
        s = {"det_auc": E.auc(y["detection"][:, 0], o["detection"][:, 0]),
             "det_recall_p95": E.recall_at_precision(y["detection"][:, 0], o["detection"][:, 0]),
             "diag_macro_f1": E.diagnosis_report(yd, pd_, names, cuts)["macro_f1"],
             "diag_macro_ap": E.macro_average_precision(yd, pd_),
             "family_f1": E.family_f1(yd, pd_, names, FAMILY_OF, cuts),
             "sev_on_fault_mae": E.severity_report(y["severity"][:, app], o["severity"][:, app])["on_fault_mae"],
             "sev_clean_mae": E.severity_report(y["severity"][:, app], o["severity"][:, app])["clean_mae"],
             "sensor_macro_f1": E.sensor_macro_f1(y["sensor"], o["sensor"].argmax(-1)),
             "health_mae": E.mae(y["health"], o["health"])}
        # Diagnosis enters through average precision (threshold-free): cut-offs are
        # tuned after training, so F1 at a fixed 0.5 would select on the wrong thing.
        s["composite"] = (s["det_auc"] + s["diag_macro_ap"] + 0.5 * s["family_f1"] + s["sensor_macro_f1"]
                          - s["sev_on_fault_mae"] - 2.0 * s["health_mae"])
        sel = self.cfg.get("select")
        s["select"] = float(sum(w * s[k] for k, w in sel.items())) if sel else s["composite"]
        return s

    # ------------------------------------------------------------------
    def _batches(self, ids, bs):
        q: queue.Queue = queue.Queue(maxsize=8)

        def work():
            for i in range(0, len(ids) - bs + 1, bs):
                seq, ctx, Er = self.cache.batch(ids[i:i + bs])
                q.put((seq, ctx, targets(self.cache, Er)))
            q.put(None)
        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            yield item

    def fit(self, epochs: int, patience: int, bs: int, out_dir: str | None = None,
            max_train_windows: int | None = None, verbose: bool = True,
            keep_current: bool = False, epoch0: int = 0) -> dict:
        """keep_current: the weights on entry are the version to beat (later
        stages), so a stage that does not improve the composite changes nothing.
        epoch0 offsets the window shuffle so later stages see fresh orders."""
        val_ids = self.cache.end_ids(["val"], jitter=False)
        if verbose:
            print(f"device: {[d.name for d in tf.config.list_physical_devices('GPU')]}", flush=True)
        best, best_ep, hist, wait = -np.inf, -1, [], 0
        best_w = None
        if keep_current:
            best, best_w = self.score(*self.predict(val_ids)[:2])["select"], self.model.get_weights()
        for ep in range(epochs):
            t0 = time.time()
            ids = self.cache.end_ids(["train"], seed=self.cfg["seed"], epoch=epoch0 + ep)
            if max_train_windows:
                ids = ids[:max_train_windows]
            tot = np.zeros(len(HEADS))
            nb = 0
            for seq, ctx, y in self._batches(ids, bs):
                tot += self.train_step(tf.constant(seq), tf.constant(ctx),
                                       {k: tf.constant(v) for k, v in y.items()}).numpy()
                nb += 1
            o, y, _ = self.predict(val_ids)
            sc = self.score(o, y)
            rec = {"epoch": ep, "seconds": round(time.time() - t0, 1),
                   "train_losses": dict(zip(HEADS, (tot / max(nb, 1)).round(4).tolist())),
                   "log_vars": self.log_vars.numpy().round(3).tolist(), "val": {k: round(v, 4) for k, v in sc.items()}}
            hist.append(rec)
            if verbose:
                print(f"ep {ep:2d} {rec['seconds']:6.1f}s  " + "  ".join(f"{k} {v:.3f}" for k, v in rec["val"].items()),
                      flush=True)
            if sc["select"] > best + 1e-4:
                best, best_ep, wait = sc["select"], ep, 0
                best_w = self.model.get_weights()
            else:
                wait += 1
                if wait >= patience:
                    break
        self.model.set_weights(best_w)
        return {"best_epoch": best_ep, "best_composite": best, "history": hist}

    def train_protocol(self, epochs: int, patience: int, bs: int, max_train_windows: int | None = None,
                       verbose: bool = True) -> dict:
        """Phase 5 task 1: joint training, then the heads alone on a frozen encoder,
        then everything unfrozen at a low learning rate with a warm-up (v4 restarted
        at 1e-3 on the whole encoder). Each later stage keeps its result only if it
        beats the validation composite so far."""
        kw = dict(max_train_windows=max_train_windows, verbose=verbose)
        steps = (max_train_windows or len(self.cache.end_ids(["train"], jitter=False))) // bs
        if self.cfg.get("lr_schedule") == "cosine":
            self.set_stage(self.cfg["lr"], freeze_encoder=False, warmup_steps=steps, decay_steps=epochs * steps)
        out = {"joint": self.fit(epochs, patience, bs, **kw)}
        n = out["joint"]["history"][-1]["epoch"] + 1
        short = max(2, epochs // 4)
        if verbose:
            print("-- stage 2: heads only, encoder frozen", flush=True)
        self.set_stage(self.cfg["lr"], freeze_encoder=True)
        out["heads_only"] = self.fit(short, 3, bs, keep_current=True, epoch0=n, **kw)
        n += len(out["heads_only"]["history"])
        if verbose:
            print("-- stage 3: all layers, low LR with warm-up", flush=True)
        self.set_stage(min(1e-4, self.cfg["lr"] / 5), freeze_encoder=False, warmup_steps=max(1, steps // 2))
        out["unfrozen"] = self.fit(short, 3, bs, keep_current=True, epoch0=n, **kw)
        out["best_composite"] = out["unfrozen"]["best_composite"]
        # history.json keeps the joint stage's shape for the notebooks
        out["history"], out["best_epoch"] = out["joint"]["history"], out["joint"]["best_epoch"]
        return out

    # ------------------------------------------------------------------
    def save(self, out_dir: str) -> None:
        """Weights + config: the model is rebuilt from code on load, so no custom
        layer ever has to be deserialised."""
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_weights(os.path.join(out_dir, "model_v5.weights.h5"))
        with open(os.path.join(out_dir, "config_v5.json"), "w") as fh:
            json.dump(self.cfg, fh, indent=1)

    @classmethod
    def load(cls, cache: Cache, out_dir: str) -> "Trainer":
        with open(os.path.join(out_dir, "config_v5.json")) as fh:
            cfg = json.load(fh)
        tr = cls(cache, cfg)
        tr.model.load_weights(os.path.join(out_dir, "model_v5.weights.h5"))
        return tr

    # ------------------------------------------------------------------
    def calibrate(self) -> dict:
        """Per-fault temperature on the calibration split, cut-offs on validation."""
        app = [FAULT_NAMES.index(n) for n in self.applicable]
        o, y, _ = self.predict(self.cache.end_ids(["cal"], jitter=False))
        temps = np.ones(len(FAULT_NAMES))
        grid_t = np.exp(np.linspace(np.log(0.3), np.log(5.0), 40))
        for j in app:
            p = np.clip(o["diagnosis"][:, j], 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            t = y["diagnosis"][:, j]
            nll = [-np.mean(t * np.log(1 / (1 + np.exp(-z / T)) + 1e-9)
                            + (1 - t) * np.log(1 - 1 / (1 + np.exp(-z / T)) + 1e-9)) for T in grid_t]
            temps[j] = float(grid_t[int(np.argmin(nll))])
        o, y, _ = self.predict(self.cache.end_ids(["val"], jitter=False))
        # Keep a fault's temperature only if it improves that fault's validation ECE.
        for j in app:
            if E.ece(y["diagnosis"][:, [j]], apply_temperature(o["diagnosis"][:, [j]], [temps[j]])) \
                    > E.ece(y["diagnosis"][:, [j]], o["diagnosis"][:, [j]]):
                temps[j] = 1.0
        pc = apply_temperature(o["diagnosis"], temps)
        cuts = np.full(len(FAULT_NAMES), 0.5)
        grid_c = np.linspace(0.01, 0.99, 99)
        for j in app:
            if y["diagnosis"][:, j].sum() == 0:
                continue
            cuts[j] = choose_cutoff(y["diagnosis"][:, j], pc[:, j], grid_c, self.cfg.get("min_recall", 0.0))
        return {"temperature": temps.tolist(), "cutoffs": cuts.tolist(),
                "val_ece_after": E.ece(y["diagnosis"][:, app], pc[:, app]),
                "val_ece_before": E.ece(y["diagnosis"][:, app], o["diagnosis"][:, app])}


# A cut-off may sit below the exact validation-F1 maximum - a sliver on a flat plateau
# that does not hold on other flights (914 v6: valve 0.87 -> test recall 0.49, where
# 0.74 costs 0.009 F1) - while BOTH bounds hold: at most CUT_F1_TOL less F1, and at
# most CUT_ERR_TOL more error (1 - F1). Each alone failed on the 914 v6 card: the
# absolute bound let near-perfect faults fall to the noise floor (combustion / bearing
# 0.03 / 0.06), the relative one let weak faults slide (oil degradation 0.39 -> 0.15).
CUT_F1_TOL = 0.01
CUT_ERR_TOL = 0.20


def choose_cutoff(y: np.ndarray, p: np.ndarray, grid: np.ndarray, min_recall: float = 0.0) -> float:
    """Per-fault cut-off: among cut-offs keeping `min_recall` (the gate's per-fault
    floor is 0.5; none keep it -> all cut-offs), the LOWEST within CUT_F1_TOL of the
    best F1 and within CUT_ERR_TOL more error than the best."""
    prf = [E.prf(y, p >= c) for c in grid]
    f1 = np.array([x[2] for x in prf])
    ok = np.array([x[1] >= min_recall for x in prf])
    f1 = np.where(ok, f1, -1.0) if ok.any() else f1
    best = f1.max()
    near = (f1 >= best - CUT_F1_TOL - 1e-12) & (1.0 - f1 <= (1.0 - best) * (1.0 + CUT_ERR_TOL) + 1e-12)
    return float(grid[int(np.argmax(near))])


ENCODER_PREFIXES = ("in_drop", "tcn", "ctx_d")


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up to `lr` over `steps`, then flat - or, with `decay_steps`,
    cosine decay to 5% of `lr` by step `decay_steps`."""

    def __init__(self, lr: float, steps: int, decay_steps: int = 0):
        self.lr, self.steps, self.decay_steps = lr, steps, decay_steps

    def __call__(self, step):
        s = tf.cast(step + 1, tf.float32)
        lr = self.lr * tf.minimum(1.0, s / self.steps)
        if self.decay_steps:
            frac = tf.clip_by_value((s - self.steps) / max(self.decay_steps - self.steps, 1), 0.0, 1.0)
            lr *= 0.05 + 0.95 * 0.5 * (1.0 + tf.cos(np.pi * frac))
        return lr

    def get_config(self):
        return {"lr": self.lr, "steps": self.steps, "decay_steps": self.decay_steps}


DEFAULT_CFG = {"channels": 64, "layers": 6, "dropout": 0.2, "input_dropout": 0.03, "head_hidden": 96,
               "lr": 5e-4, "weight_decay": 1e-4, "balance": "uncertainty", "batch": 128, "seed": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="JSON config (overrides defaults)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()
    cfg = dict(DEFAULT_CFG, **(json.load(open(a.config)) if a.config else {}))
    cfg["seed"] = set_random_seed(cfg.get("seed", 0))
    os.makedirs(a.out, exist_ok=True)
    tr = Trainer(Cache(a.cache), cfg)
    res = tr.train_protocol(a.epochs, a.patience, cfg["batch"])
    cal = tr.calibrate()
    tr.save(a.out)
    with open(os.path.join(a.out, "run_v5.json"), "w") as fh:
        json.dump({"config": cfg, "fit": res, "calibration": cal,
                   "engine_model": tr.cache.contract["engine_model"]}, fh, indent=1, default=float)
    print(json.dumps({"best_epoch": res["best_epoch"], "calibration_ece": [cal["val_ece_before"], cal["val_ece_after"]]}))


if __name__ == "__main__":
    main()
