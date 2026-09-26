"""Shared training machinery for v4.

Mirrors validation/train_common.py: the same path conventions, the same
warm-start-from-the-previous-phase pattern, the same callback set. What is new
is a set of metrics that answer the question each head is actually for, because
accuracy is meaningless on every head here.

WHY THE CUSTOM METRICS EXIST
    Every target in this dataset is imbalanced or heavy-tailed, and the default
    Keras metric is misleading on each:

    detection     rows are ~30% faulty, so accuracy is beaten by predicting
                  "faulty" always on some engines. AUC and recall at a fixed
                  precision are what matter to an operator.
    fault_mode    14 modes, each 1-8% of rows. Plain accuracy scores 95% by
                  predicting all-zero. Macro F1 over present modes, and
                  per-mode recall, are the honest numbers.
    sensor_fault  6 of 7 classes are rare; same problem.
    rul           absolute hours are not comparable between a 1200 h and a
                  2000 h engine, so error is reported as a percentage of TBO.
"""
from __future__ import annotations

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tf_data_pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data", "rotax_v4")
ART_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

BATCH_SIZE = 128
ENGINES = ["912", "914", "915", "916"]

PHASES = ["scalers", "detection", "diagnosis", "severity",
          "sensor_fault", "health", "rul"]

# Head trained by each phase, and the head it warm-starts its encoder from.
PHASE_HEADS = {
    "detection":    ("y_detection", None),
    "diagnosis":    ("y_fault_mode", "detection"),
    "severity":     ("y_fault_mode", "diagnosis"),      # same head, severity-weighted loss
    "sensor_fault": ("y_sensor_fault", "diagnosis"),
    "health":       ("y_health", "diagnosis"),
    "rul":          ("y_rul_hours", "health"),
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def data_dir(engine: str) -> str:
    return os.path.join(DATA_ROOT, engine)


def scaler_path(engine: str) -> str:
    return os.path.join(ART_ROOT, engine, "scaler.pkl")


def aux_scaler_path(engine: str) -> str:
    return os.path.join(ART_ROOT, engine, "aux_scaler.pkl")


def ckpt_path(engine: str, phase: str) -> str:
    return os.path.join(ART_ROOT, engine, f"{engine}_{phase}.keras")


def result_path(engine: str, phase: str) -> str:
    return os.path.join(ART_ROOT, engine, f"{engine}_{phase}_test.json")


def manifest(engine: str) -> dict:
    return P.load_manifest(data_dir(engine))


def tbo_of(engine: str) -> float:
    return float(manifest(engine)["tbo_hours"])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def datasets(engine: str, batch_size: int = BATCH_SIZE, shuffle_buffer: int = 4000):
    d, sc, ax = data_dir(engine), scaler_path(engine), aux_scaler_path(engine)
    return {s: P.make_dataset(d, s, sc, aux_scaler_path=ax,
                              batch_size=batch_size,
                              shuffle_buffer=shuffle_buffer,
                              repeat=(s == "train"))
            for s in ("train", "val", "test")}


def step_counts(engine: str, batch_size: int = BATCH_SIZE) -> dict:
    d = data_dir(engine)
    return {s: P.steps_per_epoch(d, s, batch_size=batch_size)
            for s in ("train", "val", "test")}


def describe(engine: str) -> dict:
    m = manifest(engine)
    sp = m["splits"]
    return {"engine": engine, "rows": m["rows"], "scenarios": m["scenarios"],
            "tbo_hours": m["tbo_hours"], "n_features": m["contract"]["n_features"],
            "splits": {k: v["rows"] for k, v in sp.items()},
            "windows": {s: P.count_windows(data_dir(engine), s)
                        for s in ("train", "val", "test")}}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class MacroF1(keras.metrics.Metric):
    """Macro F1 over multi-label fault modes, counting only modes that occur.

    A mode absent from a batch would otherwise contribute a perfect score for
    predicting nothing, which is how a 14-way head with 2% positives reports
    0.95 while detecting none of them.
    """

    def __init__(self, n_labels: int, threshold: float = 0.5, name="macro_f1", **kw):
        super().__init__(name=name, **kw)
        self.n = n_labels
        self.th = threshold
        self.tp = self.add_weight(shape=(n_labels,), initializer="zeros", name="tp")
        self.fp = self.add_weight(shape=(n_labels,), initializer="zeros", name="fp")
        self.fn = self.add_weight(shape=(n_labels,), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = tf.cast(y_true >= P.DETECTION_THRESHOLD, tf.float32)
        p = tf.cast(y_pred >= self.th, tf.float32)
        self.tp.assign_add(tf.reduce_sum(t * p, axis=0))
        self.fp.assign_add(tf.reduce_sum((1 - t) * p, axis=0))
        self.fn.assign_add(tf.reduce_sum(t * (1 - p), axis=0))

    def result(self):
        f1 = 2 * self.tp / tf.maximum(2 * self.tp + self.fp + self.fn, 1e-9)
        present = tf.cast((self.tp + self.fn) > 0, tf.float32)
        return tf.reduce_sum(f1 * present) / tf.maximum(tf.reduce_sum(present), 1.0)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros_like(v))


class FaultMae(keras.metrics.Metric):
    """Severity MAE on the cells where a fault is actually present.

    The all-cell MAE the severity gate uses is passed by a head that predicts
    zero everywhere: on the new 914 data 97.6% of severity cells are exactly
    0, so "no fault anywhere" scores ~0.015 against a 0.12 gate. This is the
    number the severity question is really asking about.
    """

    def __init__(self, name="mae_on_fault", **kw):
        super().__init__(name=name, **kw)
        self.total = self.add_weight(initializer="zeros", name="total")
        self.count = self.add_weight(initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = tf.cast(y_true, tf.float32)
        m = tf.cast(t >= P.DETECTION_THRESHOLD, tf.float32)
        self.total.assign_add(tf.reduce_sum(m * tf.abs(t - y_pred)))
        self.count.assign_add(tf.reduce_sum(m))

    def result(self):
        return self.total / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class SensorMacroF1(keras.metrics.Metric):
    """Macro F1 over the 7 sensor conditions, from the argmax per channel.

    Plain accuracy is passed by answering "none" for every channel (97.8% of
    channels on the new data, against a 0.90 gate). Macro F1 weights each
    fault kind equally with "none", and its precision term punishes a head
    that buys recall by calling healthy sensors faulty.
    """

    def __init__(self, n_classes: int = P.N_SENSOR_FAULT_TYPES, name="macro_f1", **kw):
        super().__init__(name=name, **kw)
        self.n = n_classes
        self.tp = self.add_weight(shape=(n_classes,), initializer="zeros", name="tp")
        self.fp = self.add_weight(shape=(n_classes,), initializer="zeros", name="fp")
        self.fn = self.add_weight(shape=(n_classes,), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), self.n)
        p = tf.one_hot(tf.argmax(tf.reshape(y_pred, [-1, self.n]), -1), self.n)
        self.tp.assign_add(tf.reduce_sum(t * p, axis=0))
        self.fp.assign_add(tf.reduce_sum((1 - t) * p, axis=0))
        self.fn.assign_add(tf.reduce_sum(t * (1 - p), axis=0))

    def result(self):
        f1 = 2 * self.tp / tf.maximum(2 * self.tp + self.fp + self.fn, 1e-9)
        present = tf.cast((self.tp + self.fn) > 0, tf.float32)
        return tf.reduce_sum(f1 * present) / tf.maximum(tf.reduce_sum(present), 1.0)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros_like(v))


def sensor_weighted_ce(fault_weight: float = 8.0):
    """Sparse cross-entropy with the six fault kinds weighted up.

    Each fault kind is ~0.4% of channels, ~250:1 against "none". Unweighted,
    the cheapest answer is "none" everywhere. The weight is set well below the
    full ratio for the same reason presence_bce's is: the head should name a
    broken sensor, not flag healthy ones.
    """
    def loss(y_true, y_pred):
        t = tf.cast(y_true, tf.int32)
        ce = keras.losses.sparse_categorical_crossentropy(t, y_pred)
        w = tf.where(t > 0, fault_weight, 1.0)
        return tf.reduce_mean(w * ce)
    return loss


class MaePctTbo(keras.metrics.Metric):
    """RUL error as a percentage of overhaul life.

    Absolute hours are not comparable across engines: 40 h of error is 2% of the
    914's life and 3.3% of the 915's. Every RUL number in this project is
    reported this way so the four engines can be compared at all.
    """

    def __init__(self, tbo_hours: float, name="mae_pct_tbo", **kw):
        super().__init__(name=name, **kw)
        self.tbo = float(tbo_hours)
        self.total = self.add_weight(initializer="zeros", name="total")
        self.count = self.add_weight(initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):
        err = tf.abs(tf.squeeze(y_pred, -1) - tf.cast(y_true, tf.float32))
        self.total.assign_add(tf.reduce_sum(err) * 100.0 / self.tbo)
        self.count.assign_add(tf.cast(tf.size(err), tf.float32))

    def result(self):
        return self.total / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


def presence_bce(threshold: float = 0.08, pos_weight: float = 12.0):
    """Binary cross-entropy on fault PRESENCE, for the diagnosis phase.

    THREE PROBLEMS THIS FIXES, all of them mine.

    1. Diagnosis and severity were training the SAME head with the SAME loss.
       PHASE_HEADS maps both to y_fault_mode and select_head gave both the
       severity-weighted MSE, so "diagnosis" was literally the severity run
       done twice. They are different questions: diagnosis asks WHICH component
       is failing, severity asks HOW FAR gone it is.

    2. The metric and the loss disagreed about what positive means. Labels count
       as positive at severity >= 0.08, but MacroF1 thresholds predictions at
       0.5 - and only 71% of positive labels exceed 0.5. A perfect severity
       regressor was therefore capped at 0.71 recall before training began.
       Against presence targets of exactly 0 or 1, a 0.5 threshold is the right
       one and the two agree.

    3. Positives are 2.6% of cells across 14 modes, about 37:1. Unweighted BCE
       is minimised by predicting near-zero everywhere, which is what a macro F1
       of 0.129 looks like. pos_weight is set below that 37:1 ratio on purpose -
       weighting to the full imbalance buys recall by firing constantly, and the
       point of this head is to name a component, not to raise an alarm.
    """
    def loss(y_true, y_pred):
        t = tf.cast(tf.cast(y_true, tf.float32) >= threshold, tf.float32)
        eps = 1e-7
        p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        return -tf.reduce_mean(pos_weight * t * tf.math.log(p)
                               + (1.0 - t) * tf.math.log(1.0 - p))
    return loss


def severity_weighted_mse(y_true, y_pred):
    """MSE that does not let the zeros drown the signal.

    Plain MSE is minimised by predicting zero everywhere, which scores well
    and diagnoses nothing. Non-zero targets are weighted up so the head has a
    reason to resolve severity where severity exists.

    WEIGHT 10 -> 40. The 10 was set when ~70% of targets were zero. On the new
    data 97.6% are (about 40:1), so at 10 the zeros still carried 4x the total
    weight and pulled every uncertain window toward "no severity". At 40 the
    two sides are balanced: a window that is 50% likely to carry a fault of
    severity s is pulled to ~0.98 s rather than ~0.9 s, and a cell with no
    fault still sits at zero.
    """
    t = tf.cast(y_true, tf.float32)
    w = 1.0 + 39.0 * tf.cast(t > 0.0, tf.float32)
    return tf.reduce_mean(w * tf.square(t - y_pred))


def rul_loss(tbo_hours: float):
    """Huber in TBO-fraction space.

    Squared error in hours lets a handful of near-end-of-life windows dominate
    the gradient, and those are exactly the windows one unlucky scenario can
    distort. Working in fraction-of-life also makes the loss comparable across
    engines with different TBOs.
    """
    # DELTA 0.05 -> 0.25. Huber is quadratic below delta and linear above, and
    # 0.05 of TBO is 100 h on the 914. Errors on the wear-limited windows - the
    # only ones that carry prognostic content - run 200-400 h, so every one of
    # them sat in the LINEAR region, where the gradient is a pure sign with no
    # magnitude information. The head could find the coarse linear map from
    # engine_hours_norm and then had nothing to refine with: it scored 19.89% of
    # TBO on that subset against 19.76% for predicting a constant.
    #
    # At 0.25 (500 h) those errors are inside the quadratic region and the
    # gradient scales with how wrong the prediction is, while the tail beyond
    # 500 h is still clipped - which was the original reason for Huber over MSE.
    h = keras.losses.Huber(delta=0.25)

    def loss(y_true, y_pred):
        return h(tf.cast(y_true, tf.float32) / tbo_hours,
                 tf.squeeze(y_pred, -1) / tbo_hours)
    return loss


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def warm_start(model: keras.Model, prev_path: str) -> bool:
    """Copy encoder weights from the previous phase.

    Each phase reuses the representation the previous one learned, which is why
    the chain runs detection -> diagnosis -> severity/health -> rul rather than
    training six heads from scratch. Only layers whose names and shapes match
    are copied, so a head that does not exist in the earlier model is skipped
    rather than erroring.
    """
    if not prev_path or not os.path.exists(prev_path):
        return False
    prev = keras.models.load_model(prev_path, compile=False, safe_mode=False)
    by_name = {l.name: l for l in prev.layers}
    copied = 0
    for layer in model.layers:
        src = by_name.get(layer.name)
        if src is None or not layer.weights:
            continue
        if [w.shape for w in src.weights] == [w.shape for w in layer.weights]:
            layer.set_weights(src.get_weights())
            copied += 1
    return copied > 0


def callbacks(path: str, monitor: str = "val_loss", mode: str = "min",
              patience: int = 5, lr_patience: int = 2) -> list:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(path, monitor=monitor, mode=mode,
                                        save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor=monitor, mode=mode,
                                      patience=patience, restore_best_weights=True,
                                      verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode,
                                          factor=0.5, patience=lr_patience,
                                          min_lr=1e-5, verbose=1),
        keras.callbacks.TerminateOnNaN(),
    ]


# Per-phase learning rate. Only RUL differs, and it differs because its target
# is a scenario-level scalar: ~1,186 independent examples, so a step size tuned
# for per-window supervision overshoots. At 1e-3 the validation metric went
# 7.02 -> 13.06 -> 23.40 -> 9.17 across four epochs and early stopping restored
# epoch 1, i.e. the head that had trained least.
PHASE_LR = {"rul": 2e-4}


class WearLimitedVal(keras.callbacks.Callback):
    """Per-epoch validation MAE on the wear-limited windows only.

    WHY A CALLBACK AND NOT A METRIC. A Keras metric receives (y_true, y_pred)
    and nothing else, but deciding whether a window is wear-limited needs
    engine_hours_norm, which lives in the AUX input. So this cannot be a metric
    and has to run its own pass over validation.

    WHY IT MATTERS. rul_hours_true is min(TBO, wear_out) - engine_hours. On the
    914 data ~78% of windows are TBO-limited, where the answer is one
    subtraction of an input the head is handed. Selecting the checkpoint on the
    aggregate therefore selects mostly on the easy part: the previous run picked
    epoch 3 on aggregate 7.39%, while four other epochs sat in the same
    7.4-7.9 band and their wear-limited errors were never looked at.

    Writes `val_wear_mae_pct` into logs, so EarlyStopping, ModelCheckpoint and
    ReduceLROnPlateau can all select on the part that carries prognostic
    content. It is registered FIRST in the callback list, because Keras runs
    callbacks in order and the others read logs this one writes.
    """

    def __init__(self, val_ds, val_steps: int, aux_scaler, tbo_hours: float,
                 wear_gate: float = 15.0, all_gate: float = 8.0):
        super().__init__()
        self.wear_gate = float(wear_gate)
        self.all_gate = float(all_gate)
        self.ds = val_ds
        self.steps = val_steps
        self.ax = aux_scaler
        self.tbo = float(tbo_hours)
        self.k = P.RUL_AUX_ORDER.index("engine_hours_norm")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        Y, T, H = [], [], []
        for i, (xb, yb) in enumerate(self.ds):
            if i >= self.steps:
                break
            Y.append(self.model.predict(xb, verbose=0)["y_rul_hours"].ravel())
            T.append(yb["y_rul_hours"].numpy().ravel())
            a = xb["x_rul_aux"].numpy().astype(np.float64)
            if self.ax is not None:
                a = self.ax.inverse_transform(a)
            H.append(a[:, self.k])
        y, t, h = (np.concatenate(v) for v in (Y, T, H))
        worn = t < self.tbo * (1.0 - h) - 1.0
        wear = (np.abs(y[worn] - t[worn]).mean() * 100.0 / self.tbo
                if worn.any() else float("nan"))
        allm = np.abs(y - t).mean() * 100.0 / self.tbo
        logs["val_wear_mae_pct"] = float(wear)
        logs["val_rul_mae_pct"] = float(allm)
        # WORST NORMALISED GATE VIOLATION, which is what selection should
        # minimise. Monitoring wear-limited alone swung the other way: one run
        # picked epoch 8 (wear 14.12%, all 12.61%) over epoch 11 (wear 14.17%,
        # all 8.51%) - 0.05 better on the monitored metric and 4.1 worse on the
        # one it ignored. Each error is divided by its own gate, so the score is
        # below 1.0 exactly when BOTH gates would pass, and neither can be
        # traded away for the other.
        ratio = max(wear / self.wear_gate, allm / self.all_gate)
        logs["val_gate_ratio"] = float(ratio)
        print(f"    wear-limited val: {wear:6.2f}%  (all {allm:5.2f}%, "
              f"{100 * worn.mean():.1f}% of windows)  gate ratio {ratio:5.3f}",
              flush=True)


def select_head(model: keras.Model, head: str, tbo_hours: float,
                lr: float | None = None, phase: str | None = None) -> keras.Model:
    """Compile the full model to train ONE head, freezing nothing.

    The other heads stay in the graph with zero loss weight so their weights are
    carried forward untouched and the next phase can warm-start from this
    checkpoint without shape surprises.
    """
    import model_architectures as A

    losses = dict(A.LOSSES)
    # The same head serves two phases with different questions, so the loss is
    # chosen by PHASE rather than by head.
    if head == "y_fault_mode":
        losses["y_fault_mode"] = (presence_bce() if phase == "diagnosis"
                                  else severity_weighted_mse)
    losses["y_sensor_fault"] = sensor_weighted_ce()
    losses["y_rul_hours"] = rul_loss(tbo_hours)

    weights = {k: 0.0 for k in A.LOSS_WEIGHTS}
    weights[head] = 1.0

    # FREEZE THE ENCODER FOR RUL. Every other head is supervised per window, so
    # 60,691 training windows are 60,691 training signals. RUL is not: it is a
    # SCENARIO-level scalar. Measured on the 914 test split the label varies by
    # 0.53 h within a flight and 649 h between flights, so the effective sample
    # size for this head is ~1,186 scenarios - against 248k encoder parameters.
    #
    # Letting the RUL gradient into the trunk did visible damage. Across the
    # seven epochs of the 914 run the frozen-out heads' losses exploded while
    # RUL itself barely moved:
    #     y_detection_loss      2.53 -> 57.27
    #     y_sensor_fault_loss   8.48 -> 231.54
    #     val_y_rul_hours_mae    601 -> 590
    # RUL runs last so nothing warm-starts from it, but the saved checkpoint is
    # the one that would be deployed, and its encoder was wrecked. The head has
    # its own 32-unit aux path and reads a 192-d encoding the earlier phases
    # already learned; it does not need to relearn the trunk.
    if phase == "rul":
        for layer in model.layers:
            if layer.name.startswith("tcn"):
                layer.trainable = False

    metrics = {
        "y_detection": [keras.metrics.AUC(name="auc"),
                        keras.metrics.Precision(name="prec"),
                        keras.metrics.Recall(name="rec")],
        # NO keras AUC here, deliberately. keras.metrics.AUC(multi_label=True)
        # scores a label with zero positives as 0.0 and averages it in, so on a
        # 14-mode head where only some modes appear in an evaluation batch it
        # reports a number that is not the AUC of anything. Measured on the 914
        # diagnosis checkpoint: it reported 0.4129 where the true macro AUC over
        # modes that actually occur was 0.8184 - 0.8184 x 7/14 = 0.409, which is
        # exactly the arithmetic it was doing. A metric that wrong is worse than
        # no metric, so the real value is computed with sklearn in
        # audit_diagnosis.py and in the diagnosis notebook.
        "y_fault_mode": [MacroF1(P.N_FAULT_MODES),
                         # The severity phase is gated on "mae" for this head.
                         # Without this the gate looks up a metric that was
                         # never registered and fails on a missing key rather
                         # than on the model.
                         keras.metrics.MeanAbsoluteError(name="mae"),
                         FaultMae()],
        "y_sensor_fault": [keras.metrics.SparseCategoricalAccuracy(name="acc"),
                           SensorMacroF1()],
        "y_sensor_sev": [keras.metrics.MeanAbsoluteError(name="mae")],
        "y_health": [keras.metrics.MeanAbsoluteError(name="mae")],
        "y_rul_hours": [keras.metrics.MeanAbsoluteError(name="mae"),
                        MaePctTbo(tbo_hours)],
    }
    if lr is None:
        lr = PHASE_LR.get(phase or "", 1e-3)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=losses,
                  loss_weights=weights, metrics={head: metrics[head]})
    return model


def save_result(engine: str, phase: str, payload: dict) -> str:
    path = result_path(engine, phase)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


def report_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    return f"{len(gpus)} GPU(s): {[g.name for g in gpus]}" if gpus else "CPU only"
