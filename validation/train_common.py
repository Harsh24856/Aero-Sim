"""Shared setup for the v3 training notebooks.

Exists so the five phase notebooks stay short and readable. v2 had 22
near-duplicate notebooks and the per-engine copies drifted apart - different
epoch counts, a step-count formula that was wrong in all of them, and one engine
whose generator was a different script entirely. One module, one set of
constants, engine passed as a variable.

RUN EVERYTHING WITH validation/venv/bin/python3 (or a Jupyter kernel pointing at
it). backend/.venv has no working TensorFlow and will segfault confusingly.
"""
import json
import os
import time

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

import tf_data_pipeline as P
import model_architectures as A

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(VALIDATION_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ENGINES = ["Rotax_912_ULS", "Rotax_914_ULF", "Rotax_915_iS", "Rotax_916_iS"]

# Short key used in filenames, matching backend/models/<key>/ layout.
ENGINE_KEY = {
    "Rotax_912_ULS": "912",
    "Rotax_914_ULF": "914",
    "Rotax_915_iS": "915",
    "Rotax_916_iS": "916",
}

BATCH_SIZE = 128


def data_dir(engine):
    return os.path.join(VALIDATION_DIR, f"chunks_v3_{engine.lower()}")


def index_path(engine):
    return os.path.join(data_dir(engine), "scenario_index.json")


def scaler_path(engine):
    return os.path.join(MODELS_DIR, f"scaler_{ENGINE_KEY[engine]}.pkl")


def ckpt_path(engine, phase):
    return os.path.join(MODELS_DIR, f"{phase}_{ENGINE_KEY[engine]}.keras")


def datasets(engine, batch_size=BATCH_SIZE, shuffle_buffer=2000):
    """train/val/test tf.data.Datasets for one engine."""
    ip, dd, sp = index_path(engine), data_dir(engine), scaler_path(engine)
    return (
        P.make_dataset(ip, dd, "train", sp, batch_size=batch_size, shuffle_buffer=shuffle_buffer),
        P.make_dataset(ip, dd, "val", sp, batch_size=batch_size, shuffle_buffer=0),
        P.make_dataset(ip, dd, "test", sp, batch_size=batch_size, shuffle_buffer=0),
    )


def step_counts(engine, batch_size=BATCH_SIZE):
    """TRUE steps per epoch for each split.

    v2's notebooks used rows/batch_size, which counts ROWS not WINDOWS, so an
    "epoch" was never a full pass. At 10M rows that formula asks for 78,125
    steps - roughly 65 hours per epoch. P.steps_per_epoch counts windows.
    """
    ip, dd = index_path(engine), data_dir(engine)
    return {s: P.steps_per_epoch(ip, dd, s, batch_size=batch_size)
            for s in ("train", "val", "test")}


def describe(engine):
    idx = json.load(open(index_path(engine)))
    sc = step_counts(engine)
    print(f"{engine}  ({ENGINE_KEY[engine]})")
    print(f"  physics   : {idx.get('physics_version')}   TBO {idx.get('tbo_hours')} h")
    print(f"  rows      : {idx.get('rows_total'):,} in {idx.get('n_scenarios_total')} scenarios")
    print(f"  steps/epoch: train {sc['train']:,}  val {sc['val']:,}  test {sc['test']:,}")
    print(f"  window {P.WINDOW_SIZE}  stride {P.STRIDE}  features {P.N_FEATURES}  aux {P.N_RUL_AUX}")
    return sc


# ---------------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------------
def weighted_diagnosis_loss(class_weights):
    """Categorical CE over (batch, 8 channels, 6 classes), weighted by class.

    Inverse-SQRT frequency, not inverse frequency: 'none' dominates so heavily
    that plain inverse weighting drove the head to predict faults everywhere.
    """
    cw = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=A.N_DIAG_CLASSES)
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)), axis=-1)
        w = tf.reduce_sum(y_true_oh * cw, axis=-1)
        return tf.reduce_mean(ce * w)
    return loss


def diagnosis_class_weights(engine, cap=10.0):
    """Inverse-sqrt class frequencies, computed from the TRAIN split only."""
    ip, dd, sp = index_path(engine), data_dir(engine), scaler_path(engine)
    counts = np.zeros(A.N_DIAG_CLASSES, dtype=np.float64)
    gen = P.scenario_window_generator(ip, dd, "train", sp)
    for i, (_, labels) in enumerate(gen):
        vals, c = np.unique(labels["y_diagnosis"].astype(int), return_counts=True)
        counts[vals] += c
        if i >= 20000:      # enough for a stable estimate; the full pass is slow
            break
    freq = counts / max(counts.sum(), 1.0)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    w = np.minimum(w / w.min(), cap)
    return w.astype(np.float32)


def nonzero_severity_mse(y_true, y_pred):
    """MSE over channels that actually carry stress.

    Plain MSE is dominated by the zeros and looks excellent while the head
    predicts nothing.
    """
    mask = tf.cast(y_true > 0.0, tf.float32)
    se = tf.square(y_true - y_pred) * mask
    return tf.reduce_sum(se) / tf.maximum(tf.reduce_sum(mask), 1.0)


# ---------------------------------------------------------------------------
# Metrics that are not misleading
#
# Every head here has a trivial predictor that scores well on the obvious metric.
# Measured on 3.0M rows of v3 914 data:
#
#   detection      42.1% positive        -> always-majority = 57.9% acc
#   diagnosis      87.4% class 'none'    -> always-'none'   = 87.4% acc
#   severity        9.9% cells nonzero   -> all-zeros       = 0.095 plain MSE
#   failure modes   8-17% present        -> all-zeros       = 0.012-0.032 MAE
#
# So accuracy is useless for diagnosis, plain MSE is useless for severity, and
# MAE is useless for failure modes. These are the metrics that actually move
# when the head learns something.
# ---------------------------------------------------------------------------
class DiagnosisMacroF1(keras.metrics.Metric):
    """Macro-F1 over the FAULT classes only, ignoring 'none'.

    y_true is (batch, 8) class indices; y_pred is (batch, 8, 6) softmax.
    'none' is 87.4% of cells, so including it would swamp the average and a
    do-nothing head would score ~0.93. Excluding it, a do-nothing head scores 0.
    """

    def __init__(self, n_classes=None, ignore_class=0, name="macro_f1", **kw):
        super().__init__(name=name, **kw)
        self.n_classes = n_classes or A.N_DIAG_CLASSES
        self.ignore_class = ignore_class
        # Static keep-mask, built once in numpy. Building it at result() time with
        # tf.one_hot(...) < 0.5 silently produced an all-False mask, so every class
        # was dropped and the metric returned 0/0 = 0.0 even for a perfect head.
        _keep = np.ones(self.n_classes, dtype=np.float32)
        _keep[self.ignore_class] = 0.0
        self._keep = tf.constant(_keep)
        self.tp = self.add_weight(shape=(self.n_classes,), initializer="zeros", name="tp")
        self.fp = self.add_weight(shape=(self.n_classes,), initializer="zeros", name="fp")
        self.fn = self.add_weight(shape=(self.n_classes,), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        yp = tf.argmax(tf.reshape(y_pred, [-1, self.n_classes]), axis=-1, output_type=tf.int32)
        t = tf.one_hot(yt, self.n_classes)
        p = tf.one_hot(yp, self.n_classes)
        self.tp.assign_add(tf.reduce_sum(t * p, axis=0))
        self.fp.assign_add(tf.reduce_sum((1.0 - t) * p, axis=0))
        self.fn.assign_add(tf.reduce_sum(t * (1.0 - p), axis=0))

    def result(self):
        f1 = 2.0 * self.tp / tf.maximum(2.0 * self.tp + self.fp + self.fn, 1e-7)
        return tf.reduce_sum(f1 * self._keep) / tf.maximum(tf.reduce_sum(self._keep), 1.0)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros_like(v))


class DiagnosisFaultRecall(keras.metrics.Metric):
    """Of cells that ARE faulted, what fraction did we flag as faulted at all.

    The single number that exposes a head collapsing to 'none' everywhere.
    """

    def __init__(self, name="fault_recall", **kw):
        super().__init__(name=name, **kw)
        self.hit = self.add_weight(initializer="zeros", name="hit")
        self.tot = self.add_weight(initializer="zeros", name="tot")

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        yp = tf.argmax(tf.reshape(y_pred, [-1, A.N_DIAG_CLASSES]), axis=-1, output_type=tf.int32)
        faulted = tf.cast(yt != 0, tf.float32)
        self.hit.assign_add(tf.reduce_sum(faulted * tf.cast(yp != 0, tf.float32)))
        self.tot.assign_add(tf.reduce_sum(faulted))

    def result(self):
        return self.hit / tf.maximum(self.tot, 1.0)

    def reset_state(self):
        self.hit.assign(0.0); self.tot.assign(0.0)


def nonzero_severity_mae(y_true, y_pred):
    """MAE over channels that actually carry stress. See nonzero_severity_mse."""
    mask = tf.cast(y_true > 0.0, tf.float32)
    return tf.reduce_sum(tf.abs(y_true - y_pred) * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


class ModePresenceAUC(keras.metrics.AUC):
    """AUC for 'is this failure mode present', per mode.

    The head regresses a 0..1 severity, but each mode is present in only 6-15%
    of windows, so MAE is ~0.02 for a head that always predicts zero. Binarising
    the target and scoring the predicted severity as a ranking gives a number
    that cannot be gamed that way.

    min_severity > 0 drops windows with 0 < severity < min_severity from the AUC
    (weight 0). Those are faults that have only just started: on 916 test data
    57% of cooling-degradation positives are below 0.1 and move CHT by ~1-2 C.
    Grading on them measures label noise, not the model. The threshold is the same
    DETECTION_THRESHOLD the phase-1 labels already use. The strict (>0) AUC is
    still computed and reported next to it.
    """

    def __init__(self, mode_index, name, min_severity=0.0, **kw):
        super().__init__(name=name, **kw)
        self.mode_index = mode_index
        self.min_severity = float(min_severity)

    def update_state(self, y_true, y_pred, sample_weight=None):
        sev = y_true[:, self.mode_index]
        yt = tf.cast(sev > 0.0, tf.float32)
        yp = y_pred[:, self.mode_index]
        w = None
        if self.min_severity > 0.0:
            w = tf.cast((sev <= 0.0) | (sev >= self.min_severity), tf.float32)
        return super().update_state(yt, yp, w)


class AllModesPresenceAUC(keras.metrics.AUC):
    """One aggregate presence-AUC across all four modes, pooled.

    Exists so ModelCheckpoint has a single number to monitor - per-mode AUCs are
    for reading, this is for model selection. min_severity as in ModePresenceAUC.
    """

    def __init__(self, name="auc_all", min_severity=0.0, **kw):
        super().__init__(name=name, **kw)
        self.min_severity = float(min_severity)

    def update_state(self, y_true, y_pred, sample_weight=None):
        sev = tf.reshape(y_true, [-1])
        yt = tf.cast(sev > 0.0, tf.float32)
        yp = tf.reshape(y_pred, [-1])
        w = None
        if self.min_severity > 0.0:
            w = tf.cast((sev <= 0.0) | (sev >= self.min_severity), tf.float32)
        return super().update_state(yt, yp, w)


def failure_mode_metrics():
    """Strict (>0) and detectable (>= DETECTION_THRESHOLD) presence-AUCs.

    auc_all_det is what the gate and ModelCheckpoint use; auc_all (strict) is kept
    so the harder number is never hidden.
    """
    thr = P.DETECTION_THRESHOLD
    return ([AllModesPresenceAUC(),
             AllModesPresenceAUC(name="auc_all_det", min_severity=thr)]
            + [ModePresenceAUC(i, name=f"auc_{m}") for i, m in enumerate(P.FAILURE_MODES)]
            + [ModePresenceAUC(i, name=f"auc_det_{m}", min_severity=thr)
               for i, m in enumerate(P.FAILURE_MODES)])


def mae_pct_tbo(tbo_hours):
    """RUL MAE as a percentage of TBO.

    Absolute hours are not comparable across engines - 100 h is 5% of a 2000 h
    TBO but 8.3% of a 1200 h one. This is the number to judge RUL on.
    """
    def metric(y_true, y_pred):
        # Flatten both: y_true arrives as (B,) and y_pred as (B,1), and without this
        # they broadcast to (B,B) - the mean of every pairwise difference, not MAE.
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        return 100.0 * tf.reduce_mean(tf.abs(y_true - y_pred)) / tbo_hours
    metric.__name__ = "mae_pct_tbo"
    return metric


def rul_loss(tbo_hours, delta_pct=1.0):
    """Huber on RUL measured in PERCENT OF TBO (delta 1% = 20 h on a 2000 h TBO).

    Keras Huber on raw hours has delta=1 h, so it is just MAE and prints 500+;
    in %-of-TBO the loss reads on the same scale as mae_pct_tbo.
    """
    h = keras.losses.Huber(delta=delta_pct)
    def loss(y_true, y_pred):
        return h(100.0 * y_true / tbo_hours, 100.0 * y_pred / tbo_hours)
    loss.__name__ = "rul_huber_pct_tbo"
    return loss


def tbo_of(engine):
    return float(json.load(open(index_path(engine)))["tbo_hours"])


# ---------------------------------------------------------------------------
# Warm-start + checkpoints
# ---------------------------------------------------------------------------
def warm_start(model, prev_path):
    """Copy weights from a previous phase BY LAYER NAME.

    safe_mode=False is required: build_encoder uses a Lambda.
    """
    if not os.path.exists(prev_path):
        print(f"  no previous checkpoint at {prev_path} - training from scratch")
        return 0
    prev = keras.models.load_model(prev_path, safe_mode=False, compile=False)
    by_name = {l.name: l for l in prev.layers}
    n = 0
    for layer in model.layers:
        src = by_name.get(layer.name)
        if src is not None and layer.weights and len(src.weights) == len(layer.weights):
            try:
                layer.set_weights(src.get_weights())
                n += 1
            except ValueError:
                pass          # shape changed between phases; leave it initialised
    print(f"  warm-started {n} layers from {os.path.basename(prev_path)}")
    return n


def callbacks(path, monitor="val_loss", mode="min", patience=5, lr_patience=2, min_lr=0.0):
    """Default patience raised 3 -> 5.

    The first real phase-1 run improved val_loss on epoch 10 of 10 - EarlyStopping
    never fired and the epoch cap, not convergence, ended training. Patience must
    outlast the plateaus that ReduceLROnPlateau is there to break through.
    """
    # ModelCheckpoint starts from best=None, so epoch 1 of a rerun silently
    # overwrites a good model. Keep the previous one next to it.
    if os.path.exists(path):
        import shutil
        bak = path.replace(".keras", ".prev.keras")
        shutil.copy2(path, bak)
        print(f"  existing checkpoint backed up -> {bak}")
    return [
        keras.callbacks.ModelCheckpoint(path, monitor=monitor, mode=mode,
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience,
                                      restore_best_weights=True, verbose=1),
        # lr_patience 2 is too eager for RUL: on 915 one noisy val epoch at 27.5 h cut
        # the rate at epoch 13 and it decayed to 4e-7, stuck at 25 h - the original run
        # (no working LR callback, 1e-4 throughout) was still improving at 15 h.
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.5,
                                          patience=lr_patience, min_lr=min_lr, verbose=1),
    ]


def split_labels(keep, needs_aux=False, tbo_hours=None, near_new_weight=1.0, near_new_from=0.90):
    """tf.data yields ALL inputs and ALL labels; each phase uses a subset.

    Two things this has to get right, both of which bit during the dry run:

    1. The label dict keys must match the model's OUTPUT names. A model built as
       keras.Model(xi, tensor) has no output name, and Keras then tries to coerce
       the dict key itself into a tensor - "Expected float32, but got
       y_detection of type 'str'". So every phase model is built with a dict
       output, and `keep` names those outputs.
    2. Phases that do not use x_rul_aux must be handed the bare `x` tensor, not
       the full input dict, or Keras warns that the input structure does not
       match on every single batch.

    Also emits sample_weight for the RUL output so an unlabelled window
    contributes zero RUL loss while still training the other heads.

    With tbo_hours and near_new_weight > 1, RUL windows on nearly-new engines are
    up-weighted: a linear ramp from 1x at near_new_from*TBO to near_new_weight x at a
    new engine. Only every fifth scenario starts new and wear moves out of that band
    within minutes, so those windows are ~1% of the data - and live flights all start
    there. Unweighted, the head under-predicted them by 37-109 h.
    """
    def fn(x, y):
        xx = {"x": x["x"], "x_rul_aux": x["x_rul_aux"]} if needs_aux else x["x"]
        out = {k: y[k] for k in keep}
        if "y_rul_hours" in keep:
            w = y["rul_valid"]
            if tbo_hours and near_new_weight > 1.0:
                frac = tf.clip_by_value((y["y_rul_hours"] / tbo_hours - near_new_from) / (1.0 - near_new_from), 0.0, 1.0)
                w = w * (1.0 + (near_new_weight - 1.0) * frac)
            return xx, out, {"y_rul_hours": w}
        return xx, out
    return fn


def report_gpu():
    devs = [d.device_type for d in tf.config.list_physical_devices()]
    print(f"  TF {tf.__version__}  devices {devs}")
    if "GPU" not in devs:
        print("  WARNING: no GPU. Check tensorflow-metal is installed in validation/venv.")
    return devs
