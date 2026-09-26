"""Model architectures for v4.

The encoder is deliberately unchanged from validation/model_architectures.py:
the same TCN with dilated causal convolutions, residual connections, and the
three-view pooling (last / mean / max) that v3 introduced. It performed well and
the plan is explicit that the model should not change for novelty. What changes
is what it consumes and what it predicts.

HEADS, AND WHY EACH EXISTS
    detection     one number: is anything wrong
    fault_mode    14 component faults, MULTI-LABEL rather than multi-class,
                  because two faults co-occur in roughly 8% of scenarios and a
                  softmax would force the head to pick one
    sensor_fault  per-channel instrument fault type, kept SEPARATE from the
                  engine faults so a drifting thermocouple and a real overheat
                  are distinguishable - the v3 design conflated them
    sensor_sev    per-channel instrument fault severity
    health        wear condition, monotone over life
    rul           remaining hours, with the auxiliary scalars fed in beside the
                  encoded window

The heads share one encoder. Training them jointly is what lets the detection
signal regularise the diagnosis head, which matters because individual fault
modes are each only a few percent of rows.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from tf_data_pipeline import (
    WINDOW_SIZE, N_FEATURES, N_FAULT_MODES, N_SENSOR_CHANNELS,
    N_SENSOR_FAULT_TYPES, N_RUL_AUX, FEATURE_COLS, MEASURED_COLS,
)

SENSOR_IDX = [FEATURE_COLS.index(c) for c in MEASURED_COLS]


# ---------------------------------------------------------------------------
# Serializable Lambda bodies
#
# WHY THESE ARE MODULE-LEVEL AND REGISTERED, AND NOT CLOSURES.
# Keras serializes a plain `lambda` BY VALUE - it marshals the bytecode - which
# is why the older Lambdas in this file reload under safe_mode=False. A function
# defined with `def` inside another function is serialized BY NAME instead, and
# that name cannot be resolved at load time:
#     TypeError: Could not locate function '_scale_by_calendar'
# That surfaced only in warm_start(), which calls keras.models.load_model on the
# PREVIOUS phase's checkpoint. The failure was silent in the worst way: run.py
# caught it per phase, logged "warm start from diagnosis: no checkpoint", and
# the severity, sensor_fault and health phases for the 912 went on to train from
# scratch and report passing gates on a chain that had been broken.
#
# Registering by name and passing the constants through Lambda(arguments=...)
# puts everything the layer needs into the config, so the checkpoint round-trips.
@keras.saving.register_keras_serializable(package="uav_v4")
def rul_ratio_activation(t):
    """Bounded ratio in [0, 1], reachable at finite logits."""
    return tf.clip_by_value(1.1 * tf.sigmoid(t) - 0.05, 0.0, 1.0)


@keras.saving.register_keras_serializable(package="uav_v4")
def scale_by_calendar(args, cal_idx, cal_mean, cal_scale, tbo_hours):
    """RUL = (calendar RUL) x (predicted ratio), with the aux scaler undone."""
    a, r = args
    ehn = a[:, cal_idx:cal_idx + 1] * cal_scale + cal_mean
    cal = tf.clip_by_value(1.0 - ehn, 0.0, 1.0)
    return tbo_hours * cal * r


@keras.saving.register_keras_serializable(package="uav_v4")
def sensor_channel_stats(x, idx):
    """Per-channel shape of each measured signal across the window.

    Each statistic is the signature of one instrument fault: spread (noise),
    mean step-to-step change (~0 when stuck), largest jump (spike), minimum
    (dropout), and last-minus-mean (drift). Returns (batch, 12 * 5).
    """
    s = tf.gather(x, idx, axis=-1)                       # (B, T, 12)
    d = tf.abs(s[:, 1:, :] - s[:, :-1, :])
    feats = [tf.math.reduce_std(s, axis=1), tf.reduce_mean(d, axis=1),
             tf.reduce_max(d, axis=1), tf.reduce_min(s, axis=1),
             s[:, -1, :] - tf.reduce_mean(s, axis=1)]
    return tf.concat(feats, axis=-1)


# ---------------------------------------------------------------------------
# Encoder - unchanged from v3
# ---------------------------------------------------------------------------
def tcn_block(x, channels, kernel_size, dilation, dropout, name_prefix):
    residual = x
    h = layers.Conv1D(channels, kernel_size, padding="causal", dilation_rate=dilation,
                      activation="relu", name=f"{name_prefix}_conv1")(x)
    h = layers.Dropout(dropout, name=f"{name_prefix}_drop1")(h)
    h = layers.Conv1D(channels, kernel_size, padding="causal", dilation_rate=dilation,
                      activation="relu", name=f"{name_prefix}_conv2")(h)
    h = layers.Dropout(dropout, name=f"{name_prefix}_drop2")(h)
    if residual.shape[-1] != channels:
        residual = layers.Conv1D(channels, 1, padding="same",
                                 name=f"{name_prefix}_proj")(residual)
    out = layers.Add(name=f"{name_prefix}_add")([h, residual])
    return layers.ReLU(name=f"{name_prefix}_relu")(out)


def build_encoder_input():
    return layers.Input(shape=(WINDOW_SIZE, N_FEATURES), name="x")


def build_aux_input():
    return layers.Input(shape=(N_RUL_AUX,), name="x_rul_aux")


def build_encoder(x_input, channels=64, num_layers=6, kernel_size=3, dropout=0.25):
    """Dilated causal TCN with three-view pooling.

    Channels 48 -> 64 because the feature vector grew from 25 to 29 and six of
    the new ones are residuals carrying most of the fault signal. The receptive
    field at six layers of dilation 1..32 is 127 samples, which covers the
    128-sample window exactly.

    DROPOUT 0.1 -> 0.25. The effective sample size is the number of SCENARIOS,
    not the number of windows: at stride 128 the windows do not overlap, but
    the ~50 windows cut from one flight share an engine, a fault and an
    operating profile, so 60,087 windows carry roughly 1,186 independent
    examples against 248k parameters. Detection showed it - validation loss
    bottomed at epoch 1 and rose every epoch after, while train AUC climbed to
    0.970 against a validation AUC stuck near 0.77. Early stopping then restored
    epoch 1, so the evaluated model had barely trained.
    """
    x = x_input
    for i in range(num_layers):
        x = tcn_block(x, channels, kernel_size, dilation=2 ** i,
                      dropout=dropout, name_prefix=f"tcn{i}")
    last = layers.Lambda(lambda t: t[:, -1, :], name="enc_last")(x)
    avg = layers.GlobalAveragePooling1D(name="enc_avg")(x)
    mx = layers.GlobalMaxPooling1D(name="enc_max")(x)
    return layers.Concatenate(name="enc_out")([last, avg, mx])


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------
def build_detection_head(enc, hidden=48, dropout=0.1):
    h = layers.Dense(hidden, activation="relu", name="det_d1")(enc)
    h = layers.Dropout(dropout, name="det_drop")(h)
    return layers.Dense(1, activation="sigmoid", name="y_detection")(h)


def build_fault_mode_head(enc, hidden=96, dropout=0.1):
    """14 component faults, multi-label.

    Sigmoid per mode rather than a softmax: faults co-occur, and forcing the
    head to choose one would make it wrong whenever two are present.
    """
    h = layers.Dense(hidden, activation="relu", name="fm_d1")(enc)
    h = layers.Dropout(dropout, name="fm_drop")(h)
    return layers.Dense(N_FAULT_MODES, activation="sigmoid", name="y_fault_mode")(h)


def build_sensor_fault_head(enc, x_input, hidden=96, dropout=0.1):
    """Per-channel instrument fault type: 12 channels x 7 classes.

    READS THE RAW WINDOW'S PER-CHANNEL STATISTICS BESIDE THE ENCODING. On the
    first 914 run, reading the pooled encoding alone, faulty sensors were
    called "none" 53-98% of the time and noise and drift were never caught
    (macro F1 0.275). The encoder's pooling keeps last/mean/max of learned
    features, which is what engine faults need, and loses the within-window
    spread, jumps and flat-lining that tell a broken instrument apart.
    """
    st = layers.Lambda(sensor_channel_stats, arguments={"idx": SENSOR_IDX},
                       name="sf_stats")(x_input)
    st = layers.Dense(64, activation="relu", name="sf_stats_d1")(st)
    h = layers.Concatenate(name="sf_in")([enc, st])
    h = layers.Dense(hidden, activation="relu", name="sf_d1")(h)
    h = layers.Dropout(dropout, name="sf_drop")(h)
    flat = layers.Dense(N_SENSOR_CHANNELS * N_SENSOR_FAULT_TYPES, name="sf_flat")(h)
    resh = layers.Reshape((N_SENSOR_CHANNELS, N_SENSOR_FAULT_TYPES), name="sf_reshape")(flat)
    return layers.Softmax(axis=-1, name="y_sensor_fault")(resh)


def build_sensor_sev_head(enc, hidden=64, dropout=0.1):
    h = layers.Dense(hidden, activation="relu", name="ss_d1")(enc)
    h = layers.Dropout(dropout, name="ss_drop")(h)
    return layers.Dense(N_SENSOR_CHANNELS, activation="sigmoid", name="y_sensor_sev")(h)


def build_health_head(enc, hidden=48, dropout=0.1):
    h = layers.Dense(hidden, activation="relu", name="hl_d1")(enc)
    h = layers.Dropout(dropout, name="hl_drop")(h)
    return layers.Dense(1, activation="sigmoid", name="y_health")(h)


def build_rul_head(enc, aux_input, tbo_hours, hidden=96, dropout=0.1,
                   cal_idx=None, cal_mean=None, cal_scale=None):
    """Remaining hours.

    The auxiliary scalars join AFTER the encoder rather than being appended to
    every timestep, because they are window-level facts (how far through life,
    current condition) and repeating them 128 times would let the convolutions
    treat them as a signal that varies within the window when it does not.

    The output is scaled by TBO, so the head predicts a FRACTION of overhaul
    life and the absolute hours follow. Without this the head has to learn the
    scale of the target as well as its shape, and engines with different TBOs
    (the 915 is 1200 h against 2000 h for the rest) cannot share a design.
    """
    # THE ENCODER IS DELIBERATELY NOT CONNECTED TO THIS HEAD.
    #
    # `enc` stays in the signature so the model builds uniformly, but RUL reads
    # the aux scalars alone. That is an empirical result, not a simplification.
    # Gradient-boosted trees on the 914 test split, MAE as % of TBO on the
    # wear-limited windows:
    #     aux only, 10 features                 11.06%
    #     temporal stats only, 47 features      12.96%
    #     aux + temporal, 57 features           12.44%   <- WORSE than aux alone
    # Adding per-channel std, window mean and fitted slope for every residual
    # made the model worse, which is what redundant input does. The window does
    # carry degradation signal - 12.96% on its own is far better than the 19.76%
    # of a constant - but it is signal the aux snapshot already holds in
    # health_index, margin_min, the fault severities and the residual
    # magnitudes.
    #
    # The head reading 192 frozen encoder dimensions beside 64 aux units scored
    # 16.67%, worse than a tree on the window statistics alone. It was spending
    # its capacity learning to ignore three-quarters of its input, and no
    # encoder change - LSTM included - fixes that, because the problem is not
    # how the window is encoded but that it is not incremental.
    a = layers.Dense(96, activation="relu", name="rul_aux_d1")(aux_input)
    a = layers.Dense(64, activation="relu", name="rul_aux_d2")(a)
    h = layers.Dense(hidden, activation="relu", name="rul_d1")(a)
    h = layers.Dropout(dropout, name="rul_drop")(h)
    h = layers.Dense(hidden // 2, activation="relu", name="rul_d2")(h)
    # A PLAIN SIGMOID CANNOT REACH THIS TARGET. The ratio is EXACTLY 1.0 on
    # 74.9% of windows, and 1.0 is the sigmoid's asymptote - approachable only
    # as the logit runs to infinity, with the gradient vanishing on the way. The
    # run that first used one confirmed it: wear-limited validation sat at
    # 22.10, 23.00, 23.68, 23.83, 23.35, 23.02 and early stopping restored
    # epoch 1, because no epoch ever improved.
    #
    # Two changes make the endpoints attainable at finite logits:
    #   - rescale, so sigmoid(z) = 0.955 already gives ratio 1.0 (z ~ 3.05) and
    #     sigmoid(z) = 0.045 gives 0. The clip is active only in a narrow band,
    #     and because these are SHARED weights rather than per-sample
    #     parameters, other windows in the batch keep supplying gradient there.
    #   - initialise the bias at +3.0, so the head STARTS at ratio ~ 1, i.e. at
    #     "every engine runs to overhaul". That is the right answer for 75% of
    #     windows and scores 5.07% aggregate on its own, so training begins from
    #     the trivial solution and learns the wear-out deficit from there,
    #     instead of starting at half the calendar RUL and climbing back.
    z = layers.Dense(1, name="rul_logit",
                     bias_initializer=keras.initializers.Constant(3.0))(h)
    frac = layers.Lambda(rul_ratio_activation, name="rul_frac")(z)

    # PREDICT A FRACTION OF THE CALENDAR RUL, NOT A FRACTION OF TBO.
    #
    # rul_hours_true is min(TBO, wear_out) - engine_hours, so it is bounded
    # above by the calendar interval tbo - engine_hours. The ratio
    #     rul / (tbo - engine_hours)
    # is therefore in [0, 1] BY CONSTRUCTION: exactly 1 whenever overhaul
    # arrives first, below 1 only when wear-out does.
    #
    # Predicting rul/TBO directly asks one smooth sigmoid to represent a kinked
    # function - a min() of two lines - and the head could not hold both
    # regimes at once. Across one run the wear-limited and aggregate validation
    # errors moved in opposite directions epoch to epoch:
    #     epoch  8   wear 14.12%   all 12.61%
    #     epoch 11   wear 14.17%   all  8.51%
    #     epoch 10   wear 37.56%   all 22.59%
    # Reparameterising moves the kink into the multiplication, where it is
    # exact, and leaves the network a bounded, smooth quantity to learn. The
    # TBO-limited majority becomes "saturate to 1" rather than "regress the
    # engine's age back out of a scaled input".
    #
    # engine_hours_norm arrives SCALED, so the scaler is inverted in-graph from
    # the mean and scale of that one dimension, passed in at build time.
    if cal_idx is None or cal_mean is None or cal_scale is None:
        return layers.Lambda(lambda t: t * tbo_hours, name="y_rul_hours")(frac)

    return layers.Lambda(
        scale_by_calendar,
        arguments={"cal_idx": int(cal_idx), "cal_mean": float(cal_mean),
                   "cal_scale": float(cal_scale), "tbo_hours": float(tbo_hours)},
        name="y_rul_hours")([aux_input, frac])


# ---------------------------------------------------------------------------
def build_multitask_model(tbo_hours: float, channels=64, num_layers=6,
                          dropout=0.25, cal_idx=None, cal_mean=None,
                          cal_scale=None) -> keras.Model:
    """One encoder, six heads."""
    x = build_encoder_input()
    aux = build_aux_input()
    enc = build_encoder(x, channels=channels, num_layers=num_layers, dropout=dropout)
    outs = [
        build_detection_head(enc, dropout=dropout),
        build_fault_mode_head(enc, dropout=dropout),
        build_sensor_fault_head(enc, x, dropout=dropout),
        build_sensor_sev_head(enc, dropout=dropout),
        build_health_head(enc, dropout=dropout),
        build_rul_head(enc, aux, tbo_hours, dropout=dropout, cal_idx=cal_idx,
                       cal_mean=cal_mean, cal_scale=cal_scale),
    ]
    return keras.Model(inputs={"x": x, "x_rul_aux": aux},
                          outputs={o.node.layer.name if hasattr(o, "node") else n: o
                                   for n, o in zip(
                                       ["y_detection", "y_fault_mode", "y_sensor_fault",
                                        "y_sensor_sev", "y_health", "y_rul_hours"],
                                       outs)},
                          name="uav_v4_multitask")


# Loss weights. Detection and fault-mode carry the most weight because they are
# what the operator acts on; RUL is scaled down because its loss is in hours and
# would otherwise dominate the sum by three orders of magnitude.
LOSS_WEIGHTS = {
    "y_detection": 2.0,
    "y_fault_mode": 2.0,
    "y_sensor_fault": 1.0,
    "y_sensor_sev": 0.5,
    "y_health": 1.5,
    "y_rul_hours": 0.002,
}

LOSSES = {
    "y_detection": keras.losses.BinaryCrossentropy(),
    "y_fault_mode": keras.losses.BinaryCrossentropy(),
    "y_sensor_fault": keras.losses.SparseCategoricalCrossentropy(),
    "y_sensor_sev": keras.losses.MeanSquaredError(),
    "y_health": keras.losses.MeanSquaredError(),
    # Huber on RUL: squared error lets the few near-zero-RUL windows dominate,
    # and those are exactly the ones a single bad scenario can distort.
    "y_rul_hours": keras.losses.Huber(delta=50.0),
}

METRICS = {
    "y_detection": [keras.metrics.AUC(name="auc"),
                    keras.metrics.Precision(name="prec"),
                    keras.metrics.Recall(name="rec")],
    "y_fault_mode": [keras.metrics.AUC(name="auc", multi_label=True)],
    "y_sensor_fault": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
    "y_health": [keras.metrics.MeanAbsoluteError(name="mae")],
    "y_rul_hours": [keras.metrics.MeanAbsoluteError(name="mae")],
}
