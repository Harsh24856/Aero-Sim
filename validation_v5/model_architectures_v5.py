"""The v5 model: ONE jointly trained network per engine (docs/v5_model_improvement_plan.md,
Phase 4). v4 trained six encoders in sequence, each phase weighting only its own
head while retraining the whole encoder, so the other heads decayed (the 916's
detection loss reached 96 in the diagnosis checkpoint).

Inputs
    seq  [B, 128, 39]  scaled per-second features (features_v5.FEATURE_COLS)
    ctx  [B, 45]       long-horizon residual context (42) + hours / usage / clock (3)
Encoder
    dilated causal TCN (v4's, receptive field 127 s) -> last / mean / max pooling,
    concatenated with an MLP over the context.
Heads
    detection   1 sigmoid    engine fault visible OR sensor fault visible
    diagnosis  14 sigmoid    per fault, present and visible (inapplicable masked in the loss)
    family      6 sigmoid    fault family (the UI answers at family level when unsure)
    severity   14 sigmoid    per fault, 0..1
    sensor     [12, 7]       ONE classifier shared by all channels (v4: separate
                             weights per channel and kind - ~10-15 training flights
                             per cell). Per channel: statistics built for each fault
                             kind, a small convolution over the channel's reading and
                             residual (shared weights), its long-horizon context, a
                             learned channel embedding, and the encoder's view of the
                             whole engine (so a real engine change, visible on
                             coupled channels, is not blamed on one sensor).
    health      1 sigmoid    wear condition
RUL is a separate model on this network's outputs (rul_v5.py).
"""
from __future__ import annotations

import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

import features_v5 as F  # noqa: E402
from degradation_v5 import FAULT_NAMES  # noqa: E402
from observability_v5 import FAMILIES, FAMILY_OF  # noqa: E402
from sensors_v5 import FAULTABLE_CHANNELS  # noqa: E402

N_FAULTS = len(FAULT_NAMES)
FAMILY_NAMES = list(FAMILIES)
N_FAMILIES = len(FAMILY_NAMES)
N_CH = len(FAULTABLE_CHANNELS)
N_KINDS = 7
N_CTX = F.N_LONG + 3
MEAS_IDX = [F.FEATURE_COLS.index(c) for c in FAULTABLE_CHANNELS]
RES_IDX = [F.FEATURE_COLS.index(f"res_{c}") for c in FAULTABLE_CHANNELS]
LH_IDX = [F.LONG_COLS.index(f"lh_{c}_{k}") for c in FAULTABLE_CHANNELS for k in F.LONG_KINDS]
# fault -> family membership matrix [14, 6]
FAMILY_MATRIX = [[1.0 if FAMILY_OF[f] == fam else 0.0 for fam in FAMILY_NAMES] for f in FAULT_NAMES]


def relu(z):
    """ReLU as max(z, 0) - same function, same weights. tensorflow-metal computes
    ReLU after a matmul WRONGLY inside a tf.function (up to 3.2 off; eager and CPU
    exact), and every training step is a tf.function while predict() is eager: the
    v5 models were trained on a different network from the one they were scored and
    served as. max(z, 0) is exact in both (tests/test_graph_parity_v5.py)."""
    return tf.maximum(z, 0.0)


def tcn_block(x, channels, kernel_size, dilation, dropout, name):
    res = x
    h = layers.Conv1D(channels, kernel_size, padding="causal", dilation_rate=dilation,
                      activation=relu, name=f"{name}_c1")(x)
    h = layers.SpatialDropout1D(dropout, name=f"{name}_d1")(h)
    h = layers.Conv1D(channels, kernel_size, padding="causal", dilation_rate=dilation,
                      activation=relu, name=f"{name}_c2")(h)
    h = layers.SpatialDropout1D(dropout, name=f"{name}_d2")(h)
    if res.shape[-1] != channels:
        res = layers.Conv1D(channels, 1, name=f"{name}_proj")(res)
    return layers.Activation(relu, name=f"{name}_relu")(layers.Add(name=f"{name}_add")([h, res]))


class ChannelDropout(layers.Layer):
    """Zero whole input features during training (hardens against dead sensors)."""

    def __init__(self, rate, **kw):
        super().__init__(**kw)
        self.rate = rate

    def call(self, x, training=None):
        if not training or self.rate <= 0:
            return x
        keep = tf.cast(tf.random.uniform([tf.shape(x)[0], 1, tf.shape(x)[2]]) >= self.rate, x.dtype)
        return x * keep

    def get_config(self):
        return dict(super().get_config(), rate=self.rate)


class SensorChannelFeatures(layers.Layer):
    """Per-channel fault statistics, [B, 12, S]. Each targets one fault kind:
    level shift and slope (bias, drift), flat fraction (stuck), largest jump over
    typical jump (spike), step-noise over spread (noise), minimum and jump (dropout)."""

    def call(self, seq):
        m = tf.gather(seq, MEAS_IDX, axis=-1)                   # [B, T, 12]
        r = tf.gather(seq, RES_IDX, axis=-1)
        d = m[:, 1:] - m[:, :-1]
        ad = tf.abs(d)
        t = tf.cast(tf.range(tf.shape(m)[1]), m.dtype)
        tc = t - tf.reduce_mean(t)
        denom = tf.reduce_sum(tc * tc)

        def slope(s):
            return tf.reduce_sum(tc[None, :, None] * (s - tf.reduce_mean(s, 1, keepdims=True)), 1) / denom * 128.0

        def shift(s):
            return tf.reduce_mean(s[:, -32:], 1) - tf.reduce_mean(s[:, :32], 1)

        eps = 1e-2
        # Ratios explode on flat channels (a stuck or dropped-out sensor): compress
        # every heavy-tailed statistic with signed log1p so none dominates the layer.
        slog = lambda z: tf.sign(z) * tf.math.log1p(tf.abs(z))  # noqa: E731
        feats = [
            slog(shift(m)), slog(shift(r)), slog(slope(m)), slog(slope(r)),
            tf.reduce_mean(tf.cast(ad < 1e-4, m.dtype), 1),            # flat fraction (whole)
            tf.reduce_mean(tf.cast(ad[:, -32:] < 1e-4, m.dtype), 1),   # flat fraction (recent)
            slog(tf.reduce_max(ad, 1) / (tf.reduce_mean(ad, 1) + eps)),  # spikiness
            slog(tf.math.reduce_std(d, 1) / (tf.math.reduce_std(m, 1) + eps)),
            slog(tf.math.reduce_std(m, 1)), slog(tf.reduce_min(m, 1)), slog(tf.reduce_max(ad, 1)),
            slog(tf.reduce_mean(r, 1)), slog(tf.math.reduce_std(r, 1)),
        ]
        return tf.stack(feats, axis=-1)                               # [B, 12, 13]


def build_model(channels: int = 64, num_layers: int = 6, kernel_size: int = 3,
                dropout: float = 0.2, input_dropout: float = 0.03, head_hidden: int = 96,
                ctx_hidden: int = 64, linear_regression: bool = False, enc_norm: bool = False) -> keras.Model:
    """linear_regression: severity and health end in a linear unit (clipped to 0..1
    at prediction time) instead of a sigmoid. The joint model's sigmoid severities
    saturated at 0 or 1 under the L1 loss, where their gradient vanishes.
    enc_norm: LayerNormalization on the encoder output. Without it the trained joint
    914 encoder reached |h| = 258 (20 at init): sigmoid heads saturate, linear heads
    blew up (health's training loss hit 36,499 in epoch 2)."""
    seq = layers.Input((F.WINDOW, F.N_FEATURES), name="seq")
    ctx = layers.Input((N_CTX,), name="ctx")

    x = ChannelDropout(input_dropout, name="in_drop")(seq)
    for i in range(num_layers):
        x = tcn_block(x, channels, kernel_size, 2 ** i, dropout, f"tcn{i}")
    last = layers.Lambda(lambda t: t[:, -1, :], name="enc_last")(x)
    avg = layers.GlobalAveragePooling1D(name="enc_avg")(x)
    mx = layers.GlobalMaxPooling1D(name="enc_max")(x)
    c = layers.Dense(ctx_hidden, activation=relu, name="ctx_d1")(ctx)
    c = layers.Dense(ctx_hidden, activation=relu, name="ctx_d2")(c)
    enc = layers.Concatenate(name="enc")([last, avg, mx, c])
    if enc_norm:
        enc = layers.LayerNormalization(name="enc_norm")(enc)
    enc = layers.Dropout(dropout, name="enc_drop")(enc)

    def head(name, units, act, hidden=head_hidden):
        h = layers.Dense(hidden, activation=relu, name=f"{name}_h")(enc)
        h = layers.Dropout(dropout / 2, name=f"{name}_hd")(h)
        return layers.Dense(units, activation=act, name=name)(h)

    det = head("detection", 1, "sigmoid", 48)
    diag = head("diagnosis", N_FAULTS, "sigmoid")
    fam = head("family", N_FAMILIES, "sigmoid", 48)
    reg_act = "linear" if linear_regression else "sigmoid"
    sev = head("severity", N_FAULTS, reg_act)
    health = head("health", 1, reg_act, 48)

    # -- sensor: one classifier shared across channels --------------------------
    stats = SensorChannelFeatures(name="sf_stats")(seq)                          # [B,12,13]
    stats = layers.LayerNormalization(name="sf_stats_norm")(stats)
    pair = layers.Lambda(lambda s: tf.stack([tf.gather(s, MEAS_IDX, axis=-1),
                                             tf.gather(s, RES_IDX, axis=-1)], axis=-1),
                         name="sf_pair")(seq)                                    # [B,T,12,2]
    pair = layers.Permute((2, 1, 3), name="sf_perm")(pair)                       # [B,12,T,2]
    pc = layers.Conv2D(16, (1, 5), padding="same", activation=relu, name="sf_conv1")(pair)
    pc = layers.Conv2D(16, (1, 5), padding="same", dilation_rate=(1, 4), activation=relu, name="sf_conv2")(pc)
    pc_max = layers.Lambda(lambda z: tf.reduce_max(z, axis=2), name="sf_cmax")(pc)   # [B,12,16]
    pc_avg = layers.Lambda(lambda z: tf.reduce_mean(z, axis=2), name="sf_cavg")(pc)
    lh = layers.Lambda(lambda z: tf.reshape(tf.gather(z, LH_IDX, axis=-1), [-1, N_CH, 3]),
                       name="sf_lh")(ctx)                                        # [B,12,3]
    emb = layers.Embedding(N_CH, 8, name="sf_chan_emb")(
        layers.Lambda(lambda z: tf.tile(tf.range(N_CH)[None, :], [tf.shape(z)[0], 1]), name="sf_ids")(seq))
    ctxv = layers.Dense(32, activation=relu, name="sf_enc_proj")(enc)
    ctxv = layers.Lambda(lambda z: tf.tile(z[:, None, :], [1, N_CH, 1]), name="sf_enc_tile")(ctxv)
    s = layers.Concatenate(axis=-1, name="sf_cat")([stats, pc_max, pc_avg, lh, emb, ctxv])
    s = layers.Dense(64, activation=relu, name="sf_d1")(s)
    s = layers.Dropout(dropout / 2, name="sf_dd")(s)
    s = layers.Dense(64, activation=relu, name="sf_d2")(s)
    sensor = layers.Dense(N_KINDS, name="sensor")(s)                               # logits [B,12,7]

    return keras.Model({"seq": seq, "ctx": ctx},
                       {"detection": det, "diagnosis": diag, "family": fam, "severity": sev,
                        "health": health, "sensor": sensor}, name="aerosim_v5")
