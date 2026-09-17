"""
TCN encoder + Detection/Diagnosis/Severity heads, Keras functional API.
Causal dilated Conv1D (Keras has padding="causal" built in - no manual cropping
needed, unlike the PyTorch version). Same design as before: 6 layers, dilation
doubling each layer, receptive field 253 timesteps covers the 128-step window.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Imported rather than duplicated: tf_data_pipeline is the specification, and a
# second hardcoded copy of N_FEATURES is exactly how the v2 feature contract
# drifted between files. If the import fails, fail loudly instead of guessing.
from tf_data_pipeline import (
    WINDOW_SIZE, N_FEATURES, N_CHANNELS, N_RUL_AUX, N_FAILURE_MODES, FAULT_TYPES,
)

N_DIAG_CLASSES = len(FAULT_TYPES)   # 6

def tcn_block(x, channels, kernel_size, dilation, dropout, name_prefix):
    residual = x
    h = layers.Conv1D(channels, kernel_size, padding="causal", dilation_rate=dilation,
                       activation="relu", name=f"{name_prefix}_conv1")(x)
    h = layers.Dropout(dropout, name=f"{name_prefix}_drop1")(h)
    h = layers.Conv1D(channels, kernel_size, padding="causal", dilation_rate=dilation,
                       activation="relu", name=f"{name_prefix}_conv2")(h)
    h = layers.Dropout(dropout, name=f"{name_prefix}_drop2")(h)
    if residual.shape[-1] != channels:
        residual = layers.Conv1D(channels, 1, padding="same", name=f"{name_prefix}_proj")(residual)
    out = layers.Add(name=f"{name_prefix}_add")([h, residual])
    return layers.ReLU(name=f"{name_prefix}_relu")(out)

def build_encoder_input():
    return layers.Input(shape=(WINDOW_SIZE, N_FEATURES), name="x")

def build_encoder(x_input, channels=48, num_layers=6, kernel_size=3, dropout=0.1):
    """TCN encoder.

    v2 collapsed the sequence with `t[:, -1, :]` - the LAST timestep only,
    discarding the representation of the other 127. That is enough to say what
    is happening right now, but it throws away the trend across the window,
    which is most of what distinguishes a developing fault from noise.

    v3 concatenates three views instead:
      last - the current state, preserved because a fault happening NOW must
             still be visible and this is the causal end of the sequence
      mean - the window's average behaviour
      max  - the worst moment in the window, which is what a transient
             (Spike, misfire) actually looks like

    Channels raised 32 -> 48: with ~8M training rows per engine the previous
    width was the binding constraint, not the data.
    """
    x = x_input
    for i in range(num_layers):
        x = tcn_block(x, channels, kernel_size, dilation=2**i, dropout=dropout, name_prefix=f"tcn{i}")
    last = layers.Lambda(lambda t: t[:, -1, :], name="enc_last")(x)
    avg = layers.GlobalAveragePooling1D(name="enc_avg")(x)
    mx = layers.GlobalMaxPooling1D(name="enc_max")(x)
    return layers.Concatenate(name="enc_out")([last, avg, mx])

def build_detection_head(enc_out, hidden=32, dropout=0.1):
    h = layers.Dense(hidden, activation="relu", name="det_dense1")(enc_out)
    h = layers.Dropout(dropout, name="det_drop")(h)
    return layers.Dense(1, activation="sigmoid", name="y_detection")(h)

def build_diagnosis_head(enc_out, hidden=64, dropout=0.1):
    h = layers.Dense(hidden, activation="relu", name="diag_dense1")(enc_out)
    h = layers.Dropout(dropout, name="diag_drop")(h)
    flat = layers.Dense(N_CHANNELS * N_DIAG_CLASSES, name="diag_dense2")(h)
    logits = layers.Reshape((N_CHANNELS, N_DIAG_CLASSES), name="diag_reshape")(flat)
    return layers.Softmax(axis=-1, name="y_diagnosis")(logits)

def build_severity_head(enc_out, diag_probs, hidden=64, dropout=0.1):
    diag_flat = layers.Flatten(name="diag_flat_for_sev")(diag_probs)
    combined = layers.Concatenate(name="sev_concat")([enc_out, diag_flat])
    h = layers.Dense(hidden, activation="relu", name="sev_dense1")(combined)
    h = layers.Dropout(dropout, name="sev_drop")(h)
    return layers.Dense(N_CHANNELS, activation="sigmoid", name="y_severity")(h)

def build_aux_input():
    """x_rul_aux: 6 severity/time statistics + 4 degradation residuals (v3)."""
    return layers.Input(shape=(N_RUL_AUX,), name="x_rul_aux")


def build_failure_mode_head(enc_out, hidden=64, dropout=0.1):
    """Per-mode severity for the ENGINE failure modes (v3, PS section C).

    Sigmoid per mode rather than a softmax across modes: misfire and cooling
    degradation can be present at the same time, so these are independent
    severities, not a one-of-N classification. That is the structural difference
    from the diagnosis head, which IS one-of-N per sensor channel.
    """
    # LayerNorm first. The warm-started encoder emits unnormalised activations
    # (measured on the 914 phase-3 encoder: |mean| 28.7, max 456). A fresh Dense
    # on inputs that large produced logits between -741 and -129 from the first
    # step, so sigmoid output was exactly 0.0, its derivative was 0, and the head
    # never received a gradient - loss sat at 0.0095 for all 20 epochs and every
    # AUC read 0.5000 on all three engines.
    h = layers.LayerNormalization(name="fm_norm")(enc_out)
    h = layers.Dense(hidden, activation="relu", name="fm_dense1")(h)
    h = layers.Dropout(dropout, name="fm_drop")(h)
    return layers.Dense(N_FAILURE_MODES, activation="sigmoid", name="y_failure_mode")(h)


def build_rul_head(enc_out, aux_input, hidden=64, dropout=0.1, init_hours=None, skip=None,
                   tbo_hours=None, ridge_init=None):
    """Predicts RUL in HOURS. Takes enc_out PLUS x_rul_aux. Softplus keeps RUL >= 0.

    Output stays in raw hours - this is the head that reached 33 h MAE on 914.
    Two TBO-fraction variants were tried and are worse: softplus on a 0-1 fraction
    has slope ~frac near zero, so near-overhaul windows barely train (bin 1 predicted
    222 h, MAE 63.5 h); a sharpened softplus(50z)/50 multiplies the output-layer
    gradient by 50 and diverges. In hours, softplus has slope ~1 everywhere above ~5 h.

    init_hours starts the output there (e.g. 0.3*TBO) instead of ~0.7 h, which skips
    the ~5 epochs the old run spent climbing up from zero. softplus(x) == x for x >> 1.
    """
    combined = layers.Concatenate(name="rul_concat")([enc_out, aux_input])
    h = layers.Dense(hidden, activation="relu", name="rul_dense1")(combined)
    h = layers.Dropout(dropout, name="rul_drop1")(h)
    h = layers.Dense(hidden // 2, activation="relu", name="rul_dense2")(h)
    # OUTPUT PARAMETERISATION - three modes, in increasing order of how much they
    # help Adam find the answer on this problem:
    #
    #   default          softplus(raw hours).  What the v3-scenario head uses; kept so
    #                    the old notebooks are untouched. One Adam step moves the output
    #                    by about `lr` HOURS against a 2000 h range, so training crawls
    #                    (measured: val MAE 129 -> 96 -> 79 -> 70 over four epochs).
    #   tbo_hours        TBO*sigmoid(z).  Bounded like the quantity itself, but the
    #                    gradient through the sigmoid made lr 2e-3 oscillate
    #                    (167 -> 140 -> 118 -> 146 -> 201).
    #   ridge_init       TBO*relu(z), with the linear path INITIALISED AT THE RIDGE
    #                    SOLUTION and the MLP's output kernel zeroed. Training then
    #                    starts exactly at the linear fit - 0.68% of TBO on the probe
    #                    dataset - and the network can only earn its way down from
    #                    there. This is the mode the phase-5 v3 notebooks use.
    #
    # ridge_init is (coef, intercept) in FRACTION-of-TBO units, fitted on the same
    # normalised [window stats, aux] vector that `skip` carries (rul_data.fit_baseline
    # returns exactly this).
    if ridge_init is not None:
        if skip is None:
            raise ValueError("ridge_init needs the linear skip")
        coef, intercept = ridge_init
        out_kernel = keras.initializers.Zeros()          # MLP starts contributing nothing
        bias = keras.initializers.Constant(float(intercept))
    elif tbo_hours:
        frac = 0.5 if init_hours is None else float(np.clip(init_hours/float(tbo_hours), 1e-3, 1-1e-3))
        out_kernel, bias = "glorot_uniform", keras.initializers.Constant(float(np.log(frac/(1.0 - frac))))
    else:
        out_kernel = "glorot_uniform"
        bias = "zeros" if init_hours is None else keras.initializers.Constant(float(init_hours))
    raw = layers.Dense(1, name="rul_dense_out", kernel_initializer=out_kernel,
                       bias_initializer=bias)(h)
    if skip is not None:
        skip_kernel = (keras.initializers.Constant(np.asarray(ridge_init[0], dtype="float32").reshape(-1, 1))
                       if ridge_init is not None else "glorot_uniform")
        raw = layers.Add(name="rul_out_add")(
            [raw, layers.Dense(1, name="rul_skip_dense", use_bias=False,
                               kernel_initializer=skip_kernel)(skip)])
    if ridge_init is not None:
        # relu, not sigmoid: identity over the whole 0..1 band the labels live in, so
        # the ridge initialisation is reproduced exactly, while still forbidding a
        # negative remaining life.
        return layers.Rescaling(float(tbo_hours), name="y_rul_hours")(
            layers.ReLU(name="rul_frac")(raw))
    if tbo_hours:
        return layers.Rescaling(float(tbo_hours), name="y_rul_hours")(
            layers.Activation("sigmoid", name="rul_frac")(raw))
    return layers.Activation("softplus", name="y_rul_hours")(raw)

def build_rul_lstm_encoder(x_input, units=32, num_layers=2, dropout=0.1):
    """Independent LSTM branch for RUL, processing the SAME raw window x_input but through
    entirely separate layers - zero shared weights with the TCN encoder used by Detection/
    Diagnosis/Severity. This eliminates the shared-gradient conflict found in the partial-
    unfreeze experiment (Diagnosis dropped to 87%, severity MSE spiked to 0.80) at the root,
    rather than managing it with loss_weights or freezing. LSTM chosen since sequential
    architectures are the traditional choice for RUL specifically (e.g. NASA C-MAPSS lit).
    num_layers now parameterized (was fixed at 2) so it's a real HP-search dimension,
    mirroring build_rul_tcn_encoder's flexibility."""
    h = x_input
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)  # only the LAST LSTM layer collapses the time axis
        h = layers.LSTM(units, return_sequences=return_sequences, name=f"rul_lstm{i}")(h)
        h = layers.Dropout(dropout, name=f"rul_lstm_drop{i}")(h)
    return h

def build_rul_transformer_encoder(x_input, num_heads=4, key_dim=32, ff_dim=64, dropout=0.1):
    """Independent Transformer branch for RUL - same principle as the LSTM branch: zero
    shared weights with the TCN encoder used by Detection/Diagnosis/Severity, so those
    heads stay architecturally protected. Single self-attention block (query=key=value=
    the raw window) + residual + layernorm + feedforward + residual + layernorm, then
    global average pooling to get a fixed-size representation for the RUL head."""
    n_features = x_input.shape[-1]

    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                          name="rul_mha")(x_input, x_input)
    attn_out = layers.Dropout(dropout, name="rul_mha_drop")(attn_out)
    x1 = layers.Add(name="rul_mha_add")([x_input, attn_out])
    x1 = layers.LayerNormalization(name="rul_mha_norm")(x1)

    ff = layers.Dense(ff_dim, activation="relu", name="rul_ff1")(x1)
    ff = layers.Dense(n_features, name="rul_ff2")(ff)
    ff = layers.Dropout(dropout, name="rul_ff_drop")(ff)
    x2 = layers.Add(name="rul_ff_add")([x1, ff])
    x2 = layers.LayerNormalization(name="rul_ff_norm")(x2)

    pooled = layers.GlobalAveragePooling1D(name="rul_pool")(x2)
    return pooled

def build_rul_tcn_encoder(x_input, channels=32, num_layers=6, kernel_size=3, dropout=0.1):
    """Independent TCN branch for RUL - same principle as the LSTM/Transformer branches:
    zero shared weights with the original TCN encoder used by Detection/Diagnosis/Severity,
    so those heads stay architecturally protected. Mirrors build_encoder()'s design exactly
    (6 layers, channels=32, dilation 1-32) but with rul_ prefixed names so it's a completely
    separate set of weights, not shared with the frozen path."""
    x = x_input
    ch_in = x_input.shape[-1]
    for i in range(num_layers):
        dilation = 2 ** i
        residual = x
        h = layers.Conv1D(channels, kernel_size, padding="causal", dilation_rate=dilation,
                           activation="relu", name=f"rul_tcn{i}_conv1")(x)
        h = layers.Dropout(dropout, name=f"rul_tcn{i}_drop1")(h)
        h = layers.Conv1D(channels, kernel_size, padding="causal", dilation_rate=dilation,
                           activation="relu", name=f"rul_tcn{i}_conv2")(h)
        h = layers.Dropout(dropout, name=f"rul_tcn{i}_drop2")(h)
        if residual.shape[-1] != channels:
            residual = layers.Conv1D(channels, 1, padding="same", name=f"rul_tcn{i}_proj")(residual)
        x = layers.Add(name=f"rul_tcn{i}_add")([h, residual])
        x = layers.ReLU(name=f"rul_tcn{i}_relu")(x)
    pooled = layers.Lambda(lambda t: t[:, -1, :], name="rul_tcn_last")(x)
    return pooled


# ---------------------------------------------------------------------------
# RUL v4: window statistics + LSTM hybrid
#
# Evidence for the shape of this, all measured on this project:
#   - A ridge regression on nothing but window mean/variance reached -1.1 h bias on
#     near-new engines where the deployed 2-layer LSTM sat at -80 h. The information
#     is in the window's LEVEL, and a recurrent net spends its capacity elsewhere.
#   - The LSTM still earns its place on the dynamics (settling, ripple, transient
#     response) that a pooled statistic cannot see.
# So: both, side by side, concatenated with the aux vector. Neither branch can hide
# the other's failure - the notebook ablates each one and reports the delta.
# ---------------------------------------------------------------------------
def build_rul_window_stats(x_input):
    """Per-feature mean, variance, last sample and half-to-half slope of the window.

    Standard layers only. A Lambda here cost a day of debugging twice: it needs an
    explicit output_shape, and on reload its closure had lost the `keras` global.
    Cropping1D/GlobalAveragePooling1D/Multiply/Subtract all serialise and convert to
    TFLite without special handling.
    """
    win = int(x_input.shape[1])
    half = win // 2
    mean = layers.GlobalAveragePooling1D(name="rul_win_mean")(x_input)
    sq = layers.Multiply(name="rul_win_sq")([x_input, x_input])
    sq_mean = layers.GlobalAveragePooling1D(name="rul_win_sq_mean")(sq)
    mean_sq = layers.Multiply(name="rul_win_mean_sq")([mean, mean])
    # E[x^2] - E[x]^2, floored at 0: the subtraction can go slightly negative on
    # float32 for a constant channel.
    var = layers.ReLU(name="rul_win_var")(
        layers.Subtract(name="rul_win_var_raw")([sq_mean, mean_sq]))
    # std rather than variance, so these are exactly rul_data.window_features and a
    # Normalization layer adapted on those numpy features fits this graph.
    std = layers.Lambda(lambda t: tf.sqrt(t + 1e-8), output_shape=lambda s: s,
                        name="rul_win_std")(var)
    last = layers.Flatten(name="rul_win_last")(
        layers.Cropping1D(cropping=(win - 1, 0), name="rul_win_last_crop")(x_input))
    h1 = layers.GlobalAveragePooling1D(name="rul_win_h1")(
        layers.Cropping1D(cropping=(0, win - half), name="rul_win_h1_crop")(x_input))
    h2 = layers.GlobalAveragePooling1D(name="rul_win_h2")(
        layers.Cropping1D(cropping=(half, 0), name="rul_win_h2_crop")(x_input))
    slope = layers.Subtract(name="rul_win_slope")([h2, h1])
    return layers.Concatenate(name="rul_win_stats")([mean, std, last, slope])


def build_rul_hybrid(x_input, aux_input, init_hours=None, lstm_units=32, lstm_layers=2,
                     hidden=96, dropout=0.1, aux_norm=None, use_lstm=True, use_stats=True,
                     linear_skip=True, tbo_hours=None, stats_norm=None, ridge_init=None):
    """The deployable RUL model's body: LSTM branch + window statistics + aux.

    aux_norm is a keras Normalization layer already adapted on the training aux
    (elapsed_hours is in hours, cht_excess in degrees, the ratios near 1 - feeding
    those raw into a Dense makes the degree-scaled feature dominate initialisation).

    use_lstm / use_stats exist for the notebook's branch ablation: rebuild with one
    of them off, retrain briefly, and the MAE delta says what each branch was worth.
    """
    parts, stats = [], None
    if use_lstm:
        parts.append(build_rul_lstm_encoder(x_input, units=lstm_units,
                                            num_layers=lstm_layers, dropout=dropout))
    if use_stats:
        stats = build_rul_window_stats(x_input)
        if stats_norm is not None:
            stats = stats_norm(stats)      # adapted on rul_data.window_features(train)
        parts.append(stats)
    if not parts:
        raise ValueError("build_rul_hybrid needs at least one of use_lstm/use_stats")
    body = parts[0] if len(parts) == 1 else layers.Concatenate(name="rul_merge")(parts)
    aux = aux_norm(aux_input) if aux_norm is not None else aux_input
    skip = (layers.Concatenate(name="rul_skip_in")([stats, aux])
            if (linear_skip and stats is not None) else None)
    return build_rul_head(body, aux, hidden=hidden, dropout=dropout, init_hours=init_hours,
                          skip=skip, tbo_hours=tbo_hours, ridge_init=ridge_init)
