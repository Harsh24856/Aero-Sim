"""Loader, baseline and gates for the RUL probe dataset (backend/generate_rul_dataset.py).

The phase-5 notebooks are thin on purpose - everything reusable lives here, so all
four engines are scored by exactly the same code and a metric can never drift
between them.

MEMORY. A 100k-probe training split is 100,000 x 128 x 25 x 4 B = 1.28 GB. This
machine has 8 GB and TensorFlow wants a large slice of it, so x is never loaded
whole: it stays a memmap on disk and batches are sliced out and scaled on the way
past. Scaling is affine per feature, so doing it per batch costs one multiply-add
and saves a second 1.28 GB copy.

SCALING. The engine's StandardScaler (validation/models/scaler_<key>.pkl) is the
same object tf_data_pipeline and aiv3 use, applied here as (x - mean)/scale in
FEATURE_COLS order. The probes are stored raw so a rescale never needs a
regeneration.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(VALIDATION_DIR, "models")

# Newest first: a regenerated dataset lands under a new prefix, and everything
# picks it up without editing four notebooks.
DATA_PREFIXES = ("rul_v5", "rul_v4")

ENGINE_KEY = {
    "Rotax_912_ULS": "912",
    "Rotax_914_ULF": "914",
    "Rotax_915_iS": "915",
    "Rotax_916_iS": "916",
}

# A prediction is judged near-new above this share of TBO. Every live flight starts
# at ~99%, and this band is where the deployed head reads 3.5-6.2% low.
NEAR_NEW_FROM = 0.90

# Gates. MAE and bias are in percent of TBO so the four engines are comparable
# (100 h is 5% of a 2000 h TBO but 8.3% of the 915's 1200 h).
GATE_MAE_PCT = 2.0
GATE_NEAR_NEW_BIAS_PCT = 2.0
GATE_CORR = 0.95
GATE_ABLATED_MAE_PCT = 5.0     # MAE with bsfc_ratio zeroed: the head must still work


def probe_dir(engine, prefix=None):
    """Directory holding this engine's probes. Prefers the newest prefix present.

    RUL_PREFIX in the environment pins one (e.g. RUL_PREFIX=rul_v4 while a larger
    rul_v5 is still generating). A directory only counts once meta.json is there,
    which the generator writes last - so a half-written dataset is never picked up.
    """
    prefix = prefix or os.environ.get("RUL_PREFIX")
    if prefix:
        return os.path.join(VALIDATION_DIR, f"{prefix}_{engine.lower()}")
    for p in DATA_PREFIXES:
        d = os.path.join(VALIDATION_DIR, f"{p}_{engine.lower()}")
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")):
            return d
    raise FileNotFoundError(
        f"no probe dataset for {engine} - run backend/generate_rul_dataset.py")


def load_meta(engine, prefix=None):
    with open(os.path.join(probe_dir(engine, prefix), "meta.json")) as f:
        return json.load(f)


def tbo_of(engine, prefix=None):
    return float(load_meta(engine, prefix)["tbo_hours"])


def scaler_arrays(engine):
    """(mean, scale) in FEATURE_COLS order, from the engine's fitted StandardScaler."""
    sc = joblib.load(os.path.join(MODELS_DIR, f"scaler_{ENGINE_KEY[engine]}.pkl"))
    return sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)


class Split:
    """One split, memmap-backed. .x stays on disk; .scaled_batch() does the scaling."""

    def __init__(self, engine, split, prefix=None):
        d = probe_dir(engine, prefix)
        self.engine, self.split, self.dir = engine, split, d
        self.x = np.load(os.path.join(d, f"{split}_x.npy"), mmap_mode="r")
        self.aux = np.load(os.path.join(d, f"{split}_aux.npy"))
        self.y = np.load(os.path.join(d, f"{split}_y.npy"))
        self.meta = pd.read_parquet(os.path.join(d, f"{split}_meta.parquet"))
        self.tbo = float(load_meta(engine, prefix)["tbo_hours"])
        self.mean, self.scale = scaler_arrays(engine)

    def __len__(self):
        return len(self.y)

    def scaled(self, idx):
        """Scaled windows for the given indices, as float32 (idx should be a slice
        or a sorted array - fancy-indexing a memmap reads only those rows)."""
        return ((np.asarray(self.x[idx], dtype=np.float32) - self.mean) / self.scale)

    def batches(self, batch_size=256, shuffle=False, seed=0, drop_last=False):
        """Yields ({"x", "x_rul_aux"}, y) dicts. Plain Python so it works for both
        model.fit (wrapped below) and evaluation loops."""
        n = len(self)
        order = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for s in range(0, n, batch_size):
            idx = np.sort(order[s:s + batch_size])
            if drop_last and len(idx) < batch_size:
                return
            yield ({"x": self.scaled(idx), "x_rul_aux": self.aux[idx]}, self.y[idx])


def tf_dataset(split_obj, batch_size=256, shuffle=True, seed=0, repeat=True):
    """tf.data wrapper around Split.batches, for model.fit.

    from_generator rather than from_tensor_slices: the latter would materialise the
    whole 1.28 GB array as a constant in the graph.
    """
    import tensorflow as tf
    n_feat = split_obj.x.shape[2]
    n_aux = split_obj.aux.shape[1]
    win = split_obj.x.shape[1]

    def gen():
        s = 0
        while True:
            for batch in split_obj.batches(batch_size, shuffle=shuffle, seed=seed + s,
                                           drop_last=True):
                yield batch
            s += 1
            if not repeat:
                return

    sig = (
        {"x": tf.TensorSpec(shape=(None, win, n_feat), dtype=tf.float32),
         "x_rul_aux": tf.TensorSpec(shape=(None, n_aux), dtype=tf.float32)},
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


def steps_for(split_obj, batch_size=256):
    return max(1, len(split_obj) // batch_size)


# ---------------------------------------------------------------------------
# Tabular features - the baseline's input, and the evidence that the window's
# LEVEL carries the signal (ridge on these reached -1.1 h near-new bias where the
# deployed LSTM sat at -80 h).
# ---------------------------------------------------------------------------
def window_features(split_obj, chunk=2048, with_aux=True):
    """(N, F) float32: per-feature mean, std, last and half-to-half slope of the
    scaled window, optionally with the aux vector appended. Streams the memmap in
    chunks so peak memory is one chunk."""
    n, win, nf = split_obj.x.shape
    half = win // 2
    out = []
    for s in range(0, n, chunk):
        xb = split_obj.scaled(slice(s, min(s + chunk, n)))
        feats = [xb.mean(axis=1), xb.std(axis=1), xb[:, -1, :],
                 xb[:, half:, :].mean(axis=1) - xb[:, :half, :].mean(axis=1)]
        out.append(np.concatenate(feats, axis=1).astype(np.float32))
    F = np.concatenate(out, axis=0)
    if with_aux:
        F = np.concatenate([F, split_obj.aux], axis=1).astype(np.float32)
    return F


def feature_names(split_obj, cols, with_aux=True, aux_order=None):
    names = ([f"{c}_mean" for c in cols] + [f"{c}_std" for c in cols]
             + [f"{c}_last" for c in cols] + [f"{c}_slope" for c in cols])
    if with_aux:
        names += list(aux_order or [])
    return names


def fit_baseline(train, val, alpha=1.0):
    """Ridge on the window features. This is the FLOOR: any model that cannot beat a
    linear fit on four summary statistics per channel has not earned its parameters.

    It is also the phase-5 v3 head's INITIALISATION. `info` carries the fit in the
    exact space the model's linear skip sees - features standardised by `feat_mean`
    and `feat_std`, target in fractions of TBO - so build_rul_hybrid(ridge_init=...)
    starts training at this solution rather than hunting for it.

    Returns (predict_fn, val_pred, info).
    """
    from sklearn.linear_model import Ridge

    Xtr, Xva = window_features(train), window_features(val)
    mean = Xtr.mean(axis=0).astype(np.float32)
    std = Xtr.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0                       # a constant feature contributes nothing
    Ztr = (Xtr - mean) / std
    model = Ridge(alpha=alpha).fit(Ztr, train.y / train.tbo)     # target in TBO fractions

    def predict(split_obj):
        Z = (window_features(split_obj) - mean) / std
        return (model.predict(Z) * split_obj.tbo).astype(np.float32)

    info = {
        "n_features": Xtr.shape[1], "alpha": alpha,
        "coef": model.coef_.astype(np.float32),          # fraction of TBO per std
        "intercept": float(model.intercept_),
        "feat_mean": mean, "feat_std": std,
    }
    return predict, predict(val), info


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def predict_model(model, split_obj, batch_size=512):
    """Model predictions over a whole split, batch by batch off the memmap."""
    preds = []
    for xb, _ in split_obj.batches(batch_size, shuffle=False):
        p = model.predict(xb, verbose=0)
        p = p["y_rul_hours"] if isinstance(p, dict) else p
        preds.append(np.asarray(p, dtype=np.float32).reshape(-1))
    return np.concatenate(preds)


def evaluate(y_true, y_pred, tbo, near_new_from=NEAR_NEW_FROM, bins=10):
    """Every number the gates need, plus a per-decile table."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    frac = y_true / tbo
    nn = frac >= near_new_from

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        m = (frac >= edges[i]) & (frac < edges[i + 1] if i < bins - 1 else frac <= 1.0)
        if not m.any():
            continue
        rows.append({
            "life_lo": round(100*edges[i]), "life_hi": round(100*edges[i + 1]),
            "n": int(m.sum()),
            "true_mean_h": float(y_true[m].mean()),
            "pred_mean_h": float(y_pred[m].mean()),
            "mae_h": float(np.abs(err[m]).mean()),
            "bias_h": float(err[m].mean()),
        })
    table = pd.DataFrame(rows)
    # Monotonicity: mean prediction must rise with true remaining life, bucket to
    # bucket. A head that tracks the operating point instead of the engine fails here
    # even when its MAE looks respectable - that was v2's failure mode.
    violations = int((np.diff(table["pred_mean_h"].to_numpy()) < 0).sum())

    return {
        "n": int(len(y_true)),
        "mae_h": float(np.abs(err).mean()),
        "mae_pct": float(100*np.abs(err).mean()/tbo),
        "rmse_h": float(np.sqrt((err**2).mean())),
        "bias_h": float(err.mean()),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "near_new_n": int(nn.sum()),
        "near_new_mae_h": float(np.abs(err[nn]).mean()) if nn.any() else float("nan"),
        "near_new_bias_h": float(err[nn].mean()) if nn.any() else float("nan"),
        "near_new_bias_pct": float(100*err[nn].mean()/tbo) if nn.any() else float("nan"),
        "monotonic_violations": violations,
        "table": table,
    }


def print_report(name, m, tbo):
    print(f"{name}: MAE {m['mae_h']:.1f} h ({m['mae_pct']:.2f}% TBO) | "
          f"bias {m['bias_h']:+.1f} h | corr {m['corr']:.4f} | "
          f"near-new bias {m['near_new_bias_h']:+.1f} h "
          f"({m['near_new_bias_pct']:+.2f}% TBO, n={m['near_new_n']}) | "
          f"monotonicity violations {m['monotonic_violations']}")
    t = m["table"].copy()
    t["life band"] = t.apply(lambda r: f"{r.life_lo:.0f}-{r.life_hi:.0f}% of TBO", axis=1)
    print(t[["life band", "n", "true_mean_h", "pred_mean_h", "mae_h", "bias_h"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.1f}"))


def ablate_bsfc(model, split_obj, aux_order, batch_size=512):
    """Predictions with bsfc_ratio zeroed out of the aux vector.

    physics v3 ties fuel flow to wear directly, so bsfc_ratio is the one feature a
    lazy head can read instead of the window. If MAE barely moves without it, the
    head learned the window; if it collapses, it did not.
    """
    j = list(aux_order).index("bsfc_ratio")
    preds = []
    for xb, _ in split_obj.batches(batch_size, shuffle=False):
        aux = xb["x_rul_aux"].copy()
        aux[:, j] = 0.0
        p = model.predict({"x": xb["x"], "x_rul_aux": aux}, verbose=0)
        p = p["y_rul_hours"] if isinstance(p, dict) else p
        preds.append(np.asarray(p, dtype=np.float32).reshape(-1))
    return np.concatenate(preds)


def gates(m, tbo, baseline_mae_pct=None, ablated_mae_pct=None):
    """The checks that decide whether this head ships. Returns {name: (passed, text)}."""
    g = {
        "mae": (m["mae_pct"] <= GATE_MAE_PCT,
                f"MAE {m['mae_pct']:.2f}% of TBO <= {GATE_MAE_PCT}%"),
        "near_new_bias": (abs(m["near_new_bias_pct"]) <= GATE_NEAR_NEW_BIAS_PCT,
                          f"near-new bias {m['near_new_bias_pct']:+.2f}% of TBO, "
                          f"|.| <= {GATE_NEAR_NEW_BIAS_PCT}%"),
        "corr": (m["corr"] >= GATE_CORR, f"corr {m['corr']:.4f} >= {GATE_CORR}"),
        "monotonic": (m["monotonic_violations"] == 0,
                      f"{m['monotonic_violations']} decile inversions, must be 0"),
    }
    if baseline_mae_pct is not None:
        g["beats_baseline"] = (m["mae_pct"] <= baseline_mae_pct,
                               f"model {m['mae_pct']:.2f}% vs ridge baseline "
                               f"{baseline_mae_pct:.2f}% of TBO")
    if ablated_mae_pct is not None:
        g["bsfc_ablation"] = (ablated_mae_pct <= GATE_ABLATED_MAE_PCT,
                              f"MAE without bsfc_ratio {ablated_mae_pct:.2f}% "
                              f"<= {GATE_ABLATED_MAE_PCT}% of TBO")
    return g


def print_gates(g):
    for name, (ok, text) in g.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:16s} {text}")
    print("ALL GATES PASSED" if all(ok for ok, _ in g.values()) else "SOME GATES FAILED")
    return all(ok for ok, _ in g.values())
