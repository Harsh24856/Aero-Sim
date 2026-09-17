"""Emits validation/phase5_rul_v3_<key>.ipynb, one per engine, from a single template.

Four hand-maintained copies is how the old phase notebooks drifted apart - a metric
fixed in one and not the others. These are generated, so all four are identical
except for the engine, and regenerating is one command:

    validation/venv/bin/python3 validation/make_phase5_v3_notebooks.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINES = {"Rotax_912_ULS": "912", "Rotax_914_ULF": "914",
           "Rotax_915_iS": "915", "Rotax_916_iS": "916"}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def cells(engine, key):
    return [
        md(f"""
# Phase 5 v3 - RUL head — {engine}

Trained on the **probe dataset** (`backend/generate_rul_dataset.py`), not the v3
scenario chunks. Two measured properties of the scenario data put a floor under RUL
error that no architecture could get under, and this notebook exists because of them:

| | v3 scenario windows | v4/v5 probes |
|---|---|---|
| Label inside one 128 s window | moves by **15.8 h median, 34.3 h p95** | **exact** - wear is frozen per probe |
| Rows at >= 95% of TBO | **1.6%** | ~18% (near-new deliberately over-sampled) |
| `elapsed_hours` aux vs label | correlated through the scenario's own ageing | **r = 0.000** by construction |
| `bsfc_ratio` aux vs label | the head's crutch | r ~ -0.90 with fuel-sensor error modelled |

The deployed 914 head scores MAE 22.9 h - *the same size as the old label's own
ambiguity*. That is why 80 epochs changed nothing.

**Model: LSTM + window statistics, side by side.** A ridge on window mean/variance
alone reached -1.1 h near-new bias where the deployed 2-layer LSTM sat at -80 h, so
the level of the window carries the wear signal; the LSTM still earns its place on
the dynamics. Both branches feed one head, and the ablation cells below report what
each is worth.

Serving contract is unchanged: inputs `x` (128, 25) scaled by this engine's scaler
and `x_rul_aux` (10), output `y_rul_hours`. `aiv3.py` needs no change.

> Run **one** notebook at a time - 8 GB of RAM, and two TensorFlow jobs will swap.
"""),
        code(f"""
import os, sys, json, time
sys.path.insert(0, os.path.abspath("."))
import numpy as np, tensorflow as tf
from tensorflow import keras
import rul_data as R, model_architectures as A, train_common as C

ENGINE, KEY = "{engine}", "{key}"
PHASE = "phase5_rul_v3"
C.report_gpu()
"""),
        code("""
train = R.Split(ENGINE, "train")
val   = R.Split(ENGINE, "val")
test  = R.Split(ENGINE, "test")
TBO   = train.tbo
meta  = R.load_meta(ENGINE)

print(f"{meta['dataset']}  {ENGINE}  TBO {TBO:.0f} h")
print(f"probes: train {len(train):,}  val {len(val):,}  test {len(test):,}"
      f"   ({len(train)*meta['window_size']:,} rows of telemetry in the training windows)")
frac = train.y / TBO
print("life coverage (RUL/TBO at 0/10/25/50/75/90/100 pct):",
      np.round(np.percentile(frac, [0, 10, 25, 50, 75, 90, 100]), 3).tolist())
print(f"near-new share (>= 90% of TBO): {100*(frac >= 0.90).mean():.1f}%")

# The two properties this dataset was built for, verified rather than assumed.
i_clock = meta["rul_aux_order"].index("elapsed_hours")
i_bsfc  = meta["rul_aux_order"].index("bsfc_ratio")
print(f"corr(elapsed_hours, RUL) = {np.corrcoef(train.aux[:, i_clock], train.y)[0,1]:+.3f}"
      "   <- the session clock carries no life information")
print(f"corr(bsfc_ratio,   RUL) = {np.corrcoef(train.aux[:, i_bsfc],  train.y)[0,1]:+.3f}"
      "   <- informative, not invertible (fuel-sensor error modelled)")
print(f"wear frozen per probe: {meta['wear_frozen_per_probe']}  -> the label is exact")
"""),
        md("""
### The floor

A ridge regression on four summary statistics per channel (mean, std, last, slope)
plus the aux vector. Any model that cannot beat this has not earned its parameters,
and it is the same fit that first showed the near-new bias was a *data* problem.
"""),
        code("""
t0 = time.time()
baseline_predict, val_pred_base, info = R.fit_baseline(train, val)
base = R.evaluate(val.y, val_pred_base, TBO)
R.print_report("ridge baseline (val)", base, TBO)
print(f"({info['n_features']} features, {time.time()-t0:.0f}s)")
BASELINE_MAE_PCT = base["mae_pct"]
"""),
        md("""
### The hybrid

`build_rul_hybrid` = LSTM branch (32 units x 2) + window statistics (mean, variance,
last sample, half-to-half slope, all with standard layers so it serialises and
converts to TFLite) + the normalised aux vector, into the existing `build_rul_head`.

A **linear skip** runs from the statistics straight to the output, in parallel with
the MLP. The ridge above shows the linear solution is most of the answer; without the
skip the net spends its first epochs rediscovering it (measured: val MAE 73.7 -> 57.7
over three epochs, still four times the ridge), and with it training starts near the
ridge and improves from there.

The aux `Normalization` is adapted on train only: `elapsed_hours` is in hours,
`cht_excess` in degrees and the ratios sit near 1, so raw inputs would let the
degree-scaled feature dominate initialisation.
"""),
        code("""
aux_norm = keras.layers.Normalization(axis=-1, name="rul_aux_norm")
aux_norm.adapt(train.aux)

xi, ax = A.build_encoder_input(), A.build_aux_input()
out = A.build_rul_hybrid(xi, ax, init_hours=0.5*TBO, lstm_units=32, lstm_layers=2,
                         hidden=96, dropout=0.1, aux_norm=aux_norm, linear_skip=True)
model = keras.Model({"x": xi, "x_rul_aux": ax}, {"y_rul_hours": out}, name="rul")
model.compile(optimizer=keras.optimizers.Adam(2e-3, clipnorm=1.0),
              loss={"y_rul_hours": C.rul_loss(TBO)},          # Huber in % of TBO
              metrics={"y_rul_hours": [keras.metrics.MeanAbsoluteError(name="mae"),
                                       C.mae_pct_tbo(TBO)]})
print(f"{model.count_params():,} parameters")
"""),
        code("""
BATCH = 256
ckpt = C.ckpt_path(ENGINE, PHASE)
# ModelCheckpoint insists on a .keras suffix, so the candidate keeps one.
cand = ckpt.replace(".keras", ".candidate.keras")   # promoted only if it beats what is there

hist = model.fit(
    R.tf_dataset(train, batch_size=BATCH, shuffle=True),
    validation_data=R.tf_dataset(val, batch_size=BATCH, shuffle=False),
    steps_per_epoch=R.steps_for(train, BATCH),
    validation_steps=R.steps_for(val, BATCH),
    epochs=40,     # ~0.25 s/step on this machine; EarlyStopping usually ends it sooner
    callbacks=C.callbacks(cand, monitor="val_mae", mode="min", patience=6,
                          lr_patience=3, min_lr=1e-5),
    verbose=1,
)
model.load_weights(cand)            # best epoch, not the last one
print("candidate:", cand)
"""),
        md("""
### Score it on the test split

MAE as a percentage of TBO, correlation across life stages, monotonicity by decile,
and the near-new band separately - that band is 100% of what a demo shows and where
the deployed head is 3.5-6.2% low.
"""),
        code("""
pred = R.predict_model(model, test)
m = R.evaluate(test.y, pred, TBO)
R.print_report("hybrid (test)", m, TBO)
"""),
        md("""
### Ablations - do these before trusting the number

1. **bsfc_ratio zeroed.** physics v3 ties fuel flow to wear, so this is the one
   feature a lazy head can read instead of the window.
2. **Each branch alone.** Short re-trains (8 epochs) of stats-only and LSTM-only, to
   report what the hybrid is actually buying. Skip by setting `RUN_BRANCH_ABLATION = False`.
"""),
        code("""
pred_ab = R.ablate_bsfc(model, test, meta["rul_aux_order"])
m_ab = R.evaluate(test.y, pred_ab, TBO)
print(f"with bsfc_ratio zeroed: MAE {m_ab['mae_h']:.1f} h ({m_ab['mae_pct']:.2f}% TBO), "
      f"corr {m_ab['corr']:.4f}   [full model: {m['mae_pct']:.2f}% TBO]")
ABLATED_MAE_PCT = m_ab["mae_pct"]
"""),
        code("""
RUN_BRANCH_ABLATION = True
branch = {}
if RUN_BRANCH_ABLATION:
    for name, kw in (("stats only", dict(use_lstm=False, use_stats=True)),
                     ("lstm only",  dict(use_lstm=True,  use_stats=False))):
        xi_b, ax_b = A.build_encoder_input(), A.build_aux_input()
        nrm = keras.layers.Normalization(axis=-1, name="rul_aux_norm"); nrm.adapt(train.aux)
        o = A.build_rul_hybrid(xi_b, ax_b, init_hours=0.5*TBO, aux_norm=nrm, **kw)
        mb = keras.Model({"x": xi_b, "x_rul_aux": ax_b}, {"y_rul_hours": o})
        mb.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
                   loss={"y_rul_hours": C.rul_loss(TBO)})
        mb.fit(R.tf_dataset(train, batch_size=BATCH, shuffle=True),
               steps_per_epoch=R.steps_for(train, BATCH), epochs=8, verbose=0)
        e = R.evaluate(test.y, R.predict_model(mb, test), TBO)
        branch[name] = e["mae_pct"]
        print(f"{name:10s}: MAE {e['mae_h']:6.1f} h ({e['mae_pct']:.2f}% TBO), "
              f"near-new bias {e['near_new_bias_pct']:+.2f}% TBO")
        del mb
    print(f"hybrid    : MAE {m['mae_h']:6.1f} h ({m['mae_pct']:.2f}% TBO), "
          f"near-new bias {m['near_new_bias_pct']:+.2f}% TBO")
"""),
        md("""
### Gates, then keep the better model

The candidate is promoted to the phase checkpoint only if it beats the previous
run's test MAE. A worse run leaves the good weights in place.
"""),
        code("""
g = R.gates(m, TBO, baseline_mae_pct=BASELINE_MAE_PCT, ablated_mae_pct=ABLATED_MAE_PCT)
passed = R.print_gates(g)
"""),
        code("""
report_path = os.path.join(C.MODELS_DIR, f"{PHASE}_{KEY}_report.json")
prev = json.load(open(report_path)) if os.path.exists(report_path) else None

report = {
    "engine_model": ENGINE, "key": KEY, "phase": PHASE,
    "dataset": meta["dataset"], "tbo_hours": TBO,
    "probes": {"train": len(train), "val": len(val), "test": len(test)},
    "test": {k: v for k, v in m.items() if k != "table"},
    "test_table": m["table"].to_dict(orient="records"),
    "baseline_mae_pct": BASELINE_MAE_PCT,
    "ablated_bsfc_mae_pct": ABLATED_MAE_PCT,
    "branch_ablation_mae_pct": branch,
    "gates": {k: {"passed": bool(ok), "detail": t} for k, (ok, t) in g.items()},
    "all_gates_passed": bool(passed),
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# What split_deployable_heads.py folds into the manifest, and aiv3.py serves as the
# RUL uncertainty band - including the near-new band, which widens the band on a
# fresh engine instead of pretending the overall MAE applies there.
test_json = {
    "mae_hours": m["mae_h"], "mae_pct_tbo": m["mae_pct"], "corr": m["corr"],
    "near_new_mae_hours": m["near_new_mae_h"],
    "near_new_bias_hours": m["near_new_bias_h"],
    "near_new_definition": f"true RUL >= {100*R.NEAR_NEW_FROM:.0f}% of TBO, test split",
    "dataset": meta["dataset"], "phase": PHASE,
}

better = prev is None or m["mae_pct"] <= prev["test"]["mae_pct"]
if passed and better:
    model.save(ckpt)
    json.dump(report, open(report_path, "w"), indent=2)
    json.dump(test_json, open(os.path.join(C.MODELS_DIR, f"{PHASE}_{KEY}_test.json"), "w"), indent=2)
    print(f"promoted -> {ckpt}")
    if prev:
        print(f"  (previous test MAE {prev['test']['mae_pct']:.2f}% of TBO)")
else:
    reason = "gates failed" if not passed else \\
             f"previous run was better ({prev['test']['mae_pct']:.2f}% vs {m['mae_pct']:.2f}%)"
    print(f"NOT promoted: {reason}. Candidate left at {cand}")
"""),
        md("""
### Deploying

This head replaces `<key>_rul.keras` in `backend/models_v3/<key>/`. Split and export
exactly as before - the input and output names are unchanged, so `aiv3.py` and
`validation/export_edge.py` need no edit:

```bash
validation/venv/bin/python3 validation/split_deployable_heads.py --engines KEY --rul-phase phase5_rul_v3
validation/venv/bin/python3 validation/export_edge.py --engines KEY
```

Then re-run the live check (`validation/parity_ai_v3.py`) before flying it.
"""),
    ]


def main():
    for engine, key in ENGINES.items():
        nb = {
            "cells": cells(engine, key),
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4, "nbformat_minor": 5,
        }
        path = os.path.join(HERE, f"phase5_rul_v3_{key}.ipynb")
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
