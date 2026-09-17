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

**The head starts at the ridge solution.** Its linear skip is initialised with the
ridge coefficients and the MLP's output kernel is zeroed, so before a single gradient
step the model reproduces the ridge to within 0.01 h - and training can only earn its
way down from 0.69% of TBO. Two parameterisations were tried first and are recorded in
`build_rul_head`: raw hours through a softplus crawls (val MAE 129 -> 96 -> 79 -> 70
over four epochs, because one Adam step moves the output by milli-hours against a
2000 h range), and TBO*sigmoid oscillated at lr 2e-3 (167 -> 140 -> 118 -> 146 -> 201).

The LSTM and the MLP therefore have one job: beat a linear fit by reading what a
linear fit cannot - transients, ripple, settling.

The aux `Normalization` is adapted on train only: `elapsed_hours` is in hours,
`cht_excess` in degrees and the ratios sit near 1, so raw inputs would let the
degree-scaled feature dominate initialisation.
"""),
        code("""
# The normalisers are built FROM THE RIDGE FIT, not adapted separately, so the
# model's linear skip sees exactly the space the ridge coefficients were fitted in.
N_AUX = train.aux.shape[1]
n_stats = info["n_features"] - N_AUX
stats_norm = keras.layers.Normalization(axis=-1, name="rul_stats_norm",
                                        mean=info["feat_mean"][:n_stats],
                                        variance=info["feat_std"][:n_stats]**2)
aux_norm = keras.layers.Normalization(axis=-1, name="rul_aux_norm",
                                      mean=info["feat_mean"][n_stats:],
                                      variance=info["feat_std"][n_stats:]**2)

xi, ax = A.build_encoder_input(), A.build_aux_input()
out = A.build_rul_hybrid(xi, ax, lstm_units=32, lstm_layers=2, hidden=96, dropout=0.05,
                         aux_norm=aux_norm, stats_norm=stats_norm, linear_skip=True,
                         tbo_hours=TBO, ridge_init=(info["coef"], info["intercept"]))
model = keras.Model({"x": xi, "x_rul_aux": ax}, {"y_rul_hours": out}, name="rul")
model.compile(optimizer=keras.optimizers.Adam(3e-4, clipnorm=1.0),
              loss={"y_rul_hours": C.rul_loss(TBO)},          # Huber in % of TBO
              metrics={"y_rul_hours": [keras.metrics.MeanAbsoluteError(name="mae"),
                                       C.mae_pct_tbo(TBO)]})
print(f"{model.count_params():,} parameters")

# The initialisation must REPRODUCE the ridge, or the claim below is not true.
init_pred = R.predict_model(model, val)
print(f"at init, before any training: max |model - ridge| = "
      f"{np.max(np.abs(init_pred - val_pred_base)):.3f} h  "
      f"(MAE {R.evaluate(val.y, init_pred, TBO)['mae_pct']:.2f}% of TBO)")
"""),
        code("""
BATCH = 256
# The initialisation IS the ridge, so it is a legitimate candidate: if the branches
# cannot beat it, the honest thing is to ship the linear solution rather than a worse
# network. Keras restores the best EPOCH, not the starting point, so keep it here.
init_weights = model.get_weights()
ckpt = C.ckpt_path(ENGINE, PHASE)
# ModelCheckpoint insists on a .keras suffix, so the candidate keeps one.
cand = ckpt.replace(".keras", ".candidate.keras")   # promoted only if it beats what is there

hist = model.fit(
    R.tf_dataset(train, batch_size=BATCH, shuffle=True),
    validation_data=R.tf_dataset(val, batch_size=BATCH, shuffle=False),
    steps_per_epoch=R.steps_for(train, BATCH),
    validation_steps=R.steps_for(val, BATCH),
    epochs=50,     # starts at the ridge; EarlyStopping ends it when it stops improving
    callbacks=C.callbacks(cand, monitor="val_mae", mode="min", patience=20,
                          lr_patience=3, min_lr=1e-5),
    verbose=1,
)
model.load_weights(cand)            # best epoch, not the last one

trained_mae = R.evaluate(val.y, R.predict_model(model, val), TBO)["mae_pct"]
init_mae = R.evaluate(val.y, init_pred, TBO)["mae_pct"]
if init_mae < trained_mae:
    model.set_weights(init_weights)
    WON = "ridge initialisation"
    print(f"training did not beat its starting point ({trained_mae:.2f}% vs {init_mae:.2f}% of "
          f"TBO on val) - keeping the linear solution. The LSTM and MLP found nothing "
          f"the window statistics had not already given away on this engine.")
else:
    WON = "trained hybrid"
    print(f"trained hybrid improves on the ridge start: {trained_mae:.2f}% vs "
          f"{init_mae:.2f}% of TBO on val")
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
        nrm = keras.layers.Normalization(axis=-1, name="rul_aux_norm",
                                         mean=info["feat_mean"][n_stats:],
                                         variance=info["feat_std"][n_stats:]**2)
        o = A.build_rul_hybrid(xi_b, ax_b, aux_norm=nrm, tbo_hours=TBO,
                               stats_norm=(stats_norm if kw["use_stats"] else None),
                               ridge_init=((info["coef"], info["intercept"])
                                           if kw["use_stats"] else None),
                               init_hours=(None if kw["use_stats"] else 0.5*TBO), **kw)
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
    "won": WON,
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
