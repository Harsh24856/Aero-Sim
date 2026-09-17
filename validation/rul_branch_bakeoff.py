"""Head-to-head: LSTM only vs window statistics only vs both, on the probe dataset.

    validation/venv/bin/python3 validation/rul_branch_bakeoff.py --engine Rotax_914_ULF
    RUL_PREFIX=rul_v4 validation/venv/bin/python3 validation/rul_branch_bakeoff.py \
        --engine Rotax_914_ULF --epochs 15

Same data, same batch size, same epoch budget, same early stopping for all three, so
the only thing that differs is the branch. Results go to
validation/models/rul_bakeoff_<key>.json and are printed as a table.

ONE ASYMMETRY, DELIBERATE AND UNAVOIDABLE. The ridge initialisation lives on the
window statistics, so `stats` and `hybrid` can start at the linear solution (0.69% of
TBO on 914) and `lstm` cannot - there is nothing linear to hand it. Giving the LSTM
the same start would mean giving it the statistics, which is exactly the variant it
is being compared against. Instead it gets the TBO*sigmoid output (bounded, better
conditioned than raw hours) and a higher learning rate, since it has further to
travel. That is the fairest available comparison, and the numbers should be read as
"what each branch is worth when trained the way that branch trains best", not as a
controlled experiment on a single optimiser setting.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tensorflow as tf                     # noqa: E402
from tensorflow import keras                # noqa: E402

import model_architectures as A             # noqa: E402
import rul_data as R                        # noqa: E402
import train_common as C                    # noqa: E402

VARIANTS = {
    # name: (use_lstm, use_stats, ridge_init_allowed, learning_rate)
    "lstm":   (True,  False, False, 1e-3),
    "stats":  (False, True,  True,  3e-4),
    "hybrid": (True,  True,  True,  3e-4),
}


def build(name, train, info, n_stats):
    use_lstm, use_stats, ridge_ok, lr = VARIANTS[name]
    stats_norm = aux_norm = None
    if use_stats:
        stats_norm = keras.layers.Normalization(
            axis=-1, name="rul_stats_norm",
            mean=info["feat_mean"][:n_stats], variance=info["feat_std"][:n_stats]**2)
    aux_norm = keras.layers.Normalization(
        axis=-1, name="rul_aux_norm",
        mean=info["feat_mean"][n_stats:], variance=info["feat_std"][n_stats:]**2)

    xi, ax = A.build_encoder_input(), A.build_aux_input()
    out = A.build_rul_hybrid(
        xi, ax, use_lstm=use_lstm, use_stats=use_stats,
        lstm_units=32, lstm_layers=2, hidden=96, dropout=0.05,
        aux_norm=aux_norm, stats_norm=stats_norm, tbo_hours=train.tbo,
        linear_skip=use_stats,
        ridge_init=((info["coef"], info["intercept"]) if (ridge_ok and use_stats) else None),
        init_hours=(None if use_stats else 0.5*train.tbo),
    )
    model = keras.Model({"x": xi, "x_rul_aux": ax}, {"y_rul_hours": out}, name=f"rul_{name}")
    model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
                  loss={"y_rul_hours": C.rul_loss(train.tbo)},
                  metrics={"y_rul_hours": [keras.metrics.MeanAbsoluteError(name="mae")]})
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="Rotax_914_ULF")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--variants", default="lstm,stats,hybrid")
    args = ap.parse_args()

    engine = args.engine
    key = C.ENGINE_KEY[engine]
    train, val, test = (R.Split(engine, s) for s in ("train", "val", "test"))
    tbo = train.tbo
    print(f"{R.load_meta(engine)['dataset']}  {engine}  TBO {tbo:.0f} h  "
          f"probes {len(train):,}/{len(val):,}/{len(test):,}", flush=True)

    _, val_pred_base, info = R.fit_baseline(train, val)
    base = R.evaluate(val.y, val_pred_base, tbo)
    n_stats = info["n_features"] - train.aux.shape[1]
    print(f"ridge floor (val): MAE {base['mae_h']:.1f} h ({base['mae_pct']:.2f}% TBO), "
          f"near-new bias {base['near_new_bias_pct']:+.2f}% TBO\n", flush=True)

    rows = {}
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        print(f"=== {name}", flush=True)
        model = build(name, train, info, n_stats)
        ckpt = os.path.join(C.MODELS_DIR, f"rul_bakeoff_{key}_{name}.keras")
        t0 = time.time()
        model.fit(R.tf_dataset(train, batch_size=args.batch, shuffle=True),
                  validation_data=R.tf_dataset(val, batch_size=args.batch, shuffle=False),
                  steps_per_epoch=R.steps_for(train, args.batch),
                  validation_steps=R.steps_for(val, args.batch),
                  epochs=args.epochs,
                  callbacks=C.callbacks(ckpt, monitor="val_mae", mode="min", patience=4,
                                        lr_patience=2, min_lr=1e-5),
                  verbose=2)
        train_s = time.time() - t0
        model.load_weights(ckpt)
        m = R.evaluate(test.y, R.predict_model(model, test), tbo)
        ab = R.evaluate(test.y, R.ablate_bsfc(model, test, R.load_meta(engine)["rul_aux_order"]), tbo)
        rows[name] = {
            "params": int(model.count_params()), "train_seconds": round(train_s, 1),
            "mae_h": m["mae_h"], "mae_pct": m["mae_pct"], "corr": m["corr"],
            "near_new_bias_pct": m["near_new_bias_pct"],
            "monotonic_violations": m["monotonic_violations"],
            "bsfc_ablated_mae_pct": ab["mae_pct"],
        }
        R.print_report(f"{name} (test)", m, tbo)
        print(f"  with bsfc zeroed: {ab['mae_pct']:.2f}% TBO | "
              f"{model.count_params():,} params | {train_s/60:.1f} min\n", flush=True)
        del model
        keras.backend.clear_session()

    print(f"{'variant':10s} {'MAE h':>8s} {'% TBO':>7s} {'near-new':>9s} {'corr':>7s} "
          f"{'no-bsfc':>8s} {'params':>8s} {'min':>6s}")
    print(f"{'ridge':10s} {base['mae_h']:8.1f} {base['mae_pct']:7.2f} "
          f"{base['near_new_bias_pct']:+8.2f}% {base['corr']:7.4f} {'-':>8s} {'110':>8s} {'0.1':>6s}")
    for name, r in rows.items():
        print(f"{name:10s} {r['mae_h']:8.1f} {r['mae_pct']:7.2f} {r['near_new_bias_pct']:+8.2f}% "
              f"{r['corr']:7.4f} {r['bsfc_ablated_mae_pct']:8.2f} {r['params']:8,d} "
              f"{r['train_seconds']/60:6.1f}")

    out = {"engine": engine, "key": key, "dataset": R.load_meta(engine)["dataset"],
           "epochs": args.epochs, "ridge": {k: v for k, v in base.items() if k != "table"},
           "variants": rows, "ran_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(C.MODELS_DIR, f"rul_bakeoff_{key}.json")
    json.dump(out, open(path, "w"), indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
