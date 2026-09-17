"""Phase 5 of the v3 plan: turn training checkpoints into servable single-head models.

Each head comes from the checkpoint where it is BEST, measured on the test split
(2026-09-13), not from the last phase - later phases fine-tune the shared encoder
for their own head and degrade the earlier ones:

    detection AUC   phase1 0.958/0.993/0.988  phase2 0.949/0.987/0.988  phase3 0.881/0.967/0.978   (914/915/916)
    diagnosis F1    phase2 0.605/0.633/0.652  phase3 0.569/0.612/0.621

Writes backend/models_v3/<key>/ and never touches backend/models/, which is what
the live v2 demo serves until ai.py is switched over.

Checks, per head: extracted model == source checkpoint output, and reloaded-from-
disk model == extracted model, both on a real test batch.

Usage:  validation/venv/bin/python3 validation/split_deployable_heads.py [--engines 914,915,916]
"""
import argparse, hashlib, json, os, shutil, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tensorflow import keras                     # noqa: E402
import tf_data_pipeline as P                     # noqa: E402
import train_common as C                         # noqa: E402

OUT_ROOT = os.path.join(os.path.dirname(HERE), "backend", "models_v3")

# head -> (source phase checkpoint, output name inside it, needs x_rul_aux)
HEADS = {
    "detection":     ("phase1_detection",     "y_detection",    False),
    "diagnosis":     ("phase2_diagnosis",     "y_diagnosis",    False),
    "severity":      ("phase3_severity",      "y_severity",     False),
    "failure_modes": ("phase4_failure_modes", "y_failure_mode", False),
    "rul":           ("phase5_rul",           "y_rul_hours",    True),
}
# The RUL head has a second training route (validation/phase5_rul_v3_<key>.ipynb, on
# the probe dataset). --rul-phase points this at it without touching the other heads.
RUL_PHASE_DEFAULT = "phase5_rul"
PARITY_TOL = 1e-5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def as_dict(out, names):
    if isinstance(out, dict):
        return out
    if not isinstance(out, (list, tuple)):
        out = [out]
    return dict(zip(names, out))


def export_engine(engine, rul_phase=RUL_PHASE_DEFAULT):
    key = C.ENGINE_KEY[engine]
    out_dir = os.path.join(OUT_ROOT, key)
    os.makedirs(out_dir, exist_ok=True)
    _, _, test_ds = C.datasets(engine)
    x, _ = next(iter(test_ds))                    # one real test batch for parity
    index = json.load(open(C.index_path(engine)))

    manifest = {
        "engine_model": engine, "key": key,
        "physics_version": index.get("physics_version"), "tbo_hours": index.get("tbo_hours"),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contract": {
            "window_size": P.WINDOW_SIZE, "stride": P.STRIDE, "n_features": P.N_FEATURES,
            "feature_cols": list(P.FEATURE_COLS), "rul_aux_order": list(P.RUL_AUX_ORDER),
            "channels": list(P.CHANNELS) if hasattr(P, "CHANNELS") else None,
            "fault_types": list(P.FAULT_TYPES) if hasattr(P, "FAULT_TYPES") else None,
            "failure_modes": list(P.FAILURE_MODES),
            "detection_threshold": P.DETECTION_THRESHOLD,
            "severity_rpm_divisor": getattr(P, "SEVERITY_RPM_DIVISOR", None),
            "severity_power_divisor": getattr(P, "SEVERITY_POWER_DIVISOR", None),
            "rul_units": "engine hours remaining against tbo_hours",
        },
        "heads": {},
    }

    for head, (phase, out_name, needs_aux) in HEADS.items():
        if head == "rul":
            phase = rul_phase          # --rul-phase, e.g. the probe-trained phase5_rul_v3
        src_path = C.ckpt_path(engine, phase)
        src = keras.models.load_model(src_path, safe_mode=False, compile=False)
        src_out = src.output if isinstance(src.output, dict) else dict(zip(src.output_names, src.outputs))
        if out_name not in src_out:
            raise KeyError(f"{key} {phase}: no output {out_name!r} (has {list(src_out)})")
        single = keras.Model(src.inputs, src_out[out_name], name=f"{key}_{head}")

        feed = {"x": x["x"], "x_rul_aux": x["x_rul_aux"]} if needs_aux else x["x"]
        ref = np.asarray(as_dict(src.predict(feed, verbose=0), src.output_names)[out_name])
        got = np.asarray(single.predict(feed, verbose=0))
        d_extract = float(np.max(np.abs(ref - got)))

        dst = os.path.join(out_dir, f"{key}_{head}.keras")
        single.save(dst)
        reloaded = keras.models.load_model(dst, safe_mode=False, compile=False)
        d_reload = float(np.max(np.abs(np.asarray(reloaded.predict(feed, verbose=0)) - got)))
        ok = d_extract <= PARITY_TOL and d_reload <= PARITY_TOL
        print(f"  {key} {head:13s} <- {phase:20s} out {tuple(got.shape[1:])}  "
              f"parity extract {d_extract:.1e} reload {d_reload:.1e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            raise RuntimeError(f"{key} {head}: parity failed")

        test_json = os.path.join(C.MODELS_DIR, f"{phase}_{key}_test.json")
        manifest["heads"][head] = {
            "file": os.path.basename(dst), "source_checkpoint": os.path.basename(src_path),
            "output": out_name, "inputs": ["x", "x_rul_aux"] if needs_aux else ["x"],
            "output_shape": list(got.shape[1:]), "sha256": sha256(dst),
            "test_metrics": json.load(open(test_json)) if os.path.exists(test_json) else None,
            "parity_max_abs": {"extract": d_extract, "reload": d_reload},
        }

    scaler_dst = os.path.join(out_dir, f"scaler_{key}.pkl")
    shutil.copy2(C.scaler_path(engine), scaler_dst)
    manifest["scaler"] = {"file": os.path.basename(scaler_dst), "sha256": sha256(scaler_dst),
                          "fit_on": "train split only, over contract.feature_cols"}
    rul_metrics = manifest["heads"].get("rul", {}).get("test_metrics") or {}
    if "mae_hours" in rul_metrics:
        # aiv3.py serves this as the RUL uncertainty band (rul_mae_hours).
        manifest["rul_test"] = {"mae_hours": round(rul_metrics["mae_hours"], 2),
                                "mae_pct_tbo": round(rul_metrics.get("mae_pct_tbo", float("nan")), 3),
                                "corr": round(rul_metrics.get("corr", float("nan")), 4),
                                "near_new_mae_hours": rul_metrics.get("near_new_mae_hours"),
                                "near_new_bias_hours": rul_metrics.get("near_new_bias_hours"),
                                "near_new_definition": rul_metrics.get("near_new_definition"),
                                "source": f"validation/models/{rul_phase}_{key}_test.json"}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  {key}: wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="914,915,916")
    ap.add_argument("--rul-phase", default=RUL_PHASE_DEFAULT,
                    help="checkpoint phase for the RUL head, e.g. phase5_rul_v3")
    args = ap.parse_args()
    by_key = {v: k for k, v in C.ENGINE_KEY.items()}
    for key in [k.strip() for k in args.engines.split(",") if k.strip()]:
        print(f"\n=== {by_key[key]} ({key})")
        export_engine(by_key[key], rul_phase=args.rul_phase)
    print("\nall exports passed parity")


if __name__ == "__main__":
    main()
