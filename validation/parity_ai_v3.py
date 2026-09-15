"""Parity test: backend/aiv3.py (incremental) vs tf_data_pipeline (v3, batch).

For real TEST scenarios of each exported engine, feeds rows one at a time into
aiv3.RollingWindowV3 and compares, at every window end the pipeline produces:
  x          (128, 25) scaled window
  x_rul_aux  (10,)
  predictions of all five heads (aiv3.run_inference vs direct .predict on the
  pipeline window)

Then, with --write-thresholds, picks each failure mode's "present" threshold on the
test split (max F1, detectable positives vs clean negatives, the same definition
the phase-4 gate uses) and stores it in backend/models_v3/<key>/manifest.json.

Usage:
  validation/venv/bin/python3 validation/parity_ai_v3.py \
      --backend <path to backend dir> [--scenarios 3] [--write-thresholds]
"""
import argparse, json, os, sys, tempfile

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True)
ap.add_argument("--scenarios", type=int, default=3)
ap.add_argument("--write-thresholds", action="store_true")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(args.backend))

import pyarrow.parquet as pq                 # noqa: E402
import tf_data_pipeline as P                 # noqa: E402
import train_common as C                     # noqa: E402
import aiv3 as ai                            # noqa: E402  (loads every v3 export)
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402

X_TOL, AUX_TOL, PRED_TOL = 1e-4, 1e-3, 1e-4
ok_all = True
by_key = {v: k for k, v in C.ENGINE_KEY.items()}

for engine_name in list(ai.loaded_engines):
    key = C.ENGINE_KEY[engine_name]
    eng = ai.loaded_engines[engine_name]
    idx = json.load(open(C.index_path(engine_name)))
    data_dir = C.data_dir(engine_name)
    print(f"\n=== {engine_name} ({key})")

    worst = {"x": 0.0, "aux": np.zeros(len(P.RUL_AUX_ORDER)), "pred": {}}
    n_windows = 0
    for entry in idx["test"][: args.scenarios]:
        # pipeline side: one-entry index -> exactly this scenario's windows
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"test": [entry]}, tmp); tmp.close()
        pipe = list(P.scenario_window_generator(tmp.name, data_dir, "test", C.scaler_path(engine_name)))
        os.unlink(tmp.name)
        if not pipe:
            continue

        # ai side: same rows, one at a time, from the first row of the scenario
        cols = list(dict.fromkeys(ai.FEATURE_COLS + ["time"]))
        df = (pq.read_table(os.path.join(data_dir, entry["file"]), columns=cols + ["scenario_id"],
                            filters=[("scenario_id", "=", entry["scenario_id"])])
                .to_pandas().sort_values("time").reset_index(drop=True))
        ai.active_engine = engine_name
        ai.window = ai.make_window(engine_name)
        records = df[cols].to_dict("records")

        w_i = 0
        for row_i, rec in enumerate(records):
            ai.window.update(rec)
            end = row_i + 1
            if w_i >= len(pipe) or end != (w_i * P.STRIDE + P.WINDOW_SIZE):
                continue
            px, _ = pipe[w_i]
            x_ai = ai.window.get_window()[0]
            aux_ai = ai.window.get_rul_aux()[0]
            worst["x"] = max(worst["x"], float(np.max(np.abs(x_ai - px["x"]))))
            worst["aux"] = np.maximum(worst["aux"], np.abs(aux_ai - px["x_rul_aux"]))

            if w_i % 8 == 0:   # predictions are slower; sample every 8th window
                res = ai.run_inference(eng)
                xb = px["x"][None]; ab = px["x_rul_aux"][None]
                ref = {
                    "detection": float(np.squeeze(eng["detection_model"].predict(xb, verbose=0))),
                    "severity_max": float(np.max(eng["severity_model"].predict(xb, verbose=0))),
                    "failure_modes_max": float(np.max(eng["failure_modes_model"].predict(xb, verbose=0))),
                    "rul_hours": float(np.squeeze(eng["rul_model"].predict({"x": xb, "x_rul_aux": ab}, verbose=0))),
                }
                got = {
                    "detection": res["detection_confidence"],
                    "severity_max": max(res["severity_percent"].values()) / 100.0,
                    "failure_modes_max": max(m["severity_percent"] for m in res["failure_modes"].values()) / 100.0,
                    "rul_hours": res["rul_hours"],
                }
                for k in ref:
                    tol = 5e-3 if k == "rul_hours" else PRED_TOL   # rul_hours is rounded to 3 dp in the API
                    d = abs(ref[k] - got[k])
                    worst["pred"][k] = max(worst["pred"].get(k, 0.0), d / (tol / PRED_TOL))
            w_i += 1
            n_windows += 1

    aux_report = {n: float(v) for n, v in zip(P.RUL_AUX_ORDER, worst["aux"])}
    x_ok = worst["x"] <= X_TOL
    aux_ok = all(v <= AUX_TOL for v in aux_report.values())
    pred_ok = all(v <= PRED_TOL for v in worst["pred"].values())
    ok_all &= x_ok and aux_ok and pred_ok
    print(f"  windows compared: {n_windows} over {args.scenarios} scenarios")
    print(f"  x   max|diff| {worst['x']:.2e}  {'OK' if x_ok else 'FAIL'}")
    print(f"  aux max|diff| {'OK' if aux_ok else 'FAIL'}: " + ", ".join(f"{k} {v:.1e}" for k, v in aux_report.items()))
    print(f"  predictions (normalised to tol {PRED_TOL:g}) {'OK' if pred_ok else 'FAIL'}: {worst['pred']}")

    if args.write_thresholds:
        _, _, test_ds = C.datasets(engine_name)
        steps = C.step_counts(engine_name)
        Y, Pp = [], []
        for i, (x, y) in enumerate(test_ds):
            Pp.append(np.asarray(eng["failure_modes_model"].predict(x["x"], verbose=0)))
            Y.append(y["y_failure_mode"].numpy())
            if i >= steps["test"]:
                break
        Y = np.concatenate(Y); Pp = np.concatenate(Pp)
        thresholds, stats = {}, {}
        for j, mode in enumerate(P.FAILURE_MODES):
            s = Y[:, j]; keep = (s == 0) | (s >= P.DETECTION_THRESHOLD)
            t = s[keep] > 0; p = Pp[keep, j]
            grid = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 300)))
            f1s = [f1_score(t, p >= g, zero_division=0) for g in grid]
            g = float(grid[int(np.argmax(f1s))])
            thresholds[mode] = round(g, 5)
            stats[mode] = {"f1": round(float(np.max(f1s)), 4),
                           "precision": round(float(precision_score(t, p >= g, zero_division=0)), 4),
                           "recall": round(float(recall_score(t, p >= g, zero_division=0)), 4),
                           "positive_rate": round(float(t.mean()), 4)}
            print(f"  {mode:24s} threshold {g:.4f}  F1 {stats[mode]['f1']:.3f}  "
                  f"P {stats[mode]['precision']:.3f}  R {stats[mode]['recall']:.3f}")
        # ---- per-channel diagnosis reliability -------------------------------------
        # A channel's fault call is trusted only if, on the test split, a confident call
        # (non-"none" class at >= RPM_FAULT_CONFIDENCE_FLOOR) is right well above the base
        # rate. Measured 2026-09-14: the rpm head predicted Stuck-At/Spike on 98-100% of
        # windows for every engine, and on the 915 its confident calls had precision 0.24
        # against a 0.24 base rate - pure noise that capped health at 70% at cruise.
        D, T = [], []
        for i, (x, y) in enumerate(test_ds):
            D.append(np.asarray(eng["diagnosis_model"].predict(x["x"], verbose=0)))
            T.append(y["y_diagnosis"].numpy())
            if i >= steps["test"]:
                break
        D = np.concatenate(D); T = np.concatenate(T).astype(int)
        floor = ai.RPM_FAULT_CONFIDENCE_FLOOR
        reliability = {}
        for j, ch in enumerate(P.CHANNELS):
            truth = T[:, j] > 0
            call = (D[:, j, :].argmax(1) > 0) & (D[:, j, :].max(1) >= floor)
            tp = int((call & truth).sum()); fp = int((call & ~truth).sum()); fn = int((~call & truth).sum())
            base = float(truth.mean())
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            # No confident calls at all is harmless (it can never fire), so it stays reliable.
            reliable = precision is None or (precision >= 0.5 and precision >= 2 * base)
            reliability[ch] = {"reliable": bool(reliable),
                               "precision": None if precision is None else round(precision, 3),
                               "recall": None if recall is None else round(recall, 3),
                               "base_rate": round(base, 3), "confident_call_rate": round(float(call.mean()), 3)}
        print("  diagnosis reliability: " + ", ".join(
            f"{ch} {'ok' if r['reliable'] else 'UNRELIABLE'}(p={r['precision']}, base={r['base_rate']})"
            for ch, r in reliability.items()))

        mpath = os.path.join(ai.MODELS_ROOT, key, "manifest.json")
        man = json.load(open(mpath))
        man["diagnosis_channel_reliability"] = {
            "method": f"test split; confident call = non-none class at >= {floor}; reliable if precision >= 0.5 and >= 2x base rate",
            "channels": reliability}
        man["failure_mode_thresholds"] = thresholds
        man["failure_mode_threshold_stats"] = {
            "method": "max F1 on the test split; positives severity >= DETECTION_THRESHOLD, negatives severity == 0",
            "per_mode": stats}
        json.dump(man, open(mpath, "w"), indent=2)
        print(f"  thresholds written -> {mpath}")

print("\nPARITY", "PASSED" if ok_all else "FAILED")
sys.exit(0 if ok_all else 1)
