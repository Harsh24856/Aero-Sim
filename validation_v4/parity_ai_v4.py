#!/usr/bin/env python3
"""Parity: backend/aiv4.py sees exactly what training saw.

Real test flights are fed through aiv4.RollingWindowV4 one row at a time - the way
main.py feeds it live - and at every window end compared with the training pipeline
(tf_data_pipeline.scenario_window_generator) on the same rows:

  window      the scaled 128 x 29 input must be IDENTICAL
  aux         aiv4.rul_aux given the label inputs must reproduce the pipeline's aux
              vector to 1e-5 (so the only difference live is the deliberate one:
              predicted health / severity and the measured margin)
  heads       the five window heads must give IDENTICAL predictions on both

    ../validation/venv/bin/python3 parity_ai_v4.py --engine 914
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))


import numpy as np                        # noqa: E402
import pyarrow.parquet as pq              # noqa: E402

import aiv4                               # noqa: E402  (loads the exported models)
import tf_data_pipeline as P              # noqa: E402
import train_common as C                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="914")
    ap.add_argument("--scenarios", type=int, default=4)
    args = ap.parse_args()
    os.chdir(HERE)
    e = args.engine
    name = next(n for n, k in aiv4.ENGINE_REGISTRY.items() if k == e)
    eng = aiv4.loaded_engines[name]
    if eng["placeholder"]:
        sys.exit(f"{e} has no export of its own yet - nothing to check")
    d = C.data_dir(e)
    entries = json.load(open(os.path.join(d, "index_test.json")))[:args.scenarios]
    n_windows = sum(max(0, (x["n_rows"] - P.WINDOW_SIZE) // P.STRIDE + 1) for x in entries)
    gen = P.scenario_window_generator(d, "test", C.scaler_path(e), C.aux_scaler_path(e))
    reference = [next(gen) for _ in range(n_windows)]

    cols = P.FEATURE_COLS + ["t", "engine_hours", "life_used_hours", "health_index", "margin_min"] + P.FM_COLS
    worst_x = worst_aux = worst_pred = 0.0
    k = 0
    for entry in entries:
        df = (pq.read_table(os.path.join(d, "test", entry["file"]), columns=cols,
                            filters=[("scenario_id", "=", entry["scenario_id"])])
              .to_pandas().sort_values("t").reset_index(drop=True))
        w = aiv4.RollingWindowV4(eng["scaler"])
        for i, row in enumerate(df.to_dict("records")):
            w.update(row)
            if i + 1 < P.WINDOW_SIZE or (i + 1 - P.WINDOW_SIZE) % P.STRIDE:
                continue
            ref_in, _ = reference[k]
            k += 1
            x = w.get_window()
            worst_x = max(worst_x, float(np.abs(x[0] - ref_in["x"]).max()))
            fm = np.array([row[c] for c in P.FM_COLS], np.float32)
            aux = aiv4.rul_aux(row, eng["tbo_hours"], row["health_index"], fm, row["margin_min"])
            aux = eng["aux_scaler"].transform(aux.astype(np.float64)).astype(np.float32)
            worst_aux = max(worst_aux, float(np.abs(aux[0] - ref_in["x_rul_aux"]).max()))
            for phase in ("detection", "diagnosis", "severity", "sensor_fault", "health"):
                a = aiv4._head(eng, phase, {"x": x, "x_rul_aux": aiv4._a})
                b = aiv4._head(eng, phase, {"x": ref_in["x"][None], "x_rul_aux": aiv4._a})
                worst_pred = max(worst_pred, float(np.abs(a - b).max()))
    ok = k == n_windows and worst_x == 0.0 and worst_aux <= 1e-5 and worst_pred == 0.0
    print(f"{e}: {k} windows from {len(entries)} test flights | window max diff {worst_x:.2e} | "
          f"aux max diff {worst_aux:.2e} | head max diff {worst_pred:.2e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
