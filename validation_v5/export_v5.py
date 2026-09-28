#!/usr/bin/env python3
"""Export trained v5 specialists for serving: backend/models_v5/<engine>/.

    validation/venv/bin/python validation_v5/export_v5.py 914b          # -> backend/models_v5/914/

Writes
    <specialist>/model_v5.weights.h5, config_v5.json   the five specialists (weights only:
                                                       the network is rebuilt from
                                                       model_architectures_v5, so no custom
                                                       layer is ever deserialised)
    assembly.pkl, calibration.json, rul_v5.pkl         the fitted assembly and RUL model
    contract_v5.json                                   feature order, scalers, label version
    manifest.json                                      engine, versions, git SHA, test metrics

Serving loads it with assembly_v5.Deployed, which keeps only each specialist's own
heads (a slim view of the network). tests/test_export_parity_v5.py checks the export
against the training-time Specialists on real test windows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_v5  # noqa: E402
from assembly_v5 import HEADS_OF  # noqa: E402

ROOT = os.path.dirname(HERE)
FILES = ("assembly.pkl", "calibration.json", "rul_v5.pkl")
PRIMARY = {"detection": "recall_at_p95", "diagnosis": "macro_f1", "severity": "mae_on_fault",
           "sensor_fault": "macro_f1", "health": "mae", "rul": "mae_pct_tbo_wear_limited"}


def engine_key(run_key: str) -> str:
    """914b -> 914: the run tag carries the label version, the export does not."""
    return run_key.rstrip("b")


def export(run_key: str, out_root: str | None = None) -> str:
    src = run_v5.paths(run_key)
    art = src["art"] + "_specialists"
    out = os.path.join(out_root or os.path.join(ROOT, "backend", "models_v5"), engine_key(run_key))
    for need in [*(os.path.join(art, h, "model_v5.weights.h5") for h in HEADS_OF),
                 *(os.path.join(art, f) for f in FILES), os.path.join(art, "results_test.json")]:
        if not os.path.exists(need):
            raise SystemExit(f"missing {need} - run specialists_v5.py {run_key} to the end first")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for h in HEADS_OF:
        os.makedirs(os.path.join(out, h))
        for f in ("model_v5.weights.h5", "config_v5.json"):
            shutil.copy(os.path.join(art, h, f), os.path.join(out, h, f))
    for f in FILES:
        shutil.copy(os.path.join(art, f), os.path.join(out, f))
    with open(os.path.join(src["cache"], "contract_v5.json")) as fh:
        contract = json.load(fh)
    with open(os.path.join(out, "contract_v5.json"), "w") as fh:
        json.dump(contract, fh, indent=1)
    res = json.load(open(os.path.join(art, "results_test.json")))
    sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    import sklearn
    import tensorflow as tf
    manifest = {
        "engine_model": contract["engine_model"], "key": engine_key(run_key), "model_version": "v5",
        "labels": contract.get("labels", "v5"), "run": run_key, "git_sha": sha,
        "exported": dt.datetime.now().isoformat(timespec="seconds"),
        "tbo_hours": contract["tbo_hours"], "window": contract["window"],
        "feature_cols": contract["feature_cols"], "ctx_cols": contract["ctx_cols"],
        "specialists": {h: heads for h, heads in HEADS_OF.items()},
        "versions": {"tensorflow": tf.__version__, "sklearn": sklearn.__version__},
        "test": {"flights": res["test_flights"], "windows": res["test_windows"],
                 "primary": {h: res["heads"][h][m]["point"] for h, m in PRIMARY.items() if h in res["heads"]},
                 "gates": {h: g["pass"] for h, g in res["gates"].items()}},
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="specialists run key, e.g. 914b")
    ap.add_argument("--out-root", default=None, help="default: backend/models_v5")
    a = ap.parse_args()
    print(export(a.run, a.out_root))
