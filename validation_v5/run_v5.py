#!/usr/bin/env python3
"""Run the v5 training pipeline for one engine, step by step, resuming where it stopped.

    validation/venv/bin/python validation_v5/run_v5.py 914                  # every step still to do
    validation/venv/bin/python validation_v5/run_v5.py 914 --steps train     # one step
    validation/venv/bin/python validation_v5/run_v5.py 914 --force train     # redo a step and every step after it
    validation/venv/bin/python validation_v5/run_v5.py all                   # 914, 912, 915, 916 in turn

STEPS (each writes one file; a step whose file exists is skipped)
    checks     data_checks_v5 on data/rotax_v5/<key>     artifacts/<key>/data_checks.json
    cache      window cache                              cache/<key>/contract_v5.json
    train      joint model, best epoch on validation     artifacts/<key>/model_v5.weights.h5
    calibrate  temperatures (cal), cut-offs (val)        artifacts/<key>/calibration.json
    rul        RUL model on the joint model's outputs    artifacts/<key>/rul_v5.pkl
    test       test scorecard + gates, ONCE              artifacts/<key>/results_test.json
    card       Markdown model card                       artifacts/<key>/model_card.md

The training config is, in order: --config, the HP search winner
(hp_search_v5.json), train_v5.DEFAULT_CFG.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ENGINES = ["914", "912", "915", "916"]
STEPS = ["checks", "cache", "train", "calibrate", "rul", "test", "card"]
OUTPUT = {"checks": "data_checks.json", "train": "model_v5.weights.h5", "calibrate": "calibration.json",
          "rul": "rul_v5.pkl", "test": "results_test.json", "card": "model_card.md"}


def paths(key: str) -> dict:
    return {"data": os.path.join(ROOT, "data", "rotax_v5", key),
            "cache": os.path.join(HERE, "cache", key),
            "art": os.path.join(HERE, "artifacts", key)}


def done(step: str, p: dict) -> bool:
    if step == "cache":
        return os.path.exists(os.path.join(p["cache"], "contract_v5.json"))
    if step == "checks":                       # a failed check must run again
        f = os.path.join(p["art"], OUTPUT[step])
        return os.path.exists(f) and json.load(open(f)).get("pass", False)
    return os.path.exists(os.path.join(p["art"], OUTPUT[step]))


def training_config(explicit: str | None) -> dict:
    from train_v5 import DEFAULT_CFG
    if explicit:
        with open(explicit) as fh:
            return dict(DEFAULT_CFG, **json.load(fh))
    hp = os.path.join(HERE, "hp_search_v5.json")
    if os.path.exists(hp):
        with open(hp) as fh:
            best = json.load(fh).get("best")
        if best:
            return dict(DEFAULT_CFG, **best["cfg"])
    return dict(DEFAULT_CFG)


def dump(obj, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


def run_step(step: str, p: dict, a) -> None:
    # Imports are per step: checks / cache do not need TensorFlow.
    if step == "checks":
        import data_checks_v5
        ok = data_checks_v5.run(p["data"])
        dump({"pass": ok, "time": time.strftime("%Y-%m-%d %H:%M")}, os.path.join(p["art"], OUTPUT[step]))
        if not ok:
            raise SystemExit(f"data checks FAILED for {p['data']} - fix the data before training")
        return
    if step == "cache":
        from pipeline_v5 import build_cache
        print(build_cache(p["data"], p["cache"]), flush=True)
        return

    from pipeline_v5 import Cache
    from train_v5 import Trainer
    cache = Cache(p["cache"])
    if step == "train":
        from seed import set_random_seed
        cfg = training_config(a.config)
        cfg["seed"] = set_random_seed(cfg.get("seed", 0))
        tr = Trainer(cache, cfg)
        res = tr.train_protocol(a.epochs, a.patience, cfg["batch"], max_train_windows=a.max_windows)
        tr.save(p["art"])
        dump(res, os.path.join(p["art"], "history.json"))
        return

    tr = Trainer.load(cache, p["art"])
    if step == "calibrate":
        dump(tr.calibrate(), os.path.join(p["art"], OUTPUT[step]))
    elif step == "rul":
        import rul_v5
        print(json.dumps(rul_v5.fit_rul(tr, p["art"])["chosen"]), flush=True)
    elif step == "test":
        import score_v5
        with open(os.path.join(p["art"], "calibration.json")) as fh:
            cal = json.load(fh)
        rul_path = os.path.join(p["art"], "rul_v5.pkl")
        rul = pickle.load(open(rul_path, "rb")) if os.path.exists(rul_path) else None
        dump(score_v5.score(tr, cal, rul, n_boot=a.boot), os.path.join(p["art"], OUTPUT[step]))
    elif step == "card":
        import score_v5
        with open(os.path.join(p["art"], "results_test.json")) as fh:
            res = json.load(fh)
        with open(os.path.join(p["art"], OUTPUT[step]), "w") as fh:
            fh.write(score_v5.model_card(res))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine", help="912 | 914 | 915 | 916 | all")
    ap.add_argument("--steps", default=",".join(STEPS), help="comma list (default: all)")
    ap.add_argument("--force", default="", help="comma list of steps to redo (with every later step)")
    ap.add_argument("--config", default=None, help="training config JSON")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--max-windows", type=int, default=None, help="cap training windows per epoch (smoke runs)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws for the test CIs")
    ap.add_argument("--data", default=None, help="override data dir (e.g. data/rotax_v5_pilot/914)")
    ap.add_argument("--tag", default=None, help="override the cache / artifact name (e.g. pilot_914)")
    a = ap.parse_args()
    steps = [s for s in STEPS if s in a.steps.split(",")]
    force = set(filter(None, a.force.split(",")))
    for key in ENGINES if a.engine == "all" else [a.engine]:
        p = paths(a.tag or key)
        if a.data:
            p["data"] = os.path.abspath(a.data)
        os.makedirs(p["art"], exist_ok=True)
        redo = False
        for step in steps:
            redo = redo or step in force           # a redone step makes every later output stale
            if done(step, p) and not redo:
                print(f"[{key}] {step:9s} done - skip", flush=True)
                continue
            t0 = time.time()
            print(f"[{key}] {step:9s} ...", flush=True)
            run_step(step, p, a)
            print(f"[{key}] {step:9s} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
