#!/usr/bin/env python3
"""Hyperparameter search for the joint v5 model on the 914 pilot
(docs/v5_model_improvement_plan.md, Phase 4 task 7). Optuna is not installed, so
this is random search with successive halving (the idea behind ASHA): many
configurations get a short budget, the best third get more, the best few train
to convergence. Scored on the validation flights by the composite of honest
metrics (train_v5.Trainer.score).

    validation/venv/bin/python validation_v5/hp_search_v5.py --cache validation_v5/cache/pilot_914 --trials 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pipeline_v5 import Cache  # noqa: E402
from seed import set_random_seed  # noqa: E402
from train_v5 import DEFAULT_CFG, Trainer  # noqa: E402

SPACE = {
    "channels": [48, 64, 96],
    "layers": [6, 7],
    "dropout": [0.1, 0.2, 0.3],
    "input_dropout": [0.0, 0.03, 0.08],
    "head_hidden": [64, 96, 128],
    "lr": [3e-4, 6e-4, 1e-3, 2e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "balance": ["uncertainty", "sum"],
    "batch": [64, 128],          # 256 is ~1.7x slower per window on Metal (2.2 s/step); v4 used 128
}


def sample(rng) -> dict:
    return dict(DEFAULT_CFG, **{k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--rungs", default="3,8,30", help="cumulative epoch budgets per rung")
    ap.add_argument("--out", default=os.path.join(HERE, "hp_search_v5.json"))
    a = ap.parse_args()
    rungs = [int(x) for x in a.rungs.split(",")]
    rng = np.random.default_rng(7)
    cache = Cache(a.cache)
    trials = [{"id": i, "cfg": sample(rng)} for i in range(a.trials)]
    # Resume: the draws are seeded, so a previous run's trials are the same configs;
    # keep every rung result already measured for a matching config.
    if os.path.exists(a.out):
        with open(a.out) as fh:
            old = {t["id"]: t for t in json.load(fh).get("trials", [])}
        for t in trials:
            o = old.get(t["id"])
            if o and o["cfg"] == t["cfg"]:
                t.update({k: v for k, v in o.items() if k.startswith("rung")})
    log = {"space": SPACE, "rungs": rungs, "trials": trials}

    alive = list(range(len(trials)))
    for r, budget in enumerate(rungs):
        for i in alive:
            if f"rung{r}" in trials[i]:
                print(f"rung {r} trial {i:2d} composite {trials[i][f'rung{r}']['val']['composite']:.4f} (resumed)",
                      flush=True)
                continue
            t0 = time.time()
            cfg = dict(trials[i]["cfg"], seed=1000 + i)
            set_random_seed(cfg["seed"])
            tr = Trainer(cache, cfg)
            res = tr.fit(epochs=budget, patience=max(3, budget // 3), bs=cfg["batch"], verbose=False)
            best = res["history"][res["best_epoch"]]["val"]
            trials[i][f"rung{r}"] = {"epochs": budget, "best_epoch": res["best_epoch"],
                                     "val": best, "seconds": round(time.time() - t0, 1)}
            print(f"rung {r} trial {i:2d} composite {best['composite']:.4f} "
                  f"({time.time() - t0:.0f}s)  {cfg}", flush=True)
            with open(a.out, "w") as fh:
                json.dump(log, fh, indent=1, default=float)
        keep = max(1, len(alive) // 3) if r < len(rungs) - 1 else len(alive)
        alive = sorted(alive, key=lambda i: -trials[i][f"rung{r}"]["val"]["composite"])[:keep]
    last = f"rung{len(rungs) - 1}"
    ranked = sorted([t for t in trials if last in t], key=lambda t: -t[last]["val"]["composite"])
    log["best"] = ranked[0] if ranked else None
    with open(a.out, "w") as fh:
        json.dump(log, fh, indent=1, default=float)
    print("BEST", json.dumps(log["best"], default=float))


if __name__ == "__main__":
    main()
