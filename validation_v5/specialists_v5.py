#!/usr/bin/env python3
"""One specialist network per head, assembled into one model (v5, 914 first).

WHY
    The joint model (run_v5.py) fails all six gates on the 914. On the validation
    flights its heads pull the shared encoder apart: sensor F1 rises while diagnosis
    and severity fall, and stage 1 peaks at a different epoch for each head.
    Severity predictions sit at 0 or 1 (sigmoid saturated under L1) and health is
    squeezed into 0.31-0.72 while the truth spans 0.09-0.98.

WHAT
    Five specialists, each the same proven network (model_architectures_v5), trained
    on ALL training flights with ONLY its own loss, a warm-up + cosine learning rate,
    and its epoch picked on ITS OWN validation metric:

      detection   BCE                             select  det AUC + recall@precision 0.95
      diagnosis   BCE (+ family)                  select  diagnosis macro AP + 0.5 family F1
      severity    Huber, linear output            select  -(error on faults + 0.5 error on clean)
      sensor      class-balanced cross-entropy    select  sensor macro F1
      health      Huber, linear output            select  -health error

    Then, fitted on half the calibration flights (never trained on):
      detection   a small GBT over the specialists' outputs + the largest residual,
                  kept only if it beats the network on validation recall@0.95
      health      a GBT over the network's health + hours/usage context,
                  kept only if it beats the network on validation error
      sensor      a bias on the "none" logit, trading recall for false alarms
      diagnosis   per-fault temperature and cut-offs (train_v5.Trainer.calibrate)
      severity    zeroed where calibrated diagnosis says the fault is absent (gate
                  level chosen on the same half, kept only if validation improves):
                  the 914b severity specialist left 0.044 on healthy cells (gate 0.02)
    and RUL (rul_v5) on the other half, over the assembled outputs.
    Test flights are scored once, by score_v5, against the same do-nothing baselines.

    validation/venv/bin/python validation_v5/specialists_v5.py 914
    validation/venv/bin/python validation_v5/specialists_v5.py 914 --force severity   # redo one head (and assembly)

Steps (each skipped if its output exists): train each head -> assemble -> rul -> test -> card.
Outputs: artifacts/<engine>_specialists/ ; the verdict is model_card.md there.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate as E  # noqa: E402
import run_v5  # noqa: E402
from pipeline_v5 import Cache  # noqa: E402
from assembly_v5 import (Assembly, SOURCE, apply_temperature, calibrate_severity,  # noqa: E402,F401
                         detection_features, gate_severity, health_features)
from train_v5 import HEADS, Trainer, targets  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))
from degradation_v5 import FAULT_NAMES  # noqa: E402

SPECS = {
    "detection": {"heads": ["detection"], "select": {"det_auc": 1.0, "det_recall_p95": 1.0}},
    "diagnosis": {"heads": ["diagnosis", "family"], "select": {"diag_macro_ap": 1.0, "family_f1": 0.5}},
    "severity": {"heads": ["severity"], "select": {"sev_on_fault_mae": -1.0, "sev_clean_mae": -0.5},
                 "linear": True},
    "sensor": {"heads": ["sensor"], "select": {"sensor_macro_f1": 1.0}},
    "health": {"heads": ["health"], "select": {"health_mae": -1.0}, "linear": True},
}
# lr 3e-4 with cosine decay: the joint model at a flat 6e-4 peaked by epoch 3-10 and then drifted.
BASE = {"lr": 3e-4, "lr_schedule": "cosine", "balance": "sum"}
FA_TARGET = 0.005                                   # sensor false-alarm gate


def paths(key: str) -> dict:
    p = run_v5.paths(key)
    p["art"] = p["art"] + "_specialists"
    return p


def head_config(head: str, override: str | None) -> dict:
    cfg = run_v5.training_config(override)          # the HP search winner (channels, layers, ...)
    cfg.update(BASE)
    cfg.update(SPECS[head])
    return cfg


def split_cal(cache: Cache) -> tuple[np.ndarray, np.ndarray]:
    """Calibration windows split by flight: even flights fit the stackers and the
    sensor bias, odd flights fit RUL - so RUL never learns from a stacker's
    in-sample output."""
    ids = cache.end_ids(["cal"], jitter=False)
    fl = cache.flight_of(np.asarray(cache.ends[np.sort(ids)]))[np.argsort(np.argsort(ids))]
    return ids[fl % 2 == 0], ids[fl % 2 == 1]


# ---------------------------------------------------------------------------
class Specialists(Assembly):
    """The five trained specialists (assembly_v5.Assembly) behind the Trainer
    interface score_v5 / rul_v5 use: .cache, .applicable, .model(inputs), .predict(ids)."""

    predict = Trainer.predict
    calibrate = Trainer.calibrate
    score = Trainer.score

    def __init__(self, cache: Cache, art: str):
        self.cache = cache
        self.trainers = {h: Trainer.load(cache, os.path.join(art, h)) for h in SPECS}
        self.applicable = self.trainers["detection"].applicable
        # Cut-offs keep each fault's validation recall >= 0.55: the gate floor is 0.5 on
        # test, and pure best-F1 left valve leakage at 0.47.
        self.cfg = {"min_recall": 0.55}
        self.model = self
        extra = {}
        f = os.path.join(art, "assembly.pkl")
        if os.path.exists(f):
            with open(f, "rb") as fh:
                extra = pickle.load(fh)
        super().__init__({h: t.model for h, t in self.trainers.items()}, extra)


def collect(sp: Specialists, ids: np.ndarray, bs: int = 512) -> dict:
    """Raw specialist outputs, stacking features and targets for windows `ids`."""
    acc = {"hf": [], "df": [], "sensor": [], "health": [], "det": [], "diag": [], "sev": [], "y": []}
    for i in range(0, len(ids), bs):
        seq, ctx, Er = sp.cache.batch(ids[i:i + bs])
        o = sp.raw(seq, ctx)
        y = targets(sp.cache, Er)
        acc["hf"].append(health_features(o, ctx))
        acc["df"].append(detection_features(o, seq))
        acc["sensor"].append(o["sensor"])
        acc["health"].append(o["health"][:, 0])
        acc["det"].append(o["detection"][:, 0])
        acc["diag"].append(o["diagnosis"])
        acc["sev"].append(o["severity"])
        acc["y"].append(y)
    out = {k: np.concatenate(v) for k, v in acc.items() if k != "y"}
    out["y"] = {h: np.concatenate([y[h] for y in acc["y"]]) for h in HEADS}
    return out


def tune_sensor_bias(logits: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """Bias on the 'none' logit: best macro F1, less 20 x any false-alarm rate
    above the gate's 0.5%."""
    best = None
    for b in np.linspace(-2.0, 8.0, 41):
        z = logits.copy()
        z[..., 0] += b
        rep = E.sensor_report(y, z.argmax(-1))
        obj = rep["macro_f1"] - 20.0 * max(0.0, rep["false_alarm_rate"] - FA_TARGET)
        if best is None or obj > best[0]:
            best = (obj, float(b), {"macro_f1": rep["macro_f1"], "false_alarm_rate": rep["false_alarm_rate"]})
    return best[1], best[2]


def assemble(sp: Specialists, art: str) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    cache = sp.cache
    fit_ids, _ = split_cal(cache)
    c = collect(sp, fit_ids)
    v = collect(sp, cache.end_ids(["val"], jitter=False))
    rep = {}

    # sensor: bias chosen on the calibration half, reported on validation
    b, on_cal = tune_sensor_bias(c["sensor"], c["y"]["sensor"])
    zv = v["sensor"].copy()
    zv[..., 0] += b
    before, after = E.sensor_report(v["y"]["sensor"], v["sensor"].argmax(-1)), E.sensor_report(v["y"]["sensor"], zv.argmax(-1))
    rep["sensor"] = {"bias": b, "cal": on_cal,
                     "val_before": {k: before[k] for k in ("macro_f1", "false_alarm_rate")},
                     "val_after": {k: after[k] for k in ("macro_f1", "false_alarm_rate")}}

    # health: network alone vs GBT(network health + hours / usage / long-horizon context)
    hg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=500, learning_rate=0.05,
                                       max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    hg.fit(c["hf"], c["y"]["health"][:, 0])
    yv = v["y"]["health"][:, 0]
    mae_net = E.mae(yv, v["health"])
    mae_gbt = E.mae(yv, np.clip(hg.predict(v["hf"]), 0, 1))
    rep["health"] = {"val_mae_network": mae_net, "val_mae_gbt": mae_gbt, "kept": "gbt" if mae_gbt < mae_net else "network"}

    # detection: network alone vs GBT(specialist outputs + largest residual)
    dg = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                        l2_regularization=1.0, random_state=0)
    dg.fit(c["df"], c["y"]["detection"][:, 0])
    yd = v["y"]["detection"][:, 0]
    pg = dg.predict_proba(v["df"])[:, 1]
    r_net, r_gbt = E.recall_at_precision(yd, v["det"]), E.recall_at_precision(yd, pg)
    rep["detection"] = {"val_recall_p95_network": r_net, "val_recall_p95_gbt": r_gbt,
                        "val_auc_network": E.auc(yd, v["det"]), "val_auc_gbt": E.auc(yd, pg),
                        "kept": "gbt" if r_gbt > r_net else "network"}

    sp.extra = {"sensor_bias": b,
                "health_gbt": hg if rep["health"]["kept"] == "gbt" else None,
                "detection_gbt": dg if rep["detection"]["kept"] == "gbt" else None}
    # diagnosis: temperature (calibration flights) and cut-offs (validation flights)
    rep["diagnosis"] = sp.calibrate()
    rep["severity_calibration"] = fit_severity_calibration(sp, c, v)
    sp.extra["sev_iso"] = rep["severity_calibration"].pop("iso")
    if sp.extra["sev_iso"]:                   # the gate is chosen on calibrated severities
        c["sev"], v["sev"] = calibrate_severity(c["sev"], sp.extra["sev_iso"]), calibrate_severity(v["sev"], sp.extra["sev_iso"])
    rep["severity"] = choose_severity_gate(sp, c, v, rep["diagnosis"])
    sp.extra["sev_gate"] = rep["severity"].pop("gate")
    with open(os.path.join(art, "assembly.pkl"), "wb") as fh:
        pickle.dump(sp.extra, fh)
    with open(os.path.join(art, "calibration.json"), "w") as fh:
        json.dump(rep["diagnosis"], fh, indent=1)
    return rep


def fit_severity_calibration(sp: Specialists, c: dict, v: dict) -> dict:
    """Per fault: isotonic regression of true on predicted severity over the
    calibration half's FAULT cells (true >= 0.08). Kept as a whole only if it lowers
    validation (on-fault error + |bias|)."""
    from sklearn.isotonic import IsotonicRegression
    app = [FAULT_NAMES.index(n) for n in sp.applicable]
    iso = {}
    for j in app:
        t, p = c["y"]["severity"][:, j], c["sev"][:, j]
        on = t >= E.PRESENT_SEV
        if on.sum() >= 50:
            iso[j] = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p[on], t[on])

    def cost(sev):
        r = E.severity_report(v["y"]["severity"][:, app], sev[:, app])
        return r["on_fault_mae"] + abs(r["on_fault_bias"]), r
    (c0, r0), (c1, r1) = cost(v["sev"]), cost(calibrate_severity(v["sev"], iso))
    keep = c1 < c0
    pick = lambda r: {k: r[k] for k in ("on_fault_mae", "clean_mae", "on_fault_bias")}  # noqa: E731
    return {"kept": bool(keep), "faults": len(iso), "val_before": pick(r0), "val_after": pick(r1),
            "iso": iso if keep else {}}


def choose_severity_gate(sp: Specialists, c: dict, v: dict, cal: dict) -> dict:
    """Gate = factor x each fault's diagnosis cut-off, factor on the calibration
    half by (error on faults + error on clean cells); kept only if it also lowers
    that sum on validation. Factor 0 = no gate."""
    app = [FAULT_NAMES.index(n) for n in sp.applicable]
    cuts = np.asarray(cal["cutoffs"])

    def cost(d, gate):
        sev = d["sev"] if gate is None else gate_severity(d["sev"], d["diag"], gate)
        r = E.severity_report(d["y"]["severity"][:, app], sev[:, app])
        return r["on_fault_mae"] + r["clean_mae"], r

    best = None
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        gate = {"temperature": cal["temperature"], "cut": (f * cuts).tolist()}
        cst, _ = cost(c, gate)
        if best is None or cst < best[0]:
            best = (cst, f, gate)
    _, f, gate = best
    (v0, r0), (v1, r1) = cost(v, None), cost(v, gate)
    keep = f > 0 and v1 < v0
    pick = lambda r: {k: r[k] for k in ("on_fault_mae", "clean_mae", "on_fault_bias")}  # noqa: E731
    return {"factor": f, "kept": bool(keep), "val_before": pick(r0), "val_after": pick(r1),
            "gate": gate if keep else None}


# ---------------------------------------------------------------------------
PRIMARY = [("detection", "recall_at_p95", "Detection: recall at precision 0.95", 1),
           ("diagnosis", "macro_f1", "Diagnosis: macro F1", 1),
           ("severity", "mae_on_fault", "Severity: error on real faults", -1),
           ("sensor_fault", "macro_f1", "Sensor fault: macro F1", 1),
           ("health", "mae", "Health: error", -1),
           ("rul", "mae_pct_tbo_wear_limited", "RUL: error on wear-limited engines, % of TBO", -1)]


def comparison(res: dict, joint_path: str, note: str = "") -> str:
    """Specialists vs the joint model vs the do-nothing baseline, primary metrics."""
    joint = json.load(open(joint_path))["heads"] if os.path.exists(joint_path) else {}
    lines = ["## Specialists vs the joint model (test flights)", "", *([note, ""] if note else []),
             "| Head | Specialists | Joint model | Do-nothing | Better than joint? |", "|---|---|---|---|---|"]
    for head, metric, name, sign in PRIMARY:
        s = res["heads"].get(head, {}).get(metric)
        j = joint.get(head, {}).get(metric)
        if not s:
            continue
        sv, bv = s["point"], s.get("baseline", {}).get("point")
        jv = j["point"] if j else None
        f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        better = "-" if jv is None else ("yes" if sign * (sv - jv) > 0 else "no")
        lines.append(f"| {name} | {f(sv)} | {f(jv)} | {f(bv)} | {better} |")
    return "\n".join(lines)


def training_summary(art: str) -> str:
    lines = ["## Training, per specialist (validation flights)", "",
             "| Specialist | Epochs run | Best epoch | Best selection score | Stages 2-3 score |", "|---|---|---|---|---|"]
    for h in SPECS:
        f = os.path.join(art, h, "history.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        lines.append(f"| {h} | {len(r['joint']['history'])} | {r['joint']['best_epoch']} | "
                     f"{r['joint']['best_composite']:.4f} | {r['best_composite']:.4f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine", help="912 | 914 | 915 | 916")
    ap.add_argument("--heads", default=",".join(SPECS), help="specialists to train (default: all)")
    ap.add_argument("--force", default="", help="comma list of heads or steps (assemble,rul,test,card) to redo")
    ap.add_argument("--config", default=None, help="training config JSON (default: the HP search winner)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-windows", type=int, default=20000,
                    help="windows per epoch: a fresh random draw from ALL training flights each epoch")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--verbose", type=int, default=1, choices=(0, 1, 2),
                    help="0 silent, 1 a line per epoch, 2 also training loss, best marker, quarter-epoch progress")
    a = ap.parse_args()
    force = set(filter(None, a.force.split(",")))
    p = paths(a.engine)
    os.makedirs(p["art"], exist_ok=True)
    cache = Cache(p["cache"])
    say = lambda m: print(f"[{a.engine}] {m}", flush=True)  # noqa: E731
    stale = False                                       # a retrained head makes every later step stale

    for h in [h for h in SPECS if h in a.heads.split(",")]:
        out = os.path.join(p["art"], h)
        if os.path.exists(os.path.join(out, "history.json")) and h not in force:
            say(f"{h:9s} trained - skip")
            continue
        t0 = time.time()
        cfg = head_config(h, a.config)
        say(f"{h:9s} training  lr {cfg['lr']} cosine, select {cfg['select']}")
        from seed import set_random_seed
        cfg["seed"] = set_random_seed(cfg.get("seed", 0))
        tr = Trainer(cache, cfg)
        res = tr.train_protocol(a.epochs, a.patience, cfg["batch"], max_train_windows=a.max_windows, verbose=a.verbose)
        tr.save(out)
        run_v5.dump(res, os.path.join(out, "history.json"))
        say(f"{h:9s} done in {(time.time() - t0) / 60:.1f} min, best {res['best_composite']:.4f}")
        stale = True
    if any(not os.path.exists(os.path.join(p["art"], h, "history.json")) for h in SPECS):
        say("not every specialist is trained yet - stopping before assembly")
        return

    steps = ["assemble", "rul", "test", "card"]
    outputs = {"assemble": "assembly.json", "rul": "rul_v5.pkl", "test": "results_test.json", "card": "model_card.md"}
    sp = Specialists(cache, p["art"])
    for step in steps:
        stale = stale or step in force
        if os.path.exists(os.path.join(p["art"], outputs[step])) and not stale:
            say(f"{step:9s} done - skip")
            continue
        t0 = time.time()
        say(f"{step:9s} ...")
        if step == "assemble":
            rep = assemble(sp, p["art"])
            run_v5.dump(rep, os.path.join(p["art"], "assembly.json"))
            say(f"  sensor bias {rep['sensor']['bias']:+.2f}: val F1 {rep['sensor']['val_before']['macro_f1']:.3f} -> "
                f"{rep['sensor']['val_after']['macro_f1']:.3f}, false alarms {100 * rep['sensor']['val_before']['false_alarm_rate']:.2f}% -> "
                f"{100 * rep['sensor']['val_after']['false_alarm_rate']:.2f}%")
            say(f"  health: network {rep['health']['val_mae_network']:.3f} vs GBT {rep['health']['val_mae_gbt']:.3f} -> {rep['health']['kept']}")
            say(f"  detection recall@0.95: network {rep['detection']['val_recall_p95_network']:.3f} vs GBT "
                f"{rep['detection']['val_recall_p95_gbt']:.3f} -> {rep['detection']['kept']}")
            sc = rep["severity_calibration"]
            say(f"  severity calibration ({sc['faults']} faults): val on-fault {sc['val_before']['on_fault_mae']:.3f} -> "
                f"{sc['val_after']['on_fault_mae']:.3f}, bias {sc['val_before']['on_fault_bias']:+.3f} -> "
                f"{sc['val_after']['on_fault_bias']:+.3f} -> {'kept' if sc['kept'] else 'not kept'}")
            sv = rep["severity"]
            say(f"  severity gate x{sv['factor']}: val on-fault {sv['val_before']['on_fault_mae']:.3f} -> "
                f"{sv['val_after']['on_fault_mae']:.3f}, clean {sv['val_before']['clean_mae']:.3f} -> "
                f"{sv['val_after']['clean_mae']:.3f} -> {'kept' if sv['kept'] else 'not kept'}")
        elif step == "rul":
            import rul_v5
            _, rul_ids = split_cal(cache)
            r = rul_v5.fit_rul(sp, p["art"], cal_ids=rul_ids)
            say(f"  RUL model: {r['chosen']}, val wear-limited error "
                f"{r['candidates'][r['chosen']]['wear_limited_mae_pct']:.1f}% of TBO")
        elif step == "test":
            import score_v5
            cal = json.load(open(os.path.join(p["art"], "calibration.json")))
            rul = pickle.load(open(os.path.join(p["art"], "rul_v5.pkl"), "rb"))
            run_v5.dump(score_v5.score(sp, cal, rul, n_boot=a.boot), os.path.join(p["art"], "results_test.json"))
        elif step == "card":
            import score_v5
            res = json.load(open(os.path.join(p["art"], "results_test.json")))
            relabelled = cache.contract.get("labels") == "v5b"
            joint = os.path.join(run_v5.paths(a.engine.removesuffix("b") if relabelled else a.engine)["art"],
                                 "results_test.json")
            note = ("Labels here are v5b (relabel_v5b.py); the joint model was scored on the v5 labels, "
                    "so that column is context, not a like-for-like comparison.") if relabelled else ""
            card = "\n\n".join([score_v5.model_card(res).replace("v5 test scorecard", "v5 specialists, test scorecard"),
                                comparison(res, joint, note), training_summary(p["art"])])
            with open(os.path.join(p["art"], "model_card.md"), "w") as fh:
                fh.write(card + "\n")
        stale = True
        say(f"{step:9s} finished in {(time.time() - t0) / 60:.1f} min")
    say(f"COMPLETE - {os.path.relpath(os.path.join(p['art'], 'model_card.md'), run_v5.ROOT)}")


if __name__ == "__main__":
    main()
