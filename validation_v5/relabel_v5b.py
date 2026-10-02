#!/usr/bin/env python3
"""Labels v5b: the v5 labels with three root-cause fixes, written as a new cache.

Found on the 914 specialists (2026-09-27), each measured on held-out flights:

1. SEVERITY WAS NOT IDENTIFIABLE. `fm_<fault>` is the component's damage s, but the
   engine (and every instrument) sees depth x s, and depth is drawn per engine as
   spec_depth x U(0.55, 1.0). Knowing the fault AND its exact noise-free effect still
   leaves a 0.098 error (gate 0.12; prop_erosion 0.213). v5b severity is the
   EFFECTIVE damage, s x depth / spec_depth: the fraction of the fault's full
   characteristic effect the engine actually shows - what the data can support.
2. ENGINE FAULTS LABELLED VISIBLE TOO EARLY (NOT fixed by default). A fault counts
   as present once its noise-free effect reaches 1 sigma on ANY instrument - judged
   against the same engine without the fault, which no model sees. Healthy engines
   already sit 3-31 sigma off the twin on fuel flow / rpm / oil pressure (point 3),
   so the right test is per channel against that spread. The data stores only the
   max over channels (effect_z), so a proper fix needs the generator to store
   per-channel effects and a regeneration; --z-fault raises the single threshold as
   a stopgap, default 1.0 = the v5 rule.
3. SENSOR BIAS / DRIFT LABELLED VISIBLE TOO EARLY. A bias counted from onset at any
   size (8 x severity sigma), a drift from 1 sigma. But the model sees measured minus
   a NEW-engine twin, and on HEALTHY 914 windows that residual already sits 10.6 sigma
   off (median) on oil pressure, 5.8 on rpm, 3.4 on fuel flow - normal wear varies
   between engines far more than sensor noise. v5b: a bias / drift is visible once its
   offset exceeds what healthy engines show on THAT channel - the 99th percentile of
   the healthy window-mean residual on training flights - and never below --z-sensor.

Features (X.npy) are unchanged and shared by link; only labels are rewritten.

    validation/venv/bin/python validation_v5/relabel_v5b.py 914
    -> validation_v5/cache/914b   (train with: specialists_v5.py 914b)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

import run_v5  # noqa: E402
from degradation_v4 import FAULT_MODES  # noqa: E402
from degradation_v5 import FAULT_NAMES  # noqa: E402
from pipeline_v5 import END_COLS, END_STEP, WINDOW, Cache, _read_flights  # noqa: E402
from sensors_v5 import FAULTABLE_CHANNELS, SENSOR_FAULT_TYPES  # noqa: E402

PRESENT = 0.08                                  # evaluate.PRESENT_SEV / generator FAULT_PRESENT_SEV


def sensor_offset_sigma(kind: str, severity: float, age_s: np.ndarray) -> np.ndarray:
    """Offset a bias / drift fault has put on its channel, in noise sigmas
    (sensors_v5.SensorSuite.read)."""
    if kind == "bias":
        return np.full(age_s.shape, 8.0 * severity)
    return (1.0 + 3.0 * severity) * np.maximum(age_s, 0.0) / 600.0


def healthy_offset_q99(cache: Cache, n: int = 20000, channels: list | None = None) -> dict:
    """Per faultable channel: 99th percentile of |window-mean residual| (noise sigmas)
    on healthy TRAIN windows - how far a healthy engine's reading sits from the twin."""
    from pipeline_v5 import RES_SLICE
    import features_v5 as F
    ids = cache.end_ids(["train"], jitter=False)
    ids = np.sort(np.random.default_rng(0).choice(ids, min(n, len(ids)), replace=False))
    sc = cache.contract["scaler"]
    mean, std = np.array(sc["mean"])[RES_SLICE], np.array(sc["std"])[RES_SLICE]
    wm, healthy = [], []
    for i in range(0, len(ids), 2048):
        seq, _, E = cache.batch(ids[i:i + 2048])
        wm.append(np.abs((seq[:, :, RES_SLICE] * std + mean).mean(1)))
        sf = np.stack([cache.labels(E, f"sf_{c}_flag") for c in FAULTABLE_CHANNELS], 1)
        healthy.append((cache.labels(E, "fault_present") == 0) & (sf == 0).all(1))
    wm = np.concatenate(wm)[np.concatenate(healthy)]
    return {c: float(np.quantile(wm[:, F.RESIDUAL_COLS.index(f"res_{c}")], 0.99))
            for c in (channels or FAULTABLE_CHANNELS)}


def relabel_flight(g, ends_local: np.ndarray, faults: list, sensor_faults: list, old: dict,
                   z_fault: float, z_sensor, z_effect: dict | None = None) -> dict:
    """New label columns for one flight's window ends. `old`: current columns
    (name -> array over those ends). z_sensor: one threshold, or {channel: threshold}.
    z_effect: {instrument: threshold}; with it, and data that carries the per-channel
    eff_<channel> columns (generate_dataset_v5 since 2026-10-02), an engine fault is
    visible when its effect on SOME instrument clears THAT instrument's threshold."""
    if z_effect and all(f"eff_{c}" in g.columns for c in z_effect):
        eff_rows = np.stack([np.abs(g[f"eff_{c}"].to_numpy()[ends_local]) / z_effect[c] for c in z_effect], 1)
        z = eff_rows.max(1) * z_fault                 # >= z_fault exactly when one channel clears its own bar
    else:
        z = g.effect_z.to_numpy()[ends_local]
    t = g.t.to_numpy()[ends_local]
    new = {}
    mult = {f["name"]: f["depth"] / FAULT_MODES[f["name"]]["depth"] for f in faults}
    visible = z >= z_fault
    any_fault = np.zeros(len(ends_local), bool)
    for n in FAULT_NAMES:
        eff = old[f"fm_{n}"] * mult.get(n, 1.0)
        new[f"fm_{n}"] = eff
        present = (eff >= PRESENT) & visible
        new[f"fmv_{n}"] = present.astype(np.float32)
        any_fault |= present
    new["fault_present"] = any_fault.astype(np.float32)
    for c in FAULTABLE_CHANNELS:
        flag = old[f"sf_{c}_flag"].copy()
        for f in sensor_faults:
            if f["channel"] != c or f["kind"] not in ("bias", "drift"):
                continue
            k = SENSOR_FAULT_TYPES.index(f["kind"])
            zc = z_sensor[c] if isinstance(z_sensor, dict) else z_sensor
            small = sensor_offset_sigma(f["kind"], f["severity"], t - f["onset_s"]) < zc
            flag[(flag == k) & small] = 0
        new[f"sf_{c}_flag"] = flag
    new["sensor_fault_any"] = np.stack([new[f"sf_{c}_flag"] for c in FAULTABLE_CHANNELS], 1).any(1).astype(np.float32)
    return new


def build(key: str, z_fault: float, z_sensor: float, per_channel_faults: bool = False) -> dict:
    p = run_v5.paths(key)
    dst = p["cache"] + "b"
    os.makedirs(dst, exist_ok=True)
    src = Cache(p["cache"])
    for f in ("flights.parquet",):
        shutil.copy(os.path.join(p["cache"], f), os.path.join(dst, f))
    x_link = os.path.join(dst, "X.npy")
    if not os.path.exists(x_link):
        os.symlink(os.path.realpath(os.path.join(p["cache"], "X.npy")), x_link)
    ends = np.lib.format.open_memmap(os.path.join(dst, "ends.npy"), mode="w+", dtype=np.float32,
                                     shape=src.ends.shape)
    ends[:] = src.ends[:]
    col = {c: i for i, c in enumerate(END_COLS)}
    z_ch = {c: max(z_sensor, q) for c, q in healthy_offset_q99(src).items()}
    import features_v5 as F
    z_eff = ({c: max(z_sensor, q) for c, q in healthy_offset_q99(src, channels=list(F.RESIDUAL_CHANNELS)).items()}
             if per_channel_faults else None)
    touched = ["fault_present", "sensor_fault_any"] + [f"fm_{n}" for n in FAULT_NAMES] \
        + [f"fmv_{n}" for n in FAULT_NAMES] + [f"sf_{c}_flag" for c in FAULTABLE_CHANNELS]
    before = {k: 0.0 for k in ("fmv", "sf")}
    after = dict(before)
    for fi, (r, g) in enumerate(_read_flights(p["data"])):
        fr = src.flights.iloc[fi]
        g = g.sort_values("t")
        ends_local = np.arange(WINDOW - 1, len(g), END_STEP)
        sl = slice(int(fr.end0), int(fr.end0 + fr.n_ends))
        assert len(ends_local) == fr.n_ends, f"flight {fi}: cache/raw mismatch"
        old = {k: np.array(ends[sl, col[k]]) for k in touched}      # copies: ends is written below
        new = relabel_flight(g, ends_local, json.loads(fr.faults), json.loads(fr.sensor_faults), old,
                             z_fault, z_ch, z_eff)
        for k, v in new.items():
            ends[sl, col[k]] = v
        before["fmv"] += sum(old[f"fmv_{n}"].sum() for n in FAULT_NAMES)
        after["fmv"] += sum(new[f"fmv_{n}"].sum() for n in FAULT_NAMES)
        before["sf"] += sum((old[f"sf_{c}_flag"] > 0).sum() for c in FAULTABLE_CHANNELS)
        after["sf"] += sum((new[f"sf_{c}_flag"] > 0).sum() for c in FAULTABLE_CHANNELS)
    ends.flush()
    contract = dict(src.contract, labels="v5b", relabel={"z_fault": z_fault, "z_sensor_per_channel": z_ch,
                                                          "severity": "effective (s x depth / spec_depth)"})
    with open(os.path.join(dst, "contract_v5.json"), "w") as fh:
        json.dump(contract, fh, indent=1)
    return {"cache": dst, "z_sensor_per_channel": {c: round(v, 2) for c, v in z_ch.items()},
            "diagnosis_positives": [int(before["fmv"]), int(after["fmv"])],
            "sensor_fault_positives": [int(before["sf"]), int(after["sf"])]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--z-fault", type=float, default=1.0,
                    help="engine fault visible at this noise-free effect (sigma); 1.0 = the v5 rule, unchanged")
    ap.add_argument("--per-channel-faults", action="store_true",
                    help="engine-fault visibility per instrument against its healthy spread (needs data "
                         "generated with the eff_<channel> columns; older data falls back to effect_z)")
    ap.add_argument("--z-sensor", type=float, default=3.0,
                    help="floor for the per-channel sensor bias/drift threshold (sigma)")
    a = ap.parse_args()
    print(build(a.engine, a.z_fault, a.z_sensor, a.per_channel_faults))
