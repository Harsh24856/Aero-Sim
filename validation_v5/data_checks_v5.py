#!/usr/bin/env python3
"""Checks every v5 dataset must pass before training (docs/v5_model_improvement_plan.md,
Phase 2 task 10). Exit code 1 if any check fails.

    backend/.venv/bin/python validation_v5/data_checks_v5.py data/rotax_v5_pilot/914
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backend"))

import features_v5 as F  # noqa: E402
from sensors_v5 import FAULT_KINDS, FAULTABLE_CHANNELS, SENSOR_SPEC  # noqa: E402

SPLIT_TARGET = {"train": 0.60, "cal": 0.15, "val": 0.10, "test": 0.15}


def load(dirpath: str, max_flights: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    idx = pd.read_parquet(os.path.join(dirpath, "index.parquet"))
    with open(os.path.join(dirpath, "manifest.json")) as fh:
        man = json.load(fh)
    if max_flights:
        idx = idx.head(max_flights)
    parts = []
    for fname, grp in idx.groupby("file"):
        pf = pq.ParquetFile(os.path.join(dirpath, fname))
        parts.append(pf.read_row_groups(list(grp.row_group)).to_pandas())
    return idx, pd.concat(parts, ignore_index=True), man


def _row_stats(dirpath: str, idx: pd.DataFrame, chunk: int = 64) -> dict:
    """Every row-level statistic the checks need, streamed a few flights at a time.
    A full engine is ~19.5M rows x 120 columns (~19 GB as float64): loading it whole,
    as the 600-flight pilot could, gets the process killed on this 8 GB machine."""
    sf_flag = [f"sf_{c}_flag" for c in FAULTABLE_CHANNELS]
    sf_act = [f"sf_{c}_active" for c in FAULTABLE_CHANNELS]
    st = {"rows": 0, "cols": 0, "nonfinite": 0, "turbo": 0.0, "rul_lt_oracle": 0, "rul_ne_oracle": 0,
          "rul_gt_cal": 0, "wear_limited": 0, "fp": 0, "fps": 0, "fp_gt_fps": 0, "flag_bad": 0,
          "flag_on": 0, "act_on": 0, "healthy": 0, "healthy_over": 0,
          "lo": {c: np.inf for c in F.MEASURED_COLS}, "hi": {c: -np.inf for c in F.MEASURED_COLS}}
    for fname, grp in idx.groupby("file"):
        pf = pq.ParquetFile(os.path.join(dirpath, fname))
        rgs = list(grp.row_group)
        for i in range(0, len(rgs), chunk):
            df = pf.read_row_groups(rgs[i:i + chunk]).to_pandas()
            num = df.select_dtypes("number").to_numpy()
            st["rows"] += len(df)
            st["cols"] = num.shape[1]
            st["nonfinite"] += int((~np.isfinite(num)).sum())
            del num
            st["turbo"] = max(st["turbo"], float(df["res_wastegate_position"].abs().max()),
                              float(df["wastegate_position"].abs().max()))
            for c in F.MEASURED_COLS:
                st["lo"][c] = min(st["lo"][c], float(df[c].min()))
                st["hi"][c] = max(st["hi"][c], float(df[c].max()))
            st["rul_lt_oracle"] += int((df.rul_hours < df.rul_hours_oracle - 1e-3).sum())
            st["rul_ne_oracle"] += int((df.rul_hours > df.rul_hours_oracle + 1e-3).sum())
            st["rul_gt_cal"] += int((df.rul_hours > df.rul_calendar_hours + 1e-3).sum())
            st["wear_limited"] += int((df.rul_hours < df.rul_calendar_hours - 1e-3).sum())
            st["fp"] += int(df.fault_present.sum())
            st["fps"] += int(df.fault_present_sev.sum())
            st["fp_gt_fps"] += int((df.fault_present > df.fault_present_sev).sum())
            flag, act = df[sf_flag].to_numpy(), df[sf_act].to_numpy()
            st["flag_bad"] += int(((flag != 0) & (flag != act)).sum())
            st["flag_on"] += int((flag > 0).sum())
            st["act_on"] += int((act > 0).sum())
            healthy = df.fault_present_sev == 0
            st["healthy"] += int(healthy.sum())
            st["healthy_over"] += int(df.limit_exceeded[healthy].sum())
    return st


def run(dirpath: str) -> bool:
    idx = pd.read_parquet(os.path.join(dirpath, "index.parquet"))
    with open(os.path.join(dirpath, "manifest.json")) as fh:
        man = json.load(fh)
    st = _row_stats(dirpath, idx)
    n = st["rows"]
    turbo = st["turbo"] > 0
    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))

    # 1. Finite values, instruments inside their ranges
    check("no NaN / inf", st["nonfinite"] == 0, f"{n:,} rows x {st['cols']} cols")
    bad = [c for c in F.MEASURED_COLS
           if st["lo"][c] < SENSOR_SPEC[c]["sat"][0] - 1e-6 or st["hi"][c] > SENSOR_SPEC[c]["sat"][1] + 1e-6]
    check("instruments inside sensor ranges", not bad, bad or "all")

    # 2. Sensor-fault cells balanced (flights whose FIRST sensor fault is each cell)
    cells = {}
    for sf in idx.sensor_faults:
        lst = json.loads(sf)
        if lst:
            k = (lst[0]["channel"], lst[0]["kind"])
            cells[k] = cells.get(k, 0) + 1
    counts = np.array([cells.get((c, k), 0) for c in FAULTABLE_CHANNELS for k in FAULT_KINDS])
    frac_faulty = float((idx.sensor_faults.map(lambda s: len(json.loads(s)) > 0)).mean())
    check("sensor-fault cells all covered", counts.min() > 0,
          f"flights per cell min {counts.min()} / median {int(np.median(counts))} / max {counts.max()}; "
          f"{100 * frac_faulty:.0f}% of flights faulty")
    check("sensor-fault cells balanced (max <= 3x min+1)", counts.max() <= 3 * counts.min() + 1,
          f"min {counts.min()} max {counts.max()}")

    # 3. Splits: proportions overall and per stratum family
    share = idx.split.value_counts(normalize=True).to_dict()
    dev = max(abs(share.get(s, 0) - v) for s, v in SPLIT_TARGET.items())
    check("split proportions ~60/15/10/15", dev < 0.04,
          {s: round(share.get(s, 0), 3) for s in SPLIT_TARGET})
    fam_test = idx.groupby("primary_family").split.apply(lambda s: (s == "test").mean())
    check("every fault family in every split", all(
        (idx[idx.primary_family == f].split.value_counts().reindex(SPLIT_TARGET).fillna(0) > 0).all()
        for f in idx.primary_family.unique()), fam_test.round(2).to_dict())

    # 4. RUL: never counts a future fault (known-faults label >= oracle), bounded by the calendar
    check("rul_hours >= rul_hours_oracle (no future faults)", st["rul_lt_oracle"] == 0,
          f"{100 * st['rul_ne_oracle'] / n:.1f}% of rows differ from the oracle")
    check("rul_hours <= calendar", st["rul_gt_cal"] == 0, "")
    wl = st["wear_limited"] / n
    check("wear-limited windows present", 0.05 < wl < 0.60, f"{100 * wl:.1f}% of rows")

    # 5. Visibility labels
    check("fault_present (visible) <= fault_present_sev", st["fp_gt_fps"] == 0,
          f"visible {100 * st['fp'] / n:.1f}% vs severity-only {100 * st['fps'] / n:.1f}% of rows")
    cr = n * len(FAULTABLE_CHANNELS)
    check("sensor flag only where active", st["flag_bad"] == 0,
          f"visible {100 * st['flag_on'] / cr:.2f}% vs active {100 * st['act_on'] / cr:.2f}% of channel-rows")

    # 6. Healthy engines stay inside certified limits
    rate = st["healthy_over"] / st["healthy"] if st["healthy"] else 0.0
    check("healthy rows over a certified limit < 1%", rate < 0.01, f"{100 * rate:.2f}%")

    # 7. Mix
    ls = float((idx.life_scale > 1).mean())
    check("life clock x180 on ~40% of flights", 0.30 < ls < 0.50, f"{100 * ls:.0f}%")
    ha = float((idx.mission == "high_altitude").mean())
    check("high-altitude share (turbo ~22%, else ~10%)", (0.15 < ha < 0.30) if turbo else (0.04 < ha < 0.18),
          f"{100 * ha:.0f}%")

    width = max(len(r[0]) for r in results)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    ok = all(run(p) for p in sys.argv[1:])
    sys.exit(0 if ok else 1)
