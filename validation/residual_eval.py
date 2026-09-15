"""Score backend/residual.py on physics-v3 TEST data, at the module's own SCALE.

Runs ResidualMonitor(lag="euler") - the dataset is generated at dt = 1 s - row by
row over whole test scenarios, exactly as main.py feeds it live. Physics is
deterministic, so fault-free residuals are ~0 and a percentile calibration would
only pick an arbitrary floor; the scales are sensor-tolerance sized in residual.py
and this script measures what they achieve.

Per channel:
  false_alarm   deviation rate on rows with no fault on THAT channel and no engine
                failure mode anywhere in the previous 30 s (other channels may be
                faulted - the realistic live case, which is what broke the first
                degradation index)
  on_fm         deviation rate while a failure mode is active and this channel is
                unflagged (engine-caused; advisory.py attributes these)
  tpr[type]     deviation rate while that sensor fault type is active
  confusion     for detected rows of each true type, which signature was reported
  saturated     fraction of rows pinned at the sensor range limit
Degradation index: correlation and MAE against true wear on failure-mode-free rows.

Usage:
  validation/venv/bin/python3 validation/residual_eval.py --backend <backend dir> [--rows 150000] [--workers 1]
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = {"912": "chunks_v3_rotax_912_uls", "914": "chunks_v3_rotax_914_ulf",
        "915": "chunks_v3_rotax_915_is", "916": "chunks_v3_rotax_916_is"}
TYPES = {1: "bias", 2: "drift", 3: "spike", 4: "stuck", 5: "noise"}
SIG = {None: 0, "bias": 1, "drift": 2, "spike": 3, "stuck": 4, "noise": 5}
SIG_NAMES = ["none", "bias", "drift", "spike", "stuck", "noise"]
STAT = {"settling": 0, "ok": 1, "deviation": 2, "saturated": 3}
IN_COLS = ["scenario_id", "throttle", "engine_rpm", "power_kw", "airspeed", "isa_dev_c",
           "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm_fault"]
FM_COLS = ["fm_misfire", "fm_injector_fouling", "fm_cooling_degradation", "fm_combustion_instability"]


def evaluate(args):
    key, backend, rows = args
    sys.path.insert(0, backend)
    import residual as R
    CH = R.CHANNELS
    t0 = time.time()
    flag_cols = [f"{c}_flag" for c in CH]
    cols = list(dict.fromkeys(IN_COLS + ["time", "wear"] + flag_cols + FM_COLS))
    df = (pq.read_table(os.path.join(HERE, DIRS[key], "test_chunk_01.parquet"), columns=cols)
            .to_pandas().sort_values(["scenario_id", "time"]).reset_index(drop=True))
    ends = df.groupby("scenario_id", sort=False).size().cumsum()
    cut = int(ends[ends >= rows].iloc[0]) if (ends >= rows).any() else len(df)
    df = df.iloc[:cut].reset_index(drop=True)

    flags = df[flag_cols].to_numpy(np.int8)
    fm_on = (df[FM_COLS].to_numpy() > 0).any(axis=1).astype(np.int8)
    g = df.groupby("scenario_id", sort=False)
    win = R.WINDOW
    recent_flag = np.stack([g[c].transform(lambda s: (s != 0).astype(np.int8).rolling(win, min_periods=1).max()).to_numpy()
                            for c in flag_cols], axis=1).astype(bool)
    recent_fm = (df.assign(_fm=fm_on).groupby("scenario_id", sort=False)["_fm"]
                   .transform(lambda s: s.rolling(win, min_periods=1).max()).to_numpy().astype(bool))

    n = len(df)
    status = np.zeros((n, 8), np.int8); sig = np.zeros((n, 8), np.int8)
    absres = np.zeros((n, 8), np.float32); widx = np.zeros(n, np.float32)
    mon = R.ResidualMonitor(lag="euler"); prev = None
    names = IN_COLS[1:]
    for i, row in enumerate(df[IN_COLS].itertuples(index=False, name=None)):
        if row[0] != prev:
            mon.reset(); prev = row[0]
        t = dict(zip(names, row[1:])); t["physics_version"] = "v3"
        out = mon.update(t, dt=1.0)
        widx[i] = out["degradation_index"]
        for j, c in enumerate(CH):
            ch = out["channels"][c]
            status[i, j] = STAT[ch["status"]]; sig[i, j] = SIG[ch["signature"]]; absres[i, j] = abs(ch["residual"])

    settled = status[:, 0] != 0
    dev = status == 2
    result = {"key": key, "rows": n, "scenarios": int(df.scenario_id.nunique()),
              "scales": R.SCALE, "channels": {}}
    for j, c in enumerate(CH):
        clean = settled & ~recent_flag[:, j] & ~recent_fm & (status[:, j] != 3)
        on_fm = settled & ~recent_flag[:, j] & fm_on.astype(bool) & (status[:, j] != 3)
        row = {"n_clean": int(clean.sum()),
               "false_alarm": float(dev[clean, j].mean()) if clean.any() else None,
               "clean_p99_abs_residual": float(np.percentile(absres[clean, j], 99)) if clean.any() else None,
               "on_fm": float(dev[on_fm, j].mean()) if on_fm.any() else None,
               "saturated": float((status[settled, j] == 3).mean()),
               "types": {}}
        for k, name in TYPES.items():
            act = settled & (flags[:, j] == k) & (status[:, j] != 3)
            if act.sum() < 50:
                continue
            det = act & dev[:, j]
            conf = {SIG_NAMES[s]: round(float((sig[det, j] == s).mean()), 3) for s in range(1, 6)} if det.any() else {}
            row["types"][name] = {"n": int(act.sum()), "tpr": float(dev[act, j].mean()), "confusion": conf}
        result["channels"][c] = row
    ok = settled & ~fm_on.astype(bool)
    if ok.sum() > 100:
        wear = df.wear.to_numpy()
        result["degradation"] = {"corr_with_wear": float(np.corrcoef(widx[ok], wear[ok])[0, 1]),
                                 "mae_wear": float(np.mean(np.abs(widx[ok] - wear[ok]))),
                                 "p95_abs_error": float(np.percentile(np.abs(widx[ok] - wear[ok]), 95))}
    result["minutes"] = round((time.time() - t0) / 60, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--engines", default="914,915,916")
    ap.add_argument("--rows", type=int, default=150_000)
    ap.add_argument("--workers", type=int, default=1, help="keep at 1 while a training run is using the machine")
    ap.add_argument("--json", default=os.path.join(HERE, "models", "residual_eval.json"))
    a = ap.parse_args()
    keys = [k.strip() for k in a.engines.split(",") if k.strip()]
    jobs = [(k, os.path.abspath(a.backend), a.rows) for k in keys]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(evaluate, jobs))
    else:
        results = [evaluate(j) for j in jobs]
    json.dump(results, open(a.json, "w"), indent=2)

    for r in results:
        d = r.get("degradation", {})
        print(f"\n=== {r['key']}: {r['rows']:,} rows / {r['scenarios']} scenarios, {r['minutes']} min | degradation index vs wear: "
              f"corr {d.get('corr_with_wear', float('nan')):.3f}  MAE {d.get('mae_wear', float('nan')):.3f}  p95 err {d.get('p95_abs_error', float('nan')):.3f}")
        for c, row in r["channels"].items():
            fa = "  -   " if row["false_alarm"] is None else f"{row['false_alarm']:.4f}"
            fm = "  -  " if row["on_fm"] is None else f"{row['on_fm']:.3f}"
            print(f"  {c:12s} scale {r['scales'][c]:<7g} false-alarm {fa} (n {row['n_clean']:6d}, clean p99 |r| "
                  f"{row['clean_p99_abs_residual'] if row['clean_p99_abs_residual'] is not None else float('nan'):.4g})  on-fm {fm}  saturated {row['saturated']:.3f}")
            for t, v in row["types"].items():
                top = max(v["confusion"].items(), key=lambda kv: kv[1]) if v["confusion"] else ("-", 0)
                print(f"      {t:6s} n {v['n']:6d}  detected {v['tpr']:.2f}  signature correct {v['confusion'].get(t, 0):.2f}  "
                      f"(most reported: {top[0]} {top[1]:.2f})")
    print("\nfull results ->", a.json)


if __name__ == "__main__":
    main()
