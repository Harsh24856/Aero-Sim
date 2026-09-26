"""Plumbing for the v5 notebooks (make_notebooks_v5.py), so their cells show steps,
not setup. Every heavy step goes through run_v5.run_step, the same code the
command line runs - a notebook and `run_v5.py` produce identical artifacts."""
from __future__ import annotations

import json
import os
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import Markdown, display

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

import run_v5  # noqa: E402

ENGINE_NAMES = {"912": "Rotax 912 ULS", "914": "Rotax 914 ULF", "915": "Rotax 915 iS", "916": "Rotax 916 iS"}
HEALTHY, FAULTY, TRUTH, MODEL = "#2b7bb9", "#d9480f", "#555555", "#0b7285"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

__all__ = ["plt", "np", "pd", "display", "Markdown", "ENGINE_NAMES", "HEALTHY", "FAULTY", "TRUTH", "MODEL",
           "setup", "ensure", "paths", "load_json", "show_table",
           "data_overview", "data_checks_table", "example_flight", "plot_example_flight",
           "plot_history", "calibration_view", "rul_view", "scorecard", "flight_timeline",
           "hp_search_view", "observability_view", "physics_changes"]


def setup(engine: str) -> dict:
    p = paths(engine)
    display(Markdown(f"**{ENGINE_NAMES.get(engine, engine)}** - data `{os.path.relpath(p['data'], ROOT)}`, "
                     f"artifacts `{os.path.relpath(p['art'], ROOT)}`"))
    return p


def paths(engine: str) -> dict:
    return run_v5.paths(engine)


def ensure(engine: str, step: str, force: bool = False, **kw) -> None:
    """Run one pipeline step unless its output already exists (then instant).
    kw: epochs, patience, max_windows, boot, config - as on the command line."""
    p = paths(engine)
    os.makedirs(p["art"], exist_ok=True)
    if run_v5.done(step, p) and not force:
        print(f"{step}: already done - loaded from {os.path.relpath(p['art'], ROOT)}")
        return
    a = types.SimpleNamespace(**dict({"config": None, "epochs": 40, "patience": 6, "max_windows": None,
                                      "boot": 1000}, **kw))
    run_v5.run_step(step, p, a)


def load_json(engine: str, name: str) -> dict:
    with open(os.path.join(paths(engine)["art"], name)) as fh:
        return json.load(fh)


def show_table(df: pd.DataFrame, digits: int = 3) -> None:
    """Plain DataFrame display (pandas' .style needs jinja2, not installed here)."""
    with pd.option_context("display.max_colwidth", None):
        display(df.set_index(df.columns[0]).round(digits))


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def _index(engine: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(paths(engine)["data"], "index.parquet"))


def data_overview(engine: str) -> None:
    idx = _index(engine)
    hours = idx.n_rows.sum() / 3600
    display(Markdown(f"**{len(idx):,} flights, {idx.n_rows.sum():,} seconds ({hours:,.0f} flight hours).**"))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    idx.split.value_counts().reindex(["train", "cal", "val", "test"]).plot.bar(ax=ax[0], color=MODEL)
    ax[0].set_title("flights per split")
    idx.primary_family.value_counts().plot.barh(ax=ax[1], color=FAULTY)
    ax[1].set_title("main fault family")
    idx.mission.value_counts().plot.barh(ax=ax[2], color=HEALTHY)
    ax[2].set_title("mission")
    plt.tight_layout()
    plt.show()
    n_sf = idx.sensor_faults.map(lambda s: len(json.loads(s)) if isinstance(s, str) else len(s)).gt(0).mean()
    display(Markdown(f"{100 * n_sf:.0f}% of flights carry a sensor fault; "
                     f"{100 * idx.wear_limited.mean():.0f}% of engines are removed by wear before TBO; "
                     f"{100 * (idx.life_scale > 1).mean():.0f}% run on the fast (x180) life clock."))


def data_checks_table(engine: str) -> None:
    import data_checks_v5
    ok = data_checks_v5.run(paths(engine)["data"])
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAIL - do not train on this data")


def example_flight(engine: str, family: str | None = None, split: str = "test") -> pd.DataFrame:
    """One flight's rows: the first in `split` whose main fault is `family`
    (any fault if None; 'none' for a healthy one)."""
    idx = _index(engine)
    sel = idx[idx.split == split]
    if family:
        sel = sel[sel.primary_family == family]
    elif "none" in set(sel.primary_family):
        sel = sel[sel.primary_family != "none"]
    r = sel.iloc[0]
    g = pq.ParquetFile(os.path.join(paths(engine)["data"], r.file)).read_row_group(int(r.row_group)).to_pandas()
    g.attrs["meta"] = r.to_dict()
    return g.sort_values("t").reset_index(drop=True)


def plot_example_flight(g: pd.DataFrame, channels=("res_egt", "res_cht", "res_oil_pressure", "res_engine_rpm")) -> None:
    from sensors_v5 import SENSOR_SPEC
    fm = [c for c in g.columns if c.startswith("fm_") and g[c].max() > 0]
    fig, ax = plt.subplots(len(channels) + 1, 1, figsize=(11, 1.6 * (len(channels) + 1)), sharex=True)
    t = g.t / 60
    for a, c in zip(ax, channels):
        sd = SENSOR_SPEC[c[4:]]["noise_sd"] if c[4:] in SENSOR_SPEC else 1.0
        a.plot(t, g[c] / sd, lw=0.6, color=FAULTY)
        a.axhline(0, color="k", lw=0.5)
        a.set_ylabel(c[4:] + "\n(sigma)", fontsize=8)
    for c in fm:
        ax[-1].plot(t, g[c], lw=1.2, label=c[3:])
    ax[-1].plot(t, g.health_index, lw=1.2, color=TRUTH, ls="--", label="health")
    ax[-1].set_ylabel("truth")
    ax[-1].legend(fontsize=7, ncol=4)
    ax[-1].set_xlabel("minutes into flight")
    m = g.attrs.get("meta", {})
    fig.suptitle(f"flight {m.get('scenario_id')} ({m.get('mission')}): residuals = reading - healthy twin", fontsize=10)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------
def plot_history(engine: str) -> None:
    h = load_json(engine, "history.json")
    df = pd.DataFrame([dict(epoch=r["epoch"], **r["val"]) for r in h["history"]])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    for c in ["det_auc", "diag_macro_ap", "family_f1", "sensor_macro_f1"]:
        ax[0].plot(df.epoch, df[c], marker=".", label=c)
    ax[0].set_title("validation scores (higher is better)")
    ax[0].legend(fontsize=7)
    for c in ["sev_on_fault_mae", "sev_clean_mae", "health_mae"]:
        ax[1].plot(df.epoch, df[c], marker=".", label=c)
    ax[1].set_title("validation errors (lower is better)")
    ax[1].legend(fontsize=7)
    ax[2].plot(df.epoch, df.composite, marker="o", color=MODEL)
    ax[2].axvline(h["best_epoch"], color=FAULTY, ls="--", label=f"kept: epoch {h['best_epoch']}")
    ax[2].set_title("composite (selection score)")
    ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.show()
    from train_v5 import HEADS
    lv = pd.DataFrame([r["log_vars"] for r in h["history"]], columns=HEADS)
    display(Markdown("**Learned loss weights** (exp(-s); a head the model finds noisy gets less weight):"))
    show_table(np.exp(-lv.iloc[[-1]]).assign(epoch=len(lv) - 1)[["epoch", *HEADS]])


# ---------------------------------------------------------------------------
# 3. Calibration
# ---------------------------------------------------------------------------
def calibration_view(engine: str) -> None:
    from degradation_v5 import FAULT_NAMES
    c = load_json(engine, "calibration.json")
    df = pd.DataFrame({"fault": FAULT_NAMES, "temperature": c["temperature"], "cut_off": c["cutoffs"]})
    df = df[(df.temperature != 1.0) | (df.cut_off != 0.5)]
    show_table(df, 2)
    display(Markdown(f"Validation calibration error (ECE): **{c['val_ece_before']:.4f} -> {c['val_ece_after']:.4f}** "
                     "(target <= 0.05). A temperature is kept only where it helped."))


# ---------------------------------------------------------------------------
# 4. RUL
# ---------------------------------------------------------------------------
def rul_view(engine: str) -> None:
    r = load_json(engine, "rul_v5.json")
    rows = [dict(model=k, **{m: v[m] for m in ["wear_limited_mae_pct", "calendar_wear_limited_mae_pct",
                                                "gain_vs_calendar", "tbo_limited_mae_pct"]})
            for k, v in r["candidates"].items()]
    show_table(pd.DataFrame(rows))
    display(Markdown(f"Chosen on validation flights: **{r['chosen']}**. Errors are % of TBO; "
                     "the calendar countdown (TBO - hours) is the baseline to beat."))


# ---------------------------------------------------------------------------
# 5. Test
# ---------------------------------------------------------------------------
def scorecard(engine: str) -> None:
    with open(os.path.join(paths(engine)["art"], "model_card.md")) as fh:
        display(Markdown(fh.read()))


def flight_timeline(engine: str, family: str | None = None) -> None:
    """The model's answers through one unseen (test) flight against the truth."""
    from degradation_v5 import FAULT_NAMES
    from pipeline_v5 import Cache
    from train_v5 import Trainer, apply_temperature
    p = paths(engine)
    cache = Cache(p["cache"])
    tr = Trainer.load(cache, p["art"])
    cal = load_json(engine, "calibration.json")
    fl = cache.flights[cache.flights.split == "test"]
    if family:
        fl = fl[fl.primary_family == family]
    f = fl.iloc[0]
    ids = np.arange(f.end0, f.end0 + f.n_ends)
    o, y, Er = tr.predict(ids)
    t = (cache.labels(Er, "row") - f.row0) / 60          # one row per second
    pdiag = apply_temperature(o["diagnosis"], cal["temperature"])
    top = [FAULT_NAMES.index(n) for n in tr.applicable]
    top = sorted(top, key=lambda j: -max(pdiag[:, j].max(), y["diagnosis"][:, j].max()))[:3]
    fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(t, o["detection"][:, 0], color=MODEL, label="model: something wrong")
    ax[0].fill_between(t, 0, y["detection"][:, 0], color=FAULTY, alpha=0.15, step="mid", label="truth")
    ax[0].legend(fontsize=7)
    for j in top:
        ln, = ax[1].plot(t, pdiag[:, j], label=FAULT_NAMES[j])
        ax[1].plot(t, y["diagnosis"][:, j], color=ln.get_color(), ls=":", lw=1)
    ax[1].set_ylabel("fault probability\n(dotted = truth)")
    ax[1].legend(fontsize=7)
    ax[2].plot(t, o["health"][:, 0], color=MODEL, label="model")
    ax[2].plot(t, y["health"][:, 0], color=TRUTH, ls="--", label="truth")
    ax[2].set_ylabel("health")
    ax[2].set_xlabel("minutes into flight")
    ax[2].legend(fontsize=7)
    fig.suptitle(f"test flight {f.scenario_id} ({f.mission}, main fault: {f.primary_family}) - never seen in training",
                 fontsize=10)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 00 notebooks
# ---------------------------------------------------------------------------
def hp_search_view() -> None:
    path = os.path.join(HERE, "hp_search_v5.json")
    if not os.path.exists(path):
        print("No search yet. Run:\n  validation/venv/bin/python validation_v5/hp_search_v5.py "
              "--cache validation_v5/cache/pilot_914 --trials 24 --rungs 3,8,30")
        return
    s = json.load(open(path))
    rows = []
    for t in s["trials"]:
        r = {"trial": t["id"], **{k: t["cfg"][k] for k in s["space"]}}
        for i in range(len(s["rungs"])):
            if f"rung{i}" in t:
                r[f"rung{i} ({s['rungs'][i]} ep)"] = t[f"rung{i}"]["val"]["composite"]
        rows.append(r)
    rungs = [c for c in rows[0] if c.startswith("rung")][::-1]     # furthest rung first
    df = pd.DataFrame(rows).sort_values(rungs, ascending=False, na_position="last")
    show_table(df.head(12), 4)
    if s.get("best"):
        display(Markdown(f"**Winner: trial {s['best']['id']}** - run_v5.py trains every engine with this config."))


def observability_view(engine_model: str | None = None) -> None:
    with open(os.path.join(HERE, "observability_v5.md")) as fh:
        text = fh.read()
    if engine_model:
        part = text.split(f"## {engine_model}")[1].split("\n## ")[0]
        text = f"## {engine_model}{part}"
    display(Markdown(text))


def physics_changes() -> None:
    import physics_v5 as P
    df = pd.DataFrame([{"where": k, "equation": v} for k, v in P.CHANGED_EQUATIONS.items()])
    show_table(df)
