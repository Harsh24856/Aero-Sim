#!/usr/bin/env python3
"""Generate the v4 notebooks: one per (step, engine), plus a README index.

Every notebook follows the same shape so any of the 28 reads the same way:

    the question it answers      one sentence, plus what goes in and comes out
    0  setup                     one cell
    1  look at the data          a picture of what the model learns from
    2  train or load             instant if run.py already trained it
    3  scorecard                 pass/fail against the quality bar, in words
    4  see it working            a picture of the predictions on unseen flights
    5  what this means           the result as plain sentences
       terms used                a short glossary

They are GENERATED so a fix lands in all 28 at once; plumbing lives in
nb_helpers.py so the cells show steps, not setup. Edit this file, not the
notebooks - hand edits are overwritten.

RUN
    validation_v4/make_notebooks.py
    validation_v4/make_notebooks.py --engines 914 --phases detection,rul
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINES = ["912", "914", "915", "916"]
PHASES = ["scalers", "detection", "diagnosis", "severity",
          "sensor_fault", "health", "rul"]
ENGINE_NAMES = {"912": "Rotax 912 ULS", "914": "Rotax 914 ULF",
                "915": "Rotax 915 iS", "916": "Rotax 916 iS"}

# step title, one sentence, goes in, comes out, builds on
STEP = {
    "scalers": ("Prepare the data",
                "Checks the simulated flight data is complete and consistent, then puts "
                "every input on the same scale so no reading dominates just because its "
                "numbers are bigger.",
                "Raw simulated flights, one row per second",
                "A scaler every later step uses", None),
    "detection": ("Is anything wrong?",
                  "Looks at 128 seconds of engine data at a time and says whether a fault - "
                  "in the engine or in one of its sensors - is present.",
                  "A 128-second window of 29 inputs",
                  "A probability from 0 to 1 that something is wrong", None),
    "diagnosis": ("Which part is failing?",
                  "When something is wrong, names the failing component, out of the faults "
                  "this engine can physically have. Two faults can be present at once.",
                  "The same 128-second window",
                  "One probability per fault type", "detection"),
    "severity": ("How bad is it?",
                 "For each fault that is present, estimates how far it has progressed - from "
                 "0 (just started) to 1 (fully developed).",
                 "The same 128-second window",
                 "One severity per fault type, 0 to 1", "diagnosis"),
    "sensor_fault": ("Is it the engine, or a sensor?",
                     "Checks each of the 12 sensors separately and says whether it is working "
                     "or is biased, drifting, stuck, spiking, noisy or dropping out - so a "
                     "broken instrument is not mistaken for a failing engine.",
                     "The same 128-second window",
                     "For each of 12 sensors, one of 7 conditions", "diagnosis"),
    "health": ("How worn is the engine?",
               "Estimates the engine's overall wear condition, from 1 (as new) to 0 (worn "
               "out), from how its readings compare with a healthy twin.",
               "The same 128-second window",
               "One number from 0 to 1", "diagnosis"),
    "rul": ("How many hours are left?",
            "Estimates the hours until the engine must come off the aircraft - at its "
            "scheduled overhaul, or earlier if wear gets there first.",
            "The window plus 10 life-stage numbers (age, condition, ...)",
            "Hours remaining", "health"),
}

GLOSSARY = """---
### Terms used

| Term | Meaning |
|---|---|
| **Window** | 128 seconds of flight data - the unit every model looks at. |
| **Twin** | A perfectly healthy simulated copy of the engine, flown through the exact same flight. |
| **Residual** | Real reading minus the healthy twin's reading. Near zero means the engine behaves as it should. |
| **Train / validation / test** | Flights used to learn / to pick the best version / to score it. A flight is only ever in one of them, so the score is on flights the model never saw. |
| **TBO** | Time between overhauls - the engine's scheduled life: 2000 h, or 1200 h for the 915. |
| **Warm start** | A step begins from what an earlier step already learned instead of from nothing. |
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


SETUP = """
# Finds the project folder wherever Jupyter was started, then loads everything.
import os, sys
d = os.path.abspath('.')
for _ in range(6):
    if os.path.exists(os.path.join(d, 'nb_helpers.py')): break
    if os.path.exists(os.path.join(d, 'validation_v4', 'nb_helpers.py')):
        d = os.path.join(d, 'validation_v4'); break
    d = os.path.dirname(d)
sys.path.insert(0, d)
from nb_helpers import *

ENGINE, PHASE = '{engine}', '{phase}'
setup(ENGINE, PHASE)
"""

TRAIN = """
payload = train_or_load(ENGINE, PHASE)
print(f"Done. Trained for {payload.get('epochs_run', '?')} passes over the data.")
"""

SCORE = """
scorecard(ENGINE, PHASE)
"""

# ---------------------------------------------------------------------------
# Per-step sections: (look at the data, see it working, what this means)
# ---------------------------------------------------------------------------
SECTIONS = {
"detection": [
("""### 1. Why a healthy twin makes faults visible
Raw sensor readings move with power, altitude and weather, so a fault is easy
to miss in them. Subtracting what a **healthy** engine would read at the same
moment removes all of that. Below, one reading's residual on a healthy flight
and on a flight with a developing fault:""",
"""
healthy, faulty, ch = example_flights(ENGINE)
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(healthy.t / 60, healthy[ch], color=HEALTHY, lw=0.8, label="healthy engine")
ax.plot(faulty.t / 60, faulty[ch], color=FAULTY, lw=0.8, label="engine with a fault")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("minutes into the flight")
ax.set_ylabel(f"{pretty(ch)}\\n(reading minus healthy twin)")
ax.legend(); plt.tight_layout(); plt.show()
"""),
("""### 4. See it working on flights it never saw
**Left:** every possible alarm threshold, from strict to relaxed. Higher and
further left is better; the dotted line is guessing. **Right:** the model's
fault probability for healthy windows (blue) and faulty ones (red) - the less
they overlap, the easier the call.""",
"""
from sklearn.metrics import roc_curve, roc_auc_score
r = predict_test(ENGINE, PHASE)
y, p = r["true"].ravel(), r["pred"].ravel()
hi = r["resid"] >= np.quantile(r["resid"], 0.75)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for mask, lab, col in ((np.ones_like(hi), "all windows", "#555"),
                       (hi, "engine clearly drifting from twin", FAULTY)):
    fpr, tpr, _ = roc_curve(y[mask], p[mask])
    ax[0].plot(fpr, tpr, color=col, label=f"{lab} (AUC {roc_auc_score(y[mask], p[mask]):.2f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=0.8, label="guessing")
ax[0].set_xlabel("false alarms (share of healthy windows flagged)")
ax[0].set_ylabel("faults caught (share of faulty windows flagged)")
ax[0].set_title("Catch rate vs false-alarm rate"); ax[0].legend(fontsize=8)
bins = np.linspace(0, 1, 41)
ax[1].hist(p[y == 0], bins, color=HEALTHY, alpha=0.6, density=True, label="healthy windows")
ax[1].hist(p[y == 1], bins, color=FAULTY, alpha=0.6, density=True, label="faulty windows")
ax[1].set_xlabel("model's fault probability"); ax[1].set_title("How confident the model is")
ax[1].legend(fontsize=8); plt.tight_layout(); plt.show()
"""),
"""
flag = p >= 0.5
print(f"At a 50% alarm threshold the model catches {flag[y == 1].mean():.0%} of faulty windows "
      f"and raises a false alarm on {flag[y == 0].mean():.0%} of healthy ones.")
print(f"Where the engine has clearly drifted from its twin, it catches {flag[(y == 1) & hi].mean():.0%}.")
print("The faults it misses are the ones with no visible effect yet - too early, or hidden by the "
      "engine's own controls - which a person watching the gauges would miss too.")
"""],

"diagnosis": [
("""### 1. Which faults this engine can have
Each fault type is named after the part it damages. Faults for parts the engine
does not have (a 912 has no turbo) never appear, so the model is never asked to
learn them. Grey bars would mean a fault that cannot occur here.""",
"""
df = load_rows(ENGINE, ["scenario_id"] + P.FM_COLS, split="train")
per_flight = df.groupby("scenario_id")[P.FM_COLS].max() >= P.DETECTION_THRESHOLD
counts = per_flight.sum().rename(lambda c: c[3:])
possible = applicable_faults(ENGINE)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh([pretty(f) for f in counts.index], counts.values,
        color=[FAULTY if f in possible else "#bbb" for f in counts.index])
ax.invert_yaxis(); ax.set_xlabel("training flights (one data file) carrying this fault")
ax.set_title(f"{len(possible)} fault types are possible on this engine")
plt.tight_layout(); plt.show()
"""),
("""### 4. See it working on flights it never saw
For each fault type: red is the share of real cases the model named, blue is
the share of its calls that were right. Both near 1 is ideal.""",
"""
r = predict_test(ENGINE, PHASE)
t, p = r["true"] >= P.DETECTION_THRESHOLD, r["pred"] >= 0.5
rows = []
for i, name in enumerate(P.FAULT_MODES):
    if name in applicable_faults(ENGINE) and t[:, i].any():
        hit = (t[:, i] & p[:, i]).sum()
        rows.append({"fault": pretty(name), "caught": hit / t[:, i].sum(),
                     "calls that were right": hit / max(p[:, i].sum(), 1),
                     "test windows": int(t[:, i].sum())})
tab = pd.DataFrame(rows).sort_values("caught")
fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(tab) + 1.2))
yy = np.arange(len(tab))
ax.barh(yy - 0.2, tab["caught"], 0.4, color=FAULTY, label="share of real cases named")
ax.barh(yy + 0.2, tab["calls that were right"], 0.4, color=HEALTHY, label="share of its calls that were right")
ax.set_yticks(yy, tab["fault"]); ax.set_xlim(0, 1); ax.legend(fontsize=8, loc="lower right")
ax.set_title("How well each fault is named"); plt.tight_layout(); plt.show()
tab.round(2)
"""),
"""
best, worst = tab.iloc[-1], tab.iloc[0]
print(f"Easiest to name: {best['fault']} - {best['caught']:.0%} of real cases.")
print(f"Hardest to name: {worst['fault']} - {worst['caught']:.0%}.")
if ENGINE != "912":
    print("Turbo and wastegate faults are hard by nature: below the turbo's critical altitude the "
          "wastegate simply closes further and hides them, exactly as on a real engine.")
"""],

"severity": [
("""### 1. How far faults have progressed in the data
A fault starts small and grows over many flights. This is how far along the
faults in the training data are when the model sees them.""",
"""
sev = load_rows(ENGINE, P.FM_COLS, split="train")[P.FM_COLS].max(axis=1)
sev = sev[sev >= P.DETECTION_THRESHOLD]
plt.figure(figsize=(8, 3.2)); plt.hist(sev, 40, color=FAULTY, alpha=0.8)
plt.xlabel("fault severity (0 = just started, 1 = fully developed)"); plt.ylabel("rows")
plt.title("How far faults have progressed"); plt.tight_layout(); plt.show()
"""),
("""### 4. See it working on flights it never saw
Each dot is one window with a fault: its true severity against the model's
estimate, for the main fault in that window. Dots on the dashed line are exact.""",
"""
r = predict_test(ENGINE, PHASE)
i = r["true"].argmax(1)
true = r["true"][np.arange(len(i)), i]
pred = r["pred"][np.arange(len(i)), i]
m = true >= P.DETECTION_THRESHOLD
plt.figure(figsize=(5, 5)); plt.scatter(true[m], pred[m], s=3, alpha=0.3, color=FAULTY)
plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel("true severity"); plt.ylabel("predicted severity")
plt.title("Each dot is one faulty window"); plt.tight_layout(); plt.show()
"""),
"""
err = np.abs(pred[m] - true[m])
print(f"On windows with a fault, severity is off by {err.mean():.2f} on average "
      f"(median {np.median(err):.2f}) on the 0-1 scale.")
"""],

"sensor_fault": [
("""### 1. How often sensors fail in the data
Sensors are fine almost all the time. The rare failures come in six kinds,
each with its own signature - a steady offset, a slow drift, a frozen value,
sudden spikes, extra noise, or readings dropping to zero.""",
"""
flags = load_rows(ENGINE, P.SF_FLAG_COLS, split="train").to_numpy().ravel()
share = pd.Series(np.bincount(flags, minlength=len(P.SENSOR_FAULT_TYPES)),
                  index=P.SENSOR_FAULT_TYPES) / len(flags)
share.drop("none").mul(100).plot.barh(color=FAULTY, figsize=(7, 3))
plt.xlabel("% of sensor readings with this fault")
plt.title(f"Sensors are fine {share['none']:.1%} of the time")
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
"""),
("""### 4. See it working on flights it never saw
The share of sensor channels identified correctly, split by what was really
wrong with them. Blue is a healthy sensor, red are the six failure kinds.""",
"""
r = predict_test(ENGINE, PHASE)
true, pred = r["true"].astype(int).ravel(), r["pred"].argmax(-1).ravel()
rec = pd.Series({k: (pred[true == c] == c).mean()
                 for c, k in enumerate(P.SENSOR_FAULT_TYPES) if (true == c).any()})
rec.plot.barh(color=[HEALTHY if k == "none" else FAULTY for k in rec.index], figsize=(7, 3.2))
plt.xlim(0, 1); plt.xlabel("share identified correctly"); plt.title("By sensor condition")
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
"""),
"""
faulty = true > 0
print(f"Overall, {(pred == true).mean():.1%} of sensor channels are read correctly - but most "
      f"channels are healthy, so that number flatters it.")
print(f"On channels that really had a fault, it names the kind of fault {(pred[faulty] == true[faulty]).mean():.0%} "
      f"of the time, and spots that the sensor is faulty at all {(pred[faulty] > 0).mean():.0%} of the time.")
"""],

"health": [
("""### 1. How wear builds up
Each dot is one second of flight. An engine's condition falls with age, and
falls faster when a fault is developing.""",
"""
d0 = load_rows(ENGINE, ["engine_hours", "health_index", "fault_present"], split="train")
df = d0.sample(min(20000, len(d0)), random_state=0)
plt.figure(figsize=(8, 3.6))
for v, col, lab in ((0, HEALTHY, "no fault"), (1, FAULTY, "with a fault")):
    d = df[df.fault_present == v]
    plt.scatter(d.engine_hours, d.health_index, s=2, alpha=0.4, color=col, label=lab)
plt.xlabel("engine hours"); plt.ylabel("wear condition (1 = new)"); plt.legend(markerscale=5)
plt.title("Condition falls with age, faster with a fault"); plt.tight_layout(); plt.show()
"""),
("""### 4. See it working on flights it never saw
True condition against the model's estimate. Dots on the dashed line are exact.""",
"""
r = predict_test(ENGINE, PHASE)
true, pred = r["true"].ravel(), r["pred"].ravel()
plt.figure(figsize=(5, 5)); plt.scatter(true, pred, s=3, alpha=0.25, color=HEALTHY)
plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlim(0, 1.02); plt.ylim(0, 1.02)
plt.xlabel("true condition"); plt.ylabel("predicted condition")
plt.title("Each dot is one 128-second window"); plt.tight_layout(); plt.show()
"""),
"""
err = np.abs(pred - true)
print(f"Condition is off by {err.mean():.3f} on average on the 0-1 scale - "
      f"about {100 * err.mean():.0f} points out of 100.")
"""],

"rul": [
("""### 1. What "hours left" means
Most engines run to their scheduled overhaul (blue: a straight countdown).
Some wear out first (red) - those are the ones worth predicting, because a
calendar alone gets them wrong.""",
"""
tbo = C.tbo_of(ENGINE)
d0 = load_rows(ENGINE, ["engine_hours", "rul_hours_true"], split="train")
df = d0.sample(min(20000, len(d0)), random_state=0)
worn = df.rul_hours_true < np.clip(tbo - df.engine_hours, 0, None) - 1
plt.figure(figsize=(8, 3.8))
plt.scatter(df.engine_hours[~worn], df.rul_hours_true[~worn], s=2, alpha=0.4, color=HEALTHY,
            label="reaches scheduled overhaul")
plt.scatter(df.engine_hours[worn], df.rul_hours_true[worn], s=2, alpha=0.5, color=FAULTY,
            label="wears out first")
plt.xlabel("engine hours"); plt.ylabel("hours left"); plt.legend(markerscale=5)
plt.title(f"Overhaul is due at {tbo:.0f} h"); plt.tight_layout(); plt.show()
print(f"{worn.mean():.0%} of rows belong to an engine that wears out before its overhaul.")
"""),
("""### 4. See it working on flights it never saw
True hours left against the model's estimate. Dots on the dashed line are exact.""",
"""
tbo = C.tbo_of(ENGINE)
r = predict_test(ENGINE, PHASE)
true, pred = r["true"].ravel(), r["pred"].ravel()
cal = calendar_rul(ENGINE, r["aux"])
worn = true < cal - 1
plt.figure(figsize=(5.5, 5.5))
plt.scatter(true[~worn], pred[~worn], s=3, alpha=0.25, color=HEALTHY, label="reaches overhaul")
plt.scatter(true[worn], pred[worn], s=4, alpha=0.5, color=FAULTY, label="wears out first")
plt.plot([0, tbo], [0, tbo], "k--", lw=0.8)
plt.xlabel("true hours left"); plt.ylabel("predicted hours left"); plt.legend(markerscale=4)
plt.title("Each dot is one 128-second window"); plt.tight_layout(); plt.show()
"""),
"""
e_model, e_clock = np.abs(pred - true), np.abs(cal - true)
print(f"Average error: {e_model.mean():.0f} h ({100 * e_model.mean() / tbo:.1f}% of the {tbo:.0f} h life).")
print(f"On engines that wear out early, the model is off by {e_model[worn].mean():.0f} h. "
      f"Simply counting down to the overhaul date would be off by {e_clock[worn].mean():.0f} h on those same engines.")
"""],
}

SCALERS_BODY = [
    md("""### 1. What is in the dataset
Each simulated flight goes entirely into one group, so the final score is
measured on flights the models never saw."""),
    code("""
desc = C.describe(ENGINE)
flights = {s: len(json.load(open(os.path.join(C.data_dir(ENGINE), f"index_{s}.json"))))
           for s in ("train", "val", "test")}
overview = pd.DataFrame({"Flights": flights, "Rows (1 per second)": desc["splits"],
                         "128-second windows": desc["windows"]})
overview.index = ["Train (learn from)", "Validation (pick the best)", "Test (final score)"]
overview
"""),
    md("""### 2. The 29 inputs every model sees
Only readings a real aircraft actually provides - nothing the simulator knows
but an airframe would not."""),
    code("""
groups = [("Flight condition", P.FLIGHT_COLS, "What the aircraft is doing, and the air it flies in"),
          ("Engine sensors", P.MEASURED_COLS, "The 12 readings a real engine monitor provides"),
          ("Derived", P.DERIVED_COLS, "Computed on board from the readings above"),
          ("Twin residuals", P.RESIDUAL_COLS, "Reading minus what a healthy engine would read right now")]
pd.DataFrame([{"Group": g, "Count": len(c), "Why it is there": w,
               "Inputs": ", ".join(pretty(x) for x in c)} for g, c, w in groups])
"""),
    md("""### 3. Data check - run this before trusting anything else
Every flight listed in the index must be found in the data files with exactly
the right number of rows. If this fails, stop: training would silently skip
part of the data."""),
    code("""
problems = 0
for split in ("train", "val", "test"):
    entries = json.load(open(os.path.join(C.data_dir(ENGINE), f"index_{split}.json")))
    sample = entries[::max(1, len(entries) // 40)][:40]
    ok = sum(pq.read_table(os.path.join(C.data_dir(ENGINE), split, e["file"]), columns=["scenario_id"],
                           filters=[("scenario_id", "=", e["scenario_id"])]).num_rows == e["n_rows"]
             for e in sample)
    problems += len(sample) - ok
    print(f"{split:5s}: {ok}/{len(sample)} sampled flights found with the right number of rows")
print("\\nDATA OK" if problems == 0 else "\\nPROBLEM - do not train until this is fixed")
"""),
    md("""### 4. Put every input on the same scale
Fitted on the training flights only - using the test flights here would leak
the answers into training and make every score look better than it is."""),
    code("""
R.phase_scalers(ENGINE, force=False)
print("Input scaler:      ", C.scaler_path(ENGINE))
print("Life-stage scaler: ", C.aux_scaler_path(ENGINE))
"""),
    md("""### 5. Did it work?
After scaling, every input should average about 0 with a spread of about 1."""),
    code("""
import joblib
sc = joblib.load(C.scaler_path(ENGINE))
z = sc.transform(load_rows(ENGINE, P.FEATURE_COLS, split="train").to_numpy(np.float64))
fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
yy = np.arange(len(P.FEATURE_COLS))
ax[0].barh(yy, z.mean(0), color=HEALTHY); ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_title("Average after scaling (should be about 0)")
ax[1].barh(yy, z.std(0), color=HEALTHY); ax[1].axvline(1, color="k", lw=0.8)
ax[1].set_title("Spread after scaling (should be about 1)")
ax[0].set_yticks(yy, [pretty(c) for c in P.FEATURE_COLS]); ax[0].invert_yaxis()
plt.tight_layout(); plt.show()
"""),
    md("### What this means"),
    code("""
print(f"Largest average after scaling: {np.abs(z.mean(0)).max():.2f}. Small values like this are "
      f"normal - the check uses one data file, the scaler was fitted on two million rows.")
print("The data is ready. Next: step 1, 'Is anything wrong?'")
"""),
]


def build(engine: str, phase: str, step: int) -> dict:
    title, sentence, goes_in, comes_out, prev = STEP[phase]
    builds = (f"Starts from what step {PHASES.index(prev)} (*{STEP[prev][0]}*) already learned."
              if prev else ("" if phase == "scalers" else "Learns from scratch - this is the first model."))
    howto = ("Run the cells top to bottom with **Shift+Enter**. "
             + ("" if phase == "scalers" else
                "If this step was already trained with `run.py`, training just loads the saved "
                "model in a second; otherwise it trains here."))
    cells = [
        md(f"# {ENGINE_NAMES[engine]} - Step {step}: {title}\n\n"
           f"**In one sentence:** {sentence}\n\n"
           f"| Goes in | Comes out |\n|---|---|\n| {goes_in} | {comes_out} |\n\n"
           f"{builds}\n\n{howto}"),
        md("### 0. Setup"),
        code(SETUP.format(engine=engine, phase=phase)),
    ]
    if phase == "scalers":
        cells += SCALERS_BODY
    else:
        (look_md, look_code), (see_md, see_code), mean_code = SECTIONS[phase]
        cells += [
            md(look_md), code(look_code),
            md("### 2. Train the model (or load it)"), code(TRAIN),
            md("### 3. Scorecard\nMeasured on the test flights. The quality bar catches "
               "a model that did not learn - it is a safety check, not a target."),
            code(SCORE),
            md(see_md), code(see_code),
            md("### 5. What this means"), code(mean_code),
        ]
    cells.append(md(GLOSSARY))
    return {"cells": cells,
            # The kernel must be the validation venv (Python 3.11 + TensorFlow),
            # registered as `uav-v4` - the default kernel on this machine is
            # Anaconda 3.13, which has none of the packages.
            "metadata": {"kernelspec": {"display_name": "UAV v4 (validation venv)",
                                        "language": "python", "name": "uav-v4"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def readme(engines: list, phases: list) -> str:
    rows = "\n".join(
        f"| {i} | {STEP[p][0]} | " + " | ".join(f"`phase{i}_{p}_{e}.ipynb`" for e in engines) + " |"
        for i, p in enumerate(phases))
    return (
        "# v4 notebooks - start here\n\n"
        "One notebook per step per engine. Run the steps in order: each model starts "
        "from what the previous one learned.\n\n"
        f"| Step | Question | " + " | ".join(ENGINE_NAMES[e] for e in engines) + " |\n"
        f"|---|---|" + "---|" * len(engines) + "\n" + rows + "\n\n"
        "Every notebook has the same six parts: the data, training, a scorecard, "
        "a picture of it working on unseen flights, and what it means in plain words.\n\n"
        "To train everything without opening a notebook, use `run.py` - the notebooks "
        "then just load its results. These files are generated by `make_notebooks.py`; "
        "edit that, not the notebooks.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "notebooks"))
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    n = 0
    for phase in phases:
        step = PHASES.index(phase)
        for engine in engines:
            with open(os.path.join(args.out_dir, f"phase{step}_{phase}_{engine}.ipynb"), "w") as fh:
                json.dump(build(engine, phase, step), fh, indent=1)
            n += 1
    with open(os.path.join(args.out_dir, "README.md"), "w") as fh:
        fh.write(readme(ENGINES, PHASES))
    print(f"wrote {n} notebooks and README.md to {args.out_dir}")


if __name__ == "__main__":
    main()
