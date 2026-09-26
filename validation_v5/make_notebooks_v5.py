#!/usr/bin/env python3
"""Generate the v5 notebooks: two shared ones, five per engine, and a README index.

    00_physics_observability     what changed in the engine model, which faults can be seen
    00_hp_search                 the hyper-parameter search on the 914 pilot
    1_data_<e>                   the dataset and its checks                  (run_v5 checks, cache)
    2_train_<e>                  the joint model                             (run_v5 train)
    3_calibrate_<e>              probabilities and cut-offs                  (run_v5 calibrate)
    4_rul_<e>                    hours left                                  (run_v5 rul)
    5_test_<e>                   the scorecard on unseen flights             (run_v5 test, card)

Each engine notebook reads the same way: the question, setup, the step (instant
if run_v5.py already did it), what came out, what it means. Plumbing lives in
nb_helpers_v5.py. GENERATED - edit this file, not the notebooks.

    validation/venv/bin/python validation_v5/make_notebooks_v5.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "notebooks")
ENGINES = ["914", "912", "915", "916"]
ENGINE_NAMES = {"912": "Rotax 912 ULS", "914": "Rotax 914 ULF", "915": "Rotax 915 iS", "916": "Rotax 916 iS"}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                         "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = """
# Finds validation_v5 wherever Jupyter was started, then loads the helpers.
import os, sys
d = os.path.abspath('.')
for _ in range(6):
    if os.path.exists(os.path.join(d, 'nb_helpers_v5.py')): break
    if os.path.exists(os.path.join(d, 'validation_v5', 'nb_helpers_v5.py')):
        d = os.path.join(d, 'validation_v5'); break
    d = os.path.dirname(d)
sys.path.insert(0, d)
from nb_helpers_v5 import *
"""

GLOSSARY = """---
### Terms used

| Term | Meaning |
|---|---|
| **Window** | 128 seconds of flight data - what the model looks at for one answer. |
| **Healthy twin** | A perfectly healthy simulated copy of the engine flown through the same flight. |
| **Residual** | Reading minus the healthy twin's reading, in units of the sensor's noise (sigma). |
| **Train / cal / val / test** | Flights used to learn / to calibrate / to choose / to score. A flight is in one only. |
| **Baseline** | A do-nothing rule the model must beat by a clear margin, or its gate fails. |
| **95% interval** | The range the score would take on other test flights like these (flight bootstrap). |
| **TBO** | Time between overhauls: 2000 h (1200 h for the 915). |
"""

# (title, question, cells)
STEPS = {
"1_data": ("The dataset",
"""Is the simulated data complete, balanced and physically sensible, before any
model learns from it? One flight has three engines flown side by side: the faulty
one, a healthy twin (for residuals) and a wear-only one (for the health label).""",
lambda e: [
md("### 1. What is in it"),
code("data_overview(ENGINE)"),
md("""### 2. The 14 checks
No NaN, instruments in range, split sizes, every sensor-fault cell filled, labels
consistent, healthy rows inside limits... Training refuses to start if one fails."""),
code("ensure(ENGINE, 'checks')\nprint(load_json(ENGINE, 'data_checks.json'))"),
md("""### 3. One flight, as the model sees it
Residuals sit on zero while the engine is healthy and move once a fault grows.
The bottom panel is the truth the model is trained to recover."""),
code("g = example_flight(ENGINE)\nplot_example_flight(g)"),
md("""### 4. Build the training cache
Scales every input on training flights only, and stores 128-second windows every
16 s with their long-horizon context and labels. Run once (a few minutes)."""),
code("ensure(ENGINE, 'cache')"),
md("""### What this means
If every check passes, the data is fit to train on: every fault and sensor-fault
kind has enough flights in each split to be learned and to be scored honestly."""),
]),
"2_train": ("Train the model",
"""One network answers six questions at once - is something wrong, which fault,
how bad, which sensor is lying and how, and how worn the engine is. Trained jointly
so no answer is forgotten while another is learned (v4's weakness).""",
lambda e: [
md("""### 1. Train (or load)
40 passes at most, stopping when the validation score stops improving. The kept
version is the one with the best validation composite, not the lowest loss.
Runs on the Metal GPU."""),
code("ensure(ENGINE, 'train')      # ensure(ENGINE, 'train', force=True) to retrain"),
md("### 2. How training went"),
code("plot_history(ENGINE)"),
md("""### What this means
Scores that rise and flatten mean the model learned and stopped before
memorising; the dashed line is the version kept. These are VALIDATION flights -
the honest score comes only from the test notebook."""),
]),
"3_calibrate": ("Calibrate",
"""When the model says 80%, is it right 80% of the time? And where should each
fault's alarm threshold sit?""",
lambda e: [
md("""### 1. Calibrate
A temperature per fault, fitted on the calibration flights and kept only if it
improves validation calibration; then each fault's cut-off is chosen on
validation for the best F1."""),
code("ensure(ENGINE, 'calibrate')\ncalibration_view(ENGINE)"),
md("""### What this means
Low calibration error means the percentages on the dashboard can be read as
chances. A fault with a high cut-off is one the model is shy about; a low one is
one it flags readily."""),
]),
"4_rul": ("Hours left",
"""How many hours until the engine must come off - at TBO, or earlier if a fault
or wear gets there first?""",
lambda e: [
md("""### 1. Fit
A separate model reads only what an aircraft has: the network's own answers
(predicted on flights it never trained on), the margin to limits from measured
readings, hours and usage. Gradient-boosted trees (never more life with more
hours) compete with a small neural net; the better on validation ships."""),
code("ensure(ENGINE, 'rul')\nrul_view(ENGINE)"),
md("""### What this means
The calendar countdown (TBO - hours) is right for a healthy engine and wrong for
a failing one. The model is worth having only where it beats it - on engines
that wear out early."""),
]),
"5_test": ("The scorecard",
"""How good is it on flights it has never seen? Scored once, after every choice
was made on other flights, with 95% intervals and a baseline for each answer.""",
lambda e: [
md("### 1. Score the test flights"),
code("ensure(ENGINE, 'test')\nensure(ENGINE, 'card')\nscorecard(ENGINE)"),
md("### 2. See it working on one unseen flight"),
code("flight_timeline(ENGINE)"),
md("""### What this means
A head passes only if it meets its target AND beats its baseline with the whole
95% interval on the right side. A FAIL names the reason; the plan
(docs/v5_model_improvement_plan.md) says what to do next for each."""),
]),
}


def engine_notebook(step: str, e: str) -> dict:
    title, question, body = STEPS[step]
    return notebook([
        md(f"# {ENGINE_NAMES[e]} - {title}\n\n{question.strip()}\n\n"
           f"Command-line equivalent: `validation/venv/bin/python validation_v5/run_v5.py {e}`"),
        code(SETUP + f"\nENGINE = '{e}'\nsetup(ENGINE)"),
        *body(e),
        md(GLOSSARY),
    ])


def shared_notebooks() -> dict:
    return {
        "00_physics_observability": notebook([
            md("""# Physics v5 and what can be seen

The v5 engine model is v4 plus the fixes Phase 1 needed; a healthy engine behaves
as in v4 except EGT (now load-dependent) and hot-day knock. Below: every changed
equation, then for each engine which faults move an instrument enough to be seen
(>= 2 sigma at severity 0.5) and which can be told apart from their family."""),
            code(SETUP),
            md("### 1. Changed equations"),
            code("physics_changes()"),
            md("### 2. Observability, per engine"),
            *[code(f"observability_view('{m}')") for m in
              ["Rotax_912_ULS", "Rotax_914_ULF", "Rotax_915_iS", "Rotax_916_iS"]],
            md("""### What this means
A fault unobservable in a regime cannot be diagnosed there by any model; the
recall floor in the scorecard applies only to faults observable somewhere."""),
            md(GLOSSARY),
        ]),
        "00_hp_search": notebook([
            md("""# Hyper-parameter search (914 pilot)

Random search with successive halving: 24 configurations train 3 epochs, the best
third 8, the best few 30. Scored on validation flights by the composite. The
winner is the config run_v5.py trains every engine with."""),
            code(SETUP),
            md("""### Run it (hours; do not run while data is generating)
```
validation/venv/bin/python validation_v5/hp_search_v5.py --cache validation_v5/cache/pilot_914 --trials 24 --rungs 3,8,30
```"""),
            code("hp_search_view()"),
            md(GLOSSARY),
        ]),
    }


README = """# v5 notebooks

Generated by `validation_v5/make_notebooks_v5.py` - edit that, not these.
Every step they run is the same code as `validation_v5/run_v5.py`, so a step the
command line already did loads instantly here.

| Notebook | What it answers |
|---|---|
| 00_physics_observability | What changed in the engine model; which faults can be seen |
| 00_hp_search | Which training settings to use |
{rows}
Order per engine: 1 -> 5. Start with the 914 (it has the HP search behind it).
"""


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    books = shared_notebooks()
    for e in ENGINES:
        for step in STEPS:
            books[f"{step}_{e}"] = engine_notebook(step, e)
    for name, nb in books.items():
        with open(os.path.join(OUT, f"{name}.ipynb"), "w") as fh:
            json.dump(nb, fh, indent=1)
    rows = "".join(f"| {s}_{{914,912,915,916}} | {STEPS[s][0]} |\n" for s in STEPS)
    with open(os.path.join(OUT, "README.md"), "w") as fh:
        fh.write(README.format(rows=rows))
    print(f"{len(books)} notebooks -> {OUT}")


if __name__ == "__main__":
    main()
