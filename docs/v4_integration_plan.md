# v4 integration plan: after training

**Goal.** Move the live system (FastAPI backend, AI service, Supabase and the Next.js frontend) from physics v3 / models v3 to **physics_v4 / models v4**. The v4 system has realistic Rotax physics, 13–14 component faults, 7 sensor conditions, a healthy on-board twin, and life predictions measured in real engine hours.

**Written:** 2026-09-24. Organised as ordered phases (0–8), then detailed specs (T, A, B, C, D, DB). The 914 is trained; the 912, 915 and 916 are training now (`validation_v4/logs/912_915_916_train.log`).

---

## 0. Where things stand

| Piece | State |
|---|---|
| `backend/physics_v4.py`, `degradation_v4.py`, `sensors_v4.py` | Done. Physics checked against Rotax figures. |
| Datasets `data/rotax_v4/{912,914,915,916}` | Done. About 10M rows each, and 13/13 data checks pass on every engine. |
| Notebooks `validation_v4/notebooks/` | Done. All 28 generated; all 7 of the 914 notebooks run on the trained models. |
| **914 models** | **Done. All 6 steps pass their gates.** |
| 912 / 915 / 916 models | Training (about 5–6 h per engine). |
| MATLAB twin `~/Documents/MATLAB/v4/` | Done. It matches physics_v4 to within about 2e-15. |
| Live backend / AI service / frontend | **Still on v3.** That's what this plan changes. |

### 914 results (test flights the models never saw)

| Step | Result | Gate |
|---|---|---|
| Detection | AUC 0.879; 0.952 where the engine has clearly drifted from its twin. It catches 72% of faulty windows with 2% false alarms. | ≥ 0.84 / ≥ 0.90 |
| Diagnosis | macro F1 0.571 | ≥ 0.35 |
| Severity | 0.048 over all cells; 0.216 on real faults | ≤ 0.12 |
| Sensor fault | macro F1 0.417 (up from 0.275); accuracy 0.978 | acc ≥ 0.90 |
| Health (wear) | error 0.063 (about 6 points out of 100) | ≤ 0.10 |
| RUL | 4.6% of TBO overall; 13.1% on engines that wear out early (261 h, against 462 h for simply counting down to overhaul) | ≤ 8 / ≤ 15 |

Known weak spots:
- **Wastegate fault: 1% caught.** Below the critical altitude the wastegate closes further and hides it. That's realistic, so this is a limit of the physics, not a bug.
- **Air filter: 24% caught.**
- **Sensor drift: 1% caught.** It can't be told apart from bias within a 128-second window.
- **Ignition, valve and oil faults are over-called.** Only 21–31% of those calls are right.

### Ground rules for every step below

- **v2 and v3 must keep working.** v4 is added as `AERO_PHYSICS_VERSION=v4`, the same switch v2 and v3 already use. Nothing is deleted.
- **Every change lands in both checkouts.** The live stack runs from `/Users/harsh/Documents/UAV_Engine`.
- **No git commits unless asked.**
- **Every stage ends with a check that fails if the stage is wrong.** Each stage below lists its own.

---

### Changes already made (2026-09-24)

These are in both checkouts and were used for the 914 run. The 912/915/916 run uses them too.

| File | Change | Why |
|---|---|---|
| `validation_v4/train_common.py` | New `FaultMae` metric: severity error only where a fault exists | The old all-cell error is passed by predicting zero everywhere (97.6% of cells are 0) |
| `validation_v4/train_common.py` | New `SensorMacroF1` metric and `sensor_weighted_ce` loss (faults weighted 8×) | Accuracy is passed by always answering "fine" (97.8% of channels) |
| `validation_v4/train_common.py` | Severity loss weight on real faults raised from 10× to 40× | Matches the new data's roughly 40:1 zero-to-fault ratio |
| `validation_v4/run.py` | `SELECT`: severity picks its best pass on `mae_on_fault`, sensor fault on `macro_f1` | Training now optimises the number that matters; the old gates remain as minimum checks |
| `validation_v4/run.py` | Sensor-fault pass limit raised from 10 to 20 | The 914 was still improving at pass 10 (early stopping at 18) |
| `validation_v4/model_architectures.py` | Sensor head reads 5 per-channel window statistics: spread, step change, largest jump, minimum, last minus mean | 914 sensor score went from 0.275 to 0.417; dropout caught 46% → 81%, noise 0% → 35% |
| `validation_v4/nb_helpers.py` | Scorecards show the "TRACKED" metrics | Notebooks show the honest numbers beside the gates |
| `~/Documents/MATLAB/v4/` | Simulink Rotax 914 twin (library, 100 Hz model, 1 Hz dataset model), build and verify scripts | Identical to physics_v4 to within about 2e-15 on 4 reference flights |

### Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Time scale | **×180 (3 engine hours per real minute)**, derived in Spec T. The 10–20 h/min alternative is recorded in Spec T, with its retrain route. |
| 2 | Ground-truth panel | **Shown by default**, with a toggle to hide it |
| 3 | v4 as default | **Yes**, once Phase 7 passes |
| 4 | MATLAB for 912/915/916 | **Done 2026-09-25**: `build_rotax_v4('<N>')` builds each engine's own induction and fuel path; `verify_rotax_v4` matches physics_v4 to within 4e-15 on all four engines |

---

## Progress (end of 2026-09-24)

Built on the trained 914 models, which stand in for 912/915/916 until those finish (aiv4.py serves them flagged `placeholder_models`).

| Phase | State |
|---|---|
| 1 Time scale, twin, presets | **Done.** `timescale_v4.py`, `twin_v4.py`, `scenarios_v4.py`, safety limits; tests pass. 100 Hz vs 1 Hz measured: largest feature shift 0.002 std, so one 100 Hz engine + twin pair serves display and AI. |
| 2 Finalise models | **914 done** (`export_deployable_v4.py`, `retrain_rul_live_v4.py`). Per-fault cut-offs: diagnosis 0.571 → 0.612. RUL on live inputs: **19.4% of TBO wear-limited** (387 h vs 462 h calendar), over the 15% bar - the 13.1% needed the true wear label. Repeat for each engine after training. |
| 3 AI service | **Done.** `aiv4.py`; parity exact on 95 windows (window, aux, heads: 0 difference); contract test passes. |
| 4 Database | **Done.** Migration applied; checks passed in a rolled-back transaction; `dbv4.py` + tests. |
| 5 Backend wiring | **Code done** (`main.py`, `advisory.py`, `residual.py`, `summary.py`, `can_bus.py`, `start_stack.sh`); v3 tests still pass (36/36). Live end-to-end run deferred until training ends. |
| 6 Frontend | **Code done, builds; checked in the browser (logged out).** `DiagnosticsV4`, `TwinResidualChart`, `ScenarioPanel`, `EnginesCard`, `lib/v4.ts`, v4 in `timeScale.ts`, replay/report/dashboard/Sensr/ModelBadge. Not yet run in a browser. Engine-info results text waits for final numbers. |
| 7 End-to-end, v4 default | **914: 6 of 8 presets pass** (see below). Default stays v3 until every engine has run it. |

**Done on 2026-09-25 (still on 914 models, no new engines):**

| Check | Result |
|---|---|
| Frontend production build (Phase 6 bar) | **Passes**: all 15 routes |
| Headless end-to-end, 914 (`scripts/e2e_v4.py`) | **6 of 8 presets pass**: healthy (0% false faults), bearing wear, oil pump, cooling on a hot day, prop erosion and CHT dropout are each named from the first scored window. **Fail:** turbo at 6,000 m (probability 0.74 under its 0.95 cut-off; valve leakage and air filter named instead) and EGT stuck (never recognised). Both are in `model_cards_v4.md` and the Q&A. |
| Live stack over HTTP and WebSocket | **Works**: presets, start, AI online, engine hours advancing ×180, advisory, injection, stop. Wear condition read 0.420 against a true 0.434. |
| Browser (logged out; flight started from the API) | **Renders with no console errors**: engine strip, v4 sensor list, the five questions and the advisory. The twin chart and ground-truth card need a signed-in flight to check. |
| Live database write | **Not done**: no profile matches your email, so no account was used. It will happen on your first signed-in v4 flight. |

**Bugs found and fixed today:**
- **Presets could start worn out.** A random baseline put the 1,450 h engine past its end of life before takeoff. Now only baselines that reach TBO are used, and "healthy" is at least 90% of new. Tested.
- **The first WebSocket frame** wasn't shaped like the broadcasts. Fixed.
- **Advisory limit wording** now starts with a capital and carries units.
- **`export_deployable_v4.py` defaulted to the CPU.** It now defaults to the GPU.

**Tried and rejected on validation:** requiring the severity model to agree before calling a fault. Macro F1 was 0.652 with the rule against 0.664 without. It isn't deployed, and the export repeats the test for every engine.

**Docs written:** `model_cards_v4.md`, `demo_script_v4.md` (uses only presets that passed), v4 sections in `qa_prep.md`, `architecture.md` and the README, and the engine-info page (v4 specs, limits, 914 results).

**Found on the way - must stay true:** these models give different answers on the CPU and on the Metal GPU they were trained on (diagnosis probabilities differ by up to 1.0). Every metric, the export and the AI service run on the GPU; the manifest records `metrics_device` and aiv4 reports `backend_validated` only on that device.

**After training, per engine:** `export_deployable_v4.py --engine N --gpu` → `retrain_rul_live_v4.py --engine N` → export again → `parity_ai_v4.py --engine N` → `scripts/e2e_v4.py` → then the Phase 7 live run and the v4 default.

---

## Phase-by-phase plan

Each phase ends with a check that must pass before the next phase depends on it. The phases are numbered in execution order. Two tracks can run early: **Phase 1** (no trained models needed) and the mock-driven part of **Phase 6** can start while Phase 0 is still training.

```
Phase 0  Train 912/915/916 ──────────────┐                              (GPU, running now)
Phase 1  Time scale + live twin + presets ┼──────────────────────────┐  (CPU, start now)
Phase 6a Frontend on a mock AI response ──┼──────────────────────┐   │  (start now)
                                          ▼                      │   │
Phase 2  Finalise models (cut-offs, RUL on predicted inputs, export)│   │
Phase 3  AI service aiv4.py + parity ─────────────────────────┐  │   │
Phase 4  Database migration + dbv4.py ────────────────────────┤  │   │
Phase 5  Backend wiring (main, advisory, summary, safety, CAN) ◄┘  │   │
Phase 6b Frontend on the real backend ◄──────────────────────────┘   │
Phase 7  End-to-end on every engine, then make v4 the default ◄─────┘
Phase 8  Demo script and docs
```

### Phase 0: finish training (running now)
- **Tasks:** [A1] Watch `validation_v4/logs/912_915_916_train.log`. Fix any failed gate with `run.py --engines <e> --phases <step>,<later steps> --force`.
- **Done when:** all 4 engines pass all 6 gates, and all 28 notebooks run on the real models.
- **Time:** about 15–18 GPU-hours (started 18:18 on 2026-09-24).

### Phase 1: time scale, live twin, scenarios (start now; needs no trained models)
- **Tasks:**
  - [T] Create `backend/timescale_v4.py` (`FLIGHT_HZ = 1`, `PHYSICS_HZ = 100`, `LIFE_SCALE = 180`), with `tests/test_timescale_v4.py`.
  - [B1] Create `backend/twin_v4.py` (`UAVEngineTwinV4`): the flown engine, the healthy on-board twin, sensors, both clocks.
  - [B1] Measure 100 Hz against 1 Hz on all 29 features. If any feature's mean moves by more than 0.1 of its scaler standard deviation, step the AI path at 1 Hz.
  - [B2] Create `backend/scenarios_v4.py` (8 presets, each starting close to the interesting moment); add `GET /scenarios`, `POST /scenario`, `POST /inject`.
  - [B9] Add the v4 sensor ranges and parameter limits in `safety.py`.
- **Done when:**
  - `test_timescale_v4.py` and `test_twin_v4.py` pass: residuals stay within noise when healthy, each preset moves its residuals the expected way, and there are no NaN in 30 minutes;
  - the 100 Hz / 1 Hz decision is made and recorded.
- **Time:** about 1–1.5 days.

### Phase 2: finalise the models (after Phase 0, per engine)
- **Tasks:**
  - [A2] Per-fault cut-offs, chosen on validation and reported on test.
  - [A3] Re-evaluate RUL with **predicted** health and severity and **measured** margin. If the wear-limited error is above 15%, retrain the RUL head on out-of-fold predictions.
  - [A4] Record sensor drift as future work.
  - [A5] Write `export_deployable_v4.py`, producing `backend/models_v4/<engine>/` (6 models, 2 scalers, `manifest.json`).
- **Done when:**
  - macro F1 does not drop with the cut-offs;
  - RUL on predicted inputs is ≤ 15% wear-limited on all 4 engines;
  - the manifests hold the contract, cut-offs and test metrics.
- **Time:** about 1 day, plus about 3 GPU-hours per engine if RUL needs retraining.

### Phase 3: AI service (after Phase 2)
- **Tasks:**
  - [B3] Create `backend/aiv4.py`: `RollingWindowV4`, inference in order with RUL last, the contract check at start-up, and `/step /health /select_engine /reset`.
  - [A5] Write `validation_v4/parity_ai_v4.py`.
  - [B11] Write `tests/test_ai_v4_contract.py`.
- **Done when:** parity shows the window identical to training, aux inputs within 1e-5 and predictions identical; the contract test passes; the RUL aux inputs come only from predictions and measurements.
- **Time:** about 1 day.

### Phase 4: database (after Phase 1; can run beside Phase 3)
- **Tasks:**
  - [DB] Apply migration `add_physics_v4_timescale_and_engines` to AeroSiM, **after your confirmation**.
  - [B8] Write `backend/dbv4.py`: `get_or_create_engine`, `start_simulation`, `log_telemetry` (both clocks), `log_channel_diagnostics` (v4 vocabulary), `end_simulation` plus `update_engine_hours`, and optionally `log_maintenance_event`.
- **Done when:** the 6 DB checks pass:
  1. the security and performance advisors show nothing new;
  2. a v4 round trip writes and reads back;
  3. the life clock carries over from flight to flight;
  4. a v4 run without `life_scale` is rejected;
  5. access rules isolate users;
  6. v2 and v3 pages are unchanged.
- **Time:** about half a day.

### Phase 5: backend wiring (after Phases 1, 3 and 4)
- **Tasks:**
  - [B4] `main.py`: `AERO_PHYSICS_VERSION=v4`, the feature list, RUL, wear-based failure, limit events as advisories, both clocks in `/state` and the broadcast.
  - [B5] Turn the residual monitor into a thin wrapper over the twin residuals.
  - [B6] Add the v4 advisory table, sensor-fault suppression and limit events to `advisory.py`.
  - [B7] Update the `summary.py` vocabulary.
  - [B10] Update the `can_ingest.py` channel mapping.
  - [B11] Run all backend tests, including v3 under `AERO_PHYSICS_VERSION=v3`.
- **Done when:** a scripted flight of every preset, with no browser, runs through physics, AI, advisories and the database with no errors; the v3 tests still pass.
- **Time:** about 1–1.5 days.

### Phase 6: frontend (6a can start now against the mock JSON in B3; 6b needs Phase 5)
- **6a tasks:**
  - [C1] `lib/aiTypes.ts` with a v4 type and version branching.
  - [C2] Diagnostics panel with the five questions.
  - [C3] Twin residual chart.
  - [T] The v4 section of `lib/timeScale.ts` (reads `life_scale`, never hard-codes it).
- **6b tasks:**
  - [C4] Scenario and injection panel, ground truth shown by default, two clocks on screen.
  - [C5] Report, replay and telemetry pages on the new columns (Spec DB "Frontend queries").
  - [C6] Engine info specs.
  - [C7] Model badge from `/health`.
  - A dashboard "Engines" card.
- **Done when:** every page works for a v2/v3 run, a v4 live run and a v4 replay, with no console errors, and `npm run build` passes.
- **Time:** about 2 days.

### Phase 7: end-to-end, then make v4 the default (after Phases 5 and 6)
- **Tasks:**
  - [D1] Every preset on every engine, on the live stack.
  - Set `AERO_PHYSICS_VERSION` default to `v4` in `main.py` and the launch scripts. v2 and v3 stay selectable.
- **Done when:**
  - on a healthy flight, no false component fault is present for more than 5% of the time;
  - each fault is named within 5 minutes;
  - each sensor fault is named without raising an engine fault;
  - wear-limited RUL is below the calendar countdown;
  - the old runs still open.
- **Time:** about half a day.

### Phase 8: demo and docs
- **Tasks:**
  - [D2] SIH demo script on the v4 story.
  - [D3] Update `architecture.md`, `model_cards.md` (Phase 2 numbers), `qa_prep.md` and the READMEs.
  - [D4] MATLAB for the other engines, if decision 4 says yes.
- **Done when:** the demo script's flights are reproduced live from the presets.
- **Time:** about half a day.

### Summary

| Phase | Depends on | Can start | Time |
|---|---|---|---|
| 0 Train | — | running | ~15–18 GPU-h |
| 1 Time scale, twin, presets | — | **now** | 1–1.5 d |
| 2 Finalise models | 0 | per engine as it finishes | 1 d (+ GPU if RUL retrains) |
| 3 AI service | 2 | after 2 | 1 d |
| 4 Database | 1 | beside 3 | 0.5 d |
| 5 Backend wiring | 1, 3, 4 | after 3 and 4 | 1–1.5 d |
| 6a Frontend (mock) | — | **now** | 1 d |
| 6b Frontend (real) | 5 | after 5 | 1 d |
| 7 End-to-end, v4 default | 5, 6b | after 6b | 0.5 d |
| 8 Demo, docs | 7 | after 7 | 0.5 d |

About 7–9 working days of build time after training, less where Phases 1 and 6a overlap with training.

The sections below are the detailed specs that the phases refer to by ID ([A1], [B3], [DB] …).

---

## Spec T: one time scale, everywhere

Every part of the system (physics, AI, database, frontend) runs on **one time model**, derived from the training data. Nothing is scaled per page or per feature.

### Two clocks, one constant

| Clock | Unit | Rate | Drives |
|---|---|---|---|
| **Flight clock** `t_flight_s` | seconds | **1 flight second = 1 wall-clock second.** This can't be compressed, because the models were trained on 1 Hz rows and a 128-row window means 128 real seconds. Temperatures take minutes to settle, and the shaft reacts in about 1.5 s. | Physics steps (100 per flight second), the sensor model, the AI's 128-sample window, telemetry timestamps, the mission clock and replay |
| **Life clock** `engine_hours` | engine hours | **`LIFE_SCALE = 180`: each flight second ages the engine 180 s = 0.05 h** (duty-weighted, as in training) | Degradation (`DegradationState.health_at`), wear condition, fault growth, RUL countdown, TBO and hours-to-overhaul |

`engine_hours = start_engine_hours + LIFE_SCALE × life_used_h`. Here `life_used_h` is the physics' own duty-weighted usage on the flight clock: `eng.life_used_h`, exactly as `generate_dataset_v4.py` computes it. Only the life clock is multiplied.

### Why 180: derived from the training data, not chosen for the demo

- **What training contained.** Each flight lasted at most 1.25 h, so an engine aged at most 0.036 h within any 128-second window. Faults were effectively constant within a window.
- **What compression does.** Compressing life makes a fault grow during the window the model is looking at. The fastest-growing fault in the training generator grows **0.0077 severity per engine hour**; the 95th percentile grows 0.0053. This was measured over 800 generated faults on the 914 and 915.
- **The limit.** A fault may move at most **0.05 of severity inside one window**. That's a quarter of the severity model's own error of 0.216, so the models can't tell it apart from a constant fault. It also stays below the 0.08 "fault present" threshold, so no fault can go from absent to present inside one window.
- **The result.** 0.05 / (0.0077 × 128 s) = **0.051 h per flight second**. Rounding down gives 0.05 h/s = **×180**.

| Scale | Engine hours per 128 s window | Fastest fault moves | 2,000 h TBO takes |
|---|---|---|---|
| ×36 (0.01 h/s) | 1.3 h | 0.010 | 55.6 real h |
| **×180 (0.05 h/s)** | **6.4 h** | **0.049** | **11.1 real h** |
| ×360 (old v2 scale) | 12.8 h | 0.099 | 5.6 real h |
| ×600 (10 h per real minute, proposed) | 21.3 h | 0.164 | 3.3 real h |
| ×1,200 (20 h per real minute, proposed) | 42.7 h | 0.329 | 1.7 real h |
| ×1,800 (0.5 h/s) | 64 h | 0.494 (half a fault inside one window) | 1.1 real h |

So in a 10-minute demo flight the engine ages **30 engine hours**. A fault takes a median of **476 engine hours**, and at least about 210, to go from detectable (0.08) to developed (0.90). A demo can't show a whole fault developing live at an honest scale. It shows the engine **at a chosen point in its life** (the scenario start hours, B2), with the fault visibly progressing during the flight.

### The rule that keeps model inputs honest
- **Per-flight quantities stay on the flight clock**, as in training. `life_used_norm` (an RUL input) is this flight's usage, which in training was at most 1.25 h. Scaling it would push it about 1,000× outside anything the model saw.
- **Life quantities go on the life clock:** `engine_hours_norm`, wear condition, fault severities, RUL.
- `aiv4.py` gets both fields from the twin and **never** rescales either.

### Where the scale lives
- **Defined once:** `backend/timescale_v4.py` holds `FLIGHT_HZ = 1`, `PHYSICS_HZ = 100`, `LIFE_SCALE = 180`, plus a docstring with the derivation above.
- **Served:** `GET /state` and `/health` return `{"life_scale": 180, "flight_hz": 1}`.
- **Stored with every run:** `simulations.life_scale`, so a replay always uses the scale its run was recorded with, even if the constant changes later.
- **Frontend:** `lib/timeScale.ts` gets a v4 section. It reads `life_scale` from the run row or from `/state`; it is **never hard-coded** in a component.
- **Everywhere a time is recorded:**
  - telemetry rows store `t_flight_s` **and** `engine_hours`;
  - `simulations` stores `start_engine_hours`, `end_engine_hours` and `life_scale`;
  - the UI labels them "flight time" (m:ss) and "engine hours" (h). The two are never mixed in one number.
- **Test:** `test_timescale_v4.py` checks that 3,600 flight seconds age the engine by 180 h (duty-weighted), and that no AI input goes outside its training range over a 30-minute flight.


### The proposed 10–20 engine hours per minute
**×180 (3 engine hours per real minute)** is the plan default, derived above. You proposed **10–20 engine hours per minute (×600–×1,200)**. That rate gives a more visible demo, but a fault then moves **0.11–0.33 of severity inside one 128 s window** (the fastest fault moves 0.16–0.33). That's larger than the severity model's own error of 0.216. It also lets a fault go from absent to present inside a single window, which the models never saw in training. **Route to 10–20 h/min, if wanted:** regenerate the four datasets with the engine ageing at that rate *during* each flight (`generate_dataset_v4.py` gets a `--life-scale` option; about 4 h per engine), then retrain (about 5–6 h per engine). About 1.5 days in total. The models would then have learned in-window change, and the demo scale would match the training scale exactly. `LIFE_SCALE` is defined in one place (`timescale_v4.py`) and stored per run, so switching later changes one constant and needs no code changes.

---

## Spec A: models (`validation_v4/`)

### A1. Finish 912 / 915 / 916 training
- **Watch** `logs/912_915_916_train.log`. A failed gate stops that engine's later steps. Fix it with `--phases <failed step>,<later steps> --force`.
- **Expect** the 912 to score lower on turbo-related work, because it has no turbo and so only 11 fault types. Also expect wastegate and turbo faults on the 915 and 916 to be weak below 15,000 ft, for the same physical reason as on the 914.
- **Done when:** all 4 engines pass all 6 gates, and all 28 notebooks run on the real models (`scratchpad/run_notebooks_real.py <engine>`).

### A2. A separate decision cut-off for each fault type (no retraining)
- **Why.** Every fault type is called present at a fixed probability of 0.5. The training loss weights positives 12×, so that cut-off is too low for some types. That's why ignition, valve and oil faults are over-called.
- **How.**
  1. Predict on the **validation** flights.
  2. For each fault type, choose the cut-off in 0.05–0.95 with the best F1.
  3. Apply those cut-offs to the **test** flights and report macro F1 before and after.
  4. Store the cut-offs in the engine's manifest (A5), the same way v3 stores `failure_mode_thresholds`.
- **Done when:** test macro F1 does not drop on any engine, and no applicable fault type's recall falls below 50% of its old value. The cut-offs are chosen on validation only; test is used only to report.

### A3. ⚠ RUL on inputs a real aircraft has (critical)
- **The problem.** `tf_data_pipeline.compute_rul_aux` gives the RUL head 10 values. Three of them are **simulator labels** that no aircraft can measure:
  - `health_index`, the true wear condition, which is also what the health model is trained to predict;
  - `mean_fault_severity` and `max_fault_severity`, the true fault severities;
  - `margin_min` is also computed from true physics outputs rather than sensors, which matters less because the measured values are close.
- **What that means.** The 13.1% result is the best case. In deployment these inputs have to come from the **health and severity models' predictions**.
- **Step 1: measure the gap.** Re-evaluate the trained 914 RUL head with those three inputs replaced by:
  - the health model's prediction;
  - the mean and maximum of the severity model's prediction;
  - `margin_min` recomputed from the **measured** sensors.

  Report the wear-limited error and the overall error.
- **Step 2, only if the gap is large (wear-limited error above 15%):** retrain the RUL head on predicted inputs. Generate the health and severity predictions **out-of-fold** on train (5 scenario folds), so the RUL head never trains on predictions the other models made after seeing those same flights. This is the standard fix for a model that takes other models' outputs as inputs.
- **Done when:** wear-limited RUL error with predicted inputs is ≤ 15% of TBO on all four engines. The RUL numbers on the model cards and in the demo are the predicted-input numbers.

### A4. Sensor faults (optional)
- **Where it stands.** The 914 is at 0.417 after 18 passes. The other three engines get the same window statistics and 20 passes automatically.
- **Drift is future work.** Catching drift needs a view longer than 128 seconds, for example a slow running mean of each channel's residual. Put it in "future work" rather than this release.

### A5. Export deployable models to `backend/models_v4/<engine>/`
Write `validation_v4/export_deployable_v4.py`, modelled on `validation/split_deployable_heads.py`. For each engine it writes:
- **Models:** one `.keras` file per step, each taken from the checkpoint of the step it was trained for (detection, diagnosis, severity, sensor_fault, health, rul).
- **Scalers:** `scaler.pkl` and `aux_scaler.pkl`.
- **`manifest.json`**, holding:
  - the input contract: `FEATURE_COLS` (29), `RUL_AUX_ORDER` (10), `FAULT_MODES`, `SENSOR_CHANNELS`, `SENSOR_FAULT_TYPES`, `WINDOW_SIZE` 128 and the 1 Hz sample rate;
  - the faults this engine can have (`applicable_faults`) and its `tbo_hours`;
  - the per-fault cut-offs from A2;
  - the test metrics, including the A3 RUL numbers and the RUL error band by life stage (as in v3 `rul_mae_hours`).
- **Done when:** a parity script (`validation_v4/parity_ai_v4.py`, modelled on `validation/parity_ai_v3.py`) feeds real test flights row by row through the live `RollingWindowV4` (B3). The window has to match the training pipeline exactly, the aux inputs to within 1e-5, and every prediction exactly.

---

## Spec B: backend (`backend/`)

### B1. Live v4 twin: `backend/twin_v4.py` (new)
One class, `UAVEngineTwinV4`. Its `step()` returns the same kind of flat dict `physics.UAVEngineTwin.step()` returns, so `main.py`'s loop, safety checks and broadcast barely change.

Inside it:
- `PistonEngineV4(engine_model, dt=0.01)`: the engine being flown, with the `Health` from `DegradationState`;
- a second `PistonEngineV4` with a healthy `Health()`: the **on-board twin**, fed the identical inputs;
- `SensorsV4`: noise and sensor faults, exactly as in the dataset;
- engine hours and `life_used_h`.

The output dict contains:
- the flight condition, the 12 measured channels and the derived values (`prop_rpm`, `thrust_margin`, `lift_weight_margin`);
- the 6 residuals `measured − twin` (egt, cht, oil_temp, oil_pressure, engine_rpm, fuel_flow);
- the true values, for display only: every engine output and the margins;
- ground truth for the demo only (the injected faults and severities, `health_index`). **This must never reach the AI.**

**⚠ The training data is 1 Hz; the live twin runs at 100 Hz.** The datasets were made with `dt=1.0`, where the shaft is solved directly for its balance point. The live display runs `dt=0.01`, where the shaft speeds up and slows down with its inertia. Sampling the 100 Hz engine once per second gives the same steady states, but slightly different transients.
- **Measure it.** Fly the same mission at 100 Hz (sampled at 1 Hz) and at 1 Hz, and compare the distribution of every one of the 29 features, not just the residuals. Residuals largely cancel the transients because the twin runs at the same rate, but a feature that doesn't cancel would still shift the model's inputs.
- **If any feature's mean moves by more than 0.1 of the scaler's standard deviation,** run the AI path's two engines at `dt=1.0`, stepped once per simulated second, and keep 100 Hz for display only. This costs about 2 extra engine steps per second, which is cheap.
- **Done when:** the check above passes, or the fallback is in place and the parity test (A5) passes on live data.

`main.py`: `new_twin()` builds `UAVEngineTwinV4` when `AERO_PHYSICS_VERSION=v4`. Add `v4` to the allowed versions.

### B2. Engine hours, faults and demo scenarios (on the time scale from Spec T)
Faults take hundreds of engine hours to develop, and a 10-minute flight ages the engine 30 h at ×180. So every run starts **at a chosen point in the engine's life**, and the life clock (Spec T) carries it forward from there. There is no separate fast-forward mode: Spec T's `LIFE_SCALE` is the only time compression.
- **A start state for the engine:** `start_engine_hours`, plus a random seed or a named scenario for degradation. The twin then advances `engine_hours` by `LIFE_SCALE × life_used_h`.
- **Scenario presets** in `backend/scenarios_v4.py`, a plain dict, for example:
  - "Healthy, 300 h";
  - "Bearing wear developing, 1,450 h";
  - "Oil pump degradation";
  - "Cooling degradation on a hot day (ISA +20)";
  - "Turbo degradation at FL180", which shows detection above the critical altitude;
  - "Sensor: CHT dropout";
  - "Sensor: EGT stuck";
  - "Prop erosion".

  Each preset is `start_hours` plus a list of `(fault, onset_h, rate)` or `(sensor, kind, severity)`, applied through `DegradationState` and `SensorsV4`.
- **Presets start close to the interesting moment.** Each preset puts the fault just before or just after it becomes detectable, so it visibly grows during the flight: +30 h in 10 minutes moves a median fault by about 0.08 of severity.
- **Endpoints:**
  - `POST /scenario {name}`: apply a preset;
  - `POST /inject {fault|sensor, …}`: manual injection for Q&A;
  - `GET /scenarios`: list the presets.

  The injected truth is kept server-side and is **shown by default** (decided): the "ground truth" panel sits beside the AI's reading. A toggle can hide it. The truth never reaches `aiv4.py`.

### B3. AI service: `backend/aiv4.py` (new, port 8100, run instead of `aiv3.py`)
- **`RollingWindowV4`** keeps 128 one-second rows of the 29 features and applies `scaler.pkl`. It builds the 10 RUL inputs from **model outputs and measurements, never labels** (see A3):
  - `engine_hours_norm` comes from the life clock, and `life_used_norm` from this flight's usage on the flight clock (Spec T). Neither is rescaled here;
  - `health_index` is the health model's prediction;
  - the two fault-severity inputs are the mean and maximum of the severity model's prediction;
  - `margin_min` comes from the measured sensors via the limits in `physics_v4.margins()`;
  - the two residual inputs are the mean and maximum absolute residual;
  - the two flight inputs are the throttle and altitude means.

  It then applies `aux_scaler.pkl`.
- **Inference order each second:** detection, diagnosis, severity, sensor_fault, health, then **RUL last**, because RUL needs the health and severity outputs.
- **Contract check at start-up:** it refuses to start if the manifest's feature, aux or fault lists differ from its own, as `aiv3.py` does.

**Response (`/step`):**
```json
{
  "status": "ok", "model_version": "v4", "engine_model": "Rotax_914_ULF",
  "detection": {"probability": 0.93, "fault_detected": true},
  "fault_modes": {"bearing_wear": {"probability": 0.88, "present": true, "threshold": 0.55,
                                    "severity": 0.41}, "...": {}},
  "applicable_faults": ["air_filter_fouling", "..."],
  "sensors": {"cht": {"condition": "dropout", "confidence": 0.91}, "...": {}},
  "health": {"wear_condition": 0.62},
  "rul": {"hours": 540.0, "band_hours": 260.0, "tbo_hours": 2000.0,
          "calendar_hours": 550.0, "wear_limited": false},
  "residuals": {"egt": 12.4, "cht": 3.1, "...": 0.0},
  "steps_collected": 128, "steps_needed": 128
}
```

**Endpoints:**
- `/health` reports which engines are loaded, their test metrics from the manifest, and `backend_validated`;
- `/select_engine` switches engine;
- `/reset` clears the window.

### B4. `main.py` wiring
- **Feature list:** `AI_FEATURE_COLS` must include the 29 v4 columns, with the names used in `twin_v4`.
- **RUL value:** for v4, `final_rul_hours()` returns `ai["rul"]["hours"]`.
- **Health:** `health_failure_seconds()` and `engine_failure_shutdown()` now use **wear condition**, not an operating margin.
- **Limits:** an exceeded **limit** (`margin_min ≤ 0`) is an advisory event and does not end the flight. This matches the dataset design, where only a wear condition of 0 ends a scenario.
- **Smoothing and alert holds:** `smooth_rul()` and `hold_alerts_outside_envelope()` accept the v4 response shape.
- **Diagnostics:** `/state` and `/health` report `physics_version=v4` and `ai model_version=v4`, and flag any mismatch, as they do today.

### B5. Model-free residuals (`residual.py`)
v4 already computes twin residuals in `twin_v4`. Use them directly: the residual monitor becomes a thin wrapper that smooths the six residuals and applies sensor zeroing (`/residuals/zero` keeps working). v3's hand-written expected-value formulas are not used for v4.

### B6. Maintenance advisories (`advisory.py`)
Add a v4 branch that maps each **component fault** to a maintenance action, in Rotax terms:

| Fault | Advisory action |
|---|---|
| air_filter_fouling | Inspect / replace air filter |
| compression_loss | Differential compression test |
| valve_leakage | Leak-down test, valve inspection |
| turbo_degradation (914/915/916) | Inspect turbocharger, check boost against the manifold pressure schedule |
| wastegate_fault (914/915/916) | Check the wastegate actuator and TCU / ECU boost control. Add a note that it's only observable above the critical altitude. |
| intercooler_fouling (915/916) | Inspect / clean intercooler |
| injector_fouling | 914/912: carburettor jet cleaning and synchronisation. 915/916: injector flow test. |
| ignition_degradation | Spark plugs, ignition module check |
| combustion_instability | Plugs, mixture, induction leak check |
| bearing_wear | Oil filter inspection for metal, oil analysis |
| oil_pump_degradation | Oil pressure check, pump inspection |
| oil_degradation | Oil and filter change |
| cooling_degradation | Coolant level, radiator / hoses, airflow |
| prop_erosion | Propeller inspection and balance |

Other advisory rules:
- **Sensor faults** become "check sender / wiring for `<channel>`". **That channel is then ignored** in the engine-fault advisory, so a broken sensor can't raise a false engine alarm.
- **Limit events** (margin ≤ 0) use the existing caution and warning ladder, naming which limit (CHT, EGT, oil temperature or oil pressure).
- **Confidence:** keep `CONFIDENCE_FLOOR = 0.70` for anything telling a maintainer to open the engine.

### B7. Post-flight summary (`summary.py`)
Update the prompt's vocabulary to the 14 component faults and 7 sensor conditions, and pass the v4 fields.

### B8. Persistence (`dbv4.py`)
The Supabase changes are specified in full in **Spec DB** below. `dbv4.py` keeps the same function names as `dbv3.py`: it re-exports the unchanged functions and overrides `start_simulation`, `log_telemetry` and `end_simulation`. It also adds `get_or_create_engine` and `update_engine_hours`.
- **Done when:** one v4 run is written, and read back by the mission report and replay pages (C5).

### B9. Safety limits (`safety.py`)
- **Sensor ranges:** `telemetry_problem` ranges for the new fields, taken from the `sensors_v4.SENSOR_SPEC` saturation limits.
- **Parameters:** `PARAM_LIMITS` for any new user parameters (humidity, fuel octane, ethanol share, cooling airflow, electrical load, boost limit), taken from the dataset's sampling ranges. Values outside the training range are clamped, not extrapolated.

### B10. CAN input (`can_ingest.py`)
Map incoming CAN messages to the 12 v4 measured channels (units as `sensors_v4`). While real data is live, the residuals are measured values minus the twin's.

### B11. Tests (`backend/tests/`)
- **`test_twin_v4.py`:**
  - with no faults, the residuals stay within sensor noise;
  - each preset scenario moves the expected residuals in the expected direction;
  - no NaN appears over a 30-minute run.
- **`test_ai_v4_contract.py`:** the feature, aux and fault lists agree between `aiv4.py`, the manifest and `tf_data_pipeline`.
- **Existing v3 tests still pass**, run with `AERO_PHYSICS_VERSION=v3`.

---

## Spec C: frontend (`frontend/`)

### C1. Types and version handling
- **Type:** add `AiResultV4` in `components/Diagnostics.tsx`, or better in `lib/aiTypes.ts`, matching B3.
- **Version switch:** branch on `ai.model_version === "v4"`. v2 and v3 rendering stays unchanged, because old missions in Supabase still use it.

### C2. Diagnostics panel (v4 layout)
1. **"Is anything wrong?"**: detection probability as a gauge, and an alarm when it's ≥ 0.5.
2. **"Which part?"**: the **applicable** component faults only (11 on the 912, 13 on the 914, 14 on the 915 and 916). Each shows its probability, whether it's present (per-fault cut-off from A2), and its severity bar when present. Wastegate and turbo show a hint: "observable above 15,000 ft".
3. **"Engine or sensor?"**: 12 sensor tiles, each showing its condition (fine / bias / drift / stuck / spike / noise / dropout) and confidence. A faulty sensor greys out its channel in the residual chart.
4. **"How worn?"**: the **wear condition** bar (1 = new). Keep it **separate** from the **operating margin** bar (distance to CHT, EGT or oil limits), because they measure different things.
5. **"Hours left"**: RUL in engine hours with its ± band, next to "calendar: X h to overhaul". Show a **"wear-limited"** badge when RUL is below the calendar, which is the case the model exists for.

### C3. Twin residual chart (the digital-twin visual)
- **Content:** for each of the 6 residual channels, the measured value and the **healthy twin's value** over the last 128 seconds, with the gap between them shaded.
- **Why it matters:** this makes the "digital twin" idea visible to a judge.

### C4. Scenario and injection panel (simulate page)
- **Controls:** a scenario dropdown (from `GET /scenarios`), a start-hours slider, and a manual fault and sensor injection form (B2).
- **Ground truth:** shown **by default** (decided), beside the AI's reading, to demonstrate that the AI works it out without seeing it. A toggle hides it.
- **Two clocks on screen:** "flight time m:ss" and "engine hours", both from Spec T. The life scale is shown as a small label ("×180 life") so a judge knows why engine hours move faster than the flight clock.

### C5. Mission report, replay and telemetry pages
`app/mission/report/[id]`, `app/mission/replay/[id]`, `app/telemetry/[id]` and `components/MissionReplay.tsx` should:
- read the v4 columns;
- plot wear condition, RUL, fault probabilities and residuals over time;
- keep v2 and v3 rendering for old rows.

### C6. Engine info pages
Update the specs to what physics_v4 models:
- idle about 1,400 rpm on all engines;
- critical altitude 15,000 ft for the 914, 915 and 916;
- TBO 2,000 h (915: 1,200 h);
- the oil pressure limits (0.8 bar below 3,500 rpm, 2.0 bar above);
- which faults each engine can have.

### C7. Model badge (`ModelBadge.tsx`)
Show "models v4" and each engine's test metrics, read from `aiv4 /health` (from the manifest). The numbers on screen are then always the measured ones.

- **Done when:** every page works in all three modes (v2 or v3 run, v4 live, v4 replay) with no console errors, and `npm run build` passes.

---

## Spec D: end-to-end checks, demo and docs

### D1. End-to-end run
- **Where:** the live stack on `AERO_PHYSICS_VERSION=v4`, with `aiv4.py`.
- **What:** each preset scenario on each engine.
- **Pass criteria:**
  - on a healthy flight, no component fault is present for more than 5% of the time;
  - each fault scenario is named correctly within 5 minutes;
  - each sensor scenario names the right sensor and condition, and **does not** raise an engine fault;
  - RUL on a wear-limited scenario is below the calendar countdown.

### D2. Demo script
Rewrite the SIH script with the v4 story:
1. realistic physics, checked against Rotax data and a MATLAB twin identical to about 2e-15;
2. a healthy twin running on board;
3. one flight each: healthy, bearing wear, sensor dropout, turbo at altitude;
4. the honest limits: wastegate below the critical altitude, and sensor drift.

### D3. Docs to update
- **`docs/architecture.md`:** v4 data flow (engine and twin, then sensors, then the 29 features, then 6 models, then advisories).
- **`docs/model_cards.md`:** v4 test results per engine, including the A3 RUL numbers on predicted inputs.
- **`docs/qa_prep.md`:**
  - Why can wastegate faults hide?
  - Why is drift hard?
  - Why is the 128 s window used, and why 1 Hz?
  - Why are the old gates tripwires only?
  - Why does RUL use predicted health?
- **`README.md`, `validation_v4/notebooks/README.md`:** how to run v4.

### D4. MATLAB (optional)
Build `Rotax912/915/916_v4.slx` with the same generator: the export script takes the engine as an argument. Only if it's wanted for the report.

---

## Spec DB: Supabase changes (project AeroSiM, `bplhfivqhumtvjlefmrn`)

### What is there today (read 2026-09-24)

| Table | Rows | Role | Notes |
|---|---|---|---|
| `profiles` | 6 | one per user | untouched |
| `simulations` | 262 (194 v2, 68 v3) | one per run | `model_version` check allows **only v2 and v3**; `tbo_hours`; `final_rul_hours`; `final_telemetry jsonb`; `groq_result jsonb` |
| `telemetry_logs` | 42,725 | one row per **10 flight seconds** (`DB_LOG_INTERVAL_S = 10`) | 25 typed columns (8 v3 sensors, flight state, AI summary) + v3 `failure_modes jsonb`, `degradation_index`, `residual_deviations text[]` |
| `channel_diagnostics` | 80,640 | 8 rows per telemetry row (one per v3 sensor channel) | `fault_type` vocabulary: `none, Bias, Drift, Spike, Stuck-At, Noise` |

- **Indexes:** primary keys, plus `simulations(user_id)`, `telemetry_logs(simulation_id)` and `channel_diagnostics(log_id)`.
- **Access rules (RLS):** users read and update only their own rows. The backend writes with the service-role key.
- **Migrations:** 8 so far, the last being `add_physics_v3_model_versioning`.
- **Size:** 31 MB.

### What v4 needs that the schema cannot hold

1. **`model_version = 'v4'` is rejected** by the check constraint.
2. **Nothing records an engine's life between flights.** `engine_hours` exists only inside `final_telemetry`. Section T needs a life clock that **carries over from flight to flight**: an engine that ends one flight at 1,453.5 h must start the next at 1,453.5 h, with the same faults.
3. **The time scale isn't stored.** Replaying a run needs the `life_scale` it was recorded with.
4. **v4 has 12 measured channels and 6 twin residuals.** The table has columns for only 8 v3 sensors: coolant temperature, manifold pressure and battery voltage are missing, and so are the residuals.
5. **v4's AI output has a different shape.** It has 11–14 component faults (each with a probability, whether it's present and a severity), 7 sensor conditions, wear condition separate from the operating margin, and RUL with its calendar comparison and "wear-limited" flag.
6. **The ground truth (shown by default) has nowhere to go**: neither the injected fault plan nor the true severity at each instant.

### Design

- **New table `engines`: one row per physical engine, holding its hour meter.** A run is a flight *of an engine*. The engine row holds:
  - `engine_hours`, the life clock;
  - the fault plan, which makes `DegradationState` exactly reproducible across flights: the seed plus each fault's onset hour and growth parameters;
  - the sensor-fault plan.

  At the start of a flight, `start_engine_hours` is set to `engines.engine_hours`. At the end, `engines.engine_hours` is set to `end_engine_hours`, in the same backend call. Scenario presets (B2) create a new engine at the preset's hours. "Continue flying this engine" reuses the same row.
- **`simulations` gains time and provenance:**
  - `engine_id`, `life_scale`, `start_engine_hours`, `end_engine_hours`;
  - `scenario`;
  - `model_manifest jsonb`: the test metrics and per-fault thresholds of the models used, so a report always shows the numbers of the models that actually flew.

  A check constraint **requires every v4 run to record `life_scale` and `start_engine_hours`**, so a time can't be written without its scale.
- **`telemetry_logs` records both clocks on every row:**
  - `time_offset_s` stays as the **flight clock**; it gets a comment, and its meaning doesn't change;
  - new `engine_hours` is the **life clock**.

  It also gains:
  - typed columns for what the pages plot and filter on: the three missing sensors, `margin_min`, and RUL with its calendar comparison;
  - `jsonb` for the v4 structures that are read whole: `fault_modes`, `residuals` and the per-instant ground truth `truth`.
- **`channel_diagnostics` is reused for the 12 v4 sensor channels,** with v4's vocabulary (`none, bias, drift, stuck, spike, noise, dropout`). Pages already branch on `model_version`, so the two vocabularies never mix. `severity_percent` stays `NULL` for v4: the sensor-severity head exists in the model but was never trained (no training step targets it), so nothing should be written from it.
- **Optional table `maintenance_events`:** the advisory log, keyed to engine hours. A row is written only when an advisory **appears or changes**, not every second. This gives each engine a maintenance history ("bearing wear caution at 1,461 h, oil analysis recommended") that follows the life clock across flights.
- **Old data:** existing v2/v3 rows are **not modified and not backfilled.** The new columns are nullable, so the 262 old runs and 42,725 rows stay exactly as they are and still render on the legacy path.

### Migration `add_physics_v4_timescale_and_engines` (applied 2026-09-24, after your confirmation)

```sql
-- 1. Engines: one row per physical engine; its hour meter is the life clock.
create table public.engines (
  id               bigint generated always as identity primary key,
  user_id          uuid not null references auth.users(id) on delete cascade,
  name             text not null,                       -- e.g. "914 #1"
  engine_model     text not null,
  tbo_hours        double precision not null check (tbo_hours > 0),
  engine_hours     double precision not null default 0 check (engine_hours >= 0),
  degradation_seed bigint not null,
  fault_plan       jsonb not null default '{}'::jsonb,  -- {baseline: {a, b, scale}, faults: [{name, onset_h, a, b, depth, sense}]}
  sensor_plan      jsonb not null default '[]'::jsonb,  -- [{channel, kind, onset_flight_s, severity}]
  scenario         text,
  status           text not null default 'in_service'
                   check (status in ('in_service', 'worn_out', 'overhauled', 'retired')),
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);
comment on column public.engines.engine_hours is
  'Life clock (Spec T): engine hours after the last completed flight. Advanced by LIFE_SCALE x duty-weighted flight time.';
create index engines_user_id_idx on public.engines (user_id);
alter table public.engines enable row level security;
create policy "select own engines" on public.engines for select using ((select auth.uid()) = user_id);
create policy "insert own engines" on public.engines for insert with check ((select auth.uid()) = user_id);
create policy "update own engines" on public.engines for update using ((select auth.uid()) = user_id);

-- 2. Simulations: v4, both clocks, and which engine flew.
alter table public.simulations drop constraint simulations_model_version_check;
alter table public.simulations add constraint simulations_model_version_check
  check (model_version = any (array['v2', 'v3', 'v4']));
alter table public.simulations
  add column engine_id          bigint references public.engines(id) on delete set null,
  add column life_scale         double precision check (life_scale > 0),
  add column start_engine_hours double precision check (start_engine_hours >= 0),
  add column end_engine_hours   double precision,
  add column scenario           text,
  add column model_manifest     jsonb;
alter table public.simulations add constraint simulations_v4_time_recorded
  check (model_version <> 'v4' or (life_scale is not null and start_engine_hours is not null));
comment on column public.simulations.life_scale is
  'v4: engine seconds per flight second this run was recorded with (180 at introduction). Null for v2/v3.';
create index simulations_engine_id_idx on public.simulations (engine_id);

-- 3. Telemetry: both clocks on every row, v4 channels and AI outputs.
comment on column public.telemetry_logs.time_offset_s is
  'Flight clock: flight seconds since the run started (1 flight s = 1 wall-clock s). Unscaled in every version.';
alter table public.telemetry_logs
  add column engine_hours          double precision,   -- life clock at this row
  add column coolant_temp          double precision,
  add column manifold_pressure_kpa double precision,
  add column battery_voltage       double precision,
  add column margin_min            double precision,   -- operating margin to the nearest limit, 0..1
  add column rul_calendar_hours    double precision,   -- TBO minus engine_hours: the "just count down" answer
  add column rul_band_hours        double precision,   -- test-set error band at this life stage
  add column wear_limited          boolean,            -- RUL below calendar: wear arrives before overhaul
  add column fault_modes           jsonb,              -- {fault: {p, present, severity}}
  add column residuals             jsonb,              -- {egt, cht, oil_temp, oil_pressure, engine_rpm, fuel_flow}: measured minus twin
  add column truth                 jsonb;              -- ground truth at this instant (shown by default in the UI)
comment on column public.telemetry_logs.health_percent is
  'v2/v3: model health. v4: wear condition x 100 (1 = new), NOT the operating margin (see margin_min).';

-- Replay and charts read one run ordered by time; this covers both, and replaces the single-column index.
create index telemetry_logs_sim_time_idx on public.telemetry_logs (simulation_id, time_offset_s);
drop index if exists public.telemetry_logs_simulation_id_idx;

-- 4. Optional: maintenance history on the life clock.
create table public.maintenance_events (
  id            bigint generated always as identity primary key,
  engine_id     bigint not null references public.engines(id) on delete cascade,
  simulation_id bigint references public.simulations(id) on delete set null,
  engine_hours  double precision not null,
  level         text not null check (level in ('advisory', 'caution', 'warning')),
  subject       text not null,          -- fault or sensor channel
  action        text not null,          -- e.g. 'Oil analysis; inspect filter for metal'
  acknowledged  boolean not null default false,
  created_at    timestamptz not null default now()
);
create index maintenance_events_engine_idx on public.maintenance_events (engine_id, engine_hours);
alter table public.maintenance_events enable row level security;
create policy "select own maintenance" on public.maintenance_events for select using (
  exists (select 1 from public.engines e where e.id = engine_id and e.user_id = (select auth.uid())));
create policy "update own maintenance" on public.maintenance_events for update using (
  exists (select 1 from public.engines e where e.id = engine_id and e.user_id = (select auth.uid())));
```

**Rollback** (if needed before any v4 run is kept):
```sql
drop table if exists public.maintenance_events;
alter table public.telemetry_logs drop column engine_hours, drop column coolant_temp, drop column manifold_pressure_kpa,
  drop column battery_voltage, drop column margin_min, drop column rul_calendar_hours, drop column rul_band_hours,
  drop column wear_limited, drop column fault_modes, drop column residuals, drop column truth;
create index if not exists telemetry_logs_simulation_id_idx on public.telemetry_logs (simulation_id);
drop index if exists public.telemetry_logs_sim_time_idx;
alter table public.simulations drop constraint simulations_v4_time_recorded;
alter table public.simulations drop column engine_id, drop column life_scale, drop column start_engine_hours,
  drop column end_engine_hours, drop column scenario, drop column model_manifest;
alter table public.simulations drop constraint simulations_model_version_check;
alter table public.simulations add constraint simulations_model_version_check
  check (model_version = any (array['v2', 'v3']));
drop table if exists public.engines;
```

### Who writes and reads what

| Column / table | Written by | Read by |
|---|---|---|
| `engines.*` | `dbv4.get_or_create_engine` (at the start of a run, or a preset); `update_engine_hours` (at the end of a run, in the same call as `end_simulation`) | simulate page engine picker; new "Engines" list on the dashboard |
| `simulations.life_scale, start/end_engine_hours, engine_id, scenario, model_manifest` | `dbv4.start_simulation` / `end_simulation` | report, replay, telemetry pages; `timeScale.ts` v4 helpers |
| `telemetry_logs.time_offset_s + engine_hours` | `dbv4.log_telemetry`, every 10 flight s | every chart: x-axis "flight time", secondary axis "engine hours" |
| `telemetry_logs.fault_modes, residuals, truth, margin_min, rul_*` | `dbv4.log_telemetry` | report and replay: fault timeline, twin residual chart, ground-truth overlay |
| `channel_diagnostics` (v4 vocabulary) | `dbv4.log_channel_diagnostics`, 12 rows per log | replay sensor grid |
| `maintenance_events` | `advisory.py` result via `dbv4.log_maintenance_event` (on change only) | engine page: maintenance history |

### Volume
- **v4 writes** 1 telemetry row plus 12 channel rows every 10 flight seconds, about 80 rows per flight-minute.
- **A 10-minute flight** is about 800 rows. **A 1-hour run** is about 4,700 rows, roughly 1.5 MB with the jsonb columns.
- **Current use** is 31 MB of the free tier's 500 MB, which leaves room for several hundred v4 flights.
- **If space ever matters:** move the 12 channel rows into one `sensor_conditions jsonb` column on `telemetry_logs`, and stop writing `channel_diagnostics` for v4. That's 12× fewer rows. It isn't done now, because the replay page's per-channel reads work as they are.

### Checks after applying
1. **Advisors:** run the security and performance advisors on the project. There should be no new warnings, and both new tables should have RLS enabled.
2. **v4 round trip:** a scripted v4 flight writes one `engines` row, one `simulations` row (with `life_scale = 180`), and telemetry rows with both clocks. The report and replay pages read them back.
3. **Life clock carries over:** a second flight of the same engine starts at the first flight's `end_engine_hours`.
4. **Time is always recorded:** inserting a v4 run without `life_scale` **fails**, because the constraint works.
5. **Access rules:** another user can't read the engine or its runs.
6. **Old runs still work:** a v2 run and a v3 run still open on the report, replay and telemetry pages, unchanged.

### Frontend queries to update (C5)
- **Replay** (`app/mission/replay/[id]/page.tsx`): `REPLAY_COLUMNS` gains `engine_hours, coolant_temp, manifold_pressure_kpa, battery_voltage, margin_min, fault_modes, residuals, truth, rul_calendar_hours, wear_limited`, **only for v4 runs**. v2/v3 keep today's column list, because the new columns are null for them.
- **Telemetry detail** (`app/telemetry/[id]/page.tsx`): its select adds `life_scale, start_engine_hours, end_engine_hours, engine_id, scenario`. Its log select adds `engine_hours, margin_min`.
- **Report, dashboard and mission list:** add the new `simulations` columns to their selects.
- **Dashboard:** a new "Engines" card listing `engines` with hours, TBO and status.

---

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| RUL is much worse on predicted inputs (A3) | The headline RUL number drops | Measure first; retrain the RUL head out-of-fold; report the honest number |
| The 1 Hz / 100 Hz mismatch shifts the features (B1) | Live predictions drift from test quality | Measure; fall back to stepping the AI path's engines at 1 Hz |
| Live and training features don't match | Silent garbage predictions | The start-up contract check and parity test (A5) block start-up |
| A 915/916 gate fails | That engine can't ship | Rerun the failed step; triage with that engine's notebook |
| Faults develop too slowly for a live demo | Nothing visible in 5 minutes | Presets start each run just before or after the fault becomes detectable; the ×180 life clock (Spec T) moves it during the flight |
| Old v2/v3 missions break in the UI | Lost history | Version-branched rendering; migration adds columns only |
