# v5 model improvement plan: physics, data, models, serving

**Goal.** Get the best honest results the Rotax digital twin can support, on all four engines (912, 914, 915, 916), and make every number we publish mean what it says. v4 passes its gates, but several gates are hollow, the RUL labels are partly unlearnable, the sensor-fault data is too thin, some faults are invisible in the physics, and the training recipe undoes itself across phases. v5 fixes the evaluation first, then the physics and labels, then the data pipeline and model, then retrains everything and ships it next to v4, with v4 kept as the fallback. Regenerating data, retraining all four engines and changing the architecture are all in scope.

**Written:** 2026-09-25. Same conventions as `docs/v4_integration_plan.md`: ordered phases, each ending with a "Done when" check that fails if the phase is wrong. Items marked **[H]** are hypotheses. Phase 0 or the Phase 2 pilot tests them before anything is built on them.

**Ground rules (carried over from v4).** v2, v3 and v4 keep working. v5 is added as `AERO_PHYSICS_VERSION=v5` / `models_v5`, and nothing is deleted. Every change lands in both checkouts, because the live stack runs from `/Users/harsh/Documents/UAV_Engine`. No commits unless asked. Only one GPU job runs at a time: the machine is an 8 GB Mac with one Metal GPU.

---

## 1. Where v4 stands, and why

Measured on test flights. Per engine, the 914 has 2,433 train, 334 val and 316 test flights (about 10M rows at 1 Hz). Training took about 6.3–6.7 h per engine: the 912 ran 18:18 → 01:01 and the 915 ran 01:01 → 07:18 (`validation_v4/logs/912_915_916_train.log`). The 916 stopped partway through `sensor_fault`, after epochs of 290–4,553 s.

| Head | v4 test result | What is wrong with the gate or the model |
|---|---|---|
| Detection | AUC 0.879–0.925, recall 0.69–0.78 | Zero-offset sensor faults count as positives (`tf_data_pipeline.py:224`). A fault counts as "present" at sev ≥ 0.08 whether or not it has any physical effect (`generate_dataset_v4.py:318`). |
| Diagnosis | macro F1 0.61–0.73 (916: 0.42) | Wastegate and turbo appear only as a product (`physics_v4.py:523`) that boost control hides below 4,572 m. Air filter is ≤ 4.5% (`:498`) and boost compensates for it. Oil faults share one path (`:794-803`) and ignition/combustion share one term (`:591`). The ×12 positive weight inflates probabilities, and the cut-off grid stops at 0.95 (`export_deployable_v4.py:107`). |
| Severity | on-fault error 0.16–0.37 | The gate is an all-cell MAE, and 97.6% of cells are zero (`run.py:88`). `FaultMae` ignores over-prediction (`train_common.py:163-167`). The loss weights every sev > 0 cell by ×40 (`:301`), but selection only looks at cells ≥ 0.08. |
| Sensor fault | macro F1 0.38–0.42 (914: drift 1%, spike 15%, bias 20%, noise 35%, stuck 37%, dropout 81%) | The gate is accuracy with 97.8% "none" (`run.py:89`). There are about 10–15 train flights per (channel, kind) (`sensors_v4.py:114-127`). Each channel has its own weights (`model_architectures.py:177`). Drift grows only about 0.2σ in 128 s (`sensors_v4.py:175`) but is labelled from onset. The window statistics (`model_architectures.py:82-84`) can't see level shifts. |
| Health | error 0.063–0.075 | Sound. It becomes the baseline to beat. |
| RUL | live-aux wear-limited error 18.6–21% of TBO (target 15%). 915 live: 252 h against 257 h for counting down to the overhaul date. | Fault onset is uniform over 0–0.9×TBO, and the label counts faults that haven't started yet (`degradation_v4.py:147,182-229`). Training reads true health, severities and margin, but serving feeds head outputs and measured values (`tf_data_pipeline.py:110-121` vs `aiv4.py:166-176`). The aux inputs are taken from the last row only (`:234-236`). `retrain_rul_live_v4.py:107` fits on in-sample head predictions, and on the 912 it diverged (18.6% → 29%). A GBT on label aux reaches 10.47%. |
| Whole system | 6 checkpoints, CPU and GPU disagree by up to 1.0, no seeds | Each phase weights only its own head (`train_common.py:467`), so the other heads rot, and each phase restarts Adam at 1e-3 on the whole encoder (`:519`). **[H]** A tensorflow-metal op bug, probably in the dilated causal Conv1D. |

Live 914 run: EGT stuck caught 0/67. False CHT drift at altitude (**[H]** the ×180 life clock adds trends inside a window that training never saw, `timescale_v4.py:31`). False oil-temp noise near a CHT dropout. Faults called together within families: oil degradation with bearing or oil pump, and valve leakage with turbo.

---

## 2. Targets (v5 gates)

Every gate is scored on **held-out test flights**. It is reported with a **95% flight-bootstrap CI** (resample flights, not windows, 1,000 draws) next to a **do-nothing baseline**. A gate passes only when (a) the point estimate meets the target and (b) the CI of the difference from the baseline excludes zero. Labels change in Phase 2, so v4 models are re-scored on the v5 test sets as well. v5 data keeps every v4 column (a superset), so this is always possible.

| Head | Metric (new, honest) | v4 now | v5 target | Do-nothing baseline to beat |
|---|---|---|---|---|
| Detection | recall at precision ≥ 0.95, on windows where the fault is *physically visible* (Phase 2 label) | 0.69–0.78 (old label) | **≥ 0.85**; AUC ≥ 0.95 | Rule: max \|residual\|/σ over the window, threshold tuned on val |
| Diagnosis | macro F1 over faults; per-fault recall | 0.61–0.73 (916 0.42) | **macro F1 ≥ 0.70 on every engine**. Recall **≥ 0.5 for each fault Phase 1 marks observable**. Family-level F1 ≥ 0.85. Turbo and wastegate reported separately above and below critical altitude. ECE ≤ 0.05 | GBT on window summary features; class-prior |
| Severity | MAE on fault cells (sev ≥ 0.08); MAE on clean cells; signed bias | 0.16–0.37 / not tracked | **on-fault ≤ 0.12; clean-cell ≤ 0.02; \|bias\| ≤ 0.03** | Per-fault training mean where the fault is present, 0 elsewhere |
| Sensor fault | macro F1 over 7 conditions; per-kind recall; false alarms | 0.38–0.42 | **macro F1 ≥ 0.60**. Recall floors: dropout ≥ 0.90, stuck ≥ 0.70, spike ≥ 0.60, noise ≥ 0.60, bias ≥ 0.50, drift ≥ 0.40 (on the "visible" label). **≤ 0.5% false calls per healthy channel-window** | Hand rules: flat-line run length, NaN/zero, \|Δ\| > kσ, rolling σ ratio |
| Health | MAE | 0.063–0.075 | **≤ 0.05** | engine_hours / TBO wear curve |
| RUL | MAE % of TBO on wear-limited engines, **with live-style aux only** | 18.6–21% | **≤ 12%**, and **≥ 25% lower than the calendar countdown** (TBO − hours), CI of the gain above zero. TBO-limited ≤ 4%. Monotone within a flight (no rise > 2% TBO) | Calendar countdown |
| Parity | max \|trained − served\| on stored test windows, all outputs, both on the GPU | up to 1.0 (v4 CPU vs GPU) | **≤ 1e-3** | — |
| Live | e2e presets and sorties, all engines | EGT stuck 0/67 | Every injected preset fault detected, and its family named, within 3 windows. **Zero** false sensor-fault calls on healthy sorties | v4 on the same runs |

Why these are defensible: the label-aux GBT already reaches 10.47% on RUL, so 12% on live aux is reachable once the labels stop counting future faults. The 912 already reaches 0.73 diagnosis F1 without the Phase 1 observability fixes. Sensor-fault F1 is limited mainly by data (10–15 flights per cell), and Phase 2 fixes that. If Phase 1 shows a fault is physically unobservable in a regime, its floor applies only in the regime where it is observable, and the model card says so.

---

## 3. Phases

### Phase 0: Baseline and evaluation harness

**Goal.** Fix the measuring stick before changing anything, and freeze v4 as the baseline.

**Tasks**
1. `validation_v4/run.py` gates (`:85-92`): replace all-cell severity MAE with on-fault MAE, clean-cell MAE and signed bias. Replace sensor accuracy with macro F1, per-kind recall and the false-alarm rate. Score RUL with **live-style aux** (head outputs, measured margin), not simulator labels.
2. `train_common.py:163-167`: add `FaultBias` and `CleanMae` metrics beside `FaultMae`, so over-prediction becomes visible.
3. New `validation_v4/evaluate.py`, shared by notebooks, export and e2e. It gives breakdowns per fault, per sensor kind and channel, by altitude band (below / above 4,572 m, the 914/915/916 critical altitude), by life stage (thirds of TBO), and by mission type. It also computes flight-bootstrap CIs and the do-nothing baselines from §2.
4. ~~A CPU/GPU layer-parity test~~ **Dropped 2026-09-26** (v5 trains and serves on the GPU only; v4 is left untouched). Was: a CPU/GPU layer-parity test (`tests/test_parity_v4.py`): run each layer of the exported model on the same 256 windows on `/CPU:0` and `/GPU:0`, and report the first layer that diverges. **[H]** It is the dilated causal Conv1D.
5. Seeding: `keras.utils.set_random_seed`, and record the seed in each run's JSON.
6. Run the e2e harness for all four engines (the 916 currently uses 914 placeholders), and log the live-run failures (EGT stuck, CHT drift, oil-temp noise, family co-calls) as fixed regression scenarios.
7. ~~**[H] probes**~~ **Dropped 2026-09-26 for v4** (v4 is left untouched; v5 already varies the environment within a flight, Phase 2 task 5). Was: (a) the flight-ID probe: train a classifier on `isa_dev_c`/`humidity_frac` alone to predict `scenario_id`, then retrain detection without those inputs and compare val/test gaps. (b) Read the 916 epoch-time spikes against `pmset -g thermlog` and process load to separate throttling from contention.
8. ~~Optional (Decision D8)~~ **Dropped 2026-09-26** (v4 is left untouched). Was: finish the v4 916 run (`sensor_fault`, `health`, `rul`, about 4 h overnight) so v4 is a complete fallback.
9. ~~Write `validation_v4/baseline_v4.json`~~ **Dropped 2026-09-26** (v4 is left untouched; its data was deleted). v5 is compared against the do-nothing baselines in `score_v5.py` instead. Was: write `validation_v4/baseline_v4.json` with every number above, per engine.

**Deliverables:** new gates, `evaluate.py`, the parity test, `baseline_v4.json`, a short Phase 0 findings note appended to this file.
**Done when:** v4 is re-scored on all four engines with CIs. The parity test names the diverging op, or proves there is none. The probe results are recorded. The new gates fail v4 where §1 says they should: severity on-fault, sensor F1, live RUL.
**Wall time:** about 1.5 days of work. About 2 h GPU for re-scoring and parity, plus about 4 h overnight if D8 is run.
**Depends on:** nothing.
**Status 2026-09-26:** tasks 1-3 and 5 were done in v5 form (`validation_v5/evaluate.py`, `gates_v5.py`, `seed.py`). Tasks 4 and 6-9 were dropped by the user's decision: v4 is left untouched, and v5 trains and serves on the Metal GPU only.

### Phase 1: Physics and twin fixes

**Goal.** Make each fault observable wherever a real Rotax would make it observable, and nowhere else. Realism is not traded for accuracy.

**Tasks**
1. **Turbo vs wastegate** (`physics_v4.py:515-530`). Stop multiplying the two faults together. Instead:
   - Model the wastegate actuator position that boost control commands. A leaking wastegate needs more closure for the same MAP, and runs out of authority at a lower altitude.
   - Turbo flow/efficiency loss raises the compressor outlet (airbox) temperature at a given pressure ratio, and needs more closure too.
   - Add wastegate position and airbox temperature to the twin residuals if the real engine measures them. **Check:** 914 TCU servo position feedback and airbox temperature sensor; 915 iS / 916 iS ECU wastegate position and IAT. If an engine has no such signal, keep the fault unobservable below critical altitude on that engine and say so.
2. **MAP residual**: `twin_v4.py` and `generate_dataset_v4.py:88` get `manifold_pressure_kpa` as a 7th residual channel. It is informative above critical altitude and on the 912 at full throttle.
3. **Air filter** (`:498`): keep the ≤ 4.5% loss, since clean-to-clogged ΔP of roughly 5→40 mbar is realistic. Make its signature explicit: on the 912, MAP falls with rpm² at wide-open throttle. On turbo engines, more wastegate closure and a higher airbox temperature for the same MAP. **Check** against the Rotax filter-maintenance ΔP figure if one is published; otherwise document the assumption.
4. **Oil family** (`:794-803`). Split the one oil-pressure path by physics that really differs:
   - Pump wear: flow ∝ rpm, so the deficit shows at low rpm, and the relief valve masks it at high rpm.
   - Oil degradation: a viscosity-temperature effect, so pressure drops with oil temperature, with a small temperature rise.
   - Bearing clearance: the pressure deficit grows with rpm × temperature, plus friction power, which shows as higher fuel flow at the same prop load.
   - **Check** against Rotax oil-pressure limits (min 0.8 bar below 3,500 rpm, 2–5 bar normal, 7 bar cold) and oil temperature limits.
5. **Ignition vs combustion** (`:591`). Ignition is modelled as a lost or retarded circuit: an rpm drop of the mag-check order (**check** the Rotax figure, about 100–150 rpm on one circuit), EGT up, CHT down. Combustion inefficiency is modelled through the fuel-flow-to-power and EGT relationship at fixed timing.
6. **Baseline wear vs faults** (`degradation_v4.py:75-82`): keep wear on the same modifiers (that is realistic), but define the fault labels *relative to the wear baseline* (Phase 2 task 2), so that wear alone never counts as a fault.
7. Add a per-fault **observability table** (fault × regime → observable yes/no, and through which channels). It is produced by one-at-a-time runs at fixed severity 0.5 over a grid of altitude, power and ISA conditions. It drives the §2 per-fault floors and the UI's family-level answers.
8. Update `twin_v4.py` → `twin_v5.py`, and the feature contract (29 → 30–32 inputs). Rebuild and verify the MATLAB twins with `build_rotax_v4('<N>')` → `v5` and `verify_rotax_v4` (the bar stays ≤ 4e-15 on all four engines).

**Deliverables:** `physics_v5.py` (a copy with the diff above), `twin_v5.py`, the observability table, updated MATLAB twins, and the physics test suite extended with the new checks.
**Done when:** every existing Rotax-figure test still passes. The new oil, ignition and turbo signatures each have a test. In the one-at-a-time runs, each fault's residual signature differs from its family siblings by ≥ 2σ on at least one channel in at least one regime, or the fault is marked unobservable. MATLAB matches to ≤ 4e-15.
**Wall time:** 2–3 days of work, about 1 h compute.
**Depends on:** Phase 0 (so the effect is measured with the new harness). **Risk:** no published Rotax figure for a signature → use the most conservative magnitude and label it an assumption in the model card. Don't invent observability.

### Phase 2: Data generation v5

**Goal.** Labels that can be learned, enough examples of every class, no leaks. Pilot first, then generate everything.

**Tasks**
1. **RUL label without future faults** (`degradation_v4.py:147,182-229`). Label = min(TBO − hours, the time until *already-active* faults plus baseline wear reach the limit, projected by the known progression law). Keep the old label as `rul_hours_oracle` for reference. Optionally make fault onset a Weibull hazard rising with hours instead of uniform. **Check** this is not less realistic.
2. **Fault presence by physical effect** (`generate_dataset_v4.py:318`). Run a noise-free "healthy + wear" twin alongside the faulty engine. A fault is present when its noise-free effect on any residual channel is ≥ 1σ of that sensor's noise in the window. Keep `max_sev` for the severity head. The same rule applies to detection (`tf_data_pipeline.py:224`): a sensor fault counts only once its offset is visible.
3. **Sensor-fault redesign** (`sensors_v4.py:114-127,175`):
   - 60% of flights get faults, and every (channel, kind) cell is balanced by construction, aiming for ≥ 60 train flights per cell.
   - Drift is labelled from when it becomes visible (\|offset\| ≥ 1σ), and its rate is raised to span realistic thermocouple and pressure-sensor drift, **checked** against sensor datasheets.
   - Onsets are allowed inside the first 128 s, so windows contain the transition.
   - Labels come from visibility, not onset, for every kind.
4. **Turbo coverage**: more high-altitude and high-power time (`generate_dataset_v4.py:148`). Take the high_altitude mission from about 1/8 to about 1/4 of turbo-engine flights, and add climbs through critical altitude to other missions.
5. **Environment varies within a flight** (`:192-194,310`): ISA deviation and humidity follow a slow random walk (±2–3 °C and ±0.1 over an hour), so they stop identifying the flight. Use this only if the Phase 0 probe confirms leakage **[H]**, but it is realistic either way.
6. **Life clock**: generate about 40% of flights at the live ×180 life scale (`timescale_v4.py:31`), so the trends that appear inside a window at live speed are in the training data **[H: CHT false drift]**.
7. **Splits**: by flight, **stratified** on engine life stage, wear-limited vs TBO-limited, mission and fault family. The split is 60/15/10/15 train / calibration / val / test, and the calibration split is used only by Phase 4 calibration and RUL (Decision D4). Flights: about 6,000 per engine, about 2× v4.
8. **Storage** (`:360-363`): one row group per flight, plus a per-engine `index.parquet` (flight → file, row offset, length). Phase 3 builds its cache from that.
9. **Pilot first**: 600 flights on the 914. Then check whether the targets are learnable at all: LightGBM on window summary features for detection, diagnosis, sensor kind and RUL (live-style aux). The pilot must beat its v4-data equivalent on diagnosis for the separated faults, on sensor-kind recall, and on RUL-without-future-faults. If not, go back to Phase 1 or the labels. Don't generate the full set.
10. Update `backend/data_checks` (the 13 v4 checks), and add: per-cell sensor counts, stratification balance, a check that no RUL label uses a future onset, and the visibility-label rate.

**Deliverables:** `generate_dataset_v5.py`, `data/rotax_v5/{912,914,915,916}` (about 2 GB per engine), the pilot report.
**Done when:** the pilot GBTs clear the bar above, all data checks pass on all four engines, and the v4 columns are present so v4 can be re-scored.
**Wall time:** 2 days of work. Pilot and GBT about 1.5 h. Full generation is CPU-bound: about 1.5–2.5 h with `--parallel` (v4 took about 35 min for about 3,100 flights × 4, and v5 has about 2× the flights plus the noise-free twin run), or about 6–8 h with the sequential default.
**Depends on:** Phase 1. **Risk:** 8 GB RAM with 4 parallel generators → use `--jobs 2` if swap appears.

### Phase 3: Data pipeline

**Goal.** Fast, deterministic, identical features in training and serving.

**Tasks**
1. Window cache: per-engine float16 memmaps of features and labels, plus a flight index. Windows are sliced by (flight, start), which replaces the filtered per-flight parquet reads (`tf_data_pipeline.py:203`).
2. **Per-epoch reshuffle** (`:201`) plus random start jitter within stride. This is free augmentation, since v4 windows barely overlap at about 60k windows from 7.9M rows.
3. **RUL aux**: window **mean** and robust max (95th percentile) instead of the last row (`:234-236`). Residuals in σ units, not raw units. Clip to ±6σ.
4. **Long-horizon context** (Decision D2): the twin keeps EMA residual summaries (τ = 10 min and 60 min) and a per-channel slope over the last 15 min. These are extra inputs, computed identically by `twin_v5.py` for training and serving. They are what makes slow drift visible.
5. **Input contract v5**: one `contract_v5.json` (feature order, units, scaler, window, aux definition, life scale). It is read by the generator, the pipeline, the twin and the service, and checked at startup in each of them. This extends the v4 contract checks.
6. Throughput benchmark: target **≤ 120 s per epoch** on 2× the v4 data **[H: v4 was I/O-bound at about 0.8 s/step]**.

**Deliverables:** `tf_data_pipeline_v5.py`, cache builder, `contract_v5.json`, benchmark log.
**Done when:** features from the cache match the live `twin_v5` path to ≤ 1e-5 on 20 replayed flights. Two epochs with the same seed give identical batches. The epoch-time target is met.
**Wall time:** 1–1.5 days of work, about 1 h to build the cache for all engines.
**Depends on:** Phase 2 (schema). The code can be written while Phase 2 generates.

### Phase 4: Model redesign

**Goal.** One jointly trained model, a sensor classifier that shares what it learns across channels, calibrated probabilities, and RUL as its own honest model.

**Tasks**
1. **Joint multitask model** (`model_architectures.py`): one dilated-TCN encoder with all heads trained together. Loss balancing uses **uncertainty weighting** (Kendall et al.), with GradNorm as the alternative the HP search compares. One checkpoint per engine, which replaces the six (`train_common.py:467`).
2. **Sensor head**:
   - One classifier shared by all channels (`:175-178`), run on each channel's feature vector plus a learned 8-d channel embedding.
   - Per-channel inputs: the encoder's cross-channel context, the raw channel sequence, and statistics for each fault kind:
     - level shift: mean of the last 32 s minus the first 32 s
     - OLS slope
     - longest run of identical values ÷ the expected run for that channel's quantisation and rate of change, so slow channels near quantisation don't look stuck
     - fraction of NaN or zero values
     - max \|Δ\| ÷ robust σ
     - high-pass σ ÷ the sensor's rated noise
     - agreement with the physically coupled channels (e.g. CHT ↔ EGT ↔ coolant)
     - the Phase 3 long-horizon EMA
   - Loss: focal or class-balanced by effective number, instead of a flat ×8 (`train_common.py:210`).
3. **Diagnosis**:
   - Remove the ×12 positive weight, or apply logit adjustment by log(pos_weight) at inference.
   - Per-fault temperature on the calibration split.
   - Cut-off grid 0.01–0.99 (`export_deployable_v4.py:107`).
   - An auxiliary **family head** (oil, ignition/combustion, induction/turbo, cooling, mechanical). The UI shows the family when the within-family margin is small, and the specific fault otherwise.
4. **Severity**: loss on sev ≥ 0.08 cells plus a clean-cell term, with symmetric penalties, so over-prediction is punished (`train_common.py:301`).
5. **RUL**: a separate model on the calibration split. Its inputs are live-style aux: the joint model's out-of-sample head outputs, measured margin, the Phase 3 aux, hours. Compare GBT (LightGBM, monotone constraint on hours) against a small MLP, and ship the better on val. Smooth monotonically in the service (isotonic over the flight). `retrain_rul_live_v4.py` is retired.
6. **Regularisation**: dropout / weight decay, input-channel dropout (it also hardens against sensor dropout), and mild time-warp / gain augmentation in the plausible range.
7. **HP search on the 914 pilot data first**: encoder width/depth, dilation stack, loss balancing method, LR, focal γ. Use Optuna with ASHA pruning, 20–30 short trials.

**Deliverables:** `model_architectures_v5.py`, `train_v5.py`, HP search report.
**Done when:** the best config beats the v4 baseline on the 914 pilot on every head except health, where it must match within CI. The joint model's heads don't lose > 0.02 against the single-head versions. Calibration ECE ≤ 0.05 on val.
**Wall time:** 2–3 days of work. HP search about 8–10 h GPU (overnight).
**Depends on:** Phases 0 and 3. **Risk:** joint training underfits one head → the head-freeze fine-tune in Phase 5 covers it.

### Phase 5: Training protocol

**Goal.** Train all four engines reproducibly, 914 first.

**Tasks**
1. **914 end to end first**: joint training with early stopping on a composite of the §2 metrics (val), not on loss. Then a head-only fine-tune with the encoder frozen. Then an optional unfreeze at **LR ≤ 1e-4** with a warm-up. The v4 recipe restarted at 1e-3 (`train_common.py:519`), so this replaces it.
2. Fit the RUL model and the calibration temperatures and cut-offs on the calibration split. Upgrade (Decision D4): 3-fold out-of-fold aux from 3 encoders, if the calibration split proves too small.
3. Seed variance: repeat the 914 with a second seed. If the per-head spread is larger than the CI, report it, and consider D5.
4. **Then 912, 915, 916** with the 914 config. Only the class weights change per engine. Log epoch times, and keep the machine idle during runs (no generation, `caffeinate`) **[H: contention caused the 916 spikes]**.
5. Ensembles (D5): only if a 3-seed ensemble on the 914 gains more than the CI on diagnosis or sensor F1 *and* the Phase 6 latency budget holds.

**Deliverables:** `backend/models_v5/<engine>/` checkpoints, training logs, `results_v5.json`.
**Done when:** every §2 gate passes on the 914, then on each other engine. Where one doesn't, the failure is explained by the observability table and signed off in the model card.
**Wall time:** about 3–5 h GPU per engine **[H, depends on the Phase 3 benchmark]**, so about 14–20 h for all four. A 3-seed ensemble roughly triples that.
**Depends on:** Phase 4.

### Phase 6: Export and serving

**Goal.** What runs live is exactly what was scored.

**Tasks**
1. `export_deployable_v5.py`: one SavedModel per engine, plus a manifest (`contract_v5` hash, calibration temperatures, cut-offs, family map, the observability table, the RUL model, the seed and metrics).
2. `backend/aiv5.py` (v4's `aiv4.py` stays): long-horizon features from `twin_v5`, and RUL aux as window mean with the same σ scaling (the fix for the `aiv4.py:166-176` skew). A monotone RUL smoother. Family-level output.
3. **Device: the Metal GPU, as in v4** (user decision, 2026-09-26). Training, calibration, scoring and serving all run on the GPU, so the thresholds and RUL numbers are measured on the device that serves them. `train_v5.py` refuses to run without a GPU. Budget: ≤ 50 ms per window per engine.
4. A training-vs-live parity check at startup: replay 3 stored test windows through the service, and require the output to match the exported test output to within 1e-3.
5. Selectable `AERO_PHYSICS_VERSION=v5`. v4 remains the fallback.

**Deliverables:** export script, `aiv5.py`, manifest, startup parity check.
**Done when:** the startup check passes on all four engines, the latency budget holds, and v4 still serves when selected.
**Wall time:** 1.5–2 days of work, < 1 h compute. **Depends on:** Phase 5 (the 914 can go ahead of the others).

### Phase 7: Validation and rollout

**Goal.** Prove it end to end, then document it.

**Tasks**
1. e2e: all engines × all presets and mission sorties, including the Phase 0 regression scenarios (EGT stuck, CHT drift at altitude, oil-temp noise near a CHT dropout, family co-calls).
2. Live-stack runs on each engine via `scripts/start_stack.sh`, with altitude splits for turbo faults.
3. A v5 vs v4 comparison table with CIs, from `evaluate.py`.
4. Update `docs/model_cards_v5.md`, the About page numbers (`frontend/app/about_us/page.tsx`), `docs/qa_prep.md` and the demo script. Every number cites its source JSON.
5. Make v5 the default only after the user signs off. v4 stays selectable.

**Done when:** every §2 live gate passes, the model cards match the JSONs, and the user signs off.
**Wall time:** 1–2 days of work, about 3–4 h compute.

---

## 4. Timeline

| Phase | Hands-on work | Compute wall time | Overnight-able | GPU-bound | Blocks |
|---|---|---|---|---|---|
| 0 Baseline & harness | 1.5 d | 2 h (+4 h for D8) | yes (D8) | yes | 1, 4 |
| 1 Physics & twin | 2–3 d | 1 h | — | no (MATLAB/CPU) | 2 |
| 2 Data v5 (pilot → full) | 2 d | 1.5 h pilot + 1.5–2.5 h full (parallel) | yes | no (CPU) | 3, 4 |
| 3 Pipeline & cache | 1–1.5 d | 1 h | — | no | 4, 5 |
| 4 Redesign + HP search | 2–3 d | 8–10 h | yes | yes | 5 |
| 5 Train 4 engines | 0.5 d supervision | 14–20 h (×3 with ensembles) | yes | yes | 6 |
| 6 Export & serving | 1.5–2 d | < 1 h | — | no | 7 |
| 7 Validation & rollout | 1–2 d | 3–4 h | partly | partly | — |
| **Total** | **about 12–16 working days** | **about 32–42 h** (about 60–80 h with ensembles) | | | **about 3–4 calendar weeks** |

**Parallel work:** Phase 1 code while Phase 0 re-scores or D8 runs. Phase 3 and 4 code while Phase 2 generates (CPU). Phase 6 code while Phase 5 trains the 912/915/916. **Never in parallel:** two GPU jobs, or generation during training. There is one 8 GB Metal GPU with unified memory, and Phase 0 checks whether this contention caused the 916 slowdown.

---

## 5. Decisions for the user

| # | Decision | Options | Recommended default |
|---|---|---|---|
| D1 | Accept physics changes? | (a) Phase 1 as written: new observables such as wastegate position, airbox temperature, MAP residual, and separate oil and ignition paths. (b) Keep physics_v4 and report the invisible faults as unobservable | **(a), but only where a Rotax document or datasheet supports the signal.** Anything unsupported falls back to (b) for that fault |
| D2 | Window length | (a) 128 s plus long-horizon EMA/slope features from the twin. (b) 512 s window | **(a)**: about 4× cheaper to train, and no 8.5-min wait to fill the window live |
| D3 | Joint model or one checkpoint per head | Joint + head fine-tune / v4-style per head | **Joint**. Per-head only if Phase 4 shows a head loses more than 0.02 |
| D4 | Where RUL/calibration get out-of-sample inputs | (a) 15% calibration split. (b) 3-fold OOF (about +2 training runs per engine) | **(a)**. (b) only if the calibration split's CI on RUL is wider than ±2% TBO |
| D5 | Ensembles | None / 3 seeds | **None**, unless the 914 test in Phase 5 shows a gain beyond the CI within the ≤ 50 ms per-window budget |
| D6 | Serving device | CPU after the parity fix / Metal GPU as in v4 | **Decided 2026-09-26: Metal GPU only**, for training and serving |
| D7 | Live life clock | Train on mixed ×1/×180 flights / lower the live scale | **Mixed training**: keep the agreed ×180 demo pace |
| D8 | Finish the v4 916 run (about 4 h overnight) | Yes / no | **Decided 2026-09-26: no.** v4 is left untouched |
| D9 | Family-level answers when faults are indistinguishable | Yes / always name one fault | **Yes**: honest, and matches the observability table |

---

## 6. What we keep

- **The health head** and its target recipe. It is already the strongest head, and v5 only has to hold ≤ 0.05.
- **The healthy on-board twin** idea (`twin_v4.py`) and the residual features. v5 extends them, it doesn't replace them.
- **physics_v4's Rotax-checked behaviour**. Phase 1 changes are additive and each one is tested.
- The **MATLAB/Simulink twins** and the verify workflow (≤ 4e-15).
- **Contract checks**, the e2e harness, the live stack, `timescale_v4.py`, the presets, and the version switch. v4 stays selectable.
- **Split by flight**, the zstd parquet source data, and the data-check suite (extended, not replaced).

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Phase 1 finds no real-engine signal separating turbo from wastegate (or oil siblings) | Mark the fault unobservable in that regime. Answer at family level. The per-fault floor doesn't apply there |
| The pilot GBT can't learn the new labels | Stop before full generation, revisit the labels or physics. Cost: hours, not days |
| Metal parity bug not fixable | Not a risk for v5: it is trained, scored and served on the same GPU, and the Phase 6 startup check compares the trained outputs with the served ones |
| Joint training hurts one head | Freeze the encoder and fine-tune that head only, or use a per-head checkpoint for that head (D3) |
| Epochs stay erratic | Cache in RAM-sized memmaps, run at night with nothing else running, check `pmset` thermals; smaller batches if memory pressure |
| RUL still > 12% | Ship whichever of the GBT / MLP is better if it beats the calendar with a CI above zero. Report honestly |
| v5 worse than v4 on any head in e2e | That head's v4 output stays selectable. v5 isn't made the default until the user signs off |

---

## 8. Status on 2026-09-26 and the plan to the end

**Done**
- Phases 1–4.
- Phase 2: all four datasets. Each has 6,000 flights and about 19.5M rows, and passes all 14 checks.
- Phase 3: all four caches.
- Phase 4: the settings search, finished at 08:30. The winner is trial 13: 96 channels, 6 layers, dropout 0.2, LR 6e-4, weight decay 1e-4, `balance=sum`, batch 128. Pilot validation composite 1.41.
- Phase 0: v4 tasks dropped. v4 is left untouched.

**Running:** `validation_v5/train_all_v5.sh` is training the 914 (started 08:30, 11–14 min per epoch on the GPU). After it, the 912, 915 and 916 run in turn.

| Step | What | Who | Est. finish | Done when |
|---|---|---|---|---|
| 5.1 | 914: joint → heads-only → unfrozen low-LR; then calibrate, RUL, test, card | chain | ~26 Sep 16:00 | `artifacts/914/model_card.md` exists |
| 5.2 | **914 gate review**: read every FAIL against the observability table; decide fix-and-retrain vs sign-off | Claude + user | same day | each FAIL explained or fixed |
| 5.3 | 912 → 915 → 916 (same config) | chain | ~27 Sep 13:00 | 4 model cards |
| 5.4 | 914 second seed (seed variance, Phase 5 task 3) | Claude | ~27 Sep 20:00 | spread per head vs CI reported |
| 5.5 | Ensembles (D5): only if seeds disagree by more than the CI | decision | — | — |
| 6.1 | `validation_v5/export_deployable_v5.py` → `backend/models_v5/<e>/` (weights, config, contract hash, temps, cut-offs, family map, observability, RUL model, seed, metrics, 3 parity windows + their outputs) | Claude | code during 5.3, run after | manifest per engine |
| 6.2 | `backend/twin_v5.py` (live healthy twin on physics_v5) + `backend/aiv5.py` (live features, long-horizon EMA context, RUL + in-flight smoother, family-level answers) | Claude | during 5.3 | unit tests vs cache windows |
| 6.3 | Startup parity: replay stored windows, max diff ≤ 1e-3, on the GPU; latency ≤ 50 ms/window | Claude | after 5.3 | passes on 4 engines |
| 6.4 | `AERO_PHYSICS_VERSION=v5` switch in the backend and the AI service; v4 stays the default and untouched | Claude | after 6.3 | both versions serve |
| 7.1 | e2e: all engines × presets × mission sorties, plus the known live failures (EGT stuck, CHT drift at altitude, oil-temp noise, family co-calls) | Claude | +1 day | report |
| 7.2 | Live-stack runs per engine (`scripts/start_stack.sh`), turbo faults above and below critical altitude | Claude + user | +0.5 day | screenshots / logs |
| 7.3 | v5 vs v4 table: v5 test CIs against v4's recorded numbers (v4 not rescored; different test sets, stated) | Claude | +0.5 day | table in model cards |
| 7.4 | `docs/model_cards_v5.md`, About page, `docs/qa_prep.md`, demo script; every number cites its JSON | Claude | +0.5 day | docs match JSONs |
| 7.5 | v5 becomes the default | **user sign-off** | — | — |

**Rules while it runs:**
- One GPU job at a time. Phase 6 code is written during training, but its GPU tests wait for a gap between engines.
- The machine stays plugged in with the lid open.
- No deleting data. v4 stays untouched.

**Risks seen so far:**
- Pilot validation numbers are far below the gates: diagnosis AP 0.27, health MAE 0.20. The full data has 10× the flights, but some heads may still fail at 5.2. The first fixes are in §7.
- The 914's epoch 1 dropped the composite from 1.32 to 0.89. Selection keeps the best epoch. If it keeps swinging, add an LR-on-plateau schedule and retrain (≈ 6 h).
- Per-engine time is ~6–7 h, not the 3–5 h first estimated, so all four take ≈ 26–28 h.
