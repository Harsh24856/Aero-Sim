# Plan: v5 specialists into the live stack (914 first)

**Written:** 2026-09-28. **Status (2026-10-02):** Phases 1-5 are built for the 914 (see "Progress" at the end).
Phase 6 is partly done: the side-by-side check shows the 914 ready, but the 912/915/916 still run on the
914's placeholder models, so the default stays v4 until Phase 7 trains them.
**Source of truth for the models:** `validation_v5/artifacts/914b_specialists/` (`model_card.md` there).

## Where we are

| | State |
|---|---|
| 914 v5 specialists (`specialists_v5.py 914b`) | Trained with the Metal ReLU fix and the v5b labels. On the test flights, every head's primary metric meets its target. |
| Gates passed | Detection 0.957, sensor fault 0.889 (0.10% false alarms), health 0.033 |
| Primary met, a secondary target still failing | Diagnosis 0.809 (valve leakage recall 0.47); severity 0.072 (healthy-cell error, bias); RUL 10.8% (TBO-limited 5.1% against a 4% target). The severity gate and the gated RUL are being re-scored now. |
| Live stack today | Physics **v4** by default: `main.py` + `aiv4.py` + `backend/models_v4/`. There is no v5 twin, no v5 AI service and no v5 UI. |
| 912 / 915 / 916 | Data generated. Not relabelled, not trained. |

**Rule kept from before:** finish the 914 end to end, live, before any other engine gets trained.

---

## Phase 1: Split the specialists into a deployable export

The training artifact holds five full networks. Each network has all six heads, but only one of them was trained. Serving should load only what each specialist answers.

1. **`validation_v5/export_v5.py <key>`** writes `backend/models_v5/<key>/`:
   - For each specialist, a **slim model** keeping only its own output: detection, diagnosis + family, severity, sensor and health. The weights are copied from the specialist, and the architecture is rebuilt from `model_architectures_v5` (no `.keras` deserialisation and no custom-layer registry).
   - `assembly.json` + `assembly.pkl`: the detection and health GBT stackers, the sensor "none" bias, the diagnosis temperatures and cut-offs, and the severity gate.
   - `rul.pkl` (GBT, or gated GBT plus classifier) and the RUL feature order.
   - `contract_v5.json` copied from the cache: feature order, scaler, residual sigmas, context columns and label version (`v5b`).
   - `manifest.json`: engine, model version `v5`, git SHA, training config, test metrics taken from `results_test.json`, and the label version.
2. **Parity test** (`validation_v5/tests/test_export_parity_v5.py`): the export and the training `Specialists` object must give identical outputs, within 1e-5, on 2,000 test windows, for every head plus RUL.
3. **Device parity:** the same test run on the CPU. This should now pass, because the ReLU fix makes the maths device-independent. If it does, the "GPU only" rule no longer applies to v5.

**Done when:** both parity tests pass and the export is under 20 MB.

## Phase 2: Refactor so training and serving share one assembly

Today `specialists_v5.Specialists` needs a training `Cache`. Split it:

- **`assembly_v5.py`** (in `validation_v5/`, imported by the backend the way `aiv4` imports `validation_v4`) holds a cache-free `Assembly(models, extras)` with `__call__(seq, ctx) -> heads`. It also holds the stacker features, the severity gate and the sensor bias.
- `specialists_v5.Specialists` becomes a thin wrapper: `Assembly` plus the cache plumbing for training and scoring.
- `rul_v5.features` gets a variant that takes live inputs (the assembly outputs, the measured margin, the throttle mean and the context) instead of cache rows.

**Done when:** re-running `specialists_v5.py 914b --force test` reproduces `results_test.json` exactly.

## Phase 3: Backend

1. **`backend/twin_v5.py`**, mirroring `twin_v4.py`: the flown engine `physics_v5.PistonEngineV5` + `sensors_v5.SensorBank` + a healthy on-board twin. Each second it emits `features_v5.FEATURE_COLS`, using `features_v5.residuals(...)`. `features_v5.LongHorizon` streams the 42 long-horizon context values per flight, and the twin adds hours fraction, life used and the life-clock flag (the 45 context values). The flight-state rules (scenarios, fault injection, engine records) are reused from v4 wherever the physics allows.
2. **`backend/aiv5.py`** (port 8100, run instead of `aiv4`):
   - a 128-second rolling window, scaled exactly as `pipeline_v5` does: residuals divided by the sensor sigma first, then mean/std from `contract_v5.json`;
   - the context scaled with `ctx_scaler`;
   - `Assembly` on each window; RUL from `rul.pkl`, then `rul_v5.smooth_within_flight` over the flight;
   - a response with detection, the 13 fault probabilities, present flags using the calibrated cut-offs, family-level answers, severities (gated), the sensor condition per channel, health, RUL hours with a `wear_limited` flag, and `model_version: "v5"`, `labels: "v5b"`, `placeholder_models`;
   - **placeholders:** as in `aiv4`, the 912/915/916 are served by the 914 export until their own exports exist, and every response says so;
   - a start-up contract check: the features and context must equal the manifest's lists, or the service refuses to start.
3. **`main.py`:** add `AERO_PHYSICS_VERSION=v5` as a new branch (`twin_v5`, the `aiv5` feature list, the envelope floor). **v4 stays the default** until Phase 6 passes.
4. **`dbv5.py`**, or `dbv4` with `simulations.model_version = 'v5'`: persist v5 runs next to v4 ones. No migration of v4 data.
5. **`scripts/start_stack.sh`:** `v5) AI_MODULE=aiv5`.
6. **`scripts/e2e_v5.py`:** fly a scripted mission for each fault family through the live stack, and check that the AI output matches `Assembly` offline on the same logged windows.

**Done when:** `AERO_PHYSICS_VERSION=v5 scripts/start_stack.sh` flies all of the 914's scenarios, the `e2e_v5` parity passes, and one second of AI work takes under 100 ms.

## Phase 4: Frontend

1. **`frontend/lib/v5.ts`:** the v5 result types. Six heads, 13 faults, 6 families, 7 sensor kinds on 12 channels, and RUL with `wear_limited`. Fault and sensor labels come from `lib/v4.ts`; move them to a shared `lib/labels.ts`.
2. **`DiagnosticsV5.tsx`**, extending `DiagnosticsV4`:
   - Detection, with confidence.
   - **Diagnosis at family level when unsure.** Show the fault when its calibrated probability clears its cut-off, and otherwise the family (for example "Oil system", covering oil pump / oil degradation / bearing). This matters for valve leakage (recall 0.47) and oil pump (F1 0.52).
   - Severity, only for faults diagnosed present, labelled "effective severity" (the v5b meaning).
   - A sensor-fault grid, 12 channels × condition, with bias and drift shown as "suspected" below high confidence.
   - Health and RUL: hours left, with the calendar value shown when the engine is not wear-limited.
3. **`ModelBadge`:** show `v5 · 914 own model` or `v5 · 914 model (placeholder)`, with a link to the model card metrics from the manifest.
4. **`simulate/page.tsx`:** choose between `DiagnosticsV4` and `DiagnosticsV5` from `state.ai_model_version`. No route changes.
5. **Mission report and replay:** read `model_version` per run so old v4 runs still render.

**Done when:** a v5 flight shows every panel, v4 runs still replay, and the demo scenarios read correctly. Fix the known v4 demo defects first: cooling severity, CHT dropout versus part status, and the LLM "engine stopped" line.

## Phase 5: Remaining 914 gaps (can run in parallel with Phases 3–4)

| Gap | Plan |
|---|---|
| Severity: healthy-cell error and bias | The severity gate is being re-scored now. If bias stays above 0.03, add a per-fault affine correction fitted on the calibration half. |
| RUL, TBO-limited error 5.1% against 4% | The gated RUL candidate is being re-scored now. Next lever: add the health GBT's output and the long-horizon trends as RUL features. |
| Valve leakage recall 0.47 | Regenerate with per-channel fault visibility (`relabel_v5b.py` point 2): the generator stores the per-channel noise-free effect, and visibility is judged against the healthy spread of that channel. |
| Engine-fault visibility labels | The same generator change as above. Then regenerate the 914 (about 2 h of CPU), relabel, and retrain only the specialists that change. |

## Phase 6: Switch the default to v5

- Fly the demo missions on v4 and v5 side by side; v5 must be at least as good on every panel.
- Update `README.md` / `backend/README.md` with the v5 run instructions, and with the Metal ReLU finding that replaces the old "GPU only" explanation.
- Switch the default in `main.py` and `start_stack.sh` to `v5`. Keep v4 selectable.

## Phase 7: The other engines (only after Phase 6)

For each of 912, 915 and 916: `relabel_v5b.py <key>`, then `specialists_v5.py <key>b` (about 8 h each on the GPU), then `export_v5.py`, and finally drop the placeholder. The 912 has no turbo, so its turbo channels are constant and the applicable-fault mask already handles that.

## Risks

- **Metal:** any new `Dense` + ReLU brings the bug back. `tests/test_graph_parity_v5.py` must stay in the test run, and serving must use the same code path as training.
- **Label meaning changed (v5b):** severity is now *effective* severity, and small sensor offsets count as "none". The UI text and the model card must say so.
- **Live context:** `LongHorizon` needs a warm-up of about 60 minutes before the 60-minute EMA settles. Serve with the partial context, as training saw from each flight's start, and show a "context warming" flag.

---

## Progress (2026-10-02)

| Phase | State | Evidence |
|---|---|---|
| 1 Export | done | `export_v5.py`; `tests/test_export_parity_v5.py`: export == training; CPU == GPU |
| 2 Shared assembly | done | `assembly_v5.py`; the 914b test step reproduces all 109 scorecard numbers |
| 3 Backend | done | `twin_v5.py`, `aiv5.py`, `main.py` v5 branch, `start_stack.sh` v5; backend tests; live HTTP + WebSocket flight with fault and sensor injection; `e2e_v5.py` 914: 8/8 |
| 4 Frontend | done | v5 answers in v4's shape: the v4 panel shows families ("Suspected: <family>"), effective severity, the v5 badge; `tsc` clean; checked in the browser against a live v5 stack |
| 5 914 gaps | done in code | severity isotonic calibration, RUL without the sensor-fault signal, cut-offs with a 0.55 recall floor; generator stores per-channel fault effects (`eff_<channel>`) and `relabel_v5b.py --per-channel-faults` uses them |
| 6 Default switch | not switched | `e2e_v5` vs `e2e_v4`, same flights: 914 v5 8/8 vs v4 6/8; but 915/916 on placeholders flag healthy engines - by this plan's own rule the default waits for Phase 7 |

### 2026-10-03: the per-channel relabel hid faults; root cause was the twin

The 914 was regenerated with `eff_<channel>` (identical data plus 14 columns) and
relabelled with `--per-channel-faults`. That made valve leakage 6 test windows and
oil degradation 0: the gate would pass by deleting the fault. It was reverted; the
deployed 914b models re-score to an identical card.

Root cause: the on-board twin was a brand-new engine (`P.Health()`), while every
simulated engine ages normally. In-limit wear alone put a healthy engine's rpm
residual at -5 to -13 sigma and oil pressure at -7 to -16 by mid-life, so faults that
show on rpm and oil pressure (valve leakage, compression loss, oil and bearing
faults) drowned in a wear clock.

Fix: the twin runs `degradation_v5.fleet_wear_health(hours, tbo)` - fleet-average
wear at the engine's logged hours, never the engine's own wear rate. On 120
regenerated flights, healthy q99 drops from 16.2 to 5.8 sigma on rpm, 31 to 8.1 on oil
pressure, 9.4 to 3.7 on fuel flow; medians go to ~0. The contract carries
`twin: fleet_wear`; `twin_v5` serves each export the twin it learnt (exports
without the key keep the new-engine twin, so the live 914b is unchanged until it
is retrained). Tests: `backend/tests/test_fleet_wear_v5.py`.

Every engine's residual inputs change, so all four must be regenerated and
retrained once. Use the standard v5b labels (no `--per-channel-faults`: it still
hides oil degradation).

**914 done (2026-10-03, #10):** retrained on the fleet-wear twin, 6/6 gates, exported.
On 22 identical flights against the old models: exact naming 88% -> 96%, healthy false
alarms 23/30/19% -> 0/18/11%, RUL error 167 -> 145 h. Weak spots: one valve-leakage
flight named 74% (mistaken for injector fouling), oil-pump RUL, EGT-stuck detection.

Still to run (long jobs): the 912, 915 and 916, one after another, with
`validation_v5/run_fleet.sh` (regenerate, checks, relabel, gate, retrain; ~13 h each):

    nohup caffeinate -is bash validation_v5/run_fleet.sh > validation_v5/logs/fleet.log 2>&1 &

Then per engine: review the card, `export_v5.py <key>b`, the parity test and
`e2e_v5.py --engines <key>`. Then switch the default (Phase 6).
