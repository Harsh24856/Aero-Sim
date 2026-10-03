# Fixing the v5 Weak Spots (v6: Engine History) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three known weak spots of the v5 models, then bring all four engines onto the fixed pipeline once:
- W1: shallow valve leaks are missed;
- W2: remaining life (RUL) is biased;
- W3: a stuck sensor is called about 2 minutes late.

**Architecture:** The 2026-10-03 investigation (plan.md, section W) traced W1 and W2 to one cause: the system sees one flight at a time and knows nothing about this engine's past. Two changes follow from that:
- The on-board twin is **calibrated to this engine** from a logbook-style history taken some hours earlier, not to the fleet average. That shrinks healthy rpm and oil-pressure noise (W1).
- The RUL model gets that **history as extra inputs**: how fast this engine wears compared with the fleet, and how severe its faults were then (W2).

W3 is separate: a serving-side flat-line rule in `aiv5.py`, with no retraining.

**Tech Stack:** Python 3.11, numpy, pandas/pyarrow, scikit-learn (RUL), TensorFlow-Metal (specialists), FastAPI (aiv5), bash.

**Spec:** this plan carries its own spec; there is no separate design doc. The evidence for every number is in plan.md, section W, and in the commit messages of PRs #9, #10 and #12.

## Global Constraints

- **Engine order:** one engine at a time, 914 first (user rule). An engine's models merge only if they beat what is live on the 22-flight A/B and on the test card.
- **No merge to `main` without the user's OK.** Copy every merged change into the main checkout (`~/Documents/UAV_Engine`), which runs the live stack.
- **Keep Metal ReLU as `tf.maximum`** (`model_architectures_v5.relu`). Keep `tests/test_graph_parity_v5.py` in every run.
- **Clean exports only:** `export_v5.py` must run on committed code (`git_dirty: false`).
- **Old exports keep working.** An export whose contract has no `twin` key, or `twin` = `fleet_wear`, must still get the twin and the RUL inputs it was trained with.
- **Python environments:** the generator runs on `backend/.venv/bin/python`; training and scoring on `validation/venv/bin/python`.
- **The history is a simulated logbook.** In data and in the live demo, an engine's history is computed from its own degradation state some hours earlier, plus measurement noise. It must never use information from the current flight.

## Review Focus

1. **Young engine, history taken under 50 h:** the fleet wear fraction is about 0 there, so a wear ratio would divide by about 0. Expect ratio = 1.0, the fleet twin. Test: `test_engine_history_young_engine_ratio_one` (Task 2).
2. **Engine younger than the history lag:** a live engine record at 30 h has no flight 100 h ago. Expect the lag to be clipped to the engine's hours and ratio = 1.0, never an error. Test: `test_engine_history_hours_below_lag` (Task 2).
3. **A fault already active at history time:** it inflates the wear estimate. Expect the ratio to stay clamped to [0.5, 2.0]. Test: `test_engine_history_fault_contamination_clamped` (Task 2).
4. **Old export on new code:** the live 914 v5 export (`twin: fleet_wear`, RUL without history inputs) must answer exactly as before. Test: `test_old_export_unchanged` (Task 5).
5. **Steady cruise or flat channels with the stuck rule:** a channel whose true value doesn't move (battery voltage at cruise, the 912's blank turbo channels) must never be called stuck. Test: `test_stuck_rule_ignores_steady_and_blank_channels` (Task 1).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `backend/aiv5.py` | serving | W3 flat-line stuck rule after the sensor head |
| `backend/degradation_v5.py` | wear model | `wear_fraction`, `health_from_wear`, `calibrated_wear_health`, `engine_history` |
| `backend/features_v5.py` | contract | `TWIN = "calibrated_wear"`, `HIST_COLS` |
| `backend/generate_dataset_v5.py` | data | twin runs calibrated wear; writes `HIST_COLS` per row |
| `validation_v5/pipeline_v5.py` | cache | `HIST_COLS` join `LABEL_COLS`, so they reach `END_COLS` |
| `validation_v5/rul_v5.py` | RUL | `rul_inputs(..., hist=None)` adds history inputs when given |
| `validation_v5/assembly_v5.py` | serving model | `predict_window(..., hist=None)` |
| `backend/twin_v5.py` | live twin | engine history from the engine record; `twin_mode` handles `calibrated_wear` |
| `backend/main.py` | live loop | sends `engine_history` in the AI payload |
| `validation_v5/run_fleet.sh` | pipeline | default order `914 912 915 916`; tighter gate bars |
| `frontend/app/engine_info/page.tsx` | UI | real per-engine v6 status |

---

## Track A: ship now, no retraining

### Task 1: W3, call a stuck sensor within about 20 s

**Files:**
- Modify: `backend/aiv5.py` (`run_inference`, sensor block around line 178)
- Test: `backend/tests/test_ai_v5_service.py`

**Interfaces:**
- Produces: `stuck_by_flatline(rows: np.ndarray, channels: list[str]) -> set[str]` in `aiv5.py`.
  - `rows` is `flight.rows` as an array `[n, 39]` of raw features, in `F.FEATURE_COLS` order.
  - It returns the channels whose measured value is exactly constant over the last `STUCK_FLAT_S = 20` rows **while** the twin's value (measured − residual) moves by more than `2 * SENSOR_SPEC[ch]["noise_sd"]` over those rows.
  - In `run_inference`, a channel in that set whose model condition is `none` becomes `{"condition": "stuck", "confidence": 0.99, "source": "flatline"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_stuck_rule_flags_flat_reading_against_moving_twin(self):
    rows = synthetic_rows(n=128)                  # helper in the test module: healthy rows, egt twin rising 5 sigma over the last 20 s
    rows[-20:, col("egt")] = rows[-21, col("egt")]
    self.assertEqual(aiv5.stuck_by_flatline(rows, ["egt"]), {"egt"})

def test_stuck_rule_ignores_steady_and_blank_channels(self):
    rows = synthetic_rows(n=128, steady=True)     # twin flat on every channel
    rows[-20:, col("battery_voltage")] = rows[-21, col("battery_voltage")]
    self.assertEqual(aiv5.stuck_by_flatline(rows, list(FAULTABLE_CHANNELS)), set())
```

- [ ] **Step 2:** Run `cd backend && .venv/bin/python -m unittest tests.test_ai_v5_service -k stuck`. Expected: FAIL, `stuck_by_flatline` is not defined.
- [ ] **Step 3:** Implement `stuck_by_flatline` and the override in `run_inference`. Measured is `rows[:, FEATURE_COLS.index(ch)]`; twin is measured − `rows[:, FEATURE_COLS.index(f"res_{ch}")]`.
- [ ] **Step 4:** Run the two tests plus the full v5 suite (`tests.test_ai_v5_service tests.test_main_v5 tests.test_twin_v5`). Expected: all PASS.
- [ ] **Step 5: Check end to end.**
  - Run `cd backend && ../validation/venv/bin/python3 ../scripts/e2e_v5.py --engines 914`. Expected: 8/8 PASS, and `sensor_egt_stuck` right kind ≥ 45/57 (was 32/57).
  - Rerun the 22-flight A/B (`scratchpad/ab_914.py`, Task 9). Expected: no healthy flight gains a sensor call.
- [ ] **Step 6:** Commit `feat(aiv5): flat-line rule calls a stuck sensor within 20 s`, then open a PR.

---

## Track B: v6 engine history (all four engines retrain once)

### Task 2: Engine history and the calibrated wear curve

**Files:**
- Modify: `backend/degradation_v5.py`
- Test: `backend/tests/test_fleet_wear_v5.py`, renamed to `test_engine_history_v6.py` (keep its existing tests)

**Interfaces:**
- Produces:
  - `wear_fraction(hours, tbo_hours, a, b, scale) -> float`. This is the `frac` that `baseline_health` computes today; `baseline_health` now calls it.
  - `health_from_wear(frac: float) -> Health`. `BASELINE_MODS_V5` applied at `frac`.
  - `calibrated_wear_health(hours, tbo_hours, ratio) -> Health`, equal to `health_from_wear(clip(wear_fraction(hours, tbo, *FLEET_WEAR) * ratio, 0, 1))`.
  - `engine_history(deg: DegradationStateV5, hours: float, rng: np.random.Generator, lag_h: float | None = None) -> dict` returning `{"lag_h", "wear_ratio", "sev_max"}`. Computed at `h0 = hours - lag`, where:
    - `lag` defaults to `rng.uniform(20, 200)` and is clipped to `hours`;
    - `wear_ratio = clip(observed / fleet, 0.5, 2.0)`;
    - `observed = wear_fraction(h0, deg's own a, b, scale) * (1 + rng.normal(0, 0.10)) + 0.5 * sev_max`. That is a logbook estimate: noisy, and inflated by any fault already active at `h0`;
    - `fleet = wear_fraction(h0, *FLEET_WEAR)`;
    - `wear_ratio = 1.0` when `fleet < 0.02` (young engine);
    - `sev_max = max severity at h0` (from `deg.health_at(h0)`).

  `# ponytail: 10% noise and 0.5 x severity contamination are modelling knobs; tune if live calibration data ever exists.`

- [ ] **Step 1: Write the failing tests**

```python
def test_calibrated_ratio_one_is_fleet_wear(self):
    self.assertEqual(calibrated_wear_health(900, 2000, 1.0).as_dict(), fleet_wear_health(900, 2000).as_dict())

def test_engine_history_young_engine_ratio_one(self):
    h = engine_history(deg_at(1.2), 40.0, np.random.default_rng(0))
    self.assertEqual(h["wear_ratio"], 1.0)

def test_engine_history_hours_below_lag(self):
    h = engine_history(deg_at(1.0), 30.0, np.random.default_rng(0), lag_h=100.0)
    self.assertEqual((h["lag_h"], h["wear_ratio"]), (30.0, 1.0))

def test_engine_history_fault_contamination_clamped(self):
    h = engine_history(deg_with_fault_before(800, sev=0.9), 1000.0, np.random.default_rng(0))
    self.assertLessEqual(h["wear_ratio"], 2.0)

def test_calibrated_twin_beats_fleet_twin_on_healthy_residual(self):   # reuses healthy_residuals()
    cal = np.quantile(healthy_residuals(30, lambda d, h: calibrated_wear_health(h, d.tbo, engine_history(d, h, np.random.default_rng(1))["wear_ratio"])), 0.95, axis=0)
    fleet = np.quantile(healthy_residuals(30, lambda d, h: fleet_wear_health(h, d.tbo)), 0.95, axis=0)
    for c, a, b in zip(CHANNELS, cal, fleet):
        self.assertLess(a, 0.7 * b, f"{c}: calibrated {a:.1f} vs fleet {b:.1f}")
```

- [ ] **Step 2:** Run `cd backend && .venv/bin/python -m unittest tests.test_engine_history_v6`. Expected: FAIL (imports).
- [ ] **Step 3:** Implement the four functions. `baseline_health` must keep its exact output, because `test_wear_only_is_the_shared_baseline` still passes.
- [ ] **Step 4:** Run that file plus `tests.test_twin_v5 tests.test_physics_v5`. Expected: all PASS.
- [ ] **Step 5:** Commit `feat(degradation_v5): engine history and calibrated wear curve`.

### Task 3: Generator writes calibrated-twin data and the history columns

**Files:**
- Modify: `backend/features_v5.py`, `backend/generate_dataset_v5.py`
- Test: `backend/tests/test_engine_history_v6.py`

**Interfaces:**
- Consumes: Task 2 functions.
- Produces:
  - `features_v5.TWIN = "calibrated_wear"`.
  - `features_v5.HIST_COLS = ["hist_lag_h", "hist_wear_ratio", "hist_sev_max"]`, appended to the generator's `LABEL_COLS`, so they become parquet columns, constant per flight.
  - The generator draws the history once per flight with its own RNG stream (`np.random.default_rng(seed + 2)` per scenario), so existing draws don't shift.
  - The twin runs `calibrated_wear_health(hours, tbo, hist["wear_ratio"])` for its warm start and for every step.

- [ ] **Step 1:** Write a failing test, `test_generator_rows_carry_history_and_calibrated_twin`. Generate 6 flights to a temp dir and assert:
  - `manifest["contract"]["twin"] == "calibrated_wear"`;
  - all of `HIST_COLS` are columns, each constant within a flight;
  - `0.5 <= hist_wear_ratio <= 2.0`.
- [ ] **Step 2:** Run the test. Expected: FAIL.
- [ ] **Step 3:** Implement. The twin call sites are the three lines changed in PR #9 (`fleet_wear_health(...)` becomes `calibrated_wear_health(..., hist["wear_ratio"])`).
- [ ] **Step 4:** Run the test. Expected: PASS. Then run a 120-flight sample and measure healthy residuals (same method as PR #9's check). Expected: rpm q99 ≤ 5σ (fleet twin: 7.8 on full data), oil pressure ≤ 8σ (was 12).
- [ ] **Step 5:** Commit `feat(generator): calibrated-wear twin and per-flight engine history`.

### Task 4: Cache and RUL learn from the history

**Files:**
- Modify: `validation_v5/pipeline_v5.py`, `validation_v5/rul_v5.py`, `validation_v5/assembly_v5.py`
- Test: `validation_v5/tests/test_pipeline_v5.py`, `validation_v5/tests/test_rul_smoothing_v5.py`

**Interfaces:**
- Consumes: `features_v5.HIST_COLS`.
- Produces:
  - `pipeline_v5.LABEL_COLS += F.HIST_COLS` when the data's manifest has them. The cache contract records `"hist_cols"` (an empty list for older data).
  - `rul_v5.rul_inputs(o, seq, ctx, contract, hist: np.ndarray | None = None)`. When `contract.get("hist_cols")` is set, it appends three columns: `hist_wear_ratio`, `hist_sev_max`, and `(max severity now − hist_sev_max) / max(hist_lag_h, 1)`, the progression rate. When it is unset, output is exactly as today.
  - `rul_v5.features` reads `hist` from `cache.labels(Er, c)` for `c` in `HIST_COLS`. The monotone hours column index is unchanged, because the new columns are appended at the end.
  - `Assembly.predict_window(seq, ctx, engine_hours, tbo=None, hist: dict | None = None)` passes `hist` through.

- [ ] **Step 1:** Write failing tests:
  - `test_rul_inputs_unchanged_without_hist`: an old contract gives the same array as today.
  - `test_rul_inputs_adds_three_hist_columns`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run those tests plus `test_specialists_v5.py` and `test_graph_parity_v5.py`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(rul): engine history inputs (wear ratio, past severity, progression)`.

### Task 5: Live twin and AI service use the history

**Files:**
- Modify: `backend/twin_v5.py`, `backend/main.py`, `backend/aiv5.py`
- Test: `backend/tests/test_engine_history_v6.py`, `backend/tests/test_main_v5.py`

**Interfaces:**
- Consumes: Task 2 (`engine_history`, `calibrated_wear_health`) and Task 4 (`predict_window(..., hist=)`).
- Produces:
  - `twin_mode()` returns `"new_engine" | "fleet_wear" | "calibrated_wear"`.
  - `UAVEngineTwinV5.history: dict` is set in `_load` from `engine_history(self.deg, start_engine_hours, rng(degradation_seed + 2))`. The engine record may carry `"history"`, which wins (a real logbook).
  - `twin_health(hours)` covers all three modes.
  - The twin's output gains `"engine_history": self.history`. `main.AI_FEATURE_COLS_V5` gains `"engine_history"`.
  - `aiv5.run_inference` passes `hist=last.get("engine_history")` to `predict_window` only when the export's contract has `hist_cols`.

- [ ] **Step 1:** Write failing tests:
  - `test_old_export_unchanged`: with today's `models_v5/914` (`fleet_wear`, no `hist_cols`), `run_inference` on a fixed window gives the same output as before this change.
  - `test_live_twin_calibrated_mode`: an export contract with `twin: calibrated_wear` makes the live twin's residual on a worn healthy engine smaller than in `fleet_wear` mode.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the v5 backend suite (45+ tests). Expected: PASS, and `twin_mode("Rotax_914_ULF") == "fleet_wear"` until the 914 is re-exported.
- [ ] **Step 5:** Commit `feat(live): engine history in the twin and the AI payload`.

### Task 6: Pipeline script for v6

**Files:**
- Modify: `validation_v5/run_fleet.sh`

**Interfaces:**
- Produces:
  - Default engine order `914 912 915 916`.
  - The data check accepts `twin = calibrated_wear`.
  - Gate bars: rpm ≤ 6σ, oil pressure ≤ 10σ, fuel flow ≤ 5σ (v6 target with margin).
  - The cache's `hist_cols` must be non-empty.

- [ ] **Step 1:** Edit the script, then run `bash -n`.
- [ ] **Step 2:** Run the gate's Python on the Task 3 sample cache. Expected: the residual bars pass.
- [ ] **Step 3:** Commit `chore(run_fleet): v6 order, gate and history checks`.

---

## Track C: one engine at a time - train, check against the previous models, then the next

Track B is done (branch `claude/v6-plan`, reviewed). The engines go strictly one after
another: **914 -> 912 -> 915 -> 916**. An engine starts only after the one before it has
been checked and merged (or kept back). Every engine gets the same check, against the
models it would replace:

| Engine | "Previous" it must beat |
|---|---|
| 914 | the live 914 v5 export (`backend/models_v5/914`, backup in `914b_specialists_prev`) |
| 912, 915, 916 | the 914 placeholder serving it at that moment (the 914 v6 models, once Task 8 merges) |

### Task 7: The check tool (before any training)

**Files:** Create `scripts/ab_v5.py`, from `scratchpad/ab_914.py` (the 22-flight A/B used for PR #10).

**Interfaces:** `scripts/ab_v5.py --engine <912|914|915|916> --old <checkout or models dir> --new <checkout or models dir>`.
It flies every applicable fault, healthy engines at 300 / 1,000 / 1,550 h and the sensor presets, once per side
with the same seed, and prints one table with: pass, fault named exactly, healthy false alarms, wrong-fault calls,
RUL error, and a PASS/FAIL line for the rule below. Sensor presets are scored from onset + 128 s (W3).

- [ ] Write it; run it with `--old` = `--new` = the live 914 and check both columns are identical (the tool's own test).
- [ ] Commit `feat(scripts): ab_v5.py, the old-vs-new check every engine goes through`.

**The rule an engine must pass to merge** (printed by the tool, decided by me, confirmed with the user):
1. Test card: 6/6 gates; no per-fault F1 down by more than 0.03 against the previous.
2. A/B: all flights pass; exact fault naming not lower; healthy false alarms not higher; mean RUL error not higher.
3. Export parity on GPU and CPU; `e2e_v5 --engines <key>` all pass.
If any part fails: the engine is NOT merged, the previous stays live, and we investigate before the next engine.

### Task 8: 914 on v6 (about 13 h)

- [ ] Merge `claude/v6-plan` (with the user's OK) and copy it into the main checkout.
- [ ] The user starts: `cd ~/Documents/UAV_Engine && nohup caffeinate -is bash validation_v5/run_fleet.sh 914 > validation_v5/logs/fleet_914.log 2>&1 &`
- [ ] Card against the live 914 v5. Targets on top of the rule: valve-leakage recall >= 0.65 (was 0.56); RUL bias <= +150 h when
      true life is 600+ h under the calendar (was +300); >= -40 h when 1-50 h under (was -72); wear-limited error <= 9.5% of TBO.
- [ ] `ab_v5.py --engine 914` against the live export; export to a scratch folder first, parity GPU + CPU, `e2e_v5 --engines 914`.
- [ ] Rule passed: export for real, commit the models, PR, merge with the user's OK. Failed: keep v5 live and stop to investigate.

### Task 9: 912 on v6 (about 13 h) - only after Task 8 is merged or kept back

- [ ] The user starts: `bash validation_v5/run_fleet.sh 912` (same command shape as Task 8).
- [ ] Card against the 914 placeholder; the 912 has no turbo (its turbo channels are blank) and is the most likely to
      miss the valve-leakage gate (W1) - the calibrated twin is meant to fix exactly that.
- [ ] `ab_v5.py --engine 912 --old <placeholder> --new <912 v6>`, parity, `e2e_v5 --engines 912`; merge only if the rule passes.

### Task 10: 915 on v6 - after Task 9

- [ ] Same as Task 9 with `915`. Note: TBO 1,200 h, so RUL errors read against a shorter life.

### Task 11: 916 on v6 - after Task 10

- [ ] Same as Task 9 with `916`. When it merges, no engine is on a placeholder any more.

### Task 12: Default switch, frontend, docs, tests - after Task 11

- [ ] `e2e_v4` and `e2e_v5` on the same flights, all four engines; if v5 wins on every engine, v5 becomes the default
      in `main.py` and `start_stack.sh` (v4 stays selectable).
- [ ] `engine_info` shows the real per-engine v6 status; README, plan.md and a four-engine model-card summary.
- [ ] Fix the three order-dependent backend tests and the five deferred review minors.

---

## Order and cost

| Step | What | GPU | Who |
|---|---|---|---|
| Track A (any time) | Task 1: stuck sensor in ~20 s | none | me |
| Track B | Tasks 2-6 | none | done, on `claude/v6-plan` |
| C1 | Task 7: the check tool | none | me, ~1 h |
| C2 | Task 8: 914, then check | ~13 h | the user starts, I check |
| C3 | Task 9: 912, then check | ~13 h | the user starts, I check |
| C4 | Task 10: 915, then check | ~13 h | the user starts, I check |
| C5 | Task 11: 916, then check | ~13 h | the user starts, I check |
| C6 | Task 12: default switch, UI, docs, tests | none | me, ~3 h |

Each engine waits for the previous one's check; a failed check stops the line until it is understood.
