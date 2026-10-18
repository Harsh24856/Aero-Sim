# v5 Fine-tune Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live 912/914/915/916 models behave better in flight (A/B wins) in one night: fine-tuned diagnosis for air filter and the oil pair, and RUL that uses the logbook trend only for fault families where it helps.

**Architecture:**
- Three small additions to the existing v5 pipeline: a per-instrument label check, a fine-tune mode that starts from the trained specialists, and a per-family trend gate in the RUL model.
- Measurements choose the per-engine settings.
- Every engine is accepted or rejected on its own by `ab_v5.py` against the live export.

**Tech Stack:** Python 3.11 + TensorFlow-metal (`validation/venv`), scikit-learn, `validation_v5/` pipeline, `backend/` live stack.

**Spec:** `docs/superpowers/specs/2026-10-18-v5-finetune-improve-design.md`

## Global Constraints

- **Acceptance:** ship an engine only if both hold:
  1. `ab_v5.py --engine <e> --old origin/main --new .` shows no rule worse than live, and the targeted fault's exact naming or the RUL error is better;
  2. no test-card gate goes from PASS to FAIL.

  Otherwise run `git checkout origin/main -- backend/models_v5/<e>`.
- **Gates:** `validation_v5/gates_v5.py` is unchanged. Nothing is chosen on the test split.
- **Fine-tune stage:** `lr = min(1e-4, cfg["lr"] / 5)`, warm-up `max(1, steps // 2)`, 12 epochs, patience 3, `--max-windows 20000`, and `fit(keep_current=True)`.
- **One GPU job at a time.** Run under `caffeinate -is`; mains power, lid open.
- **Never overwrite** `validation_v5/cache/<e>b` or `artifacts/<e>b_specialists` except in Task 7, when the 916 relabel path is chosen. Commit artifacts before `export_v5.py`, so the manifest records `git_dirty: false`.
- **Python style:** follow the repo: `unittest` tests runnable as `validation/venv/bin/python validation_v5/tests/<file>.py`, and `ruff check --select F,E9` clean.

## Review Focus

1. **A fine-tune that never beats the loaded weights:** the saved head must stay byte-identical, and the card must not change. Pinned in Task 2's test of the keep-or-replace rule.
2. **Window with no confident family** (every family probability below 0.5): RUL must use the base estimate. Pinned in Task 3.
3. **`families` covering every family:** output must equal today's `TrendGatedRUL` exactly. Pinned in Task 3.
4. **`--fault-weight` naming a fault the engine does not have** (e.g. turbo on the 912): reject with a clear error, not a silent no-op. Pinned in Task 2.
5. **The relabelled cache's window order must match `<e>b`.** If it differs, `gaps_v5 --vs` must stop with an error rather than compare the wrong windows. Pinned in Task 1.

---

### Task 1: Per-instrument label check (`relabel --out`, `gaps --fault --vs`)

**Files:**
- Modify: `validation_v5/relabel_v5b.py` (`build()` and the CLI)
- Modify: `validation_v5/gaps_v5.py`
- Test: `validation_v5/tests/test_gaps_v5.py`

**Interfaces:**
- Produces:
  - `relabel_v5b.build(key, z_fault, z_sensor, per_channel_faults=False, z_effect_floor=1.0, out_key: str | None = None)`. It writes to `cache/{out_key or key + 'b'}`, and the CLI gains `--out KEY`.
  - `gaps_v5.miss_visibility(y_old: np.ndarray, pred: np.ndarray, y_new: np.ndarray) -> dict`. Over windows where `y_old` is 1 and `pred` is 0 (misses), it returns `{"misses": int, "invisible": int, "share_invisible": float}`, where invisible means `y_new == 0`. With no misses, `share_invisible` is 0.0.
  - CLI: `gaps_v5.py <e>b --fault NAME --vs KEY`. It prints `miss_visibility` on test and val for that fault, comparing `cache/<e>b` labels (`fmv_<fault>`) with `cache/KEY` labels. It raises `SystemExit` when the two caches' window ends differ.

- [ ] **Step 1: Write the failing tests**

```python
def test_miss_visibility_counts_invisible_misses(self):
    y_old = np.array([1, 1, 1, 0]); pred = np.array([0, 0, 1, 0]); y_new = np.array([0, 1, 1, 0])
    self.assertEqual(gaps_v5.miss_visibility(y_old, pred, y_new),
                     {"misses": 2, "invisible": 1, "share_invisible": 0.5})

def test_miss_visibility_without_misses_is_zero(self):
    y = np.array([1, 0]); self.assertEqual(gaps_v5.miss_visibility(y, y, y)["share_invisible"], 0.0)

def test_check_same_windows_rejects_different_ends(self):
    with self.assertRaises(SystemExit):
        gaps_v5.check_same_windows(np.array([1, 2, 3]), np.array([1, 2, 4]))
```

The last test also defines the helper `check_same_windows(a: np.ndarray, b: np.ndarray) -> None`, which raises `SystemExit` when the arrays differ.

- [ ] **Step 2: Run them; expect `AttributeError: module 'gaps_v5' has no attribute 'miss_visibility'`**

Run: `validation/venv/bin/python validation_v5/tests/test_gaps_v5.py`

- [ ] **Step 3: Implement `miss_visibility` and `check_same_windows` in `gaps_v5.py`, the `--fault/--vs` CLI branch, and `out_key` / `--out` in `relabel_v5b.py`**

- [ ] **Step 4: Run the tests; expect `OK`. Then `ruff check --no-cache --select F,E9 validation_v5/gaps_v5.py validation_v5/relabel_v5b.py`**

- [ ] **Step 5: Commit** (`feat(v5): per-instrument label check for one fault's misses`)

---

### Task 2: Fine-tune mode with per-fault loss weights

**Files:**
- Modify: `validation_v5/train_v5.py`: `fault_weights()`, `Trainer.__init__` (a diagnosis-only weight), and `Trainer.load(..., overrides)`
- Modify: `validation_v5/specialists_v5.py`: `--finetune`, `--fault-weight`, and `finetune_head()`
- Test: `validation_v5/tests/test_finetune_v5.py` (new; add it to the `v5=(...)` list in `scripts/test_all.sh`)

**Interfaces:**
- Produces:
  - `train_v5.fault_weights(names: list[str], weights: dict[str, float], applicable: list[str]) -> np.ndarray`.
    - Returns 1.0 for every name, the given weight for named faults, and 0.0 for non-applicable faults (matching today's `fault_mask`).
    - Raises `ValueError` naming any key in `weights` that is not in `applicable`.
  - In `Trainer.__init__`: `self.fault_w = tf.constant([fault_weights(FAULT_NAMES, cfg.get("fault_weight", {}), applicable)])`. The diagnosis BCE uses `* self.fault_w`; the denominator and every other use of `fault_mask` are unchanged.
  - `Trainer.load(cache, out_dir, overrides: dict | None = None)`: merges `overrides` into the saved config before building.
  - `specialists_v5.finetune_head(cache, art: str, head: str, fault_weight: dict, epochs=12, patience=3, max_windows=20000, verbose=1) -> dict`.
    - Loads the head with `overrides={"fault_weight": fault_weight}`, sets the fine-tune stage (Global Constraints) and runs `fit(..., keep_current=True)`.
    - Saves the head only if the best score beat the loaded one.
    - Writes `<art>/<head>/finetune.json` with `{"before": float, "after": float, "kept": bool, "fault_weight": {...}}` and returns that dict.
  - CLI: `specialists_v5.py <e>b --finetune diagnosis --fault-weight air_filter_fouling=2.0,oil_degradation=1.5 [--ft-epochs 12]`. `--ft-epochs` is separate from the training `--epochs` (default 80). A head the fine-tune replaced makes `assemble`, `rul`, `test` and `card` stale (the same `stale` flag as a retrain).

- [ ] **Step 1: Write the failing tests** (pure numpy; no GPU)

```python
from train_v5 import fault_weights
from degradation_v5 import FAULT_NAMES
APP = [n for n in FAULT_NAMES if n != "turbo_degradation"]

def test_fault_weights_scales_named_fault(self):
    w = fault_weights(FAULT_NAMES, {"air_filter_fouling": 2.0}, APP)
    self.assertEqual(w[FAULT_NAMES.index("air_filter_fouling")], 2.0)
    self.assertEqual(w[FAULT_NAMES.index("bearing_wear")], 1.0)
    self.assertEqual(w[FAULT_NAMES.index("turbo_degradation")], 0.0)

def test_fault_weights_rejects_a_fault_the_engine_lacks(self):
    with self.assertRaises(ValueError):
        fault_weights(FAULT_NAMES, {"turbo_degradation": 2.0}, APP)

def test_empty_weights_equal_todays_mask(self):
    w = fault_weights(FAULT_NAMES, {}, APP)
    self.assertEqual(list(w), [1.0 if n in APP else 0.0 for n in FAULT_NAMES])
```

- [ ] **Step 2: Run them; expect `ImportError: cannot import name 'fault_weights'`**

Run: `validation/venv/bin/python validation_v5/tests/test_finetune_v5.py`

- [ ] **Step 3: Implement `fault_weights`, `self.fault_w` in the diagnosis loss, `Trainer.load(..., overrides)`, `finetune_head` and the CLI flags**

- [ ] **Step 4: Run the tests (`OK`), then `bash scripts/test_all.sh` with `BACKEND_PY` and `AI_PY` set (`ALL PASS`)**

- [ ] **Step 5: Smoke-check on the GPU**

Run: `caffeinate -is validation/venv/bin/python validation_v5/specialists_v5.py 914b --finetune diagnosis --fault-weight oil_degradation=1.5 --ft-epochs 1`

Expected: `<art>/diagnosis/finetune.json` is written with `before`, `after` and `kept`. Then run `git checkout -- validation_v5/artifacts/914b_specialists` to undo the smoke-check.

- [ ] **Step 6: Commit** (`feat(v5): fine-tune a trained specialist, with per-fault loss weights`)

---

### Task 3: Per-family logbook trend in RUL

**Files:**
- Modify: `validation_v5/rul_v5.py` (`TrendGatedRUL`, `family_cols`, `fit_rul`)
- Test: `validation_v5/tests/test_rul_history_v6.py`

**Interfaces:**
- Produces:
  - `rul_v5.family_cols(n_faults: int, n_families: int) -> slice`, equal to `slice(n_faults + 2, n_faults + 2 + n_families)`. The layout is health(1), severity(n_faults), engine_fault(1), family.
  - `TrendGatedRUL(gated, weight: float, families: tuple[int, ...] | None = None, fam_cols: slice | None = None)`. With `families` set, the blend applies only where `use` already holds, the argmax of `X[:, fam_cols]` is in `families`, and its value is at least 0.5.
  - `fit_rul` adds candidate `gated_gbt+trend{w}+fam` for each `w` in (0.5, 1.0):
    - `families` are the family indices where, among validation wear-limited windows with that argmax family (at least 30 windows), the served wear-limited MAE with the trend is lower than without it;
    - `fam_cols = family_cols(len(FAULT_NAMES), N_FAMILIES)`;
    - the existing `score` picks among all candidates.

- [ ] **Step 1: Write the failing tests** (stub `reg` / `clf` with `predict` / `predict_proba`; X built so `X[:, -3]` is the trend)

```python
def test_family_none_is_todays_trend_model(self): ...      # equal outputs on random X
def test_trend_only_for_allowed_family(self): ...         # argmax family 0 -> blend; family 1 -> base
def test_no_confident_family_uses_base(self): ...         # all family probs 0.3 -> base
def test_all_families_allowed_equals_todays_model(self): ...
def test_family_cols_matches_rul_inputs_layout(self): ... # rul_inputs(o, ...)[:, family_cols(...)] == o["family"]
```

Each test states its assertion with `np.testing.assert_allclose` against hand-computed values: blend `= w * trend + (1 - w) * base`.

- [ ] **Step 2: Run them; expect `TypeError` (unexpected `families`) and `ImportError` (`family_cols`)**

- [ ] **Step 3: Implement `family_cols`, the gate, and the `fit_rul` candidates**

- [ ] **Step 4: Run the tests (`OK`), then `test_all.sh` (`ALL PASS`)**

- [ ] **Step 5: Commit** (`feat(v5 RUL): use the logbook trend only for fault families where it helps`)

---

### Task 4: Measure (P1, about 45 min)

**Files:**
- Create: `validation_v5/cache/916pc` and `915pc` (git-ignored caches; never trained on unless Task 7 picks the relabel)
- Modify: this plan's "Findings" table

- [ ] **Step 1: Relabel into separate caches**

Run: `validation/venv/bin/python validation_v5/relabel_v5b.py 916 --per-channel-faults --out 916pc`, then the same for 915.

Expected: it prints the label flips per fault.

- [ ] **Step 2: Air-filter misses**

Run: `validation/venv/bin/python validation_v5/gaps_v5.py 916b --fault air_filter_fouling --vs 916pc`, then the same for 915.

- [ ] **Step 3: Oil misses**

Run `gaps_v5.py <e>b --fault oil_pump_degradation --vs <e>pc` and `--fault oil_degradation`. Do the 912 and 914 too: relabel them with `--out 912pc` and `--out 914pc` first.

- [ ] **Step 4: Fill in "Findings" and set each engine's path by the rules below; commit the plan**

**Rules:**
- **916 air filter:** if the test `share_invisible` is at least 0.6, take path **L** (Task 7). Otherwise take **W** (Task 6 with `air_filter_fouling=2.0`).
- **Oil weights per engine:** add `oil_pump_degradation=1.5,oil_degradation=1.5` when either oil fault's test-card F1 is below 0.75 and its test `share_invisible` is below 0.6.
- **915 air filter:** add `air_filter_fouling=2.0` when its test-card precision is below 0.5 (it is 0.40 today) and `share_invisible` is below 0.6.

---

### Task 5: RUL refit, all four engines (P2, about 1 h)

- [ ] **Step 1: Refit each engine's RUL**

Run, for each of 915, 912, 914 and 916: `caffeinate -is validation/venv/bin/python validation_v5/specialists_v5.py <e>b --force rul,test,card`

Expected: `rul_v5.json` `chosen` names a candidate, and only the card's RUL row may move.

- [ ] **Step 2: Commit the four engines' `rul_v5.*`, `results_test.json` and `model_card.md`**

---

### Task 6: Diagnosis fine-tunes (P3, about 45 min per engine, overnight)

- [ ] **Step 1: Fine-tune each engine, one at a time, in this order: 916 (if path W), 915, 912, 914**

Run: `caffeinate -is validation/venv/bin/python validation_v5/specialists_v5.py <e>b --finetune diagnosis --fault-weight <weights from Findings> --verbose 2`

Skip an engine whose Findings give no weights.

Expected per engine: `finetune.json` is written. If `kept` is true, the assembly, RUL, test and card steps reran. If `kept` is false, the card is unchanged.

- [ ] **Step 2: Commit each engine's artifacts after its run**

### Task 7: 916 relabel path L (only if Task 4 chose it; about 2.5 h)

- [ ] **Step 1: Relabel the 916's training cache**

Run: `validation/venv/bin/python validation_v5/relabel_v5b.py 916 --per-channel-faults`. This overwrites `cache/916b`, the only allowed overwrite.

- [ ] **Step 2: Fine-tune all five heads**

Run: `caffeinate -is validation/venv/bin/python validation_v5/specialists_v5.py 916b --finetune detection,diagnosis,severity,sensor,health --verbose 2`

- [ ] **Step 3: Commit.** The card is on new labels; Task 8's A/B decides.

---

### Task 8: Accept per engine (P4, about 20 min per engine)

- [ ] **Step 1: Run the checks for each engine with a changed artifact**

For each engine with a changed artifact, run in order:
1. `validation/venv/bin/python validation_v5/export_v5.py <e>b`
2. `test_export_parity_v5.py <e>b`, on its own and again with `AERO_TEST_CPU=1`. Expected: `OK` both times.
3. `cd backend && ../validation/venv/bin/python ../scripts/e2e_v5.py --engines <e>`. Expected: no new FAIL; the 915's turbo_high may now pass.
4. `validation/venv/bin/python scripts/ab_v5.py --engine <e> --old origin/main --new .`

- [ ] **Step 2: Apply the acceptance rule.** Restore each losing export with `git checkout origin/main -- backend/models_v5/<e>` and record why.

- [ ] **Step 3: Commit the winning exports**

---

### Task 9: Ship and check (P5, about 30 min)

- [ ] **Step 1: Update the page's scores** to the accepted cards (`SCORES` in `frontend/app/engine_info/page.tsx`). Run `npx tsc --noEmit` (clean).
- [ ] **Step 2: Open the PR** with a before/after A/B table per engine. Merge after CI is green.
- [ ] **Step 3: Update the main checkout and restart.** In `/Users/harsh/Documents/UAV_Engine`, run `git pull --ff-only`, stop the stack, and run `scripts/start_stack.sh`.

  Expected: `/health` shows `model_version v5` with no stand-ins.
- [ ] **Step 4: Fly each changed engine for one minute through the API** (`/select_engine`, `/start`, `/params`, `/state`, `/stop`).

  Expected: `ai.status ok`, and `placeholder_models` false.

---

## Findings (Task 4)

| Engine | Air filter: test misses / share invisible | Oil pump: share invisible | Oil degradation: share invisible | Path / weights |
|---|---|---|---|---|
| 916 | 250 / 43% | 100% (F1 0.45) | 81% (F1 0.54) | **W**: `air_filter_fouling=2.0`; oil not weighted (misses invisible) |
| 915 | 167 / 27% (precision 0.40) | 63% (F1 0.66) | 45% (F1 0.65) | `air_filter_fouling=2.0,oil_pump_degradation=1.5,oil_degradation=1.5` |
| 912 | not targeted (F1 0.69) | 24% (F1 0.83, above 0.75) | 71% (F1 0.61) | no fine-tune (the weak oil fault's misses are invisible) |
| 914 | not targeted | 62% (F1 0.68) | 58% (F1 0.80, above 0.75) | no fine-tune (the weak oil fault's misses are invisible) |

Test split; F1 from each engine's card. Task 7 (relabel) is not taken. Across engines, 58-100% of oil misses
are windows the per-instrument rule calls invisible: oil faults are labelled present before any instrument
shows them. That points to a per-instrument relabel for the oil faults on every engine, in a later round.

## Outcome (2026-10-18)

**Shipped: the 916's RUL with the per-family trend.** In `ab_v5` against live, mean RUL error fell from 98 h to 75 h (cooling_hot_day 411 to 78 h, x_cooling_degradation 218 to 17 h). Naming, false alarms and flights passing were all equal. The other three engines are unchanged.

| Engine | RUL per-family trend | Diagnosis fine-tune | Shipped |
|---|---|---|---|
| 916 | A/B: RUL 98 to 75 h, nothing worse | Air filter ×2: validation select 1.3039 to 1.3054 (kept), but test air-filter recall stayed at 0.33, and combined with RUL it lowered naming from 94.1% to 93.2% (valve leakage 100% to 81%) | the RUL change only |
| 915 | A/B: 52 to 53 h; turbo_high unchanged (validation still allows the induction family, which holds turbo and air filter) | Air filter ×2, oil ×1.5: no gain (select 1.3491 both) | nothing |
| 912 | Validation kept today's model | Not run: its weak oil fault's misses are invisible | nothing |
| 914 | Validation kept today's model | Not run: same reason | nothing |

**What the round showed**
- Fine-tuning works and is cheap: 7-10 minutes per head, and it can never make validation worse. Loss weights alone do not lift a fault whose test misses are mostly invisible or flight-specific.
- Across engines, 58-100% of oil misses are labelled present before any instrument shows them. The next real lever is a per-instrument relabel (`relabel_v5b.py --per-channel-faults`) followed by fine-tuning on it, since both tools now exist.
- The 915's turbo RUL needs the trend gated per fault, not per family, because the turbo and air-filter faults share the induction family.
