# v5 Model Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the known gaps in the live v5 models without regressing them:
- 916 air-filter recall, 0.33 on test against a 0.5 floor;
- the oil-pump vs oil-degradation naming mix-up on the 912, 915 and 916;
- severity bias on the 912, 915 and 916, 0.034-0.057 against a 0.03 bar;
- the 915's turbo_high RUL miss.

Then refresh the stale docs and branches.

**Architecture:** Measure first, then fix in the cheapest place that works.
- Diagnosis cut-offs and the severity gate are post-hoc steps. `specialists_v5.py <e>b --force assemble,rul,test,card` redoes them in minutes from the trained networks.
- A network retrain (`--force diagnosis`, about 2.5-3 h per engine on the GPU) happens only where the measurements show the network itself cannot separate the faults.
- An engine's new export replaces the live one only if `scripts/ab_v5.py` against the current export passes.

**Tech Stack:** Python 3.11 + TensorFlow-metal (`validation/venv`); numpy, scikit-learn; the existing v5 pipeline in `validation_v5/` and `backend/`.

**Spec:** the acceptance record and known gaps on Harsh24856/Aero-Sim#16 (comments of 2026-10-16 and 2026-10-17) and each engine's `validation_v5/artifacts/<e>b_specialists/model_card.md`.

## Global Constraints

- **Gates:** do not edit `validation_v5/gates_v5.py`. The bars stay:
  - `min_fault_recall >= 0.50`;
  - `abs:bias_on_fault <= 0.03`;
  - `mae_on_fault <= 0.12`;
  - `mae_clean <= 0.02`.
- **No tuning on test:** the test split is never used to choose anything. Cut-offs, gates and calibrations are fitted on `cal` / `val` only.
- **Device:** training runs on the Metal GPU only (`train_v5.require_gpu()`). Inference parity on GPU and CPU must still pass.
- **Acceptance rule per engine:** the new export replaces the live one only if both hold:
  1. `ab_v5.py --engine <e> --old origin/main --new .` prints `A/B RULE: PASS`;
  2. the test card has no gate that passed before and fails now.

  If the A/B fails, keep the live export and record why on the PR.
- **One engine at a time** through any GPU step: the Mac has 8 GB RAM. Run long jobs under `caffeinate -is` with the lid open.
- **Model exports:** commit the artifacts before `export_v5.py`, so the manifest records `git_dirty: false`.
- **Main checkout:** after merging, update it (`git switch main && git pull`) and restart `scripts/start_stack.sh`, because the live stack runs from there.

## Review Focus

1. **A fault with no val positives:** `choose_cutoff` is skipped and the cut-off stays 0.5. The gap report must show such faults as `n/a`, not as recall 0.
2. **The severity gate trades bias for clean error:** a fix that meets the bias bar by raising `mae_clean` above 0.02 is a regression. Task 2's test pins both.
3. **Lowering a cut-off to raise air-filter recall floods healthy flights with alarms:** the A/B's healthy-false-alarm rule and the e2e healthy preset must stay at 0%.
4. **Oil fix that only swaps the error:** improving oil-pump naming must not drop oil-degradation recall below 0.5. Task 3's acceptance checks both faults.
5. **914 must not change:** it passes 6/6 and none of these fixes target it. Run every rebuild for 912b, 915b and 916b only.

---

### Task 1: Gap report (measure before fixing)

**Files:**
- Create: `validation_v5/gaps_v5.py`
- Test: `validation_v5/tests/test_gaps_v5.py`
- Modify: `scripts/test_all.sh`: add `test_gaps_v5` to the `v5=(...)` list.

**Interfaces:**
- Produces:
  - `bias_split(true: np.ndarray, pred_ungated: np.ndarray, pred_gated: np.ndarray) -> dict`. Over fault cells (`true >= evaluate.PRESENT_SEV`) it returns `{"bias": float, "n": int, "gated_out_n": int, "gated_out_bias": float, "kept_bias": float, "share_from_gated_out": float}`.
    - `gated_out` cells are those where `pred_gated == 0` and `pred_ungated > 0`.
    - `share_from_gated_out` is `gated_out_n * gated_out_bias / (n * bias)`, or 0.0 when `bias == 0`.
  - `confusion_share(y: np.ndarray, pred: np.ndarray, a: int, b: int) -> float`: of windows where fault `a` is true, the share predicted `b` and not `a`.
  - A CLI, `gaps_v5.py <e>b`. It writes `validation_v5/artifacts/<e>b_specialists/gaps.json` and prints:
    - per applicable fault: val positives, val recall and precision at the stored cut-off, and test recall at that cut-off (test is read only for the report);
    - the severity `bias_split` on val and on test;
    - `confusion_share` both ways for `oil_pump_degradation` and `oil_degradation`.

- [ ] **Step 1: Write the failing tests**

```python
def test_bias_split_attributes_bias_to_gated_out_cells():
    t = np.array([[0.4], [0.4], [0.0]])
    ung = np.array([[0.38], [0.30], [0.05]])
    gat = np.array([[0.38], [0.00], [0.00]])
    r = gaps_v5.bias_split(t, ung, gat)
    assert r["n"] == 2 and r["gated_out_n"] == 1
    assert r["bias"] == pytest.approx((-0.02 - 0.4) / 2)
    assert r["gated_out_bias"] == pytest.approx(-0.4)
    assert r["share_from_gated_out"] == pytest.approx(0.4 / 0.42, rel=1e-6)

def test_bias_split_zero_bias_has_zero_share():
    t = np.array([[0.4]]); p = np.array([[0.4]])
    assert gaps_v5.bias_split(t, p, p)["share_from_gated_out"] == 0.0

def test_confusion_share_counts_only_misnamed_windows():
    y = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])
    pred = np.array([[0, 1], [1, 0], [0, 1], [0, 1]])
    assert gaps_v5.confusion_share(y, pred, 0, 1) == pytest.approx(2 / 3)
```

Write the tests with `unittest` (the repo's style: `python validation_v5/tests/test_gaps_v5.py`), using `assertAlmostEqual`.

- [ ] **Step 2: Run them; expect `ModuleNotFoundError: gaps_v5`**

Run: `validation/venv/bin/python validation_v5/tests/test_gaps_v5.py`

- [ ] **Step 3: Implement `bias_split`, `confusion_share` and the CLI in `validation_v5/gaps_v5.py`**

The CLI loads `Specialists(Cache(f"cache/{e}b"), art)` from `specialists_v5`. It uses:
- `sp.predict(cache.end_ids([split], jitter=False))` for assembled outputs;
- `sp.raw(seq, ctx)` with `calibrate_severity` and `extra["sev_iso"]` for severity before the gate.

Keep the import-time TensorFlow load out of the two pure functions, so the tests run on CI's CPU runner.

- [ ] **Step 4: Run the tests; expect `OK`**

- [ ] **Step 5: Run the report on 912b, 915b and 916b (the GPU is free; about 5 min each)**

Run: `for e in 912b 915b 916b; do caffeinate -is validation/venv/bin/python validation_v5/gaps_v5.py $e; done`

Expected: three `gaps.json` files. Record in this plan, under "Findings", the numbers Tasks 2-3 decide on:
- the severity `share_from_gated_out` on val;
- air filter's val positives and its val and test recall;
- the oil `confusion_share` both ways.

- [ ] **Step 6: Commit**

```bash
git add validation_v5/gaps_v5.py validation_v5/tests/test_gaps_v5.py scripts/test_all.sh validation_v5/artifacts/9*b_specialists/gaps.json docs/superpowers/plans/2026-10-17-v5-model-gaps.md
git commit -m "feat(v5): gap report - cut-off recall val vs test, severity bias by gate, oil confusion"
```

---

### Task 2: Severity bias, with the gate choosing on bias too

**Decision from Task 1:**
- If `share_from_gated_out >= 0.7` on val for an engine, the gate causes the bias. Do this task.
- Otherwise, skip to the fallback at the end of this task.

The gate today picks the factor that minimises `on_fault_mae + clean_mae` (`specialists_v5.choose_severity_gate`). Zeroing a missed fault lowers `clean_mae` and pushes the bias negative.

**Files:**
- Modify: `validation_v5/specialists_v5.py`, `choose_severity_gate`.
- Test: `validation_v5/tests/test_gaps_v5.py`, or a new `test_severity_gate_v5.py` if it needs the module's TensorFlow imports.

**Interfaces:**
- Produces: `gate_cost(report: dict, bias_bar: float = 0.03) -> float`, a module-level pure function. It returns `on_fault_mae + clean_mae + max(0.0, abs(on_fault_bias) - bias_bar)`. `choose_severity_gate` uses it in place of its inner sum on both the cal choice and the val keep check. The factor grid stays (0, 0.25, 0.5, 0.75, 1.0).

- [ ] **Step 1: Write the failing tests**

```python
def test_gate_cost_penalises_bias_beyond_the_bar():
    ok = {"on_fault_mae": 0.08, "clean_mae": 0.01, "on_fault_bias": -0.02}
    bad = {"on_fault_mae": 0.08, "clean_mae": 0.005, "on_fault_bias": -0.06}
    assert gate_cost(ok) < gate_cost(bad)

def test_gate_cost_is_the_old_sum_inside_the_bar():
    r = {"on_fault_mae": 0.08, "clean_mae": 0.01, "on_fault_bias": 0.029}
    assert gate_cost(r) == pytest.approx(0.09)
```

- [ ] **Step 2: Run them; expect `ImportError: cannot import name 'gate_cost'`**

- [ ] **Step 3: Implement `gate_cost` and use it in `choose_severity_gate`**

- [ ] **Step 4: Run the tests; expect `OK`**

- [ ] **Step 5: Rebuild the post-hoc steps for each engine Task 1 flagged; no network retraining**

Run: `caffeinate -is validation/venv/bin/python validation_v5/specialists_v5.py 912b --force assemble,rul,test,card`. Then run the same for 915b and 916b.

Expected for each card:
- `Severity` bias within 0.03;
- `mae_clean <= 0.02`;
- no other gate changed from PASS to FAIL.

- [ ] **Step 6: Commit the code and the rebuilt artifacts (`git add validation_v5/specialists_v5.py validation_v5/tests validation_v5/artifacts/9*b_specialists`)**

**Fallback** (`share_from_gated_out < 0.7`): the bias sits in severities the gate keeps.
- Add a per-fault additive offset after `sev_iso`, fitted on the even cal flights as the mean residual over fault cells. Store it as `extra["sev_offset"]` and apply it in `Assembly.__call__` after `calibrate_severity`, clipped to [0, 1].
- Tests: an offset of `+0.05` on a constant `-0.05` residual brings the bias to 0, and `export_v5` and parity still pass.
- Same rebuild and expectations as Step 5.

---

### Task 3: Air-filter recall and oil naming

**Decision from Task 1, per engine:**
- **3A, post-hoc:** use when a fault's val positives are under 300 windows and val recall at the stored cut-off is at least 0.55 but test recall is below 0.5. The validation grid is too thin. Choose cut-offs on `end_ids(["val"], jitter=True)` pooled over 4 seeds (4× the windows) instead of the single fixed grid.
- **3B, retrain diagnosis only:** use when val recall itself cannot reach 0.55 at any cut-off, or when the oil `confusion_share` is above 0.3 either way. The network cannot separate the fault.

**Files:**
- 3A: modify `validation_v5/train_v5.py`, `Trainer.calibrate`, the cut-off half only.
- 3B: `validation_v5/hp_search_v5.json` stays untouched. Add a `fault_weight` option to the training config, read in `Trainer.__init__` and multiplied into `self.fault_mask`, the per-fault weight `Trainer.losses` already applies to the diagnosis BCE.
- Test: `validation_v5/tests/test_cutoffs_v6.py`, extended.

**Interfaces:**
- 3A produces: `Trainer.calibrate(cut_seeds: int = 4)`. Cut-offs are chosen on the union of `end_ids(["val"], seed=s, jitter=True)` for `s` in `range(cut_seeds)`, de-duplicated. Temperatures stay on `cal`, unchanged.
- 3B produces: `cfg["fault_weight"]: dict[str, float]` (fault name to loss weight, default 1.0). For the 912, 915 and 916: `{"air_filter_fouling": 2.0, "oil_pump_degradation": 1.5, "oil_degradation": 1.5}`.

- [ ] **Step 1: Write the failing test**

For 3A, name it `test_calibrate_pools_jittered_val_windows`. It mocks `cache.end_ids` and asserts that `calibrate` asked for `jitter=True` with seeds 0-3.

For 3B, name it `test_fault_weight_scales_that_faults_loss`: a Trainer built with `fault_weight={"air_filter_fouling": 2.0}` has a per-fault weight vector with 2.0 at that fault's index and 1.0 elsewhere.

- [ ] **Step 2: Run it; expect FAIL**

- [ ] **Step 3: Implement the chosen variant**

- [ ] **Step 4: Run all v5 tests; expect `ALL PASS` (`bash scripts/test_all.sh` with `BACKEND_PY` and `AI_PY` set)**

- [ ] **Step 5: Rebuild, one engine at a time**

- 3A: `specialists_v5.py <e>b --force assemble,rul,test,card`, minutes.
- 3B: `specialists_v5.py <e>b --force diagnosis --config <cfg with fault_weight> --verbose 2`, about 2.5-3 h on the GPU. It redoes assembly, RUL, test and card after.

Expected on the card:
- `air_filter_fouling` recall at least 0.5;
- `oil_pump_degradation` and `oil_degradation` recall both at least 0.5;
- `Diagnosis` macro F1 not lower than before: 912 0.834, 915 0.808, 916 0.803.

- [ ] **Step 6: Commit code, config and artifacts**

---

### Task 4: Export and accept, per engine

**Files:**
- Modify: `backend/models_v5/912|915|916/`, through `export_v5.py` only.

- [ ] **Step 1: Export from a clean tree**

Run: `validation/venv/bin/python validation_v5/export_v5.py <e>b`. Expected: the manifest shows `git_dirty: false`.

- [ ] **Step 2: Check parity on both devices**

Run: `validation/venv/bin/python validation_v5/tests/test_export_parity_v5.py <e>b`, then the same with `AERO_TEST_CPU=1`. Expected: both `OK`.

- [ ] **Step 3: Flight test**

Run: `cd backend && ../validation/venv/bin/python ../scripts/e2e_v5.py --engines <e>`. Expected: no new FAIL compared with today. Today's only failure is the 915's turbo_high.

- [ ] **Step 4: A/B against the live export**

Run: `validation/venv/bin/python scripts/ab_v5.py --engine <e> --old origin/main --new .`

Expected: `A/B RULE: PASS`. If it fails:
1. `git checkout origin/main -- backend/models_v5/<e>`;
2. keep the artifacts;
3. note the result on the PR.

- [ ] **Step 5: Commit the accepted exports, open the PR with the per-engine before/after table, merge after CI, update the main checkout and restart the stack**

---

### Task 5: 915 turbo_high RUL (time-boxed investigation, half a day)

On `e2e_v5 --engines 915`, turbo_high has the turbo fault named correctly, but RUL drops below the calendar on only 19% of windows (true 147 h, calendar 222 h, predicted 110 h). The RUL value is reasonable; the wear-limited flag is not.

**Files:**
- Read: `validation_v5/rul_v5.py` (`TrendGatedRUL` and the gate classifier), `scripts/e2e_v5.py` (the `judge` rule for "RUL below calendar").

- [ ] **Step 1: Log, per window on that flight, the gate classifier's fault-limited probability and the RUL it returns. One script in the scratchpad; no repo change.**
- [ ] **Step 2: Decide.**
  - If the gate probability sits just under its threshold at high altitude, record it as a known limit on the PR.
  - A fix that changes the RUL model needs its own plan: retraining RUL alone is cheap (`--force rul,test,card`), but it moves every 915 RUL number and must pass the A/B again.

---

### Task 6: Docs and branches

**Files:**
- Modify: `docs/architecture.md`. Lines 12, 32, 57 and 81-96 describe v2 as the default and v3 as opt-in. Replace the version table with v5 as the default:
  - one row per version (v2-v5), giving the physics module, AI service, models directory and status;
  - v5 default, with each engine's own models;
  - v4 superseded;
  - v2 and v3 legacy.

  Point to `README.md` §Physics v5 rather than repeating it.

- [ ] **Step 1: Edit the doc and check every path it names exists (`git ls-files <path>`)**
- [ ] **Step 2: List remote branches already merged into main**

Run: `git branch -r --merged origin/main | grep -vE "origin/(main|HEAD|archive/|backup/)"`

- [ ] **Step 3: Show the list to the user and delete only on their yes (`git push origin --delete <branch>...`)**

Never delete `archive/v6-914-pre-rebase`: the v6 model manifests cite its commit IDs. Never delete `backup/*` either.

- [ ] **Step 4: Commit the doc; PR; merge after CI**

---

## Findings (filled in by Task 1)

| Engine | Severity bias from the gate (val) | Air-filter val positives / val recall / test recall | Oil confusion (pump→deg / deg→pump) | Task 2 path | Task 3 path |
|---|---|---|---|---|---|
| 912 | 0.52 (bias -0.065: gated-out -0.270 on 693 cells, kept -0.035) | 545 / 0.80 / 0.71 | 0.02 / 0.03 | gate_cost + kept-cell offset | none (all faults >= 0.5 on test) |
| 915 | 0.52 (bias -0.027: gated-out -0.179, kept -0.014) | 487 / 0.81 / 0.73 (precision 0.40) | 0.06 / 0.07 | gate_cost + kept-cell offset | none |
| 916 | 0.50 (bias -0.054: gated-out -0.236, kept -0.031) | 251 / 0.63 / 0.33 | 0.00 / 0.19 (test 0.29) | gate_cost + kept-cell offset | 3A (pooled jittered val windows) |

Neither cause dominates: about half the bias comes from cells the gate zeroes, half from cells it keeps. So Task 2 takes both fixes. The offset is fitted on the cells the chosen gate keeps, so the two compose. No engine meets 3B's oil rule (val confusion > 0.3), so there is no diagnosis retrain.

## Order and time

1. Task 1: about 1 h.
2. Task 2 and Task 3A: post-hoc, about 1 h for all three engines.
3. Task 4: about 30 min per engine.
4. Task 3B, only if Task 1 says so: about 3 h per engine, overnight.
5. Tasks 5 and 6: any time, independent of the rest.

## Outcome (2026-10-17)

Executed in order. **No model export changed.** The live 912, 915 and 916 models stay as they were.

| Task | Result |
|---|---|
| 1. Gap report | Shipped: `validation_v5/gaps_v5.py` plus each engine's `gaps.json`. Findings are in the table above. |
| 2. Severity bias | `gate_cost` and a kept-cell offset brought 912 test bias from 0.057 to 0.032 (error 0.082 to 0.076), but left 915 and 916 unchanged. Severity feeds the RUL inputs, and the refit RUL lost its A/B (912: 90 to 97 h), so the change was dropped. The bias that remains is faults diagnosis does not fire on, which the gate zeroes. |
| 3A. Air-filter cut-offs | Pooling 81,864 jittered val windows moved the 916 cut-off from 0.59 to 0.60 and test recall from 0.33 to 0.32. The val/test gap is flight-level, so this was reverted. 3B's trigger was not met. |
| 4. Accept per engine | Every rebuilt export failed its A/B against the live one (912 and 916 on RUL error; 915 identical), so none shipped. Parity on GPU and CPU and `e2e_v5` were as before. |
| 5. 915 turbo_high | The RUL gate is right (p = 0.99) and the base regressor is accurate: 150 h against 147-160 h true, below the calendar on 100% of windows. `TrendGatedRUL`'s logbook trend reads about 340 h for turbo degradation and drags RUL up to the calendar cap. The likely cause is that this fault is visible only above critical altitude, so inspections under-see its growth. The fix (no trend for altitude-gated faults, or a refit per family) needs its own plan. |
| 6. Docs | `docs/architecture.md` now shows v5 as the default. The branch cleanup is listed for the owner. |

**Next real levers:**
1. More air-filter flights in val and test, or a diagnosis retrain with `fault_weight`.
2. Remove the logbook trend for altitude-gated faults, then refit RUL and run the A/B.
