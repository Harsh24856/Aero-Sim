# v5 Real-World Fixes: Plan

Source: the Playwright flight of the Rotax 916 iS on 2026-10-18 (take-off, 75% cruise at
1,500 m, full power at 4,000 m and 6,000 m, then throttle 0.15 at 1,000 m and 35 m/s).

## What the flight found

| # | Problem | Side | Cause found in the code |
|---|---|---|---|
| 1 | Healthy engine shows "FAULT, 100% likely" at throttle 0.15. The RPM sensor shows as a spike fault. | AI | The training missions never command a throttle below 0.30 (`descent` ends at 0.30, `rapid_throttle` at 0.35; `generate_dataset_v4.py:141`). Throttle 0.15 is outside the training data. |
| 2 | In steady cruise, the fault probability jumps between 0.7% and 50%, and health moves ±2% every second. | AI / display | `main.py` smooths only RUL (`smooth_rul`). Detection and health go to the cockpit raw. |
| 3 | 5,890-5,970 rpm at full throttle at altitude. The limit is 5,800. | Physics | `max_rpm` exists in the spec, but nothing enforces it. The propeller is fixed pitch and has no governor. |
| 4 | Lambda is 0.89 in every phase. | Physics | `Inputs.target_lambda` is fixed at 0.88. Nothing schedules it with power. |
| 5 | CHT is 71 °C and coolant is 55 °C in cruise. CHT is 59 °C at full power at 6,000 m. | Physics | The cooling is sized for the hot-day climb, and there is no coolant thermostat (the test comment says so). |
| 6 | At full throttle at sea level: 4,980 rpm and 106 kW. A real installation reaches 5,500-5,800 rpm. | Physics | The fixed-pitch propeller is sized for altitude. It loads the engine down at sea level. |

The mission name is metadata only (`pipeline_v5.py:139`), not a model input. A new mission
therefore does not change the model's inputs or its scaler.

## P1 outcome (2026-10-18)

Offline replays corrected problem 1's cause. **Low throttle is not the cause. A fast throttle step is.**
- On a healthy 916, a steady throttle of 0.15 is clean (detection at most 0.2), and an altitude step alone is clean.
- One throttle step, either way, gives "FAULT" until the step leaves the 128 s window:
  - 0.75 to 0.15 at 1,500 m: about 15 s.
  - Full power to 0.15, with a step from 6,000 m to 1,000 m: about 80 s.
  - 0.15 to full: up to 125 s.
- The 915 behaves the same. The 914 and 912 are mostly clean.
- The generator eases the throttle (about 8 s lag) and never steps it.

Shipped instead of the planned throttle floor:
- **Hold:** a throttle move of 0.4 or more, either way, within 5 s holds alerts for the full 128 s window (`settle_reason: "throttle"`). During this hold, RUL keeps its last value and its floor.
- **Smoothing:** EMA (10 s) on detection and health. `fault_detected` follows the smoothed detector.

Replay of the live profile through the `main.py` chain: FAULT shown on the 916 for 1 s (raw detector: 202 s), on the 915 for 0 s (raw: 201 s).

Known gap: on the 914, a steady throttle of 0.15 at 6,000 m still alarms after the window clears. This is out of envelope in steady state, so no hold covers it.

For P2, the new mission must contain **throttle steps** (instant moves of 0.4-0.85, either way) and an idle descent from altitude, not only low throttle.

## Remaining work, phase by phase

Status: P1 and P1b are done in PR #29, which is open and waiting for the user to merge. Next is Step 0, then P2. P3 and P4 are still to do.

P1b result (2026-10-18). Live flights from the PR branch, with the stack started from the worktree and the same models as live, on the full profile:
- **916:** FAULT shown on 0 of 170 polls. The raw detector was at or above 0.5 on 38 polls, all inside throttle holds.
- **915:** FAULT shown on 0 of 170 polls (raw: 29).
- **In cruise:** shown detection was at most 0.06.
- **RUL:** never rose.
- **Bug found and fixed:** a frozen RUL did not count down the hours flown. It read 1,678 h beside a 1,677 h calendar.
- **Margin:** on the 916, after the advance to full throttle, the raw alarm lasted until +121 s. The 128 s hold covers it, but only just.
- **Baseline:** `scripts/transient_v5.py` and its baseline are committed (`validation_v5/artifacts/transient_v5_baseline.json`). 10 of 20 cases fail the 5 s gate. That is P2's target.

### Step 0: land P1 (user, about 15 min)

1. **User:** merge PR #29. The permission check will not let me merge to `main` without the user's review.
2. **Me:** pull into the main checkout and start the stack from there. On 2026-10-18 the stack was found stopped: no listeners on 3000, 8000 or 8100.
3. **Me:** fly the 916 once to check the merged build. Expected result as in P1b: FAULT on 0 polls.

### P2: teach the models throttle steps and idle descents (overnight, 916 first)

Engine order is 916, 915, 914, 912, by the size of the false alarm (the user chose to start with the 916). Each task ends with a commit, and each engine ends with an accept/keep line in this plan.

**P2.1 Generator (TDD, about 45 min)** — `backend/generate_dataset_v5.py`
- New argument `--start-id N`: new flights are numbered from N (6,000), each with its own RNG stream. Today's flights do not move.
- New argument `--missions a,b`: new flights draw only from the named missions.
- New missions (metadata only, so the model inputs and the scaler do not change):
  - `throttle_steps`: 1,500-3,500 m; an instant throttle move of 0.4-0.85, either way, every 60-180 s.
  - `idle_descent`: from 4,000-6,500 m down to 500 m at throttle 0.10-0.20, then full throttle again.
- At a step, the throttle bypasses the 0.12 per second lag.
- Tests:
  - a flight with an ID below 6,000 is identical to today's (compare a hash of one generated flight with and without the new arguments);
  - a `throttle_steps` flight has a throttle move of 0.4 or more within 2 s;
  - an `idle_descent` flight starts above 4,000 m and reaches throttle 0.2 or less.

**P2.2 Data (about 1 h of machine time, CPU)**
- Generate 900 flights per engine (15%) into `data/rotax_v5/<e>_steps/`, all four engines in parallel (`--jobs 4`, under `caffeinate`).
- Spot-check with `transient_v5.py`-style plots: the steps are visible in the telemetry.

**P2.3 Cache (TDD, about 1 h)** — `validation_v5/pipeline_v5.py`
- New argument `--scaler-from <live contract_v5.json>`: the live scaler is reused, so the deployed contract does not change.
- New argument `--extra <dir>`: the new flights are appended, with the same split rule (12/3/2/3 slots), into a new cache `cache/<e>s`.
- Tests:
  - the scaler equals the live one;
  - the old flights' windows are byte-identical;
  - new flights land in all four splits.
- Check the share of windows that contain a step. If it is under 8%, oversample the new flights in `end_ids` (one argument, `--extra-weight`).

**P2.4 Fine-tune (about 30-40 min per engine, GPU)**
- Run `specialists_v5.py <e>s --finetune detection,sensor,diagnosis` with 12 epochs and `keep_current=True`, so the current weights stay unless validation improves.
- The sensor head is included because the live false alarm also raised an `engine_rpm` spike flag.
- Before starting, confirm from the live export's manifest which cache the live heads came from, and fine-tune from exactly those weights.

**P2.5 Acceptance per engine (about 45 min per engine)** — all must hold:
- `transient_v5.py`: at most 5 s of detection at or above 0.5 in every case (916 today: 17-80 s);
- `ab_v5.py` against live: a win or a tie;
- no test-card gate flips from PASS to FAIL;
- e2e: no new failures (the 915 `turbo_high` failure is already known);
- export parity: the GPU and CPU give the same answers.

If an engine fails, it keeps its live models and the hold stays.

**P2.6 Ship (about 30 min)**
- Export, PR, user merges, main checkout, restart.
- For engines that pass, lower `THROTTLE_HOLD_S` to their measured need. If all four pass, remove the hold.
- Fly the Playwright profile again on the 916 and the 915.

**Risks:**
- **Step windows too rare to matter:** the oversample argument in P2.3 handles this.
- **The fine-tune trades away fault recall:** the A/B gate catches this.
- **Mac sleep:** use `caffeinate` and check `pmset` if a run stalls (memory note).

### P2 results

**916 (2026-10-18): accepted as run `916bt`.**
- **How it was built:** the detection, sensor and diagnosis heads were fine-tuned on `916sb`, which is the old flights plus 900 step flights, oversampled x2. RUL was then refitted on the old 916b flights.
- **Validation:** all three heads were kept:
  - detection 1.8881 -> 1.8911;
  - diagnosis 1.2923 -> 1.3056;
  - sensor 0.8871 -> 0.9003.
- **`transient_v5.py`** (seconds of detection at or above 0.5):

  | Case | Before | After |
  |---|---|---|
  | chop 0.75 -> 0.15 | 17 | 0 |
  | full -> 0.15 at 6,000 m | 42 | 0 |
  | full -> 0.15, 6,000 -> 1,000 m | 80 | 6 (gate 5) |
  | 0.15 -> full | 73 | 0 |
  | small step | 1 | 0 |

- **A/B against live:**
  - naming 94.1% -> 94.3%;
  - healthy false alarms 0% -> 0%;
  - flights passing 22/23 on both (air filter fails on both);
  - **mean RUL error 75 -> 77 h** (the rule fails by 2 h).
  - With the RUL refitted on `916sb` it was 104 h; with the live RUL model kept on the new heads it was 82 h.
- **Other checks:** e2e 8/8, parity OK, `test_all` passes.
- **Ruling:** accepted over the 2 h RUL miss, because it is within the per-flight noise (+-50 h) and the target false alarm went from 80 s to 6 s. Cost if wrong: about 2 h more mean RUL error. Reverting is a single commit.

**Recipe for the other engines:** use the same `<e>sb` fine-tune, then the `<e>bt` RUL refit, then export and run the checks.

**All engines (2026-10-19).** A full candidate changes the diagnosis and family outputs that RUL reads, and it cost naming or RUL on every engine. The fix is `scripts/hybrid_export_v5.py`: the live export plus only the candidate's detection and sensor nets, detection GBT and sensor bias.

Table columns:
- **transient s:** seconds of detection at or above 0.5, for chop 0.75, chop at 6,000 m, chop at 6,000 -> 1,000 m, advance, small step.
- **naming, RUL error:** A/B against the export before P2.

| Engine | Shipped | transient s (before -> after) | naming | RUL error | Throttle hold |
|---|---|---|---|---|---|
| 916 | hybrid (916bt det/sensor) | 17/42/80/73/1 -> 0/0/0/0/0 | 94.1 -> 94.1% | 75 -> 75 h | **removed** |
| 914 | hybrid (914bt det/sensor) | 0/71/0/11/0 -> 0/0/0/0/0 | 93.7 -> 93.7% | 94 -> 94 h | **removed** |
| 915 | hybrid (915bt det/sensor) | 0/13/76/60/0 -> 0/0/0/3/0 | 82.6 -> 82.6% | 52 -> 52 h | kept (see below) |
| 912 | live, unchanged | 0/1/0/6/0 | candidate 88.9 -> 87.0% | candidate 90 -> 106 h | kept |

Notes:
- **Full candidates, A/B:**
  - 916bt: naming 94.3%, RUL 77 h;
  - 915bt: naming 84.0%, RUL 63 h (61 h on the live RUL model);
  - 914bt: naming 90.5%, RUL 90 h;
  - 912bt: naming 87.0%, RUL 106 h.
- **Replay through `main.py`:** the live Playwright profile was replayed through the `main.py` chain with no hold. FAULT was shown on the 916 for 0 s, the 914 for 0 s, and the 915 for **11 s**. The 915's raw detection was about 0.6 for 23 s after full power at 6,000 m was chopped to idle with an instant 5,000 m altitude step. So the 915 keeps the hold: `THROTTLE_HOLD_ENGINES = {912, 915}`.
- **Flights failing on both live and candidate:** air filter (all engines) and the 915 `turbo_high` RUL in e2e.

### P3: physics realism (separate decision, about 2 days of machine time)

All three fixes change the training data of all four engines, so they need a full regeneration and retrain. Each fix gets its own test in `test_physics_v5.py` first.
- **Mixture** (problem 4): on EFI engines, lambda is 1.0 up to about 85% power, then enriches to 0.88 at full power. Test: lambda in 75% cruise is between 0.97 and 1.03.
- **RPM** (problems 3 and 6): a constant-speed propeller governor on the 914, 915 and 916 (rpm held at the commanded value, 5,800 at most). Test: full throttle from 0 m to 6,000 m stays at 5,800 rpm or below, and take-off rpm is at least 5,500.
- **Coolant thermostat** (problem 5): starts to open at 80 °C. Test: cruise coolant is between 80 and 95 °C, and the hot-day certification test still passes.

Then:
- regenerate all four engines (`--all --jobs 4`, about 5-6 h);
- retrain everything with `run_v5` (914 first);
- run the full test card, A/B and e2e;
- re-fly with Playwright.

The P2 missions carry over. Batch P3 with the next planned full retrain.

### P4: known model backlog (optional, can run alongside P2)

- **Oil-fault labels**: 58-100% of missed oil faults are labelled as present before any instrument sees them. Fine-tune on the per-instrument relabel (`relabel_v5b.py`, which already exists). Gate: A/B against live.
  - **Tried on the 914 (2026-10-19): rejected.** The diagnosis was fine-tuned on `914pc` (the per-instrument relabel) and validation kept it (1.3413 -> 1.3539). It was shipped as a hybrid that keeps the live detection and sensor nets.
  - A/B against live:
    - naming **93.7% -> 78.3%**;
    - oil pump named exactly 100% -> 25% (74% wrong);
    - oil degradation 53% -> 0%;
    - air filter 74% -> 0%;
    - RUL 94 -> 104 h.
  - The stricter labels teach the model to hold back on oil and air-filter faults, and on the test flights it then names the wrong one. The 912, 915 and 916 were not run (the queue was stopped) and the 914 stays live.
  - Better labels on validation do not mean better naming in flight. A next attempt needs the e2e presets judged on the same visibility rule, or only a partial relabel (for example oil degradation only).
- **915 RUL trend**: switch the trend per fault instead of per family. This fixes the e2e `turbo_high` failure (RUL below calendar on only 19%).
  - **Done 2026-10-19, no gain.** `fit_rul` now has `+fault` candidates.
  - On the 915 (run `915bp`, on the live 915b heads), validation chose `gated_gbt+trend0.5+fault` (wear-limited error 9.18% against 9.31%). But the trend stays allowed for turbo, because validation says it helps there.
  - A/B: RUL 52 -> 52 h, and `turbo_high` still fails at 19%. Not shipped.
  - **Real cause:** the wear-limited classifier calls the turbo_high flight calendar-limited 81% of the time. The trend is not the cause. The next lever is that classifier (turbo windows above critical altitude).

Keep P4 in separate runs from P2, so each A/B result has one cause.

## Order

Step 0 (merge PR #29), then P2 overnight (916 first), P4 when convenient, and P3 only as a separate decision. Every phase ends with `transient_v5.py` and a Playwright flight against the same baseline.
