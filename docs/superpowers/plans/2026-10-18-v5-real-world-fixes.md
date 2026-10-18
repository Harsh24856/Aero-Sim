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

Status: P1 and P1b are done in PR #29, which is waiting for the user to merge. P2-P4 are still to do.

P1b result (2026-10-18). Live flights from the PR branch, with the stack started from the worktree and the same models as live, on the full profile:
- **916:** FAULT shown on 0 of 170 polls. The raw detector was at or above 0.5 on 38 polls, all inside throttle holds.
- **915:** FAULT shown on 0 of 170 polls (raw: 29).
- **In cruise:** shown detection was at most 0.06.
- **RUL:** never rose.
- **Bug found and fixed:** a frozen RUL did not count down the hours flown. It read 1,678 h beside a 1,677 h calendar.
- **Margin:** on the 916, after the advance to full throttle, the raw alarm lasted until +121 s. The 128 s hold covers it, but only just.
- **Baseline:** `scripts/transient_v5.py` and its baseline are committed (`validation_v5/artifacts/transient_v5_baseline.json`). 10 of 20 cases fail the 5 s gate. That is P2's target.

### P1b: ship P1 and check it live (about 30 min)

1. When CI is green, merge PR #29. Then pull into the main checkout. This follows the memory rule: the live stack runs from the main checkout.
2. Restart the backend. The AI service does not change.
3. Fly the same Playwright profile on the 916 as in the first flight:
   - take-off;
   - 75% cruise at 1,500 m;
   - full power at 4,000 m, then at 6,000 m;
   - throttle 0.15 at 1,000 m;
   - full throttle again.
   Expected:
   - the cockpit shows "settling after a large throttle change" instead of FAULT;
   - no detection flicker in cruise;
   - remaining life never rises.
4. Also fly the 915 once through the same profile.
5. Make the offline measurement permanent: put the chop-sweep and replay scripts into
   `scripts/transient_v5.py`. They are now in the scratchpad. The script prints, for each
   engine and each throttle step, the maximum detection and how many seconds detection is
   at or above 0.5. P2 and P3 use it as their gate.

Done when: the live flight shows no false FAULT, and `transient_v5.py` is committed with a baseline.

### P2: teach the models throttle steps and idle descents (about 1.5 h of work, 3-4 h of machine time)

1. **Generator** (`generate_dataset_v5.py`). Add two arguments:
   - `--start-id N`: today's 6,000 flights are not touched. New flights are numbered from 6,000, with their own RNG streams.
   - `--missions a,b`: new flights draw only from the named missions.

   Two new missions (they are metadata only, so the model inputs and the scaler do not change):
   - `throttle_steps`: at 1,500-3,500 m, the throttle moves instantly by 0.4-0.85, in either direction, every 60-180 s. This is the measured cause.
   - `idle_descent`: from 4,000-6,500 m down to 500 m at throttle 0.10-0.20, then full throttle again. This covers the remaining 914 gap at 6,000 m.

   Both missions bypass the throttle lag (0.12 per second) at the step instants.

   Tests:
   - the first 6,000 flights are byte-identical to today's (check a hash of one file);
   - a `throttle_steps` flight contains a throttle move of 0.4 or more within 2 s.
2. **Generate** 900 flights per engine (15%), all four engines in parallel with `--jobs 4`. The 916 rate was 6,000 flights in 5.2 h, so this takes about 50-60 min if the CPU keeps up.
3. **Cache**. `pipeline_v5.py` gets a new argument, `--scaler-from <old contract>`, so the new cache keeps the live scaler. The deployed contract must not change. Test: rebuilding with `--scaler-from` gives a cache whose scaler is the old one.
4. **Fine-tune**: `specialists_v5.py <e>b --finetune detect,sensor,diag`, 12 epochs, keeping the current weights unless validation improves. Order: the 916 first, then the 915, 914 and 912 (the user's rule is 914 first through every new phase; ask whether it still applies to a fine-tune). About 30 min per engine.
5. **Acceptance per engine**:
   - `ab_v5.py` against live: win or tie.
   - `transient_v5.py`: detection at or above 0.5 for 5 s or less after each step (today the 916 measures 15-80 s).
   - No test-card gate flips from PASS to FAIL.
   - e2e: no new failures.
6. **Ship**: export, parity, PR, merge, restart. Then lower `THROTTLE_HOLD_S` to what `transient_v5.py` shows each engine needs. If every engine passes, remove the hold.

Done when: the four engines are accepted or kept, and the hold is shortened or removed.

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
- **915 RUL trend**: switch the trend per fault instead of per family. This fixes the e2e `turbo_high` failure (RUL below calendar on only 19%).

Keep P4 in separate runs from P2, so each A/B result has one cause.

## Order

P1b now, then P2 (overnight), P4 when convenient, and P3 only as a separate decision. Every phase ends with `transient_v5.py` and a Playwright flight against the same baseline.
