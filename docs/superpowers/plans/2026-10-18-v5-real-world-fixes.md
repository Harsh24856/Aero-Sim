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

## Phases

### P1: stop the false alarm on screen (about 1 h, no retraining)

1. **Reproduce offline.** A scripted flight through `ai_service` on all four engines:
   5 min cruise, then 3 min at throttle 0.15. Record detection, faults and sensor flags.
   This confirms whether problem 1 happens on every engine or only on the 916.
2. **Hold alarms outside the trained throttle envelope.** This uses the same mechanism as
   the existing ground-roll hold (`hold_alerts_outside_envelope`, `main.py:443`). The new rule:
   when the throttle is below `AI_ENVELOPE_MIN_THROTTLE = 0.25`, hold the alerts until the
   128 s AI window is clear of those samples. The cockpit already shows "stabilising" with a
   countdown for this state. Tests: one for throttle 0.15 (held), one for throttle 0.30 (not held).
3. **Smooth what the cockpit shows.** Add an EMA (about 10 s) on `detection_confidence` and
   `health_percent`. The raw values stay available as `*_raw`. To keep alarm latency
   unchanged, `fault_detected` still uses the raw detector. Test: a steady input gives a
   smooth output, and a step input reaches 90% within 25 s.
4. **Check.** Run `scripts/e2e_v5.py` (8/8), the backend tests, and the Playwright flight
   again, including the idle phase.

P1 hides problem 1. It does not fix it. P2 fixes the cause.

### P2: teach the models low power (about 4-5 h, mostly machine time)

1. **Add a mission** `idle_descent` to the generator: throttle targets 0.10-0.20 at
   35-45 m/s, a descent from 2,500 m to 500 m, with throttle steps up to 0.6 and back.
   Increase `rapid_throttle` so that some steps go down to 0.15. Lower the training throttle
   clip from 0.08 only if the physics idles cleanly there (check with the existing
   `test_ground_idle_near_1400_rpm`).
2. **Generate extra flights only.** Generate about 15% more flights per engine with the
   new missions, using the same stratified split and the same scaler (the scaler is taken
   from the train flights; rebuild the cache with the old scaler, do not refit it).
   Generation takes about 3 h for the four engines, on the CPU.
3. **Fine-tune.** Use the existing `specialists_v5.py <e>b --finetune detect,diag` on
   old + new data. The validation rule keeps the old weights unless validation improves.
4. **Accept per engine.** The engine passes when `ab_v5.py` against live wins or ties,
   the new idle replay test passes (detection < 0.5 on a healthy engine at throttle 0.15),
   and no test-card gate goes from PASS to FAIL.
5. **Ship.** Export, parity check, e2e, then PR. When P2 ships, the P1 throttle hold can
   be lowered to the new training floor.

### P3: physics realism (separate round, needs a full regeneration)

Problems 3-6 change the physics that all four engines' training data comes from. Fixing
them means a full regeneration and a full retrain of all four engines (about 2 days of
machine time, as in the v4 overhaul). Proposed fixes:

- **Lambda schedule (problem 4):** on EFI engines (912 iS, 915 iS, 916 iS), lambda 1.0 up to
  about 85% power, then rich to 0.88 at full power. Carburetted engines keep today's model.
- **RPM limit (problems 3 and 6):** model the 914, 915 and 916 with a constant-speed
  propeller (a governor holds the commanded rpm, 5,800 at maximum). Or, as a smaller
  change, keep the fixed-pitch propeller, size it for sea level, and add an rpm limit.
- **Coolant thermostat (problem 5):** a coolant thermostat that starts to open at 80 °C.
  The hot-day certification test must still pass.

Recommendation: do not start P3 until the user decides. It makes all current models
obsolete. Batch it with the next planned full retrain.

## Order and recommendation

P1 now (small, safe, visible on every descent). Then P2 overnight. P3 only as its own
decision. Each phase ends with the Playwright flight again, so we check against the same
real-world baseline.
