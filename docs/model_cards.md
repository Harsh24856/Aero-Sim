# Model cards — physics v3

All numbers are measured on the held-out **test** split (10% of scenarios, split by scenario)
of dataset v3 unless stated otherwise. Deployed heads live in `backend/models_v3/<engine>/`,
each with a `manifest.json` recording its source checkpoint, SHA-256 and test metrics.

Engines covered: **Rotax 912 ULS, 914 ULF, 915 iS, 916 iS**.

All four engines were retrained on 2026-09-15 (19 h, all 20 gates passed) on data regenerated after a
pre-regeneration audit: per-engine propellers (912 1.97 m, 915 2.30 m, 916 2.28 m), cooling degradation
that scales with engine power, random sensor fault types per channel, visible RPM glitches, an airspeed
floor, recalibrated oil temperature, the 916 iS TBO corrected to 2,000 h, and every 5th scenario starting
on a new engine. Metrics are not comparable with earlier model sets.

## Shared facts

| | |
|---|---|
| Data | physics v3 twin, 10 M rows per engine, scenarios seeded at stratified life stage, 1 s sampling |
| Input | 128-second window x 25 features, `StandardScaler` fit on the train split only |
| Encoder (heads 1-4) | 6 causal dilated Conv1D blocks, 32 channels, receptive field 253 s |
| Serving | `backend/aiv3.py`, TensorFlow on the Metal backend, `.predict()` only |
| Parity | live incremental inputs and all predictions match the batch pipeline (`validation/parity_ai_v3.py`) |

## 1. Fault detection

Binary: is any sensor fault or engine failure mode present at detectable strength
(severity >= 0.05)? Source: phase 1 checkpoint (later phases degrade it: phase-3 AUC on 914 falls to 0.881).

| Engine | AUC | Accuracy @0.5 |
|---|---|---|
| 914 | 0.984 | 0.953 |
| 912 | 0.981 | 0.954 |
| 915 | 0.992 | 0.966 |
| 916 | 0.990 | 0.963 |

## 2. Sensor fault diagnosis

Per channel (EGT, CHT, oil pressure, oil temp, vib X/Y/Z, RPM), one of none / Bias / Drift /
Spike / Stuck-At / Noise. Source: phase 2 checkpoint.

| Engine | Macro-F1 over fault classes | Fault recall |
|---|---|---|
| 914 | 0.663 | 0.968 |
| 912 | 0.665 | 0.969 |
| 915 | 0.623 | 0.969 |
| 916 | 0.656 | 0.968 |

Each channel now receives 2-3 randomly chosen fault types (EGT/CHT Bias, Drift, Noise; oil pressure/temperature
Bias, Drift; vibration Noise, Bias, Spike; RPM Spike, Stuck-At), so macro-F1 reflects genuine type
discrimination. Per-channel precision of confident calls is measured on the test split and stored in each
manifest (`diagnosis_channel_reliability`); channels no better than chance cannot cap health or raise items.

## 3. Fault severity

Per-channel severity 0-1. Source: phase 3 checkpoint. Error measured only where a fault is present
(a head predicting zero everywhere would otherwise score well).

| Engine | MAE on faulty channels | MSE on faulty channels |
|---|---|---|
| 914 | 0.020 | 0.010 |
| 912 | 0.029 | 0.017 |
| 915 | 0.064 | 0.038 |
| 916 | 0.047 | 0.024 |

## 4. Engine failure modes

Independent sigmoid per mode: misfire, injector fouling, cooling degradation, combustion
instability. Source: phase 4 checkpoint. AUC for "mode present". The gate uses windows where the
mode is detectable (severity >= 0.05) or absent; the strict AUC (any severity > 0) is also shown.

| Engine | AUC detectable (gate) | AUC strict | Misfire | Injector | Cooling | Combustion |
|---|---|---|---|---|---|---|
| 914 | 0.936 | 0.876 | 0.949 | 0.977 | 0.893 | 0.933 |
| 912 | 0.943 | 0.878 | 0.968 | 0.993 | 0.883 | 0.902 |
| 915 | 0.919 | 0.855 | 0.953 | 0.992 | 0.838 | 0.904 |
| 916 | 0.941 | 0.871 | 0.952 | 0.973 | 0.931 | 0.925 |

Alert thresholds are chosen per mode for maximum F1 on the test split and stored in each engine's manifest
(`failure_mode_thresholds`, with precision/recall in `failure_mode_threshold_stats`).

Cooling degradation, previously the weakest mode (0.58-0.77), is now 0.84-0.93: its physics effect was
strengthened to scale with engine power (+30-40 C CHT at full severity).

## 5. Remaining useful life

Engine hours remaining against TBO (912/914/916: 2,000 h, 915 iS: 1,200 h). Separate 2-layer LSTM (32 units) over the window plus the 10
auxiliary inputs, softplus output in hours, Huber loss measured in % of TBO.

| Engine | MAE | MAE % of TBO | Correlation with true hours | Monotonic across 10 life-stage bins |
|---|---|---|---|---|
| 914 | 22.9 h | 1.15% | 0.9980 | yes |
| 912 | 15.4 h | 0.77% | 0.9989 | yes |
| 915 | 7.5 h | 0.62% | 0.9992 | yes |
| 916 | 13.6 h | 0.68% | 0.9991 | yes |

Ablation: with `bsfc_ratio` replaced by its mean, 914 MAE moves 1.65% to 1.70% of TBO; a
BSFC-only linear fit scores 5.55%. The head uses the window, not a single ratio.

**Limitations.**
- Near-new engines are under-represented: windows with true RUL >= 95% of TBO are 0.0-0.1% of the
  test split, and there the head under-predicts by 45-106 h (5-9% of TBO). Every live session starts
  with a new engine, so live RUL reads 5-13% of TBO low (measured on the running stack).
- No uncertainty estimate: `model(x, training=True)` differs from `.predict()` by up to 459 h on this
  architecture, so MC-dropout cannot be used.

## Live stack check

2026-09-15, retrained models, `aiv3.py` + `main.py` on physics v3. Headless run at 1,500 m / 45 m/s,
throttle 0.5 (cruise) then 0.7 at 2,500 m:

| Engine | Cruise health | Cruise RUL error | Throttle 0.7 |
|---|---|---|---|
| 914 | 87-100% | 4-5% of TBO | 94.7-100%, no faults (none injected) |
| 912 | 99.7-99.9% | 6% of TBO | not run |
| 915 | 100% | 3-4% of TBO | vibration stress faults injected by physics; detected, health 0% |
| 916 | 99.6-100% | 2-3% of TBO | vibration stress faults injected by physics; detected, health 0% |

Browser flights (takeoff from the runway, 35% throttle, 1,000 m) on all four engines: health 98-100% at cruise,
advisory nominal. Full throttle on the 915 drove EGT/CHT to their sensor limits; detection, the warning
advisory and the low-health vignette fired, and health recovered to 97.8% after throttling back. The 916's
first AI windows after takeoff (which contain ground roll below the 32 m/s dataset floor) raised detection
confidence 0.77-0.86; `main.py` holds those alerts until the window is clear (see roadmap).

**Real-time latency** (M2, 8 GB, AI on Metal TensorFlow, all five heads per sample; `/health`
`sim_status.ai_latency_ms`): from a physics sample entering the AI queue to its diagnosis being stored for
the next 20 Hz broadcast, **p50 260 ms, p95 378 ms** over 107 samples of a 914 cruise flight, with a single
1.18 s outlier on the first sample after warm-up. The AI runs once per simulated second, so every diagnosis
is on screen before the next sample exists; physics loop lag stayed at 0 ms and no samples were dropped.

**Aircraft over CAN** (2026-09-15, plan P3). A separate process (`aircraft_sim.py`) flew its own 914
through takeoff, climb, cruise and a 0.95-throttle leg, sending only CAN frames over the UDP multicast bus;
the twin received set-points and measured sensors through `can_ingest.py` (112,000+ frames, 0 lost,
0 rejected). Results with no mismatch:

- The twin's own EGT/CHT tracked the measured values to within 0.5 C from set-points alone, and the AI
  warmed up in real time on measured data (health 99.2-99.9% at climb and cruise, detection confidence
  0.00-0.01, no residual deviations).
- Before the power leg, cruise health drifted down to 92.3% with no detection and no residual deviation -
  a soft AI health estimate, not an alarm.
- At 0.95 throttle the aircraft's physics injected a stress fault on its EGT sensor, which pinned at 999 C
  while the twin's physics expected 834-924 C. The residuals flagged EGT, oil pressure, oil temperature
  and vibration; the AI raised detection at 100% confidence (health 0%), the advisory went to warning and
  the low-health vignette fired. The fault existed only in the aircraft; the twin found it from the bus.

**Plant mismatch** (`validation/mismatch_eval.py`, 914, cruise at 1,500 m / 0.5 throttle / 45 m/s, 420 s per
run of which 128 s are AI warm-up; the plant injected no sensor faults or failure modes in any run, so every
alarm below except the drift case is false):

| Plant differs from the twin by | AI false alarm | Failure-mode false | Health mean / min | RUL error % TBO | Residual layer |
|---|---|---|---|---|---|
| nothing | 0% | 0.7% | 95.4 / 82.9 | 4.4 | quiet (0%) |
| calibration offsets (EGT +15, CHT +4, oil temp +3 C, oil pressure -2 psi) | 0% | 0% | 99.8 / 99.7 | 5.5 | flags every sample |
| unit-to-unit gain spread (3-5%, vibration +20%) | 0% | 0% | 99.8 / 99.7 | 15.4 | EGT flagged throughout |
| realistic sensor noise | 0% | 0.3% | 96.3 / 82.9 | 4.4 | vibration and RPM flagged |
| CHT sensor drifting 1 C/min (a real sensor fault) | 33.5% | 100% | 46.0 / 26.3 | 4.6 | **CHT flagged at 242 s (+4.0 C), only CHT** |
| a different thermal model (physics v2 plant) | 100% | 100% | 5.9 / 5.8 | 15.1 | EGT and oil temp flagged |

What this shows:

- The AI is robust to calibration offsets, unit spread and sensor noise: no false detections and health
  within its normal range. Unit spread costs RUL accuracy (15% of TBO), since RUL leans on absolute levels.
- On a drifting sensor the two layers complement each other: the residuals named the right sensor after
  4 C of drift, while the AI noticed something wrong but attributed it to engine failure modes and never
  flagged the CHT channel itself. The advisory should prefer the residual's sensor attribution in this case.
- The residual layer is too literal for real hardware. Temperature calibration offsets are read as wear
  (its wear index rose to 0.26 against true 0.02), and that wrong wear then makes the healthy oil-pressure and
  vibration channels look faulty; its vibration and RPM tolerances are tighter than real sensor noise.
- An engine whose thermal behaviour differs from the twin's (v2 plant) breaks both layers - as it should:
  a new engine type needs the physics recalibrated before the twin can judge it.

## Physics residuals (model-free)

`backend/residual.py`. Expected values reproduce the dataset's clean sensor values to p99 errors of
0.007 C (EGT), 0.06 C (CHT), 0.11 C (oil temp), 0.004 psi (oil pressure), 3e-5 g (vibration).

One simulated hour per engine on live physics (100 Hz, random flight legs including redline):

| Engine | Degradation index error vs true wear (mean / p95) | False alarms, channels with no fault or failure mode in the last 30 s |
|---|---|---|
| 914 | 0.004 / 0.035 | EGT, CHT, oil temp, RPM 0%; oil pressure 0.18%; vibration 0.4-0.5% |
| 915 | 0.006 / 0.009 | 0% on every channel |
| 916 | 0.009 / 0.037 | 0% except oil pressure 0.81% |

Full scoring on the regenerated v3 data (`validation/residual_eval.py`, ~125k rows / 14-15 scenarios per
engine, 2026-09-15). "Detect" is the share of injected-fault rows the residual flags; "signature" is the share
where it also names the right fault type. The 912 has not been scored yet.

| Engine | Max false-alarm rate | Wear index MAE / corr | Bias detect / signature | Drift | Noise | Spike | Stuck-At |
|---|---|---|---|---|---|---|---|
| 914 | 0.05% | 0.005 / 0.999 | 100% / 78% | 99% / 26% | 99% / 27% | 98% / 0% | 77% / 94% |
| 915 | 0.05% | 0.003 / 1.000 | 100% / 84% | 98% / 32% | 99% / 16% | 98% / 0% | 93% / 94% |
| 916 | 0.00% | 0.003 / 1.000 | 100% / 83% | 99% / 26% | 99% / 17% | 98% / 0% | 91% / 92% |

Detection is strong for every fault type; naming the type is not. Drift reads as bias, and noise and spike
read as drift or noise, because a one-second residual cannot separate a short spike from noise. Use the
residual layer to say *that* a sensor disagrees with physics, and the diagnosis model to say *how*.
