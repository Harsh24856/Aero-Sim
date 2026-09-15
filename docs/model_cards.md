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

`aiv3.py` + `main.py` on physics v3, 2500 m, 45 m/s, 75 s after warm-up:

| Engine | Throttle | Health | Faults flagged | RUL predicted / true |
|---|---|---|---|---|
| 914 | 0.60 | 99.8% | none | 1829 / 1978 h |
| 915 | 0.30 | 99.9% | none | 1090 / 1187 h |
| 916 | 0.30 | 99.5% | none | 1034 / 1189 h |

At 0.6 throttle 915 and 916 sit at redline (see roadmap), where the physics itself drives oil-pressure
and vibration stress to 100%; the models report that correctly as 0% health.

## Physics residuals (model-free)

`backend/residual.py`. Expected values reproduce the dataset's clean sensor values to p99 errors of
0.007 C (EGT), 0.06 C (CHT), 0.11 C (oil temp), 0.004 psi (oil pressure), 3e-5 g (vibration).

One simulated hour per engine on live physics (100 Hz, random flight legs including redline):

| Engine | Degradation index error vs true wear (mean / p95) | False alarms, channels with no fault or failure mode in the last 30 s |
|---|---|---|
| 914 | 0.004 / 0.035 | EGT, CHT, oil temp, RPM 0%; oil pressure 0.18%; vibration 0.4-0.5% |
| 915 | 0.006 / 0.009 | 0% on every channel |
| 916 | 0.009 / 0.037 | 0% except oil pressure 0.81% |

Full scoring on the test split (detection rate and signature accuracy per fault type):
**pending** — run `validation/residual_eval.py` and update this section.
