# Model cards: physics v4

The v3 cards are in `docs/model_cards.md`. Physics v3 and its models stay selectable, and old v3 runs still render with them.

Every number below was measured on **test flights the models never saw during training or model selection**, unless it says otherwise. The source of truth for each engine is `backend/models_v4/<engine>/manifest.json`, written by `validation_v4/export_deployable_v4.py`.

## Shared facts

| | |
|---|---|
| Physics | `backend/physics_v4.py`: layered closed-loop Rotax model, checked against Rotax figures; the MATLAB/Simulink 914 twin matches it to about 2e-15 |
| Data | About 10M rows per engine at 1 Hz, about 4,000 simulated flights, split 80/10/10 **by flight**, so no flight is in two splits |
| Inputs | 29 per second: 8 flight/atmosphere, 12 instruments, 3 derived, and 6 **residuals against a healthy on-board twin** flown on the same inputs |
| Window | 128 seconds (128 samples at 1 Hz) |
| Model | One dilated-TCN encoder with six heads. Each head is taken from the training step it was trained for: detection, diagnosis, severity, sensor fault, health, RUL |
| Faults | 13 component faults on the 914 (11 on the 912, 14 on the 915 and 916), and 6 sensor-fault kinds on each of 12 instruments |
| Device | **These weights give correct answers only on the Metal GPU they were trained on.** On the CPU the same windows give very different outputs (diagnosis probabilities differ by up to 1.0). All metrics below were measured on the GPU; `aiv4.py` reports `backend_validated` only when it serves there. |

## Rotax 914 ULF: deployed (exported 2026-09-24)

| Question | Model | Result | Gate |
|---|---|---|---|
| Is anything wrong? | detection | AUC **0.879** overall; **0.952** where the engine has clearly drifted from its twin (top quarter by residual) | ≥ 0.84 / ≥ 0.90 |
| Which part? | diagnosis | macro F1 **0.612** with per-fault cut-offs (0.571 at a flat 0.5) | ≥ 0.35 |
| How bad? | severity | error **0.216** on the 0–1 scale where a fault exists (0.048 over all cells) | ≤ 0.12 on all cells |
| Engine or sensor? | sensor fault | macro F1 **0.417** over the 7 conditions (0.275 before the window statistics were added); accuracy 0.978 | accuracy ≥ 0.90 |
| How worn? | health | wear-condition error **0.063** (about 6 points out of 100) | ≤ 0.10 |
| Hours left? | RUL | **4.6% of TBO** overall. On engines that wear out before their overhaul: **387 h**, where counting down to the overhaul date gives 462 h | ≤ 8% / ≤ 15% (see below) |

### Which part? Per fault type, with its cut-off

Cut-offs were chosen on the validation flights and reported on the test flights. Recall is the share of real cases named; precision is the share of calls that were right.

| Fault | Cut-off | F1 | Recall | Precision |
|---|---|---|---|---|
| Injector (fuel metering) fouling | 0.20 | 0.96 | 0.93 | 1.00 |
| Bearing wear | 0.60 | 0.89 | 0.88 | 0.91 |
| Cooling degradation | 0.90 | 0.87 | 0.81 | 0.93 |
| Prop erosion | 0.95 | 0.82 | 1.00 | 0.70 |
| Turbo degradation | 0.95 | 0.77 | 0.66 | 0.92 |
| Combustion instability | 0.90 | 0.75 | 0.88 | 0.65 |
| Compression loss | 0.95 | 0.73 | 0.71 | 0.75 |
| Oil pump degradation | 0.95 | 0.72 | 0.66 | 0.80 |
| Valve leakage | 0.95 | 0.53 | 0.67 | 0.44 |
| Oil degradation | 0.70 | 0.43 | 0.62 | 0.33 |
| Ignition degradation | 0.85 | 0.39 | 0.71 | 0.27 |
| Air filter fouling | 0.55 | 0.10 | 0.21 | 0.07 |
| Wastegate fault | 0.85 | 0.00 | 0.00 | 0.00 |

- **The wastegate isn't a model failure.** Below the turbo's critical altitude (15,000 ft), the wastegate closes further and holds boost, so the fault leaves no trace. A real 914 behaves the same way. The cockpit marks turbo and wastegate faults "observable above 15,000 ft".
- **A mildly fouled air filter** barely changes any reading at cruise power.

### Engine or sensor? Per condition (test flights)

| Condition | Named correctly |
|---|---|
| Dropout | 81% |
| Stuck | 37% |
| Noise | 35% |
| Bias | 20% |
| Spike | 15% |
| Drift | 1% |

- **Overall:** it spots that a sensor is faulty at all 47% of the time, names the kind 33% of the time, and calls a healthy sensor faulty only 0.9% of the time.
- **Drift** grows slowly, so over a 128-second window it looks the same as a bias. Catching it needs a longer view, for example a running mean of each residual. That's recorded as future work.

### Hours left? On inputs an aircraft actually has

The RUL head reads ten scalar inputs. As first trained, three of them were **simulator labels** that no aircraft can measure:
- the true wear condition;
- the true fault severities;
- the operating margin, computed from true rather than measured values.

On those labels it scored 13.1% of TBO on wear-limited engines. Live, the inputs come from the health model, the severity model and the sensors. Scored that way, the same head gave 20.6%. It was then fine-tuned on the live inputs (`retrain_rul_live_v4.py`), which is the version deployed:

| Engines that wear out before overhaul (22% of test windows) | Error |
|---|---|
| Counting down to the overhaul date (no model) | 23.1% of TBO (462 h) |
| RUL head as trained, fed live inputs | 20.6% |
| **RUL head fine-tuned on live inputs (deployed)** | **19.4% (387 h)** |
| RUL head fed the simulator's own labels (not possible live) | 13.1% |

- **Against the calendar:** the deployed head beats the calendar countdown but misses the 15% bar. That bar is met only with labels an aircraft can't have. This is the headline limitation of the v4 RUL model, and it is reported as such.
- **Uncertainty band** by life stage (the ± shown in the cockpit):

| Life stage (engine hours ÷ TBO) | 0–25% | 25–50% | 50–75% | 75–100% |
|---|---|---|---|---|
| Mean error | 161 h | 123 h | 81 h | 6 h |

## Rotax 912 ULS, 915 iS, 916 iS

These are training on the v4 data (`validation_v4/logs/912_915_916_train.log`). Until each is exported, `aiv4.py` serves it with the 914's models, flagged `placeholder_models`, and the cockpit says so.

The 912 so far, on test flights:
- **Diagnosis:** macro F1 0.659.
- **Severity:** error 0.156 where a fault exists.
- **Sensor condition:** macro F1 0.410.

This section is completed as each engine passes its gates and is exported.

## Live end-to-end on the 914 (`scripts/e2e_v4.py`, 2026-09-25)

Each preset was flown through the same code the live stack runs: twin, sensors, `aiv4` window, models, cut-offs, live RUL inputs and advisory. The flight was a 300 s climb, then cruise with a power change every 2 minutes, scored every 5 s once the 128 s window was full.

| Preset | Result |
|---|---|
| Healthy, 300 h | **Pass.** 0% of samples called any engine fault; RUL 1,681 h = truth = calendar |
| Bearing wear, 1,450 h | **Pass.** Named from the first scored window |
| Oil pump degradation | **Pass.** Named from the first scored window |
| Cooling degradation, hot day | **Pass.** Named from the first scored window |
| Prop erosion | **Pass.** Named from the first scored window |
| Sensor: CHT dropout | **Pass.** Recognised on 56 of 57 samples; no engine fault called |
| Turbo degradation, 6,000 m | **Fail.** Turbo probability median 0.74 (max 0.93), under its 0.95 cut-off; valve leakage (51/72) and air filter (34/72) named instead |
| Sensor: EGT stuck | **Fail.** Not recognised (0 of 57); no engine fault called, but RUL read 1,088 h against a true 1,382 h because the frozen reading feeds the RUL inputs |

Through the running stack (`aiv4` + `main.py` over HTTP and WebSocket), a bearing-wear flight gave these results:
- **Bearing wear** named on every sample.
- **Wear condition** 0.420, against a true 0.434.
- **Oil degradation** also called on 29 of 30 samples, which is a false co-call. Its probability was 0.72 against a 0.70 cut-off, while the severity model put it at only 0.07. Requiring the severity model to agree (at 0.08 or more, the labels' own definition of "present") was tested and **rejected on validation**: macro F1 0.652 with the rule, 0.664 without. So it isn't deployed; `export_deployable_v4.py` repeats that test for every engine.
- **A CHT dropout** injected during the same flight also made the sensor model flag oil temperature and fuel flow. A −40 °C reading disturbs the whole window. This is a known limitation of the sensor model.

On the preset engines, RUL matches the truth to within the model's band, because they reach their overhaul before they wear out. The wear-limited error in the table above is the harder, general case.

## Live stack check on the 914 (2026-09-25, `aiv4` + `main.py`, HTTP API)

Each preset was flown through the running services at a steady operating point: 1,500 m at 80% throttle (turbo: 6,000 m at 90%). The AI's answers were compared with the twin's injected truth every 3 s, from the moment the 128 s window was full.

| Preset | Part named (share of samples) | False calls | Wear: AI vs true | RUL: AI vs true |
|---|---|---|---|---|
| Healthy | none (0% fault flagged) | none | 0.86 vs 0.91 | 1,691 vs 1,691 h |
| Bearing wear | bearing wear 38/38 | oil degradation 38/38 | 0.32 vs 0.19 | 542 vs 542 h |
| Oil pump | oil pump 38/38 | oil degradation 18/38 | 0.31 vs 0.36 | 1,092 vs 1,092 h |
| Cooling, hot day | cooling 38/38 | none | 0.45 vs 0.54 | 831 vs 891 h |
| Prop erosion | prop erosion 37/37 | none | 0.52 vs 0.63 | 691 vs 691 h |
| Turbo, 6,000 m | turbo 38/38 | valve leakage 38/38; sensor "CHT drift" 38/38 | 0.40 vs 0.45 | 792 vs 792 h |
| Sensor: CHT dropout | CHT dropout 43/44 after onset; no engine fault | sensor "oil temp noise" 26/44 | 0.83 vs 0.81 | 1,388 vs 1,388 h |
| Sensor: EGT stuck | **not recognised** (0/67) | none | 0.73 vs 0.77 | 1,388 vs 1,388 h |

- **What works:** every injected engine fault was named on every sample, and the healthy engine raised nothing. Wear condition was within 0.02–0.13 of the truth. RUL matched the truth on 7 of 8, and was within its ±81 h band on the eighth.
- **Turbo:** named here but not in the headless run. Here the whole window is above the critical altitude; the headless run climbed gradually and was scored partly below it.
- **What's weak:**
  - false co-calls within a fault family (oil degradation beside bearing wear or oil pump; valve leakage beside turbo);
  - false sensor calls next to a real disturbance (CHT drift at altitude, oil temp noise next to a CHT dropout);
  - stuck sensors.

## Checks behind these numbers

- **Parity.** Real test flights fed row by row through the live `aiv4` window reproduce the training pipeline exactly, with zero difference in the window, the aux inputs and all five window models (`validation_v4/parity_ai_v4.py`, 95 windows).
- **Contract.** The live twin's 29 inputs, fault names and sensor names match every manifest (`backend/tests/test_ai_v4_contract.py`).
- **Time scale.** A 30-minute live flight keeps all 29 inputs within the training range (`backend/tests/test_timescale_v4.py`).
- **100 Hz against 1 Hz.** Sampling the 100 Hz live engine once a second shifts no input's mean by more than 0.002 scaler standard deviations.
