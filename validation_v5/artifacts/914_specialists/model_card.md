# Rotax 914 ULF - v5 specialists, test scorecard

900 test flights, 43,321 windows. 95% flight-bootstrap intervals in brackets.

| Head | Result | Gate | Why not |
|---|---|---|---|
| Detection: recall at precision 0.95 | 0.912 (0.894-0.928); baseline 0.619 | PASS | - |
| Diagnosis: macro F1 | 0.601 (0.551-0.634); baseline 0.000 | FAIL | macro_f1 = 0.6006 fails >= 0.7 (CI 0.5509781311366893, 0.6338523395588711); min_fault_recall = 0.3205 fails >= 0.5 (CI None, None); family_macro_f1 = 0.6671 fails >= 0.85 (CI None, None); ece = 0.08828 fails <= 0.05 (CI None, None) |
| Severity: error on real faults | 0.350 (0.320-0.380); baseline 0.199 | FAIL | mae_on_fault = 0.3503 fails <= 0.12 (CI 0.3198095351457596, 0.38006222173571585); mae_on_fault: CI of difference from baseline [0.11918556354939938, 0.1843699723482132] does not exclude 0 in the right direction (baseline 0.1993577927350998); mae_clean = 0.1597 fails <= 0.02 (CI None, None); abs:bias_on_fault = 0.307 fails <= 0.03 (CI None, None) |
| Sensor fault: macro F1 | 0.778 (0.760-0.793); baseline 0.415 | FAIL | per_kind.bias.recall = 0.3348 fails >= 0.5 (CI None, None); per_kind.drift.recall = 0.3334 fails >= 0.4 (CI None, None) |
| Health: error | 0.038 (0.035-0.040); baseline 0.118 | PASS | - |
| RUL: error on wear-limited engines, % of TBO | 11.2% (10.0-12.4%); baseline 26.2% | FAIL | mae_pct_tbo_tbo_limited = 5.338 fails <= 4 (CI None, None) |

## Diagnosis by fault

| Fault | Recall | Precision | F1 | Cut-off |
|---|---|---|---|---|
| air_filter_fouling | 0.32 | 0.09 | 0.14 | 0.98 |
| compression_loss | 0.58 | 0.44 | 0.50 | 0.94 |
| valve_leakage | 0.46 | 0.55 | 0.50 | 0.98 |
| turbo_degradation | 0.40 | 0.99 | 0.57 | 0.53 |
| wastegate_fault | 0.86 | 0.64 | 0.73 | 0.99 |
| injector_fouling | 0.90 | 0.67 | 0.77 | 0.99 |
| ignition_degradation | 0.84 | 0.55 | 0.67 | 0.99 |
| combustion_instability | 0.96 | 1.00 | 0.98 | 0.99 |
| bearing_wear | 0.81 | 0.80 | 0.81 | 0.95 |
| oil_pump_degradation | 0.36 | 0.19 | 0.25 | 0.90 |
| oil_degradation | 0.56 | 0.48 | 0.52 | 0.78 |
| cooling_degradation | 1.00 | 0.92 | 0.96 | 0.78 |
| prop_erosion | 0.86 | 0.28 | 0.42 | 0.99 |

## Sensor faults by kind (recall)

none 1.00 | bias 0.33 | drift 0.33 | stuck 0.96 | spike 0.71 | noise 0.95 | dropout 0.98
; false alarms 0.26% of healthy channel-windows.

## Specialists vs the joint model (test flights)

| Head | Specialists | Joint model | Do-nothing | Better than joint? |
|---|---|---|---|---|
| Detection: recall at precision 0.95 | 0.912 | 0.546 | 0.619 | yes |
| Diagnosis: macro F1 | 0.601 | 0.466 | 0.000 | yes |
| Severity: error on real faults | 0.350 | 0.395 | 0.199 | yes |
| Sensor fault: macro F1 | 0.778 | 0.717 | 0.415 | yes |
| Health: error | 0.038 | 0.194 | 0.118 | yes |
| RUL: error on wear-limited engines, % of TBO | 11.189 | 10.046 | 26.171 | no |

## Training, per specialist (validation flights)

| Specialist | Epochs run | Best epoch | Best selection score | Stages 2-3 score |
|---|---|---|---|---|
| detection | 35 | 24 | 1.7540 | 1.7540 |
| diagnosis | 35 | 24 | 0.8078 | 0.8078 |
| severity | 14 | 3 | -0.4131 | -0.4131 |
| sensor | 36 | 25 | 0.7895 | 0.7895 |
| health | 59 | 48 | -0.1090 | -0.1051 |
