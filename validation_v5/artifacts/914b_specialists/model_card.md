# Rotax 914 ULF - v5 specialists, test scorecard

900 test flights, 43,321 windows. 95% flight-bootstrap intervals in brackets.

| Head | Result | Gate | Why not |
|---|---|---|---|
| Detection: recall at precision 0.95 | 0.957 (0.942-0.971); baseline 0.641 | PASS | - |
| Diagnosis: macro F1 | 0.800 (0.751-0.828); baseline 0.000 | FAIL | min_fault_recall = 0.4721 fails >= 0.5 (CI None, None) |
| Severity: error on real faults | 0.078 (0.069-0.087); baseline 0.158 | PASS | - |
| Sensor fault: macro F1 | 0.889 (0.871-0.903); baseline 0.415 | PASS | - |
| Health: error | 0.033 (0.031-0.036); baseline 0.118 | PASS | - |
| RUL: error on wear-limited engines, % of TBO | 10.9% (9.7-12.1%); baseline 26.2% | PASS | - |

## Diagnosis by fault

| Fault | Recall | Precision | F1 | Cut-off |
|---|---|---|---|---|
| air_filter_fouling | 0.90 | 0.40 | 0.56 | 0.15 |
| compression_loss | 0.83 | 0.86 | 0.84 | 0.33 |
| valve_leakage | 0.47 | 0.80 | 0.59 | 0.57 |
| turbo_degradation | 0.58 | 0.92 | 0.71 | 0.67 |
| wastegate_fault | 0.93 | 0.97 | 0.95 | 0.41 |
| injector_fouling | 0.92 | 0.81 | 0.87 | 0.10 |
| ignition_degradation | 0.91 | 0.88 | 0.89 | 0.57 |
| combustion_instability | 0.98 | 0.99 | 0.99 | 0.20 |
| bearing_wear | 0.96 | 0.97 | 0.96 | 0.39 |
| oil_pump_degradation | 0.61 | 0.45 | 0.52 | 0.28 |
| oil_degradation | 0.76 | 0.47 | 0.58 | 0.15 |
| cooling_degradation | 1.00 | 0.92 | 0.96 | 0.13 |
| prop_erosion | 0.96 | 0.98 | 0.97 | 0.58 |

## Sensor faults by kind (recall)

none 1.00 | bias 0.68 | drift 0.72 | stuck 0.98 | spike 0.86 | noise 0.96 | dropout 0.98
; false alarms 0.10% of healthy channel-windows.

## Specialists vs the joint model (test flights)

Labels here are v5b (relabel_v5b.py); the joint model was scored on the v5 labels, so that column is context, not a like-for-like comparison.

| Head | Specialists | Joint model | Do-nothing | Better than joint? |
|---|---|---|---|---|
| Detection: recall at precision 0.95 | 0.957 | 0.546 | 0.641 | yes |
| Diagnosis: macro F1 | 0.800 | 0.466 | 0.000 | yes |
| Severity: error on real faults | 0.078 | 0.395 | 0.158 | yes |
| Sensor fault: macro F1 | 0.889 | 0.717 | 0.415 | yes |
| Health: error | 0.033 | 0.194 | 0.118 | yes |
| RUL: error on wear-limited engines, % of TBO | 10.881 | 10.046 | 26.171 | no |

## Training, per specialist (validation flights)

| Specialist | Epochs run | Best epoch | Best selection score | Stages 2-3 score |
|---|---|---|---|---|
| detection | 53 | 42 | 1.8329 | 1.8355 |
| diagnosis | 55 | 44 | 1.2333 | 1.2347 |
| severity | 46 | 35 | -0.1040 | -0.1030 |
| sensor | 51 | 40 | 0.8849 | 0.8849 |
| health | 39 | 28 | -0.0379 | -0.0379 |
