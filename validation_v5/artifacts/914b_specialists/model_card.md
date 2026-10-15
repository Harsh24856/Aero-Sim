# Rotax 914 ULF - v5 specialists, test scorecard

900 test flights, 43,321 windows. 95% flight-bootstrap intervals in brackets.

| Head | Result | Gate | Why not |
|---|---|---|---|
| Detection: recall at precision 0.95 | 0.978 (0.968-0.985); baseline 0.733 | PASS | - |
| Diagnosis: macro F1 | 0.842 (0.798-0.870); baseline 0.000 | PASS | - |
| Severity: error on real faults | 0.073 (0.063-0.084); baseline 0.158 | PASS | - |
| Sensor fault: macro F1 | 0.898 (0.884-0.909); baseline 0.415 | PASS | - |
| Health: error | 0.029 (0.027-0.031); baseline 0.118 | PASS | - |
| RUL: error on wear-limited engines, % of TBO | 9.6% (8.6-10.7%); baseline 26.2% | PASS | - |

## Diagnosis by fault

| Fault | Recall | Precision | F1 | Cut-off |
|---|---|---|---|---|
| air_filter_fouling | 1.00 | 0.40 | 0.57 | 0.11 |
| compression_loss | 0.86 | 0.77 | 0.81 | 0.28 |
| valve_leakage | 0.60 | 0.87 | 0.71 | 0.74 |
| turbo_degradation | 0.67 | 1.00 | 0.80 | 0.72 |
| wastegate_fault | 0.95 | 0.87 | 0.91 | 0.04 |
| injector_fouling | 0.96 | 0.86 | 0.91 | 0.08 |
| ignition_degradation | 0.94 | 0.91 | 0.92 | 0.40 |
| combustion_instability | 1.00 | 0.93 | 0.96 | 0.04 |
| bearing_wear | 1.00 | 0.93 | 0.96 | 0.14 |
| oil_pump_degradation | 0.72 | 0.64 | 0.68 | 0.24 |
| oil_degradation | 0.82 | 0.78 | 0.80 | 0.39 |
| cooling_degradation | 1.00 | 0.90 | 0.95 | 0.14 |
| prop_erosion | 0.93 | 1.00 | 0.96 | 0.57 |

## Sensor faults by kind (recall)

none 1.00 | bias 0.63 | drift 0.74 | stuck 0.99 | spike 0.85 | noise 0.95 | dropout 0.98
; false alarms 0.07% of healthy channel-windows.

## Specialists vs the joint model (test flights)

Labels here are v5b (relabel_v5b.py); the joint model was scored on the v5 labels, so that column is context, not a like-for-like comparison.

| Head | Specialists | Joint model | Do-nothing | Better than joint? |
|---|---|---|---|---|
| Detection: recall at precision 0.95 | 0.978 | 0.546 | 0.733 | yes |
| Diagnosis: macro F1 | 0.842 | 0.466 | 0.000 | yes |
| Severity: error on real faults | 0.073 | 0.395 | 0.158 | yes |
| Sensor fault: macro F1 | 0.898 | 0.717 | 0.415 | yes |
| Health: error | 0.029 | 0.194 | 0.118 | yes |
| RUL: error on wear-limited engines, % of TBO | 9.615 | 10.046 | 26.171 | yes |

## Training, per specialist (validation flights)

| Specialist | Epochs run | Best epoch | Best selection score | Stages 2-3 score |
|---|---|---|---|---|
| detection | 39 | 28 | 1.9013 | 1.9021 |
| diagnosis | 55 | 44 | 1.3110 | 1.3123 |
| severity | 58 | 47 | -0.0819 | -0.0810 |
| sensor | 67 | 56 | 0.8970 | 0.8970 |
| health | 55 | 44 | -0.0362 | -0.0362 |
