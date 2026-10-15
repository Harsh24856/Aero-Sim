# Rotax 912 ULS - v5 specialists, test scorecard

902 test flights, 43,359 windows. 95% flight-bootstrap intervals in brackets.

| Head | Result | Gate | Why not |
|---|---|---|---|
| Detection: recall at precision 0.95 | 0.967 (0.956-0.977); baseline 0.714 | PASS | - |
| Diagnosis: macro F1 | 0.834 (0.789-0.870); baseline 0.000 | PASS | - |
| Severity: error on real faults | 0.082 (0.072-0.091); baseline 0.167 | FAIL | abs:bias_on_fault = 0.05696 fails <= 0.03 (CI None, None) |
| Sensor fault: macro F1 | 0.901 (0.888-0.914); baseline 0.397 | PASS | - |
| Health: error | 0.028 (0.026-0.030); baseline 0.117 | PASS | - |
| RUL: error on wear-limited engines, % of TBO | 10.9% (9.6-12.4%); baseline 25.7% | PASS | - |

## Diagnosis by fault

| Fault | Recall | Precision | F1 | Cut-off |
|---|---|---|---|---|
| air_filter_fouling | 0.71 | 0.68 | 0.69 | 0.42 |
| compression_loss | 0.79 | 0.79 | 0.79 | 0.23 |
| valve_leakage | 0.72 | 0.93 | 0.82 | 0.63 |
| injector_fouling | 0.77 | 0.86 | 0.81 | 0.30 |
| ignition_degradation | 0.82 | 0.97 | 0.89 | 0.14 |
| combustion_instability | 1.00 | 0.97 | 0.98 | 0.11 |
| bearing_wear | 0.85 | 0.98 | 0.91 | 0.26 |
| oil_pump_degradation | 0.73 | 0.95 | 0.83 | 0.59 |
| oil_degradation | 0.78 | 0.50 | 0.61 | 0.18 |
| cooling_degradation | 0.92 | 0.95 | 0.94 | 0.09 |
| prop_erosion | 0.98 | 0.87 | 0.92 | 0.13 |

## Sensor faults by kind (recall)

none 1.00 | bias 0.65 | drift 0.73 | stuck 0.98 | spike 0.86 | noise 0.93 | dropout 0.98
; false alarms 0.03% of healthy channel-windows.

## Specialists vs the joint model (test flights)

Labels here are v5b (relabel_v5b.py); the joint model was scored on the v5 labels, so that column is context, not a like-for-like comparison.

| Head | Specialists | Joint model | Do-nothing | Better than joint? |
|---|---|---|---|---|
| Detection: recall at precision 0.95 | 0.967 | - | 0.714 | - |
| Diagnosis: macro F1 | 0.834 | - | 0.000 | - |
| Severity: error on real faults | 0.082 | - | 0.167 | - |
| Sensor fault: macro F1 | 0.901 | - | 0.397 | - |
| Health: error | 0.028 | - | 0.117 | - |
| RUL: error on wear-limited engines, % of TBO | 10.949 | - | 25.668 | - |

## Training, per specialist (validation flights)

| Specialist | Epochs run | Best epoch | Best selection score | Stages 2-3 score |
|---|---|---|---|---|
| detection | 72 | 61 | 1.9014 | 1.9014 |
| diagnosis | 58 | 47 | 1.2665 | 1.2665 |
| severity | 37 | 26 | -0.0804 | -0.0804 |
| sensor | 52 | 41 | 0.9127 | 0.9144 |
| health | 40 | 29 | -0.0361 | -0.0361 |
