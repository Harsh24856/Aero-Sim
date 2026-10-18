# Rotax 915 iS - v5 specialists, test scorecard

889 test flights, 43,069 windows. 95% flight-bootstrap intervals in brackets.

| Head | Result | Gate | Why not |
|---|---|---|---|
| Detection: recall at precision 0.95 | 0.975 (0.965-0.983); baseline 0.689 | PASS | - |
| Diagnosis: macro F1 | 0.808 (0.755-0.840); baseline 0.000 | PASS | - |
| Severity: error on real faults | 0.090 (0.076-0.107); baseline 0.166 | FAIL | abs:bias_on_fault = 0.0339 fails <= 0.03 (CI None, None) |
| Sensor fault: macro F1 | 0.900 (0.887-0.912); baseline 0.420 | PASS | - |
| Health: error | 0.030 (0.028-0.032); baseline 0.127 | PASS | - |
| RUL: error on wear-limited engines, % of TBO | 10.6% (9.2-12.0%); baseline 26.7% | PASS | - |

## Diagnosis by fault

| Fault | Recall | Precision | F1 | Cut-off |
|---|---|---|---|---|
| air_filter_fouling | 0.73 | 0.40 | 0.52 | 0.13 |
| compression_loss | 0.83 | 0.72 | 0.77 | 0.34 |
| valve_leakage | 0.59 | 0.77 | 0.67 | 0.19 |
| turbo_degradation | 0.88 | 0.75 | 0.81 | 0.20 |
| wastegate_fault | 0.82 | 0.97 | 0.89 | 0.98 |
| intercooler_fouling | 0.94 | 1.00 | 0.97 | 0.43 |
| injector_fouling | 0.96 | 0.93 | 0.95 | 0.11 |
| ignition_degradation | 0.62 | 0.89 | 0.73 | 0.45 |
| combustion_instability | 0.98 | 0.90 | 0.94 | 0.18 |
| bearing_wear | 1.00 | 0.95 | 0.97 | 0.41 |
| oil_pump_degradation | 0.55 | 0.84 | 0.66 | 0.42 |
| oil_degradation | 0.59 | 0.73 | 0.65 | 0.64 |
| cooling_degradation | 0.89 | 0.91 | 0.90 | 0.34 |
| prop_erosion | 0.95 | 0.83 | 0.89 | 0.05 |

## Sensor faults by kind (recall)

none 1.00 | bias 0.68 | drift 0.77 | stuck 0.98 | spike 0.86 | noise 0.96 | dropout 0.96
; false alarms 0.11% of healthy channel-windows.

## Specialists vs the joint model (test flights)

Labels here are v5b (relabel_v5b.py); the joint model was scored on the v5 labels, so that column is context, not a like-for-like comparison.

| Head | Specialists | Joint model | Do-nothing | Better than joint? |
|---|---|---|---|---|
| Detection: recall at precision 0.95 | 0.975 | - | 0.689 | - |
| Diagnosis: macro F1 | 0.808 | - | 0.000 | - |
| Severity: error on real faults | 0.090 | - | 0.166 | - |
| Sensor fault: macro F1 | 0.900 | - | 0.420 | - |
| Health: error | 0.030 | - | 0.127 | - |
| RUL: error on wear-limited engines, % of TBO | 10.570 | - | 26.728 | - |

## Training, per specialist (validation flights)

| Specialist | Epochs run | Best epoch | Best selection score | Stages 2-3 score |
|---|---|---|---|---|
| detection | 53 | 42 | 1.9054 | 1.9054 |
| diagnosis | 71 | 60 | 1.3491 | 1.3491 |
| severity | 44 | 33 | -0.0937 | -0.0937 |
| sensor | 46 | 35 | 0.8864 | 0.8864 |
| health | 46 | 35 | -0.0373 | -0.0373 |
