# Rotax 916 iS - v5 specialists, test scorecard

915 test flights, 44,183 windows. 95% flight-bootstrap intervals in brackets.

| Head | Result | Gate | Why not |
|---|---|---|---|
| Detection: recall at precision 0.95 | 0.971 (0.960-0.980); baseline 0.646 | PASS | - |
| Diagnosis: macro F1 | 0.803 (0.753-0.833); baseline 0.000 | FAIL | min_fault_recall = 0.3261 fails >= 0.5 (CI None, None) |
| Severity: error on real faults | 0.077 (0.065-0.091); baseline 0.169 | FAIL | abs:bias_on_fault = 0.04524 fails <= 0.03 (CI None, None) |
| Sensor fault: macro F1 | 0.892 (0.878-0.905); baseline 0.433 | PASS | - |
| Health: error | 0.027 (0.025-0.029); baseline 0.114 | PASS | - |
| RUL: error on wear-limited engines, % of TBO | 11.1% (9.7-12.6%); baseline 28.6% | PASS | - |

## Diagnosis by fault

| Fault | Recall | Precision | F1 | Cut-off |
|---|---|---|---|---|
| air_filter_fouling | 0.33 | 0.56 | 0.41 | 0.59 |
| compression_loss | 0.87 | 0.95 | 0.91 | 0.43 |
| valve_leakage | 0.72 | 0.65 | 0.68 | 0.25 |
| turbo_degradation | 0.83 | 0.96 | 0.89 | 0.45 |
| wastegate_fault | 0.95 | 0.92 | 0.93 | 0.12 |
| intercooler_fouling | 0.89 | 0.94 | 0.91 | 0.11 |
| injector_fouling | 1.00 | 0.85 | 0.92 | 0.11 |
| ignition_degradation | 0.91 | 0.88 | 0.89 | 0.39 |
| combustion_instability | 0.97 | 0.99 | 0.98 | 0.64 |
| bearing_wear | 0.91 | 1.00 | 0.95 | 0.99 |
| oil_pump_degradation | 0.83 | 0.31 | 0.45 | 0.12 |
| oil_degradation | 0.54 | 0.54 | 0.54 | 0.34 |
| cooling_degradation | 0.93 | 0.92 | 0.93 | 0.07 |
| prop_erosion | 0.92 | 0.78 | 0.85 | 0.07 |

## Sensor faults by kind (recall)

none 1.00 | bias 0.70 | drift 0.67 | stuck 0.99 | spike 0.91 | noise 0.93 | dropout 0.97
; false alarms 0.09% of healthy channel-windows.

## Specialists vs the joint model (test flights)

Labels here are v5b (relabel_v5b.py); the joint model was scored on the v5 labels, so that column is context, not a like-for-like comparison.

| Head | Specialists | Joint model | Do-nothing | Better than joint? |
|---|---|---|---|---|
| Detection: recall at precision 0.95 | 0.971 | - | 0.646 | - |
| Diagnosis: macro F1 | 0.803 | - | 0.000 | - |
| Severity: error on real faults | 0.077 | - | 0.169 | - |
| Sensor fault: macro F1 | 0.892 | - | 0.433 | - |
| Health: error | 0.027 | - | 0.114 | - |
| RUL: error on wear-limited engines, % of TBO | 11.138 | - | 28.627 | - |

## Training, per specialist (validation flights)

| Specialist | Epochs run | Best epoch | Best selection score | Stages 2-3 score |
|---|---|---|---|---|
| detection | 38 | 27 | 1.8765 | 1.8765 |
| diagnosis | 55 | 44 | 1.3019 | 1.3039 |
| severity | 37 | 26 | -0.0953 | -0.0940 |
| sensor | 50 | 39 | 0.9070 | 0.9088 |
| health | 47 | 36 | -0.0354 | -0.0354 |
