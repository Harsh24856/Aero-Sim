# Rotax 914 ULF - v5 specialists, test scorecard

900 test flights, 43,321 windows. 95% flight-bootstrap intervals in brackets.

| Head | Result | Gate | Why not |
|---|---|---|---|
| Detection: recall at precision 0.95 | 0.965 (0.952-0.977); baseline 0.639 | PASS | - |
| Diagnosis: macro F1 | 0.818 (0.771-0.846); baseline 0.000 | PASS | - |
| Severity: error on real faults | 0.073 (0.063-0.083); baseline 0.158 | PASS | - |
| Sensor fault: macro F1 | 0.901 (0.887-0.912); baseline 0.415 | PASS | - |
| Health: error | 0.032 (0.030-0.034); baseline 0.118 | PASS | - |
| RUL: error on wear-limited engines, % of TBO | 9.5% (8.4-10.6%); baseline 26.2% | PASS | - |

## Diagnosis by fault

| Fault | Recall | Precision | F1 | Cut-off |
|---|---|---|---|---|
| air_filter_fouling | 0.79 | 0.41 | 0.54 | 0.26 |
| compression_loss | 0.90 | 0.70 | 0.79 | 0.23 |
| valve_leakage | 0.56 | 0.82 | 0.67 | 0.54 |
| turbo_degradation | 0.60 | 1.00 | 0.75 | 0.85 |
| wastegate_fault | 0.91 | 0.94 | 0.93 | 0.19 |
| injector_fouling | 0.94 | 0.90 | 0.92 | 0.18 |
| ignition_degradation | 0.91 | 0.95 | 0.93 | 0.66 |
| combustion_instability | 1.00 | 0.99 | 0.99 | 0.11 |
| bearing_wear | 0.99 | 0.96 | 0.97 | 0.52 |
| oil_pump_degradation | 0.66 | 0.49 | 0.56 | 0.21 |
| oil_degradation | 0.80 | 0.59 | 0.68 | 0.25 |
| cooling_degradation | 1.00 | 0.90 | 0.95 | 0.12 |
| prop_erosion | 0.90 | 1.00 | 0.95 | 0.75 |

## Sensor faults by kind (recall)

none 1.00 | bias 0.68 | drift 0.76 | stuck 0.98 | spike 0.85 | noise 0.94 | dropout 0.98
; false alarms 0.07% of healthy channel-windows.

## Specialists vs the joint model (test flights)

Labels here are v5b (relabel_v5b.py); the joint model was scored on the v5 labels, so that column is context, not a like-for-like comparison.

| Head | Specialists | Joint model | Do-nothing | Better than joint? |
|---|---|---|---|---|
| Detection: recall at precision 0.95 | 0.965 | 0.546 | 0.639 | yes |
| Diagnosis: macro F1 | 0.818 | 0.466 | 0.000 | yes |
| Severity: error on real faults | 0.073 | 0.395 | 0.158 | yes |
| Sensor fault: macro F1 | 0.901 | 0.717 | 0.415 | yes |
| Health: error | 0.032 | 0.194 | 0.118 | yes |
| RUL: error on wear-limited engines, % of TBO | 9.483 | 10.046 | 26.171 | yes |

## Training, per specialist (validation flights)

| Specialist | Epochs run | Best epoch | Best selection score | Stages 2-3 score |
|---|---|---|---|---|
| detection | 54 | 43 | 1.8634 | 1.8642 |
| diagnosis | 51 | 40 | 1.2615 | 1.2617 |
| severity | 77 | 66 | -0.0877 | -0.0877 |
| sensor | 46 | 35 | 0.8896 | 0.8896 |
| health | 47 | 36 | -0.0350 | -0.0350 |
