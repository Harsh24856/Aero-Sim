# Rotax 914 ULF - v5 test scorecard

900 test flights, 43,321 windows. 95% flight-bootstrap intervals in brackets.

| Head | Result | Gate | Why not |
|---|---|---|---|
| Detection: recall at precision 0.95 | 0.546 (0.489-0.598); baseline 0.619 | FAIL | recall_at_p95 = 0.5459 fails >= 0.85 (CI 0.4887077663478219, 0.5977650441203184); recall_at_p95: CI of difference from baseline [-0.12958286670492677, -0.023378661028147438] does not exclude 0 in the right direction (baseline 0.6185798484423238); auc = 0.8622 fails >= 0.95 (CI None, None) |
| Diagnosis: macro F1 | 0.466 (0.426-0.492); baseline 0.000 | FAIL | macro_f1 = 0.4661 fails >= 0.7 (CI 0.42561465756701683, 0.4923134251084599); min_fault_recall = 0.4074 fails >= 0.5 (CI None, None); family_macro_f1 = 0.5485 fails >= 0.85 (CI None, None); ece = 0.1584 fails <= 0.05 (CI None, None) |
| Severity: error on real faults | 0.395 (0.367-0.422); baseline 0.199 | FAIL | mae_on_fault = 0.395 fails <= 0.12 (CI 0.36694023832678796, 0.4221504911780357); mae_on_fault: CI of difference from baseline [0.1632530178874731, 0.22733568772673607] does not exclude 0 in the right direction (baseline 0.1993577927350998); mae_clean = 0.05442 fails <= 0.02 (CI None, None); abs:bias_on_fault = 0.03077 fails <= 0.03 (CI None, None) |
| Sensor fault: macro F1 | 0.717 (0.700-0.732); baseline 0.415 | FAIL | per_kind.drift.recall = 0.2146 fails >= 0.4 (CI None, None); false_alarm_rate = 0.02134 fails <= 0.005 (CI None, None) |
| Health: error | 0.194 (0.187-0.200); baseline 0.118 | FAIL | mae = 0.1938 fails <= 0.05 (CI 0.18712186340252107, 0.20038235594885492); mae: CI of difference from baseline [0.06657740987430771, 0.08461680833590655] does not exclude 0 in the right direction (baseline 0.1183897194372101) |
| RUL: error on wear-limited engines, % of TBO | 10.0% (9.1-11.1%); baseline 26.2% | FAIL | mae_pct_tbo_tbo_limited = 7.165 fails <= 4 (CI None, None) |

## Diagnosis by fault

| Fault | Recall | Precision | F1 | Cut-off |
|---|---|---|---|---|
| air_filter_fouling | 0.80 | 0.06 | 0.12 | 0.94 |
| compression_loss | 0.67 | 0.44 | 0.53 | 0.99 |
| valve_leakage | 0.61 | 0.29 | 0.40 | 0.99 |
| turbo_degradation | 0.59 | 1.00 | 0.74 | 0.58 |
| wastegate_fault | 0.85 | 0.17 | 0.28 | 0.99 |
| injector_fouling | 0.85 | 0.46 | 0.60 | 0.98 |
| ignition_degradation | 0.77 | 0.54 | 0.63 | 0.99 |
| combustion_instability | 0.98 | 0.53 | 0.69 | 0.99 |
| bearing_wear | 0.81 | 0.40 | 0.53 | 0.97 |
| oil_pump_degradation | 0.41 | 0.20 | 0.26 | 0.94 |
| oil_degradation | 0.50 | 0.18 | 0.26 | 0.98 |
| cooling_degradation | 1.00 | 0.37 | 0.54 | 0.99 |
| prop_erosion | 0.92 | 0.32 | 0.47 | 0.99 |

## Sensor faults by kind (recall)

none 0.98 | bias 0.50 | drift 0.21 | stuck 0.93 | spike 0.69 | noise 0.90 | dropout 0.94
; false alarms 2.13% of healthy channel-windows.