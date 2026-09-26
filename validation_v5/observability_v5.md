# Fault observability, physics v5

Severity 0.5, observable = some instrument shifts >= 2.0 sigma of its noise; separable = differs from every family sibling by >= 2 sigma on some instrument. 18 regimes per engine (altitude (500.0, 3000.0, 6000.0) m x throttle (0.35, 0.7, 1.0) x ISA (0.0, 20.0)).

## Rotax_912_ULS

| Fault | Family | Observable | Separable in family | Strongest signs (cruise, 3000 m ISA) | Unobservable in |
|---|---|---|---|---|---|
| air_filter_fouling | induction | 12/18 | - (no sibling on this engine) | engine_rpm -2.3s, manifold_pressure_kpa -1.2s | 500m thr0.35 isa+0, 500m thr0.35 isa+20, 3000m thr0.35 isa+0, 3000m thr0.35 isa+20, 6000m thr0.35 isa+0, 6000m thr0.35 isa+20 |
| compression_loss | cylinder | 18/18 | 17/18 | engine_rpm -9.4s, oil_temp -2.3s, fuel_flow -2.1s | - |
| valve_leakage | cylinder | 18/18 | 17/18 | engine_rpm -6.8s, oil_temp -2.2s, fuel_flow -2.0s | - |
| injector_fouling | fuel_ignition | 18/18 | 18/18 | egt +6.4s, fuel_flow -4.9s | - |
| ignition_degradation | fuel_ignition | 18/18 | 18/18 | engine_rpm -10.4s, vibz +4.4s, egt +2.6s | - |
| combustion_instability | fuel_ignition | 18/18 | 18/18 | engine_rpm -20.4s, egt -10.2s, oil_temp -6.5s | - |
| bearing_wear | oil | 18/18 | 18/18 | oil_pressure -34.8s, engine_rpm -8.4s, vibx +6.5s | - |
| oil_pump_degradation | oil | 18/18 | 13/18 | oil_pressure -10.5s | - |
| oil_degradation | oil | 18/18 | 13/18 | oil_pressure -4.6s, engine_rpm -2.3s, oil_temp +1.1s | - |
| cooling_degradation | cooling | 18/18 | - (no sibling on this engine) | oil_pressure -12.3s, coolant_temp +6.9s, oil_temp +6.9s | - |
| prop_erosion | propeller | 18/18 | - (no sibling on this engine) | engine_rpm +16.2s, vibx +2.5s, viby +2.5s | - |

## Rotax_914_ULF

| Fault | Family | Observable | Separable in family | Strongest signs (cruise, 3000 m ISA) | Unobservable in |
|---|---|---|---|---|---|
| air_filter_fouling | induction | 10/18 | 18/18 | airbox_temp_c +2.2s, wastegate_position -1.9s, engine_rpm -1.3s | 500m thr0.35 isa+0, 500m thr0.35 isa+20, 500m thr0.7 isa+0, 500m thr0.7 isa+20, 3000m thr0.35 isa+0, 3000m thr0.35 isa+20, 6000m thr0.35 isa+0, 6000m thr0.35 isa+20 |
| compression_loss | cylinder | 18/18 | 18/18 | engine_rpm -10.7s, fuel_flow -3.4s, oil_pressure +2.6s | - |
| valve_leakage | cylinder | 18/18 | 18/18 | engine_rpm -7.8s, fuel_flow -3.3s, oil_pressure +3.2s | - |
| turbo_degradation | induction | 18/18 | 18/18 | airbox_temp_c +12.7s, wastegate_position -9.0s, engine_rpm -7.5s | - |
| wastegate_fault | induction | 18/18 | 18/18 | wastegate_position -18.4s | - |
| injector_fouling | fuel_ignition | 18/18 | 18/18 | fuel_flow -8.4s, egt +7.3s | - |
| ignition_degradation | fuel_ignition | 18/18 | 18/18 | engine_rpm -11.3s, vibz +4.4s, egt +2.8s | - |
| combustion_instability | fuel_ignition | 18/18 | 18/18 | engine_rpm -23.3s, egt -11.5s, oil_pressure +9.3s | - |
| bearing_wear | oil | 18/18 | 18/18 | oil_pressure -29.6s, vibx +7.5s, viby +7.5s | - |
| oil_pump_degradation | oil | 18/18 | 9/18 | oil_pressure -7.8s | - |
| oil_degradation | oil | 18/18 | 9/18 | oil_pressure -6.1s, engine_rpm -1.6s, oil_temp +1.2s | - |
| cooling_degradation | cooling | 18/18 | - (no sibling on this engine) | oil_pressure -10.0s, coolant_temp +9.2s, cht +7.2s | - |
| prop_erosion | propeller | 18/18 | - (no sibling on this engine) | engine_rpm +18.1s, vibx +2.8s, viby +2.8s | - |

## Rotax_915_iS

| Fault | Family | Observable | Separable in family | Strongest signs (cruise, 3000 m ISA) | Unobservable in |
|---|---|---|---|---|---|
| air_filter_fouling | induction | 8/18 | 18/18 | wastegate_position -1.8s | 500m thr0.35 isa+0, 500m thr0.35 isa+20, 500m thr0.7 isa+0, 500m thr0.7 isa+20, 500m thr1.0 isa+0, 500m thr1.0 isa+20, 3000m thr0.35 isa+0, 3000m thr0.35 isa+20, 3000m thr0.7 isa+0, 3000m thr0.7 isa+20 |
| compression_loss | cylinder | 18/18 | 15/18 | engine_rpm -10.0s, fuel_flow -4.0s, cht -1.5s | - |
| valve_leakage | cylinder | 18/18 | 15/18 | engine_rpm -7.4s, fuel_flow -4.0s, cht -1.9s | - |
| turbo_degradation | induction | 18/18 | 18/18 | wastegate_position -9.3s, airbox_temp_c +5.3s, engine_rpm -3.6s | - |
| wastegate_fault | induction | 18/18 | 18/18 | wastegate_position -19.1s | - |
| intercooler_fouling | induction | 18/18 | 18/18 | airbox_temp_c +14.5s, engine_rpm -9.4s, fuel_flow -5.1s | - |
| injector_fouling | fuel_ignition | 18/18 | 18/18 | fuel_flow -10.2s, egt +7.9s | - |
| ignition_degradation | fuel_ignition | 18/18 | 18/18 | engine_rpm -11.3s, vibz +4.4s, egt +3.2s | - |
| combustion_instability | fuel_ignition | 18/18 | 18/18 | engine_rpm -23.0s, egt -12.0s, oil_pressure +6.8s | - |
| bearing_wear | oil | 18/18 | 18/18 | oil_pressure -24.6s, vibx +7.7s, viby +7.7s | - |
| oil_pump_degradation | oil | 18/18 | 9/18 | oil_pressure -7.0s | - |
| oil_degradation | oil | 18/18 | 9/18 | oil_pressure -4.9s, engine_rpm -1.3s | - |
| cooling_degradation | cooling | 18/18 | - (no sibling on this engine) | coolant_temp +9.7s, cht +7.6s, oil_pressure -5.7s | - |
| prop_erosion | propeller | 18/18 | - (no sibling on this engine) | engine_rpm +18.5s, fuel_flow +3.1s, vibx +2.9s | - |

## Rotax_916_iS

| Fault | Family | Observable | Separable in family | Strongest signs (cruise, 3000 m ISA) | Unobservable in |
|---|---|---|---|---|---|
| air_filter_fouling | induction | 8/18 | 18/18 | wastegate_position -1.8s | 500m thr0.35 isa+0, 500m thr0.35 isa+20, 500m thr0.7 isa+0, 500m thr0.7 isa+20, 500m thr1.0 isa+0, 500m thr1.0 isa+20, 3000m thr0.35 isa+0, 3000m thr0.35 isa+20, 3000m thr0.7 isa+0, 3000m thr0.7 isa+20 |
| compression_loss | cylinder | 18/18 | 15/18 | engine_rpm -10.0s, fuel_flow -4.5s, cht -1.4s | - |
| valve_leakage | cylinder | 18/18 | 15/18 | engine_rpm -7.5s, fuel_flow -4.5s, cht -1.8s | - |
| turbo_degradation | induction | 18/18 | 18/18 | wastegate_position -9.4s, airbox_temp_c +5.8s, engine_rpm -3.6s | - |
| wastegate_fault | induction | 18/18 | 18/18 | wastegate_position -19.4s | - |
| intercooler_fouling | induction | 18/18 | 18/18 | airbox_temp_c +15.9s, engine_rpm -10.4s, fuel_flow -6.1s | - |
| injector_fouling | fuel_ignition | 18/18 | 18/18 | fuel_flow -11.4s, egt +7.9s | - |
| ignition_degradation | fuel_ignition | 18/18 | 18/18 | engine_rpm -11.3s, vibz +4.4s, egt +3.2s | - |
| combustion_instability | fuel_ignition | 18/18 | 18/18 | engine_rpm -22.7s, egt -12.0s, oil_pressure +7.4s | - |
| bearing_wear | oil | 18/18 | 18/18 | oil_pressure -24.3s, vibx +7.6s, viby +7.6s | - |
| oil_pump_degradation | oil | 18/18 | 9/18 | oil_pressure -7.1s | - |
| oil_degradation | oil | 17/18 | 9/18 | oil_pressure -4.8s, engine_rpm -1.3s | 3000m thr0.35 isa+0 |
| cooling_degradation | cooling | 18/18 | - (no sibling on this engine) | coolant_temp +9.4s, cht +7.4s, oil_pressure -5.3s | - |
| prop_erosion | propeller | 18/18 | - (no sibling on this engine) | engine_rpm +18.8s, fuel_flow +3.6s, vibx +2.9s | - |
