"""
Unified database generation for Detection + Diagnosis + RUL, all from one simulation.
Each scenario is a full engine "life": periodic randomized flight legs (not one static
random walk) with a randomized severity tendency, running until real wear-based failure
or a max-duration censoring cutoff. Detection/Diagnosis labels come from the existing
reversible auto-fault system (unchanged); RUL labels come from the new irreversible
wear model, computed in hindsight once each scenario's outcome (failed vs censored) is known.
"""
import sys, os, csv, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from physics import UAVEngineTwin

np.random.seed(7)

N_SCENARIOS = 1500
MAX_DURATION = 20000.0
DT = 1.0
LEG_MIN, LEG_MAX = 300, 800   # seconds between randomized flight-leg target resets

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "validation", "unified_database.csv")

COLUMNS = [
    "scenario_id","time","altitude","throttle","airspeed","aoa","air_density",
    "torque_available_nm","engine_rpm","prop_rpm","prop_torque","power_kw","fuel_flow",
    "thrust","lift","drag","thrust_margin","lift_weight_margin",
    "egt_healthy","cht_healthy","oil_pressure_healthy","oil_temp_healthy",
    "vibx_healthy","viby_healthy","vibz_healthy",
    "egt","cht","oil_pressure","oil_temp","vibx","viby","vibz","rpm_fault",
    "egt_flag","cht_flag","oil_pressure_flag","oil_temp_flag",
    "vibx_flag","viby_flag","vibz_flag","rpm_flag",
    "egt_stress","cht_stress","oil_pressure_stress","oil_temp_stress",
    "vibx_stress","viby_stress","vibz_stress","rpm_stress",
    "wear","is_failed_now",
    "scenario_failed","scenario_failure_time","scenario_censored","rul_true",
]

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def sample_leg_target(severity_bias):
    # severity_bias in [0,1]: 0=gentle pilot, 1=aggressive pilot - biases throttle up/airspeed down.
    # Wider range so aggressive pilots genuinely reach near-redline, not just "somewhat higher".
    throttle = clip(np.random.uniform(0.15, 0.35) + severity_bias*np.random.uniform(0.4, 0.65), 0.1, 1.0)
    airspeed = clip(np.random.uniform(45, 70) - severity_bias*np.random.uniform(20, 55), 0.0, 80.0)
    altitude = np.random.uniform(0, 8000)
    aoa = np.random.uniform(-10, 20)
    return altitude, throttle, airspeed, aoa

t_start = time.time()
f = open(OUT_PATH, "w", newline="")
writer = csv.writer(f)
writer.writerow(COLUMNS)

row_count = 0
n_failed = 0
n_censored = 0

for scen in range(N_SCENARIOS):
    twin = UAVEngineTwin(dt=DT, physics_version="v2")
    severity_bias = np.random.uniform(0.0, 1.0)   # this "pilot" personality persists all life
    altitude, throttle, airspeed, aoa = sample_leg_target(severity_bias)
    twin.altitude, twin.throttle, twin.airspeed, twin.aoa = altitude, throttle, airspeed, aoa
    next_leg_change = np.random.uniform(LEG_MIN, LEG_MAX)

    buffer = []
    t = 0.0
    while t < MAX_DURATION:
        # small continuous drift plus periodic full leg changes (new "flight")
        if t >= next_leg_change:
            altitude, throttle, airspeed, aoa = sample_leg_target(severity_bias)
            next_leg_change = t + np.random.uniform(LEG_MIN, LEG_MAX)
        # Strong reversion to the current leg target (small noise on top) - a "leg" means
        # the pilot is actually holding roughly this setting, not randomly wandering
        twin.throttle  = clip(twin.throttle  + np.random.normal(0, 0.008) + 0.15*(throttle-twin.throttle), 0.1, 1.0)
        twin.altitude  = clip(twin.altitude  + np.random.normal(0, 8.0)   + 0.05*(altitude-twin.altitude), 0.0, 9000.0)
        twin.airspeed  = clip(twin.airspeed  + np.random.normal(0, 0.4)   + 0.10*(airspeed-twin.airspeed), 0.0, 80.0)
        twin.aoa       = clip(twin.aoa       + np.random.normal(0, 0.2)   + 0.10*(aoa-twin.aoa), -15.0, 25.0)

        out = twin.step()
        h, ff, fs = out["healthy"], out["fault_flags"], out["fault_stress"]
        buffer.append([out["time"], out["altitude"], out["throttle"], out["airspeed"], out["aoa"],
            out["air_density"], out["torque_available_nm"], out["engine_rpm"], out["prop_rpm"],
            out["prop_torque"], out["power_kw"], out["fuel_flow"], out["thrust"], out["lift"],
            out["drag"], out["thrust_margin"], out["lift_weight_margin"],
            h["egt"], h["cht"], h["oil_pressure"], h["oil_temp"], h["vibx"], h["viby"], h["vibz"],
            out["egt"], out["cht"], out["oil_pressure"], out["oil_temp"],
            out["vibx"], out["viby"], out["vibz"], out["rpm_fault"],
            ff["egt"], ff["cht"], ff["oil_pressure"], ff["oil_temp"],
            ff["vibx"], ff["viby"], ff["vibz"], ff["rpm"],
            fs["egt"], fs["cht"], fs["oil_pressure"], fs["oil_temp"],
            fs["vibx"], fs["viby"], fs["vibz"], fs["rpm"],
            out["wear"], int(out["failed"]),
        ])
        t = out["time"]
        if out["failed"]:
            break

    failed = twin.failed
    failure_time = twin.failure_time if failed else None
    censored = not failed
    n_failed += int(failed)
    n_censored += int(censored)

    for row in buffer:
        rt = row[0]
        rul = (failure_time - rt) if failed else None
        writer.writerow([scen] + row + [int(failed), failure_time, int(censored), rul])
        row_count += 1

    if (scen+1) % 50 == 0:
        elapsed = time.time() - t_start
        print(f"scenario {scen+1}/{N_SCENARIOS}, rows={row_count}, failed={n_failed}, censored={n_censored}, elapsed={elapsed:.1f}s")
        sys.stdout.flush()

f.close()
print(f"DONE. rows={row_count} failed={n_failed} censored={n_censored} time={time.time()-t_start:.1f}s")
