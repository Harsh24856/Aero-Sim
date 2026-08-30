"""
Generate a large labeled training dataset from the UAV Digital Twin for AI training.
2000 scenarios x 500s x dt=1.0s = 1,000,000 rows. Each scenario randomizes initial
altitude/throttle/airspeed/AoA across the full realistic envelope, then lets them drift
naturally (bounded random walk) during the run - so the dataset covers steady cruise,
climbs, descents, throttle changes, and (via the auto-fault system) real emergent
faults across all 8 sensor channels, all from a single consistent physics engine.
"""
import sys, os, csv, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from physics import UAVEngineTwin

np.random.seed(42)

N_SCENARIOS = 2000
DURATION_S = 500
DT = 1.0

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "validation", "training_data.csv")

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
]

def clip(v, lo, hi):
    return max(lo, min(hi, v))

t_start = time.time()
f = open(OUT_PATH, "w", newline="")
writer = csv.writer(f)
writer.writerow(COLUMNS)

row_count = 0
for scen in range(N_SCENARIOS):
    twin = UAVEngineTwin(dt=DT)
    twin.altitude = np.random.uniform(0, 8000)
    twin.throttle = np.random.uniform(0.15, 1.0)
    twin.airspeed = np.random.uniform(5, 75)
    twin.aoa = np.random.uniform(-12, 22)

    n_steps = int(DURATION_S/DT)
    for i in range(n_steps):
        # bounded random walk on inputs -> realistic in-flight variation
        twin.throttle  = clip(twin.throttle  + np.random.normal(0, 0.01), 0.1, 1.0)
        twin.altitude  = clip(twin.altitude  + np.random.normal(0, 15.0), 0.0, 9000.0)
        twin.airspeed  = clip(twin.airspeed  + np.random.normal(0, 0.8), 0.0, 80.0)
        twin.aoa       = clip(twin.aoa       + np.random.normal(0, 0.3), -15.0, 25.0)

        out = twin.step()
        h = out["healthy"]
        ff = out["fault_flags"]
        fs = out["fault_stress"]
        writer.writerow([
            scen, out["time"], out["altitude"], out["throttle"], out["airspeed"], out["aoa"],
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
        ])
        row_count += 1

    if (scen+1) % 100 == 0:
        elapsed = time.time() - t_start
        print(f"scenario {scen+1}/{N_SCENARIOS}, rows={row_count}, elapsed={elapsed:.1f}s")
        sys.stdout.flush()

f.close()
print(f"DONE. Total rows: {row_count}. Total time: {time.time()-t_start:.1f}s")
