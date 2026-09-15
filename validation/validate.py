"""Validate the Python physics twin against real Simu/Users/harsh/.Trash/simulink_ground_truth.csvdel output."""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from physics import UAVEngineTwin

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "simulink_ground_truth.csv"))

twin = UAVEngineTwin(dt=0.01)
twin.omega = 3000.0 * 2*np.pi/60.0  # match Simulink initial condition

results = {k: [] for k in ["EngineRPM","PropRPM","Thrust","PropTorque","Power_kW",
    "FuelFlow","Lift","Drag","Thrust_margin","LiftWeight_margin",
    "EGT","CHT","OilPressure","OilTemp","VibX","VibY","VibZ","AirDensity","Torque_available_Nm"]}

for _, row in df.iterrows():
    twin.altitude = row["Altitude"]
    twin.throttle = row["Throttle"]
    twin.airspeed = row["Airspeed"]
    twin.aoa = row["AoA"]
    out = None
    for _ in range(10):  # 10 x dt=0.01 = 0.1s per ground-truth row
        out = twin.step()
    results["EngineRPM"].append(out["engine_rpm"])
    results["PropRPM"].append(out["prop_rpm"])
    results["Thrust"].append(out["thrust"])
    results["PropTorque"].append(out["prop_torque"])
    results["Power_kW"].append(out["power_kw"])
    results["FuelFlow"].append(out["fuel_flow"])
    results["Lift"].append(out["lift"])
    results["Drag"].append(out["drag"])
    results["Thrust_margin"].append(out["thrust_margin"])
    results["LiftWeight_margin"].append(out["lift_weight_margin"])
    results["EGT"].append(out["healthy"]["egt"])
    results["CHT"].append(out["healthy"]["cht"])
    results["OilPressure"].append(out["healthy"]["oil_pressure"])
    results["OilTemp"].append(out["healthy"]["oil_temp"])
    results["VibX"].append(out["healthy"]["vibx"])
    results["VibY"].append(out["healthy"]["viby"])
    results["VibZ"].append(out["healthy"]["vibz"])
    results["AirDensity"].append(out["air_density"])
    results["Torque_available_Nm"].append(out["torque_available_nm"])

print(f"{'Variable':<20}{'MaxAbsErr':>12}{'RMSErr':>12}{'MaxVal':>12}{'Pass':>8}")
all_pass = True
for k in results:
    py = np.array(results[k])
    ref = df[k].values
    err = np.abs(py-ref)
    maxval = max(np.max(np.abs(ref)), 1e-9)
    rel_tol = 0.02 * maxval  # 2% of max magnitude
    passed = np.max(err) < max(rel_tol, 1e-6)
    all_pass = all_pass and passed
    print(f"{k:<20}{np.max(err):>12.4g}{np.sqrt(np.mean(err**2)):>12.4g}{maxval:>12.4g}{str(passed):>8}")

print()
print("ALL VARIABLES PASS:" , all_pass)
