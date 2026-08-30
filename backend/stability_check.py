import sys; sys.path.insert(0, ".")
from physics import UAVEngineTwin
import numpy as np

twin = UAVEngineTwin(dt=0.1)
twin.throttle = 0.9
twin.airspeed = 15
rpms = []
for i in range(6000):
    out = twin.step()
    rpms.append(out["engine_rpm"])
rpms = np.array(rpms)
egt = out["egt"]; cht = out["cht"]; oilp = out["oil_pressure"]
print(f"RPM range: {rpms.min():.1f} to {rpms.max():.1f}")
print(f"Any NaN/Inf: {np.any(np.isnan(rpms))}, {np.any(np.isinf(rpms))}")
print(f"Final EGT={egt:.1f} CHT={cht:.1f} OilP={oilp:.1f}")
