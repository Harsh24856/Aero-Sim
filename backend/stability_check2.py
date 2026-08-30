import sys; sys.path.insert(0, ".")
from physics import UAVEngineTwin
import numpy as np

twin = UAVEngineTwin(dt=0.1)
twin.throttle = 0.9
twin.airspeed = 15
egts=[]; chts=[]; oilps=[]
for i in range(6000):
    out = twin.step()
    egts.append(out["egt"]); chts.append(out["cht"]); oilps.append(out["oil_pressure"])
egts=np.array(egts); chts=np.array(chts); oilps=np.array(oilps)
print(f"EGT range: {egts.min():.1f} to {egts.max():.1f} (limit 0-950)")
print(f"CHT range: {chts.min():.1f} to {chts.max():.1f} (limit 0-260)")
print(f"OilPressure range: {oilps.min():.1f} to {oilps.max():.1f} (limit 0-150)")
