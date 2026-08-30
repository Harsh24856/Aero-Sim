import sys, time; sys.path.insert(0, ".")
from physics import UAVEngineTwin

for dt in [0.01, 0.05, 0.1]:
    twin = UAVEngineTwin(dt=dt)
    twin.throttle = 0.6
    n_steps = 20000
    t0 = time.time()
    for i in range(n_steps):
        twin.step()
    elapsed = time.time() - t0
    rate = n_steps/elapsed
    sim_seconds = n_steps*dt
    print(f"dt={dt}: {n_steps} steps in {elapsed:.2f}s ({rate:.0f} steps/sec, simulated {sim_seconds:.0f}s of engine time)")
    print(f"   -> to generate 1,000,000 SECONDS of sim time: {1000000/dt/rate:.1f} sec wall-clock")
