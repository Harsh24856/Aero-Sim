"""The simulated AIRCRAFT: a separate process that flies its own engine and puts only CAN
frames on the bus. The digital twin never sees this process's state.

    aircraft_sim.py   plant    its own UAVEngineTwin (+ optional mismatch) -> CAN frames
    can_ingest.py     bridge   CAN frames -> /params (set-points) + /measured (sensors)
    main.py           twin     its own physics follows the set-points; the AI and the
                               physics residuals are scored on the MEASURED sensors

Without --mismatch the plant is the same physics as the twin, which only proves the
plumbing. The mismatch profiles make the plant deliberately NOT the twin - the question
a real engine will ask - so detection can be checked against an engine it was not built from.

Usage (from backend/, with the stack running):
    python can_ingest.py --bus udp &
    python aircraft_sim.py --engine Rotax_914_ULF --profile mission --mismatch calibration
The engine must match the one selected in the twin (the simulate page's ?engine=).
"""
import argparse
import time

import numpy as np

import can_bus
from physics import UAVEngineTwin, ENGINE_CONFIGS

# What each profile changes between the aircraft and the twin.
MISMATCH_PROFILES = {
    "none": {},
    # Installation/calibration differences: every sensor reads a little off.
    "calibration": {"offset": {"egt": 15.0, "cht": 4.0, "oil_temp": 3.0, "oil_pressure": -2.0}},
    # A different unit of the same engine type: slightly different gains everywhere.
    "engine_spread": {"gain": {"egt": 1.03, "cht": 1.04, "oil_temp": 1.02, "oil_pressure": 0.95,
                               "vibx": 1.2, "viby": 1.2, "vibz": 1.2, "fuel_flow": 1.05}},
    # Real sensor noise the simulator does not have.
    "noisy_sensors": {"noise": {"egt": 3.0, "cht": 0.8, "oil_temp": 0.5, "oil_pressure": 0.6,
                                "vibx": 0.003, "viby": 0.003, "vibz": 0.003, "rpm": 15.0}},
    # One sensor slowly drifting - a genuine fault the twin should catch.
    "sensor_drift": {"drift_per_min": {"cht": 1.0}},
    # A plant with different thermal behaviour entirely (physics v2 calibration).
    "v2_plant": {"physics_version": "v2"},
}

# Flight profiles: (start_s, end_s, values at start, values at end), linearly interpolated.
PROFILES = {
    "cruise": [
        (0, 1, dict(throttle=0.5, altitude=1500, airspeed=45, aoa=2, isa_dev_c=0), None),
    ],
    "mission": [
        (0, 60, dict(throttle=0.80, altitude=0, airspeed=12, aoa=6, isa_dev_c=0),
                dict(throttle=0.80, altitude=30, airspeed=35, aoa=6, isa_dev_c=0)),
        (60, 240, dict(throttle=0.65, altitude=30, airspeed=40, aoa=5, isa_dev_c=0),
                  dict(throttle=0.65, altitude=1500, airspeed=40, aoa=5, isa_dev_c=0)),
        (240, 600, dict(throttle=0.50, altitude=1500, airspeed=45, aoa=2, isa_dev_c=0), None),
        (600, 720, dict(throttle=0.95, altitude=1500, airspeed=45, aoa=2, isa_dev_c=0), None),
        (720, 1e9, dict(throttle=0.50, altitude=1500, airspeed=45, aoa=2, isa_dev_c=0), None),
    ],
}

SENSOR_KEYS = {"rpm": "rpm_fault", "egt": "egt", "cht": "cht", "oil_pressure": "oil_pressure",
               "oil_temp": "oil_temp", "vibx": "vibx", "viby": "viby", "vibz": "vibz",
               "fuel_flow": "fuel_flow"}


def setpoints_at(profile: list, t: float) -> dict:
    current = profile[0][2]
    for start, end, a, b in profile:
        if t < start:
            break
        if b is None or t >= end:
            current = a if b is None else b
        else:
            f = (t - start) / (end - start)
            current = {k: a[k] + f * (b[k] - a[k]) for k in a}
    return current


def apply_mismatch(values: dict, mismatch: dict, t: float, rng: np.random.Generator) -> dict:
    out = dict(values)
    for k, g in mismatch.get("gain", {}).items():
        out[k] *= g
    for k, off in mismatch.get("offset", {}).items():
        out[k] += off
    for k, rate in mismatch.get("drift_per_min", {}).items():
        out[k] += rate * t / 60.0
    for k, sd in mismatch.get("noise", {}).items():
        out[k] += rng.normal(0.0, sd)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="Rotax_914_ULF", choices=sorted(ENGINE_CONFIGS))
    ap.add_argument("--profile", default="mission", choices=sorted(PROFILES))
    ap.add_argument("--mismatch", default="none", choices=sorted(MISMATCH_PROFILES))
    ap.add_argument("--bus", default="udp", help="udp, udp://group:port, or <python-can interface>:<channel>")
    ap.add_argument("--rate", type=float, default=10.0, help="frames per second per signal")
    ap.add_argument("--duration", type=float, default=900.0, help="simulated seconds to fly")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--wear", type=float, default=None, help="initial engine wear 0..1 (default: physics default)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    mismatch = MISMATCH_PROFILES[a.mismatch]
    plant = UAVEngineTwin(dt=0.01, engine_model=a.engine,
                          physics_version=mismatch.get("physics_version", "v3"))
    if a.wear is not None:
        plant.wear = float(a.wear)
    rng = np.random.default_rng(a.seed)
    np.random.seed(a.seed)
    bus = can_bus.open_bus(a.bus)
    profile = PROFILES[a.profile]
    steps_per_frame = max(1, int(round(1.0 / (a.rate * plant.dt))))
    print(f"[aircraft] {a.engine} ({plant.physics_version}) profile={a.profile} mismatch={a.mismatch} "
          f"bus={a.bus} {a.rate:g} Hz")

    seq, frames, step = 0, 0, 0
    start = time.monotonic()
    try:
        while plant.t < a.duration:
            sp = setpoints_at(profile, plant.t)
            plant.throttle, plant.altitude, plant.airspeed = sp["throttle"], sp["altitude"], sp["airspeed"]
            plant.aoa, plant.isa_dev_c = sp["aoa"], sp["isa_dev_c"]
            out = plant.step()
            step += 1
            if step % steps_per_frame == 0:
                sensors = apply_mismatch({k: float(out[src]) for k, src in SENSOR_KEYS.items()},
                                         mismatch, plant.t, rng)
                for name, value in list(sp.items()) + list(sensors.items()):
                    bus.send(*can_bus.encode(name, value))
                bus.send(*can_bus.encode("heartbeat", seq))
                seq, frames = seq + 1, frames + len(sp) + len(sensors) + 1
            if step % 1000 == 0:
                print(f"[aircraft] t={plant.t:6.0f}s thr={sp['throttle']:.2f} alt={sp['altitude']:6.0f} "
                      f"rpm={out['engine_rpm']:5.0f} egt={out['egt']:4.0f} cht={out['cht']:5.1f} "
                      f"wear={plant.wear:.4f} frames={frames}", flush=True)
            delay = start + plant.t / a.speed - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()
    print(f"[aircraft] done: {plant.t:.0f} simulated seconds, {frames} frames")


if __name__ == "__main__":
    main()
