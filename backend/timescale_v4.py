"""The v4 time model: two clocks, one constant. Defined here and nowhere else.

FLIGHT CLOCK - flight seconds, 1 flight second = 1 wall-clock second. Cannot be
    compressed: the models were trained on 1 Hz rows, so a 128-row window means
    128 real seconds; temperatures settle over minutes and the shaft over ~1.5 s.
    Drives physics (PHYSICS_HZ steps per flight second), sensors, the AI window,
    telemetry timestamps and replay.

LIFE CLOCK - engine hours. Each flight second ages the engine LIFE_SCALE seconds,
    duty-weighted exactly as generate_dataset_v4.py counts life:
        engine_hours = start_engine_hours + LIFE_SCALE * life_used_h
    Drives degradation, wear condition, fault growth, RUL and hours-to-overhaul.

WHY 180 (docs/v4_integration_plan.md, Spec T). In training an engine aged at most
0.036 h inside a 128 s window, so faults were constant within a window. The
fastest-growing fault the generator produces grows 0.0077 severity per engine
hour (measured over 800 generated faults, 914 and 915). Holding in-window growth
to 0.05 - a quarter of the severity head's own 0.216 error, and below the 0.08
"present" threshold - gives 0.05 / (0.0077 x 128 s) = 0.051 h per flight second,
rounded down to 0.05 h/s = x180. A 10-minute flight ages the engine 30 h.

RULE THAT KEEPS MODEL INPUTS HONEST. Per-flight quantities stay on the flight
clock (life_used_norm, as in training, where it never exceeded 1.25 h); only
life quantities go on the life clock. Nothing downstream rescales either.

Every v4 run stores the LIFE_SCALE it was recorded with (simulations.life_scale),
so changing this constant later never reinterprets an old run.
"""
FLIGHT_HZ = 1            # AI samples per flight second (the training row rate)
PHYSICS_HZ = 100         # physics steps per flight second (dt = 0.01)
LIFE_SCALE = 180.0       # engine seconds per flight second


def engine_hours(start_engine_hours: float, life_used_h: float,
                 life_scale: float = LIFE_SCALE) -> float:
    """Life clock from the flight's duty-weighted usage."""
    return float(start_engine_hours) + float(life_scale) * float(life_used_h)


def describe() -> dict:
    """What /state and /health report, so the frontend never hard-codes the scale."""
    return {"life_scale": LIFE_SCALE, "flight_hz": FLIGHT_HZ, "physics_hz": PHYSICS_HZ}
