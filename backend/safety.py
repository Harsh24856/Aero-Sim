"""Safety primitives for the live simulation loop (backend/main.py).

Everything here is pure and cheap, so it can run on every physics step without
costing smoothness:

  json_safe          NaN/inf -> None. The browser's JSON.parse rejects the bare NaN
                     token Python's json.dumps emits, so one bad float froze the cockpit.
  telemetry_problem  checks a physics step before anything downstream consumes it.
  clamp_param        bounds and rejects /params input before it reaches the twin.
  CircuitBreaker     stops a failing AI service from stalling or flooding the loop,
                     and probes for recovery on its own.
"""
import math
import time
from typing import Any, Optional

# Channels a physics step must produce as finite numbers. If any of these is
# missing or non-finite the step is rejected and the twin is recovered.
REQUIRED_FINITE = (
    "time", "altitude", "throttle", "airspeed", "air_density", "engine_rpm",
    "torque_available_nm", "power_kw", "fuel_flow", "thrust", "lift", "drag",
    "egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm_fault",
)
# Physics v4 (twin_v4.py) reports brake torque and the twelve measured channels
# instead, plus the life clock every downstream consumer relies on.
REQUIRED_FINITE_V4 = (
    "time", "altitude", "throttle", "airspeed", "air_density", "engine_rpm",
    "torque_nm", "power_kw", "fuel_flow", "thrust", "lift", "drag",
    "egt", "cht", "coolant_temp", "oil_pressure", "oil_temp", "manifold_pressure_kpa",
    "vibx", "viby", "vibz", "battery_voltage", "engine_hours", "margin_min",
)

# Input envelope for /params. Matches the cockpit (components/Simulator.tsx:
# altitude 0-8000 m, speed floor 35 m/s except during the take-off roll, which
# starts at 0) and backend/can_ingest.py's accepted ranges.
PARAM_LIMITS = {
    "altitude": (0.0, 8000.0),
    "throttle": (0.0, 1.0),
    "airspeed": (0.0, 80.0),
    "aoa": (-20.0, 25.0),
    "isa_dev_c": (-30.0, 50.0),
    # physics v4 installation and fuel inputs: the ranges generate_dataset_v4.py samples,
    # so the models never see an engine state they were not trained on.
    "qnh_offset_pa": (-2700.0, 2700.0),
    "humidity_frac": (0.0, 1.0),
    "fuel_octane_mon": (91.0, 100.0),
    "fuel_ethanol_frac": (0.0, 0.10),
    "cooling_airflow_factor": (0.65, 1.30),
    "electrical_load_a": (3.0, 34.0),
    "target_lambda": (0.78, 1.02),
}


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None and numpy scalars with Python ones."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):          # numpy scalar
        try:
            return json_safe(obj.item())
        except (TypeError, ValueError):
            return None
    return obj


def telemetry_problem(out: Any) -> Optional[str]:
    """None if a physics step is usable, otherwise a short reason."""
    if not isinstance(out, dict):
        return f"step returned {type(out).__name__}, not a dict"
    for key in (REQUIRED_FINITE_V4 if out.get("physics_version") == "v4" else REQUIRED_FINITE):
        v = out.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return f"{key} missing or not numeric ({v!r})"
        if not math.isfinite(float(v)):
            return f"{key} is {v}"
    return None


def clamp_param(name: str, value: Any) -> Optional[float]:
    """The value bounded to PARAM_LIMITS, or None when it is not a finite number."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    lo, hi = PARAM_LIMITS[name]
    return min(hi, max(lo, v))


class CircuitBreaker:
    """Closed -> (N consecutive failures) -> open -> (cooldown) -> half-open probe.

    While open, callers skip the dependency entirely instead of paying a timeout on
    every sample. After the cooldown one probe is allowed through; success closes the
    breaker (and record_success reports the recovery so the caller can resync).
    """

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 5.0, clock=time.monotonic):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.failures = 0
        self.opened_at: Optional[float] = None
        self.last_error: Optional[str] = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self.opened_at >= self.cooldown_s else "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> bool:
        """Returns True when this success ends an outage."""
        recovered = self.opened_at is not None
        self.failures = 0
        self.opened_at = None
        self.last_error = None
        return recovered

    def record_failure(self, error: str = "") -> None:
        self.failures += 1
        self.last_error = error or self.last_error
        if self.opened_at is not None or self.failures >= self.failure_threshold:
            self.opened_at = self._clock()      # (re)open; a failed probe restarts the cooldown

    def reset(self) -> None:
        self.failures = 0
        self.opened_at = None
        self.last_error = None
