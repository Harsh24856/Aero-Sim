"""A12/A13 - dual-rate telemetry from the C-MAPSS engine.

THE RATE MISMATCH, AND WHY IT IS NOT A DETAIL
    C-MAPSS is ONE ROW PER FLIGHT. The paper is explicit (section V, p.5): "this
    allows us to take one measurement snapshot per flight to characterize the
    engine health during or right after that flight", taken at cruise once the
    system response has reached steady state. A "cycle" is a whole flight, not a
    sample interval.

    The MALE-UAV stack streams continuously. Emitting one C-MAPSS row per flight
    into a live dashboard would show a value every few minutes; emitting the
    cycle table at stream rate would claim a resolution the benchmark does not
    have.

    So the engine runs DUAL-RATE, which is also how NASA actually produced the
    data - a full flight was simulated and the snapshot taken at cruise:

        stream  continuous within-flight samples, for the live interface
        cycle   exactly one row per flight, at cruise steady state, in the
                released 26-column format

    Both come from the same degradation state, so the live view and the
    benchmark table can never disagree.

WITHIN-FLIGHT VARIATION IS EXPLICITLY MODELLED AS NOT DEGRADATION
    Health is held constant across a flight. The paper states that damage
    accumulation within a single flight is not directly quantifiable and that
    "sudden degradation during a flight is rather unlikely" - which is precisely
    why one snapshot per flight suffices. Within-flight movement in this module
    is therefore the flight profile and sensor noise, never wear. Wear advances
    between flights.

DATA SOURCE
    Frames are tagged `data_source = "cmapss_sim"`, alongside the existing "sim"
    and "can". The AI interface contract is untouched.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import damage as D
from . import noise as N
from .surrogate import ResponseSurface, gkey

DATA_SOURCE = "cmapss_sim"

# Fraction of a flight spent at the cruise steady state from which the C-MAPSS
# snapshot is taken. The remainder is climb-in and descent-out, which exist to
# make the stream look like a flight and are never sampled for the cycle table.
CRUISE_WINDOW = (0.35, 0.85)


@dataclass
class CMAPSSTelemetryEngine:
    """One engine unit, streaming continuously and emitting one row per flight."""
    subset: str
    rs: ResponseSurface
    gains: dict
    unit: int = 1
    samples_per_flight: int = 60
    seed: int | None = None

    # -- internal state ----------------------------------------------------
    _rng: np.random.Generator = field(init=False, repr=False)
    _params: D.UnitDamageParams = field(init=False, repr=False)
    _mode: int = field(init=False, default=0)
    _gain: float = field(init=False, default=1.0)
    _cycle: int = field(init=False, default=0)
    _health: float = field(init=False, default=1.0)
    _failed: bool = field(init=False, default=False)
    _traj: dict = field(init=False, repr=False, default_factory=dict)
    _offset: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        seed = self.seed if self.seed is not None else C.subset_seed(self.subset, "train")
        self._rng = np.random.default_rng(seed + self.unit)
        self._params = D.sample_unit_params(self.unit, self._rng)

        if self.rs.n_modes > 1:
            w = np.asarray(self.rs.mode_weights, float)
            self._mode = int(self._rng.choice(self.rs.n_modes, p=w / w.sum()))
        self._gain = self.gains[self._mode] if isinstance(self.gains, dict) else self.gains

        # Pre-compute the whole life so RUL is exact at every point. The unit is
        # a simulation, so its true remaining life is known - unlike a real
        # engine, where RUL is the thing being estimated.
        run = D.simulate_unit(self._params, max_cycles=4000, gain=self._gain)
        self._traj = run
        self._health = float(run["health"][0])

        # One sensor offset per unit, drawn once and held for life: this engine's
        # manufacturing variation and initial wear. The batch generator applies
        # the same term, and without it a streamed fleet is narrower than the
        # released data (Nc sd 17.8 against 22.1) because only the within-unit
        # noise is present.
        self._offset = {}
        for key, sds in self.rs.between_unit_sd.items():
            for s, sd in sds.items():
                self._offset[(key, s)] = float(self._rng.normal(0.0, sd)) if sd > 0 else 0.0

    # -- properties --------------------------------------------------------
    @property
    def total_life(self) -> int:
        return int(self._traj["length"])

    @property
    def rul_cycles(self) -> int:
        return max(0, self.total_life - self._cycle)

    @property
    def failed(self) -> bool:
        return self._failed

    # -- the two rates -----------------------------------------------------
    def step_flight(self) -> tuple[list[dict], dict]:
        """Fly one flight. Returns (stream frames, the one C-MAPSS cycle row).

        Health is fixed for the duration of the flight and advances only here,
        between flights - see the module docstring.
        """
        if self._failed:
            raise RuntimeError(f"unit {self.unit} has already failed at cycle {self._cycle}")

        self._cycle += 1
        idx = min(self._cycle - 1, self.total_life - 1)
        self._health = float(self._traj["health"][idx])

        regime = self._draw_regime()
        settings = self._draw_settings(regime)
        key = gkey(self._mode, regime)
        if key not in self.rs.coef:
            key = gkey(0, regime)

        n = int(self.samples_per_flight)
        lo, hi = CRUISE_WINDOW
        cruise = np.zeros(n, dtype=bool)
        cruise[int(lo * n):max(int(hi * n), int(lo * n) + 1)] = True

        # Throttle-like profile: ramps up, holds through cruise, ramps down. It
        # shapes the stream only; the cycle row is taken from the cruise hold.
        t = np.linspace(0.0, 1.0, n)
        profile = np.clip(np.minimum(t / max(lo, 1e-6), (1.0 - t) / max(1.0 - hi, 1e-6)), 0.0, 1.0)

        sensors = {}
        for s in C.SENSOR_COLS:
            if s in self.rs.constant_sensors:
                sensors[s] = np.full(n, self.rs.constant_values[key][s])
                continue
            base = float(np.polyval(self.rs.coef[key][s], self._health))
            npar = self.rs.noise_params[key].get(s, {"phi": 0.0, "var": 0.0, "kurtosis": 3.0})
            meas = N.two_component_noise(
                n, npar["phi"], npar.get("var_smooth", npar["var"]),
                npar.get("var_white", 0.0), npar["kurtosis"], self._rng)
            # Off-cruise samples sit away from the steady-state value; the
            # deviation is flight phase, not wear.
            sensors[s] = base * (0.90 + 0.10 * profile) + self._offset.get((key, s), 0.0) + meas

        frames = [self._frame(i, n, sensors, settings, regime, bool(cruise[i]))
                  for i in range(n)]

        cruise_idx = np.flatnonzero(cruise)
        row = self._cycle_row(sensors, settings, cruise_idx)

        if self._cycle >= self.total_life:
            self._failed = True
        return frames, row

    # -- helpers -----------------------------------------------------------
    def _draw_regime(self) -> int:
        n_reg = self.rs.regime_centres.shape[0]
        if n_reg == 1:
            return 0
        w = np.array([self.rs.regime_settings[r]["weight"] for r in range(n_reg)])
        return int(self._rng.choice(n_reg, p=w / w.sum()))

    def _draw_settings(self, regime: int) -> np.ndarray:
        spec = self.rs.regime_settings[regime]
        mu = np.asarray(spec["mean"], float)
        sd = np.asarray(spec["sd"], float)
        return mu + self._rng.normal(0.0, 1.0, 3) * sd

    def _frame(self, i, n, sensors, settings, regime, in_cruise) -> dict:
        f = {
            "data_source": DATA_SOURCE,
            "subset": self.subset,
            "unit": self.unit,
            "cycle": self._cycle,
            "sample": i,
            "flight_fraction": (i + 1) / n,
            "in_cruise": in_cruise,
            "operating_regime": regime,
            "fault_mode": self._mode,
            # Ground truth, available because this is a simulation.
            "health_index": self._health,
            "rul_cycles": self.rul_cycles,
            "total_life_cycles": self.total_life,
        }
        for j, c in enumerate(C.SETTING_COLS):
            f[c] = float(settings[j])
        for s in C.SENSOR_COLS:
            f[s] = float(sensors[s][i])
        return f

    def _cycle_row(self, sensors, settings, cruise_idx) -> dict:
        """The one row per flight: a single SNAPSHOT taken at cruise steady
        state, quantised to the channel's own step, in released column order.

        A snapshot, not an average over the cruise window. The paper describes
        "one measurement snapshot per flight ... collected after the system
        response reached a steady state" - one sample. Averaging n cruise
        samples would divide the measurement noise by sqrt(n) and produce a
        cycle table quieter than the released data: at 10 cruise samples it put
        T50 at sd 7.3 against the batch generator's 8.98.
        """
        row = {"unit": self.unit, "cycle": self._cycle}
        for j, c in enumerate(C.SETTING_COLS):
            row[c] = float(settings[j])
        snap = int(cruise_idx[len(cruise_idx) // 2])
        for s in C.SENSOR_COLS:
            v = float(sensors[s][snap])
            q = self.rs.quantum.get(s, 0.0)
            row[s] = float(np.round(v / q) * q) if q > 0 else v
        row["_health_index"] = self._health
        row["_rul_cycles"] = self.rul_cycles
        row["_data_source"] = DATA_SOURCE
        return row

    def run_to_failure(self, max_flights: int = 5000):
        """Fly until the failure criterion is met. Yields (frames, row)."""
        while not self._failed and self._cycle < max_flights:
            yield self.step_flight()


def load_engine(subset: str, unit: int = 1, proc_dir: str | None = None,
                **kw) -> CMAPSSTelemetryEngine:
    """Build an engine from the artefacts written by the fitting pipeline."""
    proc_dir = proc_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "cmapss", "processed")
    with open(os.path.join(proc_dir, f"calibration_{subset}.json")) as fh:
        cal = json.load(fh)
    gains = {int(k): v for k, v in cal["calibrated_parameter"]["value_by_mode"].items()}
    rs = ResponseSurface.load(os.path.join(proc_dir, f"surrogate_{subset}.json"))
    return CMAPSSTelemetryEngine(subset=subset, rs=rs, gains=gains, unit=unit, **kw)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="C-MAPSS dual-rate telemetry (A12/A13).")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--unit", type=int, default=1)
    ap.add_argument("--flights", type=int, default=5)
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    eng = load_engine(args.subset, args.unit, samples_per_flight=args.samples)
    print(f"unit {eng.unit}  subset {args.subset}  fault_mode {eng._mode}  "
          f"total life {eng.total_life} cycles  source {DATA_SOURCE}")
    print(f"{'cycle':>6} {'health':>8} {'RUL':>6} {'frames':>7} {'T50(cruise)':>12} "
          f"{'Ps30':>8}  stream T50 min..max")
    for i in range(args.flights):
        frames, row = eng.step_flight()
        t50 = [f["T50"] for f in frames]
        print(f"{row['cycle']:>6} {row['_health_index']:>8.4f} {row['_rul_cycles']:>6} "
              f"{len(frames):>7} {row['T50']:>12.2f} {row['Ps30']:>8.2f}  "
              f"{min(t50):.2f}..{max(t50):.2f}")
    n_cruise = sum(f["in_cruise"] for f in frames)
    print(f"\nlast flight: {len(frames)} stream frames, {n_cruise} in cruise "
          f"-> 1 C-MAPSS row")
    print(f"cycle row keys in released order: {list(C.ALL_COLS)}")


if __name__ == "__main__":
    main()
