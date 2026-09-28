"""Physics residuals: what each sensor reads minus what physics v3 says it should read.

PLAN PHASE 7 - the digital-twin method the models alone do not give you.

    residual = measured - physics_expected(throttle, rpm, power, airspeed, ambient, wear)

Why it matters for PS 26054:
  * Explainable by construction: "CHT reads +14 C above what physics expects at
    this power and airspeed" needs no model to justify it.
  * Model-free and cheap: plain arithmetic, no TensorFlow, no network - it runs
    onboard and keeps working while the AI service is warming up or down.
  * Sensor-drift detection (section C): a genuinely wearing engine moves EGT, CHT,
    oil temperature, oil pressure and vibration TOGETHER; a drifting or stuck
    sensor moves ONE channel. The monitor estimates a wear-like degradation index
    from the channel groups that currently agree with physics and flags any
    channel that disagrees with it.

The expected values are physics v3 exactly (backend/physics.py), including the
first-order thermal lags. Checked against the dataset's clean *_healthy columns on
400k test rows: EGT p99 error 0.007 C, CHT 0.06 C, oil temperature 0.11 C, oil
pressure 0.004 psi, vibration 3e-5 g. On live physics (100 Hz, sampled at 1 Hz)
steady residuals are EGT 0.05 C / vibration 0.0002 g, with transients up to 5.6 C /
0.0018 g in the minute after a throttle change - the scales below absorb those.

ROBUSTNESS OF THE DEGRADATION INDEX. At redline the physics faults four channel
groups at once (oil-pressure drift, oil-temperature bias, noise on all three
vibration axes, EGT drift). A plain median of five groups then breaks: the first
version pinned the index at 1.0 on a 4%-worn engine and put a 24 C false residual
on a perfectly healthy CHT. So a group only votes while its own channels are not
deviating or saturated, the index holds when fewer than two groups can vote, and
it can move at most WEAR_RATE_MAX per second - wear is physically slow, so a sudden
jump is always a sensor fault, never the engine.

Engine failure modes also move individual channels (cooling degradation -> CHT and
oil temperature, injector fouling -> EGT, ...). This module reports the deviation;
advisory.py decides whether the AI's failure-mode head explains it or whether the
sensor itself is suspect.

Only physics v3 telemetry is supported - v2's thermal model is mis-calibrated.
"""
import math
import statistics
from collections import deque

CHANNELS = ["egt", "cht", "oil_pressure", "oil_temp", "vibx", "viby", "vibz", "rpm"]
LABELS = {"egt": "EGT", "cht": "CHT", "oil_pressure": "Oil pressure", "oil_temp": "Oil temp",
          "vibx": "Vib X", "viby": "Vib Y", "vibz": "Vib Z", "rpm": "RPM"}
UNITS = {"egt": "C", "cht": "C", "oil_pressure": "psi", "oil_temp": "C",
         "vibx": "g", "viby": "g", "vibz": "g", "rpm": "rpm"}

# ---- physics v3 constants (backend/physics.py) --------------------------------
EGT_TAU, CHT_TAU, OILTEMP_TAU = 6.0, 50.0, 100.0
WEAR_EGT_RISE_C, WEAR_CHT_RISE_C, WEAR_OILTEMP_RISE_C = 45.0, 25.0, 10.0
# physics.py OILT_COEF (v3): base, throttle, rpm, airspeed cooling, ISA.
OILT_COEF = (70.0, 30.0, 0.006, 0.3, 0.5)
WEAR_OIL_PRESS_FRAC, WEAR_VIB_FRAC = 0.30, 1.50
VIB_COEF = {"vibx": (0.02, 0.00345), "viby": (0.025, 0.00310), "vibz": (0.03, 0.00379)}
# physics.py SENSOR_LIMITS (v3). A reading pinned at a stop says nothing about the
# true value beyond it, so it is reported as "saturated", never as a deviation.
SENSOR_LIMITS = {"egt": (0.0, 1000.0), "cht": (0.0, 200.0), "oil_pressure": (0.0, 150.0),
                 "oil_temp": (0.0, 180.0), "vibx": (0.0, 1.0), "viby": (0.0, 1.0),
                 "vibz": (0.0, 1.0), "rpm": (0.0, 8000.0)}

# ---- detection settings --------------------------------------------------------
# One unit of z per channel: sensor-tolerance sized and above the live transients
# measured after throttle changes. Scored by validation/residual_eval.py.
SCALE = {"egt": 1.5, "cht": 1.0, "oil_pressure": 0.25, "oil_temp": 0.5,
         "vibx": 0.0005, "viby": 0.0005, "vibz": 0.0005, "rpm": 5.0}
Z_DEVIATION = 4.0      # |median z over the last 5 samples| for bias / drift
NOISE_Z = 3.0          # residual std over the window, in scale units
DRIFT_Z = 2.0          # fitted change across the window, in scale units
WINDOW = 30            # samples (seconds at the AI cadence)
RPM_WINDOW = 5         # the physics glitches the RPM sender ~10% of seconds; a 30 s
                       # window would keep RPM flagged almost permanently
STUCK_SAMPLES = 5      # identical consecutive readings while physics says the value moved
MIN_SAMPLES = 10       # before any status other than "settling"
WEAR_RATE_MAX = 0.002  # per second; ~10x the fastest wear seen in live physics
MIN_VOTING_GROUPS = 2
ZERO_CHANNELS = ("egt", "cht", "oil_temp", "oil_pressure")   # steady offsets; not vibration (gain) or rpm
ZERO_MIN_SAMPLES = 30

_VOTE_GROUPS = {"egt": ("egt",), "cht": ("cht",), "oil_temp": ("oil_temp",),
                "oil_pressure": ("oil_pressure",), "vib": ("vibx", "viby", "vibz")}


def _slope(values):
    """Least-squares slope per sample."""
    n = len(values)
    if n < 3:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(values) / n
    num = sum((i - mx) * (v - my) for i, v in enumerate(values))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


class ResidualMonitor:
    """Stateful: carries the thermal lag states and a rolling window per channel.

    lag="exp" matches live physics integrated at 100 Hz and sampled once a second
    (main.py). lag="euler" matches the v3 dataset, generated at dt = 1 s.
    Call reset() whenever a new flight or engine starts.
    """

    def __init__(self, lag="exp", scales=None, offsets=None):
        if lag not in ("exp", "euler"):
            raise ValueError("lag must be 'exp' or 'euler'")
        self.lag = lag
        self.scales = dict(SCALE, **(scales or {}))
        # Per-installation sensor calibration (see zero()). Subtracted from the measured
        # value before anything else, and kept across reset() - it belongs to the sensors.
        self.offsets = dict(offsets or {})
        self._zeroing = None
        self.reset()

    def begin_zeroing(self, known_wear, samples=ZERO_MIN_SAMPLES):
        """Calibrate out steady sender offsets on a known-healthy ground run.

        Temperature and oil-pressure senders are routinely a few units off at installation.
        Uncalibrated, the thermal groups read that as wear, and the wrong wear then drags the
        healthy channels' expected values with it. Residuals taken mid-flight are already
        contaminated by that, so zeroing runs as its own mode: wear is FIXED at the engine's
        known value (its hour meter), temperatures are compared with the steady physics value
        (the engine must have held one operating point for several minutes - oil temperature
        lags ~100 s), and after `samples` readings the median residual of each zeroable channel
        becomes its offset. Zeroing over a real sensor fault would hide it: known-healthy only.
        """
        self.reset()
        self._zeroing = {"wear": float(known_wear), "left": int(samples),
                         "res": {c: [] for c in ZERO_CHANNELS}}

    @property
    def zeroing(self):
        return self._zeroing is not None

    def reset(self):
        self.n = 0
        self._thermal = None
        self._w_hat = 0.0
        self._hist = {c: deque(maxlen=WINDOW) for c in CHANNELS}
        self._votes = {g: deque(maxlen=WINDOW) for g in _VOTE_GROUPS}
        self._last_status = {c: "settling" for c in CHANNELS}

    def _alpha(self, dt, tau):
        return 1.0 - math.exp(-dt / tau) if self.lag == "exp" else min(1.0, dt / tau)

    def update(self, t, dt=1.0):
        if not t or t.get("physics_version") != "v3":
            return {"enabled": False, "reason": "physics residuals need physics v3 telemetry"}
        try:
            thr, rpm, kw = float(t["throttle"]), float(t["engine_rpm"]), float(t["power_kw"])
            cool = min(max(float(t["airspeed"]), 0.0), 60.0)
            isa = float(t.get("isa_dev_c") or 0.0)
            meas = {c: float(t[c]) for c in CHANNELS if c != "rpm"}
            meas["rpm"] = float(t["rpm_fault"])
            for c, off in self.offsets.items():
                meas[c] -= off
        except (KeyError, TypeError, ValueError) as e:
            return {"enabled": False, "reason": f"telemetry missing or invalid: {e}"}

        saturated = {c: not (SENSOR_LIMITS[c][0] + 1e-6 < meas[c] < SENSOR_LIMITS[c][1] - 1e-6)
                     for c in CHANNELS}
        target = {"egt": 350.0 + 250.0*thr + 0.03*rpm + 1.2*kw + 0.6*isa,
                  "cht": 75.0 + 30.0*thr + 0.004*rpm + 0.25*kw - 0.35*cool + 0.7*isa,
                  "oil_temp": OILT_COEF[0] + OILT_COEF[1]*thr + OILT_COEF[2]*rpm - OILT_COEF[3]*cool + OILT_COEF[4]*isa}
        rk = (rpm / 1000.0) ** 2
        vib0 = {c: a + b*rk for c, (a, b) in VIB_COEF.items()}

        def oil_p0(oil_temp):
            return 90.0*(1.0 - math.exp(-rpm/1500.0)) + 3.0*thr - 0.15*(oil_temp - 80.0)

        if self._zeroing is not None:
            z = self._zeroing
            w = z["wear"]
            exp_z = {"egt": target["egt"] + WEAR_EGT_RISE_C*w, "cht": target["cht"] + WEAR_CHT_RISE_C*w,
                     "oil_temp": target["oil_temp"] + WEAR_OILTEMP_RISE_C*w}
            exp_z["oil_pressure"] = oil_p0(target["oil_temp"] + WEAR_OILTEMP_RISE_C*w) * (1.0 - WEAR_OIL_PRESS_FRAC*w)
            for c in ZERO_CHANNELS:
                if not saturated[c]:
                    z["res"][c].append(meas[c] - exp_z[c])
            z["left"] -= 1
            if z["left"] > 0:
                return {"enabled": True, "zeroing": True, "samples_left": z["left"], "deviations": [],
                        "saturated": [], "channels": {}, "degradation_index": w}
            for c, vals in z["res"].items():
                if vals:
                    self.offsets[c] = round(self.offsets.get(c, 0.0) + statistics.median(vals), 4)
            self._zeroing = None
            self.reset()
            return {"enabled": True, "zeroing": False, "zeroed": dict(self.offsets), "deviations": [],
                    "saturated": [], "channels": {}, "degradation_index": w}

        def raw_votes(th, oilp_healthy):
            v = {"oil_pressure": ((1.0 - meas["oil_pressure"]/oilp_healthy)/WEAR_OIL_PRESS_FRAC
                                  if oilp_healthy > 1e-6 else None)}
            v.update({c: (meas[c]/vib0[c] - 1.0)/WEAR_VIB_FRAC for c in VIB_COEF})
            if th is not None:
                v["egt"] = (meas["egt"] - th["egt"]) / WEAR_EGT_RISE_C
                v["cht"] = (meas["cht"] - th["cht"]) / WEAR_CHT_RISE_C
                v["oil_temp"] = (meas["oil_temp"] - th["oil_temp"]) / WEAR_OILTEMP_RISE_C
            return v

        def usable(c, value):
            return (value is not None and not saturated[c]
                    and self._last_status[c] not in ("deviation", "saturated")
                    and -0.5 <= value <= 1.5)          # outside this no wear can explain it

        if self._thermal is None:
            # Start in step with the engine: oil pressure and vibration have no lag,
            # so they give a wear estimate immediately; remove that wear from the
            # measured temperatures to seed the HEALTHY lag states. No settling
            # transient, and resuming a worn engine works the same way.
            v = raw_votes(None, oil_p0(meas["oil_temp"]))
            seeds = [v[c] for c in ("oil_pressure", "vibx", "viby", "vibz") if usable(c, v[c])]
            w0 = min(1.0, max(0.0, statistics.median(seeds))) if seeds else 0.0
            rise = {"egt": WEAR_EGT_RISE_C, "cht": WEAR_CHT_RISE_C, "oil_temp": WEAR_OILTEMP_RISE_C}
            self._thermal = {k: meas[k] - rise[k]*w0 for k in target}
            self._w_hat = w0
        else:
            for k, tau in (("egt", EGT_TAU), ("cht", CHT_TAU), ("oil_temp", OILTEMP_TAU)):
                self._thermal[k] += self._alpha(dt, tau) * (target[k] - self._thermal[k])

        th = self._thermal
        oilp_healthy = oil_p0(th["oil_temp"] + WEAR_OILTEMP_RISE_C*self._w_hat)
        v = raw_votes(th, oilp_healthy)
        voting = []
        for group, chans in _VOTE_GROUPS.items():
            vals = [v[c] for c in chans if usable(c, v[c])]
            if group == "vib" and len(vals) < 2:
                continue            # one clean axis out of three is not a group consensus
            if vals:
                self._votes[group].append(statistics.median(vals))
                voting.append(group)
        if len(voting) >= MIN_VOTING_GROUPS:
            proposal = statistics.median(statistics.median(self._votes[g]) for g in voting)
            step = WEAR_RATE_MAX * dt
            self._w_hat = min(self._w_hat + step, max(self._w_hat - step, proposal))
        w = self._w_hat = min(1.0, max(0.0, self._w_hat))

        expected = {
            "egt": th["egt"] + WEAR_EGT_RISE_C*w,
            "cht": th["cht"] + WEAR_CHT_RISE_C*w,
            "oil_temp": th["oil_temp"] + WEAR_OILTEMP_RISE_C*w,
            "oil_pressure": oilp_healthy*(1.0 - WEAR_OIL_PRESS_FRAC*w),
            **{c: vib0[c]*(1.0 + WEAR_VIB_FRAC*w) for c in VIB_COEF},
            "rpm": rpm,
        }
        self.n += 1

        channels, deviations, saturated_list = {}, [], []
        for c in CHANNELS:
            r = meas[c] - expected[c]
            self._hist[c].append((meas[c], r, expected[c]))
            if saturated[c]:
                status, signature = "saturated", None
                saturated_list.append(c)
            else:
                status, signature = self._classify(c)
            if status == "deviation":
                deviations.append(c)
            self._last_status[c] = status
            nd = 4 if c.startswith("vib") else 2
            channels[c] = {"label": LABELS[c], "unit": UNITS[c],
                           "measured": round(meas[c], nd), "expected": round(expected[c], nd),
                           "residual": round(r, nd), "z": round(r / self.scales[c], 2),
                           "status": status, "signature": signature}
        return {"enabled": True, "samples": self.n,
                "degradation_index": round(w, 4),
                "voting_groups": voting,
                "deviations": deviations, "saturated": saturated_list, "channels": channels}

    def _classify(self, c):
        """(status, signature). status: settling | ok | deviation."""
        if self.n < MIN_SAMPLES:
            return "settling", None
        hist = list(self._hist[c])
        if c == "rpm":
            hist = hist[-RPM_WINDOW:]
        sc = self.scales[c]
        m = [x[0] for x in hist]
        r = [x[1] for x in hist]
        e = [x[2] for x in hist]
        recent = r[-5:]
        # Identical readings alone are NOT a stuck sensor: at steady cruise physics
        # itself repeats values exactly (vibration is a pure function of rpm, and the
        # thermal lags reach a float fixed point). Stuck means the reading froze
        # while the physics expectation moved - at true steady state a frozen
        # sensor is indistinguishable from a healthy one, and is not claimed.
        if (len(m) >= STUCK_SAMPLES and max(m[-STUCK_SAMPLES:]) == min(m[-STUCK_SAMPLES:])
                and max(e[-STUCK_SAMPLES:]) - min(e[-STUCK_SAMPLES:]) >= sc):
            return "deviation", "stuck"
        if statistics.pstdev(r) / sc >= NOISE_Z and abs(statistics.median(r)) / sc < Z_DEVIATION:
            return "deviation", "noise"
        if abs(statistics.median(recent)) / sc >= Z_DEVIATION:
            drifting = abs(_slope(r)) * (len(r) - 1) / sc >= DRIFT_Z
            return "deviation", ("drift" if drifting else "bias")
        if abs(r[-1]) / sc >= Z_DEVIATION:
            return "deviation", "spike"
        return "ok", None


# =============================================================================
# Physics v4: residuals come from the on-board twin (twin_v4.py), not from the
# hand-written expected-value formulas above - the twin IS the expected value, run
# on the same inputs with a perfect engine. This only packages them for the UI and
# the advisory: each channel in units of that sensor's own noise.
# =============================================================================
V4_DEVIATION_Z = 5.0      # |residual| in sensor-noise units that counts as a deviation


def twin_residuals_v4(out: dict, channels_: list | None = None) -> dict:
    """v4 / v5: residuals against the on-board twin, in noise sigmas. channels_: the
    twin's residual channels (v5 has 14); default the v4 six."""
    from sensors_v5 import SENSOR_SPEC          # v4's instruments plus the two turbo ones
    from twin_v4 import RESIDUAL_CHANNELS
    twin = out.get("twin") or {}
    channels = {}
    for c in channels_ or RESIDUAL_CHANNELS:
        r = out.get(f"res_{c}")
        if not isinstance(r, (int, float)):
            continue
        z = r / SENSOR_SPEC[c]["noise_sd"]
        channels[c] = {"measured": out.get(c), "twin": twin.get(c), "residual": round(r, 4),
                       "z": round(z, 2)}
    return {"enabled": True, "version": "v4", "zeroing": False, "channels": channels,
            "deviations": [c for c, v in channels.items() if abs(v["z"]) >= V4_DEVIATION_Z],
            "saturated": []}
