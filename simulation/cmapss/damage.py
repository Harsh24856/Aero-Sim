"""A6/A7 - damage propagation and health index, from the published equations.

This module is the part of the replication that is EXACT: every equation and
every constant comes from Saxena et al. PHM'08 (see config.py for citations).
Nothing here is fitted to the released data. The fitted part - the mapping from
a health state to 21 sensor readings - lives in surrogate.py and is kept
separate precisely so this boundary stays visible.

THE MODEL
    Per unit, efficiency and flow each degrade exponentially (eq. 6, p.5):

        e(t) = 1 - d_e - exp{a_e . t^(b_e)}
        f(t) = 1 - d_f - exp{a_f . t^(b_f)}

    with (a, b) sampled per unit from the published ranges and d set so the
    trajectory starts at the sampled initial deterioration.

    Those feed four operability margins, and health is their minimum (eq. 7-8):

        H(t) = min(m_fan, m_LPC, m_HPC, m_EGT)

    Failure is H = 0 (step 3, p.5). Trajectory length is therefore an OUTPUT of
    this model, never an input - which is what makes matching NASA's length
    distribution meaningful evidence rather than a tautology.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config as C
from . import noise as _N


@dataclass(frozen=True)
class UnitDamageParams:
    """The per-unit constants that fully determine one degradation trajectory.

    Sampling these and nothing else is what the paper describes: "the
    degradation trajectory parameters, a_k and b_k, corresponding to a unit
    under test k were chosen from a normal distribution. Together with e0 and
    f0, these parameters define a deterministic trajectory for degradation for a
    particular engine." (p.5)
    """
    unit: int
    e0: float          # initial efficiency,  eq. (9)
    f0: float          # initial flow,        eq. (9)
    a_e: float         # efficiency rate,     eq. (10)
    b_e: float         # efficiency exponent, eq. (10)
    a_f: float         # flow rate,           eq. (10)
    b_f: float         # flow exponent,       eq. (10)
    k: int             # fault-direction selector, eq. (10)

    def as_dict(self) -> dict:
        return asdict(self)


def _truncated_normal(rng, lo, hi, size=None):
    """Draw from a normal centred on [lo, hi] and truncated to it.

    The paper says the trajectory parameters were "chosen from a normal
    distribution" (p.5) while eq. (10) constrains them to closed intervals. A
    normal centred on the interval with sigma = range/4 puts ~95% of mass inside
    before truncation, which satisfies both statements without inventing a
    distribution shape the paper did not give. Resampling (rather than clipping)
    avoids piling probability mass on the endpoints.
    """
    mu = 0.5 * (lo + hi)
    sigma = (hi - lo) / 4.0
    n = 1 if size is None else int(np.prod(size))
    out = np.empty(n)
    filled = 0
    while filled < n:
        draw = rng.normal(mu, sigma, size=max(n - filled, 16))
        ok = draw[(draw >= lo) & (draw <= hi)]
        take = min(ok.size, n - filled)
        out[filled:filled + take] = ok[:take]
        filled += take
    return float(out[0]) if size is None else out.reshape(size)


def _correlated_pair(rng, lo, hi, rho):
    """Two draws from the same truncated normal with correlation `rho`.

    Built with a Gaussian copula and rejection so both members stay inside the
    published interval, rather than clipping, which would pile mass on the ends.
    """
    mu, sigma = 0.5 * (lo + hi), (hi - lo) / 4.0
    for _ in range(512):
        z1 = rng.normal()
        z2 = rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * rng.normal()
        x1, x2 = mu + sigma * z1, mu + sigma * z2
        if lo <= x1 <= hi and lo <= x2 <= hi:
            return float(x1), float(x2)
    return float(np.clip(x1, lo, hi)), float(np.clip(x2, lo, hi))


def sample_unit_params(unit: int, rng: np.random.Generator,
                       rho: float = 0.0) -> UnitDamageParams:
    """Sample one unit's degradation parameters within the published bounds.

    Enforces eq. (10)'s |f0 - e0| <= 1% coupling: flow is drawn relative to
    efficiency rather than independently, otherwise the pair can start up to 1%
    apart in the wrong direction and violate the constraint.

    `rho` COUPLES THE EFFICIENCY AND FLOW RATE PARAMETERS, and exists because
    the source is genuinely ambiguous about whether they are one parameter or
    two. Equation (6) writes a_e, b_e and a_f, b_f separately, while the text on
    p.5 describes "the degradation trajectory parameters, a_k and b_k,
    corresponding to a unit under test k" - singular, per unit.

    The two readings give measurably different lifetime spreads, because a
    margin mixes efficiency and flow loss and averaging two INDEPENDENT curves
    damps the variance that averaging two identical ones does not:

        rho = 0 (independent, eq. 6 read strictly)   cv 0.201
        rho = 1 (shared, the p.5 text read strictly) cv 0.264
        NASA FD001                                    cv 0.224

    NASA sits between them, so neither strict reading is right and the coupling
    is a real quantity to be recovered from the data. It is calibrated in
    calibrate.py against the lifetime standard deviation, and recorded there as
    a fitted parameter rather than a published one.
    """
    e0 = _truncated_normal(rng, *C.E0_RANGE)
    # Draw f0 within MAX_EF_DELTA of e0, still inside the published f0 range.
    lo = max(C.F0_RANGE[0], e0 - C.MAX_EF_DELTA)
    hi = min(C.F0_RANGE[1], e0 + C.MAX_EF_DELTA)
    f0 = float(rng.uniform(lo, hi))

    a_e, a_f = _correlated_pair(rng, *C.A_RANGE, rho)
    b_e, b_f = _correlated_pair(rng, *C.B_RANGE, rho)

    return UnitDamageParams(
        unit=unit, e0=e0, f0=f0,
        a_e=a_e, b_e=b_e, a_f=a_f, b_f=b_f,
        k=int(rng.choice(C.K_VALUES)),
    )


def degradation_curve(t: np.ndarray, init: float, a: float, b: float) -> np.ndarray:
    """Equation (5)/(6): one health-related index over cycles.

        x(t) = init - (exp{a . t^b} - 1)

    The paper writes d(t) = 1 - d - exp{a t^b} with d an additive initial
    deterioration term "allowing the data-generation process to start at an
    arbitrary point in the wear-space". Written literally that expression gives
    x(0) = -d, i.e. it starts near zero rather than near one, because exp(0) = 1
    is not subtracted off. The form above is the same curve shifted so that
    x(0) = init exactly, which is the behaviour the paper describes in words
    (eq. 9: "each health index trajectory starts with a number between 1 and
    0.99"). The decay term exp{a t^b} - 1 is untouched.
    """
    # Cap the exponent before exp(). Past the failure threshold the curve is
    # already far below zero and its exact value is irrelevant - the unit has
    # failed and simulate_unit() truncates there - but an uncapped exp()
    # overflows to inf for long max_cycles and floods the logs. 700 is just
    # under the float64 overflow point.
    x = np.minimum(a * np.power(t, b), 700.0)
    return init - (np.expm1(x))


def margins(e: np.ndarray, f: np.ndarray, k: int, gain: float = 1.0) -> dict[str, np.ndarray]:
    """The four operability margins, normalised to [0, 1] (section V.D, p.6).

    The paper does not publish the functional form of each margin - those come
    from the unreleased C-MAPSS thermodynamic model. What it does give is:
      - each margin is normalised so 1 = healthy and 0 = the margin has decayed
        by its specified limit (15% stall, ~2% EGT),
      - the margins are functions of efficiency e and flow f,
      - different margins respond differently to the same (e, f) pair, which is
        why the minimum is informative rather than always the same margin.

    We therefore model each margin as its loss consumed against its published
    limit, with a fixed sensitivity vector giving each margin a different
    response. The sensitivities are NOT published; they are the one modelling
    choice in this module and are recorded in SENSITIVITY below so they can be
    audited. They are constrained by the requirement that the resulting
    trajectory-length distribution matches NASA's (A16) - which is a real
    constraint, not a free parameter.

    `k` selects the fault direction (eq. 10): k=1 weights efficiency loss more,
    k=2 weights flow loss more, giving the two distinct degradation flavours the
    paper describes without changing any published constant.

    `gain` is the ONE calibrated scalar in this module - the conversion from
    efficiency/flow loss to margin loss. The paper gives the margin limits (15%
    stall, ~2% EGT) but not this conversion, because it is a property of the
    unreleased thermodynamic model. Left at 1.0 the published (a, b) ranges
    produce 20-150 cycle lives against NASA's 128-362, so the factor is real and
    must come from somewhere; see calibrate.py, which fits it to the NASA mean
    trajectory length and nothing else. The distribution SHAPE (sd, range, KS)
    is then an independent test, because one scalar cannot buy a shape.
    """
    e_loss = np.clip(1.0 - e, 0.0, None)
    f_loss = np.clip(1.0 - f, 0.0, None)

    # (efficiency weight, flow weight, limit) per margin.
    if k == 1:
        w = {"fan": (0.6, 0.4), "LPC": (0.7, 0.3), "HPC": (0.8, 0.2), "EGT": (0.9, 0.1)}
    else:
        w = {"fan": (0.4, 0.6), "LPC": (0.3, 0.7), "HPC": (0.2, 0.8), "EGT": (0.5, 0.5)}

    out = {}
    for name in C.MARGIN_NAMES:
        we, wf = w[name]
        limit = C.EGT_MARGIN_LIMIT if name == "EGT" else C.STALL_MARGIN_LIMIT
        consumed = gain * (we * e_loss + wf * f_loss) / limit
        out[name] = np.clip(1.0 - consumed, 0.0, 1.0)
    return out


# The margin sensitivities above, restated as data for auditing and for the
# assumptions doc. Marked ASSUMED: the paper gives the normalisation and the
# limits but not the per-margin response to (e, f).
SENSITIVITY_PROVENANCE = {
    "status": "ASSUMED",
    "reason": "C-MAPSS thermodynamic model unreleased; per-margin response to "
              "(efficiency, flow) is not published in Saxena et al. PHM'08.",
    "constrained_by": "trajectory-length distribution must match NASA (gate A16)",
}


# Process-noise defaults. ASSUMED - the paper describes masking the trajectory
# with a noise mixture (p.6) but gives no magnitude. Defined here rather than in
# the generator because CALIBRATION MUST SEE THE SAME NOISE THE GENERATOR ADDS:
# fitting the gain on a noise-free simulation and then generating with noise
# made FD001 calibrate to mean 206.3 / sd 49.7 and actually emit 189.9 / 36.7,
# because noise pushes health across the failure threshold early and clips the
# spread. Both paths now call simulate_unit with the same settings.
PROCESS_NOISE_SCALE = 0.01
PROCESS_NOISE_PHI = 0.8


def simulate_unit(params: UnitDamageParams, max_cycles: int = 1000,
                  gain: float = 1.0, rng: np.random.Generator | None = None,
                  process_scale: float = PROCESS_NOISE_SCALE,
                  process_phi: float = PROCESS_NOISE_PHI) -> dict:
    """Run one unit to failure and return its health trajectory.

    Returns arrays over cycles 1..N where N is the first cycle at which
    H <= FAILURE_THRESHOLD. N is emergent - no length is imposed.
    """
    t = np.arange(1, max_cycles + 1, dtype=float)
    e = degradation_curve(t, params.e0, params.a_e, params.b_e)
    f = degradation_curve(t, params.f0, params.a_f, params.b_f)

    # Process noise, applied to efficiency and flow BEFORE the margins are
    # evaluated, so it passes through the health calculation the way the paper's
    # noise passes through the engine model. Skipped when no rng is supplied.
    if rng is not None and process_scale > 0:
        e = e + _N.process_noise(max_cycles, process_scale, process_phi, rng)
        f = f + _N.process_noise(max_cycles, process_scale, process_phi, rng)

    m = margins(e, f, params.k, gain=gain)
    H = np.minimum.reduce([m[n] for n in C.MARGIN_NAMES])

    failed = np.flatnonzero(H <= C.FAILURE_THRESHOLD)
    if failed.size == 0:
        # Not a silent truncation: the caller must know the unit never failed,
        # because that would bias the length distribution if counted as a
        # complete run-to-failure trajectory.
        n = max_cycles
        reached_failure = False
    else:
        n = int(failed[0]) + 1
        reached_failure = True

    return {
        "unit": params.unit,
        "cycles": np.arange(1, n + 1, dtype=int),
        "efficiency": e[:n],
        "flow": f[:n],
        "health": H[:n],
        "margins": {k_: v[:n] for k_, v in m.items()},
        "length": n,
        "reached_failure": reached_failure,
        "params": params,
    }
