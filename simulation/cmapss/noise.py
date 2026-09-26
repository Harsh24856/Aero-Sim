"""A9 - the multi-stage noise chain, in the order the paper specifies.

THE PAPER'S CHAIN (section V.B, p.6)
    1. per-unit trajectory parameters drawn from a distribution
    2. "This trajectory was then masked by a mixture of two random distributions
       with slightly different variances."
    3. "This contaminated trajectory was fed to the engine model simulation" -
       so process noise passes THROUGH the system before being observed
    4. "Lastly, a random measurement noise component was added to all output
       channels."

    Order matters and is preserved here: process noise perturbs the health state
    BEFORE the response surface; measurement noise is added AFTER it.

TWO PROPERTIES MEASURED FROM THE REAL DATA, NOT ASSUMED
    Fitting the FD001 response surface and examining its residuals shows:

      mean lag-1 autocorrelation  +0.259   (0 would mean white noise)
      mean excess kurtosis         6.98    (3.0 would mean a single Gaussian)

    Both are exactly what the paper describes: noise filtered through system
    dynamics acquires autocorrelation, and a two-component mixture produces
    heavy tails. So rather than assuming a noise shape we measure, per sensor:
      - an AR(1) coefficient  phi   (the "filtered through system dynamics" part)
      - variance and kurtosis, from which a two-component mixture is recovered

    Per-sensor spread is wide and physically sensible: the core-speed channels
    Nc/NRc are strongly autocorrelated (phi ~ 0.8, slow-moving shaft speeds)
    while the temperatures T24/T30 are nearly white (phi ~ 0.05-0.07).

THE ONE ASSUMPTION
    The paper says "a mixture of two" but never gives the mixing weight, so the
    system is underdetermined by variance and kurtosis alone. We fix the wide
    component's weight at MIXTURE_WEIGHT and solve for the two variances. The
    value is recorded here and marked ASSUMED; the resulting kurtosis is checked
    against the target, so a bad choice shows up as a miss rather than passing
    silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Weight of the wide ("contaminating") mixture component. ASSUMED - the paper
# specifies a two-component mixture but not its weight. 0.1 sits in the usual
# contamination range and can represent kurtosis up to several hundred, which
# covers every channel observed (max ~45, P15).
MIXTURE_WEIGHT = 0.1

MIXTURE_PROVENANCE = {
    "status": "ASSUMED",
    "parameter": "mixture weight p",
    "value": MIXTURE_WEIGHT,
    "reason": "Saxena et al. specify 'a mixture of two random distributions with "
              "slightly different variances' but give no mixing weight; variance "
              "and kurtosis alone leave the system underdetermined.",
    "checked_by": "realised kurtosis is compared against the per-sensor target",
}


def solve_mixture(var: float, kurt: float, p: float = MIXTURE_WEIGHT) -> tuple[float, float]:
    """Recover two component variances from a target variance and kurtosis.

    For a zero-mean mixture (1-p).N(0,u) + p.N(0,w):
        var = (1-p)u + p w
        m4  = 3[(1-p)u^2 + p w^2],  kurt = m4 / var^2

    Eliminating u gives a quadratic in w:
        p w^2 - 2 var p w + (var^2 - A(1-p)) = 0,    A = kurt.var^2/3

    Returns (u, w) with w >= u. Falls back to a single Gaussian when the target
    kurtosis is at or below 3, where no positive-variance mixture exists.
    """
    if var <= 0:
        return 0.0, 0.0
    kurt = max(float(kurt), 3.0)
    if kurt <= 3.0 + 1e-9:
        return var, var

    A = kurt * var * var / 3.0
    disc = var * var * p * p - p * (var * var - A * (1.0 - p))
    if disc <= 0:
        return var, var
    w = var + np.sqrt(disc) / p
    u = (var - p * w) / (1.0 - p)
    if u <= 0:     # kurtosis too extreme for this weight; widen gracefully
        u = var * 1e-3
        w = (var - (1.0 - p) * u) / p
    return float(u), float(w)


def measure(resid: np.ndarray, units: np.ndarray) -> dict:  # noqa: D401
    """Measure AR(1) coefficient, variance and kurtosis of within-unit residuals.

    Residuals are de-meaned per unit first, so the between-unit offset (handled
    separately as initial-wear variation) does not leak into the within-unit
    noise statistics.
    """
    df = pd.DataFrame({"u": units, "r": resid})
    df["r"] = df["r"] - df.groupby("u")["r"].transform("mean")

    acfs = []
    for _, g in df.groupby("u"):
        x = g["r"].to_numpy()
        if x.size > 5 and x.std() > 0:
            acfs.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
    phi = float(np.clip(np.mean(acfs), 0.0, 0.95)) if acfs else 0.0

    x = df["r"].to_numpy()
    var = float(x.var())
    kurt = float(((x - x.mean()) ** 4).mean() / (x.var() ** 2)) if x.var() > 0 else 3.0

    # SMOOTH/WHITE DECOMPOSITION - why a single AR(1) is not enough.
    #
    # Matching a channel's variance and its lag-1 autocorrelation still pins its
    # cycle-to-cycle STEP SIZE, and that step size came out wrong: a classifier
    # two-sample test separated real from generated at AUC 1.000 and named the
    # mean absolute first difference of the core-speed channels as the reason
    # (NRc 3.57 real against 8.35 generated).
    #
    # The cause is that real within-unit noise is two components in proportions
    # that differ enormously by channel. Writing the residual as smooth + white
    # and estimating the white part from Var(diff)/2 - exact when the smooth
    # part changes little between consecutive cycles - gives:
    #     Nc   8.6% white     NRc  6.1% white
    #     T50 86.7% white     T24 94.0% white
    # So Nc is almost entirely slow drift and moves very little cycle to cycle,
    # while T24 is almost entirely white and moves a lot. One AR(1) cannot be
    # both. Splitting the two lets variance, autocorrelation AND step size be
    # matched at once.
    d = np.concatenate([np.diff(g["r"].to_numpy()) for _, g in df.groupby("u")
                        if len(g) > 1]) if len(df) else np.zeros(1)
    var_white = float(min(max(d.var() / 2.0, 0.0), var)) if d.size else 0.0
    var_smooth = float(max(var - var_white, 0.0))

    # THE SMOOTH COMPONENT'S TIMESCALE, not just its size.
    #
    # Splitting smooth from white fixed neither the step size nor the spread,
    # because an AR(1) at phi ~ 0.8 decorrelates in about five cycles and so
    # spends its whole variance INSIDE a short window. Real Nc does the
    # opposite: its within-30-cycle sd is 4.89 while its overall sd is 22, i.e.
    # nearly all of its variance is slow drift across the full life. Generated
    # Nc had within-window sd 13.47 - the right total variance distributed over
    # completely the wrong timescale, which a discriminator sees instantly.
    #
    # So we measure how much of the variance survives inside a window, and later
    # solve the AR(1) coefficient to reproduce that ratio (see window_ratio_phi).
    W = 30
    wv = []
    for _, g in df.groupby("u"):
        x = g["r"].to_numpy()
        for i in range(W, len(x) + 1, max(1, W // 3)):
            wv.append(x[i - W:i].var())
    within_var = float(np.mean(wv)) if wv else var
    ratio = float(np.clip(within_var / var, 1e-6, 1.0)) if var > 0 else 1.0

    return {"phi": phi, "var": var, "kurtosis": kurt,
            "var_white": var_white, "var_smooth": var_smooth,
            "within_window_var": within_var, "window_var_ratio": ratio,
            "window": W}


def _ar1_window_ratio(rho: float, W: int) -> float:
    """Fraction of an AR(1)'s variance that survives inside a window of W.

    For lag correlation rho^k, the window mean absorbs S/W^2 of the variance
    where S = W + 2.sum_{k=1}^{W-1}(W-k).rho^k, leaving 1 - S/W^2 within it.
    Decreasing in rho: a slower process puts more of itself into the window mean
    and less into within-window scatter.
    """
    k = np.arange(1, W)
    S = W + 2.0 * np.sum((W - k) * rho ** k)
    return float(max(1.0 - S / (W * W), 1e-9))


def window_ratio_phi(target_ratio: float, W: int = 30,
                     iters: int = 60) -> float:
    """AR(1) coefficient whose within-window variance fraction is `target_ratio`."""
    lo, hi = 0.0, 0.999999
    if _ar1_window_ratio(lo, W) <= target_ratio:
        return 0.0
    if _ar1_window_ratio(hi, W) >= target_ratio:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _ar1_window_ratio(mid, W) > target_ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mixture_innovations(n: int, u: float, w: float, rng: np.random.Generator,
                        p: float = MIXTURE_WEIGHT) -> np.ndarray:
    """Draw n samples from the two-component zero-mean Gaussian mixture."""
    wide = rng.random(n) < p
    out = np.empty(n)
    nw = int(wide.sum())
    if nw:
        out[wide] = rng.normal(0.0, np.sqrt(w), nw)
    if n - nw:
        out[~wide] = rng.normal(0.0, np.sqrt(u), n - nw)
    return out


def two_component_noise(n: int, phi_smooth: float, var_smooth: float,
                        var_white: float, kurt: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Slow AR(1) drift plus white noise - the real within-unit structure.

    The smooth part carries the variance and the autocorrelation; the white part
    carries the cycle-to-cycle movement. Their proportions are measured per
    channel (see `measure`), so a slow channel like Nc and a jumpy one like T24
    are both reproduced by the same code.
    """
    if n <= 0:
        return np.zeros(0)
    out = np.zeros(n)
    if var_smooth > 0:
        out += ar1_mixture_noise(n, phi_smooth, var_smooth, kurt, rng)
    if var_white > 0:
        u, w = solve_mixture(var_white, kurt)
        out += mixture_innovations(n, u, w, rng)
    return out


def ar1_mixture_noise(n: int, phi: float, var: float, kurt: float,
                      rng: np.random.Generator) -> np.ndarray:
    """AR(1) process driven by two-component mixture innovations.

    The innovation variance is scaled by (1 - phi^2) so the stationary variance
    of the output equals `var` - otherwise autocorrelation would inflate the
    spread and every channel would come out noisier than the real data.
    """
    if n <= 0:
        return np.zeros(0)
    innov_var = var * (1.0 - phi * phi)
    u, w = solve_mixture(innov_var, kurt)
    eps = mixture_innovations(n, u, w, rng)

    out = np.empty(n)
    # Start from the stationary distribution rather than 0, so early cycles are
    # not systematically quieter than late ones.
    out[0] = rng.normal(0.0, np.sqrt(var))
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def process_noise(n: int, scale: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Correlated perturbation applied to the health trajectory BEFORE the
    response surface (step 2-3 of the paper's chain).

    This is what makes efficiency and flow loss "not locally monotonic" (p.5):
    between-flight maintenance is not modelled explicitly but is absorbed here,
    letting the engine's condition improve slightly at any point.
    """
    if n <= 0 or scale <= 0:
        return np.zeros(max(n, 0))
    u, w = solve_mixture(scale * scale * (1.0 - phi * phi), 6.0)
    eps = mixture_innovations(n, u, w, rng)
    out = np.empty(n)
    out[0] = rng.normal(0.0, scale)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out
