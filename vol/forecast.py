"""Multi-step variance forecasting + the seeded simulator.

The h-step forecast is the AR(1) telescoping of the sigma2 recursion,
mean-reverting to sigma2_bar at rate persistence^(h-1) (persistence = a+b for
GARCH, a+b+gamma/2 for GJR) -- the direct analog of tinystat's AR(1)
`chain_forecast` vs `chain_forecast_closed_form`.  The
closed form is computed INDEPENDENTLY from sigma2_bar and (alpha+beta)^(h-1),
never by re-running the recursion, so the two genuinely cross-check.

IGARCH/EWMA (persistence >= 1) needs a DEDICATED flat branch: the GARCH
closed form would evaluate sigma2_bar = omega/(1-(a+b)) = 0/0 -> nan, but the
true forecast is flat (= sigma2_{t+1} at every horizon).

`simulate` returns BOTH the eps path and the internal sigma2 path, so the
exact `filter_variance(eps, seed=sigma2[0]) == sigma2` identity (the real
swapped-lag / swapped-coefficient detector) can be tested.
"""
from __future__ import annotations

import math

import numpy as np

from vol.garch import GARCHParams
from vol.gjr import GJRParams

__all__ = [
    "forecast_recursive", "forecast_closed_form", "cumulative_variance",
    "annualize_vol", "simulate",
]


def forecast_recursive(params, sigma2_tp1: float, horizon: int) -> list:
    """Recursive multi-step variance forecast:
    f_1 = sigma2_{t+1}; f_{k+1} = omega + persistence * f_k
    (persistence = alpha+beta for GARCH, alpha+beta+gamma/2 for GJR)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    out = [float(sigma2_tp1)]
    for _ in range(1, horizon):
        out.append(params.omega + params.persistence * out[-1])
    return out


def forecast_closed_form(params, sigma2_tp1: float, horizon: int) -> list:
    """Closed-form multi-step forecast, derived independently of the recursion.

    Mean-reverting case (persistence < 1), from sigma2_bar and persistence^(h-1):

        f_h = sigma2_bar + persistence^(h-1) * (sigma2_{t+1} - sigma2_bar)

    IGARCH / explosive case (persistence >= 1) has no finite sigma2_bar, so a
    DEDICATED branch is used (the mean-reverting form would divide 0/0 at
    persistence == 1).  At persistence == 1 with omega == 0 (true EWMA) this
    is FLAT; with omega > 0 it grows LINEARLY as sigma2_{t+1} + (h-1)*omega;
    persistence > 1 uses the general geometric solution.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    p = params.persistence
    if p < 1.0:
        s2_bar = params.omega / (1.0 - p)
        return [s2_bar + p ** k * (sigma2_tp1 - s2_bar) for k in range(horizon)]
    if p == 1.0:
        return [sigma2_tp1 + k * params.omega for k in range(horizon)]
    return [p ** k * sigma2_tp1 + params.omega * (p ** k - 1.0) / (p - 1.0)
            for k in range(horizon)]


def cumulative_variance(params, sigma2_tp1: float, horizon: int) -> float:
    """Sum of the per-step variance forecasts over the horizon."""
    return float(sum(forecast_recursive(params, sigma2_tp1, horizon)))


def annualize_vol(sigma2_per_period: float, periods: int = 252) -> float:
    """Annualized volatility sqrt(periods * sigma2_per_period)."""
    return math.sqrt(periods * sigma2_per_period)


def simulate(params, n: int, rng, dist: str = "gaussian", nu: float | None = None,
             burn: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a GARCH(1,1) or GJR path; return (eps, sigma2), both length n.

    sigma2[t] is the conditional variance that generated eps[t], so the
    MATCHING filter reproduces it exactly: `garch.filter_variance` for
    GARCHParams, `gjr.filter_variance` for GJRParams (the GJR path needs
    gamma, which garch.filter_variance ignores).  The path is seeded at the
    unconditional variance (requires stationarity) and the first `burn` steps
    are discarded.
    """
    total = n + burn
    if dist == "gaussian":
        z = rng.standard_normal(total)
    elif dist == "student-t":
        if nu is None or nu <= 2.0:
            raise ValueError("student-t simulation needs nu > 2")
        z = rng.standard_t(nu, size=total) * math.sqrt((nu - 2.0) / nu)
    else:
        raise ValueError("dist must be 'gaussian' or 'student-t'")

    is_gjr = isinstance(params, GJRParams)
    w, a, b = params.omega, params.alpha, params.beta
    g = params.gamma if is_gjr else 0.0

    eps = np.empty(total)
    s2 = np.empty(total)
    sig2 = params.unconditional_variance
    for t in range(total):
        s2[t] = sig2
        e = math.sqrt(sig2) * z[t]
        eps[t] = e
        ind = 1.0 if (is_gjr and e < 0.0) else 0.0
        sig2 = w + (a + g * ind) * e * e + b * sig2
    return eps[burn:], s2[burn:]
