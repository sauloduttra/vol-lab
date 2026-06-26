"""GARCH(1,1) -- the core of vol-lab (Bollerslev 1986).

The conditional variance follows an AR(1) in sigma^2 with persistence
pi = alpha + beta:

    sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}

`GARCHParams` exposes the closed-form moments and boundaries.  Two of them
-- `unconditional_variance` and `half_life` -- are GUARDED: they raise
ValueError when alpha+beta >= 1, because the bare formulas do NOT error on
their own (they would return +inf / a meaningless negative number).  The
fourth moment exists only when 1-(alpha+beta)^2-2*alpha^2 > 0, so
`kurtosis`/`excess_kurtosis` raise when it doesn't -- a model can be
covariance-stationary (alpha+beta<1) yet have no finite kurtosis.

The variance filter is seeded at the unconditional variance (stationary
case) so it stays >= omega > 0 for every step; positivity would otherwise
fail at t=0 under a backcast seed of eps[0]^2 = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["GARCHParams", "filter_variance", "acf_squared", "news_impact"]


@dataclass(frozen=True)
class GARCHParams:
    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def is_stationary(self) -> bool:
        return self.alpha + self.beta < 1.0

    @property
    def unconditional_variance(self) -> float:
        """omega / (1 - alpha - beta).  Raises if alpha+beta >= 1."""
        if self.persistence >= 1.0:
            raise ValueError(
                "unconditional variance undefined: alpha+beta >= 1 "
                f"(persistence={self.persistence})")
        return self.omega / (1.0 - self.persistence)

    @property
    def half_life(self) -> float:
        """Half-life of a variance shock: log(0.5)/log(alpha+beta).
        Raises if alpha+beta >= 1 (no mean reversion)."""
        if self.persistence >= 1.0:
            raise ValueError(
                "half-life undefined: alpha+beta >= 1 (no mean reversion)")
        return math.log(0.5) / math.log(self.persistence)

    @property
    def kurtosis_exists(self) -> bool:
        return 1.0 - self.persistence ** 2 - 2.0 * self.alpha ** 2 > 0.0

    @property
    def excess_kurtosis(self) -> float:
        denom = 1.0 - self.persistence ** 2 - 2.0 * self.alpha ** 2
        if denom <= 0.0:
            raise ValueError(
                "4th moment does not exist: 1-(alpha+beta)^2-2*alpha^2 <= 0")
        return 6.0 * self.alpha ** 2 / denom

    @property
    def kurtosis(self) -> float:
        return 3.0 + self.excess_kurtosis


def filter_variance(params: GARCHParams, eps, sigma2_0: float | None = None) -> np.ndarray:
    """Conditional-variance path sigma2_0..sigma2_{n-1} (sigma2_t = Var(eps_t)).

    sigma2_0 is the seed; sigma2_t = omega + alpha*eps_{t-1}^2 + beta*sigma2_{t-1}.
    Default seed = unconditional variance (stationary) else sample var(eps);
    the seed must be strictly positive.
    """
    eps = np.asarray(eps, dtype=float).ravel()
    n = eps.size
    if sigma2_0 is None:
        sigma2_0 = params.unconditional_variance if params.is_stationary \
            else float(np.var(eps))
    if sigma2_0 <= 0.0:
        raise ValueError("sigma2_0 must be strictly positive")
    out = np.empty(n)
    out[0] = sigma2_0
    w, a, b = params.omega, params.alpha, params.beta
    for t in range(1, n):
        e = eps[t - 1]
        out[t] = w + a * e * e + b * out[t - 1]
    return out


def acf_squared(params: GARCHParams, k: int) -> float:
    """Autocorrelation of eps^2 at lag k (volatility clustering).

        rho_1 = alpha*(1 - beta*(alpha+beta)) / (1 - (alpha+beta)^2 + alpha^2)
        rho_k = (alpha+beta)^(k-1) * rho_1      for k >= 2

    Requires the 4th moment to exist.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not params.kurtosis_exists:
        raise ValueError("eps^2 ACF undefined: 4th moment does not exist")
    a, p = params.alpha, params.persistence
    rho_1 = a * (1.0 - params.beta * p) / (1.0 - p ** 2 + a ** 2)
    if k == 1:
        return rho_1
    return p ** (k - 1) * rho_1


def news_impact(params: GARCHParams, eps, sigma2_level: float | None = None):
    """Symmetric news-impact curve omega + alpha*eps^2 + beta*sigma2_level.
    sigma2_level defaults to the unconditional variance."""
    if sigma2_level is None:
        sigma2_level = params.unconditional_variance
    eps = np.asarray(eps, dtype=float)
    return params.omega + params.alpha * eps ** 2 + params.beta * sigma2_level
