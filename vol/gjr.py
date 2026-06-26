"""GJR-GARCH -- asymmetric leverage (Glosten, Jagannathan & Runkle 1993).

A negative shock loads an extra gamma onto the variance:

    sigma2_t = omega + (alpha + gamma*1[eps_{t-1}<0]) * eps2_{t-1} + beta*sigma2_{t-1}

so bad news raises future variance more than good news of the same size
(the leverage effect).  Under SYMMETRIC innovations E[1{eps<0}eps^2] =
(1/2)E[eps^2], so the effective persistence is alpha + beta + gamma/2 (the
/2 is the adversarial catch -- it is NOT alpha+beta+gamma).

The filter is written so that gamma == 0 dispatches through the SAME
arithmetic as the GARCH(1,1) filter, giving bit-for-bit model nesting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vol.garch import GARCHParams

__all__ = ["GJRParams", "filter_variance", "news_impact", "to_garch"]


@dataclass(frozen=True)
class GJRParams:
    omega: float
    alpha: float
    beta: float
    gamma: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta + self.gamma / 2.0

    @property
    def is_stationary(self) -> bool:
        return self.persistence < 1.0

    @property
    def unconditional_variance(self) -> float:
        """omega / (1 - alpha - beta - gamma/2).  Raises if persistence >= 1."""
        if not self.is_stationary:
            raise ValueError(
                "unconditional variance undefined: alpha+beta+gamma/2 >= 1")
        return self.omega / (1.0 - self.persistence)

    @property
    def leverage_ratio(self) -> float:
        """(alpha + gamma) / alpha: downside vs upside news-impact slope."""
        return (self.alpha + self.gamma) / self.alpha


def filter_variance(params: GJRParams, eps, sigma2_0: float | None = None) -> np.ndarray:
    """GJR conditional-variance path.  At gamma==0 this is bit-identical to
    `garch.filter_variance(GARCHParams(omega,alpha,beta), eps, sigma2_0)`."""
    eps = np.asarray(eps, dtype=float).ravel()
    n = eps.size
    if sigma2_0 is None:
        sigma2_0 = params.unconditional_variance if params.is_stationary \
            else float(np.var(eps))
    if sigma2_0 <= 0.0:
        raise ValueError("sigma2_0 must be strictly positive")
    out = np.empty(n)
    out[0] = sigma2_0
    w, a, b, g = params.omega, params.alpha, params.beta, params.gamma
    for t in range(1, n):
        e = eps[t - 1]
        ind = 1.0 if e < 0.0 else 0.0
        out[t] = w + (a + g * ind) * e * e + b * out[t - 1]
    return out


def news_impact(params: GJRParams, eps, sigma2_level: float | None = None):
    """Asymmetric news-impact curve: slope (alpha+gamma) for eps<0, alpha for
    eps>=0.  sigma2_level defaults to the unconditional variance."""
    if sigma2_level is None:
        sigma2_level = params.unconditional_variance
    eps = np.asarray(eps, dtype=float)
    ind = (eps < 0.0).astype(float)
    return params.omega + (params.alpha + params.gamma * ind) * eps ** 2 \
        + params.beta * sigma2_level


def to_garch(params: GJRParams) -> GARCHParams:
    """Collapse to GARCH(1,1) -- valid only when gamma == 0."""
    if params.gamma != 0.0:
        raise ValueError("to_garch is valid only when gamma == 0")
    return GARCHParams(params.omega, params.alpha, params.beta)
