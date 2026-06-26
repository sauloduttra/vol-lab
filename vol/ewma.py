"""EWMA / RiskMetrics (J.P. Morgan/Reuters 1996).

EWMA is the IGARCH boundary -- GARCH(omega=0, alpha=1-lambda, beta=lambda)
with persistence exactly 1:

    sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * eps2_{t-1}

Being integrated, it has NO finite unconditional variance and a FLAT
multi-step forecast.  Its kernel is the geometric weights (1-lambda)*lambda^i,
whose partial sum 1 - lambda^n -> 1.  Unrolling the recursion exactly:

    sigma2_t = lambda^t * sigma2_0 + (1-lambda) * sum_{i=0}^{t-1} lambda^i * eps2_{t-1-i}

-- the decaying seed term lambda^t*sigma2_0 is part of the identity; the pure
truncated kernel matches the filter only after burn-in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vol.garch import GARCHParams

__all__ = [
    "EWMAParams", "filter_variance", "weights", "to_igarch",
    "RISKMETRICS_LAMBDA_DAILY", "RISKMETRICS_LAMBDA_MONTHLY",
]

RISKMETRICS_LAMBDA_DAILY = 0.94
RISKMETRICS_LAMBDA_MONTHLY = 0.97


@dataclass(frozen=True)
class EWMAParams:
    lam: float

    @property
    def alpha(self) -> float:
        return 1.0 - self.lam

    @property
    def beta(self) -> float:
        return self.lam

    @property
    def persistence(self) -> float:
        return 1.0

    @property
    def is_stationary(self) -> bool:
        return False

    @property
    def unconditional_variance(self) -> float:
        raise ValueError("EWMA/IGARCH has no finite unconditional variance")


def filter_variance(params: EWMAParams, eps, sigma2_0: float) -> np.ndarray:
    """EWMA conditional-variance path.  `sigma2_0` is REQUIRED (no
    unconditional variance to default to)."""
    eps = np.asarray(eps, dtype=float).ravel()
    n = eps.size
    if sigma2_0 <= 0.0:
        raise ValueError("sigma2_0 must be strictly positive")
    lam = params.lam
    out = np.empty(n)
    out[0] = sigma2_0
    for t in range(1, n):
        e = eps[t - 1]
        out[t] = lam * out[t - 1] + (1.0 - lam) * e * e
    return out


def weights(params: EWMAParams, n: int) -> np.ndarray:
    """The geometric kernel (1-lambda)*lambda^i, i = 0..n-1
    (partial sum 1 - lambda^n)."""
    i = np.arange(n)
    return (1.0 - params.lam) * params.lam ** i


def to_igarch(params: EWMAParams) -> GARCHParams:
    """The equivalent IGARCH: GARCHParams(0, 1-lambda, lambda)."""
    return GARCHParams(0.0, 1.0 - params.lam, params.lam)
