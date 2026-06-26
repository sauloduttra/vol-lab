"""ARCH(p) (Engle 1982) and the GARCH(1,1) = ARCH(infinity) bridge.

The pure ARCH(p) filter is

    sigma2_t = omega + sum_{i=0}^{p-1} alpha_i * eps2_{t-1-i}

The bridge re-derives the GARCH variance through a DIFFERENT code path (a
geometric convolution) so the GARCH recursion is cross-checked, not echoed:

    sigma2_t = omega/(1-beta) + alpha * sum_{i>=0} beta^i * eps2_{t-1-i}

Note the constant is omega/(1-beta), NOT the unconditional variance
omega/(1-alpha-beta).  The truncated-at-p reconstruction converges to the
finite-sample GARCH filter (evaluated at t >= p) only when that filter is
seeded at its unconditional variance; otherwise the two differ by an extra
seed-decay term beta^t*(sigma2_0 - sigma2_uncond).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vol.garch import GARCHParams

__all__ = ["ARCHParams", "filter_variance", "garch_as_arch_weights", "arch_inf_filter"]


@dataclass(frozen=True)
class ARCHParams:
    omega: float
    alphas: tuple

    @property
    def persistence(self) -> float:
        return float(sum(self.alphas))

    @property
    def is_stationary(self) -> bool:
        return self.persistence < 1.0

    @property
    def unconditional_variance(self) -> float:
        if not self.is_stationary:
            raise ValueError(
                "unconditional variance undefined: sum(alphas) >= 1")
        return self.omega / (1.0 - self.persistence)


def filter_variance(params: ARCHParams, eps, sigma2_0: float | None = None) -> np.ndarray:
    """ARCH(p) conditional-variance path.  sigma2_0 is the seed (= Var(eps_0));
    pre-sample lags (t-1-i < 0) are backcast at sigma2_0."""
    eps = np.asarray(eps, dtype=float).ravel()
    n = eps.size
    p = len(params.alphas)
    if sigma2_0 is None:
        sigma2_0 = params.unconditional_variance if params.is_stationary \
            else float(np.var(eps))
    if sigma2_0 <= 0.0:
        raise ValueError("sigma2_0 must be strictly positive")
    eps2 = eps * eps
    out = np.empty(n)
    out[0] = sigma2_0
    for t in range(1, n):
        s2 = params.omega
        for i in range(p):
            lag = t - 1 - i
            s2 += params.alphas[i] * (eps2[lag] if lag >= 0 else sigma2_0)
        out[t] = s2
    return out


def garch_as_arch_weights(garch: GARCHParams, p: int) -> tuple[float, np.ndarray]:
    """The ARCH(infinity) representation of a GARCH(1,1), truncated at p.

    const = omega/(1-beta); weights[i] = alpha*beta^i for i = 0..p-1.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    const = garch.omega / (1.0 - garch.beta)
    i = np.arange(p)
    weights = garch.alpha * garch.beta ** i
    return const, weights


def arch_inf_filter(garch: GARCHParams, eps, p: int) -> np.ndarray:
    """Reconstruct the GARCH variance as a truncated ARCH(p) convolution.

    Built ONLY from `garch_as_arch_weights` -- it must NOT call
    `garch.filter_variance`, so the agreement (at t >= p, GARCH seeded at its
    unconditional variance) genuinely cross-checks the GARCH recursion.
    Pre-sample lags are dropped (truncation), which is exactly the tail whose
    weight beta^i vanishes.
    """
    const, weights = garch_as_arch_weights(garch, p)
    eps = np.asarray(eps, dtype=float).ravel()
    eps2 = eps * eps
    n = eps.size
    out = np.empty(n)
    for t in range(n):
        s2 = const
        for i in range(p):
            lag = t - 1 - i
            if lag >= 0:
                s2 += weights[i] * eps2[lag]
        out[t] = s2
    return out
