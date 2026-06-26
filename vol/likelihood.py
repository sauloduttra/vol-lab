"""Gaussian & Student-t (quasi-)maximum-likelihood estimation.

The variance filter feeds the conditional log-likelihood; the optimizer
(L-BFGS-B + bounds, with a 1e10 penalty on invalid/non-stationary params)
maximizes it.

`variance_targeting` reparameterizes omega = sigma2_bar*(1-alpha-beta) (with
sigma2_bar the sample variance), which pins the unconditional variance to the
data and removes omega's weak identifiability -- the raw 3-parameter
likelihood is nearly flat in the omega direction, so unconstrained fits are
initialization-sensitive.

The Student-t log-likelihood uses the UNIT-VARIANCE standardized t (it
requires nu > 2 and reduces to the Gaussian as nu -> infinity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from vol.garch import GARCHParams, filter_variance as _garch_filter
from vol.gjr import GJRParams, filter_variance as _gjr_filter

__all__ = [
    "gaussian_loglik", "student_t_loglik", "variance_target_omega",
    "neg_loglik", "MLEResult", "fit_garch", "fit_gjr",
]

_LOG2PI = math.log(2.0 * math.pi)


def _filter(params, eps: np.ndarray) -> np.ndarray:
    if isinstance(params, GJRParams):
        return _gjr_filter(params, eps)
    return _garch_filter(params, eps)


def gaussian_loglik(params, eps) -> float:
    """Sum of conditional-normal log-densities over the filtered path."""
    eps = np.asarray(eps, dtype=float).ravel()
    s2 = _filter(params, eps)
    return float(-0.5 * np.sum(_LOG2PI + np.log(s2) + eps * eps / s2))


def student_t_loglik(params, eps, nu: float) -> float:
    """Unit-variance standardized-t log-likelihood (nu > 2).

    Innovation density (Bollerslev 1987), with z = eps/sqrt(sigma2):
        c = lnGamma((nu+1)/2) - lnGamma(nu/2) - 0.5*ln((nu-2)*pi)
        ll_t = c - 0.5*ln(sigma2) - (nu+1)/2 * ln(1 + z^2/(nu-2))
    Reduces to `gaussian_loglik` as nu -> infinity.
    """
    if nu <= 2.0:
        raise ValueError("nu > 2 required for the unit-variance standardized t")
    eps = np.asarray(eps, dtype=float).ravel()
    s2 = _filter(params, eps)
    z2 = eps * eps / s2
    c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) \
        - 0.5 * math.log((nu - 2.0) * math.pi)
    ll = c - 0.5 * np.log(s2) - ((nu + 1.0) / 2.0) * np.log1p(z2 / (nu - 2.0))
    return float(np.sum(ll))


def variance_target_omega(sigma2_bar: float, alpha: float, beta: float) -> float:
    """omega that pins the unconditional variance to sigma2_bar."""
    return sigma2_bar * (1.0 - alpha - beta)


@dataclass(frozen=True)
class MLEResult:
    params: object
    log_likelihood: float
    n_iter: int
    converged: bool
    nu: float | None = None        # fitted Student-t dof (None for Gaussian)


def _build(theta, model: str, dist: str, variance_targeting: bool,
           sigma2_bar: float):
    """Map an optimizer vector to (params, nu)."""
    theta = list(theta)
    nu = None
    if dist == "student-t":
        nu = theta[-1]
        theta = theta[:-1]
    if model == "garch":
        if variance_targeting:
            a, b = theta
            w = variance_target_omega(sigma2_bar, a, b)
        else:
            w, a, b = theta
        params = GARCHParams(w, a, b)
    elif model == "gjr":
        if variance_targeting:
            a, b, g = theta
            w = sigma2_bar * (1.0 - a - b - g / 2.0)
        else:
            w, a, b, g = theta
        params = GJRParams(w, a, b, g)
    else:
        raise ValueError("model must be 'garch' or 'gjr'")
    return params, nu


def neg_loglik(theta, eps, model: str = "garch", dist: str = "gaussian",
               variance_targeting: bool = False) -> float:
    """Negative log-likelihood for the optimizer; 1e10 on any invalid,
    non-stationary, or non-positive-variance evaluation."""
    eps = np.asarray(eps, dtype=float).ravel()
    sigma2_bar = float(np.var(eps))
    try:
        params, nu = _build(theta, model, dist, variance_targeting, sigma2_bar)
    except (ValueError, TypeError):
        return 1e10
    if params.omega <= 0.0 or params.alpha < 0.0 or params.beta < 0.0:
        return 1e10
    if isinstance(params, GJRParams) and params.gamma < 0.0:
        return 1e10
    if not params.is_stationary:
        return 1e10
    if dist == "student-t" and (nu is None or nu <= 2.0):
        return 1e10
    s2 = _filter(params, eps)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0.0):
        return 1e10
    ll = gaussian_loglik(params, eps) if dist == "gaussian" \
        else student_t_loglik(params, eps, nu)
    return -ll if np.isfinite(ll) else 1e10


def _fit(eps, model, dist, variance_targeting, x0):
    eps = np.asarray(eps, dtype=float).ravel()
    var = float(np.var(eps))

    # assemble (x0, bounds) for the chosen mode
    if model == "garch":
        core_x0 = [0.05, 0.90] if variance_targeting else [0.05 * var, 0.05, 0.90]
        core_b = [(1e-8, 0.9999), (1e-8, 0.9999)] if variance_targeting \
            else [(1e-12, None), (1e-8, 0.9999), (1e-8, 0.9999)]
    else:  # gjr
        core_x0 = [0.03, 0.90, 0.04] if variance_targeting \
            else [0.05 * var, 0.03, 0.90, 0.04]
        core_b = [(1e-8, 0.9999)] * 3 if variance_targeting \
            else [(1e-12, None)] + [(1e-8, 0.9999)] * 3
    if dist == "student-t":
        core_x0 = core_x0 + [8.0]
        core_b = core_b + [(2.05, None)]
    if x0 is not None:
        core_x0 = list(x0)

    res = minimize(neg_loglik, core_x0, args=(eps, model, dist, variance_targeting),
                   method="L-BFGS-B", bounds=core_b,
                   options={"maxiter": 500, "ftol": 1e-10})
    params, nu = _build(res.x, model, dist, variance_targeting, var)
    # Demote convergence when the optimizer is stuck on the 1e10 penalty
    # plateau (a degenerate / near-constant series has no real likelihood).
    converged = bool(res.success) and res.fun < 1e9
    return MLEResult(params=params, log_likelihood=float(-res.fun),
                     n_iter=int(res.nit), converged=converged, nu=nu)


def fit_garch(eps, dist: str = "gaussian", variance_targeting: bool = True,
              x0=None) -> MLEResult:
    """MLE of a GARCH(1,1) (variance-targeting on by default)."""
    return _fit(eps, "garch", dist, variance_targeting, x0)


def fit_gjr(eps, dist: str = "gaussian", variance_targeting: bool = True,
            x0=None) -> MLEResult:
    """MLE of a GJR-GARCH (variance-targeting on by default)."""
    return _fit(eps, "gjr", dist, variance_targeting, x0)
