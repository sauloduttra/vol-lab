"""Likelihood identities (LL-01..03).

Adversarial correction:
  * LL-02 -- Student-t -> Gaussian is O(1/nu) on the SUMMED log-likelihood
    only; the per-observation difference grows ~z^2 on the tails, so a
    per-term atol (or rtol=1e-12) is wrong.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from vol import (GARCHParams, garch, forecast, gaussian_loglik,
                 student_t_loglik, variance_target_omega)


# LL-01: Gaussian log-likelihood = explicit sum ------------------------------

def test_gaussian_loglik_matches_hand_loop():
    p = GARCHParams(1e-5, 0.1, 0.85)
    eps, _ = forecast.simulate(p, 500, np.random.default_rng(0))
    s2 = garch.filter_variance(p, eps)
    hand = -0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + eps ** 2 / s2)
    assert gaussian_loglik(p, eps) == pytest.approx(hand, rel=1e-12)


def test_gaussian_loglik_constant_variance_is_iid_normal():
    """With alpha=beta=0 the filter is constant (= omega), so the GARCH
    log-lik reduces to the iid N(0, omega) log-likelihood."""
    p = GARCHParams(4e-4, 0.0, 0.0)
    eps = np.random.default_rng(1).standard_normal(300) * 0.02
    s2c = garch.filter_variance(p, eps)
    assert np.allclose(s2c, 4e-4)                 # constant
    iid = norm(0.0, math.sqrt(4e-4)).logpdf(eps).sum()
    assert gaussian_loglik(p, eps) == pytest.approx(iid, rel=1e-12)


# LL-02: Student-t -> Gaussian limit (summed, relative) ----------------------

def test_student_t_converges_to_gaussian():
    p = GARCHParams(1e-5, 0.1, 0.85)
    eps, _ = forecast.simulate(p, 2000, np.random.default_rng(0))
    g = gaussian_loglik(p, eps)
    rel_1e5 = abs(student_t_loglik(p, eps, nu=1e5) - g) / abs(g)
    rel_1e4 = abs(student_t_loglik(p, eps, nu=1e4) - g) / abs(g)
    assert rel_1e5 < 1e-4                          # summed relative O(1/nu)
    assert rel_1e5 < rel_1e4                       # and shrinking with nu


def test_student_t_requires_nu_above_two():
    p = GARCHParams(1e-5, 0.1, 0.85)
    eps = np.random.default_rng(0).standard_normal(50) * 0.01
    with pytest.raises(ValueError):
        student_t_loglik(p, eps, nu=2.0)


# LL-03: variance targeting --------------------------------------------------

@pytest.mark.parametrize("a,b", [(0.05, 0.90), (0.10, 0.85), (0.01, 0.30), (0.2, 0.7)])
def test_variance_targeting_pins_unconditional_variance(a, b):
    s2_bar = 2e-4
    w = variance_target_omega(s2_bar, a, b)
    assert GARCHParams(w, a, b).unconditional_variance == pytest.approx(s2_bar, rel=1e-12)
