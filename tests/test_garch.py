"""GARCH(1,1) identities (GARCH-01..05).

Adversarial corrections baked in:
  * GARCH-01 -- the path MEAN does NOT catch a swapped-lag or swapped-coeff
    bug (E[sigma2]=omega/(1-a-b) is symmetric in alpha,beta); the real
    detector is the EXACT term-by-term filter==sim-path identity.
  * GARCH-02/05 -- half_life / unconditional_variance are explicit GUARDS
    (the bare formulas return inf/negative silently); positivity holds for
    t>=1 given a seed >= omega.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vol import GARCHParams, garch, forecast


# GARCH-01: unconditional variance + the exact swap/lag detector -------------

def test_unconditional_variance_closed_form():
    assert GARCHParams(1e-5, 0.1, 0.85).unconditional_variance == pytest.approx(2e-4, rel=1e-12)


def test_filter_equals_simulating_path_exact():
    """The REAL swap/lag detector: filter_variance(sim_eps, seed) reproduces
    the simulating sigma2 path bit-for-bit (a swapped lag or alpha<->beta
    would break this, though it leaves the mean unchanged)."""
    p = GARCHParams(1e-5, 0.1, 0.85)
    eps, s2 = forecast.simulate(p, 20000, np.random.default_rng(0))
    recon = garch.filter_variance(p, eps, sigma2_0=s2[0])
    assert np.allclose(recon, s2, rtol=1e-12, atol=0.0)


def test_filtered_path_mean_converges_to_unconditional():
    p = GARCHParams(1e-5, 0.1, 0.85)              # a+b = 0.95 (moderate)
    eps, _ = forecast.simulate(p, 200000, np.random.default_rng(0))
    out = garch.filter_variance(p, eps, sigma2_0=p.unconditional_variance)
    assert out.mean() == pytest.approx(2e-4, rel=5e-2)


# GARCH-02: persistence, half-life, boundary guards --------------------------

def test_persistence_and_half_life():
    p = GARCHParams(1e-5, 0.1, 0.85)
    assert p.persistence == pytest.approx(0.95, rel=1e-12)
    assert p.half_life == pytest.approx(13.513407333964874, rel=1e-12)


def test_boundary_guards_raise():
    for bad in (GARCHParams(1e-5, 0.05, 0.95),    # a+b == 1
                GARCHParams(1e-5, 0.6, 0.6)):      # a+b == 1.2
        assert bad.is_stationary is False
        with pytest.raises(ValueError):
            _ = bad.half_life
        with pytest.raises(ValueError):
            _ = bad.unconditional_variance


# GARCH-03: unconditional kurtosis + existence condition ---------------------

def test_kurtosis_closed_form():
    p = GARCHParams(1e-5, 0.08, 0.90)
    assert p.kurtosis == pytest.approx(4.432835820896, rel=1e-12)
    assert p.excess_kurtosis == pytest.approx(1.432835820896, rel=1e-12)
    assert p.excess_kurtosis == pytest.approx(p.kurtosis - 3.0, rel=1e-12)


def test_kurtosis_can_fail_to_exist_while_stationary():
    p = GARCHParams(1e-5, 0.5, 0.3)               # a+b=0.8<1 but 4th moment gone
    assert p.is_stationary is True
    assert p.kurtosis_exists is False
    with pytest.raises(ValueError):
        _ = p.kurtosis
    with pytest.raises(ValueError):
        _ = p.excess_kurtosis


def test_kurtosis_matches_simulation():
    p = GARCHParams(1e-5, 0.08, 0.90)
    eps, _ = forecast.simulate(p, 2_000_000, np.random.default_rng(0))
    m2 = (eps ** 2).mean()
    m4 = (eps ** 4).mean()
    assert m4 / m2 ** 2 == pytest.approx(4.432835820896, rel=0.15)


# GARCH-04: volatility-clustering ACF of eps^2 -------------------------------

def test_acf_squared_closed_form_and_geometric_decay():
    p = GARCHParams(1e-5, 0.08, 0.90)
    assert garch.acf_squared(p, 1) == pytest.approx(0.205217391304, rel=1e-12)
    for k in range(2, 7):
        assert garch.acf_squared(p, k) / garch.acf_squared(p, 1) == \
            pytest.approx(0.98 ** (k - 1), rel=1e-12)


def test_acf_squared_requires_fourth_moment():
    with pytest.raises(ValueError):
        garch.acf_squared(GARCHParams(1e-5, 0.5, 0.3), 1)


def test_acf_squared_matches_simulation():
    """Cross-check the closed-form eps^2 ACF against the statistical
    definition (not just the formula re-evaluated at a hardcoded base)."""
    p = GARCHParams(1e-5, 0.08, 0.90)
    eps, _ = forecast.simulate(p, 2_000_000, np.random.default_rng(1))
    x = eps ** 2 - (eps ** 2).mean()
    denom = (x * x).mean()
    for k in (1, 2, 3):
        sample = (x[k:] * x[:-k]).mean() / denom
        assert sample == pytest.approx(garch.acf_squared(p, k), rel=0.1)


def test_symmetric_news_impact():
    """The GARCH news-impact curve is symmetric (the baseline GJR's leverage
    is defined against)."""
    p = GARCHParams(5e-6, 0.06, 0.92)
    assert garch.news_impact(p, -0.05) == garch.news_impact(p, 0.05)
    assert garch.news_impact(p, 0.0) == pytest.approx(
        p.omega + p.beta * p.unconditional_variance)


# GARCH-05: positivity / numerical stability + monotonicity ------------------

def test_filter_strictly_positive_on_adversarial_input():
    p = GARCHParams(1e-5, 0.1, 0.85)
    eps = np.array([0.0, 0.0, 1e6, -1.0, 1.0, -1.0, 0.0])
    out = garch.filter_variance(p, eps)            # default = unconditional seed
    assert np.all(out > 0.0)
    assert np.all(np.isfinite(out))
    assert out[1:].min() >= p.omega - 1e-18        # floor at omega for t>=1


def test_filter_monotone_in_alpha_shared_seed():
    eps = np.random.default_rng(2).standard_normal(500) * 0.01
    seed = 2e-4
    prev = None
    for a in (0.02, 0.05, 0.10, 0.15):
        out = garch.filter_variance(GARCHParams(1e-5, a, 0.80), eps, sigma2_0=seed)
        if prev is not None:
            assert np.all(out >= prev - 1e-18)     # non-decreasing in alpha
        prev = out
