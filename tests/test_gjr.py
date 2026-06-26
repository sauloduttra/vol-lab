"""GJR-GARCH identities (GJR-01..02).

Adversarial correction:
  * GJR-01 -- persistence is alpha+beta+gamma/2 (the /2 from E[1{eps<0}]=1/2
    under symmetric innovations), NOT alpha+beta+gamma; a symmetric-innovation
    path-mean catches the gamma-vs-gamma/2 bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from vol import GJRParams, GARCHParams, gjr, garch, to_garch, forecast


# GJR-01: unconditional variance + persistence (the gamma/2) ------------------

def test_persistence_and_unconditional_variance():
    p = GJRParams(5e-6, 0.03, 0.90, 0.08)
    assert p.persistence == pytest.approx(0.97, rel=1e-12)          # a+b+g/2
    assert p.unconditional_variance == pytest.approx(1.6666666667e-4, rel=1e-9)


def test_symmetric_simulation_catches_gamma_over_two():
    """A symmetric-innovation path-mean lands on omega/(1-a-b-g/2); a
    gamma-not-gamma/2 implementation would target a different persistence."""
    p = GJRParams(5e-6, 0.03, 0.90, 0.08)
    eps, _ = forecast.simulate(p, 200000, np.random.default_rng(0))
    out = gjr.filter_variance(p, eps, sigma2_0=p.unconditional_variance)
    assert out.mean() == pytest.approx(1.6666666667e-4, rel=5e-2)


# GJR-02: model nesting + leverage -------------------------------------------

def test_gjr_gamma_zero_equals_garch_bitwise():
    eps = np.random.default_rng(1).standard_normal(500) * 0.01
    seed = 2e-4
    g0 = gjr.filter_variance(GJRParams(1e-5, 0.1, 0.85, 0.0), eps, sigma2_0=seed)
    gg = garch.filter_variance(GARCHParams(1e-5, 0.1, 0.85), eps, sigma2_0=seed)
    assert np.array_equal(g0, gg)                  # bit-for-bit


def test_to_garch_round_trip_and_guard():
    assert to_garch(GJRParams(1e-5, 0.1, 0.85, 0.0)) == GARCHParams(1e-5, 0.1, 0.85)
    with pytest.raises(ValueError):
        to_garch(GJRParams(1e-5, 0.1, 0.85, 0.05))


def test_leverage_asymmetric_news_impact():
    p = GJRParams(5e-6, 0.03, 0.90, 0.08)
    assert p.leverage_ratio == pytest.approx((0.03 + 0.08) / 0.03, rel=1e-12)
    # a negative shock raises variance more than the symmetric positive shock
    down = gjr.news_impact(p, -0.05)
    up = gjr.news_impact(p, 0.05)
    assert down > up
