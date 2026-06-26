"""EWMA / RiskMetrics identities (EWMA-01..02).

Adversarial correction:
  * EWMA-02 -- the unrolled convolution must INCLUDE the decaying seed term
    lambda^t*sigma2_0 (~6e-4 relative error at t=50 if dropped); the pure
    kernel matches only after a much longer burn-in (t > ~447 for lambda=0.94).
"""
from __future__ import annotations

import numpy as np
import pytest

from vol import EWMAParams, GARCHParams, ewma, garch, to_igarch


# EWMA-01: EWMA == IGARCH (code-path nesting) --------------------------------

def test_ewma_equals_igarch_bitwise():
    eps = np.random.default_rng(0).standard_normal(1000) * 0.01
    s0 = 2e-4
    igarch = to_igarch(EWMAParams(0.94))          # GARCHParams(0, 1-lam, lam)
    e = ewma.filter_variance(EWMAParams(0.94), eps, s0)
    g = garch.filter_variance(igarch, eps, s0)
    assert np.array_equal(e, g)                   # bit-for-bit (same 1-lam float)
    assert igarch.omega == 0.0 and igarch.alpha == 1.0 - 0.94 and igarch.beta == 0.94
    assert igarch.persistence == pytest.approx(1.0, abs=1e-15)


def test_ewma_no_unconditional_variance():
    p = EWMAParams(0.94)
    assert p.persistence == 1.0
    assert p.is_stationary is False
    with pytest.raises(ValueError):
        _ = p.unconditional_variance


# EWMA-02: geometric kernel + the seed term ----------------------------------

def test_weights_partial_sum():
    p = EWMAParams(0.94)
    for n in (10, 50, 224):
        assert ewma.weights(p, n).sum() == pytest.approx(1 - 0.94 ** n, rel=1e-12)
    assert 0.94 ** 224 < 1e-6 <= 0.94 ** 223      # first n with lambda^n < 1e-6


def test_filter_unrolls_with_seed_term():
    """filter[t] == lambda^t*s0 + (1-lambda) sum_{i<t} lambda^i eps2[t-1-i],
    EXACT at every t (the seed term lambda^t*s0 is part of the identity)."""
    lam = 0.94
    p = EWMAParams(lam)
    eps2 = (np.random.default_rng(0).standard_normal(2000)) ** 2
    eps = np.sqrt(eps2)
    s0 = eps2[0]
    filt = ewma.filter_variance(p, eps, s0)
    unrolled = np.array([
        lam ** t * s0 + (1 - lam) * sum(lam ** i * eps2[t - 1 - i] for i in range(t))
        for t in range(len(eps))
    ])
    assert np.allclose(filt, unrolled, rtol=1e-12)
    # dropping the seed term gives a large error at modest t
    no_seed_t50 = (1 - lam) * sum(lam ** i * eps2[50 - 1 - i] for i in range(50))
    assert abs(no_seed_t50 - filt[50]) / filt[50] > 1e-4
