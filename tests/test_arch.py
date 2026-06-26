"""ARCH(p) identities (ARCH-01..02).

Adversarial corrections:
  * ARCH-01 -- the ARCH(infinity) constant is omega/(1-beta), NOT the
    unconditional variance; the truncated reconstruction matches the GARCH
    filter (at t>=p) only when the GARCH filter is seeded at its
    unconditional variance, and only as p grows (never tightly at small p).
"""
from __future__ import annotations

import numpy as np
import pytest

from vol import (ARCHParams, GARCHParams, arch, garch,
                 arch_inf_filter, garch_as_arch_weights)


# ARCH-01: GARCH(1,1) = ARCH(infinity) ---------------------------------------

def test_arch_inf_constant_is_omega_over_one_minus_beta():
    g = GARCHParams(1e-5, 0.08, 0.90)
    const, w = garch_as_arch_weights(g, 5)
    assert const == pytest.approx(1e-5 / (1 - 0.90), rel=1e-12)   # NOT 1e-5/0.02
    assert np.allclose(w, [0.08 * 0.90 ** i for i in range(5)], rtol=1e-12)


def test_garch_equals_truncated_arch_as_p_grows():
    g = GARCHParams(1e-5, 0.08, 0.90)
    eps = np.random.default_rng(0).standard_normal(600) * np.sqrt(2e-4)
    gf = garch.filter_variance(g, eps)             # seeded at unconditional var
    errs = []
    for p in (20, 50, 100, 200):
        recon = arch_inf_filter(g, eps, p)
        errs.append(np.max(np.abs(recon[p:] - gf[p:])))   # evaluate at t >= p
    assert errs[-1] < 1e-10
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))   # monotone


# ARCH-02: ARCH(p) unconditional variance + ARCH(1)==GARCH(beta=0) -----------

def test_arch_unconditional_variance():
    p = ARCHParams(1e-5, (0.05, 0.04, 0.03))
    assert p.unconditional_variance == pytest.approx(1e-5 / 0.88, rel=1e-12)
    with pytest.raises(ValueError):
        _ = ARCHParams(1e-5, (0.6, 0.5)).unconditional_variance     # sum >= 1


def test_arch1_equals_garch_beta_zero():
    """ARCH(1) filter == GARCH(1,1) with beta=0, identical seed (all t)."""
    eps = np.random.default_rng(1).standard_normal(400) * 0.01
    seed = 1e-4
    a = arch.filter_variance(ARCHParams(1e-5, (0.1,)), eps, sigma2_0=seed)
    g = garch.filter_variance(GARCHParams(1e-5, 0.1, 0.0), eps, sigma2_0=seed)
    assert np.allclose(a, g, rtol=1e-12, atol=0.0)
