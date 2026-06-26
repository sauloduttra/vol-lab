"""Forecasting + sim->fit identities (FCST-01, EWMA-03, FCST-02).

Adversarial corrections:
  * FCST-01 -- convergence is the SCALE-FREE relation |f[h]-sigma2_bar| =
    |s2_tp1-sigma2_bar|*(a+b)^(h-1); a fixed 1e-6 absolute gap at H=60 is
    variance-scale dependent and wrong (it is ~1.45e-5 there).
  * EWMA-03 -- forecast_closed_form must NOT divide 0/0 for IGARCH; the
    flat forecast is the omega=0 case, while omega>0 IGARCH grows linearly.
  * FCST-02 -- sim->fit recovers (alpha,beta) under variance targeting; raw
    omega and unconstrained fits are weakly identified.
"""
from __future__ import annotations

import numpy as np
import pytest

from vol import (GARCHParams, GJRParams, forecast, fit_garch, fit_gjr,
                 to_igarch, EWMAParams)


# FCST-01: recursive == closed-form, AR(1)-chain analog ----------------------

def test_recursive_equals_closed_form_and_converges():
    p = GARCHParams(1e-5, 0.1, 0.85)
    s2_tp1, H = 3e-4, 60
    rec = forecast.forecast_recursive(p, s2_tp1, H)
    clo = forecast.forecast_closed_form(p, s2_tp1, H)
    assert np.allclose(rec, clo, rtol=1e-12, atol=1e-18)
    assert clo[0] == pytest.approx(3e-4, rel=1e-12)        # catches p^h off-by-one
    # scale-free convergence to sigma2_bar = 2e-4 at rate 0.95^(h-1)
    s2_bar = 2e-4
    gaps = np.abs(np.array(rec) - s2_bar)
    for k in range(H):
        assert gaps[k] == pytest.approx(1e-4 * 0.95 ** k, rel=1e-12)
    assert np.all(np.diff(gaps) <= 0.0)                    # monotone non-increasing


# EWMA-03: IGARCH flat (omega=0) vs linear growth (omega>0) ------------------

def test_igarch_forecast_flat_when_omega_zero():
    ig = to_igarch(EWMAParams(0.94))              # GARCHParams(0, 0.06, 0.94)
    rec = forecast.forecast_recursive(ig, 3.7e-4, 20)
    clo = forecast.forecast_closed_form(ig, 3.7e-4, 20)
    assert np.allclose(rec, [3.7e-4] * 20, rtol=1e-12)
    assert np.all(np.isfinite(clo))               # NOT nan from 0/0
    assert np.allclose(clo, [3.7e-4] * 20, rtol=1e-12)


def test_igarch_with_positive_omega_grows_linearly():
    """At persistence 1 with omega>0 the forecast plateaus-then-grows
    LINEARLY, NOT flat: f_h = s2_tp1 + (h-1)*omega."""
    p = GARCHParams(1e-5, 0.06, 0.94)             # a+b = 1, omega > 0
    rec = forecast.forecast_recursive(p, 3.7e-4, 10)
    clo = forecast.forecast_closed_form(p, 3.7e-4, 10)
    assert rec[9] == pytest.approx(3.7e-4 + 9 * 1e-5, rel=1e-9)
    assert rec[9] > rec[0]                          # genuinely grew (not flat)
    assert np.allclose(rec, clo, rtol=1e-9)        # recursive == closed form


# FCST-02: sim->fit consistency (variance targeting) -------------------------

def test_sim_fit_recovers_params_variance_targeted():
    true = GARCHParams(1e-5, 0.10, 0.85)
    eps, _ = forecast.simulate(true, 10000, np.random.default_rng(20260626), burn=1000)
    res = fit_garch(eps, variance_targeting=True)
    assert res.converged
    assert abs(res.params.alpha - 0.10) < 0.03
    assert abs(res.params.beta - 0.85) < 0.04
    assert abs(res.params.persistence - 0.95) < 0.03


def test_sim_fit_recovers_gjr_variance_targeted():
    """The GJR MLE recovers leverage; the persistence assertion pins the
    gamma/2 in the variance-targeting reparameterization."""
    true = GJRParams(5e-6, 0.03, 0.90, 0.08)      # persistence 0.97
    eps, _ = forecast.simulate(true, 10000, np.random.default_rng(20260626), burn=1000)
    res = fit_gjr(eps, variance_targeting=True)
    assert res.converged
    assert res.params.gamma > 0.0                  # leverage genuinely recovered
    assert abs(res.params.persistence - true.persistence) < 0.03


def test_cumulative_variance_equals_sum_of_recursive():
    p = GARCHParams(1e-5, 0.1, 0.85)
    assert forecast.cumulative_variance(p, 3e-4, 10) == \
        pytest.approx(sum(forecast.forecast_recursive(p, 3e-4, 10)))
    assert forecast.cumulative_variance(p, 3e-4, 1) == pytest.approx(3e-4)


def test_annualize_vol():
    assert forecast.annualize_vol(1e-4, 252) == pytest.approx(np.sqrt(252 * 1e-4))
