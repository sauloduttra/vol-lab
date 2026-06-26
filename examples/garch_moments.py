"""GARCH(1,1) closed-form moments vs a long simulation.

    PYTHONPATH=. python examples/garch_moments.py
"""
from __future__ import annotations

import numpy as np

from vol import GARCHParams, garch, forecast

p = GARCHParams(omega=1e-5, alpha=0.08, beta=0.90)

print("=" * 64)
print(" GARCH(1,1)  omega=1e-5  alpha=0.08  beta=0.90")
print("=" * 64)
print(f" persistence (alpha+beta)    = {p.persistence:.4f}")
print(f" unconditional variance      = {p.unconditional_variance:.6e}")
print(f" unconditional vol (daily)   = {np.sqrt(p.unconditional_variance)*100:.4f}%")
print(f" half-life of a vol shock    = {p.half_life:.4f} periods")
print(f" unconditional kurtosis      = {p.kurtosis:.6f}   (excess {p.excess_kurtosis:.6f})")
print(f" eps^2 ACF rho_1             = {garch.acf_squared(p,1):.6f}")
print()

print(" Closed form vs Monte Carlo (N = 3,000,000):")
eps, _ = forecast.simulate(p, 3_000_000, np.random.default_rng(0))
m2 = (eps ** 2).mean()
m4 = (eps ** 4).mean()
r1 = np.corrcoef((eps ** 2)[1:], (eps ** 2)[:-1])[0, 1]
print(f"   variance     closed {p.unconditional_variance:.4e}   sim {m2:.4e}")
print(f"   kurtosis     closed {p.kurtosis:.4f}       sim {m4/m2**2:.4f}")
print(f"   eps^2 ACF(1) closed {garch.acf_squared(p,1):.4f}       sim {r1:.4f}")
print()
print(" Covariance-stationary but NO finite kurtosis: GARCH(0.5, 0.3)")
ps = GARCHParams(1e-5, 0.5, 0.3)
print(f"   alpha+beta = {ps.persistence} < 1 (stationary={ps.is_stationary}),"
      f" kurtosis_exists = {ps.kurtosis_exists}")
