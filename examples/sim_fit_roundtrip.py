"""Simulate a GARCH(1,1) path, then recover its parameters by MLE.

    PYTHONPATH=. python examples/sim_fit_roundtrip.py
"""
from __future__ import annotations

import numpy as np

from vol import GARCHParams, forecast, fit_garch

true = GARCHParams(omega=1e-5, alpha=0.10, beta=0.85)
eps, _ = forecast.simulate(true, 10000, np.random.default_rng(20260626), burn=1000)

print("=" * 60)
print(" GARCH(1,1) sim -> MLE -> recover  (N = 10,000)")
print("=" * 60)
print(f" true:  omega={true.omega:.3e}  alpha={true.alpha:.4f}  beta={true.beta:.4f}"
      f"  persistence={true.persistence:.4f}")
print()

vt = fit_garch(eps, variance_targeting=True)
print(" variance-targeted MLE (alpha, beta free; omega pinned to sample var):")
print(f"   omega={vt.params.omega:.3e}  alpha={vt.params.alpha:.4f}"
      f"  beta={vt.params.beta:.4f}  persistence={vt.params.persistence:.4f}")
print(f"   log-likelihood={vt.log_likelihood:.2f}  converged={vt.converged}")
print()
print(" The persistence and unconditional variance are recovered tightly;")
print(" the raw omega and the alpha/beta split are weakly identified (the")
print(" GARCH likelihood is nearly flat along the alpha+beta ridge), which")
print(" is exactly why variance targeting helps.")
