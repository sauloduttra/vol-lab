"""The variance term structure: GARCH mean-reverts, EWMA stays flat.

    PYTHONPATH=. python examples/forecast_term_structure.py
"""
from __future__ import annotations

from vol import GARCHParams, EWMAParams, forecast, to_igarch

p = GARCHParams(omega=1e-5, alpha=0.10, beta=0.85)   # sigma2_bar = 2.0e-4
s2_bar = p.unconditional_variance
s2_tp1 = 4.0e-4                                       # start ABOVE the long run

print("=" * 66)
print(" Multi-step variance forecast  (GARCH omega=1e-5, a=0.10, b=0.85)")
print(f" next-step variance = {s2_tp1:.2e}   long-run sigma2_bar = {s2_bar:.2e}")
print("=" * 66)
rec = forecast.forecast_recursive(p, s2_tp1, 60)
clo = forecast.forecast_closed_form(p, s2_tp1, 60)
print(f"{'h':>4} | {'recursive':>12} | {'closed form':>12} | {'gap to bar':>12}")
for h in (1, 2, 5, 10, 20, 40, 60):
    print(f"{h:>4} | {rec[h-1]:>12.6e} | {clo[h-1]:>12.6e} | {rec[h-1]-s2_bar:>12.3e}")
print(f"\n max|recursive - closed_form| = {max(abs(a-b) for a,b in zip(rec,clo)):.2e}"
      "   (two independent derivations agree)")

print()
print(" EWMA / IGARCH forecast is FLAT (persistence = 1, no mean reversion):")
ig = to_igarch(EWMAParams(0.94))
flat = forecast.forecast_recursive(ig, s2_tp1, 60)
print(f"   h=1 -> {flat[0]:.3e}   h=20 -> {flat[19]:.3e}   h=60 -> {flat[59]:.3e}")
