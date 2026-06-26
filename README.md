# vol-lab

**The ARCH/GARCH family from first principles** in pure Python + NumPy/SciPy.
GARCH(1,1), ARCH(p), GJR-GARCH leverage, EWMA/RiskMetrics, Gaussian &
Student-t maximum likelihood, and multi-step variance forecasting — every
formula derived from its definition and pinned by an algebraic identity,
**no econometrics library underneath**.

**42/42 tests** verifying algebraic identities.

```
$ pytest tests/
=========================== 42 passed in 3.46s ============================
```

> Every mathematical identity in this repo was **adversarially verified by a
> multi-agent workflow before a line of code was written**.  17 identities
> were independently designed (three angles), synthesized, then attacked by
> one refutation agent each — **9 of 17 had a subtle error corrected**.
> Examples: the long-run *mean* of the variance path does **not** catch a
> swapped-lag or swapped-coefficient bug (E[σ²]=ω/(1−α−β) is symmetric in
> α,β) — the real detector is the exact term-by-term `filter == simulating
> path` identity; `half_life` and `unconditional_variance` need an **explicit
> guard** (the bare formulas silently return inf / a negative number at
> α+β≥1); the GARCH=ARCH(∞) bridge omits a **seed-decay term** unless the
> filter is seeded at its unconditional variance; the EWMA unrolling drops a
> **λᵗ·σ²₀ seed term** (~6e-4 error at t=50); and the IGARCH forecast closed
> form divides **0/0 → nan** without a dedicated branch.  See "How this was
> built".

## Why this exists

Every risk system runs a GARCH or an EWMA, and almost nobody implements one —
they call `arch.arch_model(...)` and trust the fit.  This repo is the
opposite: each model is a few dozen lines built straight from its variance
recursion, and every test pins an **algebraic identity** — the unconditional
variance is the fixed point of the recursion, GARCH(1,1) **is** ARCH(∞), a
GJR with γ=0 **is** a GARCH, an EWMA **is** an IGARCH, the multi-step
forecast mean-reverts to the long-run variance — rather than matching a
library's output.

It is the conditional-volatility complement to the portfolio's option labs
([`lattice-lab`](https://github.com/sauloduttra/lattice-lab),
[`convexity-lab`](https://github.com/sauloduttra/convexity-lab)): those price
at a given σ, this one **models σ through time**.

## The 6 modules

| Module | Topic | Headline API |
|---|---|---|
| `garch.py` | GARCH(1,1) core + closed-form moments | `GARCHParams`, `filter_variance`, `acf_squared`, `news_impact` |
| `arch.py` | ARCH(p) + the GARCH=ARCH(∞) bridge | `ARCHParams`, `garch_as_arch_weights`, `arch_inf_filter` |
| `gjr.py` | GJR-GARCH asymmetric leverage | `GJRParams`, `filter_variance`, `news_impact`, `to_garch` |
| `ewma.py` | EWMA / RiskMetrics (IGARCH) | `EWMAParams`, `filter_variance`, `weights`, `to_igarch` |
| `likelihood.py` | Gaussian & Student-t MLE + variance targeting | `gaussian_loglik`, `student_t_loglik`, `fit_garch`, `fit_gjr` |
| `forecast.py` | multi-step forecast + the seeded simulator | `forecast_recursive`, `forecast_closed_form`, `simulate` |

## API in 30 seconds

```python
import numpy as np
from vol import GARCHParams, garch, forecast, fit_garch, to_igarch, EWMAParams

p = GARCHParams(omega=1e-5, alpha=0.10, beta=0.85)
p.unconditional_variance        # 2.0e-4   (= omega/(1-alpha-beta))
p.half_life                     # 13.51    (= log(0.5)/log(alpha+beta))
p.kurtosis                      # fat tails, raises if the 4th moment is gone

# simulate -> filter -> the exact swap/lag detector
eps, sigma2 = forecast.simulate(p, 5000, np.random.default_rng(0))
np.allclose(garch.filter_variance(p, eps, sigma2_0=sigma2[0]), sigma2)   # True (rtol 1e-12)

# fit it back (variance targeting on by default)
fit_garch(eps).params           # recovers (alpha, beta) within sampling error

# mean-reverting variance term structure vs the flat EWMA
forecast.forecast_recursive(p, 4e-4, 60)[-1]          # 2.097e-4, -> sigma2_bar=2.0e-4 as h grows
forecast.forecast_recursive(to_igarch(EWMAParams(0.94)), 4e-4, 60)   # flat list (no mean reversion)
```

## The algebraic identities we actually test

42 tests, every one an identity the formula must satisfy. Highlights:

### GARCH core
- **Unconditional variance** ω/(1−α−β) is the fixed point; the long-run path mean converges to it
- **The exact swap/lag detector**: `filter_variance(sim_eps, seed) == simulating σ² path` (rtol 1e-12) — the mean cannot catch α↔β or a lag bug
- `half_life`, `unconditional_variance` **raise** at α+β≥1; `kurtosis` **raises** when 1−(α+β)²−2α²≤0 (stationary ≠ finite kurtosis)
- **Kurtosis** 3(1−(α+β)²)/(1−(α+β)²−2α²) and the **eps² ACF** decaying geometrically at α+β
- positivity on adversarial input (zeros, a 1e6 outlier); variance monotone in α

### ARCH / nesting
- **GARCH(1,1) = ARCH(∞)** with geometric weights αβⁱ and constant ω/(1−β) (**not** the unconditional variance) — a second code path that converges as p grows
- ARCH(1) **is** GARCH with β=0; ARCH(p) unconditional variance ω/(1−Σαᵢ)
- **GJR(γ=0) == GARCH** bit-for-bit; persistence is α+β+**γ/2** (the ½ from E[1{ε<0}]=½), with leverage (α+γ) > α on the downside
- **EWMA == IGARCH** bit-for-bit; geometric kernel (1−λ)λⁱ summing to 1−λⁿ; the exact unrolling **includes** the λᵗσ²₀ seed term

### Likelihood & forecasting
- Gaussian log-lik = the explicit per-observation sum; reduces to iid normal on a constant-variance series
- **Student-t → Gaussian** as ν→∞ (summed relative O(1/ν) — *not* a per-term bound, the tails grow ~z²)
- **Variance targeting** ω=σ̄²(1−α−β) pins the unconditional variance to the sample variance
- **Recursive == closed-form** multi-step forecast (the tinystat AR(1) analog), mean-reverting to σ̄² at rate (α+β)^(h−1); IGARCH gets a **dedicated flat/linear branch** (no 0/0)
- **sim→fit** recovers (α,β) under variance targeting within sampling tolerance

## Worked example

```bash
PYTHONPATH=. python examples/garch_moments.py          # closed form vs Monte Carlo
PYTHONPATH=. python examples/sim_fit_roundtrip.py      # simulate -> MLE -> recover
PYTHONPATH=. python examples/forecast_term_structure.py # mean reversion vs flat EWMA
```

`forecast_term_structure.py` output (textbook-exact):

```
   h |    recursive |  closed form |   gap to bar
   1 | 4.000000e-04 | 4.000000e-04 |    2.000e-04
  20 | 2.754707e-04 | 2.754707e-04 |    7.547e-05
  60 | 2.096989e-04 | 2.096989e-04 |    9.699e-06
 max|recursive - closed_form| = 3.79e-19   (two independent derivations agree)
 EWMA / IGARCH forecast is FLAT:  h=1 -> 4.000e-04   h=60 -> 4.000e-04
```

## How this was built

This repo was built with an **adversarial multi-agent workflow** before
implementation:

1. **Design panel** — three independent agents proposed the module layout
   and the algebraic identities from different angles (econometrics,
   numerical/MLE, testing-first).
2. **Synthesis** — merged into one canonical spec of 17 identities.
3. **Adversarial verification** — one agent per identity tried to refute it,
   computing counterexamples in pure NumPy. **9 of 17 were corrected** before
   any code: the swap/lag mean fallacy, the missing boundary guards, the
   ARCH(∞) seed-decay term, the EWMA seed term, the IGARCH 0/0 forecast, the
   scale-free convergence bound, the Student-t per-term vs summed limit, and
   the weak identifiability of the unconstrained GARCH MLE.
4. **Implementation** against the verified spec.
5. **Adversarial code review** — reviewers per dimension (math, numerical,
   API, tests), findings verified before applying.

The result: identities that are right because they were proven right, not
because they happened to pass.

## What's intentionally NOT here yet

- **v0.2.0** — EGARCH (Nelson 1991): log-variance, no positivity constraints
- **v0.2.0 alt** — GARCH-in-mean (risk premium in the conditional mean)
- **v0.3.0** — component / two-component GARCH (long + short-run variance)
- **v0.3.0 alt** — robust standard errors (Bollerslev-Wooldridge sandwich) + the ARCH-LM test
- **v0.4.0** — multivariate (DCC, BEKK) conditional covariance

## Related repos

- [`lattice-lab`](https://github.com/sauloduttra/lattice-lab), [`convexity-lab`](https://github.com/sauloduttra/convexity-lab) — option pricing at a given σ; this lab models σ through time
- [`var-lab`](https://github.com/sauloduttra/var-lab) — VaR/CVaR; a GARCH σ feeds the parametric and Monte Carlo methods
- [`cointegration-lab`](https://github.com/sauloduttra/cointegration-lab), [`tinystat`](https://github.com/sauloduttra/tinystat) — the time-series siblings (ADF, OU half-life, AR(1) chain forecast)
- [`hawkes-fit`](https://github.com/sauloduttra/hawkes-fit) — the MLE sim→fit round-trip pattern reused here

## References

- Engle, R. (1982). *Autoregressive Conditional Heteroscedasticity…* Econometrica 50:987–1008.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.* J. Econometrics 31:307–327.
- Bollerslev, T. (1987). *A Conditionally Heteroskedastic Time Series Model… (Student-t).* RES 69:542–547.
- Glosten, L., Jagannathan, R. & Runkle, D. (1993). *On the Relation between… Excess Return on Stocks (GJR).* J. Finance 48:1779–1801.
- Nelson, D. (1991). *Conditional Heteroskedasticity in Asset Returns (EGARCH).* Econometrica 59:347–370.
- J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, 4th ed.
- Tsay, R. *Analysis of Financial Time Series.*

## License
MIT.
