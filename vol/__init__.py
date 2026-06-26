"""vol-lab: the ARCH/GARCH family from first principles.

GARCH(1,1), ARCH(p), GJR-GARCH leverage, EWMA/RiskMetrics, Gaussian &
Student-t maximum likelihood, and multi-step variance forecasting -- every
formula derived from its definition and pinned by an algebraic identity, no
econometrics library underneath.

Each model module exposes its own `filter_variance`; the asymmetry-aware
models (garch, gjr) additionally expose a `news_impact` curve.  Access them
module-qualified, e.g. `vol.garch.filter_variance`, `vol.gjr.news_impact`.
The parameter dataclasses and the per-module helpers are re-exported here.
"""
from vol import arch, ewma, forecast, garch, gjr, likelihood

from vol.garch import GARCHParams, acf_squared
from vol.arch import ARCHParams, arch_inf_filter, garch_as_arch_weights
from vol.gjr import GJRParams, to_garch
from vol.ewma import (
    EWMAParams,
    RISKMETRICS_LAMBDA_DAILY,
    RISKMETRICS_LAMBDA_MONTHLY,
    to_igarch,
    weights,
)
from vol.likelihood import (
    MLEResult,
    fit_garch,
    fit_gjr,
    gaussian_loglik,
    neg_loglik,
    student_t_loglik,
    variance_target_omega,
)
from vol.forecast import (
    annualize_vol,
    cumulative_variance,
    forecast_closed_form,
    forecast_recursive,
    simulate,
)

__version__ = "0.1.0"
