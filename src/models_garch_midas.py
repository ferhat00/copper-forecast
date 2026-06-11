"""
models_garch_midas.py
=====================
GARCH-MIDAS volatility model (Engle, Ghysels & Sohn 2013) — the standout
interpretable family for *monthly* copper risk and calibrated intervals.

The conditional variance factorises into a short-run unit-mean GARCH component
``g_t`` and a slowly-moving long-run component ``tau_t`` driven by a MIDAS-
weighted macro variable:

    h_t = tau_t * g_t,   log tau_t = m + theta * sum_k phi_k(w) * MV_{t-k}

The headline interpretable parameter is **theta**: the signed elasticity of
copper's long-run variance to the macro driver (e.g. PPI / industrial
production).  Estimation is the standard pragmatic two-step:

    1. long run  -- OLS of log(r_t^2) on the MIDAS-weighted macro -> m, theta, tau_t
    2. short run -- GARCH(1,1) (via ``arch``) on the tau-standardised residuals

This is a *variance* model, so as a ``BaseForecaster`` its point ``predict`` is
the random-walk mean (zeros on log-returns) and the value is in
``predict_interval`` / ``forecast_variance`` — score it with the interval / CRPS
tools, not RMSE.  Falls back to an unconditional-variance band if estimation
fails.  ``arch`` is optional; without it the short-run component is skipped.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster
from src.models_midas import _beta_weights

logger = logging.getLogger(__name__)


class GarchMidasModel(BaseForecaster):
    """Two-step GARCH-MIDAS conditional-variance model.

    Parameters
    ----------
    macro_col:
        Column carrying the monthly macro driver of long-run variance.  None ->
        use the target's own lagged squared returns (a realized-variance proxy,
        i.e. a component-GARCH long run).
    n_lags:
        Number of MIDAS lags of the macro driver.
    theta2_grid:
        Beta-weight shape parameters searched (in-sample SSR of the long-run fit).
    """

    def __init__(
        self,
        macro_col: Optional[str] = None,
        n_lags: int = 12,
        theta2_grid: Optional[list[float]] = None,
    ) -> None:
        self.macro_col = macro_col
        self.n_lags = n_lags
        self.theta2_grid = theta2_grid or [1, 2, 3, 6, 9, 12]
        self.theta_ = float("nan")
        self._m = 0.0
        self._theta2 = 1.0
        self._g_next = 1.0
        self._tau_next = 1.0
        self._uncond = 1.0
        self._fitted = False

    def _macro_series(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        if self.macro_col is not None and self.macro_col in X.columns:
            return np.nan_to_num(X[self.macro_col].to_numpy(dtype=float))
        return y ** 2                                    # realized-variance proxy

    def _midas(self, mv: np.ndarray, theta2: float) -> np.ndarray:
        """MIDAS-weighted macro using lags 1..n_lags (no contemporaneous term)."""
        K = self.n_lags
        w = _beta_weights(K, theta2)
        n = len(mv)
        out = np.full(n, np.nan)
        for t in range(K, n):
            out[t] = float(w @ mv[t - K:t][::-1])        # lag1 gets weight w[0]
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GarchMidasModel":
        yv = np.asarray(y, dtype=float)
        self._uncond = float(np.var(yv)) or 1.0
        try:
            mv = self._macro_series(X, yv)
            logr2 = np.log(yv ** 2 + 1e-12)

            # 1) Long-run component: grid the Beta shape, OLS for (m, theta).
            best = (np.inf, 1.0, 0.0, 0.0)
            for th2 in self.theta2_grid:
                mid = self._midas(mv, th2)
                ok = np.isfinite(mid) & np.isfinite(logr2)
                if ok.sum() < self.n_lags + 5:
                    continue
                xm = mid[ok]
                A = np.column_stack([np.ones(ok.sum()), xm])
                coef, *_ = np.linalg.lstsq(A, logr2[ok], rcond=None)
                resid = logr2[ok] - A @ coef
                sse = float(resid @ resid)
                if sse < best[0]:
                    best = (sse, th2, float(coef[0]), float(coef[1]))
            _, self._theta2, self._m, self.theta_ = best

            mid_all = self._midas(mv, self._theta2)
            log_tau = self._m + self.theta_ * np.nan_to_num(mid_all, nan=np.nanmean(mid_all))
            tau = np.exp(np.clip(log_tau, -20, 20))

            # 2) Short-run GARCH(1,1) on tau-standardised residuals.
            u = yv / np.sqrt(np.where(tau > 0, tau, self._uncond))
            su = float(np.std(u)) or 1.0                 # residual scale (calibration gap)
            self._g_next = su ** 2
            try:
                import warnings as _w

                from arch import arch_model
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    am = arch_model(u / su, mean="Zero", vol="GARCH", p=1, q=1,
                                    dist="normal", rescale=False)   # unit-scaled -> well conditioned
                    res = am.fit(disp="off", show_warning=False)
                    fc = res.forecast(horizon=1, reindex=False)
                self._g_next = float(fc.variance.values[-1, 0]) * (su ** 2)
            except Exception as exc:
                logger.info("GarchMidas: arch short-run unavailable (%s); long-run only", exc)

            # Next-step long-run from the most recent macro lags.
            mid_next = self._midas(np.append(mv, mv[-1]), self._theta2)[-1]
            self._tau_next = float(np.exp(np.clip(self._m + self.theta_ * mid_next, -20, 20)))
            if not np.isfinite(self._tau_next) or self._tau_next <= 0:
                self._tau_next = self._uncond
            self._fitted = True
        except Exception as exc:
            logger.warning("GarchMidasModel: estimation failed (%s) — unconditional band.", exc)
            self._fitted = False
        return self

    def forecast_variance(self) -> float:
        """One-step-ahead total conditional variance ``tau_next * g_next``."""
        if not self._fitted:
            return self._uncond
        return max(self._tau_next * self._g_next, 1e-12)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))                          # variance model: RW mean

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.80) -> pd.DataFrame:
        from scipy import stats
        sd = float(np.sqrt(self.forecast_variance()))
        z = float(stats.norm.ppf(1.0 - (1.0 - alpha) / 2.0))
        n = len(X)
        return pd.DataFrame({"lower": np.full(n, -z * sd),
                             "median": np.zeros(n),
                             "upper": np.full(n, z * sd)}, index=X.index)

    @property
    def name(self) -> str:
        return "GARCH-MIDAS"
