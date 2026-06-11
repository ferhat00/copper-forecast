"""
models_markov.py
================
Markov-switching regression forecaster (Hamilton 1989) — the formal upgrade of
the repo's descriptive 3-state HMM into a *mean* model whose coefficients and
variance switch between latent regimes.

Wraps ``statsmodels`` ``MarkovRegression``: a latent state follows a Markov
chain, and conditional on the state the copper return is a (regime-dependent)
linear function of a few exogenous drivers with regime-dependent variance.  The
interpretable outputs are the regime means/variances, the transition matrix and
expected durations, and the smoothed "which-regime-are-we-in" probability path.

The h-step forecast is the regime-weighted prediction using the one-step-ahead
regime probabilities (last filtered state propagated through the transition
matrix), and ``predict_interval`` returns the Gaussian-mixture band across
regimes.  Per the literature the OOS *mean* edge over a random walk is modest —
the win is in the density and the regime narrative — so score intervals
separately.  Falls back to an intercept-only model if estimation fails.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class MarkovSwitchingForecaster(BaseForecaster):
    """Markov-switching regression with regime-weighted forecasts.

    Parameters
    ----------
    k_regimes:
        Number of latent regimes.
    exog_cols:
        Exogenous driver columns.  None -> auto-pick the ``max_exog`` columns
        most correlated with the target (keeps the parameter count sane on the
        short monthly sample).
    max_exog:
        Cap on auto-selected exogenous regressors.
    switching_variance:
        If True, each regime has its own residual variance (recommended for
        copper's calm/turbulent regimes).
    """

    def __init__(
        self,
        k_regimes: int = 2,
        exog_cols: Optional[list[str]] = None,
        max_exog: int = 3,
        switching_variance: bool = True,
    ) -> None:
        self.k_regimes = k_regimes
        self.exog_cols = exog_cols
        self.max_exog = max_exog
        self.switching_variance = switching_variance
        self._res = None
        self._exog: list[str] = []
        self._beta: list = []
        self._w_next: Optional[np.ndarray] = None
        self._intercept_only = False
        self._mean = 0.0

    def _select_exog(self, X: pd.DataFrame, y: np.ndarray) -> list[str]:
        if self.exog_cols is not None:
            return [c for c in self.exog_cols if c in X.columns]
        scores = {}
        for c in X.columns:
            s = X[c].to_numpy(dtype=float)
            if np.nanstd(s) > 0:
                sc = np.corrcoef(np.nan_to_num(s), y)[0, 1]
                scores[c] = 0.0 if not np.isfinite(sc) else abs(sc)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[: self.max_exog]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarkovSwitchingForecaster":
        yv = np.asarray(y, dtype=float)
        self._mean = float(np.nanmean(yv))
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

            self._exog = self._select_exog(X, yv)
            Xe = (np.nan_to_num(X[self._exog].to_numpy(dtype=float))
                  if self._exog else None)
            mod = MarkovRegression(
                yv, k_regimes=self.k_regimes, exog=Xe,
                switching_variance=self.switching_variance,
                switching_exog=bool(self._exog))
            self._res = mod.fit(em_iter=25, maxiter=100, disp=False)

            # One-step-ahead regime weights: last filtered state x transition.
            fp = np.asarray(self._res.filtered_marginal_probabilities)
            filt = fp[-1] if fp.shape[0] != self.k_regimes else fp[:, -1]
            filt = np.asarray(filt, dtype=float).ravel()[: self.k_regimes]
            P = np.asarray(mod.regime_transition_matrix(self._res.params))
            P = P[:, :, 0] if P.ndim == 3 else P
            w = P @ filt
            self._w_next = w / w.sum() if w.sum() > 0 else np.full(self.k_regimes, 1.0 / self.k_regimes)

            # Regime-conditional (const, betas, variance).
            par = self._res.params
            self._beta = []
            for k in range(self.k_regimes):
                ck = float(par.get(f"const[{k}]", par.get("const", 0.0)))
                bvec = np.array([
                    float(par.get(f"x{j + 1}[{k}]", par.get(f"x{j + 1}", 0.0)))
                    for j in range(len(self._exog))])
                sig = float(par.get(f"sigma2[{k}]", par.get("sigma2", np.var(yv) + 1e-8)))
                self._beta.append((ck, bvec, max(sig, 1e-10)))
            self._intercept_only = False
        except Exception as exc:
            logger.warning("MarkovSwitchingForecaster: estimation failed (%s) — "
                           "falling back to intercept-only.", exc)
            self._intercept_only = True
        return self

    def _regime_means(self, X: pd.DataFrame) -> np.ndarray:
        Xe = (np.nan_to_num(X[self._exog].to_numpy(dtype=float))
              if self._exog else np.zeros((len(X), 0)))
        means = np.empty((self.k_regimes, len(X)))
        for k, (ck, bvec, _sig) in enumerate(self._beta):
            means[k] = ck + (Xe @ bvec if bvec.size else 0.0)
        return means

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._intercept_only or self._res is None:
            return np.full(len(X), self._mean)
        means = self._regime_means(X)
        return np.nan_to_num(self._w_next @ means, nan=0.0)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.80) -> pd.DataFrame:
        if self._intercept_only or self._res is None:
            mid = self.predict(X)
            return pd.DataFrame({"lower": mid, "median": mid, "upper": mid}, index=X.index)
        from scipy import stats

        means = self._regime_means(X)
        w = self._w_next[:, None]
        var_k = np.array([s for _, _, s in self._beta])[:, None]
        m = (w * means).sum(axis=0)
        second = (w * (var_k + means ** 2)).sum(axis=0)
        var = np.maximum(second - m ** 2, 1e-12)
        z = float(stats.norm.ppf(1.0 - (1.0 - alpha) / 2.0))
        sd = np.sqrt(var)
        return pd.DataFrame({"lower": m - z * sd, "median": m, "upper": m + z * sd},
                            index=X.index)

    # -- interpretability -------------------------------------------------
    def smoothed_probabilities(self) -> Optional[np.ndarray]:
        """Smoothed P(regime | full data) — the 'which regime' path."""
        if self._res is None:
            return None
        return np.asarray(self._res.smoothed_marginal_probabilities)

    def expected_durations(self) -> Optional[np.ndarray]:
        """Expected persistence (periods) of each regime, ``1/(1 - p_kk)``."""
        if self._res is None:
            return None
        try:
            P = np.asarray(self._res.model.regime_transition_matrix(self._res.params))
            P = P[:, :, 0] if P.ndim == 3 else P
            return 1.0 / (1.0 - np.clip(np.diag(P), 0.0, 1.0 - 1e-9))
        except Exception:
            return None

    @property
    def name(self) -> str:
        return f"Markov-Switching ({self.k_regimes})"
