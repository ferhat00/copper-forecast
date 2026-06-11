"""
models_midas.py
===============
MIDAS-style restricted distributed-lag forecaster (Ghysels, Santa-Clara &
Valkanov).

Genuine MIDAS mixes a *high*-frequency predictor into a *low*-frequency target;
on the already-monthly feature matrix here it specialises to a restricted lag
polynomial: each predictor's ordered lag columns (e.g. ``dxy_ret_1m``,
``dxy_ret_3m``, ``dxy_ret_6m``, ``dxy_ret_12m``) are compressed into a single
regressor by a smooth **Beta weight curve** with one shape parameter, then a
plain OLS maps the compressed factors (plus optional extra regressors) to the
h-step copper return.

Two things make it interpretable: the slope on each compressed factor is its
elasticity, and the estimated weight curve (``weight_curve``) is plottable — it
shows *which lags matter*, the whole point of MIDAS's parsimony.  Because it
reads existing lag columns rather than re-shifting inside ``predict``, it has no
forecast-origin boundary problem in walk-forward CV.

Requires ``statsmodels`` (already a core dependency).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


def _beta_weights(k: int, theta2: float) -> np.ndarray:
    """Normalised Beta(1, theta2) weights over ``k`` ordered lags (decaying)."""
    if k <= 1:
        return np.ones(1)
    x = np.linspace(1e-6, 1.0 - 1e-6, k)
    w = (1.0 - x) ** (theta2 - 1.0)        # Beta(1, theta2): monotone decay
    s = w.sum()
    return w / s if s > 0 else np.full(k, 1.0 / k)


class MIDASForecaster(BaseForecaster):
    """Restricted distributed-lag (MIDAS) regression with Beta lag weights.

    Parameters
    ----------
    lag_groups:
        Mapping ``factor_name -> [ordered lag column names]`` (lag 0 first).
        None -> auto-detect from ``{base}_lag_{n}`` columns in ``X`` (each base
        plus its lags, ordered by ``n``).
    extra_cols:
        Optional extra regressors entered linearly (no weighting).
    theta_grid:
        Candidate Beta shape parameters searched per factor (in-sample SSR).
    """

    def __init__(
        self,
        lag_groups: Optional[dict] = None,
        extra_cols: Optional[list[str]] = None,
        theta_grid: Optional[list[float]] = None,
    ) -> None:
        self.lag_groups = lag_groups
        self.extra_cols = extra_cols
        self.theta_grid = theta_grid or [1, 2, 3, 4, 6, 8, 10, 12, 15]
        self._res = None
        self._groups: dict = {}
        self._extra: list[str] = []
        self._theta: dict = {}
        self._cols: list[str] = []
        self._intercept_only = False

    def _auto_groups(self, X: pd.DataFrame) -> dict:
        bylag: dict = {}
        for c in X.columns:
            m = re.match(r"^(.*)_lag_(\d+)$", str(c))
            if m:
                bylag.setdefault(m.group(1), {})[int(m.group(2))] = c
        groups = {}
        for base, d in bylag.items():
            if base in X.columns:
                groups[base] = [base] + [d[n] for n in sorted(d)]
        return groups

    def _design(self, X: pd.DataFrame, fit: bool, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        parts: dict = {}
        for name, cols in self._groups.items():
            cols = [c for c in cols if c in X.columns]
            if not cols:
                continue
            M = np.nan_to_num(X[cols].to_numpy(dtype=float), nan=0.0)
            if fit:
                yc = y - y.mean()
                best = (np.inf, 1.0)
                for th in self.theta_grid:
                    r = M @ _beta_weights(M.shape[1], th)
                    rc = r - r.mean()
                    denom = float(rc @ rc)
                    if denom <= 0:
                        continue
                    b = float(rc @ yc) / denom
                    resid = yc - b * rc
                    sse = float(resid @ resid)
                    if sse < best[0]:
                        best = (sse, th)
                self._theta[name] = best[1]
            w = _beta_weights(M.shape[1], self._theta.get(name, 1.0))
            parts[name] = M @ w
        design = pd.DataFrame(parts, index=X.index)
        for e in self._extra:
            if e in X.columns:
                design[e] = X[e]
        return design

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MIDASForecaster":
        import statsmodels.api as sm

        self._groups = self.lag_groups or self._auto_groups(X)
        self._extra = [c for c in (self.extra_cols or []) if c in X.columns]
        yv = np.asarray(y, dtype=float)
        design = self._design(X, fit=True, y=yv)
        self._cols = list(design.columns)
        if not self._cols:
            logger.warning("MIDASForecaster: no lag groups / extra cols found — "
                           "falling back to an intercept-only (mean) model.")
            self._intercept_only = True
            self._mean = float(yv.mean())
            return self
        frame = design.copy()
        frame["__y__"] = yv
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        self._res = sm.OLS(frame["__y__"].to_numpy(),
                           sm.add_constant(frame[self._cols].to_numpy(), has_constant="add")).fit()
        self._intercept_only = False
        return self

    def _exog(self, X: pd.DataFrame):
        import statsmodels.api as sm
        design = self._design(X, fit=False)[self._cols].fillna(0.0)
        return sm.add_constant(design.to_numpy(), has_constant="add")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._intercept_only:
            return np.full(len(X), getattr(self, "_mean", 0.0))
        if self._res is None:
            raise RuntimeError("Model not fitted yet.")
        return np.nan_to_num(self._res.predict(self._exog(X)), nan=0.0)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.80) -> pd.DataFrame:
        if self._intercept_only or self._res is None:
            mid = self.predict(X)
            return pd.DataFrame({"lower": mid, "median": mid, "upper": mid}, index=X.index)
        sf = self._res.get_prediction(self._exog(X)).summary_frame(alpha=1.0 - alpha)
        return pd.DataFrame({"lower": sf["obs_ci_lower"].to_numpy(),
                             "median": sf["mean"].to_numpy(),
                             "upper": sf["obs_ci_upper"].to_numpy()}, index=X.index)

    def weight_curve(self) -> dict:
        """The estimated Beta lag-weight vector per factor (plottable)."""
        return {n: _beta_weights(len(self._groups[n]), self._theta.get(n, 1.0))
                for n in self._groups}

    @property
    def name(self) -> str:
        return "MIDAS"
