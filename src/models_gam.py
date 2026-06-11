"""
models_gam.py
=============
Generalised Additive Model (GAM) forecaster — interpretable nonlinearity.

Models the h-step copper return as a sum of smooth, individually-plottable
functions of each driver (Wood 2017): ``y = b0 + sum_j f_j(x_j) + e``.  Each
``f_j`` is a penalised spline, so the model captures thresholds / saturation /
asymmetry while every effect stays readable as a partial-dependence curve — the
interpretable middle ground between a linear model and a black-box.

Uses ``pygam`` (true penalised GAM with prediction intervals) when available,
and falls back to a scikit-learn spline-basis + ridge model otherwise.  To stay
tractable and avoid over-fitting the short monthly sample, it fits splines on a
small, economically-screened subset of features (default: the most correlated
``max_features``).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class GAMForecaster(BaseForecaster):
    """Additive penalised-spline model on a screened feature subset.

    Parameters
    ----------
    feature_cols:
        Columns to smooth.  None -> auto-pick the ``max_features`` most
        correlated with the target.
    max_features:
        Cap on auto-selected features (keeps the additive model legible).
    n_splines:
        Spline basis size per feature.
    lam:
        Smoothing penalty (higher = smoother).  Used by both backends.
    """

    def __init__(
        self,
        feature_cols: Optional[list[str]] = None,
        max_features: int = 6,
        n_splines: int = 10,
        lam: float = 0.6,
    ) -> None:
        self.feature_cols = feature_cols
        self.max_features = max_features
        self.n_splines = n_splines
        self.lam = lam
        self._backend: Optional[str] = None
        self._cols: list[str] = []
        self._model = None
        self._resid_std = 1.0

    def _select(self, X: pd.DataFrame, y: np.ndarray) -> list[str]:
        if self.feature_cols is not None:
            return [c for c in self.feature_cols if c in X.columns]
        scores = {}
        for c in X.columns:
            s = X[c].to_numpy(dtype=float)
            if np.nanstd(s) > 0:
                sc = np.corrcoef(np.nan_to_num(s), y)[0, 1]
                scores[c] = 0.0 if not np.isfinite(sc) else abs(sc)
        return sorted(scores, key=scores.get, reverse=True)[: self.max_features]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GAMForecaster":
        yv = np.asarray(y, dtype=float)
        self._cols = self._select(X, yv) or list(X.columns[: self.max_features])
        Xc = np.nan_to_num(X[self._cols].to_numpy(dtype=float))

        try:
            from pygam import LinearGAM, s

            terms = s(0, n_splines=self.n_splines)
            for i in range(1, Xc.shape[1]):
                terms = terms + s(i, n_splines=self.n_splines)
            self._model = LinearGAM(terms, lam=self.lam).fit(Xc, yv)
            self._backend = "pygam"
        except Exception as exc:
            logger.warning("GAMForecaster: pygam unavailable/failed (%s) — "
                           "using sklearn spline+ridge fallback.", exc)
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import SplineTransformer, StandardScaler

            self._model = Pipeline([
                ("scale", StandardScaler()),
                ("spline", SplineTransformer(n_knots=min(self.n_splines, 6),
                                             degree=3, include_bias=False)),
                ("ridge", Ridge(alpha=max(self.lam, 1e-3))),
            ]).fit(Xc, yv)
            self._backend = "spline"

        resid = yv - self._predict_raw(Xc)
        self._resid_std = float(np.std(resid)) or 1.0
        return self

    def _predict_raw(self, Xc: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(Xc), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        Xc = np.nan_to_num(X[self._cols].to_numpy(dtype=float))
        return np.nan_to_num(self._predict_raw(Xc), nan=0.0)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.80) -> pd.DataFrame:
        Xc = np.nan_to_num(X[self._cols].to_numpy(dtype=float))
        mid = np.nan_to_num(self._predict_raw(Xc), nan=0.0)
        if self._backend == "pygam":
            try:
                pi = np.asarray(self._model.prediction_intervals(Xc, width=alpha))
                return pd.DataFrame({"lower": pi[:, 0], "median": mid, "upper": pi[:, 1]},
                                    index=X.index)
            except Exception:
                pass
        from scipy import stats
        z = float(stats.norm.ppf(1.0 - (1.0 - alpha) / 2.0))
        d = z * self._resid_std
        return pd.DataFrame({"lower": mid - d, "median": mid, "upper": mid + d}, index=X.index)

    def partial_dependence(self, feature: str, grid: int = 50):
        """Partial-dependence curve for one feature (pygam backend only).

        Returns ``(xs, ys)`` arrays of the fitted smooth f_j over the feature's
        range, or ``None`` if unavailable (e.g. the sklearn fallback).
        """
        if self._backend != "pygam" or feature not in self._cols:
            return None
        i = self._cols.index(feature)
        XX = self._model.generate_X_grid(term=i, n=grid)
        ys = self._model.partial_dependence(term=i, X=XX)
        return XX[:, i], np.asarray(ys)

    @property
    def name(self) -> str:
        return "GAM"
