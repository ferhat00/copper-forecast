"""
models_futures.py
=================
``KalmanFuturesForecaster`` — a time-varying-parameter Kalman fusion of the
copper futures curve (and, optionally, an analyst-consensus forecast).

The futures curve gives the single most copper-native level forecast: the
cash-to-3-month basis *is* the market's implied return to the 3-month point
(see :class:`src.models.FuturesBasisBenchmark`).  Cortazar et al. (2024) find the
LME futures curve beats the no-change random walk at every horizon, best when a
Kalman filter blends the curve with analyst consensus and lets the loading on
each drift over time.

This model regresses the forward return on the curve-implied return (and an
optional analyst-implied return) with **time-varying coefficients** evolving as a
random walk, estimated by the Kalman filter.  The final filtered coefficient is a
readable, drifting *reliability weight* on the futures curve — when the basis is
informative the loading rises toward 1; when it is noise the filter shrinks it
toward 0 (the RW).  Falls back to the random walk (zeros) when no basis column is
present, so it stays drop-in for ``compare_models``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class KalmanFuturesForecaster(BaseForecaster):
    """Kalman TVP regression of the forward return on the futures-curve carry.

    Parameters
    ----------
    horizon, tenor_periods:
        The curve-implied return is ``basis_pct * horizon / tenor_periods``
        (3 months / 3 periods for the monthly model), matching
        :class:`src.models.FuturesBasisBenchmark`.
    basis_col:
        Fractional curve-implied-return column (``copper_basis_pct``).
    analyst_col:
        Optional analyst-consensus implied-return column to fuse alongside the
        curve.  Ignored gracefully when absent.
    q:
        Process-noise scale (relative to the observation variance) governing how
        fast the coefficients drift; larger => more adaptive.
    prior_var:
        Diffuse prior variance on the coefficients.
    """

    def __init__(
        self,
        horizon: int = 1,
        tenor_periods: int = 3,
        basis_col: str = "copper_basis_pct",
        analyst_col: Optional[str] = None,
        q: float = 1e-3,
        prior_var: float = 10.0,
    ) -> None:
        self.horizon = horizon
        self.tenor_periods = max(int(tenor_periods), 1)
        self.basis_col = basis_col
        self.analyst_col = analyst_col
        self.q = q
        self.prior_var = prior_var
        self._beta: Optional[np.ndarray] = None
        self._labels: list[str] = []

    def _design(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        cols, labels = [], []
        if self.basis_col in X.columns:
            implied = np.nan_to_num(X[self.basis_col].to_numpy(dtype=float))
            cols.append(implied * (self.horizon / self.tenor_periods))
            labels.append("futures_basis")
        if self.analyst_col and self.analyst_col in X.columns:
            cols.append(np.nan_to_num(X[self.analyst_col].to_numpy(dtype=float)))
            labels.append("analyst")
        if not cols:
            return None
        self._labels = labels
        return np.column_stack(cols)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KalmanFuturesForecaster":
        Z = self._design(X)
        if Z is None:
            self._beta = None
            return self
        yv = np.asarray(y, dtype=float)
        p = Z.shape[1]
        R = float(np.nanvar(yv)) or 1e-6                 # observation noise
        Q = np.eye(p) * (self.q * R)                     # coefficient drift
        beta = np.zeros(p)
        P = np.eye(p) * self.prior_var
        for t in range(len(yv)):
            P = P + Q                                    # transition (random-walk betas)
            xt, yt = Z[t], yv[t]
            if not np.isfinite(yt) or not np.all(np.isfinite(xt)):
                continue
            S = float(xt @ P @ xt) + R
            if S <= 0:
                continue
            K = (P @ xt) / S                             # Kalman gain
            beta = beta + K * (yt - xt @ beta)
            P = P - np.outer(K, xt) @ P
        self._beta = beta
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._design(X)
        if self._beta is None or Z is None:
            return np.zeros(len(X))
        return Z @ self._beta

    @property
    def coef_(self) -> pd.Series:
        """Final filtered loadings — the drifting reliability of each signal."""
        if self._beta is None:
            return pd.Series(dtype=float)
        return pd.Series(self._beta, index=self._labels)

    @property
    def name(self) -> str:
        return "Kalman-Futures"
