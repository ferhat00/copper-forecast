"""
models_baselines.py
===================
Two strong, dependency-light *baselines to beat* for the monthly pipeline.

``AR1Model``
    The autoregressive benchmark the literature demands *alongside* the random
    walk: Wang & Zhang (2024) and most recent copper/oil studies benchmark
    machine-learning forecasters against AR(1), not just the zero-drift RW.  A
    model that beats ``NaiveModel`` but not AR(1) has only captured trivial
    persistence.  Interpretable: a single readable momentum/reversal coefficient.

``DLinearForecaster``
    The decomposition + linear model of Zeng et al. (2023, "Are Transformers
    Effective for Time Series Forecasting?") that famously matches or beats deep
    Transformers on long-horizon forecasting.  Included as a cheap, transparent
    bar: any fancy model (boosting, foundation model) that cannot clear DLinear
    *and* the RW has no real edge.

Both subclass :class:`src.models.BaseForecaster`, so they drop straight into
``compare_models`` / the config-driven lineup.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


def _lag_of(col: str) -> int:
    """Trailing integer in a return-lag column name (``copper_ret_3m`` -> 3)."""
    m = re.search(r"_(\d+)", col)
    return int(m.group(1)) if m else 10**6


class AR1Model(BaseForecaster):
    """AR(1) benchmark — forward return as a line in the last realised return.

    Regresses the forward target on the most recent realised return column
    (prefers the shortest-lag ``copper_ret_*`` column, else any ``*_ret_*``).
    Falls back to the in-sample mean (a drift term) when no return column is
    present, so it is always a strictly stronger benchmark than the zero RW.
    """

    def __init__(self, momentum_col: Optional[str] = None) -> None:
        self.momentum_col = momentum_col
        self._b0 = 0.0
        self._b1 = 0.0
        self._col: Optional[str] = None

    def _pick_col(self, X: pd.DataFrame) -> Optional[str]:
        if self.momentum_col and self.momentum_col in X.columns:
            return self.momentum_col
        cu = [c for c in X.columns if c.startswith("copper_ret_")]
        if cu:
            return sorted(cu, key=_lag_of)[0]
        ret = [c for c in X.columns if "_ret_" in c]
        return sorted(ret, key=_lag_of)[0] if ret else None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AR1Model":
        yv = np.asarray(y, dtype=float)
        self._col = self._pick_col(X)
        if self._col is None:
            self._b0 = float(np.nanmean(yv)) if np.isfinite(yv).any() else 0.0
            self._b1 = 0.0
            return self
        x = np.asarray(X[self._col], dtype=float)
        ok = np.isfinite(x) & np.isfinite(yv)
        if ok.sum() < 5:
            self._b0 = float(np.nanmean(yv)) if np.isfinite(yv).any() else 0.0
            self._b1 = 0.0
            return self
        A = np.column_stack([np.ones(ok.sum()), x[ok]])
        coef, *_ = np.linalg.lstsq(A, yv[ok], rcond=None)
        self._b0, self._b1 = float(coef[0]), float(coef[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._col is None or self._col not in X.columns:
            return np.full(len(X), self._b0)
        x = np.nan_to_num(np.asarray(X[self._col], dtype=float))
        return self._b0 + self._b1 * x

    @property
    def params_(self) -> pd.Series:
        return pd.Series({"const": self._b0, f"slope[{self._col}]": self._b1})

    @property
    def name(self) -> str:
        return "AR(1)"


class DLinearForecaster(BaseForecaster):
    """DLinear-style decomposition + linear baseline (Zeng et al. 2023).

    Uses the copper return-lag columns as a coarse lookback window, splits each
    into a trend (moving average along the lag axis) and a remainder, then fits
    two ridge linear heads (one per component) whose sum forecasts the forward
    return.  Falls back to a plain ridge on all features when fewer than two
    return-lag columns are available.
    """

    def __init__(self, kernel: int = 3, alpha: float = 1.0) -> None:
        # kernel forced odd so the moving-average keeps the lookback length.
        self.kernel = kernel if kernel % 2 == 1 else kernel + 1
        self.alpha = alpha
        self._cols: list[str] = []
        self._fallback = False
        self._model = None
        self._scaler = None

    def _lookback_cols(self, X: pd.DataFrame) -> list[str]:
        cu = [c for c in X.columns if c.startswith("copper_ret_")]
        return sorted(cu, key=_lag_of)

    def _decompose(self, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = self.kernel
        if k <= 1 or L.shape[1] < 2:
            return L.copy(), np.zeros_like(L)
        pad = k // 2
        Lp = np.pad(L, ((0, 0), (pad, pad)), mode="edge")
        kern = np.ones(k) / k
        trend = np.vstack([np.convolve(Lp[i], kern, mode="valid") for i in range(Lp.shape[0])])
        trend = trend[:, : L.shape[1]]
        return trend, L - trend

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DLinearForecaster":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self._cols = self._lookback_cols(X)
        yv = np.asarray(y, dtype=float)
        if len(self._cols) < 2:
            self._fallback = True
            feat = X.select_dtypes("number").fillna(0.0).to_numpy()
        else:
            self._fallback = False
            L = np.nan_to_num(X[self._cols].to_numpy(dtype=float))
            trend, seasonal = self._decompose(L)
            feat = np.hstack([trend, seasonal])
        self._scaler = StandardScaler().fit(feat)
        self._model = Ridge(alpha=self.alpha).fit(self._scaler.transform(feat), yv)
        return self

    def _features(self, X: pd.DataFrame) -> np.ndarray:
        if self._fallback:
            return X.select_dtypes("number").fillna(0.0).to_numpy()
        L = np.nan_to_num(X[self._cols].to_numpy(dtype=float))
        trend, seasonal = self._decompose(L)
        return np.hstack([trend, seasonal])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("DLinearForecaster not fitted yet.")
        feat = self._features(X)
        return self._model.predict(self._scaler.transform(feat))

    @property
    def name(self) -> str:
        return "DLinear"
