"""
models_arimax.py
================
ARIMAX model wrapper conforming to the BaseForecaster interface.

Uses ``statsmodels.tsa.statespace.SARIMAX`` with exogenous regressors.
Since the target is already log-returns (quasi-stationary), the default
differencing order is d=0 (ARMAX rather than ARIMAX).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)

DEFAULT_EXOG_COLS = [
    "dxy_level",
    "real_yield_level",
    "gold_copper_ratio",
    "copper_vol_22d",
]


class ARIMAXModel(BaseForecaster):
    """ARIMAX regressor with configurable exogenous features.

    Parameters
    ----------
    order:
        (p, d, q) order for SARIMAX.  Default ``(2, 0, 1)`` — since the
        target is already a log-return, d=0 is appropriate.
    exog_cols:
        Column names to use as exogenous regressors.  Must be a small
        subset (3-6 features) for ARIMAX convergence.  If *None*, uses
        ``DEFAULT_EXOG_COLS`` filtered to columns present in ``X``.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 0, 1),
        exog_cols: Optional[list[str]] = None,
    ) -> None:
        self.order = order
        self._requested_exog = exog_cols
        self._exog_cols: list[str] = []
        self._result: Any = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ARIMAXModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Resolve exog columns to those actually present in X
        requested = self._requested_exog or DEFAULT_EXOG_COLS
        self._exog_cols = [c for c in requested if c in X.columns]

        exog = X[self._exog_cols].values if self._exog_cols else None

        # Use integer-indexed endog to avoid date alignment issues
        endog = np.asarray(y.values, dtype=float)
        self._train_len = len(endog)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = SARIMAX(
                    endog,
                    exog=exog,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self._result = model.fit(disp=False, maxiter=200)
                self._fitted = True
                logger.info(
                    "ARIMAX: fitted order=%s with %d exog features (AIC=%.1f)",
                    self.order, len(self._exog_cols), self._result.aic,
                )
            except Exception as exc:
                logger.warning("ARIMAX: fit failed — %s. Falling back to zero forecast.", exc)
                self._fitted = False

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X))

        exog = X[self._exog_cols].values if self._exog_cols else None
        n = len(X)

        try:
            start = self._train_len
            end = self._train_len + n - 1
            preds = self._result.predict(start=start, end=end, exog=exog)
            return np.asarray(preds, dtype=float)
        except Exception as exc:
            logger.warning("ARIMAX: predict failed — %s", exc)
            return np.zeros(n)

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.80
    ) -> pd.DataFrame:
        """Return prediction intervals using SARIMAX built-in confidence bands.

        Parameters
        ----------
        X:
            Feature matrix for the forecast horizon.
        alpha:
            CI width (e.g. 0.80 → 10th and 90th percentile bounds).

        Returns
        -------
        pd.DataFrame
            Columns: ``lower``, ``median``, ``upper``.
        """
        if not self._fitted:
            zeros = np.zeros(len(X))
            return pd.DataFrame(
                {"lower": zeros, "median": zeros, "upper": zeros},
                index=X.index,
            )

        exog = X[self._exog_cols].values if self._exog_cols else None
        n = len(X)

        try:
            forecast = self._result.get_forecast(steps=n, exog=exog)
            ci = forecast.conf_int(alpha=1 - alpha)
            median = forecast.predicted_mean
            return pd.DataFrame(
                {
                    "lower": np.asarray(ci.iloc[:, 0].values, dtype=float),
                    "median": np.asarray(median, dtype=float),
                    "upper": np.asarray(ci.iloc[:, 1].values, dtype=float),
                },
                index=X.index,
            )
        except Exception as exc:
            logger.warning("ARIMAX: predict_interval failed — %s", exc)
            zeros = np.zeros(len(X))
            return pd.DataFrame(
                {"lower": zeros, "median": zeros, "upper": zeros},
                index=X.index,
            )

    @property
    def name(self) -> str:
        return "ARIMAX"
