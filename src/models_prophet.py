"""
models_prophet.py
=================
Facebook Prophet model wrapper conforming to the BaseForecaster interface.

Prophet expects a DataFrame with ``ds`` (datetime) and ``y`` (target) columns.
This adapter translates between the standard ``fit(X, y)`` / ``predict(X)``
interface and Prophet's native API.

Prophet is an optional heavy dependency — import errors are handled gracefully.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)

DEFAULT_REGRESSOR_COLS = [
    "dxy_level",
    "real_yield_level",
    "gold_copper_ratio",
    "copper_vol_22d",
]


class ProphetModel(BaseForecaster):
    """Prophet regressor with exogenous regressors.

    Parameters
    ----------
    regressor_cols:
        Column names to add as extra regressors.  Defaults to
        ``DEFAULT_REGRESSOR_COLS`` filtered to columns present in ``X``.
    interval_width:
        Width of the prediction interval (default 0.80 for 80% CI).
    **prophet_kwargs:
        Additional keyword arguments passed to ``Prophet()``.
    """

    def __init__(
        self,
        regressor_cols: Optional[list[str]] = None,
        interval_width: float = 0.80,
        **prophet_kwargs: Any,
    ) -> None:
        self._requested_regressors = regressor_cols
        self.interval_width = interval_width
        self._prophet_kwargs = prophet_kwargs
        self._regressor_cols: list[str] = []
        self._model: Any = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProphetModel":
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError(
                "prophet is required for ProphetModel. "
                "Install with: pip install prophet"
            ) from exc

        # Suppress verbose Prophet / cmdstanpy output
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        os.environ.setdefault("CMDSTANPY_SUPPRESS_OUTPUT", "1")

        # Resolve regressor columns
        requested = self._requested_regressors or DEFAULT_REGRESSOR_COLS
        self._regressor_cols = [c for c in requested if c in X.columns]

        # Build Prophet DataFrame
        df_prophet = pd.DataFrame({
            "ds": X.index,
            "y": y.values,
        })
        for col in self._regressor_cols:
            df_prophet[col] = X[col].values

        # Drop NaN rows
        df_prophet = df_prophet.dropna()

        # Create and configure model
        self._model = Prophet(
            interval_width=self.interval_width,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            **self._prophet_kwargs,
        )
        for col in self._regressor_cols:
            self._model.add_regressor(col)

        try:
            self._model.fit(df_prophet)
            self._fitted = True
            logger.info(
                "Prophet: fitted with %d regressors on %d observations",
                len(self._regressor_cols), len(df_prophet),
            )
        except Exception as exc:
            logger.warning("Prophet: fit failed — %s. Falling back to zero forecast.", exc)
            self._fitted = False

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X))

        future = self._make_future_df(X)

        try:
            forecast = self._model.predict(future)
            return forecast["yhat"].values
        except Exception as exc:
            logger.warning("Prophet: predict failed — %s", exc)
            return np.zeros(len(X))

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.80
    ) -> pd.DataFrame:
        """Return prediction intervals from Prophet's built-in uncertainty.

        Note: The interval width is set at model creation time via
        ``interval_width``.  The ``alpha`` parameter here is for API
        consistency; for precise control, set ``interval_width`` in the
        constructor.

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

        future = self._make_future_df(X)

        try:
            forecast = self._model.predict(future)
            return pd.DataFrame(
                {
                    "lower": forecast["yhat_lower"].values,
                    "median": forecast["yhat"].values,
                    "upper": forecast["yhat_upper"].values,
                },
                index=X.index,
            )
        except Exception as exc:
            logger.warning("Prophet: predict_interval failed — %s", exc)
            zeros = np.zeros(len(X))
            return pd.DataFrame(
                {"lower": zeros, "median": zeros, "upper": zeros},
                index=X.index,
            )

    def _make_future_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build a Prophet-compatible future DataFrame from the feature matrix."""
        future = pd.DataFrame({"ds": X.index})
        for col in self._regressor_cols:
            if col in X.columns:
                future[col] = X[col].values
            else:
                future[col] = 0.0
        return future

    @property
    def name(self) -> str:
        return "Prophet"
