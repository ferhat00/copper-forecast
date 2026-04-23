"""
models_regime.py
================
Regime-conditional forecasting router.

``RegimeRouter`` trains a separate model for each HMM regime identified by
``RegimeDetector`` and routes each prediction to the model trained on data
from that regime.  A global fallback model (trained on all data) handles
rows with unknown or NaN regime labels.

This avoids fitting a single model across structurally different market
environments (bull / sideways / bear copper regimes).
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class RegimeRouter(BaseForecaster):
    """Train one model per regime; route predictions by current regime label.

    Parameters
    ----------
    model_factory:
        Zero-argument callable that returns a fresh, unfitted
        ``BaseForecaster`` instance.  Called once per regime plus once for
        the global fallback.  Example::

            RegimeRouter(model_factory=XGBoostModel)

    regime_col:
        Name of the column in ``X`` that contains integer regime labels
        (0 = bear, 1 = sideways, 2 = bull by convention).  This column
        must be present during both ``fit()`` and ``predict()``.  It is
        **excluded** from the features passed to the inner models.
    min_regime_samples:
        Minimum number of rows required to train a dedicated regime model.
        If a regime has fewer rows, those rows are absorbed into the global
        fallback model's training set and the regime falls back to global
        predictions at inference time.
    """

    def __init__(
        self,
        model_factory: Callable[[], BaseForecaster],
        regime_col: str = "regime",
        min_regime_samples: int = 60,
    ) -> None:
        self.model_factory = model_factory
        self.regime_col = regime_col
        self.min_regime_samples = min_regime_samples

        self._regime_models: dict[int, BaseForecaster] = {}
        self._global_model: Optional[BaseForecaster] = None
        self._feature_cols: list[str] = []
        self._known_regimes: set[int] = set()

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeRouter":
        """Fit per-regime models and a global fallback.

        Steps
        -----
        1. Validate that ``regime_col`` is present.
        2. Identify feature columns (all columns except ``regime_col``).
        3. Fit a global fallback model on full ``X`` (features only).
        4. For each regime with enough samples, fit a dedicated model.
        """
        if self.regime_col not in X.columns:
            raise ValueError(
                f"RegimeRouter: column '{self.regime_col}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        self._feature_cols = [c for c in X.columns if c != self.regime_col]
        X_feats = X[self._feature_cols]

        # Global fallback — trained on all data
        self._global_model = self.model_factory()
        self._global_model.fit(X_feats, y)
        logger.info(
            "RegimeRouter: global model '%s' fitted on %d rows",
            self._global_model.name, len(X),
        )

        # Per-regime models
        regime_labels = X[self.regime_col].dropna()
        unique_regimes = sorted(regime_labels.unique().astype(int))
        self._known_regimes = set(unique_regimes)

        for regime in unique_regimes:
            mask = (X[self.regime_col] == regime) & X[self.regime_col].notna()
            X_r = X_feats[mask]
            y_r = y[mask]

            if len(X_r) < self.min_regime_samples:
                logger.warning(
                    "RegimeRouter: regime %d has only %d samples (< %d) — "
                    "will use global fallback for this regime",
                    regime, len(X_r), self.min_regime_samples,
                )
                continue

            m = self.model_factory()
            m.fit(X_r, y_r)
            self._regime_models[regime] = m
            logger.info(
                "RegimeRouter: regime %d model '%s' fitted on %d rows",
                regime, m.name, len(X_r),
            )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Route each row to the appropriate regime model.

        Rows with NaN or unrecognised regime labels are handled by the
        global fallback model.
        """
        if self._global_model is None:
            raise RuntimeError("RegimeRouter has not been fitted yet.")

        X_feats = X[self._feature_cols]
        predictions = np.full(len(X), np.nan, dtype=float)

        if self.regime_col in X.columns:
            regimes = X[self.regime_col]
        else:
            # Regime column absent at inference time — use global for all
            logger.warning(
                "RegimeRouter: '%s' not in X at predict time; "
                "using global fallback for all rows.", self.regime_col
            )
            regimes = pd.Series(np.nan, index=X.index)

        for regime, model in self._regime_models.items():
            mask = (regimes == regime) & regimes.notna()
            if mask.any():
                predictions[mask.values] = model.predict(X_feats[mask])

        # Fill any remaining NaN predictions (unknown / NaN regime) with global model
        fallback_mask = np.isnan(predictions)
        if fallback_mask.any():
            predictions[fallback_mask] = self._global_model.predict(
                X_feats.iloc[np.where(fallback_mask)[0]]
            )

        return predictions

    @property
    def name(self) -> str:
        inner = self.model_factory()
        return f"RegimeRouter({inner.name})"
