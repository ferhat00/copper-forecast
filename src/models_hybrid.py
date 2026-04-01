"""
models_hybrid.py
================
Hybrid econometric + ML model: ARIMAX backbone with ML residual correction.

Two-stage architecture:
  1. ARIMAX produces a base forecast using a small set of key macro features.
  2. An ML model (LightGBM by default) learns the residuals using the full
     feature matrix, capturing non-linear patterns the ARIMAX misses.

The final prediction is: ARIMAX_pred + ML_residual_pred.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class HybridModel(BaseForecaster):
    """Hybrid econometric + ML residual correction model.

    Parameters
    ----------
    backbone:
        The econometric backbone model (default: ARIMAXModel).
    residual_model:
        The ML model for residual correction (default: LGBMModel).
    """

    def __init__(
        self,
        backbone: Optional[BaseForecaster] = None,
        residual_model: Optional[BaseForecaster] = None,
    ) -> None:
        if backbone is None:
            from src.models_arimax import ARIMAXModel
            backbone = ARIMAXModel()
        if residual_model is None:
            from src.models import LGBMModel
            residual_model = LGBMModel()

        self.backbone = backbone
        self.residual_model = residual_model
        self._residual_std: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridModel":
        # Stage 1: fit the econometric backbone
        self.backbone.fit(X, y)

        # Get in-sample backbone predictions
        backbone_preds = self.backbone.predict(X)

        # Stage 2: compute residuals and fit the ML model on them
        residuals = y.values - backbone_preds
        self._residual_std = float(np.std(residuals))

        self.residual_model.fit(X, pd.Series(residuals, index=y.index))

        logger.info(
            "Hybrid: backbone=%s  residual_model=%s  residual_std=%.6f",
            self.backbone.name, self.residual_model.name, self._residual_std,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        backbone_preds = self.backbone.predict(X)
        residual_preds = self.residual_model.predict(X)
        return backbone_preds + residual_preds

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.80
    ) -> pd.DataFrame:
        """Return prediction intervals combining backbone CI with residual uncertainty.

        Uses the backbone's native prediction intervals widened by the
        empirical standard deviation of the ML residuals.
        """
        from scipy import stats

        point = self.predict(X)
        z = stats.norm.ppf((1 + alpha) / 2)

        # Try to get backbone intervals
        try:
            backbone_ci = self.backbone.predict_interval(X, alpha=alpha)
            # Widen by residual std
            lower = backbone_ci["lower"].values - z * self._residual_std
            upper = backbone_ci["upper"].values + z * self._residual_std
        except NotImplementedError:
            # Fall back to symmetric interval around point forecast
            lower = point - z * self._residual_std
            upper = point + z * self._residual_std

        return pd.DataFrame(
            {"lower": lower, "median": point, "upper": upper},
            index=X.index,
        )

    @property
    def name(self) -> str:
        return f"Hybrid({self.backbone.name}+{self.residual_model.name})"
