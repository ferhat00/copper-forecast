"""
models_hybrid.py
================
``ResidualBoostForecaster`` — an interpretable linear backbone plus a
gradient-boosted residual correction.

This mirrors the LightGBM-ARIMA ensembles that beat plain ARIMA/ETS for copper
(Oikonomou & Damigos 2025, *Mineral Economics*): an auditable linear model
captures the level/trend (the part you can explain to a risk committee), and a
boosting model learns *only the residual* — what the linear part missed.  The
forecast is ``base.predict(X) + residual_model.predict(X)``.

Degrades gracefully to the base model alone when the boosting library is
unavailable, so it stays drop-in for ``compare_models``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class ResidualBoostForecaster(BaseForecaster):
    """Linear backbone + gradient-boosted residual.

    Parameters
    ----------
    base_model:
        Interpretable point forecaster (default :class:`src.models.LinearModel`,
        a ridge).  Pass an ARIMAX/AdaptiveLasso instance for an ARIMA-style shell.
    residual_model:
        Model fit on the base model's residuals (default
        :class:`src.models.LGBMModel`).  If None and LightGBM is unavailable the
        forecaster falls back to the base model alone.
    """

    def __init__(self, base_model=None, residual_model=None) -> None:
        self.base_model = base_model
        self.residual_model = residual_model
        self._base = None
        self._resid = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ResidualBoostForecaster":
        from src.models import LinearModel

        self._base = self.base_model if self.base_model is not None else LinearModel()
        self._base.fit(X, y)
        resid = np.asarray(y, dtype=float) - np.asarray(self._base.predict(X), dtype=float)

        rm = self.residual_model
        if rm is None:
            try:
                from src.models import LGBMModel
                rm = LGBMModel()
            except Exception as exc:  # pragma: no cover - optional dep
                logger.warning("ResidualBoost: LightGBM unavailable (%s) — base only", exc)
                rm = None

        if rm is not None:
            try:
                idx = y.index if isinstance(y, pd.Series) else None
                rm.fit(X, pd.Series(resid, index=idx))
                self._resid = rm
            except Exception as exc:
                logger.warning("ResidualBoost: residual model failed (%s) — base only", exc)
                self._resid = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._base is None:
            raise RuntimeError("ResidualBoostForecaster not fitted yet.")
        pred = np.asarray(self._base.predict(X), dtype=float)
        if self._resid is not None:
            pred = pred + np.asarray(self._resid.predict(X), dtype=float)
        return pred

    @property
    def name(self) -> str:
        return "Residual-Boost (linear+GBM)"
