"""
dimensionality.py
=================
Dimensionality reduction for the short, wide monthly feature matrix
(≈140 columns on ≈430 rows).

`DimReducedForecaster` wraps any point forecaster behind a
``StandardScaler → PCA`` front-end, collapsing the many collinear predictors
into a handful of orthogonal factors before the model sees them.

It is deliberately a *model wrapper* rather than a global transform: the scaler
and PCA are fit inside ``fit`` on exactly the rows passed to that call, so when
``walk_forward_cv`` / ``out_of_sample_backtest`` refit per fold the factors are
re-estimated on each fold's training data only — no look-ahead.  It composes
with the curated/regularised models from item #6, e.g.::

    DimReducedForecaster(ElasticNetModel(), n_components=8)
    DimReducedForecaster(CuratedForecaster(LinearModel()), n_components=0.95)
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class DimReducedForecaster(BaseForecaster):
    """Reduce features to PCA factors, then fit a base forecaster on them.

    Parameters
    ----------
    base:
        Any point forecaster (``.fit`` / ``.predict`` → ndarray).
    n_components:
        ``int`` → keep that many principal components (clamped to the feasible
        ``min(n_samples-1, n_features)``).  ``float`` in (0, 1) → keep enough
        components to explain that fraction of variance.
    whiten:
        Pass-through to :class:`sklearn.decomposition.PCA`.
    """

    def __init__(
        self,
        base: BaseForecaster,
        n_components: Union[int, float] = 8,
        whiten: bool = False,
    ) -> None:
        self.base = base
        self.n_components = n_components
        self.whiten = whiten
        self._pipe: Optional[Pipeline] = None
        self._k: Optional[int] = None

    def _resolve_k(self, X: pd.DataFrame) -> Union[int, float]:
        if isinstance(self.n_components, int):
            feasible = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
            return feasible
        return self.n_components   # float variance threshold — PCA handles it

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DimReducedForecaster":
        k = self._resolve_k(X)
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=k, whiten=self.whiten, random_state=42)),
        ])
        scores = self._pipe.fit_transform(X)
        self._k = self._pipe.named_steps["pca"].n_components_
        factors = pd.DataFrame(
            scores, index=X.index, columns=[f"pc_{i}" for i in range(scores.shape[1])])
        self.base.fit(factors, y)
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._pipe is None:
            raise RuntimeError("Model not fitted yet.")
        scores = self._pipe.transform(X)
        return pd.DataFrame(
            scores, index=X.index, columns=[f"pc_{i}" for i in range(scores.shape[1])])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.base.predict(self._transform(X)), dtype=float)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("Model not fitted yet.")
        return self._pipe.named_steps["pca"].explained_variance_ratio_

    @property
    def name(self) -> str:
        k = self._k if self._k is not None else self.n_components
        return f"PCA{k}+{self.base.name}"
