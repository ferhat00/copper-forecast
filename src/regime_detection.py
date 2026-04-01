"""
regime_detection.py
===================
Hidden Markov Model (HMM) based market regime detection for copper.

Identifies latent market states (e.g. bull / bear / sideways) from
observable features like returns, volatility, and price z-score.
Regime labels are added as categorical features for downstream models.

Important
---------
The HMM must be refit at each cross-validation fold to prevent look-ahead
bias.  Use ``fit()`` on training data only, then ``predict()`` on both
training and test data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REGIME_FEATURES = [
    "copper_ret_1d",
    "copper_vol_22d",
    "copper_zscore_200d",
]


class RegimeDetector:
    """Gaussian HMM regime detector.

    Parameters
    ----------
    n_regimes:
        Number of hidden states (default 3: bear / sideways / bull).
    features:
        Column names to use as observable features for the HMM.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        features: Optional[list[str]] = None,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.features = features or DEFAULT_REGIME_FEATURES
        self.random_state = random_state
        self._model = None
        self._label_map: Optional[dict[int, int]] = None

    def fit(self, X: pd.DataFrame) -> "RegimeDetector":
        """Fit the HMM on training data.

        Parameters
        ----------
        X:
            Feature matrix; must contain columns listed in ``self.features``.

        Returns
        -------
        self
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise ImportError("hmmlearn is required for RegimeDetector") from exc

        avail = [f for f in self.features if f in X.columns]
        if not avail:
            raise ValueError(
                f"None of the regime features {self.features} found in X columns"
            )
        self._used_features = avail

        obs = X[avail].dropna().values
        if len(obs) < self.n_regimes * 10:
            raise ValueError(
                f"Too few observations ({len(obs)}) for {self.n_regimes} regimes"
            )

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
            tol=1e-4,
        )
        self._model.fit(obs)

        # Build label map: sort regimes by ascending mean of the first
        # feature (copper_ret_1d) so that regime 0 = bearish, N = bullish
        raw_labels = self._model.predict(obs)
        ret_col_idx = 0  # first feature is typically copper_ret_1d
        means = {}
        for state in range(self.n_regimes):
            mask = raw_labels == state
            if mask.any():
                means[state] = obs[mask, ret_col_idx].mean()
            else:
                means[state] = 0.0

        sorted_states = sorted(means, key=means.get)
        self._label_map = {old: new for new, old in enumerate(sorted_states)}

        logger.info(
            "RegimeDetector: fitted %d regimes on %d observations (%d features)",
            self.n_regimes, len(obs), len(avail),
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict regime labels for the given feature matrix.

        Parameters
        ----------
        X:
            Feature matrix (may include training or test data).

        Returns
        -------
        pd.Series
            Integer regime labels (0 = most bearish, N-1 = most bullish),
            NaN where input features are NaN.
        """
        if self._model is None:
            raise RuntimeError("RegimeDetector has not been fitted yet")

        labels = pd.Series(np.nan, index=X.index, dtype=float)
        mask = X[self._used_features].notna().all(axis=1)
        obs = X.loc[mask, self._used_features].values

        if len(obs) > 0:
            raw = self._model.predict(obs)
            mapped = np.array([self._label_map[r] for r in raw], dtype=float)
            labels.loc[mask] = mapped

        return labels

    def add_regime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add regime label and one-hot dummies to the feature matrix.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        pd.DataFrame
            ``X`` augmented with ``regime`` (int) and one-hot columns
            ``regime_0``, ``regime_1``, etc.
        """
        regime = self.predict(X)
        X_aug = X.copy()
        X_aug["regime"] = regime

        for i in range(self.n_regimes):
            X_aug[f"regime_{i}"] = (regime == i).astype(float)

        return X_aug
