"""
models_regime.py
================
Regime-conditional forecasting router.

``RegimeRouter`` trains a separate model for each HMM regime identified by
``RegimeDetector`` and routes each prediction to the model trained on data
from that regime.  A global fallback model (trained on all data) handles
rows with unknown or NaN regime labels.

``RegimeWeightedEnsemble`` is the soft alternative: instead of hard-routing to
one per-regime model (which starves each regime of data), it blends *several*
base models with weights that depend on each model's per-regime walk-forward
skill — using all the data, and degrading gracefully to global weights when a
regime is thin.

This avoids fitting a single model across structurally different market
environments (bull / sideways / bear copper regimes).
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.evaluation import walk_forward_cv
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


class RegimeWeightedEnsemble(BaseForecaster):
    """Blend base models with per-regime, skill-based (inverse-error) weights.

    Where :class:`RegimeRouter` hard-routes each row to a single model trained
    only on that regime's rows (and so starves thin regimes), this keeps *every*
    base model trained on all data and instead **weights** them by how well each
    did, out-of-fold, *within the current regime*.  A regime with too few
    out-of-fold rows falls back to the global (all-rows) weights, and an unseen
    regime at predict time does the same — so it never routes into a data-starved
    corner.

    Parameters
    ----------
    base_models:
        List of unfitted ``BaseForecaster`` instances (>= 2).
    regime_col:
        Integer-regime-label column in ``X`` (excluded from inner features).
    oof_initial_size, oof_step, horizon:
        Walk-forward parameters used to score each model's per-regime error
        (``horizon`` also purges label overlap).
    min_regime_samples:
        Minimum out-of-fold rows in a regime before trusting regime-specific
        weights; below this the regime uses the global weights.
    error_metric:
        ``"rmse"`` (default) or ``"mae"`` for the inverse-error weighting.
    """

    def __init__(
        self,
        base_models: list[BaseForecaster],
        regime_col: str = "regime",
        oof_initial_size: int = 504,
        oof_step: int = 22,
        horizon: int = 1,
        min_regime_samples: int = 30,
        error_metric: str = "rmse",
    ) -> None:
        if len(base_models) < 2:
            raise ValueError("base_models must contain at least 2 models")
        self.base_models = base_models
        self.regime_col = regime_col
        self.oof_initial_size = oof_initial_size
        self.oof_step = oof_step
        self.horizon = horizon
        self.min_regime_samples = min_regime_samples
        self.error_metric = error_metric
        self._models: list[BaseForecaster] = []
        self._model_names: list[str] = []
        self._feature_cols: list[str] = []
        self._global_weights: Optional[np.ndarray] = None
        self._regime_weights: dict[int, np.ndarray] = {}

    def _weights_from_errors(self, err_means: pd.Series) -> np.ndarray:
        """Inverse-error weights (Series indexed by model name) → full vector."""
        inv = 1.0 / err_means.replace(0, np.nan)
        if inv.notna().any():
            inv = inv.fillna(inv.max())
        else:
            inv = pd.Series(1.0, index=err_means.index)
        inv = inv / inv.sum()
        w = np.zeros(len(self._model_names))
        for i, nm in enumerate(self._model_names):
            if nm in inv.index:
                w[i] = inv[nm]
        if w.sum() == 0:
            w[:] = 1.0 / len(w)
        else:
            w = w / w.sum()
        return w

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeWeightedEnsemble":
        if self.regime_col not in X.columns:
            raise ValueError(
                f"RegimeWeightedEnsemble: column '{self.regime_col}' not in X")
        self._feature_cols = [c for c in X.columns if c != self.regime_col]
        Xf = X[self._feature_cols]
        regimes_all = X[self.regime_col]

        # 1) Out-of-fold error per model (skip any that can't produce OOF).
        oof_err: dict[str, pd.Series] = {}
        for m in self.base_models:
            try:
                cv = walk_forward_cv(
                    copy.deepcopy(m), Xf, y,
                    initial_train_size=self.oof_initial_size,
                    step_size=self.oof_step, horizon=self.horizon)
                a = cv.dropna()
                resid = a["y_true"].to_numpy() - a["y_pred"].to_numpy()
                e = np.abs(resid) if self.error_metric == "mae" else resid ** 2
                oof_err[m.name] = pd.Series(e, index=a.index)
            except Exception as exc:
                logger.warning("RegimeWeighted: OOF failed for %s — %s", m.name, exc)

        # 2) Refit all base models on the full data.
        self._models = []
        for m in self.base_models:
            m.fit(Xf, y)
            self._models.append(m)
        self._model_names = [m.name for m in self._models]
        n_models = len(self._models)

        # 3) Too little OOF info → equal global weights, no regime conditioning.
        valid = [nm for nm in self._model_names if nm in oof_err]
        if len(valid) < 2:
            logger.warning("RegimeWeighted: <2 models with OOF — equal weights")
            self._global_weights = np.full(n_models, 1.0 / n_models)
            self._regime_weights = {}
            return self

        err_df = pd.DataFrame({nm: oof_err[nm] for nm in valid}).dropna()
        reg = regimes_all.reindex(err_df.index)

        self._global_weights = self._weights_from_errors(err_df.mean())
        self._regime_weights = {}
        for r, grp in err_df.groupby(reg):
            if len(grp) >= self.min_regime_samples:
                self._regime_weights[int(r)] = self._weights_from_errors(grp.mean())
        logger.info("RegimeWeighted: regime-specific weights for %s (else global)",
                    sorted(self._regime_weights))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._models:
            raise RuntimeError("RegimeWeightedEnsemble has not been fitted yet.")
        Xf = X[self._feature_cols]
        P = np.column_stack([m.predict(Xf) for m in self._models])   # n × m
        regimes = (X[self.regime_col] if self.regime_col in X.columns
                   else pd.Series(np.nan, index=X.index))
        W = np.tile(self._global_weights, (len(X), 1))
        for i, r in enumerate(regimes.to_numpy()):
            if pd.notna(r) and int(r) in self._regime_weights:
                W[i] = self._regime_weights[int(r)]
        return (P * W).sum(axis=1)

    @property
    def name(self) -> str:
        names = "+".join(m.name for m in self.base_models)
        return f"RegimeWeighted({names})"
