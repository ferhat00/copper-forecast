"""
models_stacking.py
==================
Stacking meta-learner ensemble conforming to the BaseForecaster interface.

Uses walk-forward cross-validation to generate out-of-fold (OOF) predictions
from each base model, then trains a meta-learner (Ridge by default) on the
stacked OOF predictions.  This avoids information leakage that would occur
if in-sample predictions were used for meta-learner training.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.evaluation import walk_forward_cv
from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseForecaster):
    """Stacking ensemble with walk-forward OOF predictions and a meta-learner.

    Parameters
    ----------
    base_models:
        List of unfitted BaseForecaster instances.
    meta_learner:
        Sklearn-compatible regressor for the second stage.
        Default: ``Ridge(alpha=1.0)``.
    oof_initial_size:
        Initial training window for generating OOF predictions.
    oof_step:
        Step size for walk-forward OOF generation.
    horizon:
        Label horizon in rows.  Forwarded to :func:`walk_forward_cv` so that
        the last ``horizon - 1`` rows of each training fold are purged when
        generating OOF predictions (López de Prado, AFML Ch. 7).  Default
        1 disables purging.
    """

    def __init__(
        self,
        base_models: list[BaseForecaster],
        meta_learner: Optional[Any] = None,
        oof_initial_size: int = 504,
        oof_step: int = 22,
        horizon: int = 1,
    ) -> None:
        if not base_models:
            raise ValueError("base_models must be a non-empty list")
        self.base_models = base_models
        self.meta_learner = meta_learner or Ridge(alpha=1.0)
        self.oof_initial_size = oof_initial_size
        self.oof_step = oof_step
        self.horizon = horizon
        self._fitted_models: list[BaseForecaster] = []
        self._meta_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingEnsemble":
        """Fit the stacking ensemble.

        Steps:
          1. Generate OOF predictions for each base model via walk-forward CV.
          2. Align OOF predictions by date (keep only common indices).
          3. Train the meta-learner on stacked OOF predictions.
          4. Refit all base models on the full training set.
        """
        oof_frames: dict[str, pd.Series] = {}

        # Step 1: generate OOF predictions for each base model
        for model in self.base_models:
            model_copy = copy.deepcopy(model)
            try:
                cv_df = walk_forward_cv(
                    model_copy, X, y,
                    initial_train_size=self.oof_initial_size,
                    step_size=self.oof_step,
                    horizon=self.horizon,
                )
                oof_frames[model.name] = cv_df["y_pred"]
                logger.info(
                    "Stacking OOF: %s produced %d predictions",
                    model.name, len(cv_df),
                )
            except Exception as exc:
                logger.warning("Stacking OOF: %s failed — %s", model.name, exc)

        if len(oof_frames) < 2:
            raise ValueError(
                f"Need at least 2 base models with OOF predictions, got {len(oof_frames)}"
            )

        # Step 2: align by common index
        oof_df = pd.DataFrame(oof_frames)
        oof_df = oof_df.dropna()

        y_oof = y.reindex(oof_df.index).dropna()
        common = oof_df.index.intersection(y_oof.index)
        oof_df = oof_df.loc[common]
        y_oof = y_oof.loc[common]

        if len(oof_df) < 20:
            raise ValueError(
                f"Too few aligned OOF observations ({len(oof_df)}) for meta-learner"
            )

        # Step 3: train meta-learner
        self.meta_learner.fit(oof_df.values, y_oof.values)
        self._meta_fitted = True
        logger.info(
            "Stacking: meta-learner trained on %d observations with %d base models",
            len(oof_df), len(oof_frames),
        )

        # Step 4: refit all base models on full training data
        self._fitted_models = []
        for model in self.base_models:
            if model.name in oof_frames:
                model.fit(X, y)
                self._fitted_models.append(model)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._meta_fitted:
            raise RuntimeError("StackingEnsemble has not been fitted yet")

        base_preds = np.column_stack([
            m.predict(X) for m in self._fitted_models
        ])
        return self.meta_learner.predict(base_preds)

    @property
    def name(self) -> str:
        names = "+".join(m.name for m in self.base_models)
        return f"Stacking({names})"


class RobustCombiner(BaseForecaster):
    """Robust forecast combination — a low-variance alternative to stacking.

    Stacking trains a meta-learner on walk-forward OOF predictions, which needs
    a sizeable OOF history and overfits on thin samples — on the monthly
    notebook (~430 rows, ``initial_train_size=60``) it fails outright ("Dataset
    too small").  Simple combinations are famously hard to beat out-of-sample
    ("forecast combination puzzle"), so this class combines base-model
    predictions without a second-stage learner:

    * ``"equal"``   — unweighted mean (needs no OOF; always works).
    * ``"median"``  — per-row median (robust to a single rogue model; no OOF).
    * ``"inverse_error"`` — weights ∝ 1 / OOF error.  Estimates each model's
      walk-forward error; if CV can't run (too few rows) it **degrades
      gracefully to equal weights** instead of raising.

    Conforms to the ``BaseForecaster`` interface so it drops into
    ``compare_models`` / ``out_of_sample_backtest`` like any other model.

    Parameters
    ----------
    base_models:
        List of unfitted ``BaseForecaster`` instances.
    method:
        ``"inverse_error"`` (default), ``"equal"`` or ``"median"``.
    oof_initial_size, oof_step, horizon:
        Walk-forward CV parameters used only by ``"inverse_error"`` to estimate
        per-model error (``horizon`` also purges label overlap).
    error_metric:
        ``"rmse"`` (default) or ``"mae"`` for the inverse-error weights.
    min_oof:
        Minimum aligned OOF observations required to trust inverse-error
        weights; below this it falls back to equal weights.
    """

    def __init__(
        self,
        base_models: list[BaseForecaster],
        method: str = "inverse_error",
        oof_initial_size: int = 504,
        oof_step: int = 22,
        horizon: int = 1,
        error_metric: str = "rmse",
        min_oof: int = 20,
    ) -> None:
        if not base_models:
            raise ValueError("base_models must be a non-empty list")
        if method not in {"equal", "median", "inverse_error"}:
            raise ValueError(f"unknown method {method!r}")
        self.base_models = base_models
        self.method = method
        self.oof_initial_size = oof_initial_size
        self.oof_step = oof_step
        self.horizon = horizon
        self.error_metric = error_metric
        self.min_oof = min_oof
        self._fitted_models: list[BaseForecaster] = []
        self.weights_: Optional[np.ndarray] = None

    def _oof_errors(self, X: pd.DataFrame, y: pd.Series) -> Optional[dict[str, float]]:
        """Per-model walk-forward OOF error; None if CV is not feasible."""
        errors: dict[str, float] = {}
        for model in self.base_models:
            try:
                cv_df = walk_forward_cv(
                    copy.deepcopy(model), X, y,
                    initial_train_size=self.oof_initial_size,
                    step_size=self.oof_step, horizon=self.horizon,
                )
                aligned = cv_df.dropna()
                if len(aligned) < self.min_oof:
                    return None
                err = aligned["y_true"].to_numpy() - aligned["y_pred"].to_numpy()
                if self.error_metric == "mae":
                    errors[model.name] = float(np.mean(np.abs(err)))
                else:
                    errors[model.name] = float(np.sqrt(np.mean(err ** 2)))
            except Exception as exc:
                logger.warning("RobustCombiner OOF: %s failed — %s", model.name, exc)
                return None
        return errors

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RobustCombiner":
        # Determine weights (None => median; uniform => equal mean).
        weights = None
        if self.method == "inverse_error":
            errors = self._oof_errors(X, y)
            if errors and all(e > 0 for e in errors.values()):
                inv = np.array([1.0 / errors[m.name] for m in self.base_models])
                weights = inv / inv.sum()
                logger.info("RobustCombiner: inverse-error weights = %s",
                            dict(zip([m.name for m in self.base_models],
                                     np.round(weights, 3))))
            else:
                weights = np.full(len(self.base_models), 1.0 / len(self.base_models))
                logger.warning(
                    "RobustCombiner: OOF infeasible — falling back to equal weights")
        elif self.method == "equal":
            weights = np.full(len(self.base_models), 1.0 / len(self.base_models))

        # Fit all base models on the full training set (skip any that fail).
        self._fitted_models = []
        kept = []
        for model in self.base_models:
            try:
                model.fit(X, y)
                self._fitted_models.append(model)
                kept.append(True)
            except Exception as exc:
                logger.warning("RobustCombiner: %s failed to fit — %s", model.name, exc)
                kept.append(False)
        if not self._fitted_models:
            raise RuntimeError("RobustCombiner: no base model fitted successfully")

        if weights is not None:
            w = weights[np.array(kept)]
            self.weights_ = w / w.sum()
        else:
            self.weights_ = None  # median
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted_models:
            raise RuntimeError("RobustCombiner has not been fitted yet")
        preds = np.column_stack([m.predict(X) for m in self._fitted_models])
        if self.method == "median" or self.weights_ is None:
            return np.median(preds, axis=1)
        return preds @ self.weights_

    @property
    def name(self) -> str:
        names = "+".join(m.name for m in self.base_models)
        return f"Robust[{self.method}]({names})"
