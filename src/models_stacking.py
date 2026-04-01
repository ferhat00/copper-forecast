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
    """

    def __init__(
        self,
        base_models: list[BaseForecaster],
        meta_learner: Optional[Any] = None,
        oof_initial_size: int = 504,
        oof_step: int = 22,
    ) -> None:
        if not base_models:
            raise ValueError("base_models must be a non-empty list")
        self.base_models = base_models
        self.meta_learner = meta_learner or Ridge(alpha=1.0)
        self.oof_initial_size = oof_initial_size
        self.oof_step = oof_step
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
