"""
models.py
=========
Model definitions, training helpers, and ensemble utilities.

Models
------
NaiveModel        : Random-walk baseline (last observed value → 0 return)
LinearModel       : Ridge regression benchmark
XGBoostModel      : XGBoost regressor with Optuna hyper-parameter tuning
LGBMModel         : LightGBM regressor with Optuna hyper-parameter tuning
EnsembleModel     : Weighted / simple-average ensemble of the above
QuantileEnsemble  : Wrap any model with quantile-regression CIs
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseForecaster:
    """Minimal interface that all forecasters must satisfy."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Naive (random-walk) baseline
# ---------------------------------------------------------------------------


class NaiveModel(BaseForecaster):
    """Predicts zero log-return (copper price stays unchanged)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    @property
    def name(self) -> str:
        return "Naive (RW)"


# ---------------------------------------------------------------------------
# Linear regression benchmark
# ---------------------------------------------------------------------------


class LinearModel(BaseForecaster):
    """Ridge regression with standard scaling."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._pipe: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self._pipe.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("Model not fitted yet.")
        return self._pipe.predict(X)

    @property
    def name(self) -> str:
        return "Linear (Ridge)"


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


class XGBoostModel(BaseForecaster):
    """XGBoost regressor, optionally tuned with Optuna."""

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        self.params = params or self.DEFAULT_PARAMS.copy()
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBoostModel") from exc

        self._model = XGBRegressor(**self.params)
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        return self._model.predict(X)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        cv_splits: int = 5,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Use Optuna to search for optimal hyper-parameters.

        Returns the best parameter dict (also stored in ``self.params``).
        """
        try:
            import optuna
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("optuna and xgboost are required for tuning") from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": 0,
            }
            scores = cross_val_score(
                XGBRegressor(**params),
                X,
                y,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
            )
            return -scores.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best.update({"random_state": random_state, "n_jobs": -1, "verbosity": 0})
        self.params = best
        logger.info("XGBoost best CV RMSE: %.6f | params: %s", study.best_value, best)
        return best

    @property
    def name(self) -> str:
        return "XGBoost"


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


class LGBMModel(BaseForecaster):
    """LightGBM regressor, optionally tuned with Optuna."""

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        self.params = params or self.DEFAULT_PARAMS.copy()
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMModel":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is required for LGBMModel") from exc

        self._model = LGBMRegressor(**self.params)
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        return self._model.predict(X)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        cv_splits: int = 5,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Use Optuna to search for optimal hyper-parameters."""
        try:
            import optuna
            from lightgbm import LGBMRegressor
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        except ImportError as exc:
            raise ImportError("optuna and lightgbm are required for tuning") from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "random_state": random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            scores = cross_val_score(
                LGBMRegressor(**params),
                X,
                y,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
            )
            return -scores.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best.update({"random_state": random_state, "n_jobs": -1, "verbose": -1})
        self.params = best
        logger.info("LGBM best CV RMSE: %.6f | params: %s", study.best_value, best)
        return best

    @property
    def name(self) -> str:
        return "LightGBM"


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class EnsembleModel(BaseForecaster):
    """Weighted average of multiple fitted BaseForecaster instances.

    Weights default to equal weighting.  Pass a list of floats to
    ``weights`` to override.
    """

    def __init__(
        self,
        forecasters: list[BaseForecaster],
        weights: Optional[list[float]] = None,
    ) -> None:
        if not forecasters:
            raise ValueError("forecasters must be a non-empty list")
        self.forecasters = forecasters
        if weights is None:
            weights = [1.0 / len(forecasters)] * len(forecasters)
        if len(weights) != len(forecasters):
            raise ValueError("weights must have the same length as forecasters")
        self.weights = np.array(weights, dtype=float)
        self.weights /= self.weights.sum()   # normalise

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        for f in self.forecasters:
            f.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([f.predict(X) for f in self.forecasters])
        return preds @ self.weights

    @property
    def name(self) -> str:
        names = "+".join(f.name for f in self.forecasters)
        return f"Ensemble({names})"


# ---------------------------------------------------------------------------
# Quantile wrapper for prediction intervals
# ---------------------------------------------------------------------------


class QuantileForecaster:
    """Fit three quantile regression models (lower, median, upper) for a CI.

    Uses LightGBM's built-in quantile objective for efficiency.

    Parameters
    ----------
    alpha:
        CI width, e.g. 0.80 → 10th and 90th percentile bounds.
    base_params:
        Base LightGBM params (without 'objective' / 'alpha').
    """

    def __init__(
        self,
        alpha: float = 0.80,
        base_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self.alpha = alpha
        self._lower_q = (1 - alpha) / 2
        self._upper_q = 1 - self._lower_q
        self._base_params = base_params or LGBMModel.DEFAULT_PARAMS.copy()
        self._models: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileForecaster":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is required for QuantileForecaster") from exc

        for q, label in [
            (self._lower_q, "lower"),
            (0.5, "median"),
            (self._upper_q, "upper"),
        ]:
            params = {**self._base_params, "objective": "quantile", "alpha": q}
            params.pop("verbose", None)
            params["verbose"] = -1
            m = LGBMRegressor(**params)
            m.fit(X, y)
            self._models[label] = m

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns: lower, median, upper."""
        return pd.DataFrame(
            {
                "lower": self._models["lower"].predict(X),
                "median": self._models["median"].predict(X),
                "upper": self._models["upper"].predict(X),
            },
            index=X.index,
        )
