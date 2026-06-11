"""
models.py
=========
Model definitions, training helpers, and ensemble utilities.

Models
------
NaiveModel           : Random-walk baseline (last observed value → 0 return)
FuturesBasisBenchmark: Cost-of-carry benchmark — curve-implied return from the
                       LME cash-3M basis (stricter than RW when the curve steep)
LinearModel          : Ridge regression benchmark
ElasticNetModel      : Elastic-Net (L1+L2) regression with Optuna tuning —
                       the p≫n workhorse for the short monthly sample
AdaptiveLassoModel   : Adaptive LASSO (Zou 2006, oracle property) — consistent,
                       sparse, signed feature selection for the monthly model
CuratedForecaster    : Wrap any base model to train on a curated column subset

Functions
---------
selection_stability_report : Cross-fold selection frequency / sign stability of
                       a sparse linear model (AdaptiveLasso / ElasticNet)
XGBoostModel         : XGBoost regressor with Optuna hyper-parameter tuning
LGBMModel            : LightGBM regressor with Optuna hyper-parameter tuning
XGBoostClassifier    : XGBoost direction classifier (predict ±1)
LGBMClassifier       : LightGBM direction classifier (predict ±1)
EnsembleModel        : Weighted / simple-average ensemble of the above
QuantileForecaster   : Wrap any model with quantile-regression CIs
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
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

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.80) -> pd.DataFrame:
        """Return DataFrame with columns: lower, median, upper.

        Override in subclasses that support prediction intervals.
        """
        raise NotImplementedError(
            f"{self.name} does not support prediction intervals"
        )

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
# Futures-basis (cost-of-carry) benchmark
# ---------------------------------------------------------------------------


class FuturesBasisBenchmark(BaseForecaster):
    """Forecast the h-step return from the copper futures curve (the basis).

    Under cost-of-carry / "the futures price is the market's expected future
    spot", the cash-to-3-month basis *is* a forecast: a contango of +1% (3-month
    above cash) implies the spot is expected to rise ~1% over three months.
    This model linearly interpolates that curve-implied return to the forecast
    horizon,

        ``return_h = basis_pct * horizon / tenor_periods``

    making it a fully transparent, traded-price benchmark that is *stricter*
    than the random walk whenever the curve is steep (the literature's second
    benchmark after the RW; see Reeve & Vigfusson 2011, Chinn & Coibion 2014).

    It reads the basis from one feature column (default ``copper_basis_pct``
    from :func:`src.data_ingestion.fetch_lme_cash_3m`).  If that column is absent
    it degrades gracefully to the random walk (zeros), so it stays drop-in
    compatible with the shared-``X`` line-ups in ``compare_models``.

    Parameters
    ----------
    horizon:
        Forecast horizon in the data's own period unit (e.g. 22 trading days for
        the daily model, 1 month for the monthly model).
    tenor_periods:
        Periods spanning the basis tenor (≈ 63 trading days ~ 3 months for daily
        data; 3 for monthly data).
    basis_col:
        Name of the fractional curve-implied-return column.
    """

    def __init__(
        self,
        horizon: int = 22,
        tenor_periods: int = 63,
        basis_col: str = "copper_basis_pct",
    ) -> None:
        self.horizon = horizon
        self.tenor_periods = max(int(tenor_periods), 1)
        self.basis_col = basis_col
        self._warned = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FuturesBasisBenchmark":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.basis_col not in X.columns:
            if not self._warned:
                logger.warning(
                    "FuturesBasisBenchmark: column '%s' not found — falling back "
                    "to random walk (zeros).", self.basis_col)
                self._warned = True
            return np.zeros(len(X))
        basis = X[self.basis_col].to_numpy(dtype=float)
        preds = basis * (self.horizon / self.tenor_periods)
        return np.nan_to_num(preds, nan=0.0)

    @property
    def name(self) -> str:
        return "Futures-Basis (carry)"


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
# Elastic-Net (regularised linear; embedded feature selection)
# ---------------------------------------------------------------------------


class ElasticNetModel(BaseForecaster):
    """Elastic-Net regression (L1 + L2) with standard scaling and Optuna tuning.

    The honest workhorse when ``p >> n`` (e.g. ~140 features on ~430 monthly
    rows): the L1 term zeroes redundant predictors (embedded selection) while
    the L2 term keeps correlated groups stable.  Unlike :class:`LinearModel`
    (Ridge, ``alpha`` fixed at 1.0), both the overall strength ``alpha`` and the
    L1/L2 mix ``l1_ratio`` are tuned by walk-forward CV.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        max_iter: int = 10000,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self._pipe: Optional[Pipeline] = None

    def _make_pipe(self, alpha: float, l1_ratio: float) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                max_iter=self.max_iter, random_state=42)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetModel":
        self._pipe = self._make_pipe(self.alpha, self.l1_ratio)
        self._pipe.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("Model not fitted yet.")
        return self._pipe.predict(X)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        cv_splits: int = 5,
        random_state: int = 42,
        horizon: int = 1,
    ) -> dict[str, Any]:
        """Optuna search over ``alpha`` and ``l1_ratio`` (neg-RMSE, purged CV).

        Uses :class:`PurgedTimeSeriesSplit` so each training fold drops its last
        ``horizon - 1`` rows (label-overlap purge).  Sets ``self.alpha`` /
        ``self.l1_ratio`` to the best found and returns them.
        """
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
        except ImportError as exc:
            raise ImportError("optuna is required for tuning") from exc

        from src.evaluation import PurgedTimeSeriesSplit

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tscv = PurgedTimeSeriesSplit(n_splits=cv_splits, horizon=horizon)

        def objective(trial: "optuna.Trial") -> float:
            alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.95)
            scores = cross_val_score(
                self._make_pipe(alpha, l1_ratio), X, y,
                cv=tscv, scoring="neg_root_mean_squared_error",
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.alpha = study.best_params["alpha"]
        self.l1_ratio = study.best_params["l1_ratio"]
        logger.info("ElasticNet best CV RMSE: %.6f | alpha=%.4g l1_ratio=%.3f",
                    study.best_value, self.alpha, self.l1_ratio)
        return {"alpha": self.alpha, "l1_ratio": self.l1_ratio}

    @property
    def coef_(self) -> np.ndarray:
        """Fitted Elastic-Net coefficients (post-scaling). Raises if unfitted."""
        if self._pipe is None:
            raise RuntimeError("Model not fitted yet.")
        return self._pipe.named_steps["enet"].coef_

    @property
    def name(self) -> str:
        return "ElasticNet"


# ---------------------------------------------------------------------------
# Adaptive LASSO (oracle-property sparse linear; consistent selection)
# ---------------------------------------------------------------------------


class AdaptiveLassoModel(BaseForecaster):
    """Adaptive LASSO (Zou 2006) — sparse linear model with the oracle property.

    Plain LASSO over-shrinks large true coefficients and its selection is not
    consistent when predictors are correlated (the LME/COMEX/DXY/CNY block).
    The adaptive LASSO fixes this by penalising each coefficient by a data-driven
    weight ``w_j = 1/|beta_init_j|^gamma`` from a first-stage root-n-consistent
    fit (here ridge, robust when ``p >> n``): predictors the first stage finds
    irrelevant are penalised hard (shrunk to exactly zero) while strong ones are
    barely penalised — giving consistent variable selection and asymptotically
    unbiased estimates, i.e. a transparent, signed, sparse equation to read off.

    Implemented by the standard reparametrisation ``z_j = x_j * |beta_init_j|^gamma``
    (a plain LASSO on the rescaled design), then mapping coefficients back.  The
    honest companion to :class:`ElasticNetModel`: prefer the elastic net when
    correlated predictors should stay grouped, the adaptive LASSO when you want
    the sparsest defensible feature set and consistent selection.

    Parameters
    ----------
    alpha:
        LASSO penalty strength on the rescaled design.
    gamma:
        Adaptive-weight exponent (``> 0``; 1.0 is the common default).
    ridge_alpha:
        L2 strength of the first-stage ridge that forms the weights.
    max_iter:
        LASSO solver iterations.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        gamma: float = 1.0,
        ridge_alpha: float = 1.0,
        max_iter: int = 10000,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self._scaler: Optional[StandardScaler] = None
        self._lasso: Optional[Lasso] = None
        self._a: Optional[np.ndarray] = None       # |beta_init|^gamma per column
        self._cols: Optional[list[str]] = None

    def _fit_arrays(self, X: pd.DataFrame, y: pd.Series, alpha: float, gamma: float):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        # First-stage ridge -> initial coefficients (root-n consistent, p>>n safe).
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(Xs, y)
        a = np.abs(ridge.coef_) ** gamma            # = 1 / w_j  (adaptive rescale)
        Z = Xs * a                                   # rescaled design
        lasso = Lasso(alpha=alpha, max_iter=self.max_iter, random_state=42)
        lasso.fit(Z, y)
        return scaler, lasso, a

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AdaptiveLassoModel":
        self._cols = list(X.columns)
        self._scaler, self._lasso, self._a = self._fit_arrays(
            X, y, self.alpha, self.gamma)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._lasso is None or self._scaler is None:
            raise RuntimeError("Model not fitted yet.")
        Z = self._scaler.transform(X) * self._a
        return self._lasso.predict(Z)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        cv_splits: int = 5,
        random_state: int = 42,
        horizon: int = 1,
    ) -> dict[str, Any]:
        """Optuna search over ``alpha`` and ``gamma`` (neg-RMSE, purged CV).

        Uses :class:`PurgedTimeSeriesSplit` so each training fold drops its last
        ``horizon - 1`` rows (label-overlap purge).  Sets ``self.alpha`` /
        ``self.gamma`` to the best found and returns them.
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("optuna is required for tuning") from exc
        from sklearn.metrics import mean_squared_error

        from src.evaluation import PurgedTimeSeriesSplit

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tscv = PurgedTimeSeriesSplit(n_splits=cv_splits, horizon=horizon)
        Xv = X.reset_index(drop=True)
        yv = pd.Series(np.asarray(y, dtype=float))

        def objective(trial: "optuna.Trial") -> float:
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            gamma = trial.suggest_float("gamma", 0.5, 2.0)
            errs = []
            for tr, te in tscv.split(Xv):
                scaler, lasso, a = self._fit_arrays(
                    Xv.iloc[tr], yv.iloc[tr], alpha, gamma)
                pred = lasso.predict(scaler.transform(Xv.iloc[te]) * a)
                errs.append(mean_squared_error(yv.iloc[te], pred))
            return float(np.sqrt(np.mean(errs)))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.alpha = study.best_params["alpha"]
        self.gamma = study.best_params["gamma"]
        logger.info("AdaptiveLasso best CV RMSE: %.6f | alpha=%.4g gamma=%.3f",
                    study.best_value, self.alpha, self.gamma)
        return {"alpha": self.alpha, "gamma": self.gamma}

    @property
    def coef_(self) -> np.ndarray:
        """Effective (standardized) coefficients ``lasso.coef_ * |beta_init|^gamma``."""
        if self._lasso is None:
            raise RuntimeError("Model not fitted yet.")
        return self._lasso.coef_ * self._a

    def selected_features(self, threshold: float = 1e-8) -> list[str]:
        """Names of features with a non-zero adaptive coefficient."""
        if self._cols is None:
            raise RuntimeError("Model not fitted yet.")
        return [c for c, b in zip(self._cols, self.coef_) if abs(b) > threshold]

    @property
    def name(self) -> str:
        return "Adaptive LASSO"


def selection_stability_report(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    horizon: int = 22,
) -> pd.DataFrame:
    """Cross-fold selection frequency & sign stability of a sparse linear model.

    Refits ``model_factory()`` (any model exposing ``coef_`` aligned to
    ``X.columns``, e.g. :class:`AdaptiveLassoModel` / :class:`ElasticNetModel`)
    on each purged walk-forward training fold and records which features survive.
    A feature you can trust is one selected in most folds with a stable sign;
    coefficients that flip sign across windows are the interpretability trap the
    literature flags for penalised regression on serially-dependent data.

    Parameters
    ----------
    model_factory:
        Zero-arg callable returning a fresh, unfitted model with a ``coef_``
        attribute aligned to ``X.columns`` after ``fit``.
    X, y:
        Feature matrix and target.
    n_splits, horizon:
        Purged-CV fold count and label horizon (for label-overlap purging).

    Returns
    -------
    DataFrame indexed by feature with columns ``selection_freq`` (fraction of
    folds non-zero), ``mean_coef``, ``sign_consistency`` (max share of one sign
    among the folds where selected), ``n_folds``; sorted by ``selection_freq``.
    """
    from src.evaluation import PurgedTimeSeriesSplit

    cols = list(X.columns)
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, horizon=horizon)
    coefs = []
    for tr, _ in splitter.split(X):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        coefs.append(np.asarray(m.coef_, dtype=float))
    C = np.vstack(coefs)                            # (n_folds, p)
    nz = np.abs(C) > 1e-8
    with np.errstate(invalid="ignore"):
        pos = ((C > 0) & nz).sum(axis=0)
        neg = ((C < 0) & nz).sum(axis=0)
        tot = np.maximum(pos + neg, 1)
        sign_consistency = np.maximum(pos, neg) / tot
    out = pd.DataFrame({
        "selection_freq": nz.mean(axis=0),
        "mean_coef": C.mean(axis=0),
        "sign_consistency": sign_consistency,
        "n_folds": int(C.shape[0]),
    }, index=cols)
    return out.sort_values("selection_freq", ascending=False)


# ---------------------------------------------------------------------------
# Curated-feature wrapper (column subset → any base model)
# ---------------------------------------------------------------------------


class CuratedForecaster(BaseForecaster):
    """Train/predict a base model on an economically-curated column subset.

    A drop-in ``BaseForecaster`` so it works inside the shared-``X`` line-ups of
    ``compare_models`` / ``out_of_sample_backtest`` while only seeing the curated
    core (see :func:`src.feature_engineering.curate_features`).  The subset is
    chosen from each ``fit`` call's columns, so walk-forward refits stay
    leakage-safe.  Pairs naturally with :class:`ElasticNetModel` for the
    ``p >> n`` monthly regime: ``CuratedForecaster(ElasticNetModel())``.
    """

    def __init__(
        self,
        base: BaseForecaster,
        prefixes: Optional[tuple[str, ...]] = None,
        keep_regime: bool = True,
    ) -> None:
        self.base = base
        self.prefixes = prefixes
        self.keep_regime = keep_regime
        self._cols: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CuratedForecaster":
        from src.feature_engineering import curate_features
        self._cols = list(
            curate_features(X, self.prefixes, self.keep_regime).columns)
        self.base.fit(X[self._cols], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._cols is None:
            raise RuntimeError("Model not fitted yet.")
        return self.base.predict(X[self._cols])

    def tune(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
        """Curate first, then delegate tuning to the base model (if it tunes)."""
        from src.feature_engineering import curate_features
        self._cols = list(
            curate_features(X, self.prefixes, self.keep_regime).columns)
        if hasattr(self.base, "tune"):
            return self.base.tune(X[self._cols], y, **kwargs)
        return None

    @property
    def name(self) -> str:
        return f"Curated({self.base.name})"


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


class XGBoostModel(BaseForecaster):
    """XGBoost regressor, optionally tuned with Optuna."""

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
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
        horizon: int = 1,
    ) -> dict[str, Any]:
        """Use Optuna to search for optimal hyper-parameters.

        Uses :class:`PurgedTimeSeriesSplit` so the last ``horizon - 1`` rows
        of each training fold are dropped, preventing label-overlap leakage
        (López de Prado, AFML Ch. 7).

        Returns the best parameter dict (also stored in ``self.params``).
        """
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("optuna and xgboost are required for tuning") from exc

        from src.evaluation import PurgedTimeSeriesSplit

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        tscv = PurgedTimeSeriesSplit(n_splits=cv_splits, horizon=horizon)

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
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "min_child_samples": 30,
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
        horizon: int = 1,
    ) -> dict[str, Any]:
        """Use Optuna to search for optimal hyper-parameters.

        Uses :class:`PurgedTimeSeriesSplit` so the last ``horizon - 1`` rows
        of each training fold are dropped, preventing label-overlap leakage
        (López de Prado, AFML Ch. 7).
        """
        try:
            import optuna
            from lightgbm import LGBMRegressor
            from sklearn.model_selection import cross_val_score
        except ImportError as exc:
            raise ImportError("optuna and lightgbm are required for tuning") from exc

        from src.evaluation import PurgedTimeSeriesSplit

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tscv = PurgedTimeSeriesSplit(n_splits=cv_splits, horizon=horizon)

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
# Trimmed-mean forecast combination (robust 1/N)
# ---------------------------------------------------------------------------


class TrimmedMeanForecaster(BaseForecaster):
    """Trimmed-mean forecast combination — a robust variant of the 1/N average.

    Averages the base forecasts after dropping the most extreme ``trim`` fraction
    at each tail per observation.  Combination theory (Bates & Granger 1969) and
    the "forecast-combination puzzle" (Stock & Watson 2004; Genre et al. 2013)
    show that the equal-weight average — :class:`EnsembleModel` with default
    weights — is very hard to beat with *estimated* optimal weights once
    estimation error is paid for; the trimmed mean just adds robustness to a
    single rogue member.  Benchmark your stacking ensemble against this and the
    plain 1/N with :func:`src.evaluation.white_reality_check`.

    Parameters
    ----------
    forecasters:
        Base forecasters to combine.
    trim:
        Fraction trimmed from each tail per observation (0.1 -> drop top/bottom
        10% of the member forecasts before averaging).
    """

    def __init__(self, forecasters: list[BaseForecaster], trim: float = 0.1) -> None:
        if not forecasters:
            raise ValueError("forecasters must be a non-empty list")
        if not 0.0 <= trim < 0.5:
            raise ValueError("trim must be in [0, 0.5)")
        self.forecasters = forecasters
        self.trim = float(trim)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrimmedMeanForecaster":
        for f in self.forecasters:
            f.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.sort(np.column_stack([f.predict(X) for f in self.forecasters]), axis=1)
        k = preds.shape[1]
        cut = int(np.floor(self.trim * k))
        if 2 * cut >= k:
            return preds.mean(axis=1)
        return preds[:, cut:k - cut].mean(axis=1)

    @property
    def name(self) -> str:
        return f"TrimmedMean({len(self.forecasters)})"


# ---------------------------------------------------------------------------
# Direction classifiers (return ±1; evaluated by DA / Signal Sharpe)
# ---------------------------------------------------------------------------


class XGBoostClassifier(BaseForecaster):
    """XGBoost binary direction classifier.

    Predicts +1 (price up) or -1 (price down).  Trained with
    ``binary:logistic`` objective on the sign of the return target.
    Use Signal Sharpe and Directional Accuracy to evaluate — RMSE is
    meaningless for ±1 outputs.

    Parameters
    ----------
    params:
        Override default XGBoost parameters.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "eval_metric": "logloss",
    }

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        self.params = params or self.DEFAULT_PARAMS.copy()
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostClassifier":
        try:
            from xgboost import XGBClassifier as _XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBoostClassifier") from exc

        # Convert continuous returns to ±1 labels; 0-return → +1
        y_dir = (np.sign(y.values).astype(int) >= 0).astype(int)  # 0 or 1
        self._model = _XGBClassifier(**self.params)
        self._model.fit(X, y_dir)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        # Return ±1 to match the return-sign convention used by signal_sharpe
        proba = self._model.predict_proba(X)[:, 1]
        return np.where(proba >= 0.5, 1.0, -1.0)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        cv_splits: int = 5,
        random_state: int = 42,
        horizon: int = 1,
    ) -> dict[str, Any]:
        """Optuna search for best hyper-parameters (optimises log-loss).

        Uses :class:`PurgedTimeSeriesSplit` so the last ``horizon - 1`` rows
        of each training fold are dropped, preventing label-overlap leakage
        (López de Prado, AFML Ch. 7).
        """
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            from xgboost import XGBClassifier as _XGBClassifier
        except ImportError as exc:
            raise ImportError("optuna and xgboost are required for tuning") from exc

        from src.evaluation import PurgedTimeSeriesSplit

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        y_dir = (np.sign(y.values).astype(int) >= 0).astype(int)
        tscv = PurgedTimeSeriesSplit(n_splits=cv_splits, horizon=horizon)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": 0,
                "eval_metric": "logloss",
            }
            scores = cross_val_score(
                _XGBClassifier(**params), X, y_dir, cv=tscv, scoring="neg_log_loss"
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best.update({"random_state": random_state, "n_jobs": -1, "verbosity": 0,
                     "eval_metric": "logloss"})
        self.params = best
        return best

    @property
    def name(self) -> str:
        return "XGBoost (Classifier)"


class LGBMClassifier(BaseForecaster):
    """LightGBM binary direction classifier.

    Predicts +1 (price up) or -1 (price down).  Trained with
    ``binary`` objective on the sign of the return target.
    Use Signal Sharpe and Directional Accuracy to evaluate.

    Parameters
    ----------
    params:
        Override default LightGBM parameters.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "min_child_samples": 30,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        self.params = params or self.DEFAULT_PARAMS.copy()
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMClassifier":
        try:
            from lightgbm import LGBMClassifier as _LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is required for LGBMClassifier") from exc

        y_dir = (np.sign(y.values).astype(int) >= 0).astype(int)  # 0 or 1
        params = {**self.params, "objective": "binary"}
        self._model = _LGBMClassifier(**params)
        self._model.fit(X, y_dir)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        proba = self._model.predict_proba(X)[:, 1]
        return np.where(proba >= 0.5, 1.0, -1.0)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        cv_splits: int = 5,
        random_state: int = 42,
        horizon: int = 1,
    ) -> dict[str, Any]:
        """Optuna search for best hyper-parameters (optimises log-loss).

        Uses :class:`PurgedTimeSeriesSplit` so the last ``horizon - 1`` rows
        of each training fold are dropped, preventing label-overlap leakage
        (López de Prado, AFML Ch. 7).
        """
        try:
            import optuna
            from lightgbm import LGBMClassifier as _LGBMClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError as exc:
            raise ImportError("optuna and lightgbm are required for tuning") from exc

        from src.evaluation import PurgedTimeSeriesSplit

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        y_dir = (np.sign(y.values).astype(int) >= 0).astype(int)
        tscv = PurgedTimeSeriesSplit(n_splits=cv_splits, horizon=horizon)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 10, 60),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                "random_state": random_state,
                "n_jobs": -1,
                "verbose": -1,
                "objective": "binary",
            }
            scores = cross_val_score(
                _LGBMClassifier(**params), X, y_dir, cv=tscv, scoring="neg_log_loss"
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best.update({"random_state": random_state, "n_jobs": -1, "verbose": -1,
                     "objective": "binary"})
        self.params = best
        return best

    @property
    def name(self) -> str:
        return "LGBM (Classifier)"


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
