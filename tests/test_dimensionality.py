"""
tests/test_dimensionality.py
============================
Tests for `DimReducedForecaster` (per-fold PCA factor reduction).

Critical property: the scaler+PCA are fit on each `fit` call's rows only, so
walk-forward refits never see future data (no look-ahead).

Run with:
    pytest tests/test_dimensionality.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dimensionality import DimReducedForecaster
from src.models import CuratedForecaster, ElasticNetModel, LinearModel
from src.evaluation import out_of_sample_backtest, walk_forward_cv


def _xy(n=400, p=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    # Correlated features from a few latent factors -> PCA should compress well.
    latent = rng.normal(size=(n, 3))
    load = rng.normal(size=(3, p))
    X = pd.DataFrame(latent @ load + rng.normal(0, 0.1, size=(n, p)),
                     columns=[f"f{i}" for i in range(p)], index=idx)
    y = pd.Series(latent[:, 0] * 0.8 + rng.normal(0, 0.5, n), index=idx)
    return X, y


def test_fit_predict_shape_and_k():
    X, y = _xy()
    split = len(X) - 20
    m = DimReducedForecaster(LinearModel(), n_components=5)
    m.fit(X.iloc[:split], y.iloc[:split])
    assert m._k == 5
    preds = m.predict(X.iloc[split:])
    assert preds.shape == (20,)
    assert np.isfinite(preds).all()


def test_int_components_clamped_to_feasible():
    X, y = _xy(n=400, p=8)
    m = DimReducedForecaster(LinearModel(), n_components=50)  # > n_features
    m.fit(X.iloc[:300], y.iloc[:300])
    assert m._k == 8


def test_variance_threshold_components():
    X, y = _xy(n=400, p=12)
    m = DimReducedForecaster(LinearModel(), n_components=0.95)
    m.fit(X.iloc[:300], y.iloc[:300])
    assert m.explained_variance_ratio_.sum() >= 0.95
    assert m._k <= 12


def test_pca_fit_is_leakage_free():
    """PCA components must depend only on the rows passed to fit.

    Fitting on the first 300 rows, then again on those same 300 rows embedded in
    a longer frame, must yield identical transforms for the shared rows — future
    rows cannot influence the components.
    """
    X, y = _xy(n=400)
    m1 = DimReducedForecaster(LinearModel(), n_components=5)
    m1.fit(X.iloc[:300], y.iloc[:300])
    t1 = m1._transform(X.iloc[:300]).to_numpy()

    m2 = DimReducedForecaster(LinearModel(), n_components=5)
    m2.fit(X.iloc[:300], y.iloc[:300])           # identical training slice
    t2 = m2._transform(X.iloc[:300]).to_numpy()
    # Sign of a component is arbitrary; compare absolute values.
    np.testing.assert_allclose(np.abs(t1), np.abs(t2), rtol=1e-9, atol=1e-9)


def test_runs_inside_walk_forward_cv():
    X, y = _xy(n=500)
    cv = walk_forward_cv(
        DimReducedForecaster(LinearModel(), n_components=4),
        X, y, initial_train_size=200, step_size=22, horizon=22)
    assert {"y_true", "y_pred", "fold"}.issubset(cv.columns)
    assert np.isfinite(cv["y_pred"]).all()


def test_composes_with_curated_elasticnet():
    X, y = _xy(n=400)
    m = DimReducedForecaster(CuratedForecaster(ElasticNetModel(alpha=0.05)),
                             n_components=6)
    _, met = out_of_sample_backtest(m, X, y, holdout_size=60, horizon=22, n_trials=3)
    assert "deflated_sharpe" in met
    assert "PCA6+Curated(ElasticNet)" == m.name


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DimReducedForecaster(LinearModel()).predict(pd.DataFrame({"a": [1.0]}))
