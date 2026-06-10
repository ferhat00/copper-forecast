"""
tests/test_regime_weighting.py
==============================
Tests for `RegimeWeightedEnsemble` — soft per-regime, skill-based model blending
(the data-efficient alternative to the hard `RegimeRouter`).

Run with:
    pytest tests/test_regime_weighting.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import LinearModel, NaiveModel
from src.models_regime import RegimeWeightedEnsemble


def _regime_xy(n=900, seed=0):
    """Regime 1 carries a linear signal; regime 0 is near-noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.normal(size=(n, 4)),
                     columns=["a", "b", "c", "d"], index=idx)
    regime = (rng.random(n) < 0.5).astype(int)
    signal = 1.5 * X["a"] - 1.0 * X["b"]
    y = np.where(regime == 1, signal + rng.normal(0, 0.3, n),
                 rng.normal(0, 0.3, n))
    X = X.assign(regime=regime.astype(float))
    return X, pd.Series(y, index=idx)


def test_fit_predict_shape():
    X, y = _regime_xy()
    split = len(X) - 40
    m = RegimeWeightedEnsemble([NaiveModel(), LinearModel()],
                               oof_initial_size=250, oof_step=22, horizon=1)
    m.fit(X.iloc[:split], y.iloc[:split])
    preds = m.predict(X.iloc[split:])
    assert preds.shape == (40,)
    assert np.isfinite(preds).all()
    # regime column must be excluded from inner features
    assert "regime" not in m._feature_cols


def test_weights_are_valid_simplex():
    X, y = _regime_xy()
    m = RegimeWeightedEnsemble([NaiveModel(), LinearModel()],
                               oof_initial_size=250, oof_step=22)
    m.fit(X, y)
    assert np.isclose(m._global_weights.sum(), 1.0)
    for w in m._regime_weights.values():
        assert np.isclose(w.sum(), 1.0)
        assert len(w) == 2
        assert (w >= 0).all()


def test_linear_weighted_higher_in_signal_regime():
    X, y = _regime_xy(n=1000, seed=1)
    m = RegimeWeightedEnsemble([NaiveModel(), LinearModel()],
                               oof_initial_size=250, oof_step=22,
                               min_regime_samples=30)
    m.fit(X, y)
    # both regimes should have dedicated weights given the ~50/50 split
    assert 0 in m._regime_weights and 1 in m._regime_weights
    lin = m._model_names.index("Linear (Ridge)")
    # Linear earns more weight in the regime where the signal lives (1)
    assert m._regime_weights[1][lin] > m._regime_weights[0][lin]


def test_missing_regime_col_raises():
    X, y = _regime_xy()
    with pytest.raises(ValueError):
        RegimeWeightedEnsemble([NaiveModel(), LinearModel()]).fit(
            X.drop(columns="regime"), y)


def test_requires_two_models():
    with pytest.raises(ValueError):
        RegimeWeightedEnsemble([NaiveModel()])


def test_unseen_regime_uses_global_weights():
    X, y = _regime_xy()
    m = RegimeWeightedEnsemble([NaiveModel(), LinearModel()],
                               oof_initial_size=250, oof_step=22)
    m.fit(X, y)
    Xt = X.iloc[-10:].copy()
    Xt["regime"] = 9.0   # never seen
    preds = m.predict(Xt)
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


def test_oof_infeasible_falls_back_to_equal():
    X, y = _regime_xy(n=400)
    m = RegimeWeightedEnsemble([NaiveModel(), LinearModel()],
                               oof_initial_size=len(X) + 50)  # OOF impossible
    m.fit(X, y)
    assert m._regime_weights == {}
    np.testing.assert_allclose(m._global_weights, [0.5, 0.5])
    assert np.isfinite(m.predict(X.iloc[-5:])).all()


def test_name():
    m = RegimeWeightedEnsemble([NaiveModel(), LinearModel()])
    assert m.name == "RegimeWeighted(Naive (RW)+Linear (Ridge))"
