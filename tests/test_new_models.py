"""
tests/test_new_models.py
========================
Unit tests for ARIMAX, Prophet, Hybrid, and Stacking models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import build_features, split_features_targets
from src.models import LGBMModel, LinearModel, NaiveModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_price_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = 8000 + rng.standard_normal(n).cumsum() * 50
    dxy = 100 + rng.standard_normal(n).cumsum() * 0.5
    gold = 1800 + rng.standard_normal(n).cumsum() * 10
    aluminium = 2200 + rng.standard_normal(n).cumsum() * 30
    oil = 70 + rng.standard_normal(n).cumsum() * 1
    cny_usd = 0.155 + rng.standard_normal(n).cumsum() * 0.001
    sp500 = 4000 + rng.standard_normal(n).cumsum() * 20
    indpro = 100 + rng.standard_normal(n).cumsum() * 0.1
    real_yield = 0.5 + rng.standard_normal(n).cumsum() * 0.02
    infl_be = 2.5 + rng.standard_normal(n).cumsum() * 0.02
    return pd.DataFrame(
        {
            "copper_price": price,
            "dxy": dxy,
            "gold": gold,
            "aluminium": aluminium,
            "oil": oil,
            "cny_usd": cny_usd,
            "sp500": sp500,
            "indpro": indpro,
            "real_yield_10y": real_yield,
            "inflation_breakeven": infl_be,
        },
        index=idx,
    )


def _make_xy(n: int = 500, horizon: int = 22, seed: int = 42):
    df = _make_price_df(n=n, seed=seed)
    feats = build_features(df)
    return split_features_targets(feats, horizon=horizon)


# ---------------------------------------------------------------------------
# Prophet tests
# ---------------------------------------------------------------------------


class TestProphetModel:
    def test_fit_predict_shape(self):
        pytest.importorskip("prophet")
        from src.models_prophet import ProphetModel

        X, y, _ = _make_xy(n=400)
        m = ProphetModel()
        m.fit(X[:250], y[:250])
        preds = m.predict(X[250:270])
        assert preds.shape == (20,)

    def test_predict_interval(self):
        pytest.importorskip("prophet")
        from src.models_prophet import ProphetModel

        X, y, _ = _make_xy(n=400)
        m = ProphetModel()
        m.fit(X[:250], y[:250])
        ci = m.predict_interval(X[250:270])
        assert isinstance(ci, pd.DataFrame)
        assert "lower" in ci.columns
        assert "upper" in ci.columns

    def test_name(self):
        from src.models_prophet import ProphetModel
        assert ProphetModel().name == "Prophet"


# ---------------------------------------------------------------------------
# Stacking ensemble tests
# ---------------------------------------------------------------------------


class TestStackingEnsemble:
    def test_fit_predict_shape(self):
        from src.models_stacking import StackingEnsemble

        X, y, _ = _make_xy(n=800)
        split = len(X) - 20
        m = StackingEnsemble(
            base_models=[NaiveModel(), LinearModel()],
            oof_initial_size=200,
            oof_step=22,
        )
        m.fit(X.iloc[:split], y.iloc[:split])
        preds = m.predict(X.iloc[split:])
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_meta_learner_trained(self):
        from src.models_stacking import StackingEnsemble

        X, y, _ = _make_xy(n=800)
        split = len(X) - 20
        m = StackingEnsemble(
            base_models=[NaiveModel(), LinearModel()],
            oof_initial_size=200,
            oof_step=22,
        )
        m.fit(X.iloc[:split], y.iloc[:split])
        assert m._meta_fitted is True

    def test_empty_base_models_raises(self):
        from src.models_stacking import StackingEnsemble

        with pytest.raises(ValueError):
            StackingEnsemble(base_models=[])

    def test_name(self):
        from src.models_stacking import StackingEnsemble

        m = StackingEnsemble(base_models=[NaiveModel(), LinearModel()])
        assert "Stacking" in m.name
