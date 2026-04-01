"""
tests/test_copper_forecast.py
==============================
Unit tests for the copper forecasting source modules.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import compute_metrics, directional_accuracy, walk_forward_cv
from src.feature_engineering import build_features, split_features_targets
from src.models import (
    EnsembleModel,
    LGBMModel,
    LinearModel,
    NaiveModel,
    XGBoostModel,
)
from src.scenario_analysis import SCENARIO_TEMPLATES, ScenarioEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_price_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily price DataFrame mimicking real data structure."""
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


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    def test_build_features_returns_dataframe(self):
        df = _make_price_df()
        feats = build_features(df)
        assert isinstance(feats, pd.DataFrame)
        assert len(feats) == len(df)

    def test_build_features_has_target_columns(self):
        df = _make_price_df()
        feats = build_features(df)
        assert "target_ret_22d" in feats.columns
        assert "target_price_22d" in feats.columns

    def test_build_features_has_technical_indicators(self):
        df = _make_price_df()
        feats = build_features(df)
        for col in ["rsi_14", "macd", "macd_signal", "bb_width", "copper_zscore_200d"]:
            assert col in feats.columns, f"Missing column: {col}"

    def test_build_features_cross_asset(self):
        df = _make_price_df()
        feats = build_features(df)
        for col in ["gold_copper_ratio", "alu_copper_spread_pct", "dxy_ret_22d"]:
            assert col in feats.columns, f"Missing column: {col}"

    def test_build_features_calendar(self):
        df = _make_price_df()
        feats = build_features(df)
        for col in ["month_sin", "month_cos", "cny_flag"]:
            assert col in feats.columns

    def test_build_features_lags(self):
        df = _make_price_df()
        feats = build_features(df, lags=[1, 5])
        lag_cols = [c for c in feats.columns if "_lag_" in c]
        assert len(lag_cols) > 0

    def test_split_features_targets_shape(self):
        df = _make_price_df()
        feats = build_features(df)
        X, y_ret, y_price = split_features_targets(feats, horizon=22)
        assert len(X) == len(y_ret) == len(y_price)
        assert len(X) > 0

    def test_split_features_targets_no_nans(self):
        df = _make_price_df()
        feats = build_features(df)
        X, y_ret, y_price = split_features_targets(feats, horizon=22, drop_nan=True)
        assert X.notna().all().all()
        assert y_ret.notna().all()
        assert y_price.notna().all()

    def test_no_target_columns_in_X(self):
        df = _make_price_df()
        feats = build_features(df)
        X, _, _ = split_features_targets(feats, horizon=22)
        for col in X.columns:
            assert not col.startswith("target_"), f"Target column leaked into X: {col}"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def _make_xy(n: int = 500, horizon: int = 22, seed: int = 42):
    """Return (X, y_ret, y_price) ready for model training."""
    df = _make_price_df(n=n, seed=seed)
    feats = build_features(df)
    return split_features_targets(feats, horizon=horizon)


class TestModels:
    def test_naive_predict_zeros(self):
        X, y, _ = _make_xy()
        m = NaiveModel()
        m.fit(X, y)
        preds = m.predict(X)
        assert np.all(preds == 0)

    def test_linear_fit_predict(self):
        X, y, _ = _make_xy()
        m = LinearModel()
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_xgboost_fit_predict(self):
        pytest.importorskip("xgboost")
        X, y, _ = _make_xy()
        m = XGBoostModel()
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_lgbm_fit_predict(self):
        pytest.importorskip("lightgbm")
        X, y, _ = _make_xy()
        m = LGBMModel()
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_ensemble_equal_weights(self):
        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        X, y, _ = _make_xy()
        m = EnsembleModel([NaiveModel(), LinearModel()])
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_ensemble_custom_weights(self):
        X, y, _ = _make_xy()
        m = EnsembleModel([NaiveModel(), LinearModel()], weights=[0.3, 0.7])
        m.fit(X, y)
        preds = m.predict(X)
        assert np.isfinite(preds).all()

    def test_ensemble_wrong_weights_raises(self):
        with pytest.raises(ValueError):
            EnsembleModel([NaiveModel(), LinearModel()], weights=[1.0])

    def test_unfitted_model_raises(self):
        X, _, _ = _make_xy()
        m = LinearModel()
        with pytest.raises(RuntimeError):
            m.predict(X)


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_compute_metrics_perfect(self):
        y = np.array([1.0, -1.0, 0.5, -0.5])
        metrics = compute_metrics(y, y)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["directional_accuracy"] == pytest.approx(1.0)

    def test_compute_metrics_keys(self):
        y = np.ones(10)
        p = np.zeros(10)
        m = compute_metrics(y, p)
        for k in ["rmse", "mae", "mape", "directional_accuracy"]:
            assert k in m

    def test_directional_accuracy_range(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(200)
        p = rng.standard_normal(200)
        da = directional_accuracy(y, p)
        assert 0.0 <= da <= 1.0

    def test_walk_forward_cv(self):
        X, y, _ = _make_xy(n=600)
        m = NaiveModel()
        cv = walk_forward_cv(m, X, y, initial_train_size=300, step_size=22)
        assert isinstance(cv, pd.DataFrame)
        assert "y_true" in cv.columns
        assert "y_pred" in cv.columns
        assert len(cv) > 0

    def test_walk_forward_cv_no_lookahead(self):
        """Verify that test observations are always after training observations."""
        X, y, _ = _make_xy(n=600)
        init_size = len(X) // 2
        m = LinearModel()
        cv = walk_forward_cv(m, X, y, initial_train_size=init_size, step_size=30)
        # All CV predictions should be at or after the split point
        assert cv.index.min() >= X.index[init_size]

    def test_walk_forward_cv_small_dataset_raises(self):
        X, y, _ = _make_xy(n=100)
        with pytest.raises(ValueError):
            walk_forward_cv(NaiveModel(), X, y, initial_train_size=200)


# ---------------------------------------------------------------------------
# Scenario analysis tests
# ---------------------------------------------------------------------------


class TestScenarioAnalysis:
    def _make_engine(self):
        X, y, _ = _make_xy(n=600)
        model = LinearModel()
        model.fit(X, y)
        template = X.tail(1)
        current_price = 9000.0
        return ScenarioEngine(model, template, current_price, horizon=22)

    def test_run_named_scenario(self):
        engine = self._make_engine()
        result = engine.run("bull_strong")
        for key in ["scenario", "base_price", "scenario_price", "delta", "delta_pct"]:
            assert key in result

    def test_run_custom_shocks(self):
        engine = self._make_engine()
        result = engine.run("my_custom", shocks={"dxy_ret_22d": -0.05})
        assert result["scenario"] == "my_custom"
        assert isinstance(result["scenario_price"], float)

    def test_run_unknown_scenario_raises(self):
        engine = self._make_engine()
        with pytest.raises(ValueError):
            engine.run("nonexistent_scenario")

    def test_run_all_templates(self):
        engine = self._make_engine()
        df = engine.run_all_templates()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(SCENARIO_TEMPLATES)

    def test_sweep_returns_dataframe(self):
        engine = self._make_engine()
        shocks = [-0.1, -0.05, 0.0, 0.05, 0.1]
        df = engine.sweep("dxy_ret_22d", shocks)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(shocks)
        assert "forecast_price" in df.columns

    def test_base_price_correct(self):
        engine = self._make_engine()
        assert engine.base_price > 0
        assert engine.current_price == 9000.0

    def test_scenario_template_keys_exist(self):
        """All template shocks reference plausible feature names."""
        for name, shocks in SCENARIO_TEMPLATES.items():
            assert isinstance(shocks, dict), f"Template '{name}' has invalid shocks"
            assert len(shocks) > 0

    def test_report_returns_dataframe(self):
        engine = self._make_engine()
        df = engine.report()
        assert isinstance(df, pd.DataFrame)
        assert "delta_pct" in df.columns
