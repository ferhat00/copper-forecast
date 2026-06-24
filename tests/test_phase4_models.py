"""
tests/test_phase4_models.py
===========================
Unit tests for the Phase 4/5 additions:

  * AR1Model / DLinearForecaster        (src/models_baselines.py)
  * ResidualBoostForecaster             (src/models_hybrid.py)
  * KalmanFuturesForecaster             (src/models_futures.py)
  * FoundationModelForecaster           (src/models_foundation.py)  -- graceful skip
  * alt-data loaders + attach helper    (src/altdata.py)
  * factory wiring of all of the above  (src/model_lineup.py)

All synthetic / offline — no network, no model weights.

Run with:
    pytest tests/test_phase4_models.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.altdata import (  # noqa: E402
    attach_altdata_features,
    load_gpr_epu,
    load_news_sentiment,
)
from src.evaluation import compare_models  # noqa: E402
from src.model_lineup import build_model_lineup  # noqa: E402
from src.models_baselines import AR1Model, DLinearForecaster  # noqa: E402
from src.models_foundation import FoundationModelForecaster  # noqa: E402
from src.models_futures import KalmanFuturesForecaster  # noqa: E402
from src.models_hybrid import ResidualBoostForecaster  # noqa: E402


def _panel(n=260, seed=7):
    """Synthetic monthly panel with the column names build_features emits."""
    idx = pd.date_range("2004-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal(n) * 0.01
    r1 = rng.standard_normal(n) * 0.05
    X = pd.DataFrame(
        {
            "copper_ret_1m": r1,
            "copper_ret_3m": rng.standard_normal(n) * 0.05,
            "copper_ret_6m": rng.standard_normal(n) * 0.05,
            "copper_ret_12m": rng.standard_normal(n) * 0.05,
            "copper_basis_pct": basis,
            "ect_gold": rng.standard_normal(n) * 0.1,
            "dxy_ret_1m": rng.standard_normal(n) * 0.02,
        },
        index=idx,
    )
    return X, idx, rng, basis, r1


class TestAR1:
    def test_recovers_persistence(self):
        X, idx, rng, _basis, r1 = _panel()
        y = pd.Series(0.6 * r1 + 0.01 * rng.standard_normal(len(idx)), index=idx)
        m = AR1Model().fit(X, y)
        assert m._col == "copper_ret_1m"          # shortest-lag return chosen
        assert m._b1 > 0.3                          # positive persistence recovered
        assert np.isfinite(m.predict(X)).all()

    def test_drift_fallback_without_returns(self):
        idx = pd.date_range("2010-01-31", periods=60, freq="ME")
        X = pd.DataFrame({"foo": np.arange(60.0)}, index=idx)
        y = pd.Series(np.full(60, 0.02), index=idx)
        m = AR1Model().fit(X, y)
        assert np.allclose(m.predict(X), 0.02, atol=1e-6)


class TestDLinear:
    def test_fit_predict_finite(self):
        X, idx, rng, _b, _r = _panel()
        y = pd.Series(rng.standard_normal(len(idx)) * 0.05, index=idx)
        preds = DLinearForecaster().fit(X, y).predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_fallback_without_lookback(self):
        idx = pd.date_range("2010-01-31", periods=80, freq="ME")
        X = pd.DataFrame({"a": np.arange(80.0), "b": np.arange(80.0)[::-1]}, index=idx)
        y = pd.Series(np.arange(80.0) * 0.001, index=idx)
        m = DLinearForecaster().fit(X, y)
        assert m._fallback is True
        assert np.isfinite(m.predict(X)).all()


class TestResidualBoost:
    def test_fit_predict_and_base_passthrough(self):
        X, idx, rng, _b, r1 = _panel()
        y = pd.Series(0.5 * r1 + 0.02 * rng.standard_normal(len(idx)), index=idx)
        m = ResidualBoostForecaster().fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()
        assert m._base is not None              # linear backbone always fitted


class TestKalmanFutures:
    def test_recovers_positive_loading_on_informative_basis(self):
        X, idx, rng, basis, _r = _panel()
        implied = basis * (3 / 3)               # horizon=tenor=3 -> implied = basis
        y = pd.Series(0.5 * implied + 0.005 * rng.standard_normal(len(idx)), index=idx)
        m = KalmanFuturesForecaster(horizon=3, tenor_periods=3).fit(X, y)
        assert m.coef_["futures_basis"] > 0.1
        assert np.isfinite(m.predict(X)).all()

    def test_random_walk_fallback_without_basis(self):
        idx = pd.date_range("2010-01-31", periods=60, freq="ME")
        X = pd.DataFrame({"foo": np.arange(60.0)}, index=idx)
        y = pd.Series(np.random.default_rng(0).standard_normal(60) * 0.05, index=idx)
        m = KalmanFuturesForecaster().fit(X, y)
        assert np.allclose(m.predict(X), 0.0)   # no basis col -> RW (zeros)


class TestFoundationModel:
    def test_unavailable_backend_degrades_to_rw(self):
        idx = pd.date_range("2010-01-31", periods=40, freq="ME")
        X = pd.DataFrame({"f": np.arange(40.0)}, index=idx)
        y = pd.Series(np.arange(40.0) * 0.001, index=idx)
        m = FoundationModelForecaster(backend="definitely_not_installed").fit(X, y)
        assert m.available_ is False
        assert np.allclose(m.predict(X), 0.0)
        assert "->RW" in m.name

    def test_auto_backend_offline_is_finite(self):
        idx = pd.date_range("2010-01-31", periods=40, freq="ME")
        X = pd.DataFrame({"f": np.arange(40.0)}, index=idx)
        y = pd.Series(np.arange(40.0) * 0.001, index=idx)
        preds = FoundationModelForecaster(backend="auto").fit(X, y).predict(X)
        assert preds.shape == (40,) and np.isfinite(preds).all()


class TestAltData:
    def test_news_sentiment_csv(self, tmp_path):
        p = tmp_path / "sent.csv"
        pd.DataFrame(
            {"date": pd.date_range("2020-01-31", periods=12, freq="ME"),
             "sentiment": np.linspace(-1, 1, 12)}
        ).to_csv(p, index=False)
        out = load_news_sentiment(str(p))
        assert "news_sentiment" in out.columns and len(out) == 12

    def test_news_sentiment_missing_is_empty(self):
        assert load_news_sentiment(None).empty

    def test_gpr_from_csv(self, tmp_path):
        p = tmp_path / "gpr.csv"
        pd.DataFrame(
            {"date": pd.date_range("2020-01-31", periods=10, freq="ME"),
             "GPR": np.arange(10.0)}
        ).to_csv(p, index=False)
        out = load_gpr_epu("2020-01-01", "2020-12-31", gpr_csv_path=str(p))
        assert "gpr" in out.columns

    def test_attach_joins_enabled_sources(self, tmp_path):
        p = tmp_path / "sent.csv"
        pd.DataFrame(
            {"date": pd.date_range("2019-01-31", periods=24, freq="ME"),
             "sentiment": np.zeros(24)}
        ).to_csv(p, index=False)
        df = pd.DataFrame({"copper_price": np.arange(24.0)},
                          index=pd.date_range("2019-01-31", periods=24, freq="ME"))
        cfg = {"features": {"add_news_sentiment": True},
               "sources": {"news_sentiment_csv": str(p)}}
        out = attach_altdata_features(df, cfg, start="2019-01-01", end="2020-12-31")
        assert "news_sentiment" in out.columns


class TestFactoryWiring:
    def test_default_lineup_includes_phase4_models(self):
        models = build_model_lineup({}, horizon=3)
        names = [m.name for m in models]
        assert "AR(1)" in names
        assert "DLinear" in names
        assert "Kalman-Futures" in names
        assert any("Residual-Boost" in n for n in names)

    def test_foundation_key_builds_and_is_opt_in(self):
        models = build_model_lineup(
            {"models": {"enabled": ["naive", "chronos"]}}, horizon=3)
        assert any(isinstance(m, FoundationModelForecaster) for m in models)

    def test_phase4_lineup_runs_through_compare_models(self):
        X, idx, rng, basis, r1 = _panel(n=240)
        y = pd.Series(0.4 * r1 + 0.3 * basis + 0.02 * rng.standard_normal(len(idx)),
                      index=idx)
        models = build_model_lineup(
            {"models": {"enabled": ["naive", "ar1", "dlinear", "kalman_futures",
                                    "residual_hybrid", "combo_equal"]}},
            horizon=3)
        summary, cv = compare_models(models, X, y, initial_train_size=120,
                                     step_size=6, horizon=3, periods_per_year=12)
        assert len(summary) == len(models)
        for c in cv.values():
            assert np.isfinite(c["y_pred"]).all()
