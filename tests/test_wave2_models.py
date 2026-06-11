"""
tests/test_wave2_models.py
==========================
Unit tests for the Wave 2 model families:

    MIDASForecaster          : restricted distributed-lag (Beta weights)
    MarkovSwitchingForecaster : regime-switching mean model
    GAMForecaster            : interpretable additive splines
    GarchMidasModel          : long-run-macro volatility / intervals
    TrimmedMeanForecaster    : robust 1/N combination
    white_reality_check      : data-snooping governance test

Run with:
    pytest tests/test_wave2_models.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import walk_forward_cv, white_reality_check  # noqa: E402
from src.models import LinearModel, NaiveModel, TrimmedMeanForecaster  # noqa: E402
from src.models_gam import GAMForecaster  # noqa: E402
from src.models_garch_midas import GarchMidasModel  # noqa: E402
from src.models_markov import MarkovSwitchingForecaster  # noqa: E402
from src.models_midas import MIDASForecaster, _beta_weights  # noqa: E402


def _garch_midas_xy(n=320, seed=13):
    """Returns whose variance is driven (positively) by a lagged macro series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")
    mv = np.zeros(n)
    for t in range(1, n):
        mv[t] = 0.85 * mv[t - 1] + 0.5 * rng.standard_normal()
    sig2 = np.exp(-2.0 + 0.8 * np.concatenate([[0.0], mv[:-1]]))   # var ~ exp(b * mv_{t-1})
    y = pd.Series(np.sqrt(sig2) * rng.standard_normal(n), index=idx)
    X = pd.DataFrame({"macro": mv, "noise": rng.standard_normal(n)}, index=idx)
    return X, y


def _nonlinear_xy(n=300, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    x0 = rng.uniform(-3, 3, n)
    x1 = rng.standard_normal(n)
    y = pd.Series(0.6 * np.sin(2.0 * x0) + 0.3 * x1 + 0.1 * rng.standard_normal(n), index=idx)
    X = pd.DataFrame({"x0": x0, "x1": x1, "noise": rng.standard_normal(n)}, index=idx)
    return X, y


def _regime_xy(n=240, seed=7):
    """Two-regime data: sign of the driver effect flips by hidden regime."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    x = rng.standard_normal(n)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):                       # persistent 2-state chain
        state[t] = state[t - 1] if rng.random() < 0.92 else 1 - state[t - 1]
    eff = np.where(state == 0, 0.6, -0.4)
    noise = np.where(state == 0, 0.1, 0.25) * rng.standard_normal(n)
    y = pd.Series(eff * x + noise, index=idx)
    X = pd.DataFrame({"f0": x, "f1": rng.standard_normal(n)}, index=idx)
    return X, y


# ---------------------------------------------------------------------------
# MIDAS
# ---------------------------------------------------------------------------


def _midas_xy(n=320, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(n)
    idx = pd.date_range("2017-01-01", periods=n, freq="MS")
    cols = {"dxy": p}
    for k in (1, 3, 6):
        cols[f"dxy_lag_{k}"] = np.concatenate([np.zeros(k), p[:-k]])
    X = pd.DataFrame(cols, index=idx)
    w = np.array([0.5, 0.3, 0.15, 0.05])               # decaying true weights
    factor = X[["dxy", "dxy_lag_1", "dxy_lag_3", "dxy_lag_6"]].to_numpy() @ w
    y = pd.Series(0.8 * factor + 0.1 * rng.standard_normal(n), index=idx)
    return X, y


class TestMIDAS:
    def test_beta_weights_decay_and_sum_to_one(self):
        w = _beta_weights(5, 6.0)
        assert w.shape == (5,)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(np.diff(w) <= 0)                 # monotone decreasing

    def test_fit_predict_and_beats_naive(self):
        X, y = _midas_xy()
        m = MIDASForecaster().fit(X, y)
        preds = m.predict(X)
        assert np.isfinite(preds).all()
        rmse = float(np.sqrt(np.mean((y.to_numpy() - preds) ** 2)))
        rmse_naive = float(np.sqrt(np.mean(y.to_numpy() ** 2)))
        assert rmse < rmse_naive

    def test_weight_curve_exposed(self):
        X, y = _midas_xy()
        m = MIDASForecaster().fit(X, y)
        wc = m.weight_curve()
        assert "dxy" in wc
        assert len(wc["dxy"]) == 4
        assert wc["dxy"].sum() == pytest.approx(1.0)

    def test_explicit_lag_groups(self):
        X, y = _midas_xy()
        m = MIDASForecaster(lag_groups={"dxy": ["dxy", "dxy_lag_1", "dxy_lag_3"]}).fit(X, y)
        assert np.isfinite(m.predict(X)).all()

    def test_prediction_interval_ordered(self):
        X, y = _midas_xy()
        m = MIDASForecaster().fit(X, y)
        ci = m.predict_interval(X.tail(12), alpha=0.80)
        assert (ci["lower"] <= ci["median"]).all()
        assert (ci["median"] <= ci["upper"]).all()

    def test_intercept_only_fallback(self):
        idx = pd.date_range("2019-01-01", periods=80, freq="MS")
        X = pd.DataFrame({"foo": np.arange(80.0)}, index=idx)   # no *_lag_* columns
        y = pd.Series(np.full(80, 0.02), index=idx)
        m = MIDASForecaster().fit(X, y)
        assert m.predict(X) == pytest.approx(np.full(80, 0.02), abs=1e-6)

    def test_works_in_walk_forward_cv(self):
        X, y = _midas_xy()
        cv = walk_forward_cv(MIDASForecaster(), X, y,
                             initial_train_size=160, step_size=3, horizon=3)
        assert len(cv) > 0
        assert np.isfinite(cv["y_pred"]).all()


# ---------------------------------------------------------------------------
# Forecast-combination governance
# ---------------------------------------------------------------------------


class TestCombinationGovernance:
    def test_trimmed_mean_predict(self):
        rng = np.random.default_rng(1)
        n = 200
        X = pd.DataFrame({"f0": rng.standard_normal(n)},
                         index=pd.date_range("2018-01-01", periods=n, freq="MS"))
        y = pd.Series(0.5 * X["f0"] + rng.standard_normal(n) * 0.1, index=X.index)

        class _Const(NaiveModel):
            def __init__(self, v):
                self.v = v

            def predict(self, X):
                return np.full(len(X), self.v)

        m = TrimmedMeanForecaster([_Const(0.0), _Const(1.0), _Const(100.0)], trim=0.34)
        m.fit(X, y)
        # 3 members, trim 34% -> drop one each tail -> median (=1.0), rogue 100 ignored
        assert m.predict(X) == pytest.approx(np.full(n, 1.0))

    def test_trimmed_mean_bad_trim_raises(self):
        with pytest.raises(ValueError):
            TrimmedMeanForecaster([NaiveModel()], trim=0.6)

    def test_white_reality_check_detects_real_winner(self):
        rng = np.random.default_rng(2)
        T = 300
        # benchmark loss ~1; model 0 genuinely lower; models 1-2 ~ benchmark.
        lb = 1.0 + 0.2 * rng.standard_normal(T)
        m0 = 0.6 + 0.2 * rng.standard_normal(T)
        m1 = 1.0 + 0.2 * rng.standard_normal(T)
        m2 = 1.02 + 0.2 * rng.standard_normal(T)
        res = white_reality_check(lb, np.column_stack([m0, m1, m2]),
                                  n_boot=400, random_state=3)
        assert res["best_model"] == 0
        assert res["p_value"] < 0.05

    def test_white_reality_check_no_winner(self):
        rng = np.random.default_rng(4)
        T = 300
        lb = 1.0 + 0.2 * rng.standard_normal(T)
        # all models slightly worse than benchmark
        M = np.column_stack([1.05 + 0.2 * rng.standard_normal(T) for _ in range(3)])
        res = white_reality_check(lb, M, n_boot=400, random_state=5)
        assert res["p_value"] > 0.10


# ---------------------------------------------------------------------------
# Markov-switching
# ---------------------------------------------------------------------------


class TestMarkovSwitching:
    def test_fit_predict_finite(self):
        X, y = _regime_xy()
        m = MarkovSwitchingForecaster(k_regimes=2, exog_cols=["f0"]).fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_regime_outputs(self):
        X, y = _regime_xy()
        m = MarkovSwitchingForecaster(k_regimes=2, exog_cols=["f0"]).fit(X, y)
        sp = m.smoothed_probabilities()
        assert sp is not None
        dur = m.expected_durations()
        assert dur is not None and np.all(dur > 0)

    def test_prediction_interval_ordered(self):
        X, y = _regime_xy()
        m = MarkovSwitchingForecaster(k_regimes=2, exog_cols=["f0"]).fit(X, y)
        ci = m.predict_interval(X.tail(12), alpha=0.80)
        assert (ci["lower"] <= ci["median"]).all()
        assert (ci["median"] <= ci["upper"]).all()

    def test_auto_exog_selection(self):
        X, y = _regime_xy()
        m = MarkovSwitchingForecaster(k_regimes=2, max_exog=2).fit(X, y)
        assert np.isfinite(m.predict(X)).all()

    def test_works_in_walk_forward_cv(self):
        X, y = _regime_xy(n=180)
        cv = walk_forward_cv(MarkovSwitchingForecaster(k_regimes=2, exog_cols=["f0"]),
                             X, y, initial_train_size=120, step_size=6, horizon=3)
        assert len(cv) > 0
        assert np.isfinite(cv["y_pred"]).all()


# ---------------------------------------------------------------------------
# GAM
# ---------------------------------------------------------------------------


class TestGAM:
    def test_fit_predict_beats_naive_on_nonlinear(self):
        X, y = _nonlinear_xy()
        m = GAMForecaster(feature_cols=["x0", "x1"]).fit(X, y)
        preds = m.predict(X)
        assert np.isfinite(preds).all()
        rmse = float(np.sqrt(np.mean((y.to_numpy() - preds) ** 2)))
        rmse_naive = float(np.sqrt(np.mean(y.to_numpy() ** 2)))
        assert rmse < 0.7 * rmse_naive          # captures the nonlinearity

    def test_beats_linear_on_nonlinear(self):
        X, y = _nonlinear_xy()
        gam = GAMForecaster(feature_cols=["x0", "x1"]).fit(X, y)
        lin = LinearModel().fit(X[["x0", "x1"]], y)
        rmse_gam = float(np.sqrt(np.mean((y.to_numpy() - gam.predict(X)) ** 2)))
        rmse_lin = float(np.sqrt(np.mean((y.to_numpy() - lin.predict(X[["x0", "x1"]])) ** 2)))
        assert rmse_gam < rmse_lin

    def test_prediction_interval_ordered(self):
        X, y = _nonlinear_xy()
        m = GAMForecaster(feature_cols=["x0", "x1"]).fit(X, y)
        ci = m.predict_interval(X.tail(20), alpha=0.80)
        assert (ci["lower"] <= ci["median"]).all()
        assert (ci["median"] <= ci["upper"]).all()

    def test_partial_dependence_curve(self):
        X, y = _nonlinear_xy()
        m = GAMForecaster(feature_cols=["x0", "x1"]).fit(X, y)
        pdp = m.partial_dependence("x0")
        # pygam backend -> a (xs, ys) curve; sklearn fallback -> None
        if pdp is not None:
            xs, ys = pdp
            assert len(xs) == len(ys) > 0

    def test_auto_feature_selection_and_wfcv(self):
        X, y = _nonlinear_xy(n=180)
        cv = walk_forward_cv(GAMForecaster(max_features=4), X, y,
                             initial_train_size=120, step_size=6, horizon=3)
        assert len(cv) > 0
        assert np.isfinite(cv["y_pred"]).all()


# ---------------------------------------------------------------------------
# GARCH-MIDAS
# ---------------------------------------------------------------------------


class TestGarchMidas:
    def test_recovers_positive_macro_elasticity(self):
        X, y = _garch_midas_xy()
        m = GarchMidasModel(macro_col="macro", n_lags=12).fit(X, y)
        assert np.isfinite(m.theta_)
        assert m.theta_ > 0                      # macro positively drives long-run vol

    def test_forecast_variance_positive(self):
        X, y = _garch_midas_xy()
        m = GarchMidasModel(macro_col="macro").fit(X, y)
        v = m.forecast_variance()
        assert np.isfinite(v) and v > 0

    def test_predict_is_zero_mean(self):
        X, y = _garch_midas_xy()
        m = GarchMidasModel(macro_col="macro").fit(X, y)
        assert np.all(m.predict(X) == 0.0)

    def test_interval_ordered_and_positive_width(self):
        X, y = _garch_midas_xy()
        m = GarchMidasModel(macro_col="macro").fit(X, y)
        ci = m.predict_interval(X.tail(10), alpha=0.80)
        assert (ci["lower"] < ci["upper"]).all()
        assert (ci["lower"] <= ci["median"]).all() and (ci["median"] <= ci["upper"]).all()

    def test_rv_proxy_without_macro(self):
        X, y = _garch_midas_xy()
        m = GarchMidasModel(macro_col=None).fit(X, y)   # uses squared-return proxy
        assert np.isfinite(m.theta_)
        assert m.forecast_variance() > 0

    def test_works_in_walk_forward_cv(self):
        X, y = _garch_midas_xy(n=180)
        cv = walk_forward_cv(GarchMidasModel(macro_col="macro"), X, y,
                             initial_train_size=120, step_size=6, horizon=3)
        assert len(cv) > 0
        assert np.isfinite(cv["y_pred"]).all()
