"""
tests/test_wave1_models.py
==========================
Unit tests for the Wave 1 interpretable monthly models.

    AdaptiveLassoModel        : oracle-property sparse selection
    selection_stability_report: cross-fold selection frequency / sign stability

(ECM + M-TAR and DMA/DMS tests are appended here as those models land.)

Run with:
    pytest tests/test_wave1_models.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import compare_models, walk_forward_cv  # noqa: E402
from src.models import (  # noqa: E402
    AdaptiveLassoModel,
    FuturesBasisBenchmark,
    NaiveModel,
    selection_stability_report,
)
from src.models_dma import DMAForecaster  # noqa: E402
from src.models_ecm import ECMForecaster  # noqa: E402


def _dma_xy(n=400, p=5, seed=2):
    """Only f0 carries signal; f1..f{p-1} are noise predictors."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, p)),
        columns=[f"f{i}" for i in range(p)],
        index=pd.date_range("2017-01-01", periods=n, freq="B"),
    )
    y = pd.Series(0.6 * X["f0"] + 0.1 * rng.standard_normal(n), index=X.index)
    return X, y


def _ect(n, seed, phi=0.9, sigma=1.0):
    """A mean-zero AR(1) 'distance from equilibrium' series."""
    rng = np.random.default_rng(seed)
    e = np.empty(n)
    e[0] = 0.0
    shocks = rng.standard_normal(n) * sigma
    for t in range(1, n):
        e[t] = phi * e[t - 1] + shocks[t]
    return e


def _ecm_xy(n=400, seed=1, asym=False):
    idx = pd.date_range("2017-01-01", periods=n, freq="B")
    ect = _ect(n, seed)
    rng = np.random.default_rng(seed + 100)
    if asym:
        # Faster reversion when copper is rich (ect >= 0) than when cheap.
        y = np.where(ect >= 0, -0.4 * ect, -0.1 * ect) + 0.05 * rng.standard_normal(n)
    else:
        y = -0.25 * ect + 0.05 * rng.standard_normal(n)
    X = pd.DataFrame({"ect_gold": ect, "noise": rng.standard_normal(n)}, index=idx)
    return X, pd.Series(y, index=idx)


def _sparse_xy(n=400, p=10, seed=0):
    """y depends only on f0 (+) and f3 (-); the rest are noise predictors."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, p)),
        columns=[f"f{i}" for i in range(p)],
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )
    y = pd.Series(2.0 * X["f0"] - 1.5 * X["f3"] + 0.1 * rng.standard_normal(n),
                  index=X.index)
    return X, y


class TestAdaptiveLasso:
    def test_fit_predict_finite(self):
        X, y = _sparse_xy()
        m = AdaptiveLassoModel(alpha=0.005).fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()
        assert m.coef_.shape == (X.shape[1],)

    def test_recovers_sparse_signal(self):
        X, y = _sparse_xy()
        m = AdaptiveLassoModel(alpha=0.002, gamma=1.0).fit(X, y)
        coef = pd.Series(m.coef_, index=X.columns)
        # True drivers selected with the correct sign...
        assert coef["f0"] > 0.5
        assert coef["f3"] < -0.3
        # ...and the model is genuinely sparse (most noise features dropped).
        assert (np.abs(coef) > 1e-8).sum() <= 6
        sel = m.selected_features()
        assert "f0" in sel and "f3" in sel

    def test_unfitted_raises(self):
        X, _ = _sparse_xy()
        with pytest.raises(RuntimeError):
            AdaptiveLassoModel().predict(X)

    def test_tune_sets_params(self):
        pytest.importorskip("optuna")
        X, y = _sparse_xy(n=300)
        m = AdaptiveLassoModel()
        out = m.tune(X, y, n_trials=5, cv_splits=3, horizon=5)
        assert "alpha" in out and "gamma" in out
        assert m.alpha == out["alpha"]


class TestSelectionStability:
    def test_report_flags_true_driver(self):
        X, y = _sparse_xy()
        report = selection_stability_report(
            lambda: AdaptiveLassoModel(alpha=0.002), X, y, n_splits=4, horizon=5)
        assert set(report.columns) == {
            "selection_freq", "mean_coef", "sign_consistency", "n_folds"}
        # The genuine drivers are selected in every fold with a consistent sign.
        assert report.loc["f0", "selection_freq"] == pytest.approx(1.0)
        assert report.loc["f0", "sign_consistency"] == pytest.approx(1.0)
        assert report.loc["f3", "selection_freq"] >= 0.75
        # f0 should rank above a pure-noise feature on selection frequency.
        assert report.loc["f0", "selection_freq"] >= report.loc["f5", "selection_freq"]


class TestECM:
    def test_recovers_negative_adjustment_speed(self):
        X, y = _ecm_xy()
        m = ECMForecaster(short_run_cols=["noise"]).fit(X, y)
        params = m.params_
        # Mean-reversion => negative loading on the error-correction term.
        assert params["ect_gold"] < 0
        preds = m.predict(X)
        assert np.isfinite(preds).all()

    def test_beats_naive_on_mean_reverting_data(self):
        X, y = _ecm_xy(n=500)
        m = ECMForecaster().fit(X, y)
        rmse_ecm = float(np.sqrt(np.mean((y.to_numpy() - m.predict(X)) ** 2)))
        rmse_naive = float(np.sqrt(np.mean(y.to_numpy() ** 2)))
        assert rmse_ecm < rmse_naive

    def test_prediction_interval_ordered(self):
        X, y = _ecm_xy()
        m = ECMForecaster().fit(X, y)
        ci = m.predict_interval(X.tail(20), alpha=0.80)
        assert list(ci.columns) == ["lower", "median", "upper"]
        assert (ci["lower"] <= ci["median"]).all()
        assert (ci["median"] <= ci["upper"]).all()

    def test_mtar_asymmetry_detected(self):
        X, y = _ecm_xy(n=600, asym=True)
        m = ECMForecaster(asymmetric=True, mode="tar").fit(X, y)
        params = m.params_
        assert "ect_gold__hi" in params.index
        assert "ect_gold__lo" in params.index
        test = m.asymmetry_test()
        assert np.isfinite(test["f_stat"])
        assert test["p_value"] < 0.05  # the two speeds genuinely differ

    def test_intercept_only_fallback(self):
        idx = pd.date_range("2019-01-01", periods=100, freq="B")
        X = pd.DataFrame({"foo": np.arange(100.0)}, index=idx)
        y = pd.Series(np.full(100, 0.01), index=idx)
        m = ECMForecaster().fit(X, y)  # no ect_* columns
        preds = m.predict(X)
        assert preds == pytest.approx(np.full(100, 0.01), abs=1e-6)

    def test_works_in_walk_forward_cv(self):
        X, y = _ecm_xy(n=400)
        cv = walk_forward_cv(ECMForecaster(), X, y,
                             initial_train_size=200, step_size=22, horizon=22)
        assert len(cv) > 0
        assert np.isfinite(cv["y_pred"]).all()


class TestDMA:
    def test_fit_predict_and_probabilities(self):
        X, y = _dma_xy()
        m = DMAForecaster().fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()
        ip = m.inclusion_probabilities_
        assert ip.shape == (len(X), X.shape[1])      # singleton model per predictor
        # Model probabilities form a distribution at every step.
        assert np.allclose(ip.sum(axis=1).to_numpy(), 1.0, atol=1e-8)

    def test_upweights_informative_predictor(self):
        X, y = _dma_xy(n=500)
        m = DMAForecaster(alpha=0.99, lam=0.99).fit(X, y)
        final = m.inclusion_probabilities_.iloc[-1]
        assert final["f0"] > 0.40
        assert final["f0"] > final["f1"]
        assert m.inclusion_probability("f0").iloc[-1] == pytest.approx(final["f0"])

    def test_beats_naive_rmse(self):
        X, y = _dma_xy(n=500)
        m = DMAForecaster().fit(X, y)
        rmse_dma = float(np.sqrt(np.mean((y.to_numpy() - m.predict(X)) ** 2)))
        rmse_naive = float(np.sqrt(np.mean(y.to_numpy() ** 2)))
        assert rmse_dma < rmse_naive

    def test_prediction_interval_ordered(self):
        X, y = _dma_xy()
        m = DMAForecaster().fit(X, y)
        ci = m.predict_interval(X.tail(20), alpha=0.80)
        assert list(ci.columns) == ["lower", "median", "upper"]
        assert (ci["lower"] <= ci["median"]).all()
        assert (ci["median"] <= ci["upper"]).all()

    def test_dms_variant(self):
        X, y = _dma_xy()
        m = DMAForecaster(dms=True).fit(X, y)
        assert m.name == "DMS"
        preds = m.predict(X)
        assert np.isfinite(preds).all()

    def test_include_full_model(self):
        X, y = _dma_xy()
        m = DMAForecaster(include_full=True).fit(X, y)
        assert "FULL" in m.inclusion_probabilities_.columns

    def test_works_in_walk_forward_cv(self):
        X, y = _dma_xy(n=400)
        cv = walk_forward_cv(DMAForecaster(), X, y,
                             initial_train_size=200, step_size=22, horizon=22)
        assert len(cv) > 0
        assert np.isfinite(cv["y_pred"]).all()


class TestWave0Wave1Integration:
    """The whole interpretable line-up runs through the Wave 0 harness."""

    def _data(self, n=320, seed=3):
        idx = pd.date_range("2017-01-01", periods=n, freq="B")
        rng = np.random.default_rng(seed)
        ect = _ect(n, seed)
        X = pd.DataFrame(
            {
                "ect_gold": ect,
                "copper_basis_pct": rng.standard_normal(n) * 0.01,
                "f0": rng.standard_normal(n),
                "f1": rng.standard_normal(n),
            },
            index=idx,
        )
        y = pd.Series(-0.2 * ect + 0.3 * X["f0"] + 0.1 * rng.standard_normal(n),
                      index=idx)
        return X, y

    def test_compare_models_emits_wave0_metrics(self):
        X, y = self._data()
        models = [
            NaiveModel(),
            FuturesBasisBenchmark(horizon=22),
            AdaptiveLassoModel(alpha=0.01),
            ECMForecaster(),
            DMAForecaster(),
        ]
        summary, cv_results = compare_models(
            models, X, y, initial_train_size=160, step_size=22, horizon=22)
        assert len(summary) == len(models)
        for col in ["cw_pvalue_vs_naive", "pt_pvalue", "mcs_pvalue", "in_mcs", "rmse"]:
            assert col in summary.columns
        # Every model produced a complete out-of-sample path.
        for cv in cv_results.values():
            assert np.isfinite(cv["y_pred"]).all()
