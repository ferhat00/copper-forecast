"""
tests/test_evaluation_wave0.py
==============================
Unit tests for the Wave 0 evaluation harness additions:

    clark_west            : nested-model (vs random walk) accuracy test
    pesaran_timmermann    : directional-skill test
    model_confidence_set  : Hansen-Lunde-Nason MCS
    interval_metrics      : prediction-interval coverage / score
    crps_gaussian         : Gaussian CRPS (closed form)
    crps_from_quantiles   : CRPS from a quantile grid
    pinball_loss          : quantile loss
    subperiod_metrics     : per-sub-period / per-regime breakdown
    compare_models        : now emits CW / PT / MCS columns

Run with:
    pytest tests/test_evaluation_wave0.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (  # noqa: E402
    clark_west,
    compare_models,
    crps_from_quantiles,
    crps_gaussian,
    interval_metrics,
    model_confidence_set,
    pesaran_timmermann,
    pinball_loss,
    subperiod_metrics,
)
from src.models import LinearModel, NaiveModel  # noqa: E402


# ---------------------------------------------------------------------------
# Clark-West
# ---------------------------------------------------------------------------


class TestClarkWest:
    def test_informative_model_beats_rw(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(400) * 0.02
        # A forecast that captures part of y should beat the RW (zeros).
        pred = 0.5 * y + rng.standard_normal(400) * 0.01
        res = clark_west(y, pred, horizon=1)
        assert res["cw_stat"] > 0
        assert res["p_value"] < 0.05

    def test_zero_model_is_degenerate_tie(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(200) * 0.02
        res = clark_west(y, np.zeros_like(y), horizon=1)
        # pred == benchmark -> f is identically zero -> reported as a tie.
        assert res["cw_stat"] == pytest.approx(0.0, abs=1e-9)
        assert res["p_value"] == pytest.approx(0.5, abs=1e-9)

    def test_too_few_obs_returns_nan(self):
        res = clark_west(np.array([0.1, -0.1]), np.array([0.05, -0.05]))
        assert np.isnan(res["cw_stat"])

    def test_custom_benchmark(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(300) * 0.02
        bench = rng.standard_normal(300) * 0.02
        model = 0.6 * y + 0.1 * bench
        res = clark_west(y, model, pred_bench=bench, horizon=1)
        assert np.isfinite(res["cw_stat"])


# ---------------------------------------------------------------------------
# Pesaran-Timmermann
# ---------------------------------------------------------------------------


class TestPesaranTimmermann:
    def test_perfect_direction_is_significant(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal(300) * 0.02
        pred = y.copy()  # identical signs
        res = pesaran_timmermann(y, pred)
        assert res["hit_rate"] == pytest.approx(1.0)
        assert res["pt_stat"] > 0
        assert res["p_value"] < 0.01

    def test_random_direction_not_significant(self):
        rng = np.random.default_rng(4)
        y = rng.standard_normal(500) * 0.02
        pred = rng.standard_normal(500) * 0.02  # independent of y
        res = pesaran_timmermann(y, pred)
        assert res["p_value"] > 0.05

    def test_too_few_obs_returns_nan(self):
        res = pesaran_timmermann(np.array([0.1, -0.1]), np.array([0.1, -0.1]))
        assert np.isnan(res["pt_stat"])


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------


class TestModelConfidenceSet:
    def _losses(self, seed=5, T=400):
        rng = np.random.default_rng(seed)
        loss_a = (0.30 * rng.standard_normal(T)) ** 2   # best
        loss_b = (0.33 * rng.standard_normal(T)) ** 2   # close to best
        loss_c = (1.20 * rng.standard_normal(T)) ** 2   # clearly worst
        return pd.DataFrame({"A": loss_a, "B": loss_b, "C": loss_c})

    def test_best_in_set_worst_eliminated(self):
        losses = self._losses()
        res = model_confidence_set(losses, alpha=0.10, n_boot=300, random_state=7)
        assert "A" in res["included"]
        assert "C" not in res["included"]
        assert "C" in res["eliminated_order"]
        assert res["mcs_pvalues"]["A"] >= 0.10
        assert res["mcs_pvalues"]["C"] < 0.10

    def test_pvalues_indexed_by_model(self):
        losses = self._losses()
        res = model_confidence_set(losses, alpha=0.10, n_boot=200, random_state=7)
        assert set(res["mcs_pvalues"].index) == {"A", "B", "C"}

    def test_single_model_returns_all(self):
        res = model_confidence_set(pd.DataFrame({"only": [1.0, 2.0, 3.0]}))
        assert res["included"] == ["only"]
        assert res["mcs_pvalues"]["only"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Density / interval scoring
# ---------------------------------------------------------------------------


class TestDensityScoring:
    def test_interval_coverage_matches_nominal(self):
        rng = np.random.default_rng(8)
        y = rng.standard_normal(20000)
        z = 1.2815515594  # 80% central interval of a standard normal
        m = interval_metrics(y, np.full_like(y, -z), np.full_like(y, z),
                             coverage_level=0.80)
        assert m["coverage"] == pytest.approx(0.80, abs=0.02)
        assert m["mean_width"] == pytest.approx(2 * z, abs=1e-6)
        assert np.isfinite(m["interval_score"])

    def test_interval_score_penalises_misses(self):
        y = np.array([0.0, 0.0, 5.0])  # third point way outside
        lo = np.array([-1.0, -1.0, -1.0])
        hi = np.array([1.0, 1.0, 1.0])
        wide = interval_metrics(y, lo, hi, 0.80)
        # A miss should inflate the score above the pure width (=2.0).
        assert wide["interval_score"] > 2.0
        assert wide["coverage"] == pytest.approx(2 / 3, abs=1e-9)

    def test_crps_gaussian_closed_form(self):
        # CRPS of N(0,1) at y=0 is 2*phi(0) - 1/sqrt(pi) = 0.2336949...
        val = crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))
        assert val == pytest.approx(0.2336949, abs=1e-5)

    def test_crps_from_quantiles_matches_gaussian(self):
        from scipy import stats
        rng = np.random.default_rng(9)
        y = rng.standard_normal(2000)
        taus = np.arange(0.05, 0.96, 0.05)
        # Predictive distribution N(0,1): quantiles are the same for every row.
        q = np.tile(stats.norm.ppf(taus), (len(y), 1))
        approx = crps_from_quantiles(y, q, taus)
        exact = crps_gaussian(y, np.zeros_like(y), np.ones_like(y))
        assert approx == pytest.approx(exact, abs=0.02)

    def test_pinball_median_equals_half_mae(self):
        rng = np.random.default_rng(10)
        y = rng.standard_normal(500)
        med = np.median(y)
        pb = pinball_loss(y, np.full_like(y, med), 0.5)
        assert pb == pytest.approx(0.5 * np.mean(np.abs(y - med)), abs=1e-9)


# ---------------------------------------------------------------------------
# Sub-period breakdown
# ---------------------------------------------------------------------------


class TestSubperiodMetrics:
    def _result(self, n=500):
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        rng = np.random.default_rng(11)
        y = rng.standard_normal(n) * 0.02
        pred = 0.3 * y + rng.standard_normal(n) * 0.01
        return pd.DataFrame({"y_true": y, "y_pred": pred}, index=idx)

    def test_by_year(self):
        res = self._result()
        out = subperiod_metrics(res, horizon=1, periods_per_year=252, by="year")
        assert isinstance(out, pd.DataFrame)
        assert set(out.index) <= {2018, 2019, 2020}
        for col in ["rmse", "directional_accuracy", "cw_pvalue_vs_naive", "n"]:
            assert col in out.columns

    def test_by_regime_series(self):
        res = self._result()
        regime = pd.Series(
            np.where(np.arange(len(res)) % 2 == 0, "bull", "bear"),
            index=res.index, name="regime",
        )
        out = subperiod_metrics(res, horizon=1, by=regime)
        assert set(out.index) == {"bull", "bear"}

    def test_unknown_by_raises(self):
        with pytest.raises(ValueError):
            subperiod_metrics(self._result(), by="decade")


# ---------------------------------------------------------------------------
# compare_models wiring
# ---------------------------------------------------------------------------


class TestCompareModelsWiring:
    def test_summary_has_new_columns(self):
        rng = np.random.default_rng(12)
        n, p = 300, 6
        X = pd.DataFrame(
            rng.standard_normal((n, p)),
            columns=[f"f{i}" for i in range(p)],
            index=pd.date_range("2019-01-01", periods=n, freq="B"),
        )
        y = pd.Series(0.3 * X["f0"] + rng.standard_normal(n) * 0.5, index=X.index)
        summary, cv_results = compare_models(
            [NaiveModel(), LinearModel()], X, y,
            initial_train_size=150, step_size=22, horizon=22,
        )
        for col in ["cw_stat_vs_naive", "cw_pvalue_vs_naive",
                    "pt_stat", "pt_pvalue", "mcs_pvalue", "in_mcs"]:
            assert col in summary.columns, f"missing {col}"
        assert summary["in_mcs"].any()
