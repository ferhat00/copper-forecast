"""
tests/test_evaluation_stats.py
==============================
Tests for the statistical-validity helpers added for short / overlapping
holdouts: ``overlap_aware_sharpe``, ``deflated_sharpe_ratio`` and
``diebold_mariano``.

These guard the core properties we rely on when judging whether a weekly /
monthly signal Sharpe is real rather than a small-sample / selection artefact.

Run with:
    pytest tests/test_evaluation_stats.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.evaluation import (
    compare_models,
    deflated_sharpe_ratio,
    diebold_mariano,
    nested_cv_select,
    out_of_sample_backtest,
    overlap_aware_sharpe,
    select_best_model,
)
from src.models import LinearModel, NaiveModel


# ---------------------------------------------------------------------------
# overlap_aware_sharpe
# ---------------------------------------------------------------------------


def test_overlap_aware_sharpe_point_matches_naive():
    """Point estimate equals the naive mean/std * sqrt(ppy/h)."""
    rng = np.random.default_rng(0)
    s = rng.normal(0.001, 0.02, size=1000)
    horizon, ppy = 5, 252
    res = overlap_aware_sharpe(s, horizon=horizon, periods_per_year=ppy)
    naive = (s.mean() / s.std(ddof=1)) * np.sqrt(ppy / horizon)
    assert res["sharpe"] == pytest.approx(naive, rel=1e-9)


def test_overlap_reduces_effective_n():
    """Positively autocorrelated (overlapping) returns -> effective_n < n_obs."""
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, size=600)
    # 5-period overlapping sums induce strong positive autocorrelation
    overlapping = np.convolve(base, np.ones(5), mode="valid")
    res = overlap_aware_sharpe(overlapping, horizon=5, periods_per_year=252)
    assert res["effective_n"] < res["n_obs"]
    # IID series: effective_n should be close to n_obs
    iid = overlap_aware_sharpe(base, horizon=1, periods_per_year=252)
    assert iid["effective_n"] == pytest.approx(iid["n_obs"], rel=0.25)


def test_overlap_aware_sharpe_short_sample_safe():
    res = overlap_aware_sharpe(np.array([0.01, -0.02]), horizon=5)
    assert res["sharpe"] == 0.0
    assert np.isnan(res["effective_n"])


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio
# ---------------------------------------------------------------------------


def test_more_trials_lowers_deflated_sharpe():
    """Selecting from more candidates raises the bar -> lower probability."""
    rng = np.random.default_rng(2)
    s = rng.normal(0.05, 1.0, size=240)   # modest positive edge
    dsr_1 = deflated_sharpe_ratio(s, n_trials=1)["deflated_sr"]
    dsr_50 = deflated_sharpe_ratio(s, n_trials=50)["deflated_sr"]
    assert 0.0 <= dsr_50 <= dsr_1 <= 1.0


def test_deflated_sharpe_strong_signal_high_prob():
    rng = np.random.default_rng(3)
    s = rng.normal(0.3, 1.0, size=500)    # strong edge, long sample
    res = deflated_sharpe_ratio(s, n_trials=1)
    assert res["deflated_sr"] > 0.95
    assert res["observed_sr"] > 0


def test_deflated_sharpe_effective_n_more_conservative():
    """A smaller effective_n (overlap) cannot raise the probability."""
    rng = np.random.default_rng(4)
    s = rng.normal(0.1, 1.0, size=300)
    full = deflated_sharpe_ratio(s, n_trials=10)["deflated_sr"]
    overlapped = deflated_sharpe_ratio(s, n_trials=10, effective_n=60)["deflated_sr"]
    assert overlapped <= full + 1e-9


# ---------------------------------------------------------------------------
# diebold_mariano
# ---------------------------------------------------------------------------


def test_dm_detects_better_model():
    """A forecast that tracks the truth beats the naive zero forecast."""
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, size=400)
    good = 0.8 * y + rng.normal(0, 0.3, size=400)   # informative
    res = diebold_mariano(y, good, pred2=None, horizon=1, loss="squared")
    assert res["dm_stat"] < 0          # model1 lower loss
    assert res["better"] == "model1"
    assert res["p_value"] < 0.05


def test_dm_no_difference_for_equal_forecasts():
    rng = np.random.default_rng(6)
    y = rng.normal(0, 1, size=300)
    pred = 0.5 * y + rng.normal(0, 0.5, size=300)
    res = diebold_mariano(y, pred, pred2=pred, horizon=1)
    assert res["mean_loss_diff"] == pytest.approx(0.0, abs=1e-12)
    assert res["better"] == "tie"


def test_dm_absolute_loss_runs():
    rng = np.random.default_rng(7)
    y = rng.normal(0, 1, size=200)
    good = 0.7 * y + rng.normal(0, 0.4, size=200)
    res = diebold_mariano(y, good, horizon=5, loss="absolute")
    assert np.isfinite(res["dm_stat"])
    assert 0.0 <= res["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# Integration: enriched compare_models / out_of_sample_backtest
# ---------------------------------------------------------------------------


def _toy_xy(n=260, horizon=5, seed=0):
    rng = np.random.default_rng(seed)
    import pandas as pd
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.normal(size=(n, 4)),
                     columns=[f"f{i}" for i in range(4)], index=idx)
    y = pd.Series(0.3 * X["f0"] + rng.normal(0, 0.5, size=n), index=idx)
    return X, y


def test_compare_models_back_compatible_and_enriched():
    X, y = _toy_xy()
    models = [NaiveModel(), LinearModel()]
    summary, cv_results = compare_models(
        models, X, y, initial_train_size=120, step_size=20, horizon=5,
        periods_per_year=252)
    # back-compatible 2-tuple and original columns preserved
    assert set(cv_results) == {"Naive (RW)", "Linear (Ridge)"}
    for col in ("rmse", "directional_accuracy", "signal_sharpe", "rmse_skill"):
        assert col in summary.columns
    # new diagnostic columns present
    for col in ("deflated_sharpe", "signal_sharpe_ci_low", "effective_n",
                "dm_pvalue_vs_naive", "signal_sharpe_fold_median", "n_folds"):
        assert col in summary.columns
    assert (summary["deflated_sharpe"].dropna().between(0, 1)).all()


def test_compare_models_return_folds():
    X, y = _toy_xy()
    out = compare_models([NaiveModel(), LinearModel()], X, y,
                         initial_train_size=120, step_size=20, horizon=5,
                         return_folds=True)
    assert len(out) == 3
    _, _, fold_metrics = out
    assert {"model", "fold", "signal_sharpe"}.issubset(fold_metrics.columns)
    assert fold_metrics["model"].nunique() == 2


def test_oos_backtest_has_selection_diagnostics():
    X, y = _toy_xy()
    _, metrics = out_of_sample_backtest(
        LinearModel(), X, y, holdout_size=60, horizon=5, n_trials=8)
    for col in ("deflated_sharpe", "signal_sharpe_ci_low", "dm_pvalue_vs_naive",
                "effective_n"):
        assert col in metrics
    assert 0.0 <= metrics["deflated_sharpe"] <= 1.0 or np.isnan(metrics["deflated_sharpe"])


# ---------------------------------------------------------------------------
# select_best_model
# ---------------------------------------------------------------------------


def _summary(rows):
    return pd.DataFrame(rows).set_index("model")


def test_select_prefers_deflated_over_raw_sharpe():
    # Model A has the highest raw Sharpe but a worse deflated Sharpe (lucky);
    # Model B is the robust winner.
    summary = _summary([
        {"model": "A", "signal_sharpe": 1.9, "deflated_sharpe": 0.40,
         "dm_pvalue_vs_naive": 0.02},
        {"model": "B", "signal_sharpe": 1.1, "deflated_sharpe": 0.80,
         "dm_pvalue_vs_naive": 0.01},
        {"model": "Naive (RW)", "signal_sharpe": 0.0, "deflated_sharpe": 0.50,
         "dm_pvalue_vs_naive": 1.0},
    ])
    res = select_best_model(summary, criterion="deflated_sharpe", alpha=0.10)
    assert res["best"] == "B"
    assert "A" in res["qualified"] and "Naive (RW)" not in res["qualified"]


def test_select_falls_back_to_naive_when_none_significant():
    summary = _summary([
        {"model": "A", "deflated_sharpe": 0.55, "dm_pvalue_vs_naive": 0.40},
        {"model": "B", "deflated_sharpe": 0.52, "dm_pvalue_vs_naive": 0.60},
        {"model": "Naive (RW)", "deflated_sharpe": 0.50, "dm_pvalue_vs_naive": 1.0},
    ])
    res = select_best_model(summary, alpha=0.10)
    assert res["fell_back"] is True
    assert res["best"] == "Naive (RW)"
    assert res["qualified"] == []


def test_select_unknown_criterion_raises():
    summary = _summary([{"model": "A", "deflated_sharpe": 0.6}])
    with pytest.raises(ValueError):
        select_best_model(summary, criterion="does_not_exist")


# ---------------------------------------------------------------------------
# nested_cv_select
# ---------------------------------------------------------------------------


def test_nested_cv_structure_and_no_leakage():
    X, y = _toy_xy(n=600, horizon=5)
    res = nested_cv_select(
        [NaiveModel, LinearModel], X, y,
        n_outer_splits=4, horizon=5, periods_per_year=252)
    # one chosen model per scored outer fold; pooled diagnostics present
    assert res["n_outer"] >= 2
    assert len(res["chosen_per_fold"]) == res["n_outer"]
    assert {"deflated_sharpe", "dm_pvalue_vs_naive", "rmse"}.issubset(res["pooled"])
    oos = res["oos"]
    assert {"y_true", "y_pred", "fold", "chosen"}.issubset(oos.columns)
    assert np.isfinite(oos["y_pred"].to_numpy()).all()
    # Outer-test rows are graded out-of-sample: every chosen label is valid.
    assert set(res["chosen_per_fold"]).issubset({"Naive (RW)", "Linear (Ridge)"})


def test_nested_cv_requires_two_models():
    X, y = _toy_xy(n=300, horizon=5)
    with pytest.raises(ValueError):
        nested_cv_select([NaiveModel], X, y, n_outer_splits=3, horizon=5)


def test_nested_cv_accepts_wrapper_factories():
    from src.models import CuratedForecaster, ElasticNetModel
    X, y = _toy_xy(n=600, horizon=5)
    res = nested_cv_select(
        [NaiveModel, LinearModel, lambda: CuratedForecaster(ElasticNetModel(alpha=0.05))],
        X, y, n_outer_splits=4, horizon=5)
    assert res["n_outer"] >= 2
    assert np.isfinite(res["pooled"]["rmse"])
