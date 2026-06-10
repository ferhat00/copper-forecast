"""
tests/test_conformal.py
=======================
Tests for split-conformal prediction intervals (`ConformalForecaster`).

Core properties: marginal coverage ~ nominal, ordered bounds, drop-in column
parity with `QuantileForecaster`, monotone width in the coverage level, and a
graceful Gaussian fallback when calibration is tiny.

Run with:
    pytest tests/test_conformal.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conformal import ConformalForecaster
from src.models import LinearModel


def _xy(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.normal(size=(n, 3)),
                     columns=["a", "b", "c"], index=idx)
    y = pd.Series(0.7 * X["a"] - 0.3 * X["b"] + rng.normal(0, 0.5, n), index=idx)
    return X, y


def test_predict_columns_and_order():
    X, y = _xy()
    cf = ConformalForecaster(LinearModel(), alpha=0.80)
    cf.fit(X.iloc[:600], y.iloc[:600])
    out = cf.predict(X.iloc[600:])
    assert list(out.columns) == ["lower", "median", "upper"]
    assert (out["lower"] <= out["median"]).all()
    assert (out["median"] <= out["upper"]).all()
    assert out.index.equals(X.iloc[600:].index)


def test_marginal_coverage_near_nominal():
    X, y = _xy(n=1200, seed=1)
    cf = ConformalForecaster(LinearModel(), alpha=0.80, calib_frac=0.3)
    cf.fit(X.iloc[:800], y.iloc[:800])
    out = cf.predict(X.iloc[800:])
    yt = y.iloc[800:]
    cov = ((yt >= out["lower"]) & (yt <= out["upper"])).mean()
    # Split conformal guarantees >= nominal in expectation; allow sampling slack.
    assert 0.72 <= cov <= 0.92


def test_higher_coverage_is_wider():
    X, y = _xy()
    cf = ConformalForecaster(LinearModel(), alpha=0.80)
    cf.fit(X.iloc[:600], y.iloc[:600])
    Xt = X.iloc[600:]
    w80 = (cf.predict_interval(Xt, alpha=0.80)["upper"]
           - cf.predict_interval(Xt, alpha=0.80)["lower"]).mean()
    w95 = (cf.predict_interval(Xt, alpha=0.95)["upper"]
           - cf.predict_interval(Xt, alpha=0.95)["lower"]).mean()
    assert w95 >= w80


def test_predict_point_matches_base():
    X, y = _xy()
    cf = ConformalForecaster(LinearModel(), alpha=0.80)
    cf.fit(X.iloc[:600], y.iloc[:600])
    pt = cf.predict_point(X.iloc[600:])
    med = cf.predict(X.iloc[600:])["median"].to_numpy()
    np.testing.assert_allclose(pt, med)


def test_tiny_calibration_falls_back_without_error():
    X, y = _xy(n=40)
    cf = ConformalForecaster(LinearModel(), alpha=0.80, min_calib=5)
    cf.fit(X.iloc[:30], y.iloc[:30])
    out = cf.predict(X.iloc[30:])
    assert np.isfinite(out.to_numpy()).all()
    assert (out["upper"] >= out["lower"]).all()


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        ConformalForecaster(LinearModel(), alpha=1.5)


def test_mondrian_group_conditional_runs():
    X, y = _xy(n=1000, seed=2)
    # Inject a regime column with heteroskedastic noise per regime.
    rng = np.random.default_rng(3)
    regime = pd.Series(rng.integers(0, 2, len(X)), index=X.index, name="regime")
    y = y + regime * rng.normal(0, 1.5, len(X))   # regime 1 noisier
    X = X.assign(regime=regime.astype(float))
    cf = ConformalForecaster(LinearModel(), alpha=0.80, mondrian_by="regime",
                             calib_frac=0.4)
    cf.fit(X.iloc[:700], y.iloc[:700])
    out = cf.predict(X.iloc[700:])
    assert list(out.columns) == ["lower", "median", "upper"]
    # Mondrian stored per-group residual sets
    assert len(cf._group_resid) >= 1
    assert "mondrian" in cf.name


def test_name_includes_base():
    cf = ConformalForecaster(LinearModel(), alpha=0.80)
    assert "Conformal" in cf.name and "Ridge" in cf.name
