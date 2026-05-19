"""
tests/test_purged_cv.py
========================
Tests for purged walk-forward cross-validation (López de Prado, AFML Ch. 7).

Verifies that label-overlap leakage from forward-looking targets is removed
by the new ``PurgedTimeSeriesSplit`` splitter, ``walk_forward_cv``'s
``horizon`` parameter, and ``out_of_sample_backtest``'s purge.

Run with:
    pytest tests/test_purged_cv.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    PurgedTimeSeriesSplit,
    out_of_sample_backtest,
    walk_forward_cv,
)
from src.models import BaseForecaster


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_xy(n: int = 200, horizon: int = 22, seed: int = 0):
    """Build a toy dataset where one feature is a perfect future leak.

    ``y_t = log(p_{t+horizon} / p_t)`` is the forward log-return.  A feature
    column ``leak`` is set equal to ``y_t`` itself, so any CV that lets the
    training fold see rows whose labels overlap the test fold will achieve
    near-perfect in-sample fit on the leaked column.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(100 + rng.standard_normal(n).cumsum(), index=idx)
    y = np.log(price.shift(-horizon) / price).rename("y")
    X = pd.DataFrame({
        "noise": rng.standard_normal(n),
        "leak": y.values,
    }, index=idx)
    mask = y.notna() & X.notna().all(axis=1)
    return X.loc[mask], y.loc[mask]


class _LinReg(BaseForecaster):
    """Plain OLS wrapped in the BaseForecaster interface."""

    def __init__(self) -> None:
        self._m: LinearRegression | None = None

    def fit(self, X, y):
        self._m = LinearRegression().fit(X.values, y.values)
        return self

    def predict(self, X):
        assert self._m is not None
        return self._m.predict(X.values)


# ---------------------------------------------------------------------------
# PurgedTimeSeriesSplit
# ---------------------------------------------------------------------------


def test_purge_drops_last_h_minus_1_rows():
    """Every training fold must end at least ``horizon - 1`` bars before the
    next test fold begins."""
    n, horizon = 200, 22
    X = np.zeros((n, 3))
    splitter = PurgedTimeSeriesSplit(n_splits=4, horizon=horizon)
    folds = list(splitter.split(X))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert train_idx[-1] + 1 <= test_idx[0] - (horizon - 1), (
            f"Train ends at {train_idx[-1]} but test starts at {test_idx[0]}; "
            f"gap is only {test_idx[0] - train_idx[-1] - 1}, need >= {horizon - 1}"
        )


def test_purged_split_matches_tscv_when_horizon_1():
    """With horizon=1 the splitter must produce the same folds as
    ``sklearn.model_selection.TimeSeriesSplit`` (default args)."""
    n = 200
    X = np.zeros((n, 3))
    purged = list(PurgedTimeSeriesSplit(n_splits=5, horizon=1).split(X))
    tscv = list(TimeSeriesSplit(n_splits=5).split(X))
    assert len(purged) == len(tscv)
    for (p_tr, p_te), (t_tr, t_te) in zip(purged, tscv):
        np.testing.assert_array_equal(p_tr, t_tr)
        np.testing.assert_array_equal(p_te, t_te)


def test_purge_raises_when_purge_too_large():
    """Constructing a splitter that would purge the whole training fold must
    raise."""
    X = np.zeros((30, 1))
    splitter = PurgedTimeSeriesSplit(n_splits=5, horizon=10)
    with pytest.raises(ValueError, match="Purge"):
        list(splitter.split(X))


def test_get_n_splits_returns_n_splits():
    assert PurgedTimeSeriesSplit(n_splits=7, horizon=5).get_n_splits() == 7


# ---------------------------------------------------------------------------
# walk_forward_cv
# ---------------------------------------------------------------------------


def test_walk_forward_cv_purges_overlap():
    """Each training fold passed to ``model.fit`` must end at least
    ``horizon - 1`` rows before the test fold begins, regardless of how
    many CV iterations the loop performs.
    """
    horizon = 22
    n = 300
    initial = 100
    step = 20
    X, y = _toy_xy(n=n, horizon=horizon)

    fits: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    class _Capture(_LinReg):
        def fit(self, Xf, yf):
            fits.append((Xf.copy(), None))
            return super().fit(Xf, yf)

    cv = walk_forward_cv(
        _Capture(), X, y,
        initial_train_size=initial, step_size=step, horizon=horizon,
    )

    # walk_forward_cv predicted at least one fold.
    assert not cv.empty
    assert len(fits) >= 1

    # For every captured training fold, check it ended at least horizon-1
    # rows before the first test bar of the corresponding fold.
    purge = horizon - 1
    train_end = initial
    for X_train, _ in fits:
        first_test_idx = train_end  # test slice starts at train_end (pre-purge)
        last_train_idx = X.index.get_loc(X_train.index[-1])
        gap = first_test_idx - last_train_idx - 1
        assert gap >= purge, (
            f"Training fold ends at row {last_train_idx}; first test row at "
            f"{first_test_idx}; gap={gap} but expected >= {purge}"
        )
        train_end += step


def test_walk_forward_cv_default_horizon_is_backwards_compatible():
    """With the default ``horizon=1`` the function must produce the same
    output as the pre-purge implementation."""
    horizon = 5
    X, y = _toy_xy(n=200, horizon=horizon)

    # horizon=1 means purge = 0 → identical to the old behaviour.
    result_default = walk_forward_cv(
        _LinReg(), X, y, initial_train_size=80, step_size=10,
    )
    result_h1 = walk_forward_cv(
        _LinReg(), X, y, initial_train_size=80, step_size=10, horizon=1,
    )
    pd.testing.assert_frame_equal(result_default, result_h1)


# ---------------------------------------------------------------------------
# out_of_sample_backtest
# ---------------------------------------------------------------------------


def test_out_of_sample_backtest_gap():
    """The training data used in fit must end ``horizon - 1`` rows before
    the holdout split."""
    horizon = 22
    X, y = _toy_xy(n=400, horizon=horizon)
    holdout = 60

    captured: dict[str, pd.DataFrame] = {}

    class _Capture(_LinReg):
        def fit(self, Xf, yf):
            captured["X"] = Xf
            captured["y"] = yf
            return super().fit(Xf, yf)

    out_of_sample_backtest(
        _Capture(), X, y,
        holdout_size=holdout, horizon=horizon,
    )

    n = len(X)
    split = n - holdout
    expected_train_end = split - (horizon - 1)
    assert len(captured["X"]) == expected_train_end, (
        f"Expected training rows={expected_train_end}, got {len(captured['X'])}"
    )
    # And the last training timestamp must be at least horizon-1 rows before
    # the holdout start.
    last_train_ts = captured["X"].index[-1]
    first_test_ts = X.index[split]
    gap_rows = X.index.get_loc(first_test_ts) - X.index.get_loc(last_train_ts) - 1
    assert gap_rows >= horizon - 1


def test_out_of_sample_backtest_raises_when_purge_too_large():
    """If horizon - 1 >= split, the function must error rather than fitting
    on an empty training set."""
    X, y = _toy_xy(n=100, horizon=5)
    with pytest.raises(ValueError, match="Purge"):
        out_of_sample_backtest(
            _LinReg(), X, y,
            holdout_size=len(X) - 1,   # split = 1
            horizon=10,                # purge = 9 > split
        )
