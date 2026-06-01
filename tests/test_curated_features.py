"""
tests/test_curated_features.py
==============================
Tests for `curate_features` — the economically-motivated feature subset used to
fight the curse of dimensionality (≈140 columns on ≈430 monthly rows) ahead of
regularised-linear models.

Run with:
    pytest tests/test_curated_features.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    CURATED_PREFIXES,
    build_features,
    curate_features,
    split_features_targets,
)


def _full_df(n=420, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    walk = lambda s: 100 + rng.standard_normal(n).cumsum() * s
    return pd.DataFrame(
        {
            "copper_price": walk(2.0),
            "dxy": walk(0.3),
            "gold": walk(8.0),
            "oil": walk(1.0),
            "aluminium": walk(3.0),
            "sp500": walk(15.0),
            "indpro": walk(0.1),
            "real_yield_10y": walk(0.05),
            "inflation_breakeven": walk(0.03),
        },
        index=idx,
    )


def _X(n=420):
    feats = build_features(
        _full_df(n), lags=[1, 3], horizons=[1, 3],
        return_lags=[1, 3], vol_windows=[3, 6], horizon_unit="m",
        yoy_periods=12, include_intraday=False,
    )
    X, _, _ = split_features_targets(feats, horizon=3, horizon_unit="m")
    return X


def test_curate_is_strict_subset():
    X = _X()
    Xc = curate_features(X)
    assert Xc.shape[1] <= X.shape[1]
    assert set(Xc.columns).issubset(set(X.columns))
    assert list(Xc.columns) == [c for c in X.columns if c in Xc.columns]  # order preserved


def test_curate_keeps_core_concepts_with_lags():
    X = _X()
    Xc = curate_features(X)
    cols = list(Xc.columns)
    # copper volatility and its lag variants survive
    assert any(c.startswith("copper_vol_") for c in cols)
    assert any("copper_vol_" in c and "_lag_" in c for c in cols)
    # real-yield and dxy concepts survive
    assert any(c.startswith("real_yield") for c in cols)
    assert any(c.startswith("dxy") for c in cols)


def test_curate_drops_uncurated_columns():
    # Inject a feature not in CURATED_PREFIXES and confirm it is dropped.
    X = _X().copy()
    X["some_noise_feature"] = 1.0
    Xc = curate_features(X)
    assert "some_noise_feature" not in Xc.columns


def test_curate_keeps_regime_columns():
    X = _X().copy()
    X["regime"] = 0.0
    X["regime_1"] = 1.0
    Xc = curate_features(X, keep_regime=True)
    assert "regime" in Xc.columns and "regime_1" in Xc.columns
    Xc2 = curate_features(X, keep_regime=False)
    assert "regime" not in Xc2.columns


def test_curate_empty_match_falls_back_to_full():
    X = _X()
    Xc = curate_features(X, prefixes=("zzz_nomatch",), keep_regime=False)
    assert Xc.shape[1] == X.shape[1]      # graceful fallback, not an empty frame


def test_curated_prefixes_nonempty():
    assert len(CURATED_PREFIXES) > 5
