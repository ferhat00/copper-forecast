"""
tests/test_vol_normalised_target.py
====================================
Tests for the optional vol-normalised return target added to
``build_features`` / ``split_features_targets``.

Key property: the normalising volatility must be *ex-ante* (trailing), so the
only forward-looking component of the target is the numerator — no look-ahead.

Run with:
    pytest tests/test_vol_normalised_target.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import build_features, split_features_targets


def _price_df(n=500, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = 8000 + rng.standard_normal(n).cumsum() * 50
    return pd.DataFrame({"copper_price": price}, index=idx)


def test_default_off_no_new_columns():
    feats = build_features(_price_df(), horizons=[1, 5, 22])
    assert not any(c.startswith("target_retvol_") for c in feats.columns)


def test_vol_normalised_columns_present():
    feats = build_features(_price_df(), horizons=[1, 5, 22],
                           vol_normalise_targets=True, vol_target_window=22)
    for h in (1, 5, 22):
        assert f"target_retvol_{h}d" in feats.columns
        assert f"target_ret_{h}d" in feats.columns      # raw target retained


def test_vol_normalised_equals_manual_ratio():
    """target_retvol == forward log-return / (trailing sigma * sqrt(h))."""
    df = _price_df()
    win, h = 22, 5
    feats = build_features(df, horizons=[h], vol_normalise_targets=True,
                           vol_target_window=win)
    cp = df["copper_price"]
    sigma1 = np.log(cp / cp.shift(1)).rolling(win).std()
    expected = np.log(cp.shift(-h) / cp) / (sigma1 * np.sqrt(h))
    got = feats[f"target_retvol_{h}d"]
    pd.testing.assert_series_equal(
        got.dropna(), expected.reindex(got.index).dropna(),
        check_names=False)


def test_denominator_is_ex_ante_no_lookahead():
    """The normaliser at row t must use only returns up to t.

    Truncating the price series at t must not change the denominator implied at
    t — i.e. future prices do not enter the scaling factor.
    """
    df = _price_df(n=300)
    win, h = 22, 5
    full = build_features(df, horizons=[h], vol_normalise_targets=True,
                          vol_target_window=win)
    # Reconstruct sigma from a truncated history up to t0 and confirm it matches.
    cp = df["copper_price"]
    t0 = 200
    sigma_full = np.log(cp / cp.shift(1)).rolling(win).std().iloc[t0]
    sigma_trunc = (np.log(cp.iloc[:t0 + 1] / cp.iloc[:t0 + 1].shift(1))
                   .rolling(win).std().iloc[-1])
    assert sigma_full == pytest.approx(sigma_trunc, rel=1e-12)
    # And the implied target uses that ex-ante sigma.
    fwd = np.log(cp.shift(-h) / cp).iloc[t0]
    assert full[f"target_retvol_{h}d"].iloc[t0] == pytest.approx(
        fwd / (sigma_full * np.sqrt(h)), rel=1e-9)


def test_split_selects_retvol_and_excludes_from_features():
    feats = build_features(_price_df(), horizons=[5],
                           vol_normalise_targets=True)
    X, y, yp = split_features_targets(feats, horizon=5, target_kind="retvol")
    assert not any(c.startswith("target_") for c in X.columns)
    assert "copper_price" not in X.columns
    assert y.name == "target_retvol_5d"


def test_split_retvol_missing_raises():
    feats = build_features(_price_df(), horizons=[5])   # not vol-normalised
    with pytest.raises(KeyError):
        split_features_targets(feats, horizon=5, target_kind="retvol")


def test_split_invalid_kind_raises():
    feats = build_features(_price_df(), horizons=[5])
    with pytest.raises(ValueError):
        split_features_targets(feats, horizon=5, target_kind="bogus")
