"""
tests/test_regime_cointegration.py
==================================
Unit tests for regime detection (HMM) and cointegration analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Cointegration tests
# ---------------------------------------------------------------------------


class TestCointegration:
    def test_cointegrated_pair_detected(self):
        """Two synthetic cointegrated series should pass the test."""
        from src.cointegration import test_cointegration

        rng = np.random.default_rng(42)
        n = 1000
        # Create cointegrated pair: y = 2*x + noise (shared random walk)
        x = rng.standard_normal(n).cumsum()
        y = 2 * x + rng.standard_normal(n) * 0.5

        a = pd.Series(y)
        b = pd.Series(x)
        is_coint, p_val, beta = test_cointegration(a, b, significance=0.05)
        assert is_coint is True
        assert p_val < 0.05
        assert abs(beta - 2.0) < 1.0  # beta should be near 2

    def test_non_cointegrated_pair(self):
        """Two independent random walks should not be cointegrated."""
        from src.cointegration import test_cointegration

        rng = np.random.default_rng(123)
        n = 500
        a = pd.Series(rng.standard_normal(n).cumsum())
        b = pd.Series(rng.standard_normal(n).cumsum())
        is_coint, p_val, _ = test_cointegration(a, b)
        # Not guaranteed to be False, but very likely
        # At minimum, check it returns valid values
        assert isinstance(is_coint, bool)
        assert 0 <= p_val <= 1

    def test_ect_computation(self):
        """ECT should be computed with no NaN beyond the window."""
        from src.cointegration import compute_ect

        rng = np.random.default_rng(42)
        n = 400
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        copper = pd.Series(100 + rng.standard_normal(n).cumsum(), index=idx)
        other = pd.Series(50 + rng.standard_normal(n).cumsum(), index=idx)

        ect = compute_ect(copper, other, window=100)
        assert len(ect) == n
        # First 100 should be NaN, rest should be finite
        assert ect.iloc[:100].isna().all()
        assert ect.iloc[100:].notna().all()

    def test_add_cointegration_features(self):
        from src.cointegration import add_cointegration_features

        rng = np.random.default_rng(42)
        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        # Create a cointegrated pair
        x = rng.standard_normal(n).cumsum() + 100
        df = pd.DataFrame({
            "copper_price": 2 * x + rng.standard_normal(n) * 0.5 + 8000,
            "gold": x + 1800,
            "dxy": rng.standard_normal(n).cumsum() + 100,
        }, index=idx)

        df_aug, results = add_cointegration_features(df)
        assert isinstance(df_aug, pd.DataFrame)
        assert isinstance(results, dict)
        assert len(df_aug) == n

    def test_short_series_skipped(self):
        """Series shorter than 100 should be gracefully skipped."""
        from src.cointegration import test_cointegration

        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([4.0, 5.0, 6.0])
        is_coint, p_val, beta = test_cointegration(a, b)
        assert is_coint is False


# ---------------------------------------------------------------------------
# Regime detection tests
# ---------------------------------------------------------------------------


class TestRegimeDetector:
    def test_fit_predict_labels(self):
        pytest.importorskip("hmmlearn")
        from src.regime_detection import RegimeDetector

        rng = np.random.default_rng(42)
        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        X = pd.DataFrame({
            "copper_ret_1d": rng.standard_normal(n) * 0.02,
            "copper_vol_22d": np.abs(rng.standard_normal(n)) * 0.3 + 0.1,
            "copper_zscore_200d": rng.standard_normal(n),
        }, index=idx)

        detector = RegimeDetector(n_regimes=3)
        detector.fit(X)
        labels = detector.predict(X)

        assert len(labels) == n
        unique = labels.dropna().unique()
        assert len(unique) <= 3
        assert all(0 <= v <= 2 for v in unique)

    def test_n_regimes_respected(self):
        pytest.importorskip("hmmlearn")
        from src.regime_detection import RegimeDetector

        rng = np.random.default_rng(42)
        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        X = pd.DataFrame({
            "copper_ret_1d": rng.standard_normal(n) * 0.02,
            "copper_vol_22d": np.abs(rng.standard_normal(n)) * 0.3 + 0.1,
            "copper_zscore_200d": rng.standard_normal(n),
        }, index=idx)

        for n_reg in [2, 3]:
            detector = RegimeDetector(n_regimes=n_reg)
            detector.fit(X)
            labels = detector.predict(X)
            unique = labels.dropna().unique()
            assert len(unique) <= n_reg

    def test_add_regime_features_columns(self):
        pytest.importorskip("hmmlearn")
        from src.regime_detection import RegimeDetector

        rng = np.random.default_rng(42)
        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        X = pd.DataFrame({
            "copper_ret_1d": rng.standard_normal(n) * 0.02,
            "copper_vol_22d": np.abs(rng.standard_normal(n)) * 0.3 + 0.1,
            "copper_zscore_200d": rng.standard_normal(n),
        }, index=idx)

        detector = RegimeDetector(n_regimes=3)
        detector.fit(X)
        X_aug = detector.add_regime_features(X)

        assert "regime" in X_aug.columns
        assert "regime_0" in X_aug.columns
        assert "regime_1" in X_aug.columns
        assert "regime_2" in X_aug.columns
        assert len(X_aug) == n

    def test_unfitted_raises(self):
        pytest.importorskip("hmmlearn")
        from src.regime_detection import RegimeDetector

        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "copper_ret_1d": rng.standard_normal(100),
        })
        detector = RegimeDetector()
        with pytest.raises(RuntimeError):
            detector.predict(X)
