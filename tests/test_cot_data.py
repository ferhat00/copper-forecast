"""
tests/test_cot_data.py
======================
Unit tests for COT data ingestion and alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cot_data import _synthetic_cot, align_cot_to_daily


class TestCOTData:
    def test_synthetic_cot_shape(self):
        cot = _synthetic_cot("2020-01-01", "2021-01-01")
        assert isinstance(cot, pd.DataFrame)
        assert len(cot) > 0
        for col in ["commercial_net", "noncommercial_net", "open_interest", "spec_ratio"]:
            assert col in cot.columns

    def test_align_to_daily_ffill(self):
        """Weekly COT data should be forward-filled to daily frequency."""
        # Create weekly data
        weekly_idx = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
        cot = pd.DataFrame({
            "commercial_net": range(10),
            "noncommercial_net": range(10, 20),
        }, index=weekly_idx)

        # Create daily index
        daily_idx = pd.date_range("2020-01-02", periods=60, freq="B")

        aligned = align_cot_to_daily(cot, daily_idx)
        assert len(aligned) == len(daily_idx)
        assert aligned.index.equals(daily_idx)

        # Values should be forward-filled (no big gaps)
        non_null = aligned["commercial_net"].dropna()
        assert len(non_null) > 0

    def test_align_preserves_columns(self):
        cot = _synthetic_cot("2020-01-01", "2020-06-01")
        daily_idx = pd.date_range("2020-01-02", periods=100, freq="B")
        aligned = align_cot_to_daily(cot, daily_idx)
        assert set(aligned.columns) == set(cot.columns)

    def test_synthetic_cot_dates(self):
        """Synthetic COT should generate Friday-frequency data."""
        cot = _synthetic_cot("2020-01-01", "2020-12-31")
        # All dates should be Fridays (dayofweek == 4)
        assert all(d.dayofweek == 4 for d in cot.index)
