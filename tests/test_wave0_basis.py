"""
tests/test_wave0_basis.py
=========================
Unit tests for the Wave 0 LME cash-3M basis ingestion and the futures-basis
benchmark model:

    fetch_lme_cash_3m       : synthetic fallback + real-CSV drop-in
    FuturesBasisBenchmark   : curve-implied return + random-walk fallback

Run with:
    pytest tests/test_wave0_basis.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import (  # noqa: E402
    _assemble_curve_basis,
    _comex_copper_symbol,
    fetch_lme_cash_3m,
)
from src.evaluation import walk_forward_cv  # noqa: E402
from src.models import FuturesBasisBenchmark, NaiveModel  # noqa: E402


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class TestFetchLmeCash3m:
    def test_synthetic_fallback_shape(self):
        df = fetch_lme_cash_3m(start="2020-01-01", end="2020-06-30", try_comex=False)
        assert list(df.columns) == ["copper_cash_3m_spread", "copper_basis_pct"]
        assert len(df) > 0
        assert df.notna().all().all()
        assert df.index.min() >= pd.Timestamp("2020-01-01")
        assert df.index.max() <= pd.Timestamp("2020-06-30")

    def test_synthetic_is_deterministic(self):
        a = fetch_lme_cash_3m(start="2021-01-01", end="2021-03-31", try_comex=False)
        b = fetch_lme_cash_3m(start="2021-01-01", end="2021-03-31", try_comex=False)
        pd.testing.assert_frame_equal(a, b)

    def test_real_csv_drop_in(self, tmp_path):
        idx = pd.date_range("2022-01-03", periods=5, freq="B")
        csv = tmp_path / "lme.csv"
        pd.DataFrame(
            {"cash": [9000, 9100, 9050, 8900, 9200],
             "three_month": [9090, 9150, 9000, 8950, 9320]},
            index=idx,
        ).to_csv(csv)
        df = fetch_lme_cash_3m(start="2022-01-01", end="2022-12-31", csv_path=str(csv))
        assert len(df) == 5
        # spread = 3M - cash; first row 9090 - 9000 = 90
        assert df["copper_cash_3m_spread"].iloc[0] == pytest.approx(90.0)
        assert df["copper_basis_pct"].iloc[0] == pytest.approx(90.0 / 9000.0)

    def test_bad_csv_falls_back_to_synthetic(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2, 3]},
                     index=pd.date_range("2022-01-03", periods=3)).to_csv(csv)
        df = fetch_lme_cash_3m(start="2022-01-01", end="2022-02-28",
                               csv_path=str(csv), try_comex=False)
        # Falls back to synthetic -> still returns the two standard columns.
        assert list(df.columns) == ["copper_cash_3m_spread", "copper_basis_pct"]
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Futures-basis benchmark
# ---------------------------------------------------------------------------


class TestFuturesBasisBenchmark:
    def test_curve_implied_return(self):
        X = pd.DataFrame({"copper_basis_pct": np.full(10, 0.03)})
        m = FuturesBasisBenchmark(horizon=22, tenor_periods=63)
        m.fit(X, pd.Series(np.zeros(10)))
        preds = m.predict(X)
        assert preds == pytest.approx(np.full(10, 0.03 * 22 / 63))

    def test_monthly_scaling(self):
        # Monthly data: 1-month horizon, 3-month tenor -> 1/3 of the basis.
        X = pd.DataFrame({"copper_basis_pct": np.full(4, 0.06)})
        m = FuturesBasisBenchmark(horizon=1, tenor_periods=3)
        preds = m.predict(X)
        assert preds == pytest.approx(np.full(4, 0.02))

    def test_random_walk_fallback_when_column_missing(self):
        X = pd.DataFrame({"some_other_feature": np.arange(5)})
        m = FuturesBasisBenchmark()
        preds = m.predict(X)
        assert np.all(preds == 0.0)

    def test_nan_basis_becomes_zero(self):
        X = pd.DataFrame({"copper_basis_pct": [0.01, np.nan, 0.02]})
        m = FuturesBasisBenchmark(horizon=63, tenor_periods=63)
        preds = m.predict(X)
        assert preds[1] == 0.0
        assert preds[0] == pytest.approx(0.01)

    def test_works_in_walk_forward_cv(self):
        n = 300
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"copper_basis_pct": rng.standard_normal(n) * 0.01}, index=idx)
        y = pd.Series(rng.standard_normal(n) * 0.02, index=idx)
        cv = walk_forward_cv(FuturesBasisBenchmark(horizon=22), X, y,
                             initial_train_size=150, step_size=22, horizon=22)
        assert len(cv) > 0
        assert np.isfinite(cv["y_pred"]).all()


# ---------------------------------------------------------------------------
# COMEX futures-curve basis assembly (network-free logic)
# ---------------------------------------------------------------------------


class TestComexBasis:
    def test_contract_symbol(self):
        assert _comex_copper_symbol(2025, 12) == "HGZ25.CMX"
        assert _comex_copper_symbol(2026, 3) == "HGH26.CMX"
        assert _comex_copper_symbol(2024, 7) == "HGN24.CMX"

    def test_assemble_contango(self):
        idx = pd.bdate_range("2024-01-02", periods=20)
        contracts = {
            (2024, 3): pd.Series(4.00, index=idx),
            (2024, 6): pd.Series(4.06, index=idx),
            (2024, 9): pd.Series(4.12, index=idx),
        }
        out = _assemble_curve_basis(contracts, tenor_months=3)
        assert list(out.columns) == ["copper_cash_3m_spread", "copper_basis_pct"]
        assert len(out) > 0
        # front=Mar, deferred>=Jun -> basis = 4.06/4.00 - 1 > 0 (contango)
        assert (out["copper_basis_pct"] > 0).all()
        assert out["copper_basis_pct"].iloc[0] == pytest.approx(4.06 / 4.00 - 1, rel=1e-6)
        # spread reported in $/tonne
        assert out["copper_cash_3m_spread"].iloc[0] == pytest.approx((4.06 - 4.00) * 2204.62, rel=1e-6)

    def test_assemble_empty(self):
        assert _assemble_curve_basis({}).empty

    def test_front_rolls_as_contracts_expire(self):
        idx = pd.bdate_range("2024-02-01", periods=70)   # spans Feb -> April
        contracts = {
            (2024, 3): pd.Series(4.00, index=idx),
            (2024, 6): pd.Series(4.06, index=idx),
            (2024, 9): pd.Series(4.12, index=idx),
            (2024, 12): pd.Series(4.18, index=idx),
        }
        out = _assemble_curve_basis(contracts, tenor_months=3)
        feb = out.loc[out.index < "2024-03-01"]
        apr = out.loc[out.index >= "2024-04-01"]
        # Feb: front=Mar, deferred=Jun
        assert feb["copper_basis_pct"].iloc[0] == pytest.approx(4.06 / 4.00 - 1, rel=1e-6)
        # April (Mar in delivery): front rolled to Jun, deferred=Sep
        assert apr["copper_basis_pct"].iloc[0] == pytest.approx(4.12 / 4.06 - 1, rel=1e-6)
