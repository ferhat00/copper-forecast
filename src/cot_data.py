"""
cot_data.py
===========
Download and process CFTC Commitments of Traders (COT) data for copper.

The COT report is published weekly (Friday, for Tuesday reporting date).
This module fetches the data and aligns it to a daily frequency via
forward-fill for integration with the main feature pipeline.

Data source priority:
  1. Nasdaq Data Link (formerly Quandl) — ``CFTC/088691_FO_ALL``
  2. Synthetic fallback (for offline / no-API-key scenarios)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CFTC commodity code for copper (Comex)
COT_DATASET = "CFTC/088691_FO_ALL"


def fetch_cot_data(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download COT data for copper from Nasdaq Data Link.

    Parameters
    ----------
    start:
        ISO date string for the start of the window.
    end:
        ISO date string for the end.  Defaults to today.
    api_key:
        Nasdaq Data Link API key.  Falls back to environment variable
        ``NASDAQ_DATA_LINK_API_KEY``, then to synthetic data.

    Returns
    -------
    pd.DataFrame
        Weekly COT data with derived columns:
        ``commercial_net``, ``noncommercial_net``, ``open_interest``,
        ``spec_ratio``.
    """
    import os

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    key = api_key or os.environ.get("NASDAQ_DATA_LINK_API_KEY")
    df = None

    if key:
        try:
            import nasdaqdatalink
            nasdaqdatalink.ApiConfig.api_key = key
            raw = nasdaqdatalink.get(
                COT_DATASET,
                start_date=start,
                end_date=end,
            )
            df = _parse_cot_raw(raw)
            logger.info("COT: fetched %d weekly observations from Nasdaq Data Link", len(df))
        except Exception as exc:
            logger.warning("COT: Nasdaq Data Link fetch failed — %s", exc)

    if df is None:
        logger.warning(
            "COT: No API key or fetch failed. Generating synthetic COT data."
        )
        df = _synthetic_cot(start, end)

    return df


def _parse_cot_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract key positioning columns from the raw Nasdaq Data Link response.

    Column names vary across datasets; this handles the most common layout
    for the futures-only disaggregated report.
    """
    out: dict[str, pd.Series] = {}

    # Try common column names (exact names depend on dataset variant)
    long_cols = [c for c in raw.columns if "Long" in c and "Noncommercial" in c]
    short_cols = [c for c in raw.columns if "Short" in c and "Noncommercial" in c]
    comm_long = [c for c in raw.columns if "Long" in c and "Commercial" in c]
    comm_short = [c for c in raw.columns if "Short" in c and "Commercial" in c]
    oi_cols = [c for c in raw.columns if "Open Interest" in c]

    if long_cols and short_cols:
        out["noncommercial_net"] = raw[long_cols[0]] - raw[short_cols[0]]
    if comm_long and comm_short:
        out["commercial_net"] = raw[comm_long[0]] - raw[comm_short[0]]
    if oi_cols:
        out["open_interest"] = raw[oi_cols[0]]

    if not out:
        # Fallback: use first few numeric columns as proxies
        num_cols = raw.select_dtypes(include="number").columns[:4]
        if len(num_cols) >= 2:
            out["noncommercial_net"] = raw[num_cols[0]]
            out["commercial_net"] = raw[num_cols[1]]
        if len(num_cols) >= 3:
            out["open_interest"] = raw[num_cols[2]]

    df = pd.DataFrame(out, index=raw.index)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)

    if "open_interest" in df.columns and "noncommercial_net" in df.columns:
        oi = df["open_interest"].replace(0, np.nan)
        df["spec_ratio"] = df["noncommercial_net"] / oi

    return df


def _synthetic_cot(start: str, end: str) -> pd.DataFrame:
    """Generate synthetic weekly COT data for offline development."""
    idx = pd.date_range(start, end, freq="W-FRI")
    rng = np.random.default_rng(99)
    n = len(idx)
    return pd.DataFrame(
        {
            "commercial_net": rng.standard_normal(n).cumsum() * 1000 + 5000,
            "noncommercial_net": rng.standard_normal(n).cumsum() * 800 + 3000,
            "open_interest": np.abs(rng.standard_normal(n).cumsum() * 2000 + 150_000),
            "spec_ratio": rng.standard_normal(n).cumsum() * 0.01 + 0.02,
        },
        index=idx,
    )


def align_cot_to_daily(
    cot_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reindex weekly COT data to a daily business-day index via forward-fill.

    Parameters
    ----------
    cot_df:
        Weekly COT DataFrame (output of :func:`fetch_cot_data`).
    daily_index:
        The target daily date index (from yfinance data).

    Returns
    -------
    pd.DataFrame
        COT columns aligned to ``daily_index``, forward-filled.
    """
    # Reindex to daily and forward-fill (COT is released weekly)
    combined = cot_df.reindex(cot_df.index.union(daily_index)).sort_index()
    combined = combined.ffill()
    return combined.reindex(daily_index)
