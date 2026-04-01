"""
data_ingestion.py
=================
Downloads and harmonises raw price / macro data from:
  - yfinance  (copper futures, DXY, gold, aluminium, oil, CNY/USD)
  - FRED API  (industrial production, real yields, inflation expectations)

All series are aligned to a common daily date-index (business days),
forward-filled for non-trading gaps, and returned as a single tidy
DataFrame.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker map
# ---------------------------------------------------------------------------

YFINANCE_TICKERS: dict[str, str] = {
    # Target
    "copper_price": "HG=F",          # Comex copper front-month ($/lb → converted to $/t)
    # Macro / financial
    "dxy": "DX-Y.NYB",               # US Dollar Index
    "gold": "GC=F",                  # Gold futures ($/oz)
    "aluminium": "ALI=F",            # Aluminium futures ($/t)
    "oil": "CL=F",                   # WTI crude ($/bbl)
    "cny_usd": "CNYUSD=X",           # CNY per USD
    # Equity proxies for demand
    "sp500": "^GSPC",
    "shanghai": "000001.SS",
}

FRED_SERIES: dict[str, str] = {
    "indpro": "INDPRO",              # US Industrial Production Index
    "real_yield_10y": "DFII10",      # 10Y TIPS real yield
    "inflation_breakeven": "T10YIE", # 10Y inflation breakeven
    "ism_pmi": "MANEMP",             # Manufacturing employment as PMI proxy
    "us_m2": "M2SL",                 # M2 money supply
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def fetch_yfinance(
    tickers: Optional[dict[str, str]] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance.

    Parameters
    ----------
    tickers:
        Mapping of {column_name: yfinance_ticker}.  Defaults to
        ``YFINANCE_TICKERS``.
    start:
        ISO date string for the start of the download window.
    end:
        ISO date string for the end of the window.  Defaults to today.

    Returns
    -------
    pd.DataFrame
        Daily close prices with column names from the ``tickers`` mapping.
    """
    if tickers is None:
        tickers = YFINANCE_TICKERS
    if end is None:
        end = date.today().isoformat()

    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]

    # Rename columns
    rev = {v: k for k, v in tickers.items()}
    prices = prices.rename(columns=rev)

    # Copper is quoted in $/lb — convert to $/t (1 short ton = 2 204.62 lb)
    if "copper_price" in prices.columns:
        prices["copper_price"] = prices["copper_price"] * 2204.62

    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    logger.info("yfinance: downloaded %d rows for %d tickers", len(prices), len(tickers))
    return prices


def fetch_fred(
    series: Optional[dict[str, str]] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download macro series from FRED.

    Parameters
    ----------
    series:
        Mapping of {column_name: fred_series_id}.  Defaults to
        ``FRED_SERIES``.
    start:
        ISO date string.
    end:
        ISO date string.  Defaults to today.
    fred_api_key:
        FRED API key.  If *None* the function tries the environment variable
        ``FRED_API_KEY``; if that is also absent it falls back to synthetic
        random-walk placeholders (useful for offline testing).

    Returns
    -------
    pd.DataFrame
        Daily FRED observations, forward-filled to daily frequency.
    """
    import os

    if series is None:
        series = FRED_SERIES
    if end is None:
        end = date.today().isoformat()

    key = fred_api_key or os.environ.get("FRED_API_KEY")

    frames: dict[str, pd.Series] = {}
    if key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=key)
            for col, sid in series.items():
                try:
                    s = fred.get_series(sid, observation_start=start, observation_end=end)
                    s.name = col
                    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
                    frames[col] = s
                    logger.info("FRED: fetched %s (%d obs)", sid, len(s))
                except Exception as exc:
                    logger.warning("FRED: could not fetch %s — %s", sid, exc)
        except ImportError:
            logger.warning("fredapi not installed; FRED data will be synthetic")
            key = None

    if not key or not frames:
        logger.warning(
            "No FRED API key supplied or fredapi unavailable. "
            "Generating synthetic placeholder series."
        )
        idx = pd.date_range(start, end, freq="D")
        rng = __import__("numpy").random.default_rng(42)
        for col in series:
            frames[col] = pd.Series(
                rng.standard_normal(len(idx)).cumsum() + 100,
                index=idx,
                name=col,
            )

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def load_data(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    fred_api_key: Optional[str] = None,
    include_cot: bool = True,
    nasdaq_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch all data sources and return a single aligned DataFrame.

    Steps:
    1. Download yfinance price data.
    2. Download FRED macro data.
    3. Optionally download COT positioning data.
    4. Outer-join on date index; forward-fill up to 5 days; drop remaining NaN rows.
    5. Ensure index is sorted ascending.

    Parameters
    ----------
    start:
        Training window start date (ISO format).
    end:
        Training window end date.  Defaults to today.
    fred_api_key:
        Optional FRED API key.
    include_cot:
        If True, attempt to download COT positioning data.
    nasdaq_api_key:
        Optional Nasdaq Data Link API key for COT data.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with daily frequency and all available columns.
    """
    yf_df = fetch_yfinance(start=start, end=end)
    fred_df = fetch_fred(start=start, end=end, fred_api_key=fred_api_key)

    # Reindex FRED to the yfinance business-day calendar
    fred_daily = fred_df.reindex(yf_df.index, method="ffill")

    df = pd.concat([yf_df, fred_daily], axis=1)

    # COT positioning data
    if include_cot:
        try:
            from src.cot_data import align_cot_to_daily, fetch_cot_data
            cot = fetch_cot_data(start=start, end=end, api_key=nasdaq_api_key)
            cot_daily = align_cot_to_daily(cot, df.index)
            df = pd.concat([df, cot_daily], axis=1)
            logger.info("COT data integrated: %d columns added", cot_daily.shape[1])
        except Exception as exc:
            logger.warning("COT data unavailable: %s", exc)

    df = df.sort_index()
    df = df.ffill(limit=5)

    # Drop rows where the target is still NaN
    df = df.dropna(subset=["copper_price"])

    logger.info("Combined dataset: %d rows × %d columns", *df.shape)
    return df
