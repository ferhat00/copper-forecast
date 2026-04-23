"""
feature_engineering.py
=======================
Builds the modelling feature matrix from the raw price/macro DataFrame.

Features produced
-----------------
Price-derived
    copper_ret_{n}d        : n-day log return of copper price
    copper_vol_{n}d        : rolling n-day realised volatility (annualised)
    copper_zscore_200d     : z-score of copper price relative to 200-day MA
    rsi_14                 : 14-day Wilder RSI
    macd                   : MACD line (12-26 EMA difference)
    macd_signal            : 9-day EMA of MACD
    bb_width               : Bollinger Band width (2σ / mid)

Cross-asset
    gold_copper_ratio      : Gold/copper ratio (economic health signal)
    oil_copper_ratio       : Oil/copper ratio
    alu_copper_spread_pct  : % spread between aluminium and copper (subst. signal)
    dxy_ret_{n}d           : n-day DXY return (n = 1, 5, 22)
    sp500_ret_{n}d         : n-day S&P 500 return (n = 1, 5, 22)

Fundamental/macro
    indpro_yoy             : Year-on-year growth in industrial production
    real_yield_change_{n}d : n-day change in real 10Y yield (n = 1, 5, 22)
    infl_be_level          : Inflation breakeven level

Calendar
    month_sin / month_cos  : Cyclical encoding of calendar month
    cny_flag               : Binary flag for Chinese New Year trading week

Lagged features
    {feature}_lag_{n}      : 1, 5, 22-day lags of all numeric features

Target
    target_ret_{h}d        : h-day forward log return (forecast horizon)
    target_price_{h}d      : h-day forward copper price
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LAGS = [1, 5, 22]
DEFAULT_HORIZONS = [1, 5, 22, 66]   # 1-day, 1-week, 1-month, 3-month ahead

# Chinese New Year approximate dates (week of the holiday)
# Extend as needed; the flag covers ±3 trading days around the date
CNY_DATES = [
    "2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31",
    "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16",
    "2019-02-05", "2020-01-25", "2021-02-12", "2022-02-01",
    "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17",
]

# ---------------------------------------------------------------------------
# Technical helpers
# ---------------------------------------------------------------------------


def _wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """Return (MACD line, signal line)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _bollinger_width(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Bollinger Band width = (upper - lower) / mid."""
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return (upper - lower) / mid.replace(0, np.nan)


def _quarter_end_flag(index: pd.DatetimeIndex) -> pd.Series:
    """Flag last 5 business days of each quarter."""
    flag = pd.Series(0.0, index=index)
    # Find quarter-end business days
    for year in index.year.unique():
        for month in [3, 6, 9, 12]:
            qe = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.BMonthEnd(0)
            for delta in range(5):
                ts = qe - pd.offsets.BDay(delta)
                if ts in flag.index:
                    flag[ts] = 1.0
    return flag


def _us_holiday_flag(index: pd.DatetimeIndex) -> pd.Series:
    """Flag trading days adjacent to US federal holidays (day before + after)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    flag = pd.Series(0.0, index=index)
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=index.min(), end=index.max())
    for h in holidays:
        for delta in [-1, 0, 1]:
            ts = h + pd.offsets.BDay(delta)
            if ts in flag.index:
                flag[ts] = 1.0
    return flag


def _options_expiry_flag(index: pd.DatetimeIndex) -> pd.Series:
    """Flag 3rd Friday of each month (Comex options expiry) and the day before."""
    flag = pd.Series(0.0, index=index)
    for year in index.year.unique():
        for month in range(1, 13):
            # Find 3rd Friday: first day of month, advance to Friday, then +2 weeks
            first = pd.Timestamp(year=year, month=month, day=1)
            # Days until Friday (weekday=4)
            days_to_friday = (4 - first.weekday()) % 7
            first_friday = first + pd.Timedelta(days=days_to_friday)
            third_friday = first_friday + pd.Timedelta(weeks=2)
            for delta in [0, -1]:
                ts = third_friday + pd.offsets.BDay(delta)
                if ts in flag.index:
                    flag[ts] = 1.0
    return flag


def _cny_flag(index: pd.DatetimeIndex) -> pd.Series:
    """Binary flag: 1 during the ±3 trading-day window around CNY."""
    flag = pd.Series(0, index=index, dtype=float)
    for d in CNY_DATES:
        centre = pd.Timestamp(d)
        for delta in range(-3, 4):
            ts = centre + pd.offsets.BDay(delta)
            if ts in flag.index:
                flag[ts] = 1.0
    return flag


def _har_rv_forecast(
    log_returns: pd.Series,
    min_window: int = 252,
) -> pd.Series:
    """Compute a HAR-RV (Heterogeneous Autoregressive Realised Variance) 1-step forecast.

    The HAR model (Corsi 2009) decomposes realised variance into daily,
    weekly, and monthly components and forecasts next-day variance via OLS:

        RV_t+1 = beta_0 + beta_d * RV_d_t + beta_w * RV_w_t + beta_m * RV_m_t + eps

    Parameters
    ----------
    log_returns:
        Daily log-return series (already computed).
    min_window:
        Minimum number of rows in the rolling OLS fit (default 252 = 1 year).

    Returns
    -------
    pd.Series
        Next-day realised-variance forecast, annualised and square-rooted
        to a volatility (same scale as ``copper_vol_{n}d`` features).
        NaN for rows before the first valid OLS window.
    """
    rv_daily = log_returns ** 2  # daily RV = squared return

    rv_weekly = rv_daily.rolling(5).mean()    # 1-week avg RV
    rv_monthly = rv_daily.rolling(22).mean()  # 1-month avg RV

    forecast = pd.Series(np.nan, index=log_returns.index)
    n = len(log_returns)

    for i in range(min_window, n):
        # Regressor matrix for the training window
        slice_end = i
        slice_start = max(0, i - min_window)

        rv_d = rv_daily.iloc[slice_start:slice_end].values
        rv_w = rv_weekly.iloc[slice_start:slice_end].values
        rv_m = rv_monthly.iloc[slice_start:slice_end].values

        # Align targets: predict next-day RV
        target = rv_daily.iloc[slice_start + 1: slice_end + 1].values

        # Build design matrix; drop rows with NaN
        X_mat = np.column_stack([np.ones(len(rv_d)), rv_d, rv_w, rv_m])
        mask = np.isfinite(X_mat).all(axis=1) & np.isfinite(target)
        if mask.sum() < 30:
            continue

        X_fit = X_mat[mask]
        y_fit = target[mask]

        try:
            # OLS via least squares
            coeffs, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Predict next step using the last available values
        last_x = np.array([1.0,
                            rv_d[-1] if np.isfinite(rv_d[-1]) else 0.0,
                            rv_w.iloc[-1] if hasattr(rv_w, 'iloc') else rv_w[-1],
                            rv_m.iloc[-1] if hasattr(rv_m, 'iloc') else rv_m[-1]])

        if not np.isfinite(last_x).all():
            continue

        rv_hat = float(coeffs @ last_x)
        # Convert variance forecast to annualised volatility (same units as copper_vol)
        forecast.iloc[i] = np.sqrt(max(rv_hat, 0.0) * 252)

    return forecast


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    lags: Optional[list[int]] = None,
    horizons: Optional[list[int]] = None,
    primary_horizon: int = 22,
) -> pd.DataFrame:
    """Construct the full feature matrix from the raw DataFrame.

    Parameters
    ----------
    df:
        Output of :func:`src.data_ingestion.load_data`.
    lags:
        List of look-back periods (days) to use for lagged features.
    horizons:
        Forward horizons (days) for which to compute target variables.
    primary_horizon:
        The main forecast horizon; its target columns are not lagged.

    Returns
    -------
    pd.DataFrame
        Feature matrix with targets appended.  Rows with NaN features
        are dropped only after lag construction.
    """
    if lags is None:
        lags = DEFAULT_LAGS
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    cp = df["copper_price"]
    idx = df.index
    series: dict[str, pd.Series] = {}

    # -----------------------------------------------------------------------
    # Price-derived features
    # -----------------------------------------------------------------------
    for n in [1, 5, 22]:
        series[f"copper_ret_{n}d"] = np.log(cp / cp.shift(n))

    for n in [5, 22, 66]:
        series[f"copper_vol_{n}d"] = (
            np.log(cp / cp.shift(1)).rolling(n).std() * np.sqrt(252)
        )

    ma200 = cp.rolling(200).mean()
    std200 = cp.rolling(200).std()
    series["copper_zscore_200d"] = (cp - ma200) / std200.replace(0, np.nan)

    series["rsi_14"] = _wilder_rsi(cp)
    macd_line, signal_line = _macd(cp)
    series["macd"] = macd_line
    series["macd_signal"] = signal_line
    series["bb_width"] = _bollinger_width(cp)

    # HAR-RV 1-step-ahead volatility forecast (Corsi 2009)
    _daily_ret = np.log(cp / cp.shift(1))
    series["har_rv_forecast"] = _har_rv_forecast(_daily_ret)

    # -----------------------------------------------------------------------
    # Cross-asset features
    # -----------------------------------------------------------------------
    if "gold" in df.columns:
        series["gold_copper_ratio"] = df["gold"] / cp.replace(0, np.nan)

    if "oil" in df.columns:
        series["oil_copper_ratio"] = df["oil"] / cp.replace(0, np.nan)

    if "aluminium" in df.columns:
        series["alu_copper_spread_pct"] = (df["aluminium"] - cp) / cp.replace(0, np.nan)

    if "dxy" in df.columns:
        series["dxy_level"] = df["dxy"]
        for n in [1, 5, 22]:
            series[f"dxy_ret_{n}d"] = np.log(df["dxy"] / df["dxy"].shift(n))

    if "cny_usd" in df.columns:
        series["cny_usd_level"] = df["cny_usd"]

    if "sp500" in df.columns:
        for n in [1, 5, 22]:
            series[f"sp500_ret_{n}d"] = np.log(df["sp500"] / df["sp500"].shift(n))

    # -----------------------------------------------------------------------
    # Macro / fundamental features
    # -----------------------------------------------------------------------
    if "indpro" in df.columns:
        series["indpro_yoy"] = df["indpro"].pct_change(252)

    if "real_yield_10y" in df.columns:
        series["real_yield_level"] = df["real_yield_10y"]
        for n in [1, 5, 22]:
            series[f"real_yield_change_{n}d"] = df["real_yield_10y"].diff(n)

    if "inflation_breakeven" in df.columns:
        series["infl_be_level"] = df["inflation_breakeven"]

    # -----------------------------------------------------------------------
    # Calendar features
    # -----------------------------------------------------------------------
    month = idx.month
    series["month_sin"] = pd.Series(np.sin(2 * np.pi * month / 12), index=idx)
    series["month_cos"] = pd.Series(np.cos(2 * np.pi * month / 12), index=idx)
    series["cny_flag"] = _cny_flag(idx)
    series["quarter_end_flag"] = _quarter_end_flag(idx)
    series["us_holiday_flag"] = _us_holiday_flag(idx)
    series["options_expiry_flag"] = _options_expiry_flag(idx)

    # Build base DataFrame in one concat to avoid fragmentation
    base_df = pd.concat(series, axis=1)
    base_df.columns = list(series.keys())

    # -----------------------------------------------------------------------
    # Lagged features — build all at once via concat
    # -----------------------------------------------------------------------
    lag_frames: list[pd.DataFrame] = [base_df]
    base_cols = list(base_df.columns)
    for lag in lags:
        shifted = base_df[base_cols].shift(lag)
        shifted.columns = [f"{c}_lag_{lag}" for c in base_cols]
        lag_frames.append(shifted)

    # -----------------------------------------------------------------------
    # Target variables (forward returns / prices) — built via concat
    # -----------------------------------------------------------------------
    target_series: dict[str, pd.Series] = {}
    for h in horizons:
        target_series[f"target_ret_{h}d"] = np.log(cp.shift(-h) / cp)
        target_series[f"target_price_{h}d"] = cp.shift(-h)
    target_series["copper_price"] = cp

    target_df = pd.concat(target_series, axis=1)
    target_df.columns = list(target_series.keys())

    lag_frames.append(target_df)
    feats = pd.concat(lag_frames, axis=1)
    return feats


def split_features_targets(
    feats: pd.DataFrame,
    horizon: int = 22,
    drop_nan: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate features, target return series, and target price series.

    Parameters
    ----------
    feats:
        Output of :func:`build_features`.
    horizon:
        Forecast horizon to use as the target.
    drop_nan:
        If True, drop rows where any feature or target is NaN.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (excludes target columns and raw price).
    y_ret : pd.Series
        Forward log-return target.
    y_price : pd.Series
        Forward price target.
    """
    target_ret_col = f"target_ret_{horizon}d"
    target_price_col = f"target_price_{horizon}d"

    # Exclude all target columns and raw copper_price from features
    target_cols = [c for c in feats.columns if c.startswith("target_")] + ["copper_price"]
    feature_cols = [c for c in feats.columns if c not in target_cols]

    X = feats[feature_cols]
    y_ret = feats[target_ret_col]
    y_price = feats[target_price_col]

    if drop_nan:
        # Drop columns that are entirely NaN (e.g. a failed API fetch) so they
        # don't cause every row to be eliminated by the row-wise all() check.
        X = X.dropna(axis=1, how="all")
        mask = X.notna().all(axis=1) & y_ret.notna() & y_price.notna()
        X = X[mask]
        y_ret = y_ret[mask]
        y_price = y_price[mask]

    return X, y_ret, y_price
