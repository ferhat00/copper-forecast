"""
cointegration.py
================
Cointegration testing and error-correction term (ECT) feature generation.

Tests copper price against correlated assets (gold, aluminium, oil, DXY, CNY)
using the Engle-Granger two-step procedure.  For cointegrated pairs, a rolling
ECT is computed as a feature for the forecasting models.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)

# Default pairs to test: (column_name_in_df, label_for_ect_column)
DEFAULT_PAIRS = [
    ("gold", "gold"),
    ("aluminium", "aluminium"),
    ("oil", "oil"),
    ("dxy", "dxy"),
    ("cny_usd", "cny"),
]


def test_cointegration(
    series_a: pd.Series,
    series_b: pd.Series,
    significance: float = 0.05,
) -> tuple[bool, float, float]:
    """Run the Engle-Granger cointegration test on two price series.

    Parameters
    ----------
    series_a, series_b:
        Two price-level series (must be the same length, aligned index).
    significance:
        p-value threshold for declaring cointegration.

    Returns
    -------
    is_cointegrated : bool
    p_value : float
    beta : float
        OLS slope coefficient (series_a = alpha + beta * series_b + eps).
    """
    # Drop NaN rows
    mask = series_a.notna() & series_b.notna()
    a, b = series_a[mask].values, series_b[mask].values

    if len(a) < 100:
        logger.warning("Cointegration test: too few observations (%d), skipping", len(a))
        return False, 1.0, 0.0

    t_stat, p_value, _ = coint(a, b)

    # OLS to get beta: a = alpha + beta * b
    beta = float(np.polyfit(b, a, 1)[0])

    is_coint = bool(p_value < significance)
    logger.info(
        "Cointegration test: t=%.3f  p=%.4f  beta=%.4f  cointegrated=%s",
        t_stat, p_value, beta, is_coint,
    )
    return is_coint, float(p_value), beta


def compute_ect(
    copper: pd.Series,
    other: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Compute a rolling error-correction term (ECT) to avoid look-ahead.

    Uses an expanding window (minimum ``window`` observations) OLS:
        copper_t = alpha_t + beta_t * other_t + ect_t

    The ECT at time *t* is computed using only data up to *t*.

    Parameters
    ----------
    copper:
        Copper price series.
    other:
        Other asset price series.
    window:
        Minimum number of observations for the first OLS estimate.

    Returns
    -------
    pd.Series
        Error-correction term, NaN for the first ``window`` rows.
    """
    ect = pd.Series(np.nan, index=copper.index)
    mask = copper.notna() & other.notna()
    cu = copper[mask].values
    ot = other[mask].values
    valid_idx = copper[mask].index

    for i in range(window, len(cu)):
        # Expanding window OLS up to time i (inclusive)
        x = ot[:i + 1]
        y = cu[:i + 1]
        coeffs = np.polyfit(x, y, 1)  # [slope, intercept]
        predicted = coeffs[0] * ot[i] + coeffs[1]
        ect.loc[valid_idx[i]] = cu[i] - predicted

    return ect


def add_cointegration_features(
    df: pd.DataFrame,
    pairs: Optional[list[tuple[str, str]]] = None,
    significance: float = 0.05,
    window: int = 252,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Test copper against multiple assets and add ECT features for cointegrated pairs.

    Parameters
    ----------
    df:
        Raw data DataFrame with ``copper_price`` and other asset columns.
    pairs:
        List of (column_name, label) tuples to test.  Defaults to
        ``DEFAULT_PAIRS``.
    significance:
        p-value threshold for cointegration.
    window:
        Minimum window for rolling ECT computation.

    Returns
    -------
    df_aug : pd.DataFrame
        Original DataFrame augmented with ``ect_{label}`` columns for
        cointegrated pairs.
    results : dict
        Cointegration test results for each pair.
    """
    if pairs is None:
        pairs = DEFAULT_PAIRS

    results: dict[str, dict] = {}
    ect_cols: dict[str, pd.Series] = {}

    for col, label in pairs:
        if col not in df.columns:
            logger.info("Cointegration: column '%s' not in DataFrame, skipping", col)
            continue

        is_coint, p_val, beta = test_cointegration(
            df["copper_price"], df[col], significance=significance,
        )
        results[label] = {
            "column": col,
            "is_cointegrated": is_coint,
            "p_value": p_val,
            "beta": beta,
        }

        if is_coint:
            ect = compute_ect(df["copper_price"], df[col], window=window)
            ect_cols[f"ect_{label}"] = ect
            logger.info("Cointegration: added ect_%s (p=%.4f)", label, p_val)

    if ect_cols:
        ect_df = pd.DataFrame(ect_cols, index=df.index)
        df_aug = pd.concat([df, ect_df], axis=1)
    else:
        df_aug = df.copy()

    return df_aug, results
