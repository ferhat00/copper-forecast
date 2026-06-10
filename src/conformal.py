"""
conformal.py
============
Split-conformal prediction intervals for time-series forecasts.

Wraps any point forecaster (anything with ``.fit(X, y)`` / ``.predict(X)`` →
ndarray, i.e. the ``BaseForecaster`` contract) and turns it into a
distribution-free interval forecaster with finite-sample coverage.  Unlike
:class:`src.models.QuantileForecaster` (three LightGBM quantile fits, no coverage
guarantee, data-hungry), split conformal needs only point residuals on a held-
out calibration block and guarantees marginal coverage ≥ the requested level
under exchangeability.

Because the interval is built from *point* residuals it is agnostic to the
target's scaling — which is what makes a vol-normalised (``retvol``) headline
target usable while still producing valid price intervals.

The public surface mirrors ``QuantileForecaster`` so it is a drop-in in the
notebook interval cells:

    cf = ConformalForecaster(LinearModel(), alpha=0.80)
    cf.fit(X_train, y_train)
    cf.predict(X_test)            # -> DataFrame[lower, median, upper]

A ``mondrian_by`` column (e.g. ``"regime"``) switches on group-conditional
calibration: a separate residual quantile per group, so intervals widen in the
regimes where the model is actually less reliable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _conformal_radius(sorted_abs: np.ndarray, coverage: float) -> float:
    """Rank-based split-conformal radius for a target two-sided ``coverage``.

    Returns the ``ceil((n+1)*coverage)``-th smallest absolute residual (1-indexed,
    capped at the maximum), which guarantees marginal coverage ≥ ``coverage``.
    """
    n = sorted_abs.size
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * coverage))
    if k >= n:
        return float(sorted_abs[-1])
    return float(sorted_abs[k - 1])


def _normal_radius(resid: np.ndarray, coverage: float) -> float:
    """Gaussian fallback radius when the calibration set is too small."""
    from scipy import stats
    z = float(stats.norm.ppf(0.5 + coverage / 2.0))
    sd = float(np.std(resid)) if resid.size else 0.0
    return z * sd


class ConformalForecaster:
    """Split-conformal interval wrapper around a point forecaster.

    Parameters
    ----------
    base_model:
        Any point forecaster with ``.fit(X, y)`` and ``.predict(X) -> ndarray``.
    alpha:
        Target (two-sided) coverage, matching ``QuantileForecaster`` semantics —
        ``0.80`` ⇒ an 80% interval.
    calib_frac:
        Fraction of the (time-ordered) training rows reserved as the trailing
        calibration block.  The base model is fit on the earlier rows only.
    min_calib:
        Minimum calibration rows before falling back to a Gaussian radius.
    mondrian_by:
        Optional feature-column name to condition the radius on (group-wise
        conformal).  Groups with fewer than ``min_calib`` rows use the global
        radius.
    """

    def __init__(
        self,
        base_model,
        alpha: float = 0.80,
        calib_frac: float = 0.2,
        min_calib: int = 10,
        mondrian_by: Optional[str] = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.base_model = base_model
        self.alpha = alpha
        self.calib_frac = calib_frac
        self.min_calib = min_calib
        self.mondrian_by = mondrian_by
        self._abs_resid: Optional[np.ndarray] = None          # global, sorted
        self._group_resid: dict[object, np.ndarray] = {}      # sorted per group
        self._fallback_resid: Optional[np.ndarray] = None     # raw, for Gaussian

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConformalForecaster":
        n = len(X)
        n_calib = max(int(round(self.calib_frac * n)), self.min_calib)
        n_calib = min(n_calib, n - self.min_calib) if n > 2 * self.min_calib else max(n // 2, 1)
        split = n - n_calib

        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        X_cal, y_cal = X.iloc[split:], y.iloc[split:]

        self.base_model.fit(X_tr, y_tr)
        resid = np.asarray(y_cal, dtype=float) - np.asarray(
            self.base_model.predict(X_cal), dtype=float)
        resid = resid[np.isfinite(resid)]
        self._fallback_resid = resid
        self._abs_resid = np.sort(np.abs(resid))

        if self.mondrian_by is not None and self.mondrian_by in X_cal.columns:
            groups = X_cal[self.mondrian_by].to_numpy()
            for g in pd.unique(groups[: len(resid)]):
                mask = groups[: len(resid)] == g
                if mask.sum() >= self.min_calib:
                    self._group_resid[g] = np.sort(np.abs(resid[mask]))
        return self

    def predict_point(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.base_model.predict(X), dtype=float)

    def _radius_for(self, coverage: float, X: pd.DataFrame) -> np.ndarray:
        """Per-row interval radius (global, or group-wise if Mondrian)."""
        if self._abs_resid is None:
            raise RuntimeError("ConformalForecaster not fitted yet.")
        # Global radius (with Gaussian fallback for tiny calibration sets).
        if self._abs_resid.size < self.min_calib:
            g_rad = _normal_radius(self._fallback_resid, coverage)
        else:
            g_rad = _conformal_radius(self._abs_resid, coverage)

        radius = np.full(len(X), g_rad, dtype=float)
        if self.mondrian_by is not None and self.mondrian_by in X.columns and self._group_resid:
            col = X[self.mondrian_by].to_numpy()
            for g, arr in self._group_resid.items():
                radius[col == g] = _conformal_radius(arr, coverage)
        return radius

    def predict_interval(self, X: pd.DataFrame, alpha: Optional[float] = None) -> pd.DataFrame:
        """Return DataFrame with columns ``lower, median, upper``.

        ``alpha`` (target coverage) defaults to the value set at construction;
        pass a different value to re-derive the interval width from the stored
        calibration residuals without refitting.
        """
        coverage = self.alpha if alpha is None else alpha
        point = self.predict_point(X)
        radius = self._radius_for(coverage, X)
        return pd.DataFrame(
            {"lower": point - radius, "median": point, "upper": point + radius},
            index=X.index,
        )

    # Drop-in parity with QuantileForecaster, whose ``.predict`` returns the
    # interval DataFrame.
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.predict_interval(X)

    @property
    def name(self) -> str:
        base = getattr(self.base_model, "name", type(self.base_model).__name__)
        tag = f"Conformal[{base}]"
        return tag + (f"|mondrian:{self.mondrian_by}" if self.mondrian_by else "")
