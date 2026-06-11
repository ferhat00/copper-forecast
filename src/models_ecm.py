"""
models_ecm.py
=============
Single-equation error-correction (ECM) forecaster, with an optional
momentum-threshold (M-TAR) asymmetric variant.

This turns the repo's existing rolling error-correction terms (the ``ect_*``
columns from :func:`src.cointegration.add_cointegration_features`) from passive
features into an explicit, interpretable forecasting model:

    y_{t->t+h} = c + sum_k alpha_k * ECT_{k,t} + sum_j beta_j * z_{j,t} + e_t

where ``y`` is the h-step copper return (the pipeline's return target), each
``ECT_{k,t}`` is the gap between copper and its long-run equilibrium with a
cointegrated partner (gold / aluminium / oil / DXY / CNY), and ``z`` are
optional short-run regressors.  The headline interpretable object is the
adjustment speed ``alpha_k`` (a negative loading means copper mean-reverts
toward fair value; |alpha| is the fraction of disequilibrium closed per step).

The **M-TAR** variant (Enders & Siklos 2001; Goo & Chen 2020 for LME copper)
lets that adjustment be *asymmetric* — copper can revert faster when it is rich
than when it is cheap — by splitting each ECT loading on a Heaviside indicator,
with a formal test that the two speeds differ.

Requires ``statsmodels`` (already a core dependency).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class ECMForecaster(BaseForecaster):
    """Single-equation error-correction forecaster (optionally M-TAR asymmetric).

    Parameters
    ----------
    ect_prefix:
        Column-name prefix identifying error-correction terms.  All columns in
        ``X`` starting with this prefix are used unless ``ect_cols`` is given.
    ect_cols:
        Explicit list of ECT columns (overrides ``ect_prefix``).
    short_run_cols:
        Optional list of short-run regressor columns (e.g. recent returns of
        DXY / basis).  None -> ECT terms only.
    asymmetric:
        If True, fit a threshold (M-TAR / TAR) model that splits each ECT
        loading into a "high" and "low" regime — an asymmetric adjustment speed.
    mode:
        ``"mtar"`` (default) splits on the *momentum* of the ECT
        (``ECT.diff() >= threshold``; Enders-Siklos M-TAR); ``"tar"`` splits on
        the ECT *level* (``ECT >= threshold``).
    threshold:
        Threshold value for the indicator (default 0.0).
    """

    def __init__(
        self,
        ect_prefix: str = "ect_",
        ect_cols: Optional[list[str]] = None,
        short_run_cols: Optional[list[str]] = None,
        asymmetric: bool = False,
        mode: str = "mtar",
        threshold: float = 0.0,
    ) -> None:
        if mode not in {"mtar", "tar"}:
            raise ValueError(f"mode must be 'mtar' or 'tar', got {mode!r}")
        self.ect_prefix = ect_prefix
        self.ect_cols = ect_cols
        self.short_run_cols = short_run_cols
        self.asymmetric = asymmetric
        self.mode = mode
        self.threshold = threshold
        self._res = None
        self._ect_cols: list[str] = []
        self._design_cols: list[str] = []
        self._intercept_only = False

    # -- design matrix ----------------------------------------------------
    def _resolve_cols(self, X: pd.DataFrame) -> None:
        if self.ect_cols is not None:
            self._ect_cols = [c for c in self.ect_cols if c in X.columns]
        else:
            self._ect_cols = [c for c in X.columns if c.startswith(self.ect_prefix)]
        self._short_run = [c for c in (self.short_run_cols or []) if c in X.columns]

    def _design(self, X: pd.DataFrame, fill: bool) -> pd.DataFrame:
        """Build the regressor frame (ECT loadings + short-run terms)."""
        parts: dict[str, pd.Series] = {}
        for e in self._ect_cols:
            col = X[e]
            if self.asymmetric:
                if self.mode == "mtar":
                    ind = col.diff().fillna(0.0) >= self.threshold
                else:  # tar
                    ind = col >= self.threshold
                parts[f"{e}__hi"] = col.where(ind, 0.0)
                parts[f"{e}__lo"] = col.where(~ind, 0.0)
            else:
                parts[e] = col
        for z in self._short_run:
            parts[z] = X[z]
        design = pd.DataFrame(parts, index=X.index)
        if fill:
            design = design.fillna(0.0)
        return design

    # -- fit / predict ----------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ECMForecaster":
        import statsmodels.api as sm

        self._resolve_cols(X)
        design = self._design(X, fill=False)
        self._design_cols = list(design.columns)

        if not self._design_cols:
            logger.warning("ECMForecaster: no ECT/short-run columns found — "
                           "falling back to an intercept-only (mean) model.")
            self._intercept_only = True
            self._mean = float(np.asarray(y, dtype=float).mean())
            return self

        frame = design.copy()
        frame["__y__"] = np.asarray(y, dtype=float)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        yv = frame["__y__"].to_numpy()
        Xv = frame[self._design_cols].to_numpy()
        self._res = sm.OLS(yv, sm.add_constant(Xv, has_constant="add")).fit()
        self._intercept_only = False
        return self

    def _exog(self, X: pd.DataFrame):
        import statsmodels.api as sm
        design = self._design(X, fill=True)[self._design_cols]
        return sm.add_constant(design.to_numpy(), has_constant="add")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._intercept_only:
            return np.full(len(X), getattr(self, "_mean", 0.0))
        if self._res is None:
            raise RuntimeError("Model not fitted yet.")
        return np.nan_to_num(self._res.predict(self._exog(X)), nan=0.0)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.80) -> pd.DataFrame:
        """Gaussian prediction interval from the OLS predictive distribution."""
        if self._intercept_only or self._res is None:
            mid = self.predict(X)
            return pd.DataFrame({"lower": mid, "median": mid, "upper": mid}, index=X.index)
        pred = self._res.get_prediction(self._exog(X))
        sf = pred.summary_frame(alpha=1.0 - alpha)
        return pd.DataFrame(
            {"lower": sf["obs_ci_lower"].to_numpy(),
             "median": sf["mean"].to_numpy(),
             "upper": sf["obs_ci_upper"].to_numpy()},
            index=X.index,
        )

    # -- interpretability -------------------------------------------------
    @property
    def params_(self) -> pd.Series:
        """Fitted coefficients (``const`` + design columns) — the ECT loadings."""
        if self._res is None:
            raise RuntimeError("Model not fitted (or intercept-only).")
        return pd.Series(self._res.params, index=["const"] + self._design_cols)

    def asymmetry_test(self) -> dict[str, float]:
        """Wald F-test that high/low ECT adjustment speeds are equal (M-TAR).

        H0: ``alpha_hi == alpha_lo`` for every ECT.  A small p-value is evidence
        of genuinely asymmetric mean-reversion (the economic point of M-TAR).
        Returns NaNs when the model is symmetric or intercept-only.
        """
        out = {"f_stat": float("nan"), "p_value": float("nan"), "df_num": float("nan")}
        if not self.asymmetric or self._res is None:
            return out
        names = ["const"] + self._design_cols
        rows = []
        for e in self._ect_cols:
            hi, lo = f"{e}__hi", f"{e}__lo"
            if hi in names and lo in names:
                r = np.zeros(len(names))
                r[names.index(hi)] = 1.0
                r[names.index(lo)] = -1.0
                rows.append(r)
        if not rows:
            return out
        ftest = self._res.f_test(np.vstack(rows))
        out.update(f_stat=float(np.ravel(ftest.fvalue)[0]),
                   p_value=float(ftest.pvalue),
                   df_num=float(len(rows)))
        return out

    @property
    def name(self) -> str:
        if self.asymmetric:
            return f"ECM ({self.mode.upper()} asym)"
        return "ECM"
