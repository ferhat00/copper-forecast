"""
models_dma.py
=============
Dynamic Model Averaging (DMA) and Dynamic Model Selection (DMS) — the
best-evidenced interpretable model for *monthly* copper forecasting (Raftery,
Karny & Ettler 2010; Koop & Korobilis 2012; Buncic & Moretto 2015, who report
OOS R^2 up to ~18.5% at the 1-month horizon for copper).

Each candidate model is a time-varying-parameter (TVP) linear regression whose
coefficients evolve via a Kalman recursion with a forgetting factor ``lam``.
Across models, the model probabilities are themselves updated with a forgetting
factor ``alpha`` and the recent predictive likelihood, so the weight on each
predictor set drifts over time.  DMA forecasts with the probability-weighted
average; DMS forecasts with the single highest-probability model.

The interpretable payoff is ``inclusion_probabilities_`` — a *time series* of
how much weight each predictor carried — which answers "when did the dollar vs
inventories vs positioning drive copper?" in a way black-box ML cannot.

By default each predictor forms its own univariate TVP model (plus an optional
full model), so the model probabilities are exactly time-varying predictor
inclusion probabilities.  Pass ``model_specs`` for custom predictor subsets.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models import BaseForecaster

logger = logging.getLogger(__name__)


class DMAForecaster(BaseForecaster):
    """Dynamic Model Averaging / Selection over TVP regressions.

    Parameters
    ----------
    model_specs:
        List of predictor-column subsets (each a list of column names) defining
        the candidate models.  None -> one univariate model per column (so model
        probabilities are time-varying predictor inclusion probabilities).
    include_full:
        If True and ``model_specs`` is None, also add the all-predictors model.
    lam:
        Forgetting factor for the TVP coefficients (``0 < lam <= 1``; ~0.99
        monthly).  Lower => coefficients adapt faster.
    alpha:
        Forgetting factor for the model probabilities (``0 < alpha <= 1``;
        ~0.99).  Lower => model weights switch faster.
    kappa:
        EWMA decay for the recursive observation-variance estimate.
    prior_var:
        Prior variance of the TVP coefficients (diffuse).
    dms:
        If True, Dynamic Model *Selection* (use the single best model each step);
        otherwise Dynamic Model *Averaging*.
    max_models:
        Safety cap on the number of candidate models.
    """

    def __init__(
        self,
        model_specs: Optional[list[list[str]]] = None,
        include_full: bool = False,
        lam: float = 0.99,
        alpha: float = 0.99,
        kappa: float = 0.95,
        prior_var: float = 10.0,
        dms: bool = False,
        max_models: int = 200,
    ) -> None:
        self.model_specs = model_specs
        self.include_full = include_full
        self.lam = lam
        self.alpha = alpha
        self.kappa = kappa
        self.prior_var = prior_var
        self.dms = dms
        self.max_models = max_models
        self._scaler: Optional[StandardScaler] = None
        self._specs: list[list[int]] = []
        self._cols: list[str] = []
        self._labels: list[str] = []
        self.inclusion_probabilities_: Optional[pd.DataFrame] = None

    # -- model set --------------------------------------------------------
    def _build_specs(self, X: pd.DataFrame) -> None:
        self._cols = list(X.columns)
        idx = {c: i for i, c in enumerate(self._cols)}
        if self.model_specs is not None:
            specs, labels = [], []
            for spec in self.model_specs:
                cols = [idx[c] for c in spec if c in idx]
                if cols:
                    specs.append(cols)
                    labels.append("+".join(self._cols[i] for i in cols))
        else:
            specs = [[i] for i in range(len(self._cols))]
            labels = list(self._cols)
            if self.include_full and len(self._cols) > 1:
                specs.append(list(range(len(self._cols))))
                labels.append("FULL")
        if len(specs) > self.max_models:
            logger.warning("DMA: %d candidate models exceeds max_models=%d; truncating",
                           len(specs), self.max_models)
            specs, labels = specs[: self.max_models], labels[: self.max_models]
        self._specs, self._labels = specs, labels

    # -- the DMA recursion -----------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DMAForecaster":
        self._build_specs(X)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        yv = np.asarray(y, dtype=float)
        T = len(yv)
        K = len(self._specs)
        if K == 0:
            raise ValueError("DMAForecaster: no candidate models could be built")

        # Per-model TVP state.
        theta = [np.zeros(len(s) + 1) for s in self._specs]           # +1 intercept
        P = [np.eye(len(s) + 1) * self.prior_var for s in self._specs]
        V = [float(np.var(yv)) + 1e-6 for _ in self._specs]           # obs variance
        pi = np.full(K, 1.0 / K)                                       # model probs

        incl = np.empty((T, K))
        for t in range(T):
            xs_t = Xs[t]
            yhat = np.empty(K)
            S = np.empty(K)
            loglik = np.empty(K)
            pi_pred = pi ** self.alpha
            denom = pi_pred.sum()
            pi_pred = pi_pred / denom if denom > 0 else np.full(K, 1.0 / K)

            for k, cols in enumerate(self._specs):
                x = np.empty(len(cols) + 1)
                x[0] = 1.0
                x[1:] = xs_t[cols]
                P_pred = P[k] / self.lam
                yh = float(x @ theta[k])
                s = float(V[k] + x @ P_pred @ x)
                s = max(s, 1e-12)
                yhat[k] = yh
                S[k] = s
                e = yv[t] - yh
                loglik[k] = -0.5 * (np.log(2.0 * np.pi * s) + e * e / s)
                # Kalman update of coefficients.
                Kg = P_pred @ x / s
                theta[k] = theta[k] + Kg * e
                Pn = P_pred - np.outer(Kg, x) @ P_pred
                P[k] = 0.5 * (Pn + Pn.T)                               # keep symmetric
                V[k] = max(self.kappa * V[k] + (1.0 - self.kappa) * e * e, 1e-10)

            # Model-probability update (stable softmax on the predictive likelihood).
            w = np.log(np.maximum(pi_pred, 1e-300)) + loglik
            w -= w.max()
            pw = np.exp(w)
            pi = pw / pw.sum() if pw.sum() > 0 else np.full(K, 1.0 / K)
            incl[t] = pi

        self._theta = theta
        self._P = P
        self._V = V
        self._pi = pi
        self.inclusion_probabilities_ = pd.DataFrame(
            incl, index=X.index, columns=self._labels)
        return self

    # -- forecasting ------------------------------------------------------
    def _weights_and_components(self, X: pd.DataFrame):
        """One-step-ahead model weights and per-model mean/var for new rows."""
        Xs = self._scaler.transform(X)
        pi_pred = self._pi ** self.alpha
        denom = pi_pred.sum()
        w = pi_pred / denom if denom > 0 else np.full(len(self._pi), 1.0 / len(self._pi))
        if self.dms:                                   # selection: all weight on best
            w = np.zeros_like(w)
            w[int(np.argmax(self._pi))] = 1.0

        n = len(X)
        K = len(self._specs)
        means = np.empty((n, K))
        varis = np.empty((n, K))
        for k, cols in enumerate(self._specs):
            Xk = np.column_stack([np.ones(n), Xs[:, cols]])
            means[:, k] = Xk @ self._theta[k]
            P_pred = self._P[k] / self.lam
            varis[:, k] = self._V[k] + np.einsum("ij,jl,il->i", Xk, P_pred, Xk)
        return w, means, varis

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Model not fitted yet.")
        w, means, _ = self._weights_and_components(X)
        return np.nan_to_num(means @ w, nan=0.0)

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.80) -> pd.DataFrame:
        """Gaussian-mixture predictive interval across the candidate models."""
        from scipy import stats

        if self._scaler is None:
            raise RuntimeError("Model not fitted yet.")
        w, means, varis = self._weights_and_components(X)
        mean = means @ w
        # Mixture variance: E[var] + Var[mean] across models.
        second = ((varis + means ** 2) * w).sum(axis=1)
        var = np.maximum(second - mean ** 2, 1e-12)
        z = float(stats.norm.ppf(1.0 - (1.0 - alpha) / 2.0))
        sd = np.sqrt(var)
        return pd.DataFrame(
            {"lower": mean - z * sd, "median": mean, "upper": mean + z * sd},
            index=X.index,
        )

    # -- interpretability -------------------------------------------------
    def inclusion_probability(self, column: str) -> pd.Series:
        """Time series of total DMA weight on models containing ``column``."""
        if self.inclusion_probabilities_ is None:
            raise RuntimeError("Model not fitted yet.")
        mask = [column in lab.split("+") or lab == column for lab in self._labels]
        return self.inclusion_probabilities_.loc[:, mask].sum(axis=1)

    @property
    def name(self) -> str:
        return "DMS" if self.dms else "DMA"
