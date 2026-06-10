"""
evaluation.py
=============
Walk-forward cross-validation and forecast evaluation metrics.

Functions
---------
walk_forward_cv      : Expanding-window CV with label-overlap purging
directional_accuracy : % of predictions with correct sign of change
compute_metrics      : RMSE, MAE, MAPE, DA for a prediction series
compare_models       : Run multiple forecasters through walk-forward CV
out_of_sample_backtest : Hold-out backtest with label-overlap purging
overlap_aware_sharpe : Signal Sharpe with HAC std-error / effective-N for
                       overlapping h-step returns
deflated_sharpe_ratio : Probabilistic / Deflated Sharpe (Bailey & López de
                       Prado 2014) — discounts a Sharpe for selection bias
diebold_mariano      : Equal-predictive-accuracy test vs a benchmark forecast
select_best_model    : Robust, selection-bias-aware model picker (replaces
                       argmax of the pooled signal Sharpe)
nested_cv_select     : Nested CV that grades the *selection procedure* on
                       untouched outer folds (kills single-holdout optimism)

Classes
-------
PurgedTimeSeriesSplit : sklearn-compatible time-series splitter that drops
                        the trailing ``horizon - 1`` rows from each training
                        fold (López de Prado, *Advances in Financial Machine
                        Learning*, Ch. 7).
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
    horizon: int = 22,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE, directional accuracy, and signal Sharpe.

    Parameters
    ----------
    y_true:
        Observed values.
    y_pred:
        Predicted values (same length as ``y_true``).
    name:
        Optional label for logging.
    horizon:
        Forecast horizon expressed in the same unit as ``periods_per_year``.
        For daily data: trading days (e.g. 22 = monthly horizon, 252 days/yr).
        For monthly data: months (e.g. 3 = quarterly horizon, 12 months/yr).
    periods_per_year:
        Number of bars per year — 252 for daily (default), 12 for monthly.

    Returns
    -------
    dict with keys: rmse, mae, mape, directional_accuracy, signal_sharpe,
    information_ratio, rmse_skill
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # RMSE skill score vs the zero-prediction (random-walk on log-returns) benchmark.
    # Naive predicts 0, so its squared error is mean(y_true**2).  A skill of 0 means
    # "as good as Naive on RMSE"; > 0 beats Naive; < 0 loses to Naive.
    naive_mse = float(np.mean(y_true ** 2))
    rmse_skill = float(1.0 - (rmse ** 2) / naive_mse) if naive_mse > 0 else 0.0

    # MAPE — guard against zero actuals
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    da = directional_accuracy(y_true, y_pred)

    # Signal Sharpe: annualised Sharpe of a long/short strategy based on
    # predicted direction.  signal_returns[i] = sign(pred[i]) * actual[i]
    signal_returns = np.sign(y_pred) * y_true
    annualise = np.sqrt(periods_per_year / max(horizon, 1))
    if len(signal_returns) > 1 and np.std(signal_returns) > 0:
        signal_sharpe = float(np.mean(signal_returns) / np.std(signal_returns) * annualise)
    else:
        signal_sharpe = 0.0

    # Information ratio (vs naive/zero benchmark — identical to signal Sharpe
    # because the naive forecast is 0 and the excess return equals the signal return)
    information_ratio = signal_sharpe

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": da,
        "signal_sharpe": signal_sharpe,
        "information_ratio": information_ratio,
        "rmse_skill": rmse_skill,
    }
    if name:
        logger.info("[%s] RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  DA=%.2f%%  Sharpe=%.2f  Skill=%.4f",
                    name, rmse, mae, mape, da * 100, signal_sharpe, rmse_skill)
    return metrics


def directional_accuracy(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Fraction of predictions with the correct sign (directional accuracy).

    For return targets, a positive true return paired with a positive
    predicted return counts as correct.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


# ---------------------------------------------------------------------------
# Statistical-validity helpers for short / overlapping holdouts
# ---------------------------------------------------------------------------
#
# These exist because the headline ``signal_sharpe`` in :func:`compute_metrics`
# annualises with ``sqrt(periods_per_year / horizon)``, which is only exact for
# *non-overlapping* h-step returns.  In the walk-forward backtests the signal
# returns are sampled every ``step_size`` rows but each label looks ``horizon``
# rows forward, so consecutive observations overlap.  That deflates the naive
# standard error (inflating Sharpe / its apparent significance) and — once the
# "best" model is picked by max Sharpe across many candidates — adds selection
# bias on top.  The helpers below quantify both effects without changing any
# existing metric.


def _newey_west_long_run_var(x: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett-kernel) long-run variance of a 1-D series.

    Estimates ``gamma_0 + 2 * sum_{k=1..lag} w_k * gamma_k`` with Bartlett
    weights ``w_k = 1 - k / (lag + 1)``.  Used to get HAC standard errors for
    the mean of overlapping (autocorrelated) signal returns.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan")
    xc = x - x.mean()
    gamma0 = float(np.dot(xc, xc) / n)
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = float(np.dot(xc[k:], xc[:-k]) / n)
        lrv += 2.0 * w * gamma_k
    return max(lrv, 0.0)


def overlap_aware_sharpe(
    signal_returns: np.ndarray | pd.Series,
    horizon: int = 22,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Annualised signal Sharpe with overlap-aware (HAC) inference.

    The point estimate matches :func:`compute_metrics`' ``signal_sharpe``
    (``mean/std * sqrt(periods_per_year / horizon)``).  What this adds is an
    honest standard error and an *effective* sample size that account for the
    autocorrelation induced by overlapping h-step labels, plus a t-statistic
    and 95% CI for the **annualised** Sharpe.

    Parameters
    ----------
    signal_returns:
        Per-observation strategy returns ``sign(pred) * y_true`` at the data
        sampling frequency (one per backtest row).
    horizon:
        Label horizon in rows — the overlap span (Newey-West lag = horizon-1).
    periods_per_year:
        Bars per year (252 daily, 12 monthly).

    Returns
    -------
    dict with keys: ``sharpe`` (annualised, = naive estimate), ``sharpe_se``,
    ``sharpe_tstat``, ``sharpe_ci_low``/``_high`` (95%, annualised),
    ``n_obs``, ``effective_n`` (overlap-adjusted), ``p_value`` (one-sided
    H0: Sharpe <= 0).
    """
    from scipy import stats

    s = np.asarray(signal_returns, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    out = {
        "sharpe": 0.0, "sharpe_se": float("nan"), "sharpe_tstat": float("nan"),
        "sharpe_ci_low": float("nan"), "sharpe_ci_high": float("nan"),
        "n_obs": float(n), "effective_n": float("nan"), "p_value": float("nan"),
    }
    if n < 3:
        return out

    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    if sd == 0:
        return out

    annualise = float(np.sqrt(periods_per_year / max(horizon, 1)))
    sr_per = mu / sd                       # per-period Sharpe
    sr_ann = sr_per * annualise

    # HAC long-run variance of the per-period returns -> effective sample size.
    lag = max(int(horizon) - 1, 0)
    iid_var = float(np.var(s, ddof=0))
    lrv = _newey_west_long_run_var(s, lag)
    eff_n = float(n * iid_var / lrv) if lrv > 0 else float(n)
    eff_n = min(max(eff_n, 1.0), float(n))

    # SE of a Sharpe estimate ~ sqrt((1 + 0.5*SR^2) / N_eff), annualised.
    se_per = float(np.sqrt((1.0 + 0.5 * sr_per ** 2) / eff_n))
    se_ann = se_per * annualise
    tstat = sr_ann / se_ann if se_ann > 0 else float("nan")
    # one-sided p-value (H0: Sharpe <= 0) using t with eff_n-1 dof
    p_value = float(stats.t.sf(tstat, df=max(eff_n - 1.0, 1.0))) if np.isfinite(tstat) else float("nan")
    z = 1.959963984540054
    out.update(
        sharpe=sr_ann, sharpe_se=se_ann, sharpe_tstat=tstat,
        sharpe_ci_low=sr_ann - z * se_ann, sharpe_ci_high=sr_ann + z * se_ann,
        effective_n=eff_n, p_value=p_value,
    )
    return out


def deflated_sharpe_ratio(
    signal_returns: np.ndarray | pd.Series,
    n_trials: int = 1,
    sr_trials_std: Optional[float] = None,
    sr_benchmark: float = 0.0,
    horizon: int = 1,
    effective_n: Optional[float] = None,
) -> dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Returns the probability that the *true* (per-period) Sharpe exceeds a
    benchmark, after discounting for (a) the number of strategy
    configurations tried (selection bias), (b) non-normality of returns, and
    (c) sample length.  A value near 1.0 is strong evidence of real skill; a
    value below ~0.95 means the headline Sharpe is not distinguishable from
    the best-of-N-lucky-draws null.

    Parameters
    ----------
    signal_returns:
        Per-observation strategy returns ``sign(pred) * y_true``.
    n_trials:
        Number of independent model/parameter configurations from which the
        reported one was selected (e.g. number of rows in ``compare_models``).
    sr_trials_std:
        Std-dev of the **per-period** Sharpe ratios across those trials.  If
        None, a null sampling-variance proxy ``(1 + 0.5*SR^2)/T`` is used.
        Pass the real cross-sectional std (preferred) when available.
    sr_benchmark:
        Benchmark per-period Sharpe under H0 (default 0).
    horizon:
        Label horizon in rows; only used when ``effective_n`` is None to leave
        ``T`` at its raw value (overlap is handled via ``effective_n``).
    effective_n:
        Overlap-adjusted sample size (e.g. from :func:`overlap_aware_sharpe`).
        When provided it replaces the raw observation count in the variance
        term, giving a more conservative (honest) probability.

    Returns
    -------
    dict with keys: ``observed_sr`` (per-period), ``sr0`` (expected-max under
    null), ``deflated_sr`` (probability in [0, 1]), ``n_trials``, ``n_obs``.
    """
    from scipy import stats

    s = np.asarray(signal_returns, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    out = {
        "observed_sr": 0.0, "sr0": float("nan"), "deflated_sr": float("nan"),
        "n_trials": float(n_trials), "n_obs": float(n),
    }
    if n < 3:
        return out

    sd = float(s.std(ddof=1))
    if sd == 0:
        return out
    sr = float(s.mean()) / sd                      # observed per-period Sharpe
    skew = float(stats.skew(s, bias=False))
    kurt = float(stats.kurtosis(s, fisher=False, bias=False))  # non-excess
    t_eff = float(effective_n) if effective_n is not None else float(n)
    t_eff = max(t_eff, 2.0)

    # Expected maximum Sharpe under the null across n_trials (selection bias).
    if sr_trials_std is not None and sr_trials_std > 0:
        v_sr = float(sr_trials_std) ** 2
    else:
        v_sr = (1.0 + 0.5 * sr ** 2) / t_eff       # null sampling-variance proxy
    if n_trials > 1 and v_sr > 0:
        gamma_e = 0.5772156649015329               # Euler-Mascheroni
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr0 = sr_benchmark + np.sqrt(v_sr) * ((1.0 - gamma_e) * z1 + gamma_e * z2)
    else:
        sr0 = sr_benchmark

    denom = np.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2, 1e-12))
    dsr_stat = (sr - sr0) * np.sqrt(max(t_eff - 1.0, 1.0)) / denom
    out.update(observed_sr=sr, sr0=float(sr0), deflated_sr=float(stats.norm.cdf(dsr_stat)))
    return out


def diebold_mariano(
    y_true: np.ndarray | pd.Series,
    pred1: np.ndarray | pd.Series,
    pred2: Optional[np.ndarray | pd.Series] = None,
    horizon: int = 1,
    loss: str = "squared",
) -> dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy (with HLN correction).

    Tests H0: model 1 and model 2 have equal expected loss.  ``pred2`` defaults
    to the random-walk / naive forecast (all zeros, appropriate for log-return
    targets), so the common use is "is model 1 significantly better than
    naive?".  Uses a Newey-West HAC variance with lag ``horizon - 1`` (for
    overlapping multi-step forecasts) and the Harvey-Leybourne-Newbold (1997)
    small-sample correction.

    Parameters
    ----------
    y_true, pred1:
        Observations and model-1 forecasts.
    pred2:
        Benchmark forecasts.  None -> zeros (naive random-walk on returns).
    horizon:
        Forecast horizon in rows (sets the HAC lag).
    loss:
        ``"squared"`` (default) or ``"absolute"``.

    Returns
    -------
    dict with keys: ``dm_stat`` (negative -> model 1 more accurate),
    ``p_value`` (two-sided), ``mean_loss_diff``, ``n_obs``, ``better``
    (``"model1"``/``"model2"``/``"tie"``).
    """
    from scipy import stats

    y = np.asarray(y_true, dtype=float)
    p1 = np.asarray(pred1, dtype=float)
    p2 = np.zeros_like(y) if pred2 is None else np.asarray(pred2, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p1) & np.isfinite(p2)
    y, p1, p2 = y[mask], p1[mask], p2[mask]
    n = y.size
    out = {
        "dm_stat": float("nan"), "p_value": float("nan"),
        "mean_loss_diff": float("nan"), "n_obs": float(n), "better": "tie",
    }
    if n < 3:
        return out

    e1, e2 = y - p1, y - p2
    if loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1 ** 2 - e2 ** 2
    dbar = float(d.mean())
    out["mean_loss_diff"] = dbar
    lag = max(int(horizon) - 1, 0)
    lrv = _newey_west_long_run_var(d, lag)         # = n * Var(dbar)
    if lrv <= 0:
        # Degenerate: identical (or constant-differential) forecasts. Treat an
        # all-equal differential as a tie rather than returning NaN.
        if dbar == 0.0:
            out.update(dm_stat=0.0, p_value=1.0, better="tie")
        return out
    var_dbar = lrv / n
    dm = dbar / np.sqrt(var_dbar)
    # Harvey-Leybourne-Newbold small-sample correction.
    h = int(horizon)
    hln = np.sqrt(max((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n, 1e-12))
    dm_corr = dm * hln
    p_value = float(2.0 * stats.t.sf(abs(dm_corr), df=max(n - 1, 1)))
    better = "model1" if dbar < 0 else ("model2" if dbar > 0 else "tie")
    out.update(dm_stat=float(dm_corr), p_value=p_value,
               mean_loss_diff=dbar, better=better)
    return out


# ---------------------------------------------------------------------------
# Purged time-series splitter (López de Prado, AFML Ch. 7)
# ---------------------------------------------------------------------------


class PurgedTimeSeriesSplit:
    """Sklearn-compatible time-series splitter with label-overlap purging.

    Mirrors :class:`sklearn.model_selection.TimeSeriesSplit` fold geometry,
    but drops the trailing ``horizon - 1 + gap`` rows from each training
    fold so labels constructed as ``y_t = f(prices[t..t+horizon])`` cannot
    overlap the test fold.

    Per López de Prado (*Advances in Financial Machine Learning*, §7.4),
    an embargo on the right edge of the test fold is unnecessary when
    training always precedes testing (the walk-forward / TimeSeriesSplit
    layout), so ``gap`` defaults to 0.

    Parameters
    ----------
    n_splits:
        Number of folds (matches ``TimeSeriesSplit`` semantics).
    horizon:
        Label horizon in rows.  ``horizon - 1`` rows are purged from the
        tail of each training fold.  Use 1 to disable purging.
    gap:
        Extra rows to purge beyond ``horizon - 1`` (right-side embargo).
    """

    def __init__(self, n_splits: int = 5, horizon: int = 1, gap: int = 0) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        self.n_splits = n_splits
        self.horizon = horizon
        self.gap = gap

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        # Same fold geometry as sklearn's TimeSeriesSplit (default test_size).
        # With n_splits=k, the test folds occupy the last k * test_size rows.
        test_size = n // (self.n_splits + 1)
        if test_size < 1:
            raise ValueError(
                f"Too few samples ({n}) for n_splits={self.n_splits}"
            )

        indices = np.arange(n)
        purge = max(self.horizon - 1, 0) + self.gap

        for i in range(self.n_splits):
            test_start = n - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            train_end_purged = test_start - purge
            if train_end_purged <= 0:
                raise ValueError(
                    f"Purge ({purge}) consumed entire training fold "
                    f"(test_start={test_start}); reduce horizon, gap, or n_splits"
                )
            yield indices[:train_end_purged], indices[test_start:test_end]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------


def walk_forward_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_size: int = 504,   # ~2 years of daily data
    step_size: int = 22,             # re-fit monthly
    refit: bool = True,
    rolling_window: Optional[int] = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Walk-forward cross-validation with optional rolling window and purging.

    Parameters
    ----------
    model:
        An unfitted forecaster implementing ``.fit()`` and ``.predict()``.
    X:
        Feature matrix (time-ordered, no future leakage).
    y:
        Target series aligned with ``X``.
    initial_train_size:
        Number of rows in the first training fold.
    step_size:
        Number of new rows added to the training window each fold.
    refit:
        If True, re-fit the model on each expanded training set.
    rolling_window:
        If set, use a rolling training window of this many rows instead
        of expanding from the start.  When None (default), the classic
        expanding-window scheme is used (backward-compatible).
    horizon:
        Label horizon in rows.  The last ``horizon - 1`` rows of each
        training fold are purged so their forward-looking labels do not
        overlap the test fold (López de Prado, AFML Ch. 7).  Default 1
        disables purging — pass the active forecast horizon to enable it.

    Returns
    -------
    pd.DataFrame
        Columns: ``y_true``, ``y_pred``, ``fold``; index matches ``X``.
    """
    n = len(X)
    if n <= initial_train_size:
        raise ValueError(
            f"Dataset too small ({n} rows) for initial_train_size={initial_train_size}"
        )

    purge = max(horizon - 1, 0)
    records = []
    fold = 0
    train_end = initial_train_size

    while train_end < n:
        test_end = min(train_end + step_size, n)

        # Purge: drop the last (horizon - 1) rows of the training fold so
        # their forward-looking labels don't overlap the test fold.
        train_end_purged = train_end - purge
        if train_end_purged <= 0:
            raise ValueError(
                f"Purge ({purge}) consumed entire training fold at "
                f"train_end={train_end}; reduce horizon or grow initial_train_size"
            )

        X_train = X.iloc[:train_end_purged]
        y_train = y.iloc[:train_end_purged]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        # Rolling-window slice: only keep the most recent `rolling_window` rows
        if rolling_window is not None:
            X_train = X_train.iloc[-rolling_window:]
            y_train = y_train.iloc[-rolling_window:]

        if refit or fold == 0:
            model.fit(X_train, y_train)

        preds = model.predict(X_test)

        for i, (idx, yt, yp) in enumerate(
            zip(X_test.index, y_test.values, preds, strict=False)
        ):
            records.append({"date": idx, "y_true": yt, "y_pred": yp, "fold": fold})

        logger.debug(
            "Fold %d | train=%d (purged=%d) test=%d-%d",
            fold, train_end, train_end_purged, train_end, test_end,
        )

        train_end += step_size
        fold += 1

    result = pd.DataFrame(records).set_index("date")
    return result


def _per_fold_metrics(
    cv: pd.DataFrame, horizon: int, periods_per_year: int
) -> pd.DataFrame:
    """Per-fold metrics for one model's walk-forward CV result."""
    recs = []
    for fold, grp in cv.groupby("fold"):
        mt = compute_metrics(grp["y_true"], grp["y_pred"],
                             horizon=horizon, periods_per_year=periods_per_year)
        mt["fold"] = int(fold)
        mt["n"] = int(len(grp))
        recs.append(mt)
    return pd.DataFrame(recs)


def compare_models(
    models: list,
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_size: int = 504,
    step_size: int = 22,
    horizon: int = 22,
    rolling_window: Optional[int] = None,
    periods_per_year: int = 252,
    return_folds: bool = False,
):
    """Run several models through walk-forward CV and tabulate metrics.

    In addition to the pooled metrics (unchanged), the summary now carries
    statistical-validity diagnostics that matter on short / overlapping
    holdouts: an overlap-aware Sharpe confidence interval and effective sample
    size, a Deflated Sharpe probability (selection-bias corrected for the
    number of candidate models), a Diebold-Mariano p-value versus the naive
    forecast, and the per-fold Sharpe median / dispersion.  All are *extra*
    columns — existing columns and the ``(summary, cv_results)`` return shape
    are preserved.

    Parameters
    ----------
    models:
        List of unfitted forecaster objects.
    X, y:
        Feature matrix and target series.
    initial_train_size, step_size:
        CV parameters (see :func:`walk_forward_cv`).
    rolling_window:
        Passed through to :func:`walk_forward_cv`.  None = expanding window.
    return_folds:
        When True, return a third element: a long-format DataFrame of per-fold
        metrics (columns include ``model``, ``fold``, ``signal_sharpe`` …).

    Returns
    -------
    (summary, cv_results) by default, or (summary, cv_results, fold_metrics)
    when ``return_folds=True``.
    """
    rows = []
    cv_results: dict[str, pd.DataFrame] = {}
    fold_frames: list[pd.DataFrame] = []
    n_trials = max(len(models), 1)

    for m in models:
        logger.info("Evaluating model: %s", m.name)
        cv = walk_forward_cv(m, X, y, initial_train_size=initial_train_size,
                             step_size=step_size, rolling_window=rolling_window,
                             horizon=horizon)
        metrics = compute_metrics(cv["y_true"], cv["y_pred"], name=m.name,
                                  horizon=horizon, periods_per_year=periods_per_year)

        # --- statistical-validity diagnostics on the pooled signal returns ---
        yt = cv["y_true"].to_numpy(dtype=float)
        yp = cv["y_pred"].to_numpy(dtype=float)
        sig = np.sign(yp) * yt
        oa = overlap_aware_sharpe(sig, horizon=horizon, periods_per_year=periods_per_year)
        dsr = deflated_sharpe_ratio(sig, n_trials=n_trials,
                                    effective_n=oa["effective_n"], horizon=horizon)
        dm = diebold_mariano(yt, yp, pred2=None, horizon=horizon, loss="squared")
        fold_mt = _per_fold_metrics(cv, horizon, periods_per_year)

        metrics["signal_sharpe_tstat"] = oa["sharpe_tstat"]
        metrics["signal_sharpe_pvalue"] = oa["p_value"]
        metrics["signal_sharpe_ci_low"] = oa["sharpe_ci_low"]
        metrics["signal_sharpe_ci_high"] = oa["sharpe_ci_high"]
        metrics["effective_n"] = oa["effective_n"]
        metrics["deflated_sharpe"] = dsr["deflated_sr"]
        metrics["dm_stat_vs_naive"] = dm["dm_stat"]
        metrics["dm_pvalue_vs_naive"] = dm["p_value"]
        metrics["n_folds"] = int(fold_mt["fold"].nunique()) if len(fold_mt) else 0
        metrics["signal_sharpe_fold_median"] = (
            float(fold_mt["signal_sharpe"].median()) if len(fold_mt) else float("nan"))
        metrics["signal_sharpe_fold_std"] = (
            float(fold_mt["signal_sharpe"].std(ddof=1)) if len(fold_mt) > 1 else float("nan"))

        metrics["model"] = m.name
        rows.append(metrics)
        cv_results[m.name] = cv
        if len(fold_mt):
            fold_mt.insert(0, "model", m.name)
            fold_frames.append(fold_mt)

    summary = pd.DataFrame(rows).set_index("model")
    if return_folds:
        fold_metrics = (pd.concat(fold_frames, ignore_index=True)
                        if fold_frames else pd.DataFrame())
        return summary, cv_results, fold_metrics
    return summary, cv_results


def out_of_sample_backtest(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    holdout_size: int = 252,
    horizon: int = 22,
    periods_per_year: int = 252,
    n_trials: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train on all-but-last ``holdout_size`` rows; test on the remainder.

    Parameters
    ----------
    model:
        Forecaster to evaluate.
    X, y:
        Full feature matrix and target series (time-ordered).
    holdout_size:
        Number of rows in the out-of-sample test set (~1 year of daily data).
    n_trials:
        Number of candidate models this one is being selected among.  Used only
        to deflate the Sharpe for selection bias (pass ``len(models)`` from the
        notebook's selection loop); does not change the point metrics.

    Returns
    -------
    oos_preds : pd.DataFrame
        Columns: y_true, y_pred.
    metrics : dict
        The usual :func:`compute_metrics` keys plus overlap-aware Sharpe CI /
        effective-N, a Deflated Sharpe probability, and a Diebold-Mariano
        p-value versus the naive forecast.  On a 24-month holdout these flag how
        little of the headline Sharpe survives honest, selection-aware inference.
    """
    n = len(X)
    split = n - holdout_size

    # Purge: drop the last (horizon - 1) rows of the training set so their
    # forward-looking labels do not overlap the holdout test set
    # (López de Prado, AFML Ch. 7).
    purge = max(horizon - 1, 0)
    train_end = split - purge
    if train_end <= 0:
        raise ValueError(
            f"Purge ({purge}) consumed entire training set (split={split}); "
            f"reduce horizon or holdout_size"
        )

    model.fit(X.iloc[:train_end], y.iloc[:train_end])
    preds = model.predict(X.iloc[split:])

    oos = pd.DataFrame(
        {"y_true": y.iloc[split:].values, "y_pred": preds},
        index=y.iloc[split:].index,
    )
    metrics = compute_metrics(oos["y_true"], oos["y_pred"], name=f"{model.name} OOS",
                              horizon=horizon, periods_per_year=periods_per_year)

    # Selection-aware / overlap-aware diagnostics (extra keys, additive).
    yt = oos["y_true"].to_numpy(dtype=float)
    yp = oos["y_pred"].to_numpy(dtype=float)
    sig = np.sign(yp) * yt
    oa = overlap_aware_sharpe(sig, horizon=horizon, periods_per_year=periods_per_year)
    dsr = deflated_sharpe_ratio(sig, n_trials=max(n_trials, 1),
                                effective_n=oa["effective_n"], horizon=horizon)
    dm = diebold_mariano(yt, yp, pred2=None, horizon=horizon, loss="squared")
    metrics["signal_sharpe_tstat"] = oa["sharpe_tstat"]
    metrics["signal_sharpe_pvalue"] = oa["p_value"]
    metrics["signal_sharpe_ci_low"] = oa["sharpe_ci_low"]
    metrics["signal_sharpe_ci_high"] = oa["sharpe_ci_high"]
    metrics["effective_n"] = oa["effective_n"]
    metrics["deflated_sharpe"] = dsr["deflated_sr"]
    metrics["dm_stat_vs_naive"] = dm["dm_stat"]
    metrics["dm_pvalue_vs_naive"] = dm["p_value"]
    return oos, metrics


# ---------------------------------------------------------------------------
# Robust model selection (replaces "argmax pooled signal_sharpe")
# ---------------------------------------------------------------------------

# Metrics where a *larger* value is better.  Anything else (rmse, mae, mape) is
# treated as smaller-is-better.
_HIGHER_IS_BETTER = {
    "deflated_sharpe", "signal_sharpe", "signal_sharpe_fold_median",
    "information_ratio", "rmse_skill", "directional_accuracy",
    "signal_sharpe_tstat",
}


def select_best_model(
    summary: pd.DataFrame,
    criterion: str = "deflated_sharpe",
    alpha: float = 0.10,
    require_beats_naive: bool = True,
    naive_name: Optional[str] = None,
    one_se_rule: bool = False,
) -> dict:
    """Pick a model robustly from a :func:`compare_models` / OOS summary.

    The notebooks currently do ``summary['signal_sharpe'].idxmax()`` on a single
    short holdout — i.e. they select the *max of N noisy estimates*, which is an
    upward-biased estimator and the single biggest source of optimism on the
    weekly / monthly horizons.  This helper instead:

    1. **Gates on significance** — when ``require_beats_naive`` and a
       ``dm_pvalue_vs_naive`` column exists, only models that beat the naive
       random-walk at level ``alpha`` (Diebold-Mariano) are eligible.  If none
       qualify, it falls back to the naive model and says so in ``reason``.
    2. **Ranks by a selection-aware criterion** — defaults to ``deflated_sharpe``
       (selection-bias corrected) rather than the raw Sharpe.
    3. **Optional one-standard-error rule** — among eligible models, prefer the
       most parsimonious (here: the one with the highest deflated Sharpe) whose
       criterion is within one per-fold std of the top, reducing overfitting to
       a single lucky fold.

    Parameters
    ----------
    summary:
        DataFrame indexed by model name, as returned by :func:`compare_models`
        or assembled from :func:`out_of_sample_backtest` rows.
    criterion:
        Column to rank by.  See ``_HIGHER_IS_BETTER`` for direction handling.
    alpha:
        Significance level for the naive-beating gate.
    require_beats_naive:
        If True and ``dm_pvalue_vs_naive`` is present, restrict to significant
        models (falling back to naive when none qualify).
    naive_name:
        Index label of the naive model (used for fallback / exclusion).  When
        None, a row whose name contains "naive" is auto-detected.
    one_se_rule:
        Apply the one-standard-error tie-break (needs ``signal_sharpe_fold_std``).

    Returns
    -------
    dict with keys: ``best`` (model name), ``criterion``, ``qualified`` (list of
    eligible names), ``ranking`` (sorted DataFrame), ``fell_back`` (bool),
    ``reason`` (human-readable explanation).
    """
    if summary is None or len(summary) == 0:
        raise ValueError("summary is empty — nothing to select from")
    if criterion not in summary.columns:
        raise ValueError(
            f"criterion {criterion!r} not in summary columns {list(summary.columns)}")

    higher = criterion in _HIGHER_IS_BETTER
    ranking = summary.sort_values(criterion, ascending=not higher)

    # Identify the naive row for fallback / exclusion.
    if naive_name is None:
        cand = [ix for ix in summary.index if "naive" in str(ix).lower()]
        naive_name = cand[0] if cand else None

    eligible = ranking
    fell_back = False
    reason_bits = [f"ranked by {criterion} ({'higher' if higher else 'lower'}=better)"]

    if require_beats_naive and "dm_pvalue_vs_naive" in summary.columns:
        sig = ranking[ranking["dm_pvalue_vs_naive"] < alpha]
        # The naive model trivially ties itself — never count it as "beating".
        if naive_name is not None:
            sig = sig.drop(index=naive_name, errors="ignore")
        if len(sig) == 0:
            fell_back = True
            best = naive_name if naive_name is not None else ranking.index[0]
            reason_bits.append(
                f"no model beats naive at alpha={alpha} (DM) -> fall back to "
                f"{'naive' if naive_name else 'best-by-criterion'}")
            return {
                "best": best, "criterion": criterion,
                "qualified": [], "ranking": ranking,
                "fell_back": fell_back, "reason": "; ".join(reason_bits),
            }
        eligible = sig
        reason_bits.append(f"{len(sig)} model(s) beat naive at alpha={alpha}")

    best = eligible.index[0]

    if one_se_rule and "signal_sharpe_fold_std" in summary.columns and len(eligible) > 1:
        top_val = eligible[criterion].iloc[0]
        top_se = eligible["signal_sharpe_fold_std"].iloc[0]
        if np.isfinite(top_se) and top_se > 0:
            if higher:
                within = eligible[eligible[criterion] >= top_val - top_se]
            else:
                within = eligible[eligible[criterion] <= top_val + top_se]
            if "deflated_sharpe" in within.columns and len(within) > 1:
                best = within["deflated_sharpe"].idxmax()
                reason_bits.append(
                    f"one-SE rule: chose {best} (most robust within 1 fold-SE)")

    return {
        "best": best, "criterion": criterion,
        "qualified": list(eligible.index), "ranking": ranking,
        "fell_back": fell_back, "reason": "; ".join(reason_bits),
    }


# ---------------------------------------------------------------------------
# Nested cross-validation — estimate the *selection procedure*, not a winner
# ---------------------------------------------------------------------------


def nested_cv_select(
    model_factories: list,
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_splits: int = 5,
    horizon: int = 22,
    periods_per_year: int = 252,
    inner_initial_train_size: Optional[int] = None,
    inner_step_size: Optional[int] = None,
    inner_rolling_window: Optional[int] = None,
    criterion: str = "deflated_sharpe",
    alpha: float = 0.10,
    require_beats_naive: bool = True,
) -> dict:
    """Nested CV: pick a model on inner folds, score it on the outer-test fold.

    The single-holdout protocol (``select_best_model`` on one window, then quote
    that window's metric) is optimistic because the *same* data picks and grades
    the winner.  Nested CV removes that: for each **outer** purged fold the
    *inner* CV + :func:`select_best_model` choose a model using the outer-train
    block only, and that choice is graded on the untouched outer-test fold.  The
    pooled out-of-sample metrics therefore estimate what the **selection rule**
    earns going forward — not what the luckiest model scored in hindsight.

    Parameters
    ----------
    model_factories:
        List of zero-arg callables each returning a *fresh, unfitted* forecaster
        (e.g. ``NaiveModel``, ``LinearModel``, ``lambda: CuratedForecaster(...)``).
        Factories — not instances — because every fold needs unfitted models.
        Keep the list inner-CV-feasible (avoid stacking on short monthly data).
    X, y:
        Time-ordered feature matrix and target.
    n_outer_splits:
        Number of outer purged folds.
    horizon:
        Label horizon (rows) — purges ``horizon - 1`` rows between every
        train/test boundary, inner and outer.
    inner_initial_train_size, inner_step_size, inner_rolling_window:
        Inner :func:`walk_forward_cv` parameters.  Sensible defaults are derived
        per fold when left as None.
    criterion, alpha, require_beats_naive:
        Forwarded to :func:`select_best_model` for the inner pick.

    Returns
    -------
    dict with keys:
        ``chosen_per_fold`` — model name selected in each outer fold;
        ``oos`` — DataFrame (``y_true``, ``y_pred``, ``fold``, ``chosen``) over
        the pooled outer-test folds;
        ``pooled`` — metrics of the procedure (``compute_metrics`` keys plus
        overlap-aware Sharpe CI, deflated Sharpe, DM-vs-naive p-value);
        ``n_outer`` — number of outer folds actually scored.
    """
    if len(model_factories) < 2:
        raise ValueError("Provide at least 2 model factories")

    outer = PurgedTimeSeriesSplit(n_splits=n_outer_splits, horizon=horizon)
    records: list[dict] = []
    chosen_per_fold: list[str] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        # Inner CV parameters scaled to the outer-train length.
        n_tr = len(X_tr)
        init = inner_initial_train_size or max(int(0.5 * n_tr), horizon * 3)
        step = inner_step_size or max(horizon, 1)
        if n_tr <= init + step:
            logger.warning("nested_cv: outer fold %d too small (n=%d) — skipped",
                           fold, n_tr)
            continue

        insts = [f() for f in model_factories]
        name2factory = {m.name: f for m, f in zip(insts, model_factories)}
        try:
            inner_summary, _ = compare_models(
                insts, X_tr, y_tr, initial_train_size=init, step_size=step,
                horizon=horizon, rolling_window=inner_rolling_window,
                periods_per_year=periods_per_year)
        except Exception as exc:
            logger.warning("nested_cv: inner CV failed on fold %d — %s", fold, exc)
            continue

        sel = select_best_model(inner_summary, criterion=criterion, alpha=alpha,
                                require_beats_naive=require_beats_naive)
        chosen = sel["best"]
        chosen_per_fold.append(chosen)

        # Refit a FRESH instance of the chosen model on the (purged) outer-train
        # block, then grade it on the untouched outer-test fold.
        model = name2factory[chosen]()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        for idx, yt, yp in zip(X_te.index, y_te.values, preds, strict=False):
            records.append({"date": idx, "y_true": yt, "y_pred": yp,
                            "fold": fold, "chosen": chosen})

    if not records:
        raise ValueError("nested_cv_select scored no outer folds — data too small")

    oos = pd.DataFrame(records).set_index("date")
    yt = oos["y_true"].to_numpy(dtype=float)
    yp = oos["y_pred"].to_numpy(dtype=float)
    pooled = compute_metrics(yt, yp, horizon=horizon, periods_per_year=periods_per_year)
    sig = np.sign(yp) * yt
    oa = overlap_aware_sharpe(sig, horizon=horizon, periods_per_year=periods_per_year)
    dsr = deflated_sharpe_ratio(sig, n_trials=len(model_factories),
                                effective_n=oa["effective_n"], horizon=horizon)
    dm = diebold_mariano(yt, yp, pred2=None, horizon=horizon, loss="squared")
    pooled.update(
        signal_sharpe_ci_low=oa["sharpe_ci_low"],
        signal_sharpe_ci_high=oa["sharpe_ci_high"],
        signal_sharpe_pvalue=oa["p_value"],
        effective_n=oa["effective_n"],
        deflated_sharpe=dsr["deflated_sr"],
        dm_stat_vs_naive=dm["dm_stat"],
        dm_pvalue_vs_naive=dm["p_value"],
    )
    return {
        "chosen_per_fold": chosen_per_fold,
        "oos": oos,
        "pooled": pooled,
        "n_outer": int(oos["fold"].nunique()),
    }
