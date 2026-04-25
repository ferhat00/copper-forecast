"""
evaluation.py
=============
Walk-forward cross-validation and forecast evaluation metrics.

Functions
---------
walk_forward_cv      : Expanding-window CV returning per-fold predictions
directional_accuracy : % of predictions with correct sign of change
compute_metrics      : RMSE, MAE, MAPE, DA for a prediction series
compare_models       : Run multiple forecasters through walk-forward CV
"""

from __future__ import annotations

import logging
from typing import Optional

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
        Forecast horizon in trading days (used for annualising Sharpe).

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
    annualise = np.sqrt(252 / max(horizon, 1))
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
) -> pd.DataFrame:
    """Walk-forward cross-validation with optional rolling window.

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

    records = []
    fold = 0
    train_end = initial_train_size

    while train_end < n:
        test_end = min(train_end + step_size, n)

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
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
            "Fold %d | train=%d  test=%d-%d",
            fold, train_end, train_end, test_end,
        )

        train_end += step_size
        fold += 1

    result = pd.DataFrame(records).set_index("date")
    return result


def compare_models(
    models: list,
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_size: int = 504,
    step_size: int = 22,
    horizon: int = 22,
    rolling_window: Optional[int] = None,
) -> pd.DataFrame:
    """Run several models through walk-forward CV and tabulate metrics.

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

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = RMSE, MAE, MAPE, directional_accuracy.
    """
    rows = []
    cv_results: dict[str, pd.DataFrame] = {}

    for m in models:
        logger.info("Evaluating model: %s", m.name)
        cv = walk_forward_cv(m, X, y, initial_train_size=initial_train_size,
                             step_size=step_size, rolling_window=rolling_window)
        metrics = compute_metrics(cv["y_true"], cv["y_pred"], name=m.name, horizon=horizon)
        metrics["model"] = m.name
        rows.append(metrics)
        cv_results[m.name] = cv

    summary = pd.DataFrame(rows).set_index("model")
    return summary, cv_results


def out_of_sample_backtest(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    holdout_size: int = 252,
    horizon: int = 22,
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

    Returns
    -------
    oos_preds : pd.DataFrame
        Columns: y_true, y_pred.
    metrics : dict
    """
    n = len(X)
    split = n - holdout_size

    model.fit(X.iloc[:split], y.iloc[:split])
    preds = model.predict(X.iloc[split:])

    oos = pd.DataFrame(
        {"y_true": y.iloc[split:].values, "y_pred": preds},
        index=y.iloc[split:].index,
    )
    metrics = compute_metrics(oos["y_true"], oos["y_pred"], name=f"{model.name} OOS",
                              horizon=horizon)
    return oos, metrics
