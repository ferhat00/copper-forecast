"""
feature_pruning.py
==================
SHAP-based automatic feature importance ranking and pruning.

Computes mean |SHAP| values for each feature and removes low-importance
features below a configurable threshold.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_importance(
    model,
    X: pd.DataFrame,
    max_samples: int = 1000,
) -> pd.DataFrame:
    """Compute mean |SHAP| importance for each feature.

    Parameters
    ----------
    model:
        A fitted model.  Uses TreeExplainer for tree-based models
        (XGBoost, LightGBM), falls back to a permutation-based
        approximation for others.
    X:
        Feature matrix (a subsample is used if ``len(X) > max_samples``).
    max_samples:
        Maximum number of samples for SHAP computation.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``mean_abs_shap``, sorted descending.
    """
    import shap

    # Subsample for speed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    # Determine explainer type
    inner = getattr(model, "_model", model)
    if inner is None:
        inner = model

    # Try tree explainer first (fastest)
    try:
        explainer = shap.TreeExplainer(inner)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        # Fallback: use permutation explainer
        logger.info("SHAP: TreeExplainer unavailable, using permutation-based approximation")

        def predict_fn(x):
            return model.predict(pd.DataFrame(x, columns=X.columns))

        explainer = shap.Explainer(predict_fn, X_sample)
        shap_values = explainer(X_sample).values

    mean_abs = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance


def auto_prune_features(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: Union[str, float] = "bottom_20pct",
    max_shap_samples: int = 1000,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Fit model, compute SHAP importance, and prune low-importance features.

    Parameters
    ----------
    model:
        An unfitted model (will be fitted internally).
    X:
        Full feature matrix.
    y:
        Target series.
    threshold:
        Pruning threshold. Options:

        - ``"bottom_20pct"``: Drop features in the bottom 20% by importance.
        - ``"bottom_30pct"``: Drop features in the bottom 30%.
        - A float value: drop features with mean |SHAP| below this absolute value.
    max_shap_samples:
        Maximum samples for SHAP computation.

    Returns
    -------
    X_pruned : pd.DataFrame
        Feature matrix with low-importance features removed.
    dropped : list[str]
        Names of dropped features.
    importance : pd.DataFrame
        Full importance table (before pruning).
    """
    # Fit model
    model.fit(X, y)

    # Compute importance
    importance = compute_shap_importance(model, X, max_samples=max_shap_samples)

    # Determine cutoff
    if isinstance(threshold, str) and threshold.startswith("bottom_"):
        pct = int(threshold.split("_")[1].replace("pct", ""))
        cutoff = np.percentile(importance["mean_abs_shap"].values, pct)
    else:
        cutoff = float(threshold)

    # Identify features to drop
    to_drop = importance[importance["mean_abs_shap"] < cutoff]["feature"].tolist()
    to_keep = [c for c in X.columns if c not in to_drop]

    X_pruned = X[to_keep]

    logger.info(
        "Feature pruning: kept %d / %d features (dropped %d below threshold %.6f)",
        len(to_keep), len(X.columns), len(to_drop), cutoff,
    )

    return X_pruned, to_drop, importance
