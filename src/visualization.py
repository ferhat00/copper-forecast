"""
visualization.py
================
Interactive Plotly charts for the copper forecasting dashboard.

Functions
---------
plot_price_history        : Copper price time series with key events
plot_feature_correlations : Heatmap of feature/target correlations
plot_cv_results           : Walk-forward CV predictions vs actuals
plot_forecast_with_ci     : Forward forecast with 80% CI ribbon
plot_model_comparison     : Bar chart of model error metrics
plot_shap_summary         : SHAP beeswarm / bar summary chart
plot_scenario_tornado     : Tornado chart for scenario sensitivity
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette (consistent across all charts)
# ---------------------------------------------------------------------------

PALETTE = {
    "copper": "#b87333",
    "forecast": "#1f77b4",
    "ci_fill": "rgba(31, 119, 180, 0.2)",
    "naive": "#aec7e8",
    "actual": "#2ca02c",
    "grid": "#e0e0e0",
}


# ---------------------------------------------------------------------------
# 1. Price history
# ---------------------------------------------------------------------------


def plot_price_history(
    df: pd.DataFrame,
    price_col: str = "copper_price",
    title: str = "Copper Price History ($/t)",
) -> go.Figure:
    """Copper price time series with 50-day and 200-day moving averages."""
    price = df[price_col].dropna()
    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price.index, y=price,
        name="Copper spot ($/t)",
        line=dict(color=PALETTE["copper"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=ma50.index, y=ma50,
        name="50-day MA",
        line=dict(color="#ff7f0e", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=ma200.index, y=ma200,
        name="200-day MA",
        line=dict(color="#d62728", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($/t)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Feature correlation heatmap
# ---------------------------------------------------------------------------


def plot_feature_correlations(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 25,
    title: str = "Feature–Target Correlations",
) -> go.Figure:
    """Horizontal bar chart of Spearman correlations with the target."""
    corr = X.corrwith(y, method="spearman").dropna().sort_values()
    # Take top_n most extreme
    if len(corr) > top_n:
        idx = (
            corr.abs()
            .nlargest(top_n)
            .index
        )
        corr = corr[idx].sort_values()

    colors = [PALETTE["forecast"] if v >= 0 else "#d62728" for v in corr.values]

    fig = go.Figure(go.Bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Spearman ρ",
        template="plotly_white",
        height=max(400, 20 * len(corr)),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. CV predictions vs actuals
# ---------------------------------------------------------------------------


def plot_cv_results(
    cv_df: pd.DataFrame,
    model_name: str = "",
    title: Optional[str] = None,
) -> go.Figure:
    """Walk-forward CV: actual vs predicted log-return over time."""
    if title is None:
        title = f"Walk-Forward CV: Actual vs Predicted — {model_name}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cv_df.index, y=cv_df["y_true"],
        name="Actual",
        line=dict(color=PALETTE["actual"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=cv_df.index, y=cv_df["y_pred"],
        name="Predicted",
        line=dict(color=PALETTE["forecast"], width=1.5, dash="dot"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Log Return",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Forecast with confidence interval
# ---------------------------------------------------------------------------


def plot_forecast_with_ci(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    title: str = "Copper Price Forecast with 80% Confidence Interval",
    history_days: int = 252,
) -> go.Figure:
    """Plot recent price history followed by forecast ribbon.

    Parameters
    ----------
    history:
        Historical copper price series (pd.Series with DatetimeIndex).
    forecast_df:
        DataFrame with columns ``date``, ``lower``, ``median``, ``upper``
        containing the quantile forecasts in price space ($/t).
    history_days:
        Number of history days to display (default ~1 year).
    """
    hist = history.tail(history_days)
    fc = forecast_df.set_index("date") if "date" in forecast_df.columns else forecast_df

    fig = go.Figure()

    # Historical price
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name="Historical price",
        line=dict(color=PALETTE["copper"], width=2),
    ))

    # CI ribbon
    fig.add_trace(go.Scatter(
        x=list(fc.index) + list(fc.index[::-1]),
        y=list(fc["upper"]) + list(fc["lower"][::-1]),
        fill="toself",
        fillcolor=PALETTE["ci_fill"],
        line=dict(color="rgba(255,255,255,0)"),
        name="80% CI",
        showlegend=True,
    ))

    # Median forecast
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc["median"],
        name="Forecast (median)",
        line=dict(color=PALETTE["forecast"], width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Copper Price ($/t)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Model comparison bar chart
# ---------------------------------------------------------------------------


def plot_model_comparison(
    summary: pd.DataFrame,
    metric: str = "rmse",
    title: Optional[str] = None,
) -> go.Figure:
    """Bar chart comparing model performance on a chosen metric."""
    if title is None:
        title = f"Model Comparison — {metric.upper()}"

    df = summary[[metric]].sort_values(metric)
    colors = [PALETTE["forecast"]] * len(df)
    # Highlight the best model
    colors[0] = PALETTE["copper"]

    fig = go.Figure(go.Bar(
        x=df.index,
        y=df[metric],
        marker_color=colors,
        text=df[metric].round(4),
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=metric.upper(),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# 6. SHAP summary
# ---------------------------------------------------------------------------


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "SHAP Feature Importance (mean |SHAP|)",
) -> go.Figure:
    """Horizontal bar chart of mean absolute SHAP values."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=True)
    importance = importance.tail(top_n)

    fig = go.Figure(go.Bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        marker_color=PALETTE["copper"],
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP value|",
        template="plotly_white",
        height=max(400, 20 * top_n),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Scenario tornado chart
# ---------------------------------------------------------------------------


def plot_scenario_tornado(
    base_forecast: float,
    scenario_results: dict[str, float],
    title: str = "Scenario Sensitivity ($/t change from base)",
) -> go.Figure:
    """Tornado chart showing price impact of each scenario variable.

    Parameters
    ----------
    base_forecast:
        Baseline forecast price ($/t).
    scenario_results:
        Mapping of {scenario_name: forecast_price}.
    """
    deltas = {k: v - base_forecast for k, v in scenario_results.items()}
    s = pd.Series(deltas).sort_values()

    colors = [PALETTE["actual"] if v >= 0 else "#d62728" for v in s.values]

    fig = go.Figure(go.Bar(
        x=s.values,
        y=s.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.0f}" for v in s.values],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.update_layout(
        title=title,
        xaxis_title="Δ Price vs Baseline ($/t)",
        template="plotly_white",
        height=max(400, 30 * len(s)),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Multi-panel dashboard
# ---------------------------------------------------------------------------


def plot_dashboard(
    df: pd.DataFrame,
    cv_df: pd.DataFrame,
    model_name: str = "Ensemble",
) -> go.Figure:
    """Two-panel summary: price history on top, CV results on bottom."""
    price = df["copper_price"].dropna()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Copper Price History ($/t)", "Walk-Forward CV: Return Prediction"],
        vertical_spacing=0.12,
    )

    # Panel 1 — price
    ma200 = price.rolling(200).mean()
    fig.add_trace(
        go.Scatter(x=price.index, y=price, name="Price", line=dict(color=PALETTE["copper"])),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=ma200.index, y=ma200, name="200d MA",
                   line=dict(color="#d62728", dash="dash")),
        row=1, col=1,
    )

    # Panel 2 — CV
    fig.add_trace(
        go.Scatter(x=cv_df.index, y=cv_df["y_true"], name="Actual ret",
                   line=dict(color=PALETTE["actual"])),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=cv_df.index, y=cv_df["y_pred"], name=f"Predicted ({model_name})",
                   line=dict(color=PALETTE["forecast"], dash="dot")),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=700,
        title_text="Copper Forecasting Dashboard",
    )
    return fig
