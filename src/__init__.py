"""
Copper Price Forecasting — Source Package
==========================================
Modular components for data ingestion, feature engineering,
model training, evaluation, visualisation, and scenario analysis.
"""

from . import data_ingestion, feature_engineering, models, evaluation, visualization, scenario_analysis

__all__ = [
    "data_ingestion",
    "feature_engineering",
    "models",
    "evaluation",
    "visualization",
    "scenario_analysis",
]
