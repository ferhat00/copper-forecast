"""
scenario_analysis.py
====================
Scenario simulation module for the copper forecasting model.

Core idea
---------
Given a trained model and a current feature vector, perturb one or more
feature values by a specified shock (absolute or relative) and re-run the
model to obtain a "what-if" price forecast.

API
---
ScenarioEngine        : Main class; wraps a fitted model + current features
ScenarioEngine.run    : Run a named scenario
ScenarioEngine.sweep  : Sweep a single variable across a range
ScenarioEngine.report : Tabulate and optionally chart all scenario results
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-defined scenario templates
# ---------------------------------------------------------------------------

SCENARIO_TEMPLATES: dict[str, dict[str, float]] = {
    "bull_strong": {
        "dxy_ret_22d": -0.05,        # DXY falls 5%
        "real_yield_change_22d": -0.5,   # real yields drop 50 bps
        "indpro_yoy": 0.03,          # IP grows 3 pp more
        "sp500_ret_22d": 0.05,
    },
    "bear_strong": {
        "dxy_ret_22d": 0.05,
        "real_yield_change_22d": 0.5,
        "indpro_yoy": -0.03,
        "sp500_ret_22d": -0.05,
    },
    "china_demand_surge": {
        "sp500_ret_22d": 0.03,
        "indpro_yoy": 0.05,
        "cny_usd_level": -0.02,      # CNY appreciates (relative shock)
    },
    "supply_disruption": {
        "copper_ret_1d": 0.01,        # short-term supply squeeze
        "copper_vol_22d": 0.05,       # higher volatility
        "bb_width": 0.02,
    },
    "comex_inventory_drop_40pct": {
        "copper_zscore_200d": 1.5,    # price above long-run average
        "copper_vol_22d": 0.03,
    },
    "high_inflation": {
        "infl_be_level": 1.0,         # +100 bps in breakeven
        "real_yield_change_22d": -0.3,
        "dxy_ret_22d": -0.02,
    },
    "us_tariff_shock": {
        "dxy_ret_22d": 0.03,
        "sp500_ret_22d": -0.03,
        "copper_vol_22d": 0.04,
    },
}


# ---------------------------------------------------------------------------
# ScenarioEngine
# ---------------------------------------------------------------------------


class ScenarioEngine:
    """Apply what-if shocks to a trained forecasting model.

    Parameters
    ----------
    model:
        Fitted forecaster with a ``predict(X)`` method.
    feature_template:
        A single-row DataFrame (or Series) representing the *current*
        feature vector from which scenarios are derived.
    copper_price_current:
        Current copper price ($/t), used to convert return forecasts
        back into price space.
    horizon:
        Forecast horizon in days (used only for labelling).
    """

    def __init__(
        self,
        model: Any,
        feature_template: Union[pd.DataFrame, pd.Series],
        copper_price_current: float,
        horizon: int = 22,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.current_price = copper_price_current

        # Normalise to single-row DataFrame
        if isinstance(feature_template, pd.Series):
            self._base_features = feature_template.to_frame().T
        else:
            self._base_features = feature_template.copy()
        if len(self._base_features) != 1:
            raise ValueError("feature_template must contain exactly one row.")

        # Baseline forecast
        self.base_ret = float(model.predict(self._base_features)[0])
        self.base_price = copper_price_current * np.exp(self.base_ret)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_shocks(self, shocks: dict[str, float]) -> pd.DataFrame:
        """Return a feature vector with the specified additive shocks applied."""
        perturbed = self._base_features.copy()
        for feature, delta in shocks.items():
            if feature in perturbed.columns:
                perturbed[feature] = perturbed[feature].values[0] + delta
            else:
                logger.warning(
                    "Shock target '%s' not in feature set — skipped.", feature
                )
        return perturbed

    def _predict_price(self, features: pd.DataFrame) -> float:
        """Return forecast price ($/t) for a given feature vector."""
        ret = float(self.model.predict(features)[0])
        return self.current_price * np.exp(ret)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        scenario_name: str,
        shocks: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Run a named scenario.

        Parameters
        ----------
        scenario_name:
            A key in ``SCENARIO_TEMPLATES``, or a custom label when
            ``shocks`` is provided explicitly.
        shocks:
            Dict of {feature_name: additive_shock}.  If *None*, the
            function looks up ``scenario_name`` in ``SCENARIO_TEMPLATES``.

        Returns
        -------
        dict with keys: scenario, base_price, scenario_price, delta, delta_pct
        """
        if shocks is None:
            if scenario_name not in SCENARIO_TEMPLATES:
                raise ValueError(
                    f"Unknown scenario '{scenario_name}'. "
                    f"Available: {list(SCENARIO_TEMPLATES)}"
                )
            shocks = SCENARIO_TEMPLATES[scenario_name]

        features = self._apply_shocks(shocks)
        scenario_price = self._predict_price(features)
        delta = scenario_price - self.base_price
        delta_pct = delta / self.base_price * 100

        result = {
            "scenario": scenario_name,
            "base_price": round(self.base_price, 2),
            "scenario_price": round(scenario_price, 2),
            "delta": round(delta, 2),
            "delta_pct": round(delta_pct, 2),
        }
        logger.info(
            "Scenario '%s': base=%.0f  forecast=%.0f  delta=%+.0f (%.1f%%)",
            scenario_name, self.base_price, scenario_price, delta, delta_pct,
        )
        return result

    def run_all_templates(self) -> pd.DataFrame:
        """Run all pre-defined scenario templates and return a summary DataFrame."""
        rows = []
        for name in SCENARIO_TEMPLATES:
            rows.append(self.run(name))
        return pd.DataFrame(rows).set_index("scenario")

    def sweep(
        self,
        feature: str,
        values: list[float],
        label: Optional[str] = None,
    ) -> pd.DataFrame:
        """Sweep a single feature across a range of values.

        Parameters
        ----------
        feature:
            Feature column name to perturb (additive shock from baseline).
        values:
            List of shock magnitudes.
        label:
            Display label for the feature (defaults to ``feature``).

        Returns
        -------
        pd.DataFrame with columns: shock, forecast_price, delta, delta_pct.
        """
        if label is None:
            label = feature

        rows = []
        for v in values:
            features = self._apply_shocks({feature: v})
            price = self._predict_price(features)
            rows.append({
                "shock": v,
                "forecast_price": round(price, 2),
                "delta": round(price - self.base_price, 2),
                "delta_pct": round((price - self.base_price) / self.base_price * 100, 2),
            })

        df = pd.DataFrame(rows)
        df.name = label
        return df

    def report(
        self,
        scenarios: Optional[list[str]] = None,
        extra_shocks: Optional[dict[str, dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """Run a set of scenarios and return a tidy summary DataFrame.

        Parameters
        ----------
        scenarios:
            List of template keys to run.  Defaults to all templates.
        extra_shocks:
            Additional custom scenarios: {label: {feature: shock, ...}}.
        """
        if scenarios is None:
            scenarios = list(SCENARIO_TEMPLATES)

        rows = [self.run(s) for s in scenarios]

        if extra_shocks:
            for name, shocks in extra_shocks.items():
                rows.append(self.run(name, shocks=shocks))

        df = pd.DataFrame(rows).set_index("scenario")
        df = df.sort_values("delta_pct")
        return df
