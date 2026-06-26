"""
tests/test_model_lineup.py
==========================
Unit tests for the config-driven model lineup (src/model_lineup.py) and the
Clark-West selection gate added to src/evaluation.select_best_model.

These cover the "quick win" wiring: that config_monthly.yaml -> models.enabled
deterministically builds the headline lineup (shared by the notebook and
scripts/run_monthly_review.py), that the 1/N combiner wraps fresh base models,
that the density layer builds, and that gating on Clark-West (the correct test
for RW-nested models) can credit a model that Diebold-Mariano would reject.

Run with:
    pytest tests/test_model_lineup.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import compare_models, select_best_model  # noqa: E402
from src.model_lineup import (  # noqa: E402
    DEFAULT_ENABLED,
    build_density_layer,
    build_model_lineup,
)
from src.models import AdaptiveLassoModel, NaiveModel  # noqa: E402
from src.models_stacking import RobustCombiner  # noqa: E402


def _cfg(enabled=None, density=None):
    models: dict = {}
    if enabled is not None:
        models["enabled"] = enabled
    if density is not None:
        models["density_layer"] = density
    return {"models": models}


# A dependency-light subset that builds without pygam / prophet / boosting libs.
SAFE = ["naive", "futures_basis", "adaptive_lasso", "ecm", "dma"]


class TestBuildLineup:
    def test_config_drives_enabled(self):
        models = build_model_lineup(_cfg(SAFE), horizon=3)
        assert len(models) == len(SAFE)
        assert any(isinstance(m, NaiveModel) for m in models)

    def test_combo_equal_wraps_non_naive_bases(self):
        models = build_model_lineup(_cfg(SAFE + ["combo_equal"]), horizon=3)
        combiners = [m for m in models if isinstance(m, RobustCombiner)]
        assert len(combiners) == 1
        rc = combiners[0]
        assert rc.method == "equal"
        # Wraps every enabled point model except the trivial naive benchmark.
        assert len(rc.base_models) == len(SAFE) - 1

    def test_combiner_uses_fresh_instances(self):
        models = build_model_lineup(_cfg(["adaptive_lasso", "ecm", "combo_equal"]),
                                    horizon=3)
        rc = next(m for m in models if isinstance(m, RobustCombiner))
        standalone = [m for m in models if not isinstance(m, RobustCombiner)]
        for base in rc.base_models:
            assert all(base is not s for s in standalone)

    def test_unknown_key_skipped(self):
        models = build_model_lineup(_cfg(["naive", "does_not_exist", "dma"]), horizon=3)
        assert len(models) == 2

    def test_combo_needs_two_bases(self):
        # Only one non-naive base -> combiner cannot form -> skipped (not raised).
        models = build_model_lineup(_cfg(["naive", "adaptive_lasso", "combo_equal"]),
                                    horizon=3)
        assert not any(isinstance(m, RobustCombiner) for m in models)
        assert len(models) == 2

    def test_default_when_no_models_section(self):
        models = build_model_lineup({}, horizon=3)
        assert len(models) >= 1
        assert any(isinstance(m, NaiveModel) for m in models)
        # DEFAULT_ENABLED is the documented fallback.
        assert "naive" in DEFAULT_ENABLED and "dma" in DEFAULT_ENABLED


class TestDensityLayer:
    def test_builds_garch_and_conformal(self):
        layer = build_density_layer(
            _cfg(density=["garch_midas", "conformal"]),
            horizon=3, conformal_base=AdaptiveLassoModel(alpha=0.01))
        assert "GARCH-MIDAS" in layer
        assert "Conformal" in layer

    def test_conformal_skipped_without_base(self):
        layer = build_density_layer(
            _cfg(density=["garch_midas", "conformal"]),
            horizon=3, conformal_base=None)
        assert "GARCH-MIDAS" in layer
        assert "Conformal" not in layer


class TestLineupIntegration:
    def _data(self, n=320, seed=3):
        idx = pd.date_range("2017-01-01", periods=n, freq="B")
        rng = np.random.default_rng(seed)
        ect = np.empty(n)
        ect[0] = 0.0
        shocks = rng.standard_normal(n)
        for t in range(1, n):
            ect[t] = 0.9 * ect[t - 1] + shocks[t]
        X = pd.DataFrame(
            {
                "ect_gold": ect,
                "copper_basis_pct": rng.standard_normal(n) * 0.01,
                "f0": rng.standard_normal(n),
                "f1": rng.standard_normal(n),
            },
            index=idx,
        )
        y = pd.Series(-0.2 * ect + 0.3 * X["f0"] + 0.1 * rng.standard_normal(n),
                      index=idx)
        return X, y

    def test_factory_lineup_runs_through_compare_models(self):
        X, y = self._data()
        models = build_model_lineup(_cfg(SAFE + ["combo_equal"]), horizon=22)
        summary, cv_results = compare_models(
            models, X, y, initial_train_size=160, step_size=22, horizon=22)
        assert len(summary) == len(models)
        for col in ["cw_pvalue_vs_naive", "pt_pvalue", "rmse"]:
            assert col in summary.columns
        for cv in cv_results.values():
            assert np.isfinite(cv["y_pred"]).all()


class TestClarkWestGate:
    def _summary(self):
        # A candidate that beats the RW on Clark-West (nested test) but NOT on
        # the (mis-specified for nested models) Diebold-Mariano test.
        return pd.DataFrame(
            {
                "deflated_sharpe": [0.10, 0.60],
                "cw_pvalue_vs_naive": [1.00, 0.01],
                "dm_pvalue_vs_naive": [1.00, 0.50],
            },
            index=["Naive (RW)", "DMA"],
        )

    def test_clark_west_gate_credits_the_model(self):
        sel = select_best_model(self._summary(), gate_test="clark_west",
                                require_beats_naive=True, alpha=0.10)
        assert sel["best"] == "DMA"
        assert not sel["fell_back"]
        assert "CW" in sel["reason"]

    def test_diebold_mariano_gate_falls_back_to_naive(self):
        sel = select_best_model(self._summary(), gate_test="diebold_mariano",
                                require_beats_naive=True, alpha=0.10)
        assert sel["fell_back"]
        assert sel["best"] == "Naive (RW)"
        assert "DM" in sel["reason"]
