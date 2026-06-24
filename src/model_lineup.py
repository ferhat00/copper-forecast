"""
model_lineup.py
===============
Config-driven construction of the monthly model line-up, shared by the
notebook (``copper_forecast_kaggle_monthly.ipynb``) and
``scripts/run_monthly_review.py`` so the two cannot drift apart.

Historically the notebook hard-coded its headline ``compare_models`` lineup in
Python lists while the review script wired a *different* (richer) set, so the
interpretable copper-monthly models with the best evidence — DMA, ECM (+M-TAR),
MIDAS, Markov-switching, GAM — were only ever scored by the script, never in the
notebook's selection.  This module makes the lineup a single function driven by
``config_monthly.yaml``:

    models:
      enabled:        [naive, futures_basis, adaptive_lasso, ecm, ecm_mtar,
                       dma, midas, markov, gam, combo_equal]
      density_layer:  [garch_midas, conformal]

Each key maps to a builder returning a *fresh, unfitted* ``BaseForecaster`` (safe
to rebuild per walk-forward fold).  Unknown keys are skipped with a warning;
builders whose optional dependency is missing (pygam, statsmodels, xgboost,
lightgbm, prophet) degrade gracefully to a skip rather than raising — matching
the repo's "gracefully skipped if unavailable" philosophy.  Combiners
(``combo_equal`` / ``combo_median``) are built last over *fresh* copies of the
enabled point models, so they never share fitted state with the standalone rows.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Default headline lineup when the config carries no ``models.enabled`` list.
# Mirrors scripts/run_monthly_review.py plus the 1/N combination benchmark.
DEFAULT_ENABLED = [
    "naive", "ar1", "futures_basis", "kalman_futures", "adaptive_lasso",
    "dlinear", "ecm", "ecm_mtar", "dma", "midas", "markov", "gam",
    "residual_hybrid", "combo_equal",
]
# Default interval/variance layer (scored on CRPS / coverage, not RMSE).
DEFAULT_DENSITY = ["garch_midas", "conformal"]

# Keys that are forecast *combinations* (built last, over the other base models).
_COMBINERS = {"combo_equal", "combo_median"}
# Keys never folded into a 1/N combiner: the trivial RW, the regime meta-model,
# and the foundation models (which degrade to RW offline).
_EXCLUDE_FROM_COMBO = _COMBINERS | {"naive", "regime_combo", "ttm", "chronos", "timesfm"}


def _safe(builder: Callable, key: str):
    """Instantiate ``builder()``; on any error log and return None (skip)."""
    try:
        return builder()
    except Exception as exc:  # pragma: no cover - defensive (optional deps)
        logger.warning("model_lineup: skipping %r (%s)", key, exc)
        return None


def _base_builders(
    horizon: int,
    tenor_periods: int,
    basis_col: str,
    analyst_col: Optional[str] = None,
    oof_initial_size: int = 120,
    oof_step: int = 3,
) -> dict[str, Callable]:
    """Map each base-model key to a zero-arg builder (lazy imports inside)."""

    def naive():
        from src.models import NaiveModel
        return NaiveModel()

    def ar1():
        from src.models_baselines import AR1Model
        return AR1Model()

    def dlinear():
        from src.models_baselines import DLinearForecaster
        return DLinearForecaster()

    def futures_basis():
        from src.models import FuturesBasisBenchmark
        return FuturesBasisBenchmark(
            horizon=horizon, tenor_periods=tenor_periods, basis_col=basis_col)

    def kalman_futures():
        from src.models_futures import KalmanFuturesForecaster
        return KalmanFuturesForecaster(
            horizon=horizon, tenor_periods=tenor_periods,
            basis_col=basis_col, analyst_col=analyst_col)

    def residual_hybrid():
        from src.models_hybrid import ResidualBoostForecaster
        return ResidualBoostForecaster()

    def regime_combo():
        # Opt-in: needs a 'regime' label column in X (raises at fit otherwise),
        # so it is NOT in DEFAULT_ENABLED. The notebook's unpruned dev set has it.
        from src.models_regime import RegimeWeightedEnsemble
        from src.models import LinearModel, CuratedForecaster, ElasticNetModel
        return RegimeWeightedEnsemble(
            [LinearModel(), CuratedForecaster(ElasticNetModel())],
            regime_col="regime", horizon=horizon,
            oof_initial_size=oof_initial_size, oof_step=oof_step)

    def _foundation(backend):
        from src.models_foundation import FoundationModelForecaster
        return FoundationModelForecaster(backend=backend)

    def adaptive_lasso():
        from src.models import AdaptiveLassoModel
        return AdaptiveLassoModel(alpha=0.01)

    def elasticnet():
        from src.models import ElasticNetModel
        return ElasticNetModel()

    def curated():
        from src.models import CuratedForecaster, ElasticNetModel
        return CuratedForecaster(ElasticNetModel())

    def ecm():
        from src.models_ecm import ECMForecaster
        return ECMForecaster()

    def ecm_mtar():
        from src.models_ecm import ECMForecaster
        return ECMForecaster(asymmetric=True, mode="mtar")

    def dma():
        from src.models_dma import DMAForecaster
        return DMAForecaster()

    def dms():
        from src.models_dma import DMAForecaster
        return DMAForecaster(dms=True)

    def midas():
        from src.models_midas import MIDASForecaster
        return MIDASForecaster()

    def markov():
        from src.models_markov import MarkovSwitchingForecaster
        return MarkovSwitchingForecaster(k_regimes=2)

    def gam():
        from src.models_gam import GAMForecaster
        return GAMForecaster(max_features=6)

    def xgb():
        from src.models import XGBoostModel
        return XGBoostModel()

    def lgb():
        from src.models import LGBMModel
        return LGBMModel()

    return {
        "naive": naive,
        "ar1": ar1,
        "dlinear": dlinear,
        "futures_basis": futures_basis,
        "kalman_futures": kalman_futures,
        "adaptive_lasso": adaptive_lasso,
        "elasticnet": elasticnet,
        "curated": curated,
        "ecm": ecm,
        "ecm_mtar": ecm_mtar,
        "dma": dma,
        "dms": dms,
        "midas": midas,
        "markov": markov,
        "gam": gam,
        "residual_hybrid": residual_hybrid,
        "regime_combo": regime_combo,
        "xgb": xgb,
        "lgb": lgb,
        # Foundation-model benchmark entrants (degrade to RW if the backend
        # package is absent — see src.models_foundation). Opt-in via models.enabled.
        "ttm": lambda: _foundation("tinytimemixer"),
        "chronos": lambda: _foundation("chronos"),
        "timesfm": lambda: _foundation("timesfm"),
    }


def _make_combiner(base_models: list, method: str, horizon: int):
    from src.models_stacking import RobustCombiner
    return RobustCombiner(base_models=base_models, method=method, horizon=horizon)


def build_model_lineup(
    cfg: Optional[dict],
    horizon: int,
    *,
    tenor_periods: int = 3,
    basis_col: str = "copper_basis_pct",
    oof_initial_size: int = 120,
    oof_step: int = 3,
    extra_builders: Optional[dict[str, Callable]] = None,
) -> list:
    """Build the headline ``compare_models`` lineup from ``cfg['models']['enabled']``.

    Parameters
    ----------
    cfg:
        Parsed ``config_monthly.yaml`` (or any dict with a ``models`` section).
        When ``cfg`` is None or has no ``models.enabled`` list, falls back to
        :data:`DEFAULT_ENABLED`.
    horizon:
        Forecast horizon in periods (months) — used by the futures-basis model
        and the combiner's overlap purge.
    tenor_periods, basis_col:
        Futures-basis parameters (3 months / ``copper_basis_pct`` for monthly).
    extra_builders:
        Optional ``{key: zero-arg-callable}`` to register models defined outside
        ``src`` (e.g. the notebook's ``arimax``/``prophet`` helpers).  These take
        precedence over the built-ins for the same key.

    Returns
    -------
    list of fresh, unfitted forecasters in the order given by ``enabled``.
    """
    models_cfg = (cfg or {}).get("models", {}) or {}
    enabled = list(models_cfg.get("enabled") or DEFAULT_ENABLED)
    analyst_col = models_cfg.get("kalman_analyst_col")

    builders = _base_builders(horizon, tenor_periods, basis_col,
                              analyst_col=analyst_col,
                              oof_initial_size=oof_initial_size, oof_step=oof_step)
    if extra_builders:
        builders.update(extra_builders)

    def build_one(key: str):
        if key not in builders:
            logger.warning("model_lineup: unknown model key %r — skipped", key)
            return None
        return _safe(builders[key], key)

    # Base models a combiner averages over: every enabled point model except the
    # trivial RW, the regime meta-model, the foundation entrants and the combiners.
    combiner_keys = [k for k in enabled if k not in _EXCLUDE_FROM_COMBO]

    out: list = []
    for key in enabled:
        if key in _COMBINERS:
            method = "equal" if key == "combo_equal" else "median"
            bases = [b for b in (build_one(k) for k in combiner_keys) if b is not None]
            if len(bases) < 2:
                logger.warning(
                    "model_lineup: %r needs >=2 base models (got %d) — skipped",
                    key, len(bases))
                continue
            model = _safe(lambda: _make_combiner(bases, method, horizon), key)
        else:
            model = build_one(key)
        if model is not None:
            out.append(model)

    if not out:
        logger.warning("model_lineup: no models built — falling back to Naive only")
        from src.models import NaiveModel
        out = [NaiveModel()]
    return out


def build_density_layer(
    cfg: Optional[dict],
    *,
    horizon: int,
    conformal_base=None,
    conformal_alpha: float = 0.80,
    garch_macro_col: Optional[str] = None,
    garch_n_lags: int = 12,
) -> dict:
    """Build the interval/variance layer from ``cfg['models']['density_layer']``.

    These models are scored on CRPS / interval coverage (see
    ``src.evaluation.crps_*`` / ``interval_metrics``), *not* RMSE.

    Parameters
    ----------
    conformal_base:
        A point forecaster instance to wrap with split-conformal intervals.  If
        None, the ``conformal`` key is skipped (with a warning).
    garch_macro_col:
        Column carrying the monthly long-run-variance driver (e.g. ``"ppi_metals"``).
        None -> GARCH-MIDAS uses its realized-variance proxy (always available).

    Returns
    -------
    dict ``{label: model}`` of fresh, unfitted density models.
    """
    models_cfg = (cfg or {}).get("models", {}) or {}
    keys = list(models_cfg.get("density_layer") or DEFAULT_DENSITY)
    macro_col = garch_macro_col or models_cfg.get("garch_midas_macro_col")

    out: dict = {}
    for key in keys:
        if key == "garch_midas":
            def _garch():
                from src.models_garch_midas import GarchMidasModel
                return GarchMidasModel(macro_col=macro_col, n_lags=garch_n_lags)
            m = _safe(_garch, key)
            if m is not None:
                out["GARCH-MIDAS"] = m
        elif key == "conformal":
            if conformal_base is None:
                logger.warning(
                    "model_lineup: 'conformal' needs conformal_base — skipped")
                continue

            def _conf():
                from src.conformal import ConformalForecaster
                return ConformalForecaster(conformal_base, alpha=conformal_alpha)
            m = _safe(_conf, key)
            if m is not None:
                out["Conformal"] = m
        else:
            logger.warning("model_lineup: unknown density key %r — skipped", key)
    return out
