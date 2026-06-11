"""
scripts/run_monthly_review.py
=============================
Wave 0 + Wave 1 review checkpoint.

Builds the real monthly copper dataset (month-end calendar, ``horizon_unit='m'``)
and runs the interpretable model line-up through the upgraded evaluation harness,
printing an out-of-sample leaderboard scored the *right* way for a nested-vs-
random-walk comparison:

  * Campbell-Thompson OOS R^2 vs RW   (``rmse_skill``)
  * Clark-West p-value vs RW          (``cw_pvalue_vs_naive``) — nested test
  * Pesaran-Timmermann directional p  (``pt_pvalue``)
  * Model Confidence Set membership   (``in_mcs`` / ``mcs_pvalue``)
  * signal Sharpe + directional accuracy

Line-up: Naive/RW, Futures-Basis (carry), Adaptive-LASSO, ECM, ECM (M-TAR
asymmetric), DMA — all interpretable, all subclassing ``BaseForecaster``.

Notes
-----
* The LME cash-3M basis is SYNTHETIC unless ``lme_basis_csv`` is supplied, so the
  Futures-Basis row here is illustrative plumbing, not a real carry signal.
* Alpha Vantage / EIA are off by default (AV free tier rate-limits at 25/day);
  flip the flags below to include them.

Run from the repo root:
    python scripts/run_monthly_review.py
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.cointegration import compute_ect, test_cointegration  # noqa: E402
from src.evaluation import compare_models, subperiod_metrics  # noqa: E402
from src.feature_engineering import build_features, split_features_targets  # noqa: E402
from src.data_ingestion import load_data  # noqa: E402
from src.models import (  # noqa: E402
    AdaptiveLassoModel,
    FuturesBasisBenchmark,
    NaiveModel,
)
from src.models_dma import DMAForecaster  # noqa: E402
from src.models_ecm import ECMForecaster  # noqa: E402
from src.models_gam import GAMForecaster  # noqa: E402
from src.models_markov import MarkovSwitchingForecaster  # noqa: E402
from src.models_midas import MIDASForecaster  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ── Review configuration ────────────────────────────────────────────────────
USE_ALPHA_VANTAGE = False     # AV free tier rate-limits (25/day) — off for a quick run
USE_EIA = False               # EIA electricity series — off for a quick run
COINT_WINDOW_M = 36           # months for the rolling error-correction terms
PRIMARY_HORIZON = 3           # months (the headline monthly horizon)
PERIODS_PER_YEAR = 12
START = "1990-01-01"


def _load_cfg() -> dict:
    import yaml
    path = os.path.join(REPO, "config_monthly.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _secret(cfg: dict, section: str, env: str) -> str | None:
    return os.environ.get(env) or (cfg.get(section, {}) or {}).get("api_key") or None


# Economic anchors for the error-correction terms (real load_data column names).
ECT_ANCHORS = ["gold", "aluminium", "oil_wti", "dxy", "usd_cny"]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console is cp1252
    except Exception:
        pass

    cfg = _load_cfg()
    feat_cfg = cfg["features"]
    cv_cfg = cfg["cv"]

    print("Loading monthly data (yfinance + FRED"
          f"{' + AlphaVantage' if USE_ALPHA_VANTAGE else ''}"
          f"{' + EIA' if USE_EIA else ''}; LME basis = synthetic) …")
    df_raw = load_data(
        start=START,
        freq="M",
        fred_api_key=_secret(cfg, "fred", "FRED_API_KEY"),
        alpha_vantage_api_key=_secret(cfg, "alpha_vantage", "ALPHA_VANTAGE_API_KEY") if USE_ALPHA_VANTAGE else None,
        eia_api_key=_secret(cfg, "eia", "EIA_API_KEY") if USE_EIA else None,
        include_cot=False,
        include_lme_basis=True,
    )

    # Rolling error-correction terms vs the economic anchors that are present.
    # (Full-sample Engle-Granger is reported for transparency, but the ECM uses
    # the rolling fair-value deviation as a feature regardless of the gate.)
    df_aug = df_raw.copy()
    present = [a for a in ECT_ANCHORS
               if a in df_aug.columns and df_aug[a].notna().sum() > COINT_WINDOW_M + 24]
    print(f"Building rolling ECTs vs: {present}")
    for a in present:
        is_c, pval, _beta = test_cointegration(df_aug["copper_price"], df_aug[a])
        df_aug[f"ect_{a}"] = compute_ect(df_aug["copper_price"], df_aug[a], window=COINT_WINDOW_M)
        print(f"  ect_{a}: Engle-Granger p={pval:.3f}  (cointegrated={is_c})")
    ect_cols = [c for c in df_aug.columns if c.startswith("ect_")]

    feats = build_features(
        df_aug,
        lags=feat_cfg["lags"],
        horizons=feat_cfg["horizons"],
        primary_horizon=PRIMARY_HORIZON,
        return_lags=feat_cfg["return_lags"],
        vol_windows=feat_cfg["vol_windows"],
        ma_window=feat_cfg["ma_window"],
        annualisation_factor=feat_cfg["annualisation_factor"],
        yoy_periods=feat_cfg["yoy_periods"],
        include_intraday=False,
        horizon_unit="m",
    )

    # build_features curates its own column set, so re-attach the ECTs and the
    # basis (and any lagged ECTs) that the ECM / Futures-Basis models consume.
    extra = [c for c in df_aug.columns
             if c.startswith("ect_") or c == "copper_basis_pct"]
    feats = feats.join(df_aug[extra])

    # Drop feature columns that are mostly NaN so the row-wise dropna in
    # split_features_targets doesn't decimate the sample.
    X_all = feats[[c for c in feats.columns
                   if not c.startswith("target_") and c != "copper_price"]]
    keep = [c for c in X_all.columns if X_all[c].notna().mean() >= 0.80]
    feats = feats[keep + [c for c in feats.columns
                          if c.startswith("target_") or c == "copper_price"]]

    X, y_ret, _ = split_features_targets(
        feats, horizon=PRIMARY_HORIZON, horizon_unit="m")
    print(f"Design matrix: {X.shape[0]} monthly rows x {X.shape[1]} features "
          f"({X.index.min():%Y-%m} -> {X.index.max():%Y-%m})")
    ect_in_X = [c for c in X.columns if c.startswith("ect_")]
    print(f"ECTs in X: {ect_in_X};  basis in X: {'copper_basis_pct' in X.columns}\n")

    models = [
        NaiveModel(),
        FuturesBasisBenchmark(horizon=PRIMARY_HORIZON, tenor_periods=3),
        AdaptiveLassoModel(alpha=0.01),
        ECMForecaster(),
        ECMForecaster(asymmetric=True, mode="mtar"),
        DMAForecaster(),
        MIDASForecaster(),                          # Wave 2: restricted distributed lag
        MarkovSwitchingForecaster(k_regimes=2),     # Wave 2: regime-switching mean
        GAMForecaster(max_features=6),              # Wave 2: additive splines
        # GARCH-MIDAS is a variance model -> evaluate via intervals/CRPS, not RMSE.
    ]

    summary, cv_results = compare_models(
        models, X, y_ret,
        initial_train_size=cv_cfg["initial_train_size"],
        step_size=cv_cfg["step_size"],
        horizon=PRIMARY_HORIZON,
        rolling_window=cv_cfg.get("rolling_window"),
        periods_per_year=PERIODS_PER_YEAR,
    )

    cols = ["rmse_skill", "directional_accuracy", "cw_pvalue_vs_naive",
            "pt_pvalue", "signal_sharpe", "in_mcs", "mcs_pvalue"]
    cols = [c for c in cols if c in summary.columns]
    board = summary[cols].sort_values("rmse_skill", ascending=False)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("=" * 78)
    print(f"MONTHLY OOS LEADERBOARD  (h={PRIMARY_HORIZON}m, vs random walk)")
    print("=" * 78)
    print(board.round(4).to_string())
    print("\nModel Confidence Set (alpha=0.10):",
          [m for m in summary.index if bool(summary.loc[m, "in_mcs"])]
          if "in_mcs" in summary.columns else "n/a")

    # Interpretability read-outs from the fitted instances.
    dma = next((m for m in models if m.name == "DMA"), None)
    if dma is not None and dma.inclusion_probabilities_ is not None:
        top = dma.inclusion_probabilities_.mean().sort_values(ascending=False).head(8)
        print("\nDMA — mean inclusion probability (top predictors):")
        print(top.round(4).to_string())

    mtar = next((m for m in models if m.name == "ECM (MTAR asym)"), None)
    if mtar is not None:
        try:
            at = mtar.asymmetry_test()
            print(f"\nECM M-TAR asymmetry test: F={at['f_stat']:.3f}  "
                  f"p={at['p_value']:.4f}  (H0: symmetric adjustment)")
        except Exception as exc:
            print(f"\nECM M-TAR asymmetry test unavailable: {exc}")

    # Sub-period robustness for the best non-naive model.
    best = board.drop(index=[m for m in board.index if "Naive" in m], errors="ignore")
    if len(best):
        name = best.index[0]
        print(f"\nSub-period OOS R^2 by year — best model: {name}")
        sp = subperiod_metrics(cv_results[name], horizon=PRIMARY_HORIZON,
                               periods_per_year=PERIODS_PER_YEAR, by="year")
        print(sp[["rmse_skill", "directional_accuracy", "n"]].round(3).to_string())

    print("\nNote: the Futures-Basis row uses a SYNTHETIC basis (no real LME "
          "cash-3M feed); supply lme_basis_csv for a genuine carry signal.")


if __name__ == "__main__":
    main()
