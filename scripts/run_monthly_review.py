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
from src.evaluation import (  # noqa: E402
    compare_models, subperiod_metrics, select_best_model,
)
from src.feature_engineering import build_features, split_features_targets  # noqa: E402
from src.data_ingestion import load_data  # noqa: E402
from src.model_lineup import build_model_lineup, build_density_layer  # noqa: E402
from src.altdata import attach_altdata_features  # noqa: E402

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
    src_cfg = cfg.get("sources", {}) or {}
    df_raw = load_data(
        start=START,
        freq="M",
        fred_api_key=_secret(cfg, "fred", "FRED_API_KEY"),
        alpha_vantage_api_key=_secret(cfg, "alpha_vantage", "ALPHA_VANTAGE_API_KEY") if USE_ALPHA_VANTAGE else None,
        eia_api_key=_secret(cfg, "eia", "EIA_API_KEY") if USE_EIA else None,
        include_cot=False,
        include_lme_basis=True,
        lme_basis_csv=src_cfg.get("lme_basis_csv"),   # real cash/3M feed (D1) if provided
    )

    # Phase-4 alt-data: attach GPR/EPU, news sentiment, and a real LME basis when
    # the corresponding config flags are on (all off by default => no-op offline).
    df_raw = attach_altdata_features(
        df_raw, cfg, start=START, fred_api_key=_secret(cfg, "fred", "FRED_API_KEY"))

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

    # Headline lineup is now config-driven (config_monthly.yaml -> models.enabled),
    # shared with copper_forecast_kaggle_monthly.ipynb so the two cannot drift.
    models = build_model_lineup(cfg, PRIMARY_HORIZON, tenor_periods=3)
    print(f"Headline lineup ({len(models)}): {[m.name for m in models]}")

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

    # Robust selection — gate on Clark-West (the *correct* nested test vs the RW),
    # rank by the selection-bias-corrected deflated Sharpe. Driven by
    # config_monthly.yaml -> cv.selection so the notebook uses the same rule.
    sel_cfg = cv_cfg.get("selection") or {}
    try:
        sel = select_best_model(
            summary,
            criterion=sel_cfg.get("criterion", "deflated_sharpe"),
            alpha=sel_cfg.get("selection_alpha", 0.10),
            require_beats_naive=sel_cfg.get("require_beats_naive", True),
            one_se_rule=sel_cfg.get("one_se_rule", False),
            gate_test=sel_cfg.get("gate_test", "clark_west"),
        )
        print(f"\nSelected model: {sel['best']}\n  reason: {sel['reason']}")
    except Exception as exc:
        print(f"\nSelection step unavailable: {exc}")

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

    # ── Density / interval layer — scored on coverage + interval score, NOT RMSE.
    # The mean ties the RW at this horizon; the achievable edge is in the
    # *distribution* (calibrated intervals) and in *direction* (above).
    from src.models import AdaptiveLassoModel
    from src.evaluation import interval_metrics
    density = build_density_layer(
        cfg, horizon=PRIMARY_HORIZON,
        conformal_base=AdaptiveLassoModel(alpha=0.01),
        conformal_alpha=0.80,
    )
    if density:
        holdout = int(cv_cfg.get("holdout_months", 24))
        split = len(X) - holdout
        purge = max(PRIMARY_HORIZON - 1, 0)
        print(f"\nDensity layer — 80% interval calibration on the last "
              f"{holdout}m (coverage target 0.80):")
        for label, dmodel in density.items():
            try:
                dmodel.fit(X.iloc[:split - purge], y_ret.iloc[:split - purge])
                band = dmodel.predict_interval(X.iloc[split:], alpha=0.80)
                im = interval_metrics(y_ret.iloc[split:].to_numpy(float),
                                      band["lower"], band["upper"], coverage_level=0.80)
                print(f"  {label:<14} coverage={im['coverage']:.2f}  "
                      f"mean_width={im['mean_width']:.4f}  "
                      f"interval_score={im['interval_score']:.4f}  n={int(im['n_obs'])}")
            except Exception as exc:
                print(f"  {label:<14} unavailable: {exc}")

    print("\nNote: the Futures-Basis row uses a SYNTHETIC basis (no real LME "
          "cash-3M feed); set sources.lme_cash_3m_basis + supply a real cash/3M "
          "feed for a genuine carry signal.")


if __name__ == "__main__":
    main()
