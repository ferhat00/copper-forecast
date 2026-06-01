# Copper Forecast — Weekly/Monthly Quality Improvement Plan

## Context

`copper-forecast` forecasts COMEX copper with an ensemble of econometric + ML models, shared via `src/` and driven by three notebooks (daily 5d, monthly 1/3/6/12m, original 22d base). OOS directional accuracy sits at ~52–53% and the headline signal Sharpe (~1.9 in `outputs_1d`) is not credible. The goal is to improve **weekly and monthly** forecast quality — prioritising **monthly**, optimising **signal-led** (direction first, point-forecast + intervals secondary), changes **additive / carefully refactored** (existing notebooks keep working; shared-engine changes need test coverage), and **accuracy-first** on the interpretability axis.

The core thesis of this plan: **for the monthly horizon, the single highest-value work is "predict less, evaluate better" — fix model selection and the Sharpe/skill measurement before reaching for fancier models.** The current numbers are inflated by selection on a single short holdout and by overlapping-return Sharpe, not by a genuine modelling deficit that a new architecture would fix.

---

## Implementation status (Quick Wins 1–5 — DONE, all additive, full suite green: 86 passed / 9 pre-existing skips)

| # | What landed | Where | Tests |
|---|-------------|-------|-------|
| 1 | `overlap_aware_sharpe`, `deflated_sharpe_ratio` (Bailey/LdP), `diebold_mariano` (+ HLN correction) | `src/evaluation.py` | `tests/test_evaluation_stats.py` |
| 2 | `compare_models` enriched with deflated Sharpe, HAC Sharpe CI / `effective_n`, DM-vs-naive p-value, per-fold Sharpe median/std; opt-in `return_folds`; `out_of_sample_backtest` gets the same diagnostics + `n_trials` | `src/evaluation.py` | `tests/test_evaluation_stats.py` |
| 3 | `select_best_model` — naive-gated (DM), ranks by `deflated_sharpe`, optional one-SE rule, graceful naive fallback | `src/evaluation.py` | `tests/test_evaluation_stats.py` |
| 4 | `RobustCombiner` (equal / median / inverse-error, degrades to equal weights when OOF infeasible) — the monthly-safe alternative to stacking | `src/models_stacking.py` | `tests/test_new_models.py::TestRobustCombiner` |
| 5 | Vol-normalised target `target_retvol_{h}{u}` (ex-ante trailing σ, no look-ahead) behind `vol_normalise_targets`; `split_features_targets(target_kind=...)` | `src/feature_engineering.py` | `tests/test_vol_normalised_target.py` |

**All existing signatures preserved** (new params default to old behaviour; `compare_models` still returns the 2-tuple unless `return_folds=True`).

### Notebooks wired (DONE)
- **`copper_forecast_kaggle_monthly.ipynb` → v2**: imports the new helpers; `build_features(..., vol_normalise_targets=True)`; `CFG` gains `target_kind` / `selection_criterion` / `selection_alpha` / `include_cot`; `RobustCombiner(method='median')` added to the CV and OOS line-ups; OOS uses `out_of_sample_backtest(..., n_trials=len(oos_model_list))`; **selection switched from `oos_summary['signal_sharpe'].idxmax()` to `select_best_model(...)`** (DM-gated, deflated-Sharpe-ranked, naive fallback) and prints the robust diagnostics; dashboard reuses the selected model.
- **`copper_forecast_kaggle_weekly.ipynb` (NEW)** + **`config_weekly.yaml`**: daily bars (`freq="B"`) with a **5-trading-day primary horizon** (the repo's definition of "weekly"; `load_data` supports only B/M). Cloned from monthly v2 with the same engine wiring; COT enabled (daily), `include_intraday=true`, `periods_per_year=252`, CV `504/5/holdout 252/roll 504`. Both notebooks validated: all 34 code cells compile; the `compare_models → out_of_sample_backtest(n_trials) → select_best_model` path smoke-tested end-to-end on synthetic data.

> Note: notebook headline target stays `target_kind='ret'` (price reconstruction via `exp(return)` stays valid); `retvol` columns are emitted and available for experimentation.

---

## Step 1 — Understanding of the current pipeline (with code evidence)

### (a) Modelling pipeline
1. **Ingestion** ([src/data_ingestion.py](src/data_ingestion.py)): ~90 yfinance tickers (daily), ~30 FRED series (mixed daily/weekly/monthly), AlphaVantage monthly LME metals (incl. `PCOPPUSDM`/`COPPER` back to ~1990 → the ~430 monthly obs), EIA, optional LME/SHFE inventory + Chile exports. Macro series carry **publication lags** (`FRED_PUBLICATION_LAGS`, `data_ingestion.py:262`) applied via `.shift(lag)` (`:963`), with business-day→month conversion `math.ceil(lag_bd/21)` (`:246`). Monthly resampling uses per-column last/mean rules (`MONTHLY_AGG_MEAN`, `:195`). **Leakage handling here is sound.**
2. **Feature engineering** ([src/feature_engineering.py](src/feature_engineering.py)): ~45–55 base features (copper returns/vol/zscore/RSI/MACD/BB, cross-asset ratios, macro YoY, inventories, calendar), each lagged by `DEFAULT_LAGS=[1,5,22]` (`:407`) → ~150–250 columns. Then cointegration ECTs (`cointegration.py`, Engle-Granger), HMM regime one-hots (`regime_detection.py`, 3-state Gaussian, refit per fold), then SHAP bottom-20–30% pruning (`feature_pruning.py`). Rows with any NaN dropped; no global scaling (Ridge scales internally).
3. **Targets** (`feature_engineering.py:416`): forward **log-returns** `target_ret_{h}{u} = log(cp.shift(-h)/cp)` plus `target_price_{h}{u}`. Daily horizons `[1,5,22,66]` trading days; monthly passes `[1,3,6,12]` with unit `"m"`. Target shift is backward-looking-safe (`shift(-h)`).
4. **Model zoo** ([src/models.py](src/models.py) + friends): Naive(0), Ridge(α=1.0 **fixed**), XGBoost & LightGBM (Optuna 50 trials, RMSE objective), XGB/LGBM **direction classifiers**, EnsembleModel (equal/weighted avg), StackingEnsemble (Ridge meta on walk-forward OOF, `models_stacking.py`), RegimeRouter (per-regime models, `models_regime.py`), Prophet, QuantileForecaster (LGBM quantile, 80% band).
5. **Evaluation** ([src/evaluation.py](src/evaluation.py)): `PurgedTimeSeriesSplit` purges `horizon-1+gap` train rows (`:182`), no right embargo. `walk_forward_cv` (`:204`) expanding/rolling, `initial_train_size=504`, `step_size=22`, refit. `compare_models` (`:300`) **pools y_true/y_pred across all folds into one `compute_metrics` call** — single global number per metric, no per-fold distribution.

### (b) How the "best model" is selected
In the **notebooks** (not `src/`): rank by **`oos_summary['signal_sharpe'].idxmax()`** on a single holdout (24 months monthly / 252 days daily), then **refit the winner on ALL data including that holdout** before producing the live forecast. Signal Sharpe = `mean(sign(pred)·y_true)/std(·) · sqrt(periods_per_year/horizon)` (`evaluation.py:90`).

### (c) The 3–4 weakest links for weekly/monthly
1. **Selection overfitting + holdout reuse (biggest).** Picking max-Sharpe among ~10 candidates on one ~24-point holdout, then quoting that holdout's Sharpe as "OOS edge," is selecting on noise and reporting an optimistic number. The winner is then refit *including* the reporting holdout, so the live model has no clean OOS estimate at all.
2. **Sharpe is not trustworthy.** The annualization is the *correct* √(ppy/h) form **only for non-overlapping h-period returns**; the holdout actually contains overlapping daily/monthly observations of h-step-forward returns, which deflate `std` and inflate Sharpe. No selection-bias deflation. With ~24 monthly points and h=6/12, there are only ~4/~2 *independent* observations — **the 6m/12m horizons are effectively unevaluable on a 24-month holdout.**
3. **Curse of dimensionality.** ~120–250 features on ~430 monthly rows (fewer after purge/holdout). SHAP pruning still leaves ~100. p ≫ effective-n; this is where the macro signal drowns. Stacking literally fails on monthly ("Dataset too small").
4. **Target/lag design for monthly.** RMSE-trained GBMs on noisy macro returns optimise the wrong loss for a signal-led use; and `DEFAULT_LAGS=[1,5,22]` are trading-day counts — on a monthly frame they become 1/5/22-**month** lags (needs verification, likely a mismatch). Vol-of-target is ignored.

---

## Step 2 — Critique (weekly/monthly specifically)

- **Statistical validity / holdout size.** 24-month monthly holdout = adequate only for h=1 (~24 obs), thin for h=3 (~8 independent), useless for h=6/12 (~4/2). Weekly (5d) has more rows but heavy target overlap. Purging `horizon-1` fixes *adjacency* leakage but not the **autocorrelation/overlap** that makes the effective sample far smaller than the row count — so Sharpe/skill standard errors are badly understated.
- **Sharpe trustworthiness.** ~1.9 is implausible for copper directional bets. Drivers: overlapping returns deflate σ; no deflation for the ~10-model search; pooled single estimate hides fold-to-fold blow-ups. A √(252/5)≈7.1 or √(12/1)≈3.46 multiplier on a noisy mean amplifies the artefact.
- **Model-selection risk.** Choosing by a single high-variance metric on one short window is close to a max-of-noise estimator; expected regret is large. The refit-on-holdout step compounds it.
- **Feature/target design.** For signal-led use, h-step log-return *level* with RMSE is suboptimal vs **direction** (what you act on) or **vol-normalised return** (stabilises variance, improves loss conditioning and Sharpe interpretation). No leakage found in monthly publication-lag handling — that part is correct.
- **Sample size vs complexity.** 50-trial Optuna GBMs + stacking + regime routing + 100+ features on ~430 monthly rows is **cargo-cult complexity** for this N. Regularised linear on a curated set is the honest workhorse here. **N-BEATS/TFT are not justified** at 430 (or even at weekly N) — explicitly out of scope.

---

## Step 3 — Alternatives (each tied to a weakness)

**Evaluation upgrades (attack weaknesses #1, #2) — highest ROI**
- **Deflated Sharpe Ratio + selection-bias correction** (Bailey/López de Prado). *Why:* directly discounts the ~1.9 for the number of models tried and for non-normality. *Benefit:* honest edge estimate; kills false positives. *Effort:* S. *Risk:* will lower headline numbers (intended).
- **Per-fold metric distribution + bootstrap CIs** instead of one pooled value in `compare_models`. *Why:* exposes fold instability. *Benefit:* robust ranking. *Effort:* S. *Risk:* changes `compare_models` output shape — keep back-compatible.
- **Diebold-Mariano test vs random-walk/naive.** *Why:* is skill statistically real? *Benefit:* go/no-go gate per horizon. *Effort:* S.
- **Overlap-aware Sharpe** (compute on h-spaced non-overlapping obs, or Newey-West σ). *Why:* removes the inflation source. *Effort:* S–M.
- **Combinatorial Purged CV (CPCV)** for a *distribution* of backtest paths. *Why:* one walk-forward path ≠ evidence. *Effort:* M. *Risk:* small monthly N limits the number of combinations.

**Model-selection fix (weakness #1)**
- **Nested CV / one-standard-error rule:** select by inner-CV robust skill, evaluate once on an untouched outer window, never refit on the reporting holdout before quoting OOS. *Benefit:* removes the largest source of optimism. *Effort:* M. *Risk:* touches notebook selection cells.

**Ensembling better than thin-OOF stacking (weaknesses #1, #3)**
- **Robust combination** (equal-weight / inverse-CV-error / median of base forecasters). *Why:* stacking fails on monthly; thin OOF meta-learners overfit. *Benefit:* strictly more stable, near-free. *Effort:* S. *Risk:* low.
- **Regime-conditional weighting** using the existing HMM: weight models by regime-specific CV skill rather than hard routing. *Effort:* M.

**Target reformulation (signal-led; weakness #4)**
- **Vol-normalised return target** `target_ret/forecast_vol` (reuse `copper_vol_*`/HAR-RV). *Why:* stabilises variance, better GBM loss, cleaner Sharpe. *Effort:* S.
- **Direction-only with calibrated probability** (you already have the classifiers) ranked by AUC/Brier + economic value. *Why:* optimise the thing you trade. *Effort:* S–M.

**Models suited to short, noisy, low-N macro (weakness #3)**
- **Elastic-Net / tuned-Ridge on a curated 15–25 feature set.** *Why:* the canonical p≫n workhorse; current Ridge is α=1.0 fixed on the full matrix. *Benefit:* likely the highest-ROI *model* change for monthly. *Effort:* S.
- **Dynamic Factor Model / PCA factors** to collapse 100+ correlated predictors into a handful. *Why:* directly attacks dimensionality. *Effort:* M.
- **Bayesian structural time series / `UnobservedComponents` regression.** *Why:* small-N friendly, native intervals, partial interpretability. *Effort:* L. *Risk:* dependency/complexity — defer.
- **N-BEATS/TFT:** *not justified* at this N. Explicitly excluded.

**Feature curation (weakness #3)**
- **Curate to an economically-motivated core** (LME/SHFE inventories, real yields, DXY, China PMI/IP, term structure, COT positioning, copper momentum+vol) and **fix monthly lag units** to months. *Effort:* S–M.

**Uncertainty (secondary deliverable)**
- **Conformal prediction (split / EnbPI for time series)** for finite-sample interval coverage, replacing reliance on quantile-LGBM (no coverage guarantee, data-hungry). *Effort:* M.

---

## Step 4 — Prioritised roadmap (quality-per-effort, monthly-first)

### (1) Quick wins — S effort, do first
| # | Item | Files / functions | Test-breakage flag |
|---|------|-------------------|--------------------|
| 1 | **Deflated Sharpe + DM test + overlap-aware/Newey-West σ** as new functions | `src/evaluation.py` (add `deflated_sharpe`, `diebold_mariano`, overlap-aware sharpe helper); new `tests/test_evaluation_stats.py` | Additive — safe if `compute_metrics` keys unchanged. `test_copper_forecast.py` checks existing keys. |
| 2 | **Per-fold metric distribution + bootstrap CIs** in `compare_models` | `src/evaluation.py:300` (return per-fold table alongside pooled summary, keep old columns) | Keep return shape back-compatible or version it; notebooks consume it. |
| 3 | **Robust selection protocol**: rank by median per-fold skill / one-SE rule; stop quoting the refit-on-holdout Sharpe as OOS | Selection cells in `copper_forecast_monthly.ipynb` (+ 1d), optional helper `select_best_model()` in `evaluation.py` | Notebook-level; changes reported numbers (intended). |
| 4 | **Robust ensemble fallback** for monthly (equal / inverse-CV-error / median) | `src/models_stacking.py` (add `RobustCombiner`), wire into monthly notebook where stacking currently fails | Additive class; `test_new_models.py` unaffected. |
| 5 | **Vol-normalised return target** option + **fix monthly lag units** | `src/feature_engineering.py:416` (add `target_ret_vol_{h}` behind param), `DEFAULT_LAGS` usage for monthly | Additive params with defaults; verify against `test_copper_forecast.py` target-shape assertions. |

### (2) Medium changes — M effort
| # | Item | Files / functions | Test-breakage flag |
|---|------|-------------------|--------------------|
| 6 | **Elastic-Net (tuned α, l1_ratio) on curated feature set** as first-class monthly model | `src/models.py` (extend `LinearModel` or add `ElasticNetModel`), curated list in `feature_engineering.py` | Additive model; ensure it implements the `BaseForecaster` interface tested in `test_copper_forecast.py`. |
| 7 | **Dynamic Factor / PCA reduction** before modelling | new `src/dimensionality.py` or extend `feature_pruning.py`; wire into notebooks | Additive; `test_regime_cointegration.py` covers adjacent pruning — keep existing API. |
| 8 | **Conformal prediction intervals** | new `src/conformal.py`; replace/augment QuantileForecaster usage in notebooks | Additive module; no existing test depends on it. |
| 9 | **Nested CV** wrapper for selection | `src/evaluation.py` (new `nested_cv_select`) | Must not change `walk_forward_cv`/`PurgedTimeSeriesSplit` signatures — guarded by `test_purged_cv.py`. |
| 10 | **Regime-conditional model weighting** | `src/models_regime.py` (soft weighting variant) | Additive alongside `RegimeRouter`. |

### (3) Larger rebuilds — L effort, only if (1)+(2) show the ceiling is modelling not measurement
| # | Item | Files | Risk |
|---|------|-------|------|
| 11 | **Combinatorial Purged CV (CPCV)** for backtest-path distributions | `src/evaluation.py` (new splitter + path aggregation) | Small monthly N limits combinations; heavier compute. |
| 12 | **Bayesian structural time series** monthly model | new `src/models_bsts.py` | New dependency, slow; defer until justified. |
| 13 | **Unify selection→ensembling across all three notebooks** (nested CV everywhere) | broad `src/` + all notebooks | Touches shared engine widely; needs full test pass — highest breakage risk. |

### Cross-cutting test-suite guardrails
- `tests/test_purged_cv.py` pins `PurgedTimeSeriesSplit` / `walk_forward_cv` / `out_of_sample_backtest` geometry → **keep signatures additive** (new params with defaults).
- `tests/test_copper_forecast.py` pins `compute_metrics` keys, `build_features`, `split_features_targets`, model interfaces → **add keys, never rename/repurpose**.
- `tests/test_new_models.py`, `tests/test_regime_cointegration.py`, `tests/test_cot_data.py` → adding models/modules is safe; changing existing APIs is not.
- All three notebooks import the same `src/` → any shared-engine change must be exercised by the daily notebook too before merge.

---

## Medium items (#6–#10) — detailed implementation plan

All additive (new classes/modules or new params with old defaults), each with its own tests, none changing a signature pinned by the existing suite. **Recommended order: 6 → 8 → 7 → 9 → 10** (highest value / lowest risk first; 7 composes with 6, 9 builds on the quick-win `select_best_model`, 8 replaces the interval source in the notebooks).

### 6 — `ElasticNetModel` on a curated feature set  ✅ DONE
**Landed:** `ElasticNetModel` (scaler→ElasticNet, Optuna `tune` over `alpha`×`l1_ratio` under purged CV, `coef_` accessor) and `CuratedForecaster` (wraps any base model to train on a curated column subset — leakage-safe per-fold) in `src/models.py`; `CURATED_PREFIXES` + `curate_features()` in `src/feature_engineering.py`. Tests: `tests/test_new_models.py::TestElasticNetModel` / `::TestCuratedForecaster`, `tests/test_curated_features.py`. Both Kaggle notebooks now include `CuratedForecaster(ElasticNetModel())` in the fit / CV / OOS line-ups. Full suite: 99 passed / 10 skipped (optuna-dependent `tune` test skips locally). Original design notes below.


- **What:** a `BaseForecaster` mirroring `LinearModel`'s `StandardScaler → linear` pipeline but with `ElasticNet(alpha, l1_ratio)` and a `.tune()` method (Optuna over `alpha∈[1e-3,10]` log, `l1_ratio∈[0.1,0.95]`, scored by neg-RMSE under `PurgedTimeSeriesSplit`, same pattern as `XGBoostModel.tune`). Plus a curated core list and a `curate_features(X, base_names, lags)` helper that expands base names to the `_lag_*` columns actually present (graceful intersection).
- **Why:** the canonical p≫n workhorse. Today's `LinearModel` is Ridge with `alpha=1.0` fixed on ~140 columns; ElasticNet does embedded selection and the curated list attacks dimensionality directly — the highest-ROI *model* change for monthly.
- **Curated core (monthly):** copper momentum + `copper_vol_*`, `lme/shfe_copper_inv_chg_3` + `_level`, `real_yield_level`/`_change`, `dxy_level`/`dxy_ret_*`, `china_pmi_diffusion`, `china_mfg_yoy`, `indpro_yoy`, `t10y2y`/term spread, `gold/oil/alu` ratios, `infl_be_level`.
- **Files:** `src/models.py` (class), `src/feature_engineering.py` (`CURATED_FEATURES_MONTHLY`, `curate_features`). Notebooks: add to line-ups; optionally fit on the curated subset.
- **Tests:** `tests/test_new_models.py` — fit/predict/tune shapes; high `l1_ratio` ⇒ sparse coefficients; `curate_features` returns a subset and never errors on a missing name.
- **Risk:** must satisfy the `BaseForecaster` interface pinned by `test_copper_forecast.py`.

### 7 — Dimensionality reduction via per-fold PCA factors  ✅ DONE
**Landed:** `src/dimensionality.py::DimReducedForecaster` — wraps any base forecaster behind `StandardScaler → PCA`, fit on each `fit` call's rows only (leakage-safe per fold); supports `n_components` as an int (clamped to feasible) or a float variance threshold; exposes `explained_variance_ratio_`. Tests: `tests/test_dimensionality.py` (shape/k, clamping, variance threshold, **leakage-free PCA**, runs inside `walk_forward_cv`, composes as `DimReducedForecaster(CuratedForecaster(ElasticNetModel()))`). Both notebooks add `DimReducedForecaster(ElasticNetModel(), n_components=8)` to the fit/CV/OOS line-ups. Suite: 114 passed / 10 skipped. Original design notes below.


- **What:** `src/dimensionality.py` with `class DimReducedForecaster(BaseForecaster)` wrapping a base model + a `StandardScaler → PCA(n_components | variance threshold)`. `fit(X, y)` fits the scaler+PCA **on that call's X only**, then the base on the factor scores; `predict` transforms then delegates. `name = f"PCA{k}+{base.name}"`.
- **Why:** collapses ~140 collinear predictors into ~5–10 orthogonal factors. Implementing it as a *model wrapper* (not a global transform) means `walk_forward_cv`'s per-fold refit re-fits PCA on each fold's train only — leakage-safe by construction.
- **Files:** `src/dimensionality.py`; notebooks add it as a model (e.g. `DimReducedForecaster(LinearModel(), n_components=8)`), composes with #6 (`DimReducedForecaster(ElasticNetModel())`).
- **Tests:** `tests/test_dimensionality.py` — PCA fitted only on train rows (truncation invariance like the retvol no-leakage test); output shape; explained-variance monotonicity.
- **Risk:** additive; verify it runs inside `walk_forward_cv` and `out_of_sample_backtest`.

### 8 — Conformal prediction intervals  ✅ DONE
**Landed:** `src/conformal.py::ConformalForecaster` — split-conformal wrapper around any point forecaster, trailing-block calibration (leakage-safe), rank-based radius with a Gaussian fallback for tiny calibration sets, optional `mondrian_by` group-conditional intervals, and `.predict(X) → DataFrame[lower, median, upper]` for drop-in parity with `QuantileForecaster`; `predict_interval(alpha=...)` re-derives width from stored residuals without refitting. Tests: `tests/test_conformal.py` (coverage ≈ nominal, ordered bounds, monotone width, Mondrian, tiny-calib fallback). Both notebooks now build intervals via a `make_interval_model()` factory (conformal primary, `QuantileForecaster` fallback) in cells 37/38. Suite: 107 passed / 10 skipped; notebook smoke test gave ~87% holdout coverage at the 80% nominal (conformal is ≥-nominal by design). Original design notes below.


- **What:** `src/conformal.py` with `class ConformalForecaster(BaseForecaster)` wrapping any point model. `fit` splits the (time-ordered) training data into proper-train + a trailing calibration block, fits the base on proper-train, stores calibration residuals; `predict_interval(X, alpha)` returns a `lower/median/upper` DataFrame (same schema as `QuantileForecaster`) using the conformal residual quantile `q = ⌈(n+1)(1−α)⌉/n`. Optional Mondrian variant conditioning the quantile on `regime`.
- **Why:** distribution-free, finite-sample coverage — `QuantileForecaster` (LGBM quantile) has no coverage guarantee and is data-hungry; conformal is the right "accuracy-first + intervals secondary" tool. Because it works off point residuals it also gives **valid price intervals regardless of target scaling**, which is what makes a `retvol` headline target usable later.
- **Files:** `src/conformal.py`; notebooks swap `QuantileForecaster → ConformalForecaster(base)` in the interval cells (37/38 monthly), keeping QF as fallback.
- **Tests:** `tests/test_conformal.py` — empirical coverage ≈ nominal (e.g. 80±a few %) on synthetic; `lower ≤ median ≤ upper`; column parity with `QuantileForecaster`.
- **Risk:** match the existing `predict_interval` output columns exactly so the plotting cells are unchanged.

### 9 — Nested-CV selection wrapper  ✅ DONE
**Landed:** `src/evaluation.py::nested_cv_select` — outer `PurgedTimeSeriesSplit` folds; the inner CV + `select_best_model` choose a model on the outer-train block only, graded on the untouched outer-test fold; returns per-fold winners + pooled procedure metrics (compute_metrics keys + overlap-aware Sharpe CI, deflated Sharpe, DM-vs-naive). Takes model **factories** (fresh per fold), changes no existing signature. Tests: `tests/test_evaluation_stats.py` (structure/no-leakage, ≥2-model guard, wrapper-factory composition). Both notebooks gained a **section 9b** cell comparing the nested-CV procedure estimate vs the single-holdout pick (fast linear/curated/PCA family to stay tractable). Suite: 117 passed / 10 skipped. Original design notes below.


- **What:** `evaluation.py::nested_cv_select(models, X, y, ...)` — outer `PurgedTimeSeriesSplit` folds; inside each outer-train block run `compare_models` + `select_best_model` to choose, then score that choice on the untouched outer-test fold. Returns the per-outer-fold winners + pooled metrics **of the selection procedure**, not of a hand-picked model.
- **Why:** kills the remaining optimism of "select on the same window you then report" — estimates what the selection rule actually earns out-of-sample. Directly extends the quick-win `select_best_model`.
- **Files:** `src/evaluation.py` (new function only — must NOT touch `walk_forward_cv` / `PurgedTimeSeriesSplit` signatures pinned by `test_purged_cv.py`). Notebooks: an optional reporting cell comparing nested-CV procedure metrics vs the single-holdout OOS.
- **Tests:** extend `tests/test_evaluation_stats.py` — one selection per outer fold; pooled metrics finite; inner never sees outer-test rows (leakage check).
- **Risk:** compute cost (models × outer × inner) — keep the monthly model list small; document the cost in the cell.

### 10 — Regime-conditional model weighting  ✅ DONE
**Landed:** `src/models_regime.py::RegimeWeightedEnsemble` — blends ≥2 base models with per-regime inverse-error (out-of-fold) weights, keeping every model trained on all data and degrading to global weights when a regime is thin or unseen (no data-starved hard routing); excludes `regime_col` from inner features like `RegimeRouter`. Tests: `tests/test_regime_weighting.py` (simplex weights, **linear earns more weight in the signal regime**, missing-col guard, unseen-regime → global, OOF-infeasible → equal, naming). Notebooks: SHAP pruning now force-keeps the `regime` column, and the OOS line-up gains a **guarded** `RegimeWeightedEnsemble([LinearModel(), CuratedForecaster(ElasticNetModel())])` (appended only when `regime` is present, so it can never crash the comparison). Suite: 125 passed / 10 skipped. Original design notes below.


- **What:** `models_regime.py::class RegimeWeightedEnsemble(BaseForecaster)` — a soft alternative to `RegimeRouter`: estimate each base model's **per-regime** walk-forward skill, then at predict time blend models by their skill in the *current* regime (`regime_col` in X), falling back to global inverse-error weights when a regime is thin.
- **Why:** hard routing starves per-regime samples (`min_regime_samples=60` on ~430 rows); soft weighting uses all data and degrades gracefully. Reuses the existing 2-state HMM already in the notebooks.
- **Files:** `src/models_regime.py` (new class beside `RegimeRouter`). Notebooks: optional lineup member (the `regime` column is already added in the feature cell).
- **Tests:** `tests/test_regime_cointegration.py` — per-regime weights sum to 1; prediction shape; unseen-regime fallback.
- **Risk:** must thread/exclude `regime_col` from the inner feature matrix exactly as `RegimeRouter` does.

### Cross-cutting decision to settle before #8/#6 land
Whether to make the monthly **headline** target `retvol` (vol-normalised). It improves loss conditioning but breaks the `exp(return)` price reconstruction in the forecast cells. Cleanest path: keep `ret` as headline until **#8 (conformal)** is in, then optionally switch — conformal price intervals come from residuals and are agnostic to target scaling, so they remove the blocker. Until then `retvol` stays an opt-in experiment (columns already emitted).

## Verification
1. **Unit tests:** `pytest tests/` stays green after each item; new stats helpers get `tests/test_evaluation_stats.py` (deflated Sharpe ≤ raw Sharpe; DM p-value sane on synthetic data; overlap-aware σ ≥ naive σ on overlapping series).
2. **Selection honesty check:** on monthly, confirm the quoted OOS edge is computed on a window **not used for selection and not in the final refit**; deflated Sharpe should drop the ~1.9 substantially (a credible monthly edge is < ~0.8).
3. **Skill gate per horizon:** DM test vs random-walk must reject at h=1/3 for a model to be promoted; document that h=6/12 are under-powered on a 24m holdout (report with explicit CI, don't over-claim).
4. **Ablation:** curated-feature Elastic-Net vs full-feature GBM on identical CPCV/nested-CV paths — compare per-fold skill distributions, not single pooled Sharpe.
5. **Interval calibration:** conformal interval empirical coverage ≈ nominal (e.g. 80% band covers ~80% OOS).
6. **No regression:** re-run the daily notebook end-to-end to confirm shared-engine changes didn't break it.

## Open items — RESOLVED during implementation
- **Monthly CV config (confirmed):** `copper_forecast_monthly.ipynb` cell config = `initial_train_size: 60`, `cv_step_size: 3`, `rolling_window: 120`, `holdout_size: 24` (notebook lines 237–248). "120/3" = rolling-window 120 / step 3; the "60" is `initial_train_size`. A clean `X_dev`/`X_hold` split exists (lines 5876–5877).
- **No monthly lag-unit bug:** the monthly notebook explicitly overrides `lags=[1,3,6,12]`, `return_lags=[1,3,6,12]`, `vol_windows=[3,6,12]` (all months, lines 237–239) — `DEFAULT_LAGS` is never used on monthly frames. **Quick win #5 is therefore scoped to the vol-normalised target only** (the lag-fix is dropped as unnecessary).
