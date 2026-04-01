# copper-forecast

> **Professional-grade copper spot price forecasting model in Python**

A modular, production-quality pipeline for forecasting Comex copper prices using an ensemble
of econometric and machine-learning models, fed by a rich feature set covering supply/demand
fundamentals, macro/financial overlays, COT positioning, cointegration signals, regime
detection, and technical indicators.

---

## Key Features

| Capability | Details |
|---|---|
| **Data ingestion** | yfinance + FRED API + CFTC COT (via Nasdaq Data Link) — 15+ years of daily data from 2010 |
| **Feature engineering** | 120+ features: price-derived, cross-asset, macro, calendar (quarter-end, US holidays, options expiry), COT positioning, cointegration ECTs, HMM regime dummies |
| **Models** | Naive · Ridge · XGBoost · LightGBM · ARIMAX · Prophet · Hybrid (ARIMAX+LGBM) · Weighted Ensemble · Stacking Ensemble (meta-learner) |
| **Tuning** | Optuna walk-forward hyperparameter search |
| **Validation** | Expanding-window CV + out-of-sample backtest with signal Sharpe metric |
| **Forecast** | Point estimate + 80% CI from quantile regression, ARIMAX intervals, and Prophet uncertainty |
| **Regime detection** | 3-state Gaussian HMM (bear / sideways / bull) |
| **Cointegration** | Engle-Granger tests + rolling error-correction terms vs gold, aluminium, oil, DXY, CNY |
| **Feature pruning** | Automatic SHAP-based removal of low-importance features |
| **Explainability** | SHAP beeswarm & importance bar charts |
| **Scenario analysis** | What-if engine with 7 built-in scenarios + custom shocks + sensitivity sweeps |
| **Export** | CSV + JSON output for Power BI / Tableau / APIs |

---

## Installation

```bash
pip install -r requirements.txt
```

### Core dependencies

The following are always required: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `yfinance`, `plotly`, `optuna`.

### Optional dependencies

| Package | Used for | Install |
|---|---|---|
| `xgboost` | XGBoostModel, Ensemble, SHAP | `pip install xgboost` |
| `lightgbm` | LGBMModel, Ensemble, QuantileForecaster, Hybrid residual model | `pip install lightgbm` |
| `prophet` | ProphetModel (Bayesian structural time series) | `pip install prophet` |
| `hmmlearn` | RegimeDetector (HMM market regime classification) | `pip install hmmlearn` |
| `nasdaqdatalink` | COT positioning data from CFTC | `pip install nasdaqdatalink` |
| `shap` | Feature importance & explainability | `pip install shap` |
| `fredapi` | FRED macro data | `pip install fredapi` |

All optional dependencies degrade gracefully — the pipeline runs with synthetic fallbacks or skips the component if a package is missing.

### API keys (optional)

```bash
# FRED macro data (free key from https://fred.stlouisfed.org/docs/api/fred/)
export FRED_API_KEY="your_key_here"

# Nasdaq Data Link for COT data (free key from https://data.nasdaq.com/)
export NASDAQ_DATA_LINK_API_KEY="your_key_here"
```

If no keys are provided, synthetic placeholder series are used (safe for testing and development).

---

## Quick Start

### Option 1 — Jupyter Notebook (recommended)

```bash
jupyter notebook copper_forecast.ipynb
```

Run all cells top-to-bottom. The notebook is self-contained and will:

1. Download data from yfinance, FRED, and COT (2010-present)
2. Run cointegration tests and compute error-correction terms
3. Detect market regimes with a 3-state HMM
4. Engineer 120+ features (price, cross-asset, macro, calendar, COT, structural)
5. Auto-prune low-importance features via SHAP
6. Train and tune 9 models (set `optuna_trials=0` to skip tuning)
7. Run walk-forward CV and out-of-sample backtest with signal Sharpe
8. Produce forecast with 80% CI from multiple models
9. Generate SHAP explainability plots
10. Run scenario analysis with 7 built-in + custom scenarios
11. Export results to `./outputs/`

### Option 2 — Python scripts

```python
from src.data_ingestion import load_data
from src.feature_engineering import build_features, split_features_targets
from src.cointegration import add_cointegration_features
from src.models import XGBoostModel, LGBMModel, EnsembleModel
from src.models_arimax import ARIMAXModel
from src.models_hybrid import HybridModel
from src.evaluation import walk_forward_cv, compute_metrics

# Load data (2010-present, with COT positioning)
df = load_data(start="2010-01-01", include_cot=True)

# Add cointegration ECT features
df, coint_results = add_cointegration_features(df)

# Build features (includes 1-day, 1-week, 1-month, 3-month horizons)
feats = build_features(df, horizons=[1, 5, 22, 66])
X, y_ret, y_price = split_features_targets(feats, horizon=22)

# Train individual models
arimax = ARIMAXModel()
arimax.fit(X, y_ret)

hybrid = HybridModel()  # ARIMAX backbone + LightGBM residual correction
hybrid.fit(X, y_ret)

ensemble = EnsembleModel([XGBoostModel(), LGBMModel()])
ensemble.fit(X, y_ret)

# Forecast with prediction intervals
from src.models_arimax import ARIMAXModel
ci = arimax.predict_interval(X.tail(22), alpha=0.80)  # lower, median, upper
```

---

## Repository Structure

```
copper-forecast/
├── copper_forecast.ipynb          # Main notebook (start here)
├── requirements.txt               # Python dependencies
├── README.md
├── src/
│   ├── data_ingestion.py          # yfinance + FRED + COT download pipeline
│   ├── feature_engineering.py     # Technical, cross-asset, macro, calendar features
│   ├── models.py                  # Naive, Ridge, XGBoost, LightGBM, Ensemble, Quantile
│   ├── models_arimax.py           # ARIMAX with prediction intervals
│   ├── models_prophet.py          # Prophet with exogenous regressors
│   ├── models_hybrid.py           # Hybrid: ARIMAX backbone + ML residual correction
│   ├── models_stacking.py         # Stacking ensemble with Ridge meta-learner
│   ├── evaluation.py              # Walk-forward CV, OOS backtest, signal Sharpe
│   ├── visualization.py           # Plotly interactive charts + regime overlay
│   ├── scenario_analysis.py       # What-if scenario engine
│   ├── cointegration.py           # Engle-Granger tests + rolling ECT features
│   ├── regime_detection.py        # HMM regime detector (3-state)
│   ├── cot_data.py                # CFTC COT positioning data ingestion
│   └── feature_pruning.py         # SHAP-based automatic feature pruning
└── tests/
    ├── test_copper_forecast.py    # Core module tests (38 cases)
    ├── test_new_models.py         # ARIMAX, Prophet, Hybrid, Stacking tests (12 cases)
    ├── test_regime_cointegration.py  # Regime + cointegration tests (9 cases)
    └── test_cot_data.py           # COT data tests (4 cases)
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected: **56+ tests passed** (some tests skip if optional dependencies like xgboost, lightgbm, prophet, or hmmlearn are not installed).

---

## Data Sources

| Source | Series | Key |
|---|---|---|
| [yfinance](https://github.com/ranaroussi/yfinance) | HG=F (copper), DX-Y.NYB (DXY), GC=F (gold), ALI=F (aluminium), CL=F (oil), CNYUSD=X, ^GSPC (S&P 500) | Free |
| [FRED](https://fred.stlouisfed.org/) | INDPRO, DFII10, T10YIE, MANEMP, M2SL | Free (API key) |
| [CFTC COT](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm) | Commercial/non-commercial net positioning, open interest | Free (via Nasdaq Data Link API key) |

---

## Feature Engineering Summary

| Group | Features |
|---|---|
| **Price-derived** | Log returns (1/5/22d), realised vol (5/22/66d), z-score vs 200d MA, RSI-14, MACD, Bollinger width |
| **Cross-asset** | Gold/Cu ratio, Oil/Cu ratio, Al-Cu spread %, DXY level & 22d return, CNY/USD, S&P 500 return |
| **Macro** | Industrial production YoY, real yield level & 22d change, inflation breakeven |
| **Calendar** | Month sin/cos, Chinese New Year flag, quarter-end flag, US holiday flag, options expiry flag |
| **COT positioning** | Commercial net, non-commercial (speculative) net, open interest, speculative ratio |
| **Cointegration** | Error-correction terms (rolling 252d OLS) for cointegrated copper-asset pairs |
| **Regime** | HMM state labels (bear/sideways/bull) + one-hot encoding |
| **Lags** | 1, 5, 22-day lags of all above features |

---

## Models

| Model | Type | Prediction Intervals | Notes |
|---|---|---|---|
| **Naive (RW)** | Baseline | No | Always predicts 0 return |
| **Ridge** | Linear | No | StandardScaler + L2 regularisation |
| **XGBoost** | Tree ensemble | Via QuantileForecaster | Optuna-tuned |
| **LightGBM** | Tree ensemble | Via QuantileForecaster | Optuna-tuned |
| **ARIMAX** | Econometric | Yes (built-in) | `SARIMAX(p,0,q)` with 4 exogenous features |
| **Prophet** | Bayesian structural | Yes (built-in) | Optional dependency |
| **Hybrid** | Econometric + ML | Yes | ARIMAX backbone + LightGBM residual correction |
| **Weighted Ensemble** | Averaging | Via QuantileForecaster | Equal-weight XGBoost + LightGBM |
| **Stacking Ensemble** | Meta-learner | No | Ridge trained on walk-forward OOF predictions |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE | Root mean squared error of return predictions |
| MAE | Mean absolute error |
| MAPE | Mean absolute percentage error |
| Directional accuracy | % of predictions with correct sign |
| **Signal Sharpe** | Annualised Sharpe ratio of a long/short strategy based on predicted direction |
| **Information ratio** | Excess return vs naive benchmark per unit tracking error |

---

## Model Performance (indicative, 2010-present)

Walk-forward CV metrics depend on market regime and data window.
Typical out-of-sample results on copper 22-day return prediction:

| Model | RMSE (log ret) | Directional Acc. | Signal Sharpe |
|---|---|---|---|
| Naive (RW) | baseline | ~50% | 0.0 |
| Ridge | slightly better | ~52% | ~0.2 |
| XGBoost | better | ~54-56% | ~0.4-0.6 |
| LightGBM | competitive | ~54-56% | ~0.4-0.6 |
| ARIMAX | moderate | ~52-54% | ~0.2-0.4 |
| Hybrid | better | ~54-56% | ~0.4-0.6 |
| **Stacking Ensemble** | **best** | **~55-57%** | **~0.5-0.7** |

*Note: commodity price forecasting is inherently difficult. A directional accuracy above 53%
is economically significant at the daily/monthly frequency.*

---

## Configuration

All settings live in the `CFG` dictionary at the top of the notebook:

```python
CFG = {
    'start_date':        '2010-01-01',
    'forecast_horizon':  22,              # 1-month ahead (trading days)
    'all_horizons':      [1, 5, 22, 66],  # 1-day, 1-week, 1-month, 3-month
    'lags':              [1, 5, 22],
    'initial_train_size': 504,
    'cv_step_size':       22,
    'holdout_size':       252,
    'ci_alpha':           0.80,
    'optuna_trials':      50,             # 0 to skip tuning
    'fred_api_key':       None,           # or set FRED_API_KEY env var
    'nasdaq_api_key':     None,           # or set NASDAQ_DATA_LINK_API_KEY env var
    'random_seed':        42,
    'output_dir':         './outputs',
}
```

---

## License

MIT — see [LICENSE](LICENSE).
