# copper-forecast

> **Professional-grade copper spot price forecasting model in Python**

A modular, production-quality pipeline for forecasting LME copper prices using an ensemble
of machine-learning models, fed by a rich feature set covering supply/demand fundamentals,
macro/financial overlays, and technical indicators.

---

## ✨ Key Features

| Capability | Details |
|---|---|
| **Data ingestion** | yfinance (free) + FRED API (free) — 10+ years of daily data |
| **Feature engineering** | 100+ features: price-derived, cross-asset, macro, calendar, lagged |
| **Models** | Naive · Ridge · XGBoost · LightGBM · Ensemble |
| **Tuning** | Optuna walk-forward hyperparameter search |
| **Validation** | Expanding-window CV + out-of-sample backtest |
| **Forecast** | Point estimate + 80% confidence interval (quantile regression) |
| **Explainability** | SHAP beeswarm & importance bar charts |
| **Scenario analysis** | What-if engine with 7 built-in scenarios + custom shocks |
| **Export** | CSV + JSON output for Power BI / Tableau / APIs |

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

### Optional: FRED API key

Register for a free key at <https://fred.stlouisfed.org/docs/api/fred/>

```bash
export FRED_API_KEY="your_key_here"
```

If no key is provided, synthetic placeholder series are used (safe for testing).

---

## 🚀 Quick Start

### Option 1 — Jupyter Notebook (recommended)

```bash
jupyter notebook copper_forecast.ipynb
```

Run all cells top-to-bottom.  The notebook is self-contained and will:
1. Download data automatically
2. Engineer all features
3. Train and tune models (set `optuna_trials=0` in config to skip tuning)
4. Run walk-forward CV and OOS backtest
5. Produce forecast with 80% CI and interactive Plotly charts
6. Run scenario analysis
7. Export results to `./outputs/`

### Option 2 — Python scripts

```python
from src.data_ingestion import load_data
from src.feature_engineering import build_features, split_features_targets
from src.models import XGBoostModel, LGBMModel, EnsembleModel
from src.evaluation import walk_forward_cv, compute_metrics

# Load data
df = load_data(start="2015-01-01")

# Build features
feats = build_features(df)
X, y_ret, y_price = split_features_targets(feats, horizon=22)

# Train ensemble
xgb = XGBoostModel()
lgb = LGBMModel()
ensemble = EnsembleModel([xgb, lgb])
ensemble.fit(X, y_ret)

# Forecast
preds = ensemble.predict(X.tail(22))
```

---

## 📁 Repository Structure

```
copper-forecast/
├── copper_forecast.ipynb      # Main notebook (start here)
├── requirements.txt           # Python dependencies
├── README.md
├── src/
│   ├── data_ingestion.py      # yfinance + FRED download pipeline
│   ├── feature_engineering.py # Technical, cross-asset, macro features
│   ├── models.py              # Naive, Ridge, XGBoost, LightGBM, Ensemble, Quantile
│   ├── evaluation.py          # Walk-forward CV, OOS backtest, metrics
│   ├── visualization.py       # Plotly interactive charts
│   └── scenario_analysis.py   # What-if scenario engine
└── tests/
    └── test_copper_forecast.py  # Unit tests (pytest)
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected: **31 tests passed**.

---

## 🔑 Data Sources

| Source | Series | Key |
|---|---|---|
| [yfinance](https://github.com/ranaroussi/yfinance) | HG=F, DX-Y.NYB, GC=F, ALI=F, CL=F, CNYUSD=X, ^GSPC | Free |
| [FRED](https://fred.stlouisfed.org/) | INDPRO, DFII10, T10YIE, MANEMP, M2SL | Free (API key) |

---

## 📊 Feature Engineering Summary

| Group | Features |
|---|---|
| **Price-derived** | Log returns (1/5/22d), realised vol (5/22/66d), z-score vs 200d MA, RSI-14, MACD, Bollinger width |
| **Cross-asset** | Gold/Cu ratio, Oil/Cu ratio, Al–Cu spread %, DXY level & 22d return, CNY/USD, S&P 500 return |
| **Macro** | Industrial production YoY, real yield level & 22d change, inflation breakeven |
| **Calendar** | Month sin/cos cyclical encoding, Chinese New Year flag |
| **Lags** | 1, 5, 22-day lags of all above features |

---

## 🎯 Model Performance (indicative, 2015–present)

Walk-forward CV metrics depend on the market regime and data window.
Typical out-of-sample results on copper daily return prediction:

| Model | RMSE (log ret) | Directional Acc. |
|---|---|---|
| Naive (RW) | baseline | ~50% |
| Ridge | slightly better | ~52% |
| XGBoost | better | ~54–56% |
| LightGBM | competitive | ~54–56% |
| **Ensemble** | **best** | **~55–57%** |

*Note: commodity price forecasting is inherently difficult. A directional accuracy above 53%
is economically significant at the daily frequency.*

---

## ⚙️ Configuration

All settings live in the `CFG` dictionary at the top of the notebook:

```python
CFG = {
    'start_date':         '2015-01-01',
    'forecast_horizon':   22,      # 1-month ahead
    'all_horizons':       [5, 22, 66],
    'lags':               [1, 5, 22],
    'initial_train_size': 504,
    'cv_step_size':       22,
    'holdout_size':       252,
    'ci_alpha':           0.80,
    'optuna_trials':      50,      # 0 to skip tuning
    'fred_api_key':       None,    # or set FRED_API_KEY env var
    'random_seed':        42,
    'output_dir':         './outputs',
}
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).
