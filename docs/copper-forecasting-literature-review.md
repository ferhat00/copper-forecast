# Non-Black-Box Approaches to Weekly & Monthly Copper Price Forecasting: A Literature Review

*Internal research note for the copper-price-forecasting team. Scope: interpretable / auditable methods at the 1-week (5d), 1-month (22d) and 3-month (66d) horizons, benchmarked against the random walk and the copper futures curve. Black-box ML (XGBoost/LightGBM, deep nets) is referenced only as a complement and contrast.*

---

## 1. Framing: why copper is hard, and what "interpretable" buys us

**Copper is close to a martingale at short horizons.** The single most robust empirical fact in this literature is that the driftless random walk (RW) in log-price — tomorrow's expected log-price equals today's — is extremely hard to beat out-of-sample (OOS) at daily-to-monthly horizons. This is the direct prediction of weak-form market efficiency: in a financialized, liquid, exchange-traded market, public information is already in the price, so the conditional mean of next week's price is approximately this week's. Diaz, Hansen & Cabrera (2020, *Resources Policy* 69) demonstrate this head-on for copper: the RW produced the best OOS forecasts in the short and medium run, with tree-based learners becoming competitive only at distant (~2-year) horizons. Reeve & Vigfusson (2011, Fed IFDP 1025) find copper's 3-month futures-vs-RW relative MSE is essentially **1.00 (a tie)**, deteriorating to **1.08 (full 1990–2010 sample) / 1.20 (2003–2010 subsample)** a year out — i.e. futures were *worse* than the RW for copper in the late 2000s.

**Why copper is structurally hard:**

- **Financialization.** Since the mid-2000s, copper trades as a macro/risk asset (ETFs, CTA flows, index participation), so its price responds to the USD, real yields and global risk appetite as much as to physical balance. This both adds predictable cross-asset structure *and* injects speculative noise that decouples price from fundamentals episodically (e.g. the 2021–22 LME squeezes).
- **"Dr Copper" / China demand.** China is ~50–60% of refined demand, so copper is a barometer of global (especially Chinese) industrial activity. The relevant demand data — industrial production, PPI — are monthly, lagged and revised, which is precisely why fundamentals help at *monthly* (not weekly) horizons and why vintage discipline matters.
- **USD link.** Copper is USD-priced; a stronger dollar mechanically pressures the price and tightens financial conditions. DXY and USD/CNY are first-order, fast-moving drivers usable even at weekly frequency.
- **Regime instability.** The China super-cycle, 2008, COVID, and the 2021–22 squeezes are structural breaks that destabilize cointegrating relationships, GARCH persistence, and regression coefficients. Any static-coefficient model is fighting this.
- **Weak price seasonality.** Unlike agriculturals or natural gas, copper *price* has essentially no strong calendar seasonality (any seasonality lives in demand proxies, not the price), so SARIMA/Holt-Winters seasonal machinery adds little.

**What "non-black-box / interpretable" buys a forecasting team:**

1. **Auditability** — every coefficient, error-correction speed, regime probability, or posterior inclusion probability is inspectable and defensible to a risk committee or regulator. Contrast XGBoost feature-importance, which is opaque and unstable.
2. **Scenario analysis** — structural VARs/VECMs and supply-demand balances answer "if DXY rises 5% and China IP slows, where does copper go?" — black boxes cannot.
3. **Honest prediction intervals** — GARCH-family, GAMLSS, Gaussian processes, Bayesian state-space models and quantile regression produce *calibrated* densities and VaR you can backtest, not just point guesses.
4. **Economic intuition** — error-correction terms ("rich/cheap vs fair value"), the futures basis ("backwardation = physical tightness"), and regime labels ("85% probability high-volatility state") map to stories a PM can act on.
5. **Regulatory / governance comfort** — interpretable models are easier to validate, monitor for drift, and explain in model-risk frameworks.

The honest meta-finding: **interpretable models do beat the RW for copper, but mainly (a) at monthly+ horizons and (b) when they use exogenous drivers.** The clearest example is Buncic & Moretto (2015), whose dynamic model averaging over fundamentals + financial conditioning achieves OOS R² vs RW up to ~18.5% at 1 month. At weekly horizons, interpretable methods are best deployed for disciplined feature selection and calibrated intervals, not for large point-accuracy gains.

---

## 2. Master comparison table

| Model family | Best horizon | What's interpretable | Key copper inputs | Beats RW OOS? | Data / effort cost |
|---|---|---|---|---|---|
| **Random walk (no drift)** | 5d–22d | The last price (1 number) | Spot HG=F / LME | *Is* the benchmark | Trivial |
| **Futures-curve / cost-of-carry** | 22d–66d | Traded forward price; basis = carry − convenience yield | HG/LME forward curve, rates | **Mixed**: tie/modest at short h; conditional win when basis large | Trivial (market data) |
| **ARIMA / SARIMA** (univariate) | 22d | AR/MA coeffs, integration order *d* (nests RW) | Copper log-price only | Rarely at short h (collapses to RW) | Low |
| **ARIMAX / SARIMAX** | 22d | + exogenous elasticities, Gaussian intervals | + DXY, basis, COT, China IP, inventories | Sometimes, via the X | Low–med |
| **ETS / damped-trend / Theta** | 22d | Level, slope, smoothing params; Theta = SES+½-slope drift | Copper price only | Marginally; "gentle drift" beats RW-with-drift | Low |
| **Structural / UC state-space (Kalman)** | 22d–66d | Trend, slope, cycle, time-varying regression states + honest intervals | Copper + drivers as regression states | Modestly, monthly | Med |
| **Single-eq ECM (Engle-Granger)** | 22d | Long-run elasticities, ECT speed-of-adjustment | Gold/Al/oil/DXY/CNY, basis, inventory | Mixed; static often not | Low–med |
| **Johansen VECM** | 22d–66d | Cointegrating vectors, α-loadings, IRFs, weak-exogeneity | Base-metals complex + macro | Often *not* OOS despite in-sample fit | Med–high |
| **ARDL / NARDL bounds** | 22d | Long-run elasticities, asymmetry, ECT | Mixed I(0)/I(1): DXY, oil, real yields, IP | Mixed | Low–med |
| **DFM / FAVAR** | 22d–66d | Named factors (global demand, metals, USD), loadings, IRFs | Large macro/commodity panel | Modest (~5–7% for aggregates) | High (panel) |
| **DMA / DMS (TVP)** | 22d–66d | Time-varying inclusion probs & coefficients | ~16–18 fundamentals+financial | **Yes** — best copper result (R²≈18.5%) | Med |
| **BMA** | 22d | Posterior inclusion probabilities | Copper fundamentals/uncertainty/financial | Yes for *volatility*; thinner for level | Med |
| **BVAR (Minnesota)** | 22d–66d | Shrunk coeffs, IRFs, FEVD, scenario fans | Copper + cross-asset + macro endogenous | Fragile/horizon-dependent; best for scenarios | Med–high |
| **GARCH family (vol)** | 1d–5d | Persistence, leverage sign, tail df | Copper returns (+GARCH-X drivers) | N/A (forecasts variance) | Low |
| **HAR-RV (vol)** | 5d–22d | Daily/weekly/monthly trader-horizon betas, jumps | Intraday HG=F realized vol | **Best for daily copper RV** | Low (needs intraday) |
| **GARCH-MIDAS (vol)** | 22d–66d | θ = signed macro→long-run-vol elasticity | Daily copper + monthly PPI/IP/rates/EPU | **Best monthly vol (MCS)** | Med |
| **MIDAS (mean)** | 22d–66d | Slope + plottable lag-weight curve | Daily DXY/yields/basis/COT → monthly target | Some evidence (nowcasting) | Med |
| **Quantile / MIDAS-QR** | 5d–22d | Per-quantile driver sensitivities; VaR/ES | Realized vol, DXY, basis, COT | N/A (intervals/VaR) | Low–med |
| **Penalized regression (LASSO/EN/adaptive)** | 22d | Sparse signed coefficients; selection path | Full high-dim feature set | Strong analog evidence (oil); copper selection engine | Low |
| **GAM / penalized splines** | 22d | Per-driver smooth response curves + CI | Real yields, IP, inventory, basis, DXY | Untested on copper (gap) | Med |
| **Gaussian process regression** | 5d–22d | Kernel structure; ARD length-scales; **calibrated intervals** | Full driver set as kernel inputs | Not vs RW (autoregressive studies) | Med (O(n³)) |
| **Markov-switching AR/VAR** | 22d | Regime means/vols, transition matrix, smoothed P(state) | Copper + drivers, regime-dependent | In-sample/density yes; OOS mean weak | Med |
| **TAR/SETAR, M-TAR ECM** | 22d | Observable threshold, asymmetric adjustment speeds | ECT vs fair value, basis, momentum | In-sample asymmetry; OOS modest | Med |
| **STAR/LSTAR/ESTAR** | 22d | Transition location/speed, transition variable | Exogenous transition (USD, inflation) | **Mostly inferior to linear AR** for commodities | Med |
| **Single decision tree / RuleFit** | 22d | Decision path / sparse rule list | Threshold effects (COT, inventory, basis) | Trees **lose to RW** for copper | Low–med |
| **Symbolic regression (BSR/GP)** | 22d | Explicit closed-form equation | Driver set | **Did not beat benchmarks** (oil) | High |
| **Wavelet/EMD/VMD/STL + ARIMA** | 22d | Trend/cycle/noise bands; STL trend+seasonal | Copper price (univariate) | In-sample yes; **leakage-risk inflates** OOS | Med (leakage-safe = high) |
| **Forecast combination (1/N, GR)** | any | The weight vector | Component forecasts | Robust; "puzzle": 1/N hard to beat | Trivial |
| **Structural supply-demand balance** | 66d+ | Mine/scrap/China/inventory accounting, elasticities | ICSG/SHFE balances, China consumption | Strategic horizon only | High (data-lagged) |

---

## 3. Model-family sections

### 3.1 Univariate time-series econometrics & the random-walk benchmark

**Methods & mechanism.** The RW (no drift) sets ŷ_{t+h}=y_t in log-levels; its interval widens with √h. ARIMA(p,d,q) models the *d*-differenced log-price as AR(p)+MA(q); SARIMA adds seasonal (P,D,Q)ₛ; ARIMAX/SARIMAX add exogenous regressors X with readable elasticities. Crucially, **ARIMA(0,1,0) = the random walk**, so ARIMA literally nests the benchmark and lets you test how far copper departs from a martingale. ETS/Holt-Winters formalize exponential smoothing as state-space models (Hyndman, Koehler, Snyder & Grose 2002) with inspectable level/slope states; **damped-trend ETS** shrinks extrapolated trend toward flat, avoiding the RW-with-drift trap. The **Theta method** (Assimakopoulos & Nikolopoulos 2000) won the M3-Competition and was proven by Hyndman & Billah (2003) to equal **SES with drift = half the slope of a fitted linear trend** — a principled "gentle drift" middle ground. **Structural/unobserved-components (UC) state-space models** (Harvey 1989) decompose log-price into stochastic level + slope + cycle + irregular via the Kalman filter, with time-varying regression states — the most economically readable family, and essentially what the repo's Prophet (Bayesian structural time series) approximates.

**What's interpretable.** AR coefficients (persistence), MA coefficients (shock memory), *d* (order of integration), exogenous betas (driver elasticities), ETS smoothing parameters (how fast the model forgets), the Theta drift, and the UC trend/slope/cycle states — all with closed-form or Kalman-derived intervals.

**Weekly vs monthly.** At weekly frequency the ARMA structure of copper *returns* is weak, so pure ARIMA collapses toward ARIMA(0,1,0)=RW; value comes almost entirely from the X in ARIMAX (basis, COT, USD). At monthly frequency auto-ARIMA on log-prices and ARIMAX with macro/inventory regressors are natural, strong baselines. Damped-trend ETS and Theta are top-tier *automatic* univariate methods (M3/M4 evidence) best used as low-maintenance ensemble members. UC models suit monthly/3-month horizons where a slowly evolving trend can add over the RW — but **lean on the trend and regression states, not a fitted cycle**: Cuddington & Jerrett (2008, IMF Staff Papers) find copper's price trend is stochastic and its cyclical/"super-cycle" component is hard to identify precisely (their band-pass/UC decomposition is *consistent with* three—possibly four—super-cycles, but the component is unstable enough that cycle-based long-horizon point forecasting is fragile).

**Strengths / weaknesses.** Transparent, fast, nests the RW, calibrated Gaussian intervals; ARIMAX cleanly absorbs copper drivers. But all assume linearity and (post-differencing) homoskedasticity — violated by copper's volatility clustering and regime shifts; honest turbulent-period intervals need GARCH errors (§3.3); SARIMA/Holt-Winters seasonal terms add little for copper price.

**Copper drivers used.** ARIMAX/SARIMAX ingest LME/COMEX/SHFE inventories, DXY, USD/CNY, China IP, gold/aluminium/oil cross-prices, real yields, breakevens, CFTC COT, and the futures basis as readable exogenous regressors.

**Key evidence.** Diaz, Hansen & Cabrera (2020) — RW best in short/medium run for copper. Reeve & Vigfusson (2011) — futures beat RW "not by a large margin," and both beat RW-with-drift. Kahraman & Akay (2023, *Mineral Economics* 36(3)) — damped-trend ETS best for copper on *annual* data (a long-horizon, trend-dominated result that does **not** contradict short-horizon RW dominance). Kriechbaumer, Angus, Parsons & Rivas Casado (2014) — wavelet-ARIMA improves 1-month metal forecasts over plain ARIMA. Oikonomou & Damigos (2025, *Mineral Economics* 38(1), 37–49) — a LightGBM-ARIMA ensemble beats ARIMA/ETS alone for copper 6-month returns (ARIMA as the interpretable linear core).

> **Verifier note:** The benchmarking survey originally mis-cited as "Wang, D. et al. (2021)" is **Kwas & Rubaszek / Rubaszek, M. (2021), "Forecasting Commodity Prices: Looking for a Benchmark," *Forecasting* (MDPI) 3(2), 27** — sole/lead author Rubaszek, not Wang. The COMEX-copper NN-vs-ARIMA paper mis-cited as "Garcia & Kristjanpoller" is **Sánchez Lasheras, de Cos Juez, Suárez Sánchez, Krzemień & Riesgo Fernández (2015), *Resources Policy* 45, 37–43** — a peer-reviewed article (not a working paper) in which MLP/Elman neural nets beat ARIMA (no RW benchmark; cite only as a contrast, since NNs are the opaque method we avoid).

---

### 3.2 Multivariate & cointegration econometrics (ECM, VECM, ARDL, DFM/FAVAR)

**Methods & mechanism.** The central interpretable object is the **error-correction term (ECT)** — the gap between today's copper price and its long-run equilibrium with cointegrated partners. A negative ECT loading means mean-reversion; its magnitude is a *literal monthly speed-of-adjustment* (a loading of −0.10 ⇒ ~10% of any disequilibrium closes per month).

- **Engle-Granger single-equation ECM** (Engle & Granger 1987): OLS long-run regression → residual = "distance from fair value" → return equation on lagged ECT + short-run terms. This is exactly the repo's rolling-ECT setup vs gold/aluminium/oil/DXY/CNY.
- **Johansen VECM**: system ML estimation finding cointegration rank *r*, β-vectors (named equilibria), α-loadings (which variable adjusts), weak-exogeneity tests ("does copper lead or follow?"), and IRFs/FEVD.
- **ARDL/NARDL bounds** (Pesaran, Shin & Smith 2001): handles *mixed* I(0)/I(1) regressors — ideal for copper's heterogeneous driver set (prices ~I(1), real yields ~I(0)) — with a bounds F-test and conditional ECM; NARDL adds asymmetry (does copper react more to USD strength than weakness?).
- **DFM/FAVAR**: compress a large macro/commodity panel into a few *named* factors (global activity, metals, USD), forecast copper as a loading (DFM) or embed it in a factor-augmented VAR (FAVAR) for structural decomposition.

**Weekly vs monthly.** These are **monthly tools**. The equilibrium pull and the informative macro panels (China/global IP, real yields, breakevens) are monthly-scale; at 5d the ECT barely moves and noise dominates, so weekly value comes only via fast drivers (basis, COT, DXY). DFM/FAVAR are unsuitable weekly (informative series are monthly).

**Strengths / weaknesses.** Maximal driver usage with readable elasticities and adjustment speeds; VECM encodes genuine long-run economics and answers lead/follow questions; ARDL fits the repo's mixed feature set cheaply. But **structural breaks are the dominant failure mode** — cointegrating rank, β-vectors and adjustment speeds are unstable across regimes (China super-cycle, 2008, COVID, 2021 squeezes), so in-sample cointegration frequently fails to translate into OOS RW-beating. VECMs are parameter-hungry and overfit-prone weekly. Make ECT loadings **rolling/recursive, not static**, and pair with the repo's HMM for regime-conditional ECMs.

**Key evidence.** **Buncic & Moretto (2015, *North American Journal of Economics and Finance* 33, 1–38)** — the strongest credible monthly copper result: DMA/DMS over ~16–18 predictors beat the RW with OOS R² up to **18.5% (DMA)** / 13.7% (DMS) at 1 month (Clark-West p = 0.002/0.013), even plain expanding-window OLS-with-fundamentals beat the RW (~10% R²), with gains concentrated in 2008 and IP/convenience-yield coefficients *changing sign* over the period — but note this is **TVP/model-averaging, not a static VECM**. Galán-Gutiérrez, Labeaga & Martín-García (2023, *Resources Policy* 81) — base-metals price + futures-structure matrix is cointegrated (high prices ↔ backwardation), framed as structural/relative-value insight. Lombardi, Osbat & Schnatz (2010, ECB WP 1170) — FAVAR extracts metals & food factors (copper contributes heavily to the metals factor); exchange rates and activity drive prices, with notably *weak* oil-to-non-oil and interest-rate spillovers. Bilgin & Ellwanger (2017, BoC SAN 2017-12) and Basistha et al. (2024, *J. Forecasting*) — DFMs give interpretable demand/supply decompositions and modest OOS gains (~5–7% at 1–6m for commodity aggregates; the ~20% figure is fertilizer-specific, copper benefits via the metals block). World Bank PRWP 10611 (2023) — their reduced-form **BVAR was significantly worse than all other approaches**.

> **Verifier notes:** (i) The metals-futures cointegration paper mis-cited as "Booth & Ciner (2001), *Math. & Computers in Simulation* 56" is actually **Watkins & McAleer (2002), *Mathematics and Computers in Simulation* 59, 207–221** — a 3-month LME *copper* futures risk-premium/cost-of-carry study (a *better* copper fit than the mis-citation implied). (ii) The ARDL oil exemplar mis-cited to "Galip Afsin Ravanoğlu" is **Ben Salem, Nouira, Jeguirim & Rault (2022), *Resources Policy* 79** ("The determinants of crude oil prices: Evidence from ARDL and nonlinear ARDL approaches"). No clean *copper*-specific ARDL with a monthly RW comparison was located — treat copper ARDL forecast superiority as plausible-but-unproven. (iii) **Chinn & Coibion (2014, *J. Futures Markets* 34(7))** find base-metals (incl. copper) futures fail unbiasedness and are *no better than a RW* OOS, while **Reichsfeld & Roache (2011, IMF WP 11/254)** find futures "hard to beat" — benchmark any copper model against **both** the RW and futures on your own sample.

---

### 3.3 Volatility & density forecasting (GARCH family, HAR-RV, GARCH-MIDAS, MIDAS, quantile)

This is the family that attaches *calibrated* intervals and VaR/ES to the repo's point forecasts and lets copper macro drivers move the long-run risk level transparently.

**Univariate GARCH (GARCH, EGARCH, GJR, APARCH, FIGARCH).** Conditional variance h_t is a low-dimensional recursion: in GARCH(1,1), α = reaction to last shock, β = persistence, α+β = volatility half-life. EGARCH/GJR/APARCH add a *leverage* term — but for industrial metals the leverage sign is unstable and can be positive (vol rising on demand-driven up-moves), so **estimate it, don't assume the equity sign**. FIGARCH adds long memory (fractional *d*). With skew-t innovations you get fat-tailed one-step densities and VaR. **Best at daily/weekly**; at monthly horizon a daily-GARCH forecast decays to unconditional variance, motivating the component/MIDAS variants. Li & Li (2015, *Resources Policy* 46) show *averaging* several GARCH specs beats picking one — mirroring the repo's ensemble philosophy applied to variance.

**HAR-RV** (Corsi 2009). RV_{t+1} = c + β_d·RV_d + β_w·RV_w + β_m·RV_m, an OLS regression whose three coefficients read as daily/weekly/monthly "trader-horizon" weights; HAR-RV-J/CJ add jump and continuous components, and leverage/semivariance terms let down-moves load differently. It maps almost exactly onto the repo's 5d/22d targets. In the only dedicated COMEX-copper RV horse-race located, **Wang & Lu (2024, arXiv:2409.08356)**, HAR was the single best model for daily realized vol (QLIKE ≈ 2.39E-09), an order of magnitude below the deep nets (RNN/LSTM/GRU ≈ 6.70–6.73E-08) — which only closed the gap at hourly frequency. **Needs intraday HG=F data** to build RV.

**GARCH-MIDAS / component-GARCH** (Engle, Ghysels & Sohn 2013; Conrad & Kleen 2020; Engle & Rangel 2008 Spline-GARCH). Variance = short-run unit-GARCH g_t × long-run τ_t, where log τ_t is a MIDAS-weighted sum of *monthly macro* variables: a **single slope θ per driver gives the signed elasticity of copper's long-run risk to that macro factor** ("a one-unit rise in monthly PPI raises long-run variance by θ"). This is the standout family for **monthly/3-month** risk: Conrad & Kleen (2020, *JAE* 35(1)) show only GARCH-MIDAS models survive the Model Confidence Set at 2–3-month horizons. Copper-specific: **Wang & Li (2024, arXiv:2409.08355)** find **PPI is the most efficient long-run driver** of COMEX copper volatility (positive), with interest rates and IP entering negatively.

**MIDAS (conditional mean).** y_{t+h} = β₀ + β₁·Σ φ_k(w)·X_high_{t−k}, compressing many daily/weekly lags (DXY, real yields, basis, COT) into 1–2 parameters with a *plottable* lag-weight curve — the clean way to push the repo's daily features into the 22d/66d target without ad-hoc aggregation. A direct h-step forecaster (no iterated-error accumulation).

**Quantile / mixed-frequency quantile regression.** Minimize pinball loss for chosen quantiles; the gap between the 5% and 95% fits *is* the interval/VaR, with each β_q a readable tail sensitivity — distribution-free, fat-tail-robust, backtestable (Kupiec/Christoffersen). Čech & Baruník (2019, *J. Futures Markets* 39(9), 1167–1189) build and backtest OOS VaR/ES across commodity sectors incl. metals; Candila, Gallo & Petrella (2023, *Annals of OR*) establish MIDAS-QR for monthly VaR from daily inputs (applied to oil/gasoline — transferable, not yet copper).

**Caveats.** Several copper-specific "wins" (Fang, Zhao & Zhong 2022 HAR-RV; the copper GARCH-MIDAS working papers) lack a clean DM/RW comparison — treat magnitudes as unverified. Much MCS-based evidence is on equities/oil/gold — re-validate on HG=F/LME. Use a quantile/KDE interval machinery on **transparent** engines; the Nature *Sci Rep* (2025) copper quantile-deep-learning paper is a contrast point (opaque engine to avoid).

> **Verifier note:** In Wang & Lu (2024), the QLIKE of **5.99E-08 is *Realized* GARCH**, not plain GARCH (plain GARCH is not separately tabulated for daily RV). HAR still beats the deep nets by ~an order of magnitude.

---

### 3.4 Regime-switching & interpretable nonlinear models (MS-AR/VAR, TAR/SETAR, STAR, TVP)

**Markov-switching AR/VAR** (Hamilton 1989). A latent state s_t follows a Markov chain; conditional on s_t, returns follow a Gaussian AR(X). You read off regime-specific means/vols, the transition matrix P (and expected durations 1/(1−p_ii)), and the **smoothed probability P(s_t=k | data)** — a chartable "which-regime-are-we-in" signal. This is the natural upgrade to the repo's 3-state Gaussian HMM (an HMM on returns *is* a Markov-switching model): add AR terms and exogenous drivers (MS-ARX/MSVAR) so regimes carry *mean dynamics* and *regime-dependent driver effects*. Better for monthly point forecasts; weekly use is mainly state-tagging for risk/interval widening.

**TAR/SETAR and momentum-threshold ECM (M-TAR; Enders & Siklos 2001).** Piecewise-linear switching when an *observable* threshold variable crosses an estimated boundary. M-TAR error-correction is the most copper-relevant: it lets copper mean-revert to its gold/aluminium/oil/DXY/CNY anchor at **different speeds depending on the sign or momentum of the deviation**, with a formal asymmetry test — a direct, transparent upgrade to the repo's Engle-Granger ECM. Goo & Chen (2020, *Modern Economy* 11) document asymmetric momentum-threshold effects in LME copper futures-vs-spot under high volatility (an in-sample structural result).

**STAR/LSTAR/ESTAR.** Gradual regime change via a logistic/exponential transition function of an economic variable. **This is the weakest forecasting bet despite high interpretability:** Ubilava (2022, *Agricultural Economics* 53(5)) shows STAR is mostly *inferior* to a simple AR for multistep commodity point forecasts; Ubilava (2018, *AJAE* 100(1)) finds exogenous-transition (ENSO) STAR helps tropical agriculturals but has **no out-of-sample predictive power for metals**. Treat STAR as a diagnostic/robustness overlay, not a core forecaster.

**Time-varying-parameter regression / DMA-DMS.** Coefficients evolve via a Kalman state law; DMA averages many TVP regressions with forgetting-factor weights, exposing the **time path of each predictor's inclusion probability** ("when did inventories vs the dollar drive copper?"). This is the strongest positive copper result in the nonlinear class — **Buncic & Moretto (2015)** again — and the best interpretability match for the repo, complementing (not duplicating) the HMM.

**The honest consensus:** these models reliably improve **in-sample fit, volatility, and density/interval** forecasts, but their **OOS point-forecast edge over the RW is horizon- and regime-dependent and often small or absent at weekly frequency**. Allayioti & Venditti (2024, ECB WP 2901) crystallize this: across ten commodity indices, TVP/comovement gives only *small* point-forecast gains but *large, significant* density gains. **Evaluate the mean with rolling-origin OOS R² + DM; evaluate intervals separately with coverage/CRPS** — these models win on the distribution more than the mean.

---

### 3.5 Bayesian & model-averaging methods (DMA/DMS, BMA, BVAR, combination, BSTS/Prophet)

Every method here exposes something human-readable: shrunk VAR coefficients/IRFs (BVAR), posterior inclusion probabilities (BMA, BSTS), time-varying predictor weights (DMA/DMS), or transparent combination weights.

**DMA/DMS** (Koop-Korobilis lineage; orig. Raftery, Kárný & Ettler 2010). Fast (non-MCMC) recursion of TVP regressions reweighted by recent predictive performance. **The best interpretability-for-copper match** because its posterior inclusion probabilities are a *time series*. Validated monthly (Buncic & Moretto 2015; Naser 2016 and Drachal 2017 for oil); weekly use is an extrapolation (its informative predictors are monthly).

**BMA.** Static averaging over predictor subsets weighted by posterior model probability; the headline output is the **posterior inclusion probability (PIP)** per variable. Well-validated for copper but on **volatility, not price level**: Diaz, Hansen & Cabrera (2021, *Resources Policy* 73) find several economic variables beat an AR benchmark for monthly copper volatility, with predictability varying over the business cycle; Li & Li (2015) similar. Deploy for the interval/quantile block and feature-importance transparency.

**BVAR (Minnesota prior).** Shrinks each equation toward a RW, taming parameter explosion (Bañbura, Giannone & Reichlin 2010). The interpretable payoff is IRFs, FEVD and **coherent scenario fans** — the multivariate Bayesian generalization of the repo's Engle-Granger ECM. But its edge over the RW for metals is **fragile and horizon-dependent**: the World Bank suite found their BVAR significantly worse than all alternatives, and oil-VAR reappraisals show headline gains shrink in real time. **Its surest value is scenarios and calibrated uncertainty bands, not point-forecast wins.**

**Forecast combination & the "puzzle."** Bates-Granger (1969) variance-minimizing weights and Granger-Ramanathan regression weights are theoretically optimal, but Stock & Watson (2004) coined the **forecast-combination puzzle**: the simple equal-weighted (1/N) average routinely beats estimated "optimal" weights OOS once estimation error is accounted for. Genre, Kenny, Meyler & Timmermann (2013, *IJF* 29(1)) confirm that after a White Reality-Check multiple-comparison correction, apparent gains over the simple average mostly do not survive. **This is the most actionable governance lesson for the repo's stacking ensemble: benchmark it against 1/N and trimmed means.**

**BSTS / Prophet.** BSTS (Scott & Varian 2014) = structural state-space (local trend + seasonal) + regression with a **spike-and-slab prior** that selects predictors (PIPs) and yields proper posterior predictive intervals. Prophet (Taylor & Letham 2018) is a faster decomposable/GAM-style model (piecewise trend + changepoints + Fourier seasonality), fit in Stan (MAP by default). On volatile, weakly-seasonal copper both underperform ARIMAX/ML on point accuracy and Prophet's intervals are often poorly calibrated. **BSTS is the principled interpretable upgrade from the repo's Prophet slot** — same decomposition plus spike-and-slab variable selection and honest intervals.

> **Verifier notes:** The oil-VAR MSPE reductions are **~25% at 1 month and ~24% at 3 months** (Baumeister & Kilian 2012, real-time real-oil-price VAR), *not* the 30–32% / 20–22% sometimes quoted; gains fade past ~12 months and later real-time replications (Benyő et al. 2026, *Economic Inquiry*) question their robustness. Prophet should not be described as "BSTS-style." DMA's lineage is Koop & Korobilis (2012) in econometrics, originating with Raftery, Kárný & Ettler (2010).

---

### 3.6 Commodity-finance theory & futures-based forecasting

The most economically explicit family — and where the benchmarks are most humbling.

**Futures-curve / cost-of-carry.** Under no-arbitrage, F_{t,h} = S_t·(1 + r + storage − convenience yield); if the risk premium is zero, the futures price *is* the market's expected future spot, and the basis algebraically decomposes into carry components. Maximally interpretable (a traded price). **For copper the basis is tiny and the curve nearly flat, so the futures forecast is numerically almost identical to the RW** and rarely beats it OOS at short horizons (Reeve & Vigfusson 2011: 3-month relative MSE ≈ 1.00). It conditionally beats the RW by a wide margin *when the spot-futures gap is large* (extreme inventories/convenience yield).

**Theory of storage / basis-as-state-variable** (Working; Kaldor; Brennan; Fama & French 1987, 1988; Geman & Smith **2013**, *Resources Policy* 38(1), 18–28). The interest-adjusted basis = storage cost − convenience yield, and the convenience yield is a decreasing, convex function of inventories — so **backwardation (LME cash > 3M) flags low inventories/scarcity and predicts higher spot volatility**. Fama & French (1988) validate this for five LME base metals; **Geman & Smith (2013) show adding Chinese/SHFE inventories strengthens the inventory-basis-volatility relationship** — a direct argument to incorporate SHFE/bonded stocks. Use the LME cash-3M spread and exchange + bonded inventories as features and a volatility/regime signal.

**Hedging pressure / normal backwardation** (Keynes; Cootner; Hirshleifer; Bessembinder; Basu & Miffre 2013). The futures *risk premium* is driven by net positioning: when hedgers are net short they pay a premium to long speculators. Basu & Miffre (2013, *J. Banking & Finance* 37(7)) show hedging-pressure long-short premia are real, rise with volatility, and are distinct from carry and momentum — legitimizing **CFTC COT for COMEX copper** as an interpretable monthly signal. Heed Kolb's caveat ("normal backwardation is not normal" — backwardation alone does not guarantee positive returns) and note the premium is small, noisy, and diluted by financialization.

**Commodity factor models** — carry/basis (Koijen, Moskowitz, Pedersen & Vrugt 2018), momentum, **basis-momentum** (Boons & Prado 2019, *J. Finance* 74(1)). These are interpretable and computable from the HG/LME curve, but they are **cross-sectional relative-value premia** — for single-name copper, use carry (copper's own basis), momentum (copper trend), and basis-momentum (1st-vs-2nd-nearby curve dynamics) as *features*, not standalone forecasters.

**Structural supply-demand balance models** (IMF/World Bank/ICSG style). Explicit market-balance accounting (mine + scrap supply, China ~60% of demand, inventories) with behavioral elasticities; the price clears the balance. The most economically transparent of all, but **data-lagged, low-frequency, and best for the 66d+/strategic horizon and scenario analysis**, not weekly forecasting. Copper's short-run price elasticities are small but significant (Shojaeinia 2023, *Mineral Economics* 36(3), 509–517), so the price is sensitive to small balance errors.

> **Verifier notes — corrections applied:** (i) **World Bank PRWP 10611 is a *quarterly* exercise** (quarterly data, 1–8-quarter horizons, 2015Q1–2022Q1) — it does *not* evaluate monthly horizons; present its findings (futures/bivariate correlations best short-horizon; consensus/large macro-model best long-horizon; BVAR worst) for *quarterly* horizons. (ii) **Drop the claim that "the world output gap and USD real effective exchange rate are statistically significant for copper" as sourced to PRWP 10611** — those variables/phrases are not in that paper (its bivariate inputs were US M1, 10-year yields, the CRB Raw Materials index). (iii) **Drop the "Rubaszek et al. (2020) weekly futures beat RW" claim** — it could not be verified and is the only load-bearing support for futures-beats-RW *weekly*; the weekly-horizon case for futures is currently unsupported in the cited literature. (iv) **Gorton, Hayashi & Rouwenhorst (2007) *reject* hedging pressure** as an independent premium driver, attributing premium variation to inventories — they do *not* "unify" storage with hedging pressure; this sharpens the genuine tension with Basu & Miffre (2013), who *do* find an independent positioning premium.

---

### 3.7 Interpretable statistical learning (LASSO/EN/adaptive, GAMs, symbolic regression, RuleFit, GPR)

**Penalized regression (LASSO, ridge, elastic net, adaptive LASSO).** Best-supported interpretable approach: an L1/L2/mixed penalty yields a sparse, signed linear equation; adaptive LASSO has the oracle (consistent-selection) property; elastic net keeps correlated groups (LME/COMEX/DXY/CNY co-movement) together. The final model is a literal equation — you read off which drivers survived, their signs and magnitudes, and the selection path ranks marginal usefulness. Strong analog evidence: **Zhang, Ma & Wang (2019, *J. Empirical Finance* 54)** — LASSO/elastic net beat a wide field OOS for monthly oil on R², success ratio and economic value; post-LASSO OLS also beat competitors. **Liu, Guo & Wei (2024, *J. Commodity Markets* 34)** use LASSO over 75 copper indicators to show fundamentals drive long-run trends while financial/geopolitical factors drive short-term fluctuations (attribution, not a RW horse-race). **Highest-value addition: extend the repo's ridge to LASSO/elastic-net/adaptive-LASSO** for principled selection. Caveat: penalized-regression inference is distorted by serial dependence, and time-ordered (purged/embargoed) CV is mandatory.

**GAMs / penalized splines** (Wood 2017). Model returns as a sum of smooth, *individually plottable* functions of each driver (P-splines/thin-plate), with confidence bands — capturing thresholds/saturation/asymmetry while keeping each effect readable as a partial-dependence curve; cyclic splines encode calendar effects; **GAMLSS** models the variance/quantiles for interpretable intervals. Proven workhorses in electricity price/demand forecasting (Serinaldi 2011 GAMLSS), but **copper-specific peer-reviewed GAM-vs-RW evidence is a genuine gap** — test against the RW baseline rather than assuming a win.

**Gaussian process regression.** Nonparametric Bayesian regression returning a posterior mean *and* closed-form predictive variance; kernel structure (separable trend + seasonal + noise) and ARD length-scales (long length-scale = irrelevant feature) give semi-interpretability, with **best-in-class calibrated uncertainty**. The active metals-GPR program (Jin & Xu copper/silver/steel/coal papers) reports strong fit but is **autoregressive and rarely benchmarks against a RW** — so its accuracy is *not* evidence of a level edge. Value to the repo: **calibrated weekly intervals + ARD relevance**, dovetailing with the quantile component and HMM.

**Single decision trees & RuleFit.** Single CART trees are maximally interpretable (a flowchart of thresholds) but **lost to the RW for copper at short-to-medium horizons** (Diaz, Hansen & Cabrera 2020) and *cannot extrapolate* trending prices — use as diagnostics. **RuleFit (Friedman & Popescu 2008)** — a LASSO-pruned sparse rule ensemble — is the better middle ground: it captures threshold interactions ("backwardation + low inventory + positive China IP" as one bullish rule) while staying inspectable, a transparent contrast to the repo's XGBoost/LightGBM.

**Symbolic regression / genetic programming.** Searches for an explicit closed-form equation; Bayesian symbolic regression (BSR) returns a posterior over equations. But **Drachal (2022, *Energies* 16(1)) found BSR did *not* significantly beat LASSO/ridge/DMA/BMA/ARIMA on crude oil**, and Drachal & Pawłowski (2024, *IJFS* 12(2)) reach the same conclusion across commodities. Use only for hypothesis generation, then harden discoveries into LASSO/GAM features.

> **Verifier notes:** (i) **Drachal (2019, *Resources Policy* 64) forecasts lead, nickel and zinc — *not* copper or aluminium**; cite it only as general DMA-for-metals support, not copper-specific. (ii) Diaz et al. (2020) shows the RW beating trees at **short-to-medium horizons with the gap narrowing (not equalizing) at 2 years** — at 2 years the RW (16.89% RMSE) still beats the regression tree (20.18%); 1-year figures are **RW 11.84% vs tree 19.70%** (not "~14.4% vs ~21–24%").

---

### 3.8 Decomposition hybrids & forecast-evaluation methodology

**Decompose-and-ensemble hybrids.** A preprocessing transform splits the price into additive components, each forecast with a transparent model and recombined:
- **Wavelet (MRA) + ARIMA** (Kriechbaumer et al. 2014): named frequency bands → per-band ARIMA. Reports a **~$126/tonne one-month copper improvement over classic ARIMA** (against ARIMA only, *not* a RW), but is highly sensitive to wavelet family/levels (researcher-degrees-of-freedom risk) and suffers right-edge (forecast-origin) distortion.
- **EMD/EEMD/CEEMDAN + ARIMA**: adaptive frequency-ordered IMFs (trend vs cycle vs noise), only loosely interpretable and unstable at the right edge.
- **VMD + ARIMA**: fixed *K* band-limited modes with *named center frequencies* — the most interpretable adaptive method, but *K* and bandwidth are hyperparameters.
- **STL + ARIMAX** (Cleveland et al. 1990): trend + named seasonal + remainder on the price scale — **the most interpretable decomposition**, but copper's weak seasonality means STL is best as an honest deseasonalizer/diagnostic, not an alpha source.

**THE CRITICAL CAVEAT — information leakage.** Most decompose-and-ensemble papers run EMD/VMD/wavelet on the *whole* series before the train/test split, leaking future information into the components. **Yang, Li & Jiang (2024, *Scientific Reports* 14:28362)** show this collapses reported RMSE toward zero (~0.001) and that leakage-safe re-decomposition (sliding/expanding window) raises error by **1–2 orders of magnitude**. Any copper hybrid MUST re-decompose only on data up to each forecast origin — otherwise its OOS edge is illusory. Treat large reported MAPE reductions skeptically unless the paper states leakage-safe decomposition, a RW benchmark, and a nesting-aware significance test.

**Forecast-evaluation methodology** (see §4).

---

## 4. Forecast-evaluation methodology, data-snooping & publication-bias caveats

| Test | What it answers | When to use for copper |
|---|---|---|
| **Diebold-Mariano (1995)** | Equal predictive accuracy of two forecasts (HAC variance) | **Non-nested** pairs only |
| **Clark-West (2007)** | MSFE-adjusted equal accuracy for **nested** models | **Whenever the model nests the RW** (ARIMAX/ECM/DMA vs RW) — mandatory |
| **Campbell-Thompson OOS R²** | Skill vs RW (1 − MSFE_model/MSFE_RW) | Headline skill metric; can be small-positive yet significant |
| **Giacomini-White (2006)** | *Conditional* predictive ability (is A better given the regime?) | When predictability is regime-dependent |
| **Model Confidence Set** (Hansen, Lunde & Nason 2011) | The set of models containing the best at a confidence level | Comparing the **whole ensemble** at once |
| **Pesaran-Timmermann (1992)** | Directional/sign accuracy vs independence | Weekly/monthly trading, where RMSE gains are tiny but *direction* may be predictable |
| **Economic value** | Sharpe / certainty-equivalent **net of transaction costs** | Any trading use-case; gains often vanish after costs |
| **White Reality Check (2000) / Hansen SPA (2005)** | Does the *best of many searched* rules truly beat the benchmark? | Controlling **data-snooping** over decomposition/feature/hyperparameter search |

**The dominant methodological error in the metals literature is using plain DM where Clark-West is required** — DM is oversized (biased toward rejecting equal accuracy in favor of the larger model) when the candidate nests the RW. Many papers also report only RMSE/MAE with **no RW benchmark, no nesting correction, and no multiple-testing control**, so their "we beat the random walk" claims are fragile.

**Data-snooping & publication-bias caveats specific to this dimension:**

1. **Period-specificity.** Apparent skill is often confined to one volatile episode — Buncic & Moretto's copper gains concentrate in 2008; copper futures beat the RW pre-2000 but the RW beat futures in the late 2000s (Reeve & Vigfusson). **Split evaluation by sub-period.**
2. **In-sample ≠ OOS.** Significant in-sample cointegration/nonlinearity tests routinely fail to translate into OOS RW-beating (a recurring caution in Rubaszek 2021 and the World Bank suite). In-sample fit must never be conflated with forecasting value.
3. **Leakage** in decomposition hybrids (§3.8) and **look-ahead** from non-purged CV or non-vintage-aware macro data inflate OOS metrics.
4. **Publication bias** toward positive results: the literature systematically under-reports the (common) cases where methods fail to beat the RW. **Many methods genuinely do *not* robustly beat a random walk for copper** — this is the honest base rate, not a failure of search.
5. **Density vs point.** Regime/TVP/Bayesian methods win much more reliably on *density/interval* metrics (CRPS, log-score, PIT coverage) than on the conditional mean (Allayioti & Venditti 2024) — score intervals separately.

---

## 5. Weekly vs monthly: which families dominate, and why

**Weekly (5d) is largely random-walk territory.** Copper log-returns are close to unforecastable in the mean at one week; the RW (no drift) and the near-dated futures price (numerically almost identical to it) are very hard to beat. The informative *fundamental* panels (China/global IP, real yields, breakevens, inventories) are monthly, so they carry little weekly signal. What *can* help weekly:

- **Fast financial drivers** — DXY, USD/CNY, cross-asset ratios (copper/gold, copper/oil), the futures basis, and CFTC COT positioning — entered via ARIMAX, a small financial-only regression, or MIDAS.
- **Variance, not mean** — **HAR-RV** (best for copper realized vol at daily/weekly horizons) and skew-t GARCH for calibrated fat-tailed intervals.
- **Regime tagging** — the HMM / Markov-switching state for interval widening and position sizing.
- **Robust combination** — equal-weighted/trimmed averaging of components.
- **Calibrated intervals** — GPR, GAMLSS, quantile regression.

Do **not** expect large weekly point-accuracy gains from cointegration/VECM (the ECT barely moves in a week), DFM/FAVAR (monthly inputs), or structural balances (low-frequency).

**Monthly (22d) and 3-month (66d) is where interpretable structure earns its keep.** Genuine, documented copper predictability exists at 1–6 months and is captured by:

- **DMA/DMS and TVP regressions** over fundamentals + financial conditioning — the single best-evidenced interpretable copper result (Buncic & Moretto 2015, OOS R² up to ~18.5%).
- **Cointegration/ECM/VECM** — mean-reversion to a cross-asset/macro equilibrium is a monthly-scale process; use rolling ECTs and regime-conditioning.
- **Structural fundamentals & supply-demand balances** — increasingly relevant toward 66d and for scenarios.
- **MIDAS** — to exploit daily/weekly predictors for the monthly target without aggregation loss.
- **GARCH-MIDAS** — for the *variance* at 1–3 months (the macro-driven long-run component carries the risk signal where daily GARCH decays to a constant).

**Cross-cutting:** the random walk is the benchmark every model must clear at *both* horizons; futures/basis are a stringent secondary benchmark; and **GARCH-family/HAR/GARCH-MIDAS forecast the variance, not the mean** — deploy them for intervals and risk, not as point forecasters.

---

## 6. Pragmatic roadmap for the repo (priority order)

The repo already has: Naive/RW baseline, Ridge, ARIMAX/SARIMAX with intervals, Prophet, Engle-Granger cointegration with rolling ECTs (gold/aluminium/oil/DXY/CNY), a 3-state Gaussian HMM, quantile regression, and weighted + stacking ensembles; plus XGBoost/LightGBM (black-box) and a rich feature set (cross-asset ratios, macro, COT, calendar, technicals) from yfinance/FRED/EIA/Alpha Vantage/CFTC. Below, *Add/strengthen* items in priority order, each with rationale, horizon, and the expected pitfall.

**Priority 1 — Dynamic Model Averaging/Selection (DMA/DMS) over the existing predictor set.**
*Rationale:* the single best-evidenced interpretable copper model (Buncic & Moretto 2015, OOS R² ~18.5% at 1m), and its time-varying inclusion probabilities are a *time series* that complements (not duplicates) the HMM — you can show *when* inventories vs the dollar vs positioning drove copper. Predictors map almost one-to-one onto the repo's features.
*Horizon:* monthly (22d) and 3-month (66d); weekly is an extrapolation.
*Pitfall:* sensitive to forgetting-factor (α, λ) calibration; gains concentrate in crisis periods and can be thin in calm markets; respect monthly vintage discipline (China IP/PPI revisions).

**Priority 2 — Disciplined feature selection: LASSO / elastic-net / adaptive-LASSO (extend the existing Ridge).**
*Rationale:* highest-value, best-evidenced interpretable upgrade (Zhang, Ma & Wang 2019 for oil; Liu, Guo & Wei 2024 as the copper-driver selection engine); produces a transparent sparse equation and a stable signal set, taming the high-dimensional feature explosion that overfits the stacking layer.
*Horizon:* monthly primarily; weekly for regularization/over-fit control.
*Pitfall:* LASSO selection is unstable under multicollinearity (use elastic net) and serial dependence (use time-ordered purged CV — the repo's "cv purge" commit shows awareness); coefficient signs can flip across rolling windows, complicating the interpretable story.

**Priority 3 — GARCH-family / HAR for proper prediction intervals.**
*Rationale:* the fastest route to *calibrated, fat-tailed, backtestable* intervals on the 1d/5d forecasts (skew-t GJR/EGARCH for the mean-model errors; HAR-RV as the best copper realized-vol model where intraday HG=F is available). Li & Li (2015): average several GARCH specs rather than picking one.
*Horizon:* HAR-RV at 5d–22d; GARCH at 1d–5d.
*Pitfall:* estimate the leverage sign (don't assume the equity sign — copper can show positive asymmetry); structural breaks inflate persistence toward spurious IGARCH; HAR needs reliable intraday data (microstructure noise, overnight gaps).

**Priority 4 — GARCH-MIDAS to drive long-run risk from monthly macro.**
*Rationale:* the standout interpretable family for *monthly/3-month* risk — a single slope θ gives the signed elasticity of copper's long-run volatility to each macro driver; copper-specific evidence flags **PPI** as the most efficient driver (Wang & Li 2024), and only GARCH-MIDAS survives the MCS at 2–3 months (Conrad & Kleen 2020).
*Horizon:* 22d–66d (risk, not mean).
*Pitfall:* θ-vs-weight-shape identification is fragile in short samples; choose MIDAS weight form and macro regressors carefully; respect data vintages.

**Priority 5 — MIDAS for the conditional mean (daily/weekly predictors → monthly target).**
*Rationale:* the clean, interpretable way to push the repo's daily DXY/real-yields/basis/COT features into the 22d/66d target without ad-hoc aggregation, with a plottable lag-weight curve; a direct h-step forecaster (no iterated-error accumulation).
*Horizon:* 22d, 66d.
*Pitfall:* the restricted Beta/Exp-Almon weight shape can be mis-specified; captures level only — still need GARCH-MIDAS/quantile for intervals; copper-specific point-forecast MIDAS evidence is sparse (transfer from oil/inflation with care).

**Priority 6 — Single-equation ECM / VECM exploiting the existing cointegration ECTs (+ M-TAR asymmetry).**
*Rationale:* the repo already computes rolling ECTs; formalize them into a forecasting ECM, and add **momentum-threshold (M-TAR) asymmetric** adjustment (Enders & Siklos 2001; Goo & Chen 2020) so copper mean-reverts faster when rich than cheap (or vice versa), with a formal asymmetry test. A Johansen VECM adds weak-exogeneity ("does copper lead or follow?") and IRFs for scenario narrative.
*Horizon:* monthly (22d), some 66d.
*Pitfall:* cointegrating relationships break across regimes — keep ECTs rolling/recursive and regime-conditioned via the HMM; VECMs are parameter-hungry and overfit-prone weekly; static estimates are dangerous.

**Priority 7 — Markov-switching AR/VAR to formalize the HMM into the mean model.**
*Rationale:* upgrade the descriptive 3-state HMM to MS-ARX/MSVAR so regimes carry *mean dynamics and regime-dependent driver effects*, with readable named states, transition probabilities, expected durations, and a smoothed "which-regime-now" signal for risk/ensemble weighting.
*Horizon:* monthly (mean); weekly (state-tagging only).
*Pitfall:* regime identification is unstable at weekly frequency / with too many states; the transition matrix is assumed time-invariant; OOS *mean*-forecast gains over the RW are not robustly demonstrated for metals (the edge is in density/regime tagging) — score intervals separately.

**Priority 8 — A futures-basis / theory-of-storage benchmark model.**
*Rationale:* the basis is the single most copper-native interpretable state variable (backwardation = tightness) and a stringent secondary benchmark; add the LME cash-3M spread and the convenience yield as explicit features, and **incorporate SHFE/bonded inventories** (Geman & Smith 2013 show they strengthen the relationship). The futures curve gives free multi-horizon forecasts and option-implied intervals.
*Horizon:* 22d, 66d (and as a benchmark at all horizons).
*Pitfall:* for copper the basis is small and the curve nearly flat, so the futures point forecast ≈ RW at short horizons; it embeds a time-varying risk premium (biased predictor); the evidence that the basis adds *level* forecast value is contested (Chinn-Coibion negative vs Reichsfeld-Roache positive) — re-test on your own sample.

**Priority 9 — DMA/BMA governance + disciplined forecast combination.**
*Rationale:* BMA's PIPs give transparent feature-importance for the interval/quantile block (Diaz et al. 2021 for copper volatility); and the **forecast-combination puzzle** mandates benchmarking the repo's stacking ensemble against a simple 1/N (and trimmed-mean) average — estimated "optimal" weights frequently fail to beat 1/N OOS.
*Horizon:* horizon-agnostic (meta-layer).
*Pitfall:* estimated weights overfit and suffer multiple-comparison bias (apply White Reality Check / SPA); combination cannot help if all components share the same blind spot.

**Priority 10 — GAMs / penalized splines for interpretable nonlinearity.**
*Rationale:* smooth, plottable per-driver response curves with confidence bands (real yields, China IP, inventory, basis, DXY) capture thresholds/saturation/asymmetry while staying readable; GAMLSS upgrades the interval machinery; a natural transparent member of the ensemble.
*Horizon:* monthly (22d).
*Pitfall:* **copper-specific GAM evidence is a genuine gap** — test against the RW, don't assume a win; additivity misses interactions unless you add tensor smooths (costing interpretability/data); splines extrapolate wildly outside the observed range (dangerous for trending copper).

**Supporting additions (lower priority but high-leverage):** GPR as a probabilistic ensemble member (calibrated weekly intervals + ARD feature relevance, but *not* a level-beating point model); BSTS as the principled upgrade from Prophet (spike-and-slab PIPs + honest posterior intervals); RuleFit as an interpretable contrast to XGBoost/LightGBM. **Use STL/wavelet/EMD/VMD only with leakage-safe (sliding-window) re-decomposition** if used at all. **Avoid** standalone single decision trees and symbolic regression as accuracy engines (both lose to simpler interpretable methods for copper); keep them as diagnostics.

**Evaluation discipline for every addition:** benchmark against the existing Naive/RW (and futures) baseline with **Clark-West** (nested) + **Campbell-Thompson OOS R²**, compare the whole stable with the **Model Confidence Set**, report **Pesaran-Timmermann** directional accuracy and **net-of-cost economic value** at 5d/22d, score intervals with **coverage/CRPS**, control data-snooping with **SPA/Reality Check**, split by sub-period, and keep time-ordered purged CV.

---

## 7. Bottom line

For copper at **weekly** horizons, the random walk (and the near-flat futures curve) is very hard to beat on the level; deploy interpretable methods there for **disciplined feature selection, regime tagging, and calibrated intervals** — not for large point-accuracy gains. Genuine, documented predictability lives at **monthly-to-3-month** horizons and is captured by **interpretable, time-varying, driver-aware models** — above all **dynamic model averaging/selection** (the clearest copper-specific RW-beating result, ~18.5% OOS R² at 1 month), plus rolling cointegration/ECM, MIDAS for frequency-mixing, and GARCH-MIDAS for the variance. **Most methods do *not* robustly beat the random walk for copper**, and many published "wins" evaporate under proper benchmarking (Clark-West for nested models), leakage-safe decomposition, sub-period splits, and data-snooping controls — so the honest base rate is sobriety, and the surest gains are at monthly+ horizons via slow macro/financial drivers, in the *distribution* more than the mean, and through robust forecast combination benchmarked against a simple 1/N average. Keep the random walk and the futures curve as the bars every model must clear; keep XGBoost/LightGBM as complements; and let the interpretable models remain the auditable core.

---

## 8. Consolidated reference list (verified sources only)

**Benchmarks & copper-specific forecasting**
- Diaz, J.D., Hansen, E. & Cabrera, G. (2020). *A random walk through the trees: Forecasting copper prices using decision learning methods.* Resources Policy 69, 101545. https://www.sciencedirect.com/science/article/abs/pii/S0301420720308904
- Reeve, T.A. & Vigfusson, R.J. (2011). *Evaluating the Forecasting Performance of Commodity Futures Prices.* Federal Reserve IFDP No. 1025. https://www.federalreserve.gov/pubs/ifdp/2011/1025/ifdp1025.pdf
- Buncic, D. & Moretto, C. (2015). *Forecasting copper prices with dynamic averaging and selection models.* North American Journal of Economics and Finance 33, 1–38. https://www.danielbuncic.com/pdf/foreCopper.pdf
- Rubaszek, M. / Kwas, M. & Rubaszek, M. (2021). *Forecasting Commodity Prices: Looking for a Benchmark.* Forecasting (MDPI) 3(2), 27. https://www.mdpi.com/2571-9394/3/2/27
- Sánchez Lasheras, F., de Cos Juez, F.J., Suárez Sánchez, A., Krzemień, A. & Riesgo Fernández, P. (2015). *Forecasting the COMEX copper spot price by means of neural networks and ARIMA models.* Resources Policy 45, 37–43. https://ideas.repec.org/a/eee/jrpoli/v45y2015icp37-43.html
- Arroyo Marioli, F., Khadan, J., Ohnsorge, F. & Yamazaki, T. (2023). *Forecasting Industrial Commodity Prices: Literature Review and a Model Suite.* World Bank PRWP 10611 (quarterly). https://ideas.repec.org/p/wbk/wbrwps/10611.html

**Univariate econometrics, ETS, Theta, UC state-space**
- Assimakopoulos, V. & Nikolopoulos, K. (2000). *The theta model.* IJF 16(4), 521–530.
- Hyndman, R.J. & Billah, B. (2003). *Unmasking the Theta method.* IJF 19(2), 287–290. https://robjhyndman.com/papers/Theta.pdf
- Hyndman, R.J., Koehler, A.B., Snyder, R.D. & Grose, S.D. (2002). *A state space framework for automatic forecasting using exponential smoothing methods.* IJF 18(3), 439–454.
- Fioruci, J.A., Pellegrini, F., Louzada, F., Petropoulos, F. & Koehler, A.B. (2016). *Models for optimising the theta method…* IJF 32(4), 1151–1161.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter.* Cambridge UP.
- Cuddington, J.T. & Jerrett, D. (2008). *Super Cycles in Real Metals Prices?* IMF Staff Papers 55(4), 541–565. https://www.imf.org/external/pubs/ft/staffp/2008/04/pdf/cuddington.pdf
- Kahraman, E. & Akay, O. (2023). *Comparison of exponential smoothing methods in forecasting global prices of main metals.* Mineral Economics 36(3). https://link.springer.com/article/10.1007/s13563-022-00354-y
- Kriechbaumer, T., Angus, A., Parsons, D. & Rivas Casado, M. (2014). *An improved wavelet-ARIMA approach for forecasting metal prices.* Resources Policy 39, 32–41. https://ideas.repec.org/a/eee/jrpoli/v39y2014icp32-41.html
- Oikonomou, K. & Damigos, D. (2025). *Short term forecasting of base metals prices using a LightGBM and a LightGBM-ARIMA ensemble.* Mineral Economics 38(1), 37–49. https://link.springer.com/article/10.1007/s13563-024-00437-y

**Cointegration, VAR/VECM, ARDL, DFM/FAVAR**
- Engle, R.F. & Granger, C.W.J. (1987). *Co-integration and Error Correction…* Econometrica 55(2), 251–276.
- Pesaran, M.H., Shin, Y. & Smith, R.J. (2001). *Bounds testing approaches to the analysis of level relationships.* J. Applied Econometrics 16(3), 289–326.
- Chinn, M. & Coibion, O. (2014). *The Predictive Content of Commodity Futures.* J. Futures Markets 34(7), 607–636. https://users.ssc.wisc.edu/~mchinn/chinn_coibion_JFM2014.pdf
- Reichsfeld, D.A. & Roache, S.K. (2011). *Do Commodity Futures Help Forecast Spot Prices?* IMF WP/11/254. https://www.imf.org/external/pubs/ft/wp/2011/wp11254.pdf
- Watkins, C. & McAleer, M. (2002). *Cointegration analysis of metals futures.* Mathematics and Computers in Simulation 59, 207–221. https://www.sciencedirect.com/science/article/abs/pii/S0378475401004098
- Galán-Gutiérrez, J.A., Labeaga, J.M. & Martín-García, R. (2023). *Cointegration between high base metals prices and backwardation…* Resources Policy 81. https://ideas.repec.org/a/eee/jrpoli/v81y2023ics0301420723001216.html
- Ben Salem, S., Nouira, R., Jeguirim, K. & Rault, C. (2022). *The determinants of crude oil prices: Evidence from ARDL and nonlinear ARDL approaches.* Resources Policy 79. https://www.sciencedirect.com/science/article/abs/pii/S0301420722005281
- Lombardi, M.J., Osbat, C. & Schnatz, B. (2010). *Global commodity cycles and linkages: a FAVAR approach.* ECB WP 1170. https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1170.pdf
- Bilgin, D. & Ellwanger, R. (2017). *A Dynamic Factor Model for Commodity Prices.* Bank of Canada SAN 2017-12. https://ideas.repec.org/p/bca/bocsan/17-12.html
- Basistha, A. et al. (2024). *Measuring persistent global economic factors…* J. Forecasting 43(4). https://onlinelibrary.wiley.com/doi/10.1002/for.3139
- Apergis, N., Christou, C. & Payne, J.E. (2014). *Precious metal markets, stock markets and the macroeconomic environment: a FAVAR model approach.* Applied Financial Economics 24(10), 691–703.

**Volatility & density**
- Wang, Z. & Lu, X. (2024). *COMEX Copper Futures Volatility Forecasting: Econometric Models and Deep Learning.* arXiv:2409.08356. https://arxiv.org/html/2409.08356v1
- Wang, Z. & Li, X. (2024). *On the macroeconomic fundamentals of long-term volatilities and dynamic correlations in COMEX copper futures.* arXiv:2409.08355. https://arxiv.org/html/2409.08355v1
- Li, G. & Li, Y. (2015). *Forecasting copper futures volatility under model uncertainty.* Resources Policy 46(P2), 167–176. https://ideas.repec.org/a/eee/jrpoli/v46y2015ip2p167-176.html
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* J. Financial Econometrics 7(2), 174–196.
- Engle, R.F., Ghysels, E. & Sohn, B. (2013). *Stock Market Volatility and Macroeconomic Fundamentals.* Review of Economics and Statistics 95(3), 776–797.
- Conrad, C. & Kleen, O. (2020). *Two are better than one: Volatility forecasting using multiplicative component GARCH-MIDAS models.* J. Applied Econometrics 35(1), 19–45.
- Engle, R.F. & Rangel, J.G. (2008). *The Spline-GARCH Model…* Review of Financial Studies 21(3), 1187–1222.
- Fang, L., Chen, B., Yu, H. & Qian, Y. (2018). *The importance of global economic policy uncertainty in predicting gold futures market volatility: A GARCH-MIDAS approach.* J. Futures Markets 38(3), 413–422.
- Fang, Zhao & Zhong (2022). *Volatility Forecasting of Copper Futures Based on HAR-RV Model.* BCP Business & Management 26, 741–753.
- Čech, F. & Baruník, J. (2019). *Panel quantile regressions for estimating and predicting the Value-at-Risk of commodities.* J. Futures Markets 39(9), 1167–1189.
- Candila, V., Gallo, G.M. & Petrella, L. (2023). *Mixed-frequency quantile regressions to forecast value-at-risk and expected shortfall.* Annals of Operations Research. https://link.springer.com/article/10.1007/s10479-023-05370-x

**Regime-switching & nonlinear**
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica 57(2), 357–384.
- Enders, W. & Siklos, P.L. (2001). *Cointegration and Threshold Adjustment.* J. Business & Economic Statistics 19(2), 166–176.
- Goo, Y.J. & Chen, C.C. (2020). *Asymmetric Momentum Threshold Effect of Copper Futures Returns on Spot Returns Volatility in London Metals Exchange under High Volatility.* Modern Economy 11, 51–61.
- Ubilava, D. (2022). *A comparison of multistep commodity price forecasts using direct and iterated smooth transition autoregressive methods.* Agricultural Economics 53(5), 687–701.
- Ubilava, D. (2018). *The Role of El Niño Southern Oscillation in Commodity Price Movement and Predictability.* American J. of Agricultural Economics 100(1), 239–263.
- Allayioti, A. & Venditti, F. (2024). *The Role of Comovement and Time-Varying Dynamics in Forecasting Commodity Prices.* ECB WP 2901.

**Bayesian & combination**
- Stock, J.H. & Watson, M.W. (2004). *Combination forecasts of output growth in a seven-country data set.* J. Forecasting 23(6), 405–430.
- Genre, V., Kenny, G., Meyler, A. & Timmermann, A. (2013). *Combining expert forecasts: Can anything beat the simple average?* IJF 29(1), 108–121.
- Bates, J.M. & Granger, C.W.J. (1969). *The combination of forecasts.* Operational Research Quarterly 20(4), 451–468.
- Bañbura, M., Giannone, D. & Reichlin, L. (2010). *Large Bayesian vector autoregressions.* J. Applied Econometrics 25(1), 71–92.
- Scott, S.L. & Varian, H.R. (2014). *Predicting the present with Bayesian structural time series.* Int. J. Mathematical Modelling and Numerical Optimisation 5(1/2), 4–23.
- Taylor, S.J. & Letham, B. (2018). *Forecasting at scale.* The American Statistician 72(1), 37–45.
- Naser, H. (2016). *Estimating and forecasting the real prices of crude oil: A data-rich model using a dynamic model averaging (DMA) approach.* Energy Economics 56, 75–87.
- Drachal, K. (2017). *Forecasting Spot Oil Price Using Google Probabilities.* PMLR v58.
- Baumeister, C. & Kilian, L. (2012). *Real-Time Forecasts of the Real Price of Oil.* (real-time MSPE reductions ~25% at 1m, ~24% at 3m vs no-change; gains fade past ~1 year.)
- Diaz, J.D., Hansen, E. & Cabrera, G. (2021). *Economic drivers of commodity volatility: The case of copper.* Resources Policy 73, 102134. https://ideas.repec.org/a/eee/jrpoli/v73y2021ics030142072100235x.html

**Commodity-finance theory**
- Fama, E.F. & French, K.R. (1987). *Commodity Futures Prices: Some Evidence on Forecast Power, Premiums, and the Theory of Storage.* J. Business 60(1), 55–73.
- Fama, E.F. & French, K.R. (1988). *Business Cycles and the Behavior of Metals Prices.* J. Finance 43(5), 1075–1093.
- Geman, H. & Smith, W.O. (2013). *Theory of Storage, Inventory and Volatility in the LME Base Metals.* Resources Policy 38(1), 18–28.
- Gorton, G.B., Hayashi, F. & Rouwenhorst, K.G. (2007/2013). *The Fundamentals of Commodity Futures Returns.* NBER WP 13249 / Review of Finance 17(1), 35–105. (Rejects hedging pressure as a premium driver; attributes premia to inventories.)
- Basu, D. & Miffre, J. (2013). *Capturing the Risk Premium of Commodity Futures: The Role of Hedging Pressure.* J. Banking & Finance 37(7), 2652–2664.
- Koijen, R.S.J., Moskowitz, T.J., Pedersen, L.H. & Vrugt, E.B. (2018). *Carry.* J. Financial Economics 127(2), 197–225.
- Boons, M. & Prado, M.P. (2019). *Basis-Momentum.* J. Finance 74(1), 239–279.
- Shojaeinia, S. (2023). *Metal market analysis: an empirical model for copper supply and demand in US market.* Mineral Economics 36(3), 509–517.

**Interpretable statistical learning**
- Zhang, Y., Ma, F. & Wang, Y. (2019). *Forecasting crude oil prices with a large set of predictors: Can LASSO select powerful predictors?* J. Empirical Finance 54, 97–117.
- Liu, Y., Guo, Y. & Wei, Q. (2024). *Time-varying and multi-scale analysis of copper price influencing factors based on LASSO and EMD methods.* J. Commodity Markets 34. (DOI 10.1016/j.jcomm.2024.100388)
- Liu, C., Hu, Z., Li, Y. & Liu, S. (2017). *Forecasting copper prices by decision tree learning.* Resources Policy 52, 427–434.
- Friedman, J.H. & Popescu, B.E. (2008). *Predictive learning via rule ensembles.* Annals of Applied Statistics 2(3), 916–954.
- Drachal, K. (2022). *Forecasting the Crude Oil Spot Price with Bayesian Symbolic Regression.* Energies 16(1), 4.
- Drachal, K. & Pawłowski, M. (2024). *Forecasting Selected Commodities' Prices with the Bayesian Symbolic Regression.* Int. J. of Financial Studies 12(2), 34.
- Drachal, K. (2019). *Forecasting prices of selected metals with Bayesian data-rich models.* Resources Policy 64, 101528. (lead, nickel, zinc — not copper.)
- Jin, B. & Xu, X. (2024/2025). Gaussian-process-regression metal-price papers (copper, silver, steel index). (Autoregressive; no RW benchmark.)
- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC.

**Decomposition hybrids & evaluation methodology**
- Liu, Q., Liu, M., Zhou, H. & Yan, F. (2022). *A multi-model fusion based non-ferrous metal price forecasting.* Resources Policy 77.
- Yang, Li & Jiang (2024). *Research on information leakage in time series prediction based on empirical mode decomposition.* Scientific Reports 14:28362. https://pmc.ncbi.nlm.nih.gov/articles/PMC11569228/
- Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990). *STL: A Seasonal-Trend Decomposition Procedure Based on Loess.* J. Official Statistics 6(1), 3–73.
- Diebold, F.X. & Mariano, R.S. (1995). *Comparing Predictive Accuracy.* J. Business & Economic Statistics 13(3), 253–263.
- West, K.D. (1996). *Asymptotic Inference about Predictive Ability.* Econometrica 64(5), 1067–1084.
- Clark, T.E. & West, K.D. (2007). *Approximately normal tests for equal predictive accuracy in nested models.* J. Econometrics 138(1), 291–311.
- Giacomini, R. & White, H. (2006). *Tests of Conditional Predictive Ability.* Econometrica 74(6), 1545–1578.
- Hansen, P.R., Lunde, A. & Nason, J.M. (2011). *The Model Confidence Set.* Econometrica 79(2), 453–497.
- Pesaran, M.H. & Timmermann, A. (1992). *A Simple Nonparametric Test of Predictive Performance.* J. Business & Economic Statistics 10(4), 461–465.
- White, H. (2000). *A Reality Check for Data Snooping.* Econometrica 68(5), 1097–1126.
- Hansen, P.R. (2005). *A Test for Superior Predictive Ability.* J. Business & Economic Statistics 23(4), 365–380.

*End of report.*