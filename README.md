# U.S. Recession Probability Model — 12-Month Ahead

A probit regression model that estimates the probability of a U.S. recession occurring 12 months in the future, using NBER recession dates as the binary dependent variable and economic indicators drawn from eight macroeconomic categories. The model follows the academic framework established by Estrella and Mishkin (1996, 1998) and currently used by the New York Fed and Cleveland Fed for their public recession probability estimates.

## Quick Start (Google Colab)

1. Open `Recession_Probability_Model.ipynb` in Google Colab
2. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/fred/
3. Run all cells — the notebook handles installation, data fetching, modeling, and visualization

---

## Methodology

### Why probit regression?

The core statistical model is a **probit regression**, which maps a linear combination of economic indicators into a probability bounded between 0 and 1 using the standard normal cumulative distribution function (CDF):

**P(Recession_{t+12} = 1 | X_t) = Phi(a_0 + a_1 * X_t)**

where Phi(.) is the standard normal CDF, X_t is a vector of economic indicators observed at time t, and the forecast horizon is 12 months.

The literature overwhelmingly uses probit rather than logit, largely by convention established by Estrella and Mishkin (1998). The two produce nearly identical results in practice, but probit connects naturally to latent-variable models (the idea that there is an unobserved "economic health" variable that crosses a threshold during recessions), and since virtually every benchmark paper uses probit, comparability favors it. The implementation uses `statsmodels.discrete.discrete_model.Probit`, which provides coefficient p-values, confidence intervals, and pseudo-R-squared — none of which are available from a pure prediction tool like sklearn's `LogisticRegression`.

### Constructing the dependent variable

The dependent variable is binary: 1 if the economy is in recession, 0 otherwise, based on the NBER's official recession dating (FRED series `USREC`). However, since the model forecasts 12 months ahead, the dependent variable must be *led forward* in time. This creates a critical design choice that materially affects the model's output:

**Point-in-time approach** (`TARGET_DEFINITION = "point"`): Sets y_t = 1 if the economy is in an NBER recession specifically in month t+12. This is what the NY Fed uses. It asks: "Will we be in recession exactly 12 months from now?" Implemented as `data['USREC'].shift(-12)`.

**Any-in-window approach** (`TARGET_DEFINITION = "window"`): Sets y_t = 1 if a recession occurs at any point during months t+1 through t+12. This produces higher and more persistent probabilities because the target is "on" for a wider range of observations. Implemented as `data['USREC'].rolling(window=12).max().shift(-12)`.

The Boston Fed (2020) found "considerable dispersion in predicted recession probabilities" depending on this choice. The notebook implements both definitions, fits the model under each, and displays a side-by-side comparison so users can see the impact directly rather than relying on a hardcoded assumption.

### Why these eight indicator categories?

The indicators are organized into eight categories that collectively span the dimensions along which recessions manifest. Each category captures a different transmission mechanism:

1. **National Economic Activity** — Broad composite indexes (CFNAI) that aggregate dozens of underlying series into a single measure of whether the economy is above or below trend growth. The CFNAI is itself the first principal component of 85 indicators; its 3-month moving average below -0.70 has historically signaled recession with 86% accuracy.

2. **Industrial Indicators** — Manufacturing output, capacity utilization, and the ISM PMI. Industrial production (INDPRO) is a key coincident indicator that declines sharply during recessions. The ISM PMI is one of the most-watched leading indicators; readings below 50 indicate manufacturing contraction.

3. **Consumer Measures** — Consumer sentiment and spending. Since consumer spending represents roughly 70% of GDP, weakness here directly reduces aggregate demand. Sharp declines in the University of Michigan Consumer Sentiment Index (UMCSENT) have preceded every recession in its history.

4. **Labor Market** — Unemployment rate, initial claims, and payrolls. The labor market is primarily a coincident or slightly lagging indicator, but *changes* in labor conditions — particularly the Sahm Rule's rapid increase in unemployment and spikes in initial claims — provide timely signals. Initial claims (ICSA) are a genuine leading indicator because employers reduce hiring before layoffs show up in the unemployment rate.

5. **Inflation** — CPI and PCE measures. Rapid inflation does not directly cause recessions, but it triggers Federal Reserve tightening cycles that do. The causal chain is: inflation rises, the Fed raises rates, higher rates slow the economy, and if the tightening is aggressive enough, a recession follows. This category captures the *preconditions* for policy-induced recessions.

6. **Housing** — Housing starts, building permits, and home sales. Housing is the most interest-rate-sensitive sector of the economy and therefore the first to respond to monetary policy changes. Housing starts typically decline 12-18 months before recessions begin, making this category one of the strongest leading indicators.

7. **Banking/Credit** — Corporate bond spreads and lending conditions. The Baa corporate bond spread over Treasuries (BAA10YM) widens sharply before recessions as investors demand higher compensation for credit risk. The Federal Reserve Board's FEDS Notes (2018, 2019) found that the Gilchrist-Zakrajsek excess bond premium alone outperformed either the long- or short-term yield spread, suggesting credit risk pricing contains recession information beyond what the yield curve captures.

8. **Government Bond Yields** — The yield curve spread and the federal funds rate. The 10-year minus 3-month Treasury spread (T10Y3M) is **the single most powerful recession predictor in the academic literature**. Estrella and Mishkin (1998) showed it "typically performs better by itself out of sample than in conjunction with other variables" at horizons beyond one quarter. The yield curve inverts (long rates fall below short rates) when markets expect the Fed to cut rates in the future — which typically happens because the economy has weakened. Wright (2006) showed that adding the nominal federal funds rate *level* significantly improves the term-spread model, because the same spread at different rate levels implies different things about monetary policy stance.

### Feature selection: BIC, not hardcoding

A central problem in recession modeling is that recessions are rare events — roughly 15% of months since 1967. With 30+ candidate features and only a handful of recession episodes in the training data, overfitting is the primary risk. The literature is unambiguous on this point: Berge (2014) found that at the 12-month horizon, "it is difficult to improve on the univariate yield curve model," and equally-weighted averages of many models "on average perform worse than the best-performing univariate model."

Rather than hardcoding a feature list (which embeds the researcher's priors and prevents the data from speaking), this model uses **forward stepwise selection by BIC** (Bayesian Information Criterion). BIC penalizes model complexity more aggressively than AIC, naturally enforcing the parsimony that the literature demands at this horizon.

The procedure:
1. **Univariate screening**: Every candidate feature is fit as a single-variable probit. The resulting BIC ranking reveals which indicators carry the most recession signal on their own.
2. **Forward selection**: Starting from the yield curve spread (the strongest single predictor), features are added one at a time. At each step, the feature producing the largest BIC improvement is added. Selection stops when no addition improves BIC, or when `MAX_FEATURES_BIC` is reached.
3. **Visualization**: A BIC path chart shows exactly how each added feature changed the model's information criterion, making the selection process transparent and auditable.

This approach respects two principles simultaneously: it lets the data determine the optimal feature set, and it builds in a strong prior toward simplicity via BIC's penalty term.

### Variable transformations

Raw economic series must be transformed for stationarity before entering a regression model. The transformation applied to each series is specified in the `SERIES_CONFIG` dictionary and applied automatically — no manual transformation lists exist downstream:

- **Year-over-year percent change** (`"yoy"`): Applied to non-stationary level series like Industrial Production (INDPRO), CPI, Housing Starts, Payrolls, etc. The 12-month percent change removes trend and seasonality while preserving cyclical information.
- **Levels** (`"level"`): Applied to series that are already stationary or bounded — spread variables (T10Y3M, BAA10YM), rates (FEDFUNDS, UNRATE), and normalized indexes (CFNAI, ISM PMI, Capacity Utilization).
- **Derived features**: SPREAD (= GS10 - TB3MS) is constructed manually to provide a monthly yield curve spread with history back to the 1950s, predating the daily T10Y3M series which begins in 1982. UNRATE_CHG3 implements the Sahm-style 3-month moving average change in unemployment.

Getting transformations wrong is the single most common implementation error in recession models. A level series like Industrial Production has a strong upward trend — feeding it directly into a probit would make the model think the economy is always "improving" simply because output grows over time. The year-over-year change removes this trend and isolates the cyclical signal.

### Estimation: expanding windows, not rolling

Most recession papers use **expanding windows** rather than rolling windows for estimation. The reason is statistical power: recessions are rare, so a rolling window of 120 months might contain at most 1-2 recessions, producing wildly unstable coefficient estimates. An expanding window accumulates all available history, giving the model as much recession data as possible to learn from.

The out-of-sample procedure works as follows:
1. Starting from a minimum training window of 120 months (10 years), the probit model is estimated using only data available up to month t.
2. The estimated model generates a probability for month t+12.
3. The window expands by one month, and the process repeats.

This produces a time series of probabilities where each point was generated without any lookahead bias — the model at each date only knew what a real-time practitioner would have known. These out-of-sample probabilities are noisier than in-sample fitted values but provide an honest assessment of the model's predictive ability.

### Threshold calibration

When does a probability become actionable? The unconditional probability of being in recession is roughly 15%, so anything substantially above this is informative. The model uses two configurable thresholds:

- **30% (warning)**: Historically, probabilities in this range have preceded several recessions. Berge and Jorda (2011) found optimal cutpoints between 30% and 60% depending on the researcher's loss function (the relative cost of false alarms vs. missed recessions).
- **50% (elevated)**: At this level, the model indicates recession is more likely than not within the forecast horizon. This is the traditional decision-theoretic threshold for symmetric loss.

Both thresholds are configurable in the notebook's configuration cell.

---

## Configuration

All modeling choices are set in a single configuration cell — nothing downstream is hardcoded:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_DEFINITION` | `"point"` | `"point"` = recession at month t+12 (NY Fed approach); `"window"` = any recession in months t+1 through t+12 |
| `OBS_START` | `"1967-01-01"` | Observation start date (CFNAI begins 1967; earlier dates are possible but exclude some categories) |
| `MIN_WINDOW` | `120` | Minimum expanding-window training size in months (10 years ensures at least 1 recession in training data) |
| `MAX_FEATURES_BIC` | `9` | Maximum features for BIC forward selection (caps complexity) |
| `THRESHOLD_WARNING` | `30` | Warning probability level (%) |
| `THRESHOLD_ELEVATED` | `50` | Elevated probability level (%) |

---

## Data Universe — 37 FRED Series Across 8 Categories

All series sourced from FRED (Federal Reserve Bank of St. Louis), plus two derived features (SPREAD, UNRATE_CHG3):

| Category | Series | Transform |
|----------|--------|-----------|
| **National Activity** | CFNAI, CFNAIMA3, GDPC1, USSLIND | Level / YoY |
| **Industrial** | INDPRO, NAPM, TCU, DGORDER, IPMAN | YoY / Level |
| **Consumer** | UMCSENT, PCECC96, DSPIC96, RSAFS | Level / YoY |
| **Labor Market** | UNRATE, ICSA, PAYEMS, CIVPART, JTSJOL | Level / YoY |
| **Inflation** | CPIAUCSL, PCEPILFE, PCEPI, CPILFESL, PPIACO | YoY |
| **Housing** | HOUST, PERMIT, HSN1F, CSUSHPISA | YoY |
| **Banking/Credit** | BAA10YM, BUSLOANS, DRALACBS, DRTSCILM | Level / YoY |
| **Yields** | T10Y3M, T10Y2Y, GS10, TB3MS, FEDFUNDS | Level |

Non-monthly series are resampled: weekly (ICSA) to monthly mean, daily (T10Y3M, T10Y2Y) to month-end, quarterly (GDPC1, DRALACBS, DRTSCILM) forward-filled to monthly.

---

## Model Specifications

| Model | Features | Rationale |
|-------|----------|-----------|
| **NY Fed Baseline** | Yield curve spread only | Estrella & Mishkin (1998): the spread alone "typically performs better by itself out of sample" at horizons beyond one quarter |
| **Wright Extension** | Spread + Federal Funds Rate | Wright (2006): adding the fed funds level significantly improves fit because the same spread at different rate levels implies different policy stances |
| **BIC-Selected** | Data-driven via forward stepwise BIC | Lets the data determine the optimal feature set while BIC's penalty enforces parsimony |
| **Full Candidate Set** | All available features | Included as an overfitting benchmark — expected to have the best in-sample fit but worst out-of-sample performance |

---

## Evaluation Metrics

The notebook reports five metrics for each model:

- **AUROC** (Area Under the ROC Curve): Measures the model's ability to discriminate between recession and non-recession months. A value of 1.0 means perfect separation; 0.5 means no better than random.
- **Brier Score**: The mean squared error of the probability forecasts. Lower is better. Ranges from 0 (perfect) to 1.
- **Pseudo R-squared** (McFadden's): Compares the fitted model's log-likelihood to the intercept-only model. Higher means the indicators explain more of the recession variation.
- **AIC** (Akaike Information Criterion): Balances fit and complexity with a lighter penalty. Useful for comparison but tends to favor more complex models than BIC.
- **BIC** (Bayesian Information Criterion): Balances fit and complexity with a heavier penalty proportional to log(n). The primary selection criterion in this model.

---

## Notebook Outputs

- **Univariate BIC ranking table**: Every candidate feature ranked by its solo recession-predicting power
- **BIC selection path chart**: How each added feature changed the model's BIC
- **Dependent variable comparison**: Side-by-side charts of point-in-time vs. any-in-window probabilities
- **Main probability chart**: 12-month-ahead recession probability with NBER recession shading, both in-sample and out-of-sample
- **Model comparison chart**: All four specifications overlaid
- **Current reading**: Latest probability from all models with indicator values
- **Estrella-Mishkin quick estimate**: Closed-form probability using pre-estimated parameters from Estrella & Trubin (2006): P(recession) = Phi(-0.6045 - 0.7374 * spread)
- **Feature importance**: z-statistics and marginal effects for BIC-selected features
- **Correlation matrix**: Checks for multicollinearity among selected features
- **Historical recession detection table**: Peak probability in the 6-18 month window before each historical recession, across all models
- **CSV export**: All probabilities, indicator values, and metadata

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fredapi` | FRED API data access |
| `statsmodels` | Probit regression, marginal effects, summary statistics |
| `scikit-learn` | AUROC and Brier Score evaluation |
| `matplotlib` | All charts and visualizations |
| `pandas` | Data manipulation and time series alignment |
| `numpy` | Numerical operations |
| `scipy` | Normal CDF for Estrella-Mishkin closed-form estimate |

---

## References

1. **Estrella, A. & Mishkin, F.S. (1998)**. "Predicting U.S. Recessions: Financial Variables as Leading Indicators." *Review of Economics and Statistics*, 80(1), 45-61. — The foundational paper. Tested financial variables out of sample across 1-8 quarter horizons. Established the yield curve spread as the dominant predictor and the probit framework as the standard tool.

2. **Wright, J.H. (2006)**. "The Yield Curve and Predicting Recessions." *Federal Reserve Board FEDS Working Paper* No. 2006-07. — Extended Estrella-Mishkin by testing four nested probit models. Found that adding the federal funds rate level to the term spread significantly improves fit. Also tested for structural breaks and found no significant instability.

3. **Kauppi, H. & Saikkonen, P. (2008)**. "Predicting U.S. Recessions with Dynamic Binary Response Models." *Review of Economics and Statistics*, 90(4), 777-791. — Introduced dynamic probit models that include lagged recession status or lagged estimated probability as regressors. Their iterative approach outperformed the standard direct-horizon method.

4. **Berge, T.J. (2014)**. "Predicting Recessions with Leading Indicators: Model Averaging and Links to the Financial Crisis." *Federal Reserve Bank of Kansas City Working Paper*. — Applied Bayesian Model Averaging to recession forecasting. Key finding: at the 12-month horizon, it is difficult to improve on the univariate yield curve model. Equally-weighted model averages performed worse than the best univariate model.

5. **Berge, T.J. & Jorda, O. (2011)**. "Evaluating the Classification of Economic Activity into Recessions and Expansions." *American Economic Journal: Macroeconomics*. — Proposed optimal probability cutpoints between 30-60% based on the researcher's loss function.

6. **Federal Reserve Board FEDS Notes (2018, 2019)**. Various notes comparing six probit models. Found that a bivariate model with the term spread plus the Gilchrist-Zakrajsek excess bond premium had the highest out-of-sample AUROC.

7. **Boston Fed (2020)**. On dispersion in predicted recession probabilities from dependent variable construction choices — documented the material difference between point-in-time and any-in-window approaches.

8. **McCracken, M.W. & Ng, S. (2016)**. "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics*, 34(4), 574-589. — The curated 134-series monthly database that provides most of the indicators used in this model, with recommended transformations for stationarity.

9. **Sahm, C. (2019)**. "Direct Stimulus Payments to Individuals." *Brookings Institution*. — Introduced the Sahm Rule: a recession signal based on when the 3-month moving average of unemployment rises 0.50 percentage points above its 12-month low.

10. **Bellego, C. & Ferrara, L. (2009)**. "Forecasting Euro Area Recessions Using Time-Varying Binary Response Models for Financial Variables." *ECB Working Paper*. — Applied PCA-based dimension reduction to recession forecasting, extracting one factor per indicator category.
