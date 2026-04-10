# U.S. Recession Probability Model — 12-Month Ahead

A probit regression model that estimates the probability of a U.S. recession 12 months ahead, using NBER recession dates as the binary dependent variable and economic indicators across eight categories.

Based on the methodology of Estrella & Mishkin (1996, 1998), as used by the NY Fed, Cleveland Fed, and Columbia Threadneedle.

## Quick Start (Google Colab)

1. Open `Recession_Probability_Model.ipynb` in Google Colab
2. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/fred/
3. Run all cells — the notebook handles installation, data fetching, modeling, and visualization

## Configuration

All modeling choices are set in a single configuration cell — nothing downstream is hardcoded:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_DEFINITION` | `"point"` | `"point"` = recession at month t+12; `"window"` = any recession in months t+1 through t+12 |
| `OBS_START` | `"1967-01-01"` | Observation start date (CFNAI begins 1967) |
| `MIN_WINDOW` | `120` | Minimum expanding-window training size (months) |
| `MAX_FEATURES_BIC` | `9` | Maximum features for BIC forward selection |
| `THRESHOLD_WARNING` | `30` | Warning probability level (%) |
| `THRESHOLD_ELEVATED` | `50` | Elevated probability level (%) |

## Data Universe — 37 FRED Series Across 8 Categories

The full Columbia Threadneedle indicator universe, plus derived features (SPREAD, UNRATE_CHG3):

1. **National Activity** — CFNAI, CFNAIMA3, GDPC1, USSLIND
2. **Industrial** — INDPRO, NAPM (ISM PMI), TCU, DGORDER, IPMAN
3. **Consumer** — UMCSENT, PCECC96, DSPIC96, RSAFS
4. **Labor Market** — UNRATE, ICSA, PAYEMS, CIVPART, JTSJOL
5. **Inflation** — CPIAUCSL, PCEPILFE, PCEPI, CPILFESL, PPIACO
6. **Housing** — HOUST, PERMIT, HSN1F, CSUSHPISA
7. **Banking/Credit** — BAA10YM, BUSLOANS, DRALACBS, DRTSCILM
8. **Government Bond Yields** — T10Y3M, T10Y2Y, GS10, TB3MS, FEDFUNDS

## Models

| Model | Description | Feature Selection |
|-------|-------------|-------------------|
| NY Fed Baseline | Yield curve spread only | Fixed (Estrella & Mishkin 1998) |
| Wright Extension | Spread + Federal Funds Rate | Fixed (Wright 2006) |
| BIC-Selected | Data-driven optimal set | Forward stepwise by BIC |
| Full Candidate Set | All available features | None (included for comparison) |

## Feature Selection

Features are **not hardcoded**. A forward stepwise BIC procedure:
1. Seeds with SPREAD (strongest single predictor per Estrella & Mishkin)
2. Greedily adds the feature producing the largest BIC improvement
3. Stops when no addition improves BIC, or `MAX_FEATURES_BIC` is reached

This implements Berge (2014)'s finding: "at the 12-month horizon, parsimony dominates."

## Key Features

- All 37 indicator series fetched and transformed (YoY for levels, as-is for stationary)
- BIC-driven feature selection from the full candidate pool
- Configurable dependent variable: point-in-time vs. any-in-window (with side-by-side comparison)
- Four probit specifications: NY Fed, Wright, BIC-selected, full candidate set
- Expanding-window pseudo out-of-sample probability estimation
- AUROC, Brier Score, AIC, BIC, Pseudo R² model evaluation
- NBER recession-shaded probability charts
- Estrella-Mishkin closed-form quick estimate (Estrella & Trubin 2006 parameters)
- Feature importance (z-statistics and marginal effects)
- Historical recession detection table across all models
- Correlation matrix of selected features
- CSV export of all probabilities and indicators

## Dependencies

- `fredapi` — FRED data access
- `statsmodels` — Probit regression
- `scikit-learn` — Model evaluation metrics (AUROC, Brier)
- `matplotlib` — Visualization
- `pandas`, `numpy`, `scipy`

## References

- Estrella, A. & Mishkin, F.S. (1998). "Predicting U.S. Recessions." *Review of Economics and Statistics*
- Wright, J.H. (2006). "The Yield Curve and Predicting Recessions." *FEDS Working Paper*
- Kauppi, H. & Saikkonen, P. (2008). "Predicting U.S. Recessions with Dynamic Binary Response Models." *Review of Economics and Statistics*
- Berge, T.J. (2014). "Predicting Recessions with Leading Indicators." *Kansas City Fed Working Paper*
- Federal Reserve Board FEDS Notes (2018, 2019). Recession probability model comparisons.
- Boston Fed (2020). On dispersion from dependent variable construction choices.
- McCracken, M.W. & Ng, S. (2016). "FRED-MD." *Journal of Business & Economic Statistics*
