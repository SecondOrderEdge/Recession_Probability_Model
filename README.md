# U.S. Recession Probability Model — 12-Month Ahead

A probit regression model that estimates the probability of a U.S. recession 12 months ahead, using NBER recession dates as the binary dependent variable and economic indicators across eight categories.

Based on the methodology of Estrella & Mishkin (1996, 1998), as used by the NY Fed, Cleveland Fed, and Columbia Threadneedle.

## Quick Start (Google Colab)

1. Open `Recession_Probability_Model.ipynb` in Google Colab
2. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/fred/
3. Run all cells — the notebook handles installation, data fetching, modeling, and visualization

## Models

| Model | Description | Reference |
|-------|-------------|-----------|
| NY Fed Baseline | Yield curve spread only (T10Y-3M) | Estrella & Mishkin (1998) |
| Wright Extension | Spread + Federal Funds Rate level | Wright (2006) |
| Full Multi-Category | 9 indicators across 8 categories | Columbia Threadneedle-style |

## Eight Indicator Categories

1. **National Activity** — CFNAI (Chicago Fed National Activity Index)
2. **Industrial** — Industrial Production (YoY), ISM PMI, Capacity Utilization
3. **Consumer** — U. Michigan Consumer Sentiment, Real Disposable Income
4. **Labor Market** — Unemployment Rate, Initial Claims, Nonfarm Payrolls
5. **Inflation** — CPI (YoY), Core PCE
6. **Housing** — Housing Starts (YoY), Building Permits
7. **Banking/Credit** — Baa Corporate Bond Spread, C&I Loans
8. **Government Bond Yields** — 10Y-3M Spread, Fed Funds Rate

## Key Features

- In-sample and expanding-window out-of-sample probability estimation
- AUROC and Brier Score model evaluation
- NBER recession-shaded probability charts
- Estrella-Mishkin closed-form quick estimate
- Feature importance analysis (z-statistics and marginal effects)
- Historical recession detection performance table
- CSV export of all probabilities and indicators

## Dependencies

- `fredapi` — FRED data access
- `statsmodels` — Probit regression
- `scikit-learn` — Model evaluation metrics
- `matplotlib` — Visualization
- `pandas`, `numpy`, `scipy`

## References

- Estrella, A. & Mishkin, F.S. (1998). "Predicting U.S. Recessions." *Review of Economics and Statistics*
- Wright, J.H. (2006). "The Yield Curve and Predicting Recessions." *FEDS Working Paper*
- Kauppi, H. & Saikkonen, P. (2008). "Predicting U.S. Recessions with Dynamic Binary Response Models." *Review of Economics and Statistics*
- Berge, T.J. (2014). "Predicting Recessions with Leading Indicators." *Kansas City Fed Working Paper*