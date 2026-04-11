# U.S. Recession Probability Model — 12-Month Ahead

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SecondOrderEdge/Recession_Probability_Model/blob/main/Recession_Probability_Model.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/SecondOrderEdge/Recession_Probability_Model?style=social)](https://github.com/SecondOrderEdge/Recession_Probability_Model/stargazers)
[![Weekly Report](https://github.com/SecondOrderEdge/Recession_Probability_Model/actions/workflows/daily_recession_report.yml/badge.svg)](https://github.com/SecondOrderEdge/Recession_Probability_Model/actions/workflows/daily_recession_report.yml)

A five-model ensemble that estimates the probability of a U.S. recession occurring within the next 12 months, using 37 FRED indicators across eight macroeconomic categories. Generates an automated weekly investment committee briefing with embedded charts via Claude API.

Built on the academic framework established by Estrella and Mishkin (1996, 1998) and currently used by the New York Fed and Cleveland Fed.

---

## What This Does

Every Monday morning, a GitHub Actions pipeline:

1. Pulls the latest data from the Federal Reserve's FRED database (37 indicators)
2. Runs a probit regression with BIC-based feature selection and economic sign constraints
3. Computes a five-model ensemble probability with bootstrap confidence intervals
4. Generates 7 charts covering probability trends, indicator dashboards, model comparison, sensitivity triggers, and historical context
5. Sends Claude API the model output + charts, which writes a structured investment committee memo
6. Delivers the memo via email with all charts embedded inline

The ensemble averages across five methodologically distinct models: NY Fed spread-only, Wright two-factor, BIC-selected multivariate, Estrella-Mishkin closed-form, and Chauvet-Piger Markov-switching.

---

## Quick Start

### Option A: Google Colab (interactive exploration)

1. Open `Recession_Probability_Model.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/fred/
3. Run all cells — handles installation, data fetching, modeling, and visualization

### Option B: Automated weekly reports (fork and deploy)

1. **Fork this repository**

2. **Add GitHub Secrets** (Settings > Secrets and variables > Actions):

   | Secret | Required | Description |
   |--------|----------|-------------|
   | `FRED_API_KEY` | Yes | Free key from https://fred.stlouisfed.org/docs/api/fred/ |
   | `ANTHROPIC_API_KEY` | Yes | Claude API key from https://console.anthropic.com/ |
   | `MAIL_USERNAME` | No | Gmail address for sending reports |
   | `MAIL_PASSWORD` | No | Gmail app password (from https://myaccount.google.com/apppasswords) |
   | `MAIL_PORT` | No | SMTP port, typically `587` |
   | `EMAIL_TO` | No | Recipient email(s), comma-separated |

3. **Enable the workflow**: Actions tab > "Weekly Recession Probability Report" > Enable

4. **Test**: Click "Run workflow" to trigger immediately

Without Gmail credentials, the email report is saved as an HTML file in the workflow artifacts.

### Option C: Run locally

```bash
git clone https://github.com/SecondOrderEdge/Recession_Probability_Model.git
cd Recession_Probability_Model

pip install -r requirements.txt

export FRED_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

python automation/daily_report.py      # Fetch data, run model, generate charts
python automation/generate_email.py    # Generate email via Claude API
```

Output files are written to `automation/output/`.

---

## How It Works

### The Probit Framework

The core statistical model is a probit regression:

**P(Recession_{t+12} = 1 | X_t) = Phi(a_0 + a_1 * X_t)**

where Phi(.) is the standard normal CDF, X_t is a vector of economic indicators observed at time t, and the forecast horizon is 12 months. The probit maps a linear combination of indicators into a probability bounded between 0 and 1, connecting naturally to latent-variable models where an unobserved "economic health" index crosses a threshold during recessions.

### Five-Model Ensemble

Rather than relying on any single specification, the headline probability is an equal-weighted average of five methodologically distinct models:

| Model | Features | Methodology | Reference |
|-------|----------|-------------|-----------|
| **NY Fed Baseline** | Yield curve spread only | Re-estimated probit | Estrella & Mishkin (1998) |
| **Wright Extension** | Spread + Federal Funds Rate | Re-estimated probit | Wright (2006) |
| **BIC-Selected** | Data-driven optimal set | Re-estimated probit with sign constraints | Forward stepwise BIC |
| **Estrella-Mishkin** | Yield curve spread | Closed-form with 2006 parameters | Estrella & Trubin (2006) |
| **Chauvet-Piger** | Markov-switching model | Independent FRED series (RECPROUSM156N) | Chauvet & Piger |

This diversifies across model complexity (1 variable to 7+), estimation approach (re-estimated vs. frozen parameters vs. Markov-switching), and the fundamental question of whether the yield curve alone is sufficient or multi-factor models add value.

### Feature Selection: BIC with Sign Constraints

Recessions are rare (~15% of months since 1967). With 30+ candidate features and only ~6 recession episodes in the sample, overfitting is the primary risk. The model uses forward stepwise BIC selection with three safeguards:

1. **BIC penalty**: Penalizes complexity more aggressively than AIC, enforcing the parsimony that Berge (2014) showed dominates at the 12-month horizon
2. **Separation detection**: Rejects any feature combination that produces quasi-complete separation (coefficients > 100, nan standard errors, or pseudo R-squared > 0.99)
3. **Economic sign constraints**: Ensures coefficients align with theory before accepting a candidate:

| Indicator | Required Sign | Economic Logic |
|-----------|--------------|----------------|
| SPREAD | Negative | Lower spread = higher recession risk |
| UNRATE_CHG3 | Positive | Rising unemployment = higher recession risk |
| UMCSENT | Negative | Lower sentiment = higher recession risk |
| BUSLOANS_YOY | Negative | Credit contraction = higher recession risk |

If no valid combination passes all three checks, the model falls back to the Wright two-variable specification (spread + fed funds rate).

### Dependent Variable Construction

The dependent variable is configurable:

- **Point-in-time** (`"point"`): y_t = 1 if in recession at month t+12. Used by the NY Fed.
- **Any-in-window** (`"window"`): y_t = 1 if recession occurs at any point during months t+1 through t+12. Produces higher, more persistent probabilities.

The Boston Fed (2020) documented "considerable dispersion" between these approaches. The notebook compares both side-by-side.

### Variable Transformations

Transforms are metadata-driven via the `SERIES_CONFIG` dictionary:

- **Year-over-year percent change** (`"yoy"`): For non-stationary level series (INDPRO, CPI, Housing Starts, Payrolls). Removes trend, preserves cyclical signal.
- **Levels** (`"level"`): For stationary/bounded series (spreads, rates, normalized indexes).
- **Derived features**: SPREAD (= GS10 - TB3MS) for pre-1982 monthly history; UNRATE_CHG3 for Sahm-style unemployment momentum.

### Estimation: Expanding Windows

Uses expanding-window (not rolling) estimation because recessions are too rare for rolling windows to produce stable estimates. Minimum training window: 120 months (10 years). Each month's probability is generated from a model trained only on data available at that point — no lookahead bias.

### Robustness Checks

The notebook includes five robustness checks:

1. **Bootstrap 90% CI** — 1,000 resamples to quantify uncertainty around the point estimate
2. **Leave-one-recession-out** — Trains on all recessions except one, tests detection on the held-out episode
3. **Penalized probit** — L2-regularized logistic regression to verify quasi-separation isn't distorting estimates
4. **Model consensus** — Measures agreement across all five specifications
5. **Sensitivity triggers** — Binary search for the exact indicator value that would push probability to 30% or 50%

### Trend Attribution

The weekly report decomposes the 24-month probability change into per-indicator contributions using partial effects (coefficient x change in variable x normal PDF at the current linear index). This identifies which indicators are driving the probability up or down.

---

## Data Universe — 37 FRED Series Across 8 Categories

| Category | Series | Transform |
|----------|--------|-----------|
| **National Activity** | CFNAI, CFNAIMA3, GDPC1, USSLIND | Level / YoY |
| **Industrial** | INDPRO, BSCICP02USM460S (OECD Mfg Confidence), TCU, DGORDER, IPMAN | YoY / Level |
| **Consumer** | UMCSENT, PCECC96, DSPIC96, RSAFS | Level / YoY |
| **Labor Market** | UNRATE, ICSA, PAYEMS, JTSJOL | Level / YoY |
| **Inflation** | CPIAUCSL, PCEPILFE, PCEPI, CPILFESL, PPIACO | YoY |
| **Housing** | HOUST, PERMIT, HSN1F, CSUSHPISA | YoY |
| **Banking/Credit** | BAA10YM, BUSLOANS, DRALACBS, DRTSCILM | Level / YoY |
| **Yields** | T10Y3M, T10Y2Y, GS10, TB3MS, FEDFUNDS | Level |

Non-monthly series are resampled: weekly (ICSA) to monthly mean, daily (T10Y3M, T10Y2Y) to month-end, quarterly (GDPC1, DRALACBS, DRTSCILM) forward-filled to monthly. Features with less than 80% coverage of the target period are excluded to prevent short-history series from shrinking the sample.

---

## Weekly Email Report

The automated email includes:

| Section | Content |
|---------|---------|
| **IC Summary** | Snapshot table (probability, direction, risk driver, offset), model range, portfolio positioning (equities, fixed income, credit, hedging) |
| **Executive Summary** | Ensemble headline, model consensus, divergence framing |
| **Key Indicators** | Four macro buckets (Growth, Inflation, Policy, Market Signals) — two sentences each |
| **Model Divergence** | Structured yield curve analysis: historical track record, three structural factors (QE, foreign CB demand, Basel III), judgment call |
| **Watchlist** | Trigger levels: exact values each indicator needs to reach for 30%/50% probability, ranked by proximity |
| **Adverse Scenario** | Tail risk classification, three real-world triggers, hedging framework tied to ensemble level |
| **What Would Change Our View** | Five hardcoded thresholds (spread < -1%, core CPI < 1.5%, housing -15% YoY, claims > 300K, ensemble > 20%) |
| **Data Currency Notice** | Data-through date and lagged series disclosure |

Seven charts embedded inline: probability gauge, ensemble trend, indicator percentile dashboard, model comparison, indicator sparklines, full historical probability, and sensitivity trigger levels.

---

## Configuration

All modeling choices are set in a single configuration block:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_DEFINITION` | `"point"` | `"point"` = recession at month t+12; `"window"` = any recession in months t+1 through t+12 |
| `OBS_START` | `"1967-01-01"` | Observation start date |
| `MIN_WINDOW` | `120` | Minimum expanding-window training size (months) |
| `MAX_FEATURES_BIC` | `9` | Maximum features for BIC selection |
| `THRESHOLD_WARNING` | `30` | Warning probability level (%) |
| `THRESHOLD_ELEVATED` | `50` | Elevated probability level (%) |

---

## Project Structure

```
Recession_Probability_Model/
├── Recession_Probability_Model.ipynb   # Interactive notebook (Google Colab)
├── automation/
│   ├── daily_report.py                 # Model run, charts, summary JSON
│   ├── generate_email.py              # Claude API email composition + Gmail delivery
│   └── output/                        # Generated charts, summaries, emails
├── .github/workflows/
│   └── daily_recession_report.yml     # GitHub Actions weekly schedule
├── requirements.txt
├── .gitignore
├── LICENSE                            # MIT
└── README.md
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fredapi` | >= 0.5.0 | FRED API data access |
| `statsmodels` | >= 0.14 | Probit regression, marginal effects, summary statistics |
| `scikit-learn` | >= 1.3 | AUROC and Brier Score evaluation |
| `matplotlib` | >= 3.7 | Charts and visualizations |
| `pandas` | >= 2.0 | Data manipulation and time series alignment |
| `numpy` | >= 1.24 | Numerical operations |
| `scipy` | >= 1.10 | Normal CDF/PDF for probit predictions |
| `anthropic` | >= 0.40 | Claude API for email narrative generation |

Install all: `pip install -r requirements.txt`

---

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **AUROC** | Discrimination ability (1.0 = perfect, 0.5 = random) |
| **Brier Score** | Mean squared probability error (lower = better) |
| **Pseudo R-squared** | Variance explained vs. intercept-only model |
| **AIC** | Fit-complexity tradeoff (lighter penalty) |
| **BIC** | Fit-complexity tradeoff (heavier penalty — primary selection criterion) |

---

## Known Limitations

- Trained on ~6 independent recession episodes — limited sample for generalization
- Assumes future recessions resemble historical patterns — novel mechanisms (pandemics, geopolitical shocks) may not be captured
- 12-month horizon is long — conditions can change materially within the window
- FRED data has publication lags (1-3 months) — the latest reading may reflect conditions from 1-2 months ago
- The model assigns a negative coefficient to inflation, reflecting demand-collapse recessions — this may understate stagflation risk where inflation and growth weakness occur simultaneously
- Feature selection (BIC) is in-sample — the optimal feature set may differ in future regimes
- The model should be one input among many — not a sole basis for allocation decisions

---

## References

1. **Estrella, A. & Mishkin, F.S. (1998)**. "Predicting U.S. Recessions: Financial Variables as Leading Indicators." *Review of Economics and Statistics*, 80(1), 45-61.

2. **Wright, J.H. (2006)**. "The Yield Curve and Predicting Recessions." *Federal Reserve Board FEDS Working Paper* No. 2006-07.

3. **Kauppi, H. & Saikkonen, P. (2008)**. "Predicting U.S. Recessions with Dynamic Binary Response Models." *Review of Economics and Statistics*, 90(4), 777-791.

4. **Berge, T.J. (2014)**. "Predicting Recessions with Leading Indicators: Model Averaging and Links to the Financial Crisis." *Federal Reserve Bank of Kansas City Working Paper*.

5. **Berge, T.J. & Jorda, O. (2011)**. "Evaluating the Classification of Economic Activity into Recessions and Expansions." *American Economic Journal: Macroeconomics*.

6. **Federal Reserve Board FEDS Notes (2018, 2019)**. Comparative evaluation of six probit recession models.

7. **Boston Fed (2020)**. On dispersion in predicted recession probabilities from dependent variable construction choices.

8. **McCracken, M.W. & Ng, S. (2016)**. "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics*, 34(4), 574-589.

9. **Sahm, C. (2019)**. "Direct Stimulus Payments to Individuals." *Brookings Institution*. — Introduced the Sahm Rule.

10. **Bellego, C. & Ferrara, L. (2009)**. "Forecasting Euro Area Recessions Using Time-Varying Binary Response Models for Financial Variables." *ECB Working Paper*.

---

## Contributing

Contributions are welcome. If you'd like to extend the model:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and test locally with `python automation/daily_report.py`
4. Submit a pull request with a clear description of what changed and why

Areas where contributions would be particularly valuable:
- Additional model specifications (dynamic probit, Bayesian Model Averaging)
- Alternative data sources beyond FRED
- International recession models (EU, UK, Japan)
- Dashboard/web UI for the weekly report

---

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice, investment advice, or a recommendation to buy, sell, or hold any security or financial instrument. The model output should not be used as the sole basis for any investment decision. Past recession-prediction accuracy does not guarantee future performance. Economic models are inherently uncertain and can fail without warning, particularly during novel economic conditions. The authors assume no liability for any financial losses incurred from the use of this model. Consult a qualified financial advisor before making investment decisions.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Suggested Repository Topics

`recession` `macroeconomics` `probit` `yield-curve` `fred-api` `economic-forecasting` `recession-probability` `nber` `time-series` `python` `statsmodels` `claude-api` `github-actions`
