"""
Daily Recession Probability Report Generator

Fetches latest FRED data, runs the probit model, generates charts,
and outputs a JSON summary for the Claude API email generator.

Usage:
    python automation/daily_report.py

Required environment variables:
    FRED_API_KEY - Federal Reserve Economic Data API key

Outputs (in automation/output/):
    - recession_probability_gauge.png
    - recession_probability_history.png
    - sensitivity_chart.png
    - daily_summary.json
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from fredapi import Fred
from scipy import stats
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OBS_START = "1967-01-01"
MIN_WINDOW = 120
MAX_FEATURES_BIC = 9
THRESHOLD_WARNING = 30
THRESHOLD_ELEVATED = 50
TARGET_DEFINITION = "point"
MAX_RETRIES = 4

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FRED series definitions
# ---------------------------------------------------------------------------
SERIES_CONFIG = {
    "CFNAI":    {"name": "Chicago Fed National Activity Index",   "category": "National Activity", "transform": "level"},
    "CFNAIMA3": {"name": "CFNAI 3-Month Moving Average",         "category": "National Activity", "transform": "level"},
    "GDPC1":    {"name": "Real GDP",                              "category": "National Activity", "transform": "yoy", "freq": "Q"},
    "USSLIND":  {"name": "Leading Index for the US",              "category": "National Activity", "transform": "level"},
    "INDPRO":   {"name": "Industrial Production Index",           "category": "Industrial", "transform": "yoy"},
    "BSCICP02USM460S": {"name": "OECD Manufacturing Confidence",  "category": "Industrial", "transform": "level"},
    "TCU":      {"name": "Capacity Utilization",                  "category": "Industrial", "transform": "level"},
    "DGORDER":  {"name": "Durable Goods Orders",                  "category": "Industrial", "transform": "yoy"},
    "IPMAN":    {"name": "Industrial Production: Manufacturing",  "category": "Industrial", "transform": "yoy"},
    "UMCSENT":  {"name": "U. Michigan Consumer Sentiment",        "category": "Consumer", "transform": "level"},
    "PCECC96":  {"name": "Real Personal Consumption Expenditures","category": "Consumer", "transform": "yoy"},
    "DSPIC96":  {"name": "Real Disposable Personal Income",       "category": "Consumer", "transform": "yoy"},
    "RSAFS":    {"name": "Advance Retail Sales",                  "category": "Consumer", "transform": "yoy"},
    "UNRATE":   {"name": "Unemployment Rate",                     "category": "Labor", "transform": "level"},
    "ICSA":     {"name": "Initial Unemployment Claims",           "category": "Labor", "transform": "yoy", "freq": "W"},
    "PAYEMS":   {"name": "Total Nonfarm Payrolls",                "category": "Labor", "transform": "yoy"},
    "JTSJOL":   {"name": "Job Openings (JOLTS)",                  "category": "Labor", "transform": "yoy"},
    "CPIAUCSL": {"name": "CPI All Urban Consumers",              "category": "Inflation", "transform": "yoy"},
    "PCEPILFE": {"name": "Core PCE Price Index",                  "category": "Inflation", "transform": "yoy"},
    "PCEPI":    {"name": "PCE Chain-Type Price Index",            "category": "Inflation", "transform": "yoy"},
    "CPILFESL": {"name": "Core CPI",                              "category": "Inflation", "transform": "yoy"},
    "PPIACO":   {"name": "PPI All Commodities",                   "category": "Inflation", "transform": "yoy"},
    "HOUST":    {"name": "Housing Starts",                        "category": "Housing", "transform": "yoy"},
    "PERMIT":   {"name": "Building Permits",                      "category": "Housing", "transform": "yoy"},
    "HSN1F":    {"name": "New One-Family Houses Sold",            "category": "Housing", "transform": "yoy"},
    "CSUSHPISA":{"name": "Case-Shiller National Home Price Index","category": "Housing", "transform": "yoy"},
    "BAA10YM":  {"name": "Baa Corp Bond - 10Y Treasury Spread",  "category": "Banking", "transform": "level"},
    "BUSLOANS": {"name": "Commercial & Industrial Loans",         "category": "Banking", "transform": "yoy"},
    "DRALACBS": {"name": "Delinquency Rate, All Loans",           "category": "Banking", "transform": "level", "freq": "Q"},
    "DRTSCILM": {"name": "Tightening Standards C&I Loans",        "category": "Banking", "transform": "level", "freq": "Q"},
    "T10Y3M":   {"name": "10Y-3M Treasury Spread",               "category": "Yields", "transform": "level", "freq": "D"},
    "T10Y2Y":   {"name": "10Y-2Y Treasury Spread",               "category": "Yields", "transform": "level", "freq": "D"},
    "GS10":     {"name": "10-Year Treasury Yield",                "category": "Yields", "transform": "level"},
    "TB3MS":    {"name": "3-Month Treasury Bill Rate",            "category": "Yields", "transform": "level"},
    "FEDFUNDS": {"name": "Federal Funds Rate",                    "category": "Yields", "transform": "level"},
}

TARGET_SERIES = {
    "USREC": {"name": "NBER Recession Indicator", "category": "Target"},
    "RECPROUSM156N": {"name": "Chauvet-Piger Recession Prob", "category": "Benchmark"},
}


def fetch_fred_data(fred):
    """Fetch all series from FRED with retry logic."""
    raw_data = pd.DataFrame()
    failed = []
    all_series = {**SERIES_CONFIG, **TARGET_SERIES}

    for sid, info in all_series.items():
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                s = fred.get_series(sid, observation_start=OBS_START)
                freq = info.get("freq", "M")
                if freq == "W":
                    s = s.resample("MS").mean()
                elif freq == "D":
                    s = s.resample("MS").last()
                elif freq == "Q":
                    s = s.resample("MS").ffill()
                raw_data[sid] = s
                success = True
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Retry {sid} in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    failed.append(sid)
                    print(f"  FAILED {sid}: {e}")

    raw_data.index = pd.to_datetime(raw_data.index)
    raw_data = raw_data.resample("MS").last()
    print(f"Fetched {len(all_series) - len(failed)}/{len(all_series)} series")
    return raw_data, failed


def engineer_features(data):
    """Apply transforms and build feature columns."""
    if "GS10" in data.columns and "TB3MS" in data.columns:
        data["SPREAD"] = data["GS10"] - data["TB3MS"]

    if "UNRATE" in data.columns:
        ma3 = data["UNRATE"].rolling(3).mean()
        data["UNRATE_CHG3"] = ma3 - ma3.shift(12)

    feat_to_cat = {}
    feature_cols = []

    for sid, info in SERIES_CONFIG.items():
        if sid not in data.columns:
            continue
        if info["transform"] == "yoy":
            col = f"{sid}_YOY"
            data[col] = data[sid].pct_change(12) * 100
            feature_cols.append(col)
            feat_to_cat[col] = info["category"]
        else:
            feature_cols.append(sid)
            feat_to_cat[sid] = info["category"]

    if "SPREAD" in data.columns:
        feature_cols.append("SPREAD")
        feat_to_cat["SPREAD"] = "Yields (derived)"
    if "UNRATE_CHG3" in data.columns:
        feature_cols.append("UNRATE_CHG3")
        feat_to_cat["UNRATE_CHG3"] = "Labor (derived)"

    feature_cols = sorted(set(feature_cols))

    # Target
    if TARGET_DEFINITION == "point":
        data["TARGET"] = data["USREC"].shift(-12)
    else:
        data["TARGET"] = data["USREC"].rolling(window=12).max().shift(-12)

    return data, feature_cols, feat_to_cat


def filter_by_coverage(data, feature_cols, min_coverage=0.80):
    """Exclude short-history features."""
    target_series = data["TARGET"].dropna()
    date_range = target_series.index
    available = []
    for c in feature_cols:
        if c not in data.columns:
            continue
        coverage = data.loc[date_range, c].notna().mean()
        if coverage >= min_coverage:
            available.append(c)
    return available


def has_separation(res):
    """Check for complete separation."""
    if res.prsquared > 0.99:
        return True
    if any(abs(res.params) > 100):
        return True
    if any(np.isnan(res.bse)):
        return True
    return False


# Expected coefficient signs for economic validity
SIGN_CONSTRAINTS = {
    "SPREAD": "negative",       # lower spread = higher recession risk
    "UNRATE_CHG3": "positive",  # rising unemployment = higher recession risk
    "UMCSENT": "negative",      # lower sentiment = higher recession risk
    "BUSLOANS_YOY": "negative", # credit contraction = higher recession risk
}


def _check_sign_constraints(res, selected_feats):
    """Return True if all sign constraints are satisfied."""
    for feat in selected_feats:
        if feat in SIGN_CONSTRAINTS and feat in res.params.index:
            coef = res.params[feat]
            expected = SIGN_CONSTRAINTS[feat]
            if expected == "negative" and coef > 0:
                return False
            if expected == "positive" and coef < 0:
                return False
    return True


def forward_stepwise_bic(y, X_all, feature_names, max_features, seed=None):
    """Forward stepwise BIC selection with separation detection and sign constraints."""
    selected = list(seed) if seed else []
    remaining = [f for f in feature_names if f not in selected]

    if selected:
        X_curr = sm.add_constant(X_all[selected].astype(float))
        best_bic = sm.Probit(y, X_curr).fit(disp=False, method="bfgs", maxiter=300).bic
    else:
        X_curr = sm.add_constant(pd.DataFrame(index=X_all.index))
        best_bic = sm.Probit(y, X_curr).fit(disp=False, method="bfgs", maxiter=300).bic

    while remaining and len(selected) < max_features:
        candidates = []
        for feat in remaining:
            try:
                X_try = sm.add_constant(X_all[selected + [feat]].astype(float))
                res = sm.Probit(y, X_try).fit(disp=False, method="bfgs", maxiter=300)
                if not has_separation(res) and _check_sign_constraints(res, selected + [feat]):
                    candidates.append((feat, res.bic))
            except Exception:
                pass

        if not candidates:
            break

        best_feat, best_candidate_bic = min(candidates, key=lambda x: x[1])
        if best_candidate_bic >= best_bic:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_bic = best_candidate_bic
        print(f"  BIC step {len(selected)}: +{best_feat} (BIC={best_bic:.1f})")

    return selected


def generate_gauge_chart(prob, output_path):
    """Generate the probability gauge chart."""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    for x in range(100):
        if x < THRESHOLD_WARNING:
            color = "#2ecc71"
        elif x < THRESHOLD_ELEVATED:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        ax.axvspan(x, x + 1, alpha=0.3, color=color, lw=0)

    ax.axvline(x=prob, color="#1a1a2e", linewidth=3, zorder=5)
    ax.plot(prob, 0.5, "v", color="#1a1a2e", markersize=15, zorder=5)
    ax.text(prob, 1.3, f"{prob:.1f}%", ha="center", fontsize=20, fontweight="bold")

    ax.axvline(x=THRESHOLD_WARNING, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(x=THRESHOLD_ELEVATED, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks([])
    ax.set_xlabel("Recession Probability (%)")
    ax.set_title("12-Month-Ahead U.S. Recession Probability", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_history_chart(fitted, oos, usrec, output_path):
    """Generate historical probability chart with NBER shading."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(usrec.index, 0, 100, where=usrec.values == 1,
                    color="#e0e0e0", alpha=0.7, label="NBER Recession")
    ax.plot(fitted.index, fitted * 100, color="#1a1a2e", linewidth=1.2,
            label="BIC-Selected Model")

    if oos is not None and len(oos) > 0:
        ax.plot(oos.index, oos * 100, color="#e74c3c", linewidth=1.2,
                alpha=0.7, label="Out-of-Sample")

    ax.axhline(y=THRESHOLD_ELEVATED, color="red", linestyle="--", alpha=0.3)
    ax.axhline(y=THRESHOLD_WARNING, color="orange", linestyle=":", alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Historical 12-Month-Ahead Recession Probability", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_sensitivity_chart(scenario_data, output_path):
    """Generate watchlist chart showing current value vs trigger levels."""
    feats = []
    currents = []
    triggers_30 = []
    triggers_50 = []

    for s in scenario_data:
        feats.append(s["feature"])
        currents.append(s["current_value"])
        triggers_30.append(s.get("trigger_30pct"))
        triggers_50.append(s.get("trigger_50pct"))

    fig, axes = plt.subplots(len(feats), 1, figsize=(10, len(feats) * 1.2))
    if len(feats) == 1:
        axes = [axes]

    for ax, feat, cur, t30, t50, s in zip(axes, feats, currents, triggers_30, triggers_50, scenario_data):
        sd = s["std_dev"]
        lo = cur - 3 * sd
        hi = cur + 3 * sd

        # Background bar
        ax.barh(0, hi - lo, left=lo, height=0.6, color="#f0f0f0", edgecolor="none")

        # Current value marker
        ax.plot(cur, 0, "D", color="#1a1a2e", markersize=10, zorder=5)
        ax.annotate(f"{cur:.1f}", (cur, 0), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, fontweight="bold")

        # Trigger level markers
        if t30 is not None and lo < t30 < hi:
            ax.axvline(x=t30, color="#ff7f0e", linewidth=2, linestyle="--", alpha=0.8)
            ax.annotate(f"30%: {t30:.1f}", (t30, 0), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7, color="#ff7f0e")
        if t50 is not None and lo < t50 < hi:
            ax.axvline(x=t50, color="#d62728", linewidth=2, linestyle="--", alpha=0.8)
            ax.annotate(f"50%: {t50:.1f}", (t50, 0), textcoords="offset points",
                        xytext=(0, -22), ha="center", fontsize=7, color="#d62728")

        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.set_ylabel(feat, fontsize=8, rotation=0, ha="right", va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)

    axes[0].set_title("Watchlist: Current Value vs. Trigger Levels", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_percentile_chart(bic_selected, latest_vals, model_df, feat_to_cat, output_path):
    """Generate horizontal bar chart showing each indicator's historical percentile."""
    pctiles = []
    labels = []
    for j, feat in enumerate(bic_selected):
        pct = (model_df[feat] < latest_vals[j]).mean() * 100
        pctiles.append(pct)
        cat = feat_to_cat.get(feat, "")
        labels.append(f"{feat}\n({cat})")

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(bic_selected) * 0.7)))

    colors = []
    for p in pctiles:
        if p <= 10 or p >= 90:
            colors.append("#d62728")  # red = extreme
        elif p <= 25 or p >= 75:
            colors.append("#ff7f0e")  # orange = notable
        else:
            colors.append("#2ca02c")  # green = normal

    bars = ax.barh(labels, pctiles, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.4, label="Median")
    ax.axvline(x=10, color="red", linestyle=":", alpha=0.3)
    ax.axvline(x=90, color="red", linestyle=":", alpha=0.3)

    for bar, pct in zip(bars, pctiles):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}th", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Historical Percentile")
    ax.set_title("Indicator Dashboard: Where Are We Historically?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_model_comparison_chart(model_probs, output_path):
    """Generate horizontal bar chart comparing all model probabilities."""
    names = list(model_probs.keys())
    probs = list(model_probs.values())

    # Sort by probability
    sorted_pairs = sorted(zip(names, probs), key=lambda x: x[1])
    names = [p[0] for p in sorted_pairs]
    probs = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(names) * 0.55)))
    colors = ["#1f77b4" if n == "BIC-Selected" else "#aec7e8" for n in names]
    bars = ax.barh(names, probs, color=colors, edgecolor="white")

    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=10, fontweight="bold")

    ax.axvline(x=THRESHOLD_WARNING, color="orange", linestyle=":", alpha=0.5,
               label=f"{THRESHOLD_WARNING}% warning")
    ax.set_xlabel("Recession Probability (%)")
    ax.set_title("Model Comparison: Do They Agree?", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(max(probs) * 1.4, THRESHOLD_WARNING + 5))
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_sparklines(bic_selected, data, feat_to_cat, output_path):
    """Generate 12-month trailing sparklines for each BIC-selected indicator."""
    n_feats = len(bic_selected)
    fig, axes = plt.subplots(n_feats, 1, figsize=(10, n_feats * 1.4), sharex=True)
    if n_feats == 1:
        axes = [axes]

    for ax, feat in zip(axes, bic_selected):
        series = data[feat].dropna().tail(24)  # 24 months of context
        if len(series) < 2:
            continue

        # Color based on direction: last 3 months trend
        recent = series.tail(3)
        if len(recent) >= 2:
            trend = recent.iloc[-1] - recent.iloc[0]
        else:
            trend = 0

        color = "#d62728" if trend > 0 and feat in ["UNRATE_CHG3", "PPIACO_YOY", "UMCSENT", "UNRATE", "BAA10YM"] else \
                "#d62728" if trend < 0 and feat in ["SPREAD", "HSN1F_YOY"] else \
                "#2ca02c"

        ax.plot(series.index, series.values, color=color, linewidth=1.5)
        ax.fill_between(series.index, series.values, alpha=0.1, color=color)

        # Current value annotation
        ax.annotate(f"{series.iloc[-1]:.1f}", xy=(series.index[-1], series.iloc[-1]),
                    fontsize=9, fontweight="bold", color=color,
                    xytext=(5, 0), textcoords="offset points")

        cat = feat_to_cat.get(feat, "")
        ax.set_ylabel(feat, fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title("Indicator Trends (Trailing 24 Months)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_probability_trend(ensemble_series, bic_series, output_path):
    """Generate 24-month trailing probability trend chart (ensemble primary)."""
    recent = ensemble_series.tail(24) * 100
    bic_recent = bic_series.tail(24) * 100

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(recent.index, recent.values, color="#1a1a2e", linewidth=2,
            label="5-Model Ensemble")
    ax.fill_between(recent.index, 0, recent.values, alpha=0.15, color="#1a1a2e")
    ax.plot(bic_recent.index, bic_recent.values, color="#aec7e8", linewidth=1,
            linestyle="--", label="BIC-Selected")

    ax.axhline(y=THRESHOLD_WARNING, color="orange", linestyle=":", alpha=0.5,
               label=f"{THRESHOLD_WARNING}% warning")
    ax.axhline(y=THRESHOLD_ELEVATED, color="red", linestyle="--", alpha=0.3,
               label=f"{THRESHOLD_ELEVATED}% elevated")

    # Annotate latest
    ax.annotate(f"{recent.iloc[-1]:.1f}%",
                xy=(recent.index[-1], recent.iloc[-1]),
                fontsize=12, fontweight="bold", color="#1a1a2e",
                xytext=(-40, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#1a1a2e"))

    all_vals = pd.concat([recent, bic_recent])
    ax.set_ylim(0, max(all_vals.max() * 1.5, THRESHOLD_WARNING + 5))
    ax.set_ylabel("Probability (%)")
    ax.set_title("Probability Trend (Is Risk Rising or Falling?)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.grid(True, alpha=0.12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def run():
    """Main pipeline."""
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("ERROR: FRED_API_KEY environment variable not set")
        sys.exit(1)

    fred = Fred(api_key=fred_key)
    run_date = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Daily Recession Probability Report — {run_date} ===\n")

    # 1. Fetch data
    print("Fetching FRED data...")
    raw_data, failed = fetch_fred_data(fred)
    if "USREC" not in raw_data.columns:
        print("CRITICAL: USREC not available. Aborting.")
        sys.exit(1)

    # 2. Engineer features
    print("Engineering features...")
    data, feature_cols, feat_to_cat = engineer_features(raw_data.copy())

    # 3. Filter by coverage
    available_features = filter_by_coverage(data, feature_cols)
    model_df = data[available_features + ["TARGET", "USREC"]].dropna()
    predict_df = data[available_features].dropna()
    print(f"Model dataset: {len(model_df)} obs, {len(available_features)} features")
    print(f"Prediction range extends to: {predict_df.index.max().strftime('%Y-%m')}")

    # 4. BIC selection
    print("\nRunning BIC selection...")
    y = model_df["TARGET"].astype(float)
    seed = ["SPREAD"] if "SPREAD" in available_features else []
    bic_selected = forward_stepwise_bic(y, model_df[available_features],
                                         available_features, MAX_FEATURES_BIC, seed)
    print(f"Selected: {bic_selected}")

    # Fallback: if BIC selection returned only the seed (no valid additions),
    # fall back to Wright two-variable model
    spread_feat = "SPREAD" if "SPREAD" in available_features else available_features[0]
    if len(bic_selected) <= len(seed):
        print("WARNING: No valid BIC combination found. Falling back to Wright model.")
        bic_selected = [f for f in [spread_feat, "FEDFUNDS"] if f in available_features]

    # 5. Fit models
    print("\nFitting models...")
    models = {}

    # NY Fed
    spread_feat = "SPREAD" if "SPREAD" in available_features else available_features[0]
    X_ny = sm.add_constant(model_df[[spread_feat]].astype(float))
    res_ny = sm.Probit(y, X_ny).fit(disp=False, method="bfgs", maxiter=500)
    models["NY Fed (Spread Only)"] = {"model": res_ny, "features": [spread_feat]}

    # Wright
    wright_feats = [f for f in [spread_feat, "FEDFUNDS"] if f in available_features]
    X_wr = sm.add_constant(model_df[wright_feats].astype(float))
    res_wr = sm.Probit(y, X_wr).fit(disp=False, method="bfgs", maxiter=500)
    models["Wright (Spread + FF)"] = {"model": res_wr, "features": wright_feats}

    # BIC-selected
    X_bic = sm.add_constant(model_df[bic_selected].astype(float))
    res_bic = sm.Probit(y, X_bic).fit(disp=False, method="bfgs", maxiter=500)
    models["BIC-Selected"] = {"model": res_bic, "features": bic_selected}

    # Coefficient table diagnostic
    print("\nBIC-Selected coefficient table:")
    print(f"  {'Feature':<20s} {'Coef':>10s} {'Std Err':>10s} {'z':>8s} {'p-value':>10s}")
    print("  " + "-" * 58)
    for k in range(len(res_bic.params)):
        name = res_bic.params.index[k]
        print(f"  {name:<20s} {res_bic.params.iloc[k]:>10.4f} {res_bic.bse.iloc[k]:>10.4f} {res_bic.tvalues.iloc[k]:>8.2f} {res_bic.pvalues.iloc[k]:>10.4f}")

    # Sign constraints are enforced during BIC selection — verify they hold
    sign_warnings = []
    for feat, expected in SIGN_CONSTRAINTS.items():
        if feat in bic_selected and feat in res_bic.params.index:
            coef = res_bic.params[feat]
            violated = (expected == "negative" and coef > 0) or (expected == "positive" and coef < 0)
            if violated:
                msg = f"WARNING: {feat} sign constraint violated (coef={coef:.4f}, expected {expected})"
                print(f"\n  *** {msg}")
                sign_warnings.append(msg)

    # 6. Generate predictions on latest data
    latest_vals = predict_df[bic_selected].iloc[-1].astype(float).values
    latest_xc = np.concatenate([[1.0], latest_vals])
    bic_prob = stats.norm.cdf(latest_xc @ res_bic.params.values) * 100
    # Data-through = last observation of the most-lagged BIC-selected feature
    # (scoped to features driving the headline, not all 37 series)
    bic_last_dates = {}
    for feat in bic_selected:
        # Map derived feature back to raw FRED series
        raw_sid = feat.replace("_YOY", "") if feat.endswith("_YOY") else feat
        if raw_sid in raw_data.columns:
            last_valid = raw_data[raw_sid].last_valid_index()
            if last_valid is not None:
                bic_last_dates[feat] = last_valid
        elif feat in data.columns:
            last_valid = data[feat].last_valid_index()
            if last_valid is not None:
                bic_last_dates[feat] = last_valid
    if bic_last_dates:
        most_lagged = min(bic_last_dates, key=bic_last_dates.get)
        data_date = bic_last_dates[most_lagged].strftime("%Y-%m")
        print(f"Data through: {data_date} (most-lagged BIC feature: {most_lagged})")
    else:
        data_date = predict_df.index[-1].strftime("%Y-%m")

    # Identify series with publication lags > 30 days from run date
    lagged_series = []
    run_dt = pd.Timestamp(run_date)
    for feat, last_dt in bic_last_dates.items():
        lag_days = (run_dt - last_dt).days
        if lag_days > 30:
            lagged_series.append(f"{feat} (last: {last_dt.strftime('%Y-%m')})")
    if lagged_series:
        print(f"Lagged series (>30 days): {lagged_series}")

    model_probs = {}
    for name, m in models.items():
        feats = m["features"]
        x = predict_df[feats].iloc[-1].astype(float).values
        xc = np.concatenate([[1.0], x])
        model_probs[name] = stats.norm.cdf(xc @ m["model"].params.values) * 100

    # Estrella-Mishkin
    if "SPREAD" in data.columns:
        spread_val = data["SPREAD"].dropna().iloc[-1]
        model_probs["Estrella-Mishkin"] = stats.norm.cdf(-0.6045 - 0.7374 * spread_val) * 100

    # Chauvet-Piger smoothed recession probability (independent Markov-switching model)
    if "RECPROUSM156N" in data.columns:
        cp_val = data["RECPROUSM156N"].dropna().iloc[-1]
        model_probs["Chauvet-Piger"] = float(cp_val)

    # Ensemble: equal-weighted average of all model probabilities
    ensemble_prob = np.mean(list(model_probs.values()))
    print(f"\nEnsemble probability: {ensemble_prob:.2f}% (avg of {len(model_probs)} models)")
    for name, p in sorted(model_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<28s}: {p:.1f}%")
    print(f"  {'BIC-Selected (direct)':<28s}: {bic_prob:.1f}%")

    # 7. Bootstrap CI
    print("\nRunning bootstrap (500 iterations)...")
    np.random.seed(42)
    boot_probs = []
    for _ in range(500):
        idx = np.random.choice(len(model_df), size=len(model_df), replace=True)
        bd = model_df.iloc[idx]
        try:
            rb = sm.Probit(bd["TARGET"].astype(float),
                           sm.add_constant(bd[bic_selected].astype(float))).fit(
                           disp=False, method="bfgs", maxiter=200)
            boot_probs.append(stats.norm.cdf(latest_xc @ rb.params.values))
        except Exception:
            pass
    boot_probs = np.array(boot_probs)
    ci_lower = np.percentile(boot_probs, 5) * 100
    ci_upper = np.percentile(boot_probs, 95) * 100
    print(f"90% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")

    # 8. Sensitivity — actionable watchlist levels
    # For each indicator, find: what value would push probability to 30%? To 50%?
    # And show the +/- 1 SD impact for context.
    scenario_data = []
    for j, feat in enumerate(bic_selected):
        sd = model_df[feat].std()
        current = float(latest_vals[j])
        coef = res_bic.params.iloc[j + 1]

        # +/- 1 SD impact (traditional sensitivity)
        x_up = latest_vals.copy()
        x_up[j] += sd
        x_down = latest_vals.copy()
        x_down[j] -= sd
        xc_up = np.concatenate([[1.0], x_up])
        xc_down = np.concatenate([[1.0], x_down])
        prob_up = stats.norm.cdf(xc_up @ res_bic.params.values) * 100
        prob_down = stats.norm.cdf(xc_down @ res_bic.params.values) * 100

        # Find threshold trigger levels: what value of this indicator
        # (holding all others constant) would push probability to 30%? 50%?
        trigger_levels = {}
        for threshold in [THRESHOLD_WARNING, THRESHOLD_ELEVATED]:
            # Binary search for the indicator value that hits the threshold
            # Search in the direction that increases probability
            lo, hi = current - 6 * sd, current + 6 * sd
            for _ in range(60):
                mid = (lo + hi) / 2
                x_test = latest_vals.copy()
                x_test[j] = mid
                xc_test = np.concatenate([[1.0], x_test])
                p_test = stats.norm.cdf(xc_test @ res_bic.params.values) * 100
                if p_test < threshold:
                    if coef > 0:
                        lo = mid
                    else:
                        hi = mid
                else:
                    if coef > 0:
                        hi = mid
                    else:
                        lo = mid
            # Verify we actually found it (not at boundary)
            x_check = latest_vals.copy()
            x_check[j] = mid
            xc_check = np.concatenate([[1.0], x_check])
            p_check = stats.norm.cdf(xc_check @ res_bic.params.values) * 100
            if abs(p_check - threshold) < 1.0:
                trigger_levels[f"trigger_{threshold}pct"] = round(float(mid), 2)
                trigger_levels[f"distance_{threshold}pct"] = round(float(mid - current), 2)
            else:
                trigger_levels[f"trigger_{threshold}pct"] = None
                trigger_levels[f"distance_{threshold}pct"] = None

        scenario_data.append({
            "feature": feat,
            "category": feat_to_cat.get(feat, ""),
            "current_value": current,
            "std_dev": float(sd),
            "prob_minus_1sd": float(prob_down),
            "prob_plus_1sd": float(prob_up),
            "impact_pp": float(prob_up - prob_down),
            **trigger_levels,
        })

    # Adverse scenario
    x_adverse = latest_vals.copy()
    for j, feat in enumerate(bic_selected):
        coef = res_bic.params.iloc[j + 1]
        sd = model_df[feat].std()
        x_adverse[j] += sd if coef > 0 else -sd
    xc_adverse = np.concatenate([[1.0], x_adverse])
    adverse_prob = stats.norm.cdf(xc_adverse @ res_bic.params.values) * 100

    # Print watchlist
    print("\nWatchlist — trigger levels:")
    print(f"  {'Feature':<20s} {'Current':>10s} {'→30% at':>10s} {'Distance':>10s} {'→50% at':>10s} {'Distance':>10s}")
    print("  " + "-" * 70)
    for s in scenario_data:
        t30 = f"{s['trigger_30pct']:.2f}" if s.get('trigger_30pct') is not None else "n/a"
        d30 = f"{s['distance_30pct']:+.2f}" if s.get('distance_30pct') is not None else "n/a"
        t50 = f"{s['trigger_50pct']:.2f}" if s.get('trigger_50pct') is not None else "n/a"
        d50 = f"{s['distance_50pct']:+.2f}" if s.get('distance_50pct') is not None else "n/a"
        print(f"  {s['feature']:<20s} {s['current_value']:>10.2f} {t30:>10s} {d30:>10s} {t50:>10s} {d50:>10s}")

    # 9. Generate charts
    print("\nGenerating charts...")
    generate_gauge_chart(ensemble_prob, OUTPUT_DIR / "recession_probability_gauge.png")

    # Fitted probabilities for history chart
    X_pred_full = sm.add_constant(predict_df[bic_selected].dropna().astype(float))
    fitted_full = res_bic.predict(X_pred_full)
    usrec = data["USREC"].dropna()
    generate_history_chart(fitted_full, None, usrec,
                           OUTPUT_DIR / "recession_probability_history.png")

    generate_sensitivity_chart(scenario_data,
                                OUTPUT_DIR / "sensitivity_chart.png")

    generate_percentile_chart(bic_selected, latest_vals, model_df, feat_to_cat,
                              OUTPUT_DIR / "indicator_percentiles.png")

    generate_model_comparison_chart(model_probs,
                                    OUTPUT_DIR / "model_comparison.png")

    generate_sparklines(bic_selected, data, feat_to_cat,
                        OUTPUT_DIR / "indicator_trends.png")

    # Build ensemble probability series for the trend chart
    ensemble_series = pd.Series(index=fitted_full.index, dtype=float)
    for dt in fitted_full.index:
        probs_at_dt = []
        for name, m in models.items():
            feats = m["features"]
            if all(f in predict_df.columns for f in feats) and dt in predict_df.index:
                x = predict_df.loc[dt, feats].astype(float).values
                xc = np.concatenate([[1.0], x])
                probs_at_dt.append(stats.norm.cdf(xc @ m["model"].params.values))
        if probs_at_dt:
            ensemble_series[dt] = np.mean(probs_at_dt)

    generate_probability_trend(ensemble_series, fitted_full,
                               OUTPUT_DIR / "probability_trend.png")

    # 10. Determine signal level
    if ensemble_prob > THRESHOLD_ELEVATED:
        signal = "HIGH"
    elif ensemble_prob > THRESHOLD_WARNING:
        signal = "ELEVATED"
    else:
        signal = "LOW"

    # Consensus
    prob_values = list(model_probs.values())
    prob_range = max(prob_values) - min(prob_values)
    if prob_range < 15:
        consensus = "STRONG"
    elif prob_range < 30:
        consensus = "MODERATE"
    else:
        consensus = "WEAK"

    # 11. Build summary JSON
    summary = {
        "run_date": run_date,
        "data_through": data_date,
        "lagged_series": lagged_series,
        "ensemble_probability": round(ensemble_prob, 2),
        "bic_probability": round(bic_prob, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "signal": signal,
        "consensus": consensus,
        "model_probabilities": {k: round(v, 2) for k, v in model_probs.items()},
        "bic_selected_features": bic_selected,
        "indicator_readings": {
            feat: {
                "value": round(float(latest_vals[j]), 4),
                "category": feat_to_cat.get(feat, ""),
                "percentile": round(float((model_df[feat] < latest_vals[j]).mean() * 100), 0),
            }
            for j, feat in enumerate(bic_selected)
        },
        "sensitivity": scenario_data,
        "adverse_scenario_probability": round(adverse_prob, 2),
        "sign_warnings": sign_warnings,
        "model_metadata": {
            "training_observations": len(model_df),
            "training_start": model_df.index.min().strftime("%Y-%m"),
            "training_end": model_df.index.max().strftime("%Y-%m"),
            "pseudo_r2": round(res_bic.prsquared, 4),
            "target_definition": TARGET_DEFINITION,
        },
        "charts": [
            "recession_probability_gauge.png",
            "probability_trend.png",
            "indicator_percentiles.png",
            "model_comparison.png",
            "indicator_trends.png",
            "recession_probability_history.png",
            "sensitivity_chart.png",
        ],
    }

    summary_path = OUTPUT_DIR / "daily_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {summary_path}")
    print(f"Signal: {signal} | Ensemble: {ensemble_prob:.2f}% | BIC: {bic_prob:.2f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]")
    print("Done.")
    return summary


if __name__ == "__main__":
    run()
