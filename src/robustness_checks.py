"""
Task 5.7: Robustness checks for the speed safety analysis.

Three types of checks:
  1. Threshold sensitivity: re-compute speeding rates and logistic regression
     at 20 and 30 km/h thresholds (baseline is 25 km/h)
  2. Subsampling stability: re-run logistic regression on 5 random 50% subsamples
  3. City-specific models: separate logistic regressions for top 5 cities

Outputs:
  - data_parquet/modeling/robustness_results.json — all robustness check results
  - figures/robustness_threshold.png — threshold sensitivity figure
  - figures/robustness_subsampling.png — subsampling stability figure
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, RANDOM_SEED, FIG_DPI

MODELING_DIR = DATA_DIR / "modeling"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_trip_data() -> pd.DataFrame:
    """Load trip modeling data for robustness checks.

    Returns:
        DataFrame with trip-level data.
    """
    print("Loading trip modeling data...")
    df = pd.read_parquet(MODELING_DIR / "trip_modeling.parquet")
    print(f"  Total trips: {len(df):,}")

    # Filter to scooter modes with age data
    scooter_modes = ["SCOOTER_TUB", "SCOOTER_STD", "SCOOTER_ECO"]
    df = df[df["mode"].isin(scooter_modes) & df["age"].notna()].copy()
    print(f"  After filtering: {len(df):,} scooter trips with age data")
    return df


def prepare_logit_predictors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create predictor matrix and outcome for logistic regression.

    Args:
        df: Trip-level DataFrame.

    Returns:
        Tuple of (X predictor matrix, y outcome series).
    """
    # Age group dummies (ref: <20)
    age_dummies = pd.get_dummies(df["age_group"], prefix="age", dtype=float)
    if "age_<20" in age_dummies.columns:
        age_dummies = age_dummies.drop(columns=["age_<20"])

    # Mode dummies (ref: STD)
    mode_dummies = pd.get_dummies(df["mode_clean"], prefix="mode", dtype=float)
    if "mode_STD" in mode_dummies.columns:
        mode_dummies = mode_dummies.drop(columns=["mode_STD"])

    # Time of day dummies (ref: midday)
    tod_dummies = pd.get_dummies(df["time_of_day"], prefix="tod", dtype=float)
    if "tod_midday" in tod_dummies.columns:
        tod_dummies = tod_dummies.drop(columns=["tod_midday"])

    # Day type (weekend indicator)
    weekend = (df["day_type"] == "weekend").astype(float).rename("is_weekend")

    # Province (top 5 + Other, ref: Seoul)
    top_prov = df["province"].value_counts().head(5).index.tolist()
    prov = df["province"].where(df["province"].isin(top_prov), "Other")
    prov_dummies = pd.get_dummies(prov, prefix="prov", dtype=float)
    if "prov_Seoul" in prov_dummies.columns:
        prov_dummies = prov_dummies.drop(columns=["prov_Seoul"])

    X = pd.concat([age_dummies, mode_dummies, tod_dummies, weekend, prov_dummies],
                  axis=1)
    X = X.fillna(0)

    return X, df


def fit_logit(X: pd.DataFrame, y: pd.Series, label: str = "") -> dict:
    """Fit logistic regression and return key coefficients.

    Args:
        X: Predictor matrix.
        y: Binary outcome.
        label: Model label for output.

    Returns:
        Dict with model stats and coefficients.
    """
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const)
    try:
        result = model.fit(method="lbfgs", maxiter=100, disp=False)
    except Exception:
        result = model.fit(method="newton", maxiter=50, disp=False)

    coefs = {}
    for var in X_const.columns:
        if var == "const":
            continue
        idx = list(X_const.columns).index(var)
        coefs[var] = {
            "coef": round(float(result.params.iloc[idx]), 4),
            "or": round(float(np.exp(result.params.iloc[idx])), 4),
            "p_value": round(float(result.pvalues.iloc[idx]), 6),
            "se": round(float(result.bse.iloc[idx]), 4),
        }

    return {
        "label": label,
        "n_obs": len(y),
        "speeding_rate": round(float(y.mean()), 4),
        "pseudo_r2": round(float(result.prsquared), 4),
        "aic": round(float(result.aic), 1),
        "converged": bool(result.mle_retvals.get("converged", True)),
        "coefficients": coefs,
    }


# ---------------------------------------------------------------------------
# Check 1: Threshold sensitivity
# ---------------------------------------------------------------------------

def threshold_sensitivity(df: pd.DataFrame) -> dict:
    """Re-run analysis at different speed thresholds.

    Tests 20, 25, and 30 km/h thresholds.

    Args:
        df: Trip-level DataFrame with max_speed.

    Returns:
        Dict with results for each threshold.
    """
    print("\n" + "=" * 60)
    print("Check 1: Speed Threshold Sensitivity")
    print("=" * 60)

    thresholds = [20, 25, 30]
    results = {}

    for threshold in thresholds:
        print(f"\n  Threshold: {threshold} km/h")

        # Redefine speeding
        y = (df["max_speed_from_profile"] > threshold).astype(int)
        speeding_rate = y.mean()
        print(f"    Speeding rate: {speeding_rate:.3f}")

        # Fit logistic regression
        X, _ = prepare_logit_predictors(df)
        model_result = fit_logit(X, y, label=f"threshold_{threshold}")
        results[f"threshold_{threshold}"] = model_result

        # Print key ORs
        key_vars = ["mode_TUB", "mode_ECO", "age_25-29", "age_30-34"]
        for var in key_vars:
            if var in model_result["coefficients"]:
                c = model_result["coefficients"][var]
                print(f"    {var}: OR={c['or']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Check 2: Subsampling stability
# ---------------------------------------------------------------------------

def subsampling_stability(
    df: pd.DataFrame,
    n_subsamples: int = 5,
    frac: float = 0.5,
) -> dict:
    """Check stability of key estimates across random subsamples.

    Args:
        df: Trip-level DataFrame.
        n_subsamples: Number of subsamples to draw.
        frac: Fraction of data per subsample.

    Returns:
        Dict with results for each subsample.
    """
    print("\n" + "=" * 60)
    print("Check 2: Subsampling Stability")
    print("=" * 60)

    y_full = (df["max_speed_from_profile"] > 25).astype(int)
    X, _ = prepare_logit_predictors(df)

    results = {"n_subsamples": n_subsamples, "fraction": frac, "subsamples": []}

    key_vars = ["mode_TUB", "mode_ECO", "age_20-24", "age_25-29", "age_30-34",
                "is_weekend", "prov_Chungnam"]

    for i in range(n_subsamples):
        seed = RANDOM_SEED + i
        np.random.seed(seed)
        sample_idx = np.random.choice(len(df), size=int(len(df) * frac), replace=False)

        X_sub = X.iloc[sample_idx]
        y_sub = y_full.iloc[sample_idx]

        print(f"\n  Subsample {i+1}/{n_subsamples} (n={len(X_sub):,}, seed={seed})")
        model_result = fit_logit(X_sub, y_sub, label=f"subsample_{i+1}")

        # Extract key coefficients
        sub_coefs = {}
        for var in key_vars:
            if var in model_result["coefficients"]:
                sub_coefs[var] = model_result["coefficients"][var]["or"]

        results["subsamples"].append({
            "seed": seed,
            "n_obs": len(X_sub),
            "speeding_rate": round(float(y_sub.mean()), 4),
            "pseudo_r2": model_result["pseudo_r2"],
            "key_ors": sub_coefs,
        })

    # Compute stability metrics
    print(f"\n  Stability Summary:")
    print(f"  {'Variable':<20} {'Mean OR':>10} {'Std OR':>10} {'CV':>8}")
    print(f"  {'-'*50}")
    stability = {}
    for var in key_vars:
        ors = [s["key_ors"].get(var, np.nan) for s in results["subsamples"]]
        ors = [x for x in ors if not np.isnan(x)]
        if ors:
            mean_or = np.mean(ors)
            std_or = np.std(ors)
            cv = std_or / mean_or if mean_or > 0 else 0
            print(f"  {var:<20} {mean_or:>10.3f} {std_or:>10.4f} {cv:>8.4f}")
            stability[var] = {
                "mean_or": round(mean_or, 4),
                "std_or": round(std_or, 4),
                "cv": round(cv, 4),
            }

    results["stability"] = stability
    return results


# ---------------------------------------------------------------------------
# Check 3: City-specific models
# ---------------------------------------------------------------------------

def city_specific_models(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Run separate logistic regressions for top cities.

    Args:
        df: Trip-level DataFrame.
        top_n: Number of top cities to analyze.

    Returns:
        Dict with results for each city.
    """
    print("\n" + "=" * 60)
    print("Check 3: City-Specific Models")
    print("=" * 60)

    top_cities = df["city"].value_counts().head(top_n).index.tolist()
    print(f"Top {top_n} cities: {top_cities}")

    y_full = (df["max_speed_from_profile"] > 25).astype(int)
    results = {}

    for city in top_cities:
        city_mask = df["city"] == city
        df_city = df[city_mask].copy()
        y_city = y_full[city_mask]

        if len(df_city) < 1000:
            print(f"\n  {city}: too few trips ({len(df_city)}), skipping")
            continue

        print(f"\n  {city} (n={len(df_city):,}, speeding={y_city.mean():.3f})")

        # Simplified predictors (no province since single city)
        age_dummies = pd.get_dummies(df_city["age_group"], prefix="age", dtype=float)
        if "age_<20" in age_dummies.columns:
            age_dummies = age_dummies.drop(columns=["age_<20"])

        mode_dummies = pd.get_dummies(df_city["mode_clean"], prefix="mode", dtype=float)
        if "mode_STD" in mode_dummies.columns:
            mode_dummies = mode_dummies.drop(columns=["mode_STD"])

        tod_dummies = pd.get_dummies(df_city["time_of_day"], prefix="tod", dtype=float)
        if "tod_midday" in tod_dummies.columns:
            tod_dummies = tod_dummies.drop(columns=["tod_midday"])

        weekend = (df_city["day_type"] == "weekend").astype(float).rename("is_weekend")

        X_city = pd.concat([age_dummies, mode_dummies, tod_dummies, weekend], axis=1)
        X_city = X_city.fillna(0)

        # Drop columns with zero variance
        X_city = X_city.loc[:, X_city.std() > 0]

        model_result = fit_logit(X_city, y_city, label=city)
        results[city] = model_result

        # Print key ORs
        key_vars = ["mode_TUB", "mode_ECO", "age_30-34"]
        for var in key_vars:
            if var in model_result["coefficients"]:
                c = model_result["coefficients"][var]
                print(f"    {var}: OR={c['or']:.3f} (p={c['p_value']:.4f})")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity(threshold_results: dict) -> None:
    """Plot how key ORs change with speed threshold.

    Args:
        threshold_results: Results from threshold_sensitivity().
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    thresholds = [20, 25, 30]
    key_vars = {
        "mode_TUB": "TUB mode",
        "mode_ECO": "ECO mode",
        "age_30-34": "Age 30-34",
        "age_20-24": "Age 20-24",
    }

    # Panel 1: Speeding rates
    ax = axes[0]
    rates = [threshold_results[f"threshold_{t}"]["speeding_rate"] for t in thresholds]
    ax.bar(thresholds, rates, width=3, color="#2196F3", edgecolor="black", linewidth=0.5)
    for t, r in zip(thresholds, rates):
        ax.text(t, r + 0.01, f"{r:.1%}", ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Speed Threshold (km/h)")
    ax.set_ylabel("Speeding Rate")
    ax.set_title("(a) Speeding Rate by Threshold")
    ax.set_xticks(thresholds)

    # Panel 2: ORs across thresholds
    ax = axes[1]
    markers = ["o", "s", "^", "D"]
    colors = ["#E74C3C", "#2196F3", "#4CAF50", "#FF9800"]
    for idx, (var, label) in enumerate(key_vars.items()):
        ors = []
        for t in thresholds:
            coef = threshold_results[f"threshold_{t}"]["coefficients"].get(var, {})
            ors.append(coef.get("or", np.nan))
        ax.plot(thresholds, ors, marker=markers[idx], color=colors[idx],
                label=label, linewidth=2, markersize=8)

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Speed Threshold (km/h)")
    ax.set_ylabel("Odds Ratio")
    ax.set_title("(b) Key Odds Ratios by Threshold")
    ax.set_xticks(thresholds)
    ax.legend(fontsize=9, loc="best")
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_threshold.png", dpi=FIG_DPI,
                bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "robustness_threshold.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES_DIR / 'robustness_threshold.png'}")


def plot_subsampling_stability(subsample_results: dict) -> None:
    """Plot coefficient stability across subsamples.

    Args:
        subsample_results: Results from subsampling_stability().
    """
    key_vars = list(subsample_results["stability"].keys())
    n_vars = len(key_vars)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, var in enumerate(key_vars):
        ors = [s["key_ors"].get(var, np.nan)
               for s in subsample_results["subsamples"]]
        ors = [x for x in ors if not np.isnan(x)]
        if ors:
            mean_or = np.mean(ors)
            ax.scatter([i] * len(ors), ors, color="#2196F3", alpha=0.6, s=60, zorder=2)
            ax.scatter([i], [mean_or], color="#E74C3C", s=100, zorder=3,
                       marker="D", edgecolors="black", linewidth=0.5)

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(key_vars, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Odds Ratio (across 50% subsamples)")
    ax.set_title("Subsampling Stability of Key Odds Ratios\n"
                 "(blue = subsample, red diamond = mean)")
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_subsampling.png", dpi=FIG_DPI,
                bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "robustness_subsampling.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES_DIR / 'robustness_subsampling.png'}")


def plot_city_comparison(city_results: dict) -> None:
    """Plot city-specific ORs for key predictors.

    Args:
        city_results: Results from city_specific_models().
    """
    cities = list(city_results.keys())
    key_vars = ["mode_TUB", "mode_ECO", "age_30-34"]
    var_labels = {"mode_TUB": "TUB mode", "mode_ECO": "ECO mode",
                  "age_30-34": "Age 30-34"}

    fig, axes = plt.subplots(1, len(key_vars), figsize=(5 * len(key_vars), 6))
    colors = ["#E74C3C", "#2196F3", "#4CAF50"]

    for ax_idx, var in enumerate(key_vars):
        ax = axes[ax_idx]
        ors = []
        city_labels = []
        ci_low = []
        ci_high = []

        for city in cities:
            if var in city_results[city]["coefficients"]:
                c = city_results[city]["coefficients"][var]
                ors.append(c["or"])
                ci_low.append(np.exp(c["coef"] - 1.96 * c["se"]))
                ci_high.append(np.exp(c["coef"] + 1.96 * c["se"]))
                city_labels.append(city)

        if ors:
            y_pos = range(len(city_labels))
            ax.barh(y_pos, ors, color=colors[ax_idx], alpha=0.7,
                    edgecolor="black", linewidth=0.5, height=0.6)
            ax.errorbar(ors, y_pos, xerr=[
                [o - cl for o, cl in zip(ors, ci_low)],
                [ch - o for o, ch in zip(ors, ci_high)]
            ], fmt="none", color="black", capsize=3)
            ax.axvline(x=1, color="gray", linestyle="--", linewidth=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(city_labels, fontsize=10)
            ax.set_xlabel("Odds Ratio")
            ax.set_title(var_labels.get(var, var), fontsize=12, fontweight="bold")
            ax.invert_yaxis()

    plt.suptitle("City-Specific Odds Ratios for Speeding", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_city_models.png", dpi=FIG_DPI,
                bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "robustness_city_models.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES_DIR / 'robustness_city_models.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all robustness checks."""
    print("=" * 70)
    print("Task 5.7: Robustness Checks")
    print("=" * 70)

    # Load data
    df = load_trip_data()

    # Check 1: Threshold sensitivity
    threshold_results = threshold_sensitivity(df)

    # Check 2: Subsampling stability
    subsample_results = subsampling_stability(df, n_subsamples=5, frac=0.5)

    # Check 3: City-specific models
    city_results = city_specific_models(df, top_n=5)

    # Save all results
    all_results = {
        "threshold_sensitivity": threshold_results,
        "subsampling_stability": {
            k: v for k, v in subsample_results.items()
            if k != "subsamples"
        },
        "subsampling_detail": subsample_results["subsamples"],
        "city_models": {
            city: {k: v for k, v in res.items() if k != "coefficients"}
            for city, res in city_results.items()
        },
        "city_key_ors": {
            city: {
                var: res["coefficients"].get(var, {}).get("or", None)
                for var in ["mode_TUB", "mode_ECO", "age_30-34",
                            "age_20-24", "is_weekend"]
            }
            for city, res in city_results.items()
        },
    }

    output_path = MODELING_DIR / "robustness_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Create plots
    print("\nCreating figures...")
    plot_threshold_sensitivity(threshold_results)
    plot_subsampling_stability(subsample_results)
    if city_results:
        plot_city_comparison(city_results)

    # Summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK SUMMARY")
    print("=" * 70)

    # Threshold summary
    print("\n1. Threshold Sensitivity:")
    for t in [20, 25, 30]:
        r = threshold_results[f"threshold_{t}"]
        tub_or = r["coefficients"].get("mode_TUB", {}).get("or", "N/A")
        print(f"   {t} km/h: speeding_rate={r['speeding_rate']:.1%}, "
              f"TUB OR={tub_or}")

    # Subsampling summary
    print("\n2. Subsampling Stability (CV of key ORs):")
    for var, stats in subsample_results.get("stability", {}).items():
        print(f"   {var}: CV={stats['cv']:.4f} (mean OR={stats['mean_or']:.3f})")

    # City summary
    print("\n3. City-Specific TUB ORs:")
    for city in city_results:
        tub = city_results[city]["coefficients"].get("mode_TUB", {})
        print(f"   {city}: OR={tub.get('or', 'N/A')}, n={city_results[city]['n_obs']:,}")


if __name__ == "__main__":
    main()
