"""
Task 5.3: Mixed-effects linear regression for mean speed.

Model: mean_speed ~ age_group + mode + time_of_day + day_type + province
                   + frac_major_road + dominant_road_class
                   + (1|user_id) + (1|city)

Due to the large dataset, we use a stratified subsample of 100K trips.
Statsmodels MixedLM with random intercepts for user_id.

Outputs:
  - data_parquet/modeling/mixed_effects_results.json -- coefficients, R2, etc.
  - figures/fig_mixed_effects_coefficients.png/pdf -- coefficient plot
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
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, RANDOM_SEED, FIG_DPI

MODELING_DIR = DATA_DIR / "modeling"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 100_000


def load_and_prepare() -> pd.DataFrame:
    """Load trip modeling data merged with road class features.

    Returns:
        DataFrame ready for mixed-effects modeling.
    """
    print("Loading data...")

    # Load trip modeling data
    trip_df = pd.read_parquet(
        MODELING_DIR / "trip_modeling.parquet",
        columns=[
            "route_id", "user_id", "mode", "mode_clean", "province", "city",
            "mean_speed", "max_speed_from_profile", "speeding_rate_25",
            "start_hour", "day_of_week", "is_weekend", "distance",
            "age", "age_group", "time_of_day", "day_type", "is_speeding",
        ],
    )

    # Load road class features
    road_df = pd.read_parquet(
        DATA_DIR / "trip_road_classes.parquet",
        columns=[
            "route_id", "frac_major_road", "frac_residential", "frac_footway",
            "frac_service", "frac_cycleway", "dominant_road_class",
            "n_road_classes",
        ],
    )

    # Merge
    df = trip_df.merge(road_df, on="route_id", how="inner")
    print(f"  Merged: {len(df):,} trips")

    # Filter to scooter modes
    scooter_modes = ["SCOOTER_TUB", "SCOOTER_STD", "SCOOTER_ECO"]
    df = df[df["mode"].isin(scooter_modes)].copy()
    print(f"  Scooter-mode: {len(df):,}")

    # Drop rows without age
    df = df[df["age"].notna()].copy()
    print(f"  With age: {len(df):,}")

    # Create string vars for formulas
    df["age_group_str"] = df["age_group"].astype(str)
    df["mode_str"] = df["mode_clean"].astype(str)
    df["time_of_day_str"] = df["time_of_day"].astype(str)
    df["day_type_str"] = df["day_type"].astype(str)
    df["province_str"] = df["province"].astype(str)
    df["dominant_road_str"] = df["dominant_road_class"].astype(str)

    # Simplify road class for model (too many categories)
    road_map = {
        "motorway": "major", "trunk": "major",
        "primary": "major", "secondary": "major",
        "tertiary": "tertiary", "residential": "residential",
        "service": "service", "cycleway": "cycling",
        "footway": "footway", "unclassified": "other", "other": "other",
    }
    df["road_type"] = df["dominant_road_class"].map(road_map).fillna("other")

    print(f"  Unique users: {df['user_id'].nunique():,}")
    print(f"  Mean speed: {df['mean_speed'].mean():.1f} km/h")

    return df


def run_mixed_effects(df: pd.DataFrame) -> dict:
    """Run mixed-effects linear model on a subsample.

    Args:
        df: Prepared modeling data.

    Returns:
        Dict with model results.
    """
    np.random.seed(RANDOM_SEED)

    # Subsample for computational feasibility
    if len(df) > SAMPLE_SIZE:
        sample = df.sample(SAMPLE_SIZE, random_state=RANDOM_SEED)
        print(f"\nSubsampled to {SAMPLE_SIZE:,} trips")
    else:
        sample = df

    print(f"  Users in sample: {sample['user_id'].nunique():,}")
    print(f"  Cities in sample: {sample['city'].nunique()}")

    # Formula: mean_speed ~ fixed effects + (1|user_id)
    formula = (
        "mean_speed ~ C(age_group_str, Treatment('<20')) "
        "+ C(mode_str, Treatment('STD')) "
        "+ C(time_of_day_str, Treatment('midday')) "
        "+ C(day_type_str, Treatment('weekday')) "
        "+ C(road_type, Treatment('residential')) "
        "+ frac_major_road "
        "+ log_distance"
    )

    # Add log distance
    sample = sample.copy()
    sample["log_distance"] = np.log1p(sample["distance"])

    print(f"\nFitting MixedLM: mean_speed ~ covariates + (1|user_id)...")
    print(f"  Formula: {formula}")

    model = smf.mixedlm(formula, data=sample, groups=sample["user_id"])
    result = model.fit(reml=True)

    print(f"\n  Converged: {result.converged}")
    print(f"  Log-Likelihood: {result.llf:.1f}")
    print(f"  AIC: {result.aic:.1f}")
    print(f"  BIC: {result.bic:.1f}")

    # Random effects variance
    re_var = result.cov_re.iloc[0, 0]
    resid_var = result.scale
    icc = re_var / (re_var + resid_var)
    print(f"  Random intercept variance: {re_var:.4f}")
    print(f"  Residual variance: {resid_var:.4f}")
    print(f"  ICC (user): {icc:.4f}")

    # Extract fixed effects
    coefficients = {}
    summary = result.summary()

    for name in result.fe_params.index:
        coef = result.fe_params[name]
        se = result.bse_fe[name]
        pval = result.pvalues[name]
        ci = result.conf_int().loc[name]
        coefficients[name] = {
            "coef": float(coef),
            "se": float(se),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "pvalue": float(pval),
        }

    # Print key coefficients
    print(f"\n{'Variable':<50s} {'Coef':>8s} {'SE':>8s} {'p-value':>12s}")
    print("-" * 80)
    for name, c in coefficients.items():
        if name == "Intercept":
            continue
        print(f"{name[:50]:<50s} {c['coef']:>8.3f} {c['se']:>8.3f} {c['pvalue']:>12.2e}")

    results = {
        "model_type": "MixedLM (REML)",
        "outcome": "mean_speed",
        "n_obs": len(sample),
        "n_users": sample["user_id"].nunique(),
        "converged": result.converged,
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "random_intercept_var": float(re_var),
        "residual_var": float(resid_var),
        "icc_user": float(icc),
        "coefficients": coefficients,
    }

    return results


def run_ols_full(df: pd.DataFrame) -> dict:
    """Run OLS on the full dataset for comparison.

    Args:
        df: Full modeling data.

    Returns:
        Dict with OLS results.
    """
    print("\nRunning OLS on full dataset for comparison...")

    # Add log distance
    df = df.copy()
    df["log_distance"] = np.log1p(df["distance"])

    formula = (
        "mean_speed ~ C(age_group_str, Treatment('<20')) "
        "+ C(mode_str, Treatment('STD')) "
        "+ C(time_of_day_str, Treatment('midday')) "
        "+ C(day_type_str, Treatment('weekday')) "
        "+ C(road_type, Treatment('residential')) "
        "+ frac_major_road "
        "+ log_distance"
    )

    model = smf.ols(formula, data=df)
    result = model.fit()

    print(f"  N = {result.nobs:.0f}")
    print(f"  R-squared: {result.rsquared:.4f}")
    print(f"  Adj R-squared: {result.rsquared_adj:.4f}")

    coefficients = {}
    for name in result.params.index:
        coefficients[name] = {
            "coef": float(result.params[name]),
            "se": float(result.bse[name]),
            "ci_lower": float(result.conf_int().loc[name, 0]),
            "ci_upper": float(result.conf_int().loc[name, 1]),
            "pvalue": float(result.pvalues[name]),
        }

    return {
        "model_type": "OLS",
        "outcome": "mean_speed",
        "n_obs": int(result.nobs),
        "r_squared": float(result.rsquared),
        "adj_r_squared": float(result.rsquared_adj),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "coefficients": coefficients,
    }


def plot_coefficients(results: dict, name: str) -> None:
    """Create coefficient plot for mixed-effects model.

    Args:
        results: Model results dict.
        name: Figure filename prefix.
    """
    import re

    coeffs = results["coefficients"]

    # Clean labels
    labels = []
    values = []
    cis = []

    for raw_name, c in coeffs.items():
        if raw_name == "Intercept" or raw_name == "Group Var":
            continue

        # Parse label
        m = re.search(r"\[T\.(.+?)\]", raw_name)
        if m:
            cat = m.group(1)
            if "age_group" in raw_name:
                label = f"Age: {cat}"
            elif "mode" in raw_name:
                label = f"Mode: {cat}"
            elif "time_of_day" in raw_name:
                label = f"Time: {cat}"
            elif "day_type" in raw_name:
                label = f"Day: {cat}"
            elif "road_type" in raw_name:
                label = f"Road: {cat}"
            else:
                label = raw_name[:40]
        elif raw_name == "frac_major_road":
            label = "Frac. Major Road"
        elif raw_name == "log_distance":
            label = "log(Distance)"
        else:
            label = raw_name[:40]

        labels.append(label)
        values.append(c["coef"])
        cis.append((c["ci_lower"], c["ci_upper"]))

    # Sort by coefficient value
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    cis = [cis[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(6, len(labels) * 0.35)))

    y_pos = np.arange(len(labels))
    colors = ["#D55E00" if v > 0 else "#0072B2" for v in values]

    ax.barh(y_pos, values, color=colors, alpha=0.7, height=0.6)

    # Error bars
    for i, (ci_low, ci_high) in enumerate(cis):
        ax.plot([ci_low, ci_high], [i, i], color="black", linewidth=1)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Coefficient (km/h)")

    model_type = results["model_type"]
    n = results["n_obs"]
    if "icc_user" in results:
        ax.set_title(f"{model_type}: Mean Speed Coefficients\n"
                     f"(N={n:,}, ICC={results['icc_user']:.3f})")
    else:
        ax.set_title(f"{model_type}: Mean Speed Coefficients\n"
                     f"(N={n:,}, R2={results.get('r_squared', 0):.3f})")

    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.pdf + {name}.png")


def main() -> None:
    """Run mixed-effects analysis."""
    print("=" * 60)
    print("Task 5.3: Mixed-Effects Linear Model for Mean Speed")
    print("=" * 60)

    df = load_and_prepare()

    # Mixed-effects model (subsample)
    mixed_results = run_mixed_effects(df)

    # OLS on full dataset for comparison
    ols_results = run_ols_full(df)

    # Save results
    all_results = {
        "mixed_effects": mixed_results,
        "ols_full": ols_results,
    }
    results_path = MODELING_DIR / "mixed_effects_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")

    # Plots
    plot_coefficients(mixed_results, "fig_mixed_effects_coefficients")
    plot_coefficients(ols_results, "fig_ols_mean_speed_coefficients")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nMixed-effects (N={mixed_results['n_obs']:,}):")
    print(f"  ICC (user): {mixed_results['icc_user']:.4f}")
    print(f"  Key effects on mean speed (km/h):")
    for k, c in mixed_results["coefficients"].items():
        if abs(c["coef"]) > 0.5 and k != "Intercept":
            print(f"    {k[:60]}: {c['coef']:+.2f} km/h (p={c['pvalue']:.2e})")

    print(f"\nOLS full (N={ols_results['n_obs']:,}):")
    print(f"  R-squared: {ols_results['r_squared']:.4f}")


if __name__ == "__main__":
    main()
