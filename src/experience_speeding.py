"""
Task 1.1: Habitual driver speeding analysis.

Analyzes how riding experience affects speeding behavior using 24-month
longitudinal data. Key questions:
  - Does more experience increase or decrease speeding?
  - Is the relationship monotonic or non-linear?
  - Does experience interact with speed governor mode?

Outputs:
  - figures/fig_experience_learning_curve.pdf
  - figures/fig_experience_usage_category.pdf
  - data_parquet/modeling/experience_results.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, MODELING_DIR, FIG_DPI, RANDOM_SEED

# Paths
TRIP_EXPERIENCE = MODELING_DIR / "trip_experience.parquet"
RESULTS_PATH = MODELING_DIR / "experience_results.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELING_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (colorblind-friendly)
COLORS = {
    "STD": "#0072B2",
    "TUB": "#D55E00",
    "ECO": "#009E73",
    "none": "#CC79A7",
    "overall": "#333333",
}


def load_data() -> pd.DataFrame:
    """Load trip experience data via DuckDB."""
    con = duckdb.connect()
    path = str(TRIP_EXPERIENCE).replace("\\", "/")
    df = con.execute(f"""
        SELECT route_id, user_id, mode, start_date, trip_rank,
               days_since_first_trip, months_since_first_trip,
               experience_bin, usage_category, user_total_trips,
               is_speeding, mean_speed_from_speeds, max_speed_from_speeds,
               age, month_year
        FROM read_parquet('{path}')
        WHERE mode IN ('STD', 'TUB', 'ECO', 'none')
    """).fetchdf()
    con.close()
    print(f"Loaded {len(df):,} trips")
    return df


def analysis_1_usage_category(df: pd.DataFrame) -> dict:
    """Speeding rate by usage category with bootstrap CI.

    Returns dict with results.
    """
    print("\n--- Analysis 1: Speeding by Usage Category ---")
    results = {}

    cat_order = ["one_time", "occasional", "regular", "frequent", "heavy", "super_heavy"]
    cat_labels = ["One-time\n(1)", "Occasional\n(2-5)", "Regular\n(6-20)",
                  "Frequent\n(21-50)", "Heavy\n(51-200)", "Super-heavy\n(200+)"]

    cat_stats = []
    for cat in cat_order:
        subset = df[df["usage_category"] == cat]
        n_users = subset["user_id"].nunique()
        n_trips = len(subset)
        rate = subset["is_speeding"].mean()
        # Wilson CI
        if n_trips > 0:
            ci = sm.stats.proportion_confint(
                int(subset["is_speeding"].sum()), n_trips,
                alpha=0.05, method="wilson"
            )
        else:
            ci = (0, 0)
        cat_stats.append({
            "category": cat,
            "n_users": n_users,
            "n_trips": n_trips,
            "speeding_rate": round(rate, 4),
            "ci_lower": round(ci[0], 4),
            "ci_upper": round(ci[1], 4),
        })
        print(f"  {cat:<12}: {n_users:>8,} users, {n_trips:>10,} trips, "
              f"speeding={rate:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    results["usage_category_stats"] = cat_stats

    # Figure: bar chart with CI
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cat_order))
    rates = [s["speeding_rate"] for s in cat_stats]
    ci_lo = [s["ci_lower"] for s in cat_stats]
    ci_hi = [s["ci_upper"] for s in cat_stats]
    yerr_lo = [r - lo for r, lo in zip(rates, ci_lo)]
    yerr_hi = [hi - r for r, hi in zip(rates, ci_hi)]

    bars = ax.bar(x, rates, color="#0072B2", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, rates, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel("Speeding Rate (max speed > 25 km/h)", fontsize=11)
    ax.set_xlabel("User Category (total trips)", fontsize=11)
    ax.set_ylim(0, max(rates) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Add count annotations
    for i, s in enumerate(cat_stats):
        ax.annotate(f"n={s['n_trips']:,}",
                    xy=(i, rates[i] + yerr_hi[i] + 0.005),
                    ha="center", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_experience_usage_category.{fmt}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved: fig_experience_usage_category.pdf/png")

    return results


def analysis_2_learning_curve(df: pd.DataFrame) -> dict:
    """Within-user learning curve: speeding rate by trip_rank bin.

    Filters to users with 10+ trips, computes speeding rate
    at each trip rank bin, separately by mode.
    """
    print("\n--- Analysis 2: Learning Curve by Trip Rank ---")
    results = {}

    # Filter users with 10+ trips
    users_10plus = df.groupby("user_id").size()
    users_10plus = users_10plus[users_10plus >= 10].index
    df_10 = df[df["user_id"].isin(users_10plus)].copy()
    print(f"  Users with 10+ trips: {len(users_10plus):,}")
    print(f"  Trips from these users: {len(df_10):,}")
    results["n_users_10plus"] = len(users_10plus)

    # Create trip rank bins
    rank_bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 50000]
    rank_labels = ["1", "2-5", "6-10", "11-20", "21-50", "51-100",
                   "101-200", "201-500", "501-1K", "1K-5K", "5K+"]
    df_10["rank_bin"] = pd.cut(
        df_10["trip_rank"], bins=rank_bins, labels=rank_labels,
        include_lowest=True, right=True
    )

    # Overall learning curve
    overall_curve = (
        df_10.groupby("rank_bin", observed=True)["is_speeding"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    overall_curve.columns = ["rank_bin", "speeding_rate", "n_trips", "n_speeding"]

    # Mode-specific curves
    modes = ["STD", "TUB", "ECO"]
    mode_curves = {}
    for mode in modes:
        sub = df_10[df_10["mode"] == mode]
        if len(sub) < 100:
            continue
        curve = (
            sub.groupby("rank_bin", observed=True)["is_speeding"]
            .agg(["mean", "count"])
            .reset_index()
        )
        curve.columns = ["rank_bin", "speeding_rate", "n_trips"]
        mode_curves[mode] = curve

    # Figure: learning curves by mode
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot overall
    valid = overall_curve[overall_curve["n_trips"] >= 50]
    ax.plot(range(len(valid)), valid["speeding_rate"].values,
            "o-", color=COLORS["overall"], linewidth=2, markersize=6,
            label=f"Overall (n={len(df_10):,})", zorder=5)

    # Plot per mode
    for mode in modes:
        if mode not in mode_curves:
            continue
        curve = mode_curves[mode]
        valid_m = curve[curve["n_trips"] >= 50]
        ax.plot(range(len(valid_m)), valid_m["speeding_rate"].values,
                "s--", color=COLORS[mode], linewidth=1.5, markersize=5,
                alpha=0.8, label=f"{mode} (n={len(df_10[df_10['mode']==mode]):,})")

    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid["rank_bin"].values, fontsize=9, rotation=45)
    ax.set_xlabel("Trip Rank (chronological)", fontsize=11)
    ax.set_ylabel("Speeding Rate (max speed > 25 km/h)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_experience_learning_curve.{fmt}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved: fig_experience_learning_curve.pdf/png")

    results["overall_learning_curve"] = [
        {"rank_bin": row["rank_bin"], "speeding_rate": round(row["speeding_rate"], 4),
         "n_trips": int(row["n_trips"])}
        for _, row in overall_curve.iterrows()
    ]

    return results


def analysis_3_logistic_gee(df: pd.DataFrame) -> dict:
    """GEE logistic regression for experience effect on speeding.

    Model: is_speeding ~ log(trip_rank) + mode + age_group + time_features
    Clustered by user_id (exchangeable correlation).
    """
    print("\n--- Analysis 3: Logistic GEE for Experience Effect ---")
    results = {}

    # Prepare data — sample for GEE (very large data)
    np.random.seed(RANDOM_SEED)
    # Use a subsample of users for GEE (computationally intensive)
    unique_users = df["user_id"].unique()
    n_sample = min(50000, len(unique_users))
    sampled_users = np.random.choice(unique_users, n_sample, replace=False)
    df_gee = df[df["user_id"].isin(sampled_users)].copy()

    # Filter to scooter modes (exclude bike modes)
    df_gee = df_gee[df_gee["mode"].isin(["STD", "TUB", "ECO", "none"])].copy()

    # Create features
    df_gee["log_trip_rank"] = np.log(df_gee["trip_rank"].clip(lower=1))
    df_gee["log_trip_rank_sq"] = df_gee["log_trip_rank"] ** 2
    df_gee["is_speeding_int"] = df_gee["is_speeding"].astype(int)

    # Age groups
    df_gee["age_group"] = pd.cut(
        df_gee["age"], bins=[0, 20, 25, 30, 35, 40, 100],
        labels=["under20", "20-24", "25-29", "30-34", "35-39", "40plus"],
        right=False
    ).astype(str)
    df_gee["age_group"] = df_gee["age_group"].fillna("unknown")

    # Mode dummies
    mode_dummies = pd.get_dummies(df_gee["mode"], prefix="mode", drop_first=False, dtype=float)
    mode_dummies = mode_dummies.drop(columns=["mode_STD"], errors="ignore")  # reference

    # Build design matrix
    X_cols = ["log_trip_rank"]
    X = df_gee[X_cols].astype(float).copy()
    for col in mode_dummies.columns:
        X[col] = mode_dummies[col].astype(float).values

    # Add constant
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)
    y = df_gee["is_speeding_int"].astype(float)
    groups = df_gee["user_id"]

    # Drop rows with NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    groups = groups[mask].reset_index(drop=True)

    print(f"  GEE sample: {len(X):,} trips, {groups.nunique():,} users")

    # Fit GEE with exchangeable correlation
    try:
        family = sm.families.Binomial()
        gee_model = sm.GEE(
            y, X, groups=groups, family=family,
            cov_struct=sm.cov_struct.Exchangeable()
        )
        gee_result = gee_model.fit(maxiter=100)
        print(gee_result.summary())

        # Extract coefficients
        coef_df = pd.DataFrame({
            "coef": gee_result.params,
            "se": gee_result.bse,
            "z": gee_result.tvalues,
            "p": gee_result.pvalues,
        })
        coef_df["OR"] = np.exp(coef_df["coef"])
        coef_df["OR_lower"] = np.exp(coef_df["coef"] - 1.96 * coef_df["se"])
        coef_df["OR_upper"] = np.exp(coef_df["coef"] + 1.96 * coef_df["se"])

        results["gee_linear"] = {
            name: {
                "coef": round(row["coef"], 4),
                "se": round(row["se"], 4),
                "OR": round(row["OR"], 4),
                "OR_CI": [round(row["OR_lower"], 4), round(row["OR_upper"], 4)],
                "p": round(row["p"], 6),
            }
            for name, row in coef_df.iterrows()
        }

        # Key result
        lr_coef = coef_df.loc["log_trip_rank"]
        print(f"\n  Key: log(trip_rank) OR = {lr_coef['OR']:.4f} "
              f"[{lr_coef['OR_lower']:.4f}, {lr_coef['OR_upper']:.4f}], "
              f"p = {lr_coef['p']:.2e}")
        if lr_coef["OR"] > 1:
            print("  -> More experience INCREASES speeding")
        else:
            print("  -> More experience DECREASES speeding")

    except Exception as e:
        print(f"  GEE failed: {e}")
        results["gee_linear"] = {"error": str(e)}

    # Test quadratic: log(trip_rank) + log(trip_rank)^2
    print("\n  Testing quadratic experience effect...")
    try:
        X_quad = X.copy()
        X_quad["log_trip_rank_sq"] = df_gee.loc[mask, "log_trip_rank_sq"].astype(float).values

        gee_quad = sm.GEE(
            y, X_quad, groups=groups, family=family,
            cov_struct=sm.cov_struct.Exchangeable()
        )
        gee_quad_result = gee_quad.fit(maxiter=100)

        sq_coef = gee_quad_result.params["log_trip_rank_sq"]
        sq_p = gee_quad_result.pvalues["log_trip_rank_sq"]
        print(f"  Quadratic term: coef={sq_coef:.4f}, p={sq_p:.2e}")
        if sq_p < 0.05:
            print("  -> Non-monotonic (U-shaped or inverted-U) experience effect detected")
        else:
            print("  -> No significant quadratic effect")

        results["gee_quadratic"] = {
            "log_trip_rank_sq_coef": round(sq_coef, 4),
            "log_trip_rank_sq_p": round(sq_p, 6),
            "significant": bool(sq_p < 0.05),
        }

    except Exception as e:
        print(f"  Quadratic GEE failed: {e}")
        results["gee_quadratic"] = {"error": str(e)}

    return results


def main() -> None:
    """Run all experience-speeding analyses."""
    t_start = time.time()

    print("=" * 70)
    print("  TASK 1.1: HABITUAL DRIVER SPEEDING ANALYSIS")
    print("=" * 70)

    df = load_data()

    all_results = {}
    all_results.update(analysis_1_usage_category(df))
    all_results.update(analysis_2_learning_curve(df))
    all_results.update(analysis_3_logistic_gee(df))

    # Save results
    all_results["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print(f"  EXPERIENCE ANALYSIS COMPLETE ({all_results['runtime_seconds']:.0f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
