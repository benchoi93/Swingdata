"""
Task 5.8: Natural experiment — SCOOTER_TUB vs SCOOTER_ECO mode comparison.

Analyzes the speed governor effect by comparing the same users' behavior
under TUB (standard/turbo) mode vs ECO (speed-limited) mode. This serves
as a natural experiment since users self-select into modes but ride the
same scooter hardware.

Key comparisons:
  1. Aggregate: TUB vs ECO overall speed distributions
  2. Within-subject: Same users' behavior across modes (paired analysis)
  3. Also compare with STD (standard) mode
  4. Statistical testing (paired t-test, Wilcoxon signed-rank)

Outputs:
  - data_parquet/modeling/mode_comparison.json — comparison statistics
  - figures/mode_speed_comparison.png — speed distribution by mode
  - figures/mode_paired_analysis.png — paired within-subject analysis
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
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, RANDOM_SEED, FIG_DPI

MODELING_DIR = DATA_DIR / "modeling"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Modes to compare (scooter modes only, excluding bike modes)
SCOOTER_MODES = ["SCOOTER_TUB", "SCOOTER_STD", "SCOOTER_ECO"]
MODE_LABELS = {
    "SCOOTER_TUB": "Turbo (TUB)",
    "SCOOTER_STD": "Standard (STD)",
    "SCOOTER_ECO": "Eco (ECO)",
}
MODE_COLORS = {
    "SCOOTER_TUB": "#e74c3c",
    "SCOOTER_STD": "#3498db",
    "SCOOTER_ECO": "#2ecc71",
}


def load_trip_data() -> pd.DataFrame:
    """Load trip modeling data filtered to scooter modes only.

    Returns:
        DataFrame with scooter-mode trips.
    """
    trip_path = MODELING_DIR / "trip_modeling.parquet"
    df = pd.read_parquet(trip_path)
    df_scooter = df[df["mode"].isin(SCOOTER_MODES)].copy()
    print(f"Scooter-mode trips: {len(df_scooter):,} (out of {len(df):,} total)")
    return df_scooter


def aggregate_comparison(df: pd.DataFrame) -> dict:
    """Compare speed indicators across modes at aggregate level.

    Args:
        df: Trip DataFrame with scooter modes.

    Returns:
        Dict with aggregate comparison statistics.
    """
    print("\n--- Aggregate Mode Comparison ---")
    results = {}

    for mode in SCOOTER_MODES:
        mode_df = df[df["mode"] == mode]
        label = MODE_LABELS[mode]
        n = len(mode_df)

        stats_dict = {
            "n_trips": n,
            "n_users": mode_df["user_id"].nunique(),
            "mean_speed": float(mode_df["mean_speed"].mean()),
            "median_speed": float(mode_df["mean_speed"].median()),
            "std_speed": float(mode_df["mean_speed"].std()),
            "mean_max_speed": float(mode_df["max_speed_from_profile"].mean()),
            "mean_p85_speed": float(mode_df["p85_speed"].mean()),
            "speeding_rate": float((mode_df["speeding_rate_25"] > 0).mean()),
            "mean_speeding_fraction": float(mode_df["speeding_rate_25"].mean()),
            "mean_speed_cv": float(mode_df["speed_cv"].mean()),
            "mean_abs_accel": float(mode_df["mean_abs_accel_ms2"].mean()),
            "harsh_event_rate": float((mode_df["harsh_event_count"] > 0).mean()),
        }
        results[mode] = stats_dict

        print(f"\n  {label} ({n:,} trips, {stats_dict['n_users']:,} users):")
        print(f"    Mean speed:      {stats_dict['mean_speed']:.1f} km/h")
        print(f"    Mean max speed:  {stats_dict['mean_max_speed']:.1f} km/h")
        print(f"    P85 speed:       {stats_dict['mean_p85_speed']:.1f} km/h")
        print(f"    Speeding rate:   {100*stats_dict['speeding_rate']:.1f}% of trips")
        print(f"    Mean speed CV:   {stats_dict['mean_speed_cv']:.3f}")
        print(f"    Harsh events:    {100*stats_dict['harsh_event_rate']:.1f}% of trips")

    return results


def within_subject_comparison(df: pd.DataFrame) -> dict:
    """Compare same users' behavior across TUB and ECO modes (paired analysis).

    Args:
        df: Trip DataFrame.

    Returns:
        Dict with within-subject comparison results.
    """
    print("\n--- Within-Subject Paired Analysis (TUB vs ECO) ---")

    # Find users with trips in both TUB and ECO modes
    tub_users = set(df[df["mode"] == "SCOOTER_TUB"]["user_id"].unique())
    eco_users = set(df[df["mode"] == "SCOOTER_ECO"]["user_id"].unique())
    both_users = tub_users & eco_users
    print(f"  Users with both TUB and ECO trips: {len(both_users):,}")

    if len(both_users) < 100:
        print("  Too few users for paired analysis")
        return {"n_paired_users": len(both_users), "status": "insufficient_data"}

    # Compute per-user means for each mode
    paired_df = df[df["user_id"].isin(both_users)].copy()

    indicators = [
        "mean_speed", "max_speed_from_profile", "p85_speed",
        "speeding_rate_25", "speed_cv", "mean_abs_accel_ms2",
        "harsh_event_rate", "cruise_fraction", "zero_speed_fraction",
    ]

    user_mode_means = paired_df.groupby(["user_id", "mode"])[indicators].mean().reset_index()

    tub_means = user_mode_means[user_mode_means["mode"] == "SCOOTER_TUB"].set_index("user_id")
    eco_means = user_mode_means[user_mode_means["mode"] == "SCOOTER_ECO"].set_index("user_id")

    # Align on common users
    common_users = tub_means.index.intersection(eco_means.index)
    tub_aligned = tub_means.loc[common_users]
    eco_aligned = eco_means.loc[common_users]

    print(f"  Paired sample size: {len(common_users):,} users")

    # Paired statistical tests
    results = {
        "n_paired_users": len(common_users),
        "paired_tests": {},
    }

    print(f"\n  {'Indicator':<25} {'TUB Mean':>10} {'ECO Mean':>10} "
          f"{'Diff':>10} {'t-stat':>10} {'p-value':>12} {'Wilcoxon p':>12}")
    print("  " + "-" * 95)

    for ind in indicators:
        tub_vals = tub_aligned[ind].values
        eco_vals = eco_aligned[ind].values
        diff = tub_vals - eco_vals

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(tub_vals, eco_vals)

        # Wilcoxon signed-rank test (nonparametric)
        try:
            w_stat, w_pval = stats.wilcoxon(diff[diff != 0])
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

        # Effect size (Cohen's d for paired samples)
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0

        results["paired_tests"][ind] = {
            "tub_mean": float(tub_vals.mean()),
            "eco_mean": float(eco_vals.mean()),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std()),
            "t_stat": float(t_stat),
            "t_pval": float(t_pval),
            "wilcoxon_pval": float(w_pval) if not np.isnan(w_pval) else None,
            "cohens_d": float(cohens_d),
        }

        print(f"  {ind:<25} {tub_vals.mean():>10.3f} {eco_vals.mean():>10.3f} "
              f"{diff.mean():>+10.3f} {t_stat:>10.1f} {t_pval:>12.2e} "
              f"{w_pval:>12.2e}" if not np.isnan(w_pval) else
              f"  {ind:<25} {tub_vals.mean():>10.3f} {eco_vals.mean():>10.3f} "
              f"{diff.mean():>+10.3f} {t_stat:>10.1f} {t_pval:>12.2e} {'N/A':>12}")

    # Also compare TUB vs STD for same users
    std_users = set(df[df["mode"] == "SCOOTER_STD"]["user_id"].unique())
    tub_std_both = tub_users & std_users
    print(f"\n  Users with both TUB and STD trips: {len(tub_std_both):,}")

    if len(tub_std_both) >= 100:
        paired_tub_std = df[df["user_id"].isin(tub_std_both)].copy()
        tub_std_means = paired_tub_std.groupby(["user_id", "mode"])[indicators].mean().reset_index()
        tub_m = tub_std_means[tub_std_means["mode"] == "SCOOTER_TUB"].set_index("user_id")
        std_m = tub_std_means[tub_std_means["mode"] == "SCOOTER_STD"].set_index("user_id")
        common = tub_m.index.intersection(std_m.index)

        print(f"\n  TUB vs STD (n={len(common):,}):")
        results["tub_vs_std"] = {"n_paired_users": len(common), "tests": {}}
        for ind in ["mean_speed", "max_speed_from_profile", "speeding_rate_25"]:
            diff = tub_m.loc[common, ind].values - std_m.loc[common, ind].values
            t_stat, t_pval = stats.ttest_rel(tub_m.loc[common, ind].values, std_m.loc[common, ind].values)
            cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
            print(f"    {ind}: TUB={tub_m.loc[common, ind].mean():.3f} "
                  f"STD={std_m.loc[common, ind].mean():.3f} "
                  f"diff={diff.mean():+.3f} t={t_stat:.1f} p={t_pval:.2e} d={cohens_d:.3f}")
            results["tub_vs_std"]["tests"][ind] = {
                "tub_mean": float(tub_m.loc[common, ind].mean()),
                "std_mean": float(std_m.loc[common, ind].mean()),
                "mean_diff": float(diff.mean()),
                "t_stat": float(t_stat),
                "t_pval": float(t_pval),
                "cohens_d": float(cohens_d),
            }

    return results


def plot_speed_distributions(df: pd.DataFrame, output_path: Path) -> None:
    """Plot speed distributions by mode.

    Args:
        df: Trip DataFrame.
        output_path: Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    speed_vars = [
        ("mean_speed", "Mean Trip Speed (km/h)"),
        ("max_speed_from_profile", "Max Trip Speed (km/h)"),
        ("p85_speed", "P85 Trip Speed (km/h)"),
        ("speeding_rate_25", "Fraction of Time > 25 km/h"),
    ]

    for ax, (var, title) in zip(axes.flat, speed_vars):
        for mode in SCOOTER_MODES:
            mode_data = df[df["mode"] == mode][var].dropna()
            # Subsample for plotting efficiency
            if len(mode_data) > 50000:
                mode_data = mode_data.sample(50000, random_state=RANDOM_SEED)

            label = MODE_LABELS[mode]
            color = MODE_COLORS[mode]

            if var == "speeding_rate_25":
                # For speeding rate, use histogram
                bins = np.linspace(0, 1, 50)
                ax.hist(mode_data, bins=bins, alpha=0.4, color=color,
                        label=label, density=True)
            else:
                # KDE for speed distributions
                from scipy.stats import gaussian_kde
                if len(mode_data) > 100:
                    x_range = np.linspace(mode_data.quantile(0.01),
                                         mode_data.quantile(0.99), 200)
                    kde = gaussian_kde(mode_data.values)
                    ax.plot(x_range, kde(x_range), linewidth=2, color=color,
                            label=label)
                    ax.fill_between(x_range, kde(x_range), alpha=0.15, color=color)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add speed limit line where appropriate
        if var in ("mean_speed", "max_speed_from_profile", "p85_speed"):
            ax.axvline(25, color="red", linestyle="--", alpha=0.5, label="25 km/h limit")

    plt.suptitle("Speed Distributions by Operating Mode", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved speed distribution plot to {output_path}")


def plot_paired_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """Plot paired within-subject analysis (TUB vs ECO for same users).

    Args:
        df: Trip DataFrame.
        output_path: Path to save figure.
    """
    # Get users with both modes
    tub_users = set(df[df["mode"] == "SCOOTER_TUB"]["user_id"].unique())
    eco_users = set(df[df["mode"] == "SCOOTER_ECO"]["user_id"].unique())
    both_users = tub_users & eco_users

    if len(both_users) < 100:
        print("Too few paired users for plot")
        return

    paired_df = df[df["user_id"].isin(both_users)].copy()
    indicators = ["mean_speed", "max_speed_from_profile", "speeding_rate_25"]
    user_mode = paired_df.groupby(["user_id", "mode"])[indicators].mean().reset_index()

    tub = user_mode[user_mode["mode"] == "SCOOTER_TUB"].set_index("user_id")
    eco = user_mode[user_mode["mode"] == "SCOOTER_ECO"].set_index("user_id")
    common = tub.index.intersection(eco.index)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = [
        ("mean_speed", "Mean Speed (km/h)"),
        ("max_speed_from_profile", "Max Speed (km/h)"),
        ("speeding_rate_25", "Speeding Rate (>25 km/h)"),
    ]

    for ax, (var, title) in zip(axes, titles):
        tub_vals = tub.loc[common, var].values
        eco_vals = eco.loc[common, var].values

        # Subsample for scatter plot
        n_plot = min(5000, len(common))
        idx = np.random.RandomState(RANDOM_SEED).choice(len(common), n_plot, replace=False)

        ax.scatter(eco_vals[idx], tub_vals[idx], alpha=0.15, s=10, color="#3498db")

        # Reference line (y=x)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)

        ax.set_xlabel(f"ECO Mode: {title}", fontsize=10)
        ax.set_ylabel(f"TUB Mode: {title}", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Add mean difference annotation
        diff = tub_vals.mean() - eco_vals.mean()
        ax.annotate(
            f"Mean diff: {diff:+.2f}\n(TUB - ECO)",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Within-Subject Comparison: TUB vs ECO Mode (n={len(common):,} users)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved paired analysis plot to {output_path}")


def main() -> None:
    """Run TUB vs ECO mode natural experiment analysis."""
    print("=" * 60)
    print("Task 5.8: Mode Comparison (TUB vs ECO Natural Experiment)")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # Load data
    print("\nLoading scooter-mode trip data...")
    df = load_trip_data()

    # Aggregate comparison
    agg_results = aggregate_comparison(df)

    # Within-subject paired analysis
    paired_results = within_subject_comparison(df)

    # Plots
    print("\nGenerating plots...")
    plot_speed_distributions(df, FIGURES_DIR / "mode_speed_comparison.png")
    plot_paired_analysis(df, FIGURES_DIR / "mode_paired_analysis.png")

    # Save all results
    all_results = {
        "aggregate_comparison": agg_results,
        "within_subject_paired": paired_results,
    }

    results_path = MODELING_DIR / "mode_comparison.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved results to {results_path}")

    # Key findings summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    tub = agg_results["SCOOTER_TUB"]
    eco = agg_results["SCOOTER_ECO"]
    print(f"\n  Speed governor effect (ECO vs TUB):")
    print(f"    Mean speed reduction:     {tub['mean_speed'] - eco['mean_speed']:.1f} km/h "
          f"({100*(tub['mean_speed'] - eco['mean_speed'])/tub['mean_speed']:.1f}%)")
    print(f"    Max speed reduction:      {tub['mean_max_speed'] - eco['mean_max_speed']:.1f} km/h "
          f"({100*(tub['mean_max_speed'] - eco['mean_max_speed'])/tub['mean_max_speed']:.1f}%)")
    print(f"    Speeding rate reduction:  {tub['speeding_rate'] - eco['speeding_rate']:.3f} "
          f"({100*tub['speeding_rate']:.1f}% -> {100*eco['speeding_rate']:.1f}%)")

    if "paired_tests" in paired_results:
        pt = paired_results["paired_tests"]
        print(f"\n  Within-subject (n={paired_results['n_paired_users']:,} users):")
        for ind in ["mean_speed", "max_speed_from_profile", "speeding_rate_25"]:
            if ind in pt:
                t = pt[ind]
                print(f"    {ind}: d={t['cohens_d']:.3f}, p={t['t_pval']:.2e}")


if __name__ == "__main__":
    main()
