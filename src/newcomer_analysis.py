"""
Task 8.1.3: Newcomer vs established rider analysis.

Compares first-month riders with established riders to understand:
  - Do newcomers speed more or less than experienced riders?
  - How does speed behavior evolve over a rider's first N trips?
  - Are there adoption patterns in mode choice (STD/TUB/ECO)?

Uses trip_experience.parquet from experience_data_prep.py (Task 8.1.1).

Outputs:
  - figures/fig_newcomer_speed_trajectory.pdf
  - figures/fig_newcomer_vs_established.pdf
  - figures/fig_newcomer_mode_adoption.pdf
  - data_parquet/modeling/newcomer_results.json
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
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, MODELING_DIR, FIG_DPI, RANDOM_SEED

TRIP_EXPERIENCE = MODELING_DIR / "trip_experience.parquet"
RESULTS_PATH = MODELING_DIR / "newcomer_results.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELING_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "newcomer": "#D55E00",
    "established": "#0072B2",
    "STD": "#0072B2",
    "TUB": "#D55E00",
    "ECO": "#009E73",
    "none": "#CC79A7",
}

MAX_EARLY_TRIPS = 50


def load_data() -> pd.DataFrame:
    """Load trip experience data, filtering to trips with speed data."""
    con = duckdb.connect()
    path = str(TRIP_EXPERIENCE).replace("\\", "/")
    df = con.execute(f"""
        SELECT route_id, user_id, mode, start_date, trip_rank,
               days_since_first_trip, months_since_first_trip,
               experience_bin, usage_category, user_total_trips,
               is_speeding, mean_speed_from_speeds, max_speed_from_speeds,
               age, month_year
        FROM read_parquet('{path}')
        WHERE mean_speed_from_speeds IS NOT NULL
          AND mode IN ('STD', 'TUB', 'ECO', 'none')
    """).fetchdf()
    con.close()
    print(f"Loaded {len(df):,} trips with speed data")
    return df


def classify_rider_status(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each trip as newcomer (first 30 days) or established (31+ days)."""
    df["rider_status"] = np.where(
        df["days_since_first_trip"] <= 30, "newcomer", "established"
    )
    n_new = (df["rider_status"] == "newcomer").sum()
    n_est = (df["rider_status"] == "established").sum()
    print(f"  Newcomer trips: {n_new:,} ({n_new/len(df)*100:.1f}%)")
    print(f"  Established trips: {n_est:,} ({n_est/len(df)*100:.1f}%)")
    return df


def analysis_1_newcomer_vs_established(df: pd.DataFrame) -> dict:
    """Compare speeding rates between newcomers and established riders."""
    print("\n--- Analysis 1: Newcomer vs Established Speeding ---")

    results = {}
    for status in ["newcomer", "established"]:
        subset = df[df["rider_status"] == status]
        rate = subset["is_speeding"].mean()
        n_trips = len(subset)
        n_users = subset["user_id"].nunique()

        mean_spd = subset["mean_speed_from_speeds"].dropna().mean()
        max_spd = subset["max_speed_from_speeds"].dropna().mean()

        results[status] = {
            "n_trips": int(n_trips),
            "n_users": int(n_users),
            "speeding_rate": round(float(rate), 4),
            "mean_speed": round(float(mean_spd), 2),
            "mean_max_speed": round(float(max_spd), 2),
        }
        print(f"  {status}: {n_trips:,} trips, {n_users:,} users, "
              f"speeding={rate:.3f}, mean_speed={mean_spd:.1f}, "
              f"mean_max_speed={max_spd:.1f}")

    # Statistical test
    new_spd = df.loc[df["rider_status"] == "newcomer", "is_speeding"]
    est_spd = df.loc[df["rider_status"] == "established", "is_speeding"]

    # Mann-Whitney U test (large N, so use normal approximation)
    u_stat, u_p = scipy_stats.mannwhitneyu(
        new_spd.astype(int), est_spd.astype(int), alternative="two-sided"
    )
    results["mann_whitney_U"] = float(u_stat)
    results["mann_whitney_p"] = float(u_p)
    print(f"  Mann-Whitney U: {u_stat:.0f}, p={u_p:.2e}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (new_spd.var() * (len(new_spd) - 1) + est_spd.var() * (len(est_spd) - 1))
        / (len(new_spd) + len(est_spd) - 2)
    )
    cohens_d = (est_spd.mean() - new_spd.mean()) / pooled_std if pooled_std > 0 else 0
    results["cohens_d"] = round(float(cohens_d), 4)
    print(f"  Cohen's d: {cohens_d:.4f}")

    return results


def analysis_2_early_trip_trajectory(df: pd.DataFrame) -> dict:
    """Track speeding rate evolution over a rider's first N trips."""
    print(f"\n--- Analysis 2: Speed Trajectory (First {MAX_EARLY_TRIPS} Trips) ---")

    early = df[df["trip_rank"] <= MAX_EARLY_TRIPS].copy()
    trajectory = (
        early.groupby("trip_rank")
        .agg(
            speeding_rate=("is_speeding", "mean"),
            mean_speed=("mean_speed_from_speeds", "mean"),
            mean_max_speed=("max_speed_from_speeds", "mean"),
            n_trips=("route_id", "count"),
        )
        .reset_index()
    )

    results = {
        "trajectory": trajectory.to_dict(orient="records"),
        "trip_1_speeding": round(float(trajectory.iloc[0]["speeding_rate"]), 4),
        "trip_50_speeding": round(float(trajectory.iloc[-1]["speeding_rate"]), 4),
        "speeding_increase": round(
            float(trajectory.iloc[-1]["speeding_rate"] - trajectory.iloc[0]["speeding_rate"]),
            4,
        ),
    }
    print(f"  Trip 1 speeding: {results['trip_1_speeding']:.3f}")
    print(f"  Trip {MAX_EARLY_TRIPS} speeding: {results['trip_50_speeding']:.3f}")
    print(f"  Increase: {results['speeding_increase']:.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(trajectory["trip_rank"], trajectory["speeding_rate"] * 100,
             color="#333333", linewidth=2)
    ax1.fill_between(trajectory["trip_rank"], 0,
                     trajectory["speeding_rate"] * 100,
                     alpha=0.15, color="#333333")
    ax1.set_xlabel("Trip Number (Chronological)", fontsize=12)
    ax1.set_ylabel("Speeding Rate (%)", fontsize=12)
    ax1.set_title("Speed Violation Rate Over First 50 Trips", fontsize=13)
    ax1.set_xlim(1, MAX_EARLY_TRIPS)
    ax1.grid(True, alpha=0.3)

    ax2.plot(trajectory["trip_rank"], trajectory["mean_speed"],
             color="#0072B2", linewidth=2, label="Mean Speed")
    ax2.plot(trajectory["trip_rank"], trajectory["mean_max_speed"],
             color="#D55E00", linewidth=2, label="Mean Max Speed")
    ax2.axhline(y=25, color="red", linestyle="--", alpha=0.5, label="25 km/h limit")
    ax2.set_xlabel("Trip Number (Chronological)", fontsize=12)
    ax2.set_ylabel("Speed (km/h)", fontsize=12)
    ax2.set_title("Speed Metrics Over First 50 Trips", fontsize=13)
    ax2.set_xlim(1, MAX_EARLY_TRIPS)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_newcomer_speed_trajectory.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_newcomer_speed_trajectory.pdf/png")

    return results


def analysis_3_newcomer_by_mode(df: pd.DataFrame) -> dict:
    """Newcomer vs established speeding broken down by mode."""
    print("\n--- Analysis 3: Newcomer vs Established by Mode ---")

    modes = ["STD", "TUB", "ECO", "none"]
    results = {}

    for mode in modes:
        mode_df = df[df["mode"] == mode]
        if len(mode_df) == 0:
            continue
        for status in ["newcomer", "established"]:
            subset = mode_df[mode_df["rider_status"] == status]
            if len(subset) == 0:
                continue
            rate = subset["is_speeding"].mean()
            results[f"{mode}_{status}"] = {
                "n_trips": int(len(subset)),
                "speeding_rate": round(float(rate), 4),
            }
            print(f"  {mode:>4s} x {status:<12s}: {len(subset):>10,} trips, "
                  f"speeding={rate:.3f}")

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(modes))
    width = 0.35

    new_rates = [results.get(f"{m}_newcomer", {}).get("speeding_rate", 0) * 100
                 for m in modes]
    est_rates = [results.get(f"{m}_established", {}).get("speeding_rate", 0) * 100
                 for m in modes]

    bars1 = ax.bar(x - width / 2, new_rates, width,
                   label="Newcomer (<=30 days)", color=COLORS["newcomer"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, est_rates, width,
                   label="Established (>30 days)", color=COLORS["established"], alpha=0.85)

    ax.set_xlabel("Riding Mode", fontsize=12)
    ax.set_ylabel("Speeding Rate (%)", fontsize=12)
    ax.set_title("Newcomer vs Established Speeding Rate by Mode", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                ax.annotate(f"{height:.1f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_newcomer_vs_established.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_newcomer_vs_established.pdf/png")

    return results


def analysis_4_mode_adoption(df: pd.DataFrame) -> dict:
    """How does mode choice evolve over a rider's first trips?"""
    print(f"\n--- Analysis 4: Mode Adoption Over First {MAX_EARLY_TRIPS} Trips ---")

    early = df[df["trip_rank"] <= MAX_EARLY_TRIPS].copy()
    mode_by_rank = (
        early.groupby(["trip_rank", "mode"])
        .size()
        .unstack(fill_value=0)
    )
    mode_frac = mode_by_rank.div(mode_by_rank.sum(axis=1), axis=0)

    results = {}
    for mode in ["STD", "TUB", "ECO", "none"]:
        if mode in mode_frac.columns:
            results[f"{mode}_trip1"] = round(float(mode_frac.loc[1, mode]), 4)
            results[f"{mode}_trip{MAX_EARLY_TRIPS}"] = round(
                float(mode_frac.loc[MAX_EARLY_TRIPS, mode]), 4
            )
            print(f"  {mode}: trip 1 = {mode_frac.loc[1, mode]:.3f} "
                  f"-> trip {MAX_EARLY_TRIPS} = {mode_frac.loc[MAX_EARLY_TRIPS, mode]:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in ["STD", "TUB", "ECO", "none"]:
        if mode in mode_frac.columns:
            ax.plot(mode_frac.index, mode_frac[mode] * 100,
                    label=mode, color=COLORS[mode], linewidth=2)

    ax.set_xlabel("Trip Number (Chronological)", fontsize=12)
    ax.set_ylabel("Mode Share (%)", fontsize=12)
    ax.set_title("Riding Mode Adoption Over First 50 Trips", fontsize=13)
    ax.set_xlim(1, MAX_EARLY_TRIPS)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_newcomer_mode_adoption.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_newcomer_mode_adoption.pdf/png")

    return results


def analysis_5_age_newcomer_interaction(df: pd.DataFrame) -> dict:
    """Newcomer effect by age group."""
    print("\n--- Analysis 5: Newcomer Effect by Age Group ---")

    df_age = df[df["age"].notna() & (df["age"] > 0) & (df["age"] < 100)].copy()
    bins = [0, 20, 25, 30, 35, 40, 50, 100]
    labels = ["<20", "20-24", "25-29", "30-34", "35-39", "40-49", "50+"]
    df_age["age_group"] = pd.cut(df_age["age"], bins=bins, labels=labels, right=False)

    results = {}
    for ag in labels:
        ag_df = df_age[df_age["age_group"] == ag]
        if len(ag_df) < 100:
            continue
        for status in ["newcomer", "established"]:
            subset = ag_df[ag_df["rider_status"] == status]
            if len(subset) < 50:
                continue
            rate = subset["is_speeding"].mean()
            results[f"{ag}_{status}"] = {
                "n_trips": int(len(subset)),
                "speeding_rate": round(float(rate), 4),
            }
        new_r = results.get(f"{ag}_newcomer", {}).get("speeding_rate", 0)
        est_r = results.get(f"{ag}_established", {}).get("speeding_rate", 0)
        print(f"  {ag:>5s}: newcomer={new_r:.3f}, established={est_r:.3f}, "
              f"diff={est_r - new_r:+.3f}")

    return results


def main() -> None:
    overall_t0 = time.time()

    print("=" * 70)
    print("  TASK 8.1.3: NEWCOMER VS ESTABLISHED RIDER ANALYSIS")
    print("=" * 70)

    df = load_data()
    df = classify_rider_status(df)

    all_results = {}
    all_results["newcomer_vs_established"] = analysis_1_newcomer_vs_established(df)
    all_results["early_trip_trajectory"] = analysis_2_early_trip_trajectory(df)
    all_results["newcomer_by_mode"] = analysis_3_newcomer_by_mode(df)
    all_results["mode_adoption"] = analysis_4_mode_adoption(df)
    all_results["age_newcomer_interaction"] = analysis_5_age_newcomer_interaction(df)

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    elapsed = time.time() - overall_t0
    print("\n" + "=" * 70)
    print(f"  NEWCOMER ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
