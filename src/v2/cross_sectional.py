"""
Phase 1: Cross-Sectional Behavioral Profiles (Tasks 1.1-1.3).

Task 1.1: Multi-dimensional mode comparison (TUB vs STD vs ECO) on all 8
          primary outcomes + manipulation check. Radar chart (Fig 1).
Task 1.2: GEE for each primary outcome ~ mode + controls, clustered by user.
          Multi-panel forest plot (Fig 2).
Task 1.3: Within-subject paired comparison for multi-mode users.
          Cohen's d for each outcome.

All analyses use data_parquet/v2/trip_modeling.parquet (Feb-Nov 2023).
"""

import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, FIG_DPI

V2_DIR = DATA_DIR / "v2"
FIG_DIR = FIGURES_DIR / "v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRIP_PATH = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

# 8 primary outcomes + manipulation check
PRIMARY_OUTCOMES = {
    "harsh_accel_count": "Harsh accel\n(count/trip)",
    "harsh_decel_count": "Harsh decel\n(count/trip)",
    "speed_cv": "Speed CV",
    "distance": "Distance (m)",
    "travel_time": "Travel time (s)",
    "cruise_fraction": "Cruise fraction",
    "zero_speed_fraction": "Zero-speed\nfraction",
    "harsh_event_count": "Harsh events\n(total/trip)",
}

MANIPULATION_CHECK = {
    "mean_speed": "Mean speed\n(km/h)",
    "p85_speed": "P85 speed\n(km/h)",
    "max_speed_from_profile": "Max speed\n(km/h)",
}

MODE_COLORS = {"TUB": "#e74c3c", "STD": "#3498db", "ECO": "#2ecc71"}
MODE_ORDER = ["TUB", "STD", "ECO"]


# ─────────────────────────────────────────────────────────────────────
# Task 1.1: Multi-dimensional mode comparison
# ─────────────────────────────────────────────────────────────────────

def task_1_1(con: duckdb.DuckDBPyConnection) -> dict:
    """Compute mode-level means for all outcomes, create radar chart."""
    print("=" * 60)
    print("Task 1.1: Multi-dimensional Mode Comparison")
    print("=" * 60)

    all_outcomes = {**PRIMARY_OUTCOMES, **MANIPULATION_CHECK}

    # Compute means and SEs by mode
    agg_cols = ", ".join(
        f"ROUND(AVG({col}), 4) AS {col}_mean, "
        f"ROUND(STDDEV({col}) / SQRT(COUNT(*)), 6) AS {col}_se"
        for col in all_outcomes
    )

    stats_df = con.execute(f"""
        SELECT mode, COUNT(*) AS n, {agg_cols}
        FROM read_parquet('{TRIP_PATH}')
        GROUP BY mode ORDER BY mode
    """).fetchdf()

    # Print table
    print("\n  Mode-level means:")
    print(f"  {'Outcome':<25s} {'TUB':>10s} {'STD':>10s} {'ECO':>10s}")
    print("  " + "-" * 60)
    for col, label in all_outcomes.items():
        vals = {m: stats_df.loc[stats_df["mode"] == m, f"{col}_mean"].values[0]
                for m in MODE_ORDER}
        clean_label = label.replace("\n", " ")
        print(f"  {clean_label:<25s} {vals['TUB']:>10.3f} {vals['STD']:>10.3f} {vals['ECO']:>10.3f}")

    # Radar chart for primary outcomes (normalized)
    _make_radar_chart(stats_df, PRIMARY_OUTCOMES)

    result = {}
    for col in all_outcomes:
        result[col] = {m: float(stats_df.loc[stats_df["mode"] == m, f"{col}_mean"].values[0])
                       for m in MODE_ORDER}
    return result


def _make_radar_chart(stats_df: pd.DataFrame, outcomes: dict):
    """Create radar chart comparing modes across normalized outcome dimensions."""
    labels = list(outcomes.values())
    cols = list(outcomes.keys())
    n = len(cols)

    # Get values per mode and normalize to [0, 1]
    raw = {}
    for mode in MODE_ORDER:
        raw[mode] = [float(stats_df.loc[stats_df["mode"] == mode, f"{c}_mean"].values[0])
                     for c in cols]

    # Min-max normalize across modes for each outcome
    all_vals = np.array([raw[m] for m in MODE_ORDER])
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid div by zero

    normalized = {m: (np.array(raw[m]) - mins) / ranges for m in MODE_ORDER}

    # Plot
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for mode in MODE_ORDER:
        vals = normalized[mode].tolist() + [normalized[mode][0]]
        ax.plot(angles, vals, "o-", linewidth=2, color=MODE_COLORS[mode], label=mode)
        ax.fill(angles, vals, alpha=0.1, color=MODE_COLORS[mode])

    clean_labels = [l.replace("\n", " ") for l in labels]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(clean_labels, size=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7, alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=11)
    ax.set_title("Behavioral Profile by Operating Mode\n(normalized)", size=13, pad=20)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig1_radar_behavioral_profile.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Fig 1 saved: {FIG_DIR / 'fig1_radar_behavioral_profile.pdf'}")


# ─────────────────────────────────────────────────────────────────────
# Task 1.2: GEE for each primary outcome
# ─────────────────────────────────────────────────────────────────────

def task_1_2(con: duckdb.DuckDBPyConnection) -> dict:
    """GEE regression for each outcome ~ mode + controls, clustered by user.

    Uses a 200K subsample for computational feasibility.
    """
    print("\n" + "=" * 60)
    print("Task 1.2: GEE Models for Primary Outcomes")
    print("=" * 60)

    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Gaussian, Poisson
    from statsmodels.genmod.cov_struct import Exchangeable

    # Sample 200K trips for GEE (full 19M too slow)
    print("  Sampling 200K trips...")
    sample_df = con.execute(f"""
        SELECT
            user_id, mode, age_group, time_of_day, day_type,
            is_major_road, distance,
            harsh_accel_count, harsh_decel_count, harsh_event_count,
            speed_cv, cruise_fraction, zero_speed_fraction,
            travel_time, mean_speed
        FROM read_parquet('{TRIP_PATH}')
        USING SAMPLE 200000 ROWS (reservoir)
    """).fetchdf()
    print(f"  Sampled: {len(sample_df):,} trips, {sample_df['user_id'].nunique():,} users")

    # Prepare features
    sample_df["log_distance"] = np.log1p(sample_df["distance"])

    # Encode categoricals
    sample_df["mode_STD"] = (sample_df["mode"] == "STD").astype(int)
    sample_df["mode_ECO"] = (sample_df["mode"] == "ECO").astype(int)

    # Age dummies (reference: 25_29)
    for ag in ["under_20", "20_24", "30_34", "35_39", "40_49", "50_plus"]:
        sample_df[f"age_{ag}"] = (sample_df["age_group"] == ag).astype(int)

    # Time dummies (reference: midday)
    for tod in ["morning_rush", "afternoon_rush", "evening", "night"]:
        sample_df[f"tod_{tod}"] = (sample_df["time_of_day"] == tod).astype(int)

    sample_df["is_weekend"] = (sample_df["day_type"] == "weekend").astype(int)

    # Sort by user for GEE
    sample_df = sample_df.sort_values("user_id").reset_index(drop=True)

    # Feature columns
    x_cols = (
        ["mode_STD", "mode_ECO"]
        + [f"age_{ag}" for ag in ["under_20", "20_24", "30_34", "35_39", "40_49", "50_plus"]]
        + [f"tod_{tod}" for tod in ["morning_rush", "afternoon_rush", "evening", "night"]]
        + ["is_weekend", "is_major_road", "log_distance"]
    )

    outcomes_to_model = {
        # Primary outcomes
        "harsh_accel_count": ("Harsh accel", "poisson"),
        "harsh_decel_count": ("Harsh decel", "poisson"),
        "speed_cv": ("Speed CV", "gaussian"),
        "cruise_fraction": ("Cruise fraction", "gaussian"),
        "zero_speed_fraction": ("Zero-speed frac", "gaussian"),
        # Manipulation check
        "mean_speed": ("Mean speed", "gaussian"),
    }

    results = {}
    coef_data = []  # for forest plot

    for outcome, (label, family_type) in outcomes_to_model.items():
        print(f"\n  GEE: {label} ({outcome})...")
        t0 = time.time()

        y = sample_df[outcome].values
        X = sm.add_constant(sample_df[x_cols].values.astype(float))

        family = Poisson() if family_type == "poisson" else Gaussian()

        try:
            gee = GEE(
                y, X, groups=sample_df["user_id"],
                family=family,
                cov_struct=Exchangeable(),
            )
            fit = gee.fit(maxiter=100)

            # Extract mode coefficients (STD and ECO, ref=TUB)
            col_names = ["const"] + x_cols
            mode_std_idx = col_names.index("mode_STD")
            mode_eco_idx = col_names.index("mode_ECO")

            for mode_name, idx in [("STD vs TUB", mode_std_idx), ("ECO vs TUB", mode_eco_idx)]:
                coef = fit.params[idx]
                se = fit.bse[idx]
                pval = fit.pvalues[idx]
                ci_lo = coef - 1.96 * se
                ci_hi = coef + 1.96 * se

                results[f"{outcome}_{mode_name}"] = {
                    "coef": round(float(coef), 4),
                    "se": round(float(se), 4),
                    "p": float(pval),
                    "ci_lower": round(float(ci_lo), 4),
                    "ci_upper": round(float(ci_hi), 4),
                }

                coef_data.append({
                    "outcome": label,
                    "comparison": mode_name,
                    "coef": coef,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p": pval,
                    "family": family_type,
                })

                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"    {mode_name}: {coef:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] p={pval:.4g} {sig}")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        print(f"    ({time.time() - t0:.1f}s)")

    # Forest plot
    _make_forest_plot(coef_data)

    return results


def _make_forest_plot(coef_data: list):
    """Create multi-panel forest plot of GEE mode coefficients."""
    df = pd.DataFrame(coef_data)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    for ax_idx, comparison in enumerate(["STD vs TUB", "ECO vs TUB"]):
        ax = axes[ax_idx]
        sub = df[df["comparison"] == comparison].copy()
        sub = sub.iloc[::-1]  # reverse for bottom-up

        y_pos = range(len(sub))
        colors = ["#e74c3c" if p < 0.001 else "#f39c12" if p < 0.05 else "#95a5a6"
                  for p in sub["p"]]

        ax.barh(list(y_pos), sub["coef"], xerr=[sub["coef"] - sub["ci_lo"], sub["ci_hi"] - sub["coef"]],
                color=colors, alpha=0.7, capsize=3, height=0.6)
        ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sub["outcome"])
        ax.set_xlabel("Coefficient (vs TUB reference)")
        ax.set_title(comparison, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("GEE Mode Coefficients for Behavioral Outcomes", fontsize=14, y=1.02)
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig2_forest_gee_coefficients.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Fig 2 saved: {FIG_DIR / 'fig2_forest_gee_coefficients.pdf'}")


# ─────────────────────────────────────────────────────────────────────
# Task 1.3: Within-subject paired comparison
# ─────────────────────────────────────────────────────────────────────

def task_1_3(con: duckdb.DuckDBPyConnection) -> dict:
    """Within-subject paired tests for users who used multiple modes."""
    print("\n" + "=" * 60)
    print("Task 1.3: Within-Subject Paired Comparison")
    print("=" * 60)

    outcomes = list(PRIMARY_OUTCOMES.keys()) + ["mean_speed"]
    outcome_cols = ", ".join(f"AVG({c}) AS {c}" for c in outcomes)

    # Find users who used both TUB and STD (most common pair)
    print("  Finding multi-mode users (TUB & STD)...")
    paired_df = con.execute(f"""
        WITH user_mode_means AS (
            SELECT user_id, mode, COUNT(*) AS n_trips, {outcome_cols}
            FROM read_parquet('{TRIP_PATH}')
            WHERE mode IN ('TUB', 'STD')
            GROUP BY user_id, mode
            HAVING COUNT(*) >= 5
        ),
        tub_users AS (SELECT user_id FROM user_mode_means WHERE mode = 'TUB'),
        std_users AS (SELECT user_id FROM user_mode_means WHERE mode = 'STD'),
        both_users AS (
            SELECT user_id FROM tub_users INTERSECT SELECT user_id FROM std_users
        )
        SELECT um.*
        FROM user_mode_means um
        JOIN both_users bu ON um.user_id = bu.user_id
        ORDER BY um.user_id, um.mode
    """).fetchdf()

    n_paired = paired_df["user_id"].nunique()
    print(f"  Paired users (TUB & STD, >=5 trips each): {n_paired:,}")

    # Also find TUB & ECO pairs
    paired_eco = con.execute(f"""
        WITH user_mode_means AS (
            SELECT user_id, mode, COUNT(*) AS n_trips, {outcome_cols}
            FROM read_parquet('{TRIP_PATH}')
            WHERE mode IN ('TUB', 'ECO')
            GROUP BY user_id, mode
            HAVING COUNT(*) >= 5
        ),
        tub_users AS (SELECT user_id FROM user_mode_means WHERE mode = 'TUB'),
        eco_users AS (SELECT user_id FROM user_mode_means WHERE mode = 'ECO'),
        both_users AS (
            SELECT user_id FROM tub_users INTERSECT SELECT user_id FROM eco_users
        )
        SELECT um.*
        FROM user_mode_means um
        JOIN both_users bu ON um.user_id = bu.user_id
        ORDER BY um.user_id, um.mode
    """).fetchdf()

    n_paired_eco = paired_eco["user_id"].nunique()
    print(f"  Paired users (TUB & ECO, >=5 trips each): {n_paired_eco:,}")

    results = {}

    for pair_name, pdf, mode_a, mode_b in [
        ("TUB_vs_STD", paired_df, "TUB", "STD"),
        ("TUB_vs_ECO", paired_eco, "TUB", "ECO"),
    ]:
        print(f"\n  --- {pair_name} ---")
        pair_results = {}

        a_df = pdf[pdf["mode"] == mode_a].set_index("user_id")
        b_df = pdf[pdf["mode"] == mode_b].set_index("user_id")
        common = a_df.index.intersection(b_df.index)

        for outcome in outcomes:
            a_vals = a_df.loc[common, outcome].values.astype(float)
            b_vals = b_df.loc[common, outcome].values.astype(float)
            diff = a_vals - b_vals

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(a_vals, b_vals)

            # Cohen's d (paired)
            d = diff.mean() / diff.std() if diff.std() > 0 else 0.0

            pair_results[outcome] = {
                "n_pairs": len(common),
                "mean_a": round(float(a_vals.mean()), 4),
                "mean_b": round(float(b_vals.mean()), 4),
                "mean_diff": round(float(diff.mean()), 4),
                "cohens_d": round(float(d), 4),
                "t_stat": round(float(t_stat), 2),
                "p_value": float(p_val),
            }

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            label = PRIMARY_OUTCOMES.get(outcome, MANIPULATION_CHECK.get(outcome, outcome))
            clean_label = label.replace("\n", " ")
            print(f"    {clean_label:<25s} d={d:>7.3f}  {mode_a}={a_vals.mean():.3f} {mode_b}={b_vals.mean():.3f}  {sig}")

        results[pair_name] = pair_results

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("Phase 1: Cross-Sectional Behavioral Profiles")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    # Task 1.1
    mode_comparison = task_1_1(con)

    # Task 1.2
    gee_results = task_1_2(con)

    # Task 1.3
    paired_results = task_1_3(con)

    elapsed = time.time() - t_start

    # Save all results
    report = {
        "phase": 1,
        "tasks": ["1.1", "1.2", "1.3"],
        "mode_comparison": mode_comparison,
        "gee_results": gee_results,
        "paired_comparisons": paired_results,
        "time_s": round(elapsed, 1),
    }

    report_path = V2_DIR / "phase1_cross_sectional.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Phase 1 complete in {elapsed:.0f}s")
    print(f"  Report: {report_path}")
    print(f"  Figures: {FIG_DIR}/fig1_radar*, {FIG_DIR}/fig2_forest*")

    con.close()


if __name__ == "__main__":
    main()
