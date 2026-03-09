"""
Curvature-speeding analysis: how does road curvature relate to speeding risk?

Analyses:
  1. Speeding rate by curvature class (straight / mixed / curvy)
  2. Speeding rate by curvature quantile with LOWESS
  3. Curvature x road class interaction (heatmap)
  4. Logistic regression: speeding ~ curvature + road class + mode + controls
  5. Risk scoring: combined speed x curvature risk index

Outputs:
  - figures/fig_curvature_speeding_rate.pdf -- bar chart by curvature class
  - figures/fig_curvature_continuous.pdf    -- LOWESS speeding by curvature index
  - figures/fig_curvature_roadclass.pdf     -- interaction heatmap
  - figures/fig_curvature_risk_scatter.pdf  -- risk scatter (speed vs curvature)
  - data_parquet/modeling/curvature_speeding_results.json

Usage:
    python src/curvature_speeding.py
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import duckdb
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR, FIGURES_DIR, MODELING_DIR,
    RANDOM_SEED, FIG_DPI, SPEED_LIMIT_KR,
)

np.random.seed(RANDOM_SEED)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELING_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = MODELING_DIR / "curvature_speeding_results.json"

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
})

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "skyblue": "#56B4E9",
    "black": "#000000",
}

# Curvature classification thresholds (from compute_curvature.py)
SHARP_TURN_THRESHOLD = 45.0
MODERATE_TURN_THRESHOLD = 15.0


def load_merged_data() -> pd.DataFrame:
    """Load trip data merged with curvature and road class features.

    Returns:
        DataFrame with curvature, road class, and speeding data.
    """
    print("Loading and merging data...")
    t0 = time.time()
    con = duckdb.connect()

    df = con.execute("""
        SELECT
            m.route_id,
            m.user_id,
            m.mode_clean,
            m.city,
            m.province,
            m.distance,
            m.mean_speed,
            m.max_speed_from_profile,
            m.speeding_rate_25,
            m.is_speeding,
            m.age,
            m.age_group,
            m.start_hour,
            m.time_of_day,
            m.speed_cv,
            r.dominant_road_class,
            r.frac_major_road,
            r.frac_cycling_infra,
            c.mean_abs_turning_angle,
            c.median_abs_turning_angle,
            c.max_abs_turning_angle,
            c.curvature_index,
            c.frac_sharp_turns,
            c.frac_moderate_turns,
            c.frac_straight,
            c.n_segments
        FROM read_parquet('{trip_mod}') m
        JOIN read_parquet('{road}') r ON m.route_id = r.route_id
        JOIN read_parquet('{curv}') c ON m.route_id = c.route_id
        WHERE m.mode IN ('SCOOTER_TUB', 'SCOOTER_STD', 'SCOOTER_ECO')
    """.format(
        trip_mod=str(MODELING_DIR / "trip_modeling.parquet").replace("\\", "/"),
        road=str(DATA_DIR / "trip_road_classes.parquet").replace("\\", "/"),
        curv=str(DATA_DIR / "trip_curvature.parquet").replace("\\", "/"),
    )).fetchdf()
    con.close()

    # Derived columns
    df["is_speeding_int"] = df["is_speeding"].astype(int)
    df["log_distance"] = np.log(df["distance"] + 1)
    df["log_curvature"] = np.log(df["curvature_index"] + 0.001)

    # Curvature class
    df["curvature_class"] = np.select(
        [
            df["frac_straight"] > 0.8,
            df["frac_sharp_turns"] > 0.2,
        ],
        ["straight", "curvy"],
        default="mixed",
    )

    # Curvature quintiles
    df["curvature_quintile"] = pd.qcut(
        df["curvature_index"], 5, labels=False, duplicates="drop"
    )

    elapsed = time.time() - t0
    print(f"  Loaded {len(df):,} trips in {elapsed:.1f}s")
    print(f"  Curvature classes: {df['curvature_class'].value_counts().to_dict()}")

    return df


def analyze_speeding_by_curvature_class(df: pd.DataFrame) -> Dict[str, Any]:
    """Bar chart of speeding rate by curvature class.

    Args:
        df: Merged data.

    Returns:
        Results dictionary.
    """
    print("\n=== Analysis 1: Speeding by curvature class ===")

    class_order = ["straight", "mixed", "curvy"]
    stats_list = []

    for cls in class_order:
        sub = df[df["curvature_class"] == cls]
        n = len(sub)
        n_speed = sub["is_speeding_int"].sum()
        rate = n_speed / n
        mean_max = sub["max_speed_from_profile"].mean()
        mean_curv = sub["curvature_index"].mean()
        print(f"  {cls:10s}: n={n:>9,}, speeding={rate:.3f}, "
              f"mean_max_speed={mean_max:.1f}, mean_curvature_index={mean_curv:.4f}")
        stats_list.append({
            "class": cls,
            "n_trips": int(n),
            "speeding_rate": float(rate),
            "mean_max_speed": float(mean_max),
            "mean_curvature_index": float(mean_curv),
        })

    # Chi-square test
    contingency = pd.crosstab(df["curvature_class"], df["is_speeding_int"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
    print(f"  Chi-square: chi2={chi2:.1f}, p={p_chi2:.2e}, dof={dof}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: speeding rate
    ax = axes[0]
    rates = [s["speeding_rate"] for s in stats_list]
    colors = [COLORS["green"], COLORS["orange"], COLORS["red"]]
    bars = ax.bar(class_order, rates, color=colors, edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, stats_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{s["speeding_rate"]:.1%}\n(n={s["n_trips"]:,})',
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Speeding rate (P[max > 25 km/h])")
    ax.set_xlabel("Trip curvature class")
    ax.set_title("Speeding Rate by Curvature Class")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Right: mean max speed
    ax = axes[1]
    max_speeds = [s["mean_max_speed"] for s in stats_list]
    bars = ax.bar(class_order, max_speeds, color=colors, edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, stats_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{s["mean_max_speed"]:.1f}',
                ha="center", va="bottom", fontsize=10)
    ax.axhline(y=SPEED_LIMIT_KR, color=COLORS["black"], linestyle="--",
               linewidth=1, alpha=0.7, label=f"Speed limit ({SPEED_LIMIT_KR} km/h)")
    ax.set_ylabel("Mean max speed (km/h)")
    ax.set_xlabel("Trip curvature class")
    ax.set_title("Mean Max Speed by Curvature Class")
    ax.legend()

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_curvature_speeding_rate.{ext}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved fig_curvature_speeding_rate.pdf/png")

    return {
        "class_stats": stats_list,
        "chi_square": {"chi2": float(chi2), "pvalue": float(p_chi2), "dof": int(dof)},
    }


def analyze_continuous_curvature(df: pd.DataFrame) -> Dict[str, Any]:
    """LOWESS curve of speeding rate vs curvature index.

    Args:
        df: Merged data.

    Returns:
        Results dictionary.
    """
    print("\n=== Analysis 2: Continuous curvature effect ===")

    # Quintile stats
    quintile_stats = []
    for q in sorted(df["curvature_quintile"].dropna().unique()):
        sub = df[df["curvature_quintile"] == q]
        quintile_stats.append({
            "quintile": int(q),
            "n": int(len(sub)),
            "speeding_rate": float(sub["is_speeding_int"].mean()),
            "mean_curvature": float(sub["curvature_index"].mean()),
            "median_curvature": float(sub["curvature_index"].median()),
        })
        print(f"  Q{int(q)}: n={len(sub):,}, "
              f"curvature={sub['curvature_index'].median():.4f}, "
              f"speeding={sub['is_speeding_int'].mean():.3f}")

    # LOWESS on subsample
    sample_n = min(100_000, len(df))
    sample = df.sample(n=sample_n, random_state=RANDOM_SEED)

    lowess_result = sm_lowess(
        sample["is_speeding_int"].values,
        sample["curvature_index"].values,
        frac=0.2,
        it=3,
        return_sorted=True,
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Quintile points
    q_curv = [s["median_curvature"] for s in quintile_stats]
    q_rate = [s["speeding_rate"] for s in quintile_stats]
    ax.scatter(q_curv, q_rate, s=80, color=COLORS["blue"], zorder=3,
               edgecolors="black", linewidth=0.5, label="Quintile mean")

    # LOWESS
    ax.plot(lowess_result[:, 0], lowess_result[:, 1],
            color=COLORS["red"], linewidth=2, label="LOWESS")

    ax.axhline(y=df["is_speeding_int"].mean(), color=COLORS["black"],
               linestyle="--", alpha=0.5, linewidth=1, label="Overall mean")

    ax.set_xlabel("Curvature index (deg/m)")
    ax.set_ylabel("Speeding rate")
    ax.set_title("Speeding Rate vs. Trajectory Curvature")
    ax.legend(loc="upper right")
    ax.set_xlim(0, df["curvature_index"].quantile(0.99))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_curvature_continuous.{ext}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved fig_curvature_continuous.pdf/png")

    return {"quintile_stats": quintile_stats}


def analyze_curvature_roadclass_interaction(df: pd.DataFrame) -> Dict[str, Any]:
    """Heatmap of speeding rate by curvature class x road class.

    Args:
        df: Merged data.

    Returns:
        Results dictionary.
    """
    print("\n=== Analysis 3: Curvature x Road Class interaction ===")

    road_classes = ["residential", "tertiary", "service", "footway",
                    "primary", "secondary", "cycleway"]
    curv_classes = ["straight", "mixed", "curvy"]

    # Filter to valid road classes
    sub = df[df["dominant_road_class"].isin(road_classes)]

    pivot = sub.groupby(["curvature_class", "dominant_road_class"]).agg(
        n_trips=("is_speeding_int", "count"),
        n_speeding=("is_speeding_int", "sum"),
    ).reset_index()
    pivot["speeding_rate"] = pivot["n_speeding"] / pivot["n_trips"]

    # Create heatmap matrix
    heatmap = pivot.pivot(
        index="dominant_road_class",
        columns="curvature_class",
        values="speeding_rate",
    )
    # Order road classes by overall speeding rate
    road_order = (
        sub.groupby("dominant_road_class")["is_speeding_int"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    road_order = [r for r in road_order if r in heatmap.index]
    heatmap = heatmap.reindex(road_order)
    heatmap = heatmap[curv_classes]

    counts = pivot.pivot(
        index="dominant_road_class",
        columns="curvature_class",
        values="n_trips",
    ).reindex(road_order)[curv_classes]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = heatmap.values[i, j]
            n = counts.values[i, j]
            if np.isnan(val) or np.isnan(n):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="gray")
            else:
                text_color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.1%}\n(n={int(n):,})",
                        ha="center", va="center", fontsize=9, color=text_color)

    ax.set_xticks(range(len(curv_classes)))
    ax.set_xticklabels([c.capitalize() for c in curv_classes])
    ax.set_yticks(range(len(road_order)))
    ax.set_yticklabels([r.capitalize() for r in road_order])
    ax.set_xlabel("Curvature Class")
    ax.set_ylabel("Dominant Road Class")
    ax.set_title("Speeding Rate by Curvature and Road Class")

    cbar = fig.colorbar(im, ax=ax, label="Speeding rate", shrink=0.8)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_curvature_roadclass.{ext}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved fig_curvature_roadclass.pdf/png")

    # Serialize
    heatmap_json = {}
    for rc in road_order:
        heatmap_json[rc] = {}
        for cc in curv_classes:
            row = pivot[
                (pivot["dominant_road_class"] == rc) & (pivot["curvature_class"] == cc)
            ]
            if len(row) > 0:
                heatmap_json[rc][cc] = {
                    "speeding_rate": float(row["speeding_rate"].iloc[0]),
                    "n_trips": int(row["n_trips"].iloc[0]),
                }

    return {"heatmap": heatmap_json}


def analyze_curvature_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """Logistic regression: speeding ~ curvature + controls.

    Args:
        df: Merged data.

    Returns:
        Results dictionary with model coefficients and ORs.
    """
    print("\n=== Analysis 4: Curvature logistic regression ===")

    sample_n = min(500_000, len(df))
    sample = df.sample(n=sample_n, random_state=RANDOM_SEED).copy()
    sample = sample.dropna(subset=["age_group", "curvature_index"])
    sample["mode_str"] = sample["mode_clean"].astype(str)
    sample["age_str"] = sample["age_group"].astype(str)

    formula = (
        "is_speeding_int ~ log_curvature + frac_major_road + frac_cycling_infra "
        "+ log_distance "
        "+ C(mode_str, Treatment(reference='STD')) "
        "+ C(age_str)"
    )

    print(f"  Fitting on {len(sample):,} trips...")
    model = smf.logit(formula, data=sample).fit(disp=False, maxiter=100)
    print(f"  AIC={model.aic:.1f}, pseudo-R2={model.prsquared:.4f}")

    # Extract curvature OR
    curv_coef = model.params.get("log_curvature", None)
    curv_or = np.exp(curv_coef) if curv_coef is not None else None
    curv_p = model.pvalues.get("log_curvature", None)
    curv_ci = model.conf_int().loc["log_curvature"] if "log_curvature" in model.params.index else None

    print(f"  Curvature effect: OR={curv_or:.3f}, p={curv_p:.2e}")
    if curv_ci is not None:
        print(f"    95% CI: [{np.exp(curv_ci[0]):.3f}, {np.exp(curv_ci[1]):.3f}]")

    # All coefficients
    coefficients = {}
    for name in model.params.index:
        coefficients[name] = {
            "coef": float(model.params[name]),
            "or": float(np.exp(model.params[name])),
            "pvalue": float(model.pvalues[name]),
            "or_ci_lower": float(np.exp(model.conf_int().loc[name, 0])),
            "or_ci_upper": float(np.exp(model.conf_int().loc[name, 1])),
        }

    return {
        "sample_size": int(len(sample)),
        "aic": float(model.aic),
        "pseudo_r2": float(model.prsquared),
        "curvature_or": float(curv_or) if curv_or else None,
        "curvature_pvalue": float(curv_p) if curv_p else None,
        "coefficients": coefficients,
    }


def analyze_risk_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Composite risk score: speed x curvature.

    Higher speed on curvy roads = higher risk. The risk score combines
    the speeding intensity (how far above limit) with the curvature.

    Args:
        df: Merged data.

    Returns:
        Results dictionary.
    """
    print("\n=== Analysis 5: Speed x Curvature risk scoring ===")

    # Risk = (max_speed - speed_limit)_+ * curvature_index
    # Only for speeding trips
    df_speed = df[df["is_speeding_int"] == 1].copy()
    df_speed["speed_excess"] = df_speed["max_speed_from_profile"] - SPEED_LIMIT_KR
    df_speed["risk_score"] = df_speed["speed_excess"] * df_speed["curvature_index"]

    print(f"  Speeding trips: {len(df_speed):,}")
    print(f"  Risk score: mean={df_speed['risk_score'].mean():.4f}, "
          f"median={df_speed['risk_score'].median():.4f}, "
          f"P95={df_speed['risk_score'].quantile(0.95):.4f}")

    # Risk by curvature class
    risk_by_class = df_speed.groupby("curvature_class")["risk_score"].agg(
        ["mean", "median", "count"]
    ).reset_index()
    print("\n  Risk score by curvature class:")
    for _, row in risk_by_class.iterrows():
        print(f"    {row['curvature_class']:10s}: "
              f"mean={row['mean']:.4f}, median={row['median']:.4f}, n={int(row['count']):,}")

    # Scatter plot: speed excess vs curvature
    fig, ax = plt.subplots(figsize=(8, 6))

    # Subsample for visualization
    sample = df_speed.sample(n=min(20_000, len(df_speed)), random_state=RANDOM_SEED)

    scatter = ax.scatter(
        sample["curvature_index"],
        sample["speed_excess"],
        c=sample["risk_score"],
        cmap="YlOrRd",
        s=3,
        alpha=0.3,
        rasterized=True,
    )

    cbar = fig.colorbar(scatter, ax=ax, label="Risk score (excess speed x curvature)")

    # Add curvature class boundaries as vertical lines
    ax.set_xlabel("Curvature index (deg/m)")
    ax.set_ylabel("Speed excess above 25 km/h")
    ax.set_title("Speeding Risk: Speed Excess vs. Trajectory Curvature")
    ax.set_xlim(0, df_speed["curvature_index"].quantile(0.99))
    ax.set_ylim(0, df_speed["speed_excess"].quantile(0.99))

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_curvature_risk_scatter.{ext}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved fig_curvature_risk_scatter.pdf/png")

    risk_stats = {
        "n_speeding_trips": int(len(df_speed)),
        "risk_score_mean": float(df_speed["risk_score"].mean()),
        "risk_score_median": float(df_speed["risk_score"].median()),
        "risk_score_p95": float(df_speed["risk_score"].quantile(0.95)),
        "risk_by_class": risk_by_class.to_dict(orient="records"),
    }

    return risk_stats


def main() -> None:
    """Run all curvature-speeding analyses."""
    t0 = time.time()
    print("=" * 70)
    print("  CURVATURE-SPEEDING ANALYSIS")
    print("=" * 70)

    df = load_merged_data()

    results: Dict[str, Any] = {}
    results["curvature_class"] = analyze_speeding_by_curvature_class(df)
    results["continuous_curvature"] = analyze_continuous_curvature(df)
    results["curvature_roadclass"] = analyze_curvature_roadclass_interaction(df)
    results["curvature_regression"] = analyze_curvature_regression(df)
    results["risk_score"] = analyze_risk_score(df)

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {RESULTS_PATH}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
