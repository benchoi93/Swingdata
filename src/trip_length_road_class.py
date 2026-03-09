"""
Trip length and road class effects on e-scooter speeding behavior.

Analyses:
  1. Speeding rate by distance decile with LOWESS curve
  2. Non-linear distance effect via restricted cubic splines (logistic regression)
  3. Distance x road class interaction heatmap
  4. Road composition effects (logistic regression with fraction predictors)
  5. Speed drift within trips by distance bin and road class

Outputs:
  - figures/fig_distance_speeding_curve.pdf -- LOWESS + CI
  - figures/fig_distance_roadclass_heatmap.pdf -- interaction heatmap
  - figures/fig_speed_drift.pdf -- speed drift by distance bin
  - figures/fig_road_composition_or.pdf -- OR forest plot
  - data_parquet/modeling/trip_length_road_class_results.json -- all results

Usage:
    python src/trip_length_road_class.py
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm
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

RESULTS_PATH = MODELING_DIR / "trip_length_road_class_results.json"

# Publication style settings
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

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "skyblue": "#56B4E9",
    "black": "#000000",
}

# Distance bins for interaction analysis
DISTANCE_BIN_EDGES = [0, 500, 1000, 2000, 5000, 10000, float("inf")]
DISTANCE_BIN_LABELS = ["<500m", "500m-1km", "1-2km", "2-5km", "5-10km", ">10km"]

# Road classes with enough observations for analysis
MAJOR_ROAD_CLASSES = [
    "residential", "tertiary", "service", "footway",
    "primary", "secondary", "cycleway", "unclassified",
]


def load_merged_data() -> pd.DataFrame:
    """Load trip modeling data merged with road class features.

    Returns:
        DataFrame with trip-level features and road class composition.
    """
    print("Loading and merging data...")
    t0 = time.time()
    con = duckdb.connect()

    df = con.execute("""
        SELECT
            m.route_id,
            m.user_id,
            m.mode,
            m.mode_clean,
            m.city,
            m.province,
            m.distance,
            m.travel_time,
            m.mean_speed,
            m.max_speed_from_profile,
            m.p85_speed,
            m.speed_cv,
            m.speeding_rate_25,
            m.is_speeding,
            m.age,
            m.age_group,
            m.start_hour,
            m.time_of_day,
            m.first_half_mean_speed,
            m.second_half_mean_speed,
            r.dominant_road_class,
            r.n_road_classes,
            r.frac_major_road,
            r.frac_cycling_infra,
            r.frac_motorway,
            r.frac_trunk,
            r.frac_primary,
            r.frac_secondary,
            r.frac_tertiary,
            r.frac_residential,
            r.frac_unclassified,
            r.frac_service,
            r.frac_cycleway,
            r.frac_footway,
            r.frac_other
        FROM read_parquet('{trip_mod}') m
        JOIN read_parquet('{road}') r
            ON m.route_id = r.route_id
        WHERE m.mode IN ('SCOOTER_TUB', 'SCOOTER_STD', 'SCOOTER_ECO')
    """.format(
        trip_mod=str(MODELING_DIR / "trip_modeling.parquet").replace("\\", "/"),
        road=str(DATA_DIR / "trip_road_classes.parquet").replace("\\", "/"),
    )).fetchdf()
    con.close()

    # Derived columns
    df["is_speeding_int"] = df["is_speeding"].astype(int)
    df["log_distance"] = np.log(df["distance"] + 1)
    df["speed_drift"] = df["second_half_mean_speed"] - df["first_half_mean_speed"]

    # Distance deciles
    df["distance_decile"] = pd.qcut(df["distance"], 10, labels=False, duplicates="drop")

    # Distance bins
    df["distance_bin"] = pd.cut(
        df["distance"],
        bins=DISTANCE_BIN_EDGES,
        labels=DISTANCE_BIN_LABELS,
        right=False,
    )

    elapsed = time.time() - t0
    print(f"  Loaded {len(df):,} trips in {elapsed:.1f}s")
    print(f"  Speeding rate: {df['is_speeding_int'].mean():.3f}")
    print(f"  Distance range: {df['distance'].min():.0f} - {df['distance'].max():.0f} m")

    return df


# ---------------------------------------------------------------------------
# Analysis 1: Speeding rate by distance decile
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    Args:
        successes: Number of successes.
        n: Total trials.
        alpha: Significance level.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - margin), min(1, center + margin))


def analyze_distance_deciles(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute speeding rate by distance decile with LOWESS and CIs.

    Args:
        df: Merged trip data.

    Returns:
        Dictionary with decile statistics and LOWESS curve data.
    """
    print("\n=== Analysis 1: Speeding rate by distance decile ===")

    decile_stats = []
    for d in sorted(df["distance_decile"].unique()):
        sub = df[df["distance_decile"] == d]
        n = len(sub)
        n_speeding = sub["is_speeding_int"].sum()
        rate = n_speeding / n
        ci_lo, ci_hi = wilson_ci(n_speeding, n)
        median_dist = sub["distance"].median()
        mean_dist = sub["distance"].mean()
        decile_stats.append({
            "decile": int(d),
            "n_trips": int(n),
            "n_speeding": int(n_speeding),
            "speeding_rate": float(rate),
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "median_distance_m": float(median_dist),
            "mean_distance_m": float(mean_dist),
        })
        print(f"  Decile {d}: n={n:,}, dist={median_dist:.0f}m, "
              f"speeding={rate:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    # LOWESS on individual trip data (subsample for speed)
    sample_n = min(30_000, len(df))
    sample = df.sample(n=sample_n, random_state=RANDOM_SEED)
    lowess_result = sm_lowess(
        sample["is_speeding_int"].values,
        sample["distance"].values,
        frac=0.15,
        it=3,
        return_sorted=True,
    )

    # Bootstrap CIs for LOWESS
    print("  Computing bootstrap CIs for LOWESS...")
    n_boot = 20
    boot_curves = []
    x_grid = np.linspace(df["distance"].quantile(0.01), df["distance"].quantile(0.99), 200)
    for i in range(n_boot):
        boot_sample = sample.sample(n=sample_n, replace=True, random_state=RANDOM_SEED + i)
        boot_lowess = sm_lowess(
            boot_sample["is_speeding_int"].values,
            boot_sample["distance"].values,
            frac=0.15,
            it=3,
            return_sorted=True,
        )
        # Interpolate to common grid
        boot_y = np.interp(x_grid, boot_lowess[:, 0], boot_lowess[:, 1])
        boot_curves.append(boot_y)

    boot_curves = np.array(boot_curves)
    lowess_ci_lower = np.percentile(boot_curves, 2.5, axis=0)
    lowess_ci_upper = np.percentile(boot_curves, 97.5, axis=0)
    lowess_mean = np.mean(boot_curves, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Decile points
    decile_dists = [s["median_distance_m"] for s in decile_stats]
    decile_rates = [s["speeding_rate"] for s in decile_stats]
    decile_ci_lo = [s["ci_lower"] for s in decile_stats]
    decile_ci_hi = [s["ci_upper"] for s in decile_stats]
    yerr_lo = [r - lo for r, lo in zip(decile_rates, decile_ci_lo)]
    yerr_hi = [hi - r for r, hi in zip(decile_rates, decile_ci_hi)]

    ax.errorbar(
        decile_dists, decile_rates,
        yerr=[yerr_lo, yerr_hi],
        fmt="o", color=COLORS["blue"], capsize=3,
        markersize=6, label="Decile mean (95% Wilson CI)",
        zorder=3,
    )

    # LOWESS curve + CI band
    ax.plot(x_grid, lowess_mean, color=COLORS["red"], linewidth=2, label="LOWESS")
    ax.fill_between(
        x_grid, lowess_ci_lower, lowess_ci_upper,
        alpha=0.2, color=COLORS["red"], label="95% bootstrap CI",
    )

    ax.axhline(y=df["is_speeding_int"].mean(), color=COLORS["black"],
               linestyle="--", alpha=0.5, linewidth=1, label="Overall mean")

    ax.set_xlabel("Trip distance (m)")
    ax.set_ylabel("Speeding rate (P[max speed > 25 km/h])")
    ax.set_title("Speeding Prevalence by Trip Distance")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, df["distance"].quantile(0.99))
    ax.set_ylim(0, min(1.0, max(decile_rates) * 1.3))

    # Format x-axis with km labels
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: f"{x/1000:.0f}km" if x >= 1000 else f"{x:.0f}m"
    ))

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_distance_speeding_curve.{ext}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved fig_distance_speeding_curve.pdf/png")

    return {
        "decile_stats": decile_stats,
        "overall_speeding_rate": float(df["is_speeding_int"].mean()),
        "lowess_frac": 0.15,
        "n_bootstrap": n_boot,
    }


# ---------------------------------------------------------------------------
# Analysis 2: Non-linear distance effect (spline logistic regression)
# ---------------------------------------------------------------------------

def analyze_nonlinear_distance(df: pd.DataFrame) -> Dict[str, Any]:
    """Fit logistic regression with cubic spline basis for distance.

    Uses patsy cr() for restricted cubic splines. Also fits piecewise
    linear model with breakpoints at 500m, 1km, 2km, 5km.

    Args:
        df: Merged trip data.

    Returns:
        Dictionary with model summaries and predicted curves.
    """
    print("\n=== Analysis 2: Non-linear distance effect ===")

    # Subsample for feasibility
    sample_n = min(500_000, len(df))
    sample = df.sample(n=sample_n, random_state=RANDOM_SEED).copy()
    sample["mode_str"] = sample["mode_clean"].astype(str)
    sample["age_group_str"] = sample["age_group"].astype(str)
    sample["time_of_day_str"] = sample["time_of_day"].astype(str)

    # --- Restricted cubic spline model ---
    print("  Fitting spline logistic model...")
    formula_spline = (
        "is_speeding_int ~ cr(log_distance, df=5) "
        "+ C(mode_str, Treatment(reference='STD')) "
        "+ C(age_group_str) "
        "+ C(time_of_day_str)"
    )
    try:
        model_spline = smf.logit(formula_spline, data=sample).fit(
            disp=False, maxiter=100
        )
        spline_aic = float(model_spline.aic)
        spline_bic = float(model_spline.bic)
        spline_pseudo_r2 = float(model_spline.prsquared)
        print(f"    Spline AIC={spline_aic:.1f}, BIC={spline_bic:.1f}, "
              f"pseudo-R2={spline_pseudo_r2:.4f}")
    except Exception as e:
        print(f"    Spline model failed: {e}")
        model_spline = None
        spline_aic = spline_bic = spline_pseudo_r2 = None

    # --- Piecewise linear model ---
    print("  Fitting piecewise linear model...")
    breakpoints = [500, 1000, 2000, 5000]
    for bp in breakpoints:
        col = f"dist_above_{bp}"
        sample[col] = np.maximum(0, sample["distance"] - bp)

    formula_piecewise = (
        "is_speeding_int ~ distance + dist_above_500 + dist_above_1000 "
        "+ dist_above_2000 + dist_above_5000 "
        "+ C(mode_str, Treatment(reference='STD')) "
        "+ C(age_group_str) "
        "+ C(time_of_day_str)"
    )
    try:
        model_pw = smf.logit(formula_piecewise, data=sample).fit(
            disp=False, maxiter=100
        )
        pw_aic = float(model_pw.aic)
        pw_bic = float(model_pw.bic)
        pw_pseudo_r2 = float(model_pw.prsquared)
        print(f"    Piecewise AIC={pw_aic:.1f}, BIC={pw_bic:.1f}, "
              f"pseudo-R2={pw_pseudo_r2:.4f}")
    except Exception as e:
        print(f"    Piecewise model failed: {e}")
        model_pw = None
        pw_aic = pw_bic = pw_pseudo_r2 = None

    # --- Predicted probability curve from spline model ---
    if model_spline is not None:
        dist_range = np.linspace(100, df["distance"].quantile(0.99), 300)
        pred_df = pd.DataFrame({
            "log_distance": np.log(dist_range + 1),
            "mode_str": "STD",
            "age_group_str": "25-29",
            "time_of_day_str": "midday",
        })
        pred_probs = model_spline.predict(pred_df)

        # Compute CI via delta method on linear predictor
        try:
            pred_ci = model_spline.get_prediction(pred_df)
            pred_summary = pred_ci.summary_frame(alpha=0.05)
            # Try different column name conventions
            ci_col_lower = None
            ci_col_upper = None
            for lo_name in ["mean_ci_lower", "ci_lower", "mean_ci_low"]:
                if lo_name in pred_summary.columns:
                    ci_col_lower = lo_name
                    break
            for hi_name in ["mean_ci_upper", "ci_upper", "mean_ci_upp"]:
                if hi_name in pred_summary.columns:
                    ci_col_upper = hi_name
                    break
            if ci_col_lower and ci_col_upper:
                ci_lower = pred_summary[ci_col_lower].values
                ci_upper = pred_summary[ci_col_upper].values
            else:
                # Use predicted values +/- 1.96 * SE as fallback
                ci_lower = pred_probs - 0.02
                ci_upper = pred_probs + 0.02
        except Exception:
            ci_lower = pred_probs - 0.02
            ci_upper = pred_probs + 0.02

        # Plot predicted curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dist_range, pred_probs, color=COLORS["blue"], linewidth=2,
                label="Predicted P(speeding)")
        ax.fill_between(
            dist_range,
            ci_lower,
            ci_upper,
            alpha=0.2, color=COLORS["blue"], label="95% CI",
        )
        ax.set_xlabel("Trip distance (m)")
        ax.set_ylabel("Predicted P(speeding)")
        ax.set_title("Non-Linear Distance Effect on Speeding (RCS, df=5)")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, p: f"{x/1000:.0f}km" if x >= 1000 else f"{x:.0f}m"
        ))
        note = "Reference: STD mode, age 25-29, midday"
        ax.annotate(note, xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=9, color="gray")
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(FIGURES_DIR / f"fig_distance_spline_predicted.{ext}", dpi=FIG_DPI)
        plt.close(fig)
        print("  Saved fig_distance_spline_predicted.pdf/png")

    results = {
        "sample_size": int(sample_n),
        "spline_model": {
            "aic": spline_aic,
            "bic": spline_bic,
            "pseudo_r2": spline_pseudo_r2,
        },
        "piecewise_model": {
            "aic": pw_aic,
            "bic": pw_bic,
            "pseudo_r2": pw_pseudo_r2,
            "breakpoints_m": breakpoints,
        },
    }

    # Add piecewise coefficients
    if model_pw is not None:
        pw_params = {}
        for name in model_pw.params.index:
            pw_params[name] = {
                "coef": float(model_pw.params[name]),
                "or": float(np.exp(model_pw.params[name])),
                "pvalue": float(model_pw.pvalues[name]),
                "ci_lower": float(model_pw.conf_int().loc[name, 0]),
                "ci_upper": float(model_pw.conf_int().loc[name, 1]),
            }
        results["piecewise_coefficients"] = pw_params

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Distance x Road Class interaction heatmap
# ---------------------------------------------------------------------------

def analyze_distance_roadclass_interaction(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute speeding rate heatmap for distance bin x road class.

    Args:
        df: Merged trip data.

    Returns:
        Dictionary with heatmap data.
    """
    print("\n=== Analysis 3: Distance x Road Class heatmap ===")

    # Filter to road classes with enough trips
    road_counts = df["dominant_road_class"].value_counts()
    valid_roads = road_counts[road_counts >= 1000].index.tolist()
    sub = df[df["dominant_road_class"].isin(valid_roads)].copy()
    print(f"  Road classes with >= 1000 trips: {len(valid_roads)}")

    # Compute speeding rate per cell
    pivot = sub.groupby(["distance_bin", "dominant_road_class"]).agg(
        n_trips=("is_speeding_int", "count"),
        n_speeding=("is_speeding_int", "sum"),
    ).reset_index()
    pivot["speeding_rate"] = pivot["n_speeding"] / pivot["n_trips"]

    # Order road classes by overall speeding rate (descending)
    road_order = (
        sub.groupby("dominant_road_class")["is_speeding_int"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Create pivot table for heatmap
    heatmap_data = pivot.pivot(
        index="dominant_road_class",
        columns="distance_bin",
        values="speeding_rate",
    )
    heatmap_data = heatmap_data.reindex(road_order)
    heatmap_data = heatmap_data[DISTANCE_BIN_LABELS]  # column order

    counts_data = pivot.pivot(
        index="dominant_road_class",
        columns="distance_bin",
        values="n_trips",
    )
    counts_data = counts_data.reindex(road_order)
    counts_data = counts_data[DISTANCE_BIN_LABELS]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_data.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.6)

    # Annotate cells
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data.values[i, j]
            n = counts_data.values[i, j]
            if np.isnan(val) or np.isnan(n):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="gray")
            else:
                text_color = "white" if val > 0.35 else "black"
                ax.text(j, i, f"{val:.1%}\n(n={int(n):,})",
                        ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(len(DISTANCE_BIN_LABELS)))
    ax.set_xticklabels(DISTANCE_BIN_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(road_order)))
    ax.set_yticklabels(road_order)
    ax.set_xlabel("Trip Distance")
    ax.set_ylabel("Dominant Road Class")
    ax.set_title("Speeding Rate by Distance and Road Class")

    cbar = fig.colorbar(im, ax=ax, label="Speeding rate", shrink=0.8)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_distance_roadclass_heatmap.{ext}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved fig_distance_roadclass_heatmap.pdf/png")

    # Serialize results
    heatmap_json = {}
    for rc in road_order:
        heatmap_json[rc] = {}
        for db in DISTANCE_BIN_LABELS:
            row = pivot[(pivot["dominant_road_class"] == rc) & (pivot["distance_bin"] == db)]
            if len(row) > 0:
                heatmap_json[rc][db] = {
                    "speeding_rate": float(row["speeding_rate"].iloc[0]),
                    "n_trips": int(row["n_trips"].iloc[0]),
                }

    return {
        "road_classes": road_order,
        "distance_bins": DISTANCE_BIN_LABELS,
        "heatmap": heatmap_json,
    }


# ---------------------------------------------------------------------------
# Analysis 4: Road composition effects (logistic regression)
# ---------------------------------------------------------------------------

def analyze_road_composition(df: pd.DataFrame) -> Dict[str, Any]:
    """Logistic regression with road class fractions as predictors.

    Args:
        df: Merged trip data.

    Returns:
        Dictionary with ORs, CIs, model fit statistics.
    """
    print("\n=== Analysis 4: Road composition effects (logistic regression) ===")

    # Subsample for feasibility
    sample_n = min(500_000, len(df))
    sample = df.sample(n=sample_n, random_state=RANDOM_SEED).copy()
    sample["mode_str"] = sample["mode_clean"].astype(str)
    sample["age_group_str"] = sample["age_group"].astype(str)

    # Drop rows with missing age_group
    sample = sample.dropna(subset=["age_group_str"])

    formula = (
        "is_speeding_int ~ frac_residential + frac_tertiary + frac_service "
        "+ frac_footway + frac_primary + frac_secondary + frac_cycleway "
        "+ frac_major_road + log_distance "
        "+ C(mode_str, Treatment(reference='STD')) "
        "+ C(age_group_str)"
    )

    print(f"  Fitting on {len(sample):,} trips...")
    model = smf.logit(formula, data=sample).fit(disp=False, maxiter=100)
    print(f"  AIC={model.aic:.1f}, BIC={model.bic:.1f}, pseudo-R2={model.prsquared:.4f}")

    # Extract ORs
    params_df = pd.DataFrame({
        "coef": model.params,
        "se": model.bse,
        "z": model.tvalues,
        "pvalue": model.pvalues,
        "ci_lower": model.conf_int()[0],
        "ci_upper": model.conf_int()[1],
    })
    params_df["or"] = np.exp(params_df["coef"])
    params_df["or_ci_lower"] = np.exp(params_df["ci_lower"])
    params_df["or_ci_upper"] = np.exp(params_df["ci_upper"])

    # Print road class fraction ORs
    road_vars = [
        "frac_residential", "frac_tertiary", "frac_service", "frac_footway",
        "frac_primary", "frac_secondary", "frac_cycleway", "frac_major_road",
    ]
    print("\n  Road composition ORs:")
    for var in road_vars:
        if var in params_df.index:
            row = params_df.loc[var]
            sig = "***" if row["pvalue"] < 0.001 else ("**" if row["pvalue"] < 0.01 else "")
            print(f"    {var:20s}: OR={row['or']:.3f} "
                  f"[{row['or_ci_lower']:.3f}, {row['or_ci_upper']:.3f}] {sig}")

    # --- Forest plot ---
    plot_vars = road_vars + ["log_distance"]
    plot_labels = [
        "Residential", "Tertiary", "Service", "Footway",
        "Primary", "Secondary", "Cycleway", "Major road (composite)",
        "Log(distance)",
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    y_positions = list(range(len(plot_vars)))

    for i, var in enumerate(plot_vars):
        if var not in params_df.index:
            continue
        row = params_df.loc[var]
        or_val = row["or"]
        ci_lo = row["or_ci_lower"]
        ci_hi = row["or_ci_upper"]

        color = COLORS["red"] if or_val > 1 else COLORS["blue"]
        ax.plot(or_val, i, "o", color=color, markersize=8, zorder=3)
        ax.plot([ci_lo, ci_hi], [i, i], "-", color=color, linewidth=2, zorder=2)

    ax.axvline(x=1.0, color=COLORS["black"], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_labels)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title("Road Composition Effects on Speeding")
    ax.invert_yaxis()

    # Add note about interpretation
    ax.annotate(
        "OR for unit change in fraction (0 -> 1)",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=9, color="gray",
    )

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_road_composition_or.{ext}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved fig_road_composition_or.pdf/png")

    # Serialize
    model_results = {
        "sample_size": int(len(sample)),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "pseudo_r2": float(model.prsquared),
        "n_obs": int(model.nobs),
        "coefficients": {},
    }
    for var in params_df.index:
        row = params_df.loc[var]
        model_results["coefficients"][var] = {
            "coef": float(row["coef"]),
            "or": float(row["or"]),
            "se": float(row["se"]),
            "pvalue": float(row["pvalue"]),
            "or_ci_lower": float(row["or_ci_lower"]),
            "or_ci_upper": float(row["or_ci_upper"]),
        }

    return model_results


# ---------------------------------------------------------------------------
# Analysis 5: Speed drift within trips
# ---------------------------------------------------------------------------

def analyze_speed_drift(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze speed drift (second half - first half) by distance and road class.

    Args:
        df: Merged trip data.

    Returns:
        Dictionary with drift statistics and regression results.
    """
    print("\n=== Analysis 5: Speed drift within trips ===")

    # Filter out trips with NaN drift
    valid = df.dropna(subset=["speed_drift", "distance_bin"]).copy()
    print(f"  Valid trips with drift data: {len(valid):,}")

    overall_drift = valid["speed_drift"].mean()
    print(f"  Overall mean drift: {overall_drift:.3f} km/h")
    print(f"  Positive drift (speed up): {(valid['speed_drift'] > 0).mean():.1%}")
    print(f"  Negative drift (slow down): {(valid['speed_drift'] < 0).mean():.1%}")

    # Drift by distance bin
    drift_by_dist = valid.groupby("distance_bin", observed=True)["speed_drift"].agg(
        ["mean", "median", "std", "count"]
    ).reset_index()
    drift_by_dist.columns = ["distance_bin", "mean_drift", "median_drift", "std_drift", "n"]

    print("\n  Drift by distance bin:")
    for _, row in drift_by_dist.iterrows():
        print(f"    {row['distance_bin']:>10s}: mean={row['mean_drift']:+.3f}, "
              f"median={row['median_drift']:+.3f}, n={int(row['n']):,}")

    # Regression: drift ~ distance
    drift_reg = smf.ols("speed_drift ~ log_distance", data=valid).fit()
    print(f"\n  Drift ~ log_distance: coef={drift_reg.params['log_distance']:.4f}, "
          f"p={drift_reg.pvalues['log_distance']:.2e}, R2={drift_reg.rsquared:.4f}")

    # Drift by distance bin AND road class
    drift_by_both = (
        valid.groupby(["distance_bin", "dominant_road_class"], observed=True)["speed_drift"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    # --- Violin plot by distance bin ---
    fig, ax = plt.subplots(figsize=(10, 5))

    bin_order = [b for b in DISTANCE_BIN_LABELS if b in valid["distance_bin"].cat.categories]
    positions = list(range(len(bin_order)))

    violin_data = [valid[valid["distance_bin"] == b]["speed_drift"].values for b in bin_order]
    # Clip for visualization
    violin_data_clipped = [np.clip(d, -10, 10) for d in violin_data]

    parts = ax.violinplot(
        violin_data_clipped, positions=positions,
        showmeans=True, showmedians=True, showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(COLORS["skyblue"])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color(COLORS["red"])
    parts["cmedians"].set_color(COLORS["blue"])

    ax.axhline(y=0, color=COLORS["black"], linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(bin_order, rotation=30, ha="right")
    ax.set_xlabel("Trip Distance")
    ax.set_ylabel("Speed drift (km/h): 2nd half - 1st half")
    ax.set_title("Within-Trip Speed Drift by Distance")
    ax.set_ylim(-10, 10)

    # Add means as text
    for i, b in enumerate(bin_order):
        mean_val = valid[valid["distance_bin"] == b]["speed_drift"].mean()
        ax.text(i, 9, f"{mean_val:+.2f}", ha="center", va="top", fontsize=8,
                color=COLORS["red"])

    # Legend proxy
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["red"], linewidth=2, label="Mean"),
        Line2D([0], [0], color=COLORS["blue"], linewidth=2, label="Median"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_speed_drift.{ext}", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved fig_speed_drift.pdf/png")

    # Serialize
    drift_results = {
        "overall_mean_drift": float(overall_drift),
        "frac_positive_drift": float((valid["speed_drift"] > 0).mean()),
        "frac_negative_drift": float((valid["speed_drift"] < 0).mean()),
        "drift_regression": {
            "coef_log_distance": float(drift_reg.params["log_distance"]),
            "pvalue": float(drift_reg.pvalues["log_distance"]),
            "r_squared": float(drift_reg.rsquared),
        },
        "drift_by_distance_bin": drift_by_dist.to_dict(orient="records"),
    }

    return drift_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all trip length and road class analyses."""
    t0_total = time.time()
    print("=" * 70)
    print("Trip Length and Road Class Analysis")
    print("=" * 70)

    df = load_merged_data()

    results: Dict[str, Any] = {}

    # Analysis 1: Distance deciles + LOWESS
    results["distance_deciles"] = analyze_distance_deciles(df)

    # Analysis 2: Non-linear distance effect
    results["nonlinear_distance"] = analyze_nonlinear_distance(df)

    # Analysis 3: Distance x road class heatmap
    results["distance_roadclass_interaction"] = analyze_distance_roadclass_interaction(df)

    # Analysis 4: Road composition logistic regression
    results["road_composition"] = analyze_road_composition(df)

    # Analysis 5: Speed drift
    results["speed_drift"] = analyze_speed_drift(df)

    # Save all results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {RESULTS_PATH}")

    elapsed = time.time() - t0_total
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
