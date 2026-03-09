"""
Phase 6 (continued): Publication-quality figures for the e-scooter speed safety paper.

Creates figures that were not in the initial batch:
  - Fig 1:  Spatial distribution of trip origins across Korean cities
  - Fig 9:  Logistic regression coefficient (OR) forest plot
  - Fig 10: GEE regression coefficient (OR) forest plot
  - Fig 12: TUB vs ECO mode paired comparison

Outputs: figures/{name}.pdf and figures/{name}.png at 300 DPI
"""

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    CLEANED_PARQUET, DATA_DIR, FIGURES_DIR, MODELING_DIR,
    RANDOM_SEED, FIG_DPI, SPEED_LIMIT_KR,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
    "gray": "#999999",
}


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as both PDF and PNG."""
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.pdf + {name}.png")


# ---------------------------------------------------------------------------
# Fig 1: Spatial distribution of trip origins
# ---------------------------------------------------------------------------

def fig_spatial_distribution() -> None:
    """Create map of trip origins across Korean cities with bubble sizes."""
    print("\nFig 1: Spatial distribution of trip origins...")

    con = duckdb.connect()
    # Get city-level summary
    cities = con.execute(f"""
        SELECT
            city,
            province,
            COUNT(*) as n_trips,
            AVG(start_lat) as center_lat,
            AVG(start_lon) as center_lon,
            AVG(CASE WHEN has_speeding THEN 1 ELSE 0 END) as speeding_rate
        FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
        WHERE is_valid = true
        GROUP BY city, province
        HAVING COUNT(*) >= 1000
        ORDER BY n_trips DESC
    """).fetchdf()

    # Sample GPS points for background scatter
    sample_pts = con.execute(f"""
        SELECT start_lat, start_lon
        FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
        WHERE is_valid = true
        ORDER BY RANDOM()
        LIMIT 50000
    """).fetchdf()
    con.close()

    fig, ax = plt.subplots(figsize=(8, 10))

    # Background scatter of individual trip origins (light dots)
    ax.scatter(
        sample_pts["start_lon"], sample_pts["start_lat"],
        s=0.3, alpha=0.05, color=COLORS["blue"], rasterized=True,
    )

    # City bubbles scaled by trip count
    max_trips = cities["n_trips"].max()
    sizes = (cities["n_trips"] / max_trips) * 800 + 30

    # Color by speeding rate
    scatter = ax.scatter(
        cities["center_lon"], cities["center_lat"],
        s=sizes, c=cities["speeding_rate"] * 100,
        cmap="YlOrRd", edgecolors="black", linewidth=0.5,
        alpha=0.8, vmin=10, vmax=50, zorder=5,
    )

    # Label top 10 cities
    for _, row in cities.head(10).iterrows():
        ax.annotate(
            row["city"],
            xy=(row["center_lon"], row["center_lat"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="gray", alpha=0.8),
            zorder=10,
        )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Speeding Rate (%)")

    # Legend for bubble size
    for n_label, label in [(500000, "500K"), (100000, "100K"), (10000, "10K")]:
        size = (n_label / max_trips) * 800 + 30
        ax.scatter([], [], s=size, c="gray", alpha=0.5, edgecolors="black",
                   linewidth=0.5, label=f"{label} trips")
    ax.legend(loc="lower left", title="Trip Count", fontsize=9,
              framealpha=0.9)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Distribution of E-Scooter Trips Across South Korea\n"
                 f"(N = {cities['n_trips'].sum():,} trips, {len(cities)} cities)")
    ax.set_xlim(125.5, 131.5)
    ax.set_ylim(33.5, 38.5)
    ax.set_aspect(1.2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_figure(fig, "fig1_spatial_distribution")


# ---------------------------------------------------------------------------
# Fig 9-10: Regression coefficient (OR) forest plots
# ---------------------------------------------------------------------------

def _clean_label(raw_label: str) -> str:
    """Convert statsmodels coefficient name to readable label."""
    # Remove C(...) wrapper
    m = re.search(r"\[T\.(.+?)\]", raw_label)
    if m:
        category = m.group(1)
        if "age_group" in raw_label:
            return f"Age {category}"
        elif "mode" in raw_label:
            return f"Mode: {category}"
        elif "time_of_day" in raw_label:
            label = category.replace("_", " ").title()
            return f"Time: {label}"
        elif "day_type" in raw_label:
            return f"Day: {category.title()}"
        elif "province" in raw_label:
            return f"Prov: {category}"
    if raw_label == "Intercept":
        return "Intercept"
    return raw_label


def _group_coefficients(coeffs: dict) -> list[tuple[str, list]]:
    """Group coefficients by category for the forest plot."""
    groups = {
        "Mode": [],
        "Age": [],
        "Time of Day": [],
        "Day Type": [],
        "Province": [],
    }
    for raw_name, vals in coeffs.items():
        if raw_name == "Intercept":
            continue
        label = _clean_label(raw_name)
        entry = {
            "label": label,
            "or": vals["or"],
            "ci_low": vals["or_ci_low"],
            "ci_high": vals["or_ci_high"],
            "pvalue": vals["pvalue"],
        }
        if "mode" in raw_name:
            groups["Mode"].append(entry)
        elif "age_group" in raw_name:
            groups["Age"].append(entry)
        elif "time_of_day" in raw_name:
            groups["Time of Day"].append(entry)
        elif "day_type" in raw_name:
            groups["Day Type"].append(entry)
        elif "province" in raw_name:
            groups["Province"].append(entry)

    return [(k, v) for k, v in groups.items() if v]


def fig_regression_coefficients() -> None:
    """Create forest plots for logistic and GEE regression ORs."""
    print("\nFig 9-10: Regression coefficient plots...")

    with open(MODELING_DIR / "regression_results.json", "r") as f:
        results = json.load(f)

    for model_key, fig_name, title in [
        ("logistic_full", "fig9_logistic_OR",
         "Logistic Regression: Odds Ratios for Speeding (>25 km/h)"),
        ("gee_subsample", "fig10_gee_OR",
         "GEE Model: Odds Ratios for Speeding (>25 km/h)"),
    ]:
        model = results[model_key]
        coeffs = model["coefficients"]
        groups = _group_coefficients(coeffs)

        # Compute layout
        n_items = sum(len(items) + 1 for _, items in groups)  # +1 for group header
        fig_height = max(6, n_items * 0.35 + 1)

        fig, ax = plt.subplots(figsize=(10, fig_height))

        y_pos = 0
        y_labels = []
        y_positions = []
        group_boundaries = []

        for group_name, items in groups:
            # Group header
            y_pos += 0.5
            group_boundaries.append(y_pos)

            for entry in items:
                y_pos += 1
                or_val = entry["or"]
                ci_low = entry["ci_low"]
                ci_high = entry["ci_high"]

                # Clip for display (log scale)
                or_display = min(or_val, 200)
                ci_low_display = max(ci_low, 0.005)
                ci_high_display = min(ci_high, 200)

                # Color by significance and direction
                if entry["pvalue"] < 0.001:
                    if or_val > 1:
                        color = COLORS["red"]
                    else:
                        color = COLORS["blue"]
                elif entry["pvalue"] < 0.05:
                    color = COLORS["orange"]
                else:
                    color = COLORS["gray"]

                ax.plot(
                    [ci_low_display, ci_high_display], [y_pos, y_pos],
                    color=color, linewidth=1.5, solid_capstyle="round",
                )
                ax.plot(or_display, y_pos, "o", color=color, markersize=6)

                # Add OR text
                if or_val < 0.01 or or_val > 100:
                    or_text = f"{or_val:.1e}"
                else:
                    or_text = f"{or_val:.2f}"
                ax.text(
                    220, y_pos, or_text,
                    va="center", ha="left", fontsize=8, color=color,
                )

                y_labels.append(entry["label"])
                y_positions.append(y_pos)

        # Reference line at OR=1
        ax.axvline(x=1, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

        # Group labels
        current_pos = 0
        for (group_name, items), boundary in zip(groups, group_boundaries):
            ax.text(
                0.003, boundary + 0.5, group_name,
                fontsize=10, fontweight="bold", va="bottom",
                transform=ax.get_yaxis_transform(),
            )

        ax.set_xscale("log")
        ax.set_xlim(0.005, 250)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel("Odds Ratio (log scale)")
        ax.invert_yaxis()

        # Add model info
        n_obs = model.get("n_obs", model.get("gee_n_obs", "?"))
        info_text = f"N = {n_obs:,}" if isinstance(n_obs, int) else f"N = {n_obs}"
        if "pseudo_r2" in model:
            info_text += f" | Pseudo R2 = {model['pseudo_r2']:.3f}"
        ax.set_title(f"{title}\n({info_text})", fontsize=12)

        ax.grid(True, axis="x", alpha=0.2)

        # Legend
        legend_elements = [
            mpatches.Patch(color=COLORS["red"], label="Risk factor (p<0.001)"),
            mpatches.Patch(color=COLORS["blue"], label="Protective (p<0.001)"),
            mpatches.Patch(color=COLORS["orange"], label="p<0.05"),
            mpatches.Patch(color=COLORS["gray"], label="Not significant"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()
        save_figure(fig, fig_name)


# ---------------------------------------------------------------------------
# Fig 12: TUB vs ECO paired comparison
# ---------------------------------------------------------------------------

def fig_tub_vs_eco() -> None:
    """Create publication-quality TUB vs ECO mode comparison figure.

    Four-panel layout:
      (a) Density plots of mean speed by mode (all trips)
      (b) Density plots of max speed by mode (all trips)
      (c) Within-subject scatter: mean speed
      (d) Within-subject bar chart: effect sizes (Cohen's d)
    """
    print("\nFig 12: TUB vs ECO mode comparison...")

    # Load trip data
    trip_df = pd.read_parquet(
        MODELING_DIR / "trip_modeling.parquet",
        columns=["route_id", "user_id", "mode", "mode_clean",
                 "mean_speed", "max_speed_from_profile", "p85_speed",
                 "speeding_rate_25", "speed_cv", "mean_abs_accel_ms2",
                 "cruise_fraction", "zero_speed_fraction"],
    )

    scooter_modes = ["SCOOTER_ECO", "SCOOTER_STD", "SCOOTER_TUB"]
    df = trip_df[trip_df["mode"].isin(scooter_modes)].copy()

    mode_colors = {
        "SCOOTER_TUB": COLORS["red"],
        "SCOOTER_STD": COLORS["blue"],
        "SCOOTER_ECO": COLORS["green"],
    }
    mode_labels = {
        "SCOOTER_TUB": "Turbo (TUB)",
        "SCOOTER_STD": "Standard (STD)",
        "SCOOTER_ECO": "Eco (ECO)",
    }

    # Load paired analysis results
    with open(MODELING_DIR / "mode_comparison.json", "r") as f:
        comparison = json.load(f)
    paired = comparison["within_subject_paired"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Mean speed density by mode
    ax = axes[0, 0]
    from scipy.stats import gaussian_kde
    for mode in ["SCOOTER_ECO", "SCOOTER_STD", "SCOOTER_TUB"]:
        mode_data = df[df["mode"] == mode]["mean_speed"].dropna()
        if len(mode_data) > 50000:
            mode_data = mode_data.sample(50000, random_state=RANDOM_SEED)
        x_range = np.linspace(0, 35, 300)
        kde = gaussian_kde(mode_data.values, bw_method=0.15)
        ax.plot(x_range, kde(x_range), linewidth=2, color=mode_colors[mode],
                label=mode_labels[mode])
        ax.fill_between(x_range, kde(x_range), alpha=0.12, color=mode_colors[mode])

    ax.axvline(SPEED_LIMIT_KR, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label=f"Limit ({SPEED_LIMIT_KR} km/h)")
    ax.set_xlabel("Mean Trip Speed (km/h)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Mean speed distribution by mode")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 35)

    # (b) Max speed density by mode
    ax = axes[0, 1]
    for mode in ["SCOOTER_ECO", "SCOOTER_STD", "SCOOTER_TUB"]:
        mode_data = df[df["mode"] == mode]["max_speed_from_profile"].dropna()
        if len(mode_data) > 50000:
            mode_data = mode_data.sample(50000, random_state=RANDOM_SEED)
        x_range = np.linspace(0, 45, 300)
        kde = gaussian_kde(mode_data.values, bw_method=0.15)
        ax.plot(x_range, kde(x_range), linewidth=2, color=mode_colors[mode],
                label=mode_labels[mode])
        ax.fill_between(x_range, kde(x_range), alpha=0.12, color=mode_colors[mode])

    ax.axvline(SPEED_LIMIT_KR, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label=f"Limit ({SPEED_LIMIT_KR} km/h)")
    ax.set_xlabel("Max Trip Speed (km/h)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Max speed distribution by mode")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 45)

    # (c) Within-subject scatter: mean speed (ECO vs TUB)
    ax = axes[1, 0]
    tub_users = set(df[df["mode"] == "SCOOTER_TUB"]["user_id"].unique())
    eco_users = set(df[df["mode"] == "SCOOTER_ECO"]["user_id"].unique())
    both_users = tub_users & eco_users

    paired_df = df[df["user_id"].isin(both_users)].copy()
    user_mode = paired_df.groupby(["user_id", "mode"])[
        ["mean_speed", "max_speed_from_profile", "speeding_rate_25"]
    ].mean().reset_index()

    tub = user_mode[user_mode["mode"] == "SCOOTER_TUB"].set_index("user_id")
    eco = user_mode[user_mode["mode"] == "SCOOTER_ECO"].set_index("user_id")
    common = tub.index.intersection(eco.index)

    n_plot = min(3000, len(common))
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(common), n_plot, replace=False)
    common_arr = common[idx]

    ax.scatter(
        eco.loc[common_arr, "mean_speed"],
        tub.loc[common_arr, "mean_speed"],
        s=8, alpha=0.15, color=COLORS["blue"], rasterized=True,
    )
    lims = [0, 30]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="y = x")
    ax.set_xlabel("ECO Mode: Mean Speed (km/h)")
    ax.set_ylabel("TUB Mode: Mean Speed (km/h)")
    ax.set_title(f"(c) Within-subject comparison (n={len(common):,})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Annotate
    tub_mean = tub.loc[common, "mean_speed"].mean()
    eco_mean = eco.loc[common, "mean_speed"].mean()
    ax.annotate(
        f"TUB mean: {tub_mean:.1f} km/h\n"
        f"ECO mean: {eco_mean:.1f} km/h\n"
        f"Diff: {tub_mean - eco_mean:+.1f} km/h",
        xy=(0.05, 0.95), xycoords="axes fraction",
        va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax.legend(loc="lower right", fontsize=9)

    # (d) Effect sizes (Cohen's d) for paired comparison
    ax = axes[1, 1]
    paired_tests = paired["paired_tests"]

    indicators = [
        ("mean_speed", "Mean Speed"),
        ("max_speed_from_profile", "Max Speed"),
        ("p85_speed", "P85 Speed"),
        ("speeding_rate_25", "Speeding Rate"),
        ("mean_abs_accel_ms2", "Accel. Intensity"),
        ("speed_cv", "Speed CV"),
        ("cruise_fraction", "Cruise Fraction"),
        ("zero_speed_fraction", "Zero-Speed Frac."),
    ]

    labels = []
    d_values = []
    colors_list = []
    for key, label in indicators:
        if key in paired_tests:
            d = paired_tests[key]["cohens_d"]
            labels.append(label)
            d_values.append(d)
            colors_list.append(COLORS["red"] if d > 0 else COLORS["blue"])

    y_pos_arr = np.arange(len(labels))
    ax.barh(y_pos_arr, d_values, color=colors_list, edgecolor="white",
            linewidth=0.5, height=0.6, alpha=0.8)

    # Reference lines for effect size interpretation
    for ref, label_text in [(0.2, "Small"), (0.5, "Medium"), (0.8, "Large")]:
        ax.axvline(ref, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.axvline(-ref, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_yticks(y_pos_arr)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Cohen's d (TUB - ECO)")
    ax.set_title(f"(d) Effect sizes: TUB vs ECO (n={paired['n_paired_users']:,})")

    # Add value labels
    for i, (d, label) in enumerate(zip(d_values, labels)):
        ha = "left" if d >= 0 else "right"
        offset = 0.03 if d >= 0 else -0.03
        ax.text(d + offset, i, f"{d:.2f}", va="center", ha=ha, fontsize=8,
                fontweight="bold")

    ax.invert_yaxis()

    plt.suptitle(
        "Speed Governor Effect: TUB vs ECO Operating Mode",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save_figure(fig, "fig12_tub_vs_eco")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate remaining publication figures."""
    print("=" * 60)
    print("Phase 6 (continued): Publication Figures")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    fig_spatial_distribution()
    fig_regression_coefficients()
    fig_tub_vs_eco()

    print("\nAll additional figures created.")


if __name__ == "__main__":
    main()
