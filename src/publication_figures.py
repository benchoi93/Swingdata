"""
Phase 6: Publication-quality figures for the e-scooter speed safety paper.

Creates figures that do not require map-matching data:
  - Fig 2: Temporal distributions (hour-of-day and day-of-week)
  - Fig 3: Within-trip speed profile illustration
  - Fig 5: Speeding prevalence heatmap (hour x day_of_week)
  - Fig 10: TUB vs ECO comparison (improved version)

Outputs: figures/{name}.pdf and figures/{name}.png at 300 DPI
"""

import ast
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# Colorblind-friendly palette (based on Wong 2011)
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

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as both PDF and PNG."""
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.pdf + {name}.png")


# ---------------------------------------------------------------------------
# Fig 2: Temporal distributions
# ---------------------------------------------------------------------------

def fig_temporal_distributions() -> None:
    """Create temporal distribution figure (hour-of-day, day-of-week, speeding rates)."""
    print("\nFig 2: Temporal distributions...")

    con = duckdb.connect()
    # Hourly distribution
    hourly = con.execute(f"""
        SELECT start_hour,
               COUNT(*) as n_trips,
               AVG(CASE WHEN has_speeding THEN 1 ELSE 0 END) as speeding_rate
        FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
        WHERE is_valid = true
        GROUP BY start_hour
        ORDER BY start_hour
    """).fetchdf()

    # Day-of-week distribution
    daily = con.execute(f"""
        SELECT day_of_week,
               COUNT(*) as n_trips,
               AVG(CASE WHEN has_speeding THEN 1 ELSE 0 END) as speeding_rate
        FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
        WHERE is_valid = true
        GROUP BY day_of_week
        ORDER BY day_of_week
    """).fetchdf()
    con.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Trip count by hour
    ax = axes[0, 0]
    ax.bar(hourly["start_hour"], hourly["n_trips"] / 1000, color=COLORS["blue"],
           edgecolor="white", linewidth=0.3, width=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Trips (thousands)")
    ax.set_title("(a) Trip volume by hour")
    ax.set_xticks(range(0, 24, 3))

    # (b) Speeding rate by hour
    ax = axes[0, 1]
    ax.plot(hourly["start_hour"], hourly["speeding_rate"] * 100,
            color=COLORS["red"], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Speeding Rate (%)")
    ax.set_title("(b) Speeding rate by hour")
    ax.set_xticks(range(0, 24, 3))
    ax.axhline(y=hourly["speeding_rate"].mean() * 100, color="gray",
               linestyle="--", linewidth=0.8, alpha=0.7)

    # (c) Trip count by day of week
    ax = axes[1, 0]
    colors_dow = [COLORS["blue"]] * 5 + [COLORS["orange"]] * 2
    ax.bar(range(7), daily["n_trips"] / 1000, color=colors_dow,
           edgecolor="white", linewidth=0.3, width=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_NAMES)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Number of Trips (thousands)")
    ax.set_title("(c) Trip volume by day of week")

    # (d) Speeding rate by day of week
    ax = axes[1, 1]
    ax.bar(range(7), daily["speeding_rate"] * 100, color=colors_dow,
           edgecolor="white", linewidth=0.3, width=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_NAMES)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Speeding Rate (%)")
    ax.set_title("(d) Speeding rate by day of week")
    ax.axhline(y=daily["speeding_rate"].mean() * 100, color="gray",
               linestyle="--", linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    save_figure(fig, "fig2_temporal_distributions")


# ---------------------------------------------------------------------------
# Fig 3: Within-trip speed profile
# ---------------------------------------------------------------------------

def fig_speed_profile_example() -> None:
    """Create annotated example speed profile for a single trip."""
    print("\nFig 3: Example speed profile...")

    con = duckdb.connect()
    # Find a good example trip: medium length, has speeding, from Seoul
    trip = con.execute(f"""
        SELECT route_id, routes_raw, speeds_raw, gps_points, distance,
               mode, model, max_speed, city
        FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
        WHERE is_valid = true
          AND has_speeding = true
          AND gps_points BETWEEN 30 AND 80
          AND city = 'Seoul'
          AND mode = 'SCOOTER_TUB'
          AND max_speed BETWEEN 26 AND 35
        LIMIT 10
    """).fetchdf()
    con.close()

    if len(trip) == 0:
        print("  No suitable example trip found, skipping")
        return

    # Pick the first one
    row = trip.iloc[0]
    speeds = ast.literal_eval(row["speeds_raw"])
    if isinstance(speeds[0], list):
        speeds = speeds[0]
    speeds = np.array(speeds, dtype=float)

    # Parse trajectory for timestamps
    routes = ast.literal_eval(row["routes_raw"])
    timestamps = []
    for point in routes:
        time_str = point[1]  # 'HH:MM:SS.mmm'
        parts = time_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2])
        timestamps.append(h * 3600 + m * 60 + s)

    # Time axis (seconds from start)
    t0 = timestamps[0]
    time_s = np.array([t - t0 for t in timestamps])

    # Speed points may not align with GPS points
    n_speeds = len(speeds)
    n_gps = len(time_s)

    if n_speeds != n_gps:
        # Interpolate time axis for speeds
        speed_time = np.linspace(0, time_s[-1], n_speeds)
    else:
        speed_time = time_s

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Speed profile
    ax.plot(speed_time, speeds, color=COLORS["blue"], linewidth=1.5, label="Speed")

    # Speed limit
    ax.axhline(y=SPEED_LIMIT_KR, color=COLORS["red"], linestyle="--", linewidth=1.5,
               label=f"Speed limit ({SPEED_LIMIT_KR} km/h)")

    # Shade speeding regions
    speeding_mask = speeds > SPEED_LIMIT_KR
    ax.fill_between(speed_time, SPEED_LIMIT_KR, speeds,
                     where=speeding_mask, alpha=0.25, color=COLORS["red"],
                     label="Speeding")

    # Annotations
    # Max speed
    max_idx = np.argmax(speeds)
    ax.annotate(f"Max: {speeds[max_idx]:.0f} km/h",
                xy=(speed_time[max_idx], speeds[max_idx]),
                xytext=(speed_time[max_idx] + 20, speeds[max_idx] + 3),
                fontsize=9, color=COLORS["red"],
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1))

    # Mean speed
    mean_speed = np.mean(speeds[speeds > 0])
    ax.axhline(y=mean_speed, color=COLORS["green"], linestyle=":", linewidth=1,
               alpha=0.7, label=f"Mean: {mean_speed:.1f} km/h")

    # Cruise region (steady speed)
    # Find longest stretch where speed is within 2 km/h of a constant
    rolling_std = pd.Series(speeds).rolling(5, center=True).std().fillna(99)
    cruise_mask = rolling_std < 2
    if cruise_mask.any():
        # Find first cruise section
        cruise_runs = []
        start = None
        for i, is_cruise in enumerate(cruise_mask):
            if is_cruise and start is None:
                start = i
            elif not is_cruise and start is not None:
                if i - start >= 5:
                    cruise_runs.append((start, i))
                start = None
        if cruise_runs:
            s, e = cruise_runs[0]
            ax.axvspan(speed_time[s], speed_time[min(e, len(speed_time)-1)],
                       alpha=0.1, color=COLORS["green"])
            mid = speed_time[(s + min(e, len(speed_time)-1)) // 2]
            ax.text(mid, 2, "Cruise", ha="center", fontsize=8, color=COLORS["green"])

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Example Trip Speed Profile (Seoul, TUB mode, "
                 f"{row['gps_points']} GPS points)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, speed_time[-1])
    ax.set_ylim(0, max(speeds) + 5)

    plt.tight_layout()
    save_figure(fig, "fig3_speed_profile_example")


# ---------------------------------------------------------------------------
# Fig 5: Speeding prevalence heatmap (hour x day_of_week)
# ---------------------------------------------------------------------------

def fig_speeding_heatmap() -> None:
    """Create speeding prevalence heatmap (hour x day_of_week)."""
    print("\nFig 5: Speeding prevalence heatmap...")

    con = duckdb.connect()
    heatmap_data = con.execute(f"""
        SELECT start_hour, day_of_week,
               COUNT(*) as n_trips,
               AVG(CASE WHEN has_speeding THEN 1 ELSE 0 END) as speeding_rate
        FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
        WHERE is_valid = true
        GROUP BY start_hour, day_of_week
        ORDER BY start_hour, day_of_week
    """).fetchdf()
    con.close()

    # Create matrix
    matrix = np.zeros((24, 7))
    count_matrix = np.zeros((24, 7))
    for _, row in heatmap_data.iterrows():
        h = int(row["start_hour"])
        d = int(row["day_of_week"])
        if 0 <= h < 24 and 0 <= d < 7:
            matrix[h, d] = row["speeding_rate"] * 100
            count_matrix[h, d] = row["n_trips"]

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=matrix.max())
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_NAMES)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    ax.set_title("Speeding Prevalence (% of trips exceeding 25 km/h)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Speeding Rate (%)")

    # Annotate cells with values
    for h in range(24):
        for d in range(7):
            val = matrix[h, d]
            count = count_matrix[h, d]
            if count > 100:  # Only annotate cells with enough data
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(d, h, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.tight_layout()
    save_figure(fig, "fig5_speeding_heatmap")


# ---------------------------------------------------------------------------
# Fig 4: Speed distributions by mode
# ---------------------------------------------------------------------------

def fig_speed_by_mode() -> None:
    """Create speed distribution by mode (violin/box plots)."""
    print("\nFig 4: Speed distributions by mode...")

    con = duckdb.connect()
    # Sample to keep manageable for plotting
    df = con.execute(f"""
        SELECT mean_speed, max_speed_from_profile, mode_clean, province
        FROM read_parquet('{MODELING_DIR}/trip_modeling.parquet')
        WHERE mode_clean IN ('TUB', 'STD', 'ECO')
        USING SAMPLE 50000 ROWS
    """).fetchdf()
    con.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mode_order = ["ECO", "STD", "TUB"]
    mode_colors = [COLORS["green"], COLORS["blue"], COLORS["red"]]

    # (a) Mean speed by mode
    ax = axes[0]
    data_groups = [df[df["mode_clean"] == m]["mean_speed"].dropna().values
                   for m in mode_order]
    bp = ax.boxplot(data_groups, labels=mode_order, patch_artist=True,
                    widths=0.6, showfliers=False)
    for patch, color in zip(bp["boxes"], mode_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=SPEED_LIMIT_KR, color=COLORS["red"], linestyle="--",
               linewidth=1, alpha=0.5)
    ax.set_ylabel("Mean Trip Speed (km/h)")
    ax.set_title("(a) Mean speed by mode")

    # (b) Max speed by mode
    ax = axes[1]
    data_groups = [df[df["mode_clean"] == m]["max_speed_from_profile"].dropna().values
                   for m in mode_order]
    bp = ax.boxplot(data_groups, labels=mode_order, patch_artist=True,
                    widths=0.6, showfliers=False)
    for patch, color in zip(bp["boxes"], mode_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=SPEED_LIMIT_KR, color=COLORS["red"], linestyle="--",
               linewidth=1, alpha=0.5, label=f"Limit ({SPEED_LIMIT_KR} km/h)")
    ax.set_ylabel("Max Trip Speed (km/h)")
    ax.set_title("(b) Max speed by mode")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_figure(fig, "fig4_speed_by_mode")


# ---------------------------------------------------------------------------
# Fig 11: Rider typology radar plots (improved)
# ---------------------------------------------------------------------------

def fig_rider_typology_radar() -> None:
    """Create radar chart for the 4 GMM rider types."""
    print("\nFig 11: Rider typology radar plots...")

    with open(MODELING_DIR / "gmm_results.json", "r") as f:
        gmm = json.load(f)

    CLASS_NAMES = {
        "0": "Stop-and-Go",
        "1": "Safe Rider",
        "2": "Moderate Risk",
        "3": "Habitual Speeder",
    }

    features = [
        "user_mean_speed",
        "user_mean_max_speed",
        "speeding_propensity",
        "user_mean_abs_accel",
        "user_mean_cruise_fraction",
        "user_mean_zero_fraction",
    ]
    feature_labels = [
        "Mean Speed",
        "Max Speed",
        "Speeding Prop.",
        "Acceleration",
        "Cruise Fraction",
        "Zero Fraction",
    ]

    # Get feature values for each class and normalize to [0, 1]
    values_dict = {}
    for cls_id in ["0", "1", "2", "3"]:
        vals = [gmm["class_profiles"][cls_id]["features"][f] for f in features]
        values_dict[cls_id] = vals

    # Normalize each feature across classes
    all_vals = np.array(list(values_dict.values()))
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    norm_dict = {}
    for cls_id, vals in values_dict.items():
        norm_dict[cls_id] = [(v - mi) / r for v, mi, r in zip(vals, mins, ranges)]

    # Radar chart
    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    class_colors = [COLORS["orange"], COLORS["blue"], COLORS["skyblue"], COLORS["red"]]
    class_names_list = [CLASS_NAMES[str(i)] for i in range(4)]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    for cls_idx, cls_id in enumerate(["0", "1", "2", "3"]):
        vals = norm_dict[cls_id] + norm_dict[cls_id][:1]
        ax.plot(angles, vals, color=class_colors[cls_idx], linewidth=2,
                label=f"{class_names_list[cls_idx]} ({gmm['class_profiles'][cls_id]['pct_users']:.0f}%)")
        ax.fill(angles, vals, color=class_colors[cls_idx], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("GMM Rider Typology (K=4)", fontsize=14, fontweight="bold",
                 pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0), fontsize=10)

    plt.tight_layout()
    save_figure(fig, "fig11_rider_typology_radar")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all publication figures."""
    print("=" * 60)
    print("Phase 6: Publication Figures")
    print("=" * 60)

    fig_temporal_distributions()
    fig_speed_profile_example()
    fig_speed_by_mode()
    fig_speeding_heatmap()
    fig_rider_typology_radar()

    print("\nAll figures created.")


if __name__ == "__main__":
    main()
