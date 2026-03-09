"""
Phase 6: Spatial hotspot publication figures (Tasks 4.4, 4.6, 6.6).

Creates:
  - Fig 6: Seoul spatial hotspot map (Gi* hotspots + LISA clusters)
  - Fig 7: Daejeon spatial hotspot map (secondary city)
  - Fig S1: Multi-city Moran's I comparison

Outputs: figures/{name}.pdf and figures/{name}.png at 300 DPI
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.collections import PatchCollection
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR, FIGURES_DIR, MODELING_DIR,
    FIG_DPI, SPEED_LIMIT_KR, H3_RESOLUTION,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SPATIAL_DIR = DATA_DIR / "spatial"

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

# Colorblind-friendly palette
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

# Hotspot color scheme
HOTSPOT_COLORS = {
    "Hot Spot (99%)": "#d7191c",
    "Hot Spot (95%)": "#fdae61",
    "Hot Spot (90%)": "#fee08b",
    "Not Significant": "#e0e0e0",
    "Cold Spot (90%)": "#d1ecf1",
    "Cold Spot (95%)": "#abd9e9",
    "Cold Spot (99%)": "#2c7bb6",
}

LISA_COLORS = {
    "HH": "#d7191c",    # High-High: red
    "HL": "#fdae61",    # High-Low: orange
    "LH": "#abd9e9",    # Low-High: light blue
    "LL": "#2c7bb6",    # Low-Low: blue
    "Not Significant": "#e0e0e0",
}


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure in PDF and PNG formats."""
    fig.savefig(FIGURES_DIR / f"{name}.pdf", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: figures/{name}.pdf/png")
    plt.close(fig)


def fig6_seoul_hotspot() -> None:
    """Fig 6: Seoul spatial hotspot analysis (4-panel).

    Panel A: Trip density heatmap
    Panel B: Speeding rate heatmap
    Panel C: Getis-Ord Gi* hotspot classification
    Panel D: LISA cluster map
    """
    print("Creating Fig 6: Seoul spatial hotspot map...")
    gdf = gpd.read_file(SPATIAL_DIR / "hex_seoul.gpkg")
    print(f"  {len(gdf)} hexes loaded")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ---- Panel A: Trip density ----
    ax = axes[0, 0]
    gdf.plot(
        column="trip_count",
        cmap="YlOrRd",
        scheme="quantiles",
        k=7,
        legend=False,
        ax=ax,
        edgecolor="white",
        linewidth=0.2,
        alpha=0.9,
    )
    sm = plt.cm.ScalarMappable(
        cmap="YlOrRd",
        norm=Normalize(vmin=gdf["trip_count"].min(), vmax=gdf["trip_count"].quantile(0.95)),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Trip Count")
    ax.set_title("(a) Trip Origin Density", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ---- Panel B: Speeding rate ----
    ax = axes[0, 1]
    gdf.plot(
        column="speeding_pct",
        cmap="RdYlGn_r",
        legend=False,
        ax=ax,
        edgecolor="white",
        linewidth=0.2,
        alpha=0.9,
    )
    sm2 = plt.cm.ScalarMappable(
        cmap="RdYlGn_r",
        norm=Normalize(vmin=gdf["speeding_pct"].min(), vmax=gdf["speeding_pct"].max()),
    )
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax, shrink=0.7, pad=0.02)
    cb2.set_label("Speeding Rate (%)")
    ax.set_title("(b) Speeding Prevalence (%)", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ---- Panel C: Gi* Hotspot ----
    ax = axes[1, 0]
    hotspot_order = [
        "Hot Spot (99%)", "Hot Spot (95%)", "Hot Spot (90%)",
        "Not Significant",
        "Cold Spot (90%)", "Cold Spot (95%)", "Cold Spot (99%)",
    ]
    for cls in hotspot_order:
        subset = gdf[gdf["hotspot_class"] == cls]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=HOTSPOT_COLORS[cls],
                edgecolor="white",
                linewidth=0.2,
                alpha=0.9,
            )
    # Legend
    legend_patches = [
        mpatches.Patch(color=HOTSPOT_COLORS[cls], label=cls)
        for cls in hotspot_order if cls in gdf["hotspot_class"].values
    ]
    ax.legend(
        handles=legend_patches, loc="lower left",
        fontsize=8, framealpha=0.9, title="Gi* Classification",
        title_fontsize=9,
    )
    ax.set_title("(c) Getis-Ord Gi* Hotspots", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ---- Panel D: LISA clusters ----
    ax = axes[1, 1]
    lisa_order = ["HH", "HL", "LH", "LL", "Not Significant"]
    lisa_labels = {
        "HH": "High-High (speeding cluster)",
        "HL": "High-Low (spatial outlier)",
        "LH": "Low-High (spatial outlier)",
        "LL": "Low-Low (safe cluster)",
        "Not Significant": "Not Significant",
    }
    for cls in lisa_order:
        subset = gdf[gdf["lisa_cluster"] == cls]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=LISA_COLORS[cls],
                edgecolor="white",
                linewidth=0.2,
                alpha=0.9,
            )
    legend_patches_lisa = [
        mpatches.Patch(color=LISA_COLORS[cls], label=lisa_labels[cls])
        for cls in lisa_order if cls in gdf["lisa_cluster"].values
    ]
    ax.legend(
        handles=legend_patches_lisa, loc="lower left",
        fontsize=8, framealpha=0.9, title="LISA Cluster (p<0.05)",
        title_fontsize=9,
    )
    ax.set_title("(d) Local Moran's I (LISA)", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle("Seoul: Spatial Analysis of E-Scooter Speeding (H3 Res. 8)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig6_seoul_hotspot")


def fig7_daejeon_hotspot() -> None:
    """Fig 7: Daejeon spatial hotspot analysis (4-panel)."""
    print("Creating Fig 7: Daejeon spatial hotspot map...")
    gdf = gpd.read_file(SPATIAL_DIR / "hex_daejeon.gpkg")
    print(f"  {len(gdf)} hexes loaded")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Trip density
    ax = axes[0, 0]
    gdf.plot(
        column="trip_count", cmap="YlOrRd",
        legend=False, ax=ax, edgecolor="white", linewidth=0.2, alpha=0.9,
    )
    sm = plt.cm.ScalarMappable(
        cmap="YlOrRd",
        norm=Normalize(vmin=gdf["trip_count"].min(), vmax=gdf["trip_count"].quantile(0.95)),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Trip Count")
    ax.set_title("(a) Trip Origin Density", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Panel B: Speeding rate
    ax = axes[0, 1]
    gdf.plot(
        column="speeding_pct", cmap="RdYlGn_r",
        legend=False, ax=ax, edgecolor="white", linewidth=0.2, alpha=0.9,
    )
    sm2 = plt.cm.ScalarMappable(
        cmap="RdYlGn_r",
        norm=Normalize(vmin=gdf["speeding_pct"].min(), vmax=gdf["speeding_pct"].max()),
    )
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax, shrink=0.7, pad=0.02)
    cb2.set_label("Speeding Rate (%)")
    ax.set_title("(b) Speeding Prevalence (%)", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Panel C: Gi* Hotspot
    ax = axes[1, 0]
    hotspot_order = [
        "Hot Spot (99%)", "Hot Spot (95%)", "Hot Spot (90%)",
        "Not Significant",
        "Cold Spot (90%)", "Cold Spot (95%)", "Cold Spot (99%)",
    ]
    for cls in hotspot_order:
        subset = gdf[gdf["hotspot_class"] == cls]
        if len(subset) > 0:
            subset.plot(ax=ax, color=HOTSPOT_COLORS[cls], edgecolor="white", linewidth=0.2, alpha=0.9)
    legend_patches = [
        mpatches.Patch(color=HOTSPOT_COLORS[cls], label=cls)
        for cls in hotspot_order if cls in gdf["hotspot_class"].values
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.9,
              title="Gi* Classification", title_fontsize=9)
    ax.set_title("(c) Getis-Ord Gi* Hotspots", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Panel D: LISA
    ax = axes[1, 1]
    lisa_order = ["HH", "HL", "LH", "LL", "Not Significant"]
    lisa_labels = {
        "HH": "High-High (speeding cluster)",
        "HL": "High-Low (spatial outlier)",
        "LH": "Low-High (spatial outlier)",
        "LL": "Low-Low (safe cluster)",
        "Not Significant": "Not Significant",
    }
    for cls in lisa_order:
        subset = gdf[gdf["lisa_cluster"] == cls]
        if len(subset) > 0:
            subset.plot(ax=ax, color=LISA_COLORS[cls], edgecolor="white", linewidth=0.2, alpha=0.9)
    legend_patches_lisa = [
        mpatches.Patch(color=LISA_COLORS[cls], label=lisa_labels[cls])
        for cls in lisa_order if cls in gdf["lisa_cluster"].values
    ]
    ax.legend(handles=legend_patches_lisa, loc="lower left", fontsize=8, framealpha=0.9,
              title="LISA Cluster (p<0.05)", title_fontsize=9)
    ax.set_title("(d) Local Moran's I (LISA)", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle("Daejeon: Spatial Analysis of E-Scooter Speeding (H3 Res. 8)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig7_daejeon_hotspot")


def fig_morans_comparison() -> None:
    """Supplementary: Multi-city Moran's I comparison bar chart."""
    print("Creating supplementary: Multi-city Moran's I comparison...")

    report_path = MODELING_DIR / "spatial_analysis_report.json"
    with open(report_path) as f:
        results = json.load(f)

    cities = []
    morans_i = []
    z_scores = []
    for city_name, data in results.items():
        if city_name == "infrastructure_overlay":
            continue
        if "global_morans_I" in data:
            cities.append(city_name)
            morans_i.append(data["global_morans_I"])
            z_scores.append(data["morans_z_score"])

    # Sort by Moran's I
    order = np.argsort(morans_i)[::-1]
    cities = [cities[i] for i in order]
    morans_i = [morans_i[i] for i in order]
    z_scores = [z_scores[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Moran's I
    bars = ax1.barh(cities, morans_i, color=COLORS["blue"], alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Global Moran's I")
    ax1.set_title("(a) Spatial Autocorrelation of Speeding Rate", fontweight="bold")
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    for i, (v, z) in enumerate(zip(morans_i, z_scores)):
        ax1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    # Panel B: Z-scores
    ax2.barh(cities, z_scores, color=COLORS["red"], alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Z-score")
    ax2.set_title("(b) Statistical Significance", fontweight="bold")
    ax2.axvline(1.96, color="gray", linestyle="--", linewidth=0.8, label="z=1.96 (p<0.05)")
    ax2.legend(fontsize=9)
    for i, z in enumerate(z_scores):
        ax2.text(z + 0.3, i, f"{z:.1f}", va="center", fontsize=9)

    fig.suptitle("Spatial Autocorrelation of Speeding Across Cities",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig_morans_comparison")


def fig_hotspot_infrastructure() -> None:
    """Seoul hotspot vs infrastructure comparison."""
    print("Creating figure: Hotspot infrastructure comparison...")

    gdf = gpd.read_file(SPATIAL_DIR / "hex_seoul.gpkg")

    # Group by hotspot classification
    hot = gdf[gdf["hotspot_class"].str.contains("Hot")]
    cold = gdf[gdf["hotspot_class"].str.contains("Cold")]
    ns = gdf[gdf["hotspot_class"] == "Not Significant"]

    categories = ["Hot Spots", "Not Significant", "Cold Spots"]
    groups = [hot, ns, cold]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Speeding rate distribution
    ax = axes[0]
    data = [g["speeding_pct"].values for g in groups]
    bp = ax.boxplot(data, labels=categories, patch_artist=True, widths=0.6)
    box_colors = [COLORS["red"], COLORS["black"], COLORS["blue"]]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Speeding Rate (%)")
    ax.set_title("(a) Speeding Rate Distribution", fontweight="bold")

    # Panel B: Infrastructure comparison
    ax = axes[1]
    metrics = ["frac_major_road_mean", "frac_cycling_infra_mean"]
    metric_labels = ["Fraction Major Road", "Fraction Cycling Infra"]
    x = np.arange(len(metrics))
    width = 0.25
    for i, (label, group, color) in enumerate(zip(categories, groups, box_colors)):
        vals = [group[m].mean() for m in metrics]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Mean Fraction")
    ax.set_title("(b) Road Infrastructure by Hotspot Type", fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: Road class distribution
    ax = axes[2]
    for i, (label, group, color) in enumerate(zip(categories, groups, box_colors)):
        vc = group["hex_dominant_road_class"].value_counts(normalize=True).head(5)
        ax.barh(
            [f"{rc}" for rc in vc.index],
            vc.values * 100,
            alpha=0.6,
            label=label,
            color=color,
        )
    ax.set_xlabel("Percentage of Hexagons (%)")
    ax.set_title("(c) Dominant Road Class Distribution", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Seoul: Infrastructure Characteristics by Hotspot Classification",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig_hotspot_infrastructure")


def fig_segment_speeding_map() -> None:
    """Task 4.6: Segment-level speeding map for Seoul using hex-level data.

    Shows zoomed-in views of selected Seoul corridors with speeding patterns.
    """
    print("Creating figure: Segment-level speeding maps...")

    gdf = gpd.read_file(SPATIAL_DIR / "hex_seoul.gpkg")

    # Define corridors (approximate bounding boxes of known Seoul areas)
    corridors = {
        "Gangnam": {"lat": (37.49, 37.52), "lon": (127.02, 127.07)},
        "Hongdae/Sinchon": {"lat": (37.545, 37.565), "lon": (126.915, 126.945)},
        "Yeouido": {"lat": (37.52, 37.54), "lon": (126.91, 126.94)},
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for idx, (name, bbox) in enumerate(corridors.items()):
        ax = axes[idx]
        # Filter hexes in corridor
        mask = (
            (gdf["hex_lat"] >= bbox["lat"][0]) & (gdf["hex_lat"] <= bbox["lat"][1]) &
            (gdf["hex_lon"] >= bbox["lon"][0]) & (gdf["hex_lon"] <= bbox["lon"][1])
        )
        corridor_gdf = gdf[mask]

        if len(corridor_gdf) == 0:
            ax.text(0.5, 0.5, f"No data for\n{name}", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(f"({chr(97+idx)}) {name}", fontweight="bold")
            continue

        corridor_gdf.plot(
            column="speeding_pct",
            cmap="RdYlGn_r",
            legend=False,
            ax=ax,
            edgecolor="gray",
            linewidth=0.3,
            alpha=0.9,
        )
        # Add trip count labels for each hex
        for _, row in corridor_gdf.iterrows():
            if row["trip_count"] > 50:
                ax.annotate(
                    f"{row['speeding_pct']:.0f}%",
                    xy=(row["hex_lon"], row["hex_lat"]),
                    fontsize=6,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax.set_title(f"({chr(97+idx)}) {name} (n={len(corridor_gdf)} hexes)", fontweight="bold")
        ax.set_xlabel("Longitude")
        if idx == 0:
            ax.set_ylabel("Latitude")

    # Add shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn_r",
        norm=Normalize(vmin=gdf["speeding_pct"].min(), vmax=gdf["speeding_pct"].max()),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02, aspect=30)
    cb.set_label("Speeding Rate (%)")

    fig.suptitle("Seoul Corridor-Level Speeding Analysis (H3 Res. 8)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.94])
    save_fig(fig, "fig_corridor_speeding")


def main() -> None:
    """Generate all spatial figures."""
    fig6_seoul_hotspot()
    fig7_daejeon_hotspot()
    fig_morans_comparison()
    fig_hotspot_infrastructure()
    fig_segment_speeding_map()
    print("\nAll spatial figures created successfully.")


if __name__ == "__main__":
    main()
