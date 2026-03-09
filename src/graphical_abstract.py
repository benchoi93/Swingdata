"""
Graphical abstract for TR-C paper:
'Characterizing E-Scooter Riding Safety Through City-Scale Speed Profile Analysis'

Creates a single-page visual summary of the paper's data, methods, and key findings.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Paths
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color palette (colorblind-friendly)
COLORS = {
    "data": "#4477AA",       # blue
    "method": "#228833",     # green
    "finding": "#EE6677",    # red/coral
    "highlight": "#CCBB44",  # yellow
    "bg": "#F5F5F5",
    "text": "#333333",
    "white": "#FFFFFF",
    "light_blue": "#C6DBEF",
    "light_green": "#C7E9C0",
    "light_red": "#FDD0A2",
    "tub": "#E74C3C",
    "eco": "#27AE60",
    "std": "#3498DB",
}


def add_rounded_box(ax, x, y, w, h, text, color, fontsize=9, text_color="white",
                    alpha=1.0, bold=False):
    """Add a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=ax.transAxes, zorder=2
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=fontsize, color=text_color, weight=weight, zorder=3,
        wrap=True
    )


def add_arrow(ax, x1, y1, x2, y2, color="#666666"):
    """Add a curved arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_width=4,head_length=4",
        connectionstyle="arc3,rad=0",
        color=color, linewidth=1.5,
        transform=ax.transAxes, zorder=1
    )
    ax.add_patch(arrow)


def create_graphical_abstract():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["white"])

    # Title
    ax.text(
        0.50, 0.97,
        "Characterizing E-Scooter Riding Safety Through\nCity-Scale Speed Profile Analysis",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=14, weight="bold", color=COLORS["text"]
    )

    # --- ROW 1: DATA (top) ---
    row1_y = 0.82
    bh = 0.08

    ax.text(0.05, row1_y + bh / 2, "DATA", transform=ax.transAxes,
            fontsize=11, weight="bold", color=COLORS["data"], va="center")

    add_rounded_box(ax, 0.13, row1_y, 0.25, bh,
                    "2.78M GPS Trajectories\n52 Cities | 382K Users\nMay 2023",
                    COLORS["data"], fontsize=9, bold=False)

    add_rounded_box(ax, 0.41, row1_y, 0.25, bh,
                    "Per-Trip Speed Profiles\n~10s Sensor Readings\nFull Within-Trip Vectors",
                    COLORS["data"], fontsize=9)

    add_rounded_box(ax, 0.69, row1_y, 0.25, bh,
                    "3 Operating Modes\nTUB (no governor)\nSTD / ECO (governed)",
                    COLORS["data"], fontsize=9)

    # --- Arrows from row 1 to row 2 ---
    add_arrow(ax, 0.255, row1_y, 0.255, row1_y - 0.04, COLORS["data"])
    add_arrow(ax, 0.535, row1_y, 0.535, row1_y - 0.04, COLORS["data"])
    add_arrow(ax, 0.815, row1_y, 0.815, row1_y - 0.04, COLORS["data"])

    # --- ROW 2: METHODS ---
    row2_y = 0.64
    bh2 = 0.12

    ax.text(0.02, row2_y + bh2 / 2, "METHODS", transform=ax.transAxes,
            fontsize=11, weight="bold", color=COLORS["method"], va="center")

    methods = [
        ("Speed Indicator\nFramework\n\n- Speeding rate\n- Speed CV\n- Harsh events", 0.13),
        ("Spatial Analysis\n\n- H3 hexagonal grid\n- Getis-Ord Gi*\n- Moran's I (LISA)", 0.35),
        ("Statistical Models\n\n- Logistic + GEE\n- Mixed-effects\n- Two-part hurdle", 0.57),
        ("Rider Typology\n\n- GMM (K=4)\n- Multinomial logit\n- Natural experiment", 0.79),
    ]
    bw2 = 0.19
    for text, x in methods:
        add_rounded_box(ax, x, row2_y, bw2, bh2, text, COLORS["method"],
                        fontsize=8, text_color="white")

    # --- Arrow from row 2 to row 3 ---
    ax.annotate(
        "", xy=(0.50, 0.50), xytext=(0.50, row2_y),
        arrowprops=dict(arrowstyle="->", color="#666666", lw=2),
        xycoords="axes fraction"
    )

    # --- ROW 3: KEY FINDINGS ---
    row3_y = 0.08
    row3_top = 0.48

    ax.text(0.50, row3_top + 0.02, "KEY FINDINGS", transform=ax.transAxes,
            fontsize=12, weight="bold", color=COLORS["finding"], ha="center")

    # Finding boxes
    findings_left = [
        ("29.6% of trips exceed\n25 km/h speed limit", COLORS["finding"]),
        ("60.9% of users\nnever speed", COLORS["std"]),
        ("TUB mode: OR = 121\nfor speeding", COLORS["tub"]),
    ]
    findings_right = [
        ("Speed governor reduces\nspeeding by 95%\n(7.4% -> 0.33%)", COLORS["eco"]),
        ("Spatial clustering\nMoran's I = 0.16-0.71\n(all p < 0.001)", COLORS["method"]),
        ("4 Rider Typologies\nSafe (40%) | Moderate (26%)\nHabitual (26%) | Stop-Go (8%)", COLORS["data"]),
    ]

    box_h = 0.10
    box_w = 0.38
    gap = 0.12

    for i, (text, color) in enumerate(findings_left):
        y = row3_top - 0.04 - i * gap
        add_rounded_box(ax, 0.06, y - box_h, box_w, box_h, text, color,
                        fontsize=9, bold=True, alpha=0.85)

    for i, (text, color) in enumerate(findings_right):
        y = row3_top - 0.04 - i * gap
        add_rounded_box(ax, 0.56, y - box_h, box_w, box_h, text, color,
                        fontsize=9, bold=True, alpha=0.85)

    # --- BOTTOM: Policy implications banner ---
    banner_y = 0.01
    banner_h = 0.06
    box = FancyBboxPatch(
        (0.05, banner_y), 0.90, banner_h,
        boxstyle="round,pad=0.01",
        facecolor=COLORS["highlight"], edgecolor="none", alpha=0.9,
        transform=ax.transAxes, zorder=2
    )
    ax.add_patch(box)
    ax.text(
        0.50, banner_y + banner_h / 2,
        "POLICY: Mandatory speed governors  |  Targeted geofencing  |  Graduated enforcement",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=10, weight="bold", color=COLORS["text"], zorder=3
    )

    # Save
    for fmt in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"graphical_abstract.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=COLORS["white"])
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    create_graphical_abstract()
