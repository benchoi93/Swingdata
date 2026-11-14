"""
Task 1.5: Distribution figures — violin/box plots by mode for each outcome.

Output: figures/v2/fig_supp_distributions.pdf
"""

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

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, FIG_DPI, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
FIG_DIR = FIGURES_DIR / "v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TRIP_PATH = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

MODE_COLORS = {"TUB": "#e74c3c", "STD": "#3498db", "ECO": "#2ecc71"}
MODE_ORDER = ["TUB", "STD", "ECO"]

OUTCOMES = [
    ("harsh_accel_count", "Harsh Acceleration\n(count/trip)", (0, 5)),
    ("harsh_decel_count", "Harsh Deceleration\n(count/trip)", (0, 5)),
    ("speed_cv", "Speed CV", (0, 2)),
    ("cruise_fraction", "Cruise Fraction", (0, 1)),
    ("zero_speed_fraction", "Zero-Speed Fraction", (0, 1)),
    ("mean_speed", "Mean Speed (km/h)", (0, 30)),
    ("distance", "Trip Distance (m)", (0, 3000)),
    ("travel_time", "Travel Time (s)", (0, 1200)),
]


def main():
    print("=" * 60)
    print("Task 1.5: Distribution Figures")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()

    # Sample 100K per mode for violin plots
    cols = ", ".join(c for c, _, _ in OUTCOMES) + ", mode"
    print("  Sampling 50K per mode...")
    dfs = []
    for mode in MODE_ORDER:
        mdf = con.execute(f"""
            SELECT {cols}
            FROM read_parquet('{TRIP_PATH}')
            WHERE mode = '{mode}'
            USING SAMPLE 50000 ROWS (reservoir)
        """).fetchdf()
        dfs.append(mdf)
    df = pd.concat(dfs, ignore_index=True)
    con.close()

    print(f"  Total sampled: {len(df):,}")

    # Create multi-panel violin plot
    n_outcomes = len(OUTCOMES)
    n_cols = 4
    n_rows = (n_outcomes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, (col, label, ylim) in enumerate(OUTCOMES):
        ax = axes[i]

        data_by_mode = [df.loc[df["mode"] == m, col].clip(ylim[0], ylim[1]).values
                        for m in MODE_ORDER]

        parts = ax.violinplot(data_by_mode, positions=[1, 2, 3], showmeans=True,
                              showmedians=True, showextrema=False)

        for j, (pc, mode) in enumerate(zip(parts["bodies"], MODE_ORDER)):
            pc.set_facecolor(MODE_COLORS[mode])
            pc.set_alpha(0.6)

        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(MODE_ORDER, fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(ylim)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for i in range(n_outcomes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Outcome Distributions by Operating Mode", fontsize=14, y=1.01)
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig_supp_distributions.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t_start
    print(f"\n  Figure saved: {FIG_DIR / 'fig_supp_distributions.pdf'}")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
