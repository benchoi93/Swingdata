"""
Phase 2 continued: Event study + Demand response figure.

Task 2.3: Event study for each DiD outcome (monthly coefficients).
Task 2.4: Demand response figure (trip counts + active users by treatment intensity).

Uses city_month_panel_v2.parquet from Task 2.1.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, FIG_DPI, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
FIG_DIR = FIGURES_DIR / "v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DID_OUTCOMES = [
    "mean_speed", "harsh_accel_count", "harsh_decel_count",
    "speed_cv", "cruise_fraction", "zero_speed_fraction",
]

OUTCOME_LABELS = {
    "mean_speed": "Mean Speed (km/h)",
    "harsh_accel_count": "Harsh Accel (count/trip)",
    "harsh_decel_count": "Harsh Decel (count/trip)",
    "speed_cv": "Speed CV",
    "cruise_fraction": "Cruise Fraction",
    "zero_speed_fraction": "Zero-Speed Fraction",
}

MONTHS_ORDER = [f"2023-{m:02d}" for m in range(2, 13)]


def task_2_3(panel: pd.DataFrame) -> dict:
    """Event study: monthly treatment-intensity interactions."""
    print("=" * 60)
    print("Task 2.3: Multi-Outcome Event Study")
    print("=" * 60)

    # Reference month: Nov 2023 (last pre-treatment month)
    ref_month = "2023-11"
    months = [m for m in MONTHS_ORDER if m != ref_month]

    # Create month x treatment interactions
    for m in months:
        panel[f"treat_m_{m}"] = (panel["month_year"] == m).astype(float) * panel["tub_share_nov"]

    treat_cols = [f"treat_m_{m}" for m in months]

    city_dummies = pd.get_dummies(panel["city"], prefix="city", dtype=float, drop_first=True)
    month_dummies = pd.get_dummies(panel["month_year"], prefix="month", dtype=float, drop_first=True)

    results = {}

    for outcome in DID_OUTCOMES:
        y = panel[outcome].values
        X = np.column_stack([
            panel[treat_cols].values,
            city_dummies.values,
            month_dummies.values,
        ])
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": panel["city"]})

        coefs = []
        for i, m in enumerate(months):
            beta = model.params[1 + i]
            se = model.bse[1 + i]
            p = model.pvalues[1 + i]
            ci_lo, ci_hi = model.conf_int()[1 + i]
            coefs.append({
                "month": m,
                "beta": round(float(beta), 4),
                "se": round(float(se), 4),
                "p": float(p),
                "ci_lo": round(float(ci_lo), 4),
                "ci_hi": round(float(ci_hi), 4),
            })

        # Add reference month (zero by definition)
        coefs.append({
            "month": ref_month, "beta": 0.0, "se": 0.0,
            "p": 1.0, "ci_lo": 0.0, "ci_hi": 0.0,
        })
        coefs.sort(key=lambda x: x["month"])

        results[outcome] = coefs
        print(f"  {outcome}: {len(coefs)} monthly coefficients")

    # Clean up temp columns
    panel.drop(columns=treat_cols, inplace=True, errors="ignore")

    # Make multi-panel event study figure
    _make_event_study_figure(results)

    return results


def _make_event_study_figure(results: dict):
    """Multi-panel event study figure (Appendix)."""
    n = len(results)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for i, (outcome, coefs) in enumerate(results.items()):
        ax = axes[i]
        df = pd.DataFrame(coefs)
        x_pos = range(len(df))

        # Color: pre-treatment grey, reference green, post-treatment colored
        colors = []
        for _, row in df.iterrows():
            if row["month"] == "2023-11":
                colors.append("#2ecc71")  # reference
            elif row["month"] == "2023-12":
                colors.append("#e74c3c" if row["p"] < 0.05 else "#f39c12")
            else:
                colors.append("#3498db" if row["p"] < 0.05 else "#95a5a6")

        ax.errorbar(list(x_pos), df["beta"], yerr=[df["beta"] - df["ci_lo"], df["ci_hi"] - df["beta"]],
                     fmt="o", color="black", markersize=5, capsize=3, linewidth=1, zorder=5)

        for j, c in enumerate(colors):
            ax.plot(j, df.iloc[j]["beta"], "o", color=c, markersize=7, zorder=6)

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(x=len(df) - 2 + 0.5, color="red", linewidth=1, linestyle=":", alpha=0.7,
                   label="TUB ban")

        month_labels = [m[-2:] for m in df["month"]]
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_xlabel("Month (2023)", fontsize=9)
        ax.set_ylabel("Coefficient", fontsize=9)
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome), fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Event Study: Monthly Treatment-Intensity Interactions\n(Reference: Nov 2023)",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig_supp_event_study.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Event study figure saved: {FIG_DIR / 'fig_supp_event_study.pdf'}")


def task_2_4(panel: pd.DataFrame):
    """Demand response figure: trip counts + active users by treatment intensity."""
    print("\n" + "=" * 60)
    print("Task 2.4: Demand Response Figure")
    print("=" * 60)

    # Split cities into TUB-share quartiles (based on Nov 2023)
    city_tub = panel[panel["month_year"] == "2023-11"][["city", "tub_share_nov"]].drop_duplicates()
    city_tub["quartile"] = pd.qcut(city_tub["tub_share_nov"], 4,
                                    labels=["Q1 (Low TUB)", "Q2", "Q3", "Q4 (High TUB)"])
    panel = panel.merge(city_tub[["city", "quartile"]], on="city", how="left")
    panel = panel.dropna(subset=["quartile"])

    # Aggregate by quartile x month
    demand = panel.groupby(["quartile", "month_year"]).agg(
        trip_count=("trip_count", "mean"),
        active_users=("active_users", "mean"),
    ).reset_index()

    # Normalize to Feb 2023 baseline
    for var in ["trip_count", "active_users"]:
        base = demand[demand["month_year"] == "2023-02"].set_index("quartile")[var]
        demand[f"{var}_idx"] = demand.apply(
            lambda r: r[var] / base.get(r["quartile"], 1) * 100, axis=1
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Q1 (Low TUB)": "#3498db", "Q2": "#2ecc71", "Q3": "#f39c12", "Q4 (High TUB)": "#e74c3c"}
    quartile_order = ["Q1 (Low TUB)", "Q2", "Q3", "Q4 (High TUB)"]

    for ax, (var, label) in zip(axes, [("trip_count_idx", "Trip Count"),
                                        ("active_users_idx", "Active Users")]):
        for q in quartile_order:
            qdf = demand[demand["quartile"] == q].sort_values("month_year")
            month_nums = [MONTHS_ORDER.index(m) for m in qdf["month_year"]]
            ax.plot(month_nums, qdf[var], "o-", color=colors[q], label=q,
                    markersize=5, linewidth=1.5)

        ax.axvline(x=MONTHS_ORDER.index("2023-12") - 0.5, color="red",
                   linewidth=1, linestyle=":", alpha=0.7)
        ax.axhline(y=100, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

        ax.set_xticks(range(len(MONTHS_ORDER)))
        ax.set_xticklabels([m[-2:] for m in MONTHS_ORDER], fontsize=9)
        ax.set_xlabel("Month (2023)", fontsize=10)
        ax.set_ylabel(f"{label} (Index, Feb=100)", fontsize=10)
        ax.set_title(f"{label} by Treatment Intensity", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("Demand Response to TUB Ban by City Treatment Intensity",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig5_demand_response.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # Also compute formal DiD on normalized demand
    for var in ["trip_count", "active_users"]:
        pre_mean = panel[panel["post"] == 0].groupby("quartile")[var].mean()
        post_mean = panel[panel["post"] == 1].groupby("quartile")[var].mean()
        for q in quartile_order:
            pre = pre_mean.get(q, 0)
            post = post_mean.get(q, 0)
            pct = (post - pre) / pre * 100 if pre > 0 else 0
            print(f"  {var} {q}: pre={pre:.0f} post={post:.0f} change={pct:+.1f}%")

    print(f"\n  Demand response figure saved: {FIG_DIR / 'fig5_demand_response.pdf'}")


def main():
    t_start = time.time()

    # Load panel from Task 2.1
    panel_path = V2_DIR / "city_month_panel_v2.parquet"
    panel = pd.read_parquet(panel_path)
    print(f"  Loaded panel: {len(panel)} obs from {panel_path}")

    # Task 2.3
    event_results = task_2_3(panel)

    # Task 2.4
    task_2_4(panel)

    elapsed = time.time() - t_start

    # Save event study results
    result_path = V2_DIR / "phase2_event_study.json"
    with open(result_path, "w") as f:
        json.dump({"event_study": event_results, "time_s": round(elapsed, 1)}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Tasks 2.3-2.4 complete in {elapsed:.0f}s")
    print(f"  Results: {result_path}")


if __name__ == "__main__":
    main()
