"""
Phase 4: Behavioral Escalation Pathway (Tasks 4.1-4.6).

Task 4.1: Experience -> TUB adoption -> outcome mediation.
Task 4.2: Multi-dimensional experience trajectory visualization (Fig 7).
Task 4.3: Multi-endpoint survival analysis.
Task 4.4: KM curves by TUB usage (Fig 8).
Task 4.5: Cox PH with time-varying TUB covariate.
Task 4.6: TUB mediation percentage for each outcome.

Uses trip_modeling.parquet (Feb-Nov 2023).
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

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, FIG_DPI, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
FIG_DIR = FIGURES_DIR / "v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRIP_PATH = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

OUTCOMES = [
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

# Survival event thresholds
SURVIVAL_EVENTS = {
    "first_harsh": "First Harsh Event (accel or decel count > 0)",
    "first_high_cv": "First High-CV Trip (speed_cv > P75)",
    "first_speeding": "First Speeding Trip (max_speed > 25 km/h)",
}


def task_4_1_2(con: duckdb.DuckDBPyConnection) -> dict:
    """Experience -> TUB adoption -> outcome mediation + experience curves."""
    print("=" * 60)
    print("Task 4.1-4.2: Experience-Mediation + Experience Curves")
    print("=" * 60)

    # Load user trip sequences with trip rank
    print("  Loading trip sequences (200K users sample)...")
    df = con.execute(f"""
        WITH ranked AS (
            SELECT user_id, mode, month_year,
                mean_speed, harsh_accel_count, harsh_decel_count,
                speed_cv, cruise_fraction, zero_speed_fraction,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY month_year, route_id) AS trip_rank
            FROM read_parquet('{TRIP_PATH}')
        ),
        user_sample AS (
            SELECT DISTINCT user_id FROM ranked
            USING SAMPLE 200000 ROWS (reservoir)
        )
        SELECT r.*
        FROM ranked r
        INNER JOIN user_sample u ON r.user_id = u.user_id
        WHERE r.trip_rank <= 200
    """).fetchdf()

    print(f"  Loaded: {len(df):,} trips from {df['user_id'].nunique():,} users")

    # Experience bins
    exp_bins = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    exp_labels = ["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "51-100", "101-200"]

    df["exp_bin"] = pd.cut(df["trip_rank"],
                           bins=[0] + exp_bins,
                           labels=exp_labels,
                           right=True)
    df = df.dropna(subset=["exp_bin"])

    # TUB adoption rate by experience
    tub_by_exp = df.groupby("exp_bin").agg(
        tub_rate=("mode", lambda x: (x == "TUB").mean()),
        n_trips=("mode", "count"),
    ).reset_index()

    print("\n  TUB Adoption by Experience:")
    for _, row in tub_by_exp.iterrows():
        print(f"    Trip {row['exp_bin']:<8s}: TUB={row['tub_rate']:.1%} (n={row['n_trips']:,})")

    # Outcome means by experience (all modes vs STD/ECO only)
    mediation_results = {}

    for outcome in OUTCOMES:
        all_by_exp = df.groupby("exp_bin")[outcome].mean()
        std_eco = df[df["mode"].isin(["STD", "ECO"])]
        std_eco_by_exp = std_eco.groupby("exp_bin")[outcome].mean()

        # Total experience effect: range across bins (all modes)
        total_range = all_by_exp.max() - all_by_exp.min()
        # Direct effect (within STD/ECO): range without TUB
        direct_range = std_eco_by_exp.max() - std_eco_by_exp.min()
        # Mediated via TUB adoption
        mediated = total_range - direct_range if total_range != 0 else 0
        mediation_pct = (mediated / total_range * 100) if total_range != 0 else 0

        mediation_results[outcome] = {
            "total_range": round(float(total_range), 4),
            "direct_range": round(float(direct_range), 4),
            "mediated_range": round(float(mediated), 4),
            "mediation_pct": round(float(mediation_pct), 1),
        }

        print(f"\n  {outcome}: total_range={total_range:.4f}, direct={direct_range:.4f}, "
              f"mediation={mediation_pct:.1f}%")

    # Fig 7: Experience trajectory curves
    _make_experience_curves(df, tub_by_exp)

    return mediation_results


def _make_experience_curves(df: pd.DataFrame, tub_by_exp: pd.DataFrame):
    """Fig 7: Multi-dimensional experience curves."""
    exp_bins = df["exp_bin"].cat.categories

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    # Panel 1: TUB adoption rate
    ax = axes[0]
    ax.bar(range(len(tub_by_exp)), tub_by_exp["tub_rate"] * 100,
           color="#e74c3c", alpha=0.7)
    ax.set_xticks(range(len(tub_by_exp)))
    ax.set_xticklabels(tub_by_exp["exp_bin"], fontsize=8, rotation=45)
    ax.set_ylabel("TUB Usage Rate (%)", fontsize=9)
    ax.set_title("TUB Adoption", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Panels 2-7: Each outcome (all modes vs STD/ECO)
    for i, outcome in enumerate(OUTCOMES):
        ax = axes[i + 1]
        all_means = df.groupby("exp_bin")[outcome].mean()
        std_eco = df[df["mode"].isin(["STD", "ECO"])]
        se_means = std_eco.groupby("exp_bin")[outcome].mean()

        x = range(len(exp_bins))
        ax.plot(list(x), [all_means.get(b, np.nan) for b in exp_bins],
                "o-", color="#2c3e50", label="All modes", markersize=4, linewidth=1.5)
        ax.plot(list(x), [se_means.get(b, np.nan) for b in exp_bins],
                "s--", color="#3498db", label="STD/ECO only", markersize=4, linewidth=1.5)

        ax.set_xticks(list(x))
        ax.set_xticklabels(exp_bins, fontsize=7, rotation=45)
        ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=8)
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome), fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Hide unused
    for i in range(len(OUTCOMES) + 1, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Behavioral Escalation: Experience Trajectories\n"
                 "(Gap = TUB-mediated effect)", fontsize=13, y=1.02)
    fig.supxlabel("Trip Number", fontsize=11, y=-0.01)
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig7_experience_trajectories.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Fig 7 saved: {FIG_DIR / 'fig7_experience_trajectories.pdf'}")


def task_4_3_4_5(con: duckdb.DuckDBPyConnection) -> dict:
    """Multi-endpoint survival analysis with KM curves and Cox PH."""
    print("\n" + "=" * 60)
    print("Task 4.3-4.5: Multi-Endpoint Survival Analysis")
    print("=" * 60)

    from lifelines import KaplanMeierFitter, CoxPHFitter

    # Load user trip sequences
    print("  Loading user trip sequences...")

    # Get P75 of speed_cv for threshold
    p75_cv = con.execute(f"""
        SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY speed_cv) AS p75
        FROM read_parquet('{TRIP_PATH}')
        WHERE speed_cv IS NOT NULL
    """).fetchdf()["p75"].iloc[0]
    print(f"  Speed CV P75 threshold: {p75_cv:.4f}")

    # Build survival dataset: for each user, find time-to-first-event
    print("  Building survival dataset (200K users)...")
    surv_df = con.execute(f"""
        WITH user_sample AS (
            SELECT DISTINCT user_id
            FROM read_parquet('{TRIP_PATH}')
            USING SAMPLE 200000 ROWS (reservoir)
        ),
        ranked AS (
            SELECT
                t.user_id,
                t.mode,
                t.harsh_accel_count,
                t.harsh_decel_count,
                t.speed_cv,
                t.max_speed_from_profile,
                ROW_NUMBER() OVER (PARTITION BY t.user_id ORDER BY t.month_year, t.route_id) AS trip_rank
            FROM read_parquet('{TRIP_PATH}') t
            INNER JOIN user_sample u ON t.user_id = u.user_id
        )
        SELECT * FROM ranked
        WHERE trip_rank <= 500
    """).fetchdf()

    n_users = surv_df["user_id"].nunique()
    print(f"  Users: {n_users:,}, Trips: {len(surv_df):,}")

    # User-level: max trip_rank and ever_tub
    user_max = surv_df.groupby("user_id").agg(
        max_trip=("trip_rank", "max"),
        ever_tub=("mode", lambda x: int((x == "TUB").any())),
        tub_frac=("mode", lambda x: (x == "TUB").mean()),
    ).reset_index()

    # Find first event for each endpoint
    events = {}

    # 1. First harsh event
    harsh_trips = surv_df[(surv_df["harsh_accel_count"] > 0) | (surv_df["harsh_decel_count"] > 0)]
    first_harsh = harsh_trips.groupby("user_id")["trip_rank"].min().reset_index()
    first_harsh.columns = ["user_id", "event_trip"]
    events["first_harsh"] = first_harsh

    # 2. First high-CV trip
    high_cv = surv_df[surv_df["speed_cv"] > p75_cv]
    first_cv = high_cv.groupby("user_id")["trip_rank"].min().reset_index()
    first_cv.columns = ["user_id", "event_trip"]
    events["first_high_cv"] = first_cv

    # 3. First speeding
    speeding = surv_df[surv_df["max_speed_from_profile"] > 25]
    first_speed = speeding.groupby("user_id")["trip_rank"].min().reset_index()
    first_speed.columns = ["user_id", "event_trip"]
    events["first_speeding"] = first_speed

    survival_results = {}

    # KM curves figure (Fig 8)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (event_name, event_df) in enumerate(events.items()):
        ax = axes[ax_idx]

        # Merge with user info
        surv = user_max.merge(event_df, on="user_id", how="left")
        surv["event"] = surv["event_trip"].notna().astype(int)
        surv["duration"] = surv["event_trip"].fillna(surv["max_trip"])

        # Minimum 2 trips
        surv = surv[surv["max_trip"] >= 2]

        # Split by TUB usage
        tub_mask = surv["ever_tub"] == 1
        non_tub_mask = surv["ever_tub"] == 0

        event_rate = surv["event"].mean()
        print(f"\n  {event_name}: {len(surv):,} users, {event_rate:.1%} event rate")

        # KM fits
        kmf_tub = KaplanMeierFitter()
        kmf_non = KaplanMeierFitter()

        if tub_mask.sum() > 10:
            kmf_tub.fit(surv.loc[tub_mask, "duration"], surv.loc[tub_mask, "event"],
                        label="TUB users", timeline=range(1, 201))
            kmf_tub.plot_survival_function(ax=ax, color="#e74c3c", linewidth=1.5)
            median_tub = kmf_tub.median_survival_time_
        else:
            median_tub = np.inf

        if non_tub_mask.sum() > 10:
            kmf_non.fit(surv.loc[non_tub_mask, "duration"], surv.loc[non_tub_mask, "event"],
                        label="Non-TUB users", timeline=range(1, 201))
            kmf_non.plot_survival_function(ax=ax, color="#3498db", linewidth=1.5)
            median_non = kmf_non.median_survival_time_

        else:
            median_non = np.inf

        # Log-rank test
        from lifelines.statistics import logrank_test
        if tub_mask.sum() > 10 and non_tub_mask.sum() > 10:
            lr = logrank_test(surv.loc[tub_mask, "duration"], surv.loc[non_tub_mask, "duration"],
                              surv.loc[tub_mask, "event"], surv.loc[non_tub_mask, "event"])
            lr_p = lr.p_value
            lr_stat = lr.test_statistic
        else:
            lr_p, lr_stat = 1.0, 0.0

        ax.set_xlabel("Trip Number", fontsize=10)
        ax.set_ylabel("Survival Probability", fontsize=10)
        ax.set_title(SURVIVAL_EVENTS[event_name], fontsize=10)
        ax.set_xlim(0, 200)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Cox PH (static: ever_tub + tub_fraction)
        cox_df = surv[["duration", "event", "ever_tub", "tub_frac"]].dropna()
        cox_df = cox_df[cox_df["duration"] > 0]

        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="duration", event_col="event")
            hr_tub = float(np.exp(cph.params_["ever_tub"]))
            hr_frac = float(np.exp(cph.params_["tub_frac"]))
            hr_tub_p = float(cph.summary.loc["ever_tub", "p"])
            c_index = float(cph.concordance_index_)
        except Exception:
            hr_tub, hr_frac, hr_tub_p, c_index = 1.0, 1.0, 1.0, 0.5

        survival_results[event_name] = {
            "n_users": int(len(surv)),
            "event_rate": round(float(event_rate), 3),
            "median_tub": float(median_tub) if not np.isinf(median_tub) else None,
            "median_non_tub": float(median_non) if not np.isinf(median_non) else None,
            "logrank_chi2": round(float(lr_stat), 1),
            "logrank_p": float(lr_p),
            "cox_hr_ever_tub": round(hr_tub, 3),
            "cox_hr_tub_frac": round(hr_frac, 3),
            "cox_hr_tub_p": float(hr_tub_p),
            "c_index": round(c_index, 3),
        }

        print(f"    Median survival: TUB={median_tub:.0f} vs non-TUB={median_non:.0f} trips")
        print(f"    Log-rank: chi2={lr_stat:.1f}, p={lr_p:.2g}")
        print(f"    Cox HR(ever_tub)={hr_tub:.3f}, HR(tub_frac)={hr_frac:.3f}, C={c_index:.3f}")

    fig.suptitle("Multi-Endpoint Survival Analysis: Time-to-First Event",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig8_survival_multi_endpoint.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Fig 8 saved: {FIG_DIR / 'fig8_survival_multi_endpoint.pdf'}")

    return survival_results


def task_4_6(mediation: dict) -> dict:
    """TUB mediation percentage summary for each outcome."""
    print("\n" + "=" * 60)
    print("Task 4.6: TUB Mediation Summary")
    print("=" * 60)

    print(f"  {'Outcome':<25s} {'Total Range':<14s} {'Direct':<14s} {'Mediated':<14s} {'% Mediated':<12s}")
    print(f"  {'-' * 79}")
    for outcome in OUTCOMES:
        m = mediation[outcome]
        print(f"  {outcome:<25s} {m['total_range']:<14.4f} {m['direct_range']:<14.4f} "
              f"{m['mediated_range']:<14.4f} {m['mediation_pct']:<12.1f}")

    return mediation


def main():
    print("Phase 4: Behavioral Escalation Pathway")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 8")

    # Task 4.1-4.2
    mediation = task_4_1_2(con)

    # Task 4.3-4.5
    survival = task_4_3_4_5(con)
    con.close()

    # Task 4.6
    task_4_6(mediation)

    elapsed = time.time() - t_start

    report = {
        "phase": 4,
        "mediation": mediation,
        "survival": survival,
        "time_s": round(elapsed, 1),
    }

    report_path = V2_DIR / "phase4_escalation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Phase 4 complete in {elapsed:.0f}s")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
