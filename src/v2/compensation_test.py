"""
Phase 3: Compensation Test (Tasks 3.1-3.5).

Tests Peltzman / risk homeostasis: do riders compensate on unconstrained margins
when speed governance changes?

Task 3.1: Multi-outcome DiD restricted to STD/ECO trips only.
Task 3.2: Mode-switcher within-user DiD on all outcomes.
Task 3.3: Cohen's d effect sizes across all outcomes.
Task 3.4: Mode-switcher placebo (Oct 2023).
Task 3.5: Summary table.

Uses trip_modeling.parquet (Feb-Nov) + Dec 2023 raw CSV.
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
import statsmodels.api as sm
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, FIG_DPI, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
FIG_DIR = FIGURES_DIR / "v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path("D:/SwingData/raw")

OUTCOMES = [
    "mean_speed", "harsh_accel_count", "harsh_decel_count",
    "speed_cv", "cruise_fraction", "zero_speed_fraction",
]

OUTCOME_LABELS = {
    "mean_speed": "Mean Speed\n(km/h)",
    "harsh_accel_count": "Harsh Accel\n(count/trip)",
    "harsh_decel_count": "Harsh Decel\n(count/trip)",
    "speed_cv": "Speed CV",
    "cruise_fraction": "Cruise\nFraction",
    "zero_speed_fraction": "Zero-Speed\nFraction",
}

CITY_CENTERS = {
    "Seoul": (37.5665, 126.9780), "Busan": (35.1796, 129.0756),
    "Incheon": (37.4563, 126.7052), "Daegu": (35.8714, 128.6014),
    "Daejeon": (36.3504, 127.3845), "Gwangju": (35.1595, 126.8526),
    "Ulsan": (35.5384, 129.3114), "Sejong": (36.4800, 127.0000),
    "Suwon": (37.2636, 127.0286), "Seongnam": (37.4201, 127.1265),
    "Goyang": (37.6584, 126.8320), "Yongin": (37.2411, 127.1776),
    "Bucheon": (37.5034, 126.7660), "Ansan": (37.3219, 126.8309),
    "Anyang": (37.3943, 126.9568), "Namyangju": (37.6360, 127.2166),
    "Hwaseong": (37.1995, 126.8313), "Uijeongbu": (37.7381, 127.0338),
    "Siheung": (37.3800, 126.8028), "Gimpo": (37.6153, 126.7156),
    "Gwangmyeong": (37.4786, 126.8646), "Hanam": (37.5390, 127.2145),
    "Gunpo": (37.3614, 126.9352), "Pyeongtaek": (36.9922, 127.1129),
    "Osan": (37.1499, 127.0699), "Icheon": (37.2720, 127.4349),
    "Paju": (37.7600, 126.7800), "Yangju": (37.7852, 127.0460),
    "Guri": (37.5943, 127.1297), "Cheonan": (36.8152, 127.1139),
    "Asan": (36.7898, 127.0018), "Cheongju": (36.6424, 127.4890),
    "Jeonju": (35.8242, 127.1480), "Changwon": (35.2280, 128.6811),
    "Gimhae": (35.2285, 128.8894), "Jeju": (33.4996, 126.5312),
    "Chuncheon": (37.8813, 127.7300), "Wonju": (37.3422, 127.9202),
    "Pohang": (36.0190, 129.3435), "Gumi": (36.1195, 128.3445),
    "Gyeongju": (35.8562, 129.2247), "Geoje": (34.8806, 128.6211),
    "Yangsan": (35.3350, 129.0370), "Mokpo": (34.7937, 126.3886),
    "Yeosu": (34.7604, 127.6622), "Suncheon": (34.9506, 127.4873),
    "Gwangyang": (34.9407, 127.6959), "Iksan": (35.9483, 126.9577),
    "Gunsan": (35.9676, 126.7369), "Gimje": (35.8037, 126.8809),
    "Tongyeong": (34.8545, 128.4330), "Sacheon": (35.0037, 128.0647),
}

EXCLUDE_MODELS = ["I9"]
EXCLUDE_MODELS_SQL = ", ".join(f"'{m}'" for m in EXCLUDE_MODELS)
ACCEL_FACTOR = 1000.0 / 3600.0 / 10.0
HARSH_THRESH = 0.5


def _load_dec_trips(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load Dec 2023 trips from raw CSV with speed indicators."""
    dec_csv = str(RAW_DIR / "2023_12_Swing_Routes.csv").replace("\\", "/")

    dec = con.execute(f"""
        WITH raw_parsed AS (
            SELECT
                route_id, user_id, model,
                CASE
                    WHEN mode = 'SCOOTER_TUB' THEN 'TUB'
                    WHEN mode = 'SCOOTER_STD' THEN 'STD'
                    WHEN mode = 'SCOOTER_ECO' THEN 'ECO'
                    ELSE mode
                END AS mode,
                start_x AS start_lat, start_y AS start_lon,
                '2023-12' AS month_year,
                list_transform(
                    regexp_extract_all(CAST(speeds AS VARCHAR), '(\\d+(?:\\.\\d+)?)'),
                    x -> TRY_CAST(x AS DOUBLE)
                ) AS speed_arr
            FROM read_csv_auto('{dec_csv}', ignore_errors=true)
            WHERE model NOT IN ({EXCLUDE_MODELS_SQL})
              AND distance != -999
              AND start_x BETWEEN 33.0 AND 39.0
              AND start_y BETWEEN 124.0 AND 132.0
              AND points >= 5
              AND CAST(speeds AS VARCHAR) != '-999'
              AND speeds IS NOT NULL
        ),
        with_stats AS (
            SELECT *, len(speed_arr) AS n_pts,
                   list_avg(speed_arr) AS mean_spd
            FROM raw_parsed WHERE len(speed_arr) >= 2
        ),
        with_derived AS (
            SELECT *,
                sqrt(GREATEST(
                    list_avg(list_transform(speed_arr, x -> x * x)) - mean_spd * mean_spd, 0.0
                )) AS speed_std_val,
                list_transform(range(1, n_pts), i -> (speed_arr[i+1] - speed_arr[i]) * {ACCEL_FACTOR}) AS accel_arr
            FROM with_stats
        )
        SELECT
            user_id, mode, month_year,
            start_lat, start_lon,
            ROUND(mean_spd, 2) AS mean_speed,
            list_count(list_filter(accel_arr, x -> x > {HARSH_THRESH})) AS harsh_accel_count,
            list_count(list_filter(accel_arr, x -> x < -{HARSH_THRESH})) AS harsh_decel_count,
            ROUND(CASE WHEN mean_spd > 0 THEN speed_std_val / mean_spd ELSE 0.0 END, 4) AS speed_cv,
            ROUND(list_count(list_filter(speed_arr, x -> abs(x - mean_spd) <= 3.0)) * 1.0 / n_pts, 4) AS cruise_fraction,
            ROUND(list_count(list_filter(speed_arr, x -> x = 0.0)) * 1.0 / n_pts, 4) AS zero_speed_fraction
        FROM with_derived
        WHERE mode IN ('TUB', 'STD', 'ECO')
    """).fetchdf()

    # Assign city
    city_names = list(CITY_CENTERS.keys())
    city_coords = np.array([CITY_CENTERS[c] for c in city_names])
    tree = cKDTree(city_coords)
    coords = np.column_stack([dec["start_lat"].values, dec["start_lon"].values])
    _, idxs = tree.query(coords, k=1)
    dec["city"] = [city_names[i] for i in idxs]

    return dec


def task_3_1(panel: pd.DataFrame) -> dict:
    """Multi-outcome DiD restricted to STD/ECO trips only (city-month level)."""
    print("=" * 60)
    print("Task 3.1: DiD on STD/ECO Trips Only")
    print("=" * 60)

    # Use city_month_panel from Phase 2 but rebuild with STD/ECO only
    trip_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    # Feb-Nov STD/ECO
    feb_nov = con.execute(f"""
        SELECT city, month_year, user_id,
            mean_speed, harsh_accel_count, harsh_decel_count,
            speed_cv, cruise_fraction, zero_speed_fraction
        FROM read_parquet('{trip_path}')
        WHERE mode IN ('STD', 'ECO')
    """).fetchdf()
    print(f"  Feb-Nov STD/ECO trips: {len(feb_nov):,}")

    # Dec STD/ECO
    dec = _load_dec_trips(con)
    con.close()
    dec_std_eco = dec[dec["mode"].isin(["STD", "ECO"])]
    print(f"  Dec STD/ECO trips: {len(dec_std_eco):,}")

    all_trips = pd.concat([
        feb_nov[["city", "month_year", "user_id"] + OUTCOMES],
        dec_std_eco[["city", "month_year", "user_id"] + OUTCOMES],
    ], ignore_index=True)

    # Aggregate to city-month
    panel_se = all_trips.groupby(["city", "month_year"]).agg(
        **{o: (o, "mean") for o in OUTCOMES},
        trip_count=("user_id", "count"),
    ).reset_index()

    # Nov TUB share from the FULL panel (treatment intensity from original)
    nov_tub = panel[panel["month_year"] == "2023-11"][["city", "tub_share_nov"]].drop_duplicates()
    panel_se = panel_se.merge(nov_tub, on="city", how="left")
    panel_se["tub_share_nov"] = panel_se["tub_share_nov"].fillna(0)
    panel_se["post"] = (panel_se["month_year"] == "2023-12").astype(int)
    panel_se["treat_x_post"] = panel_se["tub_share_nov"] * panel_se["post"]

    # TWFE DiD per outcome
    city_dummies = pd.get_dummies(panel_se["city"], prefix="city", dtype=float, drop_first=True)
    month_dummies = pd.get_dummies(panel_se["month_year"], prefix="month", dtype=float, drop_first=True)

    n_tests = len(OUTCOMES)
    bonf_alpha = 0.05 / n_tests
    results = {}

    for outcome in OUTCOMES:
        y = panel_se[outcome].values
        X = np.column_stack([
            panel_se["treat_x_post"].values,
            city_dummies.values,
            month_dummies.values,
        ])
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": panel_se["city"]})

        beta = model.params[1]
        se = model.bse[1]
        p = model.pvalues[1]
        ci_lo, ci_hi = model.conf_int()[1]

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        bonf = " [Bonf]" if p < bonf_alpha else ""

        results[outcome] = {
            "beta": round(float(beta), 4),
            "se": round(float(se), 4),
            "p": float(p),
            "ci_lower": round(float(ci_lo), 4),
            "ci_upper": round(float(ci_hi), 4),
            "significant_bonferroni": p < bonf_alpha,
        }

        print(f"  {outcome:<25s} beta={beta:>8.4f} SE={se:.4f} p={p:.4g} {sig}{bonf}")

    return results


def task_3_2_3(panel: pd.DataFrame) -> dict:
    """Mode-switcher within-user DiD + Cohen's d (Tasks 3.2-3.3).

    Switchers: users who used TUB in Oct-Nov (pre), forced to STD/ECO in Dec (post).
    Control: users who NEVER used TUB.
    Outcome: user-level mean of each outcome in STD/ECO trips only.
    """
    print("\n" + "=" * 60)
    print("Task 3.2-3.3: Mode-Switcher Within-User DiD + Cohen's d")
    print("=" * 60)

    trip_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    # Get Oct-Nov user-level data from trip_modeling
    print("  Loading Oct-Nov user data...")
    pre_trips = con.execute(f"""
        SELECT user_id, mode,
            mean_speed, harsh_accel_count, harsh_decel_count,
            speed_cv, cruise_fraction, zero_speed_fraction
        FROM read_parquet('{trip_path}')
        WHERE month_year IN ('2023-10', '2023-11')
    """).fetchdf()

    # Identify switchers (used TUB in Oct-Nov) vs never-TUB
    tub_users_pre = set(pre_trips[pre_trips["mode"] == "TUB"]["user_id"].unique())
    all_modes_by_user = pre_trips.groupby("user_id")["mode"].apply(set).to_dict()

    # Never-TUB: no TUB in ANY month (use full dataset for this)
    print("  Identifying never-TUB users...")
    ever_tub = con.execute(f"""
        SELECT DISTINCT user_id
        FROM read_parquet('{trip_path}')
        WHERE mode = 'TUB'
    """).fetchdf()["user_id"].values
    ever_tub_set = set(ever_tub)

    # Pre-period STD/ECO user means (Oct-Nov)
    pre_std_eco = pre_trips[pre_trips["mode"].isin(["STD", "ECO"])]
    pre_user_means = pre_std_eco.groupby("user_id")[OUTCOMES].mean()

    # Dec STD/ECO user means
    print("  Loading Dec user data...")
    dec = _load_dec_trips(con)
    con.close()
    dec_std_eco = dec[dec["mode"].isin(["STD", "ECO"])]
    post_user_means = dec_std_eco.groupby("user_id")[OUTCOMES].mean()

    # Find users with both pre and post periods
    common_users = pre_user_means.index.intersection(post_user_means.index)

    # Classify: switcher (used TUB pre-ban) vs never-TUB
    switchers = [u for u in common_users if u in tub_users_pre]
    never_tub = [u for u in common_users if u not in ever_tub_set]

    print(f"  Switchers with both periods: {len(switchers):,}")
    print(f"  Never-TUB with both periods: {len(never_tub):,}")

    # Build user-level panel for DiD
    results = {}
    cohens_d = {}

    for outcome in OUTCOMES:
        # Switcher pre/post
        sw_pre = pre_user_means.loc[switchers, outcome].values
        sw_post = post_user_means.loc[switchers, outcome].values
        sw_delta = sw_post - sw_pre

        # Never-TUB pre/post
        nt_pre = pre_user_means.loc[never_tub, outcome].values
        nt_post = post_user_means.loc[never_tub, outcome].values
        nt_delta = nt_post - nt_pre

        # DiD = mean(switcher delta) - mean(never-TUB delta)
        did = np.mean(sw_delta) - np.mean(nt_delta)

        # SE via two-sample t-test on deltas
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(sw_delta, nt_delta, equal_var=False)

        # Cohen's d on the within-user DiD
        pooled_std = np.sqrt((np.var(sw_delta, ddof=1) * (len(sw_delta) - 1) +
                              np.var(nt_delta, ddof=1) * (len(nt_delta) - 1)) /
                             (len(sw_delta) + len(nt_delta) - 2))
        d = did / pooled_std if pooled_std > 0 else 0.0

        # Pre-existing gap (switchers vs never-TUB in STD/ECO before ban)
        pre_gap = np.mean(sw_pre) - np.mean(nt_pre)

        results[outcome] = {
            "did_estimate": round(float(did), 4),
            "t_stat": round(float(t_stat), 3),
            "p_value": float(p_val),
            "cohens_d": round(float(d), 4),
            "n_switchers": len(switchers),
            "n_never_tub": len(never_tub),
            "switcher_pre_mean": round(float(np.mean(sw_pre)), 4),
            "switcher_post_mean": round(float(np.mean(sw_post)), 4),
            "never_tub_pre_mean": round(float(np.mean(nt_pre)), 4),
            "never_tub_post_mean": round(float(np.mean(nt_post)), 4),
            "pre_gap": round(float(pre_gap), 4),
        }

        cohens_d[outcome] = d

        d_label = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        print(f"  {outcome:<25s} DiD={did:>+8.4f} d={d:>+6.3f} ({d_label}) p={p_val:.4g} {sig}")
        print(f"    Pre-gap: {pre_gap:+.4f} | Sw: {np.mean(sw_pre):.3f}->{np.mean(sw_post):.3f} | NT: {np.mean(nt_pre):.3f}->{np.mean(nt_post):.3f}")

    # Fig 6: Cohen's d comparison across outcomes
    _make_compensation_figure(cohens_d)

    return results


def _make_compensation_figure(cohens_d: dict):
    """Fig 6: Cohen's d effect sizes across all outcomes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    outcomes = list(cohens_d.keys())
    d_values = [cohens_d[o] for o in outcomes]
    y_pos = range(len(outcomes))

    colors = ["#e74c3c" if abs(d) >= 0.5 else "#f39c12" if abs(d) >= 0.2 else "#2ecc71"
              for d in d_values]

    bars = ax.barh(list(y_pos), d_values, color=colors, alpha=0.7, height=0.6)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(x=0.2, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axvline(x=-0.2, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

    labels = [OUTCOME_LABELS.get(o, o).replace("\n", " ") for o in outcomes]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Cohen's d (Mode-Switcher Within-User DiD)", fontsize=11)
    ax.set_title("Compensation Test: Effect Sizes Across Behavioral Margins\n"
                 "(Switchers vs Never-TUB, STD/ECO trips only)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.7, label="Medium/Large (|d| >= 0.5)"),
        Patch(facecolor="#f39c12", alpha=0.7, label="Small (0.2 <= |d| < 0.5)"),
        Patch(facecolor="#2ecc71", alpha=0.7, label="Negligible (|d| < 0.2)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add threshold annotations
    ax.text(0.21, len(outcomes) - 0.3, "Small effect threshold", fontsize=7, color="gray", alpha=0.7)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig6_compensation_cohens_d.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Fig 6 saved: {FIG_DIR / 'fig6_compensation_cohens_d.pdf'}")


def task_3_4(panel: pd.DataFrame) -> dict:
    """Placebo: mode-switcher DiD using Aug-Sep -> Oct (no treatment)."""
    print("\n" + "=" * 60)
    print("Task 3.4: Mode-Switcher Placebo (Oct 2023)")
    print("=" * 60)

    trip_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    # Pre = Aug-Sep, Post = Oct
    pre_trips = con.execute(f"""
        SELECT user_id, mode,
            mean_speed, harsh_accel_count, harsh_decel_count,
            speed_cv, cruise_fraction, zero_speed_fraction
        FROM read_parquet('{trip_path}')
        WHERE month_year IN ('2023-08', '2023-09')
    """).fetchdf()

    post_trips = con.execute(f"""
        SELECT user_id, mode,
            mean_speed, harsh_accel_count, harsh_decel_count,
            speed_cv, cruise_fraction, zero_speed_fraction
        FROM read_parquet('{trip_path}')
        WHERE month_year = '2023-10'
    """).fetchdf()

    # Identify TUB users (used TUB in Aug-Sep)
    tub_users = set(pre_trips[pre_trips["mode"] == "TUB"]["user_id"].unique())

    # Never-TUB (across full dataset)
    ever_tub = con.execute(f"""
        SELECT DISTINCT user_id
        FROM read_parquet('{trip_path}')
        WHERE mode = 'TUB'
    """).fetchdf()["user_id"].values
    ever_tub_set = set(ever_tub)
    con.close()

    # STD/ECO means
    pre_std = pre_trips[pre_trips["mode"].isin(["STD", "ECO"])]
    pre_means = pre_std.groupby("user_id")[OUTCOMES].mean()

    post_std = post_trips[post_trips["mode"].isin(["STD", "ECO"])]
    post_means = post_std.groupby("user_id")[OUTCOMES].mean()

    common = pre_means.index.intersection(post_means.index)
    switchers = [u for u in common if u in tub_users]
    never_tub = [u for u in common if u not in ever_tub_set]

    print(f"  Placebo switchers: {len(switchers):,}")
    print(f"  Placebo never-TUB: {len(never_tub):,}")

    results = {}
    for outcome in OUTCOMES:
        sw_delta = post_means.loc[switchers, outcome].values - pre_means.loc[switchers, outcome].values
        nt_delta = post_means.loc[never_tub, outcome].values - pre_means.loc[never_tub, outcome].values

        did = np.mean(sw_delta) - np.mean(nt_delta)
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(sw_delta, nt_delta, equal_var=False)

        pooled_std = np.sqrt((np.var(sw_delta, ddof=1) * (len(sw_delta) - 1) +
                              np.var(nt_delta, ddof=1) * (len(nt_delta) - 1)) /
                             (len(sw_delta) + len(nt_delta) - 2))
        d = did / pooled_std if pooled_std > 0 else 0.0

        status = "PASS" if p_val > 0.05 or abs(d) < 0.2 else "FAIL"
        results[outcome] = {
            "placebo_did": round(float(did), 4),
            "cohens_d": round(float(d), 4),
            "p_value": float(p_val),
            "passes": status == "PASS",
        }

        print(f"  {outcome:<25s} DiD={did:>+8.4f} d={d:>+6.3f} p={p_val:.4g} [{status}]")

    return results


def task_3_5(did_se: dict, switcher: dict, placebo: dict) -> dict:
    """Summary table: which margins show compensation and which don't."""
    print("\n" + "=" * 60)
    print("Task 3.5: Compensation Summary")
    print("=" * 60)

    summary = {}
    print(f"  {'Outcome':<25s} {'DiD(SE)':<12s} {'Switcher d':<12s} {'Placebo d':<12s} {'Verdict':<15s}")
    print(f"  {'-' * 76}")

    for outcome in OUTCOMES:
        se_p = did_se[outcome]["p"]
        sw_d = switcher[outcome]["cohens_d"]
        pl_d = placebo[outcome]["cohens_d"]

        # Compensation = significant effect in STD/ECO DiD + non-negligible switcher d
        if se_p < 0.05 and abs(sw_d) >= 0.2:
            verdict = "COMPENSATION"
        elif se_p < 0.05 and abs(sw_d) < 0.2:
            verdict = "Aggregate only"
        else:
            verdict = "NULL (no compensation)"

        summary[outcome] = {
            "did_std_eco_p": float(se_p),
            "switcher_cohens_d": float(sw_d),
            "placebo_cohens_d": float(pl_d),
            "verdict": verdict,
        }

        print(f"  {outcome:<25s} p={se_p:<10.4g} d={sw_d:<+10.3f} d={pl_d:<+10.3f} {verdict}")

    return summary


def main():
    print("Phase 3: Compensation Test")
    print("=" * 60)

    t_start = time.time()

    # Load full panel for treatment variable
    panel_path = V2_DIR / "city_month_panel_v2.parquet"
    panel = pd.read_parquet(panel_path)

    # Task 3.1
    did_se = task_3_1(panel)

    # Task 3.2-3.3
    switcher = task_3_2_3(panel)

    # Task 3.4
    placebo = task_3_4(panel)

    # Task 3.5
    summary = task_3_5(did_se, switcher, placebo)

    elapsed = time.time() - t_start

    report = {
        "phase": 3,
        "did_std_eco_only": did_se,
        "mode_switcher": switcher,
        "placebo": placebo,
        "summary": summary,
        "time_s": round(elapsed, 1),
    }

    report_path = V2_DIR / "phase3_compensation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Phase 3 complete in {elapsed:.0f}s")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
