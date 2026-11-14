"""
Phase 2: Multi-Outcome Causal DiD (Tasks 2.1-2.6).

Uses Dec 2023 TUB ban as natural experiment.

Task 2.1: Build city-month panel with all outcomes + trip count + active users.
Task 2.2: TWFE DiD on EACH outcome.
Task 2.3: Event study for each outcome.
Task 2.4: Demand response figure.
Task 2.5: Dose-response for each outcome.
Task 2.6: Placebo test (Sep 2023).

Requires Dec 2023 data -> use raw CSV for Dec, plus existing indicators for Feb-Nov.
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

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, FIG_DPI, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
FIG_DIR = FIGURES_DIR / "v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path("D:/SwingData/raw")

# City centers for assignment (same as assign_road_class_v2.py)
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

# Outcomes for DiD
DID_OUTCOMES = [
    "mean_speed", "harsh_accel_count", "harsh_decel_count",
    "speed_cv", "cruise_fraction", "zero_speed_fraction",
]
DID_DEMAND = ["trip_count", "active_users"]

EXCLUDE_MODELS = ["I9"]
EXCLUDE_MODELS_SQL = ", ".join(f"'{m}'" for m in EXCLUDE_MODELS)

ACCEL_FACTOR = 1000.0 / 3600.0 / 10.0
HARSH_THRESH = 0.5


def _assign_city_sql() -> str:
    """Generate SQL CASE expression for city assignment based on nearest city center."""
    # For simplicity in DuckDB, use a brute-force distance approach
    # This is computed in Python after fetching data instead
    return ""


def task_2_1(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build city-month panel including Dec 2023."""
    print("=" * 60)
    print("Task 2.1: Build City-Month Panel (Feb-Dec 2023)")
    print("=" * 60)

    trip_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

    # Step 1: Get Feb-Nov from trip_modeling (already has city)
    print("  Loading Feb-Nov from trip_modeling...")
    feb_nov = con.execute(f"""
        SELECT
            city, month_year, mode,
            mean_speed, harsh_accel_count, harsh_decel_count,
            speed_cv, cruise_fraction, zero_speed_fraction,
            user_id
        FROM read_parquet('{trip_path}')
        WHERE mode IN ('TUB', 'STD', 'ECO')
    """).fetchdf()
    print(f"  Feb-Nov trips: {len(feb_nov):,}")

    # Step 2: Process Dec 2023 from raw CSV
    print("  Processing Dec 2023 from raw CSV...")
    dec_csv = str(RAW_DIR / "2023_12_Swing_Routes.csv").replace("\\", "/")

    dec_raw = con.execute(f"""
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

    print(f"  Dec 2023 trips: {len(dec_raw):,}")

    # Assign city to Dec trips using KDTree
    from scipy.spatial import cKDTree
    city_names = list(CITY_CENTERS.keys())
    city_coords = np.array([CITY_CENTERS[c] for c in city_names])
    city_tree = cKDTree(city_coords)

    trip_coords = np.column_stack([dec_raw["start_lat"].values, dec_raw["start_lon"].values])
    _, idxs = city_tree.query(trip_coords, k=1)
    dec_raw["city"] = [city_names[i] for i in idxs]

    # Combine
    all_trips = pd.concat([
        feb_nov[["city", "month_year", "mode", "user_id"] + DID_OUTCOMES],
        dec_raw[["city", "month_year", "mode", "user_id"] + DID_OUTCOMES],
    ], ignore_index=True)

    print(f"  Total trips (Feb-Dec): {len(all_trips):,}")

    # Aggregate to city-month panel
    print("  Aggregating to city-month level...")
    panel = all_trips.groupby(["city", "month_year"]).agg(
        trip_count=("user_id", "count"),
        active_users=("user_id", "nunique"),
        mean_speed=("mean_speed", "mean"),
        harsh_accel_count=("harsh_accel_count", "mean"),
        harsh_decel_count=("harsh_decel_count", "mean"),
        speed_cv=("speed_cv", "mean"),
        cruise_fraction=("cruise_fraction", "mean"),
        zero_speed_fraction=("zero_speed_fraction", "mean"),
        tub_share=("mode", lambda x: (x == "TUB").mean()),
    ).reset_index()

    # Nov TUB share as treatment intensity
    nov_tub = panel[panel["month_year"] == "2023-11"][["city", "tub_share"]].rename(
        columns={"tub_share": "tub_share_nov"}
    )
    panel = panel.merge(nov_tub, on="city", how="left")
    panel["tub_share_nov"] = panel["tub_share_nov"].fillna(0)

    # Post indicator
    panel["post"] = (panel["month_year"] == "2023-12").astype(int)

    # Treatment variable
    panel["treat_x_post"] = panel["tub_share_nov"] * panel["post"]

    # Save
    panel_path = V2_DIR / "city_month_panel_v2.parquet"
    panel.to_parquet(panel_path, index=False, compression="zstd")

    print(f"\n  Panel: {len(panel)} obs ({panel['city'].nunique()} cities x {panel['month_year'].nunique()} months)")
    print(f"  Output: {panel_path}")

    return panel


def task_2_2(panel: pd.DataFrame) -> dict:
    """TWFE DiD on each outcome."""
    print("\n" + "=" * 60)
    print("Task 2.2: Multi-Outcome TWFE DiD")
    print("=" * 60)

    all_outcomes = DID_OUTCOMES + DID_DEMAND
    n_tests = len(all_outcomes)
    bonferroni_alpha = 0.05 / n_tests

    results = {}
    coef_data = []

    # City and month fixed effects
    city_dummies = pd.get_dummies(panel["city"], prefix="city", dtype=float, drop_first=True)
    month_dummies = pd.get_dummies(panel["month_year"], prefix="month", dtype=float, drop_first=True)

    for outcome in all_outcomes:
        y = panel[outcome].values
        X = np.column_stack([
            panel["treat_x_post"].values,
            city_dummies.values,
            month_dummies.values,
        ])
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": panel["city"]})

        beta = model.params[1]
        se = model.bse[1]
        p = model.pvalues[1]
        ci_lo, ci_hi = model.conf_int()[1]

        sig_bonf = p < bonferroni_alpha
        sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        bonf_str = " [Bonf]" if sig_bonf else ""

        results[outcome] = {
            "beta": round(float(beta), 4),
            "se": round(float(se), 4),
            "p": float(p),
            "ci_lower": round(float(ci_lo), 4),
            "ci_upper": round(float(ci_hi), 4),
            "significant_bonferroni": sig_bonf,
        }

        coef_data.append({
            "outcome": outcome,
            "beta": beta, "se": se, "p": p,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })

        print(f"  {outcome:<25s} beta={beta:>8.4f} SE={se:.4f} p={p:.4g} {sig_str}{bonf_str}")

    print(f"\n  Bonferroni threshold: p < {bonferroni_alpha:.4f} ({n_tests} tests)")

    # Multi-outcome DiD coefficient figure (KEY FIGURE - Fig 4)
    _make_did_coef_figure(coef_data, bonferroni_alpha)

    return results


def _make_did_coef_figure(coef_data: list, bonf_alpha: float):
    """Fig 4: Multi-outcome DiD coefficients with Bonferroni correction."""
    df = pd.DataFrame(coef_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(df))
    colors = ["#e74c3c" if p < bonf_alpha else "#f39c12" if p < 0.05 else "#95a5a6"
              for p in df["p"]]

    ax.barh(list(y_pos), df["beta"],
            xerr=[df["beta"] - df["ci_lo"], df["ci_hi"] - df["beta"]],
            color=colors, alpha=0.7, capsize=4, height=0.6)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")

    # Labels
    labels = [o.replace("_", " ").title() for o in df["outcome"]]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("DiD Coefficient (TUB share x Post)", fontsize=11)
    ax.set_title("Multi-Outcome Difference-in-Differences\n(Dec 2023 TUB Ban, Bonferroni-corrected)",
                 fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.7, label=f"Significant (Bonferroni, p<{bonf_alpha:.3f})"),
        Patch(facecolor="#f39c12", alpha=0.7, label="Significant (p<0.05)"),
        Patch(facecolor="#95a5a6", alpha=0.7, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig4_did_multi_outcome.{fmt}",
                    dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Fig 4 (KEY FIGURE) saved: {FIG_DIR / 'fig4_did_multi_outcome.pdf'}")


def task_2_5(panel: pd.DataFrame) -> dict:
    """Dose-response: city-level outcome change vs TUB share reduction."""
    print("\n" + "=" * 60)
    print("Task 2.5: Dose-Response Analysis")
    print("=" * 60)

    results = {}

    nov = panel[panel["month_year"] == "2023-11"].set_index("city")
    dec = panel[panel["month_year"] == "2023-12"].set_index("city")
    common = nov.index.intersection(dec.index)

    for outcome in DID_OUTCOMES:
        delta_y = dec.loc[common, outcome] - nov.loc[common, outcome]
        delta_tub = dec.loc[common, "tub_share"] - nov.loc[common, "tub_share"]

        valid = ~(delta_y.isna() | delta_tub.isna())
        dy = delta_y[valid].values
        dt = delta_tub[valid].values

        if len(dy) < 5:
            continue

        from scipy.stats import pearsonr
        r, p = pearsonr(dt, dy)

        # Bootstrap slope
        slopes = []
        rng = np.random.default_rng(RANDOM_SEED)
        for _ in range(1000):
            idx = rng.choice(len(dy), len(dy), replace=True)
            X_b = sm.add_constant(dt[idx])
            fit = sm.OLS(dy[idx], X_b).fit()
            slopes.append(fit.params[1])
        slope_ci = np.percentile(slopes, [2.5, 97.5])

        X = sm.add_constant(dt)
        fit = sm.OLS(dy, X).fit()

        results[outcome] = {
            "r": round(float(r), 3),
            "p": float(p),
            "slope": round(float(fit.params[1]), 4),
            "slope_ci": [round(float(slope_ci[0]), 4), round(float(slope_ci[1]), 4)],
            "n_cities": int(valid.sum()),
        }

        print(f"  {outcome:<25s} r={r:.3f} p={p:.4g} slope={fit.params[1]:.4f}")

    return results


def task_2_6(panel: pd.DataFrame) -> dict:
    """Placebo test using Sep 2023 as fake treatment date."""
    print("\n" + "=" * 60)
    print("Task 2.6: Placebo Test (Sep 2023)")
    print("=" * 60)

    # Restrict to pre-treatment period, use Sep as fake post
    pre_panel = panel[panel["month_year"].isin([
        f"2023-{m:02d}" for m in range(2, 12)  # Feb-Nov only
    ])].copy()

    pre_panel["fake_post"] = (pre_panel["month_year"] >= "2023-09").astype(int)
    pre_panel["fake_treat_x_post"] = pre_panel["tub_share_nov"] * pre_panel["fake_post"]

    city_dummies = pd.get_dummies(pre_panel["city"], prefix="city", dtype=float, drop_first=True)
    month_dummies = pd.get_dummies(pre_panel["month_year"], prefix="month", dtype=float, drop_first=True)

    results = {}
    for outcome in DID_OUTCOMES:
        y = pre_panel[outcome].values
        X = np.column_stack([
            pre_panel["fake_treat_x_post"].values,
            city_dummies.values,
            month_dummies.values,
        ])
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pre_panel["city"]})

        beta = model.params[1]
        p = model.pvalues[1]

        results[outcome] = {
            "placebo_beta": round(float(beta), 4),
            "placebo_p": float(p),
            "passes": p > 0.05,
        }

        status = "PASS" if p > 0.05 else "FAIL"
        print(f"  {outcome:<25s} beta={beta:>8.4f} p={p:.4f} [{status}]")

    return results


def main():
    print("Phase 2: Multi-Outcome Causal DiD")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 8")

    # Task 2.1
    panel = task_2_1(con)
    con.close()

    # Task 2.2
    did_results = task_2_2(panel)

    # Task 2.5
    dose_results = task_2_5(panel)

    # Task 2.6
    placebo_results = task_2_6(panel)

    elapsed = time.time() - t_start

    report = {
        "phase": 2,
        "did_results": did_results,
        "dose_response": dose_results,
        "placebo": placebo_results,
        "time_s": round(elapsed, 1),
    }

    report_path = V2_DIR / "phase2_did_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Phase 2 complete in {elapsed:.0f}s")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
