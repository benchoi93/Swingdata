"""
Distributional TWFE DiD Exploration.

Extends the mean-level TWFE DiD framework to quantile-specific effects.
Key question: Does the TUB ban disproportionately reduce extreme-risk trips?

Approach:
1. Build city-month panel with quantile summaries (p25, p50, p75, p90)
2. Run TWFE DiD at each quantile for each outcome
3. Create visualization showing how treatment effects vary across distribution

NOTE: This is a distributional TWFE DiD on city-month quantile summaries,
NOT a formal unconditional quantile treatment effect (QTE) estimator.
Formal QTE methods (Firpo, Fortin, Lemieux 2009 RIF-regression;
Callaway-Li 2019 panel QTE) estimate treatment effects at quantiles of
the unconditional outcome distribution using individual-level data.
The city-level aggregation approach here is a valid descriptive complement
but estimates a different estimand.
"""

import json
import sys
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

# Outcomes suitable for quantile analysis
# Harsh events have too many zeros for lower quantiles; focus on continuous outcomes
# and harsh events at p75/p90 only
QTE_OUTCOMES = ["speed_cv", "cruise_fraction", "zero_speed_fraction", "mean_speed"]
QTE_HARSH = ["harsh_accel_count", "harsh_decel_count"]
QUANTILES = [0.25, 0.50, 0.75, 0.90]

EXCLUDE_MODELS = ["I9"]
EXCLUDE_MODELS_SQL = ", ".join(f"'{m}'" for m in EXCLUDE_MODELS)
ACCEL_FACTOR = 1000.0 / 3600.0 / 10.0
HARSH_THRESH = 0.5


def build_quantile_panel(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build city-month panel with quantile summaries for each outcome."""
    print("=" * 60)
    print("Building City-Month Quantile Panel (Feb-Dec 2023)")
    print("=" * 60)

    trip_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

    # Step 1: Feb-Nov from trip_modeling
    print("  Loading Feb-Nov from trip_modeling...")
    feb_nov = con.execute(f"""
        SELECT city, month_year, mode, user_id,
               mean_speed, harsh_accel_count, harsh_decel_count,
               speed_cv, cruise_fraction, zero_speed_fraction
        FROM read_parquet('{trip_path}')
        WHERE mode IN ('TUB', 'STD', 'ECO')
    """).fetchdf()
    print(f"  Feb-Nov trips: {len(feb_nov):,}")

    # Step 2: Dec 2023 from raw CSV
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

    # Assign city to Dec trips
    from scipy.spatial import cKDTree
    city_names = list(CITY_CENTERS.keys())
    city_coords = np.array([CITY_CENTERS[c] for c in city_names])
    city_tree = cKDTree(city_coords)
    trip_coords = np.column_stack([dec_raw["start_lat"].values, dec_raw["start_lon"].values])
    _, idxs = city_tree.query(trip_coords, k=1)
    dec_raw["city"] = [city_names[i] for i in idxs]

    # Combine
    all_outcomes = QTE_OUTCOMES + QTE_HARSH
    all_trips = pd.concat([
        feb_nov[["city", "month_year", "mode", "user_id"] + all_outcomes],
        dec_raw[["city", "month_year", "mode", "user_id"] + all_outcomes],
    ], ignore_index=True)
    print(f"  Total trips (Feb-Dec): {len(all_trips):,}")

    # Aggregate to city-month panel with quantile summaries
    print("  Computing quantile summaries per city-month...")
    records = []
    for (city, month), grp in all_trips.groupby(["city", "month_year"]):
        rec = {"city": city, "month_year": month, "trip_count": len(grp)}
        rec["tub_share"] = (grp["mode"] == "TUB").mean()

        for outcome in all_outcomes:
            vals = grp[outcome].dropna()
            rec[f"{outcome}_mean"] = vals.mean()
            for q in QUANTILES:
                q_label = f"p{int(q*100)}"
                rec[f"{outcome}_{q_label}"] = vals.quantile(q)
        records.append(rec)

    panel = pd.DataFrame(records)

    # Add Nov TUB share as treatment intensity
    nov_tub = panel[panel["month_year"] == "2023-11"][["city", "tub_share"]].rename(
        columns={"tub_share": "tub_share_nov"}
    )
    panel = panel.merge(nov_tub, on="city", how="left")
    panel["tub_share_nov"] = panel["tub_share_nov"].fillna(0)
    panel["post"] = (panel["month_year"] == "2023-12").astype(int)
    panel["treat_x_post"] = panel["tub_share_nov"] * panel["post"]

    print(f"  Quantile panel: {len(panel)} obs ({panel['city'].nunique()} cities x {panel['month_year'].nunique()} months)")

    # Save
    panel_path = V2_DIR / "city_month_quantile_panel.parquet"
    panel.to_parquet(panel_path, index=False, compression="zstd")
    print(f"  Saved: {panel_path}")

    return panel


def run_qte_did(panel: pd.DataFrame) -> dict:
    """Run distributional TWFE DiD at each quantile for each outcome."""
    print("\n" + "=" * 60)
    print("Distributional TWFE DiD at Each Quantile")
    print("=" * 60)

    city_dummies = pd.get_dummies(panel["city"], prefix="city", dtype=float, drop_first=True)
    month_dummies = pd.get_dummies(panel["month_year"], prefix="month", dtype=float, drop_first=True)

    results = {}

    all_outcomes = QTE_OUTCOMES + QTE_HARSH
    quantile_labels = ["mean"] + [f"p{int(q*100)}" for q in QUANTILES]

    for outcome in all_outcomes:
        results[outcome] = {}
        print(f"\n  --- {outcome} ---")

        for q_label in quantile_labels:
            col = f"{outcome}_{q_label}"

            # Skip p25/p50 for harsh events (all zeros)
            if outcome in QTE_HARSH and q_label in ["p25", "p50"]:
                print(f"    {q_label}: SKIPPED (zero-inflated)")
                results[outcome][q_label] = {"beta": None, "skipped": True}
                continue

            y = panel[col].values

            # Check for zero variance
            if np.std(y) < 1e-10:
                print(f"    {q_label}: SKIPPED (no variance)")
                results[outcome][q_label] = {"beta": None, "skipped": True}
                continue

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

            sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            results[outcome][q_label] = {
                "beta": round(float(beta), 6),
                "se": round(float(se), 6),
                "p": float(p),
                "ci_lower": round(float(ci_lo), 6),
                "ci_upper": round(float(ci_hi), 6),
            }

            print(f"    {q_label}: beta={beta:>10.4f}  SE={se:.4f}  p={p:.4g}  {sig_str}")

    return results


def plot_qte(results: dict) -> None:
    """Create figure showing quantile-specific treatment effects."""
    print("\n" + "=" * 60)
    print("Creating Distributional TWFE DiD Visualization")
    print("=" * 60)

    # Focus on the 4 continuous outcomes
    outcomes_to_plot = QTE_OUTCOMES + QTE_HARSH
    outcome_labels = {
        "speed_cv": "Speed CV",
        "cruise_fraction": "Cruise Fraction",
        "zero_speed_fraction": "Zero-Speed Fraction",
        "mean_speed": "Mean Speed (km/h)",
        "harsh_accel_count": "Harsh Accel Count",
        "harsh_decel_count": "Harsh Decel Count",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, outcome in enumerate(outcomes_to_plot):
        ax = axes[idx]
        q_labels = []
        betas = []
        ci_los = []
        ci_his = []

        for q_label in ["mean", "p25", "p50", "p75", "p90"]:
            r = results[outcome].get(q_label, {})
            if r.get("skipped") or r.get("beta") is None:
                continue
            q_labels.append(q_label)
            betas.append(r["beta"])
            ci_los.append(r["ci_lower"])
            ci_his.append(r["ci_upper"])

        x = np.arange(len(q_labels))
        betas = np.array(betas)
        ci_los = np.array(ci_los)
        ci_his = np.array(ci_his)

        # Color: blue if significant (CI excludes 0), gray otherwise
        colors = []
        for lo, hi in zip(ci_los, ci_his):
            if lo > 0 or hi < 0:
                colors.append("#2166AC")  # significant
            else:
                colors.append("#BDBDBD")  # not significant

        ax.bar(x, betas, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, betas, yerr=[betas - ci_los, ci_his - betas],
                     fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=9)
        ax.set_title(outcome_labels.get(outcome, outcome), fontsize=11, fontweight="bold")
        ax.set_ylabel(r"$\beta$ (treatment effect)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    plt.suptitle("Distributional TWFE DiD: Treatment Effects by Quantile",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig_qte_exploration.{fmt}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / 'fig_qte_exploration.pdf'}")


def summarize_results(results: dict) -> None:
    """Print summary of distributional TWFE DiD findings."""
    print("\n" + "=" * 60)
    print("Distributional TWFE DiD Summary: Heterogeneity of TUB Ban Effect")
    print("=" * 60)

    for outcome in QTE_OUTCOMES:
        print(f"\n  {outcome}:")
        for q_label in ["mean", "p25", "p50", "p75", "p90"]:
            r = results[outcome].get(q_label, {})
            if r.get("skipped"):
                continue
            beta = r["beta"]
            p = r["p"]
            sig = "SIG" if p < 0.05 else "ns"
            print(f"    {q_label}: beta={beta:>10.6f}  p={p:.4g}  [{sig}]")

        # Check for tail concentration
        mean_r = results[outcome].get("mean", {})
        p90_r = results[outcome].get("p90", {})
        if mean_r.get("beta") is not None and p90_r.get("beta") is not None:
            mean_beta = abs(mean_r["beta"])
            p90_beta = abs(p90_r["beta"])
            if mean_beta > 0:
                ratio = p90_beta / mean_beta
                print(f"    -> |beta_p90| / |beta_mean| = {ratio:.2f}  "
                      f"({'tail concentration' if ratio > 1.5 else 'uniform effect'})")

    # Save results
    output_path = V2_DIR / "qte_exploration_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    con = duckdb.connect()

    # Build quantile panel
    panel = build_quantile_panel(con)

    # Run QTE DiD
    results = run_qte_did(panel)

    # Plot
    plot_qte(results)

    # Summary
    summarize_results(results)

    con.close()
    print("\nDone.")
