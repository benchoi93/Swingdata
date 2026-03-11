"""
Proportional Hazards Assumption Check for Cox PH Survival Models.

Tests the PH assumption using Schoenfeld residuals for each Cox PH model
(three endpoints: first_harsh, first_high_cv, first_speeding).

Covariates: ever_tub, tub_frac (same as escalation_pathway.py).
"""

import io
import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
TRIP_PATH = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

SURVIVAL_EVENTS = {
    "first_harsh": "First Harsh Event (accel or decel count > 0)",
    "first_high_cv": "First High-CV Trip (speed_cv > P75)",
    "first_speeding": "First Speeding Trip (max_speed > 25 km/h)",
}


def build_survival_dataset() -> tuple[dict[str, pd.DataFrame], float]:
    """Build survival datasets for each endpoint (200K user sample).

    Returns:
        Tuple of (dict mapping endpoint name -> survival DataFrame, p75_cv threshold).
    """
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 8")
    con.execute(f"SELECT setseed({RANDOM_SEED / 100})")  # DuckDB setseed takes 0-1

    # Get P75 of speed_cv from full dataset
    p75_cv = con.execute(f"""
        SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY speed_cv) AS p75
        FROM read_parquet('{TRIP_PATH}')
        WHERE speed_cv IS NOT NULL
    """).fetchdf()["p75"].iloc[0]
    print(f"  Speed CV P75 threshold: {p75_cv:.4f}")

    # Load user trip sequences (200K sample, up to 500 trips per user)
    print("  Loading trip sequences (200K users)...")
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

    # User-level covariates
    user_max = surv_df.groupby("user_id").agg(
        max_trip=("trip_rank", "max"),
        ever_tub=("mode", lambda x: int((x == "TUB").any())),
        tub_frac=("mode", lambda x: (x == "TUB").mean()),
    ).reset_index()

    # Event detection for each endpoint
    events = {}

    # 1. First harsh event
    harsh_trips = surv_df[
        (surv_df["harsh_accel_count"] > 0) | (surv_df["harsh_decel_count"] > 0)
    ]
    first_harsh = harsh_trips.groupby("user_id")["trip_rank"].min().reset_index()
    first_harsh.columns = ["user_id", "event_trip"]
    events["first_harsh"] = first_harsh

    # 2. First high-CV trip
    high_cv = surv_df[surv_df["speed_cv"] > p75_cv]
    first_cv = high_cv.groupby("user_id")["trip_rank"].min().reset_index()
    first_cv.columns = ["user_id", "event_trip"]
    events["first_high_cv"] = first_cv

    # 3. First speeding trip
    speeding = surv_df[surv_df["max_speed_from_profile"] > 25]
    first_speed = speeding.groupby("user_id")["trip_rank"].min().reset_index()
    first_speed.columns = ["user_id", "event_trip"]
    events["first_speeding"] = first_speed

    # Build Cox-ready DataFrames
    cox_datasets = {}
    for event_name, event_df in events.items():
        surv = user_max.merge(event_df, on="user_id", how="left")
        surv["event"] = surv["event_trip"].notna().astype(int)
        surv["duration"] = surv["event_trip"].fillna(surv["max_trip"])

        # Filter: minimum 2 trips, positive duration
        surv = surv[surv["max_trip"] >= 2]
        cox_df = surv[["duration", "event", "ever_tub", "tub_frac"]].dropna()
        cox_df = cox_df[cox_df["duration"] > 0].copy()

        cox_datasets[event_name] = cox_df
        event_rate = cox_df["event"].mean()
        print(f"  {event_name}: {len(cox_df):,} users, event rate = {event_rate:.1%}")

    con.close()
    return cox_datasets, float(p75_cv)


def check_ph_assumption(cox_datasets: dict[str, pd.DataFrame]) -> dict:
    """Run PH assumption checks for each Cox PH model.

    Returns:
        Dict with Schoenfeld residual test results for each endpoint/covariate.
    """
    results = {}

    for event_name, cox_df in cox_datasets.items():
        print(f"\n{'=' * 60}")
        print(f"PH Assumption Check: {SURVIVAL_EVENTS[event_name]}")
        print(f"{'=' * 60}")

        # Fit the Cox model
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="duration", event_col="event")

        # Print model summary
        print(f"\n  Model Summary:")
        print(f"  N = {len(cox_df):,}, Events = {int(cox_df['event'].sum()):,}")
        for covar in ["ever_tub", "tub_frac"]:
            hr = np.exp(cph.params_[covar])
            p = cph.summary.loc[covar, "p"]
            print(f"    {covar}: HR = {hr:.4f}, p = {p:.2e}")

        # Run PH assumption check - capture printed output
        print(f"\n  Schoenfeld Residual Test:")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            # check_assumptions returns a list of violations (or empty list if PH holds)
            # It also prints the test statistics to stdout
            ph_result = cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"    Error during check: {e}")
            results[event_name] = {"error": str(e)}
            continue
        finally:
            sys.stdout = old_stdout

        captured_output = captured.getvalue()
        print(captured_output)

        # Extract results from the proportional_hazard_test summary
        # Use the built-in method to get structured results
        try:
            from lifelines.statistics import proportional_hazard_test
            ph_test = proportional_hazard_test(cph, cox_df, time_transform="rank")
            ph_summary = ph_test.summary

            event_results = {
                "n": int(len(cox_df)),
                "n_events": int(cox_df["event"].sum()),
                "covariates": {},
            }

            for covar in ["ever_tub", "tub_frac"]:
                if covar in ph_summary.index:
                    row = ph_summary.loc[covar]
                    test_stat = float(row["test_statistic"])
                    p_value = float(row["p"])
                    ph_violated = p_value < 0.05

                    event_results["covariates"][covar] = {
                        "test_statistic": round(test_stat, 4),
                        "p_value": round(p_value, 6),
                        "ph_violated": ph_violated,
                        "hr": round(float(np.exp(cph.params_[covar])), 4),
                    }

                    status = "VIOLATED" if ph_violated else "OK"
                    print(f"    {covar}: chi2 = {test_stat:.4f}, p = {p_value:.6f} -> {status}")

            results[event_name] = event_results

        except Exception as e:
            print(f"    Error extracting structured results: {e}")
            # Fall back to capturing output only
            results[event_name] = {
                "n": int(len(cox_df)),
                "n_events": int(cox_df["event"].sum()),
                "raw_output": captured_output,
                "error": str(e),
            }

    return results


def main():
    print("PH Assumption Check for Cox PH Survival Models")
    print("=" * 60)
    t_start = time.time()

    np.random.seed(RANDOM_SEED)

    # Build survival datasets
    print("\nStep 1: Building survival datasets...")
    cox_datasets, p75_cv = build_survival_dataset()

    # Run PH checks
    print("\nStep 2: Running Schoenfeld residual tests...")
    ph_results = check_ph_assumption(cox_datasets)

    elapsed = time.time() - t_start

    # Compile full report
    report = {
        "description": "Proportional Hazards assumption check via Schoenfeld residual tests",
        "method": "lifelines proportional_hazard_test (rank transform)",
        "covariates": ["ever_tub", "tub_frac"],
        "random_seed": RANDOM_SEED,
        "p75_speed_cv": round(p75_cv, 4),
        "endpoints": ph_results,
        "time_s": round(elapsed, 1),
    }

    # Save
    out_path = V2_DIR / "ph_assumption_check.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out_path}")
    print(f"Completed in {elapsed:.0f}s")

    # Final summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY: PH Assumption Test Results")
    print(f"{'=' * 60}")
    print(f"  {'Endpoint':<20s} {'Covariate':<12s} {'chi2':<12s} {'p-value':<12s} {'Status'}")
    print(f"  {'-' * 68}")
    for endpoint, res in ph_results.items():
        if "covariates" in res:
            for covar, cres in res["covariates"].items():
                status = "VIOLATED" if cres["ph_violated"] else "OK"
                print(f"  {endpoint:<20s} {covar:<12s} {cres['test_statistic']:<12.4f} "
                      f"{cres['p_value']:<12.6f} {status}")


if __name__ == "__main__":
    main()
