"""
Task 0.6: Compute descriptive statistics for Table 1.

Organizes outcomes by: Safety x Demand x Dynamics, by mode (TUB/STD/ECO).
Computes mean, std, median, N for each outcome by mode.

Output: data_parquet/v2/descriptive_stats.json
"""

import duckdb
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR

V2_DIR = DATA_DIR / "v2"

# Outcomes organized by category
OUTCOMES = {
    "manipulation_check": {
        "mean_speed": "Mean speed (km/h)",
        "max_speed_from_profile": "Max speed (km/h)",
        "p85_speed": "P85 speed (km/h)",
    },
    "safety": {
        "harsh_accel_count": "Harsh acceleration (count/trip)",
        "harsh_decel_count": "Harsh deceleration (count/trip)",
        "speed_cv": "Speed CV",
    },
    "demand": {
        "distance": "Trip distance (m)",
        "travel_time": "Travel time (s)",
    },
    "dynamics": {
        "cruise_fraction": "Cruise fraction",
        "zero_speed_fraction": "Zero-speed fraction",
    },
}


def main():
    print("=" * 60)
    print("Task 0.6: Descriptive Statistics for Table 1")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()

    trip_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

    # Overall sample description
    print("  Computing overall sample statistics...")
    overall = con.execute(f"""
        SELECT
            COUNT(*) AS n_trips,
            COUNT(DISTINCT user_id) AS n_users,
            COUNT(DISTINCT city) AS n_cities,
            COUNT(DISTINCT month_year) AS n_months,
            SUM(CASE WHEN mode = 'TUB' THEN 1 ELSE 0 END) AS n_tub,
            SUM(CASE WHEN mode = 'STD' THEN 1 ELSE 0 END) AS n_std,
            SUM(CASE WHEN mode = 'ECO' THEN 1 ELSE 0 END) AS n_eco,
            ROUND(AVG(CAST(age AS DOUBLE)), 1) AS mean_age,
            ROUND(MEDIAN(CAST(age AS DOUBLE)), 0) AS median_age
        FROM read_parquet('{trip_path}')
    """).fetchone()
    overall_cols = [
        "n_trips", "n_users", "n_cities", "n_months",
        "n_tub", "n_std", "n_eco", "mean_age", "median_age",
    ]
    overall_dict = dict(zip(overall_cols, overall))

    print(f"    N trips: {overall_dict['n_trips']:,}")
    print(f"    N users: {overall_dict['n_users']:,}")
    print(f"    TUB: {overall_dict['n_tub']:,} | STD: {overall_dict['n_std']:,} | ECO: {overall_dict['n_eco']:,}")

    # Outcome statistics by mode
    print("\n  Computing outcome statistics by mode...")
    results = {}

    for category, outcomes in OUTCOMES.items():
        results[category] = {}
        for col, label in outcomes.items():
            stats = con.execute(f"""
                SELECT
                    mode,
                    COUNT(*) AS n,
                    ROUND(AVG({col}), 4) AS mean,
                    ROUND(STDDEV({col}), 4) AS std,
                    ROUND(MEDIAN({col}), 4) AS median,
                    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}), 4) AS q25,
                    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}), 4) AS q75
                FROM read_parquet('{trip_path}')
                GROUP BY mode
                ORDER BY mode
            """).fetchdf()

            mode_stats = {}
            for _, row in stats.iterrows():
                mode_stats[row["mode"]] = {
                    "n": int(row["n"]),
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "median": float(row["median"]),
                    "q25": float(row["q25"]),
                    "q75": float(row["q75"]),
                }
            results[category][col] = {"label": label, "by_mode": mode_stats}

    # Print summary table
    print("\n  " + "-" * 80)
    print(f"  {'Outcome':<30s} {'TUB':>12s} {'STD':>12s} {'ECO':>12s}")
    print("  " + "-" * 80)
    for category, outcomes in results.items():
        print(f"  [{category.upper()}]")
        for col, data in outcomes.items():
            tub = data["by_mode"]["TUB"]
            std = data["by_mode"]["STD"]
            eco = data["by_mode"]["ECO"]
            print(f"    {data['label']:<28s} {tub['mean']:>12.3f} {std['mean']:>12.3f} {eco['mean']:>12.3f}")
    print("  " + "-" * 80)

    # Additional: age group distribution
    age_dist = con.execute(f"""
        SELECT age_group, COUNT(*) AS n,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
        FROM read_parquet('{trip_path}')
        GROUP BY age_group ORDER BY age_group
    """).fetchdf()
    age_dict = {row["age_group"]: {"n": int(row["n"]), "pct": float(row["pct"])}
                for _, row in age_dist.iterrows()}

    # Time of day distribution
    tod_dist = con.execute(f"""
        SELECT time_of_day, COUNT(*) AS n,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
        FROM read_parquet('{trip_path}')
        GROUP BY time_of_day ORDER BY time_of_day
    """).fetchdf()
    tod_dict = {row["time_of_day"]: {"n": int(row["n"]), "pct": float(row["pct"])}
                for _, row in tod_dist.iterrows()}

    # City top 10
    city_dist = con.execute(f"""
        SELECT city, COUNT(*) AS n,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
        FROM read_parquet('{trip_path}')
        GROUP BY city ORDER BY n DESC LIMIT 10
    """).fetchdf()
    city_dict = {row["city"]: {"n": int(row["n"]), "pct": float(row["pct"])}
                 for _, row in city_dist.iterrows()}

    elapsed = time.time() - t_start

    # Save everything
    output = {
        "overall": overall_dict,
        "outcomes_by_mode": results,
        "age_distribution": age_dict,
        "time_of_day_distribution": tod_dict,
        "city_distribution_top10": city_dict,
        "time_s": round(elapsed, 1),
    }

    out_path = V2_DIR / "descriptive_stats.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Output: {out_path}")
    print(f"  Time: {elapsed:.1f}s")

    con.close()


if __name__ == "__main__":
    main()
