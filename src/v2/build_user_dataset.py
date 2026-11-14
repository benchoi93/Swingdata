"""
Task 0.5: Compute user-level aggregates from trip_modeling.parquet.

For each user: mean of all 8 primary outcomes, trip frequency per month,
active months, dominant mode, age, city.

Output: data_parquet/v2/user_modeling.parquet
"""

import duckdb
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR

V2_DIR = DATA_DIR / "v2"


def main():
    print("=" * 60)
    print("Task 0.5: Build User-Level Aggregates")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")

    trip_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")
    out_path = str(V2_DIR / "user_modeling.parquet").replace("\\", "/")

    print("  Computing user-level aggregates...")
    t0 = time.time()

    con.execute(f"""
        COPY (
            SELECT
                user_id,

                -- Trip volume
                COUNT(*) AS total_trips,
                COUNT(DISTINCT month_year) AS active_months,
                COUNT(*) * 1.0 / COUNT(DISTINCT month_year) AS trips_per_month,

                -- Mode usage
                SUM(CASE WHEN mode = 'TUB' THEN 1 ELSE 0 END) AS n_tub_trips,
                SUM(CASE WHEN mode = 'STD' THEN 1 ELSE 0 END) AS n_std_trips,
                SUM(CASE WHEN mode = 'ECO' THEN 1 ELSE 0 END) AS n_eco_trips,
                ROUND(SUM(CASE WHEN mode = 'TUB' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4)
                    AS tub_fraction,
                MODE(mode) AS dominant_mode,

                -- Demographics
                MEDIAN(age) AS age,
                MODE(city) AS primary_city,

                -- PRIMARY OUTCOME: Safety
                ROUND(AVG(harsh_accel_count), 4) AS mean_harsh_accel,
                ROUND(AVG(harsh_decel_count), 4) AS mean_harsh_decel,
                ROUND(AVG(harsh_event_count), 4) AS mean_harsh_events,
                ROUND(AVG(speed_cv), 4) AS mean_speed_cv,

                -- PRIMARY OUTCOME: Demand
                ROUND(AVG(distance), 2) AS mean_distance,
                ROUND(AVG(travel_time), 2) AS mean_travel_time,

                -- PRIMARY OUTCOME: Dynamics
                ROUND(AVG(cruise_fraction), 4) AS mean_cruise_fraction,
                ROUND(AVG(zero_speed_fraction), 4) AS mean_zero_speed_fraction,

                -- Manipulation check
                ROUND(AVG(mean_speed), 2) AS mean_mean_speed,
                ROUND(AVG(max_speed_from_profile), 2) AS mean_max_speed,
                ROUND(AVG(p85_speed), 2) AS mean_p85_speed,
                ROUND(AVG(CASE WHEN is_speeding THEN 1.0 ELSE 0.0 END), 4) AS speeding_rate,

                -- Road context
                ROUND(AVG(is_major_road), 4) AS frac_major_road

            FROM read_parquet('{trip_path}')
            GROUP BY user_id
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    elapsed = time.time() - t0
    print(f"  Aggregation complete in {elapsed:.1f}s")

    # Verify
    stats = con.execute(f"""
        SELECT
            COUNT(*) AS n_users,
            ROUND(AVG(total_trips), 1) AS avg_trips,
            ROUND(AVG(active_months), 1) AS avg_active_months,
            ROUND(AVG(trips_per_month), 1) AS avg_trips_per_month,
            SUM(CASE WHEN dominant_mode = 'TUB' THEN 1 ELSE 0 END) AS n_tub_dominant,
            SUM(CASE WHEN dominant_mode = 'STD' THEN 1 ELSE 0 END) AS n_std_dominant,
            SUM(CASE WHEN dominant_mode = 'ECO' THEN 1 ELSE 0 END) AS n_eco_dominant,
            ROUND(AVG(mean_harsh_accel), 4) AS avg_harsh_accel,
            ROUND(AVG(mean_speed_cv), 4) AS avg_speed_cv,
            ROUND(AVG(mean_cruise_fraction), 4) AS avg_cruise,
            ROUND(AVG(mean_distance), 1) AS avg_distance,
            ROUND(AVG(speeding_rate), 4) AS avg_speeding_rate
        FROM read_parquet('{out_path}')
    """).fetchone()

    cols = [
        "n_users", "avg_trips", "avg_active_months", "avg_trips_per_month",
        "n_tub_dominant", "n_std_dominant", "n_eco_dominant",
        "avg_harsh_accel", "avg_speed_cv", "avg_cruise",
        "avg_distance", "avg_speeding_rate",
    ]
    result = dict(zip(cols, stats))

    print("\n  User-level summary:")
    for k, v in result.items():
        print(f"    {k}: {v:,}" if isinstance(v, int) else f"    {k}: {v}")

    total_elapsed = time.time() - t_start
    print(f"\n  Output: {out_path}")
    print(f"  Time: {total_elapsed:.1f}s")

    report = {"task": "0.5", "stats": result, "time_s": round(total_elapsed, 1)}
    report_path = V2_DIR / "user_modeling_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    con.close()
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
