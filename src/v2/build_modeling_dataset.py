"""
Task 0.4: Build unified modeling dataset by merging:
  - trip_indicators_v2.parquet  (behavioral outcomes)
  - trip_road_classes_v2.parquet  (road class + city)
  - 2023_Swing_Scooter.csv  (user age)

Filters to scooter modes only (TUB/STD/ECO), adds derived columns
(age_group, time_of_day, day_type, frac_major_road placeholder).

Output: data_parquet/v2/trip_modeling.parquet
"""

import duckdb
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
SCOOTER_CSV = Path("D:/SwingData/raw/2023_Swing_Scooter.csv")


def main():
    print("=" * 60)
    print("Task 0.4: Build Unified Modeling Dataset")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 8")

    indicators_path = str(V2_DIR / "trip_indicators_v2.parquet").replace("\\", "/")
    road_path = str(V2_DIR / "trip_road_classes_v2.parquet").replace("\\", "/")
    scooter_path = str(SCOOTER_CSV).replace("\\", "/")
    out_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

    # Step 1: Build user age lookup from Scooter CSV
    print("  Building user age lookup...")
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE user_age AS
        SELECT user_id, MEDIAN(age) AS age
        FROM read_csv_auto('{scooter_path}', ignore_errors=true)
        WHERE age IS NOT NULL AND age > 0 AND age < 100
        GROUP BY user_id
    """)
    n_users_age = con.execute("SELECT COUNT(*) FROM user_age").fetchone()[0]
    print(f"  Users with age: {n_users_age:,} ({time.time() - t0:.1f}s)")

    # Step 2: Merge and filter
    print("  Merging indicators + road class + age...")
    t0 = time.time()

    con.execute(f"""
        COPY (
            SELECT
                i.route_id,
                i.user_id,
                i.model,
                i.mode,
                i.travel_time,
                i.start_date,
                i.start_time,
                i.start_lat,
                i.start_lon,
                i.points,
                i.distance,
                i.month_year,

                -- Speed indicators (manipulation check)
                i.n_speed_points,
                i.mean_speed,
                i.max_speed_from_profile,
                i.min_speed,
                i.p85_speed,

                -- PRIMARY: Safety
                i.speed_std,
                i.speed_cv,
                i.harsh_accel_count,
                i.harsh_decel_count,
                i.harsh_event_count,

                -- Speeding (legacy)
                i.is_speeding,
                i.speeding_rate,

                -- PRIMARY: Dynamics
                i.cruise_fraction,
                i.zero_speed_fraction,

                -- Road class
                r.start_road_class,
                r.city,
                r.is_major_road,
                r.road_match_dist_deg,

                -- User demographics
                CAST(a.age AS INTEGER) AS age,
                CASE
                    WHEN a.age < 20 THEN 'under_20'
                    WHEN a.age < 25 THEN '20_24'
                    WHEN a.age < 30 THEN '25_29'
                    WHEN a.age < 35 THEN '30_34'
                    WHEN a.age < 40 THEN '35_39'
                    WHEN a.age < 50 THEN '40_49'
                    WHEN a.age >= 50 THEN '50_plus'
                    ELSE 'unknown'
                END AS age_group,

                -- Time features
                CASE
                    WHEN HOUR(i.start_time) BETWEEN 7 AND 9
                        THEN 'morning_rush'
                    WHEN HOUR(i.start_time) BETWEEN 10 AND 15
                        THEN 'midday'
                    WHEN HOUR(i.start_time) BETWEEN 16 AND 18
                        THEN 'afternoon_rush'
                    WHEN HOUR(i.start_time) BETWEEN 19 AND 22
                        THEN 'evening'
                    ELSE 'night'
                END AS time_of_day,

                CASE
                    WHEN DAYOFWEEK(TRY_CAST(i.start_date AS DATE)) IN (0, 6)
                        THEN 'weekend'
                    ELSE 'weekday'
                END AS day_type

            FROM read_parquet('{indicators_path}') i
            LEFT JOIN read_parquet('{road_path}') r ON i.route_id = r.route_id
            LEFT JOIN user_age a ON i.user_id = a.user_id

            -- Filter to scooter modes only (exclude BIKE_TUB, BIKE_STD, none)
            WHERE i.mode IN ('TUB', 'STD', 'ECO')
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    elapsed_merge = time.time() - t0
    print(f"  Merge complete in {elapsed_merge:.1f}s")

    # Step 3: Verify
    print("\n  Verification:")
    stats = con.execute(f"""
        SELECT
            COUNT(*) AS n_trips,
            COUNT(DISTINCT user_id) AS n_users,
            COUNT(DISTINCT city) AS n_cities,
            COUNT(DISTINCT month_year) AS n_months,

            -- Mode distribution
            SUM(CASE WHEN mode = 'TUB' THEN 1 ELSE 0 END) AS n_tub,
            SUM(CASE WHEN mode = 'STD' THEN 1 ELSE 0 END) AS n_std,
            SUM(CASE WHEN mode = 'ECO' THEN 1 ELSE 0 END) AS n_eco,

            -- Age coverage
            COUNT(CASE WHEN age IS NOT NULL THEN 1 END) AS has_age,
            ROUND(AVG(CAST(age AS DOUBLE)), 1) AS mean_age,

            -- Road class coverage
            COUNT(CASE WHEN start_road_class IS NOT NULL THEN 1 END) AS has_road,

            -- Key outcomes
            ROUND(AVG(mean_speed), 2) AS avg_mean_speed,
            ROUND(AVG(speed_cv), 4) AS avg_speed_cv,
            ROUND(AVG(harsh_accel_count), 3) AS avg_harsh_accel,
            ROUND(AVG(harsh_decel_count), 3) AS avg_harsh_decel,
            ROUND(AVG(cruise_fraction), 4) AS avg_cruise,
            ROUND(AVG(zero_speed_fraction), 4) AS avg_zero_frac,
            ROUND(AVG(distance), 1) AS avg_distance

        FROM read_parquet('{out_path}')
    """).fetchone()

    cols = [
        "n_trips", "n_users", "n_cities", "n_months",
        "n_tub", "n_std", "n_eco",
        "has_age", "mean_age", "has_road",
        "avg_mean_speed", "avg_speed_cv", "avg_harsh_accel", "avg_harsh_decel",
        "avg_cruise", "avg_zero_frac", "avg_distance",
    ]
    result = dict(zip(cols, stats))

    for k, v in result.items():
        print(f"    {k}: {v:,}" if isinstance(v, int) else f"    {k}: {v}")

    # Mode-specific means
    print("\n  By mode:")
    by_mode = con.execute(f"""
        SELECT
            mode,
            COUNT(*) AS n,
            ROUND(AVG(mean_speed), 2) AS mean_speed,
            ROUND(AVG(speed_cv), 4) AS speed_cv,
            ROUND(AVG(harsh_accel_count), 3) AS harsh_accel,
            ROUND(AVG(harsh_decel_count), 3) AS harsh_decel,
            ROUND(AVG(cruise_fraction), 4) AS cruise_frac,
            ROUND(AVG(zero_speed_fraction), 4) AS zero_frac,
            ROUND(AVG(distance), 1) AS distance
        FROM read_parquet('{out_path}')
        GROUP BY mode
        ORDER BY mode
    """).fetchdf()
    print(by_mode.to_string(index=False))

    elapsed = time.time() - t_start
    print(f"\n  Output: {out_path}")
    print(f"  Total time: {elapsed:.1f}s")

    # Save report
    report = {
        "task": "0.4",
        "description": "Unified modeling dataset",
        "stats": result,
        "time_s": round(elapsed, 1),
    }
    report_path = V2_DIR / "modeling_dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    con.close()
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
