"""
Task 1.4: Compute per-city summary statistics.

Computes for each province:
  - Trip count, unique users, unique vehicles (IMEI)
  - Spatial extent (lat/lon bounding box)
  - Speed distribution (mean, median, P85, max avg_speed/max_speed)
  - Trip distance and duration distributions
  - Vehicle model and mode breakdown
  - Speeding rate (fraction of trips with max_speed > 25 km/h)

Outputs:
  - data_parquet/city_summary_stats.parquet — province-level statistics
  - Console summary table for quick review
"""

import duckdb
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, SPEED_LIMIT_KR


def compute_city_stats() -> None:
    """Compute per-city/province summary statistics using DuckDB."""
    con = duckdb.connect()
    parquet_path = str(DATA_DIR / "routes_with_cities.parquet")

    print("=" * 70)
    print("  TASK 1.4: PER-CITY SUMMARY STATISTICS")
    print("=" * 70)

    # Province-level summary
    print("\n--- Province-Level Summary ---")
    province_stats = con.execute(f"""
        SELECT
            province,
            COUNT(*) AS trip_count,
            COUNT(DISTINCT user_id) AS unique_users,
            COUNT(DISTINCT imei) AS unique_vehicles,

            -- Spatial extent
            MIN(start_x) AS lat_min,
            MAX(start_x) AS lat_max,
            MIN(start_y) AS lon_min,
            MAX(start_y) AS lon_max,

            -- Speed statistics (avg_speed)
            AVG(avg_speed) AS mean_avg_speed,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_speed) AS median_avg_speed,
            PERCENTILE_CONT(0.85) WITHIN GROUP (ORDER BY avg_speed) AS p85_avg_speed,
            MAX(avg_speed) AS max_avg_speed,

            -- Speed statistics (max_speed)
            AVG(max_speed) AS mean_max_speed,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY max_speed) AS median_max_speed,
            PERCENTILE_CONT(0.85) WITHIN GROUP (ORDER BY max_speed) AS p85_max_speed,
            MAX(max_speed) AS max_max_speed,

            -- Trip distance (meters)
            AVG(distance) AS mean_distance,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY distance) AS median_distance,

            -- Trip duration (seconds)
            AVG(travel_time) AS mean_duration,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY travel_time) AS median_duration,

            -- Speeding rate: fraction of trips with max_speed > 25
            AVG(CASE WHEN max_speed > {SPEED_LIMIT_KR} THEN 1.0 ELSE 0.0 END) AS speeding_rate,

            -- GPS points per trip
            AVG(points) AS mean_points,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY points) AS median_points

        FROM read_parquet('{parquet_path}')
        WHERE province != 'Other'
        GROUP BY province
        ORDER BY trip_count DESC
    """).fetchdf()

    # Display results
    print(f"\n{'Province':<15} {'Trips':>10} {'Users':>8} {'Vehicles':>8} "
          f"{'MeanSpd':>8} {'P85Spd':>7} {'MeanMaxS':>9} {'SpdRate':>8} "
          f"{'MeanDist':>9} {'MeanDur':>8}")
    print("-" * 105)

    for _, row in province_stats.iterrows():
        print(f"{row['province']:<15} {row['trip_count']:>10,} {row['unique_users']:>8,} "
              f"{row['unique_vehicles']:>8,} {row['mean_avg_speed']:>8.1f} "
              f"{row['p85_avg_speed']:>7.1f} {row['mean_max_speed']:>9.1f} "
              f"{row['speeding_rate']:>8.1%} {row['mean_distance']:>9.0f}m "
              f"{row['mean_duration']:>7.0f}s")

    # Save as Parquet
    output_path = DATA_DIR / "city_summary_stats.parquet"
    con.register("province_stats_df", province_stats)
    con.execute(f"""
        COPY (SELECT * FROM province_stats_df)
        TO '{output_path}'
        (FORMAT 'parquet', COMPRESSION 'zstd')
    """)
    print(f"\nSaved to {output_path}")

    # Model breakdown by province
    print("\n--- Vehicle Model Breakdown by Province ---")
    model_breakdown = con.execute(f"""
        SELECT
            province,
            model,
            COUNT(*) AS cnt,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY province) AS pct
        FROM read_parquet('{parquet_path}')
        WHERE province != 'Other'
        GROUP BY province, model
        ORDER BY province, cnt DESC
    """).fetchdf()

    current_prov = None
    for _, row in model_breakdown.iterrows():
        if row["province"] != current_prov:
            current_prov = row["province"]
            print(f"\n  {current_prov}:")
        print(f"    {row['model']:<6}: {row['cnt']:>10,} ({row['pct']:>5.1f}%)")

    # Mode breakdown by province
    print("\n--- Ride Mode Breakdown by Province ---")
    mode_breakdown = con.execute(f"""
        SELECT
            province,
            mode,
            COUNT(*) AS cnt,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY province) AS pct
        FROM read_parquet('{parquet_path}')
        WHERE province != 'Other'
        GROUP BY province, mode
        ORDER BY province, cnt DESC
    """).fetchdf()

    current_prov = None
    for _, row in mode_breakdown.iterrows():
        if row["province"] != current_prov:
            current_prov = row["province"]
            print(f"\n  {current_prov}:")
        print(f"    {row['mode']:<15}: {row['cnt']:>10,} ({row['pct']:>5.1f}%)")

    # Type breakdown by province
    print("\n--- Vehicle Type Breakdown by Province ---")
    type_breakdown = con.execute(f"""
        SELECT
            province,
            type,
            COUNT(*) AS cnt,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY province) AS pct
        FROM read_parquet('{parquet_path}')
        WHERE province != 'Other'
        GROUP BY province, type
        ORDER BY province, cnt DESC
    """).fetchdf()

    current_prov = None
    for _, row in type_breakdown.iterrows():
        if row["province"] != current_prov:
            current_prov = row["province"]
            print(f"\n  {current_prov}:")
        print(f"    {row['type']:<10}: {row['cnt']:>10,} ({row['pct']:>5.1f}%)")

    # Overall summary
    print("\n--- Overall Summary ---")
    overall = con.execute(f"""
        SELECT
            COUNT(*) AS total_trips,
            COUNT(DISTINCT user_id) AS total_users,
            COUNT(DISTINCT imei) AS total_vehicles,
            COUNT(DISTINCT province) AS num_provinces,
            COUNT(DISTINCT city) AS num_cities,
            AVG(avg_speed) AS overall_mean_speed,
            AVG(max_speed) AS overall_mean_max_speed,
            AVG(CASE WHEN max_speed > {SPEED_LIMIT_KR} THEN 1.0 ELSE 0.0 END) AS overall_speeding_rate
        FROM read_parquet('{parquet_path}')
        WHERE province != 'Other'
    """).fetchone()

    print(f"  Total trips:        {overall[0]:>12,}")
    print(f"  Unique users:       {overall[1]:>12,}")
    print(f"  Unique vehicles:    {overall[2]:>12,}")
    print(f"  Provinces covered:  {overall[3]:>12}")
    print(f"  Cities covered:     {overall[4]:>12}")
    print(f"  Mean avg speed:     {overall[5]:>12.1f} km/h")
    print(f"  Mean max speed:     {overall[6]:>12.1f} km/h")
    print(f"  Overall speeding rate: {overall[7]:>11.1%}")

    con.close()

    print("\n" + "=" * 70)
    print("  CITY STATISTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    compute_city_stats()
