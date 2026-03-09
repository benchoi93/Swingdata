"""
Task 2.4: Aggregate user-level indicators from trip-level data.

Computes per-user:
  - Trip count and frequency
  - Speed behavior: mean speed across trips, speed consistency (variance)
  - Speeding propensity: fraction of trips with any speeding (max > 25 km/h)
  - Mean speeding rate across trips
  - Harsh event propensity
  - Preferred model, mode, time of day

Outputs:
  - data_parquet/user_indicators.parquet
"""

import duckdb
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, SPEED_LIMIT_KR, USER_INDICATORS_PARQUET


def compute_user_indicators() -> None:
    """Aggregate trip indicators to user level using DuckDB."""
    con = duckdb.connect()

    trips_path = str(DATA_DIR / "cleaned" / "trips_cleaned.parquet")
    indicators_path = str(DATA_DIR / "trip_indicators.parquet")

    print("=" * 70)
    print("  TASK 2.4: USER-LEVEL INDICATORS")
    print("=" * 70)

    # Join trips with indicators and aggregate by user
    print("\nComputing user-level aggregations...")

    output_path = str(USER_INDICATORS_PARQUET)
    con.execute(f"""
        COPY (
            SELECT
                t.user_id,

                -- Trip frequency
                COUNT(*) AS trip_count,
                COUNT(DISTINCT t.start_date) AS active_days,
                COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT t.start_date), 0) AS trips_per_active_day,

                -- Speed behavior (from trip indicators)
                AVG(i.mean_speed) AS user_mean_speed,
                STDDEV(i.mean_speed) AS user_speed_std,
                AVG(i.max_speed_from_profile) AS user_mean_max_speed,
                MAX(i.max_speed_from_profile) AS user_overall_max_speed,
                AVG(i.p85_speed) AS user_mean_p85_speed,
                AVG(i.speed_cv) AS user_mean_speed_cv,

                -- Speeding propensity
                AVG(CASE WHEN t.has_speeding THEN 1.0 ELSE 0.0 END) AS speeding_propensity,
                AVG(i.speeding_rate_25) AS user_mean_speeding_rate_25,
                AVG(i.speeding_duration_25_s) AS user_mean_speeding_dur_25,
                AVG(i.max_excess_25) AS user_mean_max_excess_25,
                SUM(i.speeding_count_25) AS user_total_speeding_points,

                -- Speeding at other thresholds
                AVG(i.speeding_rate_20) AS user_mean_speeding_rate_20,
                AVG(i.speeding_rate_30) AS user_mean_speeding_rate_30,

                -- Acceleration behavior
                AVG(i.mean_abs_accel_ms2) AS user_mean_abs_accel,
                AVG(i.max_accel_ms2) AS user_mean_max_accel,
                AVG(i.max_decel_ms2) AS user_mean_max_decel,
                -- Harsh events at 0.5 m/s^2 threshold (trip level)
                AVG(CASE WHEN i.max_accel_ms2 > 0.5 THEN 1.0 ELSE 0.0 END) AS harsh_accel_propensity,
                AVG(CASE WHEN i.max_decel_ms2 < -0.5 THEN 1.0 ELSE 0.0 END) AS harsh_decel_propensity,

                -- Trip characteristics
                AVG(t.distance) AS user_mean_distance,
                AVG(t.travel_time) AS user_mean_duration,
                AVG(t.gps_points) AS user_mean_gps_points,

                -- Speed profile features
                AVG(i.cruise_fraction) AS user_mean_cruise_fraction,
                AVG(i.zero_speed_fraction) AS user_mean_zero_fraction,
                AVG(i.ramp_up_duration_s) AS user_mean_ramp_up_dur,

                -- Temporal patterns
                AVG(t.start_hour) AS user_mean_start_hour,
                AVG(CASE WHEN t.is_weekend THEN 1.0 ELSE 0.0 END) AS weekend_trip_fraction,

                -- Primary attributes (mode of categorical variables)
                MODE(t.model) AS primary_model,
                MODE(t.mode) AS primary_mode,
                MODE(t.type) AS primary_type,
                MODE(t.province) AS primary_province,
                MODE(t.city) AS primary_city,
                COUNT(DISTINCT t.province) AS provinces_visited,

                -- Strict valid trip fraction
                AVG(CASE WHEN t.is_strict_valid THEN 1.0 ELSE 0.0 END) AS strict_valid_fraction

            FROM read_parquet('{trips_path}') t
            JOIN read_parquet('{indicators_path}') i ON t.route_id = i.route_id
            GROUP BY t.user_id
        )
        TO '{output_path}'
        (FORMAT 'parquet', COMPRESSION 'zstd')
    """)

    # Verify and summarize
    print("Computing summary statistics...")
    summary = con.execute(f"""
        SELECT
            COUNT(*) as total_users,
            AVG(trip_count) as avg_trips,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trip_count) as median_trips,
            MAX(trip_count) as max_trips,
            AVG(user_mean_speed) as avg_user_speed,
            AVG(speeding_propensity) as avg_speeding_prop,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY speeding_propensity) as med_speeding_prop,
            AVG(CASE WHEN speeding_propensity = 0 THEN 1.0 ELSE 0.0 END) as never_speed_frac,
            AVG(CASE WHEN speeding_propensity > 0.5 THEN 1.0 ELSE 0.0 END) as frequent_speeder_frac,
            AVG(user_mean_abs_accel) as avg_user_accel,
            AVG(harsh_accel_propensity) as avg_harsh_accel_prop,
            AVG(weekend_trip_fraction) as avg_weekend_frac
        FROM read_parquet('{output_path}')
    """).fetchone()

    print(f"\n--- User-Level Summary ---")
    print(f"  Total users:                {summary[0]:>10,}")
    print(f"  Mean trips/user:            {summary[1]:>10.1f}")
    print(f"  Median trips/user:          {summary[2]:>10.0f}")
    print(f"  Max trips/user:             {summary[3]:>10,}")
    print(f"  Mean user avg speed:        {summary[4]:>10.2f} km/h")
    print(f"  Mean speeding propensity:   {summary[5]:>10.2%}")
    print(f"  Median speeding propensity: {summary[6]:>10.2%}")
    print(f"  Never-speed users:          {summary[7]:>10.2%}")
    print(f"  Frequent speeders (>50%):   {summary[8]:>10.2%}")
    print(f"  Mean user |accel|:          {summary[9]:>10.4f} m/s^2")
    print(f"  Mean harsh accel propensity:{summary[10]:>10.2%}")
    print(f"  Mean weekend trip fraction: {summary[11]:>10.2%}")

    # Trip count distribution
    print("\n--- Trip Count Distribution ---")
    trip_dist = con.execute(f"""
        SELECT
            CASE
                WHEN trip_count = 1 THEN '1 trip'
                WHEN trip_count BETWEEN 2 AND 5 THEN '2-5 trips'
                WHEN trip_count BETWEEN 6 AND 10 THEN '6-10 trips'
                WHEN trip_count BETWEEN 11 AND 20 THEN '11-20 trips'
                WHEN trip_count BETWEEN 21 AND 50 THEN '21-50 trips'
                WHEN trip_count BETWEEN 51 AND 100 THEN '51-100 trips'
                ELSE '100+ trips'
            END as trip_bin,
            COUNT(*) as cnt,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM read_parquet('{output_path}')) as pct
        FROM read_parquet('{output_path}')
        GROUP BY trip_bin
        ORDER BY MIN(trip_count)
    """).fetchdf()
    for _, row in trip_dist.iterrows():
        print(f"  {row['trip_bin']:<15}: {row['cnt']:>10,} ({row['pct']:>5.1f}%)")

    # Speeding propensity distribution
    print("\n--- Speeding Propensity Distribution ---")
    spd_prop = con.execute(f"""
        SELECT
            CASE
                WHEN speeding_propensity = 0 THEN 'Never speeds'
                WHEN speeding_propensity <= 0.1 THEN '0-10% trips'
                WHEN speeding_propensity <= 0.25 THEN '10-25% trips'
                WHEN speeding_propensity <= 0.5 THEN '25-50% trips'
                WHEN speeding_propensity <= 0.75 THEN '50-75% trips'
                ELSE '75-100% trips'
            END as spd_bin,
            COUNT(*) as cnt,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM read_parquet('{output_path}')) as pct
        FROM read_parquet('{output_path}')
        GROUP BY spd_bin
        ORDER BY MIN(speeding_propensity)
    """).fetchdf()
    for _, row in spd_prop.iterrows():
        print(f"  {row['spd_bin']:<20}: {row['cnt']:>10,} ({row['pct']:>5.1f}%)")

    con.close()

    print(f"\nOutput saved to {output_path}")

    print("\n" + "=" * 70)
    print("  USER INDICATORS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    compute_user_indicators()
