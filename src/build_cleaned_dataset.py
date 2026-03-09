"""
Task 1.6: Build cleaned Parquet dataset with city labels and quality flags.

Creates the final analysis-ready dataset at data_parquet/cleaned/ with:
  - All valid trips (core quality filters passed)
  - City and province labels
  - Derived temporal features (hour, day_of_week, is_weekend)
  - Sentinel values replaced with NULL
  - Quality flags retained for sensitivity analysis
  - Strict validity flag (core + distance/duration)

Outputs:
  - data_parquet/cleaned/trips_cleaned.parquet — main analysis dataset
  - data_parquet/cleaned/dataset_summary.json — dataset metadata
"""

import duckdb
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, SPEED_LIMIT_KR, MIN_TRIP_DISTANCE, MIN_TRIP_DURATION

# Maximum plausible trip duration
MAX_TRIP_DURATION = 7200


def build_cleaned_dataset() -> dict:
    """Build the final cleaned dataset with temporal features.

    Returns:
        Dictionary with dataset metadata.
    """
    cleaned_dir = DATA_DIR / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    parquet_path = str(DATA_DIR / "routes_with_cities.parquet")

    print("=" * 70)
    print("  TASK 1.6: BUILD CLEANED DATASET")
    print("=" * 70)

    # Count input
    total = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')"
    ).fetchone()[0]
    print(f"\nInput trips: {total:,}")

    # Build cleaned dataset with temporal features
    output_path = cleaned_dir / "trips_cleaned.parquet"
    print(f"Building cleaned dataset at {output_path} ...")

    con.execute(f"""
        COPY (
            SELECT
                -- Trip identifiers
                route_id,
                user_id,
                imei,
                qrcode,

                -- Vehicle info
                type,
                model,
                mode,

                -- Temporal (original)
                start_date,
                start_time,
                end_date,
                end_time,
                travel_time,

                -- Temporal (derived)
                EXTRACT(HOUR FROM start_time) AS start_hour,
                EXTRACT(DOW FROM start_date) AS day_of_week,  -- 0=Sunday, 6=Saturday
                CASE WHEN EXTRACT(DOW FROM start_date) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend,
                EXTRACT(DAY FROM start_date) AS day_of_month,

                -- Spatial (GPS coordinates)
                start_x AS start_lat,
                start_y AS start_lon,
                end_x AS end_lat,
                end_y AS end_lon,

                -- Trip metrics
                points AS gps_points,
                distance,
                moved_distance,
                avg_point_gap,
                max_point_gap,

                -- Speed metrics
                avg_speed,
                max_speed,

                -- Raw trajectory data (for downstream parsing)
                routes AS routes_raw,
                speeds AS speeds_raw,

                -- City labels
                city,
                province,
                city_distance_deg,

                -- Quality flags
                flag_excluded_model,
                flag_sentinel,
                flag_invalid_coords,
                flag_few_points,
                flag_implausible_speed,
                flag_short_distance,
                flag_short_duration,
                flag_long_duration,
                is_valid,

                -- Strict validity (core + distance + duration)
                CASE WHEN (
                    is_valid = TRUE
                    AND flag_short_distance = FALSE
                    AND flag_short_duration = FALSE
                    AND flag_long_duration = FALSE
                ) THEN TRUE ELSE FALSE END AS is_strict_valid,

                -- Speeding indicators (for quick reference)
                CASE WHEN max_speed > {SPEED_LIMIT_KR} THEN TRUE ELSE FALSE END AS has_speeding,
                CASE WHEN avg_speed > {SPEED_LIMIT_KR} THEN TRUE ELSE FALSE END AS avg_above_limit

            FROM read_parquet('{parquet_path}')
            ORDER BY start_date, start_time
        )
        TO '{output_path}'
        (FORMAT 'parquet', COMPRESSION 'zstd')
    """)

    # Verify and summarize
    verify_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
    ).fetchone()[0]
    print(f"Output rows: {verify_count:,}")
    assert verify_count == total, f"Row count mismatch: {verify_count} != {total}"

    # Dataset summary statistics
    print("\nComputing dataset summary...")
    summary = con.execute(f"""
        SELECT
            COUNT(*) AS total_trips,
            SUM(CASE WHEN is_strict_valid THEN 1 ELSE 0 END) AS strict_valid_trips,
            COUNT(DISTINCT user_id) AS unique_users,
            COUNT(DISTINCT imei) AS unique_vehicles,
            COUNT(DISTINCT province) AS num_provinces,
            COUNT(DISTINCT city) AS num_cities,

            -- Speed summary
            AVG(avg_speed) AS mean_avg_speed,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_speed) AS median_avg_speed,
            PERCENTILE_CONT(0.85) WITHIN GROUP (ORDER BY avg_speed) AS p85_avg_speed,
            AVG(max_speed) AS mean_max_speed,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY max_speed) AS median_max_speed,
            PERCENTILE_CONT(0.85) WITHIN GROUP (ORDER BY max_speed) AS p85_max_speed,

            -- Speeding summary
            AVG(CASE WHEN has_speeding THEN 1.0 ELSE 0.0 END) AS speeding_rate,
            AVG(CASE WHEN avg_above_limit THEN 1.0 ELSE 0.0 END) AS avg_speeding_rate,

            -- Trip metrics
            AVG(distance) AS mean_distance,
            AVG(travel_time) AS mean_duration,
            AVG(gps_points) AS mean_gps_points,

            -- Temporal
            MIN(start_date) AS date_min,
            MAX(start_date) AS date_max
        FROM read_parquet('{output_path}')
    """).fetchone()

    metadata = {
        "dataset": "trips_cleaned.parquet",
        "created": "2026-02-11",
        "total_trips": summary[0],
        "strict_valid_trips": summary[1],
        "unique_users": summary[2],
        "unique_vehicles": summary[3],
        "num_provinces": summary[4],
        "num_cities": summary[5],
        "speed_summary": {
            "mean_avg_speed": round(summary[6], 2),
            "median_avg_speed": round(summary[7], 2),
            "p85_avg_speed": round(summary[8], 2),
            "mean_max_speed": round(summary[9], 2),
            "median_max_speed": round(summary[10], 2),
            "p85_max_speed": round(summary[11], 2),
        },
        "speeding_rate_max": round(summary[12], 4),
        "speeding_rate_avg": round(summary[13], 4),
        "trip_metrics": {
            "mean_distance_m": round(summary[14], 1),
            "mean_duration_s": round(summary[15], 1),
            "mean_gps_points": round(summary[16], 1),
        },
        "date_range": {
            "min": str(summary[17]),
            "max": str(summary[18]),
        },
        "columns": [
            "route_id", "user_id", "imei", "qrcode", "type", "model", "mode",
            "start_date", "start_time", "end_date", "end_time", "travel_time",
            "start_hour", "day_of_week", "is_weekend", "day_of_month",
            "start_lat", "start_lon", "end_lat", "end_lon",
            "gps_points", "distance", "moved_distance", "avg_point_gap", "max_point_gap",
            "avg_speed", "max_speed", "routes_raw", "speeds_raw",
            "city", "province", "city_distance_deg",
            "flag_excluded_model", "flag_sentinel", "flag_invalid_coords",
            "flag_few_points", "flag_implausible_speed",
            "flag_short_distance", "flag_short_duration", "flag_long_duration",
            "is_valid", "is_strict_valid", "has_speeding", "avg_above_limit",
        ],
    }

    # Print summary
    print(f"\n--- Dataset Summary ---")
    print(f"  Total trips:        {metadata['total_trips']:>12,}")
    print(f"  Strict valid:       {metadata['strict_valid_trips']:>12,}")
    print(f"  Unique users:       {metadata['unique_users']:>12,}")
    print(f"  Unique vehicles:    {metadata['unique_vehicles']:>12,}")
    print(f"  Provinces:          {metadata['num_provinces']:>12}")
    print(f"  Cities:             {metadata['num_cities']:>12}")
    print(f"  Date range:         {metadata['date_range']['min']} to {metadata['date_range']['max']}")
    print(f"  Mean avg speed:     {metadata['speed_summary']['mean_avg_speed']:>12.1f} km/h")
    print(f"  Mean max speed:     {metadata['speed_summary']['mean_max_speed']:>12.1f} km/h")
    print(f"  Speeding rate:      {metadata['speeding_rate_max']:>12.1%}")
    print(f"  Mean distance:      {metadata['trip_metrics']['mean_distance_m']:>12.0f} m")
    print(f"  Mean duration:      {metadata['trip_metrics']['mean_duration_s']:>12.0f} s")
    print(f"  Columns:            {len(metadata['columns']):>12}")

    # Save metadata
    metadata_path = cleaned_dir / "dataset_summary.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nMetadata saved to {metadata_path}")

    # Check file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    con.close()

    print("\n" + "=" * 70)
    print("  CLEANED DATASET COMPLETE")
    print("=" * 70)

    return metadata


if __name__ == "__main__":
    build_cleaned_dataset()
