"""
Task 1.2: Filter GPS error trips and build quality-flagged dataset.

Filtering criteria:
  1. Remove I9 model trips (moped type, GPS errors with max_speed > 50 km/h)
  2. Replace sentinel value -999 with NULL in numeric columns
  3. Remove trips with zero/invalid GPS coordinates (outside Korea bounds)
  4. Remove trips with < 5 GPS points
  5. Remove trips with max_speed > 50 km/h (implausible for e-scooters)
  6. Flag (but retain separately) trips with short distance (<100m) or duration (<60s)
  7. Flag trips with very long duration (>2 hours)

Outputs:
  - data_parquet/routes_filtered.parquet — all trips with quality flags
  - data_parquet/filtering_report.json — filtering statistics
"""

import duckdb
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    ROUTES_CSV, DATA_DIR,
    MAX_PLAUSIBLE_SPEED, MIN_TRIP_POINTS, MIN_TRIP_DISTANCE,
    MIN_TRIP_DURATION, EXCLUDE_MODELS,
)

# Korean geographic bounds (latitude, longitude)
KOREA_LAT_MIN = 33.0
KOREA_LAT_MAX = 39.0
KOREA_LON_MIN = 124.0
KOREA_LON_MAX = 132.0

# Maximum plausible trip duration (seconds) — 2 hours
MAX_TRIP_DURATION = 7200

# Sentinel value used instead of NULL in the raw data
SENTINEL = -999

# Columns that may contain sentinel -999
SENTINEL_COLUMNS = [
    "distance", "moved_distance", "avg_point_gap",
    "max_point_gap", "avg_speed", "max_speed",
]


def run_filtering() -> dict:
    """Run the full filtering pipeline using DuckDB.

    Returns:
        Dictionary with filtering statistics.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    csv_path = str(ROUTES_CSV)
    exclude_models_str = ", ".join(f"'{m}'" for m in EXCLUDE_MODELS)

    print("=" * 70)
    print("  TASK 1.2: GPS ERROR FILTERING")
    print("=" * 70)

    # Step 1: Get total row count
    total_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{csv_path}')"
    ).fetchone()[0]
    print(f"\nTotal rows in raw CSV: {total_rows:,}")

    # Step 2: Build filtered table with quality flags in DuckDB
    # We create a view that replaces sentinels and adds quality flags
    print("\nCreating filtered dataset with quality flags...")

    query = f"""
    CREATE TABLE filtered AS
    SELECT
        -- Original columns (with sentinel replacement)
        route_id,
        user_id,
        imei,
        qrcode,
        type,
        model,
        mode,
        travel_time,
        start_date,
        start_time,
        start_x,
        start_y,
        end_date,
        end_time,
        end_x,
        end_y,
        points,
        CASE WHEN distance = {SENTINEL} THEN NULL ELSE distance END AS distance,
        CASE WHEN moved_distance = {SENTINEL} THEN NULL ELSE moved_distance END AS moved_distance,
        CASE WHEN avg_point_gap = {SENTINEL} THEN NULL ELSE avg_point_gap END AS avg_point_gap,
        CASE WHEN max_point_gap = {SENTINEL} THEN NULL ELSE max_point_gap END AS max_point_gap,
        routes,
        CASE WHEN avg_speed = {SENTINEL} THEN NULL ELSE avg_speed END AS avg_speed,
        CASE WHEN max_speed = {SENTINEL} THEN NULL ELSE max_speed END AS max_speed,
        speeds,

        -- Quality flag: excluded model (I9)
        CASE WHEN model IN ({exclude_models_str}) THEN TRUE ELSE FALSE END
            AS flag_excluded_model,

        -- Quality flag: sentinel value present in any column
        CASE WHEN (distance = {SENTINEL} OR moved_distance = {SENTINEL}
                    OR avg_point_gap = {SENTINEL} OR max_point_gap = {SENTINEL}
                    OR avg_speed = {SENTINEL} OR max_speed = {SENTINEL})
             THEN TRUE ELSE FALSE END
            AS flag_sentinel,

        -- Quality flag: invalid GPS coordinates
        CASE WHEN (start_x < {KOREA_LAT_MIN} OR start_x > {KOREA_LAT_MAX}
                    OR start_y < {KOREA_LON_MIN} OR start_y > {KOREA_LON_MAX}
                    OR end_x < {KOREA_LAT_MIN} OR end_x > {KOREA_LAT_MAX}
                    OR end_y < {KOREA_LON_MIN} OR end_y > {KOREA_LON_MAX})
             THEN TRUE ELSE FALSE END
            AS flag_invalid_coords,

        -- Quality flag: too few GPS points
        CASE WHEN points < {MIN_TRIP_POINTS} THEN TRUE ELSE FALSE END
            AS flag_few_points,

        -- Quality flag: implausible max speed
        CASE WHEN (max_speed > {MAX_PLAUSIBLE_SPEED} AND max_speed != {SENTINEL})
             THEN TRUE ELSE FALSE END
            AS flag_implausible_speed,

        -- Quality flag: short distance
        CASE WHEN (distance IS NOT NULL AND distance != {SENTINEL}
                    AND distance < {MIN_TRIP_DISTANCE} AND distance >= 0)
             THEN TRUE ELSE FALSE END
            AS flag_short_distance,

        -- Quality flag: short duration
        CASE WHEN travel_time < {MIN_TRIP_DURATION}
             THEN TRUE ELSE FALSE END
            AS flag_short_duration,

        -- Quality flag: very long duration
        CASE WHEN travel_time > {MAX_TRIP_DURATION}
             THEN TRUE ELSE FALSE END
            AS flag_long_duration,

        -- Master validity flag: trip passes all CORE quality filters
        CASE WHEN (
            model NOT IN ({exclude_models_str})
            AND NOT (distance = {SENTINEL} OR moved_distance = {SENTINEL}
                     OR avg_point_gap = {SENTINEL} OR max_point_gap = {SENTINEL}
                     OR avg_speed = {SENTINEL} OR max_speed = {SENTINEL})
            AND NOT (start_x < {KOREA_LAT_MIN} OR start_x > {KOREA_LAT_MAX}
                     OR start_y < {KOREA_LON_MIN} OR start_y > {KOREA_LON_MAX}
                     OR end_x < {KOREA_LAT_MIN} OR end_x > {KOREA_LAT_MAX}
                     OR end_y < {KOREA_LON_MIN} OR end_y > {KOREA_LON_MAX})
            AND points >= {MIN_TRIP_POINTS}
            AND NOT (max_speed > {MAX_PLAUSIBLE_SPEED} AND max_speed != {SENTINEL})
        ) THEN TRUE ELSE FALSE END
            AS is_valid

    FROM read_csv_auto('{csv_path}')
    """

    con.execute(query)

    # Step 3: Compute filtering statistics
    print("\nComputing filtering statistics...")

    stats = {}
    stats["total_rows"] = total_rows

    # Count per flag
    flag_cols = [
        "flag_excluded_model", "flag_sentinel", "flag_invalid_coords",
        "flag_few_points", "flag_implausible_speed", "flag_short_distance",
        "flag_short_duration", "flag_long_duration",
    ]

    for flag in flag_cols:
        count = con.execute(
            f"SELECT COUNT(*) FROM filtered WHERE {flag} = TRUE"
        ).fetchone()[0]
        rate = count / total_rows
        stats[flag] = {"count": count, "rate": rate}
        print(f"  {flag:<30}: {count:>10,} ({rate:.4%})")

    # Count valid trips
    valid_count = con.execute(
        "SELECT COUNT(*) FROM filtered WHERE is_valid = TRUE"
    ).fetchone()[0]
    stats["valid_trips"] = valid_count
    stats["valid_rate"] = valid_count / total_rows
    print(f"\n  Valid trips (core filters):   {valid_count:>10,} ({valid_count/total_rows:.4%})")

    # Stricter valid (core + distance + duration)
    strict_count = con.execute("""
        SELECT COUNT(*) FROM filtered
        WHERE is_valid = TRUE
            AND flag_short_distance = FALSE
            AND flag_short_duration = FALSE
            AND flag_long_duration = FALSE
    """).fetchone()[0]
    stats["strict_valid_trips"] = strict_count
    stats["strict_valid_rate"] = strict_count / total_rows
    print(f"  Strict valid (+ dist/dur):   {strict_count:>10,} ({strict_count/total_rows:.4%})")

    # Breakdown by model
    print("\n  Filtering by model:")
    model_stats = con.execute("""
        SELECT
            model,
            COUNT(*) as total,
            SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) as valid,
            AVG(CASE WHEN is_valid THEN max_speed ELSE NULL END) as avg_max_speed
        FROM filtered
        GROUP BY model
        ORDER BY total DESC
    """).fetchdf()

    for _, row in model_stats.iterrows():
        model = row["model"]
        total = row["total"]
        valid = row["valid"]
        pct = valid / total * 100 if total > 0 else 0
        avg_ms = row["avg_max_speed"]
        avg_ms_str = f"{avg_ms:.1f}" if avg_ms is not None else "N/A"
        print(f"    {model:<6}: {total:>10,} total, {valid:>10,} valid ({pct:.1f}%), "
              f"avg max_speed={avg_ms_str}")

    stats["by_model"] = {
        row["model"]: {
            "total": int(row["total"]),
            "valid": int(row["valid"]),
        }
        for _, row in model_stats.iterrows()
    }

    # Breakdown by type
    print("\n  Filtering by type:")
    type_stats = con.execute("""
        SELECT
            type,
            COUNT(*) as total,
            SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) as valid
        FROM filtered
        GROUP BY type
        ORDER BY total DESC
    """).fetchdf()

    for _, row in type_stats.iterrows():
        vtype = row["type"]
        total = row["total"]
        valid = row["valid"]
        pct = valid / total * 100 if total > 0 else 0
        print(f"    {vtype:<10}: {total:>10,} total, {valid:>10,} valid ({pct:.1f}%)")

    stats["by_type"] = {
        row["type"]: {
            "total": int(row["total"]),
            "valid": int(row["valid"]),
        }
        for _, row in type_stats.iterrows()
    }

    # Step 4: Export to Parquet
    output_path = DATA_DIR / "routes_filtered.parquet"
    print(f"\nExporting to {output_path} ...")

    con.execute(f"""
        COPY (SELECT * FROM filtered)
        TO '{output_path}'
        (FORMAT 'parquet', COMPRESSION 'zstd')
    """)

    # Verify output
    verify_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
    ).fetchone()[0]
    print(f"Parquet file rows: {verify_count:,} (expected {total_rows:,})")
    assert verify_count == total_rows, "Row count mismatch!"

    # Also export valid-only subset for convenience
    valid_output_path = DATA_DIR / "routes_valid.parquet"
    print(f"Exporting valid-only subset to {valid_output_path} ...")

    con.execute(f"""
        COPY (SELECT * FROM filtered WHERE is_valid = TRUE)
        TO '{valid_output_path}'
        (FORMAT 'parquet', COMPRESSION 'zstd')
    """)

    valid_verify = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{valid_output_path}')"
    ).fetchone()[0]
    print(f"Valid-only Parquet rows: {valid_verify:,}")

    # Save filtering report
    report_path = DATA_DIR / "filtering_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nFiltering report saved to {report_path}")

    con.close()

    print("\n" + "=" * 70)
    print("  FILTERING COMPLETE")
    print(f"  Total: {total_rows:,} -> Valid: {valid_count:,} "
          f"({valid_count/total_rows:.1%} retained)")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    run_filtering()
