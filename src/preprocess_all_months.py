"""
Preprocess all 24 monthly route CSVs (Jan 2022 - Dec 2023) into a single
consolidated Parquet dataset and build user longitudinal profiles.

Pipeline per month:
  1. Read CSV via DuckDB
  2. Replace sentinel -999 with NULL
  3. Exclude I9 model, invalid coords, few points, implausible speed
  4. Extract speed array from `speeds` column, compute mean/max/is_speeding
  5. Rename start_x -> start_lat, start_y -> start_lon
  6. Tag month_year

Outputs:
  - data_parquet/all_months/routes_all.parquet      -- consolidated trips
  - data_parquet/all_months/user_longitudinal.parquet -- per-user longitudinal profile
  - data_parquet/all_months/preprocessing_summary.json
"""

import duckdb
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR,
    MAX_PLAUSIBLE_SPEED,
    MIN_TRIP_POINTS,
    EXCLUDE_MODELS,
    SPEED_LIMIT_KR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR = Path("D:/SwingData/raw")

KOREA_LAT_MIN = 33.0
KOREA_LAT_MAX = 39.0
KOREA_LON_MIN = 124.0
KOREA_LON_MAX = 132.0

SENTINEL = -999

OUTPUT_DIR = DATA_DIR / "all_months"

EXCLUDE_MODELS_SQL = ", ".join(f"'{m}'" for m in EXCLUDE_MODELS)


def _month_parquet_path(year: int, month: int) -> Path:
    """Return temporary per-month parquet path."""
    return OUTPUT_DIR / "tmp" / f"{year}_{month:02d}.parquet"


def _build_filter_query(csv_path: str, year: int, month: int) -> str:
    """Build the DuckDB SQL that reads one monthly CSV, filters, and
    extracts speed-array indicators.

    Returns a SELECT query string (no trailing semicolon).
    """
    month_year = f"{year}-{month:02d}"
    # Use forward slashes for DuckDB path compatibility
    csv_fwd = csv_path.replace("\\", "/")

    query = f"""
    SELECT
        route_id,
        user_id,
        model,
        -- Normalize mode names: SCOOTER_TUB->TUB, SCOOTER_STD->STD, SCOOTER_ECO->ECO
        CASE
            WHEN mode = 'SCOOTER_TUB' THEN 'TUB'
            WHEN mode = 'SCOOTER_STD' THEN 'STD'
            WHEN mode = 'SCOOTER_ECO' THEN 'ECO'
            ELSE mode
        END AS mode,
        travel_time,
        start_date,
        start_time,
        start_x AS start_lat,
        start_y AS start_lon,
        points,
        CASE WHEN distance  = {SENTINEL} THEN NULL ELSE distance  END AS distance,
        CASE WHEN avg_speed = {SENTINEL} THEN NULL ELSE avg_speed END AS avg_speed,
        CASE WHEN max_speed = {SENTINEL} THEN NULL ELSE max_speed END AS max_speed,

        -- Parse speeds column: '[[s1, s2, ...]]' -> list of doubles
        -- Cast to VARCHAR first since some months have speeds as BIGINT (-999 sentinel)
        CASE
            WHEN CAST(speeds AS VARCHAR) = '-999' OR speeds IS NULL THEN NULL
            ELSE list_avg(
                list_transform(
                    regexp_extract_all(CAST(speeds AS VARCHAR), '(\\d+(?:\\.\\d+)?)'),
                    x -> TRY_CAST(x AS DOUBLE)
                )
            )
        END AS mean_speed_from_speeds,
        CASE
            WHEN CAST(speeds AS VARCHAR) = '-999' OR speeds IS NULL THEN NULL
            ELSE list_max(
                list_transform(
                    regexp_extract_all(CAST(speeds AS VARCHAR), '(\\d+(?:\\.\\d+)?)'),
                    x -> TRY_CAST(x AS DOUBLE)
                )
            )
        END AS max_speed_from_speeds,

        -- Speeding flag: use speeds array if available, otherwise fall back to max_speed column
        CASE
            WHEN CAST(speeds AS VARCHAR) != '-999' AND speeds IS NOT NULL
                 AND list_max(
                     list_transform(
                         regexp_extract_all(CAST(speeds AS VARCHAR), '(\\d+(?:\\.\\d+)?)'),
                         x -> TRY_CAST(x AS DOUBLE)
                     )
                 ) > {SPEED_LIMIT_KR} THEN TRUE
            WHEN (CAST(speeds AS VARCHAR) = '-999' OR speeds IS NULL)
                 AND max_speed != {SENTINEL} AND max_speed > {SPEED_LIMIT_KR} THEN TRUE
            ELSE FALSE
        END AS is_speeding,

        -- Flag whether speed data is available at all for this trip
        CASE WHEN avg_speed = {SENTINEL} AND max_speed = {SENTINEL}
             THEN FALSE ELSE TRUE END AS has_speed_data,

        -- Flag whether speeds array was available for this trip
        CASE WHEN CAST(speeds AS VARCHAR) = '-999' OR speeds IS NULL
             THEN FALSE ELSE TRUE END AS has_speeds_array,

        '{month_year}' AS month_year,

        -- Validity flag (same logic as filter_trips.py core filters)
        TRUE AS is_valid

    FROM read_csv_auto('{csv_fwd}', ignore_errors=true)

    WHERE
        -- Exclude I9 model
        model NOT IN ({EXCLUDE_MODELS_SQL})

        -- No sentinel in spatial/gap columns (always required)
        AND distance     != {SENTINEL}
        AND moved_distance != {SENTINEL}
        AND avg_point_gap != {SENTINEL}
        AND max_point_gap != {SENTINEL}

        -- Speed columns: allow sentinel through (2022 + 2023-01 have 100% sentinel)
        -- Trips with sentinel speed are kept for longitudinal tracking but flagged
        AND (avg_speed != {SENTINEL} OR avg_speed = {SENTINEL})

        -- Valid GPS coordinates (Korea bounds)
        AND start_x BETWEEN {KOREA_LAT_MIN} AND {KOREA_LAT_MAX}
        AND start_y BETWEEN {KOREA_LON_MIN} AND {KOREA_LON_MAX}
        AND end_x   BETWEEN {KOREA_LAT_MIN} AND {KOREA_LAT_MAX}
        AND end_y   BETWEEN {KOREA_LON_MIN} AND {KOREA_LON_MAX}

        -- Minimum GPS points
        AND points >= {MIN_TRIP_POINTS}

        -- Plausible max speed (skip check when speed is sentinel)
        AND (max_speed = {SENTINEL} OR max_speed <= {MAX_PLAUSIBLE_SPEED})
    """
    return query


def process_month(con: duckdb.DuckDBPyConnection,
                  csv_path: Path,
                  year: int,
                  month: int) -> dict:
    """Process a single monthly CSV and write to a temporary Parquet file.

    Args:
        con: DuckDB connection.
        csv_path: Path to the monthly CSV.
        year: Calendar year (e.g. 2022).
        month: Calendar month (1-12).

    Returns:
        dict with processing statistics for this month.
    """
    t0 = time.time()

    csv_str = str(csv_path)

    # Total rows in the raw CSV
    total_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{csv_str.replace(chr(92), '/')}', ignore_errors=true)"
    ).fetchone()[0]

    # Build and execute filtered query -> temp parquet
    select_query = _build_filter_query(csv_str, year, month)

    out_path = _month_parquet_path(year, month)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fwd = str(out_path).replace("\\", "/")

    copy_query = f"""
    COPY (
        SELECT
            route_id, user_id, model, mode, travel_time,
            start_date, start_time, start_lat, start_lon,
            points, distance, avg_speed, max_speed,
            mean_speed_from_speeds, max_speed_from_speeds,
            is_speeding, has_speed_data, has_speeds_array,
            month_year, is_valid
        FROM ({select_query})
    )
    TO '{out_fwd}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(copy_query)

    valid_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_fwd}')"
    ).fetchone()[0]

    elapsed = time.time() - t0

    stats = {
        "year": year,
        "month": month,
        "month_year": f"{year}-{month:02d}",
        "csv_file": csv_path.name,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "filtered_out": total_rows - valid_rows,
        "retention_rate": valid_rows / total_rows if total_rows > 0 else 0.0,
        "elapsed_sec": round(elapsed, 1),
    }

    print(f"  {year}-{month:02d}: {total_rows:>10,} raw -> "
          f"{valid_rows:>10,} valid ({stats['retention_rate']:.1%}) "
          f"[{elapsed:.1f}s]")

    return stats


def union_all_months(con: duckdb.DuckDBPyConnection,
                     month_stats: list[dict]) -> int:
    """UNION ALL temporary monthly Parquets into a single consolidated file.

    Args:
        con: DuckDB connection.
        month_stats: List of per-month stats dicts (to get file paths).

    Returns:
        Total row count in the consolidated file.
    """
    print("\nConsolidating all months into a single Parquet ...")
    t0 = time.time()

    # Build UNION ALL of all temp parquets
    parquet_paths = []
    for ms in month_stats:
        ppath = _month_parquet_path(ms["year"], ms["month"])
        parquet_paths.append(str(ppath).replace("\\", "/"))

    # DuckDB can read a list of parquet files directly
    paths_list = ", ".join(f"'{p}'" for p in parquet_paths)

    out_path = OUTPUT_DIR / "routes_all.parquet"
    out_fwd = str(out_path).replace("\\", "/")

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet([{paths_list}])
            ORDER BY user_id, start_date, start_time
        )
        TO '{out_fwd}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
    """)

    total = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_fwd}')"
    ).fetchone()[0]

    elapsed = time.time() - t0
    print(f"  Consolidated: {total:,} rows, written in {elapsed:.1f}s")
    print(f"  Output: {out_path}")

    return total


def build_user_longitudinal(con: duckdb.DuckDBPyConnection) -> int:
    """Build user-level longitudinal profile from the consolidated Parquet.

    Columns:
      - user_id
      - first_trip_date, last_trip_date
      - total_trips
      - months_active (distinct month_year count)
      - total_speeding_trips
      - speeding_rate
      - mean_max_speed, mean_avg_speed

    Returns:
        Number of unique users.
    """
    print("\nBuilding user longitudinal profiles ...")
    t0 = time.time()

    routes_fwd = str(OUTPUT_DIR / "routes_all.parquet").replace("\\", "/")
    out_path = OUTPUT_DIR / "user_longitudinal.parquet"
    out_fwd = str(out_path).replace("\\", "/")

    con.execute(f"""
        COPY (
            SELECT
                user_id,
                MIN(start_date)            AS first_trip_date,
                MAX(start_date)            AS last_trip_date,
                COUNT(*)                   AS total_trips,
                COUNT(DISTINCT month_year) AS months_active,
                SUM(CASE WHEN is_speeding THEN 1 ELSE 0 END) AS total_speeding_trips,
                ROUND(
                    SUM(CASE WHEN is_speeding THEN 1 ELSE 0 END) * 1.0 / COUNT(*),
                    4
                ) AS speeding_rate,
                ROUND(AVG(max_speed), 2)   AS mean_max_speed,
                ROUND(AVG(avg_speed), 2)   AS mean_avg_speed
            FROM read_parquet('{routes_fwd}')
            GROUP BY user_id
            ORDER BY total_trips DESC
        )
        TO '{out_fwd}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    n_users = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_fwd}')"
    ).fetchone()[0]

    elapsed = time.time() - t0
    print(f"  Users: {n_users:,} ({elapsed:.1f}s)")
    print(f"  Output: {out_path}")

    return n_users


def cleanup_temp(month_stats: list[dict]) -> None:
    """Delete temporary per-month Parquet files."""
    print("\nCleaning up temporary files ...")
    tmp_dir = OUTPUT_DIR / "tmp"
    for ms in month_stats:
        ppath = _month_parquet_path(ms["year"], ms["month"])
        if ppath.exists():
            ppath.unlink()
    if tmp_dir.exists():
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    """Run the full multi-month preprocessing pipeline."""
    overall_t0 = time.time()

    print("=" * 70)
    print("  MULTI-MONTH PREPROCESSING (Jan 2022 - Dec 2023)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover monthly CSV files
    csv_files: list[tuple[int, int, Path]] = []
    for year in (2022, 2023):
        for month in range(1, 13):
            fname = f"{year}_{month:02d}_Swing_Routes.csv"
            fpath = RAW_DIR / fname
            if fpath.exists():
                csv_files.append((year, month, fpath))

    if not csv_files:
        print(f"ERROR: No CSV files found in {RAW_DIR}")
        print("Expected pattern: YYYY_MM_Swing_Routes.csv")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} monthly CSV files in {RAW_DIR}")

    # Process each month
    con = duckdb.connect()
    # Increase memory limit for large files
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 4")

    month_stats: list[dict] = []
    print("\nProcessing months:")

    for year, month, csv_path in csv_files:
        try:
            stats = process_month(con, csv_path, year, month)
            month_stats.append(stats)
        except Exception as e:
            print(f"  ERROR processing {year}-{month:02d}: {e}")
            month_stats.append({
                "year": year,
                "month": month,
                "month_year": f"{year}-{month:02d}",
                "csv_file": csv_path.name,
                "total_rows": 0,
                "valid_rows": 0,
                "filtered_out": 0,
                "retention_rate": 0.0,
                "elapsed_sec": 0.0,
                "error": str(e),
            })

    # Union all months
    successful = [ms for ms in month_stats if "error" not in ms and ms["valid_rows"] > 0]
    if not successful:
        print("\nERROR: No months processed successfully.")
        con.close()
        sys.exit(1)

    total_consolidated = union_all_months(con, successful)

    # Build user longitudinal profiles
    n_users = build_user_longitudinal(con)

    con.close()

    # Cleanup temp files
    cleanup_temp(successful)

    # Summary
    total_raw = sum(ms["total_rows"] for ms in month_stats)
    total_valid = sum(ms["valid_rows"] for ms in month_stats)
    total_elapsed = time.time() - overall_t0

    summary = {
        "months_found": len(csv_files),
        "months_processed": len(successful),
        "months_failed": len(csv_files) - len(successful),
        "total_raw_rows": total_raw,
        "total_valid_rows": total_valid,
        "overall_retention_rate": total_valid / total_raw if total_raw > 0 else 0.0,
        "consolidated_rows": total_consolidated,
        "unique_users": n_users,
        "total_elapsed_sec": round(total_elapsed, 1),
        "per_month": month_stats,
    }

    summary_path = OUTPUT_DIR / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  MULTI-MONTH PREPROCESSING COMPLETE")
    print(f"  Months: {len(successful)}/{len(csv_files)}")
    print(f"  Raw rows:   {total_raw:>12,}")
    print(f"  Valid rows: {total_valid:>12,} ({summary['overall_retention_rate']:.1%})")
    print(f"  Users:      {n_users:>12,}")
    print(f"  Runtime:    {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  Output:     {OUTPUT_DIR / 'routes_all.parquet'}")
    print(f"  Summary:    {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
