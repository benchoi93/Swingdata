"""
Phase 0: Build unified Feb-Nov 2023 dataset with all 8 primary behavioral outcomes.

Tasks 0.1-0.2: Filter routes_all.parquet to Feb-Nov 2023, then compute trip-level
indicators from raw CSVs using DuckDB SQL (much faster than Python ast.literal_eval).

Pipeline:
  1. Filter routes_all.parquet -> trips_feb_nov.parquet (metadata only)
  2. For each month, read raw CSV with DuckDB, parse speed arrays in SQL,
     compute all indicators (harsh events, speed_cv, cruise_fraction, etc.)
  3. Combine all months -> trip_indicators_v2.parquet

Outputs:
  - data_parquet/v2/trips_feb_nov.parquet  (trip metadata, ~21.7M rows)
  - data_parquet/v2/trip_indicators_v2.parquet  (all behavioral indicators)
"""

import duckdb
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_DIR,
    MAX_PLAUSIBLE_SPEED,
    MIN_TRIP_POINTS,
    EXCLUDE_MODELS,
    SPEED_LIMIT_KR,
    HARSH_ACCEL_THRESHOLD_10S,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR = Path("D:/SwingData/raw")
V2_DIR = DATA_DIR / "v2"
ALL_MONTHS_DIR = DATA_DIR / "all_months"

KOREA_LAT_MIN, KOREA_LAT_MAX = 33.0, 39.0
KOREA_LON_MIN, KOREA_LON_MAX = 124.0, 132.0
SENTINEL = -999

MONTHS_FEB_NOV = [(2023, m) for m in range(2, 12)]  # Feb=2 through Nov=11
MONTHS_FEB_DEC = [(2023, m) for m in range(2, 13)]   # Feb=2 through Dec=12

EXCLUDE_MODELS_SQL = ", ".join(f"'{m}'" for m in EXCLUDE_MODELS)

# Speed interval between consecutive readings (~10s)
SPEED_INTERVAL_S = 10.0
ACCEL_FACTOR = 1000.0 / 3600.0 / SPEED_INTERVAL_S  # km/h -> m/s^2 at 10s interval
HARSH_THRESH = HARSH_ACCEL_THRESHOLD_10S  # 0.5 m/s^2


def _raw_csv_path(year: int, month: int) -> Path:
    return RAW_DIR / f"{year}_{month:02d}_Swing_Routes.csv"


def _build_indicator_query(csv_path: str, year: int, month: int) -> str:
    """Build DuckDB SQL that reads a monthly CSV, filters, parses speed arrays,
    and computes ALL trip-level behavioral indicators in a single pass.
    """
    csv_fwd = csv_path.replace("\\", "/")
    month_year = f"{year}-{month:02d}"

    return f"""
    WITH raw_parsed AS (
        SELECT
            route_id,
            user_id,
            model,
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
            CASE WHEN distance = {SENTINEL} THEN NULL ELSE distance END AS distance,
            '{month_year}' AS month_year,

            -- Parse speeds column into a list of doubles
            list_transform(
                regexp_extract_all(CAST(speeds AS VARCHAR), '(\\d+(?:\\.\\d+)?)'),
                x -> TRY_CAST(x AS DOUBLE)
            ) AS speed_arr

        FROM read_csv_auto('{csv_fwd}', ignore_errors=true)

        WHERE
            model NOT IN ({EXCLUDE_MODELS_SQL})
            AND distance     != {SENTINEL}
            AND moved_distance != {SENTINEL}
            AND avg_point_gap != {SENTINEL}
            AND max_point_gap != {SENTINEL}
            AND start_x BETWEEN {KOREA_LAT_MIN} AND {KOREA_LAT_MAX}
            AND start_y BETWEEN {KOREA_LON_MIN} AND {KOREA_LON_MAX}
            AND end_x   BETWEEN {KOREA_LAT_MIN} AND {KOREA_LAT_MAX}
            AND end_y   BETWEEN {KOREA_LON_MIN} AND {KOREA_LON_MAX}
            AND points >= {MIN_TRIP_POINTS}
            AND (max_speed = {SENTINEL} OR max_speed <= {MAX_PLAUSIBLE_SPEED})
            -- Only trips with speed arrays
            AND CAST(speeds AS VARCHAR) != '-999'
            AND speeds IS NOT NULL
    ),

    with_stats AS (
        SELECT
            *,
            len(speed_arr) AS n_pts,
            list_avg(speed_arr) AS mean_spd,
            list_max(speed_arr) AS max_spd,
            list_min(speed_arr) AS min_spd
        FROM raw_parsed
        WHERE len(speed_arr) >= 2
    ),

    with_derived AS (
        SELECT
            *,
            -- Speed std: sqrt(E[x^2] - E[x]^2)
            sqrt(GREATEST(
                list_avg(list_transform(speed_arr, x -> x * x)) - mean_spd * mean_spd,
                0.0
            )) AS speed_std_val,

            -- P85: sort and pick 85th percentile index
            list_sort(speed_arr)[CAST(CEIL(n_pts * 0.85) AS INTEGER)] AS p85_spd,

            -- Acceleration array: consecutive differences in m/s^2
            list_transform(
                range(1, n_pts),
                i -> (speed_arr[i + 1] - speed_arr[i]) * {ACCEL_FACTOR}
            ) AS accel_arr

        FROM with_stats
    )

    SELECT
        route_id,
        user_id,
        model,
        mode,
        travel_time,
        start_date,
        start_time,
        start_lat,
        start_lon,
        points,
        distance,
        month_year,

        -- Speed indicators (manipulation check)
        n_pts AS n_speed_points,
        ROUND(mean_spd, 2) AS mean_speed,
        ROUND(max_spd, 2) AS max_speed_from_profile,
        ROUND(min_spd, 2) AS min_speed,
        ROUND(p85_spd, 2) AS p85_speed,

        -- PRIMARY OUTCOME: Speed variability (Safety)
        ROUND(speed_std_val, 2) AS speed_std,
        ROUND(
            CASE WHEN mean_spd > 0 THEN speed_std_val / mean_spd ELSE 0.0 END,
            4
        ) AS speed_cv,

        -- Speeding flags (for legacy/survival)
        CASE WHEN max_spd > {SPEED_LIMIT_KR} THEN TRUE ELSE FALSE END AS is_speeding,
        ROUND(
            list_count(list_filter(speed_arr, x -> x > {SPEED_LIMIT_KR})) * 1.0 / n_pts,
            4
        ) AS speeding_rate,

        -- PRIMARY OUTCOME: Harsh events (Safety)
        list_count(list_filter(accel_arr, x -> x > {HARSH_THRESH}))
            AS harsh_accel_count,
        list_count(list_filter(accel_arr, x -> x < -{HARSH_THRESH}))
            AS harsh_decel_count,
        list_count(list_filter(accel_arr, x -> x > {HARSH_THRESH}))
          + list_count(list_filter(accel_arr, x -> x < -{HARSH_THRESH}))
            AS harsh_event_count,

        -- PRIMARY OUTCOME: Cruising behavior (Dynamics)
        ROUND(
            list_count(list_filter(speed_arr, x -> abs(x - mean_spd) <= 3.0)) * 1.0 / n_pts,
            4
        ) AS cruise_fraction,

        -- PRIMARY OUTCOME: Zero-speed fraction (Dynamics)
        ROUND(
            list_count(list_filter(speed_arr, x -> x = 0.0)) * 1.0 / n_pts,
            4
        ) AS zero_speed_fraction

    FROM with_derived
    """


def task_01_filter_trips(con: duckdb.DuckDBPyConnection) -> int:
    """Task 0.1: Filter routes_all.parquet to Feb-Nov 2023 with speed data."""
    print("=" * 60)
    print("Task 0.1: Filter routes_all.parquet to Feb-Nov 2023")
    print("=" * 60)

    routes_path = str(ALL_MONTHS_DIR / "routes_all.parquet").replace("\\", "/")
    out_path = str(V2_DIR / "trips_feb_nov.parquet").replace("\\", "/")

    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet('{routes_path}')
            WHERE has_speed_data = TRUE
              AND month_year IN (
                  '2023-02','2023-03','2023-04','2023-05','2023-06',
                  '2023-07','2023-08','2023-09','2023-10','2023-11'
              )
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path}')"
    ).fetchone()[0]

    elapsed = time.time() - t0
    print(f"  Rows: {count:,}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {out_path}")
    return count


def task_02_compute_indicators(con: duckdb.DuckDBPyConnection) -> int:
    """Task 0.2: Compute trip-level indicators from raw CSVs using DuckDB SQL."""
    print("=" * 60)
    print("Task 0.2: Compute trip-level indicators (Feb-Nov 2023)")
    print("=" * 60)

    out_path = str(V2_DIR / "trip_indicators_v2.parquet").replace("\\", "/")
    tmp_dir = V2_DIR / "tmp_indicators"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    month_stats = []
    total_rows = 0

    for year, month in MONTHS_FEB_NOV:
        csv_path = _raw_csv_path(year, month)
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        month_label = f"{year}-{month:02d}"
        tmp_path = str(tmp_dir / f"{month_label}.parquet").replace("\\", "/")

        print(f"\n  Processing {month_label}...", end=" ", flush=True)
        t0 = time.time()

        query = _build_indicator_query(str(csv_path), year, month)
        con.execute(f"""
            COPY ({query}) TO '{tmp_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{tmp_path}')"
        ).fetchone()[0]

        elapsed = time.time() - t0
        total_rows += count
        month_stats.append({
            "month": month_label,
            "rows": count,
            "time_s": round(elapsed, 1),
        })
        print(f"{count:,} rows in {elapsed:.1f}s")

    # Combine all months into one parquet
    print(f"\n  Combining {len(month_stats)} months...")
    t0 = time.time()

    tmp_glob = str(tmp_dir / "*.parquet").replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{tmp_glob}')
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    elapsed = time.time() - t0
    print(f"  Combined: {total_rows:,} rows in {elapsed:.1f}s")
    print(f"  Output: {out_path}")

    # Clean up tmp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return total_rows


def task_02_verify(con: duckdb.DuckDBPyConnection) -> dict:
    """Quick verification of computed indicators."""
    path = str(V2_DIR / "trip_indicators_v2.parquet").replace("\\", "/")

    stats = con.execute(f"""
        SELECT
            COUNT(*) AS n_trips,
            COUNT(DISTINCT user_id) AS n_users,
            COUNT(DISTINCT month_year) AS n_months,

            -- Mode distribution
            SUM(CASE WHEN mode = 'TUB' THEN 1 ELSE 0 END) AS n_tub,
            SUM(CASE WHEN mode = 'STD' THEN 1 ELSE 0 END) AS n_std,
            SUM(CASE WHEN mode = 'ECO' THEN 1 ELSE 0 END) AS n_eco,

            -- Safety outcomes
            ROUND(AVG(harsh_accel_count), 3) AS mean_harsh_accel,
            ROUND(AVG(harsh_decel_count), 3) AS mean_harsh_decel,
            ROUND(AVG(speed_cv), 4) AS mean_speed_cv,

            -- Demand outcomes
            ROUND(AVG(distance), 1) AS mean_distance,

            -- Dynamics outcomes
            ROUND(AVG(cruise_fraction), 4) AS mean_cruise_fraction,
            ROUND(AVG(zero_speed_fraction), 4) AS mean_zero_speed_fraction,

            -- Manipulation check
            ROUND(AVG(mean_speed), 2) AS mean_mean_speed,
            ROUND(AVG(CASE WHEN is_speeding THEN 1.0 ELSE 0.0 END), 4) AS speeding_rate

        FROM read_parquet('{path}')
    """).fetchone()

    cols = [
        "n_trips", "n_users", "n_months",
        "n_tub", "n_std", "n_eco",
        "mean_harsh_accel", "mean_harsh_decel", "mean_speed_cv",
        "mean_distance",
        "mean_cruise_fraction", "mean_zero_speed_fraction",
        "mean_mean_speed", "speeding_rate",
    ]
    result = dict(zip(cols, stats))

    print("\n  Verification:")
    for k, v in result.items():
        print(f"    {k}: {v:,}" if isinstance(v, int) else f"    {k}: {v}")

    return result


def main():
    print("Phase 0: V2 Data Preparation")
    print("=" * 60)

    V2_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    # Allow DuckDB to use available memory
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 8")

    t_start = time.time()

    # Task 0.1
    n_trips_meta = task_01_filter_trips(con)

    # Task 0.2
    n_trips_indicators = task_02_compute_indicators(con)

    # Quick verify
    verify_stats = task_02_verify(con)

    # Save summary
    summary = {
        "task_0_1_trips_feb_nov": n_trips_meta,
        "task_0_2_trip_indicators": n_trips_indicators,
        "verification": verify_stats,
        "total_time_s": round(time.time() - t_start, 1),
    }

    summary_path = V2_DIR / "data_prep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Phase 0 complete in {summary['total_time_s']:.0f}s")
    print(f"  trips_feb_nov.parquet: {n_trips_meta:,} rows")
    print(f"  trip_indicators_v2.parquet: {n_trips_indicators:,} rows")
    print(f"  Summary: {summary_path}")

    con.close()


if __name__ == "__main__":
    main()
