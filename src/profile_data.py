"""
Task 1.1: Profile full dataset schema and completeness.

Profiles both raw CSV files:
  - 2023_05_Swing_Routes.csv (GPS trajectories)
  - 2023_Swing_Scooter.csv (trip-level billing/demographics)

Outputs:
  - Console summary of schema, null rates, type issues, row counts
  - data_parquet/profile_routes.parquet — column-level stats for Routes
  - data_parquet/profile_scooter.parquet — column-level stats for Scooter
"""

import duckdb
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import ROUTES_CSV, SCOOTER_CSV, DATA_DIR


def profile_csv(con: duckdb.DuckDBPyConnection, csv_path: Path, label: str) -> dict:
    """Profile a CSV file using DuckDB for memory-efficient analysis.

    Args:
        con: DuckDB connection.
        csv_path: Path to the CSV file.
        label: Human-readable label for the dataset.

    Returns:
        Dictionary with schema, row count, null rates, and sample values.
    """
    print(f"\n{'='*70}")
    print(f"  PROFILING: {label}")
    print(f"  File: {csv_path}")
    print(f"{'='*70}")

    # --- Row count ---
    row_count = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{csv_path}')"
    ).fetchone()[0]
    print(f"\nTotal rows: {row_count:,}")

    # --- Schema detection ---
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto('{csv_path}')"
    ).fetchall()

    print(f"\nColumns ({len(schema)}):")
    print(f"  {'Column':<25} {'Type':<20} {'Nullable'}")
    print(f"  {'-'*25} {'-'*20} {'-'*10}")
    for col_name, col_type, nullable, *rest in schema:
        print(f"  {col_name:<25} {col_type:<20} {nullable}")

    # --- Null rates per column ---
    col_names = [row[0] for row in schema]
    col_types = {row[0]: row[1] for row in schema}

    null_exprs = ", ".join(
        [f"SUM(CASE WHEN \"{c}\" IS NULL THEN 1 ELSE 0 END) AS \"{c}_nulls\"" for c in col_names]
    )
    null_counts = con.execute(
        f"SELECT {null_exprs} FROM read_csv_auto('{csv_path}')"
    ).fetchone()

    print(f"\nNull rates:")
    print(f"  {'Column':<25} {'Nulls':>12} {'Rate':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10}")
    col_null_info = {}
    for i, col in enumerate(col_names):
        n = null_counts[i]
        rate = n / row_count if row_count > 0 else 0
        col_null_info[col] = {"null_count": n, "null_rate": rate}
        flag = " *** HIGH ***" if rate > 0.1 else ""
        print(f"  {col:<25} {n:>12,} {rate:>10.4%}{flag}")

    # --- Distinct counts for non-text columns (sampled) ---
    print(f"\nDistinct value counts (approx from first 100K rows):")
    for col in col_names:
        try:
            distinct = con.execute(
                f"SELECT approx_count_distinct(\"{col}\") FROM read_csv_auto('{csv_path}')"
            ).fetchone()[0]
            print(f"  {col:<25} ~{distinct:>12,}")
        except Exception as e:
            print(f"  {col:<25} ERROR: {e}")

    # --- Sample values (first 5 rows) ---
    print(f"\nSample values (first 3 rows):")
    sample = con.execute(
        f"SELECT * FROM read_csv_auto('{csv_path}') LIMIT 3"
    ).fetchdf()
    for idx, row in sample.iterrows():
        print(f"\n  --- Row {idx} ---")
        for col in col_names:
            val = str(row[col])
            if len(val) > 120:
                val = val[:120] + "..."
            print(f"    {col:<25}: {val}")

    # --- Basic stats for numeric columns ---
    print(f"\nNumeric column statistics:")
    numeric_types = {"BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "SMALLINT", "TINYINT", "HUGEINT"}
    for col in col_names:
        ctype = col_types[col].upper()
        if any(t in ctype for t in numeric_types):
            try:
                stats = con.execute(f"""
                    SELECT
                        MIN("{col}") as min_val,
                        MAX("{col}") as max_val,
                        AVG("{col}") as avg_val,
                        STDDEV("{col}") as std_val,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{col}") as median_val
                    FROM read_csv_auto('{csv_path}')
                    WHERE "{col}" IS NOT NULL
                """).fetchone()
                print(f"  {col:<25}: min={stats[0]}, max={stats[1]}, "
                      f"mean={stats[2]:.2f}, std={stats[3]:.2f}, median={stats[4]:.2f}")
            except Exception as e:
                print(f"  {col:<25}: ERROR computing stats — {e}")

    return {
        "label": label,
        "file": str(csv_path),
        "row_count": row_count,
        "column_count": len(schema),
        "columns": {r[0]: {"type": r[1], "nullable": r[2]} for r in schema},
        "null_info": col_null_info,
    }


def check_speed_column_structure(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Examine the structure of the speeds column in Routes CSV."""
    print(f"\n{'='*70}")
    print(f"  SPEEDS COLUMN DEEP-DIVE")
    print(f"{'='*70}")

    # Sample a few speed values to understand format
    speeds_sample = con.execute(f"""
        SELECT speeds FROM read_csv_auto('{csv_path}')
        WHERE speeds IS NOT NULL
        LIMIT 5
    """).fetchall()

    print("\nSample speed values (raw string):")
    for i, row in enumerate(speeds_sample):
        val = str(row[0])
        if len(val) > 200:
            val = val[:200] + "..."
        print(f"  [{i}] {val}")

    # Check length distribution
    print("\nSpeeds string length distribution:")
    len_stats = con.execute(f"""
        SELECT
            MIN(LENGTH(speeds)) as min_len,
            MAX(LENGTH(speeds)) as max_len,
            AVG(LENGTH(speeds)) as avg_len,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(speeds)) as median_len
        FROM read_csv_auto('{csv_path}')
        WHERE speeds IS NOT NULL
    """).fetchone()
    print(f"  min={len_stats[0]}, max={len_stats[1]}, mean={len_stats[2]:.0f}, median={len_stats[3]:.0f}")


def check_routes_column_structure(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Examine the structure of the routes column in Routes CSV."""
    print(f"\n{'='*70}")
    print(f"  ROUTES COLUMN DEEP-DIVE")
    print(f"{'='*70}")

    # Sample a few route values
    routes_sample = con.execute(f"""
        SELECT routes FROM read_csv_auto('{csv_path}')
        WHERE routes IS NOT NULL
        LIMIT 3
    """).fetchall()

    print("\nSample route values (raw string, truncated):")
    for i, row in enumerate(routes_sample):
        val = str(row[0])
        if len(val) > 300:
            val = val[:300] + "..."
        print(f"  [{i}] {val}")


def main() -> None:
    """Run full data profiling pipeline."""
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Profile Routes CSV
    routes_profile = profile_csv(con, ROUTES_CSV, "Swing Routes (GPS Trajectories)")
    check_speed_column_structure(con, ROUTES_CSV)
    check_routes_column_structure(con, ROUTES_CSV)

    # Profile Scooter CSV
    scooter_profile = profile_csv(con, SCOOTER_CSV, "Swing Scooter (Trips/Demographics)")

    # Save profiles as JSON for later reference
    profile_output = DATA_DIR / "data_profile.json"
    with open(profile_output, "w", encoding="utf-8") as f:
        json.dump(
            {"routes": routes_profile, "scooter": scooter_profile},
            f, indent=2, default=str
        )
    print(f"\nProfile saved to: {profile_output}")

    # --- Cross-check: are route IDs linkable between datasets? ---
    print(f"\n{'='*70}")
    print(f"  CROSS-DATASET LINKAGE CHECK")
    print(f"{'='*70}")

    # Get column names from both
    routes_cols = list(routes_profile["columns"].keys())
    scooter_cols = list(scooter_profile["columns"].keys())
    shared_cols = set(routes_cols) & set(scooter_cols)
    print(f"\nShared columns: {shared_cols if shared_cols else 'NONE'}")
    print(f"Routes columns: {routes_cols}")
    print(f"Scooter columns: {scooter_cols}")

    # Try to identify potential join keys
    for candidate in ["ride_id", "trip_id", "id", "user_id", "scooter_id"]:
        in_routes = candidate in routes_cols
        in_scooter = candidate in scooter_cols
        if in_routes or in_scooter:
            print(f"  {candidate}: Routes={'YES' if in_routes else 'NO'}, "
                  f"Scooter={'YES' if in_scooter else 'NO'}")

    con.close()
    print("\nProfiling complete.")


if __name__ == "__main__":
    main()
