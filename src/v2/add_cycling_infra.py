"""
Task 7A.1: Add is_cycling_infra column to trip_modeling.parquet.

Derives from existing start_road_class == 'cycleway'. Simple binary indicator
for whether the trip started on cycling infrastructure (OSM cycleway tag).
"""

import duckdb
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR

V2_DIR = DATA_DIR / "v2"


def main() -> None:
    print("=" * 60)
    print("Task 7A.1: Add is_cycling_infra to trip_modeling.parquet")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 8")

    in_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")
    out_path = str(V2_DIR / "trip_modeling_v2.parquet").replace("\\", "/")

    # Check current columns
    cols = con.execute(f"SELECT name FROM parquet_schema('{in_path}')").fetchall()
    col_names = [c[0] for c in cols]
    print(f"  Current columns: {len(col_names)}")

    if "is_cycling_infra" in col_names:
        print("  is_cycling_infra already exists. Skipping.")
        con.close()
        return

    # Check road class distribution first
    print("\n  Road class distribution:")
    dist = con.execute(f"""
        SELECT start_road_class, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
        FROM read_parquet('{in_path}')
        GROUP BY start_road_class
        ORDER BY n DESC
    """).fetchdf()
    print(dist.to_string(index=False))

    # Add is_cycling_infra column
    print("\n  Adding is_cycling_infra column...")
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT *,
                CASE WHEN start_road_class = 'cycleway' THEN TRUE ELSE FALSE END
                    AS is_cycling_infra
            FROM read_parquet('{in_path}')
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"  Written to temp file in {time.time() - t0:.1f}s")

    # Verify
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    n_cycling = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{out_path}') WHERE is_cycling_infra = TRUE
    """).fetchone()[0]
    pct_cycling = 100.0 * n_cycling / n_total

    print(f"\n  Verification:")
    print(f"    Total trips: {n_total:,}")
    print(f"    Cycling infra trips: {n_cycling:,} ({pct_cycling:.2f}%)")

    # Check by mode
    by_mode = con.execute(f"""
        SELECT mode,
               COUNT(*) AS n,
               SUM(CASE WHEN is_cycling_infra THEN 1 ELSE 0 END) AS n_cycling,
               ROUND(100.0 * SUM(CASE WHEN is_cycling_infra THEN 1 ELSE 0 END)
                     / COUNT(*), 2) AS pct_cycling
        FROM read_parquet('{out_path}')
        GROUP BY mode ORDER BY mode
    """).fetchdf()
    print("\n  By mode:")
    print(by_mode.to_string(index=False))

    con.close()

    # Rename: new file replaces old
    old_path = Path(V2_DIR / "trip_modeling.parquet")
    new_path = Path(V2_DIR / "trip_modeling_v2.parquet")
    backup_path = Path(V2_DIR / "trip_modeling_backup.parquet")

    old_path.rename(backup_path)
    new_path.rename(old_path)
    print(f"\n  Replaced trip_modeling.parquet (backup at trip_modeling_backup.parquet)")

    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
