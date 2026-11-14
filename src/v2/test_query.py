"""Quick test of DuckDB indicator query on a small sample."""
import duckdb
import time

con = duckdb.connect()

csv_path = "D:/SwingData/raw/2023_05_Swing_Routes.csv"

query = f"""
WITH raw_parsed AS (
    SELECT
        route_id,
        user_id,
        list_transform(
            regexp_extract_all(CAST(speeds AS VARCHAR), '(\\d+(?:\\.\\d+)?)'),
            x -> TRY_CAST(x AS DOUBLE)
        ) AS speed_arr
    FROM read_csv_auto('{csv_path}', ignore_errors=true)
    WHERE CAST(speeds AS VARCHAR) != '-999' AND speeds IS NOT NULL
    LIMIT 1000
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
        sqrt(GREATEST(
            list_avg(list_transform(speed_arr, x -> x * x)) - mean_spd * mean_spd,
            0.0
        )) AS speed_std_val,
        list_sort(speed_arr)[CAST(CEIL(n_pts * 0.85) AS INTEGER)] AS p85_spd,
        list_transform(
            range(1, n_pts),
            i -> (speed_arr[i + 1] - speed_arr[i]) * 0.02778
        ) AS accel_arr
    FROM with_stats
)
SELECT
    route_id,
    n_pts,
    ROUND(mean_spd, 2) AS mean_speed,
    ROUND(max_spd, 2) AS max_speed,
    ROUND(p85_spd, 2) AS p85_speed,
    ROUND(CASE WHEN mean_spd > 0 THEN speed_std_val / mean_spd ELSE 0.0 END, 4) AS speed_cv,
    list_count(list_filter(accel_arr, x -> x > 0.5)) AS harsh_accel,
    list_count(list_filter(accel_arr, x -> x < -0.5)) AS harsh_decel,
    ROUND(
        list_count(list_filter(speed_arr, x -> abs(x - mean_spd) <= 3.0)) * 1.0 / n_pts,
        4
    ) AS cruise_frac,
    ROUND(
        list_count(list_filter(speed_arr, x -> x = 0.0)) * 1.0 / n_pts,
        4
    ) AS zero_frac,
    CASE WHEN max_spd > 25 THEN TRUE ELSE FALSE END AS is_speeding
FROM with_derived
"""

t0 = time.time()
result = con.execute(query).fetchall()
elapsed = time.time() - t0

print(f"Query OK: {len(result)} rows in {elapsed:.2f}s")
print(f"{'route_id':>20s} pts mean   max   p85   cv    ha hd cruise zero  spd")
for r in result[:10]:
    print(f"{r[0][:20]:>20s} {r[1]:3d} {r[2]:5.1f} {r[3]:5.1f} {r[4]:5.1f} {r[5]:.3f} {r[6]:2d} {r[7]:2d} {r[8]:.3f}  {r[9]:.3f} {r[10]}")

# Verify against existing May indicators
print("\n--- Cross-check with existing trip_indicators.parquet ---")
# Pick a few route_ids and compare
route_ids = [r[0] for r in result[:5]]
ids_sql = ", ".join(f"'{rid}'" for rid in route_ids)

existing = con.execute(f"""
    SELECT route_id, mean_speed, speed_cv, harsh_accel_count, harsh_decel_count,
           cruise_fraction, zero_speed_fraction
    FROM read_parquet('data_parquet/trip_indicators.parquet')
    WHERE route_id IN ({ids_sql})
""").fetchall()

print(f"Found {len(existing)} matching routes in existing indicators")
for e in existing:
    # Find matching new result
    new = [r for r in result if r[0] == e[0]]
    if new:
        n = new[0]
        print(f"  {e[0][:20]:>20s}")
        print(f"    mean_speed: old={e[1]:.2f} new={n[2]:.2f}")
        print(f"    speed_cv:   old={e[2]:.4f} new={n[5]:.4f}")
        print(f"    harsh_a:    old={e[3]} new={n[6]}")
        print(f"    harsh_d:    old={e[4]} new={n[7]}")
        print(f"    cruise:     old={e[5]:.4f} new={n[8]:.4f}")
        print(f"    zero:       old={e[6]:.4f} new={n[9]:.4f}")

print("\nTest complete.")
con.close()
