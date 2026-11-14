"""
Task 0.7: Data quality verification for trip_modeling.parquet.

Checks: completeness, ranges, distributions, cross-consistency.
Output: reports/v2/data_verification.json
"""

import duckdb
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, REPORTS_DIR

V2_DIR = DATA_DIR / "v2"
REPORT_DIR = REPORTS_DIR / "v2"


def main():
    print("=" * 60)
    print("Task 0.7: Data Quality Verification")
    print("=" * 60)

    t_start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")
    issues = []
    checks = {}

    # 1. Row count and completeness
    print("  [1] Row count and completeness...")
    completeness = con.execute(f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(route_id) AS has_route_id,
            COUNT(user_id) AS has_user_id,
            COUNT(mode) AS has_mode,
            COUNT(mean_speed) AS has_mean_speed,
            COUNT(speed_cv) AS has_speed_cv,
            COUNT(harsh_accel_count) AS has_harsh_accel,
            COUNT(cruise_fraction) AS has_cruise,
            COUNT(distance) AS has_distance,
            COUNT(city) AS has_city,
            COUNT(age) AS has_age,
            COUNT(start_road_class) AS has_road_class,
            COUNT(time_of_day) AS has_time_of_day,
            COUNT(day_type) AS has_day_type
    FROM read_parquet('{path}')
    """).fetchone()

    n_rows = completeness[0]
    cols = ["n_rows", "has_route_id", "has_user_id", "has_mode",
            "has_mean_speed", "has_speed_cv", "has_harsh_accel", "has_cruise",
            "has_distance", "has_city", "has_age", "has_road_class",
            "has_time_of_day", "has_day_type"]
    comp_dict = dict(zip(cols, completeness))
    checks["completeness"] = comp_dict

    for col in cols[1:]:
        pct = comp_dict[col] / n_rows * 100
        status = "OK" if pct > 99 else ("WARN" if pct > 90 else "FAIL")
        if status != "OK":
            issues.append(f"{col}: {pct:.1f}% complete ({status})")
        print(f"    {col}: {comp_dict[col]:,} ({pct:.1f}%) [{status}]")

    # 2. Value ranges
    print("\n  [2] Value ranges...")
    ranges = con.execute(f"""
        SELECT
            MIN(mean_speed) AS min_mean_speed, MAX(mean_speed) AS max_mean_speed,
            MIN(speed_cv) AS min_speed_cv, MAX(speed_cv) AS max_speed_cv,
            MIN(harsh_accel_count) AS min_harsh, MAX(harsh_accel_count) AS max_harsh,
            MIN(cruise_fraction) AS min_cruise, MAX(cruise_fraction) AS max_cruise,
            MIN(zero_speed_fraction) AS min_zero, MAX(zero_speed_fraction) AS max_zero,
            MIN(distance) AS min_dist, MAX(distance) AS max_dist,
            MIN(CAST(age AS INTEGER)) AS min_age, MAX(CAST(age AS INTEGER)) AS max_age,
            MIN(start_lat) AS min_lat, MAX(start_lat) AS max_lat,
            MIN(start_lon) AS min_lon, MAX(start_lon) AS max_lon
        FROM read_parquet('{path}')
    """).fetchone()

    range_checks = {
        "mean_speed": (ranges[0], ranges[1], 0, 50),
        "speed_cv": (ranges[2], ranges[3], 0, 10),
        "harsh_accel_count": (ranges[4], ranges[5], 0, 100),
        "cruise_fraction": (ranges[6], ranges[7], 0, 1),
        "zero_speed_fraction": (ranges[8], ranges[9], 0, 1),
        "distance": (ranges[10], ranges[11], 0, 100000),
        "age": (ranges[12], ranges[13], 10, 100),
        "lat": (ranges[14], ranges[15], 33, 39),
        "lon": (ranges[16], ranges[17], 124, 132),
    }
    checks["ranges"] = {}
    for name, (mn, mx, exp_min, exp_max) in range_checks.items():
        ok = mn >= exp_min and mx <= exp_max
        checks["ranges"][name] = {"min": float(mn) if mn else None, "max": float(mx) if mx else None, "ok": ok}
        status = "OK" if ok else "WARN"
        if not ok:
            issues.append(f"{name} range [{mn}, {mx}] outside expected [{exp_min}, {exp_max}]")
        print(f"    {name}: [{mn}, {mx}] [{status}]")

    # 3. Mode distribution (should only be TUB/STD/ECO)
    print("\n  [3] Mode distribution...")
    modes = con.execute(f"""
        SELECT mode, COUNT(*) AS n
        FROM read_parquet('{path}')
        GROUP BY mode ORDER BY n DESC
    """).fetchdf()
    checks["mode_distribution"] = {r["mode"]: int(r["n"]) for _, r in modes.iterrows()}
    unexpected = set(modes["mode"]) - {"TUB", "STD", "ECO"}
    if unexpected:
        issues.append(f"Unexpected modes: {unexpected}")
    print(f"    Modes: {dict(checks['mode_distribution'])}")

    # 4. Duplicate route_ids
    print("\n  [4] Duplicate check...")
    n_unique = con.execute(f"""
        SELECT COUNT(DISTINCT route_id) FROM read_parquet('{path}')
    """).fetchone()[0]
    n_dups = n_rows - n_unique
    checks["duplicates"] = {"n_unique_routes": n_unique, "n_duplicates": n_dups}
    if n_dups > 0:
        issues.append(f"{n_dups:,} duplicate route_ids")
    print(f"    Unique routes: {n_unique:,}, Duplicates: {n_dups:,}")

    # 5. Month coverage
    print("\n  [5] Month coverage...")
    months = con.execute(f"""
        SELECT month_year, COUNT(*) AS n
        FROM read_parquet('{path}')
        GROUP BY month_year ORDER BY month_year
    """).fetchdf()
    checks["month_coverage"] = {r["month_year"]: int(r["n"]) for _, r in months.iterrows()}
    expected_months = {f"2023-{m:02d}" for m in range(2, 12)}
    actual_months = set(months["month_year"])
    missing = expected_months - actual_months
    if missing:
        issues.append(f"Missing months: {missing}")
    for _, r in months.iterrows():
        print(f"    {r['month_year']}: {r['n']:>10,}")

    # 6. Cross-consistency: is_speeding should match max_speed > 25
    print("\n  [6] Cross-consistency...")
    inconsistent = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{path}')
        WHERE (max_speed_from_profile > 25 AND NOT is_speeding)
           OR (max_speed_from_profile <= 25 AND is_speeding)
    """).fetchone()[0]
    checks["speeding_consistency"] = {"n_inconsistent": inconsistent}
    if inconsistent > 0:
        issues.append(f"{inconsistent:,} trips with inconsistent is_speeding flag")
    print(f"    Speeding flag inconsistencies: {inconsistent:,}")

    # 7. Fraction summaries
    print("\n  [7] Fraction bounds...")
    frac_check = con.execute(f"""
        SELECT
            COUNT(CASE WHEN cruise_fraction < 0 OR cruise_fraction > 1 THEN 1 END) AS bad_cruise,
            COUNT(CASE WHEN zero_speed_fraction < 0 OR zero_speed_fraction > 1 THEN 1 END) AS bad_zero,
            COUNT(CASE WHEN speed_cv < 0 THEN 1 END) AS bad_cv
        FROM read_parquet('{path}')
    """).fetchone()
    checks["fraction_bounds"] = {"bad_cruise": frac_check[0], "bad_zero": frac_check[1], "bad_cv": frac_check[2]}
    for name, val in zip(["cruise", "zero_frac", "speed_cv"], frac_check):
        if val > 0:
            issues.append(f"{val:,} trips with out-of-range {name}")
        print(f"    Out-of-range {name}: {val:,}")

    elapsed = time.time() - t_start

    # Summary
    print(f"\n  {'=' * 50}")
    print(f"  VERIFICATION SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Total rows: {n_rows:,}")
    print(f"  Issues found: {len(issues)}")
    for iss in issues:
        print(f"    - {iss}")
    if not issues:
        print("    ALL CHECKS PASSED")
    print(f"  Time: {elapsed:.1f}s")

    # Save report
    report = {
        "task": "0.7",
        "n_rows": n_rows,
        "checks": checks,
        "issues": issues,
        "all_passed": len(issues) == 0,
        "time_s": round(elapsed, 1),
    }
    out_path = REPORT_DIR / "data_verification.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    con.close()
    print(f"  Report: {out_path}")


if __name__ == "__main__":
    main()
