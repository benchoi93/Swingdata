"""
Verify all key statistics cited in the paper against actual data.

Reads the processed parquet files and checks every major number reported in the
manuscript (abstract, data description, results, conclusions). Outputs a detailed
report indicating PASS/FAIL/WARN for each claim.

Usage:
    python src/verify_paper_statistics.py
"""
import duckdb
import numpy as np
from pathlib import Path

# ---- Paths ----
ROOT = Path("C:/Users/chois/Gitsrcs/Swingdata")
DATA = ROOT / "data_parquet"

con = duckdb.connect()

results: list[tuple[str, str, str]] = []  # (section, claim, status)


def check(section: str, claim: str, actual, expected, tol: float = 0.01):
    """Compare actual vs expected within tolerance. Records result."""
    if actual is None:
        results.append((section, claim, f"FAIL  actual=None, expected={expected}"))
        return
    if isinstance(expected, (int, float)):
        diff = abs(actual - expected)
        rel = diff / max(abs(expected), 1e-9)
        if rel <= tol:
            results.append((section, claim, f"PASS  actual={actual}, expected={expected}"))
        else:
            results.append((section, claim, f"FAIL  actual={actual}, expected={expected}, diff={diff:.4f} ({rel:.1%})"))
    else:
        if actual == expected:
            results.append((section, claim, f"PASS  actual={actual}"))
        else:
            results.append((section, claim, f"FAIL  actual={actual}, expected={expected}"))


def pct(val: float) -> float:
    """Convert fraction to percentage."""
    return round(val * 100, 1)


print("=" * 70)
print("PAPER STATISTICS VERIFICATION")
print("=" * 70)

# ========================================================================
# 1. SAMPLE SIZES (Abstract, Data Description)
# ========================================================================
print("\n--- Sample Sizes ---")

# Total valid trips
n_trips = con.sql(f"""
    SELECT COUNT(*) FROM read_parquet('{DATA}/cleaned/trips_cleaned.parquet')
""").fetchone()[0]
check("Abstract", "Total valid trips = 2,780,890", n_trips, 2780890, tol=0.001)

# Unique users
n_users = con.sql(f"""
    SELECT COUNT(DISTINCT user_id) FROM read_parquet('{DATA}/cleaned/trips_cleaned.parquet')
""").fetchone()[0]
check("Abstract", "Unique users = 382,328", n_users, 382328, tol=0.001)

# Unique vehicles (qrcode = vehicle identifier)
n_vehicles = con.sql(f"""
    SELECT COUNT(DISTINCT qrcode) FROM read_parquet('{DATA}/cleaned/trips_cleaned.parquet')
""").fetchone()[0]
check("Data", "Unique vehicles = 81,525", n_vehicles, 81525, tol=0.001)

# Cities and provinces
n_cities = con.sql(f"""
    SELECT COUNT(DISTINCT city) FROM read_parquet('{DATA}/cleaned/trips_cleaned.parquet')
""").fetchone()[0]
check("Abstract", "Number of cities = 52", n_cities, 52)

n_provinces = con.sql(f"""
    SELECT COUNT(DISTINCT province) FROM read_parquet('{DATA}/cleaned/trips_cleaned.parquet')
    WHERE province != 'Other'
""").fetchone()[0]
check("Data", "Number of provinces = 17 (excl. 'Other')", n_provinces, 17)

# ========================================================================
# 2. SPEEDING RATES (Abstract, Results)
# ========================================================================
print("\n--- Speeding Rates ---")

# Trip-level binary speeding (max > 25 km/h)
trip_speeding = con.sql(f"""
    SELECT
        AVG(CASE WHEN max_speed_from_profile > 25 THEN 1 ELSE 0 END) AS pct_speeding
    FROM read_parquet('{DATA}/trip_indicators.parquet') ti
    JOIN read_parquet('{DATA}/cleaned/trips_cleaned.parquet') tc USING (route_id)
""").fetchone()[0]
check("Abstract", "29.6% of trips exceed 25 km/h", pct(trip_speeding), 29.6, tol=0.05)

# User-level: 60.9% never speed
user_never_speed = con.sql(f"""
    SELECT AVG(CASE WHEN speeding_propensity = 0 THEN 1 ELSE 0 END)
    FROM read_parquet('{DATA}/user_indicators.parquet')
""").fetchone()[0]
check("Abstract", "60.9% of users never speed", pct(user_never_speed), 60.9, tol=0.05)

# User-level: 17.3% speed on >50% of trips
user_habitual = con.sql(f"""
    SELECT AVG(CASE WHEN speeding_propensity > 0.5 THEN 1 ELSE 0 END)
    FROM read_parquet('{DATA}/user_indicators.parquet')
""").fetchone()[0]
check("Results", "17.3% of users speed on >50% of trips", pct(user_habitual), 17.3, tol=0.05)

# ========================================================================
# 3. MODE DISTRIBUTION (Data Description)
# ========================================================================
print("\n--- Mode Distribution ---")

mode_dist = con.sql(f"""
    SELECT mode, COUNT(*) as n,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
    FROM read_parquet('{DATA}/cleaned/trips_cleaned.parquet')
    GROUP BY mode ORDER BY n DESC
""").fetchall()
for mode, n, pct_val in mode_dist:
    print(f"  {mode}: {n:,} ({pct_val}%)")

# Check specific mode percentages
mode_dict = {m: p for m, _, p in mode_dist}
# Updated to match corrected paper (all valid trips)
check("Data", "TUB mode ~46.0%", mode_dict.get("SCOOTER_TUB", 0), 46.0, tol=0.05)
check("Data", "STD mode ~33.6%", mode_dict.get("SCOOTER_STD", 0), 33.6, tol=0.05)
check("Data", "ECO mode ~4.0%", mode_dict.get("SCOOTER_ECO", 0), 4.0, tol=0.10)

# ========================================================================
# 4. MODE-SPECIFIC SPEED (Results, Data Description)
# ========================================================================
print("\n--- Mode-Specific Speed ---")

mode_speeds = con.sql(f"""
    SELECT tc.mode,
           AVG(ti.mean_speed) AS mean_speed,
           AVG(CASE WHEN ti.max_speed_from_profile > 25 THEN 1 ELSE 0 END) AS speeding_rate
    FROM read_parquet('{DATA}/trip_indicators.parquet') ti
    JOIN read_parquet('{DATA}/cleaned/trips_cleaned.parquet') tc USING (route_id)
    WHERE tc.mode IN ('SCOOTER_TUB', 'SCOOTER_STD', 'SCOOTER_ECO')
    GROUP BY tc.mode
""").fetchall()
speed_dict = {m: (ms, sr) for m, ms, sr in mode_speeds}

for mode_name, (ms, sr) in speed_dict.items():
    print(f"  {mode_name}: mean_speed={ms:.1f} km/h, speeding_rate={pct(sr):.1f}%")

# Check TUB speeding rate ~54%
check("Results", "TUB speeding rate (binary) ~54%",
      pct(speed_dict.get("SCOOTER_TUB", (0, 0))[1]), 54.0, tol=0.10)
# Check ECO speeding rate ~1.1%
check("Results", "ECO speeding rate (binary) ~1.1%",
      pct(speed_dict.get("SCOOTER_ECO", (0, 0))[1]), 1.1, tol=0.20)
# Check STD speeding rate ~0.93%
check("Data", "STD speeding rate (binary) ~0.93%",
      pct(speed_dict.get("SCOOTER_STD", (0, 0))[1]), 0.93, tol=0.20)

# ========================================================================
# 5. GMM RIDER TYPOLOGY (Results)
# ========================================================================
print("\n--- GMM Rider Typology ---")

gmm_dist = con.sql(f"""
    SELECT gmm_class_name, COUNT(*) as n,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
    FROM read_parquet('{DATA}/modeling/user_classes.parquet')
    GROUP BY gmm_class_name ORDER BY n DESC
""").fetchall()
for cls, n, pct_val in gmm_dist:
    print(f"  {cls}: {n:,} ({pct_val}%)")

gmm_dict = {c: p for c, _, p in gmm_dist}
gmm_n_dict = {c: n for c, n, _ in gmm_dist}

# Check GMM class sizes (corrected 2026-02-22)
check("Results", "GMM Safe Rider n=80,052", gmm_n_dict.get("Safe Rider", 0), 80052, tol=0.005)
check("Results", "GMM Moderate Risk n=51,491", gmm_n_dict.get("Moderate Risk", 0), 51491, tol=0.005)
check("Results", "GMM Habitual Speeder n=52,223", gmm_n_dict.get("Habitual Speeder", 0), 52223, tol=0.005)
check("Results", "GMM Stop-and-Go n=15,055", gmm_n_dict.get("Stop-and-Go", 0), 15055, tol=0.005)

# ========================================================================
# 6. SPATIAL ANALYSIS (Results)
# ========================================================================
print("\n--- Spatial: H3 Hexes ---")

n_hexes = con.sql(f"""
    SELECT COUNT(DISTINCT h3_origin) FROM read_parquet('{DATA}/trip_h3_indices.parquet')
""").fetchone()[0]
check("Results", "8,144 unique origin hexes", n_hexes, 8144, tol=0.02)

# ========================================================================
# 7. ROAD CLASS (Results)
# ========================================================================
print("\n--- Road Class ---")

road_stats = con.sql(f"""
    SELECT AVG(frac_major_road) AS mean_frac_major
    FROM read_parquet('{DATA}/trip_road_classes.parquet')
""").fetchone()[0]
check("Results", "frac_major_road mean ~0.136", round(road_stats, 3), 0.136, tol=0.05)

# ========================================================================
# 8. MODELING SAMPLE (Results)
# ========================================================================
print("\n--- Modeling Samples ---")

# Modeling N = scooter-only with age data
n_modeling = con.sql(f"""
    SELECT COUNT(*) FROM read_parquet('{DATA}/modeling/trip_modeling.parquet')
    WHERE mode_clean IN ('TUB', 'STD', 'ECO') AND age IS NOT NULL AND age > 0
""").fetchone()[0]
check("Results", "Modeling dataset (scooter+age) ~2,319,886 trips", n_modeling, 2319886, tol=0.005)

# Total scooter-mode trips
n_scooter = con.sql(f"""
    SELECT COUNT(*) FROM read_parquet('{DATA}/modeling/trip_modeling.parquet')
    WHERE mode_clean IN ('TUB', 'STD', 'ECO')
""").fetchone()[0]
check("Data", "Scooter-mode trips = 2,323,154", n_scooter, 2323154, tol=0.001)

n_user_modeling = con.sql(f"""
    SELECT COUNT(*) FROM read_parquet('{DATA}/modeling/user_modeling.parquet')
""").fetchone()[0]
print(f"  User modeling N = {n_user_modeling:,}")

n_user_classes = con.sql(f"""
    SELECT COUNT(*) FROM read_parquet('{DATA}/modeling/user_classes.parquet')
""").fetchone()[0]
print(f"  User classes N = {n_user_classes:,}")

# ========================================================================
# 9. AGE DISTRIBUTION (Results / Data)
# ========================================================================
print("\n--- Age Distribution ---")

age_stats = con.sql(f"""
    SELECT
        MEDIAN(age) AS median_age,
        AVG(age) AS mean_age,
        AVG(CASE WHEN age < 25 THEN 1 ELSE 0 END) AS pct_under_25
    FROM read_parquet('{DATA}/modeling/trip_modeling.parquet')
    WHERE age IS NOT NULL AND age > 0
""").fetchall()[0]
print(f"  Median age: {age_stats[0]}, Mean age: {age_stats[1]:.1f}, % under 25: {pct(age_stats[2]):.1f}%")

# ========================================================================
# 10. PROVINCIAL VARIATION (Results)
# ========================================================================
print("\n--- Provincial Speeding Variation ---")

prov_speeding = con.sql(f"""
    SELECT province,
           AVG(is_speeding) AS speeding_rate,
           COUNT(*) AS n
    FROM read_parquet('{DATA}/modeling/trip_modeling.parquet')
    GROUP BY province
    HAVING COUNT(*) > 1000
    ORDER BY speeding_rate DESC
""").fetchall()
for prov, rate, n in prov_speeding[:3]:
    print(f"  {prov}: {pct(rate):.1f}% speeding (n={n:,})")
print("  ...")
for prov, rate, n in prov_speeding[-3:]:
    print(f"  {prov}: {pct(rate):.1f}% speeding (n={n:,})")

# Check Chungnam ~45.8% vs Ulsan ~17.6%
prov_dict = {p: r for p, r, _ in prov_speeding}
if "Chungnam" in prov_dict:
    check("Results", "Chungnam speeding ~45.8%", pct(prov_dict["Chungnam"]), 45.8, tol=0.10)
if "Gyeongbuk" in prov_dict:
    check("Results", "Gyeongbuk speeding ~42.6%", pct(prov_dict["Gyeongbuk"]), 42.6, tol=0.10)
if "Gangwon" in prov_dict:
    check("Results", "Gangwon speeding ~41.7%", pct(prov_dict["Gangwon"]), 41.7, tol=0.10)
if "Ulsan" in prov_dict:
    check("Results", "Ulsan speeding ~17.6%", pct(prov_dict["Ulsan"]), 17.6, tol=0.10)

# ========================================================================
# 11. THRESHOLD SENSITIVITY (Results)
# ========================================================================
print("\n--- Threshold Sensitivity ---")

# Threshold sensitivity: scooter-only (matches paper's robustness section)
threshold_rates = con.sql(f"""
    SELECT
        AVG(CASE WHEN max_speed_from_profile > 20 THEN 1 ELSE 0 END) AS rate_20,
        AVG(CASE WHEN max_speed_from_profile > 25 THEN 1 ELSE 0 END) AS rate_25,
        AVG(CASE WHEN max_speed_from_profile > 30 THEN 1 ELSE 0 END) AS rate_30
    FROM read_parquet('{DATA}/modeling/trip_modeling.parquet')
    WHERE mode_clean IN ('TUB', 'STD', 'ECO')
""").fetchone()
check("Results", "20 km/h threshold -> ~87.1% speeding", pct(threshold_rates[0]), 87.1, tol=0.05)
check("Results", "25 km/h threshold -> ~30.1% speeding", pct(threshold_rates[1]), 30.1, tol=0.05)
check("Results", "30 km/h threshold -> ~0.2% speeding", pct(threshold_rates[2]), 0.2, tol=0.50)

# ========================================================================
# 12. WITHIN-SUBJECT NATURAL EXPERIMENT (Results)
# ========================================================================
print("\n--- Within-Subject Mode Comparison ---")

# Users who used both TUB and ECO
dual_users = con.sql(f"""
    WITH tub_users AS (
        SELECT DISTINCT user_id FROM read_parquet('{DATA}/modeling/trip_modeling.parquet')
        WHERE mode_clean = 'TUB'
    ),
    eco_users AS (
        SELECT DISTINCT user_id FROM read_parquet('{DATA}/modeling/trip_modeling.parquet')
        WHERE mode_clean = 'ECO'
    )
    SELECT COUNT(*) FROM tub_users t JOIN eco_users e USING (user_id)
""").fetchone()[0]
check("Results", "Within-subject paired users ~12,046", dual_users, 12046, tol=0.05)

# Check mode comparison values from mode_comparison.json
import json
mc_path = DATA / "modeling" / "mode_comparison.json"
if mc_path.exists():
    mc = json.loads(mc_path.read_text())
    paired = mc["within_subject_paired"]["paired_tests"]
    check("Results/Table", "TUB-ECO mean speed diff=3.7 km/h",
          round(paired["mean_speed"]["mean_diff"], 1), 3.7, tol=0.05)
    check("Results/Table", "TUB-ECO max speed Cohen's d=2.17",
          round(paired["max_speed_from_profile"]["cohens_d"], 2), 2.17, tol=0.02)
    check("Results/Table", "TUB-ECO speed CV Cohen's d=0.22",
          round(paired["speed_cv"]["cohens_d"], 2), 0.22, tol=0.10)
    check("Results/Table", "ECO paired speeding rate=0.38%",
          round(paired["speeding_rate_25"]["eco_mean"] * 100, 2), 0.38, tol=0.15)

# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

pass_count = sum(1 for _, _, s in results if s.startswith("PASS"))
fail_count = sum(1 for _, _, s in results if s.startswith("FAIL"))
warn_count = sum(1 for _, _, s in results if s.startswith("WARN"))

print(f"\nTotal checks: {len(results)}")
print(f"  PASS: {pass_count}")
print(f"  FAIL: {fail_count}")
print(f"  WARN: {warn_count}")

print("\nDetailed Results:")
print("-" * 70)
for section, claim, status in results:
    marker = "OK" if status.startswith("PASS") else "XX" if status.startswith("FAIL") else "??"
    print(f"  [{marker}] [{section}] {claim}")
    print(f"       {status}")

if fail_count > 0:
    print(f"\n*** {fail_count} FAILURES DETECTED -- review before submission ***")
else:
    print("\n*** ALL CHECKS PASSED ***")

con.close()
