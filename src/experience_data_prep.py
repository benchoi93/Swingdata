"""
Task 1.0: Trip sequencing and experience data preparation.

Uses all 24 months of data (routes_all.parquet from preprocess_all_months.py)
to compute per-trip experience features via DuckDB window functions:
  - trip_rank (chronological order per user)
  - days_since_first_trip, months_since_first_trip
  - experience_bin (1st, 2-5, 6-20, 21-50, 51-100, 101-500, 500+)
  - usage_category (one_time, occasional, regular, frequent, heavy, super_heavy)

Cross-references with Scooter CSVs for demographics (age).

Outputs:
  - data_parquet/modeling/trip_experience.parquet
"""

import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, MODELING_DIR, RANDOM_SEED

# Paths
ALL_MONTHS_DIR = DATA_DIR / "all_months"
ROUTES_ALL = ALL_MONTHS_DIR / "routes_all.parquet"
USER_LONGITUDINAL = ALL_MONTHS_DIR / "user_longitudinal.parquet"
OUTPUT_PATH = MODELING_DIR / "trip_experience.parquet"
REPORT_PATH = MODELING_DIR / "experience_data_prep_report.json"

# Scooter CSVs for demographics
RAW_DIR = Path("D:/SwingData/raw")
SCOOTER_2022 = RAW_DIR / "2022_Swing_Scooter.csv"
SCOOTER_2023 = RAW_DIR / "2023_Swing_Scooter.csv"

MODELING_DIR.mkdir(parents=True, exist_ok=True)


def build_trip_experience() -> dict:
    """Build trip-level experience features from 24-month data.

    Returns:
        Dictionary with data prep summary statistics.
    """
    t_start = time.time()
    con = duckdb.connect()

    routes_path = str(ROUTES_ALL).replace("\\", "/")
    output_path = str(OUTPUT_PATH).replace("\\", "/")

    print("=" * 70)
    print("  TASK 1.0: TRIP EXPERIENCE DATA PREPARATION")
    print("=" * 70)

    # Verify input exists
    total_trips = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{routes_path}')"
    ).fetchone()[0]
    total_users = con.execute(
        f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{routes_path}')"
    ).fetchone()[0]
    print(f"\nInput: {total_trips:,} trips, {total_users:,} users")

    # Step 1: Extract user demographics from both Scooter CSVs
    print("\nExtracting user demographics from Scooter CSVs...")
    scooter_paths = []
    for p in [SCOOTER_2022, SCOOTER_2023]:
        if p.exists():
            scooter_paths.append(str(p).replace("\\", "/"))
            print(f"  Found: {p.name}")
        else:
            print(f"  Missing: {p.name}")

    if scooter_paths:
        union_parts = " UNION ALL ".join(
            f"SELECT user_id, age FROM read_csv_auto('{p}') WHERE age IS NOT NULL AND age > 0 AND age < 100"
            for p in scooter_paths
        )
        con.execute(f"""
            CREATE TABLE user_age AS
            SELECT user_id, MEDIAN(age)::INTEGER as age
            FROM ({union_parts})
            GROUP BY user_id
        """)
        n_with_age = con.execute("SELECT COUNT(*) FROM user_age").fetchone()[0]
        print(f"  Users with age data: {n_with_age:,}")
    else:
        con.execute("CREATE TABLE user_age (user_id VARCHAR, age INTEGER)")
        print("  WARNING: No scooter CSVs found, age data unavailable")

    # Step 2: Compute trip sequencing via window functions
    print("\nComputing trip sequencing (window functions)...")
    t1 = time.time()

    con.execute(f"""
        CREATE TABLE trip_experience AS
        WITH sequenced AS (
            SELECT
                r.*,
                ROW_NUMBER() OVER (
                    PARTITION BY r.user_id ORDER BY r.start_date, r.start_time
                ) AS trip_rank,
                MIN(r.start_date) OVER (PARTITION BY r.user_id) AS first_trip_date,
                MAX(r.start_date) OVER (PARTITION BY r.user_id) AS last_trip_date,
                COUNT(*) OVER (PARTITION BY r.user_id) AS user_total_trips
            FROM read_parquet('{routes_path}') r
        )
        SELECT
            s.route_id,
            s.user_id,
            s.start_date,
            s.start_time,
            s.model,
            s.mode,
            s.travel_time,
            s.distance,
            s.start_lat,
            s.start_lon,
            s.max_speed,
            s.avg_speed,
            s.mean_speed_from_speeds,
            s.max_speed_from_speeds,
            s.is_speeding,
            s.month_year,
            s.trip_rank,
            s.first_trip_date,
            s.last_trip_date,
            s.user_total_trips,

            -- Days since first trip
            DATEDIFF('day', s.first_trip_date, s.start_date) AS days_since_first_trip,

            -- Months since first trip
            DATEDIFF('month', s.first_trip_date, s.start_date) AS months_since_first_trip,

            -- Is first trip
            CASE WHEN s.trip_rank = 1 THEN TRUE ELSE FALSE END AS is_first_trip,

            -- Is first week (within 7 days of first trip)
            CASE WHEN DATEDIFF('day', s.first_trip_date, s.start_date) <= 7
                 THEN TRUE ELSE FALSE END AS is_first_week,

            -- Experience bin
            CASE
                WHEN s.trip_rank = 1 THEN '01_first'
                WHEN s.trip_rank BETWEEN 2 AND 5 THEN '02_2to5'
                WHEN s.trip_rank BETWEEN 6 AND 20 THEN '03_6to20'
                WHEN s.trip_rank BETWEEN 21 AND 50 THEN '04_21to50'
                WHEN s.trip_rank BETWEEN 51 AND 100 THEN '05_51to100'
                WHEN s.trip_rank BETWEEN 101 AND 500 THEN '06_101to500'
                ELSE '07_500plus'
            END AS experience_bin,

            -- Usage category based on total trips
            CASE
                WHEN s.user_total_trips = 1 THEN 'one_time'
                WHEN s.user_total_trips BETWEEN 2 AND 5 THEN 'occasional'
                WHEN s.user_total_trips BETWEEN 6 AND 20 THEN 'regular'
                WHEN s.user_total_trips BETWEEN 21 AND 50 THEN 'frequent'
                WHEN s.user_total_trips BETWEEN 51 AND 200 THEN 'heavy'
                ELSE 'super_heavy'
            END AS usage_category,

            -- User age (from scooter CSV)
            a.age

        FROM sequenced s
        LEFT JOIN user_age a ON s.user_id = a.user_id
    """)

    t_seq = time.time() - t1
    print(f"  Sequencing complete in {t_seq:.1f}s")

    # Step 3: Summary statistics
    print("\nComputing summary statistics...")
    stats = {}

    # Total counts
    result = con.execute("""
        SELECT
            COUNT(*) AS total_trips,
            COUNT(DISTINCT user_id) AS total_users,
            AVG(trip_rank) AS avg_trip_rank,
            MAX(trip_rank) AS max_trip_rank,
            AVG(user_total_trips) AS avg_user_total_trips,
            SUM(CASE WHEN is_speeding THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS overall_speeding_rate,
            SUM(CASE WHEN age IS NOT NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS age_coverage
        FROM trip_experience
    """).fetchone()

    stats["total_trips"] = result[0]
    stats["total_users"] = result[1]
    stats["avg_trip_rank"] = round(result[2], 1)
    stats["max_trip_rank"] = result[3]
    stats["avg_user_total_trips"] = round(result[4], 1)
    stats["overall_speeding_rate"] = round(result[5], 4)
    stats["age_coverage"] = round(result[6], 4)

    # Experience bin distribution
    exp_dist = con.execute("""
        SELECT experience_bin,
               COUNT(*) AS n_trips,
               COUNT(DISTINCT user_id) AS n_users,
               AVG(CASE WHEN is_speeding THEN 1.0 ELSE 0.0 END) AS speeding_rate
        FROM trip_experience
        GROUP BY experience_bin
        ORDER BY experience_bin
    """).fetchdf()
    print("\n  Experience bin distribution:")
    for _, row in exp_dist.iterrows():
        print(f"    {row['experience_bin']}: {row['n_trips']:>10,} trips, "
              f"{row['n_users']:>8,} users, speeding={row['speeding_rate']:.3f}")
    stats["experience_bins"] = {
        row["experience_bin"]: {
            "n_trips": int(row["n_trips"]),
            "n_users": int(row["n_users"]),
            "speeding_rate": round(row["speeding_rate"], 4),
        }
        for _, row in exp_dist.iterrows()
    }

    # Usage category distribution
    usage_dist = con.execute("""
        SELECT usage_category,
               COUNT(DISTINCT user_id) AS n_users,
               COUNT(*) AS n_trips,
               AVG(CASE WHEN is_speeding THEN 1.0 ELSE 0.0 END) AS speeding_rate
        FROM trip_experience
        GROUP BY usage_category
        ORDER BY n_users DESC
    """).fetchdf()
    print("\n  Usage category distribution:")
    for _, row in usage_dist.iterrows():
        print(f"    {row['usage_category']:<12}: {row['n_users']:>8,} users, "
              f"{row['n_trips']:>10,} trips, speeding={row['speeding_rate']:.3f}")
    stats["usage_categories"] = {
        row["usage_category"]: {
            "n_users": int(row["n_users"]),
            "n_trips": int(row["n_trips"]),
            "speeding_rate": round(row["speeding_rate"], 4),
        }
        for _, row in usage_dist.iterrows()
    }

    # Monthly distribution
    monthly = con.execute("""
        SELECT month_year,
               COUNT(*) AS n_trips,
               COUNT(DISTINCT user_id) AS n_users
        FROM trip_experience
        GROUP BY month_year
        ORDER BY month_year
    """).fetchdf()
    print("\n  Monthly trip counts:")
    for _, row in monthly.iterrows():
        print(f"    {row['month_year']}: {row['n_trips']:>10,} trips, "
              f"{row['n_users']:>8,} users")

    # Step 4: Export
    print(f"\nExporting to {OUTPUT_PATH}...")
    con.execute(f"""
        COPY (SELECT * FROM trip_experience)
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Verify
    verify = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
    ).fetchone()[0]
    print(f"  Exported: {verify:,} trips")
    stats["exported_trips"] = verify

    # Save report
    stats["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Report saved to {REPORT_PATH}")

    con.close()

    print("\n" + "=" * 70)
    print(f"  EXPERIENCE DATA PREP COMPLETE ({stats['runtime_seconds']:.0f}s)")
    print(f"  {stats['total_trips']:,} trips, {stats['total_users']:,} users")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    build_trip_experience()
