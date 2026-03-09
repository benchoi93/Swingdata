"""
Task 1.3: Identify cities from start coordinates.

Assigns each trip to a Korean metropolitan city/province using coordinate-based
spatial classification. Uses known city center coordinates with a KD-tree
nearest-neighbor approach.

Korean administrative divisions (시도) relevant to this dataset:
  - Seoul (서울), Busan (부산), Daegu (대구), Incheon (인천)
  - Gwangju (광주), Daejeon (대전), Ulsan (울산), Sejong (세종)
  - Gyeonggi (경기), Chungbuk (충북), Chungnam (충남)
  - Jeonbuk (전북), Jeonnam (전남), Gyeongbuk (경북), Gyeongnam (경남)
  - Gangwon (강원), Jeju (제주)

Outputs:
  - data_parquet/routes_with_cities.parquet — valid trips with city labels
  - data_parquet/city_assignment_report.json — assignment statistics
"""

import duckdb
import json
import sys
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR

# Korean metropolitan city/province centers (latitude, longitude)
# Using approximate centers of each administrative division
CITY_CENTERS = {
    "Seoul": (37.5665, 126.9780),
    "Busan": (35.1796, 129.0756),
    "Daegu": (35.8714, 128.6014),
    "Incheon": (37.4563, 126.7052),
    "Gwangju": (35.1595, 126.8526),
    "Daejeon": (36.3504, 127.3845),
    "Ulsan": (35.5384, 129.3114),
    "Sejong": (36.4800, 127.2600),
    "Suwon": (37.2636, 127.0286),         # Gyeonggi sub-region
    "Yongin": (37.2411, 127.1776),         # Gyeonggi sub-region
    "Seongnam": (37.4200, 127.1265),       # Gyeonggi sub-region
    "Goyang": (37.6584, 126.8320),         # Gyeonggi sub-region
    "Bucheon": (37.5034, 126.7660),        # Gyeonggi sub-region
    "Ansan": (37.3219, 126.8309),          # Gyeonggi sub-region
    "Anyang": (37.3943, 126.9568),         # Gyeonggi sub-region
    "Hwaseong": (37.2000, 126.8313),       # Gyeonggi sub-region
    "Pyeongtaek": (36.9920, 127.0850),     # Gyeonggi sub-region
    "Uijeongbu": (37.7377, 127.0339),      # Gyeonggi sub-region
    "Paju": (37.7590, 126.7800),           # Gyeonggi sub-region
    "Gimpo": (37.6153, 126.7156),          # Gyeonggi sub-region
    "Gwangju_Gyeonggi": (37.4095, 127.2573),  # Gwangju-si, Gyeonggi
    "Icheon": (37.2719, 127.4350),         # Gyeonggi sub-region
    "Wonju": (37.3422, 127.9202),          # Gangwon
    "Chuncheon": (37.8813, 127.7298),      # Gangwon
    "Cheongju": (36.6424, 127.4890),       # Chungbuk
    "Cheonan": (36.8151, 127.1139),        # Chungnam
    "Asan": (36.7898, 127.0018),           # Chungnam
    "Jeonju": (35.8242, 127.1480),         # Jeonbuk
    "Iksan": (35.9483, 126.9577),          # Jeonbuk
    "Gunsan": (35.9677, 126.7369),         # Jeonbuk
    "Mokpo": (34.8118, 126.3922),          # Jeonnam
    "Yeosu": (34.7604, 127.6622),          # Jeonnam
    "Suncheon": (34.9506, 127.4872),       # Jeonnam
    "Pohang": (36.0190, 129.3435),         # Gyeongbuk
    "Gumi": (36.1194, 128.3447),           # Gyeongbuk
    "Gyeongju": (35.8562, 129.2247),       # Gyeongbuk
    "Changwon": (35.2280, 128.6811),       # Gyeongnam (includes Masan, Jinhae)
    "Gimhae": (35.2285, 128.8894),         # Gyeongnam
    "Jinju": (35.1799, 128.1076),          # Gyeongnam
    "Jeju": (33.4996, 126.5312),           # Jeju
    "Yangju": (37.7854, 127.0456),         # Gyeonggi
    "Namyangju": (37.6367, 127.2165),      # Gyeonggi
    "Hanam": (37.5392, 127.2147),          # Gyeonggi
    "Gunpo": (37.3617, 126.9352),          # Gyeonggi
    "Siheung": (37.3800, 126.8030),        # Gyeonggi
    "Osan": (37.1490, 127.0770),           # Gyeonggi
    "Dongducheon": (37.9034, 127.0606),    # Gyeonggi
    "Andong": (36.5684, 128.7294),         # Gyeongbuk
    "Gangneung": (37.7519, 128.8761),      # Gangwon
    "Sokcho": (38.2070, 128.5918),         # Gangwon
    "Geoje": (34.8806, 128.6211),          # Gyeongnam
    "Tongyeong": (34.8544, 128.4331),      # Gyeongnam
    "Seosan": (36.7845, 126.4503),         # Chungnam
    "Dangjin": (36.8895, 126.6455),        # Chungnam
}

# Map sub-cities to their province for aggregation
CITY_TO_PROVINCE = {
    "Seoul": "Seoul",
    "Busan": "Busan",
    "Daegu": "Daegu",
    "Incheon": "Incheon",
    "Gwangju": "Gwangju",
    "Daejeon": "Daejeon",
    "Ulsan": "Ulsan",
    "Sejong": "Sejong",
    "Suwon": "Gyeonggi",
    "Yongin": "Gyeonggi",
    "Seongnam": "Gyeonggi",
    "Goyang": "Gyeonggi",
    "Bucheon": "Gyeonggi",
    "Ansan": "Gyeonggi",
    "Anyang": "Gyeonggi",
    "Hwaseong": "Gyeonggi",
    "Pyeongtaek": "Gyeonggi",
    "Uijeongbu": "Gyeonggi",
    "Paju": "Gyeonggi",
    "Gimpo": "Gyeonggi",
    "Gwangju_Gyeonggi": "Gyeonggi",
    "Icheon": "Gyeonggi",
    "Yangju": "Gyeonggi",
    "Namyangju": "Gyeonggi",
    "Hanam": "Gyeonggi",
    "Gunpo": "Gyeonggi",
    "Siheung": "Gyeonggi",
    "Osan": "Gyeonggi",
    "Dongducheon": "Gyeonggi",
    "Andong": "Gyeongbuk",
    "Gangneung": "Gangwon",
    "Sokcho": "Gangwon",
    "Geoje": "Gyeongnam",
    "Tongyeong": "Gyeongnam",
    "Seosan": "Chungnam",
    "Dangjin": "Chungnam",
    "Wonju": "Gangwon",
    "Chuncheon": "Gangwon",
    "Cheongju": "Chungbuk",
    "Cheonan": "Chungnam",
    "Asan": "Chungnam",
    "Jeonju": "Jeonbuk",
    "Iksan": "Jeonbuk",
    "Gunsan": "Jeonbuk",
    "Mokpo": "Jeonnam",
    "Yeosu": "Jeonnam",
    "Suncheon": "Jeonnam",
    "Pohang": "Gyeongbuk",
    "Gumi": "Gyeongbuk",
    "Gyeongju": "Gyeongbuk",
    "Changwon": "Gyeongnam",
    "Gimhae": "Gyeongnam",
    "Jinju": "Gyeongnam",
    "Jeju": "Jeju",
}

# Maximum distance (in degrees, ~approx) to assign a city
# 0.3 degrees ~ 30 km — trips farther than this get "Other"
MAX_ASSIGN_DISTANCE = 0.3


def assign_cities() -> dict:
    """Assign city labels to valid trips using KD-tree nearest-neighbor.

    Returns:
        Dictionary with assignment statistics.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    valid_parquet = str(DATA_DIR / "routes_valid.parquet")

    print("=" * 70)
    print("  TASK 1.3: CITY ASSIGNMENT")
    print("=" * 70)

    # Step 1: Build KD-tree from city centers
    city_names = list(CITY_CENTERS.keys())
    city_coords = np.array([CITY_CENTERS[c] for c in city_names])
    tree = KDTree(city_coords)

    print(f"\nCity centers loaded: {len(city_names)} cities")

    # Step 2: Read start coordinates from valid trips
    print("Reading start coordinates...")
    coords_df = con.execute(f"""
        SELECT route_id, start_x, start_y
        FROM read_parquet('{valid_parquet}')
    """).fetchdf()

    total_trips = len(coords_df)
    print(f"Total valid trips: {total_trips:,}")

    # Step 3: Query KD-tree for nearest city
    print("Running nearest-neighbor assignment...")
    query_points = coords_df[["start_x", "start_y"]].values
    distances, indices = tree.query(query_points)

    # Assign city and province
    coords_df["city"] = [city_names[i] for i in indices]
    coords_df["province"] = [CITY_TO_PROVINCE.get(city_names[i], "Other") for i in indices]
    coords_df["city_distance_deg"] = distances

    # Mark trips too far from any known city center
    far_mask = distances > MAX_ASSIGN_DISTANCE
    coords_df.loc[far_mask, "city"] = "Other"
    coords_df.loc[far_mask, "province"] = "Other"
    far_count = far_mask.sum()
    print(f"Trips assigned to 'Other' (>{MAX_ASSIGN_DISTANCE} deg from any city): {far_count:,} ({far_count/total_trips:.2%})")

    # Step 4: Register as DuckDB table and join back
    print("Joining city labels with full dataset...")
    con.register("city_labels", coords_df[["route_id", "city", "province", "city_distance_deg"]])

    output_path = DATA_DIR / "routes_with_cities.parquet"
    con.execute(f"""
        COPY (
            SELECT t.*, c.city, c.province, c.city_distance_deg
            FROM read_parquet('{valid_parquet}') t
            JOIN city_labels c ON t.route_id = c.route_id
        )
        TO '{output_path}'
        (FORMAT 'parquet', COMPRESSION 'zstd')
    """)

    # Step 5: Verify and compute statistics
    verify_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
    ).fetchone()[0]
    print(f"\nOutput rows: {verify_count:,} (expected {total_trips:,})")

    # Province-level stats
    print("\n  Province distribution:")
    province_stats = con.execute(f"""
        SELECT province, COUNT(*) as cnt,
               AVG(city_distance_deg) as avg_dist
        FROM read_parquet('{output_path}')
        GROUP BY province
        ORDER BY cnt DESC
    """).fetchdf()

    stats = {"total_trips": total_trips, "far_trips": int(far_count), "provinces": {}}
    for _, row in province_stats.iterrows():
        prov = row["province"]
        cnt = row["cnt"]
        pct = cnt / total_trips * 100
        avg_d = row["avg_dist"]
        print(f"    {prov:<15}: {cnt:>10,} ({pct:>5.1f}%) avg_dist={avg_d:.4f} deg")
        stats["provinces"][prov] = {"count": int(cnt), "pct": round(pct, 2)}

    # City-level stats
    print("\n  City distribution (top 20):")
    city_stats = con.execute(f"""
        SELECT city, province, COUNT(*) as cnt
        FROM read_parquet('{output_path}')
        GROUP BY city, province
        ORDER BY cnt DESC
        LIMIT 20
    """).fetchdf()

    stats["cities"] = {}
    for _, row in city_stats.iterrows():
        city = row["city"]
        prov = row["province"]
        cnt = row["cnt"]
        pct = cnt / total_trips * 100
        print(f"    {city:<25} ({prov:<12}): {cnt:>10,} ({pct:>5.1f}%)")
        stats["cities"][city] = {"province": prov, "count": int(cnt), "pct": round(pct, 2)}

    # Save report
    report_path = DATA_DIR / "city_assignment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    con.close()

    print("\n" + "=" * 70)
    print("  CITY ASSIGNMENT COMPLETE")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    assign_cities()
