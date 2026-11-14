"""
Task 0.3: Assign road class to trips using start-point nearest-edge matching.

Uses start_lat/start_lon from trip_indicators_v2.parquet and matches to
nearest OSM edge via KDTree. This is faster than full-trajectory matching
(~1 query/trip vs ~15/trip) and sufficient since road class is a control
variable, not a primary outcome.

Output:
  - data_parquet/v2/trip_road_classes_v2.parquet
"""

import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, OSM_NETWORKS_DIR, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"

# Road class normalization (same as v1)
ROAD_CLASS_MAP = {
    "motorway": "motorway", "motorway_link": "motorway",
    "trunk": "trunk", "trunk_link": "trunk",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
    "residential": "residential", "living_street": "residential",
    "unclassified": "unclassified",
    "service": "service",
    "cycleway": "cycleway",
    "footway": "footway", "path": "footway", "pedestrian": "footway",
}

# City centers for KDTree-based city assignment
# (same dict used in did_analysis.py / behavioral_substitution.py)
CITY_CENTERS = {
    "Seoul": (37.5665, 126.9780),
    "Busan": (35.1796, 129.0756),
    "Incheon": (37.4563, 126.7052),
    "Daegu": (35.8714, 128.6014),
    "Daejeon": (36.3504, 127.3845),
    "Gwangju": (35.1595, 126.8526),
    "Ulsan": (35.5384, 129.3114),
    "Sejong": (36.4800, 127.0000),
    "Suwon": (37.2636, 127.0286),
    "Seongnam": (37.4201, 127.1265),
    "Goyang": (37.6584, 126.8320),
    "Yongin": (37.2411, 127.1776),
    "Bucheon": (37.5034, 126.7660),
    "Ansan": (37.3219, 126.8309),
    "Anyang": (37.3943, 126.9568),
    "Namyangju": (37.6360, 127.2166),
    "Hwaseong": (37.1995, 126.8313),
    "Uijeongbu": (37.7381, 127.0338),
    "Siheung": (37.3800, 126.8028),
    "Gimpo": (37.6153, 126.7156),
    "Gwangmyeong": (37.4786, 126.8646),
    "Hanam": (37.5390, 127.2145),
    "Gunpo": (37.3614, 126.9352),
    "Pyeongtaek": (36.9922, 127.1129),
    "Osan": (37.1499, 127.0699),
    "Icheon": (37.2720, 127.4349),
    "Paju": (37.7600, 126.7800),
    "Yangju": (37.7852, 127.0460),
    "Guri": (37.5943, 127.1297),
    "Cheonan": (36.8152, 127.1139),
    "Asan": (36.7898, 127.0018),
    "Cheongju": (36.6424, 127.4890),
    "Jeonju": (35.8242, 127.1480),
    "Changwon": (35.2280, 128.6811),
    "Gimhae": (35.2285, 128.8894),
    "Jeju": (33.4996, 126.5312),
    "Chuncheon": (37.8813, 127.7300),
    "Wonju": (37.3422, 127.9202),
    "Pohang": (36.0190, 129.3435),
    "Gumi": (36.1195, 128.3445),
    "Gyeongju": (35.8562, 129.2247),
    "Geoje": (34.8806, 128.6211),
    "Yangsan": (35.3350, 129.0370),
    "Mokpo": (34.7937, 126.3886),
    "Yeosu": (34.7604, 127.6622),
    "Suncheon": (34.9506, 127.4873),
    "Gwangyang": (34.9407, 127.6959),
    "Iksan": (35.9483, 126.9577),
    "Gunsan": (35.9676, 126.7369),
    "Gimje": (35.8037, 126.8809),
    "Tongyeong": (34.8545, 128.4330),
    "Sacheon": (35.0037, 128.0647),
}


def build_global_edge_index() -> tuple[cKDTree, np.ndarray]:
    """Load all city OSM networks and build a single global KDTree.

    Returns:
        (KDTree on edge midpoints, array of road class labels)
    """
    all_coords = []
    all_classes = []

    gpkg_files = sorted(OSM_NETWORKS_DIR.glob("*.gpkg"))
    print(f"  Loading {len(gpkg_files)} city networks...")

    for gpkg_path in gpkg_files:
        try:
            edges_gdf = gpd.read_file(gpkg_path, layer="edges")
        except Exception as e:
            print(f"    WARNING: {gpkg_path.name}: {e}")
            continue

        midpoints = edges_gdf.geometry.interpolate(0.5, normalized=True)
        coords = np.column_stack([midpoints.y, midpoints.x])  # lat, lon
        all_coords.append(coords)

        road_classes = []
        for hw in edges_gdf["highway"]:
            if isinstance(hw, list):
                hw = hw[0]
            elif not isinstance(hw, str):
                hw = "unknown"
            road_classes.append(ROAD_CLASS_MAP.get(hw, "other"))
        all_classes.extend(road_classes)

    all_coords = np.vstack(all_coords)
    all_classes = np.array(all_classes)
    print(f"  Total edges: {len(all_classes):,}")

    tree = cKDTree(all_coords)
    return tree, all_classes


def assign_city(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Assign city to each trip based on nearest city center."""
    city_names = list(CITY_CENTERS.keys())
    city_coords = np.array([CITY_CENTERS[c] for c in city_names])
    city_tree = cKDTree(city_coords)

    trip_coords = np.column_stack([lats, lons])
    _, idxs = city_tree.query(trip_coords, k=1)
    return np.array([city_names[i] for i in idxs])


def main():
    print("=" * 60)
    print("Task 0.3: Road Class Assignment (start-point method)")
    print("=" * 60)

    t_start = time.time()
    np.random.seed(RANDOM_SEED)

    # Load trip start coordinates
    con = duckdb.connect()
    indicators_path = str(V2_DIR / "trip_indicators_v2.parquet").replace("\\", "/")

    print("  Loading trip start coordinates...")
    trips_df = con.execute(f"""
        SELECT route_id, start_lat, start_lon
        FROM read_parquet('{indicators_path}')
    """).fetchdf()
    print(f"  Trips: {len(trips_df):,}")

    # Build global edge index
    tree, road_class_arr = build_global_edge_index()

    # Match start points to nearest edges
    print("  Matching start points to nearest OSM edges...")
    t0 = time.time()
    query_pts = np.column_stack([trips_df["start_lat"].values, trips_df["start_lon"].values])
    dists, idxs = tree.query(query_pts, k=1)
    matched_classes = road_class_arr[idxs]
    print(f"  Matched in {time.time() - t0:.1f}s")

    # Assign city
    print("  Assigning cities...")
    cities = assign_city(trips_df["start_lat"].values, trips_df["start_lon"].values)

    # Build result
    result_df = pd.DataFrame({
        "route_id": trips_df["route_id"].values,
        "start_road_class": matched_classes,
        "city": cities,
        "road_match_dist_deg": np.round(dists, 6),
    })

    # Compute frac_major_road as binary (is start on major road?)
    major_classes = {"motorway", "trunk", "primary", "secondary"}
    result_df["is_major_road"] = result_df["start_road_class"].isin(major_classes).astype(int)

    # Save
    out_path = V2_DIR / "trip_road_classes_v2.parquet"
    result_df.to_parquet(out_path, index=False, compression="zstd")

    elapsed = time.time() - t_start
    print(f"\n  Output: {out_path}")
    print(f"  Time: {elapsed:.1f}s")

    # Stats
    print("\n  Road class distribution:")
    dist = result_df["start_road_class"].value_counts()
    for cls, n in dist.items():
        print(f"    {cls:15s} {n:>10,d} ({n*100/len(result_df):.1f}%)")

    print(f"\n  City distribution (top 10):")
    city_dist = result_df["city"].value_counts().head(10)
    for city, n in city_dist.items():
        print(f"    {city:15s} {n:>10,d} ({n*100/len(result_df):.1f}%)")

    # Save report
    report = {
        "n_trips": len(result_df),
        "n_cities": result_df["city"].nunique(),
        "road_class_distribution": dist.to_dict(),
        "city_distribution": result_df["city"].value_counts().to_dict(),
        "mean_match_dist_deg": float(result_df["road_match_dist_deg"].mean()),
        "time_s": round(elapsed, 1),
    }
    report_path = V2_DIR / "road_class_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    con.close()
    print(f"\n  Report: {report_path}")
    print(f"  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
