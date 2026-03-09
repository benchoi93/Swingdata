"""
Task 3.3-3.4: Assign road class to GPS points using nearest-edge matching.

Uses DuckDB regex to extract GPS coordinates from routes_raw (avoids slow
ast.literal_eval), then scipy KDTree for vectorized nearest-edge lookup.
Processes city-by-city, extracts highway tags, and computes per-trip
road class composition.

Outputs:
  - data_parquet/trip_road_classes.parquet -- trip-level road class features
  - data_parquet/modeling/road_class_assignment_report.json -- stats and diagnostics
"""

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    CLEANED_PARQUET,
    DATA_DIR,
    MODELING_DIR,
    OSM_NETWORKS_DIR,
    RANDOM_SEED,
)

# Output paths
TRIP_ROAD_CLASSES_PATH = DATA_DIR / "trip_road_classes.parquet"
REPORT_PATH = MODELING_DIR / "road_class_assignment_report.json"
MODELING_DIR.mkdir(parents=True, exist_ok=True)

# Road class normalization (OSM highway tag -> simplified category)
ROAD_CLASS_MAP = {
    "motorway": "motorway",
    "motorway_link": "motorway",
    "trunk": "trunk",
    "trunk_link": "trunk",
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "residential": "residential",
    "living_street": "residential",
    "unclassified": "unclassified",
    "service": "service",
    "cycleway": "cycleway",
    "footway": "footway",
    "path": "footway",
    "pedestrian": "footway",
    "track": "other",
    "busway": "other",
    "steps": "other",
    "construction": "other",
    "proposed": "other",
    "corridor": "other",
    "bridleway": "other",
}

ROAD_CATEGORIES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "unclassified", "service", "cycleway", "footway", "other",
]


def load_city_edge_index(city: str) -> Optional[tuple[cKDTree, np.ndarray]]:
    """Load OSM edges for a city and build KDTree spatial index.

    Args:
        city: City name (must match filename in osm_networks/).

    Returns:
        Tuple of (KDTree built on edge midpoints, array of road class strings),
        or None if the network file doesn't exist.
    """
    gpkg_path = OSM_NETWORKS_DIR / f"{city}.gpkg"
    if not gpkg_path.exists():
        return None

    edges_gdf = gpd.read_file(gpkg_path, layer="edges")

    # Compute edge midpoints for KDTree
    midpoints = edges_gdf.geometry.interpolate(0.5, normalized=True)
    coords = np.column_stack([midpoints.x, midpoints.y])
    tree = cKDTree(coords)

    # Extract and normalize highway tags
    road_classes = []
    for hw in edges_gdf["highway"]:
        if isinstance(hw, list):
            hw = hw[0]
        elif not isinstance(hw, str):
            hw = "unknown"
        road_classes.append(ROAD_CLASS_MAP.get(hw, "other"))
    road_class_arr = np.array(road_classes)

    return tree, road_class_arr


def extract_gps_points_duckdb(city: str) -> pd.DataFrame:
    """Extract flat (route_id, lat, lon) table using DuckDB regex.

    Uses DuckDB regexp_extract_all + UNNEST to parse coordinates from
    routes_raw strings, avoiding slow Python ast.literal_eval.

    Args:
        city: City name to filter trips.

    Returns:
        DataFrame with columns [route_id, lat, lon].
    """
    con = duckdb.connect()
    # Use regex to extract all "lat, lon" pairs from routes_raw
    # Format: [['date', 'time', 37.xxx, 126.xxx], ...]
    df = con.execute(f"""
        WITH raw AS (
            SELECT route_id,
                   regexp_extract_all(
                       routes_raw,
                       '(\\d+\\.\\d+),\\s*(\\d+\\.\\d+)'
                   ) as coords
            FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
            WHERE is_valid = true AND city = '{city}'
        ),
        unnested AS (
            SELECT route_id,
                   UNNEST(coords) as coord_str
            FROM raw
        )
        SELECT route_id,
               CAST(split_part(coord_str, ', ', 1) AS DOUBLE) as lat,
               CAST(split_part(coord_str, ', ', 2) AS DOUBLE) as lon
        FROM unnested
    """).fetchdf()
    con.close()
    return df


def compute_trip_road_features(
    trip_classes: pd.Series,
    route_ids: pd.Series,
) -> pd.DataFrame:
    """Compute per-trip road class composition from matched road classes.

    Args:
        trip_classes: Series of road class labels for each GPS point.
        route_ids: Series of route_id for each GPS point.

    Returns:
        DataFrame with one row per trip and road class fraction columns.
    """
    # Create a DataFrame for groupby
    points_df = pd.DataFrame({"route_id": route_ids.values, "road_class": trip_classes})

    # Count total points per trip
    trip_counts = points_df.groupby("route_id").size().rename("total_pts")

    # Count per road class per trip
    class_counts = (
        points_df.groupby(["route_id", "road_class"])
        .size()
        .unstack(fill_value=0)
    )

    # Compute fractions
    fractions = class_counts.div(trip_counts, axis=0)

    # Ensure all categories exist
    for cat in ROAD_CATEGORIES:
        col = cat
        if col not in fractions.columns:
            fractions[col] = 0.0

    # Build result
    result = pd.DataFrame(index=fractions.index)
    for cat in ROAD_CATEGORIES:
        result[f"frac_{cat}"] = fractions[cat].values if cat in fractions.columns else 0.0

    # Dominant road class
    result["dominant_road_class"] = class_counts.idxmax(axis=1)

    # Number of distinct road classes
    result["n_road_classes"] = (class_counts > 0).sum(axis=1)

    # Major road fraction
    major_cols = [c for c in ["motorway", "trunk", "primary", "secondary"] if c in class_counts.columns]
    if major_cols:
        result["frac_major_road"] = class_counts[major_cols].sum(axis=1) / trip_counts
    else:
        result["frac_major_road"] = 0.0

    # Cycling infra fraction
    result["frac_cycling_infra"] = (
        class_counts["cycleway"] / trip_counts if "cycleway" in class_counts.columns else 0.0
    )

    result = result.reset_index()
    return result


def process_city(
    city: str,
    tree: cKDTree,
    road_class_arr: np.ndarray,
) -> pd.DataFrame:
    """Process all valid trips in a city for road class assignment.

    Args:
        city: City name.
        tree: Pre-built KDTree for this city.
        road_class_arr: Road class labels for each edge.

    Returns:
        DataFrame with route_id and road class features.
    """
    # Extract all GPS points using DuckDB (fast)
    t0 = time.time()
    points_df = extract_gps_points_duckdb(city)
    t_extract = time.time() - t0

    if len(points_df) == 0:
        return pd.DataFrame()

    # Vectorized KDTree query
    t1 = time.time()
    query_pts = np.column_stack([
        points_df["lon"].values,
        points_df["lat"].values,
    ])
    _, idxs = tree.query(query_pts, k=1)
    matched_classes = road_class_arr[idxs]
    t_match = time.time() - t1

    # Compute per-trip features
    t2 = time.time()
    result = compute_trip_road_features(
        pd.Series(matched_classes),
        points_df["route_id"],
    )
    t_agg = time.time() - t2

    print(f"    Extract: {t_extract:.1f}s | Match: {t_match:.1f}s | "
          f"Aggregate: {t_agg:.1f}s | "
          f"Points: {len(points_df):,} | Trips: {len(result):,}")

    return result


def main() -> None:
    """Run road class assignment for all cities."""
    print("=" * 60)
    print("Task 3.3-3.4: Road Class Assignment via Nearest-Edge Matching")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    t_start = time.time()

    # Get list of cities with valid trips
    con = duckdb.connect()
    cities_df = con.execute(f"""
        SELECT city, COUNT(*) as n_trips, SUM(gps_points) as total_pts
        FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
        WHERE is_valid = true
        GROUP BY city
        ORDER BY n_trips DESC
    """).fetchdf()
    con.close()

    print(f"\nCities to process: {len(cities_df)}")
    print(f"Total trips: {cities_df['n_trips'].sum():,}")
    print(f"Total GPS points: {cities_df['total_pts'].sum():,}")

    all_results = []
    city_stats = []
    skipped_cities = []

    for idx, row in cities_df.iterrows():
        city = row["city"]
        n_trips = int(row["n_trips"])
        total_pts = int(row["total_pts"])

        print(f"\n[{idx+1}/{len(cities_df)}] {city}: "
              f"{n_trips:,} trips, {total_pts:,} GPS points")

        # Load spatial index
        index_data = load_city_edge_index(city)
        if index_data is None:
            print(f"  SKIPPED: No OSM network for {city}")
            skipped_cities.append({"city": city, "n_trips": n_trips})
            continue

        tree, road_class_arr = index_data
        print(f"  KDTree built: {len(road_class_arr):,} edges")

        # Process trips
        t0 = time.time()
        city_df = process_city(city, tree, road_class_arr)
        elapsed = time.time() - t0

        if len(city_df) > 0:
            dominant_dist = city_df["dominant_road_class"].value_counts()
            stats = {
                "city": city,
                "n_trips": n_trips,
                "n_processed": len(city_df),
                "processing_time_s": round(elapsed, 1),
                "points_per_second": round(total_pts / max(elapsed, 0.001)),
                "dominant_road_class_dist": {
                    str(k): int(v) for k, v in dominant_dist.head(5).items()
                },
                "mean_frac_major_road": round(
                    float(city_df["frac_major_road"].mean()), 3
                ),
            }
            city_stats.append(stats)
            print(f"  Top road class: {dominant_dist.index[0]} "
                  f"({dominant_dist.values[0] / len(city_df) * 100:.1f}%)")

        all_results.append(city_df)

    # Combine all cities
    print("\n" + "=" * 60)
    print("Combining results...")
    result_df = pd.concat(all_results, ignore_index=True)
    print(f"Total trips with road class: {len(result_df):,}")

    # Save to parquet
    result_df.to_parquet(TRIP_ROAD_CLASSES_PATH, index=False)
    print(f"Saved to: {TRIP_ROAD_CLASSES_PATH}")

    # Overall statistics
    valid_mask = result_df["dominant_road_class"].notna()
    overall_dominant = result_df.loc[valid_mask, "dominant_road_class"].value_counts()
    overall_frac_major = result_df["frac_major_road"].mean()

    total_time = time.time() - t_start

    # Summary report
    report = {
        "task": "3.3-3.4",
        "description": "Road class assignment via nearest-edge KDTree matching",
        "method": "DuckDB regex extraction + scipy cKDTree on edge midpoints",
        "total_trips_processed": len(result_df),
        "total_time_s": round(total_time, 1),
        "skipped_cities": skipped_cities,
        "overall_dominant_road_class_dist": {
            str(k): int(v) for k, v in overall_dominant.items()
        },
        "overall_mean_frac_major_road": round(float(overall_frac_major), 3),
        "road_class_fractions_mean": {
            f"frac_{cat}": round(float(result_df[f"frac_{cat}"].mean()), 4)
            for cat in ROAD_CATEGORIES
        },
        "city_stats": city_stats,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {REPORT_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Trips processed: {len(result_df):,}")
    print(f"Skipped cities: {len(skipped_cities)} "
          f"({sum(s['n_trips'] for s in skipped_cities):,} trips)")
    print(f"Total time: {total_time:.1f}s")
    print(f"\nOverall road class distribution (dominant per trip):")
    for k, v in overall_dominant.head(10).items():
        print(f"  {k:20s} {v:>10,} ({v / len(result_df) * 100:.1f}%)")
    print(f"\nMean fraction on major roads: {overall_frac_major:.3f}")
    for cat in ROAD_CATEGORIES:
        mean_frac = result_df[f"frac_{cat}"].mean()
        if mean_frac > 0.005:
            print(f"  frac_{cat:15s}: {mean_frac:.4f}")


if __name__ == "__main__":
    main()
