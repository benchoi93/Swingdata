"""
Task 3.1: Extract OSM road network for all identified cities using osmnx.

Downloads road networks for each city where e-scooter trips were observed.
Uses actual trip bounding boxes (with buffer) to define download areas.
Saves networks as GeoPackage files for later map-matching.

Outputs:
  - data_parquet/osm_networks/{city}.gpkg — road network per city
  - data_parquet/osm_networks/network_summary.json — extraction statistics
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import osmnx as ox
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, CLEANED_PARQUET

# Output directory
OSM_DIR = DATA_DIR / "osm_networks"
OSM_DIR.mkdir(parents=True, exist_ok=True)

# Buffer around bounding box in degrees (~500m)
BBOX_BUFFER_DEG = 0.005

# Minimum trip count to extract network for a city
MIN_TRIPS_FOR_EXTRACTION = 5000

# Network type: "all" includes drive, bike, walk; we want roads e-scooters use
NETWORK_TYPE = "all"

# Useful tags to retain on edges
USEFUL_EDGE_TAGS = [
    "highway",
    "maxspeed",
    "lanes",
    "cycleway",
    "cycleway:left",
    "cycleway:right",
    "bicycle",
    "surface",
    "oneway",
    "name",
    "width",
    "lit",
]


def get_city_bounding_boxes(min_trips: int = MIN_TRIPS_FOR_EXTRACTION) -> pd.DataFrame:
    """Query cleaned dataset to get bounding boxes for each city.

    Args:
        min_trips: Minimum number of trips to include a city.

    Returns:
        DataFrame with city, province, trip_count, min/max lat/lon, center coords.
    """
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            province,
            city,
            COUNT(*) as trip_count,
            MIN(start_lat) as min_lat,
            MAX(start_lat) as max_lat,
            MIN(start_lon) as min_lon,
            MAX(start_lon) as max_lon,
            AVG(start_lat) as center_lat,
            AVG(start_lon) as center_lon
        FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
        GROUP BY province, city
        HAVING COUNT(*) >= {min_trips}
        ORDER BY trip_count DESC
    """).fetchdf()
    con.close()
    return df


def extract_network_for_city(
    city: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    network_type: str = NETWORK_TYPE,
) -> tuple[Any, dict]:
    """Download OSM road network for a city's bounding box.

    Args:
        city: City name.
        min_lat, max_lat, min_lon, max_lon: Bounding box from trip data.
        network_type: OSM network type to download.

    Returns:
        Tuple of (networkx graph, stats dict).
    """
    # Add buffer around bounding box
    north = max_lat + BBOX_BUFFER_DEG
    south = min_lat - BBOX_BUFFER_DEG
    east = max_lon + BBOX_BUFFER_DEG
    west = min_lon - BBOX_BUFFER_DEG

    # Configure osmnx
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.useful_tags_way = USEFUL_EDGE_TAGS

    t0 = time.time()
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type)
    elapsed = time.time() - t0

    # Compute basic stats
    stats = {
        "city": city,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "bbox": {"north": north, "south": south, "east": east, "west": west},
        "download_time_s": round(elapsed, 1),
    }

    return G, stats


def save_network(G: Any, city: str, output_dir: Path = OSM_DIR) -> dict:
    """Save network as GeoPackage and extract edge attributes summary.

    Args:
        G: NetworkX graph from osmnx.
        city: City name for filename.
        output_dir: Directory to save files.

    Returns:
        Dict with file paths and edge attribute coverage.
    """
    # Convert to GeoDataFrames
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Save as GeoPackage (efficient, single file)
    gpkg_path = output_dir / f"{city}.gpkg"
    nodes_gdf.to_file(gpkg_path, layer="nodes", driver="GPKG")
    edges_gdf.to_file(gpkg_path, layer="edges", driver="GPKG")

    # Analyze edge attributes
    attr_coverage = {}
    for col in ["highway", "maxspeed", "lanes", "cycleway", "surface", "lit", "name"]:
        if col in edges_gdf.columns:
            non_null = edges_gdf[col].notna().sum()
            attr_coverage[col] = {
                "count": int(non_null),
                "pct": round(100 * non_null / len(edges_gdf), 1),
            }
        else:
            attr_coverage[col] = {"count": 0, "pct": 0.0}

    # Highway type distribution
    if "highway" in edges_gdf.columns:
        highway_dist = edges_gdf["highway"].apply(
            lambda x: x if isinstance(x, str) else (x[0] if isinstance(x, list) else str(x))
        ).value_counts().head(15).to_dict()
    else:
        highway_dist = {}

    return {
        "gpkg_path": str(gpkg_path),
        "n_nodes": len(nodes_gdf),
        "n_edges": len(edges_gdf),
        "attr_coverage": attr_coverage,
        "highway_distribution": {str(k): int(v) for k, v in highway_dist.items()},
    }


def main() -> None:
    """Extract OSM networks for all qualifying cities."""
    print("=" * 60)
    print("Task 3.1: Extract OSM Road Networks")
    print("=" * 60)

    # Get city bounding boxes
    print("\nQuerying city bounding boxes from cleaned dataset...")
    cities_df = get_city_bounding_boxes(min_trips=MIN_TRIPS_FOR_EXTRACTION)
    print(f"Found {len(cities_df)} cities with >= {MIN_TRIPS_FOR_EXTRACTION} trips")

    # Track results
    all_stats = []
    failed_cities = []

    for idx, row in cities_df.iterrows():
        city = row["city"]
        gpkg_path = OSM_DIR / f"{city}.gpkg"

        # Skip if already downloaded
        if gpkg_path.exists():
            print(f"\n[{idx+1}/{len(cities_df)}] {city} — already exists, skipping")
            # Load existing to get stats
            try:
                edges_gdf = gpd.read_file(gpkg_path, layer="edges")
                nodes_gdf = gpd.read_file(gpkg_path, layer="nodes")
                all_stats.append({
                    "city": city,
                    "province": row["province"],
                    "trip_count": int(row["trip_count"]),
                    "nodes": len(nodes_gdf),
                    "edges": len(edges_gdf),
                    "status": "cached",
                })
            except Exception:
                all_stats.append({
                    "city": city,
                    "province": row["province"],
                    "trip_count": int(row["trip_count"]),
                    "status": "cached_unreadable",
                })
            continue

        print(f"\n[{idx+1}/{len(cities_df)}] Downloading {city} "
              f"({row['province']}, {int(row['trip_count']):,} trips)...")
        print(f"  BBox: lat [{row['min_lat']:.3f}, {row['max_lat']:.3f}] "
              f"lon [{row['min_lon']:.3f}, {row['max_lon']:.3f}]")

        try:
            G, dl_stats = extract_network_for_city(
                city=city,
                min_lat=row["min_lat"],
                max_lat=row["max_lat"],
                min_lon=row["min_lon"],
                max_lon=row["max_lon"],
            )
            print(f"  Downloaded: {dl_stats['nodes']:,} nodes, "
                  f"{dl_stats['edges']:,} edges in {dl_stats['download_time_s']}s")

            save_info = save_network(G, city)
            print(f"  Saved to: {save_info['gpkg_path']}")
            print(f"  Highway coverage: {save_info['attr_coverage'].get('highway', {}).get('pct', 0)}%")
            print(f"  Maxspeed coverage: {save_info['attr_coverage'].get('maxspeed', {}).get('pct', 0)}%")

            all_stats.append({
                "city": city,
                "province": row["province"],
                "trip_count": int(row["trip_count"]),
                "nodes": dl_stats["nodes"],
                "edges": dl_stats["edges"],
                "download_time_s": dl_stats["download_time_s"],
                "attr_coverage": save_info["attr_coverage"],
                "highway_distribution": save_info["highway_distribution"],
                "status": "success",
            })

            # Be polite to OSM servers
            time.sleep(2)

        except Exception as e:
            print(f"  FAILED: {e}")
            failed_cities.append({"city": city, "error": str(e)})
            all_stats.append({
                "city": city,
                "province": row["province"],
                "trip_count": int(row["trip_count"]),
                "status": "failed",
                "error": str(e),
            })
            # Continue with next city
            time.sleep(5)

    # Save summary
    summary = {
        "total_cities_attempted": len(cities_df),
        "successful": sum(1 for s in all_stats if s["status"] in ("success", "cached")),
        "failed": len(failed_cities),
        "min_trips_threshold": MIN_TRIPS_FOR_EXTRACTION,
        "network_type": NETWORK_TYPE,
        "bbox_buffer_deg": BBOX_BUFFER_DEG,
        "cities": all_stats,
        "failed_cities": failed_cities,
    }

    summary_path = OSM_DIR / "network_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cities attempted: {len(cities_df)}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    total_nodes = sum(s.get("nodes", 0) for s in all_stats)
    total_edges = sum(s.get("edges", 0) for s in all_stats)
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total edges: {total_edges:,}")
    if failed_cities:
        print(f"\nFailed cities: {[c['city'] for c in failed_cities]}")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
