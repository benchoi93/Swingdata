"""
OSM Coverage Test for Land Use Data Source Evaluation.

Reproduces the empirical OSM coverage statistics reported in
reports/land_use_data_research.md (Section 2A). Tests landuse polygon
coverage, building footprint coverage, and POI/amenity coverage for
Seoul and Daejeon using existing OSM network GeoPackages and trip
origin data.

Results (2026-03-13):
  - Seoul landuse polygons: 4,778 features, 23.0% area coverage,
    14.5% trip origin match rate
  - Daejeon landuse polygons: 4,636 features, 9.0% area coverage
    Note: Daejeon trip bbox (~1,013 km^2) extends beyond administrative
    city limits (~539 km^2); area coverage denominator is inflated.
  - Seoul buildings (small bbox): 18,842 buildings, 99.6% trips
    within 200m, 68.3% tagged "yes" (unclassified)
  - Seoul amenities (small bbox): 4,914 POIs

Usage:
    python src/v2/osm_coverage_test.py [--skip-api]

    --skip-api: Use cached results only (no Overpass API calls).
                Useful for offline reproducibility checking.

Note: Overpass API queries may take 1-5 minutes each depending on
server load. Results are cached to data_parquet/v2/osm_coverage/
for subsequent runs.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
CACHE_DIR = V2_DIR / "osm_coverage"

# City bounding boxes derived from trip origins in trip_modeling.parquet
# Format: (south, west, north, east) for osmnx
CITY_BBOXES: dict[str, dict[str, Any]] = {
    "Seoul": {
        "description": "Trip-origin bounding box for Seoul",
        "province": "서울특별시",
    },
    "Daejeon": {
        "description": "Trip-origin bounding box for Daejeon",
        "province": "대전광역시",
    },
}

# Smaller bbox for detailed building/amenity tests (central Seoul)
SEOUL_SMALL_BBOX = {
    "south": 37.54,
    "west": 126.97,
    "north": 37.56,
    "east": 127.00,
    "description": "Central Seoul ~2x3km for building/amenity detail test",
}

np.random.seed(RANDOM_SEED)


def get_trip_bbox(con: duckdb.DuckDBPyConnection, province: str) -> dict:
    """Get bounding box of trip origins for a given province."""
    in_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")
    result = con.execute(f"""
        SELECT
            MIN(start_lat) AS south,
            MAX(start_lat) AS north,
            MIN(start_lon) AS west,
            MAX(start_lon) AS east,
            COUNT(*) AS n_trips
        FROM read_parquet('{in_path}')
        WHERE province = '{province}'
    """).fetchone()
    return {
        "south": result[0],
        "north": result[1],
        "west": result[2],
        "east": result[3],
        "n_trips": result[4],
    }


def sample_trip_origins(
    con: duckdb.DuckDBPyConnection, province: str, n: int = 50_000
) -> np.ndarray:
    """Sample trip origin coordinates for spatial matching tests."""
    in_path = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")
    df = con.execute(f"""
        SELECT start_lat, start_lon
        FROM read_parquet('{in_path}')
        WHERE province = '{province}'
        ORDER BY RANDOM()
        LIMIT {n}
    """).fetchdf()
    return df[["start_lat", "start_lon"]].values


def test_landuse_coverage(
    city: str, bbox: dict, trip_coords: np.ndarray | None = None,
    skip_api: bool = False
) -> dict:
    """Test OSM landuse polygon coverage for a city bbox.

    Parameters
    ----------
    city : str
        City name for reporting.
    bbox : dict
        Dict with south, west, north, east keys.
    trip_coords : np.ndarray | None
        Array of (lat, lon) trip origins for match rate test.
    skip_api : bool
        If True, load from cache only.

    Returns
    -------
    dict
        Coverage statistics.
    """
    cache_file = CACHE_DIR / f"landuse_{city.lower()}.gpkg"

    if cache_file.exists() and skip_api:
        print(f"  Loading cached landuse for {city}...")
        gdf = gpd.read_file(cache_file)
    elif cache_file.exists():
        print(f"  Loading cached landuse for {city}...")
        gdf = gpd.read_file(cache_file)
    else:
        print(f"  Querying Overpass API for landuse in {city}...")
        tags = {"landuse": True}
        gdf = ox.features_from_bbox(
            bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            tags=tags,
        )
        # Save cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Keep only Polygon/MultiPolygon geometries
        gdf_poly = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf_poly.to_file(cache_file, driver="GPKG")
        gdf = gdf_poly

    # Filter to polygons only
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # Compute statistics
    n_features = len(gdf)

    # Area calculation in projected CRS (EPSG:5179 for Korea)
    gdf_proj = gdf.to_crs(epsg=5179)
    total_landuse_area_km2 = gdf_proj.geometry.area.sum() / 1e6

    # Bbox area
    bbox_polygon = box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon], crs="EPSG:4326")
    bbox_area_km2 = bbox_gdf.to_crs(epsg=5179).geometry.area.sum() / 1e6

    area_coverage_pct = 100.0 * total_landuse_area_km2 / bbox_area_km2

    # Landuse tag distribution
    if "landuse" in gdf.columns:
        tag_dist = gdf["landuse"].value_counts().head(10).to_dict()
    else:
        tag_dist = {}

    result = {
        "city": city,
        "n_features": n_features,
        "landuse_area_km2": round(total_landuse_area_km2, 1),
        "bbox_area_km2": round(bbox_area_km2, 1),
        "area_coverage_pct": round(area_coverage_pct, 1),
        "top_tags": tag_dist,
    }

    # Trip origin match rate (if coordinates provided)
    if trip_coords is not None:
        from shapely.geometry import Point
        from shapely.strtree import STRtree

        print(f"  Computing trip origin match rate ({len(trip_coords)} origins)...")
        # Build spatial index
        tree = STRtree(gdf.geometry.values)
        points = [Point(lon, lat) for lat, lon in trip_coords]
        n_matched = sum(1 for p in points if len(tree.query(p, predicate="within")) > 0)
        match_pct = 100.0 * n_matched / len(trip_coords)
        result["trip_match_n"] = len(trip_coords)
        result["trip_match_count"] = n_matched
        result["trip_match_pct"] = round(match_pct, 1)

    return result


def test_building_coverage(
    bbox: dict, trip_coords: np.ndarray | None = None,
    skip_api: bool = False
) -> dict:
    """Test OSM building footprint coverage for a small bbox.

    Parameters
    ----------
    bbox : dict
        Dict with south, west, north, east keys.
    trip_coords : np.ndarray | None
        Array of (lat, lon) trip origins for proximity test.
    skip_api : bool
        If True, load from cache only.

    Returns
    -------
    dict
        Building coverage statistics.
    """
    cache_file = CACHE_DIR / "buildings_seoul_small.gpkg"

    if cache_file.exists():
        print("  Loading cached buildings...")
        gdf = gpd.read_file(cache_file)
    elif skip_api:
        print("  No cached buildings and --skip-api set. Skipping.")
        return {"skipped": True}
    else:
        print("  Querying Overpass API for buildings (small bbox)...")
        tags = {"building": True}
        gdf = ox.features_from_bbox(
            bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            tags=tags,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        gdf_save = gdf[gdf.geometry.type.isin(
            ["Polygon", "MultiPolygon", "Point"]
        )].copy()
        gdf_save.to_file(cache_file, driver="GPKG")
        gdf = gdf_save

    n_buildings = len(gdf)

    # Building type distribution
    if "building" in gdf.columns:
        type_dist = gdf["building"].value_counts().head(10).to_dict()
        n_yes = int((gdf["building"] == "yes").sum())
        pct_unclassified = round(100.0 * n_yes / n_buildings, 1)
    else:
        type_dist = {}
        pct_unclassified = None

    result = {
        "n_buildings": n_buildings,
        "type_distribution": type_dist,
        "pct_unclassified_yes": pct_unclassified,
    }

    # Proximity test: what fraction of trip origins have a building within 200m?
    if trip_coords is not None:
        print("  Computing building proximity (200m)...")
        # Get building centroids in projected CRS
        gdf_proj = gdf.to_crs(epsg=5179)
        centroids = np.array(
            [(g.centroid.x, g.centroid.y) for g in gdf_proj.geometry]
        )
        if len(centroids) > 0:
            tree = cKDTree(centroids)
            # Project trip coords
            trip_gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(trip_coords[:, 1], trip_coords[:, 0]),
                crs="EPSG:4326",
            ).to_crs(epsg=5179)
            trip_xy = np.array([(g.x, g.y) for g in trip_gdf.geometry])
            distances, _ = tree.query(trip_xy)
            n_within_200m = int((distances <= 200).sum())
            pct_within_200m = round(100.0 * n_within_200m / len(trip_coords), 1)
            result["trip_proximity_n"] = len(trip_coords)
            result["n_within_200m"] = n_within_200m
            result["pct_within_200m"] = pct_within_200m

    return result


def test_amenity_coverage(bbox: dict, skip_api: bool = False) -> dict:
    """Test OSM amenity/POI coverage for a small bbox.

    Parameters
    ----------
    bbox : dict
        Dict with south, west, north, east keys.
    skip_api : bool
        If True, load from cache only.

    Returns
    -------
    dict
        Amenity coverage statistics.
    """
    cache_file = CACHE_DIR / "amenities_seoul_small.gpkg"

    if cache_file.exists():
        print("  Loading cached amenities...")
        gdf = gpd.read_file(cache_file)
    elif skip_api:
        print("  No cached amenities and --skip-api set. Skipping.")
        return {"skipped": True}
    else:
        print("  Querying Overpass API for amenities (small bbox)...")
        tags = {"amenity": True}
        gdf = ox.features_from_bbox(
            bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            tags=tags,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache_file, driver="GPKG")

    n_amenities = len(gdf)

    # Amenity type distribution
    if "amenity" in gdf.columns:
        type_dist = gdf["amenity"].value_counts().head(10).to_dict()
    else:
        type_dist = {}

    return {
        "n_amenities": n_amenities,
        "type_distribution": type_dist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test OSM coverage for land use data source evaluation"
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Use cached results only (no Overpass API calls)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OSM Coverage Test for Land Use Data Source Evaluation")
    print("=" * 60)
    t_start = time.time()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to DuckDB for trip origin queries
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")

    results: dict[str, Any] = {}

    # --- Test 1: Landuse polygon coverage ---
    print("\n--- Test 1: Landuse Polygon Coverage ---")
    for city, info in CITY_BBOXES.items():
        print(f"\n  [{city}]")
        bbox = get_trip_bbox(con, info["province"])
        print(f"  Trips: {bbox['n_trips']:,}")
        print(f"  Bbox: S={bbox['south']:.4f} W={bbox['west']:.4f} "
              f"N={bbox['north']:.4f} E={bbox['east']:.4f}")

        # Sample trip origins for match rate (Seoul only)
        trip_coords = None
        if city == "Seoul":
            trip_coords = sample_trip_origins(con, info["province"], n=50_000)
            print(f"  Sampled {len(trip_coords)} trip origins for match test")

        landuse_result = test_landuse_coverage(
            city, bbox, trip_coords, skip_api=args.skip_api
        )
        results[f"landuse_{city.lower()}"] = landuse_result

        print(f"\n  Results for {city}:")
        print(f"    Landuse features: {landuse_result['n_features']:,}")
        print(f"    Landuse area: {landuse_result['landuse_area_km2']} km^2")
        print(f"    Bbox area: {landuse_result['bbox_area_km2']} km^2")
        print(f"    Area coverage: {landuse_result['area_coverage_pct']}%")
        if "trip_match_pct" in landuse_result:
            print(f"    Trip match rate: {landuse_result['trip_match_pct']}% "
                  f"({landuse_result['trip_match_count']}/{landuse_result['trip_match_n']})")
        if landuse_result.get("top_tags"):
            print(f"    Top tags: {landuse_result['top_tags']}")

    # --- Test 2: Building footprint coverage (Seoul small bbox) ---
    print("\n\n--- Test 2: Building Footprint Coverage (Central Seoul) ---")
    # Sample trip origins within small bbox for proximity test
    small_bbox_origins = con.execute(f"""
        SELECT start_lat, start_lon
        FROM read_parquet('{str(V2_DIR / "trip_modeling.parquet").replace(chr(92), "/")}')
        WHERE province = '서울특별시'
          AND start_lat BETWEEN {SEOUL_SMALL_BBOX['south']} AND {SEOUL_SMALL_BBOX['north']}
          AND start_lon BETWEEN {SEOUL_SMALL_BBOX['west']} AND {SEOUL_SMALL_BBOX['east']}
        ORDER BY RANDOM()
        LIMIT 5000
    """).fetchdf()[["start_lat", "start_lon"]].values

    print(f"  Small bbox: {SEOUL_SMALL_BBOX['description']}")
    print(f"  Trip origins in small bbox: {len(small_bbox_origins)}")

    building_result = test_building_coverage(
        SEOUL_SMALL_BBOX, small_bbox_origins, skip_api=args.skip_api
    )
    results["buildings_seoul_small"] = building_result

    if not building_result.get("skipped"):
        print(f"\n  Results:")
        print(f"    Buildings: {building_result['n_buildings']:,}")
        print(f"    Unclassified ('yes'): {building_result.get('pct_unclassified_yes')}%")
        if "pct_within_200m" in building_result:
            print(f"    Trips within 200m of building: {building_result['pct_within_200m']}%")
        if building_result.get("type_distribution"):
            print(f"    Top types: {building_result['type_distribution']}")

    # --- Test 3: Amenity/POI coverage (Seoul small bbox) ---
    print("\n\n--- Test 3: Amenity/POI Coverage (Central Seoul) ---")
    amenity_result = test_amenity_coverage(SEOUL_SMALL_BBOX, skip_api=args.skip_api)
    results["amenities_seoul_small"] = amenity_result

    if not amenity_result.get("skipped"):
        print(f"\n  Results:")
        print(f"    Amenities: {amenity_result['n_amenities']:,}")
        if amenity_result.get("type_distribution"):
            print(f"    Top types: {amenity_result['type_distribution']}")

    # --- Save results ---
    out_path = CACHE_DIR / "osm_coverage_results.json"

    # Convert any non-serializable types
    def make_serializable(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(
        json.dumps(results, default=make_serializable)
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # --- Summary ---
    con.close()
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"OSM Coverage Test Complete ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    print("\nKey findings:")
    print("  - OSM landuse polygon coverage is INSUFFICIENT for Korean cities")
    print("    (14.5% trip match in Seoul, likely worse elsewhere)")
    print("  - Building footprints exist but 68% have no type classification")
    print("  - POI/amenity data is the best OSM-based proxy for land use")
    print("  -> Recommend MOLIT zoning or SGIS census data instead")


if __name__ == "__main__":
    main()
