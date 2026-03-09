"""
Spatial analysis of e-scooter trip origins/destinations using H3 hexagonal grids.

Tasks 4.1-4.5: H3 aggregation, Getis-Ord Gi* hotspot detection, Moran's I
spatial autocorrelation, and infrastructure overlay.

Usage:
    python src/spatial_analysis.py
"""

import json
import time
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from libpysal.weights import KNN, Queen
from esda.getisord import G_Local
from esda.moran import Moran, Moran_Local
from shapely.geometry import Polygon

from config import (
    DATA_DIR, CLEANED_PARQUET, MODELING_DIR, REPORTS_DIR,
    H3_RESOLUTION, SPEED_LIMIT_KR, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)


# ---- Paths ----
SPATIAL_DIR = DATA_DIR / "spatial"
SPATIAL_DIR.mkdir(exist_ok=True)


def h3_to_polygon(h3_index: str) -> Polygon:
    """Convert H3 hex index to a Shapely Polygon."""
    boundary = h3.cell_to_boundary(h3_index)
    # h3 returns (lat, lon) tuples; shapely needs (lon, lat)
    coords = [(lon, lat) for lat, lon in boundary]
    coords.append(coords[0])  # close the ring
    return Polygon(coords)


def load_trip_data() -> pd.DataFrame:
    """Load trip data with coordinates and speed indicators via DuckDB."""
    print("Loading trip data...")
    con = duckdb.connect()
    df = con.execute("""
        SELECT
            c.route_id,
            c.start_lat, c.start_lon,
            c.end_lat, c.end_lon,
            c.city, c.province, c.mode,
            m.mean_speed, m.max_speed_from_profile,
            m.speeding_rate_25, m.is_speeding,
            m.speed_cv, m.harsh_event_count,
            m.p85_speed
        FROM read_parquet($cleaned) c
        JOIN read_parquet($modeling) m USING (route_id)
    """, {
        "cleaned": str(CLEANED_PARQUET / "trips_cleaned.parquet"),
        "modeling": str(MODELING_DIR / "trip_modeling.parquet"),
    }).df()
    print(f"  Loaded {len(df):,} trips")
    return df


def compute_h3_indices(df: pd.DataFrame, resolution: int = H3_RESOLUTION) -> pd.DataFrame:
    """Add H3 hex indices for trip origins and destinations."""
    print(f"Computing H3 indices at resolution {resolution}...")
    t0 = time.time()
    df["h3_origin"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(df["start_lat"], df["start_lon"])
    ]
    df["h3_dest"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(df["end_lat"], df["end_lon"])
    ]
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Unique origin hexes: {df['h3_origin'].nunique():,}")
    print(f"  Unique dest hexes: {df['h3_dest'].nunique():,}")
    return df


def aggregate_hexagons(
    df: pd.DataFrame,
    h3_col: str = "h3_origin",
    label: str = "origin",
    min_trips: int = 10,
) -> gpd.GeoDataFrame:
    """Aggregate trip-level data to H3 hexagons.

    Args:
        df: Trip dataframe with H3 indices and speed indicators.
        h3_col: Column name for H3 index.
        label: Label for the aggregation (origin/dest).
        min_trips: Minimum trips per hex for inclusion.

    Returns:
        GeoDataFrame with hex geometries and aggregated indicators.
    """
    print(f"Aggregating {label} hexagons (min_trips={min_trips})...")
    agg = df.groupby(h3_col).agg(
        trip_count=("route_id", "count"),
        mean_speed=("mean_speed", "mean"),
        median_speed=("mean_speed", "median"),
        p85_speed=("p85_speed", "mean"),
        max_speed_mean=("max_speed_from_profile", "mean"),
        speeding_rate=("speeding_rate_25", "mean"),
        speeding_trips=("is_speeding", "sum"),
        speeding_pct=("is_speeding", "mean"),
        speed_cv_mean=("speed_cv", "mean"),
        harsh_events_mean=("harsh_event_count", "mean"),
    ).reset_index()

    agg.rename(columns={h3_col: "h3_index"}, inplace=True)
    agg["speeding_pct"] = agg["speeding_pct"] * 100  # convert to percentage

    # Filter by minimum trip count
    n_before = len(agg)
    agg = agg[agg["trip_count"] >= min_trips].copy()
    print(f"  {n_before:,} hexes -> {len(agg):,} after min_trips filter")

    # Create geometries
    agg["geometry"] = agg["h3_index"].apply(h3_to_polygon)
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs="EPSG:4326")

    # Add hex centroid coordinates
    gdf["hex_lat"] = gdf.geometry.centroid.y
    gdf["hex_lon"] = gdf.geometry.centroid.x

    print(f"  Total trips in filtered hexes: {gdf['trip_count'].sum():,}")
    print(f"  Mean trips/hex: {gdf['trip_count'].mean():.1f}")
    print(f"  Mean speeding %: {gdf['speeding_pct'].mean():.1f}%")

    return gdf


def run_hotspot_analysis(
    gdf: gpd.GeoDataFrame,
    variable: str = "speeding_pct",
    k: int = 8,
    permutations: int = 999,
) -> gpd.GeoDataFrame:
    """Run Getis-Ord Gi* hotspot analysis.

    Args:
        gdf: GeoDataFrame with hex aggregated data.
        variable: Column to analyze for hotspots.
        k: Number of nearest neighbors for spatial weights.
        permutations: Number of permutations for significance testing.

    Returns:
        GeoDataFrame with Gi* z-scores and p-values added.
    """
    print(f"Running Getis-Ord Gi* on '{variable}' (k={k}, perms={permutations})...")

    # Build spatial weights using KNN (better for irregular hex grids)
    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "R"  # row-standardize

    # Run Gi*
    y = gdf[variable].values
    gi = G_Local(y, w, star=True, permutations=permutations, seed=RANDOM_SEED)

    gdf = gdf.copy()
    gdf["gi_zscore"] = gi.Zs
    gdf["gi_pvalue"] = gi.p_sim

    # Classify hotspot/coldspot at different significance levels
    gdf["hotspot_class"] = "Not Significant"
    gdf.loc[
        (gdf["gi_zscore"] > 0) & (gdf["gi_pvalue"] <= 0.01),
        "hotspot_class"
    ] = "Hot Spot (99%)"
    gdf.loc[
        (gdf["gi_zscore"] > 0) & (gdf["gi_pvalue"] > 0.01) & (gdf["gi_pvalue"] <= 0.05),
        "hotspot_class"
    ] = "Hot Spot (95%)"
    gdf.loc[
        (gdf["gi_zscore"] > 0) & (gdf["gi_pvalue"] > 0.05) & (gdf["gi_pvalue"] <= 0.10),
        "hotspot_class"
    ] = "Hot Spot (90%)"
    gdf.loc[
        (gdf["gi_zscore"] < 0) & (gdf["gi_pvalue"] <= 0.01),
        "hotspot_class"
    ] = "Cold Spot (99%)"
    gdf.loc[
        (gdf["gi_zscore"] < 0) & (gdf["gi_pvalue"] > 0.01) & (gdf["gi_pvalue"] <= 0.05),
        "hotspot_class"
    ] = "Cold Spot (95%)"
    gdf.loc[
        (gdf["gi_zscore"] < 0) & (gdf["gi_pvalue"] > 0.05) & (gdf["gi_pvalue"] <= 0.10),
        "hotspot_class"
    ] = "Cold Spot (90%)"

    # Summary
    vc = gdf["hotspot_class"].value_counts()
    print("  Hotspot classification:")
    for cls, cnt in vc.items():
        print(f"    {cls}: {cnt} ({cnt/len(gdf)*100:.1f}%)")

    return gdf


def run_morans_i(
    gdf: gpd.GeoDataFrame,
    variable: str = "speeding_pct",
    k: int = 8,
    permutations: int = 999,
) -> dict[str, Any]:
    """Compute Global and Local Moran's I for spatial autocorrelation.

    Args:
        gdf: GeoDataFrame with hex aggregated data.
        variable: Column to analyze.
        k: Number of nearest neighbors.
        permutations: Number of permutations for inference.

    Returns:
        Dictionary with global and local Moran's I results.
    """
    print(f"Computing Moran's I on '{variable}' (k={k}, perms={permutations})...")

    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "R"

    y = gdf[variable].values

    # Global Moran's I
    mi_global = Moran(y, w, permutations=permutations)
    print(f"  Global Moran's I: {mi_global.I:.4f}")
    print(f"  Expected I: {mi_global.EI:.4f}")
    print(f"  Z-score: {mi_global.z_sim:.4f}")
    print(f"  p-value: {mi_global.p_sim:.6f}")

    # Local Moran's I (LISA)
    mi_local = Moran_Local(y, w, permutations=permutations, seed=RANDOM_SEED)

    gdf = gdf.copy()
    gdf["lisa_I"] = mi_local.Is
    gdf["lisa_pvalue"] = mi_local.p_sim
    gdf["lisa_quadrant"] = mi_local.q

    # Classify LISA clusters
    sig = mi_local.p_sim <= 0.05
    quad_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    gdf["lisa_cluster"] = "Not Significant"
    for q_val, label in quad_labels.items():
        mask = sig & (mi_local.q == q_val)
        gdf.loc[mask, "lisa_cluster"] = label

    vc = gdf["lisa_cluster"].value_counts()
    print("  LISA cluster counts:")
    for cls, cnt in vc.items():
        print(f"    {cls}: {cnt} ({cnt/len(gdf)*100:.1f}%)")

    results = {
        "global_morans_I": float(mi_global.I),
        "expected_I": float(mi_global.EI),
        "z_score": float(mi_global.z_sim),
        "p_value": float(mi_global.p_sim),
        "variance": float(mi_global.VI_sim),
        "n_hexagons": len(gdf),
        "lisa_clusters": vc.to_dict(),
    }

    return results, gdf


def overlay_infrastructure(
    gdf_hex: gpd.GeoDataFrame,
    df_trips: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Overlay hotspot hexagons with road class infrastructure attributes.

    Args:
        gdf_hex: Hex GeoDataFrame with hotspot classifications.
        df_trips: Trip dataframe with H3 indices.

    Returns:
        GeoDataFrame enriched with mean road class fractions per hex.
    """
    print("Overlaying infrastructure attributes...")

    # Load road class data
    con = duckdb.connect()
    road_df = con.execute("""
        SELECT route_id,
               frac_major_road, frac_cycling_infra,
               dominant_road_class, n_road_classes
        FROM read_parquet($path)
    """, {"path": str(DATA_DIR / "trip_road_classes.parquet")}).df()

    # Merge with trips (to get h3 index)
    merged = df_trips[["route_id", "h3_origin"]].merge(road_df, on="route_id", how="inner")

    # Aggregate by hex
    road_agg = merged.groupby("h3_origin").agg(
        frac_major_road_mean=("frac_major_road", "mean"),
        frac_cycling_infra_mean=("frac_cycling_infra", "mean"),
        n_road_classes_mean=("n_road_classes", "mean"),
    ).reset_index()
    road_agg.rename(columns={"h3_origin": "h3_index"}, inplace=True)

    # Also get dominant road class mode per hex
    mode_by_hex = (
        merged.groupby("h3_origin")["dominant_road_class"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown")
        .reset_index()
    )
    mode_by_hex.columns = ["h3_index", "hex_dominant_road_class"]
    road_agg = road_agg.merge(mode_by_hex, on="h3_index", how="left")

    # Merge into hex GeoDataFrame
    gdf_out = gdf_hex.merge(road_agg, on="h3_index", how="left")

    print(f"  Matched {gdf_out['frac_major_road_mean'].notna().sum():,}/{len(gdf_out):,} hexes")
    print(f"  Mean frac_major_road in hotspots: "
          f"{gdf_out.loc[gdf_out['hotspot_class'].str.contains('Hot'), 'frac_major_road_mean'].mean():.3f}")
    print(f"  Mean frac_major_road in coldspots: "
          f"{gdf_out.loc[gdf_out['hotspot_class'].str.contains('Cold'), 'frac_major_road_mean'].mean():.3f}")

    return gdf_out


def run_city_analysis(
    df: pd.DataFrame,
    city: str,
    min_trips: int = 10,
) -> dict[str, Any]:
    """Run full spatial analysis pipeline for a single city.

    Args:
        df: Full trip dataframe with H3 indices.
        city: City name to analyze.
        min_trips: Minimum trips per hex.

    Returns:
        Dictionary with analysis results.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {city}")
    print(f"{'='*60}")

    city_df = df[df["city"] == city].copy()
    print(f"  {len(city_df):,} trips")

    if len(city_df) < 100:
        print(f"  Skipping - too few trips")
        return {}

    # Aggregate origins
    gdf_origin = aggregate_hexagons(city_df, "h3_origin", f"{city}_origin", min_trips)

    if len(gdf_origin) < 20:
        print(f"  Skipping hotspot analysis - too few hexes ({len(gdf_origin)})")
        return {"city": city, "n_trips": len(city_df), "n_hexes": len(gdf_origin)}

    # Hotspot analysis
    gdf_hotspot = run_hotspot_analysis(gdf_origin, "speeding_pct")

    # Moran's I
    moran_results, gdf_lisa = run_morans_i(gdf_hotspot, "speeding_pct")

    # Infrastructure overlay
    gdf_final = overlay_infrastructure(gdf_lisa, city_df)

    # Save city-level GeoDataFrame
    out_path = SPATIAL_DIR / f"hex_{city.lower().replace(' ', '_')}.parquet"
    # Save as parquet (drop geometry, save lat/lon + h3 for reconstruction)
    save_df = gdf_final.drop(columns=["geometry"]).copy()
    save_df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Also save as GeoPackage for GIS inspection
    gpkg_path = SPATIAL_DIR / f"hex_{city.lower().replace(' ', '_')}.gpkg"
    gdf_final.to_file(gpkg_path, driver="GPKG")
    print(f"  Saved: {gpkg_path}")

    results = {
        "city": city,
        "n_trips": len(city_df),
        "n_hexes_total": int(city_df["h3_origin"].nunique()),
        "n_hexes_filtered": len(gdf_origin),
        "global_morans_I": moran_results["global_morans_I"],
        "morans_p_value": moran_results["p_value"],
        "morans_z_score": moran_results["z_score"],
        "lisa_clusters": moran_results["lisa_clusters"],
        "hotspot_counts": gdf_hotspot["hotspot_class"].value_counts().to_dict(),
        "mean_speeding_pct": float(gdf_origin["speeding_pct"].mean()),
        "speeding_pct_std": float(gdf_origin["speeding_pct"].std()),
    }

    return results


def main() -> None:
    """Run full spatial analysis pipeline."""
    t_start = time.time()

    # Load and prepare data
    df = load_trip_data()
    df = compute_h3_indices(df)

    # Save H3 indices for later use
    h3_path = DATA_DIR / "trip_h3_indices.parquet"
    df[["route_id", "h3_origin", "h3_dest"]].to_parquet(h3_path, index=False)
    print(f"Saved H3 indices: {h3_path}")

    # ---- Task 4.1: Full-dataset H3 aggregation ----
    print("\n" + "="*60)
    print("TASK 4.1: Full-dataset H3 aggregation")
    print("="*60)

    gdf_all_origin = aggregate_hexagons(df, "h3_origin", "all_origins", min_trips=10)
    gdf_all_dest = aggregate_hexagons(df, "h3_dest", "all_dests", min_trips=10)

    # Save full-dataset aggregations
    gdf_all_origin.drop(columns=["geometry"]).to_parquet(
        SPATIAL_DIR / "hex_all_origins.parquet", index=False
    )
    gdf_all_dest.drop(columns=["geometry"]).to_parquet(
        SPATIAL_DIR / "hex_all_dests.parquet", index=False
    )

    # ---- Task 4.2-4.3: Hotspot + Moran's I per city ----
    print("\n" + "="*60)
    print("TASKS 4.2-4.3: City-level spatial analysis")
    print("="*60)

    # Primary analysis cities: Seoul (largest) + secondary
    primary_cities = ["Seoul", "Daejeon"]
    # Also run for top 5 cities for comparison
    top_cities = ["Seoul", "Daejeon", "Suwon", "Busan", "Daegu"]

    all_results = {}
    for city in top_cities:
        results = run_city_analysis(df, city, min_trips=10)
        if results:
            all_results[city] = results

    # ---- Task 4.5: Infrastructure overlay summary ----
    print("\n" + "="*60)
    print("TASK 4.5: Infrastructure overlay summary")
    print("="*60)

    # Load Seoul results for detailed summary
    seoul_gdf = gpd.read_file(SPATIAL_DIR / "hex_seoul.gpkg")
    hot = seoul_gdf[seoul_gdf["hotspot_class"].str.contains("Hot")]
    cold = seoul_gdf[seoul_gdf["hotspot_class"].str.contains("Cold")]
    ns = seoul_gdf[seoul_gdf["hotspot_class"] == "Not Significant"]

    infra_summary = {
        "Seoul_hotspot": {
            "n_hexes": len(hot),
            "mean_speeding_pct": float(hot["speeding_pct"].mean()) if len(hot) > 0 else None,
            "mean_frac_major_road": float(hot["frac_major_road_mean"].mean()) if len(hot) > 0 else None,
            "mean_frac_cycling_infra": float(hot["frac_cycling_infra_mean"].mean()) if len(hot) > 0 else None,
            "dominant_road_classes": hot["hex_dominant_road_class"].value_counts().head(5).to_dict() if len(hot) > 0 else {},
        },
        "Seoul_coldspot": {
            "n_hexes": len(cold),
            "mean_speeding_pct": float(cold["speeding_pct"].mean()) if len(cold) > 0 else None,
            "mean_frac_major_road": float(cold["frac_major_road_mean"].mean()) if len(cold) > 0 else None,
            "mean_frac_cycling_infra": float(cold["frac_cycling_infra_mean"].mean()) if len(cold) > 0 else None,
            "dominant_road_classes": cold["hex_dominant_road_class"].value_counts().head(5).to_dict() if len(cold) > 0 else {},
        },
        "Seoul_not_significant": {
            "n_hexes": len(ns),
            "mean_speeding_pct": float(ns["speeding_pct"].mean()) if len(ns) > 0 else None,
            "mean_frac_major_road": float(ns["frac_major_road_mean"].mean()) if len(ns) > 0 else None,
        },
    }
    all_results["infrastructure_overlay"] = infra_summary

    # Print infrastructure summary
    print("\nSeoul Infrastructure Overlay:")
    if len(hot) > 0:
        print(f"  Hot spots ({len(hot)} hexes):")
        print(f"    Mean speeding: {hot['speeding_pct'].mean():.1f}%")
        print(f"    Mean frac_major_road: {hot['frac_major_road_mean'].mean():.3f}")
        print(f"    Mean frac_cycling_infra: {hot['frac_cycling_infra_mean'].mean():.3f}")
        print(f"    Top road classes: {hot['hex_dominant_road_class'].value_counts().head(3).to_dict()}")
    if len(cold) > 0:
        print(f"  Cold spots ({len(cold)} hexes):")
        print(f"    Mean speeding: {cold['speeding_pct'].mean():.1f}%")
        print(f"    Mean frac_major_road: {cold['frac_major_road_mean'].mean():.3f}")
        print(f"    Mean frac_cycling_infra: {cold['frac_cycling_infra_mean'].mean():.3f}")

    # ---- Save comprehensive report ----
    report_path = MODELING_DIR / "spatial_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved report: {report_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
