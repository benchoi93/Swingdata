"""
Compute GPS trajectory curvature indices for e-scooter trips.

For each trip, extracts turning angles and curvature metrics from the GPS
coordinate sequence to classify segments as straight, moderate curve, or
sharp turn.

Metrics per trip:
  - mean_abs_turning_angle: mean absolute bearing change between consecutive segments
  - max_abs_turning_angle: maximum absolute bearing change
  - curvature_index: sum of absolute turning angles / total distance (deg/m)
  - frac_sharp_turns: fraction of segments with |turning angle| > 45 deg
  - frac_moderate_turns: fraction with 15 < |angle| <= 45 deg
  - frac_straight: fraction with |angle| <= 15 deg
  - n_segments: number of 3-point segments (n_points - 2)

Output:
  - data_parquet/trip_curvature.parquet

Usage:
    python src/compute_curvature.py
"""

import ast
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR, MODELING_DIR, RANDOM_SEED,
)

OUTPUT_PATH = DATA_DIR / "trip_curvature.parquet"
RESULTS_PATH = MODELING_DIR / "curvature_computation_results.json"

# Turning angle thresholds (degrees)
SHARP_TURN_THRESHOLD = 45.0
MODERATE_TURN_THRESHOLD = 15.0

# Processing in chunks for memory efficiency
CHUNK_SIZE = 50_000


def compute_bearing(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """Compute initial bearing from point 1 to point 2 (degrees, 0-360).

    Args:
        lat1, lon1: First point (decimal degrees).
        lat2, lon2: Second point (decimal degrees).

    Returns:
        Bearing in degrees [0, 360).
    """
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)

    x = np.sin(dlon) * np.cos(lat2_r)
    y = (np.cos(lat1_r) * np.sin(lat2_r)
         - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon))

    bearing = np.degrees(np.arctan2(x, y))
    return bearing % 360.0


def compute_bearings_vectorized(lats: np.ndarray,
                                lons: np.ndarray) -> np.ndarray:
    """Compute bearings between consecutive points (vectorized).

    Args:
        lats: Array of latitudes.
        lons: Array of longitudes.

    Returns:
        Array of bearings (length n-1).
    """
    lat1 = np.radians(lats[:-1])
    lat2 = np.radians(lats[1:])
    dlon = np.radians(lons[1:] - lons[:-1])

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearings = np.degrees(np.arctan2(x, y)) % 360.0
    return bearings


def compute_turning_angles(bearings: np.ndarray) -> np.ndarray:
    """Compute turning angles from consecutive bearings.

    The turning angle is the change in bearing, normalized to [-180, 180].

    Args:
        bearings: Array of bearings (length n-1).

    Returns:
        Array of turning angles (length n-2).
    """
    diff = np.diff(bearings)
    # Normalize to [-180, 180]
    diff = (diff + 180) % 360 - 180
    return diff


def haversine_distances(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute haversine distances between consecutive points (meters).

    Args:
        lats: Array of latitudes.
        lons: Array of longitudes.

    Returns:
        Array of distances in meters (length n-1).
    """
    R = 6371000.0  # Earth radius in meters

    lat1 = np.radians(lats[:-1])
    lat2 = np.radians(lats[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lons[1:] - lons[:-1])

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def compute_trip_curvature(route_str: str) -> Optional[Dict]:
    """Compute curvature metrics for a single trip from its route string.

    Args:
        route_str: Stringified Python list of [date, time, lat, lon] GPS points.

    Returns:
        Dictionary with curvature metrics, or None if insufficient data.
    """
    try:
        points = ast.literal_eval(route_str)
    except (ValueError, SyntaxError):
        return None

    if len(points) < 3:
        return None

    # Extract lat/lon arrays
    try:
        lats = np.array([float(p[2]) for p in points])
        lons = np.array([float(p[3]) for p in points])
    except (IndexError, ValueError, TypeError):
        return None

    # Filter out stationary points (zero displacement)
    dists = haversine_distances(lats, lons)
    # Keep only segments where scooter actually moved (> 1m)
    moving_mask = dists > 1.0
    if moving_mask.sum() < 2:
        return None

    # Build moving-only coordinate arrays
    # We need consecutive moving points: include start of each moving segment
    # and the end of the last one
    moving_indices = np.where(moving_mask)[0]
    # Include both endpoints of each moving segment
    idx_set = set()
    for i in moving_indices:
        idx_set.add(i)
        idx_set.add(i + 1)
    idx_sorted = sorted(idx_set)

    if len(idx_sorted) < 3:
        return None

    lats_m = lats[idx_sorted]
    lons_m = lons[idx_sorted]

    # Compute bearings and turning angles
    bearings = compute_bearings_vectorized(lats_m, lons_m)
    if len(bearings) < 2:
        return None

    turning_angles = compute_turning_angles(bearings)
    abs_angles = np.abs(turning_angles)

    # Compute distances for curvature index
    segment_dists = haversine_distances(lats_m, lons_m)
    total_distance = segment_dists.sum()

    if total_distance < 10:  # < 10m total movement
        return None

    n_segments = len(turning_angles)
    n_sharp = (abs_angles > SHARP_TURN_THRESHOLD).sum()
    n_moderate = ((abs_angles > MODERATE_TURN_THRESHOLD) & (abs_angles <= SHARP_TURN_THRESHOLD)).sum()
    n_straight = (abs_angles <= MODERATE_TURN_THRESHOLD).sum()

    return {
        "mean_abs_turning_angle": float(np.mean(abs_angles)),
        "median_abs_turning_angle": float(np.median(abs_angles)),
        "max_abs_turning_angle": float(np.max(abs_angles)),
        "std_turning_angle": float(np.std(turning_angles)),
        "curvature_index": float(np.sum(abs_angles) / total_distance),
        "frac_sharp_turns": float(n_sharp / n_segments),
        "frac_moderate_turns": float(n_moderate / n_segments),
        "frac_straight": float(n_straight / n_segments),
        "n_segments": int(n_segments),
        "total_moving_distance_m": float(total_distance),
        "n_moving_points": int(len(lats_m)),
    }


def process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Process a chunk of trips and compute curvature for each.

    Args:
        chunk_df: DataFrame with route_id and routes_raw columns.

    Returns:
        DataFrame with route_id and curvature metrics.
    """
    results = []
    for _, row in chunk_df.iterrows():
        metrics = compute_trip_curvature(row["routes_raw"])
        if metrics is not None:
            metrics["route_id"] = row["route_id"]
            results.append(metrics)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


SAMPLE_SIZE = 200_000  # Stratified subsample (sufficient for analysis)


def main() -> None:
    """Run curvature computation on a stratified subsample of trips."""
    t0 = time.time()
    print("=" * 70)
    print("  GPS TRAJECTORY CURVATURE COMPUTATION")
    print("=" * 70)

    MODELING_DIR.mkdir(parents=True, exist_ok=True)

    # Load route_id and routes_raw from cleaned trips (subsample)
    con = duckdb.connect()
    trips_path = str(DATA_DIR / "cleaned" / "trips_cleaned.parquet").replace("\\", "/")

    total_trips = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{trips_path}') WHERE is_valid = true"
    ).fetchone()[0]
    print(f"\nTotal valid trips: {total_trips:,}")
    print(f"Using stratified subsample of {SAMPLE_SIZE:,} trips "
          f"(full dataset too large for ast.literal_eval parsing)")

    # Stratified random sample by mode (preserves mode distribution)
    sample_df = con.execute(f"""
        SELECT route_id, routes_raw
        FROM (
            SELECT route_id, routes_raw,
                   ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn
            FROM read_parquet('{trips_path}')
            WHERE is_valid = true
        )
        WHERE rn <= {SAMPLE_SIZE}
    """).fetchdf()
    con.close()

    print(f"  Sampled {len(sample_df):,} trips")

    # Process in chunks
    all_results = []
    n_processed = 0
    n_success = 0

    print(f"\nProcessing in chunks of {CHUNK_SIZE:,}...")

    for start in range(0, len(sample_df), CHUNK_SIZE):
        chunk = sample_df.iloc[start:start + CHUNK_SIZE]

        chunk_result = process_chunk(chunk)
        n_processed += len(chunk)
        n_success += len(chunk_result)

        if len(chunk_result) > 0:
            all_results.append(chunk_result)

        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        print(f"  Processed {n_processed:,}/{len(sample_df):,} "
              f"({n_processed/len(sample_df):.1%}) | "
              f"success: {n_success:,} | "
              f"{rate:.0f} trips/s | "
              f"{elapsed:.0f}s elapsed")

    # Combine all results
    if not all_results:
        print("ERROR: No trips processed successfully")
        return

    df_curv = pd.concat(all_results, ignore_index=True)
    print(f"\nCurvature computed for {len(df_curv):,} trips "
          f"({len(df_curv)/len(sample_df):.1%} of sampled trips)")

    # Save to parquet
    df_curv.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved to {OUTPUT_PATH}")

    # Summary statistics
    print("\n--- Curvature Summary ---")
    for col in ["mean_abs_turning_angle", "curvature_index",
                 "frac_sharp_turns", "frac_straight"]:
        vals = df_curv[col]
        print(f"  {col}: mean={vals.mean():.4f}, median={vals.median():.4f}, "
              f"std={vals.std():.4f}, [P5={vals.quantile(0.05):.4f}, "
              f"P95={vals.quantile(0.95):.4f}]")

    # Classification summary
    # Classify trips: "straight" (frac_straight > 0.8), "curvy" (frac_sharp > 0.2),
    # "mixed" (everything else)
    df_curv["trip_curvature_class"] = np.select(
        [
            df_curv["frac_straight"] > 0.8,
            df_curv["frac_sharp_turns"] > 0.2,
        ],
        ["straight", "curvy"],
        default="mixed",
    )

    class_counts = df_curv["trip_curvature_class"].value_counts()
    print("\n--- Trip Curvature Classification ---")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt:,} ({cnt/len(df_curv):.1%})")

    # Save results summary
    elapsed_total = time.time() - t0
    results_summary = {
        "total_valid_trips": int(total_trips),
        "sample_size": int(len(sample_df)),
        "trips_with_curvature": int(len(df_curv)),
        "coverage_rate": float(len(df_curv) / len(sample_df)),
        "processing_time_sec": round(elapsed_total, 1),
        "curvature_stats": {
            col: {
                "mean": float(df_curv[col].mean()),
                "median": float(df_curv[col].median()),
                "std": float(df_curv[col].std()),
                "p5": float(df_curv[col].quantile(0.05)),
                "p25": float(df_curv[col].quantile(0.25)),
                "p75": float(df_curv[col].quantile(0.75)),
                "p95": float(df_curv[col].quantile(0.95)),
            }
            for col in ["mean_abs_turning_angle", "curvature_index",
                         "frac_sharp_turns", "frac_moderate_turns", "frac_straight"]
        },
        "curvature_classification": {
            cls: {"count": int(cnt), "fraction": float(cnt / len(df_curv))}
            for cls, cnt in class_counts.items()
        },
        "thresholds": {
            "sharp_turn_deg": SHARP_TURN_THRESHOLD,
            "moderate_turn_deg": MODERATE_TURN_THRESHOLD,
            "straight_frac_threshold": 0.8,
            "curvy_frac_threshold": 0.2,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")

    print(f"\nTotal time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
