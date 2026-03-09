"""
Task 1.5: Validate speeds column against displacement-derived speeds from GPS.

Approach:
  1. Sample N trips from the valid dataset
  2. Parse routes column -> extract GPS coordinates and timestamps
  3. Compute displacement between consecutive points using Haversine formula
  4. Derive speed from displacement / time_interval
  5. Compare GPS-derived speeds with reported speeds column
  6. Report correlation, bias, and error statistics

Outputs:
  - data_parquet/speed_validation.parquet — sample trip speed comparisons
  - Console report of validation statistics
"""

import ast
import duckdb
import json
import math
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, RANDOM_SEED

# Number of trips to sample for validation
VALIDATION_SAMPLE_SIZE = 10_000


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in meters between two GPS points.

    Args:
        lat1, lon1: First point coordinates (degrees).
        lat2, lon2: Second point coordinates (degrees).

    Returns:
        Distance in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def parse_routes(routes_str: str) -> list[tuple[datetime, float, float]]:
    """Parse routes string into list of (datetime, lat, lon) tuples.

    Args:
        routes_str: Stringified Python list from CSV.

    Returns:
        List of (datetime, lat, lon) tuples.
    """
    try:
        raw = ast.literal_eval(routes_str)
    except (ValueError, SyntaxError):
        return []

    points = []
    for entry in raw:
        if len(entry) < 4:
            continue
        date_str, time_str, lat, lon = entry[0], entry[1], float(entry[2]), float(entry[3])
        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S.%f")
        except ValueError:
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S")
            except ValueError:
                continue
        points.append((dt, lat, lon))
    return points


def parse_speeds(speeds_str: str) -> list[int]:
    """Parse speeds string into list of integer speed values.

    Args:
        speeds_str: Stringified nested list from CSV, e.g. '[[18, 16, 21]]'

    Returns:
        Flat list of speed values (km/h).
    """
    try:
        raw = ast.literal_eval(speeds_str)
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            return raw[0]  # Unwrap nested list
        elif isinstance(raw, list):
            return raw
    except (ValueError, SyntaxError):
        pass
    return []


def compute_gps_speeds(points: list[tuple[datetime, float, float]]) -> list[float]:
    """Compute GPS displacement-based speeds between consecutive points.

    Args:
        points: List of (datetime, lat, lon) tuples, sorted by time.

    Returns:
        List of speeds in km/h (length = len(points) - 1).
    """
    speeds = []
    for i in range(1, len(points)):
        dt1, lat1, lon1 = points[i - 1]
        dt2, lat2, lon2 = points[i]
        dt_seconds = (dt2 - dt1).total_seconds()
        if dt_seconds <= 0:
            speeds.append(0.0)
            continue
        dist_m = haversine_distance(lat1, lon1, lat2, lon2)
        speed_kmh = (dist_m / dt_seconds) * 3.6  # m/s -> km/h
        speeds.append(speed_kmh)
    return speeds


def validate_speeds() -> dict:
    """Run speed validation on a sample of trips.

    Returns:
        Dictionary with validation statistics.
    """
    con = duckdb.connect()
    parquet_path = str(DATA_DIR / "routes_with_cities.parquet")

    print("=" * 70)
    print("  TASK 1.5: SPEED COLUMN VALIDATION")
    print("=" * 70)

    # Sample trips with sufficient GPS points and valid speeds
    print(f"\nSampling {VALIDATION_SAMPLE_SIZE:,} trips for validation...")
    sample_df = con.execute(f"""
        SELECT route_id, routes, speeds, avg_speed, max_speed, points, distance, travel_time
        FROM read_parquet('{parquet_path}')
        WHERE points >= 5
            AND avg_speed > 0
            AND LENGTH(speeds) > 10
        USING SAMPLE {VALIDATION_SAMPLE_SIZE}
    """).fetchdf()

    print(f"Sampled {len(sample_df):,} trips")

    # Process each trip
    results = []
    parse_errors = 0
    mismatch_count = 0

    for idx, row in sample_df.iterrows():
        route_id = row["route_id"]
        reported_avg = row["avg_speed"]
        reported_max = row["max_speed"]
        num_points = row["points"]

        # Parse routes and speeds
        gps_points = parse_routes(row["routes"])
        reported_speeds = parse_speeds(row["speeds"])

        if len(gps_points) < 2:
            parse_errors += 1
            continue

        # Compute GPS-derived speeds
        gps_speeds = compute_gps_speeds(gps_points)

        # Compute stats
        gps_avg = np.mean(gps_speeds) if gps_speeds else 0
        gps_max = np.max(gps_speeds) if gps_speeds else 0
        gps_median = np.median(gps_speeds) if gps_speeds else 0

        reported_speed_avg = np.mean(reported_speeds) if reported_speeds else 0
        reported_speed_max = max(reported_speeds) if reported_speeds else 0

        # Compute time intervals
        intervals = [(gps_points[i][0] - gps_points[i - 1][0]).total_seconds()
                      for i in range(1, len(gps_points))]
        mean_interval = np.mean(intervals) if intervals else 0
        max_interval = np.max(intervals) if intervals else 0

        results.append({
            "route_id": route_id,
            "num_gps_points": len(gps_points),
            "num_reported_points": num_points,
            "num_speeds_reported": len(reported_speeds),
            "num_gps_speeds": len(gps_speeds),
            "gps_avg_speed": gps_avg,
            "gps_max_speed": gps_max,
            "gps_median_speed": gps_median,
            "reported_avg_speed": reported_avg,
            "reported_max_speed": reported_max,
            "speeds_col_avg": reported_speed_avg,
            "speeds_col_max": reported_speed_max,
            "mean_gps_interval_s": mean_interval,
            "max_gps_interval_s": max_interval,
            "reported_distance": row["distance"],
            "gps_total_distance": sum(
                haversine_distance(gps_points[i - 1][1], gps_points[i - 1][2],
                                   gps_points[i][1], gps_points[i][2])
                for i in range(1, len(gps_points))
            ),
        })

        # Check if speeds count matches
        if len(reported_speeds) != num_points:
            mismatch_count += 1

    results_df = pd.DataFrame(results)
    print(f"\nProcessed {len(results_df):,} trips (parse errors: {parse_errors})")
    print(f"Speed count mismatch (speeds col length != points): {mismatch_count}")

    # Validation Statistics
    print("\n--- Speed Validation Statistics ---")

    # 1. Reported avg_speed vs speeds column average
    corr_avg = results_df["reported_avg_speed"].corr(results_df["speeds_col_avg"])
    bias_avg = (results_df["reported_avg_speed"] - results_df["speeds_col_avg"]).mean()
    mae_avg = (results_df["reported_avg_speed"] - results_df["speeds_col_avg"]).abs().mean()
    print(f"\n  avg_speed vs speeds_col mean:")
    print(f"    Correlation:  {corr_avg:.4f}")
    print(f"    Mean bias:    {bias_avg:.2f} km/h")
    print(f"    MAE:          {mae_avg:.2f} km/h")

    # 2. Reported max_speed vs speeds column max
    corr_max = results_df["reported_max_speed"].corr(results_df["speeds_col_max"])
    bias_max = (results_df["reported_max_speed"] - results_df["speeds_col_max"]).mean()
    mae_max = (results_df["reported_max_speed"] - results_df["speeds_col_max"]).abs().mean()
    print(f"\n  max_speed vs speeds_col max:")
    print(f"    Correlation:  {corr_max:.4f}")
    print(f"    Mean bias:    {bias_max:.2f} km/h")
    print(f"    MAE:          {mae_max:.2f} km/h")

    # 3. GPS-derived avg speed vs reported avg_speed
    corr_gps_avg = results_df["gps_avg_speed"].corr(results_df["reported_avg_speed"])
    bias_gps_avg = (results_df["gps_avg_speed"] - results_df["reported_avg_speed"]).mean()
    mae_gps_avg = (results_df["gps_avg_speed"] - results_df["reported_avg_speed"]).abs().mean()
    print(f"\n  GPS-derived avg speed vs reported avg_speed:")
    print(f"    Correlation:  {corr_gps_avg:.4f}")
    print(f"    Mean bias:    {bias_gps_avg:.2f} km/h (positive = GPS higher)")
    print(f"    MAE:          {mae_gps_avg:.2f} km/h")

    # 4. GPS-derived max speed vs reported max_speed
    corr_gps_max = results_df["gps_max_speed"].corr(results_df["reported_max_speed"])
    bias_gps_max = (results_df["gps_max_speed"] - results_df["reported_max_speed"]).mean()
    mae_gps_max = (results_df["gps_max_speed"] - results_df["reported_max_speed"]).abs().mean()
    print(f"\n  GPS-derived max speed vs reported max_speed:")
    print(f"    Correlation:  {corr_gps_max:.4f}")
    print(f"    Mean bias:    {bias_gps_max:.2f} km/h (positive = GPS higher)")
    print(f"    MAE:          {mae_gps_max:.2f} km/h")

    # 5. GPS distance vs reported distance
    corr_dist = results_df["gps_total_distance"].corr(results_df["reported_distance"])
    bias_dist = (results_df["gps_total_distance"] - results_df["reported_distance"]).mean()
    print(f"\n  GPS distance vs reported distance:")
    print(f"    Correlation:  {corr_dist:.4f}")
    print(f"    Mean bias:    {bias_dist:.1f} m (positive = GPS higher)")

    # 6. GPS interval statistics
    print(f"\n  GPS time interval statistics (seconds):")
    print(f"    Mean interval:  {results_df['mean_gps_interval_s'].mean():.1f}s")
    print(f"    Median interval: {results_df['mean_gps_interval_s'].median():.1f}s")
    print(f"    Mean max interval: {results_df['max_gps_interval_s'].mean():.1f}s")

    # 7. Speed count relationship
    print(f"\n  Speed point counts:")
    print(f"    Mean GPS points per trip:      {results_df['num_gps_points'].mean():.1f}")
    print(f"    Mean reported points per trip:  {results_df['num_reported_points'].mean():.1f}")
    print(f"    Mean speeds col entries:        {results_df['num_speeds_reported'].mean():.1f}")
    count_match = (results_df["num_speeds_reported"] == results_df["num_reported_points"]).sum()
    print(f"    Speeds count == points count:   {count_match}/{len(results_df)} "
          f"({count_match/len(results_df):.1%})")

    # Save results
    output_path = DATA_DIR / "speed_validation.parquet"
    results_df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    print(f"\nResults saved to {output_path}")

    # Save summary stats
    stats = {
        "sample_size": len(results_df),
        "parse_errors": parse_errors,
        "avg_speed_vs_speeds_col": {
            "correlation": round(corr_avg, 4),
            "bias": round(bias_avg, 2),
            "mae": round(mae_avg, 2),
        },
        "max_speed_vs_speeds_col": {
            "correlation": round(corr_max, 4),
            "bias": round(bias_max, 2),
            "mae": round(mae_max, 2),
        },
        "gps_avg_vs_reported_avg": {
            "correlation": round(corr_gps_avg, 4),
            "bias": round(bias_gps_avg, 2),
            "mae": round(mae_gps_avg, 2),
        },
        "gps_max_vs_reported_max": {
            "correlation": round(corr_gps_max, 4),
            "bias": round(bias_gps_max, 2),
            "mae": round(mae_gps_max, 2),
        },
        "gps_distance_vs_reported": {
            "correlation": round(corr_dist, 4),
            "bias": round(bias_dist, 1),
        },
        "mean_gps_interval_s": round(results_df["mean_gps_interval_s"].mean(), 1),
        "median_gps_interval_s": round(results_df["mean_gps_interval_s"].median(), 1),
    }

    report_path = DATA_DIR / "speed_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Report saved to {report_path}")

    con.close()

    print("\n" + "=" * 70)
    print("  SPEED VALIDATION COMPLETE")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    validate_speeds()
