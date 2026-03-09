"""
Task 3.6-3.7: Compute road-class-level speed indicators.

Since we used nearest-edge matching (not full map-matching with per-point
edge assignment), we compute speed statistics by road class composition.
Also computes the relationship between road class and speeding behavior.

Outputs:
  - data_parquet/segment_indicators.parquet -- road-class-level speed stats
  - data_parquet/modeling/road_class_speed_report.json -- analysis report
"""

import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR,
    MODELING_DIR,
    RANDOM_SEED,
    SPEED_LIMIT_KR,
)

SEGMENT_INDICATORS_PATH = DATA_DIR / "segment_indicators.parquet"
REPORT_PATH = MODELING_DIR / "road_class_speed_report.json"
MODELING_DIR.mkdir(parents=True, exist_ok=True)

ROAD_CATEGORIES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "unclassified", "service", "cycleway", "footway", "other",
]


def compute_road_class_speed_stats() -> pd.DataFrame:
    """Compute speed statistics by dominant road class.

    Joins trip indicators with road class data and computes per-road-class
    speed distributions.

    Returns:
        DataFrame with road class speed statistics.
    """
    con = duckdb.connect()

    # Join trip indicators with road class data
    df = con.execute("""
        SELECT
            r.dominant_road_class,
            r.frac_major_road,
            r.n_road_classes,
            t.mean_speed,
            t.max_speed_from_profile as max_speed,
            t.p85_speed,
            t.speeding_rate_25,
            t.speed_cv,
            t.mean_abs_accel_ms2,
            t.cruise_fraction,
            CASE WHEN t.speeding_rate_25 > 0 THEN true ELSE false END as has_speeding,
            c.mode,
            c.city,
            c.province
        FROM read_parquet('data_parquet/trip_indicators.parquet') t
        JOIN read_parquet('data_parquet/trip_road_classes.parquet') r
            ON t.route_id = r.route_id
        JOIN read_parquet('data_parquet/cleaned/trips_cleaned.parquet') c
            ON t.route_id = c.route_id
        WHERE c.is_valid = true
    """).fetchdf()
    con.close()

    print(f"Joined data: {len(df):,} trips")

    # Compute stats by road class
    results = []
    for road_class in ROAD_CATEGORIES:
        mask = df["dominant_road_class"] == road_class
        subset = df[mask]
        if len(subset) < 100:
            continue

        stats = {
            "road_class": road_class,
            "n_trips": len(subset),
            "pct_of_total": len(subset) / len(df) * 100,
            # Speed statistics
            "mean_speed_mean": float(subset["mean_speed"].mean()),
            "mean_speed_median": float(subset["mean_speed"].median()),
            "mean_speed_std": float(subset["mean_speed"].std()),
            "max_speed_mean": float(subset["max_speed"].mean()),
            "max_speed_p85": float(subset["max_speed"].quantile(0.85)),
            "p85_speed_mean": float(subset["p85_speed"].mean()),
            # Speeding
            "speeding_rate": float(subset["has_speeding"].mean()),
            "mean_speeding_fraction": float(subset["speeding_rate_25"].mean()),
            # Behavior
            "speed_cv_mean": float(subset["speed_cv"].mean()),
            "mean_accel": float(subset["mean_abs_accel_ms2"].mean()),
            "cruise_fraction_mean": float(subset["cruise_fraction"].mean()),
        }
        results.append(stats)

    stats_df = pd.DataFrame(results)
    return stats_df, df


def compute_road_class_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Compute road class speed stats by city.

    Args:
        df: Joined trip + road class data.

    Returns:
        DataFrame with city x road class speed statistics.
    """
    city_road = (
        df.groupby(["city", "dominant_road_class"])
        .agg(
            n_trips=("mean_speed", "count"),
            mean_speed=("mean_speed", "mean"),
            speeding_rate=("has_speeding", "mean"),
            max_speed_mean=("max_speed", "mean"),
        )
        .reset_index()
    )
    # Filter to cells with enough data
    city_road = city_road[city_road["n_trips"] >= 100]
    return city_road


def compute_road_class_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Compute road class speed stats by mode.

    Args:
        df: Joined trip + road class data.

    Returns:
        DataFrame with mode x road class speed statistics.
    """
    mode_road = (
        df.groupby(["mode", "dominant_road_class"])
        .agg(
            n_trips=("mean_speed", "count"),
            mean_speed=("mean_speed", "mean"),
            speeding_rate=("has_speeding", "mean"),
        )
        .reset_index()
    )
    mode_road = mode_road[mode_road["n_trips"] >= 100]
    return mode_road


def main() -> None:
    """Compute and save road-class-level speed indicators."""
    print("=" * 60)
    print("Task 3.6-3.7: Road Class Speed Indicators")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    t_start = time.time()

    # Overall stats by road class
    print("\nComputing road-class-level speed statistics...")
    stats_df, joined_df = compute_road_class_speed_stats()

    print(f"\n{'Road Class':15s} {'N':>10s} {'Mean Spd':>10s} {'Max Spd':>10s} "
          f"{'Speeding%':>10s} {'SpeedCV':>8s}")
    print("-" * 65)
    for _, row in stats_df.iterrows():
        print(f"{row['road_class']:15s} {row['n_trips']:>10,} "
              f"{row['mean_speed_mean']:>10.1f} {row['max_speed_mean']:>10.1f} "
              f"{row['speeding_rate']*100:>9.1f}% {row['speed_cv_mean']:>8.3f}")

    # By city
    print("\nComputing city x road class statistics...")
    city_road_df = compute_road_class_by_city(joined_df)
    print(f"  {len(city_road_df)} city x road_class cells")

    # By mode
    print("\nComputing mode x road class statistics...")
    mode_road_df = compute_road_class_by_mode(joined_df)
    print(f"  {len(mode_road_df)} mode x road_class cells")

    # Save combined segment indicators
    segment_df = stats_df.copy()
    segment_df.to_parquet(SEGMENT_INDICATORS_PATH, index=False)
    print(f"\nSaved segment indicators to: {SEGMENT_INDICATORS_PATH}")

    # Report
    total_time = time.time() - t_start

    report = {
        "task": "3.6-3.7",
        "description": "Road class speed indicators (based on nearest-edge matching)",
        "total_trips_analyzed": len(joined_df),
        "total_time_s": round(total_time, 1),
        "road_class_stats": stats_df.to_dict(orient="records"),
        "key_findings": {
            "highest_mean_speed": stats_df.loc[stats_df["mean_speed_mean"].idxmax()].to_dict(),
            "highest_speeding_rate": stats_df.loc[stats_df["speeding_rate"].idxmax()].to_dict(),
            "lowest_speeding_rate": stats_df.loc[stats_df["speeding_rate"].idxmin()].to_dict(),
        },
        "mode_road_class_stats": mode_road_df.to_dict(orient="records"),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"Report saved to: {REPORT_PATH}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    highest_speed = stats_df.loc[stats_df["mean_speed_mean"].idxmax()]
    highest_speeding = stats_df.loc[stats_df["speeding_rate"].idxmax()]
    lowest_speeding = stats_df.loc[stats_df["speeding_rate"].idxmin()]

    print(f"\nHighest mean speed: {highest_speed['road_class']} "
          f"({highest_speed['mean_speed_mean']:.1f} km/h)")
    print(f"Highest speeding rate: {highest_speeding['road_class']} "
          f"({highest_speeding['speeding_rate']*100:.1f}%)")
    print(f"Lowest speeding rate: {lowest_speeding['road_class']} "
          f"({lowest_speeding['speeding_rate']*100:.1f}%)")

    # Mode x road class interaction
    print("\nMode x Road Class (speeding rate):")
    scooter_modes = ["SCOOTER_ECO", "SCOOTER_STD", "SCOOTER_TUB"]
    for mode in scooter_modes:
        mode_data = mode_road_df[mode_road_df["mode"] == mode]
        if not mode_data.empty:
            top = mode_data.nlargest(3, "speeding_rate")
            print(f"  {mode}:")
            for _, r in top.iterrows():
                print(f"    {r['dominant_road_class']:15s} "
                      f"speeding={r['speeding_rate']*100:.1f}% (n={r['n_trips']:,})")


if __name__ == "__main__":
    main()
