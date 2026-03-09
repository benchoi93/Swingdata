"""
Task 5.1: Prepare modeling dataset.

Merges trip-level indicators with user demographics (age from Scooter CSV),
creates categorical variables, and handles missing data for statistical modeling.

Strategy:
  - Extract user demographics (age) from Scooter CSV for May 2023
  - For users with multiple rides, take mode/median of age (should be constant per user)
  - Merge with trip_indicators.parquet and user_indicators.parquet
  - Create age groups, time-of-day categories, distance bins
  - Handle missing data (drop or impute as appropriate)

Outputs:
  - data_parquet/modeling/trip_modeling.parquet — trip-level modeling dataset
  - data_parquet/modeling/user_modeling.parquet — user-level modeling dataset
  - data_parquet/modeling/modeling_summary.json — dataset summary stats
"""

import json
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR,
    CLEANED_PARQUET,
    SCOOTER_CSV,
    TRIP_INDICATORS_PARQUET,
    USER_INDICATORS_PARQUET,
    RANDOM_SEED,
)

# Output directory
MODELING_DIR = DATA_DIR / "modeling"
MODELING_DIR.mkdir(parents=True, exist_ok=True)


def extract_user_demographics() -> pd.DataFrame:
    """Extract user demographics (age) from Scooter CSV for May 2023.

    Returns:
        DataFrame with user_id and age columns.
    """
    print("Extracting user demographics from Scooter CSV (May 2023)...")
    con = duckdb.connect()

    # Get age per user from May 2023 data
    # Age should be constant per user, but take median to be safe
    user_demo = con.execute(f"""
        SELECT
            user_id,
            MEDIAN(age) as age,
            COUNT(*) as scooter_trip_count,
            SUM(bill_amt) as total_bill,
            AVG(bill_amt) as avg_bill,
            AVG(seconds) as avg_duration_s,
            COUNT(DISTINCT smodel) as n_models_used,
            MODE(smodel) as primary_model_scooter,
            MODE(mode) as primary_mode_scooter
        FROM read_csv_auto('{SCOOTER_CSV}')
        WHERE month = 5
        AND age IS NOT NULL
        AND age >= 10 AND age <= 90
        GROUP BY user_id
    """).fetchdf()
    con.close()

    print(f"  Users with valid demographics: {len(user_demo):,}")
    print(f"  Age range: {user_demo['age'].min():.0f}-{user_demo['age'].max():.0f}")
    print(f"  Mean age: {user_demo['age'].mean():.1f}")
    print(f"  Median age: {user_demo['age'].median():.0f}")

    return user_demo


def create_age_groups(age: pd.Series) -> pd.Series:
    """Create age group categories.

    Args:
        age: Series of ages.

    Returns:
        Categorical series of age groups.
    """
    bins = [0, 20, 25, 30, 35, 40, 50, 60, 100]
    labels = ["<20", "20-24", "25-29", "30-34", "35-39", "40-49", "50-59", "60+"]
    return pd.cut(age, bins=bins, labels=labels, right=False)


def create_time_categories(hour: pd.Series) -> pd.Series:
    """Create time-of-day categories.

    Args:
        hour: Series of hour values (0-23).

    Returns:
        Categorical series of time periods.
    """
    conditions = [
        (hour >= 6) & (hour < 10),
        (hour >= 10) & (hour < 14),
        (hour >= 14) & (hour < 18),
        (hour >= 18) & (hour < 22),
        (hour >= 22) | (hour < 6),
    ]
    choices = ["morning_rush", "midday", "afternoon_rush", "evening", "night"]
    return pd.Series(
        np.select(conditions, choices, default="night"),
        index=hour.index,
        dtype="category",
    )


def create_distance_bins(distance: pd.Series) -> pd.Series:
    """Create distance category bins.

    Args:
        distance: Series of trip distances in meters.

    Returns:
        Categorical series of distance bins.
    """
    bins = [0, 500, 1000, 2000, 3000, 5000, float("inf")]
    labels = ["<500m", "500m-1km", "1-2km", "2-3km", "3-5km", ">5km"]
    return pd.cut(distance, bins=bins, labels=labels, right=False)


def build_trip_modeling_dataset(user_demo: pd.DataFrame) -> pd.DataFrame:
    """Build trip-level modeling dataset by merging indicators with demographics.

    Args:
        user_demo: User demographics DataFrame.

    Returns:
        Trip-level modeling DataFrame.
    """
    print("\nBuilding trip-level modeling dataset...")
    con = duckdb.connect()

    # Load trip indicators
    print("  Loading trip indicators...")
    trip_ind = con.execute(f"""
        SELECT * FROM read_parquet('{TRIP_INDICATORS_PARQUET}')
    """).fetchdf()
    print(f"  Trip indicators: {len(trip_ind):,} trips x {len(trip_ind.columns)} columns")

    # Load cleaned trips for metadata columns
    print("  Loading trip metadata from cleaned dataset...")
    trip_meta = con.execute(f"""
        SELECT
            route_id, user_id, model, mode, province, city,
            start_hour, day_of_week, is_weekend, day_of_month,
            distance, travel_time, gps_points
        FROM read_parquet('{CLEANED_PARQUET}/trips_cleaned.parquet')
    """).fetchdf()
    print(f"  Trip metadata: {len(trip_meta):,} trips")
    con.close()

    # Merge trip indicators with metadata
    print("  Merging trip indicators with metadata...")
    trip_df = trip_meta.merge(trip_ind, on="route_id", how="inner")
    print(f"  After merge: {len(trip_df):,} trips")

    # Merge with user demographics
    print("  Merging with user demographics...")
    trip_df = trip_df.merge(
        user_demo[["user_id", "age"]],
        on="user_id",
        how="left",
    )
    age_coverage = trip_df["age"].notna().mean() * 100
    print(f"  Age coverage: {age_coverage:.1f}% of trips have age data")

    # Create categorical variables
    print("  Creating categorical variables...")

    # Age groups
    trip_df["age_group"] = create_age_groups(trip_df["age"])

    # Time of day
    trip_df["time_of_day"] = create_time_categories(trip_df["start_hour"])

    # Distance bins
    trip_df["distance_bin"] = create_distance_bins(trip_df["distance"])

    # Day type (more granular than weekend)
    trip_df["day_type"] = trip_df["day_of_week"].map({
        0: "weekday", 1: "weekday", 2: "weekday",
        3: "weekday", 4: "weekday", 5: "saturday", 6: "sunday",
    }).astype("category")

    # Binary speeding indicator (any point > 25 km/h)
    trip_df["is_speeding"] = (trip_df["speeding_rate_25"] > 0).astype(int)

    # Speeding severity categories
    conditions = [
        trip_df["speeding_rate_25"] == 0,
        trip_df["speeding_rate_25"] <= 0.1,
        trip_df["speeding_rate_25"] <= 0.25,
        trip_df["speeding_rate_25"] <= 0.5,
        trip_df["speeding_rate_25"] > 0.5,
    ]
    choices = ["none", "minor", "moderate", "substantial", "severe"]
    trip_df["speeding_severity"] = pd.Categorical(
        np.select(conditions, choices, default="none"),
        categories=choices,
        ordered=True,
    )

    # Mode categories (clean up)
    trip_df["mode_clean"] = trip_df["mode"].replace({
        "SCOOTER_TUB": "TUB",
        "SCOOTER_STD": "STD",
        "SCOOTER_ECO": "ECO",
        "none": "NONE",
    }).astype("category")

    print(f"  Final trip modeling dataset: {len(trip_df):,} trips x {len(trip_df.columns)} columns")

    return trip_df


def build_user_modeling_dataset(
    user_demo: pd.DataFrame,
    trip_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build user-level modeling dataset.

    Args:
        user_demo: User demographics DataFrame.
        trip_df: Trip-level modeling DataFrame.

    Returns:
        User-level modeling DataFrame.
    """
    print("\nBuilding user-level modeling dataset...")
    con = duckdb.connect()

    # Load user indicators
    user_ind = con.execute(f"""
        SELECT * FROM read_parquet('{USER_INDICATORS_PARQUET}')
    """).fetchdf()
    print(f"  User indicators: {len(user_ind):,} users x {len(user_ind.columns)} columns")
    con.close()

    # Merge with demographics
    user_df = user_ind.merge(
        user_demo[["user_id", "age", "total_bill", "avg_bill"]],
        on="user_id",
        how="left",
    )
    age_coverage = user_df["age"].notna().mean() * 100
    print(f"  Age coverage: {age_coverage:.1f}% of users have age data")

    # Create age groups
    user_df["age_group"] = create_age_groups(user_df["age"])

    # User frequency categories
    user_df["usage_category"] = pd.cut(
        user_df["trip_count"],
        bins=[0, 1, 3, 10, 30, 100, float("inf")],
        labels=["one_time", "occasional", "regular", "frequent", "heavy", "super_heavy"],
        right=True,
    )

    # Speeder classification
    conditions = [
        user_df["speeding_propensity"] == 0,
        user_df["speeding_propensity"] <= 0.25,
        user_df["speeding_propensity"] <= 0.5,
        user_df["speeding_propensity"] <= 0.75,
        user_df["speeding_propensity"] > 0.75,
    ]
    choices = ["never", "rare", "occasional", "frequent", "habitual"]
    user_df["speeder_type"] = pd.Categorical(
        np.select(conditions, choices, default="never"),
        categories=choices,
        ordered=True,
    )

    # Aggregate trip-level info to user level: most common city, mode
    user_trip_agg = trip_df.groupby("user_id").agg(
        primary_city=("city", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
        primary_province=("province", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
        primary_mode_clean=("mode_clean", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
        pct_weekend_trips=("is_weekend", "mean"),
        pct_night_trips=("start_hour", lambda x: ((x >= 22) | (x < 6)).mean()),
        mean_distance=("distance", "mean"),
        total_distance=("distance", "sum"),
    ).reset_index()

    user_df = user_df.merge(user_trip_agg, on="user_id", how="left")

    print(f"  Final user modeling dataset: {len(user_df):,} users x {len(user_df.columns)} columns")

    return user_df


def compute_summary_stats(
    trip_df: pd.DataFrame,
    user_df: pd.DataFrame,
) -> dict:
    """Compute summary statistics for the modeling datasets.

    Args:
        trip_df: Trip-level modeling DataFrame.
        user_df: User-level modeling DataFrame.

    Returns:
        Dict of summary statistics.
    """
    summary = {
        "trip_dataset": {
            "n_trips": len(trip_df),
            "n_columns": len(trip_df.columns),
            "columns": list(trip_df.columns),
            "age_coverage_pct": round(trip_df["age"].notna().mean() * 100, 1),
            "speeding_rate": round(trip_df["is_speeding"].mean() * 100, 1),
            "speeding_severity_dist": trip_df["speeding_severity"].value_counts().to_dict(),
            "age_group_dist": trip_df["age_group"].value_counts().sort_index().to_dict(),
            "time_of_day_dist": trip_df["time_of_day"].value_counts().to_dict(),
            "mode_dist": trip_df["mode_clean"].value_counts().to_dict(),
            "province_dist": trip_df["province"].value_counts().head(10).to_dict(),
        },
        "user_dataset": {
            "n_users": len(user_df),
            "n_columns": len(user_df.columns),
            "columns": list(user_df.columns),
            "age_coverage_pct": round(user_df["age"].notna().mean() * 100, 1),
            "usage_category_dist": user_df["usage_category"].value_counts().sort_index().to_dict(),
            "speeder_type_dist": user_df["speeder_type"].value_counts().sort_index().to_dict(),
            "mean_speeding_propensity": round(user_df["speeding_propensity"].mean(), 3),
            "mean_trips_per_user": round(user_df["trip_count"].mean(), 1),
        },
    }
    return summary


def main() -> None:
    """Prepare modeling datasets."""
    print("=" * 60)
    print("Task 5.1: Prepare Modeling Dataset")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # Step 1: Extract user demographics
    user_demo = extract_user_demographics()

    # Step 2: Build trip-level modeling dataset
    trip_df = build_trip_modeling_dataset(user_demo)

    # Step 3: Build user-level modeling dataset
    user_df = build_user_modeling_dataset(user_demo, trip_df)

    # Step 4: Save datasets
    print("\nSaving datasets...")

    trip_path = MODELING_DIR / "trip_modeling.parquet"
    trip_df.to_parquet(trip_path, index=False)
    print(f"  Trip modeling: {trip_path} ({trip_path.stat().st_size / 1e9:.2f} GB)")

    user_path = MODELING_DIR / "user_modeling.parquet"
    user_df.to_parquet(user_path, index=False)
    print(f"  User modeling: {user_path} ({user_path.stat().st_size / 1e6:.1f} MB)")

    # Step 5: Summary statistics
    summary = compute_summary_stats(trip_df, user_df)
    summary_path = MODELING_DIR / "modeling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Summary: {summary_path}")

    # Print key stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTrip modeling dataset: {len(trip_df):,} trips x {len(trip_df.columns)} cols")
    print(f"User modeling dataset: {len(user_df):,} users x {len(user_df.columns)} cols")
    print(f"Age coverage (trips): {summary['trip_dataset']['age_coverage_pct']}%")
    print(f"Age coverage (users): {summary['user_dataset']['age_coverage_pct']}%")

    print(f"\nSpeeding rate: {summary['trip_dataset']['speeding_rate']}%")
    print(f"\nSpeeding severity distribution:")
    for k, v in summary["trip_dataset"]["speeding_severity_dist"].items():
        print(f"  {k}: {v:,} ({100*v/len(trip_df):.1f}%)")

    print(f"\nAge group distribution:")
    for k, v in summary["trip_dataset"]["age_group_dist"].items():
        print(f"  {k}: {v:,} ({100*v/len(trip_df):.1f}%)")

    print(f"\nSpeeder type distribution:")
    for k, v in summary["user_dataset"]["speeder_type_dist"].items():
        print(f"  {k}: {v:,} ({100*v/len(user_df):.1f}%)")


if __name__ == "__main__":
    main()
