"""
Task 1.4: LightGBM + SHAP for safety outcomes.

Separate models for:
  - Safety: harsh_event_count (binary: any harsh event?)
  - Dynamics: cruise_fraction (regression)
Compare feature importance rankings across categories.
Save SHAP comparison figure (Fig 3).
"""

import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, FIG_DPI, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
FIG_DIR = FIGURES_DIR / "v2"
TRIP_PATH = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")


def main():
    print("=" * 60)
    print("Task 1.4: LightGBM + SHAP Analysis")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()

    # Sample 300K for SHAP (full dataset too large for TreeExplainer)
    print("  Sampling 300K trips...")
    df = con.execute(f"""
        SELECT
            mode, age_group, time_of_day, day_type,
            is_major_road, distance, travel_time, start_road_class, city,
            n_speed_points, mean_speed,
            harsh_event_count, harsh_accel_count, harsh_decel_count,
            speed_cv, cruise_fraction, zero_speed_fraction
        FROM read_parquet('{TRIP_PATH}')
        USING SAMPLE 300000 ROWS (reservoir)
    """).fetchdf()
    con.close()

    print(f"  Sampled: {len(df):,} trips")

    # Encode features
    df["is_tub"] = (df["mode"] == "TUB").astype(int)
    df["is_eco"] = (df["mode"] == "ECO").astype(int)
    df["log_distance"] = np.log1p(df["distance"])
    df["is_weekend"] = (df["day_type"] == "weekend").astype(int)

    # One-hot encode age_group and time_of_day
    age_dummies = pd.get_dummies(df["age_group"], prefix="age", dtype=int)
    tod_dummies = pd.get_dummies(df["time_of_day"], prefix="tod", dtype=int)

    # Road class simplified
    road_dummies = pd.get_dummies(df["start_road_class"], prefix="road", dtype=int)
    # Keep top 5 road classes
    top_roads = df["start_road_class"].value_counts().head(5).index
    road_cols = [f"road_{r}" for r in top_roads if f"road_{r}" in road_dummies.columns]

    feature_df = pd.concat([
        df[["is_tub", "is_eco", "log_distance", "is_major_road",
            "is_weekend", "n_speed_points"]],
        age_dummies, tod_dummies,
        road_dummies[road_cols] if road_cols else pd.DataFrame(),
    ], axis=1)

    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(float)

    # Model 1: Safety — binary harsh event (any harsh event?)
    print("\n  Model 1: Safety (any harsh event)")
    y_safety = (df["harsh_event_count"] > 0).astype(int)
    _run_shap_model(X, y_safety, feature_names, "safety_harsh_event",
                    "Any Harsh Event (Binary)", objective="binary")

    # Model 2: Dynamics — cruise fraction
    print("\n  Model 2: Dynamics (cruise fraction)")
    y_cruise = df["cruise_fraction"].values
    _run_shap_model(X, y_cruise, feature_names, "dynamics_cruise_fraction",
                    "Cruise Fraction", objective="regression")

    # Model 3: Safety — speed_cv
    print("\n  Model 3: Safety (speed CV)")
    y_cv = df["speed_cv"].values
    _run_shap_model(X, y_cv, feature_names, "safety_speed_cv",
                    "Speed CV", objective="regression")

    elapsed = time.time() - t_start
    print(f"\n  Task 1.4 complete in {elapsed:.0f}s")


def _run_shap_model(X, y, feature_names, name, title, objective="binary"):
    """Train LightGBM and compute SHAP values."""
    t0 = time.time()

    params = {
        "objective": objective,
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }
    if objective == "binary":
        params["metric"] = "auc"

    model = lgb.LGBMClassifier(**params) if objective == "binary" else lgb.LGBMRegressor(**params)
    model.fit(X, y)

    if objective == "binary":
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        print(f"    AUC: {auc:.3f}")
    else:
        from sklearn.metrics import r2_score
        r2 = r2_score(y, model.predict(X))
        print(f"    R2: {r2:.3f}")

    # SHAP (use 5K subset for speed)
    explainer = shap.TreeExplainer(model)
    shap_subset = X[:5000]
    shap_values = explainer.shap_values(shap_subset)

    # For binary classification, shap_values may be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_subset, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title(f"SHAP: {title}", fontsize=12)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(FIG_DIR / f"fig3_shap_{name}.{fmt}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close("all")

    # Top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:10]
    print(f"    Top features:")
    for i in top_idx:
        print(f"      {feature_names[i]:<25s} SHAP={mean_shap[i]:.4f}")

    print(f"    ({time.time() - t0:.1f}s)")

    # Save result
    result = {
        "name": name,
        "title": title,
        "top_features": [
            {"feature": feature_names[i], "mean_abs_shap": round(float(mean_shap[i]), 4)}
            for i in top_idx
        ],
    }
    result_path = V2_DIR / f"shap_{name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Saved: {result_path}")


if __name__ == "__main__":
    main()
