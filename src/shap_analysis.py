"""
SHAP interpretability analysis for e-scooter speeding prediction.

Trains a LightGBM gradient boosted model for speeding prediction, then
computes SHAP values for global and local interpretability.

Outputs:
  - figures/fig_shap_summary.pdf          -- SHAP beeswarm summary plot
  - figures/fig_shap_bar.pdf              -- SHAP mean importance bar chart
  - figures/fig_shap_dependence.pdf       -- dependence plots for top features
  - figures/fig_shap_interaction.pdf      -- interaction effects for top pairs
  - data_parquet/modeling/shap_results.json -- model performance and SHAP stats

Usage:
    python src/shap_analysis.py
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR, FIGURES_DIR, MODELING_DIR,
    RANDOM_SEED, FIG_DPI, SPEED_LIMIT_KR,
)

np.random.seed(RANDOM_SEED)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELING_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = MODELING_DIR / "shap_results.json"

# Publication style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
})

# Feature names for display
FEATURE_DISPLAY_NAMES = {
    "distance": "Trip distance (m)",
    "travel_time": "Travel time (s)",
    "gps_points": "GPS points",
    "speed_cv": "Speed CV",
    "mean_abs_accel_ms2": "Mean |acceleration| (m/s2)",
    "harsh_event_rate": "Harsh event rate",
    "cruise_fraction": "Cruise fraction",
    "zero_speed_fraction": "Zero-speed fraction",
    "frac_major_road": "Major road fraction",
    "frac_cycling_infra": "Cycling infra fraction",
    "frac_residential": "Residential fraction",
    "frac_tertiary": "Tertiary road fraction",
    "frac_service": "Service road fraction",
    "frac_footway": "Footway fraction",
    "n_road_classes": "Road class diversity",
    "log_distance": "Log(distance)",
    "start_hour": "Hour of day",
    "is_weekend": "Weekend",
    "age": "Age (years)",
    "mode_TUB": "TUB mode",
    "mode_ECO": "ECO mode",
}


def load_data() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Load and prepare feature matrix and target for SHAP analysis.

    Returns:
        Tuple of (feature DataFrame, target array, feature names).
    """
    print("Loading data...")
    t0 = time.time()
    con = duckdb.connect()

    df = con.execute("""
        SELECT
            m.route_id,
            m.distance,
            m.travel_time,
            m.gps_points,
            m.speed_cv,
            m.mean_abs_accel_ms2,
            m.harsh_event_rate,
            m.cruise_fraction,
            m.zero_speed_fraction,
            m.start_hour,
            m.is_weekend,
            m.age,
            m.is_speeding,
            CASE WHEN m.mode_clean = 'TUB' THEN 1 ELSE 0 END AS mode_TUB,
            CASE WHEN m.mode_clean = 'ECO' THEN 1 ELSE 0 END AS mode_ECO,
            r.frac_major_road,
            r.frac_cycling_infra,
            r.frac_residential,
            r.frac_tertiary,
            r.frac_service,
            r.frac_footway,
            r.n_road_classes
        FROM read_parquet('{trip_mod}') m
        JOIN read_parquet('{road}') r
            ON m.route_id = r.route_id
        WHERE m.mode IN ('SCOOTER_TUB', 'SCOOTER_STD', 'SCOOTER_ECO')
          AND m.age IS NOT NULL
    """.format(
        trip_mod=str(MODELING_DIR / "trip_modeling.parquet").replace("\\", "/"),
        road=str(DATA_DIR / "trip_road_classes.parquet").replace("\\", "/"),
    )).fetchdf()
    con.close()

    # Add log distance
    df["log_distance"] = np.log(df["distance"] + 1)

    elapsed = time.time() - t0
    print(f"  Loaded {len(df):,} trips in {elapsed:.1f}s")
    print(f"  Speeding rate: {df['is_speeding'].mean():.3f}")

    # Define features
    feature_cols = [
        "log_distance", "travel_time", "gps_points",
        "speed_cv", "mean_abs_accel_ms2", "harsh_event_rate",
        "cruise_fraction", "zero_speed_fraction",
        "start_hour", "is_weekend", "age",
        "mode_TUB", "mode_ECO",
        "frac_major_road", "frac_cycling_infra",
        "frac_residential", "frac_tertiary", "frac_service",
        "frac_footway", "n_road_classes",
    ]

    # Drop rows with any NaN in features
    df_clean = df.dropna(subset=feature_cols + ["is_speeding"])
    print(f"  After dropping NaN: {len(df_clean):,} trips")

    X = df_clean[feature_cols]
    y = df_clean["is_speeding"].astype(int).values

    return X, y, feature_cols


def train_lightgbm(X: pd.DataFrame, y: np.ndarray,
                    feature_cols: List[str]) -> Tuple[lgb.LGBMClassifier, Dict]:
    """Train LightGBM classifier and evaluate performance.

    Args:
        X: Feature matrix.
        y: Binary target (is_speeding).
        feature_cols: List of feature column names.

    Returns:
        Tuple of (trained model, performance metrics dict).
    """
    print("\n=== Training LightGBM ===")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train speeding rate: {y_train.mean():.3f}")
    print(f"  Test speeding rate: {y_test.mean():.3f}")

    # LightGBM parameters
    params = {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
        "is_unbalance": True,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(period=100)],
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_estimators_used": model.n_estimators_,
        "params": {k: v for k, v in params.items() if k != "verbose"},
    }

    print(f"\n  Performance:")
    print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1: {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")

    # Feature importance (gain)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n  Top 10 features (gain):")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:,.0f}")

    metrics["feature_importance_gain"] = {
        row["feature"]: int(row["importance"])
        for _, row in importance.iterrows()
    }

    return model, metrics


def compute_shap_values(model: lgb.LGBMClassifier,
                         X: pd.DataFrame,
                         feature_cols: List[str],
                         max_samples: int = 50_000) -> Tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values using TreeExplainer.

    Args:
        model: Trained LightGBM model.
        X: Full feature matrix.
        feature_cols: Feature names.
        max_samples: Maximum samples for SHAP computation.

    Returns:
        Tuple of (SHAP explanation object, subsample DataFrame).
    """
    print(f"\n=== Computing SHAP values (n={min(max_samples, len(X)):,}) ===")
    t0 = time.time()

    # Subsample for SHAP (TreeExplainer can be slow on very large datasets)
    if len(X) > max_samples:
        idx = np.random.choice(len(X), size=max_samples, replace=False)
        X_shap = X.iloc[idx].copy()
    else:
        X_shap = X.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)

    elapsed = time.time() - t0
    print(f"  Computed in {elapsed:.1f}s")

    return shap_values, X_shap


def plot_shap_summary(shap_values: shap.Explanation,
                       X_shap: pd.DataFrame) -> None:
    """Generate SHAP beeswarm summary plot.

    Args:
        shap_values: SHAP explanation object.
        X_shap: Feature data used for SHAP.
    """
    print("\n  Generating SHAP beeswarm plot...")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=20)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(FIGURES_DIR / f"fig_shap_summary.{ext}", dpi=FIG_DPI,
                    bbox_inches="tight")
    plt.close("all")
    print("    Saved fig_shap_summary.pdf/png")


def plot_shap_bar(shap_values: shap.Explanation) -> None:
    """Generate SHAP mean absolute value bar chart.

    Args:
        shap_values: SHAP explanation object.
    """
    print("  Generating SHAP importance bar plot...")

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.bar(shap_values, show=False, max_display=15)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(FIGURES_DIR / f"fig_shap_bar.{ext}", dpi=FIG_DPI,
                    bbox_inches="tight")
    plt.close("all")
    print("    Saved fig_shap_bar.pdf/png")


def plot_shap_dependence(shap_values: shap.Explanation,
                          X_shap: pd.DataFrame,
                          feature_cols: List[str],
                          top_n: int = 6) -> None:
    """Generate SHAP dependence plots for top features.

    Args:
        shap_values: SHAP explanation object.
        X_shap: Feature data.
        feature_cols: Feature names.
        top_n: Number of top features to plot.
    """
    print(f"  Generating dependence plots for top {top_n} features...")

    # Get top features by mean |SHAP|
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_cols[i] for i in top_indices]

    ncols = 3
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if top_n > 1 else [axes]

    for i, feat in enumerate(top_features):
        ax = axes_flat[i]
        feat_idx = feature_cols.index(feat)

        display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)

        shap.plots.scatter(
            shap_values[:, feat_idx],
            ax=ax,
            show=False,
        )
        ax.set_xlabel(display_name)
        ax.set_ylabel("SHAP value")

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("SHAP Dependence Plots (Top Features)", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_shap_dependence.{ext}", dpi=FIG_DPI,
                    bbox_inches="tight")
    plt.close(fig)
    print("    Saved fig_shap_dependence.pdf/png")


def compute_shap_stats(shap_values: shap.Explanation,
                        feature_cols: List[str]) -> Dict[str, Any]:
    """Compute summary statistics from SHAP values.

    Args:
        shap_values: SHAP explanation object.
        feature_cols: Feature names.

    Returns:
        Dictionary with SHAP statistics per feature.
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    mean_shap = shap_values.values.mean(axis=0)
    std_shap = shap_values.values.std(axis=0)

    # Rank features
    rank_order = np.argsort(mean_abs_shap)[::-1]

    stats = {}
    for rank, idx in enumerate(rank_order):
        feat = feature_cols[idx]
        stats[feat] = {
            "rank": int(rank + 1),
            "mean_abs_shap": float(mean_abs_shap[idx]),
            "mean_shap": float(mean_shap[idx]),
            "std_shap": float(std_shap[idx]),
            "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
        }

    return stats


def main() -> None:
    """Run SHAP analysis pipeline."""
    t0 = time.time()
    print("=" * 70)
    print("  SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    # Load data
    X, y, feature_cols = load_data()

    # Train model
    model, model_metrics = train_lightgbm(X, y, feature_cols)

    # Compute SHAP values
    shap_values, X_shap = compute_shap_values(model, X, feature_cols)

    # Generate plots
    plot_shap_summary(shap_values, X_shap)
    plot_shap_bar(shap_values)
    plot_shap_dependence(shap_values, X_shap, feature_cols)

    # Compute SHAP stats
    shap_stats = compute_shap_stats(shap_values, feature_cols)

    # Print top features
    print("\n=== Top Features by Mean |SHAP| ===")
    sorted_feats = sorted(shap_stats.items(), key=lambda x: x[1]["rank"])
    for feat, info in sorted_feats[:10]:
        print(f"  {info['rank']:2d}. {info['display_name']:30s}: "
              f"mean|SHAP|={info['mean_abs_shap']:.4f}")

    # Save results
    elapsed = time.time() - t0
    results = {
        "model_performance": model_metrics,
        "shap_feature_importance": shap_stats,
        "n_shap_samples": int(len(X_shap)),
        "n_features": int(len(feature_cols)),
        "feature_list": feature_cols,
        "processing_time_sec": round(elapsed, 1),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {RESULTS_PATH}")

    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
