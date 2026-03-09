"""
Task 5.5: Latent class analysis / GMM on user-level speed indicators.

Uses Gaussian Mixture Models to identify distinct rider types based on
speed behavior indicators. Selects optimal K via BIC, profiles classes,
and assigns class labels to users.

Strategy:
  - Select relevant speed behavior features from user_indicators
  - Standardize features
  - Fit GMMs with K=2..8, select optimal K via BIC
  - Profile each class (mean feature values, demographics)
  - Assign class labels and save

Outputs:
  - data_parquet/modeling/gmm_results.json — BIC scores, class profiles
  - data_parquet/modeling/user_classes.parquet — user-level with class labels
  - figures/gmm_bic.png — BIC vs K plot
  - figures/gmm_profiles.png — radar/spider plots of class profiles
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    FIG_DPI,
)

MODELING_DIR = DATA_DIR / "modeling"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Features for clustering — behavioral indicators that characterize riding style
CLUSTER_FEATURES = [
    "user_mean_speed",
    "user_mean_max_speed",
    "user_mean_speed_cv",
    "speeding_propensity",
    "user_mean_speeding_rate_25",
    "user_mean_abs_accel",
    "harsh_accel_propensity",
    "harsh_decel_propensity",
    "user_mean_cruise_fraction",
    "user_mean_zero_fraction",
]

# Human-readable feature names for plots
FEATURE_LABELS = {
    "user_mean_speed": "Mean Speed",
    "user_mean_max_speed": "Mean Max Speed",
    "user_mean_speed_cv": "Speed Variability",
    "speeding_propensity": "Speeding Propensity",
    "user_mean_speeding_rate_25": "Avg Speeding Rate",
    "user_mean_abs_accel": "Mean |Accel|",
    "harsh_accel_propensity": "Harsh Accel Rate",
    "harsh_decel_propensity": "Harsh Decel Rate",
    "user_mean_cruise_fraction": "Cruise Fraction",
    "user_mean_zero_fraction": "Zero-Speed Fraction",
}

# Range of K to test
K_RANGE = range(2, 9)


def load_user_data() -> pd.DataFrame:
    """Load user modeling dataset.

    Returns:
        DataFrame with user-level indicators.
    """
    user_path = MODELING_DIR / "user_modeling.parquet"
    df = pd.read_parquet(user_path)
    print(f"Loaded {len(df):,} users with {len(df.columns)} columns")
    return df


def prepare_features(
    df: pd.DataFrame,
    min_trips: int = 3,
) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """Prepare and standardize features for GMM.

    Args:
        df: User modeling DataFrame.
        min_trips: Minimum trip count to include user in clustering.

    Returns:
        Tuple of (standardized feature array, filtered DataFrame, scaler).
    """
    # Filter users with enough trips for reliable indicators
    mask = df["trip_count"] >= min_trips
    df_filt = df[mask].copy()
    print(f"Users with >= {min_trips} trips: {len(df_filt):,} ({100*len(df_filt)/len(df):.1f}%)")

    # Select features and handle missing
    X_raw = df_filt[CLUSTER_FEATURES].copy()
    n_missing = X_raw.isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"Dropping {n_missing} users with missing features")
        valid_mask = X_raw.notna().all(axis=1)
        X_raw = X_raw[valid_mask]
        df_filt = df_filt[valid_mask]

    print(f"Final clustering sample: {len(X_raw):,} users x {len(CLUSTER_FEATURES)} features")

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw.values)

    return X_std, df_filt, scaler


def fit_gmm_range(
    X: np.ndarray,
    k_range: range = K_RANGE,
    seed: int = RANDOM_SEED,
) -> dict:
    """Fit GMMs for range of K and compute BIC/AIC.

    Args:
        X: Standardized feature array.
        k_range: Range of K values to test.
        seed: Random seed.

    Returns:
        Dict with K -> {model, bic, aic, log_likelihood} results.
    """
    results = {}
    for k in k_range:
        print(f"  Fitting GMM with K={k}...", end="", flush=True)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            max_iter=300,
            n_init=5,
            random_state=seed,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        ll = gmm.score(X) * len(X)  # total log-likelihood
        print(f" BIC={bic:.0f}, AIC={aic:.0f}, converged={gmm.converged_}")
        results[k] = {
            "model": gmm,
            "bic": bic,
            "aic": aic,
            "log_likelihood": ll,
            "converged": gmm.converged_,
            "n_iter": gmm.n_iter_,
        }

    return results


def select_optimal_k(results: dict) -> int:
    """Select optimal K based on BIC (lower is better).

    Args:
        results: Dict from fit_gmm_range.

    Returns:
        Optimal K value.
    """
    bic_scores = {k: r["bic"] for k, r in results.items()}
    optimal_k = min(bic_scores, key=bic_scores.get)
    print(f"\nOptimal K by BIC: {optimal_k}")
    return optimal_k


def plot_bic_aic(results: dict, output_path: Path) -> None:
    """Plot BIC and AIC vs K.

    Args:
        results: Dict from fit_gmm_range.
        output_path: Path to save figure.
    """
    ks = sorted(results.keys())
    bics = [results[k]["bic"] for k in ks]
    aics = [results[k]["aic"] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, bics, "o-", color="#2c3e50", linewidth=2, markersize=8, label="BIC")
    ax.plot(ks, aics, "s--", color="#e74c3c", linewidth=2, markersize=8, label="AIC")

    # Mark optimal
    opt_k = min(range(len(bics)), key=lambda i: bics[i])
    ax.axvline(ks[opt_k], color="gray", linestyle=":", alpha=0.5)
    ax.annotate(
        f"Optimal K={ks[opt_k]}",
        xy=(ks[opt_k], bics[opt_k]),
        xytext=(ks[opt_k] + 0.3, bics[opt_k]),
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlabel("Number of Components (K)", fontsize=12)
    ax.set_ylabel("Information Criterion", fontsize=12)
    ax.set_title("GMM Model Selection: BIC and AIC", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved BIC/AIC plot to {output_path}")


def profile_classes(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute mean feature values per class.

    Args:
        df: User DataFrame (filtered).
        labels: Class labels from GMM.
        feature_cols: Feature column names.

    Returns:
        DataFrame with class profiles.
    """
    df_labeled = df.copy()
    df_labeled["class_label"] = labels

    profiles = df_labeled.groupby("class_label")[feature_cols].mean()
    profiles["n_users"] = df_labeled.groupby("class_label").size()
    profiles["pct_users"] = (profiles["n_users"] / len(df_labeled) * 100).round(1)

    # Add demographics if available
    if "age" in df_labeled.columns:
        profiles["mean_age"] = df_labeled.groupby("class_label")["age"].mean()
    if "trip_count" in df_labeled.columns:
        profiles["mean_trips"] = df_labeled.groupby("class_label")["trip_count"].mean()

    return profiles


def plot_class_profiles(
    profiles: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    output_path: Path,
) -> None:
    """Plot radar/spider chart of class profiles.

    Args:
        profiles: DataFrame from profile_classes.
        feature_cols: Feature column names.
        scaler: StandardScaler used for normalization.
        output_path: Path to save figure.
    """
    n_classes = len(profiles)
    n_features = len(feature_cols)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Standardize profiles for visualization (0-1 scale)
    profile_values = profiles[feature_cols].values
    # Use min-max within profile for visualization
    pmin = profile_values.min(axis=0)
    pmax = profile_values.max(axis=0)
    prange = pmax - pmin
    prange[prange == 0] = 1  # avoid division by zero
    profile_norm = (profile_values - pmin) / prange

    # Colors for classes
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})

    for i, (cls_idx, row) in enumerate(profiles.iterrows()):
        values = profile_norm[i].tolist()
        values += values[:1]
        n_users = int(row["n_users"])
        pct = row["pct_users"]
        label = f"Class {cls_idx} (n={n_users:,}, {pct:.1f}%)"
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Feature labels
    feature_labels = [FEATURE_LABELS.get(f, f) for f in feature_cols]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=10)
    ax.set_ylim(0, 1.1)

    ax.set_title(
        f"Rider Typology: {n_classes}-Class GMM Profiles",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved class profiles plot to {output_path}")


def name_classes(profiles: pd.DataFrame) -> dict[int, str]:
    """Assign descriptive names to classes based on profile features.

    Args:
        profiles: DataFrame from profile_classes.

    Returns:
        Dict mapping class_label -> descriptive name.
    """
    names = {}
    for cls_idx, row in profiles.iterrows():
        speeding = row.get("speeding_propensity", 0)
        mean_speed = row.get("user_mean_speed", 0)
        harsh = row.get("harsh_accel_propensity", 0)
        cruise = row.get("user_mean_cruise_fraction", 0)

        if speeding > 0.5 and harsh > 0.5:
            names[cls_idx] = "Aggressive Speeder"
        elif speeding > 0.3:
            names[cls_idx] = "Frequent Speeder"
        elif speeding > 0.1 and mean_speed > 15:
            names[cls_idx] = "Moderate Risk"
        elif cruise > 0.4 and speeding < 0.1:
            names[cls_idx] = "Steady Cruiser"
        elif mean_speed < 12:
            names[cls_idx] = "Cautious Rider"
        else:
            names[cls_idx] = "Normal Rider"

    return names


def main() -> None:
    """Run latent class analysis."""
    print("=" * 60)
    print("Task 5.5: Latent Class Analysis (GMM)")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # Load data
    print("\nLoading user modeling dataset...")
    user_df = load_user_data()

    # Prepare features
    print("\nPreparing features...")
    X_std, df_filt, scaler = prepare_features(user_df, min_trips=3)

    # Fit GMM range
    print("\nFitting GMMs for K=2..8...")
    results = fit_gmm_range(X_std)

    # Select optimal K
    bic_optimal_k = select_optimal_k(results)

    # Plot BIC/AIC
    bic_path = FIGURES_DIR / "gmm_bic.png"
    plot_bic_aic(results, bic_path)

    # With large N (~200K), BIC may not plateau as even tiny fit improvements
    # are statistically significant. Use K=4 as the primary interpretive model
    # for parsimony and clear class separation. Report BIC-optimal as sensitivity.
    INTERPRETIVE_K = 4
    print(f"\nBIC-optimal K={bic_optimal_k}, but using K={INTERPRETIVE_K} "
          f"for interpretability (large N effect on BIC)")

    # Generate results for both the interpretive K and alternatives
    for k_label, use_k in [("PRIMARY", INTERPRETIVE_K), ("BIC_OPTIMAL", bic_optimal_k)]:
        if use_k not in results:
            continue
        model = results[use_k]["model"]
        labels = model.predict(X_std)
        probs = model.predict_proba(X_std)

        profiles = profile_classes(df_filt, labels, CLUSTER_FEATURES)
        class_names = name_classes(profiles)

        print(f"\n{'=' * 60}")
        print(f"CLASS PROFILES — {k_label} (K={use_k}, BIC={results[use_k]['bic']:.0f})")
        print("=" * 60)

        for cls_idx, row in profiles.iterrows():
            name = class_names.get(cls_idx, f"Class {cls_idx}")
            print(f"\n--- {name} (Class {cls_idx}) ---")
            print(f"  Users: {int(row['n_users']):,} ({row['pct_users']:.1f}%)")
            if "mean_age" in profiles.columns:
                print(f"  Mean age: {row['mean_age']:.1f}")
            if "mean_trips" in profiles.columns:
                print(f"  Mean trips: {row['mean_trips']:.1f}")
            print(f"  Mean speed: {row['user_mean_speed']:.1f} km/h")
            print(f"  Mean max speed: {row['user_mean_max_speed']:.1f} km/h")
            print(f"  Speeding propensity: {row['speeding_propensity']:.3f}")
            print(f"  Avg speeding rate: {row['user_mean_speeding_rate_25']:.3f}")
            print(f"  Speed CV: {row['user_mean_speed_cv']:.3f}")
            print(f"  Mean |accel|: {row['user_mean_abs_accel']:.3f} m/s\u00b2")
            print(f"  Harsh accel rate: {row['harsh_accel_propensity']:.3f}")
            print(f"  Cruise fraction: {row['user_mean_cruise_fraction']:.3f}")
            print(f"  Zero-speed fraction: {row['user_mean_zero_fraction']:.3f}")

        # Plot profiles
        profile_path = FIGURES_DIR / f"gmm_profiles_k{use_k}.png"
        plot_class_profiles(profiles, CLUSTER_FEATURES, scaler, profile_path)

        # Save class assignments for the primary model
        if k_label == "PRIMARY":
            print("\nSaving primary model class assignments...")
            df_filt["gmm_class"] = labels
            df_filt["gmm_class_name"] = df_filt["gmm_class"].map(class_names)
            df_filt["gmm_confidence"] = probs.max(axis=1)
            df_filt["gmm_k"] = use_k

    # Save
    user_classes_path = MODELING_DIR / "user_classes.parquet"
    df_filt.to_parquet(user_classes_path, index=False)
    print(f"  Saved: {user_classes_path}")

    # Save results JSON with all K variants
    primary_model = results[INTERPRETIVE_K]["model"]
    primary_labels = primary_model.predict(X_std)
    primary_profiles = profile_classes(df_filt.drop(
        columns=["gmm_class", "gmm_class_name", "gmm_confidence", "gmm_k"], errors="ignore"
    ), primary_labels, CLUSTER_FEATURES)
    primary_names = name_classes(primary_profiles)

    results_json = {
        "interpretive_k": INTERPRETIVE_K,
        "bic_optimal_k": bic_optimal_k,
        "note": "BIC does not plateau with large N (198K users). K=4 selected for interpretability.",
        "k_range": list(K_RANGE),
        "bic_scores": {str(k): float(r["bic"]) for k, r in results.items()},
        "aic_scores": {str(k): float(r["aic"]) for k, r in results.items()},
        "min_trips_filter": 3,
        "n_users_clustered": len(df_filt),
        "features_used": CLUSTER_FEATURES,
        "class_names": {str(k): v for k, v in primary_names.items()},
        "class_profiles": {},
    }
    for cls_idx, row in primary_profiles.iterrows():
        results_json["class_profiles"][str(cls_idx)] = {
            "n_users": int(row["n_users"]),
            "pct_users": float(row["pct_users"]),
            "mean_age": float(row.get("mean_age", 0)),
            "mean_trips": float(row.get("mean_trips", 0)),
            "features": {f: float(row[f]) for f in CLUSTER_FEATURES},
        }

    results_path = MODELING_DIR / "gmm_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_path}")

    # Report K=5 as well for comparison
    if 5 in results and 5 != INTERPRETIVE_K:
        print(f"\n--- Alternative: K=5 (BIC={results[5]['bic']:.0f}) ---")
        alt_labels = results[5]["model"].predict(X_std)
        alt_profiles = profile_classes(
            df_filt.drop(columns=["gmm_class", "gmm_class_name", "gmm_confidence", "gmm_k"], errors="ignore"),
            alt_labels, CLUSTER_FEATURES,
        )
        for cls_idx, row in alt_profiles.iterrows():
            print(f"  Class {cls_idx}: {int(row['n_users']):,} users "
                  f"({row['pct_users']:.1f}%), "
                  f"mean_speed={row['user_mean_speed']:.1f}, "
                  f"speeding={row['speeding_propensity']:.3f}")


if __name__ == "__main__":
    main()
