"""
Task 5.4: Two-part model for segment-level speeding rate.

Part 1: Logistic regression - P(any speeding | road class, trip features)
Part 2: Beta regression - E(speeding_rate | speeding > 0, road class, trip features)

The speeding_rate_25 variable is zero-inflated (70.4% zeros), so a standard
beta regression is inappropriate. We use a two-part (hurdle) model instead.

Usage:
    python src/beta_regression.py
"""

import json
import time
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR, MODELING_DIR, REPORTS_DIR,
    RANDOM_SEED, SPEED_LIMIT_KR,
)

np.random.seed(RANDOM_SEED)


def load_data() -> pd.DataFrame:
    """Load merged trip data with road class features and speed indicators."""
    print("Loading data for beta regression...")
    con = duckdb.connect()

    df = con.execute("""
        SELECT
            m.route_id,
            m.mode_clean as mode,
            m.age_group,
            m.time_of_day,
            m.day_type,
            m.province,
            m.distance,
            m.speeding_rate_25,
            m.is_speeding,
            m.mean_speed,
            r.dominant_road_class,
            r.frac_major_road,
            r.frac_cycling_infra,
            r.n_road_classes,
            r.frac_residential,
            r.frac_tertiary,
            r.frac_service,
            r.frac_footway
        FROM read_parquet($modeling) m
        JOIN read_parquet($road) r USING (route_id)
        WHERE m.mode_clean IN ('STD', 'TUB', 'ECO')
    """, {
        "modeling": str(MODELING_DIR / "trip_modeling.parquet"),
        "road": str(DATA_DIR / "trip_road_classes.parquet"),
    }).df()

    print(f"  Loaded {len(df):,} trips")
    print(f"  Speeding rate: mean={df['speeding_rate_25'].mean():.4f}, "
          f"zero={( df['speeding_rate_25'] == 0).mean()*100:.1f}%")

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create feature matrix with dummy variables."""
    print("Preparing features...")

    df = df.copy()
    df["log_distance"] = np.log1p(df["distance"])

    # Create dummies for categorical variables
    cat_vars = {
        "mode": "STD",          # reference
        "age_group": "25-29",   # reference (most common)
        "time_of_day": "midday",  # reference
        "day_type": "weekday",    # reference
        "dominant_road_class": "residential",  # reference (most common)
    }

    feature_dfs = [df[["log_distance", "frac_major_road", "frac_cycling_infra", "n_road_classes"]]]
    feature_names = ["log_distance", "frac_major_road", "frac_cycling_infra", "n_road_classes"]

    for var, ref in cat_vars.items():
        dummies = pd.get_dummies(df[var], prefix=var, drop_first=False, dtype=float)
        ref_col = f"{var}_{ref}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])
        feature_dfs.append(dummies)
        feature_names.extend(dummies.columns.tolist())

    X = pd.concat(feature_dfs, axis=1)
    X = sm.add_constant(X)

    print(f"  Features: {X.shape[1]} columns (incl. constant)")
    return X, feature_names


def run_part1_logistic(X: pd.DataFrame, y_binary: pd.Series) -> dict:
    """Part 1: Logistic regression for P(any speeding)."""
    print("\n" + "="*60)
    print("PART 1: Logistic Regression - P(speeding > 0)")
    print("="*60)

    # Use subsample for speed (full data is 2.5M)
    n_sub = 200_000
    idx = np.random.choice(len(X), size=n_sub, replace=False)
    X_sub = X.iloc[idx]
    y_sub = y_binary.iloc[idx]

    print(f"  Subsample: {n_sub:,} trips, speeding prevalence: {y_sub.mean()*100:.1f}%")

    model = sm.Logit(y_sub, X_sub)
    result = model.fit(disp=0, maxiter=100, method="newton")

    print(f"  Pseudo R2: {result.prsquared:.4f}")
    print(f"  AIC: {result.aic:.0f}")
    print(f"  Log-likelihood: {result.llf:.0f}")

    # Extract key coefficients as odds ratios
    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    coef_df = pd.DataFrame({
        "coefficient": params,
        "OR": np.exp(params),
        "OR_lower": np.exp(conf[0]),
        "OR_upper": np.exp(conf[1]),
        "p_value": pvals,
        "significant": pvals < 0.05,
    })

    print("\nKey Odds Ratios (vs reference):")
    key_vars = [c for c in coef_df.index if any(
        k in c for k in ["mode_", "dominant_road_class_", "frac_major", "frac_cycling"]
    )]
    for var in key_vars:
        row = coef_df.loc[var]
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {var}: OR={row['OR']:.3f} [{row['OR_lower']:.3f}, {row['OR_upper']:.3f}] {sig}")

    results = {
        "n_obs": int(n_sub),
        "pseudo_r2": float(result.prsquared),
        "aic": float(result.aic),
        "log_likelihood": float(result.llf),
        "coefficients": coef_df.to_dict("index"),
    }

    return results


def run_part2_beta(X: pd.DataFrame, y_rate: pd.Series, mask_positive: pd.Series) -> dict:
    """Part 2: Beta regression for speeding rate conditional on speeding > 0.

    Uses GLM with Binomial family and logit link as a quasi-beta regression,
    which is equivalent to beta regression for proportions in (0,1).
    """
    print("\n" + "="*60)
    print("PART 2: Beta Regression - E(speeding_rate | speeding > 0)")
    print("="*60)

    # Filter to trips with speeding > 0 and < 1
    valid = mask_positive & (y_rate < 1.0) & (y_rate > 0.0)
    X_pos = X[valid].copy()
    y_pos = y_rate[valid].copy()

    print(f"  Trips with 0 < speeding_rate < 1: {len(X_pos):,}")
    print(f"  Mean speeding rate (conditional): {y_pos.mean():.4f}")

    # Subsample for speed
    n_sub = min(200_000, len(X_pos))
    if n_sub < len(X_pos):
        idx = np.random.choice(len(X_pos), size=n_sub, replace=False)
        X_sub = X_pos.iloc[idx]
        y_sub = y_pos.iloc[idx]
    else:
        X_sub = X_pos
        y_sub = y_pos

    print(f"  Subsample: {n_sub:,} trips")

    # GLM with Binomial family = quasi-beta regression for proportions
    model = sm.GLM(y_sub, X_sub, family=Binomial(link=Logit()))
    result = model.fit(maxiter=100)

    print(f"  Deviance: {result.deviance:.2f}")
    print(f"  AIC: {result.aic:.0f}")
    print(f"  Pearson chi2/df: {result.pearson_chi2 / result.df_resid:.4f}")

    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    coef_df = pd.DataFrame({
        "coefficient": params,
        "exp_coef": np.exp(params),
        "ci_lower": conf[0],
        "ci_upper": conf[1],
        "p_value": pvals,
        "significant": pvals < 0.05,
    })

    print("\nKey Coefficients (logit scale):")
    key_vars = [c for c in coef_df.index if any(
        k in c for k in ["mode_", "dominant_road_class_", "frac_major", "frac_cycling"]
    )]
    for var in key_vars:
        row = coef_df.loc[var]
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {var}: coef={row['coefficient']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] {sig}")

    results = {
        "n_obs": int(n_sub),
        "deviance": float(result.deviance),
        "aic": float(result.aic),
        "pearson_chi2_df": float(result.pearson_chi2 / result.df_resid),
        "coefficients": coef_df.to_dict("index"),
    }

    return results


def main() -> None:
    """Run two-part beta regression model."""
    t0 = time.time()

    # Load data
    df = load_data()

    # Prepare features
    X, feature_names = prepare_features(df)

    # Binary outcome: any speeding
    y_binary = (df["speeding_rate_25"] > 0).astype(int)

    # Continuous outcome: speeding rate
    y_rate = df["speeding_rate_25"]

    # Part 1: Logistic
    part1_results = run_part1_logistic(X, y_binary)

    # Part 2: Beta regression (conditional on speeding > 0)
    part2_results = run_part2_beta(X, y_rate, y_binary == 1)

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Two-Part Model")
    print("="*60)
    print(f"  Part 1 (Logistic): N={part1_results['n_obs']:,}, "
          f"Pseudo R2={part1_results['pseudo_r2']:.4f}")
    print(f"  Part 2 (Beta/GLM): N={part2_results['n_obs']:,}, "
          f"AIC={part2_results['aic']:.0f}")

    # Compare road class effects across both parts
    print("\nRoad Class Effects (vs residential):")
    road_vars = [c for c in X.columns if "dominant_road_class_" in c]
    for var in sorted(road_vars):
        short_name = var.replace("dominant_road_class_", "")
        p1 = part1_results["coefficients"].get(var, {})
        p2 = part2_results["coefficients"].get(var, {})
        p1_or = p1.get("OR", float("nan"))
        p1_sig = "***" if p1.get("p_value", 1) < 0.001 else ""
        p2_coef = p2.get("coefficient", float("nan"))
        p2_sig = "***" if p2.get("p_value", 1) < 0.001 else ""
        print(f"  {short_name:15s}: OR={p1_or:6.3f}{p1_sig:3s} | "
              f"beta={p2_coef:7.4f}{p2_sig:3s}")

    # Save results
    all_results = {
        "model_type": "two-part (hurdle) model",
        "description": "Part 1: Logistic for P(speeding>0); Part 2: GLM-Binomial for E(rate|rate>0)",
        "part1_logistic": part1_results,
        "part2_beta": part2_results,
    }

    report_path = MODELING_DIR / "beta_regression_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nSaved results: {report_path}")

    elapsed = time.time() - t0
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
