"""
Task 5.6: Multinomial logistic regression predicting GMM class membership.

Predicts rider type (GMM class) from demographics and usage patterns:
  - age_group
  - primary city/province
  - usage frequency (trip count, active days)
  - primary mode
  - temporal patterns (weekend fraction, night fraction)

This tells us which demographic factors predict membership in each rider type,
complementing the behavioral clustering (Task 5.5) with explanatory factors.

Outputs:
  - data_parquet/modeling/multinomial_results.json — coefficients and fit stats
  - figures/multinomial_coefficients.png — coefficient plot
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
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, RANDOM_SEED, FIG_DPI

MODELING_DIR = DATA_DIR / "modeling"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Proper class names based on profile analysis
CLASS_NAMES = {
    0: "Stop-and-Go",
    1: "Safe Rider",
    2: "Moderate Risk",
    3: "Habitual Speeder",
}

# Reference class for multinomial logit (Safe Rider = most common, baseline)
REFERENCE_CLASS = 1


def load_user_data() -> pd.DataFrame:
    """Load user-level data with GMM class assignments and demographics.

    Returns:
        DataFrame with user demographics and GMM class labels.
    """
    df = pd.read_parquet(MODELING_DIR / "user_classes.parquet")
    print(f"Loaded {len(df):,} users with GMM class assignments")

    # Fix class names
    df["class_name"] = df["gmm_class"].map(CLASS_NAMES)
    print("\nClass distribution:")
    for cls_id, name in sorted(CLASS_NAMES.items()):
        n = (df["gmm_class"] == cls_id).sum()
        pct = 100 * n / len(df)
        print(f"  {cls_id} ({name}): {n:,} ({pct:.1f}%)")

    return df


def prepare_predictors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create predictor variables for multinomial logistic regression.

    Args:
        df: User-level DataFrame with demographics.

    Returns:
        Tuple of (predictor DataFrame with dummies, list of predictor names).
    """
    # Filter to users with complete demographic data
    df_model = df.dropna(subset=["age_group", "primary_province_y"]).copy()

    # Also filter to users with gmm_class assigned (non-null)
    df_model = df_model[df_model["gmm_class"].notna()].copy()
    print(f"\nUsers with complete data for modeling: {len(df_model):,}")

    # Create predictor variables
    # 1. Age group (reference: <20, the youngest group)
    age_dummies = pd.get_dummies(df_model["age_group"], prefix="age", dtype=float)
    age_ref = "age_<20"
    if age_ref in age_dummies.columns:
        age_dummies = age_dummies.drop(columns=[age_ref])

    # 2. Province (keep top 5 provinces, merge rest to "Other")
    top_provinces = (
        df_model["primary_province_y"]
        .value_counts()
        .head(5)
        .index.tolist()
    )
    df_model["province_grouped"] = df_model["primary_province_y"].where(
        df_model["primary_province_y"].isin(top_provinces), "Other"
    )
    prov_dummies = pd.get_dummies(
        df_model["province_grouped"], prefix="prov", dtype=float
    )
    # Reference: Seoul (largest city)
    prov_ref = "prov_Seoul"
    if prov_ref in prov_dummies.columns:
        prov_dummies = prov_dummies.drop(columns=[prov_ref])

    # 3. Scooter mode (simplify: TUB vs ECO vs STD, exclude bike modes)
    # Map primary_mode to simplified categories
    mode_map = {
        "SCOOTER_TUB": "TUB",
        "SCOOTER_STD": "STD",
        "SCOOTER_ECO": "ECO",
        "none": "STD",  # default to STD for "none"
    }
    df_model["mode_simple"] = df_model["primary_mode"].map(mode_map).fillna("Other")
    mode_dummies = pd.get_dummies(df_model["mode_simple"], prefix="mode", dtype=float)
    mode_ref = "mode_STD"
    if mode_ref in mode_dummies.columns:
        mode_dummies = mode_dummies.drop(columns=[mode_ref])

    # 4. Continuous variables
    continuous_vars = pd.DataFrame({
        "log_trip_count": np.log1p(df_model["trip_count"]),
        "pct_weekend": df_model["weekend_trip_fraction"].fillna(0),
        "pct_night": df_model["pct_night_trips"].fillna(0),
    })

    # Combine all predictors
    X = pd.concat([age_dummies, prov_dummies, mode_dummies,
                    continuous_vars], axis=1)

    # Drop any remaining NaN rows
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    df_model = df_model[valid_mask]

    print(f"Final model sample: {len(X):,} users, {X.shape[1]} predictors")
    print(f"Predictors: {list(X.columns)}")

    return X, df_model


def fit_multinomial_logit(
    X: pd.DataFrame,
    y: pd.Series,
    reference_class: int = REFERENCE_CLASS,
) -> dict:
    """Fit multinomial logistic regression.

    Args:
        X: Predictor matrix.
        y: GMM class labels (0, 1, 2, 3).
        reference_class: Reference category for multinomial logit.

    Returns:
        Dict with model results.
    """
    print(f"\nFitting multinomial logistic regression...")
    print(f"  N = {len(X):,}, K predictors = {X.shape[1]}")
    print(f"  Reference class: {reference_class} ({CLASS_NAMES[reference_class]})")

    # Add constant
    X_const = sm.add_constant(X)

    # Fit MNLogit (use newton with more iterations)
    model = sm.MNLogit(y, X_const)
    result = model.fit(method="newton", maxiter=100, disp=True, full_output=True)

    print(f"\n  Log-likelihood: {result.llf:.1f}")
    print(f"  Pseudo R-squared (McFadden): {result.prsquared:.4f}")
    print(f"  AIC: {result.aic:.1f}")
    print(f"  BIC: {result.bic:.1f}")
    print(f"  Converged: {result.mle_retvals['converged']}")

    # Extract coefficients, standard errors, p-values, and odds ratios
    # MNLogit produces J-1 sets of coefficients (excluding reference)
    classes_modeled = sorted([c for c in y.unique() if c != reference_class])

    results_dict = {
        "n_obs": len(X),
        "n_predictors": X.shape[1],
        "reference_class": reference_class,
        "reference_class_name": CLASS_NAMES[reference_class],
        "pseudo_r2": round(result.prsquared, 4),
        "log_likelihood": round(result.llf, 1),
        "aic": round(result.aic, 1),
        "bic": round(result.bic, 1),
        "converged": bool(result.mle_retvals["converged"]),
        "classes_modeled": [int(c) for c in classes_modeled],
        "class_names": CLASS_NAMES,
        "coefficients": {},
    }

    for i, cls_id in enumerate(classes_modeled):
        cls_name = CLASS_NAMES[cls_id]
        coefs = result.params.iloc[:, i]
        se = result.bse.iloc[:, i]
        pvals = result.pvalues.iloc[:, i]

        # Relative risk ratios (exponentiated coefficients)
        rrr = np.exp(coefs)

        cls_results = {}
        for var_name in X_const.columns:
            idx = list(X_const.columns).index(var_name)
            cls_results[var_name] = {
                "coef": round(float(coefs.iloc[idx]), 4),
                "se": round(float(se.iloc[idx]), 4),
                "rrr": round(float(rrr.iloc[idx]), 4),
                "p_value": round(float(pvals.iloc[idx]), 6),
                "significant": bool(pvals.iloc[idx] < 0.05),
            }

        results_dict["coefficients"][f"class_{cls_id}_{cls_name}"] = cls_results

    return results_dict, result


def plot_relative_risk_ratios(
    results_dict: dict,
    top_n: int = 15,
) -> None:
    """Plot relative risk ratios for each class vs reference.

    Args:
        results_dict: Results from fit_multinomial_logit.
        top_n: Number of top predictors to show per class.
    """
    classes = results_dict["classes_modeled"]
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 8), sharey=False)
    if n_classes == 1:
        axes = [axes]

    colors = ["#E74C3C", "#2196F3", "#FF9800"]

    for ax_idx, cls_id in enumerate(classes):
        ax = axes[ax_idx]
        cls_name = CLASS_NAMES[cls_id]
        key = f"class_{cls_id}_{cls_name}"
        coef_data = results_dict["coefficients"][key]

        # Filter significant predictors and sort by |RRR - 1|
        sig_vars = [
            (var, d["rrr"], d["coef"], d["se"], d["p_value"])
            for var, d in coef_data.items()
            if d["significant"] and var != "const"
        ]
        sig_vars.sort(key=lambda x: abs(x[1] - 1), reverse=True)
        sig_vars = sig_vars[:top_n]

        if not sig_vars:
            ax.text(0.5, 0.5, "No significant predictors",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"vs {CLASS_NAMES[REFERENCE_CLASS]}: {cls_name}")
            continue

        var_names = [v[0] for v in sig_vars]
        rrrs = [v[1] for v in sig_vars]
        coefs = [v[2] for v in sig_vars]
        ses = [v[3] for v in sig_vars]

        # Compute 95% CI for RRR
        ci_low = [np.exp(c - 1.96 * s) for c, s in zip(coefs, ses)]
        ci_high = [np.exp(c + 1.96 * s) for c, s in zip(coefs, ses)]

        y_pos = range(len(var_names))
        color = colors[ax_idx % len(colors)]

        ax.barh(y_pos, rrrs, color=color, alpha=0.7, edgecolor="black",
                linewidth=0.5, height=0.6)
        ax.errorbar(rrrs, y_pos, xerr=[
            [r - ci_l for r, ci_l in zip(rrrs, ci_low)],
            [ci_h - r for r, ci_h in zip(rrrs, ci_high)]
        ], fmt="none", color="black", capsize=3)

        ax.axvline(x=1, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_names, fontsize=9)
        ax.set_xlabel("Relative Risk Ratio (vs Safe Rider)")
        ax.set_title(f"{cls_name}\n(Class {cls_id})", fontsize=12, fontweight="bold")
        ax.invert_yaxis()

    plt.suptitle("Multinomial Logistic: Predictors of Rider Type Membership",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "multinomial_coefficients.png", dpi=FIG_DPI,
                bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "multinomial_coefficients.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {FIGURES_DIR / 'multinomial_coefficients.png'}")


def print_key_findings(results_dict: dict) -> None:
    """Print key findings from the multinomial logistic regression.

    Args:
        results_dict: Results from fit_multinomial_logit.
    """
    print("\n" + "=" * 70)
    print("KEY FINDINGS: Multinomial Logistic Regression")
    print("=" * 70)
    print(f"Reference class: {CLASS_NAMES[REFERENCE_CLASS]} (Class {REFERENCE_CLASS})")
    print(f"Pseudo R2: {results_dict['pseudo_r2']:.4f}")
    print(f"N: {results_dict['n_obs']:,}")

    for cls_id in results_dict["classes_modeled"]:
        cls_name = CLASS_NAMES[cls_id]
        key = f"class_{cls_id}_{cls_name}"
        coef_data = results_dict["coefficients"][key]

        print(f"\n--- {cls_name} (Class {cls_id}) vs Safe Rider ---")
        # Top 10 significant by |RRR - 1|
        sig = [
            (var, d["rrr"], d["p_value"])
            for var, d in coef_data.items()
            if d["significant"] and var != "const"
        ]
        sig.sort(key=lambda x: abs(x[1] - 1), reverse=True)

        print(f"  {'Variable':<30} {'RRR':>8} {'p-value':>10}")
        print(f"  {'-'*50}")
        for var, rrr, pval in sig[:10]:
            pstr = f"{pval:.4f}" if pval >= 0.0001 else "<0.0001"
            print(f"  {var:<30} {rrr:>8.3f} {pstr:>10}")


def main() -> None:
    """Run multinomial logistic regression for GMM class membership."""
    print("=" * 70)
    print("Task 5.6: Multinomial Logistic Regression for Rider Type Membership")
    print("=" * 70)

    # Step 1: Load data
    df = load_user_data()

    # Step 2: Prepare predictors
    X, df_model = prepare_predictors(df)

    # Step 3: Fit model
    y = df_model["gmm_class"].astype(int)
    results_dict, model_result = fit_multinomial_logit(X, y)

    # Step 4: Print key findings
    print_key_findings(results_dict)

    # Step 5: Save results
    output_path = MODELING_DIR / "multinomial_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert CLASS_NAMES keys to strings for JSON
        results_dict["class_names"] = {str(k): v for k, v in CLASS_NAMES.items()}
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Step 6: Plot coefficients
    print("\nCreating coefficient plot...")
    plot_relative_risk_ratios(results_dict)

    # Step 7: Prediction accuracy
    print("\nPrediction accuracy check...")
    X_const = sm.add_constant(X)
    y_pred = model_result.predict(X_const).idxmax(axis=1)
    accuracy = (y_pred == y).mean()
    print(f"  Overall accuracy: {accuracy:.1%}")

    # Per-class accuracy
    for cls_id in sorted(CLASS_NAMES.keys()):
        mask = y == cls_id
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == cls_id).mean()
            print(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): {cls_acc:.1%} "
                  f"({mask.sum():,} users)")


if __name__ == "__main__":
    main()
