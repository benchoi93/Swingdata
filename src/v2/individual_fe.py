"""
Task 7B.2: Individual fixed effects models for 6 primary outcomes.

Uses pyfixest for high-dimensional FE estimation via demeaning (Mundlak).
User FE absorbs all time-invariant traits (age, risk preference, skill),
giving within-user causal interpretation of mode effects.

Specification:
  outcome ~ mode_STD + mode_ECO + C(time_of_day) + is_weekend
            + is_major_road + is_cycling_infra + log_distance | user_id

Reference categories: TUB (mode), midday (time_of_day), weekday.
age_group absorbed by user FE (time-invariant).

Output: data_parquet/v2/phase7b_individual_fe.json
Figure: figures/v2/fig_fe_vs_gee.pdf
"""

import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, FIG_DPI, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
FIG_DIR = FIGURES_DIR / "v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRIP_PATH = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

# Outcomes: 4 continuous (feols) + 2 count (fepois) + 1 manipulation check
CONTINUOUS_OUTCOMES = {
    "speed_cv": "Speed CV",
    "cruise_fraction": "Cruise fraction",
    "zero_speed_fraction": "Zero-speed fraction",
}
COUNT_OUTCOMES = {
    "harsh_accel_count": "Harsh accel (count)",
    "harsh_decel_count": "Harsh decel (count)",
}
MANIPULATION_CHECK = {
    "mean_speed": "Mean speed (km/h)",
}

ALL_OUTCOMES = {**COUNT_OUTCOMES, **CONTINUOUS_OUTCOMES, **MANIPULATION_CHECK}

SAMPLE_SIZE = 2_000_000


def load_sample(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load and prepare sample for FE estimation."""
    print(f"  Sampling {SAMPLE_SIZE:,} trips...")
    t0 = time.time()

    df = con.execute(f"""
        SELECT
            user_id, mode, time_of_day, day_type,
            is_major_road, is_cycling_infra, distance,
            harsh_accel_count, harsh_decel_count,
            speed_cv, cruise_fraction, zero_speed_fraction,
            mean_speed
        FROM read_parquet('{TRIP_PATH}')
        USING SAMPLE {SAMPLE_SIZE} ROWS (reservoir)
    """).fetchdf()

    print(f"  Loaded {len(df):,} trips, {df['user_id'].nunique():,} users ({time.time()-t0:.1f}s)")

    # Prepare variables
    df["log_distance"] = np.log1p(df["distance"])

    # Binary mode dummies (reference: TUB)
    df["mode_STD"] = (df["mode"] == "STD").astype(np.int8)
    df["mode_ECO"] = (df["mode"] == "ECO").astype(np.int8)

    # Time of day dummies (reference: midday)
    for tod in ["morning_rush", "afternoon_rush", "evening", "night"]:
        df[f"tod_{tod}"] = (df["time_of_day"] == tod).astype(np.int8)

    df["is_weekend"] = (df["day_type"] == "weekend").astype(np.int8)
    df["is_major_road"] = df["is_major_road"].astype(np.int8)
    df["is_cycling_infra"] = df["is_cycling_infra"].astype(np.int8)

    # Filter users with only 1 trip (singleton FE — no within-variation)
    user_counts = df["user_id"].value_counts()
    multi_trip_users = user_counts[user_counts >= 2].index
    n_before = len(df)
    df = df[df["user_id"].isin(multi_trip_users)].copy()
    print(f"  After singleton removal: {len(df):,} trips ({n_before - len(df):,} dropped)")

    # Filter users with only 1 mode (no within-user mode variation)
    user_modes = df.groupby("user_id")["mode"].nunique()
    multi_mode_users = user_modes[user_modes >= 2].index
    n_multi_mode = len(multi_mode_users)
    print(f"  Users with 2+ modes: {n_multi_mode:,} ({100*n_multi_mode/df['user_id'].nunique():.1f}%)")
    print(f"  (FE still valid for all users — mode variation not required for other covariates)")

    return df


def run_fe_models(df: pd.DataFrame) -> dict:
    """Run individual FE models for all outcomes."""
    results = {}

    # Common RHS (without intercept — absorbed by FE)
    rhs = "mode_STD + mode_ECO + tod_morning_rush + tod_afternoon_rush + tod_evening + tod_night + is_weekend + is_major_road + is_cycling_infra + log_distance"

    # Continuous outcomes: feols with user FE
    for outcome, label in {**CONTINUOUS_OUTCOMES, **MANIPULATION_CHECK}.items():
        print(f"\n  FE OLS: {label} ({outcome})")
        t0 = time.time()

        fml = f"{outcome} ~ {rhs} | user_id"
        try:
            fit = pf.feols(
                fml,
                data=df,
                vcov={"CRV1": "user_id"},
                lean=True,
            )
            results[outcome] = _extract_results(fit, label, "feols")
            _print_mode_results(results[outcome])
            print(f"  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[outcome] = {"error": str(e)}

    # Count outcomes: fepois with user FE
    for outcome, label in COUNT_OUTCOMES.items():
        print(f"\n  FE Poisson: {label} ({outcome})")
        t0 = time.time()

        fml = f"{outcome} ~ {rhs} | user_id"
        try:
            fit = pf.fepois(
                fml,
                data=df,
                vcov={"CRV1": "user_id"},
                lean=True,
            )
            results[outcome] = _extract_results(fit, label, "fepois")
            _print_mode_results(results[outcome])
            print(f"  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[outcome] = {"error": str(e)}

    return results


def _extract_results(fit, label: str, model_type: str) -> dict:
    """Extract coefficients, SEs, p-values from pyfixest fit."""
    coefs = fit.coef()
    ses = fit.se()
    pvals = fit.pvalue()
    conf = fit.confint()

    result = {
        "label": label,
        "model_type": model_type,
        "n_obs": int(fit._N),
        "coefficients": {},
    }

    # Try to get R2 (may not exist for fepois)
    try:
        result["r2"] = round(float(fit._r2), 4)
        result["r2_within"] = round(float(fit._r2_within), 4)
    except (AttributeError, TypeError):
        pass

    for var in coefs.index:
        result["coefficients"][var] = {
            "coef": round(float(coefs[var]), 6),
            "se": round(float(ses[var]), 6),
            "p": float(pvals[var]),
            "ci_lower": round(float(conf.iloc[conf.index.get_loc(var), 0]), 6),
            "ci_upper": round(float(conf.iloc[conf.index.get_loc(var), 1]), 6),
        }

    return result


def _print_mode_results(result: dict) -> None:
    """Print mode coefficients nicely."""
    if "error" in result:
        return
    for var in ["mode_STD", "mode_ECO"]:
        if var in result["coefficients"]:
            c = result["coefficients"][var]
            sig = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else ""
            print(f"    {var}: {c['coef']:.4f} [{c['ci_lower']:.4f}, {c['ci_upper']:.4f}] p={c['p']:.4g} {sig}")
    print(f"    N={result['n_obs']:,}", end="")
    if "r2" in result:
        print(f", R2={result['r2']:.4f}", end="")
    print()


def load_gee_results() -> dict:
    """Load existing GEE results for comparison."""
    gee_path = V2_DIR / "phase1_cross_sectional.json"
    if not gee_path.exists():
        print("  WARNING: GEE results not found at", gee_path)
        return {}

    with open(gee_path) as f:
        data = json.load(f)

    return data.get("gee_results", {})


def compare_fe_gee(fe_results: dict, gee_results: dict) -> dict:
    """Compare FE vs GEE mode coefficients side by side."""
    print("\n" + "=" * 60)
    print("Task 7B.3: FE vs GEE Comparison")
    print("=" * 60)

    comparison = {}

    # Map GEE keys to outcomes
    # GEE results have keys like "harsh_accel_count_STD vs TUB"
    print(f"\n  {'Outcome':<20s} {'Mode':<10s} {'GEE coef':>10s} {'FE coef':>10s} {'Diff':>8s} {'Direction':>10s}")
    print("  " + "-" * 70)

    for outcome, label in ALL_OUTCOMES.items():
        if outcome not in fe_results or "error" in fe_results[outcome]:
            continue

        fe_coefs = fe_results[outcome]["coefficients"]

        for mode_var, gee_key_suffix in [("mode_STD", "STD vs TUB"), ("mode_ECO", "ECO vs TUB")]:
            gee_key = f"{outcome}_{gee_key_suffix}"

            if mode_var in fe_coefs and gee_key in gee_results:
                fe_c = fe_coefs[mode_var]["coef"]
                gee_c = gee_results[gee_key]["coef"]
                diff = fe_c - gee_c

                # Check if same direction
                same_dir = "same" if (fe_c * gee_c > 0) else "FLIPPED"

                mode_label = mode_var.replace("mode_", "")
                short_label = label.split("(")[0].strip()[:18]
                print(f"  {short_label:<20s} {mode_label:<10s} {gee_c:>10.4f} {fe_c:>10.4f} {diff:>+8.4f} {same_dir:>10s}")

                comparison[f"{outcome}_{mode_label}"] = {
                    "gee_coef": gee_c,
                    "fe_coef": fe_c,
                    "difference": round(diff, 6),
                    "same_direction": same_dir == "same",
                    "attenuation_pct": round(100 * abs(diff) / abs(gee_c), 1) if gee_c != 0 else None,
                }

    return comparison


def make_comparison_figure(fe_results: dict, gee_results: dict) -> None:
    """Create side-by-side FE vs GEE coefficient plot."""
    rows = []
    for outcome, label in ALL_OUTCOMES.items():
        if outcome not in fe_results or "error" in fe_results[outcome]:
            continue
        fe_coefs = fe_results[outcome]["coefficients"]
        for mode_var, gee_suffix in [("mode_STD", "STD vs TUB"), ("mode_ECO", "ECO vs TUB")]:
            gee_key = f"{outcome}_{gee_suffix}"
            if mode_var in fe_coefs and gee_key in gee_results:
                mode_label = mode_var.replace("mode_", "")
                short_label = label.split("(")[0].strip()
                rows.append({
                    "outcome": short_label,
                    "mode": mode_label,
                    "GEE": gee_results[gee_key]["coef"],
                    "GEE_lo": gee_results[gee_key]["ci_lower"],
                    "GEE_hi": gee_results[gee_key]["ci_upper"],
                    "FE": fe_coefs[mode_var]["coef"],
                    "FE_lo": fe_coefs[mode_var]["ci_lower"],
                    "FE_hi": fe_coefs[mode_var]["ci_upper"],
                })

    if not rows:
        print("  No comparison data for figure.")
        return

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_idx, mode in enumerate(["STD", "ECO"]):
        ax = axes[ax_idx]
        sub = df[df["mode"] == mode].iloc[::-1]
        y_pos = np.arange(len(sub))
        h = 0.35

        # GEE bars
        ax.barh(y_pos + h/2, sub["GEE"], height=h, color="#3498db", alpha=0.7,
                xerr=[sub["GEE"] - sub["GEE_lo"], sub["GEE_hi"] - sub["GEE"]],
                capsize=3, label="GEE (exchangeable)")
        # FE bars
        ax.barh(y_pos - h/2, sub["FE"], height=h, color="#e74c3c", alpha=0.7,
                xerr=[sub["FE"] - sub["FE_lo"], sub["FE_hi"] - sub["FE"]],
                capsize=3, label="Individual FE")

        ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["outcome"])
        ax.set_xlabel("Coefficient (vs TUB reference)")
        ax.set_title(f"{mode} vs TUB", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Individual Fixed Effects vs GEE: Mode Coefficients", fontsize=14, y=1.02)
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig_fe_vs_gee.{fmt}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {FIG_DIR / 'fig_fe_vs_gee.pdf'}")


def main() -> None:
    print("=" * 60)
    print("Phase 7B: Individual Fixed Effects Models")
    print("=" * 60)

    t_start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 8")

    # Load sample
    df = load_sample(con)
    con.close()

    # Run FE models
    fe_results = run_fe_models(df)

    # Load GEE for comparison
    gee_results = load_gee_results()

    # Compare
    comparison = {}
    if gee_results:
        comparison = compare_fe_gee(fe_results, gee_results)
        make_comparison_figure(fe_results, gee_results)

    elapsed = time.time() - t_start

    # Save results
    report = {
        "task": "7B.2",
        "description": "Individual FE models with pyfixest",
        "sample_size": SAMPLE_SIZE,
        "fe_results": fe_results,
        "fe_vs_gee_comparison": comparison,
        "time_s": round(elapsed, 1),
    }

    out_path = V2_DIR / "phase7b_individual_fe.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Phase 7B complete in {elapsed:.0f}s")
    print(f"  Results: {out_path}")
    print(f"  Figure: {FIG_DIR / 'fig_fe_vs_gee.pdf'}")


if __name__ == "__main__":
    main()
