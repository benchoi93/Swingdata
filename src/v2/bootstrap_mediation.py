"""
Bootstrap confidence intervals for TUB mediation percentages.

Resamples users (not trips) with replacement to preserve within-user
correlation structure. Computes mediation % for each outcome across
1000 bootstrap iterations, reports 95% percentile CIs.

Also extracts Schoenfeld residual correlation direction for Cox PH models
to determine whether TUB effect intensifies or attenuates over the trip sequence.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, RANDOM_SEED

V2_DIR = DATA_DIR / "v2"
TRIP_PATH = str(V2_DIR / "trip_modeling.parquet").replace("\\", "/")

OUTCOMES = [
    "mean_speed", "harsh_accel_count", "harsh_decel_count",
    "speed_cv", "cruise_fraction", "zero_speed_fraction",
]

N_BOOTSTRAP = 500
SAMPLE_USERS = 50000


def load_experience_data() -> pd.DataFrame:
    """Load trip sequences for 200K users with experience bins."""
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 8")
    con.execute(f"SELECT setseed({RANDOM_SEED / 100})")

    print("  Loading trip sequences (200K users)...")
    df = con.execute(f"""
        WITH ranked AS (
            SELECT user_id, mode, month_year,
                mean_speed, harsh_accel_count, harsh_decel_count,
                speed_cv, cruise_fraction, zero_speed_fraction,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY month_year, route_id) AS trip_rank
            FROM read_parquet('{TRIP_PATH}')
        ),
        user_sample AS (
            SELECT DISTINCT user_id FROM ranked
            USING SAMPLE {SAMPLE_USERS} ROWS (reservoir)
        )
        SELECT r.*
        FROM ranked r
        INNER JOIN user_sample u ON r.user_id = u.user_id
        WHERE r.trip_rank <= 200
    """).fetchdf()
    con.close()

    print(f"  Loaded: {len(df):,} trips from {df['user_id'].nunique():,} users")

    # Experience bins
    exp_bins = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    exp_labels = ["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "51-100", "101-200"]
    df["exp_bin"] = pd.cut(
        df["trip_rank"], bins=[0] + exp_bins, labels=exp_labels, right=True
    )
    df = df.dropna(subset=["exp_bin"])

    return df


def compute_mediation(df: pd.DataFrame) -> dict[str, float]:
    """Compute mediation % for each outcome from a (possibly resampled) DataFrame."""
    results = {}
    for outcome in OUTCOMES:
        all_by_exp = df.groupby("exp_bin", observed=True)[outcome].mean()
        std_eco = df[df["mode"].isin(["STD", "ECO"])]
        std_eco_by_exp = std_eco.groupby("exp_bin", observed=True)[outcome].mean()

        total_range = all_by_exp.max() - all_by_exp.min()
        direct_range = std_eco_by_exp.max() - std_eco_by_exp.min()

        if total_range != 0:
            mediation_pct = (total_range - direct_range) / total_range * 100
        else:
            mediation_pct = 0.0

        results[outcome] = float(mediation_pct)
    return results


def bootstrap_mediation(df: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> dict:
    """Bootstrap CIs for mediation percentages by resampling users.

    Pre-computes user-level summary stats to avoid expensive DataFrame
    operations inside the loop.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    user_ids = df["user_id"].unique()
    n_users = len(user_ids)

    print(f"\n  Running {n_boot} bootstrap iterations ({n_users:,} users)...")

    # Point estimate
    point = compute_mediation(df)
    print("\n  Point estimates:")
    for outcome in OUTCOMES:
        print(f"    {outcome}: {point[outcome]:.1f}%")

    # Pre-compute user-level aggregates per (user, exp_bin) and (user, exp_bin, mode)
    # This avoids re-grouping millions of rows in each iteration
    print("  Pre-computing user-level aggregates...")
    user_exp = df.groupby(["user_id", "exp_bin"], observed=True)[OUTCOMES].agg(["sum", "count"])
    user_exp.columns = [f"{o}_{s}" for o, s in user_exp.columns]
    user_exp = user_exp.reset_index()

    # STD/ECO only
    std_eco_mask = df["mode"].isin(["STD", "ECO"])
    user_exp_se = df[std_eco_mask].groupby(["user_id", "exp_bin"], observed=True)[OUTCOMES].agg(["sum", "count"])
    user_exp_se.columns = [f"{o}_{s}" for o, s in user_exp_se.columns]
    user_exp_se = user_exp_se.reset_index()

    # Create user_id -> integer index for fast lookup
    uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    user_exp["uid_idx"] = user_exp["user_id"].map(uid_to_idx)
    user_exp_se["uid_idx"] = user_exp_se["user_id"].map(uid_to_idx)

    # Bootstrap
    boot_results = {o: [] for o in OUTCOMES}

    t0 = time.time()
    for i in range(n_boot):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_boot - i - 1) / rate
            print(f"    Iteration {i + 1}/{n_boot} ({rate:.1f} iter/s, ETA {eta:.0f}s)")

        # Resample user indices with replacement
        boot_idx = rng.randint(0, n_users, size=n_users)

        # Count how many times each user is sampled
        unique_idx, counts = np.unique(boot_idx, return_counts=True)
        weight_map = dict(zip(unique_idx, counts))

        # Apply weights to pre-aggregated data
        boot_all = user_exp[user_exp["uid_idx"].isin(unique_idx)].copy()
        boot_all["_w"] = boot_all["uid_idx"].map(weight_map)

        boot_se = user_exp_se[user_exp_se["uid_idx"].isin(unique_idx)].copy()
        boot_se["_w"] = boot_se["uid_idx"].map(weight_map)

        med = {}
        for outcome in OUTCOMES:
            sum_col = f"{outcome}_sum"
            cnt_col = f"{outcome}_count"

            # Weighted group means
            all_grp = boot_all.groupby("exp_bin", observed=True).apply(
                lambda g: (g[sum_col] * g["_w"]).sum() / (g[cnt_col] * g["_w"]).sum(),
                include_groups=False,
            )
            se_grp = boot_se.groupby("exp_bin", observed=True).apply(
                lambda g: (g[sum_col] * g["_w"]).sum() / (g[cnt_col] * g["_w"]).sum(),
                include_groups=False,
            )

            total_range = all_grp.max() - all_grp.min()
            direct_range = se_grp.max() - se_grp.min()

            if total_range != 0:
                med[outcome] = float((total_range - direct_range) / total_range * 100)
            else:
                med[outcome] = 0.0

        for outcome in OUTCOMES:
            boot_results[outcome].append(med[outcome])

    elapsed = time.time() - t0
    print(f"\n  Bootstrap completed in {elapsed:.0f}s ({elapsed / n_boot:.2f}s/iter)")

    # Compute CIs
    results = {}
    print(f"\n  {'Outcome':<25s} {'Point':<10s} {'95% CI':<25s} {'SE':<10s}")
    print(f"  {'-' * 70}")
    for outcome in OUTCOMES:
        vals = np.array(boot_results[outcome])
        ci_lo = float(np.percentile(vals, 2.5))
        ci_hi = float(np.percentile(vals, 97.5))
        se = float(np.std(vals))
        results[outcome] = {
            "point": round(point[outcome], 1),
            "ci_lower": round(ci_lo, 1),
            "ci_upper": round(ci_hi, 1),
            "se": round(se, 1),
            "bootstrap_dist": [round(v, 2) for v in vals.tolist()],
        }
        print(f"  {outcome:<25s} {point[outcome]:<10.1f} [{ci_lo:.1f}, {ci_hi:.1f}]{'':<10s} {se:<10.1f}")

    return results


def schoenfeld_direction() -> dict:
    """Compute Schoenfeld residual correlation direction for each Cox PH model.

    Returns sign of correlation between Schoenfeld residuals and time for each
    covariate x endpoint combination. Positive = HR increases over trip sequence.
    """
    from lifelines import CoxPHFitter
    from scipy.stats import spearmanr

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads TO 8")
    con.execute(f"SELECT setseed({RANDOM_SEED / 100})")

    # Get P75 CV
    p75_cv = con.execute(f"""
        SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY speed_cv) AS p75
        FROM read_parquet('{TRIP_PATH}')
        WHERE speed_cv IS NOT NULL
    """).fetchdf()["p75"].iloc[0]

    # Load data
    surv_df = con.execute(f"""
        WITH user_sample AS (
            SELECT DISTINCT user_id
            FROM read_parquet('{TRIP_PATH}')
            USING SAMPLE {SAMPLE_USERS} ROWS (reservoir)
        ),
        ranked AS (
            SELECT
                t.user_id, t.mode,
                t.harsh_accel_count, t.harsh_decel_count,
                t.speed_cv, t.max_speed_from_profile,
                ROW_NUMBER() OVER (PARTITION BY t.user_id ORDER BY t.month_year, t.route_id) AS trip_rank
            FROM read_parquet('{TRIP_PATH}') t
            INNER JOIN user_sample u ON t.user_id = u.user_id
        )
        SELECT * FROM ranked WHERE trip_rank <= 500
    """).fetchdf()
    con.close()

    # User-level covariates
    user_max = surv_df.groupby("user_id").agg(
        max_trip=("trip_rank", "max"),
        ever_tub=("mode", lambda x: int((x == "TUB").any())),
        tub_frac=("mode", lambda x: (x == "TUB").mean()),
    ).reset_index()

    # Events
    events = {}
    harsh_trips = surv_df[(surv_df["harsh_accel_count"] > 0) | (surv_df["harsh_decel_count"] > 0)]
    events["first_harsh"] = harsh_trips.groupby("user_id")["trip_rank"].min().reset_index().rename(
        columns={"trip_rank": "event_trip"}
    )
    high_cv = surv_df[surv_df["speed_cv"] > p75_cv]
    events["first_high_cv"] = high_cv.groupby("user_id")["trip_rank"].min().reset_index().rename(
        columns={"trip_rank": "event_trip"}
    )
    speeding = surv_df[surv_df["max_speed_from_profile"] > 25]
    events["first_speeding"] = speeding.groupby("user_id")["trip_rank"].min().reset_index().rename(
        columns={"trip_rank": "event_trip"}
    )

    results = {}
    for event_name, event_df in events.items():
        print(f"\n  Schoenfeld direction: {event_name}")

        surv = user_max.merge(event_df, on="user_id", how="left")
        surv["event"] = surv["event_trip"].notna().astype(int)
        surv["duration"] = surv["event_trip"].fillna(surv["max_trip"])
        surv = surv[(surv["max_trip"] >= 2) & (surv["duration"] > 0)]
        cox_df = surv[["duration", "event", "ever_tub", "tub_frac"]].dropna().copy()

        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="duration", event_col="event")

        # Compute Schoenfeld residuals
        try:
            schoenfeld = cph.compute_residuals(cox_df, kind="schoenfeld")
            event_results = {}
            for covar in ["ever_tub", "tub_frac"]:
                if covar in schoenfeld.columns:
                    resid = schoenfeld[covar].values
                    times = schoenfeld.index.values  # event times
                    rho, p = spearmanr(times, resid)
                    direction = "increasing" if rho > 0 else "decreasing"
                    event_results[covar] = {
                        "spearman_rho": round(float(rho), 4),
                        "p_value": float(p),
                        "direction": direction,
                        "interpretation": (
                            f"HR {'increases' if rho > 0 else 'decreases'} "
                            f"over the trip sequence (rho={rho:.3f})"
                        ),
                    }
                    print(f"    {covar}: rho={rho:.4f} (p={p:.2g}) -> HR {direction}")
            results[event_name] = event_results
        except Exception as e:
            print(f"    Error computing residuals: {e}")
            results[event_name] = {"error": str(e)}

    return results


def main():
    print("Bootstrap Mediation CIs + Schoenfeld Direction")
    print("=" * 60)
    t_start = time.time()

    # Part 1: Bootstrap mediation CIs
    print("\nPart 1: Loading experience data...")
    df = load_experience_data()

    print("\nPart 2: Bootstrap mediation CIs...")
    mediation = bootstrap_mediation(df, n_boot=N_BOOTSTRAP)

    # Part 2: Schoenfeld direction
    print("\n\nPart 3: Schoenfeld residual direction...")
    schoenfeld = schoenfeld_direction()

    elapsed = time.time() - t_start

    # Save combined results
    report = {
        "description": f"Bootstrap mediation CIs ({N_BOOTSTRAP} iterations, user-level resampling, "
                       f"{SAMPLE_USERS:,} users) + Schoenfeld residual direction for Cox PH models",
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "mediation_ci": {
            outcome: {k: v for k, v in vals.items() if k != "bootstrap_dist"}
            for outcome, vals in mediation.items()
        },
        "schoenfeld_direction": schoenfeld,
        "time_s": round(elapsed, 1),
    }

    out_path = V2_DIR / "bootstrap_mediation.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out_path}")
    print(f"Completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
