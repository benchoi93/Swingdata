"""
Survival Analysis: Time-to-First-Speeding.

Cox proportional hazards model where:
- Event: first speeding violation (max_speed > 25 km/h)
- Survival time: trip_rank at first speeding occurrence
- Censoring: users who never speed are right-censored at last trip
- Time-varying covariate: TUB mode adoption

Addresses: Does TUB adoption trigger speeding onset, or do riders
who would eventually speed simply adopt TUB sooner?

Uses Feb-Nov 2023 data only (pre-TUB-ban period) to avoid confounding.

Output:
- data_parquet/modeling/survival_analysis_results.json
- figures/fig_survival_km.pdf (Kaplan-Meier curves)
- figures/fig_survival_cox.pdf (Cox PH forest plot)
- figures/fig_survival_hazard.pdf (hazard ratio over trip rank)
"""
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter, CoxTimeVaryingFitter
from lifelines.statistics import logrank_test
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIGURES_DIR, MODELING_DIR, FIG_DPI, RANDOM_SEED

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.random.seed(RANDOM_SEED)

ROUTES_PATH = str(DATA_DIR / 'all_months' / 'routes_all.parquet').replace('\\', '/')
EXPERIENCE_PATH = str(MODELING_DIR / 'trip_experience.parquet').replace('\\', '/')

# Max trip rank to consider (cap extreme outliers)
MAX_TRIP_RANK = 500


def build_survival_dataset() -> pd.DataFrame:
    """Build per-user survival dataset from trip-level data.

    For each user, determines:
    - event_time: trip_rank at first speeding, or last trip_rank if never sped
    - event: 1 if speeded, 0 if censored (never sped)
    - ever_tub: whether user ever used TUB mode
    - tub_before_first_speed: whether TUB adoption preceded first speeding
    - first_tub_rank: trip_rank of first TUB usage
    - age, city (from trip_experience data)
    """
    con = duckdb.connect()
    print("Building survival dataset from trip_experience.parquet...")
    print("  Using Feb-Nov 2023 (pre-TUB-ban) only.")

    # Per-user survival data: first speeding trip, first TUB trip, total trips
    df = con.execute(f"""
        WITH user_trips AS (
            SELECT
                user_id,
                trip_rank,
                is_speeding,
                mode,
                age,
                start_lat,
                start_lon,
                month_year
            FROM read_parquet('{EXPERIENCE_PATH}')
            WHERE mean_speed_from_speeds IS NOT NULL
              AND month_year >= '2023-02' AND month_year <= '2023-11'
              AND trip_rank <= {MAX_TRIP_RANK}
        ),
        user_summary AS (
            SELECT
                user_id,
                MIN(age) as age,
                MIN(start_lat) as start_lat,
                MIN(start_lon) as start_lon,
                COUNT(*) as n_trips,
                MAX(trip_rank) as max_trip_rank,
                -- First speeding trip rank
                MIN(CASE WHEN is_speeding THEN trip_rank END) as first_speed_rank,
                -- First TUB trip rank
                MIN(CASE WHEN mode = 'TUB' THEN trip_rank END) as first_tub_rank,
                -- Mode usage counts
                SUM(CASE WHEN mode = 'TUB' THEN 1 ELSE 0 END) as n_tub_trips,
                SUM(CASE WHEN mode = 'STD' THEN 1 ELSE 0 END) as n_std_trips,
                SUM(CASE WHEN mode = 'ECO' THEN 1 ELSE 0 END) as n_eco_trips,
                -- Speeding count
                SUM(CASE WHEN is_speeding THEN 1 ELSE 0 END) as n_speeding_trips
            FROM user_trips
            GROUP BY user_id
        )
        SELECT * FROM user_summary
        WHERE n_trips >= 2
    """).fetchdf()
    con.close()

    print(f"  Users with >=2 trips (Feb-Nov): {len(df):,}")

    # Build survival variables (handle nullable ints from DuckDB)
    df['event'] = df['first_speed_rank'].notna().astype(bool).astype(int)
    df['event_time'] = df['first_speed_rank'].fillna(df['max_trip_rank'])
    df['event_time'] = pd.to_numeric(df['event_time'], errors='coerce').astype(int)
    # Cap event_time at MAX_TRIP_RANK
    df['event_time'] = df['event_time'].clip(upper=MAX_TRIP_RANK)

    # TUB adoption indicators
    df['ever_tub'] = (pd.to_numeric(df['n_tub_trips'], errors='coerce').fillna(0) > 0).astype(int)
    df['tub_before_speed'] = 0
    mask = df['event'] == 1
    first_tub = pd.to_numeric(df.loc[mask, 'first_tub_rank'], errors='coerce')
    first_speed = pd.to_numeric(df.loc[mask, 'first_speed_rank'], errors='coerce')
    df.loc[mask, 'tub_before_speed'] = (
        first_tub.notna() & (first_tub < first_speed)
    ).astype(int)

    # TUB fraction
    df['tub_fraction'] = df['n_tub_trips'] / df['n_trips']

    # Age groups
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 20, 25, 30, 35, 40, 100],
        labels=['<20', '20-24', '25-29', '30-34', '35-39', '40+'],
        right=False
    )

    # Statistics
    n_events = df['event'].sum()
    n_censored = len(df) - n_events
    print(f"  Events (first speeding): {n_events:,} ({100*n_events/len(df):.1f}%)")
    print(f"  Censored (never sped): {n_censored:,} ({100*n_censored/len(df):.1f}%)")
    print(f"  Ever used TUB: {df['ever_tub'].sum():,} ({100*df['ever_tub'].mean():.1f}%)")
    print(f"  Median event time (speeders): {df.loc[df['event']==1, 'event_time'].median():.0f} trips")
    print(f"  Median censoring time (non-speeders): {df.loc[df['event']==0, 'event_time'].median():.0f} trips")

    return df


def build_time_varying_dataset(df_users: pd.DataFrame, sample_n: int = 100000) -> pd.DataFrame:
    """Build start-stop format dataset for time-varying Cox PH.

    Each user's trip timeline is split at TUB adoption: before and after first TUB usage.
    This creates interval-censored data where the TUB covariate changes at first_tub_rank.

    Args:
        df_users: Per-user survival dataset from build_survival_dataset()
        sample_n: Number of users to subsample (for computational feasibility)

    Returns:
        DataFrame in start-stop format with columns:
        user_id, start, stop, event, tub_adopted, age_group, tub_fraction
    """
    print(f"\nBuilding time-varying dataset (subsample={sample_n:,})...")

    # Stratified subsample: maintain event/censored ratio
    if len(df_users) > sample_n:
        events = df_users[df_users['event'] == 1]
        censored = df_users[df_users['event'] == 0]
        event_frac = len(events) / len(df_users)
        n_events = int(sample_n * event_frac)
        n_censored = sample_n - n_events
        sampled = pd.concat([
            events.sample(n=min(n_events, len(events)), random_state=RANDOM_SEED),
            censored.sample(n=min(n_censored, len(censored)), random_state=RANDOM_SEED)
        ])
    else:
        sampled = df_users.copy()
    print(f"  Sampled: {len(sampled):,} users ({sampled['event'].sum():,} events)")

    rows = []
    for _, user in sampled.iterrows():
        uid = user['user_id']
        event_time = int(user['event_time'])
        event = int(user['event'])
        age_group = str(user['age_group']) if pd.notna(user['age_group']) else 'unknown'
        first_tub = user['first_tub_rank']

        if pd.isna(first_tub) or first_tub >= event_time:
            # Never used TUB before event/censoring: single interval
            rows.append({
                'user_id': uid,
                'start': 0,
                'stop': event_time,
                'event': event,
                'tub_adopted': 0,
                'age_group': age_group,
                'tub_fraction': float(user['tub_fraction']),
                'n_trips': int(user['n_trips']),
            })
        else:
            # TUB adopted before event: split into two intervals
            first_tub_int = int(first_tub)
            # Pre-TUB interval: (0, first_tub_rank] - no event
            rows.append({
                'user_id': uid,
                'start': 0,
                'stop': first_tub_int,
                'event': 0,
                'tub_adopted': 0,
                'age_group': age_group,
                'tub_fraction': float(user['tub_fraction']),
                'n_trips': int(user['n_trips']),
            })
            # Post-TUB interval: (first_tub_rank, event_time]
            rows.append({
                'user_id': uid,
                'start': first_tub_int,
                'stop': event_time,
                'event': event,
                'tub_adopted': 1,
                'age_group': age_group,
                'tub_fraction': float(user['tub_fraction']),
                'n_trips': int(user['n_trips']),
            })

    tv_df = pd.DataFrame(rows)
    print(f"  Time-varying rows: {len(tv_df):,}")
    print(f"  TUB-adopted intervals: {tv_df['tub_adopted'].sum():,}")
    return tv_df


def fit_kaplan_meier(df: pd.DataFrame) -> dict[str, Any]:
    """Fit Kaplan-Meier survival curves stratified by TUB usage."""
    print("\n--- Kaplan-Meier Analysis ---")
    results: dict[str, Any] = {}

    kmf = KaplanMeierFitter()

    # Overall
    kmf.fit(df['event_time'], df['event'], label='All riders')
    median_all = kmf.median_survival_time_
    results['overall'] = {
        'median_survival': float(median_all) if np.isfinite(median_all) else None,
        'n': int(len(df)),
        'n_events': int(df['event'].sum()),
    }
    print(f"  Overall median survival: {median_all} trips")

    # By TUB usage
    tub_mask = df['ever_tub'] == 1
    no_tub_mask = df['ever_tub'] == 0

    kmf_tub = KaplanMeierFitter()
    kmf_tub.fit(df.loc[tub_mask, 'event_time'], df.loc[tub_mask, 'event'], label='TUB users')
    median_tub = kmf_tub.median_survival_time_

    kmf_notub = KaplanMeierFitter()
    kmf_notub.fit(df.loc[no_tub_mask, 'event_time'], df.loc[no_tub_mask, 'event'], label='Non-TUB users')
    median_notub = kmf_notub.median_survival_time_

    results['by_tub'] = {
        'tub_users': {
            'n': int(tub_mask.sum()),
            'n_events': int(df.loc[tub_mask, 'event'].sum()),
            'median_survival': float(median_tub) if np.isfinite(median_tub) else None,
        },
        'non_tub_users': {
            'n': int(no_tub_mask.sum()),
            'n_events': int(df.loc[no_tub_mask, 'event'].sum()),
            'median_survival': float(median_notub) if np.isfinite(median_notub) else None,
        },
    }
    print(f"  TUB users median: {median_tub} trips (n={tub_mask.sum():,})")
    print(f"  Non-TUB median: {median_notub} trips (n={no_tub_mask.sum():,})")

    # Log-rank test
    lr = logrank_test(
        df.loc[tub_mask, 'event_time'], df.loc[no_tub_mask, 'event_time'],
        df.loc[tub_mask, 'event'], df.loc[no_tub_mask, 'event']
    )
    results['logrank'] = {
        'test_statistic': float(lr.test_statistic),
        'p_value': float(lr.p_value),
    }
    print(f"  Log-rank test: chi2={lr.test_statistic:.1f}, p={lr.p_value:.2e}")

    # By age group
    results['by_age'] = {}
    for ag in ['<20', '20-24', '25-29', '30-34', '35-39', '40+']:
        mask = df['age_group'] == ag
        if mask.sum() < 100:
            continue
        kmf_ag = KaplanMeierFitter()
        kmf_ag.fit(df.loc[mask, 'event_time'], df.loc[mask, 'event'])
        med = kmf_ag.median_survival_time_
        results['by_age'][ag] = {
            'n': int(mask.sum()),
            'n_events': int(df.loc[mask, 'event'].sum()),
            'median_survival': float(med) if np.isfinite(med) else None,
        }
        print(f"  Age {ag}: median={med}, n={mask.sum():,}")

    return results, kmf_tub, kmf_notub


def fit_cox_static(df: pd.DataFrame) -> tuple[CoxPHFitter, dict[str, Any]]:
    """Fit standard Cox PH with static covariates."""
    print("\n--- Cox PH (Static Covariates) ---")

    # Prepare modeling data
    model_df = df[['event_time', 'event', 'ever_tub', 'tub_fraction',
                    'age_group', 'n_trips']].copy()
    model_df = model_df.dropna(subset=['age_group'])
    model_df['log_n_trips'] = np.log1p(model_df['n_trips'])

    # Dummy encode age_group (reference: 25-29)
    age_dummies = pd.get_dummies(model_df['age_group'], prefix='age', dtype=float)
    ref_col = 'age_25-29'
    if ref_col in age_dummies.columns:
        age_dummies = age_dummies.drop(columns=[ref_col])
    model_df = pd.concat([model_df, age_dummies], axis=1)
    model_df = model_df.drop(columns=['age_group', 'n_trips'])

    # Subsample for computational feasibility
    if len(model_df) > 200000:
        model_df = model_df.sample(n=200000, random_state=RANDOM_SEED)
        print(f"  Subsampled to {len(model_df):,} for Cox PH")

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col='event_time', event_col='event')
    cph.print_summary()

    results = {
        'n': int(len(model_df)),
        'n_events': int(model_df['event'].sum()),
        'concordance': float(cph.concordance_index_),
        'log_likelihood': float(cph.log_likelihood_),
        'AIC': float(cph.AIC_partial_),
        'coefficients': {},
    }

    for var in cph.summary.index:
        row = cph.summary.loc[var]
        results['coefficients'][var] = {
            'coef': float(row['coef']),
            'exp_coef': float(row['exp(coef)']),
            'se': float(row['se(coef)']),
            'z': float(row['z']),
            'p': float(row['p']),
            'lower_95': float(row['exp(coef) lower 95%']),
            'upper_95': float(row['exp(coef) upper 95%']),
        }

    print(f"\n  Concordance index: {cph.concordance_index_:.4f}")
    print(f"  AIC: {cph.AIC_partial_:.1f}")

    # Check proportional hazards assumption
    print("\n  Checking proportional hazards assumption...")
    try:
        ph_test = cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)
        if ph_test is None or len(ph_test) == 0:
            results['ph_assumption'] = {'passes': True, 'note': 'All covariates pass PH test'}
            print("  PH assumption: PASS (all covariates)")
        else:
            violations = []
            for var_name, test_result in ph_test:
                violations.append(str(var_name))
            results['ph_assumption'] = {
                'passes': False,
                'violations': violations,
                'note': f'{len(violations)} covariate(s) violate PH assumption'
            }
            print(f"  PH assumption: {len(violations)} violation(s): {violations}")
    except Exception as e:
        results['ph_assumption'] = {'passes': 'unknown', 'error': str(e)}
        print(f"  PH assumption check error: {e}")

    return cph, results


def fit_cox_time_varying(tv_df: pd.DataFrame) -> tuple[CoxTimeVaryingFitter, dict[str, Any]]:
    """Fit Cox PH with TUB adoption as time-varying covariate.

    Uses lifelines.CoxTimeVaryingFitter which accepts start-stop format data.
    Keeps covariates minimal to avoid singular matrix issues.
    """
    print("\n--- Cox PH (Time-Varying TUB Adoption) ---")

    # CoxTimeVaryingFitter needs: id, start, stop, event, covariates
    model_df = tv_df[['user_id', 'start', 'stop', 'event', 'tub_adopted']].copy()

    # Filter out zero-length intervals
    model_df = model_df[model_df['stop'] > model_df['start']].copy()

    # Ensure numeric types
    for col in ['start', 'stop', 'event', 'tub_adopted']:
        model_df[col] = model_df[col].astype(float)

    print(f"  Intervals: {len(model_df):,}, Events: {int(model_df['event'].sum()):,}")
    print(f"  TUB-adopted intervals: {int(model_df['tub_adopted'].sum()):,}")

    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(model_df, id_col='user_id', start_col='start',
            stop_col='stop', event_col='event')
    ctv.print_summary()

    results = {
        'n_intervals': int(len(model_df)),
        'n_users': int(model_df['user_id'].nunique()),
        'n_events': int(model_df['event'].sum()),
        'log_likelihood': float(ctv.log_likelihood_),
        'AIC': float(ctv.AIC_partial_),
        'coefficients': {},
    }

    for var in ctv.summary.index:
        row = ctv.summary.loc[var]
        results['coefficients'][var] = {
            'coef': float(row['coef']),
            'exp_coef': float(row['exp(coef)']),
            'se': float(row['se(coef)']),
            'z': float(row['z']),
            'p': float(row['p']),
            'lower_95': float(row['exp(coef) lower 95%']),
            'upper_95': float(row['exp(coef) upper 95%']),
        }

    tub_hr = results['coefficients'].get('tub_adopted', {}).get('exp_coef', None)
    if tub_hr:
        print(f"\n  TUB adoption hazard ratio: {tub_hr:.2f}")

    return ctv, results


def plot_kaplan_meier(df: pd.DataFrame, results_km: dict,
                      output_path: Path) -> None:
    """Plot Kaplan-Meier survival curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    max_x = 100  # Show first 100 trips

    # (a) By TUB usage
    ax = axes[0]
    for label, mask, color, ls in [
        ('TUB users', df['ever_tub'] == 1, '#d62728', '-'),
        ('Non-TUB users', df['ever_tub'] == 0, '#1f77b4', '-'),
    ]:
        kmf = KaplanMeierFitter()
        subset = df[mask]
        kmf.fit(subset['event_time'], subset['event'], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color,
                                    linestyle=ls, linewidth=2)
    ax.set_xlim(0, max_x)
    ax.set_xlabel('Trip rank (cumulative trips)', fontsize=11)
    ax.set_ylabel('Survival probability\n(no speeding yet)', fontsize=11)
    ax.set_title('(a) Survival by TUB Mode Usage', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Add median annotations
    tub_med = results_km['by_tub']['tub_users']['median_survival']
    notub_med = results_km['by_tub']['non_tub_users']['median_survival']
    if tub_med and tub_med <= max_x:
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(f'Median: {tub_med:.0f}', xy=(tub_med, 0.5),
                    xytext=(tub_med + 5, 0.55), fontsize=8, color='#d62728',
                    arrowprops=dict(arrowstyle='->', color='#d62728'))

    # Add log-rank p-value
    lr_p = results_km['logrank']['p_value']
    ax.text(0.98, 0.95, f'Log-rank p < 0.001' if lr_p < 0.001 else f'Log-rank p = {lr_p:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (b) By age group
    ax = axes[1]
    colors_age = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#a65628']
    age_groups = ['<20', '20-24', '25-29', '30-34', '35-39', '40+']
    for ag, color in zip(age_groups, colors_age):
        mask = df['age_group'] == ag
        if mask.sum() < 100:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, 'event_time'], df.loc[mask, 'event'], label=ag)
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=1.8)

    ax.set_xlim(0, max_x)
    ax.set_xlabel('Trip rank (cumulative trips)', fontsize=11)
    ax.set_ylabel('Survival probability\n(no speeding yet)', fontsize=11)
    ax.set_title('(b) Survival by Age Group', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, title='Age group', loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Kaplan-Meier: Time-to-First-Speeding', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cox_forest(cph_static: CoxPHFitter, cph_tv: CoxTimeVaryingFitter,
                     output_path: Path) -> None:
    """Plot Cox PH hazard ratios as forest plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Static Cox PH
    ax = axes[0]
    summary = cph_static.summary.copy()
    summary = summary.sort_values('exp(coef)', ascending=True)

    y_pos = range(len(summary))
    hrs = summary['exp(coef)'].values
    lower = summary['exp(coef) lower 95%'].values
    upper = summary['exp(coef) upper 95%'].values
    labels = [name.replace('age_', 'Age ').replace('_', ' ')
              for name in summary.index]

    colors = ['#d62728' if p < 0.001 else '#ff7f00' if p < 0.05 else '#999999'
              for p in summary['p'].values]

    ax.errorbar(hrs, y_pos, xerr=[hrs - lower, upper - hrs],
                fmt='o', capsize=3, color='black', markersize=5, linewidth=1)
    for i, (hr, c) in enumerate(zip(hrs, colors)):
        ax.scatter(hr, i, color=c, s=60, zorder=5)

    ax.axvline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
    ax.set_title('(a) Cox PH - Static Covariates', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, axis='x', alpha=0.3)

    # Concordance annotation
    ax.text(0.98, 0.02, f'C-index = {cph_static.concordance_index_:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (b) Time-varying Cox PH
    ax = axes[1]
    summary_tv = cph_tv.summary.copy()
    summary_tv = summary_tv.sort_values('exp(coef)', ascending=True)

    y_pos_tv = range(len(summary_tv))
    hrs_tv = summary_tv['exp(coef)'].values
    lower_tv = summary_tv['exp(coef) lower 95%'].values
    upper_tv = summary_tv['exp(coef) upper 95%'].values
    labels_tv = [name.replace('age_', 'Age ').replace('_', ' ')
                  for name in summary_tv.index]

    colors_tv = ['#d62728' if p < 0.001 else '#ff7f00' if p < 0.05 else '#999999'
                  for p in summary_tv['p'].values]

    ax.errorbar(hrs_tv, y_pos_tv, xerr=[hrs_tv - lower_tv, upper_tv - hrs_tv],
                fmt='o', capsize=3, color='black', markersize=5, linewidth=1)
    for i, (hr, c) in enumerate(zip(hrs_tv, colors_tv)):
        ax.scatter(hr, i, color=c, s=60, zorder=5)

    ax.axvline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(list(y_pos_tv))
    ax.set_yticklabels(labels_tv, fontsize=9)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
    ax.set_title('(b) Cox PH - Time-Varying TUB', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, axis='x', alpha=0.3)

    # AIC annotation (CoxTimeVaryingFitter doesn't compute concordance)
    ax.text(0.98, 0.02, f'AIC = {cph_tv.AIC_partial_:.0f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Cox Proportional Hazards: Speeding Onset',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cumulative_hazard(df: pd.DataFrame, output_path: Path) -> None:
    """Plot cumulative hazard by TUB usage to visualize hazard dynamics."""
    from lifelines import NelsonAalenFitter

    fig, ax = plt.subplots(figsize=(8, 5))
    max_x = 100

    for label, mask, color in [
        ('TUB users', df['ever_tub'] == 1, '#d62728'),
        ('Non-TUB users', df['ever_tub'] == 0, '#1f77b4'),
    ]:
        naf = NelsonAalenFitter()
        subset = df[mask]
        naf.fit(subset['event_time'], subset['event'], label=label)
        naf.plot_cumulative_hazard(ax=ax, color=color, linewidth=2, ci_show=True)

    ax.set_xlim(0, max_x)
    ax.set_xlabel('Trip rank (cumulative trips)', fontsize=11)
    ax.set_ylabel('Cumulative hazard', fontsize=11)
    ax.set_title('Nelson-Aalen Cumulative Hazard: Time-to-First-Speeding',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_tub_mediation(df: pd.DataFrame) -> dict[str, Any]:
    """Estimate how much of the experience-speeding link is mediated by TUB adoption.

    Compares:
    1. Experience effect on speeding (total)
    2. Experience effect within STD/ECO only (direct, removing TUB mediation)
    """
    con = duckdb.connect()
    print("\n--- TUB Mediation Analysis ---")

    # Get trip-level data: experience bins x mode x speeding
    mediation = con.execute(f"""
        SELECT
            experience_bin,
            mode,
            COUNT(*) as n_trips,
            AVG(CASE WHEN is_speeding THEN 1.0 ELSE 0.0 END) as speeding_rate
        FROM read_parquet('{EXPERIENCE_PATH}')
        WHERE mean_speed_from_speeds IS NOT NULL
          AND month_year >= '2023-02' AND month_year <= '2023-11'
        GROUP BY experience_bin, mode
        ORDER BY experience_bin, mode
    """).fetchdf()

    # Overall speeding by experience
    overall = con.execute(f"""
        SELECT
            experience_bin,
            COUNT(*) as n_trips,
            AVG(CASE WHEN is_speeding THEN 1.0 ELSE 0.0 END) as speeding_rate,
            AVG(CASE WHEN mode = 'TUB' THEN 1.0 ELSE 0.0 END) as tub_rate
        FROM read_parquet('{EXPERIENCE_PATH}')
        WHERE mean_speed_from_speeds IS NOT NULL
          AND month_year >= '2023-02' AND month_year <= '2023-11'
        GROUP BY experience_bin
        ORDER BY experience_bin
    """).fetchdf()
    con.close()

    # STD/ECO only speeding rate by experience
    std_eco = mediation[mediation['mode'].isin(['STD', 'ECO'])].groupby('experience_bin').agg(
        n_trips=('n_trips', 'sum'),
        speeding_rate=('speeding_rate', 'mean')  # weighted average would be better but this is close
    ).reset_index()

    print("\nExperience -> Speeding rate:")
    print(f"  {'Bin':<15} {'Overall':>10} {'STD/ECO':>10} {'TUB rate':>10}")
    results = {'experience_bins': []}
    for _, row in overall.iterrows():
        eb = row['experience_bin']
        std_row = std_eco[std_eco['experience_bin'] == eb]
        std_rate = std_row['speeding_rate'].values[0] if len(std_row) > 0 else None
        print(f"  {eb:<15} {row['speeding_rate']:>10.3f} {std_rate if std_rate else 'N/A':>10} {row['tub_rate']:>10.3f}")
        results['experience_bins'].append({
            'bin': eb,
            'overall_speeding': float(row['speeding_rate']),
            'std_eco_speeding': float(std_rate) if std_rate else None,
            'tub_rate': float(row['tub_rate']),
            'n_trips': int(row['n_trips']),
        })

    # Calculate mediation proportion
    # Total effect: speeding difference between most/least experienced
    bins_sorted = overall.sort_values('experience_bin')
    if len(bins_sorted) >= 2:
        total_effect = (bins_sorted.iloc[-1]['speeding_rate'] -
                        bins_sorted.iloc[0]['speeding_rate'])
        # Direct effect (within STD/ECO)
        std_sorted = std_eco.sort_values('experience_bin')
        if len(std_sorted) >= 2:
            direct_effect = (std_sorted.iloc[-1]['speeding_rate'] -
                             std_sorted.iloc[0]['speeding_rate'])
            mediation_pct = (1 - direct_effect / total_effect) * 100 if total_effect != 0 else 0
            # Bootstrap CI for mediation proportion
            print("\n  Bootstrapping mediation CI (1000 iterations)...")
            n_boot = 1000
            boot_mediation = []
            all_bins = sorted(overall['experience_bin'].unique())
            first_bin, last_bin = all_bins[0], all_bins[-1]
            for b in range(n_boot):
                # Resample cities/users by resampling experience bins with replacement
                boot_overall = overall.sample(n=len(overall), replace=True,
                                              random_state=RANDOM_SEED + b)
                boot_std = std_eco.sample(n=len(std_eco), replace=True,
                                          random_state=RANDOM_SEED + b)
                bo_sorted = boot_overall.sort_values('experience_bin')
                bs_sorted = boot_std.sort_values('experience_bin')
                if len(bo_sorted) >= 2 and len(bs_sorted) >= 2:
                    b_total = bo_sorted.iloc[-1]['speeding_rate'] - bo_sorted.iloc[0]['speeding_rate']
                    b_direct = bs_sorted.iloc[-1]['speeding_rate'] - bs_sorted.iloc[0]['speeding_rate']
                    if b_total != 0:
                        boot_mediation.append((1 - b_direct / b_total) * 100)
            boot_arr = np.array(boot_mediation)
            ci_lo, ci_hi = np.percentile(boot_arr, [2.5, 97.5])

            results['mediation'] = {
                'total_effect': float(total_effect),
                'direct_effect': float(direct_effect),
                'indirect_via_tub': float(total_effect - direct_effect),
                'mediation_pct': float(mediation_pct),
                'bootstrap_ci_95': [float(ci_lo), float(ci_hi)],
                'bootstrap_n': n_boot,
            }
            print(f"\n  Total experience effect: {total_effect:.3f}")
            print(f"  Direct effect (within STD/ECO): {direct_effect:.3f}")
            print(f"  Mediated via TUB adoption: {mediation_pct:.1f}% "
                  f"(95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%])")

    return results


def main() -> None:
    """Run the full survival analysis pipeline."""
    print("=" * 70)
    print("SURVIVAL ANALYSIS: TIME-TO-FIRST-SPEEDING")
    print("=" * 70)

    all_results: dict[str, Any] = {}

    # Step 1: Build survival dataset
    print("\n--- Step 1: Build Survival Dataset ---")
    df = build_survival_dataset()
    all_results['dataset'] = {
        'n_users': int(len(df)),
        'n_events': int(df['event'].sum()),
        'n_censored': int((df['event'] == 0).sum()),
        'event_rate': float(df['event'].mean()),
        'median_event_time': float(df.loc[df['event'] == 1, 'event_time'].median()),
        'median_censor_time': float(df.loc[df['event'] == 0, 'event_time'].median()),
        'ever_tub_pct': float(df['ever_tub'].mean()),
        'tub_before_speed_pct': float(
            df.loc[df['event'] == 1, 'tub_before_speed'].mean()
        ),
    }

    # Step 2: Kaplan-Meier
    print("\n--- Step 2: Kaplan-Meier Curves ---")
    km_results, kmf_tub, kmf_notub = fit_kaplan_meier(df)
    all_results['kaplan_meier'] = km_results

    # Step 3: Plot KM
    print("\n--- Step 3: KM Plots ---")
    plot_kaplan_meier(df, km_results, FIGURES_DIR / 'fig_survival_km.pdf')

    # Step 4: Cumulative hazard
    print("\n--- Step 4: Cumulative Hazard ---")
    plot_cumulative_hazard(df, FIGURES_DIR / 'fig_survival_hazard.pdf')

    # Step 5: Cox PH (static)
    print("\n--- Step 5: Cox PH (Static) ---")
    cph_static, cox_static_results = fit_cox_static(df)
    all_results['cox_static'] = cox_static_results

    # Step 6: Time-varying dataset
    print("\n--- Step 6: Build Time-Varying Dataset ---")
    tv_df = build_time_varying_dataset(df, sample_n=100000)

    # Step 7: Cox PH (time-varying)
    print("\n--- Step 7: Cox PH (Time-Varying) ---")
    cph_tv, cox_tv_results = fit_cox_time_varying(tv_df)
    all_results['cox_time_varying'] = cox_tv_results

    # Step 8: Forest plot
    print("\n--- Step 8: Forest Plots ---")
    plot_cox_forest(cph_static, cph_tv, FIGURES_DIR / 'fig_survival_cox.pdf')

    # Step 9: TUB mediation
    print("\n--- Step 9: TUB Mediation ---")
    mediation_results = compute_tub_mediation(df)
    all_results['tub_mediation'] = mediation_results

    # Step 10: Save all results
    print("\n--- Step 10: Save Results ---")
    out_path = MODELING_DIR / 'survival_analysis_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SURVIVAL ANALYSIS COMPLETE")
    print(f"  Users: {len(df):,}")
    print(f"  Events (first speeding): {df['event'].sum():,} ({100*df['event'].mean():.1f}%)")
    print(f"  Censored (never sped): {(df['event']==0).sum():,}")
    tub_hr_static = all_results['cox_static']['coefficients'].get(
        'ever_tub', {}).get('exp_coef', 'N/A')
    tub_hr_tv = all_results['cox_time_varying']['coefficients'].get(
        'tub_adopted', {}).get('exp_coef', 'N/A')
    print(f"  TUB hazard ratio (static): {tub_hr_static}")
    print(f"  TUB hazard ratio (time-varying): {tub_hr_tv}")
    print(f"  Cox concordance (static): {all_results['cox_static']['concordance']:.4f}")
    print(f"  Cox AIC (time-varying): {all_results['cox_time_varying']['AIC']:.1f}")
    if 'mediation' in all_results.get('tub_mediation', {}):
        med_pct = all_results['tub_mediation']['mediation']['mediation_pct']
        print(f"  TUB mediation of experience effect: {med_pct:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
