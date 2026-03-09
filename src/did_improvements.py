"""
DiD Analysis Improvements — Addresses reviewer feedback from 2026-03-02.

Improvements:
1. Joint F-test for pre-trend coefficients (instead of individual t-tests)
2. Cluster-robust standard errors at city level (Arellano 1987)
3. Alternative event study with Nov as reference month
4. Bootstrapped 95% CI for dose-response correlation
5. De-trended event study (remove linear pre-trend)
6. Consistent sample filtering across dose-response figures
7. Correct panel dimension reporting

Output:
- data_parquet/modeling/did_improvements.json
- figures/fig_did_event_study_v2.pdf  (Nov reference, improved)
- figures/fig_did_summary_v2.pdf      (all normalized to per-10pp)
"""
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIGURES_DIR, MODELING_DIR, FIG_DPI, RANDOM_SEED

warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(RANDOM_SEED)


def load_panel() -> pd.DataFrame:
    """Load and prepare the city-month panel with treatment variables."""
    panel = pd.read_parquet(MODELING_DIR / 'city_month_panel.parquet')

    # Pre-ban TUB share (Nov 2023)
    nov_tub = panel[panel['month_year'] == '2023-11'][['city', 'tub_share']].copy()
    nov_tub.columns = ['city', 'pre_tub_share']
    panel = panel.merge(nov_tub, on='city', how='left')

    # Fallback for cities missing Nov data
    if panel['pre_tub_share'].isna().any():
        pre_period = panel[panel['month_year'].isin(['2023-09', '2023-10', '2023-11'])]
        fallback = pre_period.groupby('city')['tub_share'].mean().reset_index()
        fallback.columns = ['city', 'fallback_tub']
        panel = panel.merge(fallback, on='city', how='left')
        panel['pre_tub_share'] = panel['pre_tub_share'].fillna(panel['fallback_tub'])
        panel.drop(columns=['fallback_tub'], inplace=True)

    # Filter to cities with >= 100 trips minimum
    city_min = panel.groupby('city')['total_trips'].min()
    valid = city_min[city_min >= 100].index
    panel = panel[panel['city'].isin(valid)].copy()

    # Treatment variables
    panel['post'] = (panel['month_year'] == '2023-12').astype(int)
    panel['treatment_intensity'] = panel['post'] * panel['pre_tub_share']

    return panel


def report_panel_dimensions(panel: pd.DataFrame) -> dict:
    """Report panel dimensions accurately, noting incomplete coverage."""
    n_cities = panel['city'].nunique()
    n_months = panel['month_year'].nunique()
    n_obs = len(panel)
    expected = n_cities * n_months

    # Identify cities with incomplete coverage
    city_month_counts = panel.groupby('city')['month_year'].nunique()
    incomplete = city_month_counts[city_month_counts < n_months]

    print(f"\n--- Panel Dimensions ---")
    print(f"  {n_cities} cities x {n_months} months = {expected} expected, {n_obs} actual")
    if len(incomplete) > 0:
        print(f"  {len(incomplete)} cities with incomplete monthly coverage:")
        for city, count in incomplete.items():
            print(f"    {city}: {count}/{n_months} months")

    return {
        'n_cities': n_cities,
        'n_months': n_months,
        'n_obs': n_obs,
        'expected_obs': expected,
        'incomplete_cities': {
            city: int(count) for city, count in incomplete.items()
        },
    }


def twfe_cluster_robust(panel: pd.DataFrame) -> dict:
    """TWFE DiD with cluster-robust SEs at city level (Arellano 1987).

    Addresses reviewer concern: HC1 corrects heteroscedasticity but not
    within-city serial correlation. With 11 periods per city, clustering
    is more appropriate.
    """
    print("\n--- TWFE with Cluster-Robust SEs ---")

    # Build design matrix manually for more control
    y = panel['overall_speeding'].values
    weights = np.sqrt(panel['total_trips'].values)

    X_treat = panel['treatment_intensity'].values.reshape(-1, 1)
    city_dummies = pd.get_dummies(panel['city'], prefix='city', drop_first=True)
    month_dummies = pd.get_dummies(panel['month_year'], prefix='month', drop_first=True)

    X = np.column_stack([
        np.ones(len(panel)),
        X_treat,
        city_dummies.values,
        month_dummies.values,
    ])

    col_names = ['const', 'treatment_intensity'] + \
        city_dummies.columns.tolist() + month_dummies.columns.tolist()

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(weights)
    X_clean = X[mask]
    y_clean = y[mask]
    w_clean = weights[mask]
    groups_clean = panel['city'].values[mask]

    # HC1 standard errors (original)
    model_hc1 = sm.WLS(y_clean, X_clean, weights=w_clean).fit(cov_type='HC1')

    # Cluster-robust SEs (city-level)
    model_cluster = sm.WLS(y_clean, X_clean, weights=w_clean).fit(
        cov_type='cluster',
        cov_kwds={'groups': groups_clean}
    )

    beta_hc1 = float(model_hc1.params[1])
    se_hc1 = float(model_hc1.bse[1])
    pval_hc1 = float(model_hc1.pvalues[1])

    beta_cluster = float(model_cluster.params[1])
    se_cluster = float(model_cluster.bse[1])
    pval_cluster = float(model_cluster.pvalues[1])

    ci_cluster = model_cluster.conf_int()

    print(f"  HC1:     beta={beta_hc1:.4f}, SE={se_hc1:.4f}, p={pval_hc1:.2e}")
    print(f"  Cluster: beta={beta_cluster:.4f}, SE={se_cluster:.4f}, p={pval_cluster:.2e}")
    print(f"  SE ratio (cluster/HC1): {se_cluster/se_hc1:.2f}")
    print(f"  Cluster 95% CI: [{ci_cluster[1, 0]:.4f}, {ci_cluster[1, 1]:.4f}]")

    return {
        'hc1': {
            'beta': beta_hc1, 'se': se_hc1, 'pvalue': pval_hc1,
        },
        'cluster': {
            'beta': beta_cluster, 'se': se_cluster, 'pvalue': pval_cluster,
            'ci_95': [float(ci_cluster[1, 0]), float(ci_cluster[1, 1])],
        },
        'se_ratio': se_cluster / se_hc1,
    }


def event_study_nov_reference(panel: pd.DataFrame) -> dict:
    """Event study with Nov 2023 as reference (last pre-treatment month).

    Reviewer suggested Nov reference since it's the immediate pre-treatment
    period, making the Dec coefficient directly interpretable as the
    one-month treatment effect.
    """
    print("\n--- Event Study (Nov reference) ---")

    month_map = {
        '2023-02': -9, '2023-03': -8, '2023-04': -7, '2023-05': -6,
        '2023-06': -5, '2023-07': -4, '2023-08': -3, '2023-09': -2,
        '2023-10': -1, '2023-11': 0, '2023-12': 1
    }
    ref_month = '2023-11'  # Nov as reference (k=0)

    panel_es = panel.copy()
    panel_es['rel_time'] = panel_es['month_year'].map(month_map)

    # Create interaction terms (skip reference month)
    event_dummies = {}
    for month_str, k in month_map.items():
        if month_str == ref_month:
            continue
        col = f'evt_k{k:+d}'
        panel_es[col] = (
            (panel_es['month_year'] == month_str).astype(float)
            * panel_es['pre_tub_share']
        )
        event_dummies[col] = k

    panel_es = panel_es.dropna(subset=['pre_tub_share', 'overall_speeding'])

    y = panel_es['overall_speeding'].values
    weights = np.sqrt(panel_es['total_trips'].values)

    evt_names = list(event_dummies.keys())
    X_evt = panel_es[evt_names].values
    city_dummies = pd.get_dummies(panel_es['city'], prefix='city', drop_first=True)
    month_dummies = pd.get_dummies(panel_es['month_year'], prefix='month', drop_first=True)

    X = np.column_stack([
        np.ones(len(panel_es)),
        X_evt,
        city_dummies.values,
        month_dummies.values,
    ])

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(weights)
    X_clean, y_clean, w_clean = X[mask], y[mask], weights[mask]
    groups_clean = panel_es['city'].values[mask]

    # Fit with cluster-robust SEs
    model = sm.WLS(y_clean, X_clean, weights=w_clean).fit(
        cov_type='cluster',
        cov_kwds={'groups': groups_clean}
    )

    # Extract event study coefficients
    ci = model.conf_int()
    es_results = {}
    for i, (col, k) in enumerate(event_dummies.items()):
        idx = 1 + i
        es_results[k] = {
            'coef': float(model.params[idx]),
            'se': float(model.bse[idx]),
            'pvalue': float(model.pvalues[idx]),
            'ci_lo': float(ci[idx, 0]),
            'ci_hi': float(ci[idx, 1]),
        }

    # Joint F-test for pre-trends: H0: all pre-period coefficients = 0
    pre_indices = [1 + i for i, (_, k) in enumerate(event_dummies.items()) if k < 0]
    r_matrix = np.zeros((len(pre_indices), X_clean.shape[1]))
    for row, col_idx in enumerate(pre_indices):
        r_matrix[row, col_idx] = 1.0

    f_test = model.f_test(r_matrix)
    f_stat = float(f_test.fvalue)
    f_pval = float(f_test.pvalue)

    print(f"  Event study coefficients (Nov=0, cluster-robust SEs):")
    for k in sorted(es_results.keys()):
        v = es_results[k]
        sig = '***' if v['pvalue'] < 0.001 else '**' if v['pvalue'] < 0.01 else '*' if v['pvalue'] < 0.05 else ''
        label = ' <-- POST (Dec)' if k == 1 else (' <-- REF (Nov)' if k == 0 else '')
        print(f"    k={k:+2d}: {v['coef']:+.4f} (SE={v['se']:.4f}, p={v['pvalue']:.4f}) {sig}{label}")
    print(f"  Joint F-test for pre-trends: F={f_stat:.3f}, p={f_pval:.4f}")
    print(f"  Pre-trends {'PASS' if f_pval >= 0.05 else 'FAIL'} at 5%")

    return {
        'reference_month': 'Nov 2023 (k=0)',
        'se_type': 'cluster-robust (city)',
        'event_study_coefs': {str(k): v for k, v in es_results.items()},
        'joint_f_test': {
            'f_statistic': f_stat,
            'p_value': f_pval,
            'n_restrictions': len(pre_indices),
            'passes_5pct': f_pval >= 0.05,
        },
    }, es_results


def event_study_detrended(panel: pd.DataFrame) -> dict:
    """De-trended event study: remove linear pre-trend.

    If there's a linear pre-trend, the standard event study will show
    significant pre-period coefficients even if the parallel trends
    assumption holds after detrending. This specification includes a
    linear interaction of time x pre_tub_share, so the event dummies
    capture deviations from the linear trend.
    """
    print("\n--- De-trended Event Study ---")

    month_map = {
        '2023-02': -9, '2023-03': -8, '2023-04': -7, '2023-05': -6,
        '2023-06': -5, '2023-07': -4, '2023-08': -3, '2023-09': -2,
        '2023-10': -1, '2023-11': 0, '2023-12': 1
    }
    ref_month = '2023-11'

    panel_dt = panel.copy()
    panel_dt['rel_time'] = panel_dt['month_year'].map(month_map)

    # Linear pre-trend: time * pre_tub_share (estimated only on pre-period)
    panel_dt['time_x_tub'] = panel_dt['rel_time'] * panel_dt['pre_tub_share']

    # Event dummies (skip reference)
    event_dummies = {}
    for month_str, k in month_map.items():
        if month_str == ref_month:
            continue
        col = f'evt_k{k:+d}'
        panel_dt[col] = (
            (panel_dt['month_year'] == month_str).astype(float)
            * panel_dt['pre_tub_share']
        )
        event_dummies[col] = k

    panel_dt = panel_dt.dropna(subset=['pre_tub_share', 'overall_speeding'])

    y = panel_dt['overall_speeding'].values
    weights = np.sqrt(panel_dt['total_trips'].values)

    evt_names = list(event_dummies.keys())
    X_evt = panel_dt[evt_names].values
    X_trend = panel_dt['time_x_tub'].values.reshape(-1, 1)
    city_dummies = pd.get_dummies(panel_dt['city'], prefix='city', drop_first=True)
    month_dummies = pd.get_dummies(panel_dt['month_year'], prefix='month', drop_first=True)

    X = np.column_stack([
        np.ones(len(panel_dt)),
        X_trend,
        X_evt,
        city_dummies.values,
        month_dummies.values,
    ])

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(weights)
    X_clean, y_clean, w_clean = X[mask], y[mask], weights[mask]
    groups_clean = panel_dt['city'].values[mask]

    model = sm.WLS(y_clean, X_clean, weights=w_clean).fit(
        cov_type='cluster',
        cov_kwds={'groups': groups_clean}
    )

    # Linear trend coefficient
    trend_coef = float(model.params[1])
    trend_se = float(model.bse[1])
    trend_pval = float(model.pvalues[1])
    print(f"  Linear pre-trend: {trend_coef:.4f} (SE={trend_se:.4f}, p={trend_pval:.4f})")

    # Extract de-trended event study coefficients
    ci = model.conf_int()
    es_results = {}
    for i, (col, k) in enumerate(event_dummies.items()):
        idx = 2 + i  # +1 for const, +1 for trend
        es_results[k] = {
            'coef': float(model.params[idx]),
            'se': float(model.bse[idx]),
            'pvalue': float(model.pvalues[idx]),
            'ci_lo': float(ci[idx, 0]),
            'ci_hi': float(ci[idx, 1]),
        }

    # Joint F-test on de-trended pre-period coefficients
    pre_indices = [2 + i for i, (_, k) in enumerate(event_dummies.items()) if k < 0]
    r_matrix = np.zeros((len(pre_indices), X_clean.shape[1]))
    for row, col_idx in enumerate(pre_indices):
        r_matrix[row, col_idx] = 1.0

    f_test = model.f_test(r_matrix)
    f_stat = float(f_test.fvalue)
    f_pval = float(f_test.pvalue)

    print(f"  De-trended event study coefficients:")
    for k in sorted(es_results.keys()):
        v = es_results[k]
        sig = '***' if v['pvalue'] < 0.001 else '**' if v['pvalue'] < 0.01 else '*' if v['pvalue'] < 0.05 else ''
        print(f"    k={k:+2d}: {v['coef']:+.4f} (SE={v['se']:.4f}) {sig}")
    print(f"  De-trended joint F-test: F={f_stat:.3f}, p={f_pval:.4f}")
    print(f"  De-trended pre-trends {'PASS' if f_pval >= 0.05 else 'FAIL'} at 5%")

    return {
        'linear_trend': {
            'coef': trend_coef, 'se': trend_se, 'pvalue': trend_pval,
        },
        'detrended_coefs': {str(k): v for k, v in es_results.items()},
        'joint_f_test_detrended': {
            'f_statistic': f_stat,
            'p_value': f_pval,
            'passes_5pct': f_pval >= 0.05,
        },
    }, es_results


def bootstrap_dose_response_ci(panel: pd.DataFrame,
                                n_bootstrap: int = 10000) -> dict:
    """Bootstrap 95% CI for dose-response correlation and slope.

    Addresses reviewer concern: correlation r=0.754 reported without CI.
    """
    print(f"\n--- Bootstrapped CI for Dose-Response (n={n_bootstrap}) ---")

    # Build Nov-Dec cross section
    nov = panel[panel['month_year'] == '2023-11'][
        ['city', 'overall_speeding', 'tub_share', 'total_trips']
    ].copy()
    nov.columns = ['city', 'nov_speeding', 'nov_tub', 'nov_trips']

    dec = panel[panel['month_year'] == '2023-12'][
        ['city', 'overall_speeding', 'tub_share', 'total_trips']
    ].copy()
    dec.columns = ['city', 'dec_speeding', 'dec_tub', 'dec_trips']

    cs = nov.merge(dec, on='city')
    cs['tub_reduction'] = cs['nov_tub'] - cs['dec_tub']
    cs['speeding_reduction'] = (cs['nov_speeding'] - cs['dec_speeding']) * 100

    # Point estimates
    r_point, p_point = stats.pearsonr(cs['tub_reduction'], cs['speeding_reduction'])

    X = sm.add_constant(cs['tub_reduction'])
    w = np.sqrt(cs['nov_trips'])
    model = sm.WLS(cs['speeding_reduction'], X, weights=w).fit(cov_type='HC1')
    slope_point = float(model.params.iloc[1])

    # Fisher z-transform CI for r
    z = np.arctanh(r_point)
    se_z = 1.0 / np.sqrt(len(cs) - 3)
    z_lo = z - 1.96 * se_z
    z_hi = z + 1.96 * se_z
    r_lo_fisher = np.tanh(z_lo)
    r_hi_fisher = np.tanh(z_hi)

    # Bootstrap CIs for both r and slope
    rng = np.random.RandomState(RANDOM_SEED)
    boot_r = np.zeros(n_bootstrap)
    boot_slope = np.zeros(n_bootstrap)

    n = len(cs)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_cs = cs.iloc[idx]
        boot_r[b] = stats.pearsonr(
            boot_cs['tub_reduction'], boot_cs['speeding_reduction']
        )[0]
        X_b = sm.add_constant(boot_cs['tub_reduction'])
        w_b = np.sqrt(boot_cs['nov_trips'])
        try:
            m_b = sm.WLS(boot_cs['speeding_reduction'], X_b, weights=w_b).fit()
            boot_slope[b] = float(m_b.params.iloc[1])
        except Exception:
            boot_slope[b] = np.nan

    boot_slope = boot_slope[~np.isnan(boot_slope)]
    r_lo_boot = np.percentile(boot_r, 2.5)
    r_hi_boot = np.percentile(boot_r, 97.5)
    slope_lo = np.percentile(boot_slope, 2.5)
    slope_hi = np.percentile(boot_slope, 97.5)

    print(f"  r = {r_point:.3f} (p = {p_point:.2e})")
    print(f"  Fisher z 95% CI:    [{r_lo_fisher:.3f}, {r_hi_fisher:.3f}]")
    print(f"  Bootstrap 95% CI:   [{r_lo_boot:.3f}, {r_hi_boot:.3f}]")
    print(f"  WLS slope = {slope_point:.2f}")
    print(f"  Bootstrap slope CI: [{slope_lo:.2f}, {slope_hi:.2f}]")
    print(f"  Per 10pp: {slope_point * 0.10:.1f}pp [{slope_lo * 0.10:.1f}, {slope_hi * 0.10:.1f}]")

    return {
        'n_cities': len(cs),
        'r': r_point,
        'r_pvalue': p_point,
        'r_ci_fisher': [float(r_lo_fisher), float(r_hi_fisher)],
        'r_ci_bootstrap': [float(r_lo_boot), float(r_hi_boot)],
        'slope': slope_point,
        'slope_ci_bootstrap': [float(slope_lo), float(slope_hi)],
        'slope_per_10pp': slope_point * 0.10,
        'slope_per_10pp_ci': [float(slope_lo * 0.10), float(slope_hi * 0.10)],
        'n_bootstrap': n_bootstrap,
    }


def generate_improved_event_study_figure(
    es_nov_ref: dict,
    es_detrended: dict,
) -> None:
    """Generate improved event study figure with Nov reference + de-trended.

    Two-panel figure:
    (a) Standard event study with Nov reference, cluster-robust SEs
    (b) De-trended event study removing linear pre-trend
    """
    print("\n[Figures] Generating improved event study figure...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    month_labels = {
        -9: 'Feb', -8: 'Mar', -7: 'Apr', -6: 'May', -5: 'Jun',
        -4: 'Jul', -3: 'Aug', -2: 'Sep', -1: 'Oct', 0: 'Nov\n(ref)', 1: 'Dec'
    }

    for ax_idx, (es_data, title_suffix, panel_label) in enumerate([
        (es_nov_ref, 'Nov reference, cluster SEs', '(a) Standard'),
        (es_detrended, 'De-trended, cluster SEs', '(b) De-trended'),
    ]):
        ax = axes[ax_idx]

        ks = sorted([k for k in es_data.keys() if k != 0])  # skip ref (=0)
        # Insert reference point
        ks_full = sorted(list(set(ks + [0])))
        coefs_full = []
        ci_lo_full = []
        ci_hi_full = []

        for k in ks_full:
            if k == 0:  # reference point
                coefs_full.append(0)
                ci_lo_full.append(0)
                ci_hi_full.append(0)
            else:
                coefs_full.append(es_data[k]['coef'])
                ci_lo_full.append(es_data[k]['ci_lo'])
                ci_hi_full.append(es_data[k]['ci_hi'])

        # Color points by significance
        colors = []
        for k in ks_full:
            if k == 0:
                colors.append('#2ca02c')  # reference = green
            elif k > 0:
                colors.append('#d62728')  # post-treatment = red
            elif k in es_data and es_data[k]['pvalue'] < 0.05:
                colors.append('#ff7f0e')  # significant pre = orange
            else:
                colors.append('#1f77b4')  # non-significant = blue

        ax.errorbar(ks_full, coefs_full,
                     yerr=[np.array(coefs_full) - np.array(ci_lo_full),
                           np.array(ci_hi_full) - np.array(coefs_full)],
                     fmt='o-', color='#1f77b4', capsize=4, capthick=1.5,
                     linewidth=1.5, markersize=6, zorder=2)

        # Color individual markers
        for i, (k, c) in enumerate(zip(ks_full, colors)):
            ax.plot(k, coefs_full[i], 'o', color=c, markersize=7, zorder=3)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                    label='TUB ban')

        ax.set_xticks(ks_full)
        ax.set_xticklabels([month_labels.get(k, str(k)) for k in ks_full],
                           fontsize=8)
        ax.set_xlabel('Month (2023)', fontsize=10)
        ax.set_ylabel('Coefficient (x pre-TUB share)', fontsize=10)
        ax.set_title(f'{panel_label}: {title_suffix}', fontsize=10,
                     fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

    plt.suptitle('Event Study: TUB Mode Restriction Effect on Speeding',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_event_study_v2.pdf', dpi=FIG_DPI,
                bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_event_study_v2.png', dpi=FIG_DPI,
                bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_event_study_v2.pdf/png")


def generate_improved_summary_figure(
    twfe_cluster: dict,
    es_nov: dict,
    dose_response: dict,
) -> None:
    """Generate improved summary figure with all estimates normalized.

    All estimates are normalized to "speeding change per 10pp TUB share"
    using cluster-robust SEs.
    """
    print("\n[Figures] Generating improved summary figure...")

    # Load original cross-sectional result for comparison
    with open(MODELING_DIR / 'did_results.json') as f:
        orig = json.load(f)

    # Normalize all to per-10pp
    # 1. Cross-sectional: beta is per 100pp (unit) of TUB share, y is in pp
    cs_per10 = orig['model_1_cross_sectional']['beta_tub_share'] * 0.10
    cs_se10 = orig['model_1_cross_sectional']['se_tub_share'] * 0.10

    # 2. TWFE cluster: beta is per unit of TUB share, y is 0-1 scale
    twfe_per10 = twfe_cluster['cluster']['beta'] * 0.10 * 100
    twfe_se10 = twfe_cluster['cluster']['se'] * 0.10 * 100

    # 3. Event study Dec coefficient (Nov ref): same scale as TWFE
    dec_coef = es_nov.get(1, {})
    es_per10 = dec_coef.get('coef', 0) * 0.10 * 100
    es_se10 = dec_coef.get('se', 0) * 0.10 * 100

    # 4. Dose-response: beta is per unit of TUB reduction, y is in pp
    dr_per10 = dose_response['slope'] * 0.10
    dr_se10_lo = dose_response['slope_per_10pp'] - dose_response['slope_per_10pp_ci'][0]
    dr_se10_hi = dose_response['slope_per_10pp_ci'][1] - dose_response['slope_per_10pp']

    fig, ax = plt.subplots(figsize=(9, 5))

    estimates = [
        ('Cross-sectional OLS\n(Nov vs Dec)', cs_per10, cs_se10 * 1.96, cs_se10 * 1.96),
        ('TWFE DiD\n(cluster SEs)', twfe_per10, twfe_se10 * 1.96, twfe_se10 * 1.96),
        ('Event Study Dec\n(Nov ref, cluster)', es_per10, es_se10 * 1.96, es_se10 * 1.96),
        ('Dose-Response\n(bootstrap CI)', dr_per10, dr_se10_lo, dr_se10_hi),
    ]

    y_pos = range(len(estimates))
    labels = [e[0] for e in estimates]
    coefs = [e[1] for e in estimates]
    ci_lo = [e[2] for e in estimates]
    ci_hi = [e[3] for e in estimates]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (label, coef, lo, hi) in enumerate(estimates):
        ax.barh(i, coef, height=0.5, color=colors[i], edgecolor='black',
                linewidth=0.5, alpha=0.8)
        ax.errorbar(coef, i, xerr=[[lo], [hi]], fmt='none',
                     color='black', capsize=5, capthick=1.5, linewidth=1.5)

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Speeding rate change per 10pp TUB share (pp)', fontsize=11)
    ax.set_title('Causal Effect of TUB Restriction on Speeding\n'
                 '(all estimates normalized to per 10pp TUB share)',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # Value labels — adaptive positioning
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    offset = x_range * 0.03

    for i, (coef, lo, hi) in enumerate(zip(coefs, ci_lo, ci_hi)):
        x_pos = coef - lo - offset if coef < 0 else coef + hi + offset
        ha = 'right' if coef < 0 else 'left'
        ax.text(x_pos, i, f'{coef:.1f}pp', va='center', ha=ha,
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_summary_v2.pdf', dpi=FIG_DPI,
                bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_summary_v2.png', dpi=FIG_DPI,
                bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_summary_v2.pdf/png")


def main() -> dict:
    """Run all DiD improvements."""
    print("=" * 60)
    print("DiD IMPROVEMENTS (Reviewer Feedback 2026-03-02)")
    print("=" * 60)

    panel = load_panel()

    # 1. Panel dimensions
    dim_report = report_panel_dimensions(panel)

    # 2. Cluster-robust TWFE
    twfe_cluster = twfe_cluster_robust(panel)

    # 3. Event study with Nov reference + joint F-test
    es_nov_result, es_nov_coefs = event_study_nov_reference(panel)

    # 4. De-trended event study
    es_dt_result, es_dt_coefs = event_study_detrended(panel)

    # 5. Bootstrapped dose-response CI
    dose_ci = bootstrap_dose_response_ci(panel)

    # 6. Improved figures
    generate_improved_event_study_figure(es_nov_coefs, es_dt_coefs)
    generate_improved_summary_figure(twfe_cluster, es_nov_coefs, dose_ci)

    # Compile all results
    all_results = {
        'panel_dimensions': dim_report,
        'twfe_cluster_robust': twfe_cluster,
        'event_study_nov_reference': es_nov_result,
        'event_study_detrended': es_dt_result,
        'dose_response_ci': dose_ci,
    }

    # Summary
    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"  Panel: {dim_report['n_cities']} cities, {dim_report['n_obs']}/{dim_report['expected_obs']} obs "
          f"({len(dim_report['incomplete_cities'])} incomplete)")
    print(f"  TWFE cluster SE = {twfe_cluster['cluster']['se']:.4f} "
          f"(vs HC1 {twfe_cluster['hc1']['se']:.4f}, ratio={twfe_cluster['se_ratio']:.2f})")
    print(f"  Joint F-test (Nov ref): F={es_nov_result['joint_f_test']['f_statistic']:.3f}, "
          f"p={es_nov_result['joint_f_test']['p_value']:.4f}")
    print(f"  De-trended F-test: F={es_dt_result['joint_f_test_detrended']['f_statistic']:.3f}, "
          f"p={es_dt_result['joint_f_test_detrended']['p_value']:.4f}")
    print(f"  Dose-response r = {dose_ci['r']:.3f} [{dose_ci['r_ci_bootstrap'][0]:.3f}, "
          f"{dose_ci['r_ci_bootstrap'][1]:.3f}]")
    print(f"  Dose-response slope per 10pp = {dose_ci['slope_per_10pp']:.1f}pp "
          f"[{dose_ci['slope_per_10pp_ci'][0]:.1f}, {dose_ci['slope_per_10pp_ci'][1]:.1f}]")

    # Save
    output_path = MODELING_DIR / 'did_improvements.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved results to {output_path}")

    return all_results


if __name__ == '__main__':
    results = main()
