"""
Difference-in-Differences Analysis: December 2023 TUB Mode Restriction.

Exploits the sudden, near-universal ban of TUB (turbo) mode in December 2023
as a natural experiment to estimate the causal effect of speed governor policy
on e-scooter speeding behavior.

Design:
- Treatment: TUB mode restriction (Dec 2023)
- Treatment intensity: Pre-ban TUB share (varies by city, 32-70%)
- Outcome: City-level speeding rate
- Panel: 54 cities x 11 months (2023-02 to 2023-12)

Models:
1. Simple pre-post with OLS (cross-sectional)
2. Two-way fixed effects (TWFE) panel DiD
3. Event study with monthly leads/lags
4. Continuous treatment intensity (dose-response)

Output:
- data_parquet/modeling/did_results.json
- figures/fig_did_*.pdf
"""
import json
import sys
import warnings
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIGURES_DIR, MODELING_DIR, FIG_DPI, RANDOM_SEED

warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(RANDOM_SEED)


def build_panel() -> pd.DataFrame:
    """Build city-month panel with treatment variables."""
    panel = pd.read_parquet(MODELING_DIR / 'city_month_panel.parquet')

    # Create time variables
    panel['year_month'] = pd.to_datetime(panel['month_year'] + '-01')
    panel['month_num'] = (
        (panel['year_month'].dt.year - 2023) * 12
        + panel['year_month'].dt.month
    )  # 2=Feb, 3=Mar, ..., 12=Dec

    # Treatment: post-December 2023
    panel['post'] = (panel['month_year'] == '2023-12').astype(int)

    # Treatment intensity: pre-ban TUB share (Nov 2023 TUB share per city)
    nov_tub = panel[panel['month_year'] == '2023-11'][['city', 'tub_share']].copy()
    nov_tub.columns = ['city', 'pre_tub_share']
    panel = panel.merge(nov_tub, on='city', how='left')

    # Handle cities without Nov data
    # Use average TUB share from Sep-Nov as fallback
    if panel['pre_tub_share'].isna().any():
        pre_period = panel[panel['month_year'].isin(['2023-09', '2023-10', '2023-11'])]
        fallback = pre_period.groupby('city')['tub_share'].mean().reset_index()
        fallback.columns = ['city', 'fallback_tub']
        panel = panel.merge(fallback, on='city', how='left')
        panel['pre_tub_share'] = panel['pre_tub_share'].fillna(panel['fallback_tub'])
        panel.drop(columns=['fallback_tub'], inplace=True)

    # Continuous treatment: interaction of post x pre_tub_share
    panel['treatment_intensity'] = panel['post'] * panel['pre_tub_share']

    # Log transform outcome for robustness
    panel['log_speeding'] = np.log(panel['overall_speeding'].clip(lower=1e-6))

    # City and month fixed effects (as categories)
    panel['city_fe'] = pd.Categorical(panel['city'])
    panel['month_fe'] = pd.Categorical(panel['month_year'])

    # Filter to cities with enough observations (at least 1000 trips in most months)
    city_min_trips = panel.groupby('city')['total_trips'].min()
    valid_cities = city_min_trips[city_min_trips >= 100].index
    panel = panel[panel['city'].isin(valid_cities)].copy()

    print(f"Panel: {panel['city'].nunique()} cities x {panel['month_year'].nunique()} months "
          f"= {len(panel)} obs")
    return panel


def model_1_cross_sectional(panel: pd.DataFrame) -> dict:
    """Model 1: Simple cross-sectional pre-post regression.

    Speeding_change_i = alpha + beta * TUB_share_pre_i + epsilon_i
    """
    print("\n--- Model 1: Cross-Sectional Pre-Post ---")

    # Build cross-section: Nov vs Dec per city
    nov = panel[panel['month_year'] == '2023-11'][
        ['city', 'overall_speeding', 'tub_share', 'eco_share', 'total_trips']
    ].copy()
    nov.columns = ['city', 'nov_speeding', 'nov_tub', 'nov_eco', 'nov_trips']

    dec = panel[panel['month_year'] == '2023-12'][
        ['city', 'overall_speeding', 'tub_share', 'eco_share', 'total_trips']
    ].copy()
    dec.columns = ['city', 'dec_speeding', 'dec_tub', 'dec_eco', 'dec_trips']

    cs = nov.merge(dec, on='city')
    cs['speeding_change'] = cs['dec_speeding'] - cs['nov_speeding']
    cs['tub_change'] = cs['dec_tub'] - cs['nov_tub']
    cs['speeding_change_pp'] = cs['speeding_change'] * 100  # percentage points

    # Weighted by trips
    weights = np.sqrt(cs['nov_trips'])

    # OLS: speeding_change ~ pre_tub_share
    X = sm.add_constant(cs['nov_tub'])
    model = sm.WLS(cs['speeding_change_pp'], X, weights=weights).fit(
        cov_type='HC1'
    )
    print(model.summary().tables[1])

    result = {
        'n_cities': len(cs),
        'beta_tub_share': float(model.params.iloc[1]),
        'se_tub_share': float(model.bse.iloc[1]),
        'pvalue_tub_share': float(model.pvalues.iloc[1]),
        'r_squared': float(model.rsquared),
        'interpretation': (
            f"A 10pp higher pre-ban TUB share is associated with "
            f"{abs(model.params.iloc[1] * 0.10):.1f}pp larger speeding reduction"
        ),
    }
    print(f"  R2 = {model.rsquared:.3f}")
    print(f"  {result['interpretation']}")
    return result, cs


def model_2_twfe(panel: pd.DataFrame) -> dict:
    """Model 2: Two-Way Fixed Effects panel DiD.

    Speeding_ct = alpha_c + gamma_t + beta * (Post_t x TUB_share_c) + epsilon_ct
    """
    print("\n--- Model 2: Two-Way Fixed Effects DiD ---")

    # TWFE with city + month FE
    formula = 'overall_speeding ~ treatment_intensity + C(city_fe) + C(month_fe)'
    model = smf.wls(
        formula, data=panel,
        weights=np.sqrt(panel['total_trips'])
    ).fit(cov_type='HC1')

    # Extract treatment coefficient
    coef = model.params['treatment_intensity']
    se = model.bse['treatment_intensity']
    pval = model.pvalues['treatment_intensity']
    ci_lo = model.conf_int().loc['treatment_intensity', 0]
    ci_hi = model.conf_int().loc['treatment_intensity', 1]

    result = {
        'beta_treatment': float(coef),
        'se': float(se),
        'pvalue': float(pval),
        'ci_95': [float(ci_lo), float(ci_hi)],
        'r_squared': float(model.rsquared),
        'n_obs': int(model.nobs),
        'interpretation': (
            f"TWFE DiD: Post x TUB_share coefficient = {coef:.4f} "
            f"(SE={se:.4f}, p={pval:.2e}). "
            f"A city with 50% pre-ban TUB share sees {abs(coef * 0.5) * 100:.1f}pp "
            f"speeding reduction due to the ban."
        ),
    }
    print(f"  beta(Post x TUB_share) = {coef:.4f} (SE={se:.4f}, p={pval:.2e})")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  R2 = {model.rsquared:.3f}")
    print(f"  {result['interpretation']}")
    return result


def model_3_event_study(panel: pd.DataFrame) -> dict:
    """Model 3: Event study with monthly leads and lags.

    Speeding_ct = alpha_c + gamma_t + sum_k(beta_k * 1(t=k) * TUB_share_c) + epsilon_ct

    Reference period: November 2023 (k=-1).
    """
    print("\n--- Model 3: Event Study ---")

    # Create relative time variable (Dec = 0, Nov = -1, Oct = -2, etc.)
    month_map = {
        '2023-02': -9, '2023-03': -8, '2023-04': -7, '2023-05': -6,
        '2023-06': -5, '2023-07': -4, '2023-08': -3, '2023-09': -2,
        '2023-10': -1, '2023-11': 0, '2023-12': 1
    }
    # Using Nov as reference (k=0) and Dec as post (k=1)
    # Actually, let's use Oct (k=-1) as the omitted reference
    # and let Nov (k=0) and Dec (k=1) show the effect
    # Better: use standard pre-treatment as reference
    ref_month = '2023-10'  # Last pre-treatment month we'll omit

    panel_es = panel.copy()
    panel_es['rel_time'] = panel_es['month_year'].map(month_map)

    # Create interaction terms for each period (except reference)
    event_dummies = {}
    for month_str, k in month_map.items():
        if month_str == ref_month:
            continue
        col = f'evt_{abs(k)}{"post" if k > 0 else "pre" if k < 0 else "ref"}'
        panel_es[col] = (panel_es['month_year'] == month_str).astype(float) * panel_es['pre_tub_share']
        event_dummies[col] = k

    # Drop rows with missing pre_tub_share (cities without Nov data)
    panel_es = panel_es.dropna(subset=['pre_tub_share', 'overall_speeding']).copy()

    # Use manual matrix construction to avoid patsy naming issues
    y = panel_es['overall_speeding'].values
    weights = np.sqrt(panel_es['total_trips'].values)

    # Event study interaction columns
    evt_names = list(event_dummies.keys())
    X_evt = panel_es[evt_names].values

    # City fixed effects (dummies, drop first)
    city_dummies = pd.get_dummies(panel_es['city'], prefix='city', drop_first=True)
    # Month fixed effects (dummies, drop first)
    month_dummies = pd.get_dummies(panel_es['month_year'], prefix='month', drop_first=True)

    X = np.column_stack([
        np.ones(len(panel_es)),  # constant
        X_evt,
        city_dummies.values,
        month_dummies.values,
    ])
    col_names = ['const'] + evt_names + city_dummies.columns.tolist() + month_dummies.columns.tolist()

    # Verify no NaN/inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(weights)
    X, y, weights = X[mask], y[mask], weights[mask]
    print(f"  Event study: {X.shape[0]} obs, {X.shape[1]} regressors")

    model = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')

    # Extract event study coefficients
    # Map column indices to names
    param_series = pd.Series(model.params, index=col_names)
    bse_series = pd.Series(model.bse, index=col_names)
    pval_series = pd.Series(model.pvalues, index=col_names)
    ci = model.conf_int()

    es_results = {}
    for i, (col, k) in enumerate(event_dummies.items()):
        col_idx = 1 + list(event_dummies.keys()).index(col)  # +1 for constant
        es_results[k] = {
            'coef': float(model.params[col_idx]),
            'se': float(model.bse[col_idx]),
            'pvalue': float(model.pvalues[col_idx]),
            'ci_lo': float(ci[col_idx, 0]),
            'ci_hi': float(ci[col_idx, 1]),
        }

    # Check pre-trends
    pre_coefs = [v['coef'] for k, v in es_results.items() if k < -1]
    pre_pvals = [v['pvalue'] for k, v in es_results.items() if k < -1]
    pre_trend_significant = any(p < 0.05 for p in pre_pvals)

    result = {
        'event_study_coefs': es_results,
        'pre_trend_significant': pre_trend_significant,
        'post_coef': es_results.get(1, {}),
        'reference_month': ref_month,
        'interpretation': (
            f"Event study: Pre-treatment coefficients "
            f"{'FAIL' if pre_trend_significant else 'PASS'} parallel trends test "
            f"(any p<0.05: {pre_trend_significant}). "
            f"Post-treatment (Dec) coefficient: {es_results.get(1, {}).get('coef', 'N/A'):.4f}"
        ),
    }

    print(f"  Event study coefficients (interaction with pre-TUB share):")
    for k in sorted(es_results.keys()):
        v = es_results[k]
        sig = '***' if v['pvalue'] < 0.001 else '**' if v['pvalue'] < 0.01 else '*' if v['pvalue'] < 0.05 else ''
        marker = ' <-- POST' if k == 1 else (' <-- REF' if k == -1 else '')
        print(f"    k={k:+2d}: {v['coef']:+.4f} (SE={v['se']:.4f}) {sig}{marker}")
    print(f"  Pre-trends significant? {pre_trend_significant}")
    print(f"  {result['interpretation']}")

    return result, es_results


def model_4_dose_response(panel: pd.DataFrame) -> dict:
    """Model 4: Dose-response with continuous treatment intensity.

    Estimate the marginal effect of TUB share reduction on speeding reduction.
    """
    print("\n--- Model 4: Dose-Response (Nov vs Dec) ---")

    # Build Nov-Dec cross section
    nov = panel[panel['month_year'] == '2023-11'][
        ['city', 'overall_speeding', 'tub_share', 'eco_share', 'total_trips']
    ].copy()
    nov.columns = ['city', 'nov_speeding', 'nov_tub', 'nov_eco', 'nov_trips']

    dec = panel[panel['month_year'] == '2023-12'][
        ['city', 'overall_speeding', 'tub_share', 'eco_share', 'total_trips']
    ].copy()
    dec.columns = ['city', 'dec_speeding', 'dec_tub', 'dec_eco', 'dec_trips']

    cs = nov.merge(dec, on='city')
    cs['tub_reduction'] = cs['nov_tub'] - cs['dec_tub']  # positive = large reduction
    cs['speeding_reduction'] = cs['nov_speeding'] - cs['dec_speeding']  # positive = improvement

    # OLS: speeding_reduction ~ tub_reduction
    X = sm.add_constant(cs['tub_reduction'])
    weights = np.sqrt(cs['nov_trips'])
    model = sm.WLS(cs['speeding_reduction'] * 100, X, weights=weights).fit(cov_type='HC1')

    print(model.summary().tables[1])

    # Per 10pp TUB reduction, how many pp speeding reduction?
    beta = model.params.iloc[1]
    result = {
        'beta_dose': float(beta),
        'se': float(model.bse.iloc[1]),
        'pvalue': float(model.pvalues.iloc[1]),
        'r_squared': float(model.rsquared),
        'interpretation': (
            f"Dose-response: each 10pp TUB share reduction causes "
            f"{abs(beta * 0.10):.1f}pp speeding rate reduction "
            f"(beta={beta:.2f}, SE={model.bse.iloc[1]:.2f}, p={model.pvalues.iloc[1]:.2e})"
        ),
    }
    print(f"  {result['interpretation']}")
    return result, cs


def generate_figures(panel: pd.DataFrame, cs_data: pd.DataFrame,
                     es_results: dict) -> None:
    """Generate publication-quality DiD figures."""
    print("\n[Figures] Generating DiD analysis plots...")

    # ---- Figure 1: Event Study Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))

    ks = sorted(es_results.keys())
    coefs = [es_results[k]['coef'] for k in ks]
    ci_los = [es_results[k]['ci_lo'] for k in ks]
    ci_his = [es_results[k]['ci_hi'] for k in ks]

    # Insert the reference point (k=-1, coef=0)
    ref_k = -1
    ks_full = sorted(list(ks) + [ref_k]) if ref_k not in ks else ks
    coefs_full = []
    ci_lo_full = []
    ci_hi_full = []
    for k in ks_full:
        if k == ref_k and ref_k not in es_results:
            coefs_full.append(0)
            ci_lo_full.append(0)
            ci_hi_full.append(0)
        else:
            coefs_full.append(es_results[k]['coef'])
            ci_lo_full.append(es_results[k]['ci_lo'])
            ci_hi_full.append(es_results[k]['ci_hi'])

    ax.errorbar(ks_full, coefs_full,
                yerr=[np.array(coefs_full) - np.array(ci_lo_full),
                      np.array(ci_hi_full) - np.array(coefs_full)],
                fmt='o-', color='#1f77b4', capsize=4, capthick=1.5,
                linewidth=2, markersize=7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
               label='TUB ban (Dec 2023)')
    ax.fill_between([-0.5, max(ks_full) + 0.5], -0.5, 0.5,
                     color='red', alpha=0.05)

    # Month labels
    month_labels = {
        -9: 'Feb', -8: 'Mar', -7: 'Apr', -6: 'May', -5: 'Jun',
        -4: 'Jul', -3: 'Aug', -2: 'Sep', -1: 'Oct\n(ref)', 0: 'Nov', 1: 'Dec'
    }
    ax.set_xticks(ks_full)
    ax.set_xticklabels([month_labels.get(k, str(k)) for k in ks_full], fontsize=9)

    ax.set_xlabel('Months relative to TUB ban', fontsize=11)
    ax.set_ylabel('Coefficient (interaction with pre-TUB share)', fontsize=11)
    ax.set_title('Event Study: Effect of TUB Mode Restriction on Speeding', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_event_study.pdf', dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_event_study.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_event_study.pdf/png")

    # ---- Figure 2: Dose-Response Scatter ----
    fig, ax = plt.subplots(figsize=(7, 6))

    cs_data['tub_reduction'] = cs_data['nov_tub'] - cs_data['dec_tub']
    cs_data['speeding_reduction_pp'] = (cs_data['nov_speeding'] - cs_data['dec_speeding']) * 100

    # Size by trip count
    sizes = np.sqrt(cs_data['nov_trips']) / 2
    sizes = sizes.clip(upper=200)

    ax.scatter(cs_data['tub_reduction'] * 100, cs_data['speeding_reduction_pp'],
              s=sizes, alpha=0.7, c='#d62728', edgecolors='black', linewidth=0.5)

    # Best fit line (WLS with trip-count weights, matching paper text)
    X_wls = sm.add_constant(cs_data['tub_reduction'] * 100)
    wls_model = sm.WLS(cs_data['speeding_reduction_pp'], X_wls,
                       weights=cs_data['nov_trips']).fit(cov_type='HC1')
    slope_wls = wls_model.params.iloc[1]
    x_fit = np.linspace(cs_data['tub_reduction'].min() * 100,
                        cs_data['tub_reduction'].max() * 100, 100)
    y_fit = wls_model.params.iloc[0] + slope_wls * x_fit
    ax.plot(x_fit, y_fit, 'k--', linewidth=2, alpha=0.7,
            label=f'WLS fit (slope={slope_wls:.2f})')

    # Label notable cities
    for _, row in cs_data.iterrows():
        if row['tub_reduction'] > 0.65 or row['speeding_reduction_pp'] > 30:
            ax.annotate(row['city'],
                       (row['tub_reduction'] * 100, row['speeding_reduction_pp']),
                       fontsize=7, ha='left', va='bottom')

    r, p = stats.pearsonr(cs_data['tub_reduction'], cs_data['speeding_reduction_pp'])
    ax.text(0.05, 0.95, f'r = {r:.3f}, p = {p:.2e}\nn = {len(cs_data)} cities',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('TUB share reduction (pp, Nov -> Dec 2023)', fontsize=11)
    ax.set_ylabel('Speeding rate reduction (pp)', fontsize=11)
    ax.set_title('Dose-Response: TUB Restriction and Speeding Reduction', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_dose_response.pdf', dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_dose_response.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_dose_response.pdf/png")

    # ---- Figure 3: Parallel trends (top 6 cities) ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    top6 = panel.groupby('city')['total_trips'].sum().nlargest(6).index.tolist()

    for idx, city in enumerate(top6):
        ax = axes[idx // 3, idx % 3]
        city_data = panel[panel['city'] == city].sort_values('month_year')
        months = city_data['month_year'].tolist()
        x = range(len(months))

        ax.plot(x, city_data['overall_speeding'] * 100, 'o-', color='#1f77b4',
                linewidth=2, markersize=5)
        ax.axvline(x=len(months) - 1.5, color='red', linestyle='--', alpha=0.7)

        # Add TUB share as secondary axis
        ax2 = ax.twinx()
        ax2.fill_between(x, 0, city_data['tub_share'] * 100,
                        alpha=0.15, color='#d62728', label='TUB share')
        ax2.set_ylim(0, 80)
        if idx % 3 == 2:
            ax2.set_ylabel('TUB share (%)', fontsize=9, color='#d62728')
        else:
            ax2.set_yticklabels([])

        ax.set_title(f'{city} (n={city_data["total_trips"].sum():,.0f})',
                    fontsize=10, fontweight='bold')
        ax.set_ylabel('Speeding rate (%)' if idx % 3 == 0 else '', fontsize=9)
        ax.set_xticks(x)
        month_abbr = {'02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May',
                      '06': 'Jun', '07': 'Jul', '08': 'Aug', '09': 'Sep',
                      '10': 'Oct', '11': 'Nov', '12': 'Dec'}
        ax.set_xticklabels([month_abbr.get(m[-2:], m[-2:]) for m in months], fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('City-Level Speeding Trends and TUB Share (Feb-Dec 2023)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_parallel_trends.pdf', dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_parallel_trends.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_parallel_trends.pdf/png")

    # Note: Summary coefficient plot is generated in main() with actual model results


def main() -> dict:
    """Run all DiD models and generate outputs."""
    print("=" * 60)
    print("DiD ANALYSIS: December 2023 TUB Mode Restriction")
    print("=" * 60)

    # Build panel
    panel = build_panel()

    # Run models
    m1_result, cs_data = model_1_cross_sectional(panel)
    m2_result = model_2_twfe(panel)
    m3_result, es_results = model_3_event_study(panel)
    m4_result, cs_dose = model_4_dose_response(panel)

    # Generate figures
    generate_figures(panel, cs_data, es_results)

    # Generate summary coefficient plot — ALL normalized to "per 10pp TUB share"
    fig, ax = plt.subplots(figsize=(9, 4))

    # Normalize TWFE and event study to per-10pp scale
    # TWFE beta is per unit (100pp), so multiply by 0.10 and convert to pp
    twfe_per10 = m2_result['beta_treatment'] * 0.10 * 100  # per 10pp, in pp
    twfe_se10 = m2_result['se'] * 0.10 * 100 * 1.96
    # Event study: same normalization
    es_coef = m3_result['post_coef'].get('coef', 0)
    es_ci_hi = m3_result['post_coef'].get('ci_hi', 0)
    es_per10 = es_coef * 0.10 * 100
    es_se10 = (es_ci_hi - es_coef) * 0.10 * 100

    estimates = [
        ('Cross-sectional OLS',
         m1_result['beta_tub_share'] * 0.10,
         m1_result['se_tub_share'] * 0.10 * 1.96),
        ('TWFE DiD\n(city + month FE)',
         twfe_per10, twfe_se10),
        ('Event Study\n(Dec 2023, k=+1)',
         es_per10, es_se10),
        ('Dose-Response\n(Nov->Dec change)',
         m4_result['beta_dose'] * 0.10,
         m4_result['se'] * 0.10 * 1.96),
    ]

    y_pos = range(len(estimates))
    labels = [e[0] for e in estimates]
    coefs = [e[1] for e in estimates]
    cis = [e[2] for e in estimates]

    ax.barh(y_pos, coefs, xerr=cis, height=0.5,
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            edgecolor='black', linewidth=0.5, capsize=5, alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Speeding rate change per 10pp TUB share (pp)', fontsize=11)
    ax.set_title('Summary: Causal Effect of TUB Restriction on Speeding\n'
                 '(all estimates normalized to per 10pp TUB share)', fontsize=11)
    ax.grid(alpha=0.3, axis='x')

    for i, (coef, ci) in enumerate(zip(coefs, cis)):
        ax.text(coef - ci - 0.3 if coef < 0 else coef + ci + 0.3, i,
                f'{coef:.1f}pp', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_summary.pdf', dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_summary.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_summary.pdf/png")

    # Compile results
    all_results = {
        'model_1_cross_sectional': m1_result,
        'model_2_twfe': m2_result,
        'model_3_event_study': m3_result,
        'model_4_dose_response': m4_result,
        'panel_stats': {
            'n_cities': int(panel['city'].nunique()),
            'n_months': int(panel['month_year'].nunique()),
            'n_obs': int(len(panel)),
            'mean_pre_tub_share': float(panel['pre_tub_share'].mean()),
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Model 1 (Cross-sectional): R2={m1_result['r_squared']:.3f}")
    print(f"    {m1_result['interpretation']}")
    print(f"  Model 2 (TWFE DiD): R2={m2_result['r_squared']:.3f}")
    print(f"    {m2_result['interpretation']}")
    print(f"  Model 3 (Event Study): Pre-trends {'FAIL' if m3_result['pre_trend_significant'] else 'PASS'}")
    print(f"    {m3_result['interpretation']}")
    print(f"  Model 4 (Dose-Response): R2={m4_result['r_squared']:.3f}")
    print(f"    {m4_result['interpretation']}")

    # Save
    output_path = MODELING_DIR / 'did_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved results to {output_path}")

    return all_results


if __name__ == '__main__':
    results = main()
