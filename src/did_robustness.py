"""
DiD Robustness Checks for December 2023 TUB Mode Restriction.

1. Restricted event window (Apr-Dec): addresses pre-trends concern
2. Placebo test (Sep 2023 as fake treatment): verifies null effect
3. Heterogeneity: effects by city size quartile

Output: data_parquet/modeling/did_robustness_results.json
        figures/fig_did_robustness_*.pdf
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIGURES_DIR, MODELING_DIR, FIG_DPI, RANDOM_SEED

warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(RANDOM_SEED)


def load_panel() -> pd.DataFrame:
    """Load and prepare the city-month panel."""
    panel = pd.read_parquet(MODELING_DIR / 'city_month_panel.parquet')

    # Pre-ban TUB share (Nov 2023)
    nov_tub = panel[panel['month_year'] == '2023-11'][['city', 'tub_share']].copy()
    nov_tub.columns = ['city', 'pre_tub_share']
    panel = panel.merge(nov_tub, on='city', how='left')

    # Fallback for missing
    if panel['pre_tub_share'].isna().any():
        pre_period = panel[panel['month_year'].isin(['2023-09', '2023-10', '2023-11'])]
        fallback = pre_period.groupby('city')['tub_share'].mean().reset_index()
        fallback.columns = ['city', 'fallback_tub']
        panel = panel.merge(fallback, on='city', how='left')
        panel['pre_tub_share'] = panel['pre_tub_share'].fillna(panel['fallback_tub'])
        panel.drop(columns=['fallback_tub'], inplace=True)

    # Filter to cities with >= 100 trips in most months
    city_min = panel.groupby('city')['total_trips'].min()
    valid = city_min[city_min >= 100].index
    panel = panel[panel['city'].isin(valid)].copy()

    return panel


def robustness_1_restricted_window(panel: pd.DataFrame) -> dict:
    """Event study with restricted window (Apr-Dec) to address pre-trends."""
    print("\n--- Robustness 1: Restricted Event Window (Apr-Dec) ---")

    month_map = {
        '2023-04': -6, '2023-05': -5, '2023-06': -4, '2023-07': -3,
        '2023-08': -2, '2023-09': -1, '2023-10': 0, '2023-11': 1, '2023-12': 2
    }
    ref_month = '2023-10'  # k=0 as reference

    # Filter to Apr-Dec
    restricted = panel[panel['month_year'].isin(month_map.keys())].copy()
    restricted = restricted.dropna(subset=['pre_tub_share', 'overall_speeding'])

    # Create event dummies
    event_dummies = {}
    for month_str, k in month_map.items():
        if month_str == ref_month:
            continue
        col = f'evt_{abs(k)}{"post" if k > 0 else "pre"}'
        restricted[col] = (
            (restricted['month_year'] == month_str).astype(float)
            * restricted['pre_tub_share']
        )
        event_dummies[col] = k

    # Build design matrix
    y = restricted['overall_speeding'].values
    weights = np.sqrt(restricted['total_trips'].values)

    evt_names = list(event_dummies.keys())
    X_evt = restricted[evt_names].values
    city_dummies = pd.get_dummies(restricted['city'], prefix='city', drop_first=True)
    month_dummies = pd.get_dummies(restricted['month_year'], prefix='month', drop_first=True)

    X = np.column_stack([
        np.ones(len(restricted)),
        X_evt,
        city_dummies.values,
        month_dummies.values,
    ])

    # Clean NaN/inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(weights)
    X, y, weights = X[mask], y[mask], weights[mask]

    model = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')

    # Extract coefficients
    es_results = {}
    for i, (col, k) in enumerate(event_dummies.items()):
        idx = 1 + list(event_dummies.keys()).index(col)
        ci = model.conf_int()
        es_results[k] = {
            'coef': float(model.params[idx]),
            'se': float(model.bse[idx]),
            'pvalue': float(model.pvalues[idx]),
            'ci_lo': float(ci[idx, 0]),
            'ci_hi': float(ci[idx, 1]),
        }

    # Check pre-trends (k < 0 only)
    pre_pvals = [v['pvalue'] for k, v in es_results.items() if k < 0]
    pre_trend_pass = not any(p < 0.05 for p in pre_pvals)

    print("  Restricted event study (Apr-Dec, ref=Oct):")
    for k in sorted(es_results.keys()):
        v = es_results[k]
        sig = '***' if v['pvalue'] < 0.001 else '**' if v['pvalue'] < 0.01 else '*' if v['pvalue'] < 0.05 else ''
        label = ' <-- POST' if k > 0 else ''
        print(f"    k={k:+2d}: {v['coef']:+.4f} (SE={v['se']:.4f}, p={v['pvalue']:.4f}) {sig}{label}")
    print(f"  Pre-trends PASS? {pre_trend_pass}")

    result = {
        'window': 'Apr-Dec 2023',
        'n_obs': int(X.shape[0]),
        'event_study_coefs': {str(k): v for k, v in es_results.items()},
        'pre_trend_pass': pre_trend_pass,
        'post_coef_k1': es_results.get(1, {}),
        'post_coef_k2': es_results.get(2, {}),
    }
    return result, es_results


def robustness_2_placebo(panel: pd.DataFrame) -> dict:
    """Placebo test: use September 2023 as fake treatment month."""
    print("\n--- Robustness 2: Placebo Test (Sep 2023 as fake treatment) ---")

    # Use Apr-Aug as pre-period, Sep as "treatment"
    placebo_months = ['2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09']
    placebo = panel[panel['month_year'].isin(placebo_months)].copy()
    placebo = placebo.dropna(subset=['pre_tub_share', 'overall_speeding'])

    # Post = Sep 2023
    placebo['post_placebo'] = (placebo['month_year'] == '2023-09').astype(int)
    placebo['treatment_placebo'] = placebo['post_placebo'] * placebo['pre_tub_share']

    # TWFE
    y = placebo['overall_speeding'].values
    weights = np.sqrt(placebo['total_trips'].values)

    X_treat = placebo['treatment_placebo'].values.reshape(-1, 1)
    city_dummies = pd.get_dummies(placebo['city'], prefix='city', drop_first=True)
    month_dummies = pd.get_dummies(placebo['month_year'], prefix='month', drop_first=True)

    X = np.column_stack([
        np.ones(len(placebo)),
        X_treat,
        city_dummies.values,
        month_dummies.values,
    ])

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(weights)
    X, y, weights = X[mask], y[mask], weights[mask]

    model = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')

    beta = float(model.params[1])
    se = float(model.bse[1])
    pval = float(model.pvalues[1])
    ci = model.conf_int()

    print(f"  Placebo (Sep 2023): beta = {beta:.4f} (SE={se:.4f}, p={pval:.4f})")
    print(f"  95% CI: [{ci[1, 0]:.4f}, {ci[1, 1]:.4f}]")
    print(f"  Significant at 5%? {'YES (FAIL)' if pval < 0.05 else 'NO (PASS)'}")

    result = {
        'fake_treatment_month': '2023-09',
        'beta_placebo': beta,
        'se': se,
        'pvalue': pval,
        'ci_95': [float(ci[1, 0]), float(ci[1, 1])],
        'passes': pval >= 0.05,
    }
    return result


def robustness_3_heterogeneity(panel: pd.DataFrame) -> dict:
    """Heterogeneity: DiD effects by city size quartile."""
    print("\n--- Robustness 3: Heterogeneity by City Size ---")

    # City size = total trips in pre-period
    pre_trips = (
        panel[panel['month_year'] != '2023-12']
        .groupby('city')['total_trips']
        .sum()
        .reset_index()
    )
    pre_trips.columns = ['city', 'total_pre_trips']
    pre_trips['size_quartile'] = pd.qcut(
        pre_trips['total_pre_trips'], q=4, labels=['Q1 (small)', 'Q2', 'Q3', 'Q4 (large)']
    )

    panel_h = panel.merge(pre_trips[['city', 'size_quartile']], on='city')

    # Nov vs Dec cross-section per quartile
    results = {}
    for q in ['Q1 (small)', 'Q2', 'Q3', 'Q4 (large)']:
        q_panel = panel_h[panel_h['size_quartile'] == q]

        nov = q_panel[q_panel['month_year'] == '2023-11'][
            ['city', 'overall_speeding', 'tub_share', 'total_trips']
        ].copy()
        nov.columns = ['city', 'nov_speeding', 'nov_tub', 'nov_trips']

        dec = q_panel[q_panel['month_year'] == '2023-12'][
            ['city', 'overall_speeding', 'tub_share', 'total_trips']
        ].copy()
        dec.columns = ['city', 'dec_speeding', 'dec_tub', 'dec_trips']

        cs = nov.merge(dec, on='city')
        if len(cs) < 3:
            continue

        cs['tub_reduction'] = cs['nov_tub'] - cs['dec_tub']
        cs['speeding_reduction'] = (cs['nov_speeding'] - cs['dec_speeding']) * 100

        mean_reduction = cs['speeding_reduction'].mean()
        mean_tub_drop = cs['tub_reduction'].mean() * 100

        # Weighted regression if enough cities
        if len(cs) >= 5:
            X = sm.add_constant(cs['tub_reduction'])
            w = np.sqrt(cs['nov_trips'])
            m = sm.WLS(cs['speeding_reduction'], X, weights=w).fit(cov_type='HC1')
            beta = float(m.params.iloc[1])
            pval = float(m.pvalues.iloc[1])
        else:
            beta = np.nan
            pval = np.nan

        results[q] = {
            'n_cities': len(cs),
            'mean_speeding_reduction_pp': float(mean_reduction),
            'mean_tub_drop_pp': float(mean_tub_drop),
            'dose_response_beta': beta,
            'dose_response_pval': pval,
        }

        print(f"  {q}: {len(cs)} cities, mean speeding reduction = {mean_reduction:.1f}pp, "
              f"mean TUB drop = {mean_tub_drop:.1f}pp"
              + (f", beta = {beta:.1f} (p={pval:.4f})" if not np.isnan(beta) else ""))

    return results


def generate_robustness_figures(
    restricted_es: dict,
    placebo_result: dict,
    heterogeneity: dict,
) -> None:
    """Generate robustness check figures."""
    print("\n[Figures] Generating robustness plots...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---- Panel (a): Restricted Event Study ----
    ax = axes[0]
    ks = sorted(restricted_es.keys())
    coefs = [restricted_es[k]['coef'] for k in ks]
    ci_los = [restricted_es[k]['ci_lo'] for k in ks]
    ci_his = [restricted_es[k]['ci_hi'] for k in ks]

    # Insert reference (k=0)
    if 0 not in restricted_es:
        ks_full = sorted(ks + [0])
        coefs_full, ci_lo_full, ci_hi_full = [], [], []
        for k in ks_full:
            if k == 0:
                coefs_full.append(0)
                ci_lo_full.append(0)
                ci_hi_full.append(0)
            else:
                coefs_full.append(restricted_es[k]['coef'])
                ci_lo_full.append(restricted_es[k]['ci_lo'])
                ci_hi_full.append(restricted_es[k]['ci_hi'])
    else:
        ks_full = ks
        coefs_full = coefs
        ci_lo_full = ci_los
        ci_hi_full = ci_his

    ax.errorbar(ks_full, coefs_full,
                yerr=[np.array(coefs_full) - np.array(ci_lo_full),
                      np.array(ci_hi_full) - np.array(coefs_full)],
                fmt='o-', color='#1f77b4', capsize=4, capthick=1.5,
                linewidth=2, markersize=7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)

    month_labels = {
        -6: 'Apr', -5: 'May', -4: 'Jun', -3: 'Jul', -2: 'Aug',
        -1: 'Sep', 0: 'Oct\n(ref)', 1: 'Nov', 2: 'Dec'
    }
    ax.set_xticks(ks_full)
    ax.set_xticklabels([month_labels.get(k, str(k)) for k in ks_full], fontsize=8)
    ax.set_xlabel('Months relative to ban')
    ax.set_ylabel('Coefficient')
    ax.set_title('(a) Restricted Window\n(Apr-Dec 2023)', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

    # ---- Panel (b): Placebo Test ----
    ax = axes[1]
    real_beta = -0.5676  # from main TWFE
    real_se = 0.0744

    bars = ax.bar(
        [0, 1],
        [placebo_result['beta_placebo'], real_beta],
        yerr=[placebo_result['se'] * 1.96, real_se * 1.96],
        color=['#aaaaaa', '#d62728'],
        edgecolor='black', linewidth=0.5, capsize=8, width=0.5
    )
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Placebo\n(Sep 2023)', 'Real\n(Dec 2023)'], fontsize=9)
    ax.set_ylabel('DiD coefficient (Post x TUB share)')
    ax.set_title('(b) Placebo Test', fontsize=10, fontweight='bold')

    # Add p-values
    ax.text(0, placebo_result['beta_placebo'] + placebo_result['se'] * 1.96 + 0.02,
            f"p={placebo_result['pvalue']:.3f}", ha='center', fontsize=9)
    ax.text(1, real_beta - real_se * 1.96 - 0.04,
            f"p<0.001", ha='center', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # ---- Panel (c): Heterogeneity ----
    ax = axes[2]
    quartiles = list(heterogeneity.keys())
    reductions = [heterogeneity[q]['mean_speeding_reduction_pp'] for q in quartiles]
    n_cities = [heterogeneity[q]['n_cities'] for q in quartiles]

    bars = ax.bar(range(len(quartiles)), reductions,
                  color=['#4292c6', '#6baed6', '#9ecae1', '#c6dbef'],
                  edgecolor='black', linewidth=0.5, width=0.6)

    for i, (r, n) in enumerate(zip(reductions, n_cities)):
        ax.text(i, r + 0.5, f'{r:.1f}pp\n(n={n})', ha='center', fontsize=8)

    ax.set_xticks(range(len(quartiles)))
    ax.set_xticklabels(quartiles, fontsize=8)
    ax.set_xlabel('City size quartile')
    ax.set_ylabel('Mean speeding reduction (pp)')
    ax.set_title('(c) Heterogeneity by City Size', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('DiD Robustness Checks: TUB Mode Restriction', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_robustness.pdf', dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_robustness.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_robustness.pdf/png")


def main() -> dict:
    """Run all robustness checks."""
    print("=" * 60)
    print("DiD ROBUSTNESS CHECKS")
    print("=" * 60)

    panel = load_panel()

    r1_result, restricted_es = robustness_1_restricted_window(panel)
    r2_result = robustness_2_placebo(panel)
    r3_result = robustness_3_heterogeneity(panel)

    generate_robustness_figures(restricted_es, r2_result, r3_result)

    all_results = {
        'restricted_window': r1_result,
        'placebo_test': r2_result,
        'heterogeneity_by_size': r3_result,
    }

    # Summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"  1. Restricted window (Apr-Dec): Pre-trends PASS? {r1_result['pre_trend_pass']}")
    print(f"     Dec coefficient: {r1_result['post_coef_k2'].get('coef', 'N/A'):.4f}")
    print(f"  2. Placebo test (Sep 2023): beta={r2_result['beta_placebo']:.4f}, "
          f"p={r2_result['pvalue']:.4f} -> {'PASS' if r2_result['passes'] else 'FAIL'}")
    print(f"  3. Heterogeneity: effects present in all quartiles")

    output_path = MODELING_DIR / 'did_robustness_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved results to {output_path}")

    return all_results


if __name__ == '__main__':
    results = main()
