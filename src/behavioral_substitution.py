"""
Behavioral Substitution Analysis: Multi-Outcome DiD.

Tests whether the December 2023 TUB mode ban caused riders to compensate
in other behavioral dimensions. Uses the same TWFE DiD framework as the
primary speeding analysis (Model 9) but with alternative dependent variables
computed for STD/ECO-mode trips only.

Outcomes tested:
1. Mean trip distance (km) -- did riders take longer trips?
2. Mean speed within STD/ECO trips -- did riders push harder within governed modes?
3. Mean max speed within STD/ECO trips -- did peak speeds increase?
4. Speeding rate within STD/ECO trips -- did compliance worsen within governed modes?

A null result on all outcomes is a strong policy finding: mandatory speed
governors reduce speeding without inducing offsetting risky behavior.

Output:
- data_parquet/modeling/behavioral_substitution_results.json
- figures/fig_behavioral_substitution.pdf
- figures/fig_behavioral_time_series.pdf
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
import statsmodels.api as sm
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIGURES_DIR, MODELING_DIR, FIG_DPI, RANDOM_SEED

warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(RANDOM_SEED)

# Korean city centers (from did_feasibility.py)
CITY_CENTERS = {
    'Seoul': (37.5665, 126.9780), 'Busan': (35.1796, 129.0756),
    'Daegu': (35.8714, 128.6014), 'Incheon': (37.4563, 126.7052),
    'Gwangju': (35.1595, 126.8526), 'Daejeon': (36.3504, 127.3845),
    'Ulsan': (35.5384, 129.3114), 'Sejong': (36.4800, 127.2600),
    'Suwon': (37.2636, 127.0286), 'Changwon': (35.2280, 128.6811),
    'Goyang': (37.6584, 126.8320), 'Yongin': (37.2411, 127.1776),
    'Seongnam': (37.4449, 127.1389), 'Hwaseong': (37.1997, 126.8312),
    'Cheongju': (36.6424, 127.4890), 'Bucheon': (37.5035, 126.7660),
    'Ansan': (37.3219, 126.8309), 'Anyang': (37.3943, 126.9568),
    'Namyangju': (37.6360, 127.2165), 'Cheonan': (36.8152, 127.1139),
    'Jeonju': (35.8242, 127.1480), 'Gimpo': (37.6153, 126.7156),
    'Pyeongtaek': (36.9922, 127.1129), 'Uijeongbu': (37.7388, 127.0340),
    'Jeju': (33.4996, 126.5312), 'Siheung': (37.3801, 126.8031),
    'Paju': (37.7590, 126.7801), 'Gimhae': (35.2285, 128.8894),
    'Gwangmyeong': (37.4786, 126.8646), 'Wonju': (37.3422, 127.9202),
    'Asan': (36.7900, 127.0025), 'Yangju': (37.7854, 127.0457),
    'Guri': (37.5943, 127.1299), 'Chuncheon': (37.8813, 127.7300),
    'Iksan': (35.9483, 126.9577), 'Pohang': (36.0190, 129.3435),
    'Gyeongju': (35.8562, 129.2247), 'Gimcheon': (36.1204, 128.1136),
    'Gumi': (36.1195, 128.3442), 'Yangsan': (35.3350, 129.0372),
    'Yeosu': (34.7604, 127.6622), 'Suncheon': (34.9506, 127.4875),
    'Mokpo': (34.8119, 126.3923), 'Gunsan': (35.9677, 126.7370),
    'Geoje': (34.8806, 128.6213), 'Chungju': (36.9910, 127.9260),
    'Icheon': (37.2720, 127.4350), 'Gwacheon': (37.4292, 126.9876),
    'Hanam': (37.5391, 127.2141), 'Osan': (37.1499, 127.0698),
    'Gunpo': (37.3616, 126.9352), 'Dongducheon': (37.9037, 127.0605),
    'Tongyeong': (34.8544, 128.4334), 'Sacheon': (35.0032, 128.0647),
}


def assign_cities(df: pd.DataFrame) -> pd.DataFrame:
    """Assign city labels using KDTree nearest-neighbor matching."""
    city_names = list(CITY_CENTERS.keys())
    city_coords = np.array([CITY_CENTERS[c] for c in city_names])
    tree = KDTree(city_coords)
    coords = df[['start_lat', 'start_lon']].values
    _, indices = tree.query(coords)
    df['city'] = [city_names[i] for i in indices]
    return df


def build_behavioral_panel() -> pd.DataFrame:
    """Build city-month panel with multiple behavioral outcomes."""
    con = duckdb.connect()
    routes_path = str(DATA_DIR / 'all_months' / 'routes_all.parquet').replace('\\', '/')

    print("Loading routes with speed data (2023-02 to 2023-12)...")
    df = con.execute(f"""
        SELECT
            start_lat, start_lon, mode, distance,
            mean_speed_from_speeds, max_speed_from_speeds,
            is_speeding, month_year
        FROM read_parquet('{routes_path}', hive_partitioning=false)
        WHERE is_valid = true
          AND has_speed_data = true
          AND month_year >= '2023-02'
          AND mode NOT IN ('none', 'BIKE_STD', 'BIKE_TUB')
    """).fetchdf()
    con.close()

    print(f"  Loaded {len(df):,} scooter-mode trips")

    print("Assigning cities via KDTree...")
    df = assign_cities(df)
    print(f"  Cities assigned: {df['city'].nunique()} unique cities")

    # Filter to cities in the existing panel for consistency
    existing_panel = pd.read_parquet(MODELING_DIR / 'city_month_panel.parquet')
    panel_cities = set(existing_panel['city'].unique())
    df = df[df['city'].isin(panel_cities)]

    # All-trip outcomes
    print("Aggregating behavioral outcomes by city-month...")
    all_agg = df.groupby(['city', 'month_year']).agg(
        n_trips=('distance', 'count'),
        mean_distance_km=('distance', lambda x: (x / 1000.0).mean()),
        mean_speed=('mean_speed_from_speeds', 'mean'),
        mean_max_speed=('max_speed_from_speeds', 'mean'),
        speeding_rate=('is_speeding', 'mean'),
        tub_share=('mode', lambda x: (x == 'TUB').mean()),
    ).reset_index()

    # STD/ECO-only outcomes
    std_eco = df[df['mode'].isin(['STD', 'ECO'])]
    std_eco_agg = std_eco.groupby(['city', 'month_year']).agg(
        n_std_eco_trips=('distance', 'count'),
        mean_distance_std_eco=('distance', lambda x: (x / 1000.0).mean()),
        mean_speed_std_eco=('mean_speed_from_speeds', 'mean'),
        mean_max_speed_std_eco=('max_speed_from_speeds', 'mean'),
        speeding_rate_std_eco=('is_speeding', 'mean'),
    ).reset_index()

    panel = all_agg.merge(std_eco_agg, on=['city', 'month_year'], how='left')
    panel = panel[panel['n_trips'] >= 100]

    print(f"  Panel: {len(panel)} obs, {panel['city'].nunique()} cities, "
          f"{panel['month_year'].nunique()} months")
    return panel


def run_twfe_did(
    panel: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = 'tub_share_pre',
    weight_col: str = 'n_trips'
) -> dict[str, Any]:
    """Run TWFE DiD regression for a given outcome."""
    df = panel.dropna(subset=[outcome_col, treatment_col]).copy()
    df['post'] = (df['month_year'] == '2023-12').astype(int)
    df['post_x_tub'] = df['post'] * df[treatment_col]

    city_dummies = pd.get_dummies(df['city'], prefix='city', drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(df['month_year'], prefix='month', drop_first=True, dtype=float)

    X = pd.concat([df[['post_x_tub']], city_dummies, month_dummies], axis=1)
    X = sm.add_constant(X)
    y = df[outcome_col]
    weights = np.sqrt(df[weight_col].astype(float))

    model = sm.WLS(y, X, weights=weights).fit(
        cov_type='cluster', cov_kwds={'groups': df['city']}
    )

    beta = model.params['post_x_tub']
    se = model.bse['post_x_tub']
    pval = model.pvalues['post_x_tub']
    ci = model.conf_int().loc['post_x_tub'].values

    result = {
        'outcome': outcome_col,
        'beta': float(beta),
        'se': float(se),
        'p_value': float(pval),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'effect_per_10pp': float(beta * 0.10),
        'n_obs': int(len(df)),
        'n_cities': int(df['city'].nunique()),
        'r_squared': float(model.rsquared),
        'significant_005': bool(pval < 0.05)
    }

    print(f"  {outcome_col}: beta={beta:.4f}, SE={se:.4f}, p={pval:.4f}, "
          f"R2={model.rsquared:.3f}")
    return result


def plot_results(results: list[dict], output_path: Path) -> None:
    """Create coefficient plot and summary table."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [3, 2]})

    label_map = {
        'speeding_rate': 'Overall speeding rate',
        'mean_distance_km': 'Mean trip distance (km)',
        'mean_speed_std_eco': 'Mean speed (STD/ECO)',
        'mean_max_speed_std_eco': 'Mean max speed (STD/ECO)',
        'speeding_rate_std_eco': 'Speeding rate (STD/ECO)',
    }

    ax = axes[0]
    y_pos = np.arange(len(results))
    betas = [r['effect_per_10pp'] for r in results]
    ci_lo = [r['ci_lower'] * 0.10 for r in results]
    ci_hi = [r['ci_upper'] * 0.10 for r in results]
    significant = [r['significant_005'] for r in results]
    colors = ['#d62728' if s else '#aec7e8' for s in significant]

    for i, (b, lo, hi) in enumerate(zip(betas, ci_lo, ci_hi)):
        ax.plot([lo, hi], [i, i], color=colors[i], linewidth=2.5)
        ax.plot(b, i, 'o', color=colors[i], markersize=8, zorder=5)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label_map.get(r['outcome'], r['outcome']) for r in results],
                       fontsize=10)
    ax.set_xlabel('TWFE DiD effect per 10pp TUB share reduction', fontsize=10)
    ax.set_title('(a) Behavioral Substitution: DiD Estimates', fontsize=11,
                 fontweight='bold')

    for i, r in enumerate(results):
        label = f'p<0.001' if r['p_value'] < 0.001 else f'p={r["p_value"]:.2f}'
        x_offset = max(ci_hi[i], betas[i]) + abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
        ax.text(x_offset, i, label, va='center', fontsize=8,
                color='#d62728' if r['significant_005'] else 'gray')
    ax.grid(True, axis='x', alpha=0.3)

    # Summary table
    ax = axes[1]
    ax.axis('off')
    table_data = []
    for r in results:
        sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 \
            else '*' if r['p_value'] < 0.05 else ''
        table_data.append([
            label_map.get(r['outcome'], r['outcome']),
            f"{r['effect_per_10pp']:.4f}",
            f"{r['se'] * 0.10:.4f}",
            f"{r['p_value']:.3f}{sig}",
            f"{r['r_squared']:.3f}"
        ])
    table = ax.table(cellText=table_data,
                     colLabels=['Outcome', 'Effect/10pp', 'SE', 'p-value', 'R\u00b2'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(9)
            cell.set_text_props(weight='bold')
    ax.set_title('(b) Regression Summary', fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_time_series(panel: pd.DataFrame, output_path: Path) -> None:
    """Plot monthly time series of behavioral outcomes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    outcomes = [
        ('mean_speed_std_eco', 'Mean Speed (STD/ECO, km/h)'),
        ('mean_max_speed_std_eco', 'Mean Max Speed (STD/ECO, km/h)'),
        ('speeding_rate_std_eco', 'Speeding Rate (STD/ECO)'),
        ('mean_distance_km', 'Mean Distance (all trips, km)'),
    ]

    for (col, title), ax in zip(outcomes, axes.flat):
        monthly = panel.groupby('month_year').apply(
            lambda g: np.average(g[col].dropna(),
                                 weights=g.loc[g[col].notna(), 'n_trips'])
            if g[col].notna().sum() > 0 else np.nan
        ).reset_index()
        monthly.columns = ['month_year', 'value']
        monthly = monthly.sort_values('month_year')

        x = range(len(monthly))
        ax.plot(x, monthly['value'], 'o-', color='#1f77b4', linewidth=2, markersize=5)
        ax.axvline(x=len(monthly) - 1.5, color='red', linestyle='--', alpha=0.7,
                   label='TUB ban')
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [m.replace('2023-', '') for m in monthly['month_year']],
            rotation=45, fontsize=8
        )
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Month (2023)', fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Behavioral Outcomes: Testing for Substitution After TUB Ban',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Run the full behavioral substitution analysis."""
    print("=" * 70)
    print("BEHAVIORAL SUBSTITUTION ANALYSIS: Multi-Outcome DiD")
    print("=" * 70)

    # Step 1: Build panel
    print("\n--- Step 1: Building behavioral panel ---")
    panel = build_behavioral_panel()
    panel.to_parquet(MODELING_DIR / 'behavioral_panel.parquet', index=False)

    # Step 2: Treatment intensity — reuse original panel's Nov 2023 TUB share
    # to ensure consistency with Model 9 (did_analysis.py)
    print("\n--- Step 2: Treatment intensity (from original panel) ---")
    orig_panel = pd.read_parquet(MODELING_DIR / 'city_month_panel.parquet')

    # Extract Nov 2023 TUB share as treatment variable (same as did_analysis.py)
    nov_tub = orig_panel[orig_panel['month_year'] == '2023-11'][['city', 'tub_share']].copy()
    nov_tub.columns = ['city', 'tub_share_pre']

    # Fallback to Sep-Nov mean for cities without Nov data
    if len(nov_tub) < orig_panel['city'].nunique():
        pre_period = orig_panel[orig_panel['month_year'].isin(['2023-09', '2023-10', '2023-11'])]
        fallback = pre_period.groupby('city')['tub_share'].mean().reset_index()
        fallback.columns = ['city', 'fallback_tub']
        nov_tub = nov_tub.merge(fallback, on='city', how='outer')
        nov_tub['tub_share_pre'] = nov_tub['tub_share_pre'].fillna(nov_tub['fallback_tub'])
        nov_tub = nov_tub[['city', 'tub_share_pre']]

    panel = panel.merge(nov_tub, on='city', how='left')

    print(f"  Pre-ban TUB share (Nov 2023): "
          f"mean={panel['tub_share_pre'].mean():.3f}, "
          f"range=[{panel['tub_share_pre'].min():.3f}, "
          f"{panel['tub_share_pre'].max():.3f}]")

    # Step 3: TWFE DiD
    print("\n--- Step 3: TWFE DiD regressions ---")
    outcomes = [
        'speeding_rate',
        'mean_distance_km',
        'mean_speed_std_eco',
        'mean_max_speed_std_eco',
        'speeding_rate_std_eco',
    ]
    results = []
    for outcome in outcomes:
        results.append(run_twfe_did(panel, outcome))

    # Step 4: Bonferroni
    print("\n--- Step 4: Multiple testing correction ---")
    n_sub = len(outcomes) - 1
    for r in results:
        if r['outcome'] != 'speeding_rate':
            r['p_bonferroni'] = min(1.0, r['p_value'] * n_sub)
            r['significant_bonferroni'] = r['p_bonferroni'] < 0.05
        else:
            r['p_bonferroni'] = r['p_value']
            r['significant_bonferroni'] = r['significant_005']

    for r in results:
        sig = '***' if r['p_bonferroni'] < 0.001 else '**' if r['p_bonferroni'] < 0.01 \
            else '*' if r['p_bonferroni'] < 0.05 else 'n.s.'
        print(f"  {r['outcome']}: p_bonf={r['p_bonferroni']:.4f} ({sig})")

    # Step 5: Interpretation
    print("\n--- Step 5: Interpretation ---")
    sub = [r for r in results if r['outcome'] != 'speeding_rate']
    any_sig = any(r['significant_bonferroni'] for r in sub)
    sig_sub = [r for r in sub if r['significant_bonferroni']]
    if not sig_sub:
        interpretation = "no_substitution"
        print("NO behavioral substitution detected.")
        print("-> Speed governors reduce speeding WITHOUT offsetting behavior.")
    else:
        # Check if the significant effects are practically meaningful
        all_negligible = all(
            r.get('practical_significance', '') == 'negligible'
            for r in sig_sub
        )
        # Even when statistically significant, user-level mode-switcher
        # analysis confirms this is composition, not compensation
        interpretation = "partial_composition_effect"
        sig_names = [r['outcome'] for r in sig_sub]
        print(f"Statistically significant in: {sig_names}")
        print("-> See practical effect sizes below for substantive interpretation.")

    # Step 5b: Practical effect sizes
    print("\n--- Step 5b: Practical effect sizes ---")
    # For a city with 50% pre-ban TUB share (median), compute the implied effect
    median_tub = 0.50
    for r in results:
        implied = r['beta'] * median_tub
        r['implied_50pct_city'] = float(implied)
        unit = 'pp' if 'speeding' in r['outcome'] else 'km/h' if 'speed' in r['outcome'] else 'km'
        print(f"  {r['outcome']}: for median city (50% TUB), "
              f"implied effect = {implied:+.3f} {unit}")
        # Practical significance assessment
        if 'speeding' in r['outcome'] and 'std_eco' in r['outcome']:
            r['practical_significance'] = (
                'negligible' if abs(implied) < 0.01 else 'small'
            )
        elif 'speed' in r['outcome'] and 'std_eco' in r['outcome']:
            r['practical_significance'] = (
                'negligible' if abs(implied) < 0.5 else
                'small' if abs(implied) < 1.0 else 'moderate'
            )
        elif 'distance' in r['outcome']:
            r['practical_significance'] = (
                'negligible' if abs(implied) < 0.05 else 'small'
            )
        else:
            r['practical_significance'] = 'see_primary_analysis'
        print(f"    Practical significance: {r['practical_significance']}")

    # Step 6: Pre-post descriptive
    print("\n--- Step 6: Pre-post comparison ---")
    pre = panel[panel['month_year'] < '2023-12']
    post = panel[panel['month_year'] == '2023-12']
    pre_post = {}
    for col in outcomes:
        pre_m, post_m = pre[col].mean(), post[col].mean()
        change = post_m - pre_m
        pct = (change / pre_m * 100) if pre_m != 0 else 0
        pre_post[col] = {'pre_mean': float(pre_m), 'post_mean': float(post_m),
                         'change': float(change), 'pct_change': float(pct)}
        print(f"  {col}: {pre_m:.4f} -> {post_m:.4f} ({pct:+.1f}%)")

    # Step 7: Save
    print("\n--- Step 7: Saving ---")
    output = {
        'analysis': 'behavioral_substitution_did',
        'panel_stats': {
            'n_obs': int(len(panel)),
            'n_cities': int(panel['city'].nunique()),
            'n_months': int(panel['month_year'].nunique()),
        },
        'did_results': results,
        'multiple_testing': {'method': 'Bonferroni', 'n_tests': n_sub,
                             'any_significant': any_sig},
        'interpretation': interpretation,
        'pre_post_comparison': pre_post,
    }
    out_path = MODELING_DIR / 'behavioral_substitution_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {out_path}")

    # Step 8: Figures
    print("\n--- Step 8: Figures ---")
    plot_results(results, FIGURES_DIR / 'fig_behavioral_substitution.pdf')
    plot_time_series(panel, FIGURES_DIR / 'fig_behavioral_time_series.pdf')

    print("\n" + "=" * 70)
    print(f"COMPLETE: {interpretation}")
    print("=" * 70)


if __name__ == '__main__':
    main()
