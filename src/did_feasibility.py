"""
Difference-in-Differences Feasibility Assessment for Speed Governor Effects.

Investigates whether the rollout of TUB/STD/ECO modes was staggered across
cities, which would enable a staggered DiD design for causal identification
of speed governor effects on speeding behavior.

Key questions:
1. When did each mode first appear in each city?
2. Is there meaningful variation in rollout timing across cities?
3. What happened in December 2023 (TUB -> ECO transition)?
4. Can we construct valid treatment/control groups?

Output: data_parquet/modeling/did_feasibility_results.json
        figures/fig_did_feasibility_*.pdf
"""
import json
import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, FIGURES_DIR, MODELING_DIR, FIG_DPI, RANDOM_SEED
)

np.random.seed(RANDOM_SEED)

# Korean city centers (same as used in city assignment)
CITY_CENTERS = {
    'Seoul': (37.5665, 126.9780),
    'Busan': (35.1796, 129.0756),
    'Daegu': (35.8714, 128.6014),
    'Incheon': (37.4563, 126.7052),
    'Gwangju': (35.1595, 126.8526),
    'Daejeon': (36.3504, 127.3845),
    'Ulsan': (35.5384, 129.3114),
    'Sejong': (36.4800, 127.2600),
    'Suwon': (37.2636, 127.0286),
    'Changwon': (35.2280, 128.6811),
    'Goyang': (37.6584, 126.8320),
    'Yongin': (37.2411, 127.1776),
    'Seongnam': (37.4449, 127.1389),
    'Hwaseong': (37.1997, 126.8312),
    'Cheongju': (36.6424, 127.4890),
    'Bucheon': (37.5035, 126.7660),
    'Ansan': (37.3219, 126.8309),
    'Anyang': (37.3943, 126.9568),
    'Namyangju': (37.6360, 127.2165),
    'Cheonan': (36.8152, 127.1139),
    'Jeonju': (35.8242, 127.1480),
    'Gimpo': (37.6153, 126.7156),
    'Pyeongtaek': (36.9922, 127.1129),
    'Uijeongbu': (37.7388, 127.0340),
    'Jeju': (33.4996, 126.5312),
    'Siheung': (37.3801, 126.8031),
    'Paju': (37.7590, 126.7801),
    'Gimhae': (35.2285, 128.8894),
    'Gwangmyeong': (37.4786, 126.8646),
    'Wonju': (37.3422, 127.9202),
    'Asan': (36.7900, 127.0025),
    'Yangju': (37.7854, 127.0457),
    'Guri': (37.5943, 127.1299),
    'Chuncheon': (37.8813, 127.7300),
    'Iksan': (35.9483, 126.9577),
    'Pohang': (36.0190, 129.3435),
    'Gyeongju': (35.8562, 129.2247),
    'Gimcheon': (36.1204, 128.1136),
    'Gumi': (36.1195, 128.3442),
    'Yangsan': (35.3350, 129.0372),
    'Yeosu': (34.7604, 127.6622),
    'Suncheon': (34.9506, 127.4875),
    'Mokpo': (34.8119, 126.3923),
    'Gunsan': (35.9677, 126.7370),
    'Geoje': (34.8806, 128.6213),
    'Chungju': (36.9910, 127.9260),
    'Icheon': (37.2720, 127.4350),
    'Gwacheon': (37.4292, 126.9876),
    'Hanam': (37.5391, 127.2141),
    'Osan': (37.1499, 127.0698),
    'Gunpo': (37.3616, 126.9352),
    'Dongducheon': (37.9037, 127.0605),
    'Tongyeong': (34.8544, 128.4334),
    'Sacheon': (35.0032, 128.0647),
}


def assign_cities_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Assign city labels using KDTree nearest-neighbor matching."""
    city_names = list(CITY_CENTERS.keys())
    city_coords = np.array([CITY_CENTERS[c] for c in city_names])
    tree = KDTree(city_coords)

    coords = df[['start_lat', 'start_lon']].values
    _, indices = tree.query(coords)
    df['city'] = [city_names[i] for i in indices]
    return df


def run_feasibility_assessment() -> dict:
    """Run full DiD feasibility assessment on multi-month data."""
    con = duckdb.connect()
    results = {}

    print("=" * 60)
    print("DiD FEASIBILITY ASSESSMENT")
    print("=" * 60)

    # ---- 1. Mode rollout timing by city ----
    print("\n[1/5] Querying mode rollout by city (first appearance)...")

    # Sample 2M trips per month for city assignment (memory-efficient)
    # Focus on months with mode data (2023-02 through 2023-12)
    df_mode = con.execute("""
        SELECT month_year, mode, start_lat, start_lon,
               is_speeding, has_speed_data,
               max_speed_from_speeds
        FROM read_parquet('data_parquet/all_months/routes_all.parquet')
        WHERE is_valid = true
          AND month_year >= '2023-02'
          AND mode NOT IN ('none', 'BIKE_STD', 'BIKE_TUB')
    """).fetchdf()

    print(f"  Loaded {len(df_mode):,} scooter-mode trips (2023-02 to 2023-12)")

    # Assign cities
    print("  Assigning cities via KDTree...")
    df_mode = assign_cities_fast(df_mode)

    # First mode appearance per city
    first_appearance = (
        df_mode.groupby(['city', 'mode'])['month_year']
        .min()
        .unstack(fill_value='never')
    )
    print("\n  First mode appearance per city (top 20 cities by trip count):")
    top_cities = df_mode['city'].value_counts().head(20).index
    print(first_appearance.loc[first_appearance.index.isin(top_cities)].to_string())

    # Check if rollout is staggered
    unique_first_months = {}
    for mode in ['TUB', 'STD', 'ECO']:
        if mode in first_appearance.columns:
            vals = first_appearance[mode]
            vals = vals[vals != 'never']
            unique_first_months[mode] = vals.unique().tolist()

    results['first_appearance_by_city'] = first_appearance.to_dict()
    results['unique_rollout_months'] = unique_first_months

    all_same = all(len(v) == 1 for v in unique_first_months.values())
    print(f"\n  Unique rollout months per mode: {unique_first_months}")
    print(f"  All cities same rollout month? {all_same}")

    if all_same:
        print("  >> STAGGERED DiD NOT FEASIBLE: All modes rolled out simultaneously")
        print("     across all cities in the same month (2023-02)")
        results['staggered_did_feasible'] = False
    else:
        print("  >> POTENTIAL for staggered DiD: Different cities got modes at different times")
        results['staggered_did_feasible'] = True

    # ---- 2. December 2023 TUB->ECO transition ----
    print("\n[2/5] Analyzing December 2023 mode transition...")

    monthly_mode = (
        df_mode.groupby(['month_year', 'mode'])
        .size()
        .unstack(fill_value=0)
    )
    monthly_mode['total'] = monthly_mode.sum(axis=1)
    for m in ['TUB', 'STD', 'ECO']:
        if m in monthly_mode.columns:
            monthly_mode[f'{m}_pct'] = (monthly_mode[m] / monthly_mode['total'] * 100).round(1)

    print(monthly_mode[['TUB', 'STD', 'ECO', 'TUB_pct', 'STD_pct', 'ECO_pct']].to_string())

    # Check Dec 2023 specifically
    if '2023-11' in monthly_mode.index and '2023-12' in monthly_mode.index:
        nov = monthly_mode.loc['2023-11']
        dec = monthly_mode.loc['2023-12']
        tub_drop = nov.get('TUB', 0) - dec.get('TUB', 0)
        eco_rise = dec.get('ECO', 0) - nov.get('ECO', 0)
        print(f"\n  Nov->Dec TUB drop: {tub_drop:,} trips")
        print(f"  Nov->Dec ECO rise: {eco_rise:,} trips")
        results['dec2023_tub_drop'] = int(tub_drop)
        results['dec2023_eco_rise'] = int(eco_rise)

    # ---- 3. City-level mode shares over time ----
    print("\n[3/5] Computing city-month mode shares...")

    city_month_mode = (
        df_mode.groupby(['city', 'month_year', 'mode'])
        .agg(
            n_trips=('mode', 'size'),
            speeding_rate=('is_speeding', 'mean'),
            mean_max_speed=('max_speed_from_speeds', lambda x: x.dropna().mean())
        )
        .reset_index()
    )

    # City-month totals
    city_month_total = (
        df_mode.groupby(['city', 'month_year'])
        .agg(
            total_trips=('mode', 'size'),
            overall_speeding=('is_speeding', 'mean')
        )
        .reset_index()
    )

    # TUB share per city-month
    tub_trips = city_month_mode[city_month_mode['mode'] == 'TUB'][
        ['city', 'month_year', 'n_trips']
    ].rename(columns={'n_trips': 'tub_trips'})

    eco_trips = city_month_mode[city_month_mode['mode'] == 'ECO'][
        ['city', 'month_year', 'n_trips']
    ].rename(columns={'n_trips': 'eco_trips'})

    panel = city_month_total.merge(tub_trips, on=['city', 'month_year'], how='left')
    panel = panel.merge(eco_trips, on=['city', 'month_year'], how='left')
    panel['tub_trips'] = panel['tub_trips'].fillna(0)
    panel['eco_trips'] = panel['eco_trips'].fillna(0)
    panel['tub_share'] = panel['tub_trips'] / panel['total_trips']
    panel['eco_share'] = panel['eco_trips'] / panel['total_trips']

    results['n_city_month_obs'] = len(panel)
    results['n_cities'] = panel['city'].nunique()
    results['n_months'] = panel['month_year'].nunique()

    print(f"  Panel: {panel['city'].nunique()} cities x {panel['month_year'].nunique()} months = {len(panel)} obs")

    # ---- 4. Alternative DiD: Dec 2023 ECO mandate as natural experiment ----
    print("\n[4/5] Assessing Dec 2023 ECO transition as natural experiment...")

    # Check which cities had TUB in Nov 2023 and lost it in Dec 2023
    nov_data = panel[panel['month_year'] == '2023-11'][['city', 'tub_share', 'eco_share', 'overall_speeding']]
    nov_data.columns = ['city', 'nov_tub_share', 'nov_eco_share', 'nov_speeding']

    dec_data = panel[panel['month_year'] == '2023-12'][['city', 'tub_share', 'eco_share', 'overall_speeding']]
    dec_data.columns = ['city', 'dec_tub_share', 'dec_eco_share', 'dec_speeding']

    transition = nov_data.merge(dec_data, on='city')
    transition['tub_share_change'] = transition['dec_tub_share'] - transition['nov_tub_share']
    transition['eco_share_change'] = transition['dec_eco_share'] - transition['nov_eco_share']
    transition['speeding_change'] = transition['dec_speeding'] - transition['nov_speeding']

    # Cities with large TUB drop
    big_drop = transition[transition['tub_share_change'] < -0.1].sort_values('tub_share_change')
    print(f"  Cities with >10pp TUB share drop (Nov->Dec): {len(big_drop)}")
    if len(big_drop) > 0:
        print(big_drop[['city', 'nov_tub_share', 'dec_tub_share', 'tub_share_change',
                         'nov_speeding', 'dec_speeding', 'speeding_change']].round(3).to_string(index=False))

    results['cities_with_tub_drop'] = len(big_drop)
    results['dec_transition'] = transition[['city', 'tub_share_change', 'eco_share_change',
                                             'speeding_change']].to_dict('records')

    # ---- 5. Check variation in ECO adoption intensity ----
    print("\n[5/5] Checking cross-city variation in ECO adoption intensity...")

    # Compute treatment intensity: ECO share in Dec 2023 minus ECO share in Feb 2023
    feb_data = panel[panel['month_year'] == '2023-02'][['city', 'eco_share']].rename(
        columns={'eco_share': 'feb_eco_share'})
    dec_eco = panel[panel['month_year'] == '2023-12'][['city', 'eco_share']].rename(
        columns={'eco_share': 'dec_eco_share'})

    intensity = feb_data.merge(dec_eco, on='city', how='outer')
    intensity = intensity.fillna(0)
    intensity['eco_adoption_change'] = intensity['dec_eco_share'] - intensity['feb_eco_share']
    intensity = intensity.sort_values('eco_adoption_change', ascending=False)

    print("  Top 15 cities by ECO adoption change (Feb -> Dec 2023):")
    print(intensity.head(15).round(3).to_string(index=False))

    eco_variation = intensity['eco_adoption_change'].describe()
    print(f"\n  ECO adoption change stats:")
    print(f"    Mean: {eco_variation['mean']:.3f}")
    print(f"    Std:  {eco_variation['std']:.3f}")
    print(f"    Min:  {eco_variation['min']:.3f}")
    print(f"    Max:  {eco_variation['max']:.3f}")

    results['eco_adoption_variation'] = {
        'mean': float(eco_variation['mean']),
        'std': float(eco_variation['std']),
        'min': float(eco_variation['min']),
        'max': float(eco_variation['max']),
    }

    # ---- Generate figures ----
    print("\n[Figures] Generating DiD feasibility plots...")

    # Figure 1: Mode shares over time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    monthly_pct = monthly_mode[['TUB_pct', 'STD_pct', 'ECO_pct']].copy()
    months = monthly_pct.index.tolist()
    x = range(len(months))

    ax = axes[0]
    ax.plot(x, monthly_pct['TUB_pct'], 'o-', color='#d62728', label='TUB (turbo)', linewidth=2)
    ax.plot(x, monthly_pct['STD_pct'], 's-', color='#1f77b4', label='STD (standard)', linewidth=2)
    ax.plot(x, monthly_pct['ECO_pct'], '^-', color='#2ca02c', label='ECO (eco)', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mode share (%)')
    ax.set_title('(a) Monthly mode shares')
    ax.legend(fontsize=9)
    ax.axvline(x=len(months)-1, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('Dec 2023\nTUB restriction', xy=(len(months)-1, 50),
                fontsize=8, ha='center', color='gray')
    ax.grid(alpha=0.3)

    # Figure 2: Speeding rate by month
    monthly_speeding = (
        df_mode.groupby('month_year')['is_speeding']
        .mean() * 100
    )
    ax = axes[1]
    ax.plot(range(len(monthly_speeding)), monthly_speeding.values, 'ko-', linewidth=2)
    ax.set_xticks(range(len(monthly_speeding)))
    ax.set_xticklabels(monthly_speeding.index.tolist(), rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Speeding rate (%)')
    ax.set_title('(b) Overall speeding rate over time')
    ax.axvline(x=len(monthly_speeding)-1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_mode_trends.pdf', dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_mode_trends.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_mode_trends.pdf/png")

    # Figure 3: City-level Dec 2023 transition scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    # Filter to cities with enough data
    big_cities = transition[transition.apply(
        lambda r: panel[(panel['city'] == r['city']) & (panel['month_year'] == '2023-11')]['total_trips'].values[0] > 1000
        if len(panel[(panel['city'] == r['city']) & (panel['month_year'] == '2023-11')]) > 0 else False,
        axis=1
    )]
    if len(big_cities) > 0:
        sc = ax.scatter(
            big_cities['tub_share_change'] * 100,
            big_cities['speeding_change'] * 100,
            s=60, alpha=0.7, c='#d62728', edgecolors='black', linewidth=0.5
        )
        for _, row in big_cities.iterrows():
            if abs(row['tub_share_change']) > 0.15 or abs(row['speeding_change']) > 0.05:
                ax.annotate(row['city'], (row['tub_share_change']*100, row['speeding_change']*100),
                           fontsize=7, ha='left', va='bottom')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('TUB share change (pp, Nov -> Dec 2023)')
        ax.set_ylabel('Speeding rate change (pp, Nov -> Dec 2023)')
        ax.set_title('Dec 2023 TUB Restriction: City-Level Impact')

        # Add correlation
        from scipy import stats
        r, p = stats.pearsonr(big_cities['tub_share_change'], big_cities['speeding_change'])
        ax.text(0.05, 0.95, f'r = {r:.3f}, p = {p:.4f}\nn = {len(big_cities)} cities',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        results['dec_transition_correlation'] = {'r': float(r), 'p': float(p), 'n': len(big_cities)}

    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_dec_transition.pdf', dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_dec_transition.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_dec_transition.pdf/png")

    # Figure 4: City-level speeding trends (top 6 cities)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    top6 = df_mode['city'].value_counts().head(6).index.tolist()
    colors = {'TUB': '#d62728', 'STD': '#1f77b4', 'ECO': '#2ca02c'}

    for idx, city in enumerate(top6):
        ax = axes[idx // 3, idx % 3]
        city_data = city_month_mode[city_month_mode['city'] == city]

        for mode, color in colors.items():
            mode_data = city_data[city_data['mode'] == mode].sort_values('month_year')
            if len(mode_data) > 0:
                ax.plot(range(len(mode_data)), mode_data['speeding_rate'] * 100,
                       'o-', color=color, label=mode, linewidth=1.5, markersize=4)

        ax.set_title(city, fontsize=10, fontweight='bold')
        ax.set_ylabel('Speeding rate (%)' if idx % 3 == 0 else '')
        months_list = sorted(city_data['month_year'].unique())
        ax.set_xticks(range(len(months_list)))
        ax.set_xticklabels(months_list, rotation=45, ha='right', fontsize=6)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Mode-Specific Speeding Rates by City', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_did_city_trends.pdf', dpi=FIG_DPI, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig_did_city_trends.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print("  Saved fig_did_city_trends.pdf/png")

    # ---- Summary and recommendations ----
    print("\n" + "=" * 60)
    print("FEASIBILITY SUMMARY")
    print("=" * 60)

    summary_lines = []
    if all_same:
        summary_lines.append(
            "1. STAGGERED DiD (mode rollout): NOT FEASIBLE - all modes rolled out "
            "simultaneously in 2023-02 across all cities."
        )
    else:
        summary_lines.append(
            "1. STAGGERED DiD (mode rollout): POTENTIALLY FEASIBLE - "
            "mode rollout timing varies across cities."
        )

    summary_lines.append(
        f"2. DEC 2023 NATURAL EXPERIMENT: {results.get('cities_with_tub_drop', 0)} cities "
        f"had >10pp TUB share drop. "
        f"Correlation between TUB drop and speeding change: "
        f"r={results.get('dec_transition_correlation', {}).get('r', 'N/A'):.3f}"
    )

    summary_lines.append(
        f"3. ECO ADOPTION VARIATION: Mean change = "
        f"{results['eco_adoption_variation']['mean']:.3f}, "
        f"Std = {results['eco_adoption_variation']['std']:.3f}. "
        f"{'Sufficient' if results['eco_adoption_variation']['std'] > 0.05 else 'Insufficient'} "
        f"cross-city variation for continuous treatment DiD."
    )

    summary_lines.append(
        "4. ALTERNATIVE: Pre-post comparison (2023-11 vs 2023-12) with city-level "
        "variation in TUB->ECO transition intensity as continuous treatment."
    )

    for line in summary_lines:
        print(f"  {line}")

    results['summary'] = summary_lines

    # Save panel data for potential DiD analysis
    panel.to_parquet(DATA_DIR / 'modeling' / 'city_month_panel.parquet', index=False)
    print(f"\n  Saved city-month panel ({len(panel)} obs) to modeling/city_month_panel.parquet")

    # Save results
    output_path = MODELING_DIR / 'did_feasibility_results.json'
    # Convert non-serializable types
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable_results[k] = {
                str(kk): (str(vv) if not isinstance(vv, (int, float, bool, str, list)) else vv)
                for kk, vv in v.items()
            }
        elif isinstance(v, list):
            serializable_results[k] = v
        else:
            serializable_results[k] = v

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"  Saved results to {output_path}")

    return results


if __name__ == '__main__':
    results = run_feasibility_assessment()
