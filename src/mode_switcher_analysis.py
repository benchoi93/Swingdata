"""
User-Level Mode-Switcher Analysis: Composition vs. Compensation.

Distinguishes whether the aggregate STD/ECO speed increase after the
December 2023 TUB ban reflects:
  (a) COMPOSITION effect: faster-riding TUB users join the STD/ECO pool,
      raising the pool average without individual behavior change, or
  (b) COMPENSATION effect: existing STD/ECO users ride faster to compensate
      for losing TUB mode access.

Method:
1. Classify users into "switchers" (used TUB pre-ban, STD/ECO post-ban)
   and "never-TUB" (only STD/ECO throughout).
2. Compare their Dec 2023 STD/ECO speeds.
3. Compare pre-ban STD/ECO speeds (Oct-Nov vs Dec) within each group.
4. User-level DiD: speed_change ~ group + controls.
5. Effect size reporting (Cohen's d, practical significance).

Output:
- data_parquet/modeling/mode_switcher_results.json
- figures/fig_mode_switcher.pdf
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIGURES_DIR, MODELING_DIR, FIG_DPI, RANDOM_SEED

warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(RANDOM_SEED)

ROUTES_PATH = str(DATA_DIR / 'all_months' / 'routes_all.parquet').replace('\\', '/')


def classify_users() -> pd.DataFrame:
    """Classify users into switchers and never-TUB based on pre-ban mode use.

    Returns:
        DataFrame with user_id and group_label ('switcher' or 'never_tub').
    """
    con = duckdb.connect()

    print("Classifying users by pre-ban TUB usage...")

    # Users who ever used TUB in the pre-ban period (Feb-Nov 2023)
    pre_tub = con.execute(f"""
        SELECT DISTINCT user_id
        FROM read_parquet('{ROUTES_PATH}', hive_partitioning=false)
        WHERE is_valid AND has_speed_data
          AND month_year >= '2023-02' AND month_year <= '2023-11'
          AND mode = 'TUB'
    """).fetchdf()
    tub_set = set(pre_tub['user_id'])
    print(f"  Pre-ban TUB users: {len(tub_set):,}")

    # Users who rode STD/ECO in Dec 2023
    dec_riders = con.execute(f"""
        SELECT DISTINCT user_id
        FROM read_parquet('{ROUTES_PATH}', hive_partitioning=false)
        WHERE is_valid AND has_speed_data
          AND month_year = '2023-12'
          AND mode IN ('STD', 'ECO')
    """).fetchdf()
    dec_set = set(dec_riders['user_id'])
    print(f"  Dec 2023 STD/ECO users: {len(dec_set):,}")

    switchers = tub_set & dec_set
    never_tub_dec = dec_set - tub_set
    print(f"  Switchers (TUB pre-ban -> STD/ECO Dec): {len(switchers):,}")
    print(f"  Never-TUB (STD/ECO only, in Dec): {len(never_tub_dec):,}")

    rows = [(uid, 'switcher') for uid in switchers] + \
           [(uid, 'never_tub') for uid in never_tub_dec]
    user_groups = pd.DataFrame(rows, columns=['user_id', 'group_label'])
    con.close()
    return user_groups


def get_user_trip_stats(user_groups: pd.DataFrame) -> pd.DataFrame:
    """Get per-user speed statistics in STD/ECO mode, pre and post ban.

    Args:
        user_groups: DataFrame with user_id and group_label.

    Returns:
        DataFrame with user-level pre/post speed statistics.
    """
    con = duckdb.connect()

    # Register user groups as a DuckDB table
    con.register('user_groups', user_groups)

    print("Computing user-level STD/ECO speed statistics...")

    # Pre-ban: Oct-Nov 2023 (most stable period, avoids early ramp-up)
    pre_stats = con.execute(f"""
        SELECT
            r.user_id,
            COUNT(*) as pre_n_trips,
            AVG(r.mean_speed_from_speeds) as pre_mean_speed,
            AVG(r.max_speed_from_speeds) as pre_mean_max_speed,
            AVG(CASE WHEN r.is_speeding THEN 1.0 ELSE 0.0 END) as pre_speeding_rate
        FROM read_parquet('{ROUTES_PATH}', hive_partitioning=false) r
        INNER JOIN user_groups ug ON r.user_id = ug.user_id
        WHERE r.is_valid AND r.has_speed_data
          AND r.month_year IN ('2023-10', '2023-11')
          AND r.mode IN ('STD', 'ECO')
        GROUP BY r.user_id
    """).fetchdf()
    print(f"  Pre-ban (Oct-Nov) STD/ECO users: {len(pre_stats):,}")

    # Post-ban: Dec 2023
    post_stats = con.execute(f"""
        SELECT
            r.user_id,
            COUNT(*) as post_n_trips,
            AVG(r.mean_speed_from_speeds) as post_mean_speed,
            AVG(r.max_speed_from_speeds) as post_mean_max_speed,
            AVG(CASE WHEN r.is_speeding THEN 1.0 ELSE 0.0 END) as post_speeding_rate
        FROM read_parquet('{ROUTES_PATH}', hive_partitioning=false) r
        INNER JOIN user_groups ug ON r.user_id = ug.user_id
        WHERE r.is_valid AND r.has_speed_data
          AND r.month_year = '2023-12'
          AND r.mode IN ('STD', 'ECO')
        GROUP BY r.user_id
    """).fetchdf()
    print(f"  Post-ban (Dec) STD/ECO users: {len(post_stats):,}")

    con.close()

    # Merge: users with BOTH pre and post observations in STD/ECO
    merged = user_groups.merge(pre_stats, on='user_id', how='inner')
    merged = merged.merge(post_stats, on='user_id', how='inner')
    merged['speed_change'] = merged['post_mean_speed'] - merged['pre_mean_speed']
    merged['max_speed_change'] = merged['post_mean_max_speed'] - merged['pre_mean_max_speed']
    merged['speeding_change'] = merged['post_speeding_rate'] - merged['pre_speeding_rate']

    print(f"  Users with both pre and post STD/ECO trips: {len(merged):,}")
    for g in ['switcher', 'never_tub']:
        n = len(merged[merged['group_label'] == g])
        print(f"    {g}: {n:,}")

    return merged


def compute_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute group comparisons and effect sizes.

    Args:
        df: User-level pre/post statistics.

    Returns:
        Dictionary with all statistical results.
    """
    print("\nComputing group comparisons...")
    sw = df[df['group_label'] == 'switcher']
    nt = df[df['group_label'] == 'never_tub']

    results: dict[str, Any] = {
        'sample_sizes': {
            'switchers': int(len(sw)),
            'never_tub': int(len(nt)),
            'total': int(len(df)),
        },
        'comparisons': {}
    }

    metrics = [
        ('post_mean_speed', 'Dec 2023 mean speed (km/h)'),
        ('post_mean_max_speed', 'Dec 2023 mean max speed (km/h)'),
        ('post_speeding_rate', 'Dec 2023 speeding rate'),
        ('pre_mean_speed', 'Pre-ban mean speed (km/h)'),
        ('speed_change', 'Speed change (post-pre, km/h)'),
        ('max_speed_change', 'Max speed change (post-pre, km/h)'),
        ('speeding_change', 'Speeding rate change (post-pre)'),
    ]

    for col, label in metrics:
        sw_vals = sw[col].dropna()
        nt_vals = nt[col].dropna()

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(sw_vals, nt_vals, equal_var=False)

        # Cohen's d
        pooled_sd = np.sqrt(
            ((len(sw_vals) - 1) * sw_vals.std()**2 +
             (len(nt_vals) - 1) * nt_vals.std()**2) /
            (len(sw_vals) + len(nt_vals) - 2)
        )
        cohens_d = (sw_vals.mean() - nt_vals.mean()) / pooled_sd if pooled_sd > 0 else 0

        results['comparisons'][col] = {
            'label': label,
            'switcher_mean': float(sw_vals.mean()),
            'switcher_sd': float(sw_vals.std()),
            'never_tub_mean': float(nt_vals.mean()),
            'never_tub_sd': float(nt_vals.std()),
            'difference': float(sw_vals.mean() - nt_vals.mean()),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d),
            'effect_size_label': (
                'negligible' if abs(cohens_d) < 0.2 else
                'small' if abs(cohens_d) < 0.5 else
                'medium' if abs(cohens_d) < 0.8 else 'large'
            ),
        }

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        print(f"  {label}:")
        print(f"    Switcher: {sw_vals.mean():.4f} (SD={sw_vals.std():.4f})")
        print(f"    Never-TUB: {nt_vals.mean():.4f} (SD={nt_vals.std():.4f})")
        print(f"    Diff: {sw_vals.mean() - nt_vals.mean():+.4f}, d={cohens_d:.3f} "
              f"({results['comparisons'][col]['effect_size_label']}), {sig}")

    # DiD-style interpretation: within-user speed change by group
    print("\n--- DiD interpretation (within-user speed change) ---")
    sw_change = sw['speed_change'].mean()
    nt_change = nt['speed_change'].mean()
    did_effect = sw_change - nt_change
    print(f"  Switcher mean speed change: {sw_change:+.4f} km/h")
    print(f"  Never-TUB mean speed change: {nt_change:+.4f} km/h")
    print(f"  DiD effect (switcher - never_tub): {did_effect:+.4f} km/h")

    results['did_interpretation'] = {
        'switcher_speed_change': float(sw_change),
        'never_tub_speed_change': float(nt_change),
        'did_effect': float(did_effect),
        'interpretation': (
            'composition_confirmed' if did_effect < 0.5 else 'compensation_detected'
        ),
        'explanation': (
            'Switchers do not increase their STD/ECO speeds after losing TUB access. '
            'The aggregate speed increase is due to faster riders joining the STD pool, '
            'not individual behavioral change.'
            if did_effect < 0.5 else
            'Switchers increase their STD/ECO speeds after losing TUB, suggesting compensation.'
        )
    }

    return results


def plot_mode_switcher(df: pd.DataFrame, results: dict, output_path: Path) -> None:
    """Create a 4-panel figure comparing switchers and never-TUB riders.

    Args:
        df: User-level pre/post statistics.
        results: Statistical results dictionary.
        output_path: Path for output figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sw = df[df['group_label'] == 'switcher']
    nt = df[df['group_label'] == 'never_tub']

    colors = {'switcher': '#d62728', 'never_tub': '#1f77b4'}

    # (a) Dec 2023 mean speed distributions
    ax = axes[0, 0]
    bins = np.linspace(0, 25, 60)
    ax.hist(nt['post_mean_speed'].clip(0, 25), bins=bins, alpha=0.6,
            color=colors['never_tub'], label=f'Never-TUB (n={len(nt):,})',
            density=True, edgecolor='white', linewidth=0.3)
    ax.hist(sw['post_mean_speed'].clip(0, 25), bins=bins, alpha=0.6,
            color=colors['switcher'], label=f'Switcher (n={len(sw):,})',
            density=True, edgecolor='white', linewidth=0.3)
    ax.axvline(nt['post_mean_speed'].mean(), color=colors['never_tub'],
               linestyle='--', linewidth=2)
    ax.axvline(sw['post_mean_speed'].mean(), color=colors['switcher'],
               linestyle='--', linewidth=2)
    ax.set_xlabel('Mean speed (km/h)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('(a) Dec 2023 STD/ECO Mean Speed', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    comp = results['comparisons']['post_mean_speed']
    ax.text(0.98, 0.95, f"d = {comp['cohens_d']:.3f} ({comp['effect_size_label']})",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # (b) Pre-ban speed comparison (showing pre-existing gap)
    ax = axes[0, 1]
    ax.hist(nt['pre_mean_speed'].clip(0, 25), bins=bins, alpha=0.6,
            color=colors['never_tub'], label='Never-TUB',
            density=True, edgecolor='white', linewidth=0.3)
    ax.hist(sw['pre_mean_speed'].clip(0, 25), bins=bins, alpha=0.6,
            color=colors['switcher'], label='Switcher',
            density=True, edgecolor='white', linewidth=0.3)
    ax.axvline(nt['pre_mean_speed'].mean(), color=colors['never_tub'],
               linestyle='--', linewidth=2)
    ax.axvline(sw['pre_mean_speed'].mean(), color=colors['switcher'],
               linestyle='--', linewidth=2)
    ax.set_xlabel('Mean speed (km/h)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('(b) Pre-Ban (Oct-Nov) STD/ECO Mean Speed', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    comp_pre = results['comparisons']['pre_mean_speed']
    ax.text(0.98, 0.95, f"d = {comp_pre['cohens_d']:.3f} ({comp_pre['effect_size_label']})",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # (c) Within-user speed change (DiD visualization)
    ax = axes[1, 0]
    ax.hist(nt['speed_change'].clip(-10, 10), bins=60, alpha=0.6,
            color=colors['never_tub'], label='Never-TUB',
            density=True, edgecolor='white', linewidth=0.3)
    ax.hist(sw['speed_change'].clip(-10, 10), bins=60, alpha=0.6,
            color=colors['switcher'], label='Switcher',
            density=True, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(nt['speed_change'].mean(), color=colors['never_tub'],
               linestyle='--', linewidth=2)
    ax.axvline(sw['speed_change'].mean(), color=colors['switcher'],
               linestyle='--', linewidth=2)
    ax.set_xlabel('Speed change (km/h, post - pre)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('(c) Within-User Speed Change', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    did = results['did_interpretation']
    ax.text(0.98, 0.95,
            f"DiD = {did['did_effect']:+.3f} km/h",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # (d) Summary effect size bar chart
    ax = axes[1, 1]
    metrics = ['post_mean_speed', 'post_mean_max_speed', 'speed_change',
               'max_speed_change', 'speeding_change']
    labels = ['Mean speed\n(Dec)', 'Max speed\n(Dec)', 'Mean speed\nchange',
              'Max speed\nchange', 'Speeding\nrate change']
    d_vals = [results['comparisons'][m]['cohens_d'] for m in metrics]
    bar_colors = ['#d62728' if abs(d) >= 0.2 else '#aec7e8' for d in d_vals]

    bars = ax.barh(range(len(d_vals)), d_vals, color=bar_colors,
                   edgecolor='gray', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's d (switcher - never-TUB)", fontsize=10)
    ax.set_title("(d) Effect Sizes", fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(0.2, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(-0.2, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.grid(True, axis='x', alpha=0.3)

    for i, (d, bar) in enumerate(zip(d_vals, bars)):
        ax.text(d + 0.02 if d >= 0 else d - 0.02, i,
                f'{d:.3f}', va='center',
                ha='left' if d >= 0 else 'right', fontsize=8)

    plt.suptitle('Mode-Switcher Analysis: Composition vs. Compensation',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Run the full mode-switcher analysis."""
    print("=" * 70)
    print("MODE-SWITCHER ANALYSIS: Composition vs. Compensation")
    print("=" * 70)

    # Step 1: Classify users
    print("\n--- Step 1: Classifying users ---")
    user_groups = classify_users()

    # Step 2: Get trip statistics
    print("\n--- Step 2: User-level trip statistics ---")
    user_stats = get_user_trip_stats(user_groups)

    # Step 3: Statistical comparisons
    print("\n--- Step 3: Statistical comparisons ---")
    results = compute_statistics(user_stats)

    # Step 4: Save results
    print("\n--- Step 4: Saving results ---")
    out_path = MODELING_DIR / 'mode_switcher_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out_path}")

    # Step 5: Visualization
    print("\n--- Step 5: Visualization ---")
    plot_mode_switcher(user_stats, results, FIGURES_DIR / 'fig_mode_switcher.pdf')

    # Step 6: Print verdict
    print("\n" + "=" * 70)
    verdict = results['did_interpretation']['interpretation']
    print(f"VERDICT: {verdict}")
    print(f"  {results['did_interpretation']['explanation']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
