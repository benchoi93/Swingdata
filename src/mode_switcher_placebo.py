"""
User-Level Placebo Test for Mode-Switcher Analysis.

Replicates the composition vs. compensation analysis from
mode_switcher_analysis.py using a PLACEBO treatment date:
October 2023 (instead of December 2023).

Design:
- User classification: same as original (TUB users Feb-Nov vs never-TUB)
- "Pre-placebo": Aug-Sep 2023 STD/ECO speeds
- "Post-placebo": Oct 2023 STD/ECO speeds
- Expected: DiD ~ 0 (no differential speed change between groups)

If the placebo DiD is negligible, it confirms that the December finding
reflects the TUB ban (treatment) rather than systematic seasonal
differences between switcher and never-TUB user groups.

Output:
- data_parquet/modeling/mode_switcher_placebo_results.json
- figures/fig_mode_switcher_placebo.pdf
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
    """Classify users into switchers and never-TUB (same as original)."""
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

    # For placebo: users who rode STD/ECO in Oct 2023 (placebo "post")
    oct_riders = con.execute(f"""
        SELECT DISTINCT user_id
        FROM read_parquet('{ROUTES_PATH}', hive_partitioning=false)
        WHERE is_valid AND has_speed_data
          AND month_year = '2023-10'
          AND mode IN ('STD', 'ECO')
    """).fetchdf()
    oct_set = set(oct_riders['user_id'])
    print(f"  Oct 2023 STD/ECO users: {len(oct_set):,}")

    switchers = tub_set & oct_set
    never_tub_oct = oct_set - tub_set
    print(f"  Switchers in Oct: {len(switchers):,}")
    print(f"  Never-TUB in Oct: {len(never_tub_oct):,}")

    rows = [(uid, 'switcher') for uid in switchers] + \
           [(uid, 'never_tub') for uid in never_tub_oct]
    user_groups = pd.DataFrame(rows, columns=['user_id', 'group_label'])
    con.close()
    return user_groups


def get_placebo_stats(user_groups: pd.DataFrame) -> pd.DataFrame:
    """Get per-user speed statistics for placebo periods.

    Pre-placebo: Aug-Sep 2023 (STD/ECO only)
    Post-placebo: Oct 2023 (STD/ECO only)
    """
    con = duckdb.connect()
    con.register('user_groups', user_groups)

    print("Computing user-level STD/ECO speed statistics (placebo)...")

    # Pre-placebo: Aug-Sep 2023
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
          AND r.month_year IN ('2023-08', '2023-09')
          AND r.mode IN ('STD', 'ECO')
        GROUP BY r.user_id
    """).fetchdf()
    print(f"  Pre-placebo (Aug-Sep) STD/ECO users: {len(pre_stats):,}")

    # Post-placebo: Oct 2023
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
          AND r.month_year = '2023-10'
          AND r.mode IN ('STD', 'ECO')
        GROUP BY r.user_id
    """).fetchdf()
    print(f"  Post-placebo (Oct) STD/ECO users: {len(post_stats):,}")

    con.close()

    merged = user_groups.merge(pre_stats, on='user_id', how='inner')
    merged = merged.merge(post_stats, on='user_id', how='inner')
    merged['speed_change'] = merged['post_mean_speed'] - merged['pre_mean_speed']
    merged['max_speed_change'] = merged['post_mean_max_speed'] - merged['pre_mean_max_speed']
    merged['speeding_change'] = merged['post_speeding_rate'] - merged['pre_speeding_rate']

    print(f"  Users with both pre and post placebo STD/ECO trips: {len(merged):,}")
    for g in ['switcher', 'never_tub']:
        n = len(merged[merged['group_label'] == g])
        print(f"    {g}: {n:,}")

    return merged


def compute_placebo_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute group comparisons for the placebo test."""
    print("\nComputing placebo group comparisons...")
    sw = df[df['group_label'] == 'switcher']
    nt = df[df['group_label'] == 'never_tub']

    results: dict[str, Any] = {
        'placebo_design': {
            'pre_period': 'Aug-Sep 2023',
            'post_period': 'Oct 2023',
            'treatment': 'None (placebo)',
            'user_classification': 'Same as original (TUB users Feb-Nov vs never-TUB)',
        },
        'sample_sizes': {
            'switchers': int(len(sw)),
            'never_tub': int(len(nt)),
            'total': int(len(df)),
        },
        'comparisons': {}
    }

    metrics = [
        ('pre_mean_speed', 'Pre-placebo mean speed (km/h)'),
        ('speed_change', 'Speed change (Oct - AugSep, km/h)'),
        ('max_speed_change', 'Max speed change (Oct - AugSep, km/h)'),
        ('speeding_change', 'Speeding rate change (Oct - AugSep)'),
    ]

    for col, label in metrics:
        sw_vals = sw[col].dropna()
        nt_vals = nt[col].dropna()

        t_stat, p_val = stats.ttest_ind(sw_vals, nt_vals, equal_var=False)

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

    # DiD interpretation
    sw_change = sw['speed_change'].mean()
    nt_change = nt['speed_change'].mean()
    did_effect = sw_change - nt_change
    print(f"\n--- Placebo DiD ---")
    print(f"  Switcher mean speed change: {sw_change:+.4f} km/h")
    print(f"  Never-TUB mean speed change: {nt_change:+.4f} km/h")
    print(f"  Placebo DiD effect: {did_effect:+.4f} km/h")

    # Load original DiD from mode_switcher_results.json
    orig_results_path = MODELING_DIR / 'mode_switcher_results.json'
    with open(orig_results_path) as f:
        orig_results = json.load(f)
    original_did = orig_results['did_interpretation']['did_effect']
    results['did_interpretation'] = {
        'switcher_speed_change': float(sw_change),
        'never_tub_speed_change': float(nt_change),
        'placebo_did_effect': float(did_effect),
        'original_did_effect': float(original_did),
        'placebo_passes': bool(abs(did_effect) < abs(original_did)),
        'interpretation': (
            'PASS: Placebo DiD is smaller than treatment DiD, '
            'confirming the December effect is treatment-driven.'
            if abs(did_effect) < abs(original_did) else
            'FAIL: Placebo DiD is comparable to treatment DiD.'
        ),
    }

    return results


def plot_placebo(df: pd.DataFrame, results: dict, output_path: Path) -> None:
    """Create a 2-panel placebo comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sw = df[df['group_label'] == 'switcher']
    nt = df[df['group_label'] == 'never_tub']

    colors = {'switcher': '#d62728', 'never_tub': '#1f77b4'}

    # (a) Within-user speed change (placebo)
    ax = axes[0]
    bins = np.linspace(-10, 10, 60)
    ax.hist(nt['speed_change'].clip(-10, 10), bins=bins, alpha=0.6,
            color=colors['never_tub'], label=f'Never-TUB (n={len(nt):,})',
            density=True, edgecolor='white', linewidth=0.3)
    ax.hist(sw['speed_change'].clip(-10, 10), bins=bins, alpha=0.6,
            color=colors['switcher'], label=f'Switcher (n={len(sw):,})',
            density=True, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(nt['speed_change'].mean(), color=colors['never_tub'],
               linestyle='--', linewidth=2)
    ax.axvline(sw['speed_change'].mean(), color=colors['switcher'],
               linestyle='--', linewidth=2)
    ax.set_xlabel('Speed change (km/h, Oct - AugSep)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('(a) Placebo: Within-User Speed Change\n(Aug-Sep -> Oct 2023)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    did = results['did_interpretation']
    ax.text(0.98, 0.95,
            f"Placebo DiD = {did['placebo_did_effect']:+.3f} km/h",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # (b) Comparison: placebo vs treatment DiD
    ax = axes[1]
    bar_labels = ['Placebo\n(Aug-Sep -> Oct)', 'Treatment\n(Oct-Nov -> Dec)']
    bar_vals = [did['placebo_did_effect'], did['original_did_effect']]
    bar_colors_list = ['#2ca02c', '#d62728']

    bars = ax.bar(range(2), bar_vals, color=bar_colors_list,
                  edgecolor='gray', linewidth=0.8, width=0.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel('DiD Effect (km/h)', fontsize=10)
    ax.set_title('(b) Placebo vs. Treatment DiD', fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='y', alpha=0.3)

    for i, (val, bar) in enumerate(zip(bar_vals, bars)):
        ax.text(i, val + 0.01 if val >= 0 else val - 0.01,
                f'{val:+.3f}', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=11, fontweight='bold')

    plt.suptitle('Mode-Switcher Placebo Test', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Run the user-level placebo test."""
    print("=" * 70)
    print("MODE-SWITCHER PLACEBO TEST")
    print("Placebo treatment: Oct 2023 (no actual policy change)")
    print("=" * 70)

    # Step 1: Classify users (same as original)
    print("\n--- Step 1: Classifying users ---")
    user_groups = classify_users()

    # Step 2: Get placebo period trip statistics
    print("\n--- Step 2: Placebo period trip statistics ---")
    user_stats = get_placebo_stats(user_groups)

    # Step 3: Statistical comparisons
    print("\n--- Step 3: Placebo statistical comparisons ---")
    results = compute_placebo_statistics(user_stats)

    # Step 4: Save results
    print("\n--- Step 4: Saving results ---")
    out_path = MODELING_DIR / 'mode_switcher_placebo_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out_path}")

    # Step 5: Visualization
    print("\n--- Step 5: Visualization ---")
    plot_placebo(user_stats, results, FIGURES_DIR / 'fig_mode_switcher_placebo.pdf')

    # Step 6: Print verdict
    print("\n" + "=" * 70)
    did = results['did_interpretation']
    status = "PASS" if did['placebo_passes'] else "FAIL"
    print(f"PLACEBO TEST: {status}")
    print(f"  Placebo DiD = {did['placebo_did_effect']:+.4f} km/h")
    print(f"  Treatment DiD = {did['original_did_effect']:+.4f} km/h")
    print(f"  {did['interpretation']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
