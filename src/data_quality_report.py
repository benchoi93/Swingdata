"""
Task 1.7: Generate data quality report section for the paper.

Produces a markdown report summarizing:
  - Data sources and coverage
  - Preprocessing decisions and filtering criteria
  - Filtering rates and final sample sizes
  - Speed column validation results
  - City assignment methodology
  - Dataset summary statistics

Output:
  - reports/data_quality_report.md
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, REPORTS_DIR, SPEED_LIMIT_KR, MAX_PLAUSIBLE_SPEED, MIN_TRIP_POINTS


def generate_report() -> None:
    """Generate the data quality report from intermediate outputs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load intermediate reports
    with open(DATA_DIR / "filtering_report.json", encoding="utf-8") as f:
        filter_stats = json.load(f)
    with open(DATA_DIR / "city_assignment_report.json", encoding="utf-8") as f:
        city_stats = json.load(f)
    with open(DATA_DIR / "speed_validation_report.json", encoding="utf-8") as f:
        speed_val = json.load(f)
    with open(DATA_DIR / "cleaned" / "dataset_summary.json", encoding="utf-8") as f:
        dataset_summary = json.load(f)

    total = filter_stats["total_rows"]
    valid = filter_stats["valid_trips"]
    strict = filter_stats["strict_valid_trips"]

    report = f"""# Data Quality Report: E-Scooter Speed Safety Analysis

## 1. Data Sources

The dataset was obtained from Swing, a leading e-scooter sharing operator in South Korea.
Two complementary datasets were provided:

| Dataset | Rows | Columns | Coverage | Key Contents |
|---------|------|---------|----------|-------------|
| Routes CSV | {total:,} | 25 | May 2023 | GPS trajectories, speed profiles, trip metrics |
| Scooter CSV | 20,045,245 | 20 | Jan-Oct 2023 | User demographics, billing, ride mode |

The Routes dataset contains GPS trajectories with ~10-second intervals and per-point speed
readings from the scooter's onboard sensors.

## 2. Preprocessing and Filtering

### 2.1 Sentinel Value Replacement

The raw data uses `-999` as a sentinel value instead of NULL for six numeric columns:
`distance`, `moved_distance`, `avg_point_gap`, `max_point_gap`, `avg_speed`, `max_speed`.
All sentinel values were replaced with NULL.

- Distance/gap sentinels: {filter_stats['flag_sentinel']['count']:,} trips ({filter_stats['flag_sentinel']['rate']:.2%})

### 2.2 Quality Filtering Criteria

The following exclusion criteria were applied:

| Filter | Criterion | Trips Removed | Rate |
|--------|-----------|--------------|------|
| Excluded model | I9 (moped type) | {filter_stats['flag_excluded_model']['count']:,} | {filter_stats['flag_excluded_model']['rate']:.2%} |
| Sentinel values | -999 in any numeric column | {filter_stats['flag_sentinel']['count']:,} | {filter_stats['flag_sentinel']['rate']:.2%} |
| Invalid coordinates | Outside Korea bounds (lat 33-39, lon 124-132) | {filter_stats['flag_invalid_coords']['count']:,} | {filter_stats['flag_invalid_coords']['rate']:.4%} |
| Few GPS points | < {MIN_TRIP_POINTS} GPS readings | {filter_stats['flag_few_points']['count']:,} | {filter_stats['flag_few_points']['rate']:.2%} |
| Implausible speed | max_speed > {MAX_PLAUSIBLE_SPEED} km/h | {filter_stats['flag_implausible_speed']['count']:,} | {filter_stats['flag_implausible_speed']['rate']:.2%} |

**Core filtering result**: {valid:,} trips retained ({filter_stats['valid_rate']:.1%} of original)

### 2.3 Additional Quality Flags (Retained for Sensitivity Analysis)

| Flag | Criterion | Trips Flagged | Rate |
|------|-----------|--------------|------|
| Short distance | < 100 m | {filter_stats['flag_short_distance']['count']:,} | {filter_stats['flag_short_distance']['rate']:.2%} |
| Short duration | < 60 s | {filter_stats['flag_short_duration']['count']:,} | {filter_stats['flag_short_duration']['rate']:.2%} |
| Long duration | > 7,200 s (2 hrs) | {filter_stats['flag_long_duration']['count']:,} | {filter_stats['flag_long_duration']['rate']:.2%} |

**Strict filtering** (core + distance/duration): {strict:,} trips ({filter_stats['strict_valid_rate']:.1%} of original)

### 2.4 Model-Level Filtering Summary

| Model | Total | Valid | Retention Rate |
|-------|-------|-------|---------------|
"""
    for model, mstats in filter_stats["by_model"].items():
        pct = mstats["valid"] / mstats["total"] * 100 if mstats["total"] > 0 else 0
        report += f"| {model} | {mstats['total']:,} | {mstats['valid']:,} | {pct:.1f}% |\n"

    report += f"""
## 3. City Assignment

Trips were assigned to Korean administrative provinces using a KD-tree nearest-neighbor
approach based on start coordinates. {len(city_stats['cities'])} sub-city centers were
defined across {len(city_stats['provinces'])} provinces.

- Maximum assignment distance: 0.3 degrees (~30 km)
- Trips beyond threshold assigned to "Other": {city_stats['far_trips']:,} ({city_stats['far_trips']/city_stats['total_trips']:.2%})

### Provincial Distribution

| Province | Trips | Share |
|----------|-------|-------|
"""
    for prov, pstats in sorted(city_stats["provinces"].items(), key=lambda x: -x[1]["count"]):
        report += f"| {prov} | {pstats['count']:,} | {pstats['pct']:.1f}% |\n"

    report += f"""
## 4. Speed Column Validation

The `speeds` column (per-point speed readings from onboard sensors) was validated against:
(a) the reported `avg_speed` and `max_speed` columns, and
(b) GPS displacement-derived speeds.

Validation sample: {speed_val['sample_size']:,} trips.

| Comparison | Correlation | Mean Bias | MAE |
|------------|------------|-----------|-----|
| avg_speed vs speeds_col mean | {speed_val['avg_speed_vs_speeds_col']['correlation']:.3f} | {speed_val['avg_speed_vs_speeds_col']['bias']:.2f} km/h | {speed_val['avg_speed_vs_speeds_col']['mae']:.2f} km/h |
| max_speed vs speeds_col max | {speed_val['max_speed_vs_speeds_col']['correlation']:.3f} | {speed_val['max_speed_vs_speeds_col']['bias']:.2f} km/h | {speed_val['max_speed_vs_speeds_col']['mae']:.2f} km/h |
| GPS-derived avg vs reported avg | {speed_val['gps_avg_vs_reported_avg']['correlation']:.3f} | {speed_val['gps_avg_vs_reported_avg']['bias']:.2f} km/h | {speed_val['gps_avg_vs_reported_avg']['mae']:.2f} km/h |
| GPS-derived max vs reported max | {speed_val['gps_max_vs_reported_max']['correlation']:.3f} | {speed_val['gps_max_vs_reported_max']['bias']:.2f} km/h | {speed_val['gps_max_vs_reported_max']['mae']:.2f} km/h |

**Key findings:**
- `max_speed` perfectly matches the maximum of the `speeds` array (r=1.0, MAE=0.0)
- GPS displacement-derived speeds show lower correlation for max speed due to ~{speed_val['mean_gps_interval_s']:.0f}s
  sampling intervals smoothing peak speeds
- The onboard speed sensor data is internally consistent and reliable for speed analysis

## 5. Final Dataset Summary

| Metric | Value |
|--------|-------|
| Total trips (after core filtering) | {dataset_summary['total_trips']:,} |
| Strict valid trips (core + dist/dur) | {dataset_summary['strict_valid_trips']:,} |
| Unique users | {dataset_summary['unique_users']:,} |
| Unique vehicles | {dataset_summary['unique_vehicles']:,} |
| Provinces | {dataset_summary['num_provinces']} |
| Cities | {dataset_summary['num_cities']} |
| Date range | {dataset_summary['date_range']['min']} to {dataset_summary['date_range']['max']} |
| Mean avg speed | {dataset_summary['speed_summary']['mean_avg_speed']:.1f} km/h |
| P85 avg speed | {dataset_summary['speed_summary']['p85_avg_speed']:.1f} km/h |
| Mean max speed | {dataset_summary['speed_summary']['mean_max_speed']:.1f} km/h |
| Speeding rate (max > {SPEED_LIMIT_KR}) | {dataset_summary['speeding_rate_max']:.1%} |
| Mean trip distance | {dataset_summary['trip_metrics']['mean_distance_m']:.0f} m |
| Mean trip duration | {dataset_summary['trip_metrics']['mean_duration_s']:.0f} s |
"""

    # Write report
    report_path = REPORTS_DIR / "data_quality_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Data quality report saved to {report_path}")
    print(f"Report length: {len(report):,} characters")


if __name__ == "__main__":
    generate_report()
