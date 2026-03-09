# Data Quality Report: E-Scooter Speed Safety Analysis

## 1. Data Sources

The dataset was obtained from Swing, a leading e-scooter sharing operator in South Korea.
Two complementary datasets were provided:

| Dataset | Rows | Columns | Coverage | Key Contents |
|---------|------|---------|----------|-------------|
| Routes CSV | 2,830,461 | 25 | May 2023 | GPS trajectories, speed profiles, trip metrics |
| Scooter CSV | 20,045,245 | 20 | Jan-Oct 2023 | User demographics, billing, ride mode |

The Routes dataset contains GPS trajectories with ~10-second intervals and per-point speed
readings from the scooter's onboard sensors.

## 2. Preprocessing and Filtering

### 2.1 Sentinel Value Replacement

The raw data uses `-999` as a sentinel value instead of NULL for six numeric columns:
`distance`, `moved_distance`, `avg_point_gap`, `max_point_gap`, `avg_speed`, `max_speed`.
All sentinel values were replaced with NULL.

- Distance/gap sentinels: 20,844 trips (0.74%)

### 2.2 Quality Filtering Criteria

The following exclusion criteria were applied:

| Filter | Criterion | Trips Removed | Rate |
|--------|-----------|--------------|------|
| Excluded model | I9 (moped type) | 10,556 | 0.37% |
| Sentinel values | -999 in any numeric column | 20,844 | 0.74% |
| Invalid coordinates | Outside Korea bounds (lat 33-39, lon 124-132) | 162 | 0.0057% |
| Few GPS points | < 5 GPS readings | 34,820 | 1.23% |
| Implausible speed | max_speed > 50 km/h | 6,790 | 0.24% |

**Core filtering result**: 2,780,890 trips retained (98.2% of original)

### 2.3 Additional Quality Flags (Retained for Sensitivity Analysis)

| Flag | Criterion | Trips Flagged | Rate |
|------|-----------|--------------|------|
| Short distance | < 100 m | 110,456 | 3.90% |
| Short duration | < 60 s | 57,813 | 2.04% |
| Long duration | > 7,200 s (2 hrs) | 1,210 | 0.04% |

**Strict filtering** (core + distance/duration): 2,664,263 trips (94.1% of original)

### 2.4 Model-Level Filtering Summary

| Model | Total | Valid | Retention Rate |
|-------|-------|-------|---------------|
| S9 | 2,147,382 | 2,138,292 | 99.6% |
| W9 | 321,867 | 320,396 | 99.5% |
| S11 | 181,179 | 177,272 | 97.8% |
| S7 | 169,477 | 144,930 | 85.5% |
| I9 | 10,556 | 0 | 0.0% |

## 3. City Assignment

Trips were assigned to Korean administrative provinces using a KD-tree nearest-neighbor
approach based on start coordinates. 20 sub-city centers were
defined across 18 provinces.

- Maximum assignment distance: 0.3 degrees (~30 km)
- Trips beyond threshold assigned to "Other": 3,370 (0.12%)

### Provincial Distribution

| Province | Trips | Share |
|----------|-------|-------|
| Gyeonggi | 1,120,604 | 40.3% |
| Seoul | 577,133 | 20.8% |
| Daejeon | 154,075 | 5.5% |
| Chungnam | 119,434 | 4.3% |
| Gyeongnam | 114,830 | 4.1% |
| Gyeongbuk | 107,458 | 3.9% |
| Busan | 96,273 | 3.5% |
| Daegu | 84,556 | 3.0% |
| Ulsan | 81,987 | 3.0% |
| Jeonbuk | 81,036 | 2.9% |
| Incheon | 58,024 | 2.1% |
| Gangwon | 56,211 | 2.0% |
| Jeonnam | 43,742 | 1.6% |
| Chungbuk | 42,580 | 1.5% |
| Sejong | 19,519 | 0.7% |
| Jeju | 11,306 | 0.4% |
| Gwangju | 8,752 | 0.3% |
| Other | 3,370 | 0.1% |

## 4. Speed Column Validation

The `speeds` column (per-point speed readings from onboard sensors) was validated against:
(a) the reported `avg_speed` and `max_speed` columns, and
(b) GPS displacement-derived speeds.

Validation sample: 9,884 trips.

| Comparison | Correlation | Mean Bias | MAE |
|------------|------------|-----------|-----|
| avg_speed vs speeds_col mean | 0.844 | -0.95 km/h | 1.65 km/h |
| max_speed vs speeds_col max | 1.000 | 0.00 km/h | 0.00 km/h |
| GPS-derived avg vs reported avg | 0.671 | -1.05 km/h | 2.54 km/h |
| GPS-derived max vs reported max | 0.138 | 16.34 km/h | 18.21 km/h |

**Key findings:**
- `max_speed` perfectly matches the maximum of the `speeds` array (r=1.0, MAE=0.0)
- GPS displacement-derived speeds show lower correlation for max speed due to ~10s
  sampling intervals smoothing peak speeds
- The onboard speed sensor data is internally consistent and reliable for speed analysis

## 5. Final Dataset Summary

| Metric | Value |
|--------|-------|
| Total trips (after core filtering) | 2,780,890 |
| Strict valid trips (core + dist/dur) | 2,664,263 |
| Unique users | 382,328 |
| Unique vehicles | 81,525 |
| Provinces | 18 |
| Cities | 52 |
| Date range | 2023-05-01 to 2023-05-31 |
| Mean avg speed | 13.9 km/h |
| P85 avg speed | 18.6 km/h |
| Mean max speed | 22.9 km/h |
| Speeding rate (max > 25) | 29.6% |
| Mean trip distance | 878 m |
| Mean trip duration | 377 s |
