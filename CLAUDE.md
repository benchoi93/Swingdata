# CLAUDE.md — Swingdata (E-Scooter Speed Governance Paper)

## Project Overview

Research project analyzing e-scooter GPS trajectories from Swing (Korean operator) to study **behavioral responses to speed governance** across multiple safety dimensions. Target venue: **Analytic Methods in Accident Research (AMAR)**.

**Paper title (working):** Selected after results — candidates in NEWPLAN.md

**Core insight:** Speed governors don't just cap top speed — they reshape the entire riding behavioral profile. The paper analyzes 6 behavioral dimensions (speed level, speed variability, harsh events, cruising behavior, trip characteristics, distributional shape) across cross-sectional, causal, and survival frameworks.

**Full plan:** See `NEWPLAN.md` for paper design. See `TASKS.md` for task list.

## Directory Structure

- `src/` — Legacy analysis scripts (May 2023 single-month, Phases 1-12)
- `src/v2/` — **NEW: Scripts for AMAR paper** (Feb-Nov 2023 full dataset)
- `src/paper/` — Paper LaTeX files (synced with Overleaf via git subtree)
- `data_parquet/` — Legacy processed data
- `data_parquet/v2/` — **NEW: Feb-Nov 2023 unified dataset**
- `figures/v2/` — **NEW: Figures for AMAR paper**
- `figures/` — Legacy figures (v1 paper)
- `reports/` — Analysis reports
- `archive/paper_v1_trc/` — Archived TR-C paper (v1)
- `NEWPLAN.md` — Paper design document
- `TASKS.md` — Master task list for AMAR paper

## Data Plan

**Cross-sectional analyses:** Use `data_parquet/v2/trip_modeling.parquet` (Feb-Nov 2023, ~21M trips with speed data). ALL cross-sectional analyses must use this same dataset. Must include all 6 behavioral dimensions.

**Causal analyses (DiD):** Use `data_parquet/v2/city_month_panel.parquet` (Feb-Dec 2023, includes Dec for post-treatment). Multi-outcome TWFE on all 6 dimensions.

**Compensation test:** User-level mode-switcher analysis on all 6 dimensions, Nov-Dec 2023.

**Survival analysis:** Feb-Nov 2023 user-trip sequences. Multiple event types: first speeding, first harsh event, first high-CV trip.

**Raw data location:** `D:/SwingData/raw/` (24 monthly CSVs, 2022-01 to 2023-12)

## Data Pipeline (v2)

```
routes_all.parquet (44.1M trips, all months)
  |-- filter to Feb-Nov 2023, has_speed_data=True
  v
trips_feb_nov.parquet (~21M trips)
  |-- compute ALL 6 behavioral dimensions, assign road class, merge demographics
  v
trip_modeling.parquet (unified modeling dataset with 6 dimensions)
  |-- used by ALL cross-sectional analyses (Block 1, Block 4)
  |-- aggregated to city-month panel for DiD (Block 2, Block 3)
```

## 6 Behavioral Dimensions

| Dimension | Trip-level metrics | Column names |
|-----------|-------------------|--------------|
| Speed level | Mean speed, P85, max speed | mean_speed, p85_speed, max_speed |
| Speed variability | Speed CV, speed std | speed_cv, speed_std |
| Harsh events | Harsh accel count, harsh decel count | harsh_accel_count, harsh_decel_count |
| Cruising behavior | Cruise fraction, zero-speed fraction | cruise_fraction, zero_speed_fraction |
| Trip characteristics | Distance, duration | distance, duration |
| Distributional shape | Skewness, kurtosis of within-trip speeds | speed_skewness, speed_kurtosis |

## Key Conventions

- All data in Parquet format (via DuckDB or pandas)
- DuckDB for large-scale queries; pandas/geopandas for smaller operations
- Figures: matplotlib, 300 DPI, PDF + PNG, colorblind-friendly palettes
- Config in `src/config.py` (paths, thresholds, constants)
- Korean GPS bounds: lat 33-39, lon 124-132
- Speed limit: 25 km/h regulatory threshold for e-scooters in Korea
- Sentinel value -999 used instead of NULL in raw data
- v2 scripts go in `src/v2/`, v2 data in `data_parquet/v2/`, v2 figures in `figures/v2/`

## Paper Template

- Elsevier CAS single-column: `cas-sc.cls` + `cas-common.sty` + `cas-model2-names.bst`
- Overleaf remote: `overleaf` -> `https://git@git.overleaf.com/69af522801dd1b8a205557ae`
- Push to Overleaf: `git subtree push --prefix=src/paper overleaf master`
- Pull from Overleaf: `git subtree pull --prefix=src/paper overleaf master --squash`
- **ALWAYS commit before pushing to Overleaf** (subtree push only sends committed content)

## Relevant Skills

### `/figure-maker` — Figures
All 12+ figures for AMAR paper. Multi-dimensional profiles, multi-outcome DiD, quantile shifts. Output to `figures/v2/`.

### `/paper-writer` — Paper Writing
All paper sections. AMAR style: methods-forward, econometric rigor, safety application. Target ~8,000-10,000 words.

### `/experiment-code` — Analysis
Data preparation, multi-outcome models, DiD, survival. All code in `src/v2/`.

### `/commit` — Version Control
Commit completed work with descriptive messages.

## Technical Notes

- DuckDB `USING SAMPLE N`: samples N rows as percentage, NOT N rows
- DuckDB read_parquet: no `columns=` param; use SELECT
- Cleaned dataset: `start_lat`/`start_lon` (not `start_x`/`start_y`)
- Harsh accel threshold: 0.5 m/s^2 (not 2.0; 10s GPS intervals)
- osmnx 2.0.7 `graph_from_bbox`: takes `bbox=(west, south, east, north)`
- OSM maxspeed coverage in Korea: very low (1-12%); highway tags 100%
- `np.select` on Windows: provide explicit string default to avoid dtype error
- Unicode arrows cause UnicodeEncodeError on Windows cp1252; use `->`
- 2022-01 to 2023-01: speed columns are sentinel -999 (NO speed data)
- Speed data available: Feb 2023 - Dec 2023 only
- Mode names in raw CSVs: SCOOTER_TUB/SCOOTER_STD/SCOOTER_ECO (normalize to TUB/STD/ECO)
- ast.literal_eval on GPS strings: slow (~1300 trips/s); use DuckDB regex for scale
- GeoPackage -> osmnx graph: set_index([u,v,key]) for edges, set_index(osmid) for nodes
- MNLogit: use newton optimizer; lbfgs may fail on hessian
- statsmodels get_prediction summary_frame: CI column names vary by version; check dynamically
- DiD treatment variable: use Nov 2023 TUB share (not all-month mean)
- lifelines CoxTimeVaryingFitter: needs minimal covariates to avoid singular matrix
- city_summary_stats.parquet is at PROVINCE level; use CITY_CENTERS dict + KDTree for city assignment
- scipy.stats.skew/kurtosis for within-trip speed distributional shape metrics
