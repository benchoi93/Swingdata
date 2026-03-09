# Analysis Code for "Characterizing E-Scooter Riding Safety Through City-Scale Speed Profile Analysis"

**Authors:** Seongjin Choi
**Target Venue:** Transportation Research Part C: Emerging Technologies
**Date:** February 2026

## Overview

This directory contains all Python analysis code used to produce the results in the manuscript. The pipeline is organized into sequential stages matching the paper's methodology.

## Requirements

- Python 3.10+
- Key packages: `duckdb`, `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `geopandas`, `osmnx`, `h3`, `matplotlib`, `seaborn`, `libpysal`, `esda`

Install all dependencies:
```bash
pip install duckdb pandas numpy scipy statsmodels scikit-learn geopandas osmnx h3 matplotlib seaborn libpysal esda splot mapclassify
```

## Data Requirements

The analysis requires two CSV files from Swing (not included due to privacy restrictions):

1. `2023_05_Swing_Routes.csv` (~2.83M rows) - GPS trajectories with speed profiles
2. `2023_Swing_Scooter.csv` (~20M rows) - Trip-level data with user demographics

## Pipeline Execution Order

### Phase 1: Data Preprocessing

| Script | Description | Output |
|--------|-------------|--------|
| `config.py` | Project configuration (paths, constants) | -- |
| `profile_data.py` | Profile raw dataset schema and completeness | `data_parquet/data_profile.json` |
| `filter_trips.py` | Apply quality filters (I9 removal, GPS errors, min points) | `data_parquet/routes_filtered.parquet` |
| `assign_cities.py` | Assign cities via KD-tree nearest-neighbor | `data_parquet/routes_with_cities.parquet` |
| `city_stats.py` | Compute per-city summary statistics | `data_parquet/city_summary_stats.parquet` |
| `validate_speeds.py` | Cross-validate sensor speeds vs GPS-derived speeds | `data_parquet/speed_validation.parquet` |
| `build_cleaned_dataset.py` | Build final cleaned dataset with all quality flags | `data_parquet/cleaned/trips_cleaned.parquet` |
| `data_quality_report.py` | Generate data quality report | `reports/data_quality_report.md` |

### Phase 2: Speed Indicators

| Script | Description | Output |
|--------|-------------|--------|
| `compute_indicators.py` | Trip-level speed indicators (mean, max, P85, speeding rate, accel, etc.) | `data_parquet/trip_indicators.parquet` |
| `user_indicators.py` | User-level aggregated indicators | `data_parquet/user_indicators.parquet` |

### Phase 3: Road Network & Road Class Assignment

| Script | Description | Output |
|--------|-------------|--------|
| `extract_osm_networks.py` | Download OSM road networks for 50 cities | `data_parquet/osm_networks/*.gpkg` |
| `evaluate_map_matching.py` | Evaluate map-matching approaches (Leuven vs nearest-edge) | `figures/map_matching_evaluation.pdf` |
| `assign_road_class.py` | Assign road class to GPS points via nearest-edge | `data_parquet/trip_road_classes.parquet` |
| `road_class_speed_analysis.py` | Speed indicators by road class | `data_parquet/segment_indicators.parquet` |

### Phase 4: Spatial Analysis

| Script | Description | Output |
|--------|-------------|--------|
| `spatial_analysis.py` | H3 hexagonal aggregation, Moran's I, Getis-Ord Gi* | `data_parquet/spatial/*.parquet` |
| `spatial_figures.py` | Spatial hotspot maps (Seoul, Daejeon, corridor maps) | `figures/fig6_*.pdf`, `figures/fig7_*.pdf` |

### Phase 5: Statistical Modeling

| Script | Description | Output |
|--------|-------------|--------|
| `prepare_modeling_data.py` | Merge trip indicators with demographics, create modeling vars | `data_parquet/modeling/trip_modeling.parquet` |
| `regression_models.py` | Logistic regression (GEE), hurdle model | `figures/fig9_*.pdf`, `figures/fig10_*.pdf` |
| `mixed_effects_model.py` | Mixed-effects linear model, OLS | `figures/fig_mixed_effects_*.pdf` |
| `beta_regression.py` | Two-part hurdle model (logistic + beta/GLM) | Model results in reports/ |
| `latent_class_analysis.py` | GMM rider typology (K=4) | `data_parquet/modeling/user_classes.parquet` |
| `multinomial_class_model.py` | Multinomial logit for class membership | `figures/multinomial_coefficients.pdf` |
| `mode_comparison.py` | TUB vs ECO natural experiment | `figures/fig12_*.pdf` |
| `robustness_checks.py` | Threshold sensitivity, subsampling, city-specific models | `figures/robustness_*.pdf` |

### Phase 6: Publication Figures

| Script | Description | Output |
|--------|-------------|--------|
| `publication_figures.py` | Figures 1-5, 8, 11 (distribution plots, heatmaps, etc.) | `figures/fig1-5,8,11_*.pdf` |
| `publication_figures_2.py` | Figures 9-10 (regression OR plots) | `figures/fig9-10_*.pdf` |
| `graphical_abstract.py` | Graphical abstract for submission | `figures/graphical_abstract.pdf` |

### Verification

| Script | Description |
|--------|-------------|
| `verify_paper_statistics.py` | Automated verification of all key statistics cited in the paper |

## Configuration

All paths, constants, and thresholds are centralized in `config.py`:

- `SPEED_LIMIT_KR = 25` (Korean regulatory limit, km/h)
- `HARSH_ACCEL_THRESHOLD_10S = 0.5` (m/s^2, calibrated for ~10s GPS intervals)
- `MIN_TRIP_POINTS = 5`, `MIN_TRIP_DISTANCE = 100` m, `MIN_TRIP_DURATION = 60` s
- `MAX_PLAUSIBLE_SPEED = 50` km/h (GPS error filter)
- `RANDOM_SEED = 42`
- `H3_RESOLUTION = 8` (~461m hexagons)

## Key Technical Notes

1. **DuckDB** is used for large-scale queries to avoid loading the full dataset into memory
2. **Sensor-based speeds** (from `speeds` column) are used throughout, not GPS-derived speeds
3. **Harsh acceleration threshold** is 0.5 m/s^2 (not the automotive standard of 2.0 m/s^2), calibrated for the ~10-second GPS recording interval
4. The `speeds` column format is `[[s1, s2, ...]]` (nested list); parsed via DuckDB regex or `ast.literal_eval`
5. **I9 model** trips are excluded (moped-class vehicle with implausible speeds)
6. **Sentinel value** -999 appears in 6 numeric columns and is replaced with NULL

## Reproducibility

All scripts use `RANDOM_SEED = 42` for reproducible random sampling. DuckDB queries are deterministic. The full pipeline can be re-executed by running scripts in the order listed above.

## Output Structure

```
data_parquet/
  cleaned/trips_cleaned.parquet    # Main cleaned dataset (2.78M trips)
  trip_indicators.parquet          # Trip-level speed indicators
  user_indicators.parquet          # User-level aggregated indicators
  trip_road_classes.parquet        # Road class composition per trip
  segment_indicators.parquet       # Road-class-level speed indicators
  trip_h3_indices.parquet          # H3 hexagonal indices for origins/destinations
  modeling/
    trip_modeling.parquet           # Modeling-ready dataset with demographics
    user_modeling.parquet           # User-level modeling dataset
    user_classes.parquet            # GMM class assignments
  spatial/
    hex_*.parquet                   # H3 hex-aggregated spatial data per city
  osm_networks/
    *.gpkg                          # OSM road networks per city (GeoPackage)
figures/
  fig1-12_*.pdf                    # Main manuscript figures
  fig_*.pdf                        # Supplementary figures
  robustness_*.pdf                 # Robustness check figures
  graphical_abstract.pdf           # Graphical abstract
```
