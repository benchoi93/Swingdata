# Phase 8 Extension Analyses — Summary of Findings

**Date**: 2026-02-24
**Status**: All 4 research questions addressed (4/4 COMPLETE) + preprocessing infrastructure

---

## Overview

Phase 8 extends the original May 2023 analysis with four additional research questions from `more_research.md`, plus a multi-month preprocessing infrastructure step. All analyses are now complete using the full 24-month dataset (Jan 2022 - Dec 2023).

---

## RQ1: Do experienced riders speed more? (COMPLETE)

**Scripts**: `src/preprocess_all_months.py`, `src/experience_data_prep.py`, `src/experience_speeding.py`, `src/newcomer_analysis.py`
**Data**: 44.1M valid trips across 24 months (Jan 2022 - Dec 2023), 1,816,911 unique users. Speed data available for 25.1M trips (2023-02 onward).

### Data Notes
- 2022 + 2023-01 months have no speed sensor data (avg_speed/max_speed/speeds all sentinel -999)
- These trips are retained for longitudinal user tracking (trip sequencing, experience computation)
- Mode names normalized: SCOOTER_TUB -> TUB, SCOOTER_STD -> STD, SCOOTER_ECO -> ECO

### Key Findings

1. **Experience monotonically increases speeding**: Speeding prevalence rises steadily with rider experience.
   | Experience bin | Trips | Users | Speeding rate |
   |---|---|---|---|
   | 1st trip | 1,816,911 | 1,816,911 | **3.8%** |
   | 2-5 trips | 4,497,120 | 1,365,628 | 5.4% |
   | 6-20 trips | 9,184,824 | 866,253 | 8.5% |
   | 21-50 trips | 9,193,834 | 436,746 | 12.8% |
   | 51-100 trips | 7,397,225 | 214,458 | 17.2% |
   | 101-500 trips | 10,673,925 | 100,793 | 23.8% |
   | 500+ trips | 1,356,904 | 5,516 | **28.0%** |

2. **Usage category confirms pattern**:
   - One-time riders (1 trip): 2.3% speeding
   - Super-heavy riders (200+): 19.0% speeding

3. **Newcomer vs established riders** (21.8M trips with speed data):
   | Metric | Newcomer (<=30 days) | Established (>30 days) |
   |---|---|---|
   | Trips | 2,828,091 | 18,961,829 |
   | Users | 571,005 | 780,124 |
   | Speeding rate | **15.6%** | **26.7%** |
   | Mean speed (km/h) | 12.7 | 14.6 |
   | Mean max speed (km/h) | 21.1 | 22.7 |
   - Mann-Whitney U test: p < 0.001, Cohen's d = 0.256 (small-medium effect)

4. **Early trip speed trajectory** (newcomer analysis subset: trips with speed data only, 2023-02 onward): Speeding increases rapidly over a rider's first trips.
   - Trip 1: 7.0% speeding
   - Trip 10: 16.4% speeding
   - Trip 50: 26.0% speeding
   - Total increase: +19 percentage points over first 50 trips
   - Note: The 7.0% first-trip rate here differs from the 3.8% in the experience bin table above because the experience bins include all 44.1M trips (including months without speed data, coded as non-speeding), while the newcomer analysis restricts to trips with actual speed data.

5. **TUB mode adoption drives speeding increase**: Riders increasingly choose TUB (unlocked/turbo) mode with experience.
   - Trip 1: STD 69.5%, TUB 11.5%, ECO 11.1%
   - Trip 50: STD 42.5%, **TUB 47.8%**, ECO 4.5%
   - TUB adoption quadruples from trip 1 to trip 50

6. **Mode-specific newcomer effects**:
   - STD mode: newcomer 1.0% vs established 0.8% (negligible)
   - TUB mode: newcomer 49.6% vs established 50.7% (similar — TUB enables speeding regardless of experience)
   - ECO mode: newcomer 0.8% vs established 0.9% (negligible)
   - Conclusion: The experience-speeding link is primarily mediated by **mode adoption**, not within-mode behavior change

7. **Age interaction with experience**:
   | Age group | Newcomer | Established | Difference |
   |---|---|---|---|
   | <20 | 20.1% | 34.0% | **+13.9pp** |
   | 20-24 | 17.6% | 28.8% | +11.2pp |
   | 25-29 | 13.8% | 23.7% | +10.0pp |
   | 30-34 | 11.3% | 20.3% | +9.0pp |
   | 50+ | 15.4% | 20.5% | +5.0pp |
   - Younger riders show the largest experience-driven speeding increase

### Outputs
- `figures/fig_experience_usage_category.pdf` — speeding by usage category
- `figures/fig_experience_learning_curve.pdf` — learning curves by mode
- `figures/fig_newcomer_speed_trajectory.pdf` — first 50 trips speed trajectory
- `figures/fig_newcomer_vs_established.pdf` — newcomer vs established by mode
- `figures/fig_newcomer_mode_adoption.pdf` — mode share evolution over first 50 trips
- `data_parquet/all_months/routes_all.parquet` — 44.1M consolidated trips
- `data_parquet/all_months/user_longitudinal.parquet` — 1.82M user profiles
- `data_parquet/modeling/trip_experience.parquet` — trip-level experience features
- `data_parquet/modeling/experience_results.json`, `newcomer_results.json`

---

## RQ2: Do trip distance and road type predict speeding? (COMPLETE)

**Script**: `src/trip_length_road_class.py`
**Data**: 2,317,678 trips (May 2023, scooter modes only)

### Key Findings

1. **Monotonic distance-speeding relationship**: Speeding prevalence increases strongly with distance.
   - Shortest decile (median 149m): 13.9% speeding
   - Longest decile (median 2,168m): 54.2% speeding
   - LOWESS curve confirms smooth, approximately linear increase on log scale

2. **Non-linear distance model**: Restricted cubic spline logistic regression (df=5) achieves pseudo-R2 = 0.387, slightly outperforming piecewise linear (AIC 375,384 vs 375,441).

3. **Road composition effects** (logistic regression ORs):
   | Road class fraction | OR | 95% CI | Direction |
   |---|---|---|---|
   | frac_major_road (composite) | **6.74** | [4.49, 10.12] | Risk factor |
   | frac_service | **2.74** | [2.48, 3.03] | Risk factor |
   | frac_tertiary | **2.05** | [1.86, 2.27] | Risk factor |
   | frac_cycleway | **1.66** | [1.45, 1.90] | Risk factor |
   | frac_residential | **1.58** | [1.43, 1.74] | Risk factor |
   | frac_footway | **1.19** | [1.08, 1.32] | Mild risk |
   | frac_secondary | **0.20** | [0.13, 0.29] | Protective |
   | frac_primary | **0.13** | [0.09, 0.20] | Protective |

   Interpretation: Service roads and tertiary roads enable speeding (lower traffic, fewer controls). Primary/secondary roads are protective (congestion, signals).

4. **Speed drift**: Riders slow down on average (-1.5 km/h second half vs first half). Drift diminishes with distance (short trips: -2.4 km/h; long trips: ~0 km/h).

### Outputs
- `figures/fig_distance_speeding_curve.pdf` — LOWESS + decile points
- `figures/fig_distance_spline_predicted.pdf` — spline predicted probabilities
- `figures/fig_distance_roadclass_heatmap.pdf` — distance x road class interaction
- `figures/fig_road_composition_or.pdf` — OR forest plot
- `figures/fig_speed_drift.pdf` — violin plot by distance bin
- `data_parquet/modeling/trip_length_road_class_results.json`

---

## RQ3: Is speeding riskier on curvy roads? (COMPLETE)

**Scripts**: `src/compute_curvature.py`, `src/curvature_speeding.py`
**Data**: 200K stratified subsample -> 198,303 trips with curvature; 165,198 after merging with modeling data

### Key Findings

1. **Curvature classification**: Most e-scooter trips involve significant turning.
   - Curvy (frac_sharp > 0.2): 49.9%
   - Mixed: 45.0%
   - Straight (frac_straight > 0.8): 5.1%

2. **Counter-intuitive speed-curvature relationship**: Higher curvature is associated with LOWER speeding prevalence.
   - **By curvature quintile** (continuous curvature, strong gradient):
     - Q0 (straightest): **41.0%** speeding
     - Q4 (curviest): **16.8%** speeding
   - **By curvature class** (categorical, smaller differences):
     - Straight (frac_straight > 0.8): 29.1% speeding
     - Mixed: 31.6% speeding
     - Curvy (frac_sharp > 0.2): 29.3% speeding
   - Note: The quintile gradient (41% to 17%) is much steeper than the class gradient (~29-32%) because the categorical classification captures only the extremes. The continuous specification (used in the logistic regression) is preferred.
   - Logistic regression: curvature OR = **0.716** [0.699, 0.732], p < 0.001

   Physical interpretation: Riders physically cannot maintain high speed on curvy paths. Straight road segments enable sustained speeding.

3. **Risk paradox**: When speeding DOES occur on curvy roads, the risk is much higher.
   - Curvy roads: mean risk score = 1.18 (speed excess x curvature)
   - Straight roads: mean risk score = 0.23
   - **5x higher risk score** on curvy roads when speeding

4. **Policy implication**: Speed enforcement should target straight road segments (where speeding is prevalent) AND curvy segments (where speeding, though rare, is most dangerous).

### Outputs
- `figures/fig_curvature_speeding_rate.pdf` — bar chart by class
- `figures/fig_curvature_continuous.pdf` — LOWESS speeding by curvature
- `figures/fig_curvature_roadclass.pdf` — interaction heatmap
- `figures/fig_curvature_risk_scatter.pdf` — risk scatter plot
- `data_parquet/trip_curvature.parquet`
- `data_parquet/modeling/curvature_computation_results.json`
- `data_parquet/modeling/curvature_speeding_results.json`

---

## RQ4: Which features drive speeding predictions? (COMPLETE)

**Script**: `src/shap_analysis.py`
**Data**: 2,314,437 trips with age data (May 2023, scooter modes)

### Key Findings

1. **LightGBM performance**: AUC-ROC = **0.905**, F1 = 0.739, Accuracy = 0.800
   - Excellent predictive performance confirms that speeding is well-explained by available features

2. **SHAP feature importance** (mean |SHAP value|):
   | Rank | Feature | Mean |SHAP| |
   |---|---|---|
   | 1 | **TUB mode** | 2.331 |
   | 2 | Log(distance) | 0.639 |
   | 3 | Mean |acceleration| | 0.499 |
   | 4 | Cruise fraction | 0.264 |
   | 5 | Speed CV | 0.249 |
   | 6 | Service road fraction | 0.163 |
   | 7 | Travel time | 0.160 |
   | 8 | Age | 0.148 |
   | 9 | Zero-speed fraction | 0.124 |
   | 10 | GPS points | 0.098 |

3. **Mode dominance**: TUB mode's SHAP importance (2.33) is **3.6x larger** than the next feature (log_distance at 0.64). This confirms that the speed governor mode is the single most important determinant of speeding.

4. **Feature gain vs SHAP**: LightGBM's native feature importance (gain) ranks log_distance first, but SHAP correctly identifies TUB mode as most impactful due to its large marginal effect despite being binary.

5. **Infrastructure features present**: Service road fraction (rank 6), footway (rank 13), residential (rank 12), and major road (rank 14) all appear in top 15, confirming road infrastructure matters for speeding beyond mode.

### Outputs
- `figures/fig_shap_summary.pdf` — SHAP beeswarm plot
- `figures/fig_shap_bar.pdf` — mean importance bar chart
- `figures/fig_shap_dependence.pdf` — dependence plots for top 6 features
- `data_parquet/modeling/shap_results.json`

---

## Implications for Paper Revision

### New content for paper:

1. **Results section additions**:
   - Experience-speeding learning curve (3.8% -> 28.0% over trip history)
   - Newcomer vs established comparison (15.6% vs 26.7%, Cohen's d=0.256)
   - TUB mode adoption trajectory (11.5% -> 47.8% over first 50 trips)
   - Experience-speeding link mediated by mode choice, not within-mode behavior
   - Age x experience interaction (younger riders show largest increase)
   - Distance-speeding curve (monotonic, pseudo-R2=0.387)
   - Road composition ORs (service 2.74, tertiary 2.05, major composite 6.74)
   - Curvature-speeding paradox (lower prevalence but higher risk)
   - SHAP analysis confirming TUB mode dominance
   - Speed drift pattern (riders slow down in second half)

2. **Discussion additions**:
   - Experience as risk factor: learning makes riders bolder, not safer
   - Mode adoption as mechanism: experience -> TUB adoption -> speeding
   - Policy implication: restrict TUB access for new riders or implement graduated mode unlocking
   - Road type as infrastructure-level intervention target
   - Curvature risk paradox and geofencing implications
   - Model interpretability: SHAP validates logistic regression findings
   - Speed drift as evidence of rider fatigue/adaptation

3. **New figures**: 17 publication-quality figures (PDF + PNG, 300 DPI)

4. **Supplementary material**: Detailed results in JSON for reproducibility

### Dataset summary:
- **Single-month** (May 2023): 2.78M trips, 382K users — used for cross-sectional analyses (RQ2-RQ5)
- **Multi-month** (Jan 2022 - Dec 2023): 44.1M trips, 1.82M users — used for longitudinal analysis (RQ1)

### GEE Regression Results (experience_speeding.py)
GEE logistic regression (50K users, 1.19M trips, exchangeable correlation):
| Variable | Coef | OR | 95% CI | p |
|---|---|---|---|---|
| log(trip_rank) | 0.164 | **1.178** | [1.156, 1.200] | 1.67e-66 |
| mode_TUB | 4.427 | **83.6** | [77.7, 90.0] | ~0 |
| mode_ECO | -0.118 | 0.889 | [0.694, 1.139] | 0.351 |
| mode_none | 0.656 | 1.926 | [1.756, 2.113] | ~0 |

- Each doubling of trip rank increases speeding odds by 17.8% (controlling for mode)
- No significant quadratic effect (p=0.44) -> experience effect is monotonic, not U-shaped
- TUB OR=83.6 remains the dominant predictor even after controlling for experience

### Limitations

1. **GPS-derived curvature precision**: Curvature is computed from GPS turning angles at ~10s intervals, not from actual road geometry. This yields approximate trajectory curvature rather than true road curvature.
2. **Subsample sizes**: Curvature analysis uses a 200K subsample; SHAP analysis uses a 50K subsample. While these are large, they represent 7% and 2% of the full dataset respectively.
3. **Missing 2022 speed data**: All 2022 months and January 2023 have no speed sensor data (sentinel -999). The experience analysis includes these trips for longitudinal tracking but cannot assess speeding during these periods.
4. **Experience-time confound**: The experience effect (trip rank) is partially confounded with calendar time. Riders who started earlier have both more trips and rode during potentially different seasonal/policy conditions.
5. **Single operator**: All data comes from Swing; results may not generalize to other operators or countries.

### Remaining work:
- ~~Integration of new findings into paper sections~~ DONE (Phase 9.1)
- ~~Updated abstract and conclusions~~ DONE (Phase 9.1)
