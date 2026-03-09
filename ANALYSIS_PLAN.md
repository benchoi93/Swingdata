# Extended Analysis Plan: New Research Questions

**Date:** 2026-02-16
**Status:** PLAN (pending approval before execution)

---

## Data Availability Summary

From exploration of existing datasets:

| Dataset | Rows | Key Columns |
|---------|------|-------------|
| `trip_modeling.parquet` | 2,780,890 | mode, age, speed indicators, temporal, spatial |
| `trips_cleaned.parquet` | 2,780,890 | start/end lat/lon, moved_distance, type (SCOOTER/BIKE), speeds_raw |
| `trip_h3_indices.parquet` | 2,780,890 | route_id, h3_origin, h3_dest (res 8) |
| `user_indicators.parquet` | 198,821 | user-level aggregated speed behavior |
| OSM GeoPackages (50 cities) | 3.57M edges | highway, lanes, surface, lit, cycleway (NO tunnel/bridge) |

**Mode distribution:**
| Mode | N trips | Speeding (%) |
|------|---------|-------------|
| SCOOTER_TUB | 1,277,925 | 55.7% |
| SCOOTER_STD | 933,212 | 0.5% |
| BIKE_TUB | 247,312 | 30.9% |
| none (S7) | 144,323 | 16.7% |
| SCOOTER_ECO | 112,017 | ~1.0% |
| BIKE_STD | 66,101 | 31.8% |

**Vehicle type:** SCOOTER = 2,460,494 (88.5%); BIKE = 320,396 (11.5%)

---

## Task 4: Routine Riding Patterns & Speeding

### Research Question
Do riders who routinely ride at the same time and location speed more?

### Hypothesis
Habitual riders develop comfort/overconfidence on familiar conditions, leading to higher speeding propensity.

### Methodology
1. **Compute user-level regularity metrics** (from trip_modeling):
   - Temporal entropy: `H_time = -sum(p_h * log(p_h))` over hour-of-day distribution per user
   - Spatial entropy: `H_space = -sum(p_hex * log(p_hex))` over h3_origin distribution per user
   - Schedule regularity: standard deviation of start_hour across trips
   - Low entropy = routine rider (always same time, same place)

2. **Define routine rider** (users with >= 10 trips):
   - Tertile split on combined temporal + spatial entropy
   - Bottom tertile = "routine" riders

3. **Statistical analysis**:
   - Compare speeding propensity across routine vs. non-routine (t-test, Mann-Whitney)
   - Logistic regression: `is_speeding ~ routine_indicator + mode + age + time_of_day + road_class`
   - Control for mode (critical confounder: routine commuters may prefer specific modes)

### Data Requirements
- trip_modeling.parquet: user_id, start_hour, is_speeding, mode, age
- trip_h3_indices.parquet: h3_origin per trip
- Filter: users with >= 10 trips for reliable entropy computation

### Expected Output
- Table: speeding rate by routine tertile
- Figure: speeding propensity vs. temporal/spatial entropy scatter
- Logistic regression coefficient for routine_indicator

### Feasibility: HIGH
All data available. Computation is straightforward (groupby + entropy).

---

## Task 5: Trip Distance Effect on Speeding

### Research Question
Does trip distance (or OD straight-line distance) affect speeding behavior?

### Preliminary Finding (from exploration)
| Distance Bin | N trips | Speeding Rate |
|-------------|---------|--------------|
| < 1 km | 1,509,386 | 3.7% |
| 1-3 km | 1,115,184 | 6.0% |
| 3-5 km | 116,857 | 8.3% |
| > 5 km | 39,463 | 7.9% |

Speeding increases with distance up to 3-5 km, then plateaus. But this may be confounded by mode (longer trips may be more likely TUB).

### Methodology
1. **Compute distance metrics** (from trips_cleaned):
   - `moved_distance`: actual GPS trace distance (already available)
   - `od_distance`: Haversine straight-line from start to end coords
   - `circuity`: moved_distance / od_distance (route directness)

2. **Descriptive analysis**:
   - Speeding rate by distance decile, stratified by mode
   - Mean speed vs. distance curve (does speed increase for longer trips?)
   - Circuity vs. speeding (do direct trips speed more?)

3. **Regression**:
   - Add `log(moved_distance)` to existing logistic model
   - Interaction: `log(distance) x mode` (does distance effect differ by mode?)
   - Control for road class composition (longer trips traverse more road types)

### Data Requirements
- trips_cleaned.parquet: moved_distance, start_lat/lon, end_lat/lon (join on route_id)
- trip_modeling.parquet: is_speeding, mode, age, time_of_day

### Expected Output
- Figure: speeding rate vs. distance by mode (line plot with CI)
- Regression table with distance coefficient and mode interaction
- Circuity analysis table

### Feasibility: HIGH
All data available. Simple join + regression. `distance_bin` already precomputed.

---

## Task 6: Route Familiarity Effect on Speeding

### Research Question
Do riders speed more on routes they've taken before?

### Preliminary Finding (from exploration)
- 58.8% of trips are on repeated OD hex pairs
- Familiar routes: 31.8% speeding vs. novel routes: 26.5% speeding
- But confounded: TUB mode users ride more frequently and speed more

### Methodology
1. **Define route familiarity** (from trip_h3_indices):
   - OD pair = (h3_origin, h3_dest) at resolution 8
   - For each user, sort trips chronologically
   - `familiarity_count`: how many prior times user took this OD pair
   - `is_familiar`: familiarity_count >= 2

2. **Within-subject design** (strongest causal inference):
   - Identify users with both novel and familiar trips
   - Paired comparison: same user, same mode, novel vs. familiar
   - Controls for all user-level confounders

3. **Regression analysis**:
   - `is_speeding ~ log(familiarity_count + 1) + mode + age + time_of_day + distance`
   - Within-user fixed effects (or GEE with user clustering)
   - Interaction: `familiarity x mode` (does familiarity effect differ by mode?)

4. **Dose-response**: Is there a gradient? (1st time vs 3rd vs 10th)

### Data Requirements
- trip_h3_indices.parquet: h3_origin, h3_dest
- trip_modeling.parquet: is_speeding, mode, age, time_of_day
- Need trip timestamps for chronological ordering (from trips_cleaned or route metadata)

### Challenge
- Need trip date/time for ordering. Check if trip_modeling has a date column or if we need to join from another source.
- trips_cleaned.parquet should have temporal ordering info.

### Expected Output
- Figure: speeding rate vs. familiarity count (dose-response curve)
- Within-subject paired comparison (like TUB vs ECO analysis)
- Regression table with familiarity coefficient

### Feasibility: HIGH
H3 OD pairs available. 1.6M familiar trips provides strong statistical power. Within-subject design requires chronological ordering.

---

## Task 7: Speeding on Tunnels and Bridges

### Research Question
Is speeding elevated on tunnels and bridges?

### Hypothesis
Tunnels and bridges have less pedestrian mixing, straighter geometry, possible downhill sections -> higher speeds.

### Data Gap
**OSM GeoPackages do NOT contain `tunnel` or `bridge` tags.** The original osmnx extraction did not include these in `useful_tags_way`.

### Methodology
1. **Re-extract OSM data for target cities** (not all 50; focus on top 5):
   - Add `tunnel`, `bridge` to `ox.settings.useful_tags_way`
   - Re-extract Seoul, Daejeon, Busan, Daegu, Suwon
   - Save updated GeoPackages

2. **Identify tunnel/bridge segments**:
   - Filter edges where `tunnel='yes'` or `bridge='yes'`
   - Count: how many tunnel/bridge edges per city
   - Compute total length of tunnel/bridge segments

3. **Trip-level analysis**:
   - For each trip, compute fraction of GPS points near tunnel/bridge edges (KDTree snap)
   - `frac_tunnel`, `frac_bridge` as trip-level covariates
   - Compare mean speed on tunnel/bridge segments vs. regular segments

4. **Within-trip analysis**:
   - For trips that pass through both tunnel and non-tunnel segments
   - Paired comparison of speed at tunnel points vs. non-tunnel points (same trip)

### Data Requirements
- Re-extracted OSM networks with tunnel/bridge tags
- trips_cleaned.parquet: GPS trajectories (routes_raw) for spatial matching
- This requires the same KDTree approach used for road class assignment

### Expected Output
- Table: speeding rate on tunnel vs. bridge vs. regular segments
- Figure: speed differential (tunnel - regular) per trip
- Map: tunnel/bridge locations with speeding overlay

### Feasibility: MEDIUM
Requires re-extracting OSM data (30-60 min for 5 cities). GPS point matching is computationally intensive but proven approach (Task 3 road class used same method). Risk: tunnel/bridge coverage may be sparse for scooter routes (scooters avoid highways).

---

## Task 8: Scooter vs. Bike Factor Comparison

### Research Question
How do speeding determinants differ across scooter, bike, and e-bike modes?

### Key Data
| Vehicle | Modes | N trips | Speeding |
|---------|-------|---------|----------|
| Scooter (S9, S11) | TUB, STD, ECO | 2,323,154 | varies by mode |
| Scooter (S7) | none | 144,323 | 16.7% |
| Bike (W9) | TUB, STD | 313,413 | 30.9-31.8% |

**Notable:** BIKE_TUB (30.9%) and BIKE_STD (31.8%) have nearly identical speeding rates, while SCOOTER_TUB (55.7%) vs SCOOTER_STD (0.5%) differ dramatically. This suggests the TUB/STD speed governor works very differently on bikes.

### Methodology
1. **Separate models by vehicle type**:
   - Scooter model: `is_speeding ~ mode + age + time_of_day + day_type + province + road_class`
   - Bike model: same covariates (but mode = TUB/STD only, no ECO)
   - Compare OR for each covariate

2. **Pooled interaction model**:
   - `is_speeding ~ vehicle_type * (age + time_of_day + province) + mode`
   - Test significance of interaction terms
   - Shows which factors are universal vs. vehicle-specific

3. **Key comparisons**:
   - Age effect: same direction/magnitude for scooter vs. bike?
   - Temporal effect: same peak hours for both?
   - Provincial effect: same geographic variation?
   - Mode effect: TUB/STD on scooter vs. TUB/STD on bike

4. **Visualization**: side-by-side OR forest plots (like robustness_city_models figure)

### Data Requirements
- trip_modeling.parquet: all columns (mode, age, is_speeding, etc.)
- trips_cleaned.parquet: `type` column for SCOOTER vs BIKE classification

### Expected Output
- Figure: side-by-side OR forest plot (scooter vs. bike)
- Table: regression coefficients for separate scooter and bike models
- Interaction significance table

### Feasibility: HIGH
All data available. Bike sample (313K trips) is large enough for reliable estimation. No new data extraction needed.

---

## Task 9: Nighttime Speeding Risk by Vehicle Type

### Research Question
Do scooter riders maintain unsafe speeds at night due to their smaller, less visible vehicles?

### Hypothesis
Scooters are smaller and harder to see at night. If riders don't slow down, nighttime risk is disproportionately higher for scooters vs. bikes.

### Methodology
1. **Day/night speeding comparison by vehicle type**:
   - Define: day = 07:00-19:00, night = 19:00-07:00
   - Compute speeding rate for scooter_day, scooter_night, bike_day, bike_night
   - Key metric: night-day speed differential per vehicle type

2. **Street lighting analysis** (OSM `lit` tag):
   - Seoul OSM edges have `lit` column (yes/no)
   - Check coverage: what fraction of edges have `lit` data?
   - For trips on lit vs. unlit roads at night: compare speeding rate
   - Interaction: `night x unlit x vehicle_type`

3. **Regression**:
   - `is_speeding ~ night x vehicle_type + age + mode + province`
   - Test: is the night coefficient larger for scooters than bikes?
   - If significant interaction: scooter riders don't slow down enough at night

4. **Speed differential analysis**:
   - For users who ride both day and night: within-subject comparison
   - Do bike riders reduce speed at night while scooter riders don't?

### Data Requirements
- trip_modeling.parquet: time_of_day (or start_hour), is_speeding, mode
- trips_cleaned.parquet: type (SCOOTER/BIKE)
- OSM Seoul edges: `lit` column (need to check coverage)

### Pre-check Needed
- What fraction of Seoul OSM edges have `lit` tag? If too sparse, use only time-of-day analysis.

### Expected Output
- Table: speeding rate by vehicle_type x day/night
- Figure: night-day speeding differential by vehicle type
- Regression table with night x vehicle_type interaction
- (If lit data sufficient) speeding on lit vs. unlit roads at night

### Feasibility: HIGH (time-of-day), MEDIUM (lit tag depends on coverage)

---

## Task 10: Speeding on Curved Roads

### Research Question
Do scooter riders reduce speed sufficiently on curved road segments?

### Hypothesis
Scooters have smaller wheels and higher center of gravity, making curves more dangerous. Riders who don't slow down on curves are at higher risk.

### Methodology
1. **Compute road curvature from OSM geometry**:
   - For each OSM edge, compute curvature = total angle change / length
   - Classify: straight (< 5 deg/100m), gentle curve (5-15), sharp curve (> 15)
   - Use edge LineString geometry from GeoPackage

2. **Assign curvature to trips**:
   - Same KDTree snap approach as road class assignment
   - For each GPS point, find nearest edge, get its curvature
   - Trip-level: `frac_curved`, `max_curvature`

3. **Within-trip analysis**:
   - For trips passing through both straight and curved segments
   - Compare speed at curved points vs. straight points
   - Speed reduction ratio = speed_curve / speed_straight

4. **Vehicle type comparison**:
   - Do scooter riders reduce speed on curves as much as bike riders?
   - Interaction: `curvature x vehicle_type` in regression

5. **Night x curve interaction**:
   - Curves at night: do riders slow down less? (visibility of curve)

### Data Requirements
- OSM edge geometries (already in GeoPackage)
- trips_cleaned.parquet: GPS trajectories (routes_raw) for point-level matching
- Curvature computation from shapely LineString coordinates

### Computational Cost
- Curvature computation: fast (geometry processing)
- GPS-to-edge matching: heavy (105M points, 463s for road class - similar cost here)
- May want to limit to top 5 cities

### Expected Output
- Table: mean speed by curvature class x vehicle type
- Figure: speed reduction ratio on curves (scooter vs. bike)
- Within-trip speed profiles showing curve behavior

### Feasibility: MEDIUM-HIGH
Curvature computable from existing geometry. GPS matching is proven but computationally expensive. Limit to top 5 cities for manageable runtime.

---

## Execution Priority & Dependencies

| Priority | Task | Feasibility | New Data? | Est. Time |
|----------|------|-------------|-----------|-----------|
| 1 | Task 5 (Distance) | HIGH | No (join only) | 1 hr |
| 2 | Task 8 (Scooter vs Bike) | HIGH | No (filter only) | 2 hr |
| 3 | Task 6 (Route Familiarity) | HIGH | No (H3 OD pairs) | 2 hr |
| 4 | Task 4 (Routine Patterns) | HIGH | No (entropy calc) | 2 hr |
| 5 | Task 9 (Nighttime Risk) | HIGH | No (time filter) | 1.5 hr |
| 6 | Task 10 (Curves) | MED-HIGH | Curvature from OSM | 3 hr |
| 7 | Task 7 (Tunnels/Bridges) | MEDIUM | Re-extract OSM | 4 hr |

**Dependency:** Tasks 7 and 10 both require OSM geometry work. Task 7 needs re-extraction with tunnel/bridge tags. Can be batched together.

**Recommended execution order:** 5 -> 8 -> 6 -> 4 -> 9 -> 10 -> 7

---

## Script Organization

All new analysis scripts in `src/analysis/`:
```
src/analysis/
  task5_distance_effect.py
  task6_route_familiarity.py
  task4_routine_patterns.py
  task7_tunnel_bridge.py
  task8_vehicle_comparison.py
  task9_nighttime_risk.py
  task10_curve_speeding.py
```

Output: figures in `figures/`, tables/reports in `reports/`.
