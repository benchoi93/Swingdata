# Paper Outline: Characterizing E-Scooter Riding Safety Through City-Scale Speed Profile Analysis

**Target venue:** Transportation Research Part C: Emerging Technologies
**Authors:** [TBD — Choi, ...]
**Status:** Outline draft (2026-02-10)

---

## Draft Abstract (250 words)

We present the first large-scale empirical characterization of e-scooter riding speed behavior using 2.83 million GPS trajectories with per-point speed profiles and 20 million trip records from Swing, a major Korean e-scooter sharing operator, covering multiple cities throughout 2023. Unlike prior studies limited to small samples or aggregate trip statistics, our dataset provides full within-trip speed vectors at approximately 10-second intervals, enabling fine-grained analysis of speed dynamics along individual rides. We define and operationalize a suite of safety-relevant speed indicators — including speeding exposure rate, speed variability index, maximum sustained speed duration, and harsh acceleration/deceleration frequency — and compute these at the trip, user, spatial, and temporal levels. Using map-matched trajectories overlaid on OpenStreetMap road network attributes, we identify spatial hotspots where riders systematically exhibit unsafe speed behavior and quantify the association between road infrastructure characteristics (road class, speed limit, cycling infrastructure presence, intersection density) and observed riding speed. We estimate mixed-effects regression models to disentangle the contributions of rider demographics (age), vehicle model (S7 vs. S9), trip purpose proxies (time of day, day of week), city-level regulatory environments, and road-level infrastructure on speeding propensity and speed variability. Our results reveal substantial heterogeneity in speed behavior across user groups, vehicle types, and urban contexts, with specific infrastructure configurations consistently associated with elevated risk. We provide actionable recommendations for speed limit enforcement, geofencing design, infrastructure investment, and vehicle fleet management. This work contributes both a methodological framework for speed-based micromobility safety assessment and empirical evidence to inform data-driven regulation.

---

## Research Questions

**RQ1.** What are the distributional characteristics of e-scooter riding speed at the trip and within-trip levels, and how prevalent is speeding (exceeding posted or regulatory speed limits) across a city-scale dataset?

**RQ2.** Where do spatially concentrated speed-related safety risks occur, and what road infrastructure attributes (road class, speed limit, cycling infrastructure, intersection density, land use) are associated with elevated speeding and speed variability?

**RQ3.** How do rider demographics (age group), vehicle characteristics (model, mode), and temporal factors (time of day, day of week, month) influence riding speed behavior and speeding propensity?

**RQ4.** Can we identify distinct rider speed behavior typologies (e.g., consistently cautious vs. habitual speeders), and how do these typologies distribute across cities and demographic groups?

---

## Detailed Paper Outline

### 1. Introduction (3-4 pages)

**Key content:**
- The rapid global expansion of shared e-scooters and mounting safety concerns (injury rates, fatalities, conflicts with pedestrians and vehicles)
- The regulatory gap: cities imposing speed limits (e.g., 25 km/h in Korea, 20 km/h in some EU cities) without empirical evidence on actual compliance
- Limitations of prior work: small samples, survey-based, aggregate statistics, single-city, no within-trip speed dynamics
- Our contribution: first city-scale analysis with full speed profiles from operator data, covering multiple cities and vehicle types
- Brief overview of dataset, methods, and key findings
- Paper organization

**Motivation framing (for TR-C):**
- Position as both a transportation safety contribution and a data science / emerging technology contribution (novel use of operator telemetry data at unprecedented scale)

### 2. Literature Review (3-4 pages)

**2.1 E-Scooter Safety: Epidemiological and Behavioral Evidence**
- Injury statistics and crash patterns (Trivedi et al., 2019; Bloom et al., 2021; Yang et al., 2020)
- Naturalistic riding studies with instrumented scooters (small N)
- Survey-based studies on perceived safety and risk factors
- Gap: lack of large-scale behavioral data from actual trips

**2.2 Speed as a Safety Indicator for Micromobility**
- Speed-crash severity relationship established for motor vehicles (Elvik, 2013) and applicability to micromobility
- Regulatory speed limits for e-scooters across jurisdictions (Korea: 25 km/h; Germany: 20 km/h; various US cities)
- Speed compliance studies (limited: mostly small GPS logger studies or video observation)
- Within-trip speed variability as a risk proxy (from automotive safety literature)

**2.3 Large-Scale GPS Trajectory Analysis for Transportation Safety**
- Taxi/ride-hailing GPS data for safety analysis (precedent)
- Bike-share GPS studies (speed behavior, route choice)
- E-scooter GPS studies: Jiao & Bai (2020), McKenzie (2019), Bai & Jiao (2020) — mostly spatial distribution, not speed profiles
- Map-matching techniques for GPS trajectories on road networks

**2.4 Gaps and Contributions**
- No prior study with millions of trajectories AND per-point speed profiles
- No multi-city comparison of e-scooter speed behavior
- No analysis linking infrastructure attributes to within-trip speed dynamics at scale

### 3. Data Description (3-4 pages)

**3.1 Data Source and Collection**
- Swing operator overview: market share in Korea, fleet size, operational cities
- Data sharing agreement and IRB/ethical considerations
- Two datasets:
  - **Trajectory dataset** (May 2023): 2.83M trips with GPS routes (~10s interval), per-point timestamps, and per-trip speed vector (`speeds` column), `avg_speed`, `max_speed`
  - **Trip-level dataset** (full year 2023): 20M records with `user_id`, `age`, `vehicletype`, `smodel` (S7, S9), `mode` (SCOOTER_TUB, SCOOTER_STD), billing data (`bill_amt`, `disc_amt`), temporal attributes

**3.2 Data Schema and Key Variables**
- Trajectory-level: `route_id`, `user_id`, `model`, `mode`, `travel_time`, `distance`, `moved_distance`, `avg_speed`, `max_speed`, `speeds` (vector), `routes` (GPS point list with timestamps), `start_x/y`, `end_x/y`, `avg_point_gap`, `max_point_gap`, `points`
- Trip-level: `ride_id`, `date`, `user_id`, `age`, `smodel`, `mode`, `bill_amt`, `disc_amt`, `cpn_nm`, `pass_nm`, temporal columns

**3.3 Data Quality and Preprocessing**
- GPS noise filtering: outlier speed removal (physical impossibility threshold, e.g., >50 km/h for e-scooters)
- Trajectory filtering: minimum trip length (>100m), minimum duration (>60s), minimum GPS points (>5)
- Handling of GPS gaps (`avg_point_gap`, `max_point_gap` used for quality screening)
- Speed vector validation: cross-check `speeds` vector against computed speeds from GPS displacement/time
- Missing data summary

**3.4 Descriptive Statistics**
- Trip count by city, month, day of week, hour of day
- Trip length and duration distributions
- User demographics (age distribution)
- Vehicle model and mode distributions
- Repeat usage patterns (trips per user)

> **Table 1.** Summary statistics of the trajectory dataset (May 2023) and the trip-level dataset (full year 2023).
> **Table 2.** Data coverage by city: number of trips, unique users, unique vehicles, and spatial coverage area.
> **Figure 1.** Spatial distribution of trip origins across Korean cities (map).
> **Figure 2.** Temporal distribution of trips: (a) by hour of day, (b) by day of week, (c) by day of month.

### 4. Methodology (5-6 pages)

**4.1 Map-Matching GPS Trajectories to Road Network**
- Road network source: OpenStreetMap (OSM) via `osmnx`
- Map-matching algorithm: Hidden Markov Model-based approach (Newson & Krumm, 2009), implemented via `Leuven Map Matching` or `FMM` (Fast Map Matching; Yang & Gidofalvi, 2018)
- Rationale for HMM over geometric nearest-road: handles GPS noise, maintains topological consistency
- Road attribute extraction after matching: road class (`highway` tag), posted speed limit, number of lanes, presence of cycling infrastructure (`cycleway` tag), one-way status
- Intersection density computed from OSM node degree within buffer zones
- Validation: sample manual inspection, comparison of matched distance vs. reported `distance`

**4.2 Safety-Relevant Speed Indicators**
- **Trip-level indicators:**
  - Mean speed (from `avg_speed` or recomputed from `speeds`)
  - Maximum speed (from `max_speed`)
  - 85th percentile speed (P85) within the trip
  - Speed variability: standard deviation of within-trip speed vector, coefficient of variation
  - Speeding rate: fraction of speed observations exceeding the regulatory limit (25 km/h) or the road-specific posted speed limit
  - Harsh event frequency: count of acceleration/deceleration events exceeding threshold (e.g., |a| > 2 m/s^2), computed from consecutive speed differences divided by time interval

- **Segment-level indicators (after map-matching):**
  - Mean segment speed (average speed of all trips traversing a road segment)
  - Speeding exposure: fraction of traversals where mean segment speed exceeds the posted limit
  - Speed variance across traversals (indicates inconsistent behavior)

- **User-level indicators (aggregated across trips):**
  - User mean speed, user max speed (across all trips)
  - User speeding propensity: fraction of trips with any speeding event
  - User speed consistency: variance of trip-level mean speeds

- **Spatial indicators:**
  - Grid-cell level (e.g., 250m x 250m) aggregation of speeding rate and speed variability
  - Kernel density estimation (KDE) of high-speed events for hotspot mapping

> **Table 3.** Definitions, formulas, and rationale for all safety-relevant speed indicators.
> **Figure 3.** Illustration of within-trip speed profile extraction and indicator computation for a sample trip.

**4.3 Spatial Analysis of Speed Behavior**
- Spatial aggregation at two resolutions:
  - Road segment level (from map-matching)
  - Grid cell level (250m hexagonal or square grid)
- Hotspot detection: Getis-Ord Gi* statistic to identify statistically significant clusters of high speeding rates
- Spatial autocorrelation: Global and local Moran's I for speed indicators
- Overlay with infrastructure attributes and land use (from OSM and, if available, Korean land use data)

**4.4 Statistical Modeling of Speed Behavior**

**4.4.1 Model 1: Trip-Level Speeding Propensity (Binary Outcome)**
- Dependent variable: whether a trip contains any speeding event (binary)
- Model: Mixed-effects logistic regression with random intercepts for `user_id` and `city`
- Fixed effects: `age` (categorical: <20, 20-29, 30-39, 40-49, 50+), `model` (S7/S9), `mode` (TUB/STD), `hour_of_day` (categorical or spline), `day_of_week`, `trip_distance`, `city`
- Rationale for mixed effects: repeated measures per user, city-level clustering

**4.4.2 Model 2: Trip-Level Mean Speed (Continuous Outcome)**
- Dependent variable: trip-level mean speed
- Model: Linear mixed-effects model with random intercepts for `user_id` and `city`
- Same fixed effects as Model 1, plus road-class composition of the trip (fraction of trip on primary/secondary/tertiary/residential roads)

**4.4.3 Model 3: Segment-Level Speeding Rate (Continuous, Bounded)**
- Dependent variable: fraction of traversals exceeding speed limit on a road segment
- Model: Beta regression (for proportions in (0,1))
- Predictors: road class, posted speed limit, number of lanes, cycling infrastructure presence, intersection density, land use type, mean trip volume (exposure control)

**4.4.4 Model 4: User-Level Speed Behavior Typology**
- Latent class analysis or Gaussian mixture model on user-level speed indicators (mean speed, speed variability, speeding propensity, harsh event rate)
- Determine number of classes via BIC
- Profile each class by demographics and usage patterns
- Multinomial logistic regression: predict class membership from age, city, trip frequency

> **Table 4.** Variable definitions and descriptive statistics for all model covariates.
> **Table 5.** Model specification summary for Models 1-4.

### 5. Results (6-8 pages)

**5.1 Descriptive Speed Behavior Characterization**
- Distribution of trip-level mean speed, max speed, P85 speed
- Distribution of within-trip speed variability
- Speeding prevalence: % of trips with at least one speeding event; % of total riding time spent above limit
- Comparison across vehicle models (S7 vs. S9) and modes (TUB vs. STD)
- Comparison across cities

> **Figure 4.** Distributions of trip-level mean speed and max speed: (a) overall, (b) by vehicle model, (c) by city.
> **Figure 5.** Distribution of within-trip speed variability (CV of speed) by vehicle model and city.
> **Figure 6.** Speeding prevalence by hour of day and day of week (heatmap).
> **Table 6.** Summary of speed indicators by city, vehicle model, and mode.

**5.2 Spatial Patterns of Speed Behavior**
- City-level maps of mean speed and speeding rate by grid cell
- Hotspot analysis results: Getis-Ord Gi* maps
- Road-segment level analysis: speeding rate by road class
- Case study: detailed spatial analysis for Seoul (largest city) and one smaller city (Daejeon or Ulsan)

> **Figure 7.** Spatial heatmaps of mean riding speed for Seoul: (a) grid-cell mean speed, (b) Gi* hotspot map.
> **Figure 8.** Mean speeding rate by road class (box plots) across all cities.
> **Figure 9.** Detailed segment-level speeding map for a selected urban corridor.

**5.3 Infrastructure and Contextual Associations (Model 3 Results)**
- Beta regression results: which road attributes are significantly associated with segment-level speeding rate
- Effect sizes and interpretation
- Interaction effects (e.g., cycling infrastructure x road class)

> **Table 7.** Beta regression results for segment-level speeding rate (Model 3).
> **Figure 10.** Marginal effects plots for key infrastructure variables on speeding rate.

**5.4 Rider and Trip Characteristics (Models 1 & 2 Results)**
- Mixed-effects logistic regression: odds ratios for speeding propensity
- Mixed-effects linear regression: coefficients for mean speed
- Key findings: age effect, vehicle model effect, temporal patterns
- Random effects: user-level heterogeneity (ICC), city-level variation

> **Table 8.** Mixed-effects logistic regression results for trip-level speeding propensity (Model 1).
> **Table 9.** Mixed-effects linear regression results for trip-level mean speed (Model 2).
> **Figure 11.** Predicted speeding probability by age group and hour of day (interaction plot).
> **Figure 12.** Random intercept distribution for users: illustrating rider heterogeneity.

**5.5 Rider Speed Behavior Typologies (Model 4 Results)**
- Optimal number of latent classes (BIC plot)
- Profile of each class: speed characteristics, demographics, usage frequency
- Cross-city and cross-demographic distribution of classes
- Characterization of the "habitual speeder" class

> **Figure 13.** BIC plot for latent class selection.
> **Figure 14.** Radar/spider plots of speed indicator profiles for each rider typology.
> **Table 10.** Rider typology profiles: mean characteristics and demographic composition.
> **Table 11.** Multinomial logistic regression results predicting typology membership.

### 6. Discussion (3-4 pages)

**6.1 Key Findings and Interpretation**
- Speeding is [expected: prevalent / concentrated among specific groups] — contextualize against regulatory assumptions
- Infrastructure matters: [expected: certain road types consistently enable/discourage speeding]
- Vehicle design matters: differences between S7 and S9 reflect hardware-level safety implications
- Young riders [expected: higher risk] — but quantify the effect size
- Substantial user-level heterogeneity — a small fraction of habitual speeders may account for disproportionate risk

**6.2 Policy Implications**
- **Speed limit enforcement:** Evidence on compliance rates informs whether current limits (25 km/h in Korea) are respected and whether hardware-level speed governors are effective
- **Geofencing and dynamic speed management:** Spatial hotspot results directly inform where geofencing zones should be deployed
- **Infrastructure investment:** Road attributes associated with safer riding inform bike lane design and traffic calming priorities
- **Fleet management:** Vehicle model differences suggest operator-level interventions (e.g., deploying speed-limited models in high-risk areas)
- **User-targeted interventions:** Rider typology results enable targeted safety messaging or progressive penalties for habitual speeders

**6.3 Comparison with Prior Literature**
- How our speeding rates compare to smaller-scale studies in the US, Europe, and Asia
- Whether age-speed relationships align with prior survey-based findings
- Novel insights not available from prior work (within-trip dynamics, multi-city comparison)

**6.4 Limitations**
- Single operator (Swing) may not represent all e-scooter usage in Korea
- GPS accuracy (~10s interval) limits detection of very short-duration events
- Speed from operator data vs. directly measured speed (potential smoothing)
- May 2023 trajectory data only — seasonal variation not fully captured in spatial analysis (though trip-level data covers full year)
- No crash/incident data for direct safety outcome linkage
- Korea-specific regulatory and infrastructure context may limit generalizability
- Self-selection: users who choose e-scooters may differ from general population

### 7. Conclusions (1-2 pages)

- Restate the four research questions and summarize answers
- Emphasize the methodological contribution: a replicable framework for speed-based micromobility safety assessment using operator telemetry data
- Emphasize the empirical contribution: largest-ever characterization of e-scooter speed behavior
- Highlight top 3 actionable policy recommendations
- Future work:
  - Linking speed behavior to crash data when available
  - Real-time safety scoring for dynamic geofencing
  - Extending to other micromobility modes (e-bikes)
  - Deep learning approaches for trajectory-level risk prediction
  - Causal inference designs (e.g., before/after infrastructure changes)

---

## Expected Contributions

1. **Unprecedented empirical scale:** We provide the first city-scale characterization of e-scooter speed behavior using 2.83M trajectories with full within-trip speed profiles, two orders of magnitude larger than any prior study.

2. **Comprehensive safety indicator framework:** We define and operationalize a multi-level suite of speed-based safety indicators (trip, segment, user, spatial) that can be directly adopted by other researchers and operators for standardized micromobility safety assessment.

3. **Infrastructure-speed nexus:** We quantify the association between road infrastructure attributes (road class, cycling infrastructure, intersection density) and observed riding speed behavior through map-matched trajectories, providing evidence for infrastructure investment decisions.

4. **Multi-factor statistical models:** We disentangle the relative contributions of rider demographics, vehicle characteristics, temporal context, and infrastructure to speeding propensity using mixed-effects regression models that properly account for the hierarchical data structure (observations nested within users nested within cities).

5. **Rider typology for targeted intervention:** We identify empirically grounded rider speed behavior typologies through latent class analysis, enabling targeted safety interventions rather than one-size-fits-all regulations.

6. **Actionable policy recommendations:** We translate statistical findings into specific, implementable recommendations for speed limit calibration, geofencing zone design, fleet management, and infrastructure prioritization.

---

## Key Methodological Choices and Justifications

### 1. Map-Matching Approach

- **Method:** Hidden Markov Model (HMM)-based map-matching using the Fast Map Matching (FMM) library (Yang & Gidofalvi, 2018) or Leuven Map Matching.
- **Road network:** OpenStreetMap (OSM) extracted via `osmnx` (Boeing, 2017).
- **Why HMM over simple nearest-road:** GPS points at ~10s intervals can be 50-150m apart; simple snapping to the nearest road fails at intersections and parallel roads. HMM considers the sequence of observations and transition probabilities between road segments, maintaining topological consistency.
- **Why FMM:** Designed for large-scale trajectory matching (millions of trajectories), C++ backend with Python bindings, handles the computational scale of 2.83M trajectories.
- **Validation strategy:** (1) Compare matched path distance to reported `distance` and `moved_distance` fields; (2) manual inspection of 500 randomly sampled trajectories across cities; (3) report match rate (fraction of points successfully matched) and exclude trips below a quality threshold.
- **Alternative considered:** GraphHopper map-matching API — rejected due to rate limits and lack of control over parameters at this scale.

### 2. Safety Indicator Definitions

- **Speeding threshold:** Primary analysis uses the Korean regulatory limit of 25 km/h. Sensitivity analysis at 20 km/h and 30 km/h thresholds. Road-specific posted speed limits from OSM `maxspeed` tags used for segment-level analysis.
- **Speed variability:** Coefficient of variation (CV) of the within-trip speed vector, preferred over standard deviation because it normalizes for mean speed (a trip at mean 20 km/h with SD 5 km/h is more variable than one at mean 10 km/h with SD 5 km/h).
- **Harsh acceleration/deceleration:** Computed as (v_{t+1} - v_t) / delta_t where delta_t is the GPS interval. Threshold: |a| > 2 m/s^2 (approximately 0.2g, consistent with automotive safety literature for non-motorized modes). Note: ~10s GPS interval means this captures sustained rather than instantaneous events — we acknowledge this limitation.
- **Speeding exposure rate:** Time-weighted (fraction of trip duration above threshold) rather than point-count-weighted, to avoid bias from irregular GPS intervals.

### 3. Statistical Models

- **Mixed-effects models (Models 1 & 2):** Chosen because the data has a natural hierarchical structure — multiple trips per user, users nested within cities. Fixed effects capture population-average associations; random intercepts for users capture individual-level heterogeneity in baseline speed behavior; random intercepts for cities capture unmeasured city-level factors (e.g., topography, enforcement intensity, local culture). Software: `statsmodels` or `lme4` via `rpy2`.
- **Beta regression (Model 3):** Segment-level speeding rate is a proportion bounded in (0,1). OLS is inappropriate (can predict outside bounds, heteroscedastic by construction). Beta regression directly models proportions with a logit link. Software: `betareg` in R via `rpy2` or `statsmodels`.
- **Latent class analysis (Model 4):** Data-driven identification of distinct behavioral typologies without imposing a priori groupings. Gaussian mixture model with full covariance, number of components selected by BIC. Validated by examining class separation (entropy, average posterior probabilities). Software: `scikit-learn` GaussianMixture.
- **Alternative considered for Model 4:** k-means clustering — rejected because it assumes spherical clusters and does not provide probabilistic class membership. Also considered: latent profile analysis via `poLCA` in R.

### 4. Spatial Analysis

- **Grid resolution:** 250m hexagonal grid (H3 index system, Uber) for spatial aggregation. Hexagons preferred over squares because they have uniform adjacency (6 neighbors, equidistant) and reduce edge effects.
- **Hotspot detection:** Getis-Ord Gi* statistic with distance-based spatial weights (threshold: 500m). Reports both z-scores and FDR-corrected p-values to control for multiple testing.
- **Software:** `PySAL` / `esda` for spatial statistics, `geopandas` + `folium`/`matplotlib` for mapping.

### 5. Handling of Multiple Cities

- **City identification:** Reverse geocoding of trip origin coordinates to assign city, or spatial join with Korean administrative boundary polygons (available from Korean Statistical Geographic Information Service, SGIS).
- **Separate vs. pooled analysis:** Primary analysis pools all cities with city-level random effects. Supplementary analysis runs city-specific models to check for effect heterogeneity.
- **City sample size threshold:** Only include cities with >10,000 trips in May 2023 for robust estimation.

---

## Suggested Figure and Table List (Summary)

| # | Type | Description |
|---|------|-------------|
| Fig. 1 | Map | Spatial distribution of trip origins across Korean cities |
| Fig. 2 | Time series | Temporal distribution: (a) hourly, (b) daily, (c) monthly |
| Fig. 3 | Diagram | Illustration of within-trip speed profile and indicator extraction |
| Fig. 4 | Histogram/violin | Trip-level mean and max speed distributions by model and city |
| Fig. 5 | Violin/box | Speed variability (CV) by vehicle model and city |
| Fig. 6 | Heatmap | Speeding prevalence by hour x day of week |
| Fig. 7 | Map | Spatial heatmaps for Seoul: (a) mean speed, (b) Gi* hotspots |
| Fig. 8 | Box plot | Speeding rate by road class |
| Fig. 9 | Map | Segment-level speeding map for selected corridor |
| Fig. 10 | Marginal effects | Infrastructure variable effects on speeding rate |
| Fig. 11 | Interaction plot | Speeding probability by age group and hour |
| Fig. 12 | Histogram | Random intercept distribution for users |
| Fig. 13 | Line plot | BIC plot for latent class selection |
| Fig. 14 | Radar plot | Speed indicator profiles per rider typology |
| Tab. 1 | Summary | Dataset summary statistics |
| Tab. 2 | Summary | City-level data coverage |
| Tab. 3 | Definition | Safety indicator definitions and formulas |
| Tab. 4 | Summary | Model covariate definitions and statistics |
| Tab. 5 | Summary | Model specification summary |
| Tab. 6 | Results | Speed indicators by city, model, and mode |
| Tab. 7 | Results | Beta regression for segment-level speeding (Model 3) |
| Tab. 8 | Results | Mixed-effects logistic regression (Model 1) |
| Tab. 9 | Results | Mixed-effects linear regression (Model 2) |
| Tab. 10 | Results | Rider typology profiles |
| Tab. 11 | Results | Multinomial regression for typology membership |

---

## Preliminary Timeline

- [ ] Data preprocessing and quality control (2 weeks)
- [ ] Map-matching pipeline development and validation (2 weeks)
- [ ] Speed indicator computation (1 week)
- [ ] Descriptive analysis and visualization (1 week)
- [ ] Spatial analysis and hotspot detection (1 week)
- [ ] Statistical modeling (Models 1-4) (2 weeks)
- [ ] Writing: Introduction, Literature Review, Data, Methods (2 weeks)
- [ ] Writing: Results, Discussion, Conclusion (2 weeks)
- [ ] Internal review and revision (1 week)
- [ ] Submission to TR Part C

---

## Key References (Partial List)

- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126-139.
- Elvik, R. (2013). A re-parameterisation of the Power Model of the relationship between the speed of traffic and the number of accidents and accident victims. *Accident Analysis & Prevention*, 50, 854-860.
- Jiao, J., & Bai, S. (2020). Understanding the shared e-scooter travels in Austin, TX. *ISPRS International Journal of Geo-Information*, 9(2), 135.
- McKenzie, G. (2019). Spatiotemporal comparative analysis of scooter-share and bike-share usage patterns in Washington, D.C. *Journal of Transport Geography*, 78, 19-28.
- Newson, P., & Krumm, J. (2009). Hidden Markov map matching through noise and sparseness. *Proceedings of the 17th ACM SIGSPATIAL*, 336-343.
- Trivedi, B., et al. (2019). Injuries associated with standing electric scooter use. *JAMA Network Open*, 2(1), e187381.
- Yang, C., & Gidofalvi, G. (2018). Fast map matching, an algorithm integrating hidden Markov model with precomputation. *International Journal of Geographical Information Science*, 32(3), 547-570.
- Choi, S., Kim, J., & Yeo, H. (2021). TrajGAIL: Generating Urban Vehicle Trajectories using Generative Adversarial Imitation Learning. *Transportation Research Part C*, 128, 103091.
- Bloom, M. B., et al. (2021). Standing electric scooter injuries: Impact on a community. *The American Journal of Surgery*, 221(1), 227-232.

---

*This outline was drafted on 2026-02-10. To be refined as exploratory data analysis proceeds.*
