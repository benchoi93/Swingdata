# Paper Plan: Speed Governor Mandate for E-Scooters

**Target venue:** Analytic Methods in Accident Research (AMAR)
**Template:** Elsevier CAS single-column (`cas-sc.cls`)
**Target length:** ~8,000-10,000 words (AMAR typical; methods-heavy papers welcome)
**Working directory:** `src/paper/` (synced to Overleaf via git subtree)

---

## Title

**"Do Speed Governors Work? Evidence from 21 Million E-Scooter Trips and a Natural Experiment in South Korea"**

## Core Thesis

Software speed governors are the single most effective intervention for e-scooter speed safety. We demonstrate this through converging evidence: (1) cross-sectional analysis of 21M trips showing governors reduce speeding 121-fold, (2) a natural experiment (Dec 2023 TUB ban) providing causal evidence of a 28pp speeding reduction with no behavioral compensation, and (3) survival analysis showing governors delay speeding onset 5.4x.

## Story Arc (Evidence Funnel)

```
Descriptive -> Associative -> Causal -> Survival -> Policy
   "How bad?"   "What drives?"  "Does banning   "Does it     "Mandate
                                 TUB help?"    prevent       governors"
                                               onset?"
```

## Data Plan

| Analysis | Dataset | Period | N | Notes |
|----------|---------|--------|---|-------|
| Cross-sectional (descriptive, regression, spatial, SHAP) | Full trips with speed data | Feb-Nov 2023 | ~21M trips | Re-run all from scratch |
| DiD causal | City-month panel | Feb-Dec 2023 | 53 x 12 = ~583 obs | Include Dec for post-treatment |
| Behavioral substitution | User-level pre/post | Nov-Dec 2023 | ~240K users | Mode-switcher analysis |
| Survival (time-to-first-speeding) | User-trip sequences | Feb-Nov 2023 | ~864K users | Pre-ban only |

**Critical rule:** All cross-sectional analyses use the SAME dataset (Feb-Nov 2023, ~21M trips with speed data). Only DiD/behavioral-sub use Dec 2023.

## Paper Structure

### 1. Introduction (~800 words)
- E-scooter injury epidemic and speed as primary risk factor
- Regulatory landscape: 25 km/h limits exist but compliance is unknown
- The speed governor question: does technological enforcement work?
- Gap: no large-scale evidence + no causal evidence
- Our contribution: (1) largest GPS speed study (21M trips), (2) natural experiment from TUB ban
- Brief overview of data, methods, key findings

### 2. Background (~1,000 words)
- 2.1 E-scooter safety and speed-related injuries
- 2.2 Speed governance: regulations vs technological enforcement
- 2.3 Natural experiments in transportation safety (DiD precedent in safety lit)
- Clear positioning of our contribution

### 3. Data and Setting (~800 words)
- 3.1 Swing platform: TUB (turbo, no governor), STD (standard), ECO (economy, strict governor)
- 3.2 Dataset: 21M trips, Feb-Nov 2023, 52 cities, sensor-based speeds at ~10s intervals
- 3.3 The December 2023 TUB ban (natural experiment setup)
- 3.4 Data quality, filtering, road class assignment
- Table 1: Descriptive statistics by mode

### 4. Methods (~1,200 words)
- 4.1 Speed indicators: speeding (>25 km/h), mean speed, speed CV, harsh events
- 4.2 Cross-sectional: Logistic GEE (speeding ~ mode + age + time + city + road_class, clustered by user)
- 4.3 Feature importance: LightGBM + SHAP
- 4.4 Experience analysis: GEE on trip rank, TUB adoption mediation
- 4.5 DiD: TWFE with continuous treatment intensity (pre-ban TUB share), event study, dose-response
- 4.6 Robustness: placebo test (Sep 2023), behavioral substitution (5 outcomes), mode-switcher (within-user)
- 4.7 Survival: KM curves, Cox PH static + time-varying, TUB mediation of experience

### 5. Results (~2,000 words)
- 5.1 Descriptive patterns: 29.6% speeding rate, mode distribution, temporal/spatial
- 5.2 Cross-sectional evidence: TUB OR=121 (GEE), SHAP dominance (3.6x), experience-as-TUB-adoption (98.4% mediation)
- 5.3 Causal evidence: TWFE beta=-0.568 (p<0.001), dose-response r=0.754, placebo PASS
- 5.4 No risk compensation: STD/ECO speeding unchanged (p=0.59), mode-switcher within-user d=0.069
- 5.5 Speeding onset: TUB median 15 trips vs non-TUB 81 trips, Cox TV HR=10.41

### 6. Discussion (~800 words)
- 6.1 Converging evidence for governor mandates (cross-sectional + causal + survival)
- 6.2 Policy recommendations: (1) mandatory governors, (2) graduated mode access, (3) geofencing
- 6.3 Comparison with prior e-scooter safety literature
- 6.4 Limitations: single operator, no crash data, Korean context

### 7. Conclusions (~400 words)

## Figures (~10)

1. Fig 1: Spatial distribution of trips across Korea (context)
2. Fig 2: Speed distributions by mode (violin/box — THE key visual)
3. Fig 3: Speeding rate temporal heatmap (hour x day)
4. Fig 4: Logistic OR forest plot (mode dominates)
5. Fig 5: SHAP bar plot (TUB >> everything)
6. Fig 6: Experience curve + TUB adoption overlay (dual-axis)
7. Fig 7: DiD mode trends (TUB ban event, before/after)
8. Fig 8: Dose-response scatter (TUB share vs speeding reduction)
9. Fig 9: Mode-switcher / behavioral substitution summary
10. Fig 10: Kaplan-Meier survival curves (TUB vs non-TUB)

Appendix figures: event study coefficients, robustness checks, Cox forest plot, DiD parallel trends

## What's IN vs OUT

**IN** (supports governor mandate story):
- Mode effect on speeding (OR=121)
- Experience as TUB adoption mechanism
- SHAP feature importance
- DiD causal analysis (all robustness)
- Behavioral substitution / mode-switcher
- Survival time-to-first-speeding
- Spatial hotspots (brief, in descriptive)
- Threshold sensitivity robustness

**OUT** (save for separate papers):
- GMM rider typology (segmentation paper)
- Curvature risk paradox (infrastructure paper)
- Detailed road class regression coefficients
- Mixed-effects model (mention only, not main)
- Detailed road class speed analysis
- Two-part hurdle model (mention only)
- Multinomial logit for cluster membership

## Key Technical Decisions

1. **Unified dataset**: Build ONE modeling parquet from Feb-Nov 2023 (~21M trips) with all needed features
2. **GEE for main model**: Logistic GEE handles within-user correlation at scale
3. **SHAP on subsample**: LightGBM + SHAP on ~2M stratified subsample (computational limit)
4. **DiD panel**: Reuse existing city_month_panel.parquet (already Feb-Dec 2023)
5. **Survival**: Reuse existing analysis (already Feb-Nov 2023)
6. **All figures**: PDF + PNG, 300 DPI, colorblind-friendly
