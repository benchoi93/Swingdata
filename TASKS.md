# Task List: E-Scooter Speed Governance Paper (AMAR)

**Status legend:** `[ ]` pending, `[~]` in progress, `[x]` done, `[!]` blocked

**Paper:** Behavioral Responses to E-Scooter Speed Governance (title TBD after results)
**Target:** Analytic Methods in Accident Research (AMAR)
**Plan:** See `NEWPLAN.md` for full paper design
**Design doc:** `docs/superpowers/specs/2026-03-09-amar-paper-design.md`

---

## Phase 0: Data Preparation (Priority: CRITICAL)

Build unified Feb-Nov 2023 dataset with all 8 genuinely behavioral outcomes.

- [x] 0.1 Filter `routes_all.parquet` to Feb-Nov 2023 with `has_speed_data=True`. Output: `data_parquet/v2/trips_feb_nov.parquet`. 23,728,759 rows (all modes incl. BIKE).
- [x] 0.2 Compute trip-level indicators for 8 primary outcomes + manipulation check. Output: `data_parquet/v2/trip_indicators_v2.parquet`. 23,718,206 rows. Script: `src/v2/prepare_data.py`.
- [x] 0.3 Road class assignment (start-point KDTree). Output: `data_parquet/v2/trip_road_classes_v2.parquet`. 50 city networks, 3.57M edges. Script: `src/v2/assign_road_class_v2.py`.
- [x] 0.4 Merge indicators + road class + city + age -> unified modeling dataset. Output: `data_parquet/v2/trip_modeling.parquet`. 19,454,309 scooter trips (TUB/STD/ECO only), 1,001,459 users, 52 cities. Script: `src/v2/build_modeling_dataset.py`.
- [x] 0.5 User-level aggregates. Output: `data_parquet/v2/user_modeling.parquet`. 1,001,459 users, avg 19.4 trips, 3.2 active months. Script: `src/v2/build_user_dataset.py`.
- [x] 0.6 Descriptive statistics for Table 1. Output: `data_parquet/v2/descriptive_stats.json`. Script: `src/v2/descriptive_stats.py`.
- [x] 0.7 Data quality verification. 3 minor warnings (extreme outliers). Output: `reports/v2/data_verification.json`. Script: `src/v2/verify_data.py`.

## Phase 1: Block 1 — Cross-Sectional Behavioral Profiles (Priority: HIGH)

All analyses use `data_parquet/v2/trip_modeling.parquet` (Feb-Nov 2023).

- [x] 1.1 Multi-dimensional mode comparison: TUB vs STD vs ECO on all 8 primary outcomes. Radar chart saved. Script: `src/v2/cross_sectional.py`. Fig: `figures/v2/fig1_radar_behavioral_profile.pdf`.
- [x] 1.2 GEE for EACH primary outcome (200K sample, reservoir): Poisson for counts, Gaussian for continuous, Exchangeable correlation. Script: `src/v2/cross_sectional.py`. Fig: `figures/v2/fig2_forest_gee_coefficients.pdf`.
- [x] 1.3 Within-subject paired comparison: 76,726 TUB-STD pairs, 10,710 TUB-ECO pairs. Cohen's d: harsh events d=1.30 (TUB vs ECO), cruise d=-1.11, speed_cv d=0.36. Script: `src/v2/cross_sectional.py`.
- [x] 1.4 LightGBM + SHAP: 3 models (harsh event AUC=0.734, cruise R2=0.248, speed_cv R2=0.464). is_tub dominates harsh event (SHAP=0.49). Script: `src/v2/shap_analysis.py`.
- [x] 1.5 Distribution figures: violin plots for 8 outcomes by mode. Script: `src/v2/distribution_figures.py`. Fig: `figures/v2/fig_supp_distributions.pdf`.

## Phase 2: Block 2 — Causal Multi-Outcome DiD (Priority: HIGH)

Uses Dec 2023 TUB ban as natural experiment.

- [x] 2.1 City-month panel built: 549 obs (52 cities x 11 months incl Dec 2023). Script: `src/v2/did_multi_outcome.py`. Output: `data_parquet/v2/city_month_panel_v2.parquet`.
- [x] 2.2 TWFE DiD: harsh_accel beta=-0.156 (Bonf. sig), harsh_decel beta=-0.190 (Bonf. sig), speed_cv/cruise/zero_speed NULL, trip_count/active_users NULL (no demand cost). Script: `src/v2/did_multi_outcome.py`. Fig: `figures/v2/fig4_did_multi_outcome.pdf`.
- [x] 2.3 Event study: 6 outcomes x 11 monthly coefficients (ref=Nov). Script: `src/v2/did_event_study.py`. Fig: `figures/v2/fig_supp_event_study.pdf`.
- [x] 2.4 Demand response: seasonal Dec decline across all quartiles (not treatment-related). Script: `src/v2/did_event_study.py`. Fig: `figures/v2/fig5_demand_response.pdf`.
- [x] 2.5 Dose-response: r=0.567 (speed), r=0.561 (harsh_accel), r=0.621 (harsh_decel), r=-0.516 (cruise). Script: `src/v2/did_multi_outcome.py`.
- [x] 2.6 Placebo test: ALL 6 outcomes PASS (all p>0.14). Script: `src/v2/did_multi_outcome.py`.

## Phase 3: Block 3 — Compensation Test (Priority: HIGH)

- [x] 3.1 STD/ECO-only DiD: mean_speed/harsh events increase (composition), speed_cv/cruise/zero_speed NULL. Script: `src/v2/compensation_test.py`.
- [x] 3.2 Mode-switcher within-user DiD: 24,357 switchers vs 79,700 never-TUB. ALL d negligible (max |d|=0.133). Script: `src/v2/compensation_test.py`.
- [x] 3.3 Cohen's d: ALL 6 outcomes negligible (|d| < 0.2). NO compensation on ANY margin. Fig: `figures/v2/fig6_compensation_cohens_d.pdf`.
- [x] 3.4 Placebo (Aug-Sep -> Oct): ALL 6 PASS (all |d| < 0.06). Script: `src/v2/compensation_test.py`.
- [x] 3.5 Summary: 3 "aggregate only" (composition), 3 NULL. Zero compensation. Report: `data_parquet/v2/phase3_compensation.json`.

## Phase 4: Block 4 — Behavioral Escalation Pathway (Priority: HIGH)

- [x] 4.1 Mediation: harsh_accel 70.0% [65.3, 74.8], mean_speed 63.2% [58.2, 66.0], harsh_decel 57.0% [53.4, 60.4] mediated via TUB (bootstrap 500 iter, 50K users). Dynamics: speed_cv 26.0%, cruise 20.6%, zero_speed 18.4%. Script: `src/v2/escalation_pathway.py`, `src/v2/bootstrap_mediation.py`.
- [x] 4.2 Experience curves: TUB adoption 23% (trip 1) -> 69.5% (trip 101-200). Fig: `figures/v2/fig7_experience_trajectories.pdf`.
- [x] 4.3 Survival: 3 endpoints (harsh event, high-CV, speeding). 127K users, 200K sample. Script: `src/v2/escalation_pathway.py`.
- [x] 4.4 KM curves: speeding median TUB=7 vs non-TUB=inf. Fig: `figures/v2/fig8_survival_multi_endpoint.pdf`.
- [x] 4.5 Cox PH: speeding HR(ever_tub)=5.73, C=0.819. Harsh event HR(tub_frac)=2.13. High-CV HR(tub_frac)=0.80 (protective — TUB caps speed mechanically).
- [x] 4.6 Mediation summary: safety outcomes 57-70% mediated by TUB adoption (bootstrap CIs all exclude zero); dynamics 18-27%. Report: `data_parquet/v2/phase4_escalation.json`, `data_parquet/v2/bootstrap_mediation.json`.

## Phase 5: Narrative Selection & Paper Writing (Priority: HIGH — after Phases 1-4)

- [x] 5.0 **DECISION**: Selected narrative C ("One Lever, Many Margins" — anti-Peltzman). Title: "Behavioral Responses to E-Scooter Speed Governance: Multi-Outcome Evidence from 21 Million Trips".
- [x] 5.1 Introduction: Already written (from previous session). Tautology framing, unconstrained margins.
- [x] 5.2 Background: Already written. Peltzman, natural experiments, research gaps.
- [x] 5.3 Data and Setting: Updated with Table 1 (descriptive stats by mode x outcome category).
- [x] 5.4 Methods: Rewritten for multi-outcome framework (8 outcomes, 4 blocks, Bonferroni, mode-switcher, multi-endpoint survival).
- [x] 5.5 Results: Complete rewrite with all v2 numbers. Tables: GEE, DiD, compensation, survival.
- [x] 5.6 Discussion: Written. Anti-Peltzman, selective response, escalation pathway, 3 policy recs.
- [x] 5.7 Conclusions: Written (~400 words).
- [x] 5.8 Compile: 21 pages, 5,642 words text + 395 captions (~6,037 total). No undefined refs. 16 figures/tables.
- [x] 5.9 Push to Overleaf: pushed successfully.

## Phase 6: Quality Assurance (Priority: MEDIUM)

- [x] 6.1 Bib verification: ALL 54 entries verified (done in previous session + 3 new entries added).
- [x] 6.2 Figure cross-refs: all resolve (confirmed via pdflatex with 0 undefined ref warnings).
- [x] 6.3 Word count: 5,642 text + 395 captions = ~6,037 total. Slightly below 8K target — room for expansion.
- [x] 6.4 Proofread for AMAR style: methods-forward, econometric rigor. 6 fixes: title consistency (Verify→Validate), P4 prediction justification, survival HR footnote, highlights.tex update, GEE table footnote, TWFE criticism acknowledgment + Goodman-Bacon (2021) ref added.
- [x] 6.5 Final Overleaf push: done.

## Phase 7: Model Enhancements (Priority: HIGH — collaborator feedback)

Suggested by 한양대 collaborators: add built environment controls + individual FE.

### 7A: Cycling Infrastructure Variable (Easy — restore from v1)

**Context for agent:** v1 computed `frac_cycling_infra` in `src/assign_road_class.py:206-209` by counting GPS points matching OSM `cycleway` edges. v2 (`src/v2/assign_road_class_v2.py`) simplified to start-point-only matching with `is_major_road` binary. The 50 city OSM GeoPackage networks are already at `data_parquet/osm_networks/{city}.gpkg`. The `ROAD_CLASS_MAP` in v2 already maps `cycleway -> cycling`. **Approach:** For each trip's start point (already matched via KDTree), check if `start_road_class == "cycling"` to create `is_cycling_infra` binary. This is a 1-line derivation from existing data — no need to re-run the full GPS trace matching.

- [x] 7A.1 Add `is_cycling_infra` column to `trip_modeling.parquet`: `start_road_class == 'cycleway'`. 255,047 trips (1.31%) on cycling infra. Script: `src/v2/add_cycling_infra.py`.
- [x] 7A.2 Verify distribution: 1.31% overall (TUB 1.34%, STD 1.27%, ECO 1.31%). Within 1-3% expectation.
- [x] 7A.3 Add `is_cycling_infra` as control in FE models: cruise +0.6% (p<0.001), zero_speed +0.5% (p<0.001), mean_speed +0.08 (p=0.003). Included in all 6 FE models.

### 7B: Individual Fixed Effects (Medium — methodological upgrade)

**Context for agent:** Current GEE (Task 1.2 in `src/v2/cross_sectional.py:155-260`) uses 200K sample with exchangeable correlation. Individual FE absorbs all time-invariant user traits (risk preference, skill, demographics), giving within-user causal interpretation. With 1M users, use `pyfixest` (demeaning approach, not dummy variables). Key spec: `outcome ~ mode_STD + mode_ECO + tod_* + is_weekend + is_major_road + is_cycling_infra + log_distance | user_id`. Reference categories: TUB (mode), midday (tod), weekday, 25-29 (age absorbed by FE). **Note:** age_group is time-invariant so it gets absorbed by user FE — cannot estimate its effect (this is expected and fine). Use full 19M dataset if pyfixest can handle it; otherwise sample 1-2M.

- [x] 7B.1 Install `pyfixest` v0.40.1. Compatible with Python environment.
- [x] 7B.2 Run individual FE models for 6 outcomes (2M sample, 1.8M after singleton removal). All converged, all mode effects p=0. FE OLS: speed_cv R2=0.335, cruise R2=0.296, zero_speed R2=0.347, mean_speed R2=0.540. FE Poisson: harsh_accel/decel converged. Script: `src/v2/individual_fe.py`. Output: `data_parquet/v2/phase7b_individual_fe.json`.
- [x] 7B.3 Compare FE vs GEE: all 12 comparisons SAME direction. Safety attenuates ~15-18% (endogeneity via user selection), dynamics STRONGER under FE. Fig: `figures/v2/fig_fe_vs_gee.pdf`.
- [x] 7B.4 Hausman-style: FE and GEE diverge substantially (mean_speed STD: GEE -1.94 vs FE -0.85; harsh_accel STD: GEE -0.61 vs FE -0.52). Endogeneity confirmed -> FE preferred. Key finding: mode effects SURVIVE individual FE, confirming causal interpretation.
- [ ] 7B.5 Update paper Methods (4.2) and Results with FE results. Add FE vs GEE comparison table.

### 7C: Land Use Variables (Hard — needs external data)

**Context for agent:** We have NO land use data currently. Korean public data options: (1) 국토교통부 토지이용현황도 from data.go.kr — SHP files by municipality, (2) OSM `landuse` tags — likely sparse in Korea, (3) population density from 통계청 SGIS, (4) POI density. **Agent should RESEARCH ONLY in 7C.1 — do NOT download or implement without PI approval on data source.** Focus on 7A and 7B first; 7C is lower priority.

- [ ] 7C.1 Research: what Korean land use data sources exist? Check OSM landuse tag coverage for our 52 cities (can query existing GeoPackages). Write findings to `reports/land_use_data_research.md`. **Research only — no implementation.**
- [ ] 7C.2 PI decision: which land use proxy to use. **BLOCKED until 7C.1 reviewed.**
- [ ] 7C.3 If approved: extract land use features per trip origin. Merge into modeling dataset.
- [ ] 7C.4 Add land use controls to FE/GEE models.

### 7D: Update Paper with New Variables

- [x] 7D.1 Update Data & Setting section with new variables. Added `is_cycling_infra` description (1.3% of trips).
- [x] 7D.2 Update Methods section with individual FE specification. Added Eq. 4 (FE), pyfixest citation, Hausman rationale.
- [x] 7D.3 Update Results with FE estimates. Added Table 5 (FE vs GEE, 12 comparisons). Updated discussion limitations. Added pyfixest bib entry.
- [ ] 7D.4 Push to Overleaf.

---

## Daily Agent Instructions

1. Read this file and `NEWPLAN.md` to determine current state
2. Work on the **lowest-numbered incomplete task** in the **lowest-numbered incomplete phase**
3. Update checkboxes as tasks complete
4. All analysis code goes in `src/v2/` (new scripts for v2 data)
5. All data outputs go in `data_parquet/v2/`
6. All figures go in `figures/v2/` (PDF + PNG, 300 DPI)
7. Paper files go in `src/paper/` (synced with Overleaf)
8. At session end, write a brief report to `reports/YYYY-MM-DD.md`
9. **After any paper file change**: commit + push to Overleaf
