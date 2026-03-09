# Task List: Speed Governor E-Scooter Paper (AAP)

**Status legend:** `[ ]` pending, `[~]` in progress, `[x]` done, `[!]` blocked

**Paper:** "Do Speed Governors Work? Evidence from 21M E-Scooter Trips and a Natural Experiment"
**Target:** Accident Analysis & Prevention (AAP)
**Plan:** See `NEWPLAN.md` for full paper design

---

## Phase 0: Data Preparation (Priority: CRITICAL)

Build unified Feb-Nov 2023 dataset for all cross-sectional analyses.

- [ ] 0.1 Filter `routes_all.parquet` to Feb-Nov 2023 with `has_speed_data=True`. Output: `data_parquet/v2/trips_feb_nov.parquet`. Record row count (expect ~21M).
- [ ] 0.2 Re-compute trip-level speed indicators on full Feb-Nov dataset: mean_speed, max_speed, p85_speed, speed_cv, speeding_rate, speeding_duration, harsh_accel_count, cruise_fraction, zero_speed_fraction. Output: `data_parquet/v2/trip_indicators.parquet`.
- [ ] 0.3 Re-run road class assignment (DuckDB regex + KDTree) on Feb-Nov trips. Output: `data_parquet/v2/trip_road_classes.parquet`.
- [ ] 0.4 Merge trip indicators + road class + city assignment + user demographics (age from Scooter CSV). Create unified modeling dataset. Output: `data_parquet/v2/trip_modeling.parquet`.
- [ ] 0.5 Compute user-level aggregates from Feb-Nov data: speeding propensity, mode usage, trip count, experience measures. Output: `data_parquet/v2/user_modeling.parquet`.
- [ ] 0.6 Compute descriptive statistics for Table 1: by mode (TUB/STD/ECO), overall. Save to `data_parquet/v2/descriptive_stats.json`.
- [ ] 0.7 Verify data quality: check for nulls, sentinel values, GPS bounds, speed ranges. Write verification report to `reports/v2/data_verification.json`.

## Phase 1: Cross-Sectional Analysis (Priority: HIGH)

All analyses use `data_parquet/v2/trip_modeling.parquet` (Feb-Nov 2023).

- [ ] 1.1 Logistic GEE: speeding ~ mode + age_group + time_of_day + day_type + city + frac_major_road + log(distance), clustered by user_id. Report OR, 95% CI, p-values. Save results JSON + OR forest plot (Fig 4).
- [ ] 1.2 LightGBM + SHAP: Train on stratified 2M subsample. Compute SHAP values. Save SHAP bar plot (Fig 5) and summary plot. Report AUC, top-10 features.
- [ ] 1.3 Experience GEE: speeding ~ log(trip_rank) + mode + age_group + controls, clustered by user. Test within-mode experience effect. Compute TUB adoption curve. Save experience + adoption overlay figure (Fig 6).
- [ ] 1.4 Descriptive figures: (a) spatial distribution map (Fig 1), (b) speed by mode violin (Fig 2), (c) temporal heatmap (Fig 3). All from full Feb-Nov data.
- [ ] 1.5 Spatial analysis: H3 hex aggregation + Getis-Ord Gi* for top 5 cities. Brief summary for paper (not main focus). Save stats for text.

## Phase 2: Causal Analysis (Priority: HIGH)

Uses Dec 2023 TUB ban as natural experiment.

- [ ] 2.1 Rebuild city-month panel from `routes_all.parquet` for Feb-Dec 2023. Ensure consistent city assignment. Output: `data_parquet/v2/city_month_panel.parquet`.
- [ ] 2.2 TWFE DiD: speeding_rate ~ post * tub_share_nov + city_FE + month_FE. Report beta, SE, p-value. Save DiD mode trends figure (Fig 7).
- [ ] 2.3 Event study: monthly coefficients relative to Nov 2023. Check pre-trends. Save event study figure (Appendix).
- [ ] 2.4 Dose-response: scatter of TUB share reduction vs speeding reduction across cities. Report r, slope, bootstrap CI. Save dose-response figure (Fig 8).
- [ ] 2.5 Placebo test: use Sep 2023 as fake treatment date. Report beta and p-value.
- [ ] 2.6 Behavioral substitution: multi-outcome DiD on 5 outcomes (overall speeding, trip distance, STD/ECO speeding, STD/ECO mean speed, STD/ECO max speed). Report all betas.
- [ ] 2.7 Mode-switcher analysis: identify TUB->STD/ECO switchers vs never-TUB users. Within-user DiD on STD/ECO speed. Report Cohen's d. Save mode-switcher figure (Fig 9).
- [ ] 2.8 Mode-switcher placebo: repeat with Oct 2023 as placebo. Confirm null result.

## Phase 3: Survival Analysis (Priority: MEDIUM)

Uses Feb-Nov 2023 user-trip sequences.

- [ ] 3.1 Rebuild survival dataset from v2 trip data. Users with >=2 trips, Feb-Nov 2023. Record event/censored counts.
- [ ] 3.2 Kaplan-Meier: stratified by TUB-ever vs never-TUB. Log-rank test. Save KM figure (Fig 10).
- [ ] 3.3 Cox PH static: HR for ever_tub, tub_fraction, age, log_n_trips. Report C-index.
- [ ] 3.4 Cox time-varying: TUB adoption as time-varying covariate. Report HR for TUB adoption.
- [ ] 3.5 TUB mediation: compute % of experience-speeding gradient mediated by TUB adoption.

## Phase 4: Paper Writing (Priority: HIGH — after Phases 0-3)

All writing in `src/paper/`. Uses CAS template (`cas-sc.cls`).

- [ ] 4.1 Write Introduction (~800 words). Focus on safety problem, governor question, our evidence.
- [ ] 4.2 Write Background (~1,000 words). E-scooter safety, speed governance, natural experiments in safety.
- [ ] 4.3 Write Data and Setting (~800 words). Swing platform, dataset, Dec 2023 event, data quality. Table 1.
- [ ] 4.4 Write Methods (~1,200 words). All 7 methods sections per NEWPLAN.md.
- [ ] 4.5 Write Results (~2,000 words). All 5 results sections per NEWPLAN.md. Reference all figures/tables.
- [ ] 4.6 Write Discussion (~800 words). Converging evidence, 3 policy recs, comparison, limitations.
- [ ] 4.7 Write Conclusions (~400 words).
- [ ] 4.8 Compile references.bib with verified entries from existing bib + any new AAP-relevant refs.
- [ ] 4.9 Compile paper: pdflatex + bibtex. Fix all warnings. Target ~7,000 words.
- [ ] 4.10 Push to Overleaf: `git subtree push --prefix=src/paper overleaf master`.

## Phase 5: Quality Assurance (Priority: MEDIUM)

- [ ] 5.1 Verify all bib entries against academic databases.
- [ ] 5.2 Check all figure cross-references resolve.
- [ ] 5.3 Word count check (target 6,000-8,000 for AAP).
- [ ] 5.4 Proofread for AAP style: safety-focused language, policy recommendations prominent.
- [ ] 5.5 Prepare highlights (5 bullet points, AAP format).
- [ ] 5.6 Final Overleaf push.

---

## Daily Agent Instructions

1. Read this file and `NEWPLAN.md` to determine current state
2. Work on the **lowest-numbered incomplete task** in the **lowest-numbered incomplete phase**
3. Update checkboxes as tasks complete
4. All analysis code goes in `src/v2/` (new scripts for v2 data)
5. All data outputs go in `data_parquet/v2/`
6. All figures go in `figures/v2/` (PDF + PNG, 300 DPI)
7. Paper files go in `src/paper/` (synced with Overleaf)
8. At session end, write a report to `reports/YYYY-MM-DD.md`
