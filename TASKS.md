# Task List: E-Scooter Speed Governance Paper (AMAR)

**Status legend:** `[ ]` pending, `[~]` in progress, `[x]` done, `[!]` blocked

**Paper:** Behavioral Responses to E-Scooter Speed Governance (title TBD after results)
**Target:** Analytic Methods in Accident Research (AMAR)
**Plan:** See `NEWPLAN.md` for full paper design
**Design doc:** `docs/superpowers/specs/2026-03-09-amar-paper-design.md`

---

## Phase 0: Data Preparation (Priority: CRITICAL)

Build unified Feb-Nov 2023 dataset with all 8 genuinely behavioral outcomes.

- [ ] 0.1 Filter `routes_all.parquet` to Feb-Nov 2023 with `has_speed_data=True`. Output: `data_parquet/v2/trips_feb_nov.parquet`. Record row count (expect ~21M).
- [ ] 0.2 Compute trip-level indicators for 8 primary outcomes + manipulation check:
  - **Safety**: harsh_accel_count, harsh_decel_count, speed_cv
  - **Demand**: distance, duration (trip frequency computed at user-month level)
  - **Dynamics**: cruise_fraction, zero_speed_fraction
  - **Manipulation check**: mean_speed, max_speed, p85_speed
  - Also: binary speeding (>25 km/h), speeding_rate (for legacy/survival)
  Output: `data_parquet/v2/trip_indicators.parquet`.
- [ ] 0.3 Re-run road class assignment (DuckDB regex + KDTree) on Feb-Nov trips. Output: `data_parquet/v2/trip_road_classes.parquet`.
- [ ] 0.4 Merge trip indicators + road class + city assignment + user demographics (age from Scooter CSV). Create unified modeling dataset. Output: `data_parquet/v2/trip_modeling.parquet`.
- [ ] 0.5 Compute user-level aggregates: all 8 outcomes + trip frequency per month + active months. Output: `data_parquet/v2/user_modeling.parquet`.
- [ ] 0.6 Compute descriptive statistics for Table 1: outcomes organized by safety x demand x dynamics, by mode (TUB/STD/ECO). Save to `data_parquet/v2/descriptive_stats.json`.
- [ ] 0.7 Verify data quality. Write report to `reports/v2/data_verification.json`.

## Phase 1: Block 1 — Cross-Sectional Behavioral Profiles (Priority: HIGH)

All analyses use `data_parquet/v2/trip_modeling.parquet` (Feb-Nov 2023).

- [ ] 1.1 Multi-dimensional mode comparison: TUB vs STD vs ECO on all 8 primary outcomes. Table + visualization. Save radar/parallel coordinates figure (Fig 1).
- [ ] 1.2 GEE for EACH primary outcome: harsh_accel_count, harsh_decel_count, speed_cv, distance, cruise_fraction, zero_speed_fraction ~ mode + age_group + time_of_day + day_type + city + frac_major_road + log(distance), clustered by user. Also run manipulation check (mean_speed). Report coefficients. Save multi-panel forest plot (Fig 2).
- [ ] 1.3 Within-subject paired comparison (users who used multiple modes): paired tests on all 8 outcomes. Report Cohen's d for each.
- [ ] 1.4 LightGBM + SHAP: separate models for safety outcomes (harsh events, speed_cv), demand (distance), dynamics (cruise_fraction). Compare feature importance rankings across categories. Save SHAP comparison figure (Fig 3).
- [ ] 1.5 Distribution figures: violin/box by mode for each outcome category (Fig supplementary).

## Phase 2: Block 2 — Causal Multi-Outcome DiD (Priority: HIGH)

Uses Dec 2023 TUB ban as natural experiment.

- [ ] 2.1 Rebuild city-month panel with all 8 primary outcomes aggregated at city-month level + trip count + active user count. Output: `data_parquet/v2/city_month_panel_v2.parquet`.
- [ ] 2.2 TWFE DiD on EACH outcome: Y_ct ~ post * tub_share_nov + city_FE + month_FE. Include trip count and active user count as demand outcomes. Report all betas, SEs, p-values. Apply Bonferroni correction. Save multi-outcome DiD coefficient figure (Fig 4 — KEY FIGURE).
- [ ] 2.3 Event study for each outcome (monthly coefficients). Save multi-panel event study (Appendix).
- [ ] 2.4 Demand response figure: trip counts + active users before/after ban, by treatment intensity (Fig 5).
- [ ] 2.5 Dose-response for each outcome. Save multi-panel dose-response (Fig 9).
- [ ] 2.6 Placebo test (Sep 2023) for each outcome. Report all betas and p-values.

## Phase 3: Block 3 — Compensation Test (Priority: HIGH)

- [ ] 3.1 Multi-outcome DiD restricted to STD/ECO trips only, on all 8 outcomes.
- [ ] 3.2 Mode-switcher within-user DiD on ALL 8 outcomes. 120K switchers vs 120K never-TUB.
- [ ] 3.3 Cohen's d effect sizes across all outcomes. Save compensation summary figure (Fig 6).
- [ ] 3.4 Mode-switcher placebo (Oct 2023) on all outcomes.
- [ ] 3.5 Summary table: which margins show compensation and which don't.

## Phase 4: Block 4 — Behavioral Escalation Pathway (Priority: HIGH)

- [ ] 4.1 Experience -> mode adoption -> outcome changes: mediation analysis for each primary outcome. How much of the experience effect on each outcome is mediated by TUB adoption?
- [ ] 4.2 Experience trajectory visualization: multi-dimensional experience curves by outcome category (Fig 7).
- [ ] 4.3 Survival: time-to-first events for 4 endpoints:
  - Time-to-first-harsh-event (harsh_accel_count > 0 or harsh_decel_count > 0)
  - Time-to-first-high-CV-trip (speed_cv > 75th percentile)
  - Time-to-first-speeding (legacy, secondary)
  - Time-to-platform-churn (demand — NEW)
- [ ] 4.4 KM curves for each endpoint, stratified by TUB usage. Save multi-panel KM figure (Fig 8).
- [ ] 4.5 Cox PH with time-varying TUB covariate for each endpoint. Report HRs.
- [ ] 4.6 TUB mediation percentage for each outcome.

## Phase 5: Narrative Selection & Paper Writing (Priority: HIGH — after Phases 1-4)

- [ ] 5.0 **DECISION POINT**: Review all Block 1-4 results. Select narrative framing:
  - A: "Behavioral Fingerprint" — methods-forward (if multi-outcome framework is strongest)
  - B: "Behavioral Cost" — policy-forward (if demand response or null compensation is key)
  - C: "One Lever, Many Margins" — theory-forward (if Peltzman test is most compelling)
  Select title from NEWPLAN.md candidates.
- [ ] 5.1 Write Introduction (~800 words). Tautology framing, unconstrained margins.
- [ ] 5.2 Write Background (~1,000 words). Peltzman, demand elasticity, natural experiments.
- [ ] 5.3 Write Data and Setting (~800 words). Table 1 organized by safety x demand x dynamics.
- [ ] 5.4 Write Methods (~1,500 words). Multi-outcome framework, 8 outcomes, 4 blocks.
- [ ] 5.5 Write Results (~2,500 words). Manipulation check -> safety -> demand -> dynamics -> causal -> compensation -> escalation.
- [ ] 5.6 Write Discussion (~1,000 words). Synthesis, Peltzman comparison, policy.
- [ ] 5.7 Write Conclusions (~400 words).
- [ ] 5.8 Compile: pdflatex + bibtex. Fix all warnings. Target ~8,000-10,000 words.
- [ ] 5.9 Push to Overleaf.

## Phase 6: Quality Assurance (Priority: MEDIUM)

- [ ] 6.1 Verify all bib entries against academic databases.
- [ ] 6.2 Check all figure cross-references resolve.
- [ ] 6.3 Word count check (target 8,000-10,000 for AMAR).
- [ ] 6.4 Proofread for AMAR style: methods-forward, econometric rigor.
- [ ] 6.5 Final Overleaf push.

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
