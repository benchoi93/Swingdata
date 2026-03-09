# Review: reports/phase8_summary.md

**Reviewer**: Research Report Reviewer Agent
**Date**: 2026-02-25
**Report reviewed**: Phase 8 Extension Analyses Summary (dated 2026-02-24)

---

## Score: B

- **B: Minor issues, can proceed** -- The report is comprehensive, well-structured, and statistically rigorous overall. Most statistics validate against the underlying JSON outputs. Several minor discrepancies and presentational issues are noted below that should be corrected before paper submission, but none undermine the core findings.

---

## Issues Found

### 1. [MAJOR] Experience bin speeding rates in summary table do not match the JSON source

The Phase 8 summary (RQ1 table) reports the following experience bin speeding rates:
- 1st trip: **3.8%**
- 2-5 trips: **5.4%**
- 500+ trips: **28.0%**

The `experience_data_prep_report.json` records:
- 01_first: **3.76%** (rounds to 3.8% -- OK)
- 02_2to5: **5.38%** (rounds to 5.4% -- OK)
- 07_500plus: **27.99%** (rounds to 28.0% -- OK)

However, the `experience_results.json` `overall_learning_curve` uses different binning (rank bins of 1, 2-5, 6-10, 11-20, etc. vs. 1, 2-5, 6-20, 21-50, etc. in the data prep report). The learning curve in `experience_results.json` shows rank bin "1" with speeding rate of **2.5%** and "501-1K" with **28.06%**. The discrepancy between 3.8% (data prep) and 2.5% (experience results) for first trips arises because the data prep report includes ALL 44.1M trips (including those without speed data, where speeding is coded as absent), while the experience results script filters to users with 10+ trips. This is a methodological concern worth clarifying in the report -- the speeding rate for first trips depends heavily on the analysis subset. The report does not explicitly state which subset produces the experience bin table.

- **Suggested fix**: Add a footnote or parenthetical specifying which subset (full 44.1M vs. users with 10+ trips with speed data) generates each table. The 3.8% figure from the data prep report uses all trips including months without speed data, which dilutes the speeding rate. Consider reporting only trips with `has_speed_data=True`.

### 2. [MAJOR] Curvature-class speeding rates do NOT match the quintile-level statistics cited

The summary states: "Q0 (straightest): **41.0%** speeding; Q4 (curviest): **16.8%** speeding." These match the `curvature_speeding_results.json` quintile stats (Q0: 0.4103, Q4: 0.1682) -- **VERIFIED**.

However, the curvature class-level statistics in the same JSON tell a different story:
- straight: **29.1%** speeding
- mixed: **31.6%** speeding
- curvy: **29.3%** speeding

The class-level differences are far smaller than the quintile-level differences. This is because the "straight" class (frac_straight > 0.8) captures only 5.1% of trips and is not the same as the straightest curvature quintile. The summary report refers to "curvature class" results but primarily showcases quintile results, which could mislead readers into thinking straight/mixed/curvy classes show a 41%-to-17% gradient. The class-level chi-square is significant (p < 2.4e-24) but the actual class rates are nearly equal (~29-32%).

- **Suggested fix**: The report should explicitly distinguish between curvature quintile results (which show a strong gradient) and curvature class results (which show much smaller differences). The paper text should favor the continuous/quintile specification, and the class-level results should be discussed as supplementary.

### 3. [MINOR] Early trip trajectory: summary says "Trip 1: 7.0%" but data prep report says "1st trip: 3.8%"

The summary reports "Trip 1: 7.0% speeding" and "Trip 10: 17.2% speeding" under the early trip trajectory section. From `newcomer_results.json`, trip_rank=1 has speeding_rate=0.07004 (7.0%) and trip_rank=10 has 0.16351 (16.4%, not 17.2%). The 17.2% figure at trip 10 does not match -- the JSON shows 16.4%.

Meanwhile, the experience bin table above lists "1st trip: 3.8% speeding." This creates an apparent contradiction within the same RQ1 section. The 7.0% figure comes from the newcomer analysis (trips with speed data only, from 2023-02 onward), while the 3.8% comes from data prep (all 44.1M trips including no-speed-data months). Both are correct for their subsets, but the report does not clarify this.

- **Suggested fix**: (a) Correct "Trip 10: 17.2%" to "Trip 10: 16.4%". (b) Add explicit subset labels to distinguish the data prep (all trips) vs. newcomer analysis (speed-data trips only) figures.

### 4. [MINOR] Trip 50 speeding rate: summary says 26.0%, JSON says 26.04%

The summary states "Trip 50: 26.0% speeding." The JSON shows 0.2604, which rounds to 26.0%. This is acceptable but the preceding Trip 10 value (claimed 17.2% vs. actual 16.4%) suggests the summary may have been written from an earlier run.

- **Suggested fix**: Verify all trip trajectory values were drawn from the final `newcomer_results.json`.

### 5. [MINOR] Road composition OR for frac_major_road reported as [4.49, 10.12] but JSON shows [4.49, 10.12]

The summary reports frac_major_road OR = **6.74** [4.49, 10.12]. The JSON shows OR = 6.738, CI lower = 4.489, CI upper = 10.115. This matches within rounding -- **VERIFIED**.

### 6. [MINOR] SHAP F1 score reported as 0.739, JSON shows 0.7385

The summary says F1 = 0.739, the JSON says 0.7385. Acceptable rounding. **VERIFIED**.

### 7. [MINOR] Curvature OR reported as 0.716, JSON shows 0.7156

The summary says curvature OR = **0.716** [0.699, 0.732]. The JSON shows OR = 0.7156, CI = [0.6994, 0.7322]. Match within rounding. **VERIFIED**.

### 8. [MINOR] Risk score "5x higher" claim

The summary states "5x higher risk score on curvy roads when speeding." The JSON shows: curvy mean risk = 1.18, straight mean risk = 0.233. Ratio: 1.18/0.233 = **5.06x**. **VERIFIED**.

### 9. [MINOR] Missing section: RQ5 not addressed

The summary overview states "Phase 8 extends the original May 2023 analysis with five additional research questions." The five RQs listed are RQ1 (experience), RQ2 (distance/road), RQ3 (curvature), RQ4 (SHAP). There is no explicit RQ5 section. TASKS.md Phase 8 has 5 sub-phases (8.0-8.4), and the summary header says "5 research questions addressed (5/5 COMPLETE)" -- but only 4 RQ sections are written. It appears Phase 8.0 (preprocessing) is counted as a research question, but it is infrastructure, not a research question per se.

- **Suggested fix**: Either re-number to "4 research questions" or add an explicit RQ5 section (e.g., the GEE regression results that appear at the bottom could be elevated to a standalone RQ on experience-mode interaction).

### 10. [MINOR] Report has no Limitations subsection

Scientific reports should acknowledge limitations. Phase 8 analyses have several: (a) curvature based on GPS turning angles at 10s intervals, not actual road geometry; (b) 200K subsample for curvature vs. full dataset; (c) SHAP computed on 50K subsample; (d) speed data missing for 2022 + 2023-01; (e) experience analysis confounds time effects with experience effects (riders who started early have more trips AND rode during potentially different conditions).

- **Suggested fix**: Add a brief "Limitations" paragraph at the end of the summary.

---

## Verified

- [x] All 17 figure files exist in both PDF and PNG format in `figures/`
- [x] All referenced JSON output files exist in `data_parquet/modeling/`
- [x] TASKS.md Phase 8 tasks all marked `[x]` complete (8.0.1 through 8.5.2)
- [x] GEE results: log(trip_rank) OR=1.178 [1.156, 1.200] matches JSON (OR=1.1782, CI=[1.1564, 1.2004])
- [x] GEE TUB OR=83.6 matches JSON (OR=83.7088, rounds to 83.7 -- close, report says 83.6)
- [x] GEE quadratic p=0.44 matches JSON (p=0.4398)
- [x] Newcomer vs. established: 15.6% vs. 26.7%, Cohen's d=0.256 -- all match JSON exactly
- [x] Mode adoption: STD 69.5%->42.5%, TUB 11.5%->47.8%, ECO 11.1%->4.5% -- all match JSON
- [x] Age x experience interaction: <20 newcomer 20.1%, established 34.0% (+13.9pp) -- match JSON
- [x] Distance deciles: 13.9% at 149m, 54.2% at 2168m -- match JSON exactly
- [x] Spline pseudo-R2=0.387 matches JSON (0.3869, rounds to 0.387)
- [x] Spline AIC (375,384) vs. piecewise AIC (375,441) -- match JSON
- [x] Road composition ORs: service 2.74, tertiary 2.05, cycleway 1.66, residential 1.58, footway 1.19, secondary 0.20, primary 0.13 -- all match JSON CIs
- [x] Speed drift: overall mean -1.5 km/h, short trips -2.4 km/h -- match JSON (-1.500, -2.431)
- [x] Curvature classification: curvy 49.9%, mixed 45.0%, straight 5.1% -- match JSON (0.499, 0.450, 0.051)
- [x] LightGBM AUC-ROC=0.905, accuracy=0.800 -- match JSON (0.9054, 0.8003)
- [x] SHAP TUB mode: 2.331 mean |SHAP|, rank 1 -- match JSON exactly
- [x] SHAP log_distance: 0.639, rank 2 -- match JSON (0.6393)
- [x] SHAP top 5 ordering: TUB, log_distance, mean_accel, cruise_fraction, speed_CV -- match JSON ranks 1-5
- [x] Mode-specific newcomer effects: STD 1.0% vs 0.8%, TUB 49.6% vs 50.7%, ECO 0.8% vs 0.9% -- match JSON
- [x] 25 km/h speed limit threshold used correctly throughout
- [x] Sentinel value -999 handling documented for 2022 + 2023-01 months
- [x] Mode names normalized (SCOOTER_TUB -> TUB, etc.) documented
- [x] 44.1M total trips, 1,816,911 users -- match JSON
- [x] 198,303 trips with curvature out of 200K sample (99.2%) -- match JSON (198,303, coverage 0.9915)
- [x] Curvature subsample merged to 165,198 for regression -- JSON shows sample_size=165,001 (minor discrepancy: 165,198 vs 165,001; likely report used a different merge)
- [ ] Trip 10 speeding rate: summary says 17.2%, JSON shows 16.4% -- needs correction
- [ ] "5 research questions" claim: only 4 RQ sections are written
- [ ] No limitations section in the summary

---

## Recommendations

1. **Correct the Trip 10 speeding rate** from 17.2% to 16.4% (or re-verify against the final data).

2. **Clarify subset definitions** for the experience analysis. The 3.8% (first trip, all months) and 7.0% (first trip, speed-data months only) figures appear in the same RQ1 section without explanation. Readers will be confused by these seemingly contradictory numbers.

3. **Distinguish curvature quintiles from curvature classes** when presenting the curvature-speeding relationship. The quintile gradient (41% -> 17%) is much steeper than the class gradient (29% -> 29%). The current text risks overstating the class-level differences.

4. **Add a Limitations paragraph** addressing GPS-derived curvature precision, SHAP subsample size, missing 2022 speed data, and the experience-time confound.

5. **Resolve the "5 research questions" framing** -- either restructure to 4 RQs or add an explicit RQ5.

6. **Report the curvature regression sample size discrepancy**: the summary says 165,198 trips after merging, but the JSON shows 165,001. This is a 197-row difference that should be investigated (likely a different random seed or merge key mismatch).

7. **Consider reporting the GEE TUB OR more precisely**: the summary rounds to 83.6 but the JSON gives 83.71. Given that other Phase 5 models report TUB OR = 121-127, the discrepancy between 83.6 (GEE with experience control) and ~126 (logistic without experience) is scientifically meaningful and should be discussed. Controlling for experience reduces the TUB OR from ~126 to ~84, suggesting that part of the mode effect was confounded with experience.

---

## Suggested Research Directions

### 1. **Deep Learning on Within-Trip Speed Sequences for Rider Risk Classification** (Feasibility: HIGH)

- **Method**: Train a 1D-CNN or Transformer encoder on the raw per-trip speed vectors (variable-length sequences of 5-50+ readings per trip). Frame as binary classification (speeding trip vs. safe trip) or multi-class (the 4 GMM rider types). Use the existing `speeds` column, which contains the full speed time series. Compare against the logistic regression and LightGBM baselines already established (AUC = 0.905).
- **Data needed**: The `speeds` column from `routes_all.parquet` (already available for 25.1M trips with speed data). No new data collection required.
- **Expected finding**: A sequence model should capture temporal patterns within trips (e.g., gradual acceleration toward speeding, speed oscillations, end-of-trip surge) that point-estimate features like mean_speed or speed_CV cannot represent. Expected AUC improvement of 2-5% over LightGBM. More importantly, the learned representations could reveal distinct speed profile archetypes (e.g., "steady speeder" vs. "burst speeder") via clustering of the latent embeddings.
- **Target venue**: Transportation Research Part C or IEEE T-ITS. Aligns with the PI's expertise in deep generative models and trajectory analysis (TrajGAIL).
- **Connection to current work**: Builds directly on RQ4 (SHAP analysis) by moving from hand-crafted features to learned representations of the speed sequence. The existing SHAP analysis shows that speed dynamics features (cruise_fraction, speed_CV, mean_accel) collectively rival TUB mode in importance -- a sequence model could capture these dynamics more naturally.
- **Key references**: Yurdakul (2025) applied ML/DL to e-scooter usage prediction ([Wiley](https://onlinelibrary.wiley.com/doi/10.1155/atr/8794166)); the naturalistic e-scooter maneuver recognition study used federated contrastive learning on rider interactions ([ACM UbiComp](https://dl.acm.org/doi/10.1145/3570345)); Kim et al. (2022) developed a safety monitoring system for personal mobility using deep learning ([Oxford Academic](https://academic.oup.com/jcde/article/9/4/1397/6650219)).

### 2. **Causal Inference for Speed Governor Effectiveness Using Staggered Adoption as a Natural Experiment** (Feasibility: HIGH)

- **Method**: Exploit the 24-month longitudinal data to identify the staggered rollout of TUB and ECO modes across cities and time periods. Apply a staggered difference-in-differences (DiD) design (Callaway & Sant'Anna, 2021) or synthetic control method to estimate the causal effect of TUB mode introduction on city-level speeding rates. The key identification strategy: some cities received TUB mode earlier than others, and mode availability changed over the observation window. Alternatively, use the within-user paired analysis (n=12,046 users who used both STD and TUB) as an instrumental variable or regression discontinuity around the first TUB trip.
- **Data needed**: `routes_all.parquet` (44.1M trips with timestamps and modes), `user_longitudinal.parquet` (user first/last dates). Need to identify city-level mode rollout dates from the data (when TUB first appears in each city).
- **Expected finding**: The causal estimate of TUB mode on speeding should be somewhat lower than the observational OR (83-127) due to selection bias (risk-seeking riders select TUB). Expect causal effect in the range of OR = 30-60. The ECO mode causal effect (speed reduction) should be robust since it is less subject to selection bias.
- **Target venue**: Transportation Science or Transportation Research Part C. Causal inference in micromobility is a major gap in the literature.
- **Connection to current work**: The current analysis reports TUB OR = 83.6-126.6 across models, but these are associational. A causal estimate would dramatically strengthen the policy recommendation (speed governor mandates). The existing within-subject analysis (Task 5.8) is a step toward causality but does not account for trip-level selection.
- **Key references**: The geofencing-based speed regulation study by Campisi et al. (2023) evaluated safety aspects of speed limits but used simulation, not causal inference on observational data ([Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/15472450.2023.2201681)); Neshagaran et al. (2025) provided safety insights from behavioral observations but lacked longitudinal data for causal claims ([SAGE](https://journals.sagepub.com/doi/10.1177/03611981241279301)); the "Learning from the evidence" review by Orozco-Fontalvo et al. (2024) explicitly called for more rigorous causal evidence on e-scooter regulation effectiveness ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0967070X24000982)).

### 3. **Generative Speed Profile Synthesis Using Conditional Diffusion Models for Privacy-Preserving Data Sharing** (Feasibility: MEDIUM)

- **Method**: Train a conditional denoising diffusion probabilistic model (DDPM) on the per-trip speed sequences, conditioned on trip metadata (mode, distance, road class, city, time of day). The model would generate synthetic but realistic speed profiles that preserve the statistical properties of the original data (speed distributions, speeding rates, temporal dynamics) while ensuring individual privacy. Evaluate fidelity using: (a) distributional distance (Wasserstein, MMD) between real and synthetic speed profiles; (b) downstream utility (train speeding classifier on synthetic data, test on real data); (c) privacy (membership inference attack resistance).
- **Data needed**: `speeds` column from `routes_all.parquet` plus trip-level features. All data already available.
- **Expected finding**: A conditional diffusion model should generate speed profiles that maintain realistic dynamics (acceleration patterns, speed drift, mode-specific speed distributions) and preserve the key statistical relationships (TUB mode effect, distance-speeding curve, curvature-speed relationship). This would enable sharing of e-scooter speed data without privacy concerns, which is a major barrier for micromobility research.
- **Target venue**: NeurIPS or ICML (methodological contribution) or Transportation Research Part C (domain application). Directly leverages the PI's expertise in TrajGAIL and deep generative models for transportation (Choi et al., 2025 TR-C tutorial).
- **Connection to current work**: The comprehensive characterization of speed behavior in Phases 5-8 provides the ground truth for evaluating synthetic data quality. The SHAP analysis (RQ4) identifies which features must be faithfully reproduced. The 25.1M trips with speed data constitute a uniquely large training set for generative modeling of micromobility speed profiles.
- **Key references**: The PI's own tutorial on deep generative models in transportation (Choi et al., 2025, TR-C) provides the methodological foundation; GPS-based route choice analysis using city-scale data (Scientific Reports, 2025) demonstrates the value of large-scale micromobility datasets ([Nature](https://www.nature.com/articles/s41598-025-06938-2)); the multimodal sensing study on e-scooter conflicts (ACM UbiComp, 2025) highlights the need for larger datasets that generative models could provide ([ACM](https://dl.acm.org/doi/10.1145/3712284)).

---

## Summary Assessment

The Phase 8 summary report is thorough, well-organized, and covers all planned extension analyses. The vast majority of statistics have been verified against the underlying JSON outputs and are accurate. The two major issues (experience bin subset ambiguity and curvature class/quintile conflation) are presentational rather than analytical -- the underlying analyses are sound. The minor numerical discrepancy at Trip 10 (16.4% vs. 17.2%) and the "5 research questions" framing should be corrected.

The findings are scientifically compelling: the experience-speeding monotonic relationship mediated by TUB mode adoption, the curvature risk paradox, and the SHAP-confirmed dominance of speed governor mode all contribute meaningfully to the e-scooter safety literature. The three suggested research directions build directly on these findings and exploit the unique strengths of this dataset (scale, longitudinal depth, within-trip speed dynamics).

**Overall**: Proceed to paper integration with the corrections noted above. No re-analysis is required.
