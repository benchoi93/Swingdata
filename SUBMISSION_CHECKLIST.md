# Submission File Inventory — TR Part C

**Manuscript:** Characterizing E-Scooter Riding Safety Through City-Scale Speed Profile Analysis
**Target:** Transportation Research Part C: Emerging Technologies
**Date prepared:** 2026-02-25 (final compilation and cross-reference verification)

## Manuscript Files

| # | File | Description | Status |
|---|------|-------------|--------|
| 1 | `src/paper/main.tex` | Main manuscript (includes all sections via \input) | Ready |
| 2 | `src/paper/introduction.tex` | Section 1: Introduction (~2100 words, 8 contributions) | **Updated 2026-02-24** |
| 3 | `src/paper/literature_review.tex` | Section 2: Literature Review (~2700 words) | Ready |
| 4 | `src/paper/data_description.tex` | Section 3: Data Description (~2600 words, Tables 1-2, 24-mo data) | **Updated 2026-02-24** |
| 5 | `src/paper/methodology.tex` | Section 4: Methodology (~4200 words, Tables 3-5, Models 6-8) | **Updated 2026-02-24** |
| 6 | `src/paper/results.tex` | Section 5: Results (~5500 words, 4 new subsections) | **Updated 2026-02-24** |
| 7 | `src/paper/discussion.tex` | Section 6: Discussion (~2900 words, Phase 8 interpretations) | **Updated 2026-02-24** |
| 8 | `src/paper/conclusions.tex` | Section 7: Conclusions (~1000 words, 6 dimensions, 4 policies) | **Updated 2026-02-24** |
| 9 | `src/paper/appendix.tex` | Supplementary Figures (27 figures: 10 original + 17 Phase 8) | **Updated 2026-02-24** |
| 10 | `src/paper/references.bib` | Bibliography (49 entries, +SHAP) | **Updated 2026-02-24** |
| 11 | `paper_compiled.pdf` | Compiled manuscript (89 pages) | **Compiled 2026-02-25** |

## Supplementary Submission Files

| # | File | Description | Status |
|---|------|-------------|--------|
| 12 | `src/paper/cover_letter.tex` | Cover letter to Editor-in-Chief | Ready |
| 13 | `src/paper/highlights.tex` | 5 highlights (max 85 chars each) | Ready |
| 14 | `figures/graphical_abstract.pdf` | Graphical abstract | Ready |

## Main Figures (12)

| Fig | File | Description |
|-----|------|-------------|
| 1 | `figures/fig1_spatial_distribution.pdf` | Trip origin bubble map |
| 2 | `figures/fig2_temporal_distributions.pdf` | Hour-of-day and day-of-week patterns |
| 3 | `figures/fig3_speed_profile_example.pdf` | Annotated within-trip speed profile |
| 4 | `figures/fig4_speed_by_mode.pdf` | Speed distributions by operating mode |
| 5 | `figures/fig5_speeding_heatmap.pdf` | 24h x 7-day speeding prevalence heatmap |
| 6 | `figures/fig6_seoul_hotspot.pdf` | Seoul spatial hotspot analysis (4-panel) |
| 7 | `figures/fig7_daejeon_hotspot.pdf` | Daejeon spatial hotspot analysis |
| 8 | `figures/fig8_speed_by_road_class.pdf` | Speed safety by road class (4-panel) |
| 9 | `figures/fig9_logistic_OR.pdf` | Logistic regression odds ratio forest plot |
| 10 | `figures/fig10_gee_OR.pdf` | GEE regression odds ratio forest plot |
| 11 | `figures/fig11_rider_typology_radar.pdf` | GMM rider typology radar plots |
| 12 | `figures/fig12_tub_vs_eco.pdf` | TUB vs ECO mode comparison (4-panel) |

## Supplementary Figures (10)

| Fig | File | Description |
|-----|------|-------------|
| A.1 | `figures/fig_mixed_effects_coefficients.pdf` | Mixed-effects model coefficients |
| A.2 | `figures/fig_ols_mean_speed_coefficients.pdf` | OLS regression coefficients |
| A.3 | `figures/multinomial_coefficients.pdf` | Multinomial logit coefficients |
| A.4 | `figures/fig_morans_comparison.pdf` | Moran's I comparison across 5 cities |
| A.5 | `figures/fig_hotspot_infrastructure.pdf` | Infrastructure profiles of hot/cold spots |
| A.6 | `figures/fig_corridor_speeding.pdf` | Seoul corridor-level speeding maps |
| A.7 | `figures/map_matching_evaluation.pdf` | Map-matching validation |
| B.1 | `figures/robustness_threshold.pdf` | Threshold sensitivity (20/25/30 km/h) |
| B.2 | `figures/robustness_subsampling.pdf` | Subsampling stability |
| B.3 | `figures/robustness_city_models.pdf` | City-specific model comparison |

## Extension Analysis Figures (17) — Added 2026-02-24

| Fig | File | Description |
|-----|------|-------------|
| C.1 | `figures/fig_experience_learning_curve.pdf` | Speeding learning curves by mode |
| C.2 | `figures/fig_newcomer_speed_trajectory.pdf` | First 50 trips speed trajectory |
| C.3 | `figures/fig_newcomer_mode_adoption.pdf` | Mode share evolution over first 50 trips |
| C.4 | `figures/fig_newcomer_vs_established.pdf` | Newcomer vs established by mode |
| C.5 | `figures/fig_experience_usage_category.pdf` | Speeding by usage category |
| C.6 | `figures/fig_distance_speeding_curve.pdf` | LOWESS distance-speeding curve |
| C.7 | `figures/fig_distance_spline_predicted.pdf` | Spline predicted probabilities |
| C.8 | `figures/fig_road_composition_or.pdf` | Road class composition OR forest plot |
| C.9 | `figures/fig_distance_roadclass_heatmap.pdf` | Distance x road class heatmap |
| C.10 | `figures/fig_speed_drift.pdf` | Within-trip speed drift |
| C.11 | `figures/fig_curvature_speeding_rate.pdf` | Speeding by curvature class |
| C.12 | `figures/fig_curvature_continuous.pdf` | LOWESS curvature-speeding |
| C.13 | `figures/fig_curvature_risk_scatter.pdf` | Curvature risk paradox scatter |
| C.14 | `figures/fig_shap_bar.pdf` | SHAP mean importance |
| C.15 | `figures/fig_shap_summary.pdf` | SHAP beeswarm plot |
| C.16 | `figures/fig_shap_dependence.pdf` | SHAP dependence plots (top 6) |
| C.17 | `figures/fig_curvature_roadclass.pdf` | Curvature x road class heatmap |

## Analysis Code (Supplementary)

| File | Description |
|------|-------------|
| `src/CODE_README.md` | Documentation of analysis pipeline |
| `src/config.py` | Project configuration |
| `src/*.py` (27 scripts) | Complete analysis pipeline |

## Pre-Submission Checklist

- [x] All numerical claims verified against data (26/26 checks pass)
- [x] Mode distribution corrected in Data Description (2026-02-20)
- [x] Provincial speeding ranking corrected in Results (Gyeongnam/Daejeon -> Gyeongbuk/Gangwon) (2026-02-20)
- [x] STD speeding rate corrected (2026-02-19)
- [x] Abstract trimmed to ~290 words (updated 2026-02-25 with Phase 8 content)
- [x] All figure files present (40 PDF + 40 PNG = 80 files, including 17 Phase 8)
- [x] Cross-references verified (zero LaTeX errors)
- [x] Line numbering enabled (review format)
- [x] Bibliography: 49 entries, elsarticle-harv style (updated 2026-02-24)
- [x] Highlights: 5 items, all under 85 characters (updated 2026-02-25 with Phase 8 content)
- [x] Cover letter addressed to TR-C Editor-in-Chief
- [x] Graphical abstract created
- [x] Paper recompiled after 2026-02-21 corrections (71 pages, zero undefined refs/citations)
- [x] Appendix cross-references added (6 unreferenced appendix figures now cited in main text) (2026-02-21)
- [x] 'none' mode footnote added in Data Description (2026-02-21)
- [x] GMM rider typology corrected (class counts, percentages, feature values) (2026-02-22)
- [x] Mode comparison statistics corrected (speed CV, speeding rate, Cohen's d) (2026-02-22)
- [x] Multinomial RRR for Moderate-Risk corrected (8.6 -> 1.76) (2026-02-22)
- [x] "22-fold" reduction updated to "20-fold" across all sections (2026-02-22)
- [x] Phase 8 findings integrated into all paper sections (2026-02-24)
- [x] Abstract updated with longitudinal data, SHAP, experience, curvature (2026-02-24)
- [x] Introduction expanded from 6 to 8 contributions (2026-02-24)
- [x] 17 new appendix figures added (2026-02-24)
- [x] SHAP reference added to bibliography (49 entries) (2026-02-24)
- [x] Paper recompiled after Phase 8 integration (89 pages, 0 warnings, 0 undefined refs) (2026-02-25)
- [x] Final visual check of compiled PDF (all 38 figures, 10 tables render correctly) (2026-02-25)
- [x] Verify all figure cross-references resolve (added 8 missing \ref citations, all labels now referenced) (2026-02-25)
- [x] Phase 8 summary corrected per reviewer feedback (Trip 10 rate, curvature quintile/class distinction, limitations added) (2026-02-25)
- [x] highlights.tex updated to match main.tex highlights (2026-02-25)
- [x] Reference verification: 10 fabricated/incorrect .bib entries corrected against Crossref/Semantic Scholar (2026-02-26)
- [x] Curvature class figure caption corrected (mixed 31.6% highest, not straight) (2026-02-26)
- [x] Literature review citation contexts updated for replaced references (2026-02-26)
- [x] Paper recompiled after reference corrections (90 pages, 0 undefined refs) (2026-02-26)
- [x] Full verification of all 49 .bib entries: 11 additional errors corrected (3 wrong venues, 5 fabricated author names, 3 missing authors). 2 unverifiable entries flagged for PI. (2026-02-27)
- [x] Deceleration rates (3.8 vs 6.0 m/s^2) removed from text; Vetturi et al. paper suggests different findings. PI to verify. (2026-02-27)
- [x] 8 orphan PNG files removed from figures/ (intermediate outputs not in paper) (2026-02-27)
- [x] Paper recompiled after all corrections (91 pages, 0 undefined refs) (2026-02-27)
- [x] 3 previously unverifiable references replaced with verified alternatives (quddus2009, chen2016, chen2017) (2026-02-28)
- [x] BibTeX key renaming pass: 14 mismatched keys renamed across all .tex and .bib files (2026-02-28)
- [x] Citation contexts adjusted for 2 replaced references (chen2017route, chen2015bicycle) (2026-02-28)
- [x] Paper recompiled after key renaming (91 pages, 0 undefined refs/citations) (2026-02-28)
- [x] chen2017route year field corrected: year={2018} -> year={2017} to match key and Crossref DOI date (2026-03-01)
- [x] Methodology covariate descriptions updated to match results: 8 age groups, 5 time periods, 3 day types (2026-03-01)
- [x] Missing appendix figure C.17 (fig_curvature_roadclass) added to appendix.tex with \ref in results (2026-03-01)
- [x] 2 unreferenced figure labels (fig:spatial_distribution, fig:tub_eco) now cited in text (2026-03-01)
- [x] Paper recompiled: 92 pages, 0 undefined refs, 0 warnings, 0 errors (2026-03-01)
- [ ] Co-author list finalized (pending confirmation)
- [ ] CRediT author statement completed
- [ ] Upload to Elsevier Editorial Manager

## Data Corrections Log

| Date | Section | Issue | Fix |
|------|---------|-------|-----|
| 2026-02-19 | Data Description | STD speeding rate reported as 3.2% | Corrected to 0.9% |
| 2026-02-19 | Data Description | STD mean speed reported as 12.1 km/h | Corrected to 14.2 km/h |
| 2026-02-19 | Data Description | ECO mean speed reported as 10.4 km/h | Corrected to 11.0 km/h |
| 2026-02-20 | Data Description | Mode distribution: "TUB 42.3%, STD 35.1%, ECO 11.3%" | Corrected to actual values: TUB 46.0%, STD 33.6%, ECO 4.0%, etc. |
| 2026-02-20 | Results | Provincial ranking: "Gyeongnam (42.1%), Daejeon (41.3%)" | Corrected to Gyeongbuk (42.6%), Gangwon (41.7%) |
| 2026-02-21 | Multiple | 6 appendix figures not cross-referenced from main text | Added \ref{} citations in Results and Methodology |
| 2026-02-21 | Data Description | "none" mode (5.2%) not explained | Added footnote explaining S7 model / Ulsan origin |
| 2026-02-22 | Results | GMM Moderate-Risk: n=51,737 (26.0%), propensity=0.58, max_speed=26.0 | Corrected to n=51,491 (25.9%), propensity=0.30, max_speed=23.6 |
| 2026-02-22 | Results | GMM Habitual Speeder: n=51,977 (26.1%), propensity=0.98, mean_speed=17.3, max_speed=27.3 | Corrected to n=52,223 (26.3%), propensity=0.59, mean_speed=17.9, max_speed=25.1 |
| 2026-02-22 | Results | Multinomial TUB RRR for Moderate-Risk: 8.6 | Corrected to 1.76 |
| 2026-02-22 | Results/Table | Speed CV TUB=0.49, ECO=0.49, d=0.01, p=0.614 | Corrected to TUB=0.54, ECO=0.48, d=0.22, p<0.001 |
| 2026-02-22 | Results/Table | Speeding rate ECO=0.33%, d=0.96 | Corrected to ECO=0.38%, d=0.79 |
| 2026-02-22 | Multiple | "22-fold" speeding reduction | Corrected to "20-fold" (7.4/0.38=19.5) |
| 2026-02-22 | Multiple | "without altering variability structure" | Softened to "with only modest effect on variability structure" |
| 2026-02-22 | Discussion | Habitual Speeder "speeds on 98% of trips" | Corrected to "speeds on 59% of trips" |
| 2026-02-24 | All sections | Phase 8 findings not in paper | Integrated experience, distance, curvature, SHAP into all sections |
| 2026-02-24 | Introduction | 6 contributions | Expanded to 8 (added experience + SHAP/curvature) |
| 2026-02-24 | Abstract | No longitudinal/SHAP content | Updated with 44.1M trips, SHAP AUC=0.905, experience effect, curvature paradox |
| 2026-02-24 | Data Description | No 24-month data description | Added paragraph on longitudinal dataset |
| 2026-02-24 | Methodology | Models 1-5 only | Added Models 6-8 (GEE experience, spline distance, LightGBM+SHAP) + curvature |
| 2026-02-24 | Results | No Phase 8 results | Added 4 subsections with Table 5 (experience) and 17 figure references |
| 2026-02-24 | Discussion | No experience/curvature/SHAP discussion | Added 4 paragraphs + graduated mode access policy |
| 2026-02-24 | Conclusions | 4 dimensions, 3 policies | Expanded to 6 dimensions, 4 policies |
| 2026-02-24 | Appendix | 10 figures | Added 17 Phase 8 figures (now 27 total) |
| 2026-02-24 | References | 48 entries | Added Lundberg & Lee 2017 (SHAP), now 49 |
| 2026-02-25 | Results | 8 unreferenced appendix figures | Added \ref citations for fig:daejeon_hotspot, fig:gee_forest, fig:experience_curve, fig:curvature_rate, fig:usage_category, fig:distance_road_heatmap, fig:shap_beeswarm, fig:curvature_continuous, fig:distance_spline, fig:newcomer_established |
| 2026-02-25 | Abstract | ~310 words (over 300 limit) | Trimmed to ~290 words |
| 2026-02-25 | highlights.tex | Out of sync with main.tex | Updated to match main.tex Phase 8 highlights |
| 2026-02-25 | phase8_summary.md | Trip 10: 17.2% | Corrected to 16.4% per reviewer |
| 2026-02-25 | phase8_summary.md | "5 research questions" | Corrected to "4 research questions + preprocessing" |
| 2026-02-25 | phase8_summary.md | No limitations section | Added 5-item limitations paragraph |
| 2026-02-25 | phase8_summary.md | Curvature quintile/class conflation | Added explicit distinction between quintile gradient (41-17%) and class gradient (29-32%) |
| 2026-02-26 | references.bib | denver2024escooter: fabricated authors/journal | Corrected to Kahan et al. (2024) Clinical Orthopaedics and Related Research |
| 2026-02-26 | references.bib | che2020users: fabricated paper | Replaced with Ma et al. (2021) TR Part D municipal guidelines |
| 2026-02-26 | references.bib | allem2024escooter: fabricated paper | Replaced with Sexton et al. (2023) Transport Reviews safety review |
| 2026-02-26 | references.bib | campisi2023geofencing: wrong authors | Corrected to Caggiani et al. (2023) |
| 2026-02-26 | references.bib | li2025latent: wrong authors | Corrected to Jena et al. (2025) |
| 2026-02-26 | references.bib | dozza2020pedelec: wrong authors/DOI | Corrected to Twisk et al. (2021), DOI 10.1016/j.aap.2020.105940 |
| 2026-02-26 | references.bib | zarwi2023braking: wrong authors/venue | Corrected to Vetturi et al. (2023) TR Procedia |
| 2026-02-26 | references.bib | lee2021escooter: unverifiable paper | Replaced with Shah et al. (2021) J. Safety Research |
| 2026-02-26 | references.bib | cicchino2024speed: wrong year/pages | Corrected to 2023, vol 2678(8):171-183 |
| 2026-02-26 | references.bib | spota2024injury: wrong volume/pages | Corrected to vol 90(6):1702-1713 |
| 2026-02-26 | appendix.tex | Curvature class caption: "straight highest speeding" | Corrected: modest 29-32% range across classes, quintile gradient stronger |
| 2026-02-26 | literature_review.tex | lee2021escooter citation context | Updated to match Shah et al. crash typology paper |
| 2026-02-26 | literature_review.tex | brown2021geofencing citation context | Updated to match Neshagaran et al. observational study |
| 2026-02-27 | references.bib | trivedi2019injuries: wrong first names | Corrected "Anna Linda M"→"Anna Liza M", "Vince"→"Vanessa" |
| 2026-02-27 | references.bib | bloom2021standing: wrong first names | Corrected Areg→Ali, Chun→Carol, Margaret→Milton, Elias→Ernest |
| 2026-02-27 | references.bib | badeau2019emergency: ALL first names wrong | Corrected Amy→Austin, Casey→Chad, Mark→Michael, Jay→Jacob, Mark→Margaret |
| 2026-02-27 | references.bib | james2019pedestrians: wrong venue/metadata | Corrected TRR→Sustainability, vol 11(20):5591, DOI added |
| 2026-02-27 | references.bib | singh2022impact: wrong venue/metadata | Corrected Foot & Ankle Specialist→Bone & Joint Open, vol 3(9):674-683, DOI added |
| 2026-02-27 | references.bib | haworth2021comparing: missing author | Added Twisk, Divera as 3rd author, title corrected |
| 2026-02-27 | references.bib | tuncer2020notes: missing 2 authors | Added Laurier, Eric and Licoppe, Christian, DOI added |
| 2026-02-27 | references.bib | liu2020taxi: ALL authors fabricated | Corrected to Liu, Haiyue; Fu, Chuanyun; Jiang, Chaozhe; Zhou, Yue; Mao, Chengyuan; Zhang, Jining |
| 2026-02-27 | references.bib | strauss2015speed: wrong venue/year/metadata | Corrected TRR 2015→TR Part D 2017, vol 57:155-171, DOI added |
| 2026-02-27 | references.bib | degele2018identifying: ALL first names fabricated | Corrected 9 of 10 first names + Korber→Kormann |
| 2026-02-27 | references.bib | jin2022transformer: missing author | Added Kim, Jiwon; reordered to match Crossref |
| 2026-02-27 | literature_review.tex | zarwi2023braking deceleration values | Removed unverifiable 3.8 vs 6.0 m/s^2; generalized claim |
| 2026-02-27 | figures/ | 8 orphan PNGs | Removed gee_coefficients, gmm_bic, gmm_profiles (x3), mode_paired/speed, regression_coefficients |
| 2026-02-28 | references.bib | quddus2009dependence: unverifiable (not in Crossref) | Replaced with verified Quddus (2013) JTSS, DOI 10.1080/19439962.2012.705232 |
| 2026-02-28 | references.bib | chen2016hotspot: unverifiable (TRR vol 2587 pages mismatch) | Replaced with verified Chen (2015) Safety Science, DOI 10.1016/j.ssci.2015.06.016 |
| 2026-02-28 | references.bib | chen2017bicycle: fabricated title/venue | Replaced with verified Chen/Shen/Childress (2017) IJST, DOI 10.1080/15568318.2017.1349222 |
| 2026-02-28 | literature_review.tex | chen2017bicycle citation context | Adjusted to describe real paper (route preferences, not speed patterns) |
| 2026-02-28 | literature_review.tex | chen2016hotspot citation context | Adjusted to describe real paper (bicycle crash spatial statistics) |
| 2026-02-28 | All .tex + .bib | 14 mismatched BibTeX keys | Renamed: che2020users->ma2021municipal, allem2024escooter->sexton2023shared, cicchino2024speed->cicchino2023speed, campisi2023geofencing->caggiani2023geofencing, brown2021geofencing->neshagaran2024safety, dozza2020pedelec->twisk2021speed, zarwi2023braking->vetturi2023kinematic, strauss2015speed->strauss2017cycling, denver2024escooter->kahan2024escooter, lee2021escooter->shah2021crash, li2025latent->jena2025latent, chen2017bicycle->chen2017route, chen2016hotspot->chen2015bicycle, quddus2009dependence->quddus2013speed |
| 2026-02-28 | Paper | Recompiled | 91 pages, 0 undefined refs/citations, 0 LaTeX errors |
| 2026-03-01 | references.bib | chen2017route year inconsistency (key=2017, year field=2018) | Changed year={2018} to year={2017} to match DOI date |
| 2026-03-01 | methodology.tex | Covariate descriptions didn't match results table | Updated: 6->8 age groups, 4->5 time periods, 2->3 day types |
| 2026-03-01 | appendix.tex | C.17 fig_curvature_roadclass missing from appendix | Added figure entry with caption and label |
| 2026-03-01 | data_description.tex | fig:spatial_distribution never referenced | Added \ref{fig:spatial_distribution} in city assignment text |
| 2026-03-01 | results.tex | fig:tub_eco never referenced | Added \ref{fig:tub_eco} in speed governor results |
| 2026-03-01 | Paper | Recompiled | 92 pages, 0 undefined refs, 0 warnings, 0 errors |
