# Paper Plan: Behavioral Responses to E-Scooter Speed Governance

**Target venue:** Analytic Methods in Accident Research (AMAR)
**Template:** Elsevier CAS single-column (`cas-sc.cls`)
**Target length:** ~8,000-10,000 words (AMAR: methods-heavy papers welcome)
**Working directory:** `src/paper/` (synced to Overleaf via git subtree)
**Design doc:** `docs/superpowers/specs/2026-03-09-amar-paper-design.md`

---

## Core Insight

A speed governor is a **single-dimensional intervention** (caps max speed) that produces **multi-dimensional behavioral responses**. "Does a speed governor reduce speeding?" is tautological. The paper asks: what happens on the *unconstrained* behavioral margins?

```
Single policy lever          Multi-dimensional response
─────────────────           ──────────────────────────
                        ┌→  Safety: harsh events, speed variability
Speed governor ────────┼→  Demand: trip frequency, distance, retention
                        └→  Dynamics: cruising, acceleration patterns
```

## Title (working — finalize after results)

Candidates (select based on which story the data supports best):
- A: **"Behavioral Fingerprint of Speed Governance: Multi-Dimensional Evidence from 21 Million E-Scooter Trips"** (methods-forward)
- B: **"The Behavioral Cost of Speed Governance: A Multi-Outcome Natural Experiment"** (policy-forward)
- C: **"One Lever, Many Margins: How Speed Governance Reshapes E-Scooter Riding Behavior"** (Peltzman/behavioral economics)

**Decision rule:** Run all analyses first, then pick whichever narrative the data supports best.

## Outcome Variables

### Primary Outcomes (genuinely behavioral — the paper's core)

| Category | Outcome | Metric | Why non-tautological |
|----------|---------|--------|---------------------|
| **Safety** | Harsh acceleration | Count per trip | Throttle aggressiveness independent of speed cap |
| **Safety** | Harsh deceleration | Count per trip | Braking aggressiveness independent of speed cap |
| **Safety** | Speed variability | CV of within-trip speeds | Riding smoothness is a choice within any range |
| **Demand** | Trip frequency | Trips per user per month | Governor doesn't force ridership changes |
| **Demand** | Trip distance | km per trip | Route/destination choice is behavioral |
| **Demand** | User retention | Active months / churn | Platform abandonment is behavioral |
| **Dynamics** | Cruising fraction | % time at steady speed | Riding pattern is a choice |
| **Dynamics** | Zero-speed fraction | % time stopped | Stop-and-go is behavioral |

### Manipulation Check (mechanical — report briefly, not a "finding")

- Mean speed, max speed, P85 speed
- Purpose: confirm the governor works, then move to what's interesting

### Dropped (mechanical artifacts)

- Speed std (range shrinks mechanically with governor)
- Skewness, kurtosis (truncation artifact, not behavioral)

## Data Plan

| Analysis | Dataset | Period | N | Notes |
|----------|---------|--------|---|-------|
| Cross-sectional (Blocks 1, 4) | trip_modeling.parquet | Feb-Nov 2023 | ~21M trips | All 8 primary outcomes |
| DiD causal (Block 2) | city_month_panel_v2.parquet | Feb-Dec 2023 | 53 x 12 ~636 obs | Multi-outcome TWFE |
| Compensation (Block 3) | User-level pre/post | Nov-Dec 2023 | ~240K users | All 8 outcomes |
| Survival (Block 4) | User-trip sequences | Feb-Nov 2023 | ~864K users | Multiple endpoints |

**Critical rule:** All cross-sectional analyses use the SAME dataset (Feb-Nov 2023, ~21M trips with speed data). Only DiD/compensation use Dec 2023.

## Analysis Blocks

### Block 1: Cross-Sectional Behavioral Profiles

- 1a. Multi-dimensional mode comparison: TUB vs STD vs ECO on all 8 primary outcomes (Table + visualization)
- 1b. GEE per outcome: `outcome ~ mode + age_group + time_of_day + day_type + city + frac_major_road + log(distance)`, clustered by user
- 1c. Within-subject paired comparison (12K multi-mode users): paired tests + Cohen's d on each outcome
- 1d. LightGBM + SHAP: separate models per outcome category, compare feature importance rankings
- **Question answered:** How do behavioral profiles differ across modes, beyond speed?

### Block 2: Causal Multi-Outcome DiD

- 2a. TWFE on each primary outcome + Bonferroni correction (city-month panel, Dec 2023 TUB ban)
- 2b. Trip count per city-month as DiD outcome (demand response — NEW)
- 2c. Active user count per city-month as DiD outcome (retention — NEW)
- 2d. Event study for each outcome
- 2e. Dose-response for each outcome
- 2f. Placebo test (Sep 2023) for each outcome
- **Question answered:** Which behavioral dimensions causally respond to removing the ungoverned mode?

### Block 3: Compensation Test

- 3a. Multi-outcome DiD restricted to STD/ECO trips only
- 3b. Mode-switcher within-user DiD on all 8 outcomes (120K switchers vs 120K never-TUB)
- 3c. Cohen's d effect sizes across all dimensions
- 3d. Placebo validation (Oct 2023) on all outcomes
- 3e. Summary: which margins show compensation (if any) and which don't?
- **Question answered:** Do riders compensate on ANY unconstrained margin?

### Block 4: Behavioral Escalation Pathway

- 4a. Experience -> TUB adoption -> outcome changes (mediation per outcome)
- 4b. Survival with multiple endpoints:
  - Time-to-first-harsh-event
  - Time-to-first-high-CV-trip
  - Time-to-first-speeding (secondary/legacy)
  - Time-to-platform-churn (demand — NEW)
- 4c. Cox PH with time-varying TUB covariate for each endpoint
- 4d. TUB adoption mediation percentage for each outcome
- **Question answered:** Does ungoverned mode availability create a behavioral escalation pipeline?

## Paper Structure

### 1. Introduction (~800 words)
- E-scooter safety crisis + speed as primary risk factor
- "Do governors reduce speeding?" is tautological
- Right question: does governance reshape behavior on unconstrained margins? (safety, demand, dynamics)
- Gap: no multi-margin evidence, no demand-side evidence, no causal identification beyond speed
- Contribution: first multi-outcome causal evaluation across safety, demand, and dynamics

### 2. Background (~1,000 words)
- 2.1 E-scooter injuries and speed as risk factor
- 2.2 Speed governance: regulation vs technology
- 2.3 Peltzman / risk homeostasis — theoretical expectation of compensation
- 2.4 Demand elasticity of safety regulation (seatbelt -> more driving, helmet laws -> less cycling)
- 2.5 Natural experiments in safety evaluation
- 2.6 Research gaps

### 3. Data & Setting (~800 words)
- 3.1 Swing platform: TUB/STD/ECO on identical hardware
- 3.2 Dataset: 21M trips, 52 cities, per-point speed profiles
- 3.3 Dec 2023 TUB ban (natural experiment)
- Table 1: Descriptive stats organized by safety x demand x dynamics

### 4. Methods (~1,500 words)
- 4.1 Outcome definitions (8 primary + manipulation check)
- 4.2 Cross-sectional: GEE per outcome
- 4.3 Feature importance: LightGBM + SHAP
- 4.4 Multi-outcome DiD: TWFE + Bonferroni
- 4.5 Compensation test: within-user mode-switcher
- 4.6 Behavioral escalation: experience mediation + multi-endpoint survival
- 4.7 Robustness: placebo, heterogeneity

### 5. Results (~2,500 words)
- 5.1 Manipulation check (speed drops — brief)
- 5.2 Safety margins (harsh events, speed CV)
- 5.3 Demand margins (trip frequency, distance, retention)
- 5.4 Riding dynamics (cruising, zero-speed)
- 5.5 Causal effects (DiD across all margins)
- 5.6 Compensation test
- 5.7 Escalation pathway

### 6. Discussion (~1,000 words)
- 6.1 Synthesis: which margins respond, which don't
- 6.2 Policy: mandate governors, graduated access, demand implications
- 6.3 Comparison with Peltzman literature
- 6.4 Limitations

### 7. Conclusions (~400 words)

## Figures (~10-12)

1. Multi-dimensional behavioral profiles by mode (radar or parallel coordinates)
2. GEE coefficient comparison across outcomes (multi-panel forest plot by category)
3. SHAP importance comparison across outcome categories
4. **KEY FIGURE**: Multi-outcome DiD coefficients (which margins respond causally)
5. Demand response: trip counts + active users before/after ban
6. Compensation test summary: Cohen's d across all margins
7. Experience -> TUB adoption -> outcome mediation diagram
8. Multi-endpoint KM survival curves
9. Dose-response across outcomes
10. Robustness panel (placebo, heterogeneity)

Appendix: event studies per outcome, Cox forest plots, within-subject paired details

## What's IN vs OUT

**IN** (multi-dimensional behavioral analysis on unconstrained margins):
- 8 genuinely behavioral outcomes across all 4 blocks
- Multi-outcome DiD with Bonferroni correction
- Demand-side outcomes (trip count, retention) — novel
- Compensation test on all unconstrained margins
- Multi-endpoint survival (including platform churn)
- SHAP across outcome categories

**OUT** (save for separate papers):
- GMM rider typology
- Curvature risk paradox
- Detailed road class regressions
- Two-part hurdle model
- Multinomial logit for cluster membership
- Spatial hotspot analysis (mention briefly only)

## Key Technical Decisions

1. **Unified dataset**: ONE modeling parquet from Feb-Nov 2023 (~21M trips) with all outcomes
2. **GEE per outcome**: Separate GEE (not multivariate GEE — simpler, interpretable)
3. **SHAP per category**: Separate LightGBM models, compare feature rankings across categories
4. **Multi-outcome DiD**: Separate TWFE per outcome + Bonferroni correction
5. **Demand in DiD**: City-level trip counts and active user counts as outcomes (novel)
6. **Multiple survival endpoints**: Harsh event, high-CV, speeding, churn
7. **Manipulation check**: Speed outcomes reported but NOT primary findings
8. **Narrative selection**: Run all blocks first, then select framing (A/B/C) based on results
