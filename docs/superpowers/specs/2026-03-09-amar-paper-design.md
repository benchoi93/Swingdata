# Design: AMAR Paper — Behavioral Responses to E-Scooter Speed Governance

**Date:** 2026-03-09
**Target venue:** Analytic Methods in Accident Research (AMAR)
**Target length:** ~8,000-10,000 words
**Status:** Approved by PI

---

## Core Insight

A speed governor is a **single-dimensional intervention** (caps max speed) that can produce **multi-dimensional behavioral responses**. Asking "does a speed governor reduce speeding?" is tautological — the device mechanically prevents it. The paper asks: what happens on the *unconstrained* behavioral margins?

```
Single policy lever          Multi-dimensional response
─────────────────           ──────────────────────────
                        ┌→  Safety: harsh events, speed variability
Speed governor ────────┼→  Demand: trip frequency, distance, retention
                        └→  Dynamics: cruising, acceleration patterns
```

## Candidate Narratives (decide after results)

- **A: "Multi-Outcome Policy Evaluation"** — Methods-forward. The framework IS the contribution. Transferable to seatbelts, helmets, AEB.
- **B: "Behavioral Cost of Speed Governance"** — Policy-forward. Are there hidden costs? Works regardless of result direction.
- **C: "One Lever, Many Margins"** — Peltzman/behavioral economics. Theory-driven multi-margin compensation test.

**Decision rule:** Run all analyses, then pick whichever narrative the data supports best.

## Outcome Variables

### Primary Outcomes (genuinely behavioral)

| Category | Outcome | Metric | Why non-tautological |
|----------|---------|--------|---------------------|
| Safety | Harsh acceleration | Count per trip | Throttle aggressiveness independent of speed cap |
| Safety | Harsh deceleration | Count per trip | Braking aggressiveness independent of speed cap |
| Safety | Speed variability | CV of within-trip speeds | Riding smoothness is a choice within any range |
| Demand | Trip frequency | Trips per user per month | Governor doesn't force ridership changes |
| Demand | Trip distance | km per trip | Route/destination choice is behavioral |
| Demand | User retention | Active months / churn | Platform abandonment is behavioral |
| Dynamics | Cruising fraction | % time at steady speed | Riding pattern is a choice |
| Dynamics | Zero-speed fraction | % time stopped | Stop-and-go is behavioral |

### Manipulation Check (mechanical — report briefly)

- Mean speed, max speed, P85 speed
- Purpose: confirm the governor works, then move to what's interesting

### Dropped (mechanical/artifact)

- Speed std (range shrinks mechanically)
- Skewness, kurtosis (truncation artifact)

## Analysis Blocks

### Block 1: Cross-Sectional Behavioral Profiles

- Multi-dimensional comparison: TUB vs STD vs ECO on all 8 primary outcomes
- GEE per outcome: `outcome ~ mode + age_group + time_of_day + day_type + city + frac_major_road + log(distance)`, clustered by user
- Within-subject paired comparison (12K multi-mode users): paired tests + Cohen's d
- LightGBM + SHAP: separate models per outcome category
- **Question:** How do behavioral profiles differ across modes, beyond speed?

### Block 2: Causal Multi-Outcome DiD

- City-month panel (Feb-Dec 2023), Dec 2023 TUB ban
- TWFE on each primary outcome + Bonferroni correction
- **NEW**: Trip count per city-month as DiD outcome (demand)
- **NEW**: Active user count per city-month (retention)
- Event study + dose-response + placebo for each
- **Question:** Which behavioral dimensions causally respond to removing the ungoverned mode?

### Block 3: Compensation Test

- Multi-outcome DiD restricted to STD/ECO trips only
- Mode-switcher within-user DiD on all outcomes (120K switchers vs 120K never-TUB)
- Cohen's d effect sizes across all dimensions
- Placebo validation (Oct 2023)
- **Question:** Do riders compensate on ANY unconstrained margin?

### Block 4: Behavioral Escalation Pathway

- Experience -> TUB adoption -> outcome changes (mediation per outcome)
- Survival with multiple endpoints:
  - Time-to-first-harsh-event
  - Time-to-first-high-CV-trip
  - Time-to-first-speeding (secondary/legacy)
  - **NEW**: Time-to-platform-churn (demand)
- Cox PH with time-varying TUB covariate
- **Question:** Does ungoverned mode availability create a behavioral escalation pipeline?

## Data Plan

| Analysis | Dataset | Period | N | Notes |
|----------|---------|--------|---|-------|
| Cross-sectional (Blocks 1, 4) | trip_modeling.parquet | Feb-Nov 2023 | ~21M trips | All 8 outcomes |
| DiD causal (Block 2) | city_month_panel_v2.parquet | Feb-Dec 2023 | 53 x 12 ~636 obs | Multi-outcome TWFE |
| Compensation (Block 3) | User-level pre/post | Nov-Dec 2023 | ~240K users | All 8 outcomes |
| Survival (Block 4) | User-trip sequences | Feb-Nov 2023 | ~864K users | Multiple endpoints |

## Paper Structure

### 1. Introduction (~800 words)
- E-scooter safety crisis
- "Do governors reduce speeding?" — tautological
- Right question: does governance reshape behavior on unconstrained margins?
- Gap + contribution

### 2. Background (~1,000 words)
- 2.1 E-scooter injuries and speed as risk factor
- 2.2 Speed governance: regulation vs technology
- 2.3 Peltzman / risk homeostasis theory
- 2.4 Demand elasticity of safety regulation (seatbelt/helmet parallels)
- 2.5 Natural experiments in safety evaluation
- 2.6 Gaps

### 3. Data & Setting (~800 words)
- Swing platform, TUB/STD/ECO
- 21M trips, 52 cities
- Dec 2023 TUB ban
- Table 1: descriptive stats by safety x demand x dynamics

### 4. Methods (~1,500 words)
- 4.1 Outcome definitions
- 4.2 GEE per outcome
- 4.3 LightGBM + SHAP
- 4.4 Multi-outcome DiD + Bonferroni
- 4.5 Compensation test: within-user mode-switcher
- 4.6 Escalation: mediation + multi-endpoint survival
- 4.7 Robustness

### 5. Results (~2,500 words)
- 5.1 Manipulation check (speed — brief)
- 5.2 Safety margins
- 5.3 Demand margins
- 5.4 Riding dynamics
- 5.5 Causal effects (DiD)
- 5.6 Compensation test
- 5.7 Escalation pathway

### 6. Discussion (~1,000 words)
- Synthesis, policy, Peltzman comparison, limitations

### 7. Conclusions (~400 words)

## Figures (~10-12)

1. Multi-dimensional profiles by mode (radar/parallel coords)
2. GEE coefficients across outcomes (forest plot by category)
3. SHAP importance comparison across categories
4. **KEY FIGURE**: Multi-outcome DiD coefficients (which margins respond)
5. Demand response: trip counts + active users before/after
6. Compensation test: Cohen's d across all margins
7. Experience -> TUB adoption -> outcome mediation
8. Multi-endpoint KM survival curves
9. Dose-response across outcomes
10. Robustness panel (placebo, heterogeneity)

Appendix: event studies, Cox forest, within-subject paired details

## What's IN vs OUT

**IN:**
- 8 genuinely behavioral outcomes across all 4 blocks
- Multi-outcome DiD with Bonferroni
- Demand-side outcomes (trip count, user retention) — novel
- Compensation test on all margins
- Multi-endpoint survival (including churn)
- SHAP across outcome categories

**OUT (separate papers):**
- GMM rider typology
- Curvature risk paradox
- Road class regressions
- Two-part hurdle model
- Spatial hotspot analysis
- Multinomial logit for cluster membership
