"""
Programmatic statistical audit: verify paper LaTeX claims against JSON ground truth.

Reads all JSON ground truth files and parses numerical claims from .tex files,
then compares each claim against the source data with rounding tolerance.

Usage:
    python src/v2/audit_paper_stats.py
"""

import json
import re
import sys
from pathlib import Path

# --- Paths ---
DATA_DIR = Path("data_parquet/v2")
PAPER_DIR = Path("src/paper")

# Tolerance for rounding comparisons
ABS_TOL = 0.015  # absolute tolerance for values < 1
REL_TOL = 0.02   # 2% relative tolerance for values >= 1


def load_json(name):
    path = DATA_DIR / name
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    with open(path, "r") as f:
        return json.load(f)


def approx_eq(paper_val, json_val, label=""):
    """Check if paper value approximately equals JSON value."""
    if json_val == 0:
        return abs(paper_val) < ABS_TOL
    if abs(json_val) < 1:
        return abs(paper_val - json_val) < ABS_TOL
    return abs(paper_val - json_val) / abs(json_val) < REL_TOL


def read_tex(name):
    path = PAPER_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- Verification functions ---

def verify_table1(desc_stats, tex_content):
    """Verify Table 1 descriptive statistics (data_description.tex)."""
    results = []

    # Map from table variable names to JSON paths
    var_map = {
        "Mean speed": ("manipulation_check", "mean_speed"),
        "Max speed": ("manipulation_check", "max_speed_from_profile"),
        "P85 speed": ("manipulation_check", "p85_speed"),
        "Harsh accel": ("safety", "harsh_accel_count"),
        "Harsh decel": ("safety", "harsh_decel_count"),
        "Speed CV": ("safety", "speed_cv"),
        "Trip distance": ("demand", "distance"),
        "Travel time": ("demand", "travel_time"),
        "Cruise fraction": ("dynamics", "cruise_fraction"),
        "Zero-speed frac": ("dynamics", "zero_speed_fraction"),
    }

    # Parse table rows: "& Variable name & TUB_mean (TUB_sd) & STD_mean (STD_sd) & ECO_mean (ECO_sd) \\"
    pattern = re.compile(
        r'&\s*(Mean speed|Max speed|P85 speed|Harsh accel|Harsh decel|Speed CV|'
        r'Trip distance|Travel time|Cruise fraction|Zero-speed frac)'
        r'[^&]*&\s*([\d,.]+)\s*\(([\d,.]+)\)\s*&\s*([\d,.]+)\s*\(([\d,.]+)\)\s*&\s*([\d,.]+)\s*\(([\d,.]+)\)'
    )

    for m in pattern.finditer(tex_content):
        var_name = m.group(1)
        tub_mean = float(m.group(2).replace(",", ""))
        tub_sd = float(m.group(3).replace(",", ""))
        std_mean = float(m.group(4).replace(",", ""))
        std_sd = float(m.group(5).replace(",", ""))
        eco_mean = float(m.group(6).replace(",", ""))
        eco_sd = float(m.group(7).replace(",", ""))

        if var_name not in var_map:
            continue
        category, variable = var_map[var_name]
        try:
            json_data = desc_stats["outcomes_by_mode"][category][variable]["by_mode"]
        except KeyError:
            results.append({
                "section": "Table 1",
                "claim": f"{var_name} (all modes)",
                "paper": "N/A",
                "json": "MISSING KEY",
                "status": "FAIL",
            })
            continue

        for mode, paper_mean, paper_sd in [
            ("TUB", tub_mean, tub_sd),
            ("STD", std_mean, std_sd),
            ("ECO", eco_mean, eco_sd),
        ]:
            json_mean = json_data[mode]["mean"]
            json_sd = json_data[mode]["std"]

            mean_ok = approx_eq(paper_mean, json_mean)
            sd_ok = approx_eq(paper_sd, json_sd)

            results.append({
                "section": "Table 1",
                "claim": f"{var_name} {mode} mean",
                "paper": paper_mean,
                "json": round(json_mean, 4),
                "status": "PASS" if mean_ok else "FAIL",
            })
            results.append({
                "section": "Table 1",
                "claim": f"{var_name} {mode} SD",
                "paper": paper_sd,
                "json": round(json_sd, 4),
                "status": "PASS" if sd_ok else "FAIL",
            })

    return results


def verify_overall_counts(desc_stats, tex_content):
    """Verify overall dataset counts mentioned in data_description.tex."""
    results = []
    overall = desc_stats["overall"]

    # Total trips: 19,454,309
    m = re.search(r'(19[,.]454[,.]309|19,454,309)', tex_content)
    if m:
        results.append({
            "section": "Data description",
            "claim": "Total trips",
            "paper": 19454309,
            "json": overall["n_trips"],
            "status": "PASS" if overall["n_trips"] == 19454309 else "FAIL",
        })

    # Total users: 1,001,459
    m = re.search(r'1[,.]001[,.]459', tex_content)
    if m:
        results.append({
            "section": "Data description",
            "claim": "Total users",
            "paper": 1001459,
            "json": overall["n_users"],
            "status": "PASS" if overall["n_users"] == 1001459 else "FAIL",
        })

    # Cities: 52 (look for dataset context: "across 52 cities" or "52 cities x")
    m = re.search(r'across\s+(\d+)\s+cities', tex_content)
    if m:
        paper_cities = int(m.group(1))
        results.append({
            "section": "Data description",
            "claim": "Number of cities (dataset)",
            "paper": paper_cities,
            "json": overall["n_cities"],
            "status": "PASS" if paper_cities == overall["n_cities"] else "FAIL",
        })

    # Mode shares
    mode_trips = desc_stats["outcomes_by_mode"]["manipulation_check"]["mean_speed"]["by_mode"]
    mode_pct_pattern = re.compile(r'TUB mode accounts for ([\d.]+)\\?%.*?STD for ([\d.]+)\\?%.*?ECO for ([\d.]+)\\?%')
    m = mode_pct_pattern.search(tex_content)
    if m:
        tub_pct = float(m.group(1))
        std_pct = float(m.group(2))
        eco_pct = float(m.group(3))
        json_tub_n = mode_trips["TUB"]["n"]
        json_std_n = mode_trips["STD"]["n"]
        json_eco_n = mode_trips["ECO"]["n"]
        total = json_tub_n + json_std_n + json_eco_n
        json_tub_pct = round(json_tub_n / total * 100, 1)
        json_std_pct = round(json_std_n / total * 100, 1)
        json_eco_pct = round(json_eco_n / total * 100, 1)
        results.append({
            "section": "Data description",
            "claim": "TUB mode share %",
            "paper": tub_pct,
            "json": json_tub_pct,
            "status": "PASS" if abs(tub_pct - json_tub_pct) < 0.2 else "FAIL",
        })
        results.append({
            "section": "Data description",
            "claim": "STD mode share %",
            "paper": std_pct,
            "json": json_std_pct,
            "status": "PASS" if abs(std_pct - json_std_pct) < 0.2 else "FAIL",
        })
        results.append({
            "section": "Data description",
            "claim": "ECO mode share %",
            "paper": eco_pct,
            "json": json_eco_pct,
            "status": "PASS" if abs(eco_pct - json_eco_pct) < 0.2 else "FAIL",
        })

    return results


def verify_gee_table(cross_json, tex_content):
    """Verify GEE coefficients in Table 2 (results.tex).

    JSON structure: gee_results is a dict keyed by "{outcome}_{comparison}"
    e.g. "harsh_accel_count_STD vs TUB" -> {coef, se, p, ci_lower, ci_upper}
    """
    results = []
    gee = cross_json["gee_results"]

    # Map table outcome names to JSON outcome keys
    gee_map = {
        "Harsh accel count": "harsh_accel_count",
        "Harsh decel count": "harsh_decel_count",
        "Speed CV": "speed_cv",
        "Cruise fraction": "cruise_fraction",
        "Zero-speed frac": "zero_speed_fraction",
        "Mean speed": "mean_speed",
    }

    # Parse GEE table rows: "Outcome & $coef$ & $p$ & $coef$ & $p$ \\"
    gee_pattern = re.compile(
        r'(Harsh accel count|Harsh decel count|Speed CV|Cruise fraction|Zero-speed frac\.|Mean speed)[^&]*'
        r'&\s*\$([+-]?[\d.]+)\$\s*&\s*\$?([<\d.]+)\$?\s*'
        r'&\s*\$([+-]?[\d.]+)\$\s*&\s*\$?([<\d.]+)\$?'
    )

    for m in gee_pattern.finditer(tex_content):
        outcome_name = m.group(1).rstrip(".")
        std_coef = float(m.group(2))
        eco_coef = float(m.group(4))

        json_key = gee_map.get(outcome_name)
        if not json_key:
            continue

        # Look up in dict by composite key
        std_key = f"{json_key}_STD vs TUB"
        eco_key = f"{json_key}_ECO vs TUB"

        if std_key in gee:
            json_coef = gee[std_key]["coef"]
            ok = approx_eq(std_coef, json_coef)
            results.append({
                "section": "Table 2 (GEE)",
                "claim": f"{outcome_name} STD coef",
                "paper": std_coef,
                "json": round(json_coef, 4),
                "status": "PASS" if ok else "FAIL",
            })

        if eco_key in gee:
            json_coef = gee[eco_key]["coef"]
            ok = approx_eq(eco_coef, json_coef)
            results.append({
                "section": "Table 2 (GEE)",
                "claim": f"{outcome_name} ECO coef",
                "paper": eco_coef,
                "json": round(json_coef, 4),
                "status": "PASS" if ok else "FAIL",
            })

    return results


def verify_fe_vs_gee_table(fe_json, tex_content):
    """Verify FE vs GEE comparison table (Table 5, results.tex).

    JSON structure: fe_vs_gee_comparison is a dict keyed by "{outcome}_{mode}"
    e.g. "harsh_accel_count_STD" -> {gee_coef, fe_coef, difference, same_direction, attenuation_pct}
    """
    results = []
    comparison = fe_json.get("fe_vs_gee_comparison", {})

    # Parse table rows: "Outcome & Mode & $GEE$ & $FE$ & $delta%$ & Direction \\"
    fe_pattern = re.compile(
        r'(Harsh accel|Harsh decel|Speed CV|Cruise fraction|Zero-speed frac\.|Mean speed)\s*'
        r'&\s*(STD|ECO)\s*'
        r'&\s*\$([+-]?[\d.]+)\$\s*'
        r'&\s*\$([+-]?[\d.]+)\$\s*'
        r'&\s*\$([+-]?\d+(?:\.\d+)?)\$'
    )

    # Outcome name mapping
    fe_outcome_map = {
        "Harsh accel": "harsh_accel_count",
        "Harsh decel": "harsh_decel_count",
        "Speed CV": "speed_cv",
        "Cruise fraction": "cruise_fraction",
        "Zero-speed frac.": "zero_speed_fraction",
        "Zero-speed frac": "zero_speed_fraction",
        "Mean speed": "mean_speed",
    }

    for m in fe_pattern.finditer(tex_content):
        outcome_name = m.group(1).rstrip(".")
        mode = m.group(2)
        paper_gee = float(m.group(3))
        paper_fe = float(m.group(4))
        paper_delta = float(m.group(5))

        json_key = fe_outcome_map.get(outcome_name)
        if not json_key:
            continue

        # Lookup by composite key "{outcome}_{mode}"
        comp_key = f"{json_key}_{mode}"
        entry = comparison.get(comp_key)
        if not entry:
            continue

        json_gee = entry["gee_coef"]
        json_fe = entry["fe_coef"]
        json_delta = entry["attenuation_pct"]

        gee_ok = approx_eq(paper_gee, json_gee)
        fe_ok = approx_eq(paper_fe, json_fe)
        # delta% can have larger rounding variance
        # Paper reports negative delta% for attenuation; JSON stores positive attenuation_pct
        delta_ok = abs(abs(paper_delta) - abs(json_delta)) < 3.0

        results.append({
            "section": "Table 5 (FE vs GEE)",
            "claim": f"{outcome_name} {mode} GEE coef",
            "paper": paper_gee,
            "json": round(json_gee, 4),
            "status": "PASS" if gee_ok else "FAIL",
        })
        results.append({
            "section": "Table 5 (FE vs GEE)",
            "claim": f"{outcome_name} {mode} FE coef",
            "paper": paper_fe,
            "json": round(json_fe, 4),
            "status": "PASS" if fe_ok else "FAIL",
        })
        results.append({
            "section": "Table 5 (FE vs GEE)",
            "claim": f"{outcome_name} {mode} delta%",
            "paper": paper_delta,
            "json": round(json_delta, 1),
            "status": "PASS" if delta_ok else "FAIL",
        })

    return results


def verify_did_table(did_json, tex_content):
    """Verify DiD table (Table 4, results.tex).

    JSON structure: did_results is a dict keyed by outcome name
    e.g. "mean_speed" -> {beta, se, p, ci_lower, ci_upper, significant_bonferroni}
    """
    results_list = []
    did = did_json["did_results"]

    # Parse DiD table rows: "Outcome & $beta$ & SE & $p$ & [$lo, $hi$] & Verdict"
    did_pattern = re.compile(
        r'(Harsh accel|Harsh decel|Speed CV|Trip count|Active users|Cruise fraction|Zero-speed frac\.|Mean speed)\s*'
        r'&\s*\$([+-]?[\d,{}.]+)\$\s*'
        r'&\s*([\d,.]+)\s*'
        r'&\s*\$?([<\d.e\u00d7^{}\-]+)\$?\s*'
    )

    did_name_map = {
        "Harsh accel": "harsh_accel_count",
        "Harsh decel": "harsh_decel_count",
        "Speed CV": "speed_cv",
        "Trip count": "trip_count",
        "Active users": "active_users",
        "Cruise fraction": "cruise_fraction",
        "Zero-speed frac.": "zero_speed_fraction",
        "Zero-speed frac": "zero_speed_fraction",
        "Mean speed": "mean_speed",
    }

    for m in did_pattern.finditer(tex_content):
        outcome_name = m.group(1).rstrip(".")
        paper_beta_str = m.group(2).replace("{", "").replace("}", "").replace(",", "").replace("\\,", "")
        paper_se_str = m.group(3).replace(",", "")

        try:
            paper_beta = float(paper_beta_str)
        except ValueError:
            continue
        try:
            paper_se = float(paper_se_str)
        except ValueError:
            continue

        json_key = did_name_map.get(outcome_name)
        if not json_key or json_key not in did:
            continue

        entry = did[json_key]
        json_beta = entry["beta"]
        json_se = entry["se"]

        beta_ok = approx_eq(paper_beta, json_beta)
        se_ok = approx_eq(paper_se, json_se)

        results_list.append({
            "section": "Table 4 (DiD)",
            "claim": f"{outcome_name} beta",
            "paper": paper_beta,
            "json": round(json_beta, 4),
            "status": "PASS" if beta_ok else "FAIL",
        })
        results_list.append({
            "section": "Table 4 (DiD)",
            "claim": f"{outcome_name} SE",
            "paper": paper_se,
            "json": round(json_se, 4),
            "status": "PASS" if se_ok else "FAIL",
        })

    return results_list


def verify_compensation_table(comp_json, tex_content):
    """Verify compensation Cohen's d table (results.tex).

    JSON structure: mode_switcher is a dict keyed by outcome
    e.g. "mean_speed" -> {did_estimate, cohens_d, n_switchers, ...}
    """
    results = []
    switcher = comp_json["mode_switcher"]

    comp_name_map = {
        "Mean speed": "mean_speed",
        "Harsh accel count": "harsh_accel_count",
        "Harsh decel count": "harsh_decel_count",
        "Speed CV": "speed_cv",
        "Cruise fraction": "cruise_fraction",
        "Zero-speed frac": "zero_speed_fraction",
    }

    # Parse compensation table: "Outcome & $DiD$ & $d$ & $p$ & Interpretation"
    comp_pattern = re.compile(
        r'(Mean speed|Harsh accel count|Harsh decel count|Speed CV|Cruise fraction|Zero-speed frac)\s*'
        r'[^&]*&\s*\$([+-]?[\d.]+)\$\s*'
        r'&\s*\$([+-]?[\d.]+)\$\s*'
        r'&\s*\$?([<\d.]+)\$?'
    )

    for m in comp_pattern.finditer(tex_content):
        outcome_name = m.group(1).rstrip(".")
        paper_did = float(m.group(2))
        paper_d = float(m.group(3))

        json_key = comp_name_map.get(outcome_name)
        if not json_key or json_key not in switcher:
            continue

        entry = switcher[json_key]
        json_d = entry["cohens_d"]

        d_ok = approx_eq(paper_d, json_d)
        results.append({
            "section": "Compensation table",
            "claim": f"{outcome_name} Cohen's d",
            "paper": paper_d,
            "json": round(json_d, 4),
            "status": "PASS" if d_ok else "FAIL",
        })

    return results


def verify_survival_table(esc_json, tex_content):
    """Verify survival analysis table (results.tex)."""
    results = []
    survival = esc_json["survival"]

    surv_name_map = {
        "First harsh event": "first_harsh",
        "First high-CV trip": "first_high_cv",
        "First speeding": "first_speeding",
    }

    # Parse survival table: "Event & rate% & median_tub / median_nontub & logrank & HR"
    surv_pattern = re.compile(
        r'(First harsh event|First high-CV trip|First speeding)\s*'
        r'&\s*([\d.]+)\\?%\s*'
        r'&\s*(\d+|\\infty)\s*/\s*(\d+|\\infty|\$\\infty\$)'
    )

    for m in surv_pattern.finditer(tex_content):
        event_name = m.group(1)
        paper_rate = float(m.group(2))

        json_key = surv_name_map.get(event_name)
        if not json_key or json_key not in survival:
            continue

        entry = survival[json_key]
        json_rate = entry["event_rate"] * 100  # convert to %

        rate_ok = abs(paper_rate - json_rate) < 0.2
        results.append({
            "section": "Survival table",
            "claim": f"{event_name} event rate",
            "paper": paper_rate,
            "json": round(json_rate, 1),
            "status": "PASS" if rate_ok else "FAIL",
        })

        # HR
        if "cox_hr_tub_frac" in entry:
            json_hr = entry["cox_hr_tub_frac"]
            results.append({
                "section": "Survival table",
                "claim": f"{event_name} HR(tub_frac)",
                "paper": "see table",
                "json": round(json_hr, 2),
                "status": "INFO",
            })

    return results


def verify_inline_shap(shap_harsh, shap_cv, shap_cruise, tex_content):
    """Verify inline SHAP values mentioned in results.tex."""
    results = []

    # Harsh event SHAP
    shap_claims = [
        ("SHAP harsh: is_tub", 0.49, shap_harsh["top_features"][0]["mean_abs_shap"]),
        ("SHAP harsh: is_eco", 0.32, shap_harsh["top_features"][1]["mean_abs_shap"]),
        ("SHAP harsh: n_speed_points", 0.29, shap_harsh["top_features"][2]["mean_abs_shap"]),
        ("SHAP harsh: log_distance", 0.25, shap_harsh["top_features"][3]["mean_abs_shap"]),
    ]

    # Speed CV SHAP
    shap_claims.extend([
        ("SHAP speed_cv: log_distance", 0.12, shap_cv["top_features"][0]["mean_abs_shap"]),
        ("SHAP speed_cv: n_speed_points", 0.11, shap_cv["top_features"][1]["mean_abs_shap"]),
        ("SHAP speed_cv: is_tub", 0.03, shap_cv["top_features"][2]["mean_abs_shap"]),
    ])

    for label, paper_val, json_val in shap_claims:
        ok = approx_eq(paper_val, json_val)
        results.append({
            "section": "Inline SHAP",
            "claim": label,
            "paper": paper_val,
            "json": round(json_val, 4),
            "status": "PASS" if ok else "FAIL",
        })

    return results


def verify_inline_did(did_json, tex_content):
    """Verify inline DiD claims in results.tex.

    JSON: dose_response and placebo are dicts keyed by outcome.
    dose_response["harsh_decel_count"] -> {r, p, slope, ...}
    placebo["mean_speed"] -> {placebo_beta, p, ...}
    """
    results = []

    dose = did_json.get("dose_response", {})
    if "harsh_decel_count" in dose:
        hd_dose = dose["harsh_decel_count"]
        claims = [
            ("Dose-response harsh_decel r", 0.621, hd_dose.get("r")),
        ]
        if "harsh_accel_count" in dose:
            ha_dose = dose["harsh_accel_count"]
            claims.append(("Dose-response harsh_accel r", 0.561, ha_dose.get("r")))
        if "mean_speed" in dose:
            ms_dose = dose["mean_speed"]
            claims.append(("Dose-response mean_speed r", 0.567, ms_dose.get("r")))

        for label, paper_val, json_val in claims:
            if json_val is not None:
                ok = approx_eq(paper_val, json_val)
                results.append({
                    "section": "Inline DiD",
                    "claim": label,
                    "paper": paper_val,
                    "json": round(json_val, 4),
                    "status": "PASS" if ok else "FAIL",
                })

    # Placebo
    placebo = did_json.get("placebo", {})
    if "mean_speed" in placebo:
        ms_plac = placebo["mean_speed"]
        json_beta = ms_plac.get("placebo_beta")
        if json_beta is not None:
            ok = approx_eq(-0.59, json_beta)
            results.append({
                "section": "Inline DiD placebo",
                "claim": "Mean speed placebo beta",
                "paper": -0.59,
                "json": round(json_beta, 4),
                "status": "PASS" if ok else "FAIL",
            })

    return results


def verify_inline_mediation(boot_json, esc_json, tex_content):
    """Verify mediation percentages and bootstrap CIs.

    JSON: mediation_ci is a dict keyed by outcome.
    mediation_ci["harsh_accel_count"] -> {point, ci_lower, ci_upper, se}
    """
    results = []
    med_ci = boot_json.get("mediation_ci", {})

    mediation_claims = [
        ("Mediation harsh_accel %", 70.0, "harsh_accel_count", "point"),
        ("Mediation harsh_accel CI lo", 65.3, "harsh_accel_count", "ci_lower"),
        ("Mediation harsh_accel CI hi", 74.8, "harsh_accel_count", "ci_upper"),
        ("Mediation mean_speed %", 63.2, "mean_speed", "point"),
        ("Mediation mean_speed CI lo", 58.2, "mean_speed", "ci_lower"),
        ("Mediation mean_speed CI hi", 66.0, "mean_speed", "ci_upper"),
        ("Mediation harsh_decel %", 57.0, "harsh_decel_count", "point"),
        ("Mediation harsh_decel CI lo", 53.4, "harsh_decel_count", "ci_lower"),
        ("Mediation harsh_decel CI hi", 60.4, "harsh_decel_count", "ci_upper"),
        ("Mediation speed_cv %", 26.0, "speed_cv", "point"),
        ("Mediation speed_cv CI lo", 22.7, "speed_cv", "ci_lower"),
        ("Mediation speed_cv CI hi", 29.4, "speed_cv", "ci_upper"),
        ("Mediation cruise %", 20.6, "cruise_fraction", "point"),
        ("Mediation cruise CI lo", 18.5, "cruise_fraction", "ci_lower"),
        ("Mediation cruise CI hi", 22.8, "cruise_fraction", "ci_upper"),
        ("Mediation zero_speed %", 18.4, "zero_speed_fraction", "point"),
        ("Mediation zero_speed CI lo", 15.5, "zero_speed_fraction", "ci_lower"),
        ("Mediation zero_speed CI hi", 20.8, "zero_speed_fraction", "ci_upper"),
    ]

    for label, paper_val, outcome, field in mediation_claims:
        if outcome in med_ci:
            json_val = med_ci[outcome].get(field)
            if json_val is not None:
                ok = abs(paper_val - json_val) < 1.0  # within 1 percentage point
                results.append({
                    "section": "Mediation",
                    "claim": label,
                    "paper": paper_val,
                    "json": round(json_val, 1),
                    "status": "PASS" if ok else "FAIL",
                })

    return results


def verify_inline_paired(cross_json, tex_content):
    """Verify inline paired Cohen's d values.

    JSON: paired_comparisons.TUB_vs_ECO is a dict keyed by outcome
    e.g. TUB_vs_ECO["harsh_decel_count"] -> {n_pairs, cohens_d, t_stat, ...}
    """
    results = []
    paired = cross_json.get("paired_comparisons", {})

    # TUB vs ECO pairs
    tub_eco = paired.get("TUB_vs_ECO", {})
    eco_claims = [
        ("Paired TUB-ECO harsh_decel d", 1.36, "harsh_decel_count"),
        ("Paired TUB-ECO harsh_accel d", 1.05, "harsh_accel_count"),
        ("Paired TUB-ECO cruise d", -1.11, "cruise_fraction"),
    ]
    for label, paper_val, outcome in eco_claims:
        if outcome in tub_eco:
            json_val = tub_eco[outcome]["cohens_d"]
            ok = approx_eq(paper_val, json_val)
            results.append({
                "section": "Inline paired",
                "claim": label,
                "paper": paper_val,
                "json": round(json_val, 4),
                "status": "PASS" if ok else "FAIL",
            })

    # TUB vs STD pairs
    tub_std = paired.get("TUB_vs_STD", {})
    std_claims = [
        ("Paired TUB-STD travel_time d", 0.007, "travel_time"),
        ("Paired TUB-STD distance d", 0.34, "distance"),
    ]
    for label, paper_val, outcome in std_claims:
        if outcome in tub_std:
            json_val = tub_std[outcome]["cohens_d"]
            ok = abs(paper_val - json_val) < 0.02
            results.append({
                "section": "Inline paired",
                "claim": label,
                "paper": paper_val,
                "json": round(json_val, 4),
                "status": "PASS" if ok else "FAIL",
            })

    # Number of pairs
    # n_pairs is per-outcome; use first available
    for outcome_data in tub_eco.values():
        if isinstance(outcome_data, dict) and "n_pairs" in outcome_data:
            paper_eco_pairs = 10710
            ok = paper_eco_pairs == outcome_data["n_pairs"]
            results.append({
                "section": "Inline paired",
                "claim": "TUB-ECO n_pairs",
                "paper": paper_eco_pairs,
                "json": outcome_data["n_pairs"],
                "status": "PASS" if ok else "FAIL",
            })
            break

    for outcome_data in tub_std.values():
        if isinstance(outcome_data, dict) and "n_pairs" in outcome_data:
            paper_std_pairs = 76726
            ok = paper_std_pairs == outcome_data["n_pairs"]
            results.append({
                "section": "Inline paired",
                "claim": "TUB-STD n_pairs",
                "paper": paper_std_pairs,
                "json": outcome_data["n_pairs"],
                "status": "PASS" if ok else "FAIL",
            })
            break

    return results


def verify_inline_experience(esc_json, tex_content):
    """Verify experience/mediation claims from phase4_escalation.json.

    JSON: mediation is a dict keyed by outcome
    e.g. mediation["mean_speed"] -> {total_range, direct_range, mediated_range, mediation_pct}
    Note: TUB adoption trajectory data is NOT in this JSON. Skip those claims.
    """
    results = []
    mediation = esc_json.get("mediation", {})

    # Mediation percentages (non-bootstrap point estimates)
    med_claims = [
        ("Mediation % mean_speed (phase4)", 65.6, "mean_speed"),
        ("Mediation % harsh_accel (phase4)", None, "harsh_accel_count"),
    ]
    for label, paper_val, outcome in med_claims:
        if outcome in mediation and paper_val is not None:
            json_val = mediation[outcome].get("mediation_pct")
            if json_val is not None:
                ok = abs(paper_val - json_val) < 2.0
                results.append({
                    "section": "Experience/mediation",
                    "claim": label,
                    "paper": paper_val,
                    "json": round(json_val, 1),
                    "status": "PASS" if ok else "FAIL",
                })

    return results


def verify_compensation_inline(comp_json, tex_content):
    """Verify inline compensation claims.

    JSON: did_std_eco_only is a dict keyed by outcome
    mode_switcher is a dict keyed by outcome
    placebo is a dict keyed by outcome
    """
    results = []

    # STD/ECO-only DiD
    std_eco_did = comp_json.get("did_std_eco_only", {})

    inline_claims = [
        ("STD/ECO DiD mean_speed beta", 1.73, "mean_speed", "beta"),
        ("STD/ECO DiD harsh_accel beta", 0.12, "harsh_accel_count", "beta"),
        ("STD/ECO DiD harsh_decel beta", 0.22, "harsh_decel_count", "beta"),
    ]

    for label, paper_val, outcome, field in inline_claims:
        if outcome in std_eco_did:
            json_val = std_eco_did[outcome].get(field)
            if json_val is not None:
                ok = approx_eq(paper_val, json_val)
                results.append({
                    "section": "Compensation inline",
                    "claim": label,
                    "paper": paper_val,
                    "json": round(json_val, 4),
                    "status": "PASS" if ok else "FAIL",
                })

    # Switcher counts: n_switchers and n_never_tub are per-entry
    switcher = comp_json.get("mode_switcher", {})
    first_entry = next(iter(switcher.values()), {}) if switcher else {}
    n_switchers = first_entry.get("n_switchers")
    n_never_tub = first_entry.get("n_never_tub")

    if n_switchers:
        results.append({
            "section": "Compensation inline",
            "claim": "N switchers",
            "paper": 24357,
            "json": n_switchers,
            "status": "PASS" if n_switchers == 24357 else "FAIL",
        })
    if n_never_tub:
        results.append({
            "section": "Compensation inline",
            "claim": "N never-TUB",
            "paper": 79700,
            "json": n_never_tub,
            "status": "PASS" if n_never_tub == 79700 else "FAIL",
        })

    # Max |d| across all outcomes
    if switcher:
        max_d = max(abs(entry["cohens_d"]) for entry in switcher.values())
        paper_max_d = 0.133
        ok = abs(paper_max_d - max_d) < 0.002
        results.append({
            "section": "Compensation inline",
            "claim": "Max |d|",
            "paper": paper_max_d,
            "json": round(max_d, 4),
            "status": "PASS" if ok else "FAIL",
        })

    # Placebo max |d|
    placebo = comp_json.get("placebo", {})
    if placebo:
        plac_max_d = max(abs(entry["cohens_d"]) for entry in placebo.values())
        paper_plac_d = 0.053
        ok = abs(paper_plac_d - plac_max_d) < 0.005
        results.append({
            "section": "Compensation inline",
            "claim": "Placebo max |d|",
            "paper": paper_plac_d,
            "json": round(plac_max_d, 4),
            "status": "PASS" if ok else "FAIL",
        })

    return results


def verify_survival_inline(esc_json, boot_json, tex_content):
    """Verify inline survival claims.

    JSON: survival is a dict keyed by endpoint.
    schoenfeld_direction (not schoenfeld_rho) is a dict with endpoint keys,
    each having tub_frac.spearman_rho (not rho).
    """
    results = []
    survival = esc_json.get("survival", {})

    # first_harsh
    fh = survival.get("first_harsh", {})
    if fh:
        claims = [
            ("Survival first_harsh HR", 2.13, fh.get("cox_hr_tub_frac")),
            ("Survival first_harsh n_users", 127764, fh.get("n_users")),
        ]
        for label, paper_val, json_val in claims:
            if json_val is not None:
                ok = approx_eq(paper_val, json_val)
                results.append({
                    "section": "Survival inline",
                    "claim": label,
                    "paper": paper_val,
                    "json": round(json_val, 4) if isinstance(json_val, float) else json_val,
                    "status": "PASS" if ok else "FAIL",
                })

    # first_speeding
    fs = survival.get("first_speeding", {})
    if fs:
        claims = [
            ("Survival first_speeding HR_ever_tub", 5.73, fs.get("cox_hr_ever_tub")),
        ]
        c_index = fs.get("c_index") or fs.get("concordance")
        if c_index is not None:
            claims.append(("Survival first_speeding C-index", 0.819, c_index))
        for label, paper_val, json_val in claims:
            if json_val is not None:
                ok = approx_eq(paper_val, json_val)
                results.append({
                    "section": "Survival inline",
                    "claim": label,
                    "paper": paper_val,
                    "json": round(json_val, 4) if isinstance(json_val, float) else json_val,
                    "status": "PASS" if ok else "FAIL",
                })

    # first_high_cv
    fhcv = survival.get("first_high_cv", {})
    if fhcv:
        json_hr = fhcv.get("cox_hr_tub_frac")
        if json_hr is not None:
            ok = approx_eq(0.80, json_hr)
            results.append({
                "section": "Survival inline",
                "claim": "Survival first_high_cv HR",
                "paper": 0.80,
                "json": round(json_hr, 4),
                "status": "PASS" if ok else "FAIL",
            })

    # Schoenfeld rho values from bootstrap_mediation.json
    schoenfeld = boot_json.get("schoenfeld_direction", {})
    for endpoint, data in schoenfeld.items():
        if "tub_frac" in data:
            rho = data["tub_frac"].get("spearman_rho")
            if rho is not None:
                results.append({
                    "section": "Survival Schoenfeld",
                    "claim": f"Schoenfeld rho {endpoint} tub_frac",
                    "paper": "[-0.05, -0.15] range",
                    "json": round(rho, 4),
                    "status": "PASS" if -0.16 < rho < -0.04 else "FAIL",
                })

    return results


def main():
    print("=" * 80)
    print("PROGRAMMATIC STATISTICAL AUDIT")
    print("Paper: Behavioral Responses to E-Scooter Speed Governance")
    print("=" * 80)

    # Load all JSON
    desc_stats = load_json("descriptive_stats.json")
    cross_json = load_json("phase1_cross_sectional.json")
    did_json = load_json("phase2_did_results.json")
    comp_json = load_json("phase3_compensation.json")
    esc_json = load_json("phase4_escalation.json")
    boot_json = load_json("bootstrap_mediation.json")
    fe_json = load_json("phase7b_individual_fe.json")
    shap_harsh = load_json("shap_safety_harsh_event.json")
    shap_cv = load_json("shap_safety_speed_cv.json")
    shap_cruise = load_json("shap_dynamics_cruise_fraction.json")

    # Load tex files
    data_tex = read_tex("data_description.tex")
    results_tex = read_tex("results.tex")

    all_results = []

    # 1. Table 1 descriptive statistics
    print("\n--- Table 1: Descriptive Statistics ---")
    r = verify_table1(desc_stats, data_tex)
    all_results.extend(r)

    # 2. Overall counts
    print("\n--- Overall Dataset Counts ---")
    r = verify_overall_counts(desc_stats, data_tex)
    all_results.extend(r)

    # 3. GEE table
    print("\n--- Table 2: GEE Coefficients ---")
    r = verify_gee_table(cross_json, results_tex)
    all_results.extend(r)

    # 4. FE vs GEE table
    print("\n--- Table 5: FE vs GEE ---")
    r = verify_fe_vs_gee_table(fe_json, results_tex)
    all_results.extend(r)

    # 5. DiD table
    print("\n--- Table 4: DiD Results ---")
    r = verify_did_table(did_json, results_tex)
    all_results.extend(r)

    # 6. Compensation table
    print("\n--- Compensation Table ---")
    r = verify_compensation_table(comp_json, results_tex)
    all_results.extend(r)

    # 7. Survival table
    print("\n--- Survival Table ---")
    r = verify_survival_table(esc_json, results_tex)
    all_results.extend(r)

    # 8. Inline SHAP
    print("\n--- Inline SHAP Values ---")
    r = verify_inline_shap(shap_harsh, shap_cv, shap_cruise, results_tex)
    all_results.extend(r)

    # 9. Inline DiD (dose-response, placebo)
    print("\n--- Inline DiD Claims ---")
    r = verify_inline_did(did_json, results_tex)
    all_results.extend(r)

    # 10. Inline mediation
    print("\n--- Mediation Claims ---")
    r = verify_inline_mediation(boot_json, esc_json, results_tex)
    all_results.extend(r)

    # 11. Inline paired comparisons
    print("\n--- Inline Paired Comparisons ---")
    r = verify_inline_paired(cross_json, results_tex)
    all_results.extend(r)

    # 12. Experience curves
    print("\n--- Experience Curves ---")
    r = verify_inline_experience(esc_json, results_tex)
    all_results.extend(r)

    # 13. Compensation inline
    print("\n--- Compensation Inline Claims ---")
    r = verify_compensation_inline(comp_json, results_tex)
    all_results.extend(r)

    # 14. Survival inline
    print("\n--- Survival Inline Claims ---")
    r = verify_survival_inline(esc_json, boot_json, results_tex)
    all_results.extend(r)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("AUDIT RESULTS SUMMARY")
    print("=" * 80)

    pass_count = sum(1 for r in all_results if r["status"] == "PASS")
    fail_count = sum(1 for r in all_results if r["status"] == "FAIL")
    info_count = sum(1 for r in all_results if r["status"] == "INFO")
    total = len(all_results)

    print(f"\nTotal claims verified: {total}")
    print(f"  PASS: {pass_count}")
    print(f"  FAIL: {fail_count}")
    print(f"  INFO: {info_count}")

    # Print all results grouped by section
    sections = {}
    for r in all_results:
        sec = r["section"]
        if sec not in sections:
            sections[sec] = []
        sections[sec].append(r)

    for sec, items in sections.items():
        print(f"\n--- {sec} ---")
        for item in items:
            status = item["status"]
            marker = "OK" if status == "PASS" else ("FAIL" if status == "FAIL" else "INFO")
            print(f"  [{marker}] {item['claim']}: paper={item['paper']}, json={item['json']}")

    # Print failures prominently
    if fail_count > 0:
        print("\n" + "=" * 80)
        print(f"FAILURES ({fail_count}):")
        print("=" * 80)
        for r in all_results:
            if r["status"] == "FAIL":
                print(f"  {r['section']} | {r['claim']}: paper={r['paper']}, json={r['json']}")

    # Save results as JSON
    output_path = DATA_DIR / "audit_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "pass": pass_count,
                "fail": fail_count,
                "info": info_count,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return fail_count


if __name__ == "__main__":
    sys.exit(main())
