"""
physassistbench/pipeline/ehr_prefetch.py — Pre-fetch all available EHR data for a patient
before benchmark generation begins.

Purpose:
  The user question agent and planner agent must only generate questions about
  data that ACTUALLY EXISTS in the EHR.  This module calls every relevant EHR
  tool once, resolves raw itemids → human-readable labels using the MIMIC-IV
  dictionary tables, and returns a structured text snapshot.

  The snapshot is injected into `context["ehr_snapshot"]` and rendered into
  `context["context_str"]`, replacing the lightweight placeholder that
  `context_builder.py` produces.

Usage:
    from physassistbench.pipeline.ehr_prefetch import build_ehr_snapshot
    snapshot_text = build_ehr_snapshot(subject_id, hadm_id)
    context["ehr_snapshot"] = snapshot_text
"""

from __future__ import annotations

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ── Lookup-table paths ────────────────────────────────────────────────────────
# Central MIMIC-IV dictionary files (not split per patient)
from physassistbench.paths import MIMIC_REF_ROOT as _MIMIC_ROOT
_D_LABITEMS_PATH = os.path.join(_MIMIC_ROOT, "hosp", "d_labitems.csv.gz")
_D_ICD_DX_PATH   = os.path.join(_MIMIC_ROOT, "hosp", "d_icd_diagnoses.csv.gz")

# Singleton caches — loaded once per process
_LAB_LABEL_MAP: dict[int, str] | None = None
_ICD_LABEL_MAP: dict[tuple[str, int], str] | None = None


def _get_lab_label_map() -> dict[int, str]:
    global _LAB_LABEL_MAP
    if _LAB_LABEL_MAP is None:
        try:
            df = pd.read_csv(_D_LABITEMS_PATH, compression="gzip", low_memory=False)
            _LAB_LABEL_MAP = dict(zip(df["itemid"].astype(int), df["label"].astype(str)))
        except Exception as e:
            logger.warning(f"Could not load d_labitems: {e}")
            _LAB_LABEL_MAP = {}
    return _LAB_LABEL_MAP


def _get_icd_label_map() -> dict[tuple[str, int], str]:
    global _ICD_LABEL_MAP
    if _ICD_LABEL_MAP is None:
        try:
            df = pd.read_csv(_D_ICD_DX_PATH, compression="gzip", low_memory=False)
            _ICD_LABEL_MAP = {
                (str(row["icd_code"]).strip(), int(row["icd_version"])): str(row["long_title"])
                for _, row in df.iterrows()
            }
        except Exception as e:
            logger.warning(f"Could not load d_icd_diagnoses: {e}")
            _ICD_LABEL_MAP = {}
    return _ICD_LABEL_MAP


# ── Per-patient data reader ───────────────────────────────────────────────────

from physassistbench.paths import MIMIC_PATIENT_ROOT as _DATA_ROOT


def _read_patient_table(subject_id: int, table: str) -> pd.DataFrame | None:
    fpath = os.path.join(_DATA_ROOT, str(subject_id), f"{table}.csv.csv.gz")
    if not os.path.exists(fpath):
        return None
    try:
        return pd.read_csv(fpath, compression="gzip", low_memory=False)
    except Exception:
        return None


# ── Section builders ──────────────────────────────────────────────────────────

def _build_labs_section(subject_id: int, hadm_id: int | None) -> str:
    df = _read_patient_table(subject_id, "hosp_labevents")
    if df is None or df.empty:
        return ""

    if hadm_id is not None and "hadm_id" in df.columns:
        df = df[df["hadm_id"] == hadm_id]

    if df.empty:
        return ""

    lab_map = _get_lab_label_map()
    lines = []

    # Group by itemid — show one representative row per test
    for itemid, grp in df.groupby("itemid"):
        label = lab_map.get(int(itemid), f"itemid:{itemid}")
        # Most recent non-null value
        grp_sorted = grp.sort_values("charttime", ascending=False) if "charttime" in grp.columns else grp
        row = grp_sorted.iloc[0]
        value = row.get("value", "")
        unit  = row.get("valueuom", "")
        flag  = row.get("flag", "")
        flag_str = f" [{flag}]" if pd.notna(flag) and str(flag).strip() else ""
        lines.append(f"  {label}: {value} {unit}{flag_str}".rstrip())

    if not lines:
        return ""
    return "Lab Results (available tests):\n" + "\n".join(sorted(lines)) + "\n"


def _build_diagnoses_section(subject_id: int, hadm_id: int | None) -> str:
    df = _read_patient_table(subject_id, "hosp_diagnoses_icd")
    if df is None or df.empty:
        return ""

    if hadm_id is not None and "hadm_id" in df.columns:
        df = df[df["hadm_id"] == hadm_id]

    if df.empty:
        return ""

    icd_map = _get_icd_label_map()
    lines = []
    for _, row in df.sort_values("seq_num").iterrows() if "seq_num" in df.columns else df.iterrows():
        code    = str(row.get("icd_code", "")).strip()
        version = int(row.get("icd_version", 9))
        title   = icd_map.get((code, version), code)
        lines.append(f"  ICD-{version} {code}: {title}")

    if not lines:
        return ""
    return "Diagnoses:\n" + "\n".join(lines) + "\n"


def _build_prescriptions_section(subject_id: int, hadm_id: int | None) -> str:
    df = _read_patient_table(subject_id, "hosp_prescriptions")
    if df is None or df.empty:
        return ""

    if hadm_id is not None and "hadm_id" in df.columns:
        df = df[df["hadm_id"] == hadm_id]

    if df.empty:
        return ""

    drugs = []
    for _, row in df.iterrows():
        drug  = str(row.get("drug", "")).strip()
        dose  = str(row.get("dose_val_rx", "")).strip()
        unit  = str(row.get("dose_unit_rx", "")).strip()
        route = str(row.get("route", "")).strip()
        if drug:
            drugs.append(f"  {drug} {dose} {unit} {route}".rstrip())

    if not drugs:
        return ""
    # Deduplicate
    drugs = list(dict.fromkeys(drugs))
    return "Prescriptions (active medications):\n" + "\n".join(drugs) + "\n"


def _build_microbiology_section(subject_id: int, hadm_id: int | None) -> str:
    df = _read_patient_table(subject_id, "hosp_microbiologyevents")
    if df is None or df.empty:
        return ""

    if hadm_id is not None and "hadm_id" in df.columns:
        df = df[df["hadm_id"] == hadm_id]

    if df.empty:
        return ""

    lines = []
    for _, row in df.iterrows():
        spec    = str(row.get("spec_type_desc", "")).strip()
        test    = str(row.get("test_name", "")).strip()
        org     = str(row.get("org_name", "")).strip()
        ab      = str(row.get("ab_name", "")).strip()
        interp  = str(row.get("interpretation", "")).strip()
        comment = str(row.get("comments", "")).strip()

        parts = [f"{spec} — {test}"]
        if org and org != "nan":
            parts.append(f"organism: {org}")
        if ab and ab != "nan":
            parts.append(f"antibiotic: {ab} ({interp})")
        if comment and comment != "nan":
            parts.append(comment[:120])
        lines.append("  " + "; ".join(parts))

    if not lines:
        return ""
    return "Microbiology:\n" + "\n".join(lines) + "\n"


def _build_vitals_section(subject_id: int) -> str:
    df = _read_patient_table(subject_id, "hosp_omr")
    if df is None or df.empty:
        return ""

    lines = []
    for _, row in df.sort_values("chartdate", ascending=False).head(10).iterrows() if "chartdate" in df.columns else df.head(10).iterrows():
        result_name  = str(row.get("result_name", "")).strip()
        result_value = str(row.get("result_value", "")).strip()
        if result_name and result_value:
            lines.append(f"  {result_name}: {result_value}")

    if not lines:
        return ""
    return "Outpatient Vitals (most recent):\n" + "\n".join(lines) + "\n"


def _build_radiology_section(subject_id: int, hadm_id: int | None) -> str:
    df = _read_patient_table(subject_id, "note_radiology")
    if df is None or df.empty:
        return ""

    if hadm_id is not None and "hadm_id" in df.columns:
        df = df[df["hadm_id"] == hadm_id]

    if df.empty:
        return ""

    lines = []
    for _, row in df.iterrows():
        note_type = str(row.get("note_type", "")).strip()
        text      = str(row.get("text", "")).strip()[:200]
        lines.append(f"  [{note_type}] {text}...")

    if not lines:
        return ""
    return f"Radiology Reports ({len(lines)} reports available):\n" + "\n".join(lines[:3]) + "\n"


# ── Tool → section mapping ────────────────────────────────────────────────────

# Maps tool names to the section builder that produces queryable data
_TOOL_TO_SECTION = {
    "get_lab_results":          "labs",
    "get_lab_trends":           "labs",
    "get_diagnoses":            "diagnoses",
    "get_prescriptions":        "prescriptions",
    "get_microbiology_results": "microbiology",
    "get_vital_signs_outpatient": "vitals",
    "get_radiology_report":     "radiology",
    # These don't add new queryable items to the snapshot
    "get_patient_info":         None,
    "get_admissions":           None,
    "get_admission_details":    None,
    "get_patient_timeline":     None,
    "ask_user_for_required_parameters": None,
    "prepare_to_answer":        None,
}

# ── PHM patient section ───────────────────────────────────────────────────────

def _build_phm_patient_section(subject_id: int) -> str:
    """
    Extract a brief patient-reported summary from the PHM YAML.
    Used by the session planner to decide if/which patient interview tools are useful.
    Returns empty string if PHM is unavailable or has no relevant data.
    """
    import os as _os
    # ehr_prefetch.py lives at physassistbench/pipeline/ehr_prefetch.py
    # output/ lives at PhysAssistBench/output/ — three levels up from this file
    _physassistbench_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _repo_root = _os.path.dirname(_physassistbench_dir)
    phm_path = _os.path.join(_repo_root, "output", f"PHM_{subject_id}.yaml")
    if not _os.path.exists(phm_path):
        return ""
    try:
        import yaml
        with open(phm_path, encoding="utf-8") as f:
            phm = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"PHM load failed for {subject_id}: {e}")
        return ""

    lines = ["Patient-Reported Information (available for patient interview tools):"]

    # Medications with adherence status
    meds = phm.get("medications", [])
    if meds:
        lines.append("  Medications (patient self-report):")
        for m in meds[:10]:
            drug = m.get("drug", "")
            if not drug:
                continue
            # Shorten to first meaningful segment (before first comma or newline)
            drug_short = drug.split(",")[0].strip()
            adherence = m.get("adherence", "unknown")
            lines.append(f"    - {drug_short}  [adherence: {adherence}]")

    # Diagnoses in patient-friendly terms (shows what patient understands/can describe)
    diagnoses = phm.get("diagnoses", [])
    if diagnoses:
        lines.append("  Patient-understood diagnoses:")
        for d in diagnoses[:6]:
            pt = d.get("patient_term", "") or d.get("medical_term", "")
            if pt:
                lines.append(f"    - {str(pt)[:120]}")

    # Warning signs ≈ symptoms the patient is aware of / should watch for
    warnings = phm.get("warning_signs", [])
    if warnings:
        lines.append("  Known symptoms / warning signs:")
        for w in warnings[:4]:
            cond = w.get("medical_condition", "") or w.get("condition", "")
            if cond:
                lines.append(f"    - {cond}")

    if len(lines) == 1:  # only header, no content
        return ""

    lines.append(
        "  NOTE: Use patient.get_medication_adherence(drug=X) only for drugs listed above. "
        "Use patient.get_symptom_history(query=X) only for symptoms/conditions listed above."
    )
    return "\n".join(lines) + "\n"


# ── Clinical scoring opportunities detector ───────────────────────────────────

# Lab itemid → (display name, SOFA/scoring component)
_SCORE_LAB_MAP = {
    # Bilirubin (SOFA liver, Child-Pugh)
    50885: ("Bilirubin Total", "SOFA-liver / Child-Pugh"),
    50883: ("Bilirubin Direct", "SOFA-liver"),
    # Platelets (SOFA coagulation, Child-Pugh)
    51265: ("Platelet Count", "SOFA-coagulation"),
    # Creatinine (SOFA renal, Cockcroft-Gault, CURB-65)
    50912: ("Creatinine", "SOFA-renal / CrCl"),
    # BUN (CURB-65)
    51842: ("BUN", "CURB-65"),
    # Albumin (Child-Pugh)
    50862: ("Albumin", "Child-Pugh"),
    # PT/INR (Child-Pugh, MELD)
    51274: ("PT",  "Child-Pugh / MELD"),
    51237: ("INR", "Child-Pugh / MELD"),
    # WBC (SIRS)
    51301: ("WBC", "SIRS"),
    # Lactate (sepsis severity)
    50813: ("Lactate", "sepsis severity"),
}

# ICD prefix → scoring system hint
_ICD_SCORE_HINTS = {
    "I48":   "→ CHA₂DS₂-VASc (afib stroke risk) + HAS-BLED applicable",
    "42731": "→ CHA₂DS₂-VASc (afib stroke risk) + HAS-BLED applicable",
    "K74":   "→ Child-Pugh / MELD (liver disease severity) applicable",
    "5712":  "→ Child-Pugh / MELD (liver disease severity) applicable",
    "5715":  "→ Child-Pugh / MELD (liver disease severity) applicable",
    "J18":   "→ CURB-65 (pneumonia severity) applicable",
    "486":   "→ CURB-65 (pneumonia severity) applicable",
    "A41":   "→ qSOFA / SOFA (sepsis severity) applicable",
    "99591": "→ qSOFA / SOFA (sepsis severity) applicable",
}


def _build_scoring_opportunities(subject_id: int, hadm_id: int | None) -> str:
    """
    Detect which validated clinical scoring systems are computable from
    this patient's available lab data and diagnoses.

    This section is injected into the EHR snapshot so the session planner
    knows explicitly which scores to compute — bridging the gap between
    prefilter (data exists) and session planning (score gets calculated).
    """
    labs = _read_patient_table(subject_id, "hosp_labevents")
    if labs is None or labs.empty:
        return ""
    if hadm_id is not None and "hadm_id" in labs.columns:
        labs = labs[labs["hadm_id"] == hadm_id]
    if labs.empty:
        return ""

    lab_map = _get_lab_label_map()
    present_items = set(labs["itemid"].dropna().astype(int))

    # Build per-score availability
    score_lines: list[str] = []

    # ── SOFA (partial) ────────────────────────────────────────────────────────
    sofa_parts = []
    sofa_item_map = {
        frozenset([50885, 50883]): "Bilirubin",
        frozenset([51265]):        "Platelets",
        frozenset([50912, 51081]): "Creatinine",
    }
    sofa_vals: dict[str, str] = {}
    for item_set, label in sofa_item_map.items():
        matched = present_items & item_set
        if matched:
            row = labs[labs["itemid"].isin(matched)].sort_values(
                "charttime", ascending=False
            ).iloc[0] if "charttime" in labs.columns else labs[labs["itemid"].isin(matched)].iloc[0]
            val  = row.get("value", "")
            unit = row.get("valueuom", "")
            sofa_vals[label] = f"{val} {unit}".strip()
    if len(sofa_vals) >= 2:
        parts_str = ", ".join(f"{k}={v}" for k, v in sofa_vals.items())
        score_lines.append(
            f"  SOFA (partial, {len(sofa_vals)}/6 components): {parts_str}\n"
            f"    → Plan a KG turn to compute the partial SOFA score and interpret organ dysfunction."
        )

    # ── SIRS ─────────────────────────────────────────────────────────────────
    wbc_items = {51301, 51755, 51756} & present_items
    if wbc_items:
        wbc_row = labs[labs["itemid"].isin(wbc_items)].iloc[0]
        wbc_val = wbc_row.get("value", "?")
        score_lines.append(
            f"  SIRS criteria: WBC={wbc_val} K/μL available\n"
            f"    → Plan a KG turn to evaluate SIRS criteria (temp/HR/RR/WBC) for infection vs. SIRS diagnosis."
        )

    # ── Child-Pugh ────────────────────────────────────────────────────────────
    cp_items = {50885, 50883, 50862, 51274, 51237} & present_items
    if len(cp_items) >= 3:
        cp_vals = []
        for iid, label in [(50885, "Bili"), (50862, "Alb"), (51237, "INR")]:
            if iid in present_items:
                row = labs[labs["itemid"] == iid].iloc[0]
                cp_vals.append(f"{label}={row.get('value','?')}")
        score_lines.append(
            f"  Child-Pugh score: {', '.join(cp_vals)} available\n"
            f"    → Plan a KG turn to compute Child-Pugh score (also need ascites/encephalopathy from patient interview)."
        )

    # ── CURB-65 ───────────────────────────────────────────────────────────────
    if {51842} & present_items and {50912} & present_items:
        bun_row = labs[labs["itemid"] == 51842].iloc[0]
        score_lines.append(
            f"  CURB-65: BUN={bun_row.get('value','?')} available\n"
            f"    → Plan a KG turn to compute CURB-65 score for pneumonia severity (also need age/RR/BP)."
        )

    # ── Diagnosis-triggered scores ────────────────────────────────────────────
    dx = _read_patient_table(subject_id, "hosp_diagnoses_icd")
    if dx is not None and not dx.empty:
        if hadm_id is not None and "hadm_id" in dx.columns:
            dx = dx[dx["hadm_id"] == hadm_id]
        if not dx.empty and "icd_code" in dx.columns:
            codes = dx["icd_code"].dropna().astype(str).tolist()
            seen_hints: set[str] = set()
            for code in codes:
                for prefix, hint in _ICD_SCORE_HINTS.items():
                    if code.startswith(prefix) and hint not in seen_hints:
                        score_lines.append(f"  Diagnosis ({code}) {hint}")
                        seen_hints.add(hint)

    if not score_lines:
        return ""

    header = (
        "CLINICAL SCORING OPPORTUNITIES (computable from available data):\n"
        "The following validated scoring systems CAN be computed for this patient.\n"
        "Session planner: PRIORITISE a Clinical Reasoning turn that computes one of these scores.\n"
    )
    return header + "\n".join(score_lines) + "\n"


# ── Public API ────────────────────────────────────────────────────────────────

def build_ehr_snapshot(subject_id: int, hadm_id: int | None,
                       available_tool_names: list[str] | None = None) -> str:
    """
    Build a human-readable EHR snapshot for a patient/admission.

    Always extracts from ALL tables regardless of available_tool_names,
    so the user agent has maximum grounding data.

    Returns a plain-text string with actual item names (not itemid numbers).
    """
    sections = []

    try:
        s = _build_labs_section(subject_id, hadm_id)
        if s:
            sections.append(s)
    except Exception as e:
        logger.warning(f"Labs section failed: {e}")

    try:
        s = _build_diagnoses_section(subject_id, hadm_id)
        if s:
            sections.append(s)
    except Exception as e:
        logger.warning(f"Diagnoses section failed: {e}")

    try:
        s = _build_prescriptions_section(subject_id, hadm_id)
        if s:
            sections.append(s)
    except Exception as e:
        logger.warning(f"Prescriptions section failed: {e}")

    try:
        s = _build_microbiology_section(subject_id, hadm_id)
        if s:
            sections.append(s)
    except Exception as e:
        logger.warning(f"Microbiology section failed: {e}")

    try:
        s = _build_vitals_section(subject_id)
        if s:
            sections.append(s)
    except Exception as e:
        logger.warning(f"Vitals section failed: {e}")

    try:
        s = _build_radiology_section(subject_id, hadm_id)
        if s:
            sections.append(s)
    except Exception as e:
        logger.warning(f"Radiology section failed: {e}")

    if not sections:
        return "[No EHR data available for this patient/admission]"

    # Append PHM patient section — gives planner grounding for patient interview turns
    try:
        s = _build_phm_patient_section(subject_id)
        if s:
            sections.append(s)
    except Exception as e:
        logger.warning(f"PHM patient section failed: {e}")

    # Append scoring opportunities section — explicitly signals which clinical
    # scoring systems are computable from this patient's labs and diagnoses.
    # This is the key bridge: prefilter ensures data exists; this section tells
    # the session planner WHICH scores to compute and with WHAT values.
    try:
        s = _build_scoring_opportunities(subject_id, hadm_id)
        if s:
            sections.append(s)
    except Exception as e:
        logger.warning(f"Scoring opportunities section failed: {e}")

    header = (
        f"=== EHR Snapshot: subject_id={subject_id}, hadm_id={hadm_id} ===\n"
        "IMPORTANT: Only generate questions about items listed below. "
        "Do NOT ask about tests, drugs, or diagnoses not present in this snapshot.\n\n"
    )
    return header + "\n".join(sections)


def has_queryable_data(subject_id: int, hadm_id: int | None,
                       available_tool_names: list[str] | None = None) -> bool:
    """
    Return True if the patient has at least some EHR data available.
    Used to skip patients with no data before generation begins.
    """
    snapshot = build_ehr_snapshot(subject_id, hadm_id)
    return "[No EHR data available" not in snapshot
