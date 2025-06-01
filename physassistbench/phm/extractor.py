"""
Phase 2 — Structured Field Extraction (Hard Rules only).

Medication list, diagnosis list, adherence evidence, and lab trends are all
extracted via regex or direct DataFrame queries — no LLM involved.
"""
from __future__ import annotations
import re
from statistics import mean
from typing import Optional

import pandas as pd

from .schema import Medication, LabTrend
from .section_parser import DEID_PATTERN

# ---------------------------------------------------------------------------
# Medication parsing
# ---------------------------------------------------------------------------

# Matches lines like:
#   1. Furosemide 40 mg PO DAILY
#   2. Lactulose 30 mL PO TID
#   3. Trimethoprim-Sulfamethoxazole 160-800 mg PO 3X/WEEK
MEDICATION_LINE_PATTERN = re.compile(
    r"^\s*\d+[.)]\s+"           # numbered item
    r"(.+?)\s+"                 # drug name (greedy up to dose)
    r"([\d.,/-]+)\s*"           # dose value(s)
    r"(mg|mL|mcg|units?|tabs?|caps?|gm|g|mg/kg|%|IU|mEq|PUFF|NEB|SPRAY|DROP)\s*"  # dose unit
    r"(PO|IV|IH|SC|TD|SL|PR|NG|IM|TOP|INH|OTIC|NASAL)?\s*"    # optional route
    r"(.+)?$",                  # frequency / remainder
    re.IGNORECASE
)

# Fallback: numbered item without a cleanly parsable dose
MEDICATION_FALLBACK_PATTERN = re.compile(
    r"^\s*\d+[.)]\s+(.+)$"
)


def parse_medication_list(section_text: str) -> list[dict]:
    """
    Parse a Discharge Medications or Medications on Admission section into
    a list of structured medication dicts.

    Each dict has keys: drug, dose_value, dose_unit, route, frequency, raw_line.
    """
    results: list[dict] = []
    for line in section_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = MEDICATION_LINE_PATTERN.match(line)
        if m:
            results.append({
                "drug":       m.group(1).strip(),
                "dose_value": m.group(2).strip(),
                "dose_unit":  m.group(3).strip(),
                "route":      (m.group(4) or "").strip(),
                "frequency":  (m.group(5) or "").strip(),
                "raw_line":   line,
            })
        else:
            fb = MEDICATION_FALLBACK_PATTERN.match(line)
            if fb:
                results.append({
                    "drug":       fb.group(1).strip(),
                    "dose_value": "",
                    "dose_unit":  "",
                    "route":      "",
                    "frequency":  "",
                    "raw_line":   line,
                })
    return results


# ---------------------------------------------------------------------------
# Adherence evidence extraction (Hard Rules)
# ---------------------------------------------------------------------------

ADHERENCE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Use \s+ to handle line breaks within phrases (common in MIMIC-IV wrapped text)
    ("never_filled",      re.compile(r"never\s+filled|did\s+not\s+fill|hasn[''`]t\s+filled", re.I)),
    ("never_filled",      re.compile(r"never\s+pick(?:ed)?\s+up|did\s+not\s+pick\s+up", re.I)),
    ("self_discontinued", re.compile(r"self[- ]?discontinu\w+", re.I)),
    ("ran_out",           re.compile(r"ran\s+out\s+of", re.I)),
    ("poor",              re.compile(r"(?:has\s+not\s+been|haven[''`]t\s+been|not\s+been|stopped)\s+taking", re.I)),
    ("poor",              re.compile(r"not\s+(?:taking|compliant\s+with)", re.I)),
    ("poor",              re.compile(r"except\s+for\s+\w", re.I)),   # "taking all except for X"
    ("good",              re.compile(r"(?:compliant\s+with|taking\s+all|adherent\s+to)", re.I)),
]

# Capture all verbatim double-quoted strings in HPI
_QUOTE_PATTERN = re.compile(r'"([^"]{5,300})"')


def extract_adherence_evidence(hpi_text: str, drug_name: str) -> dict:
    """
    Search HPI text for adherence signals related to a specific drug.

    Returns {adherence_type, evidence_quotes} where evidence_quotes is a list
    of verbatim text snippets supporting the classification.
    """
    drug_lower = drug_name.lower().split()[0]   # use first word for fuzzy match
    evidence_quotes: list[str] = []
    classified_type: str = "unknown"

    for adh_type, pattern in ADHERENCE_PATTERNS:
        for m in pattern.finditer(hpi_text):
            # Check whether this match mentions the drug
            context = hpi_text[max(0, m.start() - 80):m.end() + 80]
            if drug_lower in context.lower() or not drug_name:
                evidence_quotes.append(m.group(0).strip())
                if classified_type == "unknown":
                    classified_type = adh_type

    # Collect direct quotes that mention the drug
    for m in _QUOTE_PATTERN.finditer(hpi_text):
        q = m.group(1)
        if drug_lower in q.lower():
            evidence_quotes.append(f'"{q}"')

    # De-duplicate
    evidence_quotes = list(dict.fromkeys(evidence_quotes))
    return {"adherence_type": classified_type, "evidence_quotes": evidence_quotes}


def extract_all_adherence_evidence(hpi_text: str) -> dict[str, dict]:
    """
    Scan HPI for all adherence signals without targeting a specific drug.
    Returns {drug_fragment: {adherence_type, evidence_quotes}}.
    """
    all_evidence: dict[str, dict] = {}
    for adh_type, pattern in ADHERENCE_PATTERNS:
        for m in pattern.finditer(hpi_text):
            drug_fragment = m.group(len(m.groups())).strip().split()[0] if m.lastindex else "medication"
            key = drug_fragment.lower()
            if key not in all_evidence:
                all_evidence[key] = {"adherence_type": adh_type, "evidence_quotes": []}
            all_evidence[key]["evidence_quotes"].append(m.group(0).strip())
    return all_evidence


# ---------------------------------------------------------------------------
# Diagnosis list extraction from Discharge Diagnosis section
# ---------------------------------------------------------------------------

_DIAG_LINE = re.compile(r"^\s*(?:\d+[.)]\s+|[-•*]\s+)?(.+)$")
_ICD_CODE = re.compile(r"\b[A-Z]\d{2}(?:\.\d+)?\b")   # e.g. K74.60


def parse_diagnosis_list(dc_diagnosis_text: str) -> list[str]:
    """
    Extract a list of diagnosis strings from the Discharge Diagnosis section.
    Returns raw medical terms; patient_term translation is done by LLM later.
    """
    diagnoses: list[str] = []
    for line in dc_diagnosis_text.splitlines():
        line = line.strip()
        if not line or DEID_PATTERN.search(line):
            continue
        m = _DIAG_LINE.match(line)
        if m:
            diag = m.group(1).strip()
            # skip lines that are only ICD codes or trivially short
            if diag and len(diag) > 3 and not _ICD_CODE.fullmatch(diag):
                diagnoses.append(diag)
    return diagnoses


# ---------------------------------------------------------------------------
# Lab trends — hard rules querying hosp_labevents
# ---------------------------------------------------------------------------

# Key lab itemids in MIMIC-IV (D_LABITEMS standard values)
LAB_ITEM_KEYWORDS: dict[str, list[str]] = {
    "sodium":         ["sodium", "Na"],
    "potassium":      ["potassium", "K"],
    "creatinine":     ["creatinine"],
    "bilirubin":      ["bilirubin", "bili"],
    "albumin":        ["albumin"],
    "INR":            ["INR", "PT"],
    "ammonia":        ["ammonia", "NH3"],
    "hemoglobin":     ["hemoglobin", "Hgb", "Hb"],
    "platelets":      ["platelet", "plt"],
    "WBC":            ["wbc", "white blood"],
    "ALT":            ["ALT", "alanine"],
    "AST":            ["AST", "aspartate"],
    "glucose":        ["glucose"],
}


def build_lab_trends(
    labevents_df: pd.DataFrame,
    item_keywords: Optional[dict[str, list[str]]] = None,
    n_recent: int = 3,
) -> list[LabTrend]:
    """
    Compute lab trend summaries from the labevents DataFrame.

    Uses the 'comments' column (which contains test names in MIMIC-IV) for
    keyword matching. Falls back to itemid if comments are missing.
    """
    if labevents_df is None or labevents_df.empty:
        return []

    keywords = item_keywords or LAB_ITEM_KEYWORDS
    has_comments = "comments" in labevents_df.columns
    df = labevents_df.copy()
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

    trends: list[LabTrend] = []
    for item_name, kws in keywords.items():
        # Build mask: any keyword matches comments or itemid string
        mask = pd.Series([False] * len(df), index=df.index)
        for kw in kws:
            if has_comments:
                mask |= df["comments"].str.contains(kw, case=False, na=False)
            mask |= df["itemid"].astype(str).str.contains(kw, case=False, na=False)

        subset = df[mask].copy()
        if subset.empty:
            continue

        if "charttime" in subset.columns:
            subset = subset.sort_values("charttime", ascending=False)

        recent = subset.head(n_recent)
        values = recent["valuenum"].dropna().tolist() if "valuenum" in recent.columns else []
        unit = recent["valueuom"].dropna().iloc[0] if ("valueuom" in recent.columns and not recent["valueuom"].dropna().empty) else ""

        recent_records = []
        for _, row in recent.iterrows():
            recent_records.append({
                "charttime": str(row.get("charttime", "")),
                "value":     row.get("value", ""),
                "valuenum":  row.get("valuenum"),
                "flag":      row.get("flag", ""),
                "hadm_id":   row.get("hadm_id"),
            })

        trend = LabTrend(
            item_name=item_name,
            itemid=int(subset["itemid"].iloc[0]) if not subset.empty else None,
            unit=str(unit),
            recent_values=recent_records,
            value_min=float(min(values)) if values else None,
            value_max=float(max(values)) if values else None,
            trend_direction=_classify_trend(values),
        )
        trends.append(trend)
    return trends


def _classify_trend(values: list[float]) -> str:
    """Classify a short numeric series as improving/worsening/stable/fluctuating."""
    if len(values) < 2:
        return "stable"
    # values are sorted newest-first, so reverse for chronological order
    chron = list(reversed(values))
    diffs = [chron[i + 1] - chron[i] for i in range(len(chron) - 1)]
    if all(d > 0 for d in diffs):
        return "worsening"   # monotonically rising (might be bad, context-dependent)
    if all(d < 0 for d in diffs):
        return "improving"   # monotonically falling
    if max(abs(d) for d in diffs) < 0.1 * (max(chron) - min(chron) + 1e-9):
        return "stable"
    return "fluctuating"


# ---------------------------------------------------------------------------
# Cross-validation helpers (used by verifier)
# ---------------------------------------------------------------------------

def fuzzy_drug_match(drug_name: str, prescriptions_df: pd.DataFrame) -> Optional[dict]:
    """
    Return the best-matching prescription row for a drug name, or None.
    Matching: first word of drug_name must appear in the 'drug' column (case-insensitive).
    """
    if prescriptions_df is None or prescriptions_df.empty or "drug" not in prescriptions_df.columns:
        return None
    key = drug_name.lower().split()[0]
    mask = prescriptions_df["drug"].str.lower().str.contains(key, na=False)
    hits = prescriptions_df[mask]
    if hits.empty:
        return None
    return hits.iloc[0].where(pd.notnull(hits.iloc[0]), None).to_dict()


def align_with_icd(medical_term: str, diagnoses_icd_df: pd.DataFrame) -> Optional[str]:
    """
    Find a matching ICD long_title in hosp_diagnoses_icd for a free-text medical_term.
    Returns the standardised ICD title if found, else None.
    """
    if diagnoses_icd_df is None or diagnoses_icd_df.empty:
        return None
    # The diagnoses table may have 'long_title' from d_icd_diagnoses join,
    # or just icd_code. Try both.
    key_words = set(medical_term.lower().split())
    for col in ["long_title", "icd_title", "icd_code"]:
        if col not in diagnoses_icd_df.columns:
            continue
        for val in diagnoses_icd_df[col].dropna():
            val_words = set(str(val).lower().split())
            overlap = len(key_words & val_words) / max(len(key_words), 1)
            if overlap >= 0.5:
                return str(val)
    return None
