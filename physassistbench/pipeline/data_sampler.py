"""
data_sampler.py — Sample real MIMIC-IV patient data to ground user question generation.

Called before user_agent for Lookup/Data Gathering turns to provide confirmed facts about
the patient, ensuring generated questions are grounded in data that actually exists.

Returns a compact human-readable string of facts, e.g.:
  "Lab results: Hemoglobin 11.6 g/dL (abnormal), INR 1.6 (abnormal), Creatinine 0.5 mg/dL
   Diagnoses (ICD-9): Portal hypertension (572.3), Chronic liver disease (571.5)
   Medications: Furosemide 40mg PO, Potassium Chloride 40mEq PO"
"""

import os
import logging
from functools import lru_cache
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Reference table paths ────────────────────────────────────────────────────

_REF_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "MIMIC-IV-split-by-patient"
)

# Try the physionet path used on this server; override via env var if needed
from physassistbench.paths import MIMIC_REF_ROOT as _PHYSIONET_ROOT

from physassistbench.paths import MIMIC_PATIENT_ROOT as _PATIENT_ROOT

_MAX_FACTS = 6   # max items per category to include in grounding string


# ─── Lazy-loaded reference tables ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_d_labitems() -> pd.DataFrame:
    path = os.path.join(_PHYSIONET_ROOT, "hosp", "d_labitems.csv.gz")
    if not os.path.exists(path):
        logger.warning("d_labitems not found at %s", path)
        return pd.DataFrame(columns=["itemid", "label", "category"])
    return pd.read_csv(path, compression="gzip")[["itemid", "label", "category"]]


@lru_cache(maxsize=1)
def _load_d_icd_diagnoses() -> pd.DataFrame:
    path = os.path.join(_PHYSIONET_ROOT, "hosp", "d_icd_diagnoses.csv.gz")
    if not os.path.exists(path):
        logger.warning("d_icd_diagnoses not found at %s", path)
        return pd.DataFrame(columns=["icd_code", "icd_version", "long_title"])
    return pd.read_csv(path, compression="gzip")[["icd_code", "icd_version", "long_title"]]


@lru_cache(maxsize=1)
def _load_d_items() -> pd.DataFrame:
    path = os.path.join(_PHYSIONET_ROOT, "icu", "d_items.csv.gz")
    if not os.path.exists(path):
        logger.warning("d_items not found at %s", path)
        return pd.DataFrame(columns=["itemid", "label", "category", "unitname"])
    return pd.read_csv(path, compression="gzip")[["itemid", "label", "category", "unitname"]]


# ─── Per-patient file reader ───────────────────────────────────────────────────

def _read(subject_id: int, table: str) -> Optional[pd.DataFrame]:
    path = os.path.join(_PATIENT_ROOT, str(subject_id), f"{table}.csv.csv.gz")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, compression="gzip", low_memory=False)
    except Exception as e:
        logger.debug("Error reading %s for patient %s: %s", table, subject_id, e)
        return None


# ─── Domain-specific samplers ─────────────────────────────────────────────────

def _sample_labs(subject_id: int, hadm_id: Optional[int]) -> str:
    """Return up to _MAX_FACTS lab results with real values."""
    lab = _read(subject_id, "hosp_labevents")
    if lab is None or lab.empty:
        return ""

    d_lab = _load_d_labitems()

    # Filter to this admission if specified
    if hadm_id and "hadm_id" in lab.columns:
        adm_lab = lab[lab["hadm_id"] == hadm_id]
        if adm_lab.empty:
            adm_lab = lab   # fall back to all admissions
    else:
        adm_lab = lab

    # Merge with item names, keep numeric values only
    merged = adm_lab.merge(d_lab, on="itemid", how="left")
    merged = merged[merged["valuenum"].notna() & merged["label"].notna()]

    if merged.empty:
        return ""

    # Prioritise abnormal results; deduplicate by label (keep most recent)
    merged["is_abnormal"] = merged["flag"].str.lower().str.contains("abnormal", na=False)
    merged = merged.sort_values(["is_abnormal", "charttime"], ascending=[False, False])
    merged = merged.drop_duplicates(subset="label").head(_MAX_FACTS)

    parts = []
    for _, row in merged.iterrows():
        flag_str = " (abnormal)" if row["is_abnormal"] else ""
        unit = str(row["valueuom"]) if pd.notna(row.get("valueuom")) else ""
        parts.append(f"{row['label']}: {row['value']} {unit}{flag_str}".strip())

    return "Lab results: " + ", ".join(parts) if parts else ""


def _sample_diagnoses(subject_id: int, hadm_id: Optional[int]) -> str:
    """Return up to _MAX_FACTS ICD diagnoses with descriptions."""
    dx = _read(subject_id, "hosp_diagnoses_icd")
    if dx is None or dx.empty:
        return ""

    d_icd = _load_d_icd_diagnoses()

    if hadm_id and "hadm_id" in dx.columns:
        adm_dx = dx[dx["hadm_id"] == hadm_id]
        if adm_dx.empty:
            adm_dx = dx
    else:
        adm_dx = dx

    # Sort by seq_num (principal first), merge description
    adm_dx = adm_dx.sort_values("seq_num") if "seq_num" in adm_dx.columns else adm_dx

    # Normalise key types for merge
    d_icd = d_icd.copy()
    d_icd["icd_code"] = d_icd["icd_code"].astype(str).str.strip()
    d_icd["icd_version"] = d_icd["icd_version"].astype(str)
    adm_dx = adm_dx.copy()
    adm_dx["icd_code"] = adm_dx["icd_code"].astype(str).str.strip()
    adm_dx["icd_version"] = adm_dx["icd_version"].astype(str)

    merged = adm_dx.merge(d_icd, on=["icd_code", "icd_version"], how="left")
    merged = merged.head(_MAX_FACTS)

    parts = []
    for _, row in merged.iterrows():
        ver = f"ICD-{row['icd_version']}" if pd.notna(row.get("icd_version")) else "ICD"
        code = row["icd_code"]
        title = str(row["long_title"]) if pd.notna(row.get("long_title")) else code
        parts.append(f"{ver} {code}: {title}")

    return "Diagnoses: " + "; ".join(parts) if parts else ""


def _sample_medications(subject_id: int, hadm_id: Optional[int]) -> str:
    """Return up to _MAX_FACTS unique medications with dose and route."""
    rx = _read(subject_id, "hosp_prescriptions")
    if rx is None or rx.empty:
        return ""

    if hadm_id and "hadm_id" in rx.columns:
        adm_rx = rx[rx["hadm_id"] == hadm_id]
        if adm_rx.empty:
            adm_rx = rx
    else:
        adm_rx = rx

    # Keep MAIN drug entries, deduplicate by drug name
    if "drug_type" in adm_rx.columns:
        adm_rx = adm_rx[adm_rx["drug_type"] == "MAIN"]
    adm_rx = adm_rx.dropna(subset=["drug"])
    adm_rx = adm_rx.sort_values("starttime", ascending=False) if "starttime" in adm_rx.columns else adm_rx
    adm_rx = adm_rx.drop_duplicates(subset="drug").head(_MAX_FACTS)

    parts = []
    for _, row in adm_rx.iterrows():
        drug = row["drug"]
        dose = str(row.get("dose_val_rx", "")) if pd.notna(row.get("dose_val_rx")) else ""
        unit = str(row.get("dose_unit_rx", "")) if pd.notna(row.get("dose_unit_rx")) else ""
        route = str(row.get("route", "")) if pd.notna(row.get("route")) else ""
        parts.append(f"{drug} {dose}{unit} {route}".strip())

    return "Medications: " + ", ".join(parts) if parts else ""


def _sample_icu(subject_id: int, hadm_id: Optional[int]) -> str:
    """Return ICU stay info and sample of vital signs."""
    stays = _read(subject_id, "icu_icustays")
    lines = []

    if stays is not None and not stays.empty:
        if hadm_id and "hadm_id" in stays.columns:
            adm_stays = stays[stays["hadm_id"] == hadm_id]
            if adm_stays.empty:
                adm_stays = stays
        else:
            adm_stays = stays

        for _, row in adm_stays.head(3).iterrows():
            unit = str(row.get("first_careunit", "ICU"))
            los = round(float(row.get("los", 0)), 1) if pd.notna(row.get("los")) else "?"
            stay_id = int(row.get("stay_id", 0))
            lines.append(f"{unit} (stay_id={stay_id}, LOS={los} days)")

    # Sample a few numeric vital signs from chartevents
    chart = _read(subject_id, "icu_chartevents")
    if chart is not None and not chart.empty:
        d_items = _load_d_items()
        vital_cats = {"Routine Vital Signs"}
        vital_items = d_items[d_items["category"].isin(vital_cats)][["itemid", "label", "unitname"]]

        merged = chart.merge(vital_items, on="itemid", how="inner")
        merged = merged[merged["valuenum"].notna()]
        merged = merged.sort_values("charttime", ascending=False)
        merged = merged.drop_duplicates(subset="label").head(5)

        vitals = []
        for _, row in merged.iterrows():
            unit = str(row["unitname"]) if pd.notna(row.get("unitname")) else ""
            vitals.append(f"{row['label']}: {row['valuenum']} {unit}".strip())
        if vitals:
            lines.append("Vitals: " + ", ".join(vitals))

    return "ICU: " + " | ".join(lines) if lines else ""


def _sample_notes(subject_id: int, hadm_id: Optional[int]) -> str:
    """Return counts of available note types."""
    lines = []
    for table, label in [("note_discharge", "discharge summaries"),
                         ("note_radiology", "radiology reports")]:
        df = _read(subject_id, table)
        if df is not None and not df.empty:
            if hadm_id and "hadm_id" in df.columns:
                n = len(df[df["hadm_id"] == hadm_id])
                n_total = len(df)
                lines.append(f"{n} {label} (this admission), {n_total} total")
            else:
                lines.append(f"{len(df)} {label}")
    return "Notes: " + "; ".join(lines) if lines else ""


# ─── Domain dispatch ──────────────────────────────────────────────────────────

_DOMAIN_SAMPLERS = {
    "LabInterp": lambda sid, hid: "\n".join(filter(None, [
        _sample_labs(sid, hid),
        _sample_diagnoses(sid, hid),
    ])),
    "DiagCode": lambda sid, hid: "\n".join(filter(None, [
        _sample_diagnoses(sid, hid),
        _sample_medications(sid, hid),
        _sample_labs(sid, hid),
        _sample_notes(sid, hid),
    ])),
    "MedRecon": lambda sid, hid: "\n".join(filter(None, [
        _sample_medications(sid, hid),
        _sample_labs(sid, hid),
    ])),
    "ICUReasoning": lambda sid, hid: "\n".join(filter(None, [
        _sample_icu(sid, hid),
        _sample_labs(sid, hid),
    ])),
    "WorkflowQuery": lambda sid, hid: "\n".join(filter(None, [
        _sample_notes(sid, hid),
        _sample_diagnoses(sid, hid),
        _sample_medications(sid, hid),
    ])),
    "DischargePlan": lambda sid, hid: "\n".join(filter(None, [
        _sample_diagnoses(sid, hid),
        _sample_medications(sid, hid),
        _sample_labs(sid, hid),
    ])),
}


def sample_grounding_facts(
    subject_id: int,
    hadm_id: Optional[int],
    task_domain: str,
) -> str:
    """
    Return a compact string of real patient facts for the given domain.
    Returns empty string if data unavailable or on error.
    """
    sampler = _DOMAIN_SAMPLERS.get(task_domain)
    if sampler is None:
        return ""
    try:
        result = sampler(subject_id, hadm_id)
        return result.strip()
    except Exception as e:
        logger.debug("data_sampler error for %s subject=%s: %s", task_domain, subject_id, e)
        return ""
