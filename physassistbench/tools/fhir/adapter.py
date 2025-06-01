"""
tools/fhir/adapter.py — CSV → HL7 FHIR R4 Resource format conversion.

Converts per-patient MIMIC-IV CSV files into FHIR-structured dicts without
requiring a FHIR server.  All returned objects conform to FHIR R4 resource
shapes so downstream tools can use standard FHIR semantics.

Supported resources:
  Patient, Encounter, Condition, Observation (lab / vital-signs),
  MedicationRequest, MedicationAdministration, Procedure,
  DiagnosticReport, DocumentReference, AllergyIntolerance, CarePlan
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import pandas as pd

# ── data paths ────────────────────────────────────────────────────────────────

from physassistbench.paths import (
    MIMIC_PATIENT_ROOT as _PATIENT_ROOT,
    MIMIC_REF_ROOT as _PHYSIONET_ROOT,
)


def _read(subject_id: int, table: str) -> Optional[pd.DataFrame]:
    path = os.path.join(_PATIENT_ROOT, str(subject_id), f"{table}.csv.csv.gz")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, compression="gzip", low_memory=False)
    except Exception:
        return None


def _str(val) -> Optional[str]:
    return str(val) if pd.notna(val) else None


def _float(val) -> Optional[float]:
    try:
        return float(val) if pd.notna(val) else None
    except (TypeError, ValueError):
        return None


# ── reference table helpers ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _lab_label_map() -> dict[int, str]:
    path = os.path.join(_PHYSIONET_ROOT, "hosp", "d_labitems.csv.gz")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    return dict(zip(df["itemid"].astype(int), df["label"].astype(str)))


@lru_cache(maxsize=1)
def _icd_desc_map() -> dict[tuple, str]:
    path = os.path.join(_PHYSIONET_ROOT, "hosp", "d_icd_diagnoses.csv.gz")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    return {
        (str(r["icd_code"]).strip(), str(r["icd_version"])): str(r["long_title"])
        for _, r in df.iterrows()
    }


@lru_cache(maxsize=1)
def _icu_item_map() -> dict[int, tuple[str, str]]:
    """itemid → (label, unitname)"""
    path = os.path.join(_PHYSIONET_ROOT, "icu", "d_items.csv.gz")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    return {
        int(r["itemid"]): (_str(r.get("label", "")) or "", _str(r.get("unitname", "")) or "")
        for _, r in df.iterrows()
    }


# ── FHIR Bundle wrapper ───────────────────────────────────────────────────────

def _bundle(resources: list[dict], resource_type: str) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(resources),
        "entry": [{"resource": r} for r in resources],
        "_fhir_resource_type": resource_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Patient
# ─────────────────────────────────────────────────────────────────────────────

def patient_to_fhir(subject_id: int) -> Optional[dict]:
    """Convert hosp_patients row → FHIR Patient resource."""
    df = _read(subject_id, "hosp_patients")
    if df is None or df.empty:
        return None
    row = df.iloc[0]

    gender_map = {"M": "male", "F": "female"}
    gender = gender_map.get(str(row.get("gender", "")).upper(), "unknown")

    resource: dict = {
        "resourceType": "Patient",
        "id": str(subject_id),
        "identifier": [{"system": "mimic-iv", "value": str(subject_id)}],
        "gender": gender,
    }
    anchor_age = _float(row.get("anchor_age"))
    if anchor_age is not None:
        resource["_anchor_age"] = int(anchor_age)
    dod = _str(row.get("dod"))
    if dod:
        resource["deceasedDateTime"] = dod
    return resource


# ─────────────────────────────────────────────────────────────────────────────
# Encounter
# ─────────────────────────────────────────────────────────────────────────────

def _admission_to_encounter(row, subject_id: int) -> dict:
    hadm_id = _str(row.get("hadm_id"))
    enc: dict = {
        "resourceType": "Encounter",
        "id": f"encounter-{hadm_id}",
        "status": "finished",
        "class": {"code": "IMP", "display": "inpatient encounter"},
        "subject": {"reference": f"Patient/{subject_id}"},
    }
    admit = _str(row.get("admittime"))
    disch = _str(row.get("dischtime"))
    if admit or disch:
        enc["period"] = {}
        if admit:
            enc["period"]["start"] = admit
        if disch:
            enc["period"]["end"] = disch
    if hadm_id:
        enc["identifier"] = [{"system": "mimic-iv/hadm", "value": hadm_id}]
    for field in ["admission_type", "admission_location", "discharge_location",
                  "insurance", "language", "marital_status", "race"]:
        val = _str(row.get(field))
        if val:
            enc.setdefault("_mimic", {})[field] = val
    expire = row.get("hospital_expire_flag")
    if pd.notna(expire):
        enc.setdefault("_mimic", {})["hospital_expire_flag"] = int(expire)
    return enc


def encounters_to_fhir(subject_id: int, hadm_id: Optional[int] = None) -> list[dict]:
    df = _read(subject_id, "hosp_admissions")
    if df is None or df.empty:
        return []
    if hadm_id:
        df = df[df["hadm_id"] == hadm_id]
    return [_admission_to_encounter(row, subject_id) for _, row in df.iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# Condition (Diagnoses)
# ─────────────────────────────────────────────────────────────────────────────

def conditions_to_fhir(
    subject_id: int,
    hadm_id: Optional[int] = None,
    code: Optional[str] = None,
) -> list[dict]:
    df = _read(subject_id, "hosp_diagnoses_icd")
    if df is None or df.empty:
        return []
    if hadm_id:
        df = df[df["hadm_id"] == hadm_id]
    if "seq_num" in df.columns:
        df = df.sort_values(["hadm_id", "seq_num"])
    icd_map = _icd_desc_map()

    resources = []
    for _, row in df.iterrows():
        icd_code = str(row.get("icd_code", "")).strip()
        icd_ver  = str(row.get("icd_version", "10"))
        if code and not icd_code.startswith(code.upper()):
            continue
        desc = icd_map.get((icd_code, icd_ver), icd_code)
        sys_url = (
            "http://hl7.org/fhir/sid/icd-10-cm"
            if icd_ver == "10"
            else "http://hl7.org/fhir/sid/icd-9-cm"
        )
        hadm_val = _str(row.get("hadm_id"))
        resource: dict = {
            "resourceType": "Condition",
            "id": f"condition-{subject_id}-{icd_code}-{hadm_val}",
            "clinicalStatus": {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                             "code": "active"}]
            },
            "category": [{
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category",
                             "code": "encounter-diagnosis",
                             "display": "Encounter Diagnosis"}]
            }],
            "code": {
                "coding": [{"system": sys_url, "code": icd_code, "display": desc}],
                "text": desc,
            },
            "subject": {"reference": f"Patient/{subject_id}"},
        }
        if hadm_val:
            resource["encounter"] = {"reference": f"Encounter/encounter-{hadm_val}"}
        seq = row.get("seq_num")
        if pd.notna(seq):
            resource["_seq_num"] = int(seq)
        resources.append(resource)
    return resources


# ─────────────────────────────────────────────────────────────────────────────
# Observation — Laboratory
# ─────────────────────────────────────────────────────────────────────────────

def lab_observations_to_fhir(
    subject_id: int,
    hadm_id: Optional[int] = None,
    item_name: Optional[str] = None,
    abnormal_only: bool = False,
    _count: int = 20,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict]:
    df = _read(subject_id, "hosp_labevents")
    if df is None or df.empty:
        return []
    if hadm_id:
        df = df[df["hadm_id"] == hadm_id]

    lab_map = _lab_label_map()

    if item_name:
        matched = {iid for iid, lbl in lab_map.items()
                   if isinstance(lbl, str) and item_name.lower() in lbl.lower()}
        if matched:
            df = df[df["itemid"].isin(matched)]

    if abnormal_only:
        df = df[df["flag"].str.upper().isin(["ABNORMAL", "DELTA"], )]

    if date_from and "charttime" in df.columns:
        df = df[df["charttime"] >= date_from]
    if date_to and "charttime" in df.columns:
        df = df[df["charttime"] <= date_to]

    if "charttime" in df.columns:
        df = df.sort_values("charttime", ascending=False)

    df = df.head(_count)

    resources = []
    for _, row in df.iterrows():
        itemid = row.get("itemid")
        label  = lab_map.get(int(itemid), f"itemid:{itemid}") if pd.notna(itemid) else "unknown"
        labevent_id = _str(row.get("labevent_id"))
        hadm_val    = _str(row.get("hadm_id"))

        obs: dict = {
            "resourceType": "Observation",
            "id": f"lab-{labevent_id}" if labevent_id else f"lab-{subject_id}-{itemid}",
            "status": "final",
            "category": [{
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                             "code": "laboratory", "display": "Laboratory"}]
            }],
            "code": {
                "coding": [{"system": "mimic-iv/itemid", "code": str(itemid), "display": label}],
                "text": label,
            },
            "subject": {"reference": f"Patient/{subject_id}"},
        }
        if hadm_val:
            obs["encounter"] = {"reference": f"Encounter/encounter-{hadm_val}"}
        charttime = _str(row.get("charttime"))
        if charttime:
            obs["effectiveDateTime"] = charttime

        value_num = _float(row.get("valuenum"))
        value_str = _str(row.get("value"))
        uom = _str(row.get("valueuom"))
        if value_num is not None:
            obs["valueQuantity"] = {"value": value_num, "unit": uom or ""}
        elif value_str:
            obs["valueString"] = value_str

        flag = _str(row.get("flag"))
        if flag and flag.upper() in ("ABNORMAL", "DELTA"):
            obs["interpretation"] = [{
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                             "code": "A", "display": "Abnormal"}]
            }]

        ref_low  = _float(row.get("ref_range_lower"))
        ref_high = _float(row.get("ref_range_upper"))
        if ref_low is not None or ref_high is not None:
            rr: dict = {}
            if ref_low  is not None:
                rr["low"]  = {"value": ref_low,  "unit": uom or ""}
            if ref_high is not None:
                rr["high"] = {"value": ref_high, "unit": uom or ""}
            obs["referenceRange"] = [rr]

        resources.append(obs)
    return resources


# ─────────────────────────────────────────────────────────────────────────────
# Observation — ICU Vital Signs
# ─────────────────────────────────────────────────────────────────────────────

def vital_observations_to_fhir(
    subject_id: int,
    stay_id: Optional[int] = None,
    vital_name: Optional[str] = None,
    _count: int = 50,
) -> list[dict]:
    resources: list[dict] = []

    # ── Source 1: ICU chartevents ─────────────────────────────────────────────
    df = _read(subject_id, "icu_chartevents")
    if df is not None and not df.empty:
        if stay_id is not None:
            df = df[df["stay_id"] == stay_id]
        if vital_name:
            item_map = _icu_item_map()
            matched = {iid for iid, (lbl, _) in item_map.items()
                       if isinstance(lbl, str) and vital_name.lower() in lbl.lower()}
            if matched:
                df = df[df["itemid"].isin(matched)]
        if "charttime" in df.columns:
            df = df.sort_values("charttime", ascending=False)
        df = df.head(_count)

        item_map = _icu_item_map()
        for _, row in df.iterrows():
            itemid = row.get("itemid")
            label, unit = item_map.get(int(itemid), (f"itemid:{itemid}", "")) if pd.notna(itemid) else ("unknown", "")
            obs: dict = {
                "resourceType": "Observation",
                "id": f"vital-{subject_id}-{itemid}-{_str(row.get('charttime','')) or ''}",
                "status": "final",
                "category": [{
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                 "code": "vital-signs", "display": "Vital Signs"}]
                }],
                "code": {
                    "coding": [{"system": "mimic-iv/icu-itemid", "code": str(itemid), "display": label}],
                    "text": label,
                },
                "subject": {"reference": f"Patient/{subject_id}"},
            }
            stay_val = _str(row.get("stay_id"))
            if stay_val:
                obs["_stay_id"] = stay_val
            charttime = _str(row.get("charttime"))
            if charttime:
                obs["effectiveDateTime"] = charttime
            val_num = _float(row.get("valuenum"))
            val_str = _str(row.get("value"))
            if val_num is not None:
                obs["valueQuantity"] = {"value": val_num, "unit": unit}
            elif val_str:
                obs["valueString"] = val_str
            resources.append(obs)

    # ── Source 2: Outpatient measurement results (hosp_omr) ───────────────────
    # Covers non-ICU patients whose vitals are only in outpatient records.
    if not resources:
        omr = _read(subject_id, "hosp_omr")
        if omr is not None and not omr.empty:
            if vital_name:
                omr = omr[omr["result_name"].str.contains(vital_name, case=False, na=False)]
            if "chartdate" in omr.columns:
                omr = omr.sort_values("chartdate", ascending=False)
            omr = omr.head(_count)
            for idx, row in omr.iterrows():
                name = _str(row.get("result_name")) or "unknown"
                val  = _str(row.get("result_value"))
                date = _str(row.get("chartdate"))
                obs = {
                    "resourceType": "Observation",
                    "id": f"omr-{subject_id}-{idx}",
                    "status": "final",
                    "category": [{
                        "coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                     "code": "vital-signs", "display": "Vital Signs"}]
                    }],
                    "code": {
                        "coding": [{"system": "mimic-iv/omr", "display": name}],
                        "text": name,
                    },
                    "subject": {"reference": f"Patient/{subject_id}"},
                }
                if date:
                    obs["effectiveDateTime"] = date
                if val:
                    obs["valueString"] = val
                resources.append(obs)

    return resources


# ─────────────────────────────────────────────────────────────────────────────
# MedicationRequest (Prescriptions)
# ─────────────────────────────────────────────────────────────────────────────

def medication_requests_to_fhir(
    subject_id: int,
    hadm_id: Optional[int] = None,
    medication: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict]:
    df = _read(subject_id, "hosp_prescriptions")
    if df is None or df.empty:
        return []
    if hadm_id:
        df = df[df["hadm_id"] == hadm_id]
    if medication:
        df = df[df["drug"].str.contains(medication, case=False, na=False)]
    if "starttime" in df.columns:
        df = df.sort_values("starttime", ascending=False)

    resources = []
    for idx, row in df.iterrows():
        hadm_val   = _str(row.get("hadm_id"))
        drug       = _str(row.get("drug")) or "unknown"
        dose_val   = _str(row.get("dose_val_rx"))
        dose_unit  = _str(row.get("dose_unit_rx"))
        route      = _str(row.get("route"))
        start      = _str(row.get("starttime"))
        stop       = _str(row.get("stoptime"))
        drug_type  = _str(row.get("drug_type"))

        # Infer status from times
        fhir_status = "unknown"
        if stop and start:
            fhir_status = "completed"
        elif start:
            fhir_status = "active"
        if status and fhir_status != status:
            continue

        resource: dict = {
            "resourceType": "MedicationRequest",
            "id": f"rx-{subject_id}-{idx}",
            "status": fhir_status,
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [{"system": "mimic-iv/drug", "display": drug}],
                "text": drug,
            },
            "subject": {"reference": f"Patient/{subject_id}"},
        }
        if hadm_val:
            resource["encounter"] = {"reference": f"Encounter/encounter-{hadm_val}"}

        dosage: dict = {}
        text_parts = []
        if dose_val and dose_unit:
            text_parts.append(f"{dose_val}{dose_unit}")
            dosage["doseAndRate"] = [{
                "doseQuantity": {"value": _float(dose_val), "unit": dose_unit}
            }]
        if route:
            text_parts.append(route)
            dosage["route"] = {"text": route}
        if start or stop:
            timing: dict = {}
            if start:
                timing["boundsPeriod"] = {"start": start}
                if stop:
                    timing["boundsPeriod"]["end"] = stop
            dosage["timing"] = {"repeat": timing}
        if text_parts:
            dosage["text"] = " ".join(text_parts)
        if dosage:
            resource["dosageInstruction"] = [dosage]
        if drug_type:
            resource["_drug_type"] = drug_type
        resources.append(resource)
    return resources


# ─────────────────────────────────────────────────────────────────────────────
# MedicationAdministration (eMAR)
# ─────────────────────────────────────────────────────────────────────────────

def medication_administrations_to_fhir(
    subject_id: int,
    hadm_id: Optional[int] = None,
    medication: Optional[str] = None,
    _count: int = 100,
) -> list[dict]:
    df = _read(subject_id, "hosp_emar")
    if df is None or df.empty:
        return []
    if hadm_id:
        df = df[df["hadm_id"] == hadm_id]
    if medication:
        df = df[df["medication"].str.contains(medication, case=False, na=False)]
    if "charttime" in df.columns:
        df = df.sort_values("charttime", ascending=False)
    if not medication and not hadm_id:
        df = df.head(_count)

    resources = []
    for idx, row in df.iterrows():
        hadm_val = _str(row.get("hadm_id"))
        med      = _str(row.get("medication")) or "unknown"
        chart    = _str(row.get("charttime"))
        event    = _str(row.get("event_txt"))

        resource: dict = {
            "resourceType": "MedicationAdministration",
            "id": f"emar-{subject_id}-{idx}",
            "status": "completed",
            "medicationCodeableConcept": {"text": med},
            "subject": {"reference": f"Patient/{subject_id}"},
        }
        if hadm_val:
            resource["context"] = {"reference": f"Encounter/encounter-{hadm_val}"}
        if chart:
            resource["effectiveDateTime"] = chart
        if event:
            resource["_event_txt"] = event
        resources.append(resource)
    return resources


# ─────────────────────────────────────────────────────────────────────────────
# Procedure
# ─────────────────────────────────────────────────────────────────────────────

def procedures_to_fhir(
    subject_id: int,
    hadm_id: Optional[int] = None,
) -> list[dict]:
    df = _read(subject_id, "hosp_procedures_icd")
    if df is None or df.empty:
        return []
    if hadm_id:
        df = df[df["hadm_id"] == hadm_id]

    # Reuse ICD description map for procedures too (hosp covers ICD procedure codes)
    resources = []
    for idx, row in df.iterrows():
        icd_code = str(row.get("icd_code", "")).strip()
        icd_ver  = str(row.get("icd_version", "10"))
        hadm_val = _str(row.get("hadm_id"))
        sys_url  = (
            "http://hl7.org/fhir/sid/icd-10-pcs"
            if icd_ver == "10"
            else "http://hl7.org/fhir/sid/icd-9-cm/procedure"
        )
        resource: dict = {
            "resourceType": "Procedure",
            "id": f"proc-{subject_id}-{idx}",
            "status": "completed",
            "code": {
                "coding": [{"system": sys_url, "code": icd_code}],
                "text": icd_code,
            },
            "subject": {"reference": f"Patient/{subject_id}"},
        }
        if hadm_val:
            resource["encounter"] = {"reference": f"Encounter/encounter-{hadm_val}"}
        if "seq_num" in row and pd.notna(row.get("seq_num")):
            resource["_seq_num"] = int(row["seq_num"])
        resources.append(resource)
    return resources


# ─────────────────────────────────────────────────────────────────────────────
# DiagnosticReport (Radiology + Microbiology)
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_reports_to_fhir(
    subject_id: int,
    hadm_id: Optional[int] = None,
    report_type: Optional[str] = None,
    _count: int = 5,
) -> list[dict]:
    resources = []

    # ── Radiology reports (note_radiology) ────────────────────────────────────
    df = _read(subject_id, "note_radiology")
    if df is not None and not df.empty:
        if hadm_id and "hadm_id" in df.columns:
            df = df[df["hadm_id"] == hadm_id]
        if report_type:
            # Normalise common synonyms so queries like "X-RAY" find "CHEST (PORTABLE AP)"
            # and "CXR" finds chest radiographs, etc.
            _REPORT_ALIASES: dict[str, list[str]] = {
                "X-RAY":   ["CHEST", "PORTABLE", "AP", "PA", "RADIOGRAPH"],
                "CXR":     ["CHEST", "PORTABLE", "AP", "PA"],
                "CHEST":   ["CHEST"],
                "CT":      ["CT"],
                "MRI":     ["MRI", "MR "],
                "XRAY":    ["CHEST", "PORTABLE", "AP", "RADIOGRAPH"],
                "ULTRASOUND": ["US ", "ULTRASOUND", "SONO"],
                "ECHO":    ["ECHO"],
                "CULTURE": [],   # handled by microbiology branch
            }
            kw = report_type.upper().strip()
            search_terms = [kw] + _REPORT_ALIASES.get(kw, [])

            def _matches(row) -> bool:
                nt    = str(row.get("note_type", "")).upper()
                title = str(row.get("text", ""))[:200].upper()
                return any(t in nt or t in title for t in search_terms)

            mask = df.apply(_matches, axis=1)
            df = df[mask]
        if "charttime" in df.columns:
            df = df.sort_values("charttime", ascending=False)
        df = df.head(_count)

        for _, row in df.iterrows():
            note_id   = _str(row.get("note_id"))
            chart     = _str(row.get("charttime"))
            note_type = _str(row.get("note_type")) or "Radiology"
            text      = str(row.get("text", ""))[:3000]
            hadm_val  = _str(row.get("hadm_id"))
            # Extract exam title from the first non-empty line of text
            first_line = next((ln.strip() for ln in text.split('\n')
                               if ln.strip() and not ln.strip().startswith('INDICATION')), note_type)
            title_text = first_line[:120] if first_line else note_type

            resource: dict = {
                "resourceType": "DiagnosticReport",
                "id": f"dr-{note_id or subject_id}",
                "status": "final",
                "category": [{"coding": [{"system": "http://loinc.org",
                                           "code": "LP29684-5", "display": "Radiology"}]}],
                "code": {"text": title_text},
                "subject": {"reference": f"Patient/{subject_id}"},
                "presentedForm": [{"contentType": "text/plain", "data": text}],
            }
            if hadm_val:
                resource["encounter"] = {"reference": f"Encounter/encounter-{hadm_val}"}
            if chart:
                resource["effectiveDateTime"] = chart
            resources.append(resource)

    # ── Microbiology results (hosp_microbiologyevents) ────────────────────────
    # Include when report_type is None (all reports) or matches "microbiology"/"culture"
    _micro_keywords = {"micro", "culture", "blood", "urine", "sputum", "csf", "wound"}
    _include_micro = (report_type is None or
                      any(kw in report_type.lower() for kw in _micro_keywords))
    if _include_micro:
        mdf = _read(subject_id, "hosp_microbiologyevents")
        if mdf is not None and not mdf.empty:
            if hadm_id and "hadm_id" in mdf.columns:
                mdf = mdf[mdf["hadm_id"] == hadm_id]
            if report_type:
                kw = report_type.upper()
                def _micro_matches(row) -> bool:
                    spec = str(row.get("spec_type_desc", "")).upper()
                    org  = str(row.get("org_name", "")).upper()
                    return kw in spec or kw in org
                mdf = mdf[mdf.apply(_micro_matches, axis=1)]
            if "charttime" in mdf.columns:
                mdf = mdf.sort_values("charttime", ascending=False)
            # Group by specimen + charttime to collapse rows
            seen: set = set()
            for _, row in mdf.iterrows():
                spec_type = _str(row.get("spec_type_desc")) or "Specimen"
                chart     = _str(row.get("charttime"))
                org_name  = _str(row.get("org_name")) or "No growth / pending"
                ab_name   = _str(row.get("ab_name"))
                interp    = _str(row.get("interpretation"))
                key       = (spec_type, chart)
                if key in seen:
                    continue
                seen.add(key)
                if len(resources) >= _count * 2:
                    break
                detail = f"Specimen: {spec_type} | Organism: {org_name}"
                if ab_name:
                    detail += f" | Antibiotic: {ab_name} ({interp or '?'})"
                resource: dict = {
                    "resourceType": "DiagnosticReport",
                    "id": f"micro-{subject_id}-{chart or len(resources)}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://loinc.org",
                                               "code": "LP7820-4", "display": "Microbiology"}]}],
                    "code": {"text": spec_type},
                    "subject": {"reference": f"Patient/{subject_id}"},
                    "presentedForm": [{"contentType": "text/plain", "data": detail}],
                }
                if hadm_id:
                    resource["encounter"] = {"reference": f"Encounter/encounter-{hadm_id}"}
                if chart:
                    resource["effectiveDateTime"] = chart
                resources.append(resource)

    return resources


# ─────────────────────────────────────────────────────────────────────────────
# DocumentReference (Discharge summary + other notes)
# ─────────────────────────────────────────────────────────────────────────────

def document_references_to_fhir(
    subject_id: int,
    hadm_id: Optional[int] = None,
    type_code: Optional[str] = None,
    keyword: Optional[str] = None,
    _count: int = 5,
) -> list[dict]:
    resources = []
    tables = [
        ("note_discharge", "discharge-summary", "Discharge Summary"),
        ("note_radiology", "radiology",          "Radiology Report"),
    ]
    for table, code, display in tables:
        if type_code and code not in type_code.lower() and type_code.lower() not in code:
            continue
        df = _read(subject_id, table)
        if df is None or df.empty:
            continue
        if hadm_id and "hadm_id" in df.columns:
            df = df[df["hadm_id"] == hadm_id]
        if keyword and "text" in df.columns:
            df = df[df["text"].str.contains(keyword, case=False, na=False)]
        if "charttime" in df.columns:
            df = df.sort_values("charttime", ascending=False)
        df = df.head(_count)

        for _, row in df.iterrows():
            note_id  = _str(row.get("note_id"))
            chart    = _str(row.get("charttime"))
            text     = str(row.get("text", ""))[:4000]
            hadm_val = _str(row.get("hadm_id"))

            resource: dict = {
                "resourceType": "DocumentReference",
                "id": f"docref-{note_id or subject_id}",
                "status": "current",
                "type": {
                    "coding": [{"system": "http://loinc.org", "code": code, "display": display}],
                    "text": display,
                },
                "subject": {"reference": f"Patient/{subject_id}"},
                "content": [{"attachment": {"contentType": "text/plain", "data": text}}],
            }
            if hadm_val:
                resource["context"] = {"encounter": [{"reference": f"Encounter/encounter-{hadm_val}"}]}
            if chart:
                resource["date"] = chart
            resources.append(resource)
    return resources


# ─────────────────────────────────────────────────────────────────────────────
# AllergyIntolerance
# ─────────────────────────────────────────────────────────────────────────────

def care_plans_to_fhir(
    subject_id: int,
    hadm_id: Optional[int] = None,
    category: Optional[str] = None,
) -> list[dict]:
    """
    Extract FHIR CarePlan resources from MIMIC-IV discharge notes.

    MIMIC-IV has no dedicated care-plan table; care plan content is mined from
    structured sections of discharge summaries:
      - "Discharge Instructions" / "Discharge Condition" → discharge-planning
      - "Followup Instructions" / "Follow-Up" → follow-up care
      - Free-text "Plan:" paragraphs → treatment plan

    Args:
        subject_id: MIMIC-IV patient identifier.
        hadm_id:    Scope to one admission (optional).
        category:   Filter by category code: "discharge-planning" | "followup" |
                    "treatment".  None returns all.
    """
    import re

    df = _read(subject_id, "note_discharge")
    if df is None or df.empty:
        return []
    if hadm_id and "hadm_id" in df.columns:
        df = df[df["hadm_id"] == hadm_id]
    if df.empty:
        return []

    # Patterns: (category_code, display, regex capturing the section body)
    _SECTION_PATTERNS = [
        (
            "discharge-planning",
            "Discharge Instructions",
            re.compile(
                r"(?i)discharge\s+instructions?\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z]|\Z)",
                re.DOTALL,
            ),
        ),
        (
            "discharge-planning",
            "Discharge Condition",
            re.compile(
                r"(?i)discharge\s+condition\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z]|\Z)",
                re.DOTALL,
            ),
        ),
        (
            "followup",
            "Followup Instructions",
            re.compile(
                r"(?i)follow[\-\s]?up\s+instructions?\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z]|\Z)",
                re.DOTALL,
            ),
        ),
        (
            "treatment",
            "Treatment Plan",
            re.compile(
                r"(?i)\bplan\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Z]|\Z)",
                re.DOTALL,
            ),
        ),
    ]

    resources: list[dict] = []
    for _, row in df.head(3).iterrows():
        text     = str(row.get("text", ""))
        note_id  = _str(row.get("note_id"))
        chart    = _str(row.get("charttime"))
        hadm_val = _str(row.get("hadm_id"))

        for cat_code, cat_display, pattern in _SECTION_PATTERNS:
            if category and cat_code != category:
                continue
            m = pattern.search(text)
            if not m:
                continue
            body = m.group(1).strip()[:1000]
            if not body:
                continue

            resource: dict = {
                "resourceType": "CarePlan",
                "id": f"careplan-{note_id or subject_id}-{cat_code}",
                "status": "completed",
                "intent": "plan",
                "category": [{
                    "coding": [{
                        "system": "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
                        "code": cat_code,
                        "display": cat_display,
                    }],
                    "text": cat_display,
                }],
                "subject": {"reference": f"Patient/{subject_id}"},
                "note": [{"text": body}],
            }
            if hadm_val:
                resource["encounter"] = {"reference": f"Encounter/encounter-{hadm_val}"}
            if chart:
                resource["period"] = {"end": chart}

            resources.append(resource)

    return resources


def allergies_to_fhir(subject_id: int, hadm_id: Optional[int] = None) -> list[dict]:
    """
    MIMIC-IV doesn't have a dedicated allergy table; we approximate from
    the discharge note 'Allergies' section if available.
    Returns an empty list when no data exists (normal for MIMIC-IV).
    """
    df = _read(subject_id, "note_discharge")
    if df is None or df.empty:
        return []
    if hadm_id and "hadm_id" in df.columns:
        df = df[df["hadm_id"] == hadm_id]
    if df.empty:
        return []

    import re
    resources = []
    for _, row in df.head(1).iterrows():
        text = str(row.get("text", ""))
        m = re.search(r"(?i)allerg(?:ies|y)[:\s]*(.*?)(?=\n[A-Z]|\Z)", text, re.DOTALL)
        if not m:
            continue
        allergy_text = m.group(1).strip()[:500]
        if not allergy_text or allergy_text.lower() in ("nka", "nkda", "none", "no known allergies"):
            resource: dict = {
                "resourceType": "AllergyIntolerance",
                "id": f"allergy-{subject_id}-nka",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "verificationStatus": {"coding": [{"code": "confirmed"}]},
                "code": {"coding": [{"system": "http://snomed.info/sct",
                                      "code": "716186003",
                                      "display": "No known allergy"}]},
                "patient": {"reference": f"Patient/{subject_id}"},
            }
            resources.append(resource)
        else:
            # Return as a free-text note allergy entry
            resource = {
                "resourceType": "AllergyIntolerance",
                "id": f"allergy-{subject_id}-1",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "code": {"text": allergy_text},
                "patient": {"reference": f"Patient/{subject_id}"},
                "note": [{"text": allergy_text}],
            }
            resources.append(resource)
    return resources
