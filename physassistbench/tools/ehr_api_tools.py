"""
EHR API Tools — 28 implementations reading real MIMIC-IV .csv.gz files.

Each function returns a dict (JSON-serializable) matching the tool's schema.
subject_id is always int. hadm_id / stay_id are optional ints.
All functions handle missing-file gracefully (return empty dict with 'error' key).
"""

import gzip
import os
import re
from typing import Optional

import pandas as pd

from physassistbench.paths import MIMIC_PATIENT_ROOT as DATA_ROOT


# ─── helpers ────────────────────────────────────────────────────────────────

def _patient_dir(subject_id: int) -> str:
    return os.path.join(DATA_ROOT, str(subject_id))


def _read(subject_id: int, table: str) -> Optional[pd.DataFrame]:
    """Read a per-patient compressed CSV. table = e.g. 'hosp_labevents'."""
    pat_dir = _patient_dir(subject_id)
    # naming convention: {table}.csv.csv.gz  (double .csv is intentional in dataset)
    fname = f"{table}.csv.csv.gz"
    fpath = os.path.join(pat_dir, fname)
    if not os.path.exists(fpath):
        return None
    try:
        return pd.read_csv(fpath, compression="gzip", low_memory=False)
    except Exception:
        return None


def _df_to_records(df: pd.DataFrame, max_rows: int | None = None) -> list:
    """Convert DataFrame to JSON-serializable list of dicts.

    If max_rows is None (default), all rows are returned.
    If max_rows is set, rows are capped and the list ends with a sentinel
    dict ``{"_truncated": True, "total_rows": N}`` so callers can detect that
    the result was cut short.
    """
    total = len(df)
    if max_rows is not None and total > max_rows:
        df = df.head(max_rows)
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        records.append({"_truncated": True, "total_rows": total, "returned_rows": max_rows})
        return records
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def _error(msg: str) -> dict:
    return {"error": msg, "data": []}


# ─── GROUP 1: Hospital / General ────────────────────────────────────────────

def get_patient_info(subject_id: int) -> dict:
    """Return demographic information for a patient."""
    df = _read(subject_id, "hosp_patients")
    if df is None:
        return _error(f"No patient record found for subject_id={subject_id}")
    row = df.iloc[0].where(pd.notnull(df.iloc[0]), None).to_dict()
    return {"subject_id": subject_id, "demographics": row}


def get_admissions(subject_id: int) -> dict:
    """Return all hospital admissions for a patient."""
    df = _read(subject_id, "hosp_admissions")
    if df is None:
        return _error(f"No admissions found for subject_id={subject_id}")
    cols = ["hadm_id", "admittime", "dischtime", "deathtime",
            "admission_type", "admission_location", "discharge_location",
            "insurance", "language", "marital_status", "race",
            "hospital_expire_flag"]
    cols = [c for c in cols if c in df.columns]
    # Admissions list is small — return all; cap at 20 only for patients with many re-admissions
    return {"subject_id": subject_id, "admissions": _df_to_records(df[cols], max_rows=20)}


def get_admission_details(subject_id: int, hadm_id: int) -> dict:
    """Return details for a specific hospital admission."""
    df = _read(subject_id, "hosp_admissions")
    if df is None:
        return _error(f"No admissions found for subject_id={subject_id}")
    row = df[df["hadm_id"] == hadm_id]
    if row.empty:
        return _error(f"hadm_id={hadm_id} not found for subject_id={subject_id}")
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "admission": _df_to_records(row)[0]}


def get_diagnoses(subject_id: int, hadm_id: Optional[int] = None) -> dict:
    """Return ICD diagnosis codes for a patient (all admissions or a specific one)."""
    df = _read(subject_id, "hosp_diagnoses_icd")
    if df is None:
        return _error(f"No diagnoses found for subject_id={subject_id}")
    if hadm_id is not None:
        df = df[df["hadm_id"] == hadm_id]
    df = df.sort_values(["hadm_id", "seq_num"])
    # Diagnoses lists are naturally small; return all when scoped to one admission
    row_limit = None if hadm_id is not None else 100
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "diagnoses": _df_to_records(df, max_rows=row_limit)}


from physassistbench.paths import MIMIC_REF_ROOT as _REF_ROOT_FOR_LABITEMS
_D_LABITEMS_PATH = os.environ.get(
    "MIMIC_D_LABITEMS",
    os.path.join(_REF_ROOT_FOR_LABITEMS, "hosp", "d_labitems.csv.gz"),
)
_LAB_LABEL_MAP: dict[int, str] | None = None


def _get_lab_label_map() -> dict[int, str]:
    """Load itemid → label mapping from d_labitems (cached)."""
    global _LAB_LABEL_MAP
    if _LAB_LABEL_MAP is None:
        try:
            df = pd.read_csv(_D_LABITEMS_PATH, compression="gzip", low_memory=False)
            _LAB_LABEL_MAP = dict(zip(df["itemid"].astype(int), df["label"].astype(str)))
        except Exception:
            _LAB_LABEL_MAP = {}
    return _LAB_LABEL_MAP


def get_lab_results(subject_id: int,
                    hadm_id: Optional[int] = None,
                    item_name: Optional[str] = None,
                    abnormal_only: bool = False) -> dict:
    """Return lab results. Optionally filter by admission, test name, or abnormal flag."""
    df = _read(subject_id, "hosp_labevents")
    if df is None:
        return _error(f"No lab events found for subject_id={subject_id}")
    if hadm_id is not None:
        df = df[df["hadm_id"] == hadm_id]
    if item_name:
        lab_map = _get_lab_label_map()
        # Match itemids whose label contains item_name (case-insensitive)
        matched_itemids = {
            iid for iid, label in lab_map.items()
            if item_name.lower() in str(label).lower()
        }
        if matched_itemids:
            mask = df["itemid"].isin(matched_itemids)
            df = df[mask] if mask.any() else df
        else:
            # Fallback: fuzzy match on comments
            mask = df["comments"].str.contains(item_name, case=False, na=False)
            df = df[mask] if mask.any() else df
    if abnormal_only:
        df = df[df["flag"].notna() & (df["flag"].str.upper().isin(["ABNORMAL", "DELTA"]))]
    cols = ["labevent_id", "hadm_id", "charttime", "itemid", "value",
            "valuenum", "valueuom", "ref_range_lower", "ref_range_upper",
            "flag", "priority", "comments"]
    cols = [c for c in cols if c in df.columns]
    df = df.sort_values("charttime", ascending=False) if "charttime" in df.columns else df
    # Add label column for readability
    lab_map = _get_lab_label_map()
    # When filtering by item_name: return ALL matching rows (typically <20 per admission).
    # When no item filter: cap at 50 most-recent rows to avoid overwhelming the context
    # window, and include a _truncated sentinel if data was cut.
    row_limit = None if item_name else 50
    result_records = _df_to_records(df[cols], max_rows=row_limit)
    for rec in result_records:
        iid = rec.get("itemid")
        if iid is not None and not rec.get("_truncated"):
            rec["item_label"] = lab_map.get(int(iid), f"itemid:{iid}")
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "lab_results": result_records}


def get_lab_trends(subject_id: int, item_name: str, n_recent: int = 5) -> dict:
    """Return the N most recent lab results for a specific test."""
    df = _read(subject_id, "hosp_labevents")
    if df is None:
        return _error(f"No lab events found for subject_id={subject_id}")
    # Match by d_labitems label first
    lab_map = _get_lab_label_map()
    matched_itemids = {
        iid for iid, label in lab_map.items()
        if item_name.lower() in str(label).lower()
    }
    if matched_itemids:
        mask = df["itemid"].isin(matched_itemids)
        df_match = df[mask]
    else:
        mask = df["comments"].str.contains(item_name, case=False, na=False)
        df_match = df[mask]
    if df_match.empty:
        return _error(f"No lab results found for item_name='{item_name}' for subject_id={subject_id}")
    df_match = df_match.sort_values("charttime", ascending=False).head(n_recent)
    cols = ["charttime", "hadm_id", "itemid", "value", "valuenum",
            "valueuom", "ref_range_lower", "ref_range_upper", "flag"]
    cols = [c for c in cols if c in df_match.columns]
    result_records = _df_to_records(df_match[cols])
    for rec in result_records:
        iid = rec.get("itemid")
        if iid is not None:
            rec["item_label"] = lab_map.get(int(iid), f"itemid:{iid}")
    return {"subject_id": subject_id, "item_name": item_name,
            "n_recent": n_recent, "trends": result_records}


def get_microbiology_results(subject_id: int,
                              hadm_id: Optional[int] = None) -> dict:
    """Return microbiology culture results."""
    df = _read(subject_id, "hosp_microbiologyevents")
    if df is None:
        return _error(f"No microbiology events for subject_id={subject_id}")
    if hadm_id is not None:
        df = df[df["hadm_id"] == hadm_id]
    cols = ["microevent_id", "hadm_id", "chartdate", "charttime",
            "spec_type_desc", "test_name", "org_name", "ab_name",
            "interpretation", "comments"]
    cols = [c for c in cols if c in df.columns]
    # Microbiology results are naturally small; return all when hadm_id-scoped
    row_limit = None if hadm_id is not None else 50
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "microbiology": _df_to_records(df[cols], max_rows=row_limit)}


def get_prescriptions(subject_id: int,
                      hadm_id: Optional[int] = None) -> dict:
    """Return medication prescriptions."""
    df = _read(subject_id, "hosp_prescriptions")
    if df is None:
        return _error(f"No prescriptions found for subject_id={subject_id}")
    if hadm_id is not None:
        df = df[df["hadm_id"] == hadm_id]
    cols = ["hadm_id", "starttime", "stoptime", "drug_type", "drug",
            "dose_val_rx", "dose_unit_rx", "route", "doses_per_24_hrs"]
    cols = [c for c in cols if c in df.columns]
    if "starttime" in df.columns:
        df = df.sort_values("starttime", ascending=False)
    # When scoped to one admission return all prescriptions; otherwise cap at 100
    row_limit = None if hadm_id is not None else 100
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "prescriptions": _df_to_records(df[cols], max_rows=row_limit)}


def get_medication_administration(subject_id: int,
                                  hadm_id: Optional[int] = None,
                                  medication: Optional[str] = None) -> dict:
    """Return medication administration records (eMAR)."""
    df = _read(subject_id, "hosp_emar")
    if df is None:
        return _error(f"No eMAR found for subject_id={subject_id}")
    if hadm_id is not None:
        df = df[df["hadm_id"] == hadm_id]
    if medication:
        df = df[df["medication"].str.contains(medication, case=False, na=False)]
    cols = ["hadm_id", "emar_seq", "charttime", "medication",
            "event_txt", "scheduletime"]
    cols = [c for c in cols if c in df.columns]
    if "charttime" in df.columns:
        df = df.sort_values("charttime", ascending=False)
    # When filtering by medication or hadm_id return all; otherwise cap at 100
    row_limit = None if (medication or hadm_id is not None) else 100
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "medication_administration": _df_to_records(df[cols], max_rows=row_limit)}


def get_procedures(subject_id: int,
                   hadm_id: Optional[int] = None) -> dict:
    """Return ICD procedure codes."""
    df = _read(subject_id, "hosp_procedures_icd")
    if df is None:
        return _error(f"No procedures found for subject_id={subject_id}")
    if hadm_id is not None:
        df = df[df["hadm_id"] == hadm_id]
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "procedures": _df_to_records(df)}


def get_drg_info(subject_id: int,
                 hadm_id: Optional[int] = None) -> dict:
    """Return DRG codes with severity and mortality weights."""
    df = _read(subject_id, "hosp_drgcodes")
    if df is None:
        return _error(f"No DRG codes found for subject_id={subject_id}")
    if hadm_id is not None:
        df = df[df["hadm_id"] == hadm_id]
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "drg_codes": _df_to_records(df)}


def get_service_history(subject_id: int,
                        hadm_id: Optional[int] = None) -> dict:
    """Return hospital service transfer history."""
    df = _read(subject_id, "hosp_services")
    if df is None:
        return _error(f"No service history for subject_id={subject_id}")
    if hadm_id is not None:
        df = df[df["hadm_id"] == hadm_id]
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "services": _df_to_records(df)}


# ─── GROUP 2: ICU ────────────────────────────────────────────────────────────

def get_icu_stays(subject_id: int) -> dict:
    """Return all ICU stay records."""
    df = _read(subject_id, "icu_icustays")
    if df is None:
        return _error(f"No ICU stays for subject_id={subject_id}")
    return {"subject_id": subject_id, "icu_stays": _df_to_records(df)}


def get_icu_vitals(subject_id: int,
                   stay_id: Optional[int] = None,
                   vital_name: Optional[str] = None) -> dict:
    """Return ICU chart events (vitals, measurements). Optionally filter by stay or vital name."""
    df = _read(subject_id, "icu_chartevents")
    if df is None:
        return _error(f"No ICU chart events for subject_id={subject_id}")
    if stay_id is not None:
        df = df[df["stay_id"] == stay_id]
    if vital_name:
        df = df[df["itemid"].astype(str).str.contains(vital_name, case=False, na=False)]
    cols = ["stay_id", "charttime", "itemid", "value", "valuenum", "valueuom", "warning"]
    cols = [c for c in cols if c in df.columns]
    df = df.sort_values("charttime", ascending=False) if "charttime" in df.columns else df
    # ICU chartevents can be very large (p90=25k rows); cap at 100 most-recent measurements.
    # When vital_name filter narrows the results, still cap at 100 to limit context size.
    return {"subject_id": subject_id, "stay_id": stay_id,
            "icu_vitals": _df_to_records(df[cols], max_rows=100)}


def get_icu_fluids_in(subject_id: int,
                      stay_id: Optional[int] = None) -> dict:
    """Return ICU fluid/medication inputs."""
    df = _read(subject_id, "icu_inputevents")
    if df is None:
        return _error(f"No ICU input events for subject_id={subject_id}")
    if stay_id is not None:
        df = df[df["stay_id"] == stay_id]
    cols = ["stay_id", "starttime", "endtime", "itemid", "amount",
            "amountuom", "rate", "rateuom", "ordercategoryname", "statusdescription"]
    cols = [c for c in cols if c in df.columns]
    return {"subject_id": subject_id, "stay_id": stay_id,
            "icu_inputs": _df_to_records(df[cols])}


def get_icu_output(subject_id: int,
                   stay_id: Optional[int] = None) -> dict:
    """Return ICU fluid outputs."""
    df = _read(subject_id, "icu_outputevents")
    if df is None:
        return _error(f"No ICU output events for subject_id={subject_id}")
    if stay_id is not None:
        df = df[df["stay_id"] == stay_id]
    cols = ["stay_id", "charttime", "itemid", "value", "valueuom"]
    cols = [c for c in cols if c in df.columns]
    return {"subject_id": subject_id, "stay_id": stay_id,
            "icu_outputs": _df_to_records(df[cols])}


# ─── GROUP 3: Emergency Department ─────────────────────────────────────────

def get_ed_visits(subject_id: int) -> dict:
    """Return all ED visit records."""
    df = _read(subject_id, "ed_edstays")
    if df is None:
        return _error(f"No ED visits for subject_id={subject_id}")
    return {"subject_id": subject_id, "ed_visits": _df_to_records(df)}


def get_ed_triage(subject_id: int,
                  stay_id: Optional[int] = None) -> dict:
    """Return ED triage assessment including vitals and chief complaint."""
    df = _read(subject_id, "ed_triage")
    if df is None:
        return _error(f"No ED triage for subject_id={subject_id}")
    if stay_id is not None:
        df = df[df["stay_id"] == stay_id]
    return {"subject_id": subject_id, "stay_id": stay_id,
            "triage": _df_to_records(df)}


def get_ed_vital_signs(subject_id: int,
                       stay_id: Optional[int] = None) -> dict:
    """Return ED vital sign time series."""
    df = _read(subject_id, "ed_vitalsign")
    if df is None:
        return _error(f"No ED vitals for subject_id={subject_id}")
    if stay_id is not None:
        df = df[df["stay_id"] == stay_id]
    df = df.sort_values("charttime") if "charttime" in df.columns else df
    return {"subject_id": subject_id, "stay_id": stay_id,
            "ed_vitals": _df_to_records(df)}


def get_ed_diagnoses(subject_id: int,
                     stay_id: Optional[int] = None) -> dict:
    """Return ED diagnoses (ICD codes with titles)."""
    df = _read(subject_id, "ed_diagnosis")
    if df is None:
        return _error(f"No ED diagnoses for subject_id={subject_id}")
    if stay_id is not None:
        df = df[df["stay_id"] == stay_id]
    return {"subject_id": subject_id, "stay_id": stay_id,
            "ed_diagnoses": _df_to_records(df)}


def get_ed_medications(subject_id: int,
                       stay_id: Optional[int] = None) -> dict:
    """Return ED medication reconciliation and pyxis dispensing records."""
    med = _read(subject_id, "ed_medrecon")
    pyx = _read(subject_id, "ed_pyxis")
    result = {}
    for name, df in [("medrecon", med), ("pyxis", pyx)]:
        if df is None:
            continue
        if stay_id is not None:
            df = df[df["stay_id"] == stay_id]
        result[name] = _df_to_records(df)
    if not result:
        return _error(f"No ED medication data for subject_id={subject_id}")
    return {"subject_id": subject_id, "stay_id": stay_id, **result}


# ─── GROUP 4: Clinical Notes ────────────────────────────────────────────────

def _read_note(subject_id: int, table: str,
               hadm_id: Optional[int] = None) -> Optional[pd.DataFrame]:
    df = _read(subject_id, table)
    if df is None:
        return None
    if hadm_id is not None and "hadm_id" in df.columns:
        df = df[df["hadm_id"] == hadm_id]
    return df


def get_discharge_summary(subject_id: int,
                          hadm_id: Optional[int] = None) -> dict:
    """Return the full discharge summary note text."""
    df = _read_note(subject_id, "note_discharge", hadm_id)
    if df is None or df.empty:
        return _error(f"No discharge summary for subject_id={subject_id}, hadm_id={hadm_id}")
    row = df.sort_values("charttime").iloc[-1]
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "note_id": row.get("note_id"),
            "charttime": row.get("charttime"),
            "text": str(row.get("text", ""))[:4000]}  # truncate to 4k chars


def get_discharge_section(subject_id: int,
                          hadm_id: Optional[int] = None,
                          section_name: str = "Assessment and Plan") -> dict:
    """Return a specific section of the discharge summary."""
    full = get_discharge_summary(subject_id, hadm_id)
    if "error" in full:
        return full
    text = full.get("text", "")
    # Extract section by looking for section header
    pattern = re.compile(
        rf"(?i)({re.escape(section_name)}[:\s]*)(.*?)(?=\n[A-Z][A-Z ]+:|$)",
        re.DOTALL
    )
    m = pattern.search(text)
    section_text = m.group(2).strip() if m else f"Section '{section_name}' not found."
    return {"subject_id": subject_id, "hadm_id": hadm_id,
            "section": section_name,
            "content": section_text[:2000]}


def get_radiology_report(subject_id: int,
                         hadm_id: Optional[int] = None,
                         report_type: Optional[str] = None) -> dict:
    """Return radiology report text."""
    df = _read_note(subject_id, "note_radiology", hadm_id)
    if df is None or df.empty:
        return _error(f"No radiology reports for subject_id={subject_id}")
    if report_type and "note_type" in df.columns:
        df = df[df["note_type"].str.contains(report_type, case=False, na=False)]
    if df.empty:
        return _error(f"No radiology reports matching type={report_type}")
    df = df.sort_values("charttime") if "charttime" in df.columns else df
    records = []
    for _, row in df.head(3).iterrows():
        records.append({
            "note_id": row.get("note_id"),
            "charttime": str(row.get("charttime")),
            "note_type": row.get("note_type"),
            "text": str(row.get("text", ""))[:2000],
        })
    return {"subject_id": subject_id, "hadm_id": hadm_id, "reports": records}


def search_notes(subject_id: int, keyword: str) -> dict:
    """Search across all clinical notes for a keyword."""
    results = []
    for table in ["note_discharge", "note_radiology"]:
        df = _read(subject_id, table)
        if df is None:
            continue
        if "text" not in df.columns:
            continue
        mask = df["text"].str.contains(keyword, case=False, na=False)
        hits = df[mask]
        for _, row in hits.head(3).iterrows():
            snippet_start = str(row["text"]).lower().find(keyword.lower())
            snippet = str(row["text"])[max(0, snippet_start - 100):snippet_start + 200]
            results.append({
                "table": table,
                "note_id": row.get("note_id"),
                "charttime": str(row.get("charttime", "")),
                "snippet": snippet,
            })
    if not results:
        return _error(f"Keyword '{keyword}' not found in notes for subject_id={subject_id}")
    return {"subject_id": subject_id, "keyword": keyword, "matches": results}


# ─── GROUP 5: Utilities ─────────────────────────────────────────────────────

def get_vital_signs_outpatient(subject_id: int) -> dict:
    """Return outpatient vital signs and measurements (OMR table)."""
    df = _read(subject_id, "hosp_omr")
    if df is None:
        return _error(f"No outpatient vitals for subject_id={subject_id}")
    df = df.sort_values("chartdate") if "chartdate" in df.columns else df
    return {"subject_id": subject_id, "outpatient_vitals": _df_to_records(df)}


def get_patient_timeline(subject_id: int,
                         event_type: Optional[str] = None) -> dict:
    """Return a chronological timeline of clinical events across all tables."""
    events = []

    def _add(df, ts_col, etype, desc_col=None):
        if df is None:
            return
        for _, r in df.iterrows():
            ts = r.get(ts_col, "")
            desc = str(r.get(desc_col, "")) if desc_col else ""
            events.append({"timestamp": str(ts), "type": etype, "description": desc})

    _add(_read(subject_id, "hosp_admissions"), "admittime", "admission", "admission_type")
    _add(_read(subject_id, "hosp_admissions"), "dischtime", "discharge", "discharge_location")
    _add(_read(subject_id, "icu_icustays"), "intime", "icu_admit", "first_careunit")
    _add(_read(subject_id, "ed_edstays"), "intime", "ed_visit", "disposition")
    _add(_read(subject_id, "hosp_diagnoses_icd"), "hadm_id", "diagnosis", "icd_code")

    # filter by type if requested
    if event_type:
        events = [e for e in events if event_type.lower() in e["type"].lower()]

    events.sort(key=lambda e: e["timestamp"])
    return {"subject_id": subject_id, "event_type": event_type,
            "timeline": events[:50]}


def prepare_to_answer() -> dict:
    """Signal that the planner has finished tool calls and is ready to answer.
    This is a special sentinel tool identical in purpose to WildToolBench's prepare_to_answer."""
    return {"status": "ready"}
