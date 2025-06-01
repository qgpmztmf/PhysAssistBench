"""
context_builder.py — Build a concise patient context summary for the LLM agents.

The context is provided to the user_agent and planner_agent to ground their
generation in real patient data. It does NOT include diagnosis labels (to avoid
leakage) but includes demographics, admission info, and data availability.
"""

import os
from typing import Optional

import pandas as pd

from physassistbench.paths import MIMIC_PATIENT_ROOT as DATA_ROOT


def _read(subject_id: int, table: str) -> Optional[pd.DataFrame]:
    fpath = os.path.join(DATA_ROOT, str(subject_id), f"{table}.csv.csv.gz")
    if not os.path.exists(fpath):
        return None
    try:
        return pd.read_csv(fpath, compression="gzip", low_memory=False)
    except Exception:
        return None


def _latest_event_date(subject_id: int) -> Optional[str]:
    """Return the most recent event date (YYYY-MM-DD) across the patient's
    time-series tables, or None if none are available.

    Used to anchor current_date when a patient has no hosp_admissions row: the
    per-patient MIMIC anchor_year marks the *start* of a shifted timeline, so
    anchoring there (anchor_year-03-15) leaves the patient's later data in the
    future and leaks it through the tools. Anchoring at the latest real event
    date instead keeps every observation at or before "now".
    """
    tables = {
        "hosp_labevents": "charttime",
        "icu_chartevents": "charttime",
        "hosp_emar": "charttime",
        "hosp_microbiologyevents": "charttime",
        "hosp_omr": "chartdate",
        "note_radiology": "charttime",
    }
    latest = None
    for table, col in tables.items():
        df = _read(subject_id, table)
        if df is None or df.empty or col not in df.columns:
            continue
        ts = pd.to_datetime(df[col], errors="coerce").dropna()
        if ts.empty:
            continue
        m = ts.max()
        if latest is None or m > latest:
            latest = m
    return latest.strftime("%Y-%m-%d") if latest is not None else None


def build_context(subject_id: int,
                  hadm_id: Optional[int] = None,
                  task_domain: str = "WorkflowQuery") -> dict:
    """
    Return a dict with patient context for LLM generation.
    Keys: demographics, admissions, available_data_summary, env_info, subject_id, hadm_id
    """
    ctx = {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "task_domain": task_domain,
    }

    # Demographics
    pat = _read(subject_id, "hosp_patients")
    demo_row = pat.iloc[0] if (pat is not None and not pat.empty) else None
    if demo_row is not None:
        ctx["demographics"] = {
            "gender": demo_row.get("gender", "Unknown"),
            "anchor_age": int(demo_row.get("anchor_age", 0)),
            "anchor_year_group": str(demo_row.get("anchor_year_group", "")),
            "dod": str(demo_row.get("dod", "")) if pd.notna(demo_row.get("dod")) else None,
        }
    else:
        ctx["demographics"] = {}

    # Most recent admission — loaded before env_info so admittime can anchor current_date
    adm = _read(subject_id, "hosp_admissions")
    if adm is not None and not adm.empty:
        if hadm_id:
            adm_row = adm[adm["hadm_id"] == hadm_id]
        else:
            adm_row = adm.sort_values("admittime", ascending=False)
        if not adm_row.empty:
            r = adm_row.iloc[0]
            ctx["admission"] = {
                "hadm_id": int(r.get("hadm_id", 0)),
                "admittime": str(r.get("admittime", "")),
                "dischtime": str(r.get("dischtime", "")),
                "admission_type": str(r.get("admission_type", "")),
                "admission_location": str(r.get("admission_location", "")),
                "discharge_location": str(r.get("discharge_location", "")),
            }
            if hadm_id is None:
                ctx["hadm_id"] = int(r.get("hadm_id", 0))

    # current_date: use dischtime of the selected admission so all observations
    # within that admission are visible, but future admissions are excluded.
    # Fall back to admittime, then anchor_year-03-15 if admission data is missing.
    current_date: str
    adm_info = ctx.get("admission", {})
    dischtime = adm_info.get("dischtime", "")
    admittime = adm_info.get("admittime", "")
    if dischtime and dischtime not in ("", "NaT", "None", "nan"):
        current_date = dischtime[:10]
    elif admittime and admittime not in ("", "NaT", "None", "nan"):
        current_date = admittime[:10]
    else:
        # No admission for this subject: anchor at the latest real event date so
        # the patient's own data is not left in the future (see _latest_event_date).
        latest = _latest_event_date(subject_id)
        if latest is not None:
            current_date = latest
        elif demo_row is not None:
            anchor_year = int(demo_row.get("anchor_year", 2180))
            current_date = f"{anchor_year}-03-15"
        else:
            current_date = "2180-03-15"

    ctx["current_date"] = current_date

    if demo_row is not None:
        ctx["env_info"] = (
            f"Current date: {current_date}. "
            f"Patient is a {int(demo_row.get('anchor_age', 0))}-year-old "
            f"{demo_row.get('gender', '')}"
        )
    else:
        ctx["env_info"] = f"Current date: {current_date}."

    # Available data summary
    available = []
    table_labels = {
        "hosp_labevents": "lab results",
        "hosp_diagnoses_icd": "ICD diagnoses",
        "hosp_prescriptions": "prescriptions",
        "hosp_emar": "medication administration records",
        "hosp_microbiologyevents": "microbiology cultures",
        "icu_icustays": "ICU stays",
        "icu_chartevents": "ICU vital signs",
        "icu_inputevents": "ICU fluid inputs",
        "ed_edstays": "ED visits",
        "ed_triage": "ED triage records",
        "note_discharge": "discharge summaries",
        "note_radiology": "radiology reports",
        "hosp_omr": "outpatient vital signs",
        "hosp_drgcodes": "DRG codes",
        "hosp_services": "service history",
    }
    for table, label in table_labels.items():
        df = _read(subject_id, table)
        if df is not None and not df.empty:
            available.append(f"{label} ({len(df)} records)")
    ctx["available_data"] = available

    # Build natural language context string for prompt injection
    demo = ctx.get("demographics", {})
    adm_info = ctx.get("admission", {})
    ctx["context_str"] = (
        f"Patient ID: {subject_id}\n"
        f"Demographics: {demo.get('anchor_age','?')}-year-old {demo.get('gender','')}\n"
        f"Admission ID: {adm_info.get('hadm_id', hadm_id or 'N/A')}\n"
        f"Admission type: {adm_info.get('admission_type','')}, "
        f"admitted: {adm_info.get('admittime','')[:10]}\n"
        f"Available data: {', '.join(available[:8])}\n"
        f"{ctx['env_info']}"
    )

    return ctx
