"""
patient_selector.py — Select eligible patients per task domain and split.

Eligibility: patient must have ALL required tables for the given task.
Split: based on anchor_year_group from hosp_patients.
"""

import os
import random
from typing import List, Optional, Tuple

import pandas as pd

from physassistbench.paths import MIMIC_PATIENT_ROOT as DATA_ROOT

SPLIT_YEAR_GROUPS = {
    "train": ["2008 - 2010", "2011 - 2013", "2014 - 2016"],
    "val":   ["2017 - 2019"],
    "test":  ["2020 - 2022"],
}

TASK_REQUIRED_TABLES = {
    "LabInterp":     ["hosp_labevents", "hosp_patients"],
    "MedRecon":      ["hosp_prescriptions", "hosp_emar", "hosp_patients"],
    "DiagCode":      ["hosp_diagnoses_icd", "hosp_labevents", "hosp_patients"],
    "WorkflowQuery": ["hosp_admissions", "hosp_patients"],
    "ICUReasoning":  ["icu_icustays", "icu_chartevents", "hosp_patients"],
    "DischargePlan": ["note_discharge", "hosp_admissions", "hosp_patients"],
}


def _has_table(patient_dir: str, table: str) -> bool:
    return os.path.exists(os.path.join(patient_dir, f"{table}.csv.csv.gz"))


def _read_patient_year_group(subject_id: str) -> Optional[str]:
    fpath = os.path.join(DATA_ROOT, subject_id, "hosp_patients.csv.csv.gz")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, compression="gzip", usecols=["anchor_year_group"])
        if df.empty:
            return None
        return str(df["anchor_year_group"].iloc[0])
    except Exception:
        return None


def _get_primary_hadm_id(subject_id: str) -> Optional[int]:
    """Return the most recent hadm_id for a patient."""
    fpath = os.path.join(DATA_ROOT, subject_id, "hosp_admissions.csv.csv.gz")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, compression="gzip",
                         usecols=["hadm_id", "admittime"])
        if df.empty:
            return None
        df = df.sort_values("admittime", ascending=False)
        return int(df["hadm_id"].iloc[0])
    except Exception:
        return None


def select_patients(
    task_domain: str,
    split: str = "test",  # kept for backward compatibility, no longer used
    n: int = 15,
    seed: int = 42,
) -> List[Tuple[int, Optional[int]]]:
    """
    Return up to n (subject_id, hadm_id) tuples eligible for the given task.
    All patients are considered regardless of anchor_year_group (no train/val/test split).
    The `split` parameter is retained for backward compatibility but ignored.
    """
    required_tables = TASK_REQUIRED_TABLES.get(task_domain, [])

    all_patients = sorted(os.listdir(DATA_ROOT))
    random.seed(seed)
    random.shuffle(all_patients)

    candidates = []
    for pid in all_patients:
        if len(candidates) >= n * 5:  # over-sample then filter
            break
        pat_dir = os.path.join(DATA_ROOT, pid)
        if not os.path.isdir(pat_dir):
            continue
        # check required tables
        if not all(_has_table(pat_dir, t) for t in required_tables):
            continue
        hadm_id = _get_primary_hadm_id(pid)
        # For ICU tasks, we also need a stay_id
        if task_domain == "ICUReasoning":
            icu_path = os.path.join(pat_dir, "icu_icustays.csv.csv.gz")
            try:
                icu_df = pd.read_csv(icu_path, compression="gzip", usecols=["stay_id"])
                if icu_df.empty:
                    continue
            except Exception:
                continue
        candidates.append((int(pid), hadm_id))

    random.shuffle(candidates)
    return candidates[:n]
