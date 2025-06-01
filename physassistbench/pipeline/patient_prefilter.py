"""
physassistbench/pipeline/patient_prefilter.py — Two-stage patient eligibility filter.

Stage 1 (instant, CSV-based):
    Uses pre-computed patient_info_stats.csv byte-size columns as data-volume
    proxies. Eliminates patients with clearly insufficient records without
    reading any patient file.

Stage 2 (content-based, ~0.3 s/patient):
    Reads gzipped patient CSV files to verify specific lab itemids, drug-lab
    pairings (med_safety), and antibiotic coverage (treatment_response).

Public API
----------
load_stats_index(csv_path)                             -> dict[int, dict]
stage1_qualify(stats_row, scenario, difficulty)        -> bool
stage2_qualify(subject_id, hadm_id, scenario, diff)   -> bool
prefilter_pool(candidates, scenario, diff, n, ...)     -> list[tuple]
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# ── Data paths ────────────────────────────────────────────────────────────────
from physassistbench.paths import MIMIC_PATIENT_ROOT as DATA_ROOT

STATS_CSV = os.environ.get("MIMIC_PATIENT_STATS_CSV", "")

# ── Lab item-ID sets (MIMIC-IV d_labitems) ────────────────────────────────────
_CREATININE  = {50912, 51081, 51977, 52024, 52546}
_POTASSIUM   = {50822, 50833, 50971, 52452, 52610}
_GLUCOSE     = {50809, 50931, 51981, 52027, 52569}
_BUN         = {51842}
_ALT         = {50861}
_AST         = {50878}
_WBC         = {51301, 51755, 51756}
_HEMOGLOBIN  = {51222}
_PLATELET    = {51265}
_NEUTROPHILS = {51256, 52075, 53159, 53164}
_INR         = {51237, 51675}
_LACTATE     = {50813, 52442, 53154}
_HBA1C       = {50852, 51631}
_SODIUM      = {50824, 50983, 52455, 52623}
_BICARBONATE = {50803, 50882, 52039}

_MONITORING_LABS   = _CREATININE | _POTASSIUM | _GLUCOSE | _BUN | _INR | _HBA1C
_INFECTION_MARKERS = _WBC | _NEUTROPHILS | _LACTATE

# ── Multi-system lab groups (diagnostic_workup) ───────────────────────────────
# Each group represents one organ system. Patients must have labs spanning
# multiple groups to qualify for differential-diagnosis workup scenarios.
_ALP         = {50863}
_BILIRUBIN   = {50885, 50883}
_ALBUMIN     = {50862}
_LDH         = {50954}
_LYMPHOCYTES = {51244, 52769}
_CALCIUM     = {50893}
_PHOSPHATE   = {50970}
_TROPONIN    = {51002, 52642}
_CK          = {50911}
_BNP         = {51214}
_CRP         = {50889}
_PROCALCITONIN = {52189}
_FERRITIN    = {50924}
_PTT         = {51275}
_PT          = {51274}
_FIBRINOGEN  = {50884}

_LAB_SYSTEMS: dict[str, set[int]] = {
    "hematologic":  _WBC | _PLATELET | _HEMOGLOBIN | _NEUTROPHILS | _LYMPHOCYTES | _LDH,
    "renal":        _CREATININE | _BUN,
    "hepatic":      _ALT | _AST | _ALP | _BILIRUBIN | _ALBUMIN,
    "metabolic":    _GLUCOSE | _HBA1C | _SODIUM | _POTASSIUM | _BICARBONATE | _CALCIUM | _PHOSPHATE,
    "coagulation":  _INR | _PTT | _PT | _FIBRINOGEN,
    "inflammatory": _CRP | _FERRITIN | _PROCALCITONIN | _LDH,
    "cardiac":      _TROPONIN | _CK | _BNP,
}

# Biomarker groups used for lab_trend trend-count checks
_LAB_GROUPS = [
    _CREATININE, _POTASSIUM, _GLUCOSE, _BUN, _ALT, _AST,
    _WBC, _PLATELET, _HEMOGLOBIN, _INR, _LACTATE, _NEUTROPHILS,
    _SODIUM, _BICARBONATE, _HBA1C,
]

# ── Scoring-system lab requirements ──────────────────────────────────────────
# Defined after all individual lab sets are available.

# Child-Pugh / MELD: bilirubin + albumin + PT/INR (hepatic scoring)
_CHILD_PUGH_LABS = _BILIRUBIN | _ALBUMIN | _PT | _INR

# SOFA: 3 of 6 organ components measurable from labevents
# (PaO₂ and GCS require chartevents — excluded here)
_SOFA_LABS = _PLATELET | _BILIRUBIN | _CREATININE

# CURB-65: BUN + creatinine (age from admissions, BP/RR from chartevents)
_CURB65_LABS = _BUN | _CREATININE

# ICD code prefixes for scoring-relevant diagnoses
_ICD_AFIB      = {"42731", "I48"}
_ICD_CIRRHOSIS = {"5712", "5715", "5716",
                  "K740", "K741", "K742", "K743", "K744", "K745", "K746"}
_ICD_PNEUMONIA = {"480", "481", "482", "483", "484", "485", "486",
                  "J09", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18"}
_ICD_SEPSIS    = {"99591", "99592", "A40", "A41", "R6520", "R6521"}


def _has_icd(dx_df, icd_prefixes: set[str]) -> bool:
    """Return True if any ICD code starts with one of the given prefixes."""
    if dx_df is None or dx_df.empty:
        return False
    codes = dx_df["icd_code"].dropna().astype(str)
    return any(code.startswith(tuple(icd_prefixes)) for code in codes)


# ── Drug keyword sets (med_safety) ────────────────────────────────────────────
# 25 core monitoring drugs across 5 drug-lab categories
_DRUG_CATEGORIES: dict[str, set[str]] = {
    "renal_nephrotoxic": {
        "vancomycin", "gentamicin", "amikacin", "tobramycin",
        "ibuprofen", "naproxen", "tacrolimus", "cyclosporine",
        "digoxin", "methotrexate",
    },
    "renal_acei_arb": {
        "lisinopril", "enalapril", "ramipril", "captopril",
        "losartan", "valsartan", "irbesartan", "candesartan",
    },
    "glucose_dm": {
        "metformin", "insulin", "glipizide", "glyburide", "glimepiride",
        "empagliflozin", "dapagliflozin", "liraglutide", "sitagliptin",
    },
    "electrolyte_diuretic": {
        "spironolactone", "furosemide", "hydrochlorothiazide",
        "eplerenone", "triamterene", "amiloride", "bumetanide",
    },
    "coagulation": {
        "warfarin",
    },
}

# Required lab itemids for each drug category to form a valid monitoring pair
_CATEGORY_LABS: dict[str, set[int]] = {
    "renal_nephrotoxic":   _CREATININE | _BUN,
    "renal_acei_arb":      _CREATININE | _POTASSIUM,
    "glucose_dm":          _GLUCOSE | _HBA1C,
    "electrolyte_diuretic": _POTASSIUM,
    "coagulation":         _INR,
}

# ── Antibiotic classes (treatment_response) ───────────────────────────────────
_AB_CLASSES: dict[str, set[str]] = {
    "glycopeptide":    {"vancomycin"},
    "beta_lactam":     {"piperacillin", "meropenem", "cefepime", "ceftriaxone",
                        "cefazolin", "ampicillin", "imipenem", "ertapenem"},
    "fluoroquinolone": {"levofloxacin", "ciprofloxacin", "moxifloxacin"},
    "macrolide":       {"azithromycin", "clarithromycin"},
    "nitroimidazole":  {"metronidazole"},
    "aminoglycoside":  {"gentamicin", "tobramycin", "amikacin"},
    "antifungal":      {"micafungin", "fluconazole", "caspofungin"},
    "oxazolidinone":   {"linezolid"},
    "lipopeptide":     {"daptomycin"},
}
_ALL_AB = set().union(*_AB_CLASSES.values())

# ── Stage 1: byte-size thresholds ─────────────────────────────────────────────
# Each entry is a list of (column_name, min_value) — ALL must be satisfied.
_STAGE1: dict[tuple[str, int], list[tuple[str, int]]] = {
    # diagnostic_workup: requires multi-system labs + radiology + diagnoses
    # Stage-1 thresholds calibrated so that patients passing Stage-1 have
    # a >60% chance of passing the multi-system Stage-2 check.
    ("diagnostic_workup", 1): [
        ("bytes_hosp_labevents.csv",    1_000),  # ≥ basic metabolic + CBC
        ("bytes_note_radiology.csv",      500),  # ≥ 1 short radiology report
        ("rows_hosp_diagnoses_icd.csv",     3),  # ≥ 3 ICD codes
    ],
    ("diagnostic_workup", 2): [
        ("bytes_hosp_labevents.csv",    4_000),  # richer lab panel
        ("bytes_note_radiology.csv",    2_000),  # ≥ 2 reports
        ("rows_hosp_diagnoses_icd.csv",     5),
    ],
    ("diagnostic_workup", 3): [
        ("bytes_hosp_labevents.csv",   10_000),  # extensive lab workup
        ("bytes_note_radiology.csv",    5_000),  # ≥ 3 substantial reports
        ("rows_hosp_diagnoses_icd.csv",     8),
    ],

    ("lab_trend", 1): [("bytes_hosp_labevents.csv", 500)],
    ("lab_trend", 2): [("bytes_hosp_labevents.csv", 3_000)],
    ("lab_trend", 3): [("bytes_hosp_labevents.csv", 8_000)],

    ("med_safety", 1): [("bytes_hosp_labevents.csv",    500),
                        ("bytes_hosp_prescriptions.csv", 300)],
    ("med_safety", 2): [("bytes_hosp_labevents.csv",    2_000),
                        ("bytes_hosp_prescriptions.csv", 1_000)],
    ("med_safety", 3): [("bytes_hosp_labevents.csv",    5_000),
                        ("bytes_hosp_prescriptions.csv", 3_000)],

    ("treatment_response", 1): [("bytes_hosp_labevents.csv", 500),
                                ("bytes_hosp_emar.csv",       300)],
    ("treatment_response", 2): [("bytes_hosp_labevents.csv", 3_000),
                                ("bytes_hosp_emar.csv",       800)],
    ("treatment_response", 3): [("bytes_hosp_labevents.csv", 6_000),
                                ("bytes_hosp_emar.csv",       1_500),
                                ("rows_hosp_diagnoses_icd.csv", 5)],

    # discharge_planning: true difficulty stratification by data volume
    # L1: ≥10 drugs, ≥5 dx, labs present
    # L2: ≥20 drugs, ≥12 dx, >50 lab rows
    # L3: ≥30 drugs, ≥18 dx, >150 lab rows
    ("discharge_planning", 1): [("bytes_note_discharge.csv",    2_000),
                                ("bytes_hosp_prescriptions.csv", 800),
                                ("rows_hosp_diagnoses_icd.csv",  4),
                                ("bytes_hosp_labevents.csv",     200)],
    ("discharge_planning", 2): [("bytes_note_discharge.csv",    5_000),
                                ("bytes_hosp_prescriptions.csv", 1_500),
                                ("rows_hosp_diagnoses_icd.csv",  10),
                                ("bytes_hosp_labevents.csv",     3_000)],
    ("discharge_planning", 3): [("bytes_note_discharge.csv",    10_000),
                                ("bytes_hosp_prescriptions.csv", 2_500),
                                ("rows_hosp_diagnoses_icd.csv",  15),
                                ("bytes_hosp_labevents.csv",     8_000)],
}

# ── CSV helpers ───────────────────────────────────────────────────────────────

def _read(subject_id: int, table: str,
          usecols: list[str] | None = None) -> pd.DataFrame | None:
    path = os.path.join(DATA_ROOT, str(subject_id), f"{table}.csv.csv.gz")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, compression="gzip",
                           low_memory=False, usecols=usecols)
    except Exception:
        return None


def _filter_hadm(df: pd.DataFrame, hadm_id: int | None) -> pd.DataFrame:
    if hadm_id is not None and "hadm_id" in df.columns:
        return df[df["hadm_id"] == hadm_id]
    return df


def get_primary_hadm(subject_id: int) -> int | None:
    """Return the most recent hadm_id for a patient."""
    df = _read(subject_id, "hosp_admissions",
               usecols=["hadm_id", "admittime"])
    if df is None or df.empty:
        return None
    try:
        return int(df.sort_values("admittime", ascending=False)
                   ["hadm_id"].iloc[0])
    except Exception:
        return None


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def stage1_qualify(stats_row: dict, scenario: str, difficulty: int) -> bool:
    """Return True if stats_row passes all byte-size thresholds for this slot."""
    thresholds = _STAGE1.get((scenario, difficulty))
    if not thresholds:
        return True
    for col, min_val in thresholds:
        try:
            if float(stats_row.get(col, 0) or 0) < min_val:
                return False
        except (TypeError, ValueError):
            return False
    return True


# ── Stage 2: per-scenario content checks ─────────────────────────────────────

def _qualify_lab_trend(subject_id: int, hadm_id: int | None,
                       difficulty: int) -> bool:
    labs = _read(subject_id, "hosp_labevents",
                 usecols=["hadm_id", "itemid", "charttime"])
    if labs is None or labs.empty:
        return False
    labs = _filter_hadm(labs, hadm_id)
    all_key = set().union(*_LAB_GROUPS)
    labs = labs[labs["itemid"].isin(all_key)]
    if labs.empty:
        return False
    if difficulty == 1:
        return True

    counts = labs.groupby("itemid").size()
    if difficulty == 2:
        return bool((counts >= 3).any())

    # L3: ≥2 distinct biomarker groups each with ≥3 measurements
    groups_ok = sum(
        1 for grp in _LAB_GROUPS
        if counts.reindex(list(grp)).dropna().sum() >= 3
    )
    return groups_ok >= 2


def _qualify_med_safety(subject_id: int, hadm_id: int | None,
                        difficulty: int) -> bool:
    labs = _read(subject_id, "hosp_labevents",
                 usecols=["hadm_id", "itemid"])
    if labs is None or labs.empty:
        return False
    labs = _filter_hadm(labs, hadm_id)
    lab_items = set(labs["itemid"].dropna().astype(int))
    if not (lab_items & _MONITORING_LABS):
        return False

    rx = _read(subject_id, "hosp_prescriptions",
               usecols=["hadm_id", "drug"])
    if rx is None or rx.empty:
        return False
    rx = _filter_hadm(rx, hadm_id)
    if rx.empty:
        return False

    rx_drugs = rx["drug"].dropna().str.lower()

    def _has_drug(kws: set[str]) -> bool:
        return rx_drugs.str.contains("|".join(kws), regex=True).any()

    if difficulty == 1:
        return True

    matched = [
        cat for cat, kws in _DRUG_CATEGORIES.items()
        if _has_drug(kws) and (_CATEGORY_LABS[cat] & lab_items)
    ]
    if difficulty == 2:
        return len(matched) >= 1
    # L3: ≥2 drug-lab pairs + CHA₂DS₂-VASc capability (afib + anticoagulant)
    # OR ≥2 drug-lab pairs + Child-Pugh capability (cirrhosis + hepatic labs)
    if len(matched) < 2:
        return False
    dx = _read(subject_id, "hosp_diagnoses_icd", usecols=["hadm_id", "icd_code"])
    dx_h = _filter_hadm(dx, hadm_id) if dx is not None else None
    has_afib_scoring = (
        _has_icd(dx_h, _ICD_AFIB) and
        _has_drug({"warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban"})
    )
    has_liver_scoring = (
        _has_icd(dx_h, _ICD_CIRRHOSIS) and
        bool(lab_items & _CHILD_PUGH_LABS)
    )
    return has_afib_scoring or has_liver_scoring or len(matched) >= 2


def _qualify_treatment_response(subject_id: int, hadm_id: int | None,
                                difficulty: int) -> bool:
    labs = _read(subject_id, "hosp_labevents",
                 usecols=["hadm_id", "itemid", "charttime"])
    if labs is None or labs.empty:
        return False
    labs = _filter_hadm(labs, hadm_id)
    if labs[labs["itemid"].isin(_INFECTION_MARKERS)].empty:
        return False

    emar = _read(subject_id, "hosp_emar",
                 usecols=["hadm_id", "medication"])
    if emar is None or emar.empty:
        return False
    emar = _filter_hadm(emar, hadm_id)
    if emar.empty:
        return False
    if difficulty == 1:
        return True

    wbc = labs[labs["itemid"].isin(_WBC)]
    wbc_count = (wbc["charttime"].nunique()
                 if "charttime" in wbc.columns else len(wbc))

    meds = emar["medication"].dropna().str.lower()
    has_ab = meds.str.contains("|".join(_ALL_AB), regex=True).any()

    if difficulty == 2:
        return wbc_count >= 2 and has_ab

    # L3: ≥3 WBC timepoints + ≥2 antibiotic classes + SOFA-capable labs
    # SOFA capability = platelets + bilirubin + creatinine all present
    # (the 3 most reliably measurable SOFA components in MIMIC-IV labevents)
    if wbc_count < 3 or not has_ab:
        return False
    n_classes = sum(
        1 for drugs in _AB_CLASSES.values()
        if meds.str.contains("|".join(drugs), regex=True).any()
    )
    if n_classes < 2:
        return False
    dx = _read(subject_id, "hosp_diagnoses_icd", usecols=["hadm_id", "icd_code"])
    if dx is None or dx.empty:
        return False
    dx_h = _filter_hadm(dx, hadm_id)
    if dx_h.empty:
        return False
    # Require SOFA-capable lab panel: PLT + bilirubin + creatinine
    lab_items = set(labs["itemid"].dropna().astype(int))
    has_sofa_labs = bool(lab_items & _PLATELET) and \
                    bool(lab_items & _BILIRUBIN) and \
                    bool(lab_items & _CREATININE)
    # Also accept patients with sepsis diagnosis even without full SOFA labs
    has_sepsis_dx = _has_icd(dx_h, _ICD_SEPSIS)
    return has_sofa_labs or has_sepsis_dx


def _qualify_discharge_planning(subject_id: int, hadm_id: int | None,
                                difficulty: int) -> bool:
    """
    Discharge-planning eligibility.

    Guarantees the patient's primary admission has enough structured data to
    support all four benchmark turns without returning empty results:
      T0 Information Lookup  — Encounter / DocumentReference
      T1 Data Gathering     — CarePlan + MedicationRequest
      T2 Data Gathering/mix — patient interview grounded in discharge note
      T3 Action     — write orders based on clinical findings

    Difficulty scaling — true data-volume stratification:
      L1: ≥10 distinct drugs  +  ≥5  diagnoses  +  labevents present
      L2: ≥20 distinct drugs  +  ≥12 diagnoses  +  >50  lab rows
      L3: ≥30 distinct drugs  +  ≥18 diagnoses  +  >150 lab rows
    """
    # Exclude patients who died in hospital — discharge planning is meaningless for them
    admissions = _read(subject_id, "hosp_admissions",
                       usecols=["hadm_id", "discharge_location"])
    if admissions is not None and not admissions.empty:
        adm = _filter_hadm(admissions, hadm_id)
        if not adm.empty:
            loc = str(adm["discharge_location"].iloc[0]).strip().upper()
            _DEAD = {"DIED", "DEAD", "EXPIRED", "HOSPICE", "DIED IN HOSPITAL"}
            if any(kw in loc for kw in _DEAD):
                return False

    # Discharge note mandatory at every level (grounds T2 patient interview)
    note = _read(subject_id, "note_discharge")
    if note is None or note.empty:
        return False

    # Prescriptions scoped to this admission
    rx = _read(subject_id, "hosp_prescriptions", usecols=["hadm_id", "drug"])
    if rx is None or rx.empty:
        return False
    rx = _filter_hadm(rx, hadm_id)
    if rx.empty:
        return False
    n_drugs = rx["drug"].dropna().str.lower().nunique()

    # Diagnoses scoped to this admission
    dx = _read(subject_id, "hosp_diagnoses_icd", usecols=["hadm_id", "icd_code"])
    if dx is None or dx.empty:
        return False
    dx = _filter_hadm(dx, hadm_id)
    n_dx = len(dx)

    # Labs scoped to this admission (required at all levels)
    labs = _read(subject_id, "hosp_labevents", usecols=["hadm_id", "itemid"])
    if labs is None or labs.empty:
        return False
    n_labs = len(_filter_hadm(labs, hadm_id))

    if difficulty == 1:
        return n_drugs >= 10 and n_dx >= 5 and n_labs > 0

    if difficulty == 2:
        return n_drugs >= 20 and n_dx >= 12 and n_labs > 50

    # L3: polypharmacy + complex multi-system disease + rich lab record
    return n_drugs >= 30 and n_dx >= 18 and n_labs > 150


def _count_lab_systems(item_ids: set[int]) -> int:
    """Return number of distinct organ systems covered by the given lab itemid set."""
    return sum(1 for items in _LAB_SYSTEMS.values() if items & item_ids)


def _qualify_diagnostic_workup(subject_id: int, hadm_id: int | None,
                               difficulty: int) -> bool:
    """
    Diagnostic workup eligibility — multi-system strategy.

    Requires genuine multi-system involvement (not just single-domain labs)
    plus imaging reports for differential diagnosis grounding:

      L1: labs span ≥2 organ systems  +  ≥1 radiology report  +  ≥3 diagnoses
      L2: labs span ≥3 organ systems  +  ≥2 radiology reports  +  ≥5 diagnoses
      L3: labs span ≥4 organ systems  +  ≥3 radiology reports  +  ≥8 diagnoses

    Organ systems tracked: hematologic, renal, hepatic, metabolic,
                           coagulation, inflammatory, cardiac  (7 total)
    """
    labs = _read(subject_id, "hosp_labevents", usecols=["hadm_id", "itemid"])
    if labs is None or labs.empty:
        return False
    labs = _filter_hadm(labs, hadm_id)
    if labs.empty:
        return False
    lab_items = set(labs["itemid"].dropna().astype(int))
    n_systems = _count_lab_systems(lab_items)

    radiology = _read(subject_id, "note_radiology")
    if radiology is None or radiology.empty:
        n_reports = 0
    else:
        # Scope to current admission — reports from other admissions are not
        # queryable via DiagnosticReport.search(hadm_id=...) and must not count.
        radiology_hadm = _filter_hadm(radiology, hadm_id)
        n_reports = len(radiology_hadm)

    dx = _read(subject_id, "hosp_diagnoses_icd", usecols=["hadm_id", "icd_code"])
    n_dx = 0
    if dx is not None and not dx.empty:
        n_dx = len(_filter_hadm(dx, hadm_id))

    dx_h = _filter_hadm(
        _read(subject_id, "hosp_diagnoses_icd", usecols=["hadm_id","icd_code"]),
        hadm_id
    ) if _read(subject_id, "hosp_diagnoses_icd", usecols=["hadm_id","icd_code"]) is not None else None

    if difficulty == 1:
        return n_systems >= 2 and n_reports >= 1 and n_dx >= 3

    if difficulty == 2:
        # L2: multi-source + scoring capability (SIRS, CURB-65, or Wells context)
        basic_ok = n_systems >= 3 and n_reports >= 2 and n_dx >= 5
        # Bonus: CURB-65 capable (pneumonia dx + BUN + creatinine)
        has_curb65 = (bool(lab_items & _CURB65_LABS) and
                      _has_icd(dx_h, _ICD_PNEUMONIA))
        # OR: Child-Pugh capable (cirrhosis + hepatic labs)
        has_child_pugh = (bool(lab_items & _CHILD_PUGH_LABS) and
                          _has_icd(dx_h, _ICD_CIRRHOSIS))
        return basic_ok or has_curb65 or has_child_pugh

    # L3: four organ systems + scoring-capable patient
    # Child-Pugh OR Wells (thrombosis context) required for advanced kn.-grounded turns
    basic_l3 = n_systems >= 4 and n_reports >= 3 and n_dx >= 8
    has_child_pugh = (bool(lab_items & _CHILD_PUGH_LABS) and
                      _has_icd(dx_h, _ICD_CIRRHOSIS))
    has_sepsis_scoring = (bool(lab_items & _INFECTION_MARKERS) and
                          _has_icd(dx_h, _ICD_SEPSIS))
    return basic_l3 or has_child_pugh or has_sepsis_scoring


_STAGE2_FNS = {
    "diagnostic_workup":   _qualify_diagnostic_workup,
    "lab_trend":           _qualify_lab_trend,
    "med_safety":          _qualify_med_safety,
    "treatment_response":  _qualify_treatment_response,
    "discharge_planning":  _qualify_discharge_planning,
}


def stage2_qualify(subject_id: int, hadm_id: int | None,
                   scenario: str, difficulty: int) -> bool:
    """Return True if this patient's EHR content satisfies scenario×difficulty."""
    if hadm_id is None:
        hadm_id = get_primary_hadm(subject_id)
    fn = _STAGE2_FNS.get(scenario)
    if fn is None:
        return True
    try:
        return fn(subject_id, hadm_id, difficulty)
    except Exception as exc:
        logger.debug(f"Stage2 error subject={subject_id}: {exc}")
        return False


# ── Stats index ───────────────────────────────────────────────────────────────
_stats_cache: dict[int, dict] | None = None


def load_stats_index(csv_path: str = STATS_CSV) -> dict[int, dict]:
    """Load patient_info_stats.csv into a dict keyed by subject_id (cached)."""
    global _stats_cache
    if _stats_cache is not None:
        return _stats_cache
    logger.info(f"Loading stats index from {csv_path} …")
    df = pd.read_csv(csv_path, low_memory=False)
    _stats_cache = {
        int(row["subject_id"]): row.to_dict()
        for _, row in df.iterrows()
    }
    logger.info(f"Loaded {len(_stats_cache):,} patient records")
    return _stats_cache


# ── Combined entry point (used by prefilter_patients.py) ─────────────────────

def prefilter_pool(
    candidates: list[tuple[int, int | None]],
    scenario: str,
    difficulty: int,
    n: int,
    stats_index: dict[int, dict] | None = None,
    max_stage2_scan: int = 5_000,
) -> list[tuple[int, int | None]]:
    """
    Run Stage 1 then Stage 2 on candidates; return up to n qualified pairs.
    """
    if stats_index is None:
        stats_index = load_stats_index()

    s1 = [
        (sid, hid) for sid, hid in candidates
        if stage1_qualify(stats_index.get(sid, {}), scenario, difficulty)
    ]
    logger.info(
        f"[{scenario}/L{difficulty}] Stage1: {len(s1)}/{len(candidates)} passed"
    )

    qualified: list[tuple[int, int | None]] = []
    for sid, hid in s1[:max_stage2_scan]:
        if len(qualified) >= n:
            break
        if stage2_qualify(sid, hid, scenario, difficulty):
            qualified.append((sid, hid))

    logger.info(
        f"[{scenario}/L{difficulty}] Stage2: {len(qualified)} qualified"
    )
    return qualified
