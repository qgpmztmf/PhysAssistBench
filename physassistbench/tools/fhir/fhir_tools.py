"""
tools/fhir/fhir_tools.py — 13 FHIR R4 tool functions (EHR side).

Each function:
  - Accepts HL7 FHIR-style search parameters
  - Calls the adapter layer to convert MIMIC-IV CSV → FHIR Resources
  - Returns a FHIR Bundle dict (JSON-serializable)

Tool set (13 EHR tools):
  Patient.read
  Encounter.search
  Condition.search
  Observation.search
  MedicationRequest.search
  MedicationAdministration.search
  Procedure.search
  DiagnosticReport.search
  DocumentReference.search
  AllergyIntolerance.search
  CarePlan.search
  Patient.everything
  prepare_to_answer              (pass-through, same as legacy)
"""

from __future__ import annotations

import functools
import inspect
from typing import Optional

from physassistbench.tools.fhir.adapter import (
    _bundle,
    patient_to_fhir,
    encounters_to_fhir,
    conditions_to_fhir,
    lab_observations_to_fhir,
    vital_observations_to_fhir,
    medication_requests_to_fhir,
    medication_administrations_to_fhir,
    procedures_to_fhir,
    diagnostic_reports_to_fhir,
    document_references_to_fhir,
    allergies_to_fhir,
    care_plans_to_fhir,
)


def _error(msg: str) -> dict:
    return {"resourceType": "OperationOutcome",
            "issue": [{"severity": "error", "diagnostics": msg}]}


# Params that must NEVER be coerced to None (required ids or structural args).
_KEEP_PARAMS = {"subject_id", "_count", "_sort", "abnormal_only", "answer_type"}


def _normalize_optional_params(fn):
    """Coerce placeholder optional args to None before the tool runs.

    Some models (notably GPT-5.x) fill EVERY optional parameter with a
    placeholder — hadm_id=0, code="", date_from="" — instead of omitting it.
    Downstream filters then match nothing (e.g. hadm_id=0 → admission 0 → empty
    Bundle → "No observations found"), which unfairly tanks retrieval. Treat such
    falsy placeholders ('' / whitespace / 0) as "not provided".
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, val in list(bound.arguments.items()):
            if name in _KEEP_PARAMS:
                continue
            if isinstance(val, bool):
                continue
            if isinstance(val, str):
                bound.arguments[name] = val.strip() or None
            elif isinstance(val, int) and val == 0:
                bound.arguments[name] = None
        return fn(*bound.args, **bound.kwargs)

    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Patient.read
# ─────────────────────────────────────────────────────────────────────────────

def fhir_patient_read(subject_id: int) -> dict:
    """
    Read demographic information for a patient as a FHIR Patient resource.
    Returns gender, anchor_age, and deceased date if applicable.
    """
    resource = patient_to_fhir(subject_id)
    if resource is None:
        return _error(f"Patient/{subject_id} not found")
    return resource


# ─────────────────────────────────────────────────────────────────────────────
# Encounter.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_encounter_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
) -> dict:
    """
    Search hospital encounters (admissions) for a patient.
    Optionally filter by hadm_id to get a single encounter.
    Returns a FHIR Bundle of Encounter resources with admission/discharge
    times, admission type, location, and discharge disposition.
    """
    resources = encounters_to_fhir(subject_id, hadm_id)
    if not resources:
        return _error(f"No encounters found for Patient/{subject_id}")
    return _bundle(resources, "Encounter")


# ─────────────────────────────────────────────────────────────────────────────
# Condition.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_condition_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
    code: Optional[str] = None,
    clinical_status: Optional[str] = None,
) -> dict:
    """
    Search ICD diagnosis conditions for a patient.
    Filter by hadm_id (specific admission), ICD code prefix, or clinical_status.
    Results are ordered by seq_num (principal diagnosis first).
    Returns a FHIR Bundle of Condition resources.
    """
    resources = conditions_to_fhir(subject_id, hadm_id, code)
    if clinical_status:
        resources = [
            r for r in resources
            if any(c.get("code") == clinical_status
                   for coding in r.get("clinicalStatus", {}).get("coding", [])
                   for c in [coding])
        ]
    if not resources:
        return _error(f"No conditions found for Patient/{subject_id}")
    return _bundle(resources, "Condition")


# ─────────────────────────────────────────────────────────────────────────────
# Observation.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_observation_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
    category: Optional[str] = None,
    code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    _sort: str = "-date",
    _count: int = 20,
) -> dict:
    """
    Search Observation resources for a patient.
    category: 'laboratory' | 'vital-signs' | 'microbiology'
    code: lab test name (e.g. 'Potassium') or LOINC code
    date_from / date_to: ISO datetime strings for time filtering
    _count: maximum results to return (default 20, applies even when code is specified)
    Results are sorted newest-first by default (_sort='-date').
    Returns a FHIR Bundle of Observation resources.
    """
    cat = (category or "").lower()

    if cat in ("vital-signs", "vital_signs", "vitals"):
        resources = vital_observations_to_fhir(
            subject_id=subject_id,
            stay_id=None,
            vital_name=code,
            _count=_count,
        )
    else:
        # Default: laboratory (also handles microbiology via item_name filter)
        resources = lab_observations_to_fhir(
            subject_id=subject_id,
            hadm_id=hadm_id,
            item_name=code,
            abnormal_only=False,
            _count=_count,
            date_from=date_from,
            date_to=date_to,
        )
        if cat == "microbiology":
            # Microbiology items typically have category info in item labels
            resources = [
                r for r in resources
                if "micro" in str(r.get("code", {}).get("text", "")).lower()
                or "culture" in str(r.get("code", {}).get("text", "")).lower()
                or "bacteria" in str(r.get("code", {}).get("text", "")).lower()
            ]

    if not resources:
        return _error(
            f"No observations found for Patient/{subject_id}"
            + (f" category={category}" if category else "")
            + (f" code={code}" if code else "")
        )
    return _bundle(resources, "Observation")


# ─────────────────────────────────────────────────────────────────────────────
# MedicationRequest.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_medication_request_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
    medication: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    """
    Search prescription (MedicationRequest) resources for a patient.
    medication: drug name substring filter (e.g. 'Furosemide')
    status: 'active' | 'completed' | 'stopped'
    When hadm_id is provided, returns all prescriptions for that admission.
    Returns a FHIR Bundle of MedicationRequest resources sorted by start time (newest first).
    """
    resources = medication_requests_to_fhir(subject_id, hadm_id, medication, status)
    if not resources:
        return _error(f"No medication requests found for Patient/{subject_id}")
    return _bundle(resources, "MedicationRequest")


# ─────────────────────────────────────────────────────────────────────────────
# MedicationAdministration.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_medication_administration_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
    medication: Optional[str] = None,
) -> dict:
    """
    Search medication administration (eMAR) records for a patient.
    medication: drug name substring filter
    Returns a FHIR Bundle of MedicationAdministration resources.
    """
    resources = medication_administrations_to_fhir(subject_id, hadm_id, medication)
    if not resources:
        return _error(f"No medication administrations found for Patient/{subject_id}")
    return _bundle(resources, "MedicationAdministration")


# ─────────────────────────────────────────────────────────────────────────────
# Procedure.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_procedure_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
) -> dict:
    """
    Search ICD procedure codes for a patient.
    Returns a FHIR Bundle of Procedure resources ordered by seq_num.
    """
    resources = procedures_to_fhir(subject_id, hadm_id)
    if not resources:
        return _error(f"No procedures found for Patient/{subject_id}")
    return _bundle(resources, "Procedure")


# ─────────────────────────────────────────────────────────────────────────────
# DiagnosticReport.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_diagnostic_report_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
    report_type: Optional[str] = None,
) -> dict:
    """
    Search radiology and other diagnostic reports for a patient.
    report_type: partial match on note_type (e.g. 'CT', 'MRI', 'X-ray')
    Returns a FHIR Bundle of DiagnosticReport resources (newest first, max 5).
    Full report text is included in presentedForm.data.
    """
    resources = diagnostic_reports_to_fhir(subject_id, hadm_id, report_type)
    if not resources:
        return _error(f"No diagnostic reports found for Patient/{subject_id}")
    return _bundle(resources, "DiagnosticReport")


# ─────────────────────────────────────────────────────────────────────────────
# DocumentReference.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_document_reference_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
    type_code: Optional[str] = None,
    keyword: Optional[str] = None,
) -> dict:
    """
    Search clinical document references (discharge summaries, radiology notes).
    type_code: 'discharge-summary' | 'radiology' (partial match)
    keyword: free-text search within note content
    Returns a FHIR Bundle of DocumentReference resources.
    Full note text is included in content[].attachment.data (truncated to 4000 chars).
    """
    resources = document_references_to_fhir(subject_id, hadm_id, type_code, keyword)
    if not resources:
        return _error(f"No documents found for Patient/{subject_id}")
    return _bundle(resources, "DocumentReference")


# ─────────────────────────────────────────────────────────────────────────────
# AllergyIntolerance.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_allergy_intolerance_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
) -> dict:
    """
    Search allergy/intolerance records for a patient.
    Note: MIMIC-IV does not have a dedicated allergy table; allergies are
    extracted from the 'Allergies' section of discharge notes.
    Returns a FHIR Bundle of AllergyIntolerance resources, or an empty bundle
    if no allergy information is available.
    """
    resources = allergies_to_fhir(subject_id, hadm_id)
    if not resources:
        return _bundle([], "AllergyIntolerance")
    return _bundle(resources, "AllergyIntolerance")


# ─────────────────────────────────────────────────────────────────────────────
# CarePlan.search
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_care_plan_search(
    subject_id: int,
    hadm_id: Optional[int] = None,
    category: Optional[str] = None,
) -> dict:
    """
    Search care plan records for a patient extracted from MIMIC-IV discharge notes.

    MIMIC-IV has no dedicated care plan table; care plan content is parsed from
    structured sections of discharge summaries:
      - "discharge-planning": Discharge Instructions, Discharge Condition
      - "followup": Followup Instructions / Follow-Up
      - "treatment": Free-text Plan sections

    Args:
        subject_id: MIMIC-IV patient identifier.
        hadm_id:    Scope to one hospital admission (optional).
        category:   Filter by plan type: "discharge-planning" | "followup" |
                    "treatment".  Omit to return all categories.

    Returns:
        FHIR Bundle of CarePlan resources.  Returns an empty bundle when no
        discharge notes are available for this patient.
    """
    resources = care_plans_to_fhir(subject_id, hadm_id, category)
    return _bundle(resources, "CarePlan")


# ─────────────────────────────────────────────────────────────────────────────
# Patient.everything
# ─────────────────────────────────────────────────────────────────────────────

@_normalize_optional_params
def fhir_patient_everything(subject_id: int, hadm_id: Optional[int] = None) -> dict:
    """
    Patient.$everything — Returns a summary Bundle of all available FHIR
    resources for the patient: Patient, Encounters, Conditions, Observations
    (recent labs + vitals), MedicationRequests, and Procedures.
    Scoped to one admission when hadm_id is provided.
    Useful for getting a complete clinical picture in a single call.
    """
    all_resources: list[dict] = []

    pat = patient_to_fhir(subject_id)
    if pat:
        all_resources.append(pat)

    all_resources.extend(encounters_to_fhir(subject_id, hadm_id))
    all_resources.extend(conditions_to_fhir(subject_id, hadm_id))
    all_resources.extend(
        lab_observations_to_fhir(subject_id, hadm_id, _count=20)
    )
    all_resources.extend(
        vital_observations_to_fhir(subject_id, _count=10)
    )
    all_resources.extend(medication_requests_to_fhir(subject_id, hadm_id))
    all_resources.extend(procedures_to_fhir(subject_id, hadm_id))

    return _bundle(all_resources, "Patient")


# ─────────────────────────────────────────────────────────────────────────────
# prepare_to_answer (pass-through — same semantics as legacy tool)
# ─────────────────────────────────────────────────────────────────────────────

def fhir_prepare_to_answer(answer_type: str = "tool") -> dict:
    """Signal that the agent has gathered sufficient information to answer."""
    return {"status": "ready", "answer_type": answer_type}
