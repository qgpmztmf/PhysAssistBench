"""
Tool registry: maps tool name strings → callable functions from ehr_api_tools,
patient_interview_tools, and the new FHIR tool layer.

Includes argument normalization: strips unknown kwargs and maps common aliases
so the LLM can't break tool calls with wrong argument names.

Two registries are provided:
  TOOL_REGISTRY      — legacy tools (28 EHR + 6 Patient) for backward compat
  FHIR_TOOL_REGISTRY — 13 FHIR R4 tools + 3 Patient tools
  TOOL_REGISTRY      — merged (FHIR names take precedence for overlapping keys)
"""
import inspect
from physassistbench.tools.patient_interview_tools import (
    patient_get_chief_complaint,
    patient_get_symptom_history,
    patient_get_medication_adherence,
    patient_get_social_history,
    patient_get_functional_status,
    patient_get_pain_assessment,
)
from physassistbench.phm.patient_agent_runtime import register_session, get_session, reset_all_sessions  # noqa: F401
from physassistbench.tools.fhir.fhir_tools import (
    fhir_patient_read,
    fhir_encounter_search,
    fhir_condition_search,
    fhir_observation_search,
    fhir_medication_request_search,
    fhir_medication_administration_search,
    fhir_procedure_search,
    fhir_diagnostic_report_search,
    fhir_document_reference_search,
    fhir_allergy_intolerance_search,
    fhir_care_plan_search,
    fhir_patient_everything,
    fhir_prepare_to_answer,
)
from physassistbench.tools.ehr_api_tools import (
    get_patient_info, get_admissions, get_admission_details,
    get_diagnoses, get_lab_results, get_lab_trends,
    get_microbiology_results, get_prescriptions,
    get_medication_administration, get_procedures,
    get_drg_info, get_service_history,
    get_icu_stays, get_icu_vitals, get_icu_fluids_in, get_icu_output,
    get_ed_visits, get_ed_triage, get_ed_vital_signs,
    get_ed_diagnoses, get_ed_medications,
    get_discharge_summary, get_discharge_section,
    get_radiology_report, search_notes,
    get_vital_signs_outpatient, get_patient_timeline,
    prepare_to_answer,
)

TOOL_REGISTRY = {
    "get_patient_info": get_patient_info,
    "get_admissions": get_admissions,
    "get_admission_details": get_admission_details,
    "get_diagnoses": get_diagnoses,
    "get_lab_results": get_lab_results,
    "get_lab_trends": get_lab_trends,
    "get_microbiology_results": get_microbiology_results,
    "get_prescriptions": get_prescriptions,
    "get_medication_administration": get_medication_administration,
    "get_procedures": get_procedures,
    "get_drg_info": get_drg_info,
    "get_service_history": get_service_history,
    "get_icu_stays": get_icu_stays,
    "get_icu_vitals": get_icu_vitals,
    "get_icu_fluids_in": get_icu_fluids_in,
    "get_icu_output": get_icu_output,
    "get_ed_visits": get_ed_visits,
    "get_ed_triage": get_ed_triage,
    "get_ed_vital_signs": get_ed_vital_signs,
    "get_ed_diagnoses": get_ed_diagnoses,
    "get_ed_medications": get_ed_medications,
    "get_discharge_summary": get_discharge_summary,
    "get_discharge_section": get_discharge_section,
    "get_radiology_report": get_radiology_report,
    "search_notes": search_notes,
    "get_vital_signs_outpatient": get_vital_signs_outpatient,
    "get_patient_timeline": get_patient_timeline,
    "prepare_to_answer": prepare_to_answer,
    # ── Patient Interview Tools ──────────────────────────────────────────────
    "patient.get_chief_complaint": patient_get_chief_complaint,
    "patient.get_symptom_history": patient_get_symptom_history,
    "patient.get_medication_adherence": patient_get_medication_adherence,
    "patient.get_social_history": patient_get_social_history,
    "patient.get_functional_status": patient_get_functional_status,
    "patient.get_pain_assessment": patient_get_pain_assessment,
}

# ── FHIR Tool Registry ────────────────────────────────────────────────────────
# HL7 FHIR R4 style tools (Resource.operation naming convention)

FHIR_TOOL_REGISTRY: dict = {
    "Patient.read":                    fhir_patient_read,
    "Encounter.search":                fhir_encounter_search,
    "Condition.search":                fhir_condition_search,
    "Observation.search":              fhir_observation_search,
    "MedicationRequest.search":        fhir_medication_request_search,
    "MedicationAdministration.search": fhir_medication_administration_search,
    "Procedure.search":                fhir_procedure_search,
    "DiagnosticReport.search":         fhir_diagnostic_report_search,
    "DocumentReference.search":        fhir_document_reference_search,
    "AllergyIntolerance.search":       fhir_allergy_intolerance_search,
    "CarePlan.search":                 fhir_care_plan_search,
    "Patient.everything":              fhir_patient_everything,
    "prepare_to_answer":               fhir_prepare_to_answer,
    # ── Patient Interview Tools (shared with legacy) ─────────────────────────
    "patient.get_chief_complaint":     patient_get_chief_complaint,
    "patient.get_symptom_history":     patient_get_symptom_history,
    "patient.get_medication_adherence":patient_get_medication_adherence,
    "patient.get_social_history":      patient_get_social_history,
    "patient.get_functional_status":   patient_get_functional_status,
    "patient.get_pain_assessment":     patient_get_pain_assessment,
}

# ── Unified registry (FHIR + legacy, FHIR names take precedence) ─────────────
_ALL_REGISTRIES = {**TOOL_REGISTRY, **FHIR_TOOL_REGISTRY}

# Pre-compute valid parameter names for each tool from their actual signatures
_VALID_PARAMS: dict[str, set[str]] = {
    name: set(inspect.signature(fn).parameters.keys())
    for name, fn in _ALL_REGISTRIES.items()
}

# FHIR aliases: common LLM mistakes on FHIR tool parameter names
_FHIR_ARG_ALIASES: dict[tuple[str, str], str | None] = {
    # Observation.search
    ("Observation.search", "test_name"):   "code",
    ("Observation.search", "lab_test"):    "code",
    ("Observation.search", "item_name"):   "code",
    ("Observation.search", "vital_name"):  "code",
    ("Observation.search", "type"):        "category",
    ("Observation.search", "limit"):       "_count",
    # Condition.search
    ("Condition.search", "icd_code"):      "code",
    ("Condition.search", "diagnosis"):     "code",
    # MedicationRequest.search
    ("MedicationRequest.search", "drug"):  "medication",
    ("MedicationRequest.search", "med"):   "medication",
    # DiagnosticReport.search
    ("DiagnosticReport.search", "type"):   "report_type",
    # DocumentReference.search
    ("DocumentReference.search", "type"):  "type_code",
    ("DocumentReference.search", "query"): "keyword",
}

# Aliases: (tool_name, wrong_arg_name) -> correct_arg_name (or None to drop)
# These cover the most common LLM hallucinations observed in generation logs.
_ARG_ALIASES: dict[tuple[str, str], str | None] = {
    # get_lab_results aliases
    ("get_lab_results", "test_name"):    "item_name",
    ("get_lab_results", "test_type"):    None,   # drop — not a real param
    ("get_lab_results", "limit"):        None,
    ("get_lab_results", "item_ids"):     None,
    ("get_lab_results", "test_items"):   None,
    ("get_lab_results", "lab_items"):    None,
    ("get_lab_results", "test_names"):   None,
    ("get_lab_results", "lab_name"):     "item_name",
    ("get_lab_results", "lab_test"):     "item_name",
    ("get_lab_results", "test"):         "item_name",
    ("get_lab_results", "medication"):   None,

    # get_lab_trends aliases
    ("get_lab_trends", "test_name"):     "item_name",
    ("get_lab_trends", "test_itemid"):   "item_name",
    ("get_lab_trends", "lab_itemid"):    "item_name",
    ("get_lab_trends", "lab_test"):      "item_name",
    ("get_lab_trends", "lab_name"):      "item_name",
    ("get_lab_trends", "test"):          "item_name",
    ("get_lab_trends", "hadm_id"):       None,   # drop — not accepted
    ("get_lab_trends", "n_results"):     "n_recent",
    ("get_lab_trends", "num_recent"):    "n_recent",
    ("get_lab_trends", "count"):         "n_recent",

    # get_patient_timeline aliases
    ("get_patient_timeline", "hadm_id"): None,   # drop — not accepted

    # get_icu_vitals aliases
    ("get_icu_vitals", "vital_sign"):    "vital_name",
    ("get_icu_vitals", "vitals"):        None,

    # get_discharge_section aliases
    ("get_discharge_section", "section"): "section_name",

    # get_radiology_report aliases
    ("get_radiology_report", "type"):    "report_type",

    # get_medication_administration aliases
    ("get_medication_administration", "drug"): "medication",
    ("get_medication_administration", "med"):  "medication",

    # search_notes aliases
    ("search_notes", "query"):           "keyword",
    ("search_notes", "search_term"):     "keyword",
}

# Merge FHIR aliases into the main alias table
_ARG_ALIASES.update(_FHIR_ARG_ALIASES)


def _normalize_args(tool_name: str, args: dict) -> dict:
    """
    Normalize arguments for a tool call:
    1. Apply known aliases (wrong_name -> correct_name or drop).
    2. Strip any remaining unknown arguments.
    """
    valid = _VALID_PARAMS.get(tool_name, set())
    normalized: dict = {}

    for k, v in args.items():
        if k in valid:
            # Already correct
            normalized[k] = v
        elif (tool_name, k) in _ARG_ALIASES:
            correct = _ARG_ALIASES[(tool_name, k)]
            if correct is not None and correct not in normalized:
                normalized[correct] = v
            # else: drop the argument
        else:
            # Unknown arg not in aliases — silently drop it
            pass

    return normalized


# ── Active-session date gate ─────────────────────────────────────────────────
# Single-threaded: stores the current_date for the active patient session so
# call_tool can auto-inject date_to into time-ordered FHIR tools, preventing
# future observations (from later admissions) from leaking into results.

_active_date: str | None = None

# Tools whose results are sorted newest-first and support date_to filtering.
# Only Observation.search exposes date_to today; others are bounded by hadm_id.
_DATE_BOUNDED_TOOLS: frozenset[str] = frozenset({
    "Observation.search",
})


def set_active_date(date: str | None) -> None:
    """Set the current_date upper bound for the active patient session."""
    global _active_date
    _active_date = date


def get_active_date() -> str | None:
    return _active_date


def call_tool(name: str, arguments: dict, tool_set: str = "auto") -> dict:
    """Execute a tool by name with given arguments. Returns observation dict.

    tool_set:
      "auto"   — try FHIR registry first, then legacy (default)
      "fhir"   — FHIR registry only
      "legacy" — legacy registry only
    """
    if tool_set == "fhir":
        registry = FHIR_TOOL_REGISTRY
    elif tool_set == "legacy":
        registry = TOOL_REGISTRY
    else:
        registry = _ALL_REGISTRIES

    # Write tools (MedicationRequest.create / ServiceRequest.create / Flag.create)
    # are not in the read registries — route them to the write simulator so the
    # agent receives a successful "resource created" confirmation (mirrors how the
    # benchmark was generated). Without this the agent sees "Unknown tool" and
    # reports failure, tanking Write/Update rubric scores.
    if name in ("MedicationRequest.create", "ServiceRequest.create", "Flag.create"):
        from physassistbench.pipeline.generate_discharge_planning import _exec_write_tool
        return _exec_write_tool(name, arguments)

    fn = registry.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        clean_args = _normalize_args(name, arguments)
        # Auto-inject date_to upper bound to prevent future-admission data leakage.
        # Only applied when the caller has not already specified date_to and an
        # active session date has been set via set_active_date().
        if (
            name in _DATE_BOUNDED_TOOLS
            and _active_date is not None
            and "date_to" not in clean_args
            and "date_to" in _VALID_PARAMS.get(name, set())
        ):
            clean_args["date_to"] = _active_date
        result = fn(**clean_args)
        # Patient tools return str — wrap in a dict for uniform observation format
        if isinstance(result, str):
            obs = {"patient_response": result}
            # Also generate responses for all 3 health literacy levels.
            # The session's persona literacy is preserved; we temporarily swap it
            # for each variant, then restore.
            session_id = clean_args.get("session_id", "")
            if session_id and name.startswith("patient."):
                try:
                    from physassistbench.phm.patient_agent_runtime import get_session as _gs
                    rt = _gs(session_id)
                    orig_literacy = rt.persona.get("health_literacy", "medium")
                    _qt_map = {
                        "patient.get_chief_complaint":      ("get_chief_complaint",     {}),
                        "patient.get_symptom_history":      ("get_symptom_history",     {"query": clean_args.get("query", "")}),
                        "patient.get_medication_adherence": ("get_medication_adherence", {"drug": clean_args.get("drug", "")}),
                        "patient.get_social_history":       ("get_social_history",       {}),
                        "patient.get_functional_status":    ("get_functional_status",    {}),
                        "patient.get_pain_assessment":      ("get_pain_assessment",      {}),
                    }
                    qt, qt_kwargs = _qt_map.get(name, (None, {}))
                    if qt:
                        # Generate the HIGH-literacy response once (most precise, all
                        # clinical facts), then STYLE-REWRITE it down to medium/low so
                        # all three variants share identical clinical facts and differ
                        # only in expression. This avoids fact drift across variants.
                        if orig_literacy == "high":
                            high_resp = result
                        else:
                            rt.persona["health_literacy"] = "high"
                            try:
                                high_resp = rt.respond(qt, **qt_kwargs)
                            except Exception:
                                high_resp = result
                            rt.persona["health_literacy"] = orig_literacy

                        all_lit = {
                            "high":   high_resp,
                            "medium": rt.rewrite_literacy(high_resp, "medium"),
                            "low":    rt.rewrite_literacy(high_resp, "low"),
                        }
                        obs["patient_responses_all_literacy"] = all_lit
                except Exception:
                    pass   # best-effort; don't break main flow
            return obs
        return result
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"Tool execution error for {name}: {e}"}
